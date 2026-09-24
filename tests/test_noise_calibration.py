"""Per-observation noise terms, per-spectrum calibration and the profiled polynomial
calibration (v1.0.7).

* ``log_err_scale`` / ``log_jitter`` / ``log_f_calib`` / ``log_f_data`` are named per
  observation like the outlier mixture: ``<name>_<kind>`` (one observation of the kind) or
  ``<name>_<kind>_<obs.name>``; the old shared names are refused with a teaching error.
* ``spectrum_scaling`` / ``spectrum_calib``: the plain name for a single Spectrum,
  ``spectrum_scaling_<obs.name>`` for each of several.
* ``Spectrum(polynomial_order=M)``: the calibration polynomial is solved by weighted least
  squares inside the likelihood (Prospector's PolyOptCal), reproduced against Prospector's
  own ``compute_response``; equal to the maximum over the sampled calibration; differentiable.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from _gridfixture import require_test_grid

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology, fitSED, read_result_h5
from ceridwen.broadening import Instrument
from ceridwen.fit import _likelihood_for
from ceridwen.likelihood.likelihood import (MultiObservationLikelihood, observation_data,
                                            DiagonalGaussianLikelihood)
from ceridwen.likelihood.noise_model import DiagonalNoiseModel
from ceridwen.likelihood.poly_calibration import (PolynomialCalibration,
                                                  chebyshev_design_matrix)
from ceridwen.csp.spectrum_calibration import legendre_design_matrix
from ceridwen.observation import Photometry, Spectrum
from ceridwen.sampler.priors import Uniform

REF = pathlib.Path(__file__).resolve().parent / "reference" / "prospector_polyopt.npz"
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
ZRED = 0.3


@pytest.fixture(scope="module")
def csp():
    ssp = SSPData.load(str(require_test_grid()))
    return CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 9.0, 4), zh_const=True,
                    add_dust=True, add_diffuse_dust=True, add_neb=False, add_igm=False,
                    verbose=False, cosmo=Cosmology.planck18())


def _spec(name, lo, hi, n=120, **kw):
    w = np.linspace(lo, hi, n)
    return Spectrum(wavelength=w, flux=np.full(n, 1e-29), uncertainty=np.full(n, 1e-30),
                    instrument=Instrument.sigma_kms(150.0), name=name, **kw)


def _phot():
    return Photometry(filters=FILTERS, flux=[1e-9] * 4, uncertainty=[1e-10] * 4, name="p")


def _model(csp, obs, extra_init=None, priors=None, transforms=None):
    init = {"logmass": jnp.array([10.0])}
    init.update(extra_init or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SedModel(csp, obs, priors=dict(priors or {}), free_param_init=init,
                     transforms=dict(transforms or {}), zred=ZRED)
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    for o in m.observations:                     # data = the model at theta_init, 5 % errors
        mu = np.asarray(m.predict(th)[o.name])
        o.flux = jnp.asarray(mu * (1.0 + 0.03 * np.cos(np.arange(mu.size))))
        o.uncertainty = jnp.asarray(0.05 * np.abs(mu) + 1e-40)
    return m


def _lhs(m):
    keys = tuple(m.obs_dict)
    return MultiObservationLikelihood(keys=keys, likelihoods=tuple(
        _likelihood_for(m.obs_dict[k], m.param_names, model=m) for k in keys))


def _lnl(m, lh, th):
    lnprob = lh.make_lnprobfn(m.obs_dict, m, _NoPrior())
    return lnprob(th)


class _NoPrior:
    @staticmethod
    def log_prob(theta):
        return 0.0


def _gauss_lnl(y, mu, var, mask):
    y, mu, var = (np.asarray(a, float) for a in (y, mu, var))
    m = np.asarray(mask, bool)
    return float(np.sum((-0.5 * (y - mu) ** 2 / var - 0.5 * np.log(2 * np.pi * var))[m]))


# ------------------------------------------------------------ per-observation noise -----
def test_single_spectrum_kind_names(csp):
    m = _model(csp, [_spec("s", 5000, 8000), _phot()],
               extra_init={"log_jitter_spec": jnp.array([-66.0]),
                           "log_err_scale_phot": jnp.array([0.3])})
    lh = dict(zip(m.obs_dict, _lhs(m).likelihoods))
    assert lh["s"].noise_model.nuisance_param_names == ("log_jitter_spec",)
    assert lh["p"].noise_model.nuisance_param_names == ("log_err_scale_phot",)
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    th["log_jitter_spec"] = jnp.array([np.log(4e-30)])
    pred = m.predict(th)
    want = 0.0
    for k, o in m.obs_dict.items():
        y, sig, mask, _c, _u = observation_data(o)
        var = np.asarray(sig) ** 2
        if k == "s":
            var = var + 4e-30 ** 2
        else:
            var = var * np.exp(2 * 0.3)
        want += _gauss_lnl(y, pred[k], var, mask)
    assert float(_lnl(m, _lhs(m), th)) == pytest.approx(want, rel=1e-10, abs=0.0)


def test_two_spectra_each_their_own(csp):
    obs = [_spec("blue", 4000, 6000), _spec("red", 6000, 9000)]
    m = _model(csp, obs, extra_init={"log_jitter_spec_blue": jnp.array([np.log(1e-30)]),
                                     "log_jitter_spec_red": jnp.array([np.log(5e-30)])})
    lh = dict(zip(m.obs_dict, _lhs(m).likelihoods))
    assert lh["blue"].noise_model.nuisance_param_names == ("log_jitter_spec_blue",)
    assert lh["red"].noise_model.nuisance_param_names == ("log_jitter_spec_red",)
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    pred = m.predict(th)
    want = 0.0
    for k, j in (("blue", 1e-30), ("red", 5e-30)):
        y, sig, mask, _c, _u = observation_data(m.obs_dict[k])
        want += _gauss_lnl(y, pred[k], np.asarray(sig) ** 2 + j ** 2, mask)
    assert float(_lnl(m, _lhs(m), th)) == pytest.approx(want, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("old", ["log_jitter", "log_err_scale", "log_f_calib", "log_f_data"])
def test_old_shared_name_refused(csp, old):
    m = _model(csp, [_spec("s", 5000, 8000), _phot()], extra_init={old: jnp.array([-3.0])})
    with pytest.raises(ValueError, match=rf"{old}_spec.*{old}_phot|{old}_phot.*{old}_spec") as e:
        _lhs(m)
    assert "is not a parameter name" in str(e.value) and "maggies" in str(e.value)


@pytest.mark.parametrize("names, match", [
    (["log_jitter_spec"], "ambiguous"),
    (["log_jitter_spec_nope"], "match no observation"),
    (["log_jitter_spec_blue", "log_jitter_spec_blue_x"], "match no observation"),
    (["log_f_data_lines"], "match no observation"),
])
def test_bad_names_refused(csp, names, match):
    obs = [_spec("blue", 4000, 6000), _spec("red", 6000, 9000)]
    m = _model(csp, obs, extra_init={n: jnp.array([-3.0]) for n in names})
    with pytest.raises(ValueError, match=match):
        _lhs(m)


def test_plain_and_own_together_refused(csp):
    m = _model(csp, [_spec("s", 5000, 8000)],
               extra_init={"log_jitter_spec": jnp.array([-3.0]),
                           "log_jitter_spec_s": jnp.array([-3.0])})
    with pytest.raises(ValueError, match="keep one"):
        _lhs(m)


def test_constant_transform_fixes_a_noise_term(csp):
    """A noise term fixed by a constant transform is used (before v1.0.7 fitSED ignored it)."""
    fixed = _model(csp, [_spec("s", 5000, 8000)],
                   transforms={"log_jitter_spec": lambda th: jnp.array([np.log(3e-30)])})
    sampled = _model(csp, [_spec("s", 5000, 8000)],
                     extra_init={"log_jitter_spec": jnp.array([np.log(3e-30)])})
    nm = dict(zip(fixed.obs_dict, _lhs(fixed).likelihoods))["s"].noise_model
    assert nm.nuisance_param_names == () and nm.jitter_key == pytest.approx(np.log(3e-30))
    th_f = {k: jnp.asarray(v) for k, v in fixed.theta_init.items()}
    th_s = {k: jnp.asarray(v) for k, v in sampled.theta_init.items()}
    assert float(_lnl(fixed, _lhs(fixed), th_f)) == pytest.approx(
        float(_lnl(sampled, _lhs(sampled), th_s)), rel=1e-12, abs=0.0)


def test_noise_model_default_keys_unchanged():
    """The low-level DiagonalNoiseModel keeps reading the historical keys by default."""
    nm = DiagonalNoiseModel(use_jitter=True, use_error_scale=True)
    assert nm.nuisance_param_names == ("log_err_scale", "log_jitter")
    out = nm.compute(jnp.ones(3), jnp.ones(3), jnp.ones(3, bool),
                     {"log_jitter": jnp.array([0.0]), "log_err_scale": jnp.array([0.0])})
    np.testing.assert_allclose(np.asarray(out.inv_var), 0.5)


# ------------------------------------------------------------ per-spectrum calibration --
def test_per_spectrum_scaling_and_calib(csp):
    obs = [_spec("blue", 4000, 6000), _spec("red", 6000, 9000), _phot()]
    base = _model(csp, obs)
    th = {k: jnp.asarray(v) for k, v in base.theta_init.items()}
    p0 = base.predict(th)
    m = _model(csp, [_spec("blue", 4000, 6000), _spec("red", 6000, 9000), _phot()],
               extra_init={"spectrum_scaling_blue": jnp.array([0.5]),
                           "spectrum_calib_red": jnp.array([0.1, -0.05])})
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    p1 = m.predict(th)
    np.testing.assert_allclose(np.asarray(p1["blue"]), 0.5 * np.asarray(p0["blue"]), rtol=1e-6)
    P = legendre_design_matrix(m.obs_dict["red"].wavelength, 2)
    np.testing.assert_allclose(np.asarray(p1["red"]),
                               np.asarray(p0["red"]) * (1 + P @ np.array([0.1, -0.05])),
                               rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(p1["p"]), np.asarray(p0["p"]))


@pytest.mark.parametrize("names, match", [
    ({"spectrum_scaling": jnp.array([1.0])}, "ambiguous"),
    ({"spectrum_calib": jnp.zeros(2)}, "ambiguous"),
    ({"spectrum_scaling_green": jnp.array([1.0])}, "match no observation"),
])
def test_spectrum_calibration_names_refused(csp, names, match):
    with pytest.raises(ValueError, match=match):
        _model(csp, [_spec("blue", 4000, 6000), _spec("red", 6000, 9000)], extra_init=names)


def test_single_spectrum_plain_name_unchanged(csp):
    m = _model(csp, [_spec("s", 5000, 8000)], extra_init={"spectrum_scaling": jnp.array([0.7])})
    b = _model(csp, [_spec("s", 5000, 8000)])
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    thb = {k: jnp.asarray(v) for k, v in b.theta_init.items()}
    np.testing.assert_allclose(np.asarray(m.predict(th)["s"]),
                               0.7 * np.asarray(b.predict(thb)["s"]), rtol=1e-6)


# ------------------------------------------------------------ profiled polynomial -------
@pytest.mark.parametrize("tag", ["o1", "o3_holes", "o6_reg"])
def test_profiled_polynomial_equals_prospector(tag):
    z = np.load(REF)
    g = lambda k: z[f"{tag}/{k}"]                                 # noqa: E731
    pc = PolynomialCalibration(chebyshev_design_matrix(g("wave"), g("mask"), int(g("order"))),
                               g("reg"))
    c, resp = pc.solve(jnp.asarray(g("flux")), jnp.asarray(g("spec")),
                       jnp.asarray(1.0 / g("unc") ** 2), jnp.asarray(g("mask")))
    np.testing.assert_allclose(np.asarray(c), g("coeffs"), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(np.asarray(resp), g("response"), rtol=1e-10, atol=0.0)


def test_order_zero_is_the_current_likelihood(csp):
    a = _model(csp, [_spec("s", 5000, 8000, polynomial_order=0), _phot()])
    b = _model(csp, [_spec("s", 5000, 8000), _phot()])
    la, lb = _lhs(a), _lhs(b)
    assert all(getattr(x, "poly_calibration", None) is None for x in la.likelihoods)
    assert la == lb
    th = {k: jnp.asarray(v) for k, v in a.theta_init.items()}
    assert float(_lnl(a, la, th)) == float(_lnl(b, lb, th))


def test_recovers_known_polynomial(csp):
    m = _model(csp, [_spec("s", 5000, 8000, n=300, polynomial_order=3)])
    o = m.obs_dict["s"]
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    mu = np.asarray(m.predict(th)["s"])
    A = chebyshev_design_matrix(o.wavelength, o.mask, 3)
    c_true = np.array([0.05, -0.1, 0.03, 0.02])
    o.flux = jnp.asarray(mu * (1.0 + A @ c_true))
    o.uncertainty = jnp.asarray(0.01 * mu)
    lh = dict(zip(m.obs_dict, _lhs(m).likelihoods))["s"]
    y, sig, mask, _c, _u = observation_data(o)
    nout = lh.noise_model.compute(sig, jnp.asarray(mu), mask, th, data=y)
    c, _ = lh.poly_calibration.solve(y, jnp.asarray(mu), nout.inv_var, mask)
    np.testing.assert_allclose(np.asarray(c), c_true, rtol=0, atol=1e-10)
    # noisy mock: within the least-squares errors
    rng = np.random.default_rng(3)
    yn = mu * (1.0 + A @ c_true) + 0.01 * mu * rng.standard_normal(mu.size)
    c_n, _ = lh.poly_calibration.solve(jnp.asarray(yn), jnp.asarray(mu), nout.inv_var, mask)
    D = mu[:, None] * A
    cov = np.linalg.inv(D.T @ (np.asarray(nout.inv_var)[:, None] * D))
    assert np.all(np.abs(np.asarray(c_n) - c_true) < 4.0 * np.sqrt(np.diag(cov)))


def test_profile_is_the_maximum_over_the_sampled_calibration(csp):
    """ln L(theta, c_hat(theta)) of the profiled route equals max over (spectrum_scaling,
    spectrum_calib) of the sampled route: both span the same degree-M polynomials."""
    M = 3
    prof = _model(csp, [_spec("s", 5000, 8000, n=200, polynomial_order=M)])
    o = prof.obs_dict["s"]
    th = {k: jnp.asarray(v) for k, v in prof.theta_init.items()}
    mu = np.asarray(prof.predict(th)["s"])
    A = chebyshev_design_matrix(o.wavelength, o.mask, M)
    rng = np.random.default_rng(5)
    o.flux = jnp.asarray(mu * (1 + A @ np.array([0.02, 0.05, -0.03, 0.01]))
                         * (1 + 0.02 * rng.standard_normal(mu.size)))
    lnl_prof = float(_lnl(prof, _lhs(prof), th))
    # the sampled route: pred = s mu (1 + P c); its maximum is a linear least-squares problem
    y, sig, mask, _c, _u = observation_data(o)
    y, w = np.asarray(y), np.where(np.asarray(mask), 1.0 / np.asarray(sig) ** 2, 0.0)
    P = np.column_stack([np.ones(mu.size), legendre_design_matrix(o.wavelength, M)])
    D = mu[:, None] * P
    beta = np.linalg.solve(D.T @ (w[:, None] * D), D.T @ (w * y))     # beta = (s, s c_1..c_M)
    samp = _model(csp, [_spec("s", 5000, 8000, n=200)],
                  extra_init={"spectrum_scaling": jnp.array([beta[0]]),
                              "spectrum_calib": jnp.asarray(beta[1:] / beta[0])})
    samp.obs_dict["s"].flux, samp.obs_dict["s"].uncertainty = o.flux, o.uncertainty
    ths = {k: jnp.asarray(v) for k, v in samp.theta_init.items()}
    lnl_samp = float(_lnl(samp, _lhs(samp), ths))
    assert lnl_prof == pytest.approx(lnl_samp, rel=1e-9, abs=0.0)
    ths2 = dict(ths, spectrum_calib=ths["spectrum_calib"] * 1.1)       # any other c is lower
    assert float(_lnl(samp, _lhs(samp), ths2)) < lnl_samp


def _poly_model(csp):
    obs = [_spec("s", 5000, 8000, n=150, polynomial_order=2), _phot()]
    m = _model(csp, obs, extra_init={"log_jitter_spec": jnp.array([np.log(1e-30)])},
               priors={"logmass": Uniform(low=9.0, high=11.0)})
    o = m.obs_dict["s"]
    o.mask = jnp.asarray(np.arange(o.wavelength.size) % 17 != 0)
    o.flux = o.flux.at[::17].set(jnp.nan)             # masked non-finite data
    return m


def test_profiled_gradients_finite_and_match_fd(csp):
    m = _poly_model(csp)
    lh = _lhs(m)
    f = jax.jit(lh.make_lnprobfn(m.obs_dict, m, _NoPrior()))
    th = {k: jnp.asarray(v, dtype=float) for k, v in m.theta_init.items()}
    g = jax.grad(f)(th)
    for k, v in g.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    # full model: the forward model has float32 stages, so central differences scatter at
    # ~1e-3 even without the polynomial (measured: logmass 3.6e-2 at h = 1e-3 with order 0);
    # h = 1e-3 on the dust optical depth, where it is smooth
    k, h = "diffuse_tau_kc", 1e-3
    e = jnp.zeros_like(th[k]).at[0].set(h)
    fd = (float(f(dict(th, **{k: th[k] + e}))) - float(f(dict(th, **{k: th[k] - e})))) / (2 * h)
    assert fd == pytest.approx(float(np.asarray(g[k])[0]), rel=1e-3)
    # jit(value_and_grad) under vmap equals the loop
    vg = jax.jit(jax.value_and_grad(f))
    B = 3
    batch = {k: jnp.stack([v + 0.01 * i for i in range(B)]) for k, v in th.items()}
    vals, _ = jax.vmap(vg)(batch)
    loop = [float(vg({k: v[i] for k, v in batch.items()})[0]) for i in range(B)]
    np.testing.assert_allclose(np.asarray(vals), loop, rtol=1e-7)


def test_profiled_kernel_gradients_exact():
    """float64 kernel (profiled solve + noise model + Gaussian): check_grads in the model
    spectrum and the sampled jitter, forward and reverse mode."""
    from jax.test_util import check_grads
    rng = np.random.default_rng(11)
    n = 60
    wave = np.linspace(5000.0, 8000.0, n)
    mask = np.ones(n, bool); mask[::9] = False
    mu0 = 1.0 + 0.3 * np.sin(wave / 400.0)
    y = mu0 * (1 + 0.05 * np.cos(wave / 900.0)) + 0.02 * rng.standard_normal(n)
    sig = np.full(n, 0.02)
    lh = DiagonalGaussianLikelihood(
        noise_model=DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec"),
        poly_calibration=PolynomialCalibration(chebyshev_design_matrix(wave, mask, 3)))

    def f(mu, lj):
        return lh(jnp.asarray(y), mu, jnp.asarray(sig), jnp.asarray(mask),
                  {"log_jitter_spec": lj})[0]
    check_grads(f, (jnp.asarray(mu0), jnp.array([np.log(0.01)])), order=1, modes=("fwd", "rev"),
                atol=1e-6, rtol=1e-6)


def test_noise_model_weights_enter_the_solve(csp):
    """Prospector weights by 1/sigma^2; CERIDWEN by the noise model's 1/sigma_eff^2."""
    m = _poly_model(csp)
    lh = dict(zip(m.obs_dict, _lhs(m).likelihoods))["s"]
    o = m.obs_dict["s"]
    y, sig, mask, _c, _u = observation_data(o)
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    mu = m.predict(th)["s"]
    pc = lh.poly_calibration
    jit = 5.0 * float(np.median(np.asarray(sig)))          # comparable to the errors
    big = dict(th, log_jitter_spec=jnp.array([np.log(jit)]))
    c_nm, _ = pc.solve(y, mu, lh.noise_model.compute(sig, mu, mask, big, data=y).inv_var, mask)
    var = np.asarray(sig) ** 2 + jit ** 2
    c_ref, _ = pc.solve(y, mu, jnp.asarray(1.0 / var), mask)
    np.testing.assert_allclose(np.asarray(c_nm), np.asarray(c_ref), rtol=1e-10)
    c_raw, _ = pc.solve(y, mu, 1.0 / sig ** 2, mask)
    assert not np.allclose(np.asarray(c_nm), np.asarray(c_raw), rtol=1e-6)


@pytest.mark.parametrize("extra", [{"spectrum_scaling": jnp.array([1.0])},
                                   {"spectrum_calib": jnp.zeros(2)}])
def test_profiled_with_sampled_calibration_refused(csp, extra):
    m = _model(csp, [_spec("s", 5000, 8000, polynomial_order=2)], extra_init=extra)
    with pytest.raises(ValueError, match="degenerate"):
        _lhs(m)


@pytest.mark.parametrize("bad", [-1, 1.5, "2"])
def test_bad_polynomial_order(bad):
    with pytest.raises((ValueError, TypeError), match="polynomial_order"):
        _spec("s", 5000, 8000, polynomial_order=bad)


def test_result_file_and_postprocess(csp, tmp_path):
    from ceridwen.postprocess import PostProcess
    m = _poly_model(csp)
    for name in m.param_names:
        v = float(np.ravel(np.asarray(m.theta_init[name]))[0])
        m.priors.setdefault(name, Uniform(low=v - 0.05, high=v + 0.05))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fitSED(m, output_dir=tmp_path, rng_key=jax.random.PRNGKey(1), verbose=False,
                     sampler_kwargs={"num_live": 16, "num_delete": 4, "num_inner_steps": 3,
                                     "logZ_tol": -0.5})
        out = PostProcess(m, res, n_samples=3, seed=0, uv=False, ionizing=False).run()
    r = read_result_h5(tmp_path / "ceridwen_result.h5")
    cfg = r["obs"]["s"]["likelihood"]
    assert cfg["poly_calibration"]["order"] == 2
    assert cfg["noise_model"]["jitter_key"] == "log_jitter_spec"
    pred = out["prediction"]["spectra"]["s"]
    resp = out["prediction"]["calibration"]["s"]
    assert pred.shape == resp.shape == (3, m.obs_dict["s"].wavelength.size)
    th = {k: jnp.asarray(np.asarray(out["theta"][k])[0]).reshape(np.shape(m.theta_init[k]))
          for k in m.param_names}
    lh = dict(zip(m.obs_dict, _lhs(m).likelihoods))["s"]
    y, sig, mask, _c, _u = observation_data(m.obs_dict["s"])
    mu = m.predict(m.apply_transforms(th) if m.transforms else th)["s"]
    want = lh.poly_calibration.calibrate(y, mu, sig, mask, th, lh.noise_model)
    np.testing.assert_allclose(pred[0], np.asarray(want), rtol=1e-5)
