"""The compiled Gaussian-process likelihood of a spectrum (``ceridwen.likelihood.gp_likelihood``)
and its ``fitSED`` wiring.

1. the kernel equals the independent numpy ``GaussianProcess.log_likelihood`` (plus the
   per-pixel normalisation), tends to the diagonal Gaussian as a -> 0 by the analytic
   eps term, has correct gradients and is jit / vmap safe;
2. ``fitSED`` setup (``fit._likelihood_for``): sampled names, fixed names, the
   ``GaussianProcess`` object, and every refusal -- with a stand-in model, no grid needed;
3. a mock spectrum with GP noise: the lnL profile over (a, l) peaks at the truth, and a
   short nested-sampling fit runs and round-trips the GP configuration (test grid).
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from ceridwen.likelihood import (DiagonalNoiseModel, DiagonalGaussianLikelihood,
                                 MultiObservationLikelihood, lnlike_diag_gaussian)
from ceridwen.likelihood.gp_likelihood import (GP_JITTER, GPGaussianLikelihood, gp_sqdist,
                                               lnlike_gp_gaussian, gp_conditional_mean)
from ceridwen.observation import GaussianProcess, Photometry, Spectrum

jax.config.update("jax_enable_x64", True)

GP_NAMES = ("log_gp_amp_spec", "log_gp_length_spec")


def _data(n=160, seed=0, frac_masked=0.25):
    rng = np.random.default_rng(seed)
    w = np.sort(rng.uniform(5000.0, 5600.0, n))
    mu = 1.0 + 0.2 * np.sin(w / 50.0)
    sig = rng.uniform(0.05, 0.15, n)
    y = mu + sig * rng.normal(size=n)
    mask = rng.uniform(size=n) > frac_masked
    mask[:3] = False                                  # masked pixels at the edge too
    return w, y, mu, sig, mask


def _noise(sig, mu, mask, nm=None, y=None, params=None):
    nm = DiagonalNoiseModel() if nm is None else nm
    return nm.compute(jnp.asarray(sig), jnp.asarray(mu), jnp.asarray(mask), params,
                      data=None if y is None else jnp.asarray(y))


def _reference(w, y, mu, sig_eff, mask, a, l, eps=GP_JITTER):
    """Independent numpy value: GaussianProcess.log_likelihood on the whitened residuals
    plus -sum ln sigma_eff (Spectrum.log_likelihood's normalisation)."""
    r = (y - mu) / sig_eff
    return (GaussianProcess(a, l, jitter=eps).log_likelihood(r, w, mask)
            - 0.5 * np.sum(np.log(sig_eff[mask] ** 2)))


# ============================================================================
# 1. the kernel
# ============================================================================

@pytest.mark.parametrize("a, l", [(0.7, 30.0), (2.5, 4.0), (0.05, 200.0)])
def test_equals_numpy_gaussian_process(a, l):
    w, y, mu, sig, mask = _data()
    out = _noise(sig, mu, mask)
    lnl, aux = lnlike_gp_gaussian(y, mu, out.inv_var, out.log_det, mask, gp_sqdist(w),
                                  np.log(a), np.log(l))
    ref = _reference(w, y, mu, sig, mask, a, l)
    assert float(lnl) == pytest.approx(ref, rel=1e-10, abs=0.0)
    # the pointwise decomposition sums to the total and is 0 on masked pixels
    assert float(jnp.sum(aux.lnl_pointwise)) == pytest.approx(float(lnl), rel=1e-12, abs=0.0)
    assert np.all(np.asarray(aux.lnl_pointwise)[~mask] == 0.0)
    assert int(aux.ndof) == int(mask.sum())
    assert np.all(np.asarray(aux.residuals)[~mask] == 0.0)
    assert np.all(np.asarray(aux.chi)[~mask] == 0.0)
    np.testing.assert_allclose(np.asarray(aux.chi)[mask], ((y - mu) / sig)[mask], rtol=1e-14)


def test_equals_numpy_with_noise_terms():
    """sigma_eff from the noise model (jitter, fractional, error scale) enters as in the
    diagonal likelihood; the numpy reference gets the same sigma_eff."""
    w, y, mu, sig, mask = _data(seed=3)
    nm = DiagonalNoiseModel(use_jitter=True, use_fractional=True, use_error_scale=True,
                            noise_floor=0.02, jitter_key="lj", f_calib_key="lf",
                            err_scale_key="le")
    th = {"lj": jnp.array(-3.0), "lf": jnp.array(-2.5), "le": jnp.array(0.2)}
    var = (sig ** 2 * np.exp(0.4) + (0.02 * mu) ** 2 + (np.exp(-2.5) * mu) ** 2
           + np.exp(-3.0) ** 2)
    lh = GPGaussianLikelihood(nm, gp_sqdist(w), log_amp=np.log(1.3), log_len=np.log(12.0))
    lnl = float(lh(jnp.asarray(y), jnp.asarray(mu), jnp.asarray(sig), jnp.asarray(mask), th)[0])
    assert lnl == pytest.approx(_reference(w, y, mu, np.sqrt(var), mask, 1.3, 12.0),
                                rel=1e-10, abs=0.0)


def test_spectrum_log_likelihood_equals_compiled_path():
    """Spectrum(noise=GaussianProcess).log_likelihood (post-fit, host-side) and the
    likelihood fitSED builds for the same spectrum agree (sky, noise floor and a fixed
    calibration vector included)."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood.likelihood import observation_data
    w, y, mu, sig, mask = _data(seed=5)
    sky = 0.01 * np.cos(w / 30.0)
    cal = 1.0 + 0.01 * (w - 5300.0) / 300.0
    spec = Spectrum(wavelength=w, flux=y + sky, uncertainty=sig, mask=mask, sky=sky,
                    calibration=cal, noise_floor=0.03, name="s",
                    noise=GaussianProcess(0.8, 25.0))
    lh = _likelihood_for(spec)
    assert isinstance(lh, GPGaussianLikelihood)
    yy, ss, mm, calib, ul = observation_data(spec)
    compiled = float(lh(yy, jnp.asarray(mu) * calib, ss, mm)[0])
    assert compiled == pytest.approx(spec.log_likelihood(mu), rel=1e-10, abs=0.0)


def test_small_amplitude_tends_to_diagonal_by_the_eps_term():
    """ln a = -30 (a^2 ~ 1e-26): K = (1 + eps) I on the unmasked pixels, so
    lnL_gp - lnL_diag = eps/(1+eps) * sum r^2 / 2 - n ln(1+eps) / 2 exactly (to rounding);
    with eps = 0 the two are equal."""
    w, y, mu, sig, mask = _data(seed=7)
    out = _noise(sig, mu, mask)
    D = gp_sqdist(w)
    diag = float(lnlike_diag_gaussian(y, mu, out.inv_var, out.log_det, mask)[0])
    r2 = float(np.sum((((y - mu) / sig)[mask]) ** 2))
    n = int(mask.sum())
    for eps in (GP_JITTER, 1e-3):
        gp = float(lnlike_gp_gaussian(y, mu, out.inv_var, out.log_det, mask, D,
                                      -30.0, np.log(30.0), eps)[0])
        expect = 0.5 * r2 * eps / (1.0 + eps) - 0.5 * n * np.log1p(eps)
        assert gp - diag == pytest.approx(expect, rel=1e-6, abs=1e-11)
        assert abs(expect) > 1e-6            # the eps term is resolved, not rounding
    gp0 = float(lnlike_gp_gaussian(y, mu, out.inv_var, out.log_det, mask, D,
                                   -30.0, np.log(30.0), 0.0)[0])
    assert gp0 == pytest.approx(diag, rel=1e-13, abs=0.0)


def test_masked_pixels_contribute_exactly_nothing():
    """Changing data, model or wavelength-independent sigma on masked pixels (even to NaN
    through finite_data) leaves the value unchanged; equals the kernel on the unmasked
    subset alone."""
    w, y, mu, sig, mask = _data(seed=9)
    out = _noise(sig, mu, mask)
    D = gp_sqdist(w)
    lnl = float(lnlike_gp_gaussian(y, mu, out.inv_var, out.log_det, mask, D, 0.1, 3.0)[0])
    for bad in (1e6, np.nan):
        y2 = np.where(mask, y, bad)
        lnl2 = float(lnlike_gp_gaussian(y2, mu, out.inv_var, out.log_det, mask, D, 0.1, 3.0)[0])
        assert lnl2 == lnl
    m = mask
    out_s = _noise(sig[m], mu[m], np.ones(m.sum(), bool))
    sub = float(lnlike_gp_gaussian(y[m], mu[m], out_s.inv_var, out_s.log_det,
                                   np.ones(m.sum(), bool), gp_sqdist(w[m]), 0.1, 3.0)[0])
    assert sub == pytest.approx(lnl, rel=1e-12, abs=0.0)


def _lnl_fn(w, y, sig, mask):
    D = gp_sqdist(w)
    nm = DiagonalNoiseModel(use_jitter=True, jitter_key="lj")

    def f(log_amp, log_len, shift, lj):
        mu = 1.0 + 0.2 * jnp.sin(jnp.asarray(w) / 50.0) + shift
        out = nm.compute(jnp.asarray(sig), mu, jnp.asarray(mask), {"lj": lj})
        return lnlike_gp_gaussian(jnp.asarray(y), mu, out.inv_var, out.log_det,
                                  jnp.asarray(mask), D, log_amp, log_len)[0]
    return f


@pytest.mark.parametrize("log_amp", [0.3, -4.0, -12.0])
def test_gradients_finite_and_correct(log_amp):
    from jax.test_util import check_grads
    w, y, mu, sig, mask = _data(n=60, seed=11)
    y = np.where(mask, y, np.nan)                     # NaN on masked pixels, as real data
    ys, ss = np.where(np.isfinite(y), y, 0.0), sig
    f = _lnl_fn(w, ys, ss, mask)
    args = (jnp.array(log_amp), jnp.array(np.log(20.0)), jnp.array(0.01), jnp.array(-3.0))
    g = jax.grad(f, argnums=(0, 1, 2, 3))(*args)
    assert all(np.isfinite(float(x)) for x in g)
    check_grads(f, args, order=1, modes=("rev",), eps=1e-5, atol=1e-5, rtol=1e-5)


def test_jit_and_vmap_equal_python_loop():
    w, y, mu, sig, mask = _data(n=120, seed=13)
    f = _lnl_fn(w, y, sig, mask)
    rng = np.random.default_rng(0)
    batch = (jnp.asarray(rng.uniform(-2, 1, 7)), jnp.asarray(rng.uniform(1, 4, 7)),
             jnp.asarray(rng.normal(0, 0.01, 7)), jnp.asarray(rng.uniform(-4, -2, 7)))
    loop = np.array([float(f(*(b[i] for b in batch))) for i in range(7)])
    vm = np.asarray(jax.jit(jax.vmap(f))(*batch))
    jt = np.array([float(jax.jit(f)(*(b[i] for b in batch))) for i in range(7)])
    np.testing.assert_allclose(vm, loop, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(jt, loop, rtol=1e-12, atol=0.0)


def test_class_theta_lookup_pytree_and_make_lnprobfn():
    w, y, mu, sig, mask = _data(n=50, seed=17)
    lh = GPGaussianLikelihood(DiagonalNoiseModel(), gp_sqdist(w), log_amp="a", log_len=np.log(9.0))
    assert lh.gp_param_names == ("a",)
    th = {"a": jnp.array([0.2])}
    v = float(lh(jnp.asarray(y), jnp.asarray(mu), jnp.asarray(sig), jnp.asarray(mask), th)[0])
    assert v == pytest.approx(_reference(w, y, mu, sig, mask, np.exp(0.2), 9.0), rel=1e-10, abs=0.0)
    with pytest.raises(KeyError, match="theta\\['a'\\]"):
        lh(jnp.asarray(y), jnp.asarray(mu), jnp.asarray(sig), jnp.asarray(mask), {})
    leaves, tree = jax.tree_util.tree_flatten(lh)
    assert len(leaves) == 1 and leaves[0].shape == (50, 50)
    lh2 = jax.tree_util.tree_unflatten(tree, leaves)
    assert (lh2.log_amp, lh2.log_len, lh2.eps) == (lh.log_amp, lh.log_len, lh.eps)
    assert "GPGaussianLikelihood" in repr(lh) and "n_pix=50" in repr(lh)
    # the single-observation make_lnprobfn and the multi-observation path agree
    obs = SimpleNamespace(flux=jnp.asarray(y), uncertainty=jnp.asarray(sig),
                          mask=jnp.asarray(mask), name="spec")
    model = SimpleNamespace(predict=lambda t: jnp.asarray(mu) * jnp.exp(t["s"][0]))
    prior = SimpleNamespace(log_prob=lambda t: jnp.zeros(()))
    th = {"a": jnp.array([0.2]), "s": jnp.array([0.01])}
    one = float(lh.make_lnprobfn(obs, model, prior)(th))
    multi = MultiObservationLikelihood(keys=("spec",), likelihoods=(lh,))
    model_d = SimpleNamespace(predict=lambda t: {"spec": model.predict(t)})
    many = float(multi.make_lnprobfn({"spec": obs}, model_d, prior)(th))
    assert one == pytest.approx(many, rel=1e-13, abs=0.0)
    for bad in (dict(log_amp=None, log_len=1.0), dict(log_amp=1.0, log_len=np.nan),
                dict(log_amp="", log_len=1.0)):
        with pytest.raises((TypeError, ValueError)):
            GPGaussianLikelihood(DiagonalNoiseModel(), gp_sqdist(w), **bad)
    with pytest.raises(ValueError, match="outlier"):
        GPGaussianLikelihood(DiagonalNoiseModel(f_outlier=0.1), gp_sqdist(w), 0.0, 1.0)


def test_conditional_mean_recovers_the_correlated_component():
    """With a strong, smooth GP component the conditional mean tracks it far better than
    zero (a plotting diagnostic)."""
    rng = np.random.default_rng(2)
    w = np.linspace(5000.0, 5600.0, 300)
    sig = np.full(300, 0.1)
    D = np.asarray(gp_sqdist(w))
    Kc = 2.0 ** 2 * np.exp(-0.5 * D / 40.0 ** 2)
    g = np.linalg.cholesky(Kc + 1e-9 * np.eye(300)) @ rng.normal(size=300) * sig
    y = 1.0 + g + sig * rng.normal(size=300)
    mask = np.ones(300, bool)
    out = _noise(sig, np.ones(300), mask)
    cm = np.asarray(gp_conditional_mean(y, np.ones(300), out.inv_var, mask, D,
                                        np.log(2.0), np.log(40.0)))
    assert np.std(cm - g) < 0.35 * np.std(g)


# ============================================================================
# 2. fitSED setup (stand-in model: no grid, no FSPS)
# ============================================================================

def _spec(name="spec", **kw):
    w, y, mu, sig, mask = _data(n=80, seed=19)
    return Spectrum(wavelength=w, flux=y, uncertainty=sig, mask=mask, name=name, **kw)


def _fake_model(observations, sampled=(), transforms=None):
    init = {n: jnp.array([0.0]) for n in sampled}
    init["logmass"] = jnp.array([10.0])
    return SimpleNamespace(param_names=list(init), transforms=dict(transforms or {}),
                           observations=list(observations), theta_init=init, priors={})


def _lh(obs, model):
    from ceridwen.fit import _likelihood_for
    return _likelihood_for(obs, model.param_names, model=model)


def test_setup_off_by_default_is_the_unchanged_diagonal():
    s = _spec()
    lh = _lh(s, _fake_model([s]))
    assert type(lh) is DiagonalGaussianLikelihood


def test_setup_sampled_names():
    from ceridwen.fit import _describe_likelihood, _likelihood_config
    s = _spec()
    lh = _lh(s, _fake_model([s], sampled=GP_NAMES))
    assert isinstance(lh, GPGaussianLikelihood)
    assert (lh.log_amp, lh.log_len, lh.eps) == ("log_gp_amp_spec", "log_gp_length_spec", GP_JITTER)
    assert lh.n_pix == 80
    assert "a = exp(theta['log_gp_amp_spec']) (sampled)" in _describe_likelihood(s, lh)
    cfg = _likelihood_config(lh)
    assert cfg["class"] == "GPGaussianLikelihood"
    assert cfg["gp"]["log_amp"] == "log_gp_amp_spec" and cfg["gp"]["n_pix"] == 80
    assert set(GP_NAMES) <= set(cfg["noise_model"]["sampled_parameters"])


def test_setup_per_observation_names_and_fixed_by_transform():
    s0, s1 = _spec("s0"), _spec("s1")
    tr = {"log_gp_amp_spec_s1": lambda th: jnp.array([np.log(0.5)]),
          "log_gp_length_spec_s1": lambda th: jnp.array([np.log(40.0)])}
    m = _fake_model([s0, s1], sampled=("log_gp_amp_spec_s0", "log_gp_length_spec_s0"),
                    transforms=tr)
    l0, l1 = _lh(s0, m), _lh(s1, m)
    assert (l0.log_amp, l0.log_len) == ("log_gp_amp_spec_s0", "log_gp_length_spec_s0")
    assert l1.log_amp == pytest.approx(np.log(0.5), rel=1e-12, abs=0.0)
    assert l1.log_len == pytest.approx(np.log(40.0), rel=1e-12, abs=0.0)
    assert l1.gp_param_names == ()


def test_setup_gaussian_process_object_gives_fixed_values():
    s = _spec(noise=GaussianProcess(0.6, 15.0, jitter=1e-5))
    lh = _lh(s, _fake_model([s]))
    assert lh.log_amp == pytest.approx(np.log(0.6), rel=1e-14, abs=0.0)
    assert lh.log_len == pytest.approx(np.log(15.0), rel=1e-14, abs=0.0)
    assert lh.eps == 1e-5 and lh.gp_param_names == ()


@pytest.mark.parametrize("case, exc, match", [
    ("object_and_names", ValueError, "either fixed by the GaussianProcess object"),
    ("only_amp", ValueError, "not its partner"),
    ("ambiguous", ValueError, "ambiguous"),
    ("photometry_name", ValueError, "match no observation"),
    ("unknown_obs", ValueError, "match no observation"),
    ("zero_amp_object", ValueError, "must be > 0"),
    ("not_a_gp", TypeError, "takes a ceridwen.observation.GaussianProcess"),
    ("gp_on_photometry", NotImplementedError, "only a Spectrum"),
    ("derived_transform", NotImplementedError, "constant scalar"),
])
def test_setup_refusals_names(case, exc, match):
    kw = {}
    sampled = GP_NAMES
    obs = None
    tr = None
    if case == "object_and_names":
        kw = dict(noise=GaussianProcess(1.0, 10.0))
    elif case == "only_amp":
        sampled = GP_NAMES[:1]
    elif case == "ambiguous":
        obs = [_spec("a"), _spec("b")]
    elif case == "photometry_name":
        sampled = GP_NAMES + ("log_gp_amp_phot",)
    elif case == "unknown_obs":
        sampled = ("log_gp_amp_spec_nope", "log_gp_length_spec_nope")
    elif case == "zero_amp_object":
        kw, sampled = dict(noise=GaussianProcess(0.0, 10.0)), ()
    elif case == "not_a_gp":
        kw, sampled = dict(noise=DiagonalNoiseModel()), ()
    elif case == "derived_transform":
        sampled = GP_NAMES[:1]
        tr = {"log_gp_length_spec": lambda th: th["logmass"] - 7.0}
    if case == "gp_on_photometry":
        p = Photometry(filters=["sdss_g0"], flux=np.ones(1), uncertainty=np.ones(1),
                       name="phot", noise=GaussianProcess(1.0, 10.0))
        s = _spec()
        with pytest.raises(exc, match=match):
            _lh(p, _fake_model([s, p]))
        return
    s = _spec(**kw)
    obs = obs or [s]
    m = _fake_model(obs, sampled=sampled, transforms=tr)
    with pytest.raises(exc, match=match):
        _lh(obs[0], m)


@pytest.mark.parametrize("case, match", [
    ("outlier", "outlier mixture"),
    ("upper_limit", "upper limits"),
    ("marginalize_elines", "marginalize_elines"),
    ("logify", "logify_spectrum"),
    ("polynomial", "profiled calibration polynomial"),
])
@pytest.mark.parametrize("route", ["names", "object"])
def test_setup_refused_combinations(case, match, route):
    """Decision 7: each combination raises at setup, before any likelihood call, for both
    ways of switching the GP on."""
    from ceridwen.broadening import Instrument
    kw = {} if route == "names" else dict(noise=GaussianProcess(1.0, 10.0))
    sampled = GP_NAMES if route == "names" else ()
    tr = None
    if case == "outlier":
        tr = {"f_outlier_spec": lambda th: jnp.array([0.1])}
    elif case == "marginalize_elines":
        if route == "object":        # refused already at Spectrum construction
            with pytest.raises(ValueError, match="GaussianProcess noise model is not supported"):
                _spec(marginalize_elines=True, instrument=Instrument.R_fwhm(2000.0), **kw)
            return
        kw.update(marginalize_elines=True, instrument=Instrument.R_fwhm(2000.0))
    elif case == "logify":
        kw.update(logify_spectrum=True)
    elif case == "polynomial":
        kw.update(polynomial_order=2)
    s = _spec(**kw)
    if case == "upper_limit":
        s.upper_limit = jnp.zeros(s.flux.shape, bool).at[5].set(True)
    m = _fake_model([s], sampled=sampled, transforms=tr)
    with pytest.raises(NotImplementedError, match=match):
        _lh(s, m)


def test_setup_warns_above_the_pixel_threshold():
    from ceridwen.fit import GP_WARN_NPIX
    n = GP_WARN_NPIX + 1
    w = np.linspace(5000.0, 9000.0, n)
    s = Spectrum(wavelength=w, flux=np.ones(n), uncertainty=np.ones(n), name="big")
    with pytest.warns(UserWarning, match="dense"):
        _lh(s, _fake_model([s], sampled=GP_NAMES))


# ============================================================================
# 3. mock spectrum with GP noise (test SSP grid)
# ============================================================================

A_TRUE, L_TRUE, LOGM_TRUE, ZRED = 1.0, 30.0, 10.5, 0.1


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    from _gridfixture import find_test_grid
    path = find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (tests/_gridfixture.py)")
    from ceridwen import CSPBasis, SSPData, SedModel, Cosmology
    from ceridwen.sampler.priors import Uniform
    ssp = SSPData.load(str(path))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 10.0, 4), zh_const=True,
                   sfh_interp="step", add_dust=False, add_diffuse_dust=False, add_neb=False,
                   add_igm=False, verbose=False, cosmo=Cosmology.planck18())
    n = 600
    w = np.linspace(5000.0, 6200.0, n)
    fixed = {"sfh": lambda th: jnp.ones(4), "logzsol": lambda th: jnp.array([-0.3])}
    priors = {"logmass": Uniform(low=10.0, high=11.0),
              "log_gp_amp_spec": Uniform(low=-3.0, high=1.5),
              "log_gp_length_spec": Uniform(low=1.0, high=5.5)}
    init = {"logmass": jnp.array([LOGM_TRUE]),
            "log_gp_amp_spec": jnp.array([np.log(A_TRUE)]),
            "log_gp_length_spec": jnp.array([np.log(L_TRUE)])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp = Spectrum(wavelength=w, flux=np.ones(n), uncertainty=np.ones(n), name="spec")
        m0 = SedModel(csp, [tmp], priors=priors, transforms=fixed, free_param_init=init,
                      zred=ZRED)
    truth = {k: jnp.asarray(v) for k, v in init.items()}
    mu = np.asarray(m0.predict(truth)["spec"], dtype=float)
    assert np.all(np.isfinite(mu)) and np.all(mu > 0)
    sig = mu / 30.0
    rng = np.random.default_rng(42)
    D = np.asarray(gp_sqdist(w))
    C = A_TRUE ** 2 * np.exp(-0.5 * D / L_TRUE ** 2) + (1.0 + GP_JITTER) * np.eye(n)
    y = mu + sig * (np.linalg.cholesky(C) @ rng.normal(size=n))
    mask = np.ones(n, bool)
    mask[250:270] = False
    spec = Spectrum(wavelength=w, flux=y, uncertainty=sig, mask=mask, name="spec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, [spec], priors=priors, transforms=fixed, free_param_init=init,
                         zred=ZRED)
    return SimpleNamespace(model=model, truth=truth, out=tmp_path_factory.mktemp("gp"))


def test_mock_profile_peaks_at_truth(mock):
    from ceridwen.optimize import build_lnprob
    f = jax.jit(build_lnprob(mock.model))
    la = np.log(A_TRUE) + np.linspace(-1.0, 1.0, 11)
    ll = np.log(L_TRUE) + np.linspace(-1.0, 1.0, 11)
    th = dict(mock.truth)
    grid = np.array([[float(f({**th, "log_gp_amp_spec": jnp.array([a]),
                                "log_gp_length_spec": jnp.array([l])}))
                      for l in ll] for a in la])
    assert np.all(np.isfinite(grid))
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    assert abs(la[i] - np.log(A_TRUE)) <= 0.4 and abs(ll[j] - np.log(L_TRUE)) <= 0.4
    # far from the truth the likelihood is clearly lower (the GP is identified)
    assert grid.max() - grid[0, 0] > 10.0 and grid.max() - grid[-1, -1] > 10.0


def test_mock_nested_fit_and_result_roundtrip(mock):
    from ceridwen import fitSED
    from ceridwen.fit import read_result_h5
    res = fitSED(mock.model, output_dir=mock.out, rng_key=jax.random.PRNGKey(3),
                 verbose=False, mfrac=False,
                 sampler_kwargs={"num_live": 40, "num_delete": 8, "num_inner_steps": 8,
                                 "logZ_tol": -0.5})
    assert np.isfinite(float(res.log_evidence))
    ll = np.asarray(res.log_likelihoods)
    assert np.all(np.isfinite(ll))
    r = read_result_h5(mock.out / "ceridwen_result.h5")
    cfg = r["obs"]["spec"]["likelihood"]
    assert cfg["class"] == "GPGaussianLikelihood"
    assert cfg["gp"] == {"kernel": "squared_exponential", "log_amp": "log_gp_amp_spec",
                         "log_len": "log_gp_length_spec", "eps": GP_JITTER, "n_pix": 600,
                         "wavelength": "observed frame, Angstrom",
                         "amplitude_units": "sigma_eff",
                         "sampled_parameters": ["log_gp_amp_spec", "log_gp_length_spec"]}
    assert "log_gp_amp_spec" in r["model"]["param_names"]
    log = (mock.out / "ceridwen_result.log").read_text()
    assert "GP (squared exponential, 600 pixels)" in log
