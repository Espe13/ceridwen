"""Analytic marginalisation of the calibration polynomial
(``Spectrum(polynomial_order=M, polynomial_mode="marginalize", polynomial_prior_sigma=s)``).

* the closed form equals the dense Gaussian ``N(y - mu; D c0, C + D Lambda D^T)`` (numpy
  ``slogdet`` / ``solve``) with a mask, and a Gauss-Hermite integral over c for k = 1;
* s -> 0 is the uncalibrated diagonal likelihood; the conditional mean is the profiled
  solution with ``reg = 1/s``; gradients, jit and vmap;
* every refused combination raises at construction or at setup;
* a mock spectrum with a known order-3 calibration: a short nested-sampling fit recovers the
  physical parameters and the polynomial, consistently with the sampled route
  (``spectrum_scaling`` + ``spectrum_calib``); the result file records the mode.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ceridwen.broadening import Instrument
from ceridwen.likelihood.likelihood import (DiagonalGaussianLikelihood, lnlike_diag_gaussian,
                                            MultiObservationLikelihood, observation_data)
from ceridwen.likelihood.noise_model import DiagonalNoiseModel
from ceridwen.likelihood.poly_calibration import PolynomialCalibration, chebyshev_design_matrix
from ceridwen.likelihood.poly_marginal import (PolyMarginalGaussianLikelihood,
                                               PolynomialMarginal, poly_marginal_loglike,
                                               poly_marginal_posterior)
from ceridwen.observation import Photometry, Spectrum


# ------------------------------------------------------------------ small problems -----
def _problem(seed, n=80, k=4, sig_prior=None, masked=True):
    rng = np.random.default_rng(seed)
    wave = np.linspace(4000.0, 7000.0, n)
    mask = (rng.random(n) > 0.2) if masked else np.ones(n, bool)
    A = chebyshev_design_matrix(wave, mask, k - 1)
    mu = 1.0 + 0.3 * np.sin(wave / 500.0)
    sig = 0.03 + 0.02 * rng.random(n)
    s = np.asarray(sig_prior if sig_prior is not None else 0.02 + 0.2 * rng.random(k))
    y = mu * (1.0 + A @ (s * rng.standard_normal(k))) + sig * rng.standard_normal(n)
    y = np.where(mask, y, np.nan)                      # masked data may be anything
    return dict(wave=wave, mask=mask, A=A, mu=mu, sig=sig, s=s, y=y)


def _kernel(p, **kw):
    y = jnp.asarray(np.where(p["mask"], p["y"], 0.0))
    iv = jnp.asarray(1.0 / p["sig"] ** 2)
    ld = jnp.asarray(0.5 * np.log(2 * np.pi * p["sig"] ** 2))
    return poly_marginal_loglike(y, jnp.asarray(p["mu"]), iv, ld, jnp.asarray(p["mask"]),
                                 p["A"], p["s"], **kw)


def _dense(p):
    """ln N(r; 0, C + D Lambda D^T) on the unmasked pixels, and the posterior (mean, cov)."""
    m = p["mask"]
    D = (p["mu"][:, None] * p["A"])[m]
    C = np.diag(p["sig"][m] ** 2)
    Lam = np.diag(p["s"] ** 2)
    r = (p["y"] - p["mu"])[m]
    S = C + D @ Lam @ D.T
    _sgn, logdet = np.linalg.slogdet(2.0 * np.pi * S)
    lnl = -0.5 * r @ np.linalg.solve(S, r) - 0.5 * logdet
    # posterior of c: Lambda D^T S^-1 r, Lambda - Lambda D^T S^-1 D Lambda
    mean = Lam @ D.T @ np.linalg.solve(S, r)
    cov = Lam - Lam @ D.T @ np.linalg.solve(S, D @ Lam)
    return lnl, mean, cov


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_form_equals_dense_gaussian(seed):
    p = _problem(seed)
    lnl, aux, mean, cov = _kernel(p, return_posterior=True)
    want, m_want, c_want = _dense(p)
    assert float(lnl) == pytest.approx(want, rel=1e-10, abs=0.0)
    np.testing.assert_allclose(np.asarray(mean), m_want, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cov), c_want, rtol=1e-7, atol=1e-14)
    # the pointwise decomposition documented in poly_marginal_loglike
    s = p["s"]
    m = p["mask"]
    D = (p["mu"][:, None] * p["A"])[m]
    G = D.T @ np.diag(p["sig"][m] ** -2) @ D
    P = np.diag(s) @ G @ np.diag(s) + np.eye(len(s))
    extra = -0.5 * np.sum(np.asarray(mean) ** 2 / s ** 2) - 0.5 * np.linalg.slogdet(P)[1]
    assert float(np.sum(np.asarray(aux.lnl_pointwise))) + extra == pytest.approx(
        float(lnl), rel=1e-10)
    assert np.all(np.asarray(aux.lnl_pointwise)[~m] == 0.0)
    assert int(aux.ndof) == int(m.sum())


def test_masked_pixels_contribute_nothing():
    p = _problem(3)
    q = dict(p)
    q["y"] = np.where(p["mask"], p["y"], 1e30)          # garbage under the mask
    q["mu"] = np.where(p["mask"], p["mu"], 7.0)
    assert float(_kernel(p)[0]) == float(_kernel(q)[0])
    m = p["mask"]
    r = dict(p, y=p["y"][m], mu=p["mu"][m], sig=p["sig"][m], A=p["A"][m], mask=np.ones(m.sum(), bool))
    assert float(_kernel(r)[0]) == pytest.approx(float(_kernel(p)[0]), rel=1e-12)


def test_heterogeneous_and_pinned_prior_widths():
    p = _problem(4, sig_prior=[0.0, 1e-3, 0.5, 30.0])   # pinned, tight, loose, very loose
    want, m_want, _ = _dense(p)
    lnl, _aux, mean, _cov = _kernel(p, return_posterior=True)
    assert float(lnl) == pytest.approx(want, rel=1e-10)
    assert float(mean[0]) == 0.0
    np.testing.assert_allclose(np.asarray(mean), m_want, rtol=1e-7, atol=1e-12)


def test_k1_equals_gauss_hermite_integral():
    """One coefficient (the grey level): the closed form equals the integral over c of the
    diagonal likelihood times the Gaussian prior, by Gauss-Hermite quadrature."""
    rng = np.random.default_rng(5)
    n = 40
    mu = 1.0 + 0.2 * rng.random(n)
    sig = np.full(n, 0.2)
    s = 0.15
    y = mu * (1.0 + 0.1) + sig * rng.standard_normal(n)
    mask = np.ones(n, bool)
    A = np.ones((n, 1))

    def lnl_c(c):
        r = y - mu * (1.0 + c)
        return np.sum(-0.5 * r ** 2 / sig ** 2 - 0.5 * np.log(2 * np.pi * sig ** 2))

    def lnpost(c):
        return lnl_c(c) - 0.5 * c ** 2 / s ** 2 - 0.5 * np.log(2 * np.pi * s ** 2)

    # centre and width from a numerical quadratic fit (independent of the closed form)
    from scipy.optimize import minimize_scalar
    c0 = minimize_scalar(lambda c: -lnpost(c), bracket=(-1, 1)).x
    h = 1e-3
    curv = -(lnpost(c0 + h) - 2 * lnpost(c0) + lnpost(c0 - h)) / h ** 2
    w = 1.0 / np.sqrt(curv)
    x, wt = np.polynomial.hermite.hermgauss(40)
    vals = np.array([lnpost(c0 + np.sqrt(2) * w * xi) + xi ** 2 for xi in x])
    ref = np.log(np.sqrt(2) * w) + np.max(vals) + np.log(np.sum(wt * np.exp(vals - np.max(vals))))
    got = poly_marginal_loglike(jnp.asarray(y), jnp.asarray(mu), jnp.asarray(1 / sig ** 2),
                                jnp.asarray(0.5 * np.log(2 * np.pi * sig ** 2)),
                                jnp.asarray(mask), A, np.array([s]))[0]
    assert float(got) == pytest.approx(ref, rel=1e-8, abs=1e-8)


def test_zero_width_is_the_uncalibrated_likelihood():
    p = _problem(6)
    y = jnp.asarray(np.where(p["mask"], p["y"], 0.0))
    iv = jnp.asarray(1.0 / p["sig"] ** 2)
    ld = jnp.asarray(0.5 * np.log(2 * np.pi * p["sig"] ** 2))
    want = float(lnlike_diag_gaussian(y, jnp.asarray(p["mu"]), iv, ld, jnp.asarray(p["mask"]))[0])
    assert float(_kernel(dict(p, s=np.zeros(4)))[0]) == pytest.approx(want, rel=1e-14)
    assert float(_kernel(dict(p, s=np.full(4, 1e-9)))[0]) == pytest.approx(want, rel=1e-10)


def test_conditional_mean_is_the_profiled_solution():
    p = _problem(7)
    y = jnp.asarray(np.where(p["mask"], p["y"], 0.0))
    iv = jnp.asarray(1.0 / p["sig"] ** 2)
    mean, _cov = poly_marginal_posterior(y, jnp.asarray(p["mu"]), iv, jnp.asarray(p["mask"]),
                                         p["A"], p["s"])
    c_prof, _resp = PolynomialCalibration(p["A"], regularization=1.0 / p["s"]).solve(
        y, jnp.asarray(p["mu"]), iv, jnp.asarray(p["mask"]))
    np.testing.assert_allclose(np.asarray(mean), np.asarray(c_prof), rtol=1e-9, atol=1e-13)


def _lh_with_jitter(p):
    return PolyMarginalGaussianLikelihood(
        noise_model=DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec"),
        poly_marginal=PolynomialMarginal(p["A"], p["s"]))


def test_gradients_jit_vmap():
    from jax.test_util import check_grads
    p = _problem(8)
    lh = _lh_with_jitter(p)
    y = jnp.asarray(np.where(p["mask"], p["y"], np.nan))   # NaN under the mask
    sig, mask = jnp.asarray(p["sig"]), jnp.asarray(p["mask"])

    def f(mu, lj):
        return lh(jnp.where(mask, y, 0.0), mu, sig, mask, {"log_jitter_spec": lj})[0]

    args = (jnp.asarray(p["mu"]), jnp.array([np.log(0.01)]))
    g = jax.grad(f, argnums=(0, 1))(*args)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in g)
    check_grads(f, args, order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)
    fj = jax.jit(f)
    B = 4
    mus = jnp.stack([args[0] * (1 + 0.01 * i) for i in range(B)])
    ljs = jnp.stack([args[1] + 0.1 * i for i in range(B)])
    v = jax.vmap(fj)(mus, ljs)
    loop = [float(f(mus[i], ljs[i])) for i in range(B)]
    np.testing.assert_allclose(np.asarray(v), loop, rtol=1e-12)


def test_pytree_repr_and_multi_observation():
    p = _problem(9)
    lh = _lh_with_jitter(p)
    leaves, tree = jax.tree_util.tree_flatten(lh)
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves) == lh
    assert "PolyMarginalGaussianLikelihood" in repr(lh) and "prior_sigma" in repr(lh)
    assert PolynomialMarginal(p["A"], p["s"]) == PolynomialMarginal(p["A"].copy(), p["s"].copy())
    y = jnp.asarray(np.where(p["mask"], p["y"], 0.0))
    th = {"log_jitter_spec": jnp.array([np.log(0.005)])}
    one = lh(y, jnp.asarray(p["mu"]), jnp.asarray(p["sig"]), jnp.asarray(p["mask"]), th)[0]
    multi = MultiObservationLikelihood(keys=("s",), likelihoods=(lh,))
    tot, aux = multi({"s": y}, {"s": jnp.asarray(p["mu"])}, {"s": jnp.asarray(p["sig"])},
                     {"s": jnp.asarray(p["mask"])}, th)
    assert float(tot) == float(one) and set(aux) == {"s"}
    with pytest.raises(NotImplementedError, match="outlier"):
        PolyMarginalGaussianLikelihood(noise_model=DiagonalNoiseModel(f_outlier="f_outlier_spec"),
                                       poly_marginal=PolynomialMarginal(p["A"], p["s"]))


# --------------------------------------------------------------- construction errors -----
def _spec(**kw):
    w = np.linspace(5000.0, 8000.0, 60)
    kw.setdefault("instrument", Instrument.sigma_kms(150.0))
    return Spectrum(wavelength=w, flux=np.ones(60), uncertainty=np.full(60, 0.1), name="s", **kw)


MARG = dict(polynomial_order=3, polynomial_mode="marginalize")


@pytest.mark.parametrize("kw, match", [
    (dict(MARG), "needs polynomial_prior_sigma"),
    (dict(MARG, polynomial_prior_sigma=np.inf), "flat prior"),
    (dict(MARG, polynomial_prior_sigma=[0.1, 0.1, np.inf, 0.1]), "flat prior"),
    (dict(MARG, polynomial_prior_sigma=-0.1), "finite and >= 0"),
    (dict(MARG, polynomial_prior_sigma=np.nan), "finite and >= 0"),
    (dict(MARG, polynomial_prior_sigma=[0.1, 0.1]), "one per coefficient"),
    (dict(MARG, polynomial_prior_sigma=0.1, polynomial_regularization=10.0),
     "polynomial_prior_sigma = 1 / polynomial_regularization"),
    (dict(MARG, polynomial_prior_sigma=0.1, logify_spectrum=True), "logify_spectrum"),
    (dict(MARG, polynomial_prior_sigma=0.1, noise=object()), "GaussianProcess"),
    (dict(MARG, polynomial_prior_sigma=0.1, marginalize_elines=True), "bilinear"),
    (dict(polynomial_order=0, polynomial_mode="marginalize", polynomial_prior_sigma=0.1),
     "polynomial_order >= 1"),
    (dict(polynomial_order=3, polynomial_prior_sigma=0.1), "only acts with"),
    (dict(polynomial_order=3, polynomial_mode="marginalise"), "'profile' or 'marginalize'"),
])
def test_construction_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        _spec(**kw)


def test_construction_accepts_and_defaults():
    s = _spec(**MARG, polynomial_prior_sigma=[0.0, 0.1, 0.1, 0.05])
    assert s.polynomial_mode == "marginalize"
    np.testing.assert_array_equal(s.polynomial_prior_sigma, [0.0, 0.1, 0.1, 0.05])
    d = _spec(polynomial_order=2)
    assert d.polynomial_mode == "profile" and d.polynomial_prior_sigma is None
    assert PolynomialCalibration.for_spectrum(s) is None
    assert PolynomialMarginal.for_spectrum(d) is None
    assert PolynomialMarginal.for_spectrum(s).order == 3


# ------------------------------------------------------------------ model-level tests -----
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
ZRED = 0.3
TRUE = {"logmass": 10.0, "diffuse_tau_kc": 0.6}
C_TRUE = np.array([0.03, 0.06, -0.04, 0.025])          # injected Chebyshev T_0..T_3
WAVE = np.linspace(4500.0, 8500.0, 200)


@pytest.fixture(scope="module")
def csp():
    from _gridfixture import require_test_grid
    from ceridwen import SSPData, CSPBasis, Cosmology
    ssp = SSPData.load(str(require_test_grid()))
    return CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 9.0, 4), zh_const=True,
                    add_dust=True, add_diffuse_dust=True, add_neb=False, add_igm=False,
                    verbose=False, cosmo=Cosmology.planck18())


def _model(csp, spec_kw, init=None, priors=None, transforms=None, data=None):
    from ceridwen import SedModel
    from ceridwen.sampler.priors import Uniform
    sp = Spectrum(wavelength=WAVE, flux=np.full(WAVE.size, 1e-29),
                  uncertainty=np.full(WAVE.size, 1e-30), instrument=Instrument.sigma_kms(150.0),
                  name="s", **spec_kw)
    ph = Photometry(filters=FILTERS, flux=[1e-9] * 4, uncertainty=[1e-10] * 4, name="p")
    fixed = {"sfh": jnp.ones(4), "logzsol": jnp.array([-0.3]), "tau_pow": jnp.array([1.0]),
             "alpha_pow": jnp.array([-1.0]), "diffuse_dust_index": jnp.array([0.0])}
    tr = {k: (lambda v: (lambda th: v))(v) for k, v in fixed.items()}
    tr.update(transforms or {})
    fi = {k: jnp.array([v]) for k, v in TRUE.items()}
    fi.update(init or {})
    pr = {"logmass": Uniform(low=9.5, high=10.5), "diffuse_tau_kc": Uniform(low=0.0, high=1.5)}
    pr.update(priors or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SedModel(csp, [sp, ph], priors=pr, free_param_init=fi, transforms=tr, zred=ZRED)
    if data is not None:
        for o in m.observations:
            o.flux, o.uncertainty = (jnp.asarray(a) for a in data[o.name])
    return m


def _mock(csp):
    """Data of the true model with the injected calibration on the spectrum; S/N 30 on the
    spectrum, 3 % on the photometry, seeded noise."""
    m = _model(csp, {})
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    pred = m.predict(m.apply_transforms(th))
    rng = np.random.default_rng(42)
    A = chebyshev_design_matrix(WAVE, np.ones(WAVE.size, bool), 3)
    mu = np.asarray(pred["s"]) * (1.0 + A @ C_TRUE)
    e = np.abs(mu) / 30.0
    phot = np.asarray(pred["p"])
    ep = 0.03 * phot
    return {"s": (mu + e * rng.standard_normal(mu.size), e),
            "p": (phot + ep * rng.standard_normal(phot.size), ep)}


def _lhs(m):
    from ceridwen.fit import _likelihood_for
    return {o.name: _likelihood_for(o, m.param_names, model=m) for o in m.observations}


MARG_KW = dict(polynomial_order=3, polynomial_mode="marginalize", polynomial_prior_sigma=0.1)


def test_setup_builds_marginal_likelihood(csp):
    m = _model(csp, MARG_KW)
    lh = _lhs(m)
    assert isinstance(lh["s"], PolyMarginalGaussianLikelihood)
    assert isinstance(lh["p"], DiagonalGaussianLikelihood)
    assert lh["s"].poly_marginal.order == 3
    # default: the profiled route is unchanged
    lp = _lhs(_model(csp, {"polynomial_order": 3}))["s"]
    assert type(lp) is DiagonalGaussianLikelihood and lp.poly_calibration.order == 3


@pytest.mark.parametrize("extra, match", [
    (dict(transforms={"f_outlier_spec": lambda th: jnp.array([0.1])}), "outlier"),
    (dict(init={"spectrum_calib": jnp.zeros(2)}), "second polynomial"),
    (dict(init={"spectrum_scaling": jnp.array([1.0])}), "pin T_0"),
])
def test_setup_refusals(csp, extra, match):
    m = _model(csp, MARG_KW, **extra)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _lhs(m)


def test_upper_limits_refused(csp):
    m = _model(csp, MARG_KW)
    m.obs_dict["s"].upper_limit = jnp.zeros(WAVE.size, bool).at[3].set(True)
    with pytest.raises(NotImplementedError, match="upper limits"):
        _lhs(m)


def test_scaling_allowed_with_pinned_level(csp):
    m = _model(csp, dict(MARG_KW, polynomial_prior_sigma=[0.0, 0.1, 0.1, 0.1]),
               init={"spectrum_scaling": jnp.array([1.0])})
    assert isinstance(_lhs(m)["s"], PolyMarginalGaussianLikelihood)


def test_lnprob_equals_kernel(csp):
    data = _mock(csp)
    m = _model(csp, MARG_KW, data=data)
    lh = _lhs(m)
    multi = MultiObservationLikelihood(keys=tuple(lh), likelihoods=tuple(lh.values()))

    class _NoPrior:
        @staticmethod
        def log_prob(theta):
            return 0.0
    f = jax.jit(multi.make_lnprobfn(m.obs_dict, m, _NoPrior()))
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    pred = m.predict(m.apply_transforms(th))
    want = 0.0
    for k, o in m.obs_dict.items():
        y, sig, mask, _c, _u = observation_data(o)
        want += float(lh[k](y, pred[k], sig, mask, th)[0])
    y, sig, mask, _c, _u = observation_data(m.obs_dict["s"])
    direct = _dense(dict(mask=np.asarray(mask), mu=np.asarray(pred["s"]),
                         A=chebyshev_design_matrix(WAVE, np.asarray(mask), 3),
                         sig=np.asarray(sig), s=np.full(4, 0.1), y=np.asarray(y)))[0]
    assert float(lh["s"](y, pred["s"], sig, mask, th)[0]) == pytest.approx(direct, rel=1e-10)
    # jitted vs eager forward model: its float32 stages differ in the last bits (1.6e-10 rel)
    assert float(f(th)) == pytest.approx(want, rel=1e-8)
    g = jax.grad(f)(th)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())


def _weighted(res):
    w = np.exp(np.asarray(res.log_weights) - np.max(res.log_weights))
    return w / w.sum()


def _stats(res, k):
    w = _weighted(res)
    x = np.asarray(res.samples[k]).reshape(len(w), -1)[:, 0]
    mean = np.sum(w * x)
    return mean, np.sqrt(np.sum(w * (x - mean) ** 2))


@pytest.fixture(scope="module")
def fits(csp, tmp_path_factory):
    """One short nested-sampling fit per route on the same mock."""
    from ceridwen import fitSED
    from ceridwen.sampler.priors import Uniform
    data = _mock(csp)
    out = {}
    kw = {"num_live": 120, "num_delete": 20, "num_inner_steps": 10}
    m = _model(csp, MARG_KW, data=data)
    d = tmp_path_factory.mktemp("marg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["marg"] = (m, fitSED(m, output_dir=d, rng_key=jax.random.PRNGKey(0), verbose=False,
                                 sampler_kwargs=kw), d)
    ms = _model(csp, {}, data=data,
                init={"spectrum_scaling": jnp.array([1.0]), "spectrum_calib": jnp.zeros(3)},
                priors={"spectrum_scaling": Uniform(low=0.7, high=1.3),
                        # scalar bounds broadcast over the vector (per-element bounds cannot be
                        # serialised into the result file: FINDINGS G2-001)
                        "spectrum_calib": Uniform(low=-0.3, high=0.3)})
    d = tmp_path_factory.mktemp("samp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["samp"] = (ms, fitSED(ms, output_dir=d, rng_key=jax.random.PRNGKey(0),
                                  verbose=False, sampler_kwargs=kw), d)
    return out


def test_mock_recovery_and_sampled_route(fits):
    _m, res, _d = fits["marg"]
    _ms, res_s, _d = fits["samp"]
    for k, v in TRUE.items():
        mean, sd = _stats(res, k)
        assert abs(mean - v) < 4 * sd + 1e-3, (k, mean, sd, v)
        mean_s, sd_s = _stats(res_s, k)
        # same data, same polynomial space: posteriors agree within the sampling error
        assert abs(mean - mean_s) < 3 * np.hypot(sd, sd_s), (k, mean, sd, mean_s, sd_s)
        assert sd == pytest.approx(sd_s, rel=0.5), (k, sd, sd_s)


def test_postprocess_conditional_polynomial_and_result_file(fits):
    from ceridwen import read_result_h5
    from ceridwen.postprocess import PostProcess
    m, res, d = fits["marg"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = PostProcess(m, res, n_samples=40, seed=0, uv=False, ionizing=False).run()
    blk = out["extras"]["calibration"]["s"]
    assert blk["mean"].shape == blk["sd"].shape == blk["draws"].shape == (40, 4)
    assert blk["cov"].shape == (40, 4, 4)
    # the conditional-mean polynomial matches the injected one within its posterior spread
    A = chebyshev_design_matrix(WAVE, np.ones(WAVE.size, bool), 3)
    resp = out["prediction"]["calibration"]["s"]
    np.testing.assert_allclose(resp, 1.0 + blk["mean"] @ A.T, rtol=1e-10)
    true_resp = 1.0 + A @ C_TRUE
    spread = np.std(1.0 + blk["draws"] @ A.T, axis=0)
    assert np.max(np.abs(np.median(resp, axis=0) - true_resp) / spread) < 4.0
    # the grey level T_0 trades against logmass within the photometric level (3 % in 4 bands;
    # measured: a ~1.8 % level offset), so the shape is checked with the level divided out
    ratio = np.median(resp, axis=0) / true_resp
    assert np.max(np.abs(ratio / np.mean(ratio) - 1.0)) < 0.01
    # the prediction carries the response, recomputed from the likelihood at draw 0
    th = {k: jnp.asarray(np.asarray(out["theta"][k])[0]).reshape(np.shape(m.theta_init[k]))
          for k in m.param_names}
    lh = _lhs(m)["s"]
    y, sig, mask, _c, _u = observation_data(m.obs_dict["s"])
    mu = m.predict(m.apply_transforms(th))["s"]
    mean, _cov, want = lh.conditional(y, mu, sig, mask, th)
    np.testing.assert_allclose(blk["mean"][0], np.asarray(mean), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(out["prediction"]["spectra"]["s"][0], np.asarray(mu * want),
                               rtol=1e-6)
    # result file: mode, order and prior widths
    r = read_result_h5(d / "ceridwen_result.h5")
    cfg = r["obs"]["s"]["likelihood"]
    assert cfg["class"] == "PolyMarginalGaussianLikelihood"
    assert cfg["poly_calibration"]["mode"] == "marginalize"
    assert cfg["poly_calibration"]["order"] == 3
    assert cfg["poly_calibration"]["prior_sigma"] == [0.1] * 4
    assert r["obs"]["p"]["likelihood"]["class"] == "DiagonalGaussianLikelihood"


def test_calibrated_prediction_and_log_line(csp):
    from ceridwen.fit import _describe_likelihood
    from ceridwen.likelihood.poly_calibration import calibrated_prediction
    m = _model(csp, MARG_KW, data=_mock(csp))
    lh = _lhs(m)["s"]
    y, sig, mask, _c, _u = observation_data(m.obs_dict["s"])
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    mu = m.predict(m.apply_transforms(th))["s"]
    _mean, _cov, resp = lh.conditional(y, mu, sig, mask, th)
    np.testing.assert_array_equal(np.asarray(calibrated_prediction(lh, y, mu, sig, mask, th)),
                                  np.asarray(mu * resp))
    line = _describe_likelihood(m.obs_dict["s"], lh)
    assert "marginalised calibration polynomial, order 3" in line and "0.1" in line
