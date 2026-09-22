"""Outlier mixture likelihood (Hogg, Bovy & Lang 2010; Prospector's ``NoiseModel.lnlike``
branch ``f_outlier > 0``), ``DiagonalNoiseModel(f_outlier=..., nsigma_outlier=...)``.

1. equals a NumPy transcription of Prospector's branch for every noise-term combination;
2. equals the real Prospector (``tests/reference/prospector_outlier.npz``, written by
   ``run_prospector_outlier.py``), and the live ``prospect`` when it imports;
3. identities: nsigma = 1 and f -> 0 give the Gaussian; f = 0 / None leave the graph unchanged;
4. gradients finite (masked NaN data, chi = 1e4) and equal to central finite differences;
5. jit compiles once, vmap equals the loop, float32 agrees with float64;
6. fitSED wiring: one fraction per observation (plain name for a single observation of a
   kind, f_outlier_<kind>_<name> otherwise), Lines, upper limits (mixture on detections),
   the line marginalisation (mixture only outside its system), fixed values, refusals;
7. non-finite data: warned at construction when not masked, never in a value or gradient.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ceridwen.likelihood import (                                          # noqa: E402
    DiagonalNoiseModel, DiagonalGaussianLikelihood, DiagonalGaussianLikelihoodWithUpperLimits,
    MultiObservationLikelihood, lnlike_diag_gaussian, lnlike_diag_outlier, outlier_probability,
    lnlike_diag_gaussian_with_upper_limits, lnlike_diag_outlier_with_upper_limits)

REF = pathlib.Path(__file__).resolve().parent / "reference" / "prospector_outlier.npz"


# Prospector bd-j/prospector commit a78d15362fc177c43b7cde919d26409806e50fd7,
# prospect/likelihood/noise_model.py:56-62 (NoiseModel.lnlike, branch f_outlier > 0), verbatim
# apart from self.* -> arguments; Sigma is the noise model's diagonal variance.
def prospector_outlier_branch(flux, pred, mask, Sigma, f_outlier, n_sigma_outlier):
    delta = flux[mask] - pred[mask]
    var = Sigma
    lnp = -0.5*((delta**2 / var) + np.log(2*np.pi*var))
    var_bad = var * (n_sigma_outlier**2)
    lnp_bad = -0.5*((delta**2 / var_bad) + np.log(2*np.pi*var_bad))
    lnp_tot = np.logaddexp(lnp + np.log(1 - f_outlier), lnp_bad + np.log(f_outlier))
    return np.sum(lnp_tot)


def _data(n=400, seed=0, n_out=12, nan_masked=False):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(1.0, 5.0, n)
    sig = rng.uniform(0.05, 0.3, n)
    y = mu + sig * rng.standard_normal(n)
    idx = rng.choice(n, n_out, replace=False)
    y[idx] += rng.choice([-1.0, 1.0], n_out) * 20.0 * sig[idx]
    mask = rng.uniform(size=n) > 0.1
    if nan_masked:
        y[~mask] = np.nan
    return y, mu, sig, mask


NOISE_TERMS = {                       # every DiagonalNoiseModel variance term, alone and together
    "plain": {}, "floor": dict(noise_floor=0.03), "err_scale": dict(use_error_scale=True),
    "jitter": dict(use_jitter=True), "f_calib": dict(use_fractional=True),
    "f_data": dict(use_data_fractional=True),
    "all": dict(noise_floor=0.03, use_error_scale=True, use_jitter=True, use_fractional=True,
                use_data_fractional=True)}
NUIS = {"log_err_scale": jnp.array([0.2]), "log_jitter": jnp.array([np.log(0.07)]),
        "log_f_calib": jnp.array([np.log(0.02)]), "log_f_data": jnp.array([np.log(0.015)])}


def _sigma2_independent(kw, sig, mu, y):
    """sigma_eff^2 recomputed from the documented terms (not from DiagonalNoiseModel)."""
    v = sig ** 2
    if kw.get("use_error_scale"):
        v = v * np.exp(2 * 0.2)
    v = v + (kw.get("noise_floor", 0.0) * np.abs(mu)) ** 2
    if kw.get("use_fractional"):
        v = v + (0.02 * np.abs(mu)) ** 2
    if kw.get("use_data_fractional"):
        v = v + (0.015 * np.abs(y)) ** 2
    if kw.get("use_jitter"):
        v = v + 0.07 ** 2
    return v


# ============================================================================
# 1-2. equality with Prospector
# ============================================================================

@pytest.mark.parametrize("terms", list(NOISE_TERMS))
@pytest.mark.parametrize("f,nsigma", [(0.01, 50.0), (0.25, 4.0), (1e-5, 50.0)])
def test_equals_prospector_transcription(terms, f, nsigma):
    y, mu, sig, mask = _data(seed=list(NOISE_TERMS).index(terms))
    kw = NOISE_TERMS[terms]
    lh = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=f, nsigma_outlier=nsigma, **kw))
    lnl, aux = lh(jnp.asarray(y), jnp.asarray(mu), jnp.asarray(sig), jnp.asarray(mask), NUIS)
    Sigma = _sigma2_independent(kw, sig, mu, y)[mask]
    ref = prospector_outlier_branch(y, mu, mask, Sigma, f, nsigma)
    assert float(lnl) == pytest.approx(ref, rel=1e-12, abs=0.0)
    assert float(jnp.sum(aux.lnl_pointwise)) == pytest.approx(float(lnl), rel=1e-14, abs=0.0)
    assert int(aux.ndof) == int(mask.sum())


def test_equals_prospector_reference_file():
    ref = np.load(REF)
    tags = sorted({k.split("/")[0] for k in ref.files if "/" in k})
    assert len(tags) == 6
    for tag in tags:
        g = {k: ref[f"{tag}/{k}"] for k in ("flux", "unc", "mask", "pred", "f", "nsigma", "lnl")}
        args = [jnp.asarray(g[k]) for k in ("flux", "pred", "unc", "mask")]
        f = float(g["f"])
        lh = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=f,
                                                           nsigma_outlier=float(g["nsigma"])))
        lnl = float(lh(*args)[0])
        if f > 0.0:
            assert lnl == pytest.approx(float(g["lnl"]), rel=1e-12, abs=0.0), tag
        else:
            # Prospector's f == 0 branch is the defective plain likelihood (noise_model.py:90):
            # -0.5 [ln(2 pi) chi^2 + sum ln sigma^2].  CERIDWEN gives the correct Gaussian.
            chi2 = np.sum(((g["flux"] - g["pred"]) / g["unc"])[g["mask"]] ** 2)
            defect = -0.5 * (np.log(2 * np.pi) * chi2 + np.sum(np.log(g["unc"][g["mask"]] ** 2)))
            assert float(g["lnl"]) == pytest.approx(defect, rel=1e-12, abs=0.0), tag
            gauss = float(DiagonalGaussianLikelihood()(*args)[0])
            assert lnl == gauss
            assert abs(lnl - float(g["lnl"])) > 1.0


def test_equals_live_prospect():
    try:
        from prospect.likelihood import NoiseModel
        from prospect.observation import Spectrum as PSpectrum
    except Exception as exc:        # absent, or broken here (np.trapz under NumPy 2, $SPS_HOME)
        pytest.skip(f"prospect not usable here ({type(exc).__name__}: {exc}); the stored "
                    "reference (test_equals_prospector_reference_file) covers it")
    y, mu, sig, mask = _data(seed=11)
    nm = NoiseModel(frac_out_name="f_outlier_spec", nsigma_out_name="nsigma_outlier_spec")
    obs = PSpectrum(wavelength=np.linspace(4e3, 7e3, y.size), flux=y, uncertainty=sig,
                    mask=mask, noise=nm)
    nm.update(f_outlier_spec=np.array([0.03]), nsigma_outlier_spec=np.array([20.0]))
    ref = float(np.squeeze(nm.lnlike(mu, obs)))
    lh = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=0.03, nsigma_outlier=20.0))
    assert float(lh(y, mu, sig, mask)[0]) == pytest.approx(ref, rel=1e-12, abs=0.0)


# ============================================================================
# 3. identities
# ============================================================================

@pytest.mark.parametrize("f", [1e-4, 0.3, 0.9])
def test_nsigma_one_is_gaussian(f):
    y, mu, sig, mask = (jnp.asarray(a) for a in _data(seed=3))
    g = float(DiagonalGaussianLikelihood()(y, mu, sig, mask)[0])
    m = float(DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=f, nsigma_outlier=1.0))(
        y, mu, sig, mask)[0])
    assert m == pytest.approx(g, rel=1e-14, abs=0.0)


def test_small_f_tends_to_gaussian_without_outliers():
    y, mu, sig, mask = (jnp.asarray(a) for a in _data(seed=4, n_out=0))
    g = float(DiagonalGaussianLikelihood()(y, mu, sig, mask)[0])
    m = float(DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=1e-12))(
        y, mu, sig, mask)[0])
    # per datum the difference is f (exp(lnp_bad - lnp_good) - 1) ~ 1e-12 at |chi| < 4
    assert abs(m - g) < 1e-12 * float(jnp.sum(mask)) * 10.0
    assert m != g


def test_f_zero_and_none_are_the_unchanged_gaussian():
    assert DiagonalNoiseModel(f_outlier=0.0).f_outlier is None
    assert DiagonalNoiseModel(f_outlier=0.0) == DiagonalNoiseModel()
    y, mu, sig, mask = (jnp.asarray(a) for a in _data(seed=5))
    off = DiagonalGaussianLikelihood()
    zero = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=0.0))
    fn = lambda lh: (lambda m: lh(y, m, sig, mask)[0])                        # noqa: E731
    assert str(jax.make_jaxpr(fn(zero))(mu)) == str(jax.make_jaxpr(fn(off))(mu))
    a, b = off(y, mu, sig, mask), zero(y, mu, sig, mask)
    for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.array_equal(np.asarray(u), np.asarray(v))
    direct, _ = lnlike_diag_gaussian(y, mu, *(lambda o: (o.inv_var, o.log_det))(
        DiagonalNoiseModel().compute(sig, mu, mask, data=y)), mask)
    assert np.array_equal(np.asarray(a[0]), np.asarray(direct))


def test_outlier_probability_flags_the_outliers():
    y, mu, sig, mask = _data(seed=6, n_out=10)
    lh = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=0.02))
    p = np.asarray(lh.outlier_probability(y, mu, sig, mask))
    chi = np.abs((y - mu) / sig)
    assert np.all(p[~mask] == 0.0)
    assert np.all(p[mask & (chi > 15)] > 0.999)
    # Bayes' rule in closed form: p = f N(chi; n) / [(1-f) N(chi; 1) + f N(chi; n)]
    good = (1 - 0.02) * np.exp(-0.5 * chi ** 2)
    bad = 0.02 / 50.0 * np.exp(-0.5 * chi ** 2 / 50.0 ** 2)
    np.testing.assert_allclose(p[mask], (bad / (good + bad))[mask], rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="off"):
        DiagonalGaussianLikelihood().outlier_probability(y, mu, sig, mask)


# ============================================================================
# 4. gradients
# ============================================================================

def _sampled():
    return DiagonalGaussianLikelihood(DiagonalNoiseModel(
        f_outlier="f_outlier_spec", nsigma_outlier="nsigma_outlier_spec", use_jitter=True))


def test_gradients_finite_with_nan_masked_data_and_huge_chi():
    y, mu, sig, mask = _data(seed=7, nan_masked=True)
    y[np.flatnonzero(mask)[:2]] = mu[np.flatnonzero(mask)[:2]] + 1e4 * sig[np.flatnonzero(mask)[:2]]
    lh = _sampled()

    def lnl(f, ns, lj, m):
        return lh(jnp.asarray(y), m, jnp.asarray(sig), jnp.asarray(mask),
                  {"f_outlier_spec": f, "nsigma_outlier_spec": ns, "log_jitter": lj})[0]
    args = (jnp.array([0.05]), jnp.array([30.0]), jnp.array([-4.0]), jnp.asarray(mu))
    assert np.isfinite(float(lnl(*args)))
    for g in jax.grad(lnl, argnums=(0, 1, 2, 3))(*args):
        assert np.all(np.isfinite(np.asarray(g)))


def test_gradients_equal_central_finite_differences():
    y, mu, sig, mask = (jnp.asarray(a) for a in _data(seed=8))
    lh = _sampled()

    def lnl(v):          # v = (f, nsigma, log_jitter, mu shift)
        return lh(y, mu + v[3] * sig, sig, mask,
                  {"f_outlier_spec": v[0:1], "nsigma_outlier_spec": v[1:2],
                   "log_jitter": v[2:3]})[0]
    v0 = jnp.array([0.04, 25.0, -3.0, 0.1])
    g = np.asarray(jax.grad(lnl)(v0))
    for i, h in enumerate([1e-6, 1e-4, 1e-5, 1e-6]):
        e = jnp.zeros(4).at[i].set(h)
        fd = (float(lnl(v0 + e)) - float(lnl(v0 - e))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-6, abs=0.0), i
    from jax.test_util import check_grads
    check_grads(lnl, (v0,), order=2, modes=("rev",), eps=1e-5)


def test_gradient_in_f_is_exact_and_finite_at_zero():
    """d lnL/df = sum_i (e^b_i - e^a_i) / L_i (the custom JVP): at f = 0 it is
    sum_i (p_bad_i / p_good_i - 1), finite; a datum beyond ~37 sigma makes it +inf."""
    y, mu, sig, mask = _data(seed=9, n_out=0)
    lh = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier="f"))
    args = [jnp.asarray(a) for a in (y, mu, sig)]
    lnl = lambda f: lh(*args, jnp.asarray(mask), {"f": f})[0]                # noqa: E731
    chi2 = ((y - mu) / sig)[mask] ** 2
    ratio = np.exp(0.5 * chi2 * (1 - 1 / 50.0 ** 2)) / 50.0          # p_bad / p_good
    g0 = float(jax.grad(lnl)(jnp.array([0.0]))[0])
    assert g0 == pytest.approx(float(np.sum(ratio - 1.0)), rel=1e-12, abs=0.0)
    assert float(lnl(jnp.array([0.0]))) == float(DiagonalGaussianLikelihood()(
        *args, jnp.asarray(mask))[0])
    for f in (1e-3, 0.2):                                    # forward and reverse agree
        fwd = jax.jvp(lnl, (jnp.array([f]),), (jnp.array([1.0]),))[1]
        assert float(fwd) == pytest.approx(float(jax.grad(lnl)(jnp.array([f]))[0]),
                                           rel=1e-12, abs=0.0)
    y2 = y.copy(); i = int(np.flatnonzero(mask)[0]); y2[i] = mu[i] + 40.0 * sig[i]
    lnl2 = lambda f: lh(jnp.asarray(y2), *args[1:], jnp.asarray(mask), {"f": f})[0]  # noqa: E731
    assert float(jax.grad(lnl2)(jnp.array([0.0]))[0]) == np.inf
    assert np.isfinite(float(jax.grad(lnl2)(jnp.array([1e-5]))[0]))


def test_upper_limits_keep_one_sided_penalty_detections_get_mixture():
    y, mu, sig, mask = _data(seed=14)
    ul = np.zeros(y.size, bool); ul[np.flatnonzero(mask)[:25]] = True
    a = [jnp.asarray(v) for v in (y, mu, sig)]
    nm = DiagonalNoiseModel(f_outlier=0.05, nsigma_outlier=30.0)
    lnl, aux = DiagonalGaussianLikelihoodWithUpperLimits(nm)(*a, jnp.asarray(mask),
                                                             is_upper_limit=jnp.asarray(ul))
    out = nm.compute(a[2], a[1], jnp.asarray(mask), data=a[0])
    _, aux_ul = lnlike_diag_gaussian_with_upper_limits(*a[:2], out.inv_var, out.log_det,
                                                        jnp.asarray(mask), jnp.asarray(ul))
    _, aux_mix = lnlike_diag_outlier(*a[:2], out.inv_var, out.log_det, jnp.asarray(mask),
                                     0.05, 30.0)
    lp = np.asarray(aux.lnl_pointwise)
    np.testing.assert_array_equal(lp[ul], np.asarray(aux_ul.lnl_pointwise)[ul])
    np.testing.assert_array_equal(lp[~ul], np.asarray(aux_mix.lnl_pointwise)[~ul])
    none = DiagonalGaussianLikelihoodWithUpperLimits(nm)(*a, jnp.asarray(mask))
    assert float(none[0]) == float(DiagonalGaussianLikelihood(nm)(*a, jnp.asarray(mask))[0])
    Sigma = sig ** 2
    det = mask & ~ul
    ref = prospector_outlier_branch(y, mu, det, Sigma[det], 0.05, 30.0) + np.sum(
        -0.5 * np.maximum(mu - y, 0.0)[mask & ul] ** 2 / Sigma[mask & ul]
        - 0.5 * np.log(2 * np.pi * Sigma[mask & ul]))
    assert float(lnl) == pytest.approx(ref, rel=1e-12, abs=0.0)


# ============================================================================
# 5. jit, vmap, float32
# ============================================================================

def test_jit_compiles_once_and_vmap_equals_loop():
    y, mu, sig, mask = (jnp.asarray(a) for a in _data(seed=10))
    lh = _sampled()
    traces = []

    @jax.jit
    def lnl(th):
        traces.append(1)
        return lh(y, mu * th["s"], sig, mask, th)[0]
    rng = np.random.default_rng(0)
    B = 100
    batch = {"f_outlier_spec": jnp.asarray(rng.uniform(1e-4, 0.3, (B, 1))),
             "nsigma_outlier_spec": jnp.asarray(rng.uniform(5.0, 60.0, (B, 1))),
             "log_jitter": jnp.asarray(rng.uniform(-6.0, -2.0, (B, 1))),
             "s": jnp.asarray(rng.uniform(0.98, 1.02, (B,)))}
    loop = np.array([float(lnl({k: v[i] for k, v in batch.items()})) for i in range(B)])
    assert len(traces) == 1
    vm = np.asarray(jax.jit(jax.vmap(lnl))(batch))
    np.testing.assert_allclose(vm, loop, rtol=1e-13, atol=0.0)


def test_single_observation_make_lnprobfn_uses_the_mixture():
    from types import SimpleNamespace
    y, mu, sig, mask = (jnp.asarray(a) for a in _data(seed=13))
    obs = SimpleNamespace(flux=y, uncertainty=sig, mask=mask)
    model = SimpleNamespace(predict=lambda th: mu * th["s"])
    prior = SimpleNamespace(log_prob=lambda th: -0.5 * jnp.sum(th["s"] - 1.0) ** 2)
    lh = _sampled()
    th = {"s": jnp.array([1.001]), "f_outlier_spec": jnp.array([0.03]),
          "nsigma_outlier_spec": jnp.array([30.0]), "log_jitter": jnp.array([-4.0])}
    got = float(lh.make_lnprobfn(obs, model, prior)(th))
    want = float(lh(y, mu * th["s"], sig, mask, th)[0]) + float(prior.log_prob(th))
    assert got == pytest.approx(want, rel=1e-14, abs=0.0)
    gauss = DiagonalGaussianLikelihood(DiagonalNoiseModel(use_jitter=True))
    assert got != pytest.approx(float(gauss.make_lnprobfn(obs, model, prior)(th)), rel=1e-6)


def test_float32_matches_float64():
    y, mu, sig, mask = _data(seed=12, n_out=20)
    y[np.flatnonzero(mask)[0]] = mu[np.flatnonzero(mask)[0]] + 1e4 * sig[np.flatnonzero(mask)[0]]
    lh = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier="f"))
    th = {"f": jnp.array([1e-5])}                   # log f = -11.5
    ref = float(lh(*(jnp.asarray(a) for a in (y, mu, sig)), jnp.asarray(mask), th)[0])
    a32 = [jnp.asarray(a, dtype=jnp.float32) for a in (y, mu, sig)]
    lnl32 = lh(*a32, jnp.asarray(mask), {"f": jnp.array([1e-5], dtype=jnp.float32)})[0]
    assert lnl32.dtype == jnp.float32
    assert float(lnl32) == pytest.approx(ref, rel=1e-5, abs=0.0)


# ============================================================================
# 6. construction and fitSED wiring
# ============================================================================

def test_noise_model_validation_and_pytree():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            DiagonalNoiseModel(f_outlier=bad)
    for bad in (0.0, -2.0):
        with pytest.raises(ValueError):
            DiagonalNoiseModel(f_outlier=0.1, nsigma_outlier=bad)
    with pytest.raises(TypeError):
        DiagonalNoiseModel(f_outlier=True)
    nm = DiagonalNoiseModel(f_outlier="f_outlier_spec", use_jitter=True)
    assert nm.use_outlier and nm.outlier_param_names == ("f_outlier_spec",)
    assert nm.nuisance_param_names == ("log_jitter",)
    leaves, tree = jax.tree_util.tree_flatten(DiagonalGaussianLikelihood(nm))
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).noise_model == nm
    with pytest.raises(KeyError, match="f_outlier_spec"):
        DiagonalGaussianLikelihood(nm)(jnp.ones(3), jnp.ones(3), jnp.ones(3),
                                       jnp.ones(3, bool), {"log_jitter": jnp.array([-3.0])})
    DiagonalGaussianLikelihoodWithUpperLimits(DiagonalNoiseModel(f_outlier=0.1))  # allowed


@pytest.fixture(scope="module")
def grid():
    from _gridfixture import find_test_grid
    path = find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (tests/_gridfixture.py)")
    from ceridwen import SSPData
    return SSPData.load(str(path))


ZRED = 0.8


def _model(ssp, priors=None, transforms=None, init=None, two_spectra=False, phot=True,
           upper_limit=False, lines=False, phot_flux=None, phot_mask=None, marginalize=False):
    from ceridwen import CSPBasis, SedModel, Cosmology
    from ceridwen.broadening import Kinematics, Instrument
    from ceridwen.model.transforms import logsfr_ratios_to_sfh
    from ceridwen.observation import Photometry, Spectrum, Lines
    from ceridwen.sampler.priors import Uniform
    cosmo = Cosmology.planck18()
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), 5),
                   zh_const=True, sfh_interp="step", add_dust=True, add_diffuse_dust=True,
                   add_neb=False, add_igm=False, verbose=False, cosmo=cosmo)
    t_yr = np.array(csp.sfh_times)
    wave = (np.linspace(4700.0, 6800.0, 2500) if marginalize            # lines need pixels
            else np.linspace(3800.0, 8800.0, 300)) * (1 + ZRED)
    rng = np.random.default_rng(1)
    obs = []
    if phot:
        f = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
        ul = np.array([False, False, False, False, True]) if upper_limit else None
        pf = rng.uniform(1e-9, 3e-9, 5) if phot_flux is None else phot_flux
        kw = {} if phot_mask is None else {"mask": phot_mask}
        obs.append(Photometry(filters=f, flux=pf, uncertainty=np.full(5, 2e-10), name="phot",
                              upper_limit=ul, **kw))
    for i in range(2 if two_spectra else 1):
        extra = ({"marginalize_elines": True} if (marginalize and i == 0) else {})
        obs.append(Spectrum(wavelength=wave, flux=rng.uniform(0.8e-9, 1.2e-9, wave.size),
                            uncertainty=np.full(wave.size, 1e-10),
                            instrument=Instrument.sigma_kms(150.0), name=f"spec{i}", **extra))
    if lines:
        obs.append(Lines(line_ind=np.arange(3), line_names=["Hbeta", "OIII5007", "Halpha"],
                         wavelength=np.array([4862.71, 5008.24, 6564.61]),
                         flux=np.full(3, 1e-17), uncertainty=np.full(3, 1e-18), name="lines"))
    base_init = {"logsfr_ratios": jnp.zeros(4), "logmass": jnp.array([10.0])}
    base_init.update(init or {})
    tr = {"sfh": lambda th: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=t_yr)}
    tr.update(transforms or {})
    pr = {"logmass": Uniform(low=9.0, high=11.0)}
    pr.update(priors or {})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, obs, priors=pr, transforms=tr, free_param_init=base_init,
                        zred=ZRED, kinematics=Kinematics(sigma_gal=200.0))


def _lhs(model):
    from ceridwen.fit import _likelihood_for
    return {o.name: _likelihood_for(o, model.param_names, model=model) for o in model.observations}


def _sampled_f(*names, low=1e-5):
    from ceridwen.sampler.priors import TopHat
    return dict(priors={n: TopHat(low=low, high=0.5) for n in names},
                init={n: jnp.array([0.01]) for n in names})


def test_wiring_sampled_spec_only(grid):
    from ceridwen.fit import _describe_likelihood
    m = _model(grid, **_sampled_f("f_outlier_spec"))
    lh = _lhs(m)
    assert lh["spec0"].noise_model.f_outlier == "f_outlier_spec"
    assert lh["spec0"].noise_model.nsigma_outlier == 50.0
    assert not lh["phot"].noise_model.use_outlier          # phot and spec do not share it
    d = _describe_likelihood(m.obs_dict["spec0"], lh["spec0"])
    assert "outlier mixture f = f_outlier_spec (sampled), nsigma = 50 (fixed)" in d
    assert "outlier mixture off" in _describe_likelihood(m.obs_dict["phot"], lh["phot"])
    # the sampled log-posterior uses the mixture on the spectrum and the Gaussian on phot
    multi = MultiObservationLikelihood(keys=tuple(lh), likelihoods=tuple(lh.values()))
    lnprob = multi.make_lnprobfn(m.obs_dict, m, m)
    th = dict(m.theta_init)
    pred = m.predict(th)
    manual = 0.0
    for k, o in m.obs_dict.items():
        manual += float(lh[k](o.flux, pred[k], o.uncertainty, o.mask, th)[0])
    gauss_spec = float(DiagonalGaussianLikelihood()(m.obs_dict["spec0"].flux, pred["spec0"],
                       m.obs_dict["spec0"].uncertainty, m.obs_dict["spec0"].mask)[0])
    assert float(lnprob(th)) == pytest.approx(manual + float(m.log_prob(th)), rel=1e-12, abs=0.0)
    assert float(lh["spec0"](m.obs_dict["spec0"].flux, pred["spec0"],
                             m.obs_dict["spec0"].uncertainty, m.obs_dict["spec0"].mask,
                             th)[0]) != gauss_spec


def test_wiring_one_fraction_per_spectrum(grid):
    m = _model(grid, two_spectra=True, **_sampled_f("f_outlier_spec_spec0"),
               transforms={"f_outlier_spec_spec1": lambda th: jnp.array([0.2]),
                           "nsigma_outlier_spec_spec1": lambda th: jnp.array([10.0])})
    lh = _lhs(m)
    assert lh["spec0"].noise_model.f_outlier == "f_outlier_spec_spec0"
    assert lh["spec0"].noise_model.nsigma_outlier == 50.0
    assert (lh["spec1"].noise_model.f_outlier, lh["spec1"].noise_model.nsigma_outlier) == (0.2, 10.0)
    assert not lh["phot"].noise_model.use_outlier
    with pytest.raises(ValueError, match="ambiguous"):          # the plain name, 2 spectra
        _lhs(_model(grid, two_spectra=True, **_sampled_f("f_outlier_spec")))
    # one spectrum: plain and per-observation name both work, not both at once
    one = _lhs(_model(grid, **_sampled_f("f_outlier_spec_spec0")))
    assert one["spec0"].noise_model.f_outlier == "f_outlier_spec_spec0"
    with pytest.raises(ValueError, match="keep one"):
        _lhs(_model(grid, **_sampled_f("f_outlier_spec", "f_outlier_spec_spec0")))


def test_wiring_lines_and_upper_limits(grid):
    m = _model(grid, lines=True, upper_limit=True, **_sampled_f("f_outlier_lines", "f_outlier_phot",
                                                             low=0.0))
    lh = _lhs(m)
    assert lh["lines"].noise_model.f_outlier == "f_outlier_lines"
    assert type(lh["phot"]).__name__ == "DiagonalGaussianLikelihoodWithUpperLimits"
    assert lh["phot"].noise_model.f_outlier == "f_outlier_phot"
    assert not lh["spec0"].noise_model.use_outlier


def test_default_is_off_for_every_kind_and_fixed_zero_is_off(grid):
    """All outlier fractions default to 0: (a) no f_outlier_* key -> off for Photometry,
    Spectrum and Lines; (b) every fraction fixed at 0 -> off, ln L array_equal to the plain
    Gaussian; (c) the fitSED log line says so."""
    from ceridwen.fit import _describe_likelihood
    zero = {n: (lambda th: jnp.array([0.0])) for n in
            ("f_outlier_phot", "f_outlier_spec_spec0", "f_outlier_spec_spec1", "f_outlier_lines")}
    for kw in ({}, {"transforms": zero}):
        m = _model(grid, two_spectra=True, lines=True, **kw)
        lh = _lhs(m)
        assert {o.kind for o in m.observations} == {"photometry", "spectrum", "lines"}
        for k, o in m.obs_dict.items():
            assert lh[k].noise_model == DiagonalNoiseModel(), k                   # (a), (b)
            assert "outlier mixture off" in _describe_likelihood(o, lh[k]), k      # (c)
        m = _model(grid, two_spectra=True, **({"transforms": {k: v for k, v in zero.items()
                                                               if k != "f_outlier_lines"}}
                                               if kw else {}))
        lh = _lhs(m)                          # (b) without Lines: predict needs no nebular grid
        th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
        pred = m.predict(th)
        for k in ("phot", "spec0", "spec1"):
            o = m.obs_dict[k]
            got = lh[k](o.flux, pred[k], o.uncertainty, o.mask, th)
            ref = DiagonalGaussianLikelihood()(o.flux, pred[k], o.uncertainty, o.mask)
            for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref)):
                assert np.array_equal(np.asarray(a), np.asarray(b)), k


def test_wiring_fixed_by_transform_and_zero_is_off(grid):
    m = _model(grid, transforms={"f_outlier_phot": lambda th: jnp.array([0.05]),
                                 "nsigma_outlier_phot": lambda th: jnp.array([20.0])})
    nm = _lhs(m)["phot"].noise_model
    assert nm.f_outlier == 0.05 and nm.nsigma_outlier == 20.0
    m0 = _model(grid, transforms={"f_outlier_phot": lambda th: jnp.array([0.0])})
    assert _lhs(m0)["phot"].noise_model == DiagonalNoiseModel()
    md = _model(grid, transforms={"f_outlier_spec": lambda th: 0.01 * jnp.exp(th["logmass"] - 10)})
    with pytest.raises(NotImplementedError, match="constant"):
        _lhs(md)


def test_wiring_refusals(grid):
    from ceridwen.sampler.priors import TopHat, Normal
    with pytest.raises(ValueError, match="match no observation"):
        _lhs(_model(grid, phot=False, **_sampled_f("f_outlier_phot")))
    with pytest.raises(ValueError, match="match no observation"):          # wrong obs name
        _lhs(_model(grid, **_sampled_f("f_outlier_spec_specX")))
    with pytest.raises(ValueError, match="without an f_outlier"):
        _lhs(_model(grid, priors={"nsigma_outlier_spec": TopHat(low=2.0, high=80.0)},
                    init={"nsigma_outlier_spec": jnp.array([50.0])}))
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        _lhs(_model(grid, priors={"f_outlier_spec": Normal(mean=0.1, sigma=0.05)},
                    init={"f_outlier_spec": jnp.array([0.1])}))
    with pytest.raises(ValueError, match="positive lower bound"):
        _lhs(_model(grid, priors={"f_outlier_spec": TopHat(low=1e-5, high=0.5),
                                  "nsigma_outlier_spec": TopHat(low=0.0, high=80.0)},
                    init={"f_outlier_spec": jnp.array([0.01]),
                          "nsigma_outlier_spec": jnp.array([50.0])}))


def test_eline_refusal_only_inside_the_system():
    from ceridwen.likelihood.eline_marginal import refuse_outlier_with_elines
    mix = DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier=0.1))
    with pytest.raises(NotImplementedError, match="outside the"):
        refuse_outlier_with_elines(("spec0", "phot"), (DiagonalGaussianLikelihood(), mix),
                                   ("spec0", "phot"))
    refuse_outlier_with_elines(("spec0", "spec1"), (DiagonalGaussianLikelihood(), mix),
                               ("spec0",))


@pytest.mark.skipif(not os.environ.get("SPS_HOME"),
                    reason="line marginalisation without a nebular grid reads $SPS_HOME")
def test_mixture_outside_the_marginalised_system(grid, tmp_path):
    """spec0 marginalises its lines (system = spec0 alone, no photometry); spec1 outside
    carries a mixture: the joint ln L changes by exactly spec1's mixture - Gaussian."""
    from ceridwen import fitSED
    from ceridwen.likelihood.eline_marginal import joint_loglike, _static_data
    m = _model(grid, phot=False, two_spectra=True, marginalize=True,
               **_sampled_f("f_outlier_spec_spec1"))
    assert tuple(m._eline_system.keys) == ("spec0",)
    lh = _lhs(m)
    keys = tuple(lh)
    gauss = dict(lh, spec1=DiagonalGaussianLikelihood())
    th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
    sd = _static_data(m, keys)
    a = float(joint_loglike(m, keys, tuple(lh[k] for k in keys), sd, th))
    b = float(joint_loglike(m, keys, tuple(gauss[k] for k in keys), sd, th))
    y, sig, mask, _, _ = sd["spec1"]
    mu = m.predict_with_elines(th)[0]["spec1"]
    d = float(lh["spec1"](y, mu, sig, mask, th)[0]) - float(gauss["spec1"](y, mu, sig, mask)[0])
    assert d != 0.0 and a - b == pytest.approx(d, rel=1e-9, abs=0.0)
    inside = _model(grid, phot=False, two_spectra=True, marginalize=True,
                    **_sampled_f("f_outlier_spec_spec0"))
    with pytest.raises(NotImplementedError, match="outside the"):
        fitSED(inside, output_dir=tmp_path, verbose=False)


def test_unmasked_nan_warns_and_masked_nan_is_irrelevant(grid):
    pf = np.array([1.5e-9, np.nan, 2e-9, 2.5e-9, 1.8e-9])
    from ceridwen.observation import Photometry
    with pytest.warns(UserWarning, match=r"were not in the mask \(indices \[1\]\)"):
        Photometry(filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"], flux=pf,
                   uncertainty=np.full(5, 2e-10), name="p")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # masked by the user: no warning
        Photometry(filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"], flux=pf,
                   uncertainty=np.full(5, 2e-10), name="p", mask=np.isfinite(pf))
    # gradients of the fitSED log-posterior are finite with the NaN in the data, for the
    # Gaussian and the mixture, with a sampled error scale too
    from ceridwen.sampler.priors import TopHat
    for extra in ({}, _sampled_f("f_outlier_phot")):
        pr = dict(extra.get("priors", {}), log_err_scale=TopHat(low=-1.0, high=1.0))
        ini = dict(extra.get("init", {}), log_err_scale=jnp.array([0.1]))
        m = _model(grid, phot_flux=pf, priors=pr, init=ini)
        lh = _lhs(m)
        lnprob = MultiObservationLikelihood(keys=tuple(lh), likelihoods=tuple(lh.values())
                                            ).make_lnprobfn(m.obs_dict, m, m)
        th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
        v, g = jax.value_and_grad(lnprob)(th)
        assert np.isfinite(float(v))
        assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(g))


def test_fitsed_logs_the_mixture(grid, tmp_path):
    from ceridwen import fitSED
    from ceridwen.sampler.priors import TopHat, Uniform, StudentT
    m = _model(grid, priors={"f_outlier_spec": TopHat(low=1e-5, high=0.5),
                             "logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
                             "logzsol": Uniform(low=-1.8, high=0.19),
                             "tau_pow": Uniform(low=0.0, high=2.0),
                             "alpha_pow": Uniform(low=-2.0, high=0.0),
                             "diffuse_tau_kc": Uniform(low=0.0, high=2.0),
                             "diffuse_dust_index": Uniform(low=-1.0, high=0.4)},
               init={"f_outlier_spec": jnp.array([0.01])})
    m.priors = {k: v for k, v in m.priors.items() if k in m.param_names}
    res = fitSED(m, sampler="nested", rng_key=jax.random.PRNGKey(0), output_dir=tmp_path,
                 sampler_kwargs={"num_live": 20, "num_delete": 4, "num_inner_steps": 3,
                                 "logZ_tol": 1e3}, verbose=False)
    log = (tmp_path / "ceridwen_result.log").read_text()
    assert "spec0: DiagonalGaussianLikelihood, outlier mixture f = f_outlier_spec (sampled)" in log
    assert "phot: DiagonalGaussianLikelihood, outlier mixture off" in log
    assert log.count("outlier mixture f =") == 1
    f = np.asarray(res.samples["f_outlier_spec"])
    assert np.all((f >= 1e-5) & (f <= 0.5)) and np.all(np.isfinite(res.log_likelihoods))
