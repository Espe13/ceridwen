"""Emission-line marginalisation (``Spectrum(marginalize_elines=True)``).

1. the kernel is the exact marginal likelihood: numerical quadrature over the line fluxes
   (1 and 2 lines; flat, Gaussian and mixed priors) and least squares for the posterior mean;
2. against Prospector's ``SpecModel.fit_mle_elines`` on identical inputs
   (``tests/reference/prospector_elines.npz``, written by ``run_prospector_elines.py``);
3. limits: zero flux, prior width -> 0 (= the CLOUDY lines, the ordinary likelihood),
   prior width -> infinity (= flat prior plus the prior-volume term);
4. jit, vmap at the production batch width, gradients against finite differences;
5. every refusal happens at construction.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from scipy import integrate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_emission_lines as T                                           # noqa: E402

from ceridwen import SedModel                                             # noqa: E402
from ceridwen.broadening import Kinematics, Instrument                    # noqa: E402
from ceridwen.fit import _likelihood_for                                  # noqa: E402
from ceridwen.likelihood.likelihood import MultiObservationLikelihood     # noqa: E402
from ceridwen.likelihood.eline_marginal import (                          # noqa: E402
    eline_marginal_loglike, eline_line_fluxes, _check_conditioning, ElineSystem)
from ceridwen.model.transforms import logsfr_ratios_to_sfh               # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines             # noqa: E402
from ceridwen.sampler.priors import Uniform                               # noqa: E402

REF = pathlib.Path(__file__).resolve().parent / "reference" / "prospector_elines.npz"
LOG2PI = np.log(2.0 * np.pi)
W_PROD = 100          # BlackJAX NSS batch = num_delete = num_live // 5 at fitSED defaults


# ============================================================================
# 1. kernel == exact marginal likelihood
# ============================================================================

def _synthetic(n_lines, seed=0):
    """Unit-flux Gaussian columns on 200 pixels, known noise, residual with lines + noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-10.0, 10.0, 200)
    centres = [-1.0, 1.5][:n_lines]
    A = np.stack([np.exp(-0.5 * (x - c) ** 2 / 0.8 ** 2) for c in centres], axis=1)
    sig = np.full(x.size, 0.3) * (1 + 0.2 * np.sin(x))
    true = np.array([2.0, -0.7])[:n_lines]
    r = A @ true + sig * rng.standard_normal(x.size)
    w = 1.0 / sig ** 2
    lognorm = float(np.sum(0.5 * np.log(2 * np.pi * sig ** 2)))
    return r, A, w, lognorm


def _lnlike(alpha, r, A, w, lognorm):
    res = r - A @ np.atleast_1d(alpha)
    return -0.5 * np.sum(w * res ** 2) - lognorm


def _kernel(r, A, w, lognorm, mean, sd, flat, post=False):
    return eline_marginal_loglike([(jnp.asarray(r), jnp.asarray(A), jnp.asarray(w), lognorm)],
                                  jnp.asarray(mean), jnp.asarray(sd), np.asarray(flat),
                                  return_posterior=post)


def _ln_gauss(a, m, s):
    return -0.5 * ((a - m) / s) ** 2 - np.log(s) - 0.5 * LOG2PI


@pytest.mark.parametrize("prior", ["flat", "gauss"])
def test_one_line_kernel_equals_quadrature(prior):
    r, A, w, ln0 = _synthetic(1)
    mean, sd = np.array([1.6]), np.array([0.3])
    flat = np.array([prior == "flat"])
    lnz = float(_kernel(r, A, w, ln0, mean, sd, flat))
    ahat = np.linalg.lstsq(A * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None)[0][0]
    peak = _lnlike(ahat, r, A, w, ln0)

    def f(a):
        lp = 0.0 if prior == "flat" else _ln_gauss(a, mean[0], sd[0])
        return np.exp(_lnlike(a, r, A, w, ln0) + lp - peak)
    val, err = integrate.quad(f, ahat - 30, ahat + 30, epsabs=0, epsrel=1e-13, limit=400,
                              points=[ahat, mean[0]])
    assert abs(peak + np.log(val) - lnz) < 1e-9


@pytest.mark.parametrize("prior", ["flat", "gauss", "mixed"])
def test_two_line_kernel_equals_quadrature(prior):
    r, A, w, ln0 = _synthetic(2, seed=3)
    mean, sd = np.array([1.5, -0.2]), np.array([0.4, 0.5])
    flat = {"flat": [True, True], "gauss": [False, False], "mixed": [True, False]}[prior]
    flat = np.asarray(flat)
    lnz = float(_kernel(r, A, w, ln0, mean, sd, flat))
    ahat = np.linalg.lstsq(A * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None)[0]
    peak = _lnlike(ahat, r, A, w, ln0)

    def f(a1, a0):
        lp = sum(0.0 if flat[j] else _ln_gauss(a, mean[j], sd[j])
                 for j, a in enumerate((a0, a1)))
        return np.exp(_lnlike([a0, a1], r, A, w, ln0) + lp - peak)
    val, err = integrate.dblquad(f, ahat[0] - 8, ahat[0] + 8, ahat[1] - 8, ahat[1] + 8,
                                 epsabs=0, epsrel=1e-11)
    assert abs(peak + np.log(val) - lnz) < 1e-8


def test_flat_posterior_mean_is_weighted_least_squares():
    r, A, w, ln0 = _synthetic(2, seed=5)
    _, mean, cov = _kernel(r, A, w, ln0, np.zeros(2), np.ones(2), np.array([True, True]), post=True)
    Aw = A * np.sqrt(w)[:, None]
    ahat = np.linalg.lstsq(Aw, r * np.sqrt(w), rcond=None)[0]
    np.testing.assert_allclose(np.asarray(mean), ahat, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(cov), np.linalg.inv(Aw.T @ Aw), rtol=1e-10)


# ============================================================================
# 2. Prospector fit_mle_elines on identical inputs
# ============================================================================

def _ref_cases():
    if not REF.is_file():
        return []
    with np.load(REF) as d:
        return sorted({k.split("/")[0] for k in d.files if "/" in k})


@pytest.mark.prospector
@pytest.mark.parametrize("case", _ref_cases() or ["<missing>"])
def test_kernel_matches_prospector_fit_mle_elines(case):
    if not REF.is_file():
        pytest.skip(f"MISSING {REF.name}: run tests/reference/run_prospector_elines.py in the "
                    "prospector environment to create it; the Prospector comparison did NOT run")
    with np.load(REF) as d:
        g = {k.split("/", 1)[1]: d[k] for k in d.files if k.startswith(case + "/")}
    A, delta, unc = g["A"], g["delta"], g["unc"]
    w = 1.0 / unc ** 2
    ln0 = float(np.sum(0.5 * np.log(2 * np.pi * unc ** 2)))
    m = A.shape[1]
    abreve = g["alpha_breve"]
    sbreve = np.sqrt(np.diag(g["sigma_breve"]))

    lnz_f, mean_f, cov_f = _kernel(delta, A, w, ln0, abreve, sbreve, np.ones(m, bool), post=True)
    lnz_p, mean_p, cov_p = _kernel(delta, A, w, ln0, abreve, sbreve, np.zeros(m, bool), post=True)
    mean_f, cov_f, mean_p, cov_p = map(np.asarray, (mean_f, cov_f, mean_p, cov_p))
    chi2_hat = np.sum(w * (delta - A @ mean_f) ** 2)
    base = -0.5 * chi2_hat - ln0                 # ln L at the ML amplitudes (what Prospector scores)

    scale = np.abs(g["alpha_hat"]).max()
    np.testing.assert_allclose(mean_f, g["alpha_hat"], rtol=1e-10, atol=1e-10 * scale)
    np.testing.assert_allclose(cov_f, g["sigma_hat"], rtol=1e-9,
                               atol=1e-10 * np.abs(g["sigma_hat"]).max())
    np.testing.assert_allclose(mean_p, g["alpha_bar"], rtol=1e-10, atol=1e-10 * scale)
    np.testing.assert_allclose(cov_p, g["sigma_bar"], rtol=1e-9,
                               atol=1e-10 * np.abs(g["sigma_bar"]).max())
    np.testing.assert_allclose(A @ mean_f, g["line_spec"], rtol=1e-10,
                               atol=1e-10 * np.abs(g["line_spec"]).max())
    # with a prior both codes compute the exact marginal: same penalty
    assert float(lnz_p) - base == pytest.approx(float(g["K"]), rel=1e-10, abs=1e-10)
    # without: Prospector's penalty has the opposite sign of the flat-prior marginal (D1)
    assert float(lnz_f) - base == pytest.approx(-float(g["K_flat"]), rel=1e-10, abs=1e-10)


# ============================================================================
# 3-5. the full model
# ============================================================================

def _model(csp, *, width=0.0, marg=True, phot=True, lines=True, fit=None, fix=None,
           ignore=None, zred_prior=None, extra_transforms=None, sample_gas=False,
           spectra=1, noise_seed=1, line_boost=None):
    """Nebular CSP at z=0.3 (nebular parameters fixed by constant transforms), Spectrum
    (+ Photometry + Lines) with mock data from the grid model, lines optionally boosted."""
    t = np.array(csp.sfh_times)
    tr = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)}
    if not sample_gas:
        tr.update({"gas_logu": lambda th: jnp.array([-2.23]),
                   "gas_logz": lambda th: jnp.array([-0.37])})
    tr.update(extra_transforms or {})
    init = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.0])}
    kw = dict(zred=T.ZRED)
    priors = {}
    if zred_prior is not None:
        init["zred"] = jnp.array([T.ZRED])
        priors["zred"] = zred_prior
        kw = {}
    truth = _truth(csp)
    rng = np.random.default_rng(noise_seed)
    y = np.asarray(truth["spec"]) * (1.0 if line_boost is None else 1.0)
    e = 0.03 * np.median(y) * np.ones_like(y)
    specs = [Spectrum(wavelength=T.WAVE_OBS, flux=y + e * rng.standard_normal(y.size),
                      uncertainty=e, instrument=Instrument.R_fwhm(T.R_FWHM),
                      name="spec" if i == 0 else f"spec{i}", marginalize_elines=marg,
                      eline_prior_width=width if marg else 0.0,
                      elines_to_fit=fit, elines_to_fix=fix, elines_to_ignore=ignore)
             for i in range(spectra)]
    obs = list(specs)
    if phot:
        yp = np.asarray(truth["phot"], float)
        obs.append(Photometry(filters=T.FILTERS, flux=yp, uncertainty=0.05 * yp, name="phot"))
    if lines:
        yl = np.asarray(truth["lines"])
        obs.append(Lines(line_ind=np.arange(5), wavelength=[w for _, w in T.LINES] + [T.OII[0]],
                         components=[(w,) for _, w in T.LINES] + [T.OII], flux=yl,
                         uncertainty=0.1 * yl, name="lines"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, obs, transforms=tr, free_param_init=init, priors=priors,
                         kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS), **kw)
    lh = MultiObservationLikelihood(
        keys=tuple(model.obs_dict),
        likelihoods=tuple(_likelihood_for(o, model.param_names) for o in model.observations))
    return model, lh


_TRUTH = {}


def _truth(csp):
    if id(csp) not in _TRUTH:
        s, p, l = T._obs()
        m = T._model(csp, (s, p, l))
        _TRUTH[id(csp)] = m.predict(T._theta(m))
    return _TRUTH[id(csp)]


class _NoPrior:
    def log_prob(self, theta):
        return 0.0


def _lnl(model, lh):
    return lh.make_lnprobfn(model.obs_dict, model, _NoPrior())


def _theta(model, **over):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th["logzsol"] = jnp.array([-0.3])                  # the metallicity of the mock (T._theta)
    th.update({k: jnp.asarray(v) for k, v in over.items()})
    return th


@pytest.fixture(scope="module")
def csp():
    return T._csp(True)


def test_marginalisation_off_changes_nothing(csp):
    model, lh = _model(csp, marg=False)
    assert model._eline_system is None
    ref, lh_ref = _model(csp, marg=False)
    th = _theta(model)
    for k, v in model.predict(th).items():
        np.testing.assert_array_equal(np.asarray(v), np.asarray(ref.predict(th)[k]))


def test_zero_prior_width_is_the_ordinary_likelihood(csp):
    """Every line pinned at its CLOUDY flux: the joint marginal likelihood over Spectrum,
    Photometry and Lines equals the ordinary likelihood of the grid model (limit 3b)."""
    std, lh_std = _model(csp, marg=False)
    pin, lh_pin = _model(csp, width=1e-12)
    th = _theta(pin)
    # component by component: the predictions with the lines put back at their CLOUDY fluxes
    ref = std.predict(_theta(std))
    pred, aux = pin.predict_with_elines(th)
    back = {k: np.asarray(pred[k], float) + np.asarray(aux["cols"][k]) @ np.asarray(aux["prior_mean"])
            for k in pred}
    np.testing.assert_allclose(back["spec"], np.asarray(ref["spec"]), rtol=1e-11,
                               atol=1e-11 * float(np.abs(ref["spec"]).max()))
    np.testing.assert_array_equal(back["lines"], np.asarray(ref["lines"]))
    # the ordinary photometric line path is float32 (G @ F in obs._T's dtype); ours is float64
    np.testing.assert_allclose(back["phot"], np.asarray(ref["phot"], float), rtol=1e-6)
    a = float(_lnl(std, lh_std)(_theta(std)))
    b = float(_lnl(pin, lh_pin)(th))
    assert b == pytest.approx(a, rel=1e-9, abs=0.0)
    post = eline_line_fluxes(pin, th, lh_pin)
    np.testing.assert_allclose(np.asarray(post["mean"]), np.asarray(post["cloudy"]), rtol=1e-9)


def test_wide_prior_tends_to_flat_plus_prior_volume(csp):
    """Prior width -> infinity: ln Z_prior - ln Z_flat -> -sum ln s_j - (m/2) ln 2 pi (limit 3c)."""
    flat, lh_f = _model(csp, width=0.0)
    th = _theta(flat)
    zf = float(_lnl(flat, lh_f)(th))
    width = 1e20        # s_j >> sigma_hat_j even for lines with CLOUDY flux ~1e-27
    wide, lh_w = _model(csp, width=width)
    zw = float(_lnl(wide, lh_w)(th))
    F = np.asarray(eline_line_fluxes(wide, th, lh_w)["cloudy"])
    s = width * np.abs(F)
    expect = -np.sum(np.log(s)) - 0.5 * F.size * LOG2PI
    assert zw - zf == pytest.approx(expect, rel=1e-6, abs=1e-6)


def test_noise_free_mock_recovers_the_true_fluxes(csp):
    """Flat prior, data = the grid model: the posterior line fluxes are the CLOUDY fluxes
    (and a line with zero true flux is recovered as zero), limit 3a."""
    model, lh = _model(csp, width=0.0, noise_seed=None)
    model.obs_dict["spec"].flux = jnp.asarray(_truth(csp)["spec"])
    post = eline_line_fluxes(model, _theta(model), lh)
    mean, sd, F = map(np.asarray, (post["mean"], post["sd"], post["cloudy"]))
    assert np.all(np.abs(mean - F) < 1e-6 * sd)


def test_line_fluxes_differ_from_cloudy_only_through_the_data(csp):
    """Boost [O III] 5007 in the data by 3x: its posterior flux follows the data (within
    4 sigma of 3 F_cloudy), and ln L prefers the marginalised model over the grid lines."""
    model, lh = _model(csp, width=0.0)
    th = _theta(model)
    es = model._eline_system
    j = es.names.index("[O III] 5007")
    proj = model.obs_dict["spec"]._proj
    F = np.asarray(eline_line_fluxes(model, th, lh)["cloudy"])
    extra = 2.0 * F[j] * np.asarray(proj.line_basis(T.SIGMA_GAS, proj.opz_ref))[:, es.fit_pos[j]]
    spec = model.obs_dict["spec"]
    spec.flux = spec.flux + jnp.asarray(extra)
    post = eline_line_fluxes(model, th, lh)
    mean, sd = np.asarray(post["mean"]), np.asarray(post["sd"])
    assert abs(mean[j] - 3.0 * F[j]) < 4.0 * sd[j]
    std, lh_std = _model(csp, marg=False)
    std.obs_dict["spec"].flux = spec.flux
    assert float(_lnl(model, lh)(th)) > float(_lnl(std, lh_std)(_theta(std)))


def test_fixed_and_ignored_lines(csp):
    fit = ["Ba-beta 4861", "[O III] 5007", "Ba-alpha 6563"]
    model, lh = _model(csp, fit=fit, fix=["[O III] 4959"], ignore=["[S II] 6716"])
    es = model._eline_system
    assert list(es.names) == fit
    assert es.ignored == ("[S II] 6716",)
    rows = T._rows(csp, [6718.29])
    assert es.keep_grid[rows[0]] == 0.0                 # ignored: removed everywhere
    assert es.keep_grid[T._rows(csp, [4960.295])[0]] == 1.0  # fixed: CLOUDY flux kept


def test_jit_vmap_production_width_and_gradients(csp):
    model, lh = _model(csp, width=0.2)
    f = _lnl(model, lh)
    th = _theta(model)
    x0 = float(th["logmass"][0])
    xs = x0 + jnp.linspace(-0.2, 0.2, W_PROD)
    batch = {k: jnp.stack([v] * W_PROD) for k, v in th.items()}
    batch["logmass"] = xs[:, None]
    vm = np.asarray(jax.jit(jax.vmap(f))(batch))
    for i in (0, 37, W_PROD - 1):
        # 3e-7: the CSP and photometry have float32 stages whose summation order vmap changes;
        # the ordinary likelihood shows the same (8.7e-9 measured at this theta); 1.35e-7
        # measured here with the default (dust-free, ZAU_ND) CLOUDY grid, < 1e-7 with ZAU_WD
        assert vm[i] == pytest.approx(float(f(dict(th, logmass=jnp.array([xs[i]])))),
                                      rel=3e-7, abs=0.0)
    g = float(jax.grad(lambda x: f(dict(th, logmass=jnp.array([x]))))(x0))
    h = 1e-4            # mass enters as float32(10**logmass): steps must exceed its resolution
    fd = (float(f(dict(th, logmass=jnp.array([x0 + h]))))
          - float(f(dict(th, logmass=jnp.array([x0 - h]))))) / (2 * h)
    # the full model has float32 stages: its central differences scatter by ~1e-3 over
    # h = 1e-3..1e-5, for the ordinary likelihood as well (measured: AD 4757.76, FD 4731.5 /
    # 4757.4 / 4760.8); the float64 parts are checked exactly in test_kernel_gradients
    assert np.isfinite(g) and g == pytest.approx(fd, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("flat", [(True, True), (False, False), (True, False)])
def test_kernel_gradients(flat):
    """Forward- and reverse-mode derivatives of the kernel with respect to the residual, the
    design, the weights, the prior mean and width: jax.test_util.check_grads (float64)."""
    from jax.test_util import check_grads
    r, A, w, ln0 = _synthetic(2, seed=7)
    flat = np.asarray(flat)

    def f(r, A, w, mean, sd):
        return eline_marginal_loglike([(r, A, w, ln0)], mean, sd, flat)
    args = tuple(map(jnp.asarray, (r, A, w, np.array([1.5, -0.2]), np.array([0.4, 0.5]))))
    check_grads(f, args, order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)


def test_gradient_in_redshift_spectrum_only(csp):
    """d lnZ / d zred through the line basis and the kernel, in float64 (continuum held at
    its value): AD == central differences.  The full model is checked for finiteness; its
    continuum has float32 stages and a piecewise-linear dependence on z (free-z projector)."""
    model, lh = _model(csp, width=0.0, phot=False, lines=False,
                       zred_prior=Uniform(low=0.29, high=0.31))
    th = _theta(model)
    es = model._eline_system
    spec = model.obs_dict["spec"]
    preds, aux = model.predict_with_elines(th)
    r = np.where(np.asarray(spec.mask), np.asarray(spec.flux) - np.asarray(preds["spec"]), 0.0)
    w = np.where(np.asarray(spec.mask), 1.0 / np.asarray(spec.uncertainty) ** 2, 0.0)
    proj = spec._proj

    def lnz(z):
        A = proj.line_basis(T.SIGMA_GAS, 1.0 + z)[:, es.fit_pos]
        return eline_marginal_loglike([(jnp.asarray(r), A, jnp.asarray(w), 0.0)],
                                      aux["prior_mean"], aux["prior_mean"], es.is_flat)
    z0 = 0.3000123
    g = float(jax.grad(lnz)(z0))
    h = 1e-7
    fd = (float(lnz(z0 + h)) - float(lnz(z0 - h))) / (2 * h)
    assert np.isfinite(g) and g == pytest.approx(fd, rel=1e-5, abs=0.0)
    f = _lnl(model, lh)
    assert np.isfinite(float(jax.grad(lambda z: f(dict(th, zred=jnp.array([z]))))(z0)))


def test_eline_delta_zred_shifts_all_lines(csp):
    """eline_delta_zred moves every painted line to (1 + z + dz) lambda (D3)."""
    model, lh = _model(csp, width=0.0, extra_transforms={
        "eline_delta_zred": lambda th: jnp.array([2e-4])})
    th = _theta(model)
    preds, aux = model.predict_with_elines(th)
    proj = model.obs_dict["spec"]._proj
    A = np.asarray(aux["cols"]["spec"])
    j = model._eline_system.names.index("Ba-alpha 6563")
    col = A[:, j] / T.WAVE_OBS
    centre = np.sum(col * np.log(T.WAVE_OBS)) / np.sum(col)
    assert centre == pytest.approx(np.log(6564.723 * (1 + T.ZRED + 2e-4)), abs=2e-7)


# ---------------------------------------------------------------------------
# 5. refusals, all at construction
# ---------------------------------------------------------------------------

def _noneb_model(csp0, flux=None, phot_flux=None, **spec_kw):
    """Spectrum + Photometry on a CSPBasis without a nebular model (add_neb=False)."""
    t = np.array(csp0.sfh_times)
    unc = None if flux is None else 0.03 * np.median(flux) * np.ones_like(flux)
    obs = [Spectrum(wavelength=T.WAVE_OBS, flux=flux, uncertainty=unc, name="spec",
                    instrument=Instrument.R_fwhm(T.R_FWHM), marginalize_elines=True, **spec_kw),
           Photometry(filters=T.FILTERS, flux=phot_flux, name="phot",
                      uncertainty=None if phot_flux is None else 0.05 * np.abs(phot_flux))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp0, obs, zred=T.ZRED, kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS),
                        transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
                        free_param_init={"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.0])})


def test_without_a_nebular_model_lines_come_from_emlines_info():
    """add_neb=False: the lines come from $SPS_HOME/data/emlines_info.dat with a flat prior,
    and injected lines are recovered exactly, jointly over Spectrum and Photometry."""
    from ceridwen.likelihood.eline_marginal import line_table_for
    if not __import__("os").environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (the line list is $SPS_HOME/data/emlines_info.dat)")
    csp0 = T._csp(add_neb=False)
    gen = _noneb_model(csp0)
    es = gen._eline_system
    assert not es.has_grid and np.all(es.is_flat)
    tab = line_table_for(csp0)
    for n in ("Ba-beta 4861", "[O III] 5007", "Ba-alpha 6563"):
        assert es.wave_rest[es.names.index(n)] == tab["wave"][tab["names"].index(n)]
    th = {k: jnp.asarray(v) for k, v in gen.theta_init.items()}
    th["logzsol"] = jnp.array([-0.3])
    pred, aux = gen.predict_with_elines(th)
    F = np.abs(np.random.default_rng(2).normal(1.0, 0.5, es.m)) * 1e-17
    y = np.asarray(pred["spec"]) + np.asarray(aux["cols"]["spec"]) @ F
    yp = np.asarray(pred["phot"], float) + np.asarray(aux["cols"]["phot"]) @ F
    model = _noneb_model(csp0, y, yp)
    assert model._eline_system.static is not None
    lh = MultiObservationLikelihood(keys=tuple(model.obs_dict), likelihoods=tuple(
        _likelihood_for(o, model.param_names) for o in model.observations))
    post = eline_line_fluxes(model, th, lh)
    assert np.all(np.abs(np.asarray(post["mean"]) - F) < 1e-6 * np.asarray(post["sd"]))


@pytest.mark.parametrize("kw,match", [
    (dict(eline_prior_width=0.2), r"no nebular model \(add_neb=False\).*flat prior"),
    (dict(elines_to_fix=["[O III] 5007"]), "elines_to_fix.*no nebular model"),
])
def test_without_a_nebular_model_refuses_what_needs_cloudy_fluxes(kw, match):
    csp0 = T._csp(add_neb=False)
    with pytest.raises(ValueError, match=match):
        _noneb_model(csp0, **kw)


def test_refuses_sampled_nebular_parameters(csp):
    with pytest.raises(ValueError, match=r"nebular parameter\(s\) \['gas_logu', 'gas_logz'\] sampled"):
        _model(csp, sample_gas=True)


def test_refusals_happen_before_any_trace(csp, monkeypatch):
    """The guards raise in SedModel construction: no jit/trace is ever entered."""
    traced = []
    monkeypatch.setattr(jax, "jit", lambda *a, **k: traced.append(1) or (lambda *x: None))
    with pytest.raises(ValueError):
        _model(csp, sample_gas=True)
    assert traced == []


@pytest.mark.parametrize("kw,err,match", [
    (dict(instrument=None), ValueError, "needs the instrument"),
    (dict(eline_sigma=100.0), TypeError, "no eline_sigma"),
    (dict(logify_spectrum=True), ValueError, "logify_spectrum"),
    (dict(elines_to_fit=["[O III] 5007"], elines_to_fix=["[O III] 5007"]), ValueError, "in both"),
    (dict(eline_prior_width=-0.1), ValueError, "finite fraction"),
])
def test_spectrum_construction_refusals(kw, err, match):
    base = dict(wavelength=T.WAVE_OBS, flux=np.ones_like(T.WAVE_OBS),
                uncertainty=np.ones_like(T.WAVE_OBS), instrument=Instrument.R_fwhm(1000.0),
                marginalize_elines=True)
    base.update(kw)
    with pytest.raises(err, match=match):
        Spectrum(**base)


def test_options_without_marginalisation_are_refused():
    with pytest.raises(ValueError, match="only acts with marginalize_elines=True"):
        Spectrum(wavelength=T.WAVE_OBS, instrument=Instrument.R_fwhm(1000.0),
                 elines_to_fit=["[O III] 5007"])


def test_unknown_line_name_suggests_the_fsps_name(csp):
    with pytest.raises(ValueError, match=r"did you mean.*\[O III\] 5007"):
        _model(csp, fit=["[OIII] 5007"])


def test_explicit_line_outside_the_spectrum_is_refused(csp):
    with pytest.raises(ValueError, match=r"\[O II\] 3726 \(centre outside the spectrum\)"):
        _model(csp, fit=["[O II] 3726", "[O III] 5007"])


def test_only_one_marginalising_spectrum(csp):
    with pytest.raises(ValueError, match="only one marginalising Spectrum"):
        _model(csp, spectra=2)


def test_photometry_with_sampled_redshift_is_refused(csp):
    with pytest.raises(NotImplementedError, match="sampled zred"):
        _model(csp, zred_prior=Uniform(low=0.29, high=0.31))


def test_degenerate_lines_are_refused():
    class _Spec:
        name = "spec"
        uncertainty = np.ones(50)

    class _Proj:
        opz_ref = 1.0
        def line_basis(self, s, opz):
            c = np.exp(-0.5 * (np.arange(50) - 25.0) ** 2 / 9.0)
            return np.stack([c, c * (1 + 1e-9)], axis=1)
    es = ElineSystem(spec_key="spec", phot_keys=(), lines_keys=(), fit_rows=np.arange(2),
                     fit_pos=np.arange(2), names=("a", "b"), wave_rest=np.ones(2),
                     is_flat=np.ones(2, bool), prior_width=0.0, keep_grid=np.ones(2))
    with pytest.raises(ValueError, match="nearly degenerate.*'a' / 'b'"):
        _check_conditioning(es, _Spec(), _Proj(), 80.0, np.ones(50, bool))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-ra"]))


def test_fitsed_writes_line_fluxes_and_postprocess_uses_them(csp, tmp_path):
    """fitSED (nested, tiny run) stores /elines per draw; read_result_h5 returns them equal to a
    direct evaluation; PostProcess predictions carry each draw's posterior-mean lines."""
    from ceridwen import fitSED, read_result_h5
    from ceridwen.postprocess import PostProcess
    from ceridwen.sampler.priors import Normal
    model, lh = _model(csp, width=0.0)
    model.priors.update({"logmass": Uniform(low=8.8, high=9.2),
                         "logsfr_ratios": Normal(mean=0.0, sigma=0.3)})
    for name in model.param_names:          # every other CSP key is sampled too: narrow boxes
        if name not in model.priors:
            v = float(np.ravel(np.asarray(model.theta_init[name]))[0])
            model.priors[name] = Uniform(low=v - 0.05, high=v + 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fitSED(model, output_dir=tmp_path, rng_key=jax.random.PRNGKey(0), verbose=False,
                     sampler_kwargs={"num_live": 24, "num_delete": 6, "num_inner_steps": 4,
                                     "logZ_tol": -1.0})
    r = read_result_h5(tmp_path / "ceridwen_result.h5")
    el = r["elines"]
    assert el["names"] == list(model._eline_system.names)
    n = r["samples"]["log_likelihoods"].shape[0]
    assert el["mean"].shape == (n, model._eline_system.m)
    i = n // 2
    th = {p: jnp.asarray(np.asarray(res.samples[p][i]).reshape(np.shape(model.theta_init[p])))
          for p in model.theta_init}
    direct = eline_line_fluxes(model, th, lh)
    # stored values come from a vmapped batch: the continuum's float32 stages differ from the
    # single call at ~1e-26 per pixel, i.e. ~1e-5 of the line-flux sd (measured)
    sd = np.asarray(direct["sd"])
    assert np.all(np.abs(el["mean"][i] - np.asarray(direct["mean"])) < 1e-3 * sd)
    np.testing.assert_allclose(el["sd"][i], sd, rtol=1e-8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = PostProcess(model, res, n_samples=4, seed=0).run()
    e = out["extras"]["elines"]
    assert e["mean"].shape == (4, model._eline_system.m)
    spec_pred = out["prediction"]["spectra"]["spec"]
    assert spec_pred.shape == (4, T.WAVE_OBS.size) and np.all(np.isfinite(spec_pred))


@pytest.mark.parametrize("width,phot,lines", [(0.0, False, False), (0.0, True, True),
                                               (0.2, False, False), (0.2, True, True)])
def test_static_fast_path_equals_general_path(csp, width, phot, lines):
    """The precomputed path (fixed z, width, noise and calibration) gives the same ln L and
    gradient as the per-call path, flat and Gaussian priors, with and without Photometry/Lines."""
    model, lh = _model(csp, width=width, phot=phot, lines=lines)
    assert model._eline_system.static is not None
    th = _theta(model)
    f_fast = _lnl(model, lh)
    a = float(f_fast(th))
    ga = float(jax.grad(lambda x: f_fast(dict(th, logmass=jnp.array([x]))))(th["logmass"][0]))
    static = model._eline_system.static
    model._eline_system.static = None
    try:
        f_gen = _lnl(model, lh)
        b = float(f_gen(th))
        gb = float(jax.grad(lambda x: f_gen(dict(th, logmass=jnp.array([x]))))(th["logmass"][0]))
    finally:
        model._eline_system.static = static
    assert a == pytest.approx(b, rel=1e-11, abs=0.0)
    assert ga == pytest.approx(gb, rel=1e-6, abs=0.0)


def test_static_fast_path_is_off_when_the_weights_depend_on_the_model(csp):
    t = np.array(csp.sfh_times)
    y = np.asarray(_truth(csp)["spec"])
    spec = Spectrum(wavelength=T.WAVE_OBS, flux=y, uncertainty=0.03 * np.median(y) * np.ones_like(y),
                    instrument=Instrument.R_fwhm(T.R_FWHM), name="spec", marginalize_elines=True,
                    noise_floor=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, [spec], zred=T.ZRED, kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS),
                         transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t),
                                     "gas_logu": lambda th: jnp.array([-2.23]),
                                     "gas_logz": lambda th: jnp.array([-0.37])},
                         free_param_init={"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.0])})
    assert model._eline_system.static is None


# ---------------------------------------------------------------------------
# alpha-enhanced basis: no nebular grid, line list from emlines_info.dat, flat prior
# ---------------------------------------------------------------------------

def _afe_csp():
    grid = T.find_test_grid()
    grid = None if grid is None else grid.parent / "amist_c3k_lr_chab_afe.h5"
    if grid is None or not grid.is_file():
        pytest.skip("alpha-enhanced grid (amist_c3k_lr_chab_afe.h5) not found")
    if not __import__("os").environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (emlines_info.dat)")
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe
    from ceridwen import Cosmology
    return CSPBasis_afe(SSPDataAfe.load(str(grid)), lookback_time=jnp.linspace(0.0, 12.0, 5),
                        zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                        verbose=False, cosmo=Cosmology.planck18())


AFE_Z = 0.1
AFE_WAVE = np.exp(np.arange(np.log(4750 * (1 + AFE_Z)), np.log(6800 * (1 + AFE_Z)), 1 / (2.3548 * 500) / 2.5))


def _afe_model(csp, flux=None, phot_flux=None, **spec_kw):
    unc = None if flux is None else 0.02 * np.median(flux) * np.ones_like(flux)
    obs = [Spectrum(wavelength=AFE_WAVE, flux=flux, uncertainty=unc, name="spec",
                    instrument=Instrument.R_fwhm(500.0), marginalize_elines=True, **spec_kw),
           Photometry(filters=["sdss_g0", "sdss_r0", "sdss_i0"], flux=phot_flux,
                      uncertainty=None if phot_flux is None else 0.03 * np.abs(phot_flux), name="phot")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, obs, zred=AFE_Z, kinematics=Kinematics(sigma_gal=200.0, sigma_gas=150.0),
                        free_param_init={"logmass": jnp.array([10.5])})


def test_afe_photometric_profiles_equal_the_nebular_model(csp):
    """line_profiles_on_grid (used without a nebular grid) == NebularModel.line_profiles."""
    from ceridwen.likelihood.eline_marginal import line_profiles_on_grid
    pos = np.asarray(csp.neb.nebem_line_pos)
    for s_kms in (0.0, 150.0):
        np.testing.assert_allclose(line_profiles_on_grid(np.asarray(csp.wave), pos, s_kms),
                                   csp.neb.line_profiles(s_kms), rtol=1e-12, atol=0.0)


def test_afe_marginalisation_recovers_injected_lines():
    """CSPBasis_afe: the line list comes from emlines_info.dat, the prior is flat, and a
    noise-free mock (alpha-enhanced continuum + injected lines, spectrum and photometry) is
    recovered exactly, jointly."""
    from ceridwen.likelihood.eline_marginal import line_table_for
    acsp = _afe_csp()
    gen = _afe_model(acsp)
    es = gen._eline_system
    assert not es.has_grid and np.all(es.is_flat)
    tab = line_table_for(acsp)
    for n in ("Ba-beta 4861", "[O III] 5007", "Ba-alpha 6563", "[N II] 6584", "[S II] 6716"):
        assert n in es.names
        assert es.wave_rest[es.names.index(n)] == tab["wave"][tab["names"].index(n)]
    th = {k: jnp.asarray(v) for k, v in gen.theta_init.items()}
    th["afe"] = jnp.array([0.3])
    pred, aux = gen.predict_with_elines(th)
    rng = np.random.default_rng(4)
    F = np.abs(rng.normal(1.0, 0.5, es.m)) * 3e-17
    y = np.asarray(pred["spec"]) + np.asarray(aux["cols"]["spec"]) @ F
    yp = np.asarray(pred["phot"], float) + np.asarray(aux["cols"]["phot"]) @ F
    model = _afe_model(acsp, y, yp)
    lh = MultiObservationLikelihood(keys=tuple(model.obs_dict), likelihoods=tuple(
        _likelihood_for(o, model.param_names) for o in model.observations))
    assert model._eline_system.static is not None      # fast path with data attached
    post = eline_line_fluxes(model, th, lh)
    mean, sd = np.asarray(post["mean"]), np.asarray(post["sd"])
    assert np.all(np.abs(mean - F) < 1e-6 * sd)
    assert np.all(np.isnan(np.asarray(post["cloudy"])))            # no grid prediction
    assert np.isfinite(float(lh.make_lnprobfn(model.obs_dict, model, _NoPrior())(th)))


@pytest.mark.parametrize("kw,match", [
    (dict(eline_prior_width=0.2), "CSPBasis_afe has no nebular grid.*flat prior"),
    (dict(elines_to_fix=["[O III] 5007"]), "elines_to_fix.*no nebular grid"),
])
def test_afe_refuses_what_needs_the_cloudy_fluxes(kw, match):
    acsp = _afe_csp()
    with pytest.raises(ValueError, match=match):
        _afe_model(acsp, **kw)


def test_afe_line_list_needs_sps_home(monkeypatch):
    acsp = _afe_csp()
    monkeypatch.setattr(acsp, "sps_home", None)
    monkeypatch.delenv("SPS_HOME", raising=False)
    with pytest.raises(ValueError, match="emlines_info.dat.*SPS_HOME"):
        _afe_model(acsp)


def test_static_fast_path_refuses_a_mask_changed_after_setup(csp):
    """B2-015: the precomputed weights froze the masks at setup_observations; a mask changed
    afterwards is refused (it was silently ignored by the marginalised block), and
    setup_observations() rebuilds the system with the new mask."""
    model, lh = _model(csp)
    assert model._eline_system.static is not None
    spec = model.obs_dict["spec"]
    spec.mask_wavelength_range(float(spec.wavelength[10]), float(spec.wavelength[20]))
    with pytest.raises(ValueError, match="changed its mask or uncertainty"):
        _lnl(model, lh)(_theta(model))
    model.setup_observations()
    assert np.isfinite(float(_lnl(model, lh)(_theta(model))))
