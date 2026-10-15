"""The dtype contract of docs/conventions.md ("Precision"): spectra, curves and projections are
float32 work; photometry predictions are float32; spectrum predictions come back float64 (the
float32 projection scaled by an exact power of two); line fluxes (a handful of numbers) and the
likelihood's arithmetic are float64.

Asserted at each boundary: the SSP cube, the attenuation curves, the model-grid spectrum, the
photometry matrix, the line apertures, ``predict_line_fluxes``, ``SedModel.predict`` (spec,
phot, lines) and ``PostProcess``'s prediction arrays; and a finite gradient through the float32
photometric broadening far from the data. The float64 data and likelihood are tested in
``test_observation_dtype.py`` and ``likelihood/test_likelihood_dtype.py``.

Grid-dependent tests skip without the test SSP grid; the nebular ones also without $SPS_HOME.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology            # noqa: E402
from ceridwen.broadening import Kinematics, Instrument                  # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh             # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines           # noqa: E402

F32, F64 = np.dtype(np.float32), np.dtype(np.float64)
ZRED = 0.8
N_TIME = 5
LINES = {"Hbeta": 4862.71, "OIII5007": 5008.24, "Halpha": 6564.61}


def _model(ssp_grid_path, add_neb):
    ssp = SSPData.load(str(ssp_grid_path))
    cosmo = Cosmology.planck18()
    kw = dict(lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME), zh_const=True,
              add_dust=True, add_diffuse_dust=True, add_neb=add_neb, add_igm=True,
              add_dust_emission=False, verbose=False, cosmo=cosmo)
    if add_neb:
        kw["sps_home"] = os.environ["SPS_HOME"]
    csp = CSPBasis(ssp, **kw)
    t_yr = np.array(csp.sfh_times)
    wave = np.linspace(3800.0, 8800.0, 600) * (1.0 + ZRED)
    obs = [Photometry(filters=["sdss_g0", "sdss_r0", "sdss_i0"], flux=[2e-9] * 3,
                      uncertainty=[2e-10] * 3, name="phot"),
           Spectrum(wavelength=wave, flux=np.full(wave.size, 1e-29),
                    uncertainty=np.full(wave.size, 1e-30),
                    instrument=Instrument.sigma_kms(150.0), name="spec")]
    if add_neb:
        obs.append(Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                         wavelength=np.array(list(LINES.values())),
                         flux=np.full(len(LINES), 1e-17),
                         uncertainty=np.full(len(LINES), 1e-18), name="lines"))
    def sfh(th):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=t_yr)
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    with warnings.catch_warnings():           # no priors: flat, as the bit-identity harness
        warnings.simplefilter("ignore")
        model = SedModel(csp, obs, priors={}, transforms={"sfh": sfh}, free_param_init=init,
                         zred=ZRED, kinematics=Kinematics(sigma_gal=250.0))
    return model


def _theta(model):
    return {k: jnp.asarray(v) for k, v in model.theta_init.items()}


def _check_grid_and_predictions(model):
    csp = model.csp
    th = _theta(model)
    t = model.apply_transforms(th)
    t["zred"] = jnp.array([ZRED])
    assert csp.flux.dtype == F32                                   # SSP cube
    attn, attn_diffuse = csp.attenuate_dust(csp._wave32, t)        # attenuation curves
    assert attn.dtype == F32 and attn_diffuse.dtype == F32
    assert csp.get_spectrum(t, include_lines=True).dtype == F32    # model grid
    phot = model.obs_dict["phot"]
    assert np.asarray(phot._T).dtype == F32                        # photometry matrix
    pred = model.predict(th)
    for k, v in pred.items():                                      # predict/*
        # spectra: float32 work returned in float64 (broadening._scaled_to_float64); line fluxes
        # float64; photometry float32
        assert v.dtype == (F32 if k == "phot" else F64), (k, v.dtype)
    return t, pred


def test_float32_boundaries_plain(ssp_grid_path):
    model = _model(ssp_grid_path, add_neb=False)
    _, pred = _check_grid_and_predictions(model)
    assert set(pred) == {"phot", "spec"}


@pytest.mark.fsps
def test_float32_boundaries_nebular_and_postprocess(ssp_grid_path):
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    from ceridwen.postprocess import PostProcess
    from ceridwen.sampler.runner import SamplingResult

    model = _model(ssp_grid_path, add_neb=True)
    t, pred = _check_grid_and_predictions(model)
    assert set(pred) == {"phot", "spec", "lines"}
    assert model.csp.predict_line_fluxes(t).dtype == F64           # line fluxes (small)
    lines = model.obs_dict["lines"]
    assert np.asarray(lines._W).dtype == F32                       # line apertures

    th = _theta(model)
    n = 3
    samples = {k: jnp.repeat(jnp.atleast_1d(v)[None], n, axis=0) for k, v in th.items()}
    samples = {k: (v[:, 0] if v.shape[1:] == (1,) else v) for k, v in samples.items()}
    ll = jnp.arange(n, dtype=jnp.float64)
    res = SamplingResult(samples=samples, log_evidence=float("nan"),
                         log_evidence_err=float("nan"), log_weights=jnp.zeros(n),
                         log_likelihoods=ll, param_names=list(model.param_names),
                         n_likelihood_calls=n, wall_time_s=0.0, sampler_name="fake",
                         log_likelihoods_birth=ll - 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = PostProcess(model, res, n_samples=n, seed=0, batch_size=2).run()
    p = out["prediction"]
    for slot, name, dt in (("spectra", "spec", F64), ("photometry", "phot", F64),
                           ("lines", "lines", F64)):     # stored float64 (postprocess.py)
        assert np.asarray(p[slot][name]).dtype == dt, (slot, p[slot][name].dtype)
    for key in ("spectra_model", "spectra_observed", "spectra_intrinsic"):
        assert np.asarray(p[key]).dtype == F32, key


def test_gradient_finite_with_broadened_photometry_at_cgs_scale(ssp_grid_path):
    """The NUTS/MAP path: lnL and its gradient through float32 photometric broadening (default
    kinematics, 300 km/s) at a theta far from the data, as a prior draw at the start of a fit
    can be: a young population at z = 2 predicting ~1e-1 maggies against data of 2e-9 +- 2e-10,
    so residual / sigma^2 is ~1e18 per band and, through the photometry matrix, the FFT's
    backward pass overflows float32. Before the FFT scaling (broadening.make_gaussian_fft)
    every gradient here was NaN."""
    if not os.environ.get("SPS_HOME"):          # CSPBasis builds its nebular model by default
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    ssp = SSPData.load(str(ssp_grid_path))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 0.5, 5), zh_const=True, verbose=False,
                   cosmo=Cosmology.planck18(), add_igm=True)
    obs = [Photometry(filters=["jwst_f090w", "jwst_f150w", "jwst_f444w", "jwst_f770w"],
                      flux=[2e-9] * 4, uncertainty=[2e-10] * 4, name="phot")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, obs, priors={}, free_param_init={"logmass": jnp.array([10.0])},
                         zred=2.0)
    assert model.obs_dict["phot"]._broadener is not None
    th = _theta(model)
    assert np.min(np.asarray(model.predict(th)["phot"])) > 1e6 * 2e-9          # far away

    def lnl(t):
        return -0.5 * jnp.sum(((model.predict(t)["phot"] - 2e-9) / 2e-10) ** 2)

    grad = jax.jit(jax.grad(lnl))(th)
    for k, g in grad.items():
        assert np.all(np.isfinite(np.asarray(g))), k
