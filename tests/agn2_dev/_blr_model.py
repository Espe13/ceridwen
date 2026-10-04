"""Shared model builder of the BLR development tests (session BLR)."""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_TESTS = pathlib.Path(__file__).resolve().parents[1]
for p in (_TESTS, _TESTS / "agn2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import _ref  # noqa: E402  (tests/agn2/_ref.py: loud skips, FWHM, nu integral)
import test_emission_lines as T  # noqa: E402

from ceridwen import CSPBasis, Cosmology, SSPData, SedModel  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics  # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh  # noqa: E402
from ceridwen.observation import Lines, Photometry, Spectrum  # noqa: E402

ZRED = 0.3
HA, HB = 6564.7229, 4862.7629
R_FWHM = 2000.0
WAVE_OBS = np.exp(np.arange(np.log(5600 * (1 + ZRED)), np.log(7600 * (1 + ZRED)),
                            1.0 / (2.3548 * R_FWHM) / 2.5))
FILTERS = ["sdss_i0", "sdss_z0"]


def need(test):
    if T.find_test_grid() is None:
        _ref.missing("test SSP grid", test)
    if not os.environ.get("SPS_HOME"):
        _ref.missing("$SPS_HOME", test)


_CACHE = {}


def build(test, mode="corrected", duste=False, dust4=False, sigma_max=2000.0, phot=True):
    """(model, csp, spec) with a Spectrum (R = 2000 around H alpha), Lines (H alpha, H beta)
    and, when ``phot``, Photometry; BLR only (no torus, disc or narrow lines)."""
    need(test)
    key = (mode, duste, dust4, sigma_max, phot)
    if key in _CACHE:
        return _CACHE[key]
    cosmo = Cosmology.planck18()
    if mode == "corrected":
        cfg = AGNConfig(agn_mode="corrected", torus=False, disk=False, lines=False, blr=True,
                        dust4_type=1 if dust4 else 0)
    else:
        cfg = AGNConfig(agn_mode="cigale", torus=False, blr=True)
    csp = CSPBasis(SSPData.load(str(T.find_test_grid())),
                   lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(ZRED))]),
                   zh_const=True, sfh_interp="step", add_dust=duste, add_diffuse_dust=True,
                   add_neb=True, add_igm=False, add_dust_emission=duste, verbose=False,
                   cosmo=cosmo, sps_home=os.environ["SPS_HOME"], agn=cfg)
    spec = Spectrum(wavelength=WAVE_OBS, flux=np.ones_like(WAVE_OBS),
                    uncertainty=np.ones_like(WAVE_OBS), instrument=Instrument.R_fwhm(R_FWHM),
                    name="spec")
    lines = Lines(line_ind=np.arange(2), wavelength=[HA, HB], line_names=["Halpha", "Hbeta"],
                  flux=np.ones(2), uncertainty=np.ones(2), name="lines")
    obs = [spec, lines]
    if phot:
        obs.append(Photometry(filters=FILTERS, flux=np.ones(len(FILTERS)),
                              uncertainty=np.ones(len(FILTERS)), name="phot"))
    t = np.array(csp.sfh_times)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, obs,
                         transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(
                             th["logsfr_ratios"], sfh_times_yr=_t)},
                         free_param_init={"logsfr_ratios": jnp.zeros(len(t) - 1),
                                          "logmass": jnp.array([9.0])},
                         zred=ZRED, kinematics=Kinematics(sigma_gal=100.0, sigma_gas=60.0,
                                                          sigma_max=sigma_max))
    _CACHE[key] = (model, csp, spec)
    return _CACHE[key]


def theta(model, **kw):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th["logzsol"] = jnp.array([-0.3])
    th.update({k: jnp.array([float(v)]) for k, v in kw.items()})
    return th


def mtheta(model, th):
    mt = dict(model.apply_transforms(th))
    mt["zred"] = jnp.array([ZRED])
    return mt


def broad(model, **kw):
    """BLR-only part of every prediction: predict(blr_lum) - predict(blr_lum = 0), float64."""
    lum = kw.pop("blr_lum", 1.0)
    on = model.predict(theta(model, blr_lum=lum, **kw))
    off = model.predict(theta(model, blr_lum=0.0, **kw))
    return {k: np.asarray(on[k], np.float64) - np.asarray(off[k], np.float64) for k in on}


def expected_fluxes(model, csp, **kw):
    """(n_lines,) observed integrated BLR fluxes [erg/s/cm^2] recomputed in NumPy from the
    definitions: blr_lum x ratio x M_formed(W) x 10^logmass x flux factor / (1 + z) x T_gal."""
    lum = kw.get("blr_lum", 1.0)
    mt = mtheta(model, theta(model, **kw))
    msum = float(jnp.sum(csp.calculate_ssp_weights(theta=mt)))
    dc = np.exp(-np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64))
    blr = csp._agn.model.blr
    t = np.interp(blr.line_wave, np.asarray(csp.wave, np.float64), dc)
    ff = float(csp._flux_factor(mt))
    return lum * blr.ratios * msum * 10 ** float(mt["logmass"][0]) * ff / (1 + ZRED) * t
