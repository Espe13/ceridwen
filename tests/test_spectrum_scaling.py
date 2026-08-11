"""Spectrophotometric normalisation ``spectrum_scaling`` and its independence from
the emission-line aperture correction ``eline_scaling``.

``spectrum_scaling`` (added 2026-08) is a scalar multiplicative recalibration applied
to ``Spectrum`` predictions ONLY, inside ``CSPBasis._project_observations``
(and the ``CSPBasis_afe`` twin). It rescales the model spectrum onto the
photometric flux scale; photometry (and lines) are left untouched so they
anchor the absolute flux. It must be fully independent of ``eline_scaling``,
which scales the emission-LINE component only.

These are pure construction/projection tests: they load a committed SSP grid
(skipped if unavailable) but need no sampler and run on CPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen import SSPData, CSPBasis
from ceridwen.observation import Photometry, Spectrum

FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.linspace(4000.0, 8000.0, 120)     # observed-frame vacuum A
ZRED = 0.1


def _base_theta(csp):
    """A minimal, valid theta for csp.predict: grid defaults + mass + zred."""
    theta = dict(csp.all_params)
    theta["logmass"] = jnp.array([10.0])
    theta["zred"] = jnp.array([ZRED])
    return theta


def _build_csp(ssp_grid_path):
    ssp = SSPData.load(ssp_grid_path)
    return CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, 5),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=False,
        sigma_losvd_kms=0.0, verbose=False,
    )


def _setup(obs_list, csp):
    """Bind every observation to the model grid.

    ``csp.predict`` requires each Observation to have had
    ``setup_for_model`` called first (``SedModel.__init__`` does this
    automatically; here we call ``csp.predict`` directly, so we must do it
    ourselves).  ``Spectrum.predict`` hard-raises otherwise.
    """
    for o in obs_list:
        if hasattr(o, "setup_for_model"):
            o.setup_for_model(csp.wave, zred=ZRED)
    return obs_list


def test_spectrum_scaling_scales_spectrum_not_photometry(ssp_grid_path):
    csp = _build_csp(ssp_grid_path)
    phot = Photometry(filters=FILTERS, name="phot")
    spec = Spectrum(wavelength=SPEC_WAVE, resolution=100.0,
                    smoothtype="vel", name="spec")
    obs = _setup([phot, spec], csp)

    theta = _base_theta(csp)
    pred0 = csp.predict(theta, obs)

    factor = 0.5
    theta_s = dict(theta); theta_s["spectrum_scaling"] = jnp.array([factor])
    pred1 = csp.predict(theta_s, obs)

    # Spectrum scales by exactly spectrum_scaling ...
    np.testing.assert_allclose(np.asarray(pred1["spec"]),
                               factor * np.asarray(pred0["spec"]), rtol=1e-5)
    # ... photometry is untouched (anchors the absolute flux).
    np.testing.assert_allclose(np.asarray(pred1["phot"]),
                               np.asarray(pred0["phot"]), rtol=1e-6)


def test_spectrum_scaling_absent_is_identity(ssp_grid_path):
    csp = _build_csp(ssp_grid_path)
    spec = Spectrum(wavelength=SPEC_WAVE, resolution=100.0,
                    smoothtype="vel", name="spec")
    _setup([spec], csp)
    theta = _base_theta(csp)
    pred_no = csp.predict(theta, [spec])
    pred_one = csp.predict({**theta, "spectrum_scaling": jnp.array([1.0])}, [spec])
    np.testing.assert_allclose(np.asarray(pred_no["spec"]),
                               np.asarray(pred_one["spec"]), rtol=1e-6)


def test_eline_scaling_does_not_touch_the_spectrum(ssp_grid_path):
    """eline_scaling must never rescale the Spectrum (decoupled from spectrum_scaling).

    With no nebular module the emission-line component is identically zero, so
    the continuum spectrum must be bit-for-bit independent of eline_scaling.
    """
    csp = _build_csp(ssp_grid_path)
    spec = Spectrum(wavelength=SPEC_WAVE, resolution=100.0,
                    smoothtype="vel", name="spec")
    _setup([spec], csp)
    theta = _base_theta(csp)
    pred0 = csp.predict(theta, [spec])
    pred_e = csp.predict({**theta, "eline_scaling": jnp.array([0.3])}, [spec])
    np.testing.assert_allclose(np.asarray(pred0["spec"]),
                               np.asarray(pred_e["spec"]), rtol=1e-6)


# --------------------------------------------------------------------------
# Alpha-enhanced twin: spectrum_scaling must work identically in CSPBasis_afe.
# --------------------------------------------------------------------------
def _find_afe_grid():
    here = pathlib.Path(__file__).resolve().parent
    cands = [
        here.parent / "ceridwen" / "data" / "test_data"
        / "amist_c3k_lr_chab_afe.h5",
        here / "fixtures" / "amist_c3k_lr_chab_afe.h5",
        here.parent / "examples" / "amist_c3k_lr_chab_afe.h5",
    ]
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def test_spectrum_scaling_in_csp_afe():
    grid = _find_afe_grid()
    if grid is None:
        pytest.skip("alpha-enhanced grid (amist_c3k_lr_chab_afe.h5) not found")
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe

    ssp = SSPDataAfe.load(grid)
    csp = CSPBasis_afe(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, 5),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True,
        sigma_losvd_kms=0.0, verbose=False,
    )
    phot = Photometry(filters=FILTERS, name="phot")
    spec = Spectrum(wavelength=SPEC_WAVE, resolution=100.0,
                    smoothtype="vel", name="spec")
    obs = _setup([phot, spec], csp)

    theta = dict(csp.all_params)
    theta["logmass"] = jnp.array([10.0])
    theta["zred"] = jnp.array([ZRED])
    theta["afe"] = jnp.array([0.3])

    pred0 = csp.predict(theta, obs)
    factor = 0.7
    pred1 = csp.predict({**theta, "spectrum_scaling": jnp.array([factor])}, obs)

    np.testing.assert_allclose(np.asarray(pred1["spec"]),
                               factor * np.asarray(pred0["spec"]), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(pred1["phot"]),
                               np.asarray(pred0["phot"]), rtol=1e-6)
