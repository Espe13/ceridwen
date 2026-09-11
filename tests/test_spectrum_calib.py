"""Polynomial spectrophotometric calibration ``spectrum_calib``.

``spectrum_calib`` (added 2026-08, ceridwen 0.2.4) is a vector of Legendre
coefficients ``c_1..c_order`` applied multiplicatively to ``Spectrum``
predictions ONLY, inside ``_project_observations`` of ``CSPBasis`` and
``CSPBasis_afe``:

    pred *= spectrum_scaling * (1 + sum_k c_k P_k(x)),
    x = 2 (lam - lam_min)/(lam_max - lam_min) - 1  over obs.wavelength.

These tests pin: the exact factor (against an independent NumPy Legendre
evaluation), photometry invariance, identity when absent / all-zero,
composition with ``spectrum_scaling``, the fitted-sigma_gal path, jit,
and the alpha twin.  They load a committed SSP grid (skipped if unavailable),
need no sampler, and run on CPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen import SSPData, CSPBasis
from ceridwen.observation import Photometry, Spectrum
from ceridwen.broadening import Instrument, Kinematics
from ceridwen.csp.spectrum_calibration import (
    legendre_design_matrix, spectrum_calibration_factor,
)
from ceridwen.cosmology import Cosmology

FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.linspace(4000.0, 8000.0, 120)     # observed-frame vacuum A
ZRED = 0.1
COEFF = np.array([0.05, -0.03, 0.02])


def _expected_factor(wave, coeff):
    """Independent reference: numpy.polynomial.legendre on x in [-1, 1]."""
    x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
    return 1.0 + np.polynomial.legendre.legval(x, np.concatenate([[0.0], coeff]))


def _base_theta(csp):
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
        verbose=False,
        cosmo=Cosmology.planck18(),
    )


def _setup(obs_list, csp):
    for o in obs_list:
        if hasattr(o, "setup_for_model"):
            if isinstance(o, Spectrum):
                o.setup_for_model(
                    csp.wave, zred=ZRED, kinematics=Kinematics.none(),
                    lib_resolution=getattr(csp, "lib_resolution", None))
            else:
                o.setup_for_model(csp.wave, zred=ZRED)
    return obs_list


# --------------------------------------------------------------------------
# Pure helper tests (no grid needed)
# --------------------------------------------------------------------------
def test_design_matrix_matches_numpy_legendre():
    D = legendre_design_matrix(SPEC_WAVE, 3)
    assert D.shape == (SPEC_WAVE.size, 3)
    ref = _expected_factor(SPEC_WAVE, COEFF)
    np.testing.assert_allclose(1.0 + D @ COEFF, ref, rtol=1e-12)
    # P_1 is exactly -1 .. +1 at the pixel-range ends, P_2 = +1 at both.
    np.testing.assert_allclose(D[[0, -1], 0], [-1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(D[[0, -1], 1], [1.0, 1.0], atol=1e-12)


def test_factor_absent_is_none_and_level_only_is_scalar():
    spec = Spectrum(wavelength=SPEC_WAVE, name="spec")
    assert spectrum_calibration_factor(spec, {}) is None
    f = spectrum_calibration_factor(spec, {"spectrum_scaling": jnp.array([0.7])})
    assert np.ndim(f) == 0 and float(f) == pytest.approx(0.7)


def test_factor_shape_and_composition():
    spec = Spectrum(wavelength=SPEC_WAVE, name="spec")
    f = spectrum_calibration_factor(
        spec, {"spectrum_scaling": jnp.array([0.7]),
               "spectrum_calib": jnp.asarray(COEFF)})
    assert f.shape == (SPEC_WAVE.size,)
    np.testing.assert_allclose(np.asarray(f), 0.7 * _expected_factor(SPEC_WAVE, COEFF),
                               rtol=1e-12)


def test_factor_rejects_empty_coefficients():
    spec = Spectrum(wavelength=SPEC_WAVE, name="spec")
    with pytest.raises(ValueError):
        spectrum_calibration_factor(spec, {"spectrum_calib": jnp.zeros(0)})


def test_factor_is_jittable_and_differentiable():
    spec = Spectrum(wavelength=SPEC_WAVE, name="spec")

    def f(c):
        return spectrum_calibration_factor(spec, {"spectrum_calib": c})

    out = jax.jit(f)(jnp.asarray(COEFF))
    np.testing.assert_allclose(np.asarray(out), _expected_factor(SPEC_WAVE, COEFF),
                               rtol=1e-12)
    # d(factor)/d(c_k) = P_k(x): the design matrix itself.
    J = jax.jacobian(f)(jnp.asarray(COEFF))
    np.testing.assert_allclose(np.asarray(J), legendre_design_matrix(SPEC_WAVE, 3),
                               rtol=1e-12)


# --------------------------------------------------------------------------
# Projection tests (grid)
# --------------------------------------------------------------------------
def test_spectrum_calib_scales_spectrum_not_photometry(ssp_grid_path):
    csp = _build_csp(ssp_grid_path)
    phot = Photometry(filters=FILTERS, name="phot")
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(100.0), name="spec")
    obs = _setup([phot, spec], csp)
    theta = _base_theta(csp)
    pred0 = csp.predict(theta, obs)
    pred1 = csp.predict({**theta, "spectrum_calib": jnp.asarray(COEFF)}, obs)

    np.testing.assert_allclose(np.asarray(pred1["spec"]),
                               _expected_factor(SPEC_WAVE, COEFF) * np.asarray(pred0["spec"]),
                               rtol=1e-5)
    np.testing.assert_allclose(np.asarray(pred1["phot"]),
                               np.asarray(pred0["phot"]), rtol=1e-6)


def test_spectrum_calib_zero_is_identity(ssp_grid_path):
    csp = _build_csp(ssp_grid_path)
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(100.0), name="spec")
    _setup([spec], csp)
    theta = _base_theta(csp)
    pred_no = csp.predict(theta, [spec])
    pred_zero = csp.predict({**theta, "spectrum_calib": jnp.zeros(2)}, [spec])
    np.testing.assert_allclose(np.asarray(pred_no["spec"]),
                               np.asarray(pred_zero["spec"]), rtol=1e-6)


def test_spectrum_calib_composes_with_spectrum_scaling(ssp_grid_path):
    csp = _build_csp(ssp_grid_path)
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(100.0), name="spec")
    _setup([spec], csp)
    theta = _base_theta(csp)
    pred0 = csp.predict(theta, [spec])
    pred = csp.predict({**theta, "spectrum_scaling": jnp.array([0.8]),
                        "spectrum_calib": jnp.asarray(COEFF)}, [spec])
    np.testing.assert_allclose(
        np.asarray(pred["spec"]),
        0.8 * _expected_factor(SPEC_WAVE, COEFF) * np.asarray(pred0["spec"]),
        rtol=1e-5)


def test_spectrum_calib_with_fitted_sigma_gal(ssp_grid_path):
    """A sampled sigma_gal (Kinematics key) composes with the calibration too."""
    csp = _build_csp(ssp_grid_path)
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(100.0), name="spec")
    kin = Kinematics(sigma_gal="sigma_gal")
    spec.setup_for_model(csp.wave, zred=ZRED, kinematics=kin,
                         lib_resolution=getattr(csp, "lib_resolution", None))
    theta = _base_theta(csp)
    theta["sigma_gal"] = jnp.array([220.0])
    pred0 = csp.predict(theta, [spec], kinematics=kin)
    pred1 = csp.predict({**theta, "spectrum_calib": jnp.asarray(COEFF)}, [spec], kinematics=kin)
    np.testing.assert_allclose(np.asarray(pred1["spec"]),
                               _expected_factor(SPEC_WAVE, COEFF) * np.asarray(pred0["spec"]),
                               rtol=1e-5)


def test_spectrum_calib_under_jit(ssp_grid_path):
    csp = _build_csp(ssp_grid_path)
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(100.0), name="spec")
    _setup([spec], csp)
    theta = _base_theta(csp)
    f = jax.jit(lambda th: csp.predict(th, [spec])["spec"])
    eager = csp.predict({**theta, "spectrum_calib": jnp.asarray(COEFF)}, [spec])["spec"]
    jitted = f({**theta, "spectrum_calib": jnp.asarray(COEFF)})
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-6)


# --------------------------------------------------------------------------
# Alpha-enhanced twin
# --------------------------------------------------------------------------
def _find_afe_grid():
    here = pathlib.Path(__file__).resolve().parent
    cands = [
        here.parent / "ceridwen" / "data" / "test_data" / "amist_c3k_lr_chab_afe.h5",
        here / "fixtures" / "amist_c3k_lr_chab_afe.h5",
        here.parent / "examples" / "amist_c3k_lr_chab_afe.h5",
    ]
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def test_spectrum_calib_in_csp_afe():
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
        verbose=False,
        cosmo=Cosmology.planck18(),
    )
    phot = Photometry(filters=FILTERS, name="phot")
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(100.0), name="spec")
    kin = Kinematics(sigma_gal="sigma_gal")
    phot.setup_for_model(csp.wave, zred=ZRED)
    spec.setup_for_model(csp.wave, zred=ZRED, kinematics=kin,
                         lib_resolution=getattr(csp, "lib_resolution", None))
    obs = [phot, spec]

    theta = dict(csp.all_params)
    theta["logmass"] = jnp.array([10.0])
    theta["zred"] = jnp.array([ZRED])
    theta["afe"] = jnp.array([0.3])
    theta["sigma_gal"] = jnp.array([260.0])

    pred0 = csp.predict(theta, obs, kinematics=kin)
    pred1 = csp.predict({**theta, "spectrum_scaling": jnp.array([0.9]),
                         "spectrum_calib": jnp.asarray(COEFF)}, obs, kinematics=kin)
    np.testing.assert_allclose(
        np.asarray(pred1["spec"]),
        0.9 * _expected_factor(SPEC_WAVE, COEFF) * np.asarray(pred0["spec"]),
        rtol=1e-5)
    np.testing.assert_allclose(np.asarray(pred1["phot"]),
                               np.asarray(pred0["phot"]), rtol=1e-6)
