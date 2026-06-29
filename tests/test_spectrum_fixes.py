"""
Tests for the three Spectrum-class bug fixes:

1. ``inres`` is now honoured in ``smoothtype="lsf"`` mode.
2. ``self.calibration`` is applied inside chi_sq / residuals / log_likelihood.
3. ``mask_lines`` accepts ``zred`` and redshifts line centres before masking.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.observation.spectrum import Spectrum


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_test_grids(n_obs=400, n_model=2048, wmin=4000.0, wmax=8000.0):
    """A linear observed grid and a denser model grid covering the same range."""
    wave_obs   = np.linspace(wmin, wmax, n_obs)
    wave_model = np.linspace(wmin - 50.0, wmax + 50.0, n_model)
    return wave_obs, wave_model


def _model_spectrum(wave_model, lines=((5500.0, 8.0), (6000.0, 5.0))):
    """A smooth continuum + a couple of narrow Gaussian emission lines."""
    spec = 1.0 + 0.1 * np.sin(wave_model / 200.0)
    for lam0, sigma in lines:
        spec = spec + 0.5 * np.exp(-0.5 * ((wave_model - lam0) / sigma) ** 2)
    return jnp.asarray(spec, dtype=jnp.float64)


# =============================================================================
# Issue 1 — LSF inres
# =============================================================================

class TestLSFInres:

    def test_inres_equivalent_to_quadrature_reduced_sigma(self):
        """A Spectrum with constant σ_LSF = c1 and inres = i must match a
        Spectrum with σ_LSF = sqrt(c1² − i²) and inres = 0, to numerical
        precision."""
        wave_obs, wave_model = _make_test_grids()
        sigma_target = 4.0   # Å
        inres        = 2.5   # Å
        sigma_eff    = float(np.sqrt(sigma_target**2 - inres**2))

        sigma_lsf_a = np.full(wave_obs.size, sigma_target)
        sigma_lsf_b = np.full(wave_obs.size, sigma_eff)

        flux_dummy = np.zeros_like(wave_obs)
        unc_dummy  = np.ones_like(wave_obs)

        spec_a = Spectrum(wavelength=wave_obs, flux=flux_dummy,
                          uncertainty=unc_dummy, mask=np.ones_like(wave_obs, bool),
                          resolution=sigma_lsf_a, smoothtype="lsf", inres=inres)
        spec_b = Spectrum(wavelength=wave_obs, flux=flux_dummy,
                          uncertainty=unc_dummy, mask=np.ones_like(wave_obs, bool),
                          resolution=sigma_lsf_b, smoothtype="lsf", inres=0.0)

        spec_a.setup_for_model(wave_model)
        spec_b.setup_for_model(wave_model)

        m = _model_spectrum(wave_model)
        out_a = np.asarray(spec_a.predict(m, jnp.asarray(wave_model)))
        out_b = np.asarray(spec_b.predict(m, jnp.asarray(wave_model)))

        assert np.allclose(out_a, out_b, rtol=1e-6, atol=1e-8), (
            f"LSF inres path disagrees with explicit-quadrature path. "
            f"max abs diff = {np.max(np.abs(out_a - out_b)):.3e}"
        )

    def test_inres_floor_no_nans(self):
        """When inres > σ_LSF at every pixel, σ_eff floors at 0 and the
        smoother must not produce NaNs."""
        wave_obs, wave_model = _make_test_grids()
        sigma_lsf = np.full(wave_obs.size, 1.0)   # 1 Å
        inres     = 5.0                           # > all σ_LSF

        spec = Spectrum(wavelength=wave_obs, flux=np.zeros_like(wave_obs),
                        uncertainty=np.ones_like(wave_obs),
                        mask=np.ones_like(wave_obs, bool),
                        resolution=sigma_lsf, smoothtype="lsf", inres=inres)
        spec.setup_for_model(wave_model)

        m   = _model_spectrum(wave_model)
        out = np.asarray(spec.predict(m, jnp.asarray(wave_model)))

        assert np.all(np.isfinite(out)), "LSF predict produced non-finite values"


# =============================================================================
# Issue 2 — self.calibration applied in chi_sq / residuals
# =============================================================================

class TestCalibrationApplied:

    def test_constant_calibration_matches_scaled_model(self):
        wave_obs = np.linspace(4000.0, 8000.0, 200)
        flux     = 1.0 + 0.05 * np.sin(wave_obs / 300.0)
        unc      = 0.01 * np.ones_like(wave_obs)
        mask     = np.ones_like(wave_obs, dtype=bool)
        calib    = np.full_like(wave_obs, 1.2)
        model    = 0.9 + 0.04 * np.sin(wave_obs / 300.0)

        spec_cal = Spectrum(wavelength=wave_obs, flux=flux, uncertainty=unc,
                            mask=mask, calibration=calib)
        spec_ref = Spectrum(wavelength=wave_obs, flux=flux, uncertainty=unc,
                            mask=mask, calibration=None)

        chi2_cal = spec_cal.chi_sq(jnp.asarray(model))
        chi2_ref = spec_ref.chi_sq(jnp.asarray(1.2 * model))
        assert np.isclose(chi2_cal, chi2_ref, rtol=1e-10, atol=1e-10), (
            f"chi_sq with calibration={chi2_cal} != chi_sq on pre-scaled model={chi2_ref}"
        )

    def test_nonconstant_calibration_zero_residual_when_perfect(self):
        wave_obs = np.linspace(4000.0, 8000.0, 200)
        calib    = 1.0 + 0.3 * np.sin(wave_obs / 800.0)
        # Choose a model and synthesise data = calib * model exactly
        model    = 1.0 + 0.05 * np.cos(wave_obs / 250.0)
        data     = calib * model
        unc      = 0.01 * np.ones_like(wave_obs)

        spec = Spectrum(wavelength=wave_obs, flux=data, uncertainty=unc,
                        mask=np.ones_like(wave_obs, bool), calibration=calib)
        res = np.asarray(spec.residuals(jnp.asarray(model)))
        assert np.allclose(res, 0.0, atol=1e-12), (
            f"residuals not zero where data = calib * model; max |res| = {np.max(np.abs(res)):.3e}"
        )

    def test_calibration_none_is_unchanged_path(self):
        """Smoke-check: with calibration=None, residuals reduce to (data-model)/σ."""
        wave_obs = np.linspace(4000.0, 8000.0, 200)
        model    = 1.0 + 0.05 * np.cos(wave_obs / 250.0)
        data     = model + 0.02 * np.random.default_rng(0).standard_normal(wave_obs.size)
        unc      = 0.02 * np.ones_like(wave_obs)
        spec = Spectrum(wavelength=wave_obs, flux=data, uncertainty=unc,
                        mask=np.ones_like(wave_obs, bool), calibration=None)
        res = np.asarray(spec.residuals(jnp.asarray(model)))
        expected = (data - model) / unc
        assert np.allclose(res, expected, rtol=1e-10, atol=1e-12)


# =============================================================================
# Issue 3 — mask_lines redshift handling
# =============================================================================

class TestMaskLinesRedshift:

    def test_zred_shifts_centre_to_observed_frame(self):
        # Observed-frame grid covering redshifted Hα at z = 1.5
        # λ_obs = (1 + z) × 6562.8 ≈ 16407 Å
        zred = 1.5
        lam_rest = 6562.8
        lam_obs  = (1.0 + zred) * lam_rest

        wave_obs = np.linspace(lam_obs - 200.0, lam_obs + 200.0, 401)
        spec = Spectrum(wavelength=wave_obs,
                        flux=np.zeros_like(wave_obs),
                        uncertainty=np.ones_like(wave_obs),
                        mask=np.ones_like(wave_obs, bool))

        spec.mask_lines([lam_rest], dv=500.0, zred=zred)

        mask = np.asarray(spec.mask)
        wave = np.asarray(spec.wavelength)
        c_kms = 2.998e5
        dlam  = lam_obs * 500.0 / c_kms

        # Pixels inside ±dlam of lam_obs should be masked off
        expected_masked = (wave >= lam_obs - dlam) & (wave <= lam_obs + dlam)
        assert np.array_equal(~mask, expected_masked), (
            "mask_lines with zred=1.5 did not mask pixels centred on the "
            f"observed-frame Hα (~{lam_obs:.1f} Å)."
        )

    def test_zred_zero_is_backward_compatible(self):
        # Caller passing observed-frame wavelength with default zred=0
        wave_obs = np.linspace(6300.0, 6800.0, 501)
        spec = Spectrum(wavelength=wave_obs,
                        flux=np.zeros_like(wave_obs),
                        uncertainty=np.ones_like(wave_obs),
                        mask=np.ones_like(wave_obs, bool))
        spec.mask_lines([6562.8], dv=500.0)  # default zred=0.0
        mask = np.asarray(spec.mask)
        wave = np.asarray(spec.wavelength)
        c_kms = 2.998e5
        dlam = 6562.8 * 500.0 / c_kms
        expected_masked = (wave >= 6562.8 - dlam) & (wave <= 6562.8 + dlam)
        assert np.array_equal(~mask, expected_masked)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
