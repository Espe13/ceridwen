"""
Spectrum: library-resolution subtraction through ``Instrument``, ``calibration``
inside chi_sq / residuals / log_likelihood, ``mask_lines`` with ``zred``, and
the Instrument unit conventions.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.observation.spectrum import Spectrum
from ceridwen.broadening import Instrument, Kinematics

_KIN = Kinematics(sigma_gal=0.0, sigma_max=10.0)   # test grid has a 50 A buffer only


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
# Library subtraction
# =============================================================================

class TestLibrarySubtraction:

    def test_library_subtraction_equals_quadrature_reduced_width(self):
        """An Instrument of sigma c1 with the library width i subtracted must
        match an Instrument of sigma sqrt(c1^2 - i^2) with no subtraction."""
        wave_obs, wave_model = _make_test_grids()
        sigma_target = 4.0   # A
        inres        = 2.5   # A
        sigma_eff    = float(np.sqrt(sigma_target**2 - inres**2))
        lib_kms = inres / wave_model * 2.99792458e5
        spec_a = Spectrum(wavelength=wave_obs, flux=np.zeros_like(wave_obs),
                          uncertainty=np.ones_like(wave_obs), mask=np.ones_like(wave_obs, bool),
                          instrument=Instrument.sigma_aa(np.full(wave_obs.size, sigma_target), wave=wave_obs))
        spec_b = Spectrum(wavelength=wave_obs, flux=np.zeros_like(wave_obs),
                          uncertainty=np.ones_like(wave_obs), mask=np.ones_like(wave_obs, bool),
                          instrument=Instrument.sigma_aa(np.full(wave_obs.size, sigma_eff), wave=wave_obs),
                          subtract_library=False)
        spec_a.setup_for_model(wave_model, kinematics=_KIN, lib_resolution=(wave_model, lib_kms))
        spec_b.setup_for_model(wave_model, kinematics=_KIN)
        m = _model_spectrum(wave_model)
        out_a = np.asarray(spec_a.predict(m, jnp.asarray(wave_model)))
        out_b = np.asarray(spec_b.predict(m, jnp.asarray(wave_model)))
        assert np.allclose(out_a, out_b, rtol=1e-6, atol=1e-8), (
            f"max abs diff = {np.max(np.abs(out_a - out_b)):.3e}")

    def test_library_wider_than_instrument_warns_and_is_finite(self):
        """Instrument narrower than the library: warning, width floored at 0, no NaN."""
        wave_obs, wave_model = _make_test_grids()
        lib_kms = 5.0 / wave_model * 2.99792458e5
        spec = Spectrum(wavelength=wave_obs, flux=np.zeros_like(wave_obs),
                        uncertainty=np.ones_like(wave_obs), mask=np.ones_like(wave_obs, bool),
                        instrument=Instrument.sigma_aa(1.0))
        with pytest.warns(UserWarning, match="narrower than the SSP library"):
            spec.setup_for_model(wave_model, kinematics=_KIN, lib_resolution=(wave_model, lib_kms))
        out = np.asarray(spec.predict(_model_spectrum(wave_model), jnp.asarray(wave_model)))
        assert np.all(np.isfinite(out))


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
        with pytest.warns(UserWarning, match="without zred"):
            spec.mask_lines([6562.8], dv=500.0)  # no zred: rest = observed, with a warning
        mask = np.asarray(spec.mask)
        wave = np.asarray(spec.wavelength)
        c_kms = 2.998e5
        dlam = 6562.8 * 500.0 / c_kms
        expected_masked = (wave >= 6562.8 - dlam) & (wave <= 6562.8 + dlam)
        assert np.array_equal(~mask, expected_masked)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# -----------------------------------------------------------------------------
# Resolution convention (sigma vs FWHM)
# -----------------------------------------------------------------------------
class TestInstrumentConvention:
    """R_fwhm and R_sigma differ by 2.3548; the removed keywords are refused."""

    CKMS = 2.99792458e5
    F = 2.0 * np.sqrt(2.0 * np.log(2.0))

    def _spec(self, **kw):
        wave = np.linspace(4000.0, 8000.0, 100)
        return Spectrum(wavelength=wave, flux=np.ones(100),
                        uncertainty=np.ones(100), name="c", **kw)

    def test_removed_keywords_raise(self):
        for kw in ({"resolution": 1000.0, "smoothtype": "R"}, {"res_convention": "sigma"},
                   {"sigma_losvd": 200.0}, {"fit_sigma_smooth": True}, {"inres": 0.0}):
            with pytest.raises(TypeError, match="is not an argument"):
                self._spec(**kw)

    def test_non_instrument_raises(self):
        with pytest.raises(TypeError, match="Instrument"):
            self._spec(instrument=100.0)

    def test_R_sigma_vs_fwhm_factor(self):
        w = np.array([5000.0])
        sig = Instrument.R_sigma(1000.0).sigma_kms_at(w)[0]
        fwm = Instrument.R_fwhm(1000.0).sigma_kms_at(w)[0]
        assert np.isclose(sig, self.CKMS / 1000.0, rtol=1e-12)
        assert np.isclose(sig / fwm, self.F, rtol=1e-12)

    def test_fwhm_kms_equals_sigma_over_2p3548(self):
        w = np.array([5000.0])
        assert np.isclose(Instrument.sigma_kms(100.0).sigma_kms_at(w)[0],
                          Instrument.fwhm_kms(100.0 * self.F).sigma_kms_at(w)[0], rtol=1e-12)

    def test_fwhm_aa_array_converted(self):
        wave = np.linspace(4000.0, 8000.0, 100)
        res = np.full(100, 2.3548200450309493)
        out = Instrument.fwhm_aa(res, wave=wave).sigma_kms_at(wave)
        assert np.allclose(out, self.CKMS / wave, rtol=1e-10)
