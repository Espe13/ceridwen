"""Spectroscopic observation container: instrument LSF, projection through
``ceridwen.broadening.SpectralProjector``, masking, chi-squared and
polynomial-calibration helpers."""

import jax.numpy as jnp
import numpy as np
from .base import Observation
from ..broadening import Instrument, Kinematics, SpectralProjector


class Spectrum(Observation):
    """Spectroscopic observation on an observed-frame pixel grid.

    Parameters
    ----------
    wavelength : array-like (n_pix,), Å, vacuum, OBSERVED frame.
    flux, uncertainty : array-like (n_pix,) -- same units as the model spectra
        passed to ``chi_sq`` / ``residuals``.
    mask : array-like of bool or slice -- True for pixels that are USED.
    instrument : Instrument or None -- the spectrograph LSF; unit and convention
        are the constructor name (``Instrument.R_fwhm(2700)``,
        ``Instrument.fwhm_aa(2.5)``, ``Instrument.sigma_kms(arr, wave=w)``, ...).
        None = no instrumental broadening (and no library subtraction).
    subtract_library : bool -- remove the SSP library resolution in quadrature
        from the instrument width (default True; needs the grid's resolution curve).
    calibration : array-like (n_pix,) -- multiplicative model correction
        (model * calibration ~ data), applied in chi_sq/residuals/log_likelihood.
    logify_spectrum : bool -- residuals in log-flux space.
    sky : array-like (n_pix,) -- subtracted from data in chi_sq/residuals/
        log_likelihood only (not in ``predict``).
    noise_floor : float -- fractional floor: sigma_eff^2 = sigma^2 + (floor*|model|)^2.
    noise : GaussianProcess -- adds a correlated-residual term in ``log_likelihood``
        only.
    zred_range : (z_min, z_max) -- support of a SAMPLED redshift; otherwise taken from
        the finite bounds of the model's ``zred`` prior.

    The galaxy's velocity dispersions are not a property of the observation:
    they are set once on the model (``SedModel(kinematics=Kinematics(...))``).
    """

    _kind = "spectrum"
    logify_spectrum = False
    alias = dict(
        spectrum     = "flux",
        spectrum_unc = "uncertainty",
        spec_mask    = "mask",
    )
    _meta = ("kind", "name")
    _data = ("wavelength", "flux", "uncertainty", "mask")
    _removed_kwargs = {
        "resolution": "instrument=Instrument.<unit>(...)",
        "smoothtype": "instrument=Instrument.<unit>(...)",
        "res_convention": "Instrument.R_fwhm(...) or Instrument.R_sigma(...)",
        "inres": "subtract_library=True/False",
        "sigma_losvd": "SedModel(kinematics=Kinematics(sigma_gal=...))",
        "fit_sigma_smooth": "SedModel(kinematics=Kinematics(sigma_gal='sigma_gal'))",
    }

    def __init__(
        self,
        wavelength   = None,
        flux         = None,
        uncertainty  = None,
        mask         = slice(None),
        noise        = None,
        name         = None,
        instrument   = None,
        subtract_library = True,
        calibration  = None,
        logify_spectrum = False,
        sky          = None,
        noise_floor  = 0.0,
        zred_range   = None,
        **kwargs,
    ):
        for k in list(kwargs):
            if k in self._removed_kwargs:
                raise TypeError(
                    f"Spectrum(): '{k}' was removed; use {self._removed_kwargs[k]}")
        if instrument is not None and not isinstance(instrument, Instrument):
            raise TypeError(
                "Spectrum(instrument=...) takes a ceridwen.broadening.Instrument "
                "(Instrument.R_fwhm(2700), Instrument.fwhm_aa(2.5), "
                "Instrument.sigma_kms(120.0), ...), got "
                f"{type(instrument).__name__}")
        self._wavelength    = (
            None if wavelength is None
            else jnp.asarray(wavelength, dtype=float)
        )
        self.instrument       = instrument
        self.subtract_library = bool(subtract_library)
        self.calibration    = (
            None if calibration is None
            else jnp.asarray(calibration, dtype=float)
        )
        self.logify_spectrum = logify_spectrum
        self.sky             = (None if sky is None
                                else jnp.asarray(sky, dtype=float))
        self.noise_floor     = float(noise_floor)
        self.zred_range      = (None if zred_range is None
                                else (float(zred_range[0]), float(zred_range[1])))
        self._proj = None

        super().__init__(
            flux        = flux,
            uncertainty = uncertainty,
            mask        = mask,
            noise       = noise,
            name        = name,
            **kwargs,
        )

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = (
            None if value is None
            else jnp.asarray(value, dtype=float)
        )

    def setup_for_model(self, wave_model, zred: float = 0.0,
                        kinematics=None, lib_resolution=None,
                        line_wave_rest=None, zred_range=None):
        """Build the projector from the rest-frame model grid ``wave_model`` [Å]
        redshifted by (1 + zred) onto ``self.wavelength``; call once, outside JIT.

        kinematics : Kinematics -- galaxy widths (None = ``Kinematics.none()``)
        lib_resolution : (wave_rest [Å], sigma_v [km/s]) -- SSP library
            resolution curve on ``wave_model`` (NaN = unknown); used when
            ``subtract_library`` and an instrument are set
        line_wave_rest : (n_lines,) rest wavelengths of the nebular grid lines
            in ``predict_line_fluxes`` order, or None (no lines painted)
        zred_range : (z_min, z_max) -- for a SAMPLED redshift (default ``self.zred_range``):
            the projection is then read at ``theta['zred']`` on every call and ``zred``
            is the reference redshift inside the range
        """
        if self._wavelength is None:
            raise ValueError(
                "Spectrum.setup_for_model() needs the observed pixel "
                "wavelength grid, but this Spectrum has wavelength=None. "
                "Pass wavelength= (Å, vacuum, observed frame) at "
                "construction — flux/uncertainty may be added later (e.g. a "
                "predictive container for mock generation)."
            )
        if kinematics is None:
            kinematics = Kinematics.none()
        if zred_range is None:
            zred_range = self.zred_range
        lib = None
        if lib_resolution is not None and self.instrument is not None and self.subtract_library:
            lw = np.asarray(lib_resolution[0], dtype=np.float64)
            ls = np.asarray(lib_resolution[1], dtype=np.float64)
            wm = np.asarray(wave_model, dtype=np.float64)
            if lw.shape == wm.shape and np.allclose(lw, wm):
                lib = ls
            else:
                fin = np.isfinite(ls)
                lib = (np.interp(wm, lw[fin], ls[fin], left=np.nan, right=np.nan)
                       if fin.any() else None)
        self._proj = SpectralProjector.build(
            kinematics, self.instrument, wave_model, self._wavelength, zred,
            lib_sigma_kms=lib, line_wave_rest=line_wave_rest,
            subtract_library=self.subtract_library, zred_range=zred_range)

    def predict(self, spectrum, wave_model=None, line_flux=None, theta=None):
        """Model F_nu on the observed pixels (n_pix,): the rest-frame CONTINUUM
        ``spectrum`` (n_wave,) broadened by sigma_gal, the instrument LSF (library
        width removed) and resampled, plus the emission lines painted from their
        observed-frame integrated fluxes ``line_flux`` (all grid lines, or None)
        with sigma_gas + instrument.  ``theta`` supplies the free widths and, for a
        projector built with ``zred_range``, the redshift."""
        if self._proj is None:
            raise RuntimeError(
                "Spectrum.predict() called before setup_for_model(): the "
                "projector has not been built. Call "
                "spec.setup_for_model(wave_model, zred=..., kinematics=...) once "
                "(before the first predict / JIT trace)."
            )
        return self._proj.predict(spectrum, line_flux, {} if theta is None else theta)

    def synthetic_photometry(self, filterset):
        """Synthetic maggies (n_filters,) of this F_nu spectrum through
        ``filterset``; None if there is no data."""
        if self.flux is None or self._wavelength is None:
            return None
        _c        = jnp.array(2.998e18)   # Å/s
        flux_flam = self.flux * _c / self._wavelength**2
        return filterset.get_sed_maggies(flux_flam, sourcewave=self._wavelength)

    def mask_wavelength_range(self, wave_min, wave_max):
        """Mask pixels with wavelength in [wave_min, wave_max] Å (inclusive)."""
        if self._wavelength is None:
            return
        in_range  = (self._wavelength >= wave_min) & (self._wavelength <= wave_max)
        self.mask = self.mask & ~in_range

    def mask_lines(self, line_waves, dv=1000.0, zred=0.0):
        """Mask +/- ``dv`` km/s around each rest-frame line wavelength [Å],
        redshifted by (1 + zred) onto the observed grid."""
        if self._wavelength is None:
            return
        c_kms = 2.998e5
        opz   = 1.0 + float(zred)
        for lam0_rest in np.asarray(line_waves).ravel():
            lam0_obs = opz * float(lam0_rest)
            dlam     = lam0_obs * dv / c_kms
            self.mask_wavelength_range(lam0_obs - dlam, lam0_obs + dlam)

    def _sky_corrected_data(self):
        if self.sky is not None:
            return self.flux - self.sky
        return self.flux

    def _effective_sigma(self, model_flux):
        if self.noise_floor > 0.0:
            return jnp.sqrt(
                self.uncertainty ** 2
                + (self.noise_floor * jnp.abs(model_flux)) ** 2
            )
        return self.uncertainty

    def _compute_residuals(self, model_flux, return_sigma=False):
        """(sky-subtracted data - calibration * model) / sigma_eff per pixel."""
        mf    = jnp.asarray(model_flux, dtype=float)
        if self.calibration is not None:
            mf = self.calibration * mf
        data  = self._sky_corrected_data()
        sigma = self._effective_sigma(mf)
        if self.logify_spectrum:
            denom = sigma / data
            resid = (jnp.log(data) - jnp.log(mf)) / denom
        else:
            denom = sigma
            resid = (data - mf) / denom
        if return_sigma:
            return resid, denom
        return resid

    def chi_sq(self, model_flux):
        """Sum of squared normalised residuals over unmasked pixels for
        ``model_flux`` (n_pix,) on the observed grid."""
        resid = self._compute_residuals(model_flux)
        return float(jnp.sum(jnp.where(self.mask, resid ** 2, 0.0)))

    def residuals(self, model_flux):
        """Per-pixel normalised residuals (n_pix,); masked pixels are NaN."""
        resid = self._compute_residuals(model_flux)
        return jnp.where(self.mask, resid, jnp.nan)

    def log_likelihood(self, model_flux):
        """Gaussian log-likelihood of ``model_flux`` (n_pix,), including the
        sigma_eff normalisation and the GP term when ``self.noise`` is set."""
        resid, sigma_r = self._compute_residuals(model_flux, return_sigma=True)
        mask = self.mask

        safe_sigma = jnp.where(mask, sigma_r, 1.0)
        lognorm    = float(
            -0.5 * jnp.sum(jnp.where(mask, jnp.log(safe_sigma ** 2), 0.0))
        )

        if self.noise is not None and self._wavelength is not None:
            # GP term already includes the white-noise identity and -n/2 ln(2 pi)
            lnL_struct = float(self.noise.log_likelihood(
                np.array(resid),
                np.array(self._wavelength),
                np.array(mask),
            ))
        else:
            r = jnp.where(mask, resid, 0.0)
            n = float(jnp.sum(mask))
            lnL_struct = float(
                -0.5 * jnp.sum(r ** 2) - 0.5 * n * jnp.log(2.0 * jnp.pi)
            )

        return float(lnL_struct + lognorm)

    def fit_polynomial_calibration(self, model_flux, order: int = 3):
        """Weighted least-squares Chebyshev calibration P(lambda) with
        data ~ P * model_flux over unmasked pixels; returns
        (coeffs (order+1,), P * model_flux (n_pix,))."""
        mf     = np.asarray(model_flux, dtype=np.float64)
        data   = np.asarray(self._sky_corrected_data(), dtype=np.float64)
        sigma  = np.asarray(self.uncertainty, dtype=np.float64)
        wav    = np.asarray(self._wavelength,  dtype=np.float64)
        mask   = np.asarray(self.mask,         dtype=bool)

        wav_mid  = 0.5 * (wav.max() + wav.min())
        wav_half = 0.5 * (wav.max() - wav.min())
        x        = (wav - wav_mid) / (wav_half if wav_half > 0 else 1.0)

        A = np.polynomial.chebyshev.chebvander(x, order)  # (n_pix, order+1)

        A_w = (A * mf[:, None]) / sigma[:, None]
        y_w = data / sigma

        A_wm = A_w[mask]
        y_wm = y_w[mask]

        coeffs, _, _, _ = np.linalg.lstsq(A_wm, y_wm, rcond=None)

        poly_vals       = A @ coeffs
        calibrated_flux = jnp.asarray(poly_vals * mf)

        return coeffs, calibrated_flux

    def __str__(self):
        wmin = (
            "none" if self._wavelength is None
            else f"{float(jnp.min(self._wavelength)):.1f}"
        )
        wmax = (
            "none" if self._wavelength is None
            else f"{float(jnp.max(self._wavelength)):.1f}"
        )
        ins = self.instrument
        if ins is None:
            res_str = "none"
        elif ins.value.ndim == 0:
            res_str = f"{ins.kind} = {float(ins.value):g}"
        else:
            res_str = f"{ins.kind} array [{ins.value.min():g}, {ins.value.max():g}]"

        lines = [
            f"Spectrum ({self.name})",
            f"  ndata         : {self.ndata}",
            f"  ndof          : {self.ndof}",
            f"  wavelength    : {wmin} – {wmax} Å",
            f"  instrument    : {res_str}",
            f"  subtract lib. : {self.subtract_library}",
            f"  projector     : " + ("not built" if self._proj is None else
                                    (f"built, zred sampled in [{self._proj.zred_range[0]:g}, {self._proj.zred_range[1]:g}]"
                                     if self._proj.free_z else f"built, zred = {self._proj.opz_ref - 1:g}")),
            f"  logify        : {self.logify_spectrum}",
            f"  calibration   : {'provided' if self.calibration is not None else 'none'}",
            f"  sky           : {'provided' if self.sky is not None else 'none'}",
            f"  noise_floor   : {self.noise_floor:.4f}",
            f"  noise model   : {repr(self.noise) if self.noise is not None else 'none'}",
            f"  masked pixels : {self.ndata - self.ndof} / {self.ndata}",
        ]
        return "\n".join(lines)

    def _display_str(self, max_rows: int = 20) -> str:
        header = str(self)
        if self._wavelength is None or self.flux is None:
            return header + "\n  (no wavelength / flux vector)"

        wave = np.asarray(self._wavelength)
        flux = np.asarray(self.flux)
        unc  = (np.asarray(self.uncertainty)
                if self.uncertainty is not None
                else np.full_like(flux, np.nan))
        mask = np.asarray(self.mask)
        n    = len(wave)

        col = f"  {'i':<6}  {'λ [Å]':>12}  {'flux':>14}  {'σ':>14}  {'mask':>5}"
        sep = "  " + "-" * (len(col) - 2)
        out = [header, "", col, sep]

        def _row(i):
            w = float(wave[i])
            f = float(flux[i]) if i < len(flux) else float("nan")
            u = float(unc[i])  if i < len(unc)  else float("nan")
            m = bool(mask[i])  if i < len(mask) else False
            return (f"  {i:<6d}  {w:>12.2f}  {f:>14.4e}  "
                    f"{u:>14.4e}  {str(m):>5}")

        if n <= max_rows:
            idxs = list(range(n))
        else:
            head = list(range(max_rows // 2))
            tail = list(range(n - max_rows // 2, n))
            idxs = head + [None] + tail

        for i in idxs:
            if i is None:
                out.append("  ...")
            else:
                out.append(_row(i))

        good = mask.astype(bool) & np.isfinite(flux) & np.isfinite(unc)
        if good.any():
            f_lo, f_md, f_hi = np.percentile(flux[good], [0, 50, 100])
            snr_mean = float(np.nanmean(np.abs(flux[good] / unc[good])))
            out.append(sep)
            out.append(
                f"  [stats over {int(good.sum())}/{n} unmasked, "
                f"finite pixels]  "
                f"flux min/med/max = {f_lo:.3e} / {f_md:.3e} / {f_hi:.3e}  "
                f"mean |S/N| = {snr_mean:.2f}"
            )
        return "\n".join(out)
