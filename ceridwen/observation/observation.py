"""
ceridwen/observation/observation.py
====================================
Data containers for observed SEDs.

Classes
-------
Observation
    Abstract base class.  Stores flux, uncertainty, mask, noise model.

Photometry(Observation)
    Broadband photometry in AB maggies.  Uses a sedpy_jax FilterSet for
    filter-convolution of model spectra.

Spectrum(Observation)
    Spectroscopic observation.  Stores a dense wavelength array, optional
    resolution and calibration vectors, and helper methods for masking,
    chi-squared computation, and synthetic photometry.
"""

import json
import jax.numpy as jnp
import numpy as np          # only for dtype and IO, not computation
from sedpy_jax.observate import FilterSet


# =====================================================================
# Base class
# =====================================================================

class Observation:
    """
    Base class for a single observed dataset.

    Parameters
    ----------
    flux : array-like, optional
        Observed data vector (maggies, F_nu, F_lambda, …).
    uncertainty : array-like, optional
        1-sigma uncertainty, same units as `flux`.
    mask : array-like of bool or slice
        True where a datum is *used* (unmasked).  Defaults to all-True.
    noise : object, optional
        Noise model instance (used by a fitter; stored but not applied here).
    name : str, optional
        Human-readable label.

    Notes
    -----
    `wavelength` is *not* set by the base class – subclasses must define it
    either as a property (Photometry: derived from filter effective wavelengths;
    Spectrum: stored directly) or as an instance attribute.
    """

    _kind          = "observation"
    logify_spectrum = False
    alias          = {}
    _meta          = ("kind", "name")
    _data          = ("wavelength", "flux", "uncertainty", "mask")

    # Class-level sentinel so `self.wavelength` is always defined without
    # raising AttributeError on bare Observation instances.
    wavelength = None

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 noise=None,
                 name=None,
                 **kwargs):

        self.flux        = None if flux        is None else jnp.asarray(flux,        dtype=float)
        self.uncertainty = None if uncertainty is None else jnp.asarray(uncertainty, dtype=float)

        # Initialise mask (bool array matching flux length, or empty)
        if isinstance(mask, slice):
            if self.flux is None:
                self.mask = jnp.array([], dtype=bool)
            else:
                m = np.zeros(len(self.flux), dtype=bool)
                m[mask] = True
                self.mask = jnp.asarray(m)
        else:
            self.mask = jnp.asarray(mask, dtype=bool)

        self.noise = noise

        if name is None:
            addr      = f"{id(self):016x}"
            self.name = f"{self.kind[:4]}-{addr[-6:]}"
        else:
            self.name = name

        self.rectify()

    # ------------------------------------------------------------------
    def __str__(self):
        wmin = (None if self.wavelength is None
                else float(jnp.min(self.wavelength)))
        wmax = (None if self.wavelength is None
                else float(jnp.max(self.wavelength)))
        lines = [
            f"Observation ({self._kind}, {self.name})",
            f"  ndata         : {self.ndata}",
            f"  ndof          : {self.ndof}",
            f"  wavelength    : {wmin} – {wmax} Å",
            f"  masked points : {self.ndata - self.ndof} / {self.ndata}",
            f"  flux finite   : "
            f"{int(jnp.sum(jnp.isfinite(self.flux))) if self.flux is not None else 0}"
            f" / {self.ndata}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' ndof={self.ndof}>"

    def __getitem__(self, item):
        k = self.alias.get(item, item)
        return getattr(self, k)

    def get(self, item, default=None):
        try:
            return self[item]
        except AttributeError:
            return default

    # ------------------------------------------------------------------
    def rectify(self):
        """
        Validate arrays and build the boolean mask.
        Called automatically at the end of ``__init__``.
        """
        if self.flux is None:
            # Subclasses may define wavelength as a read-only property;
            # wrap the assignment in a try/except to avoid AttributeError.
            try:
                self.wavelength = None
            except AttributeError:
                pass
            return

        assert self.flux.ndim == 1,        "flux must be 1-D"
        assert self.uncertainty is not None, "uncertainty is required when flux is provided"
        assert self.uncertainty.ndim == 1, "uncertainty must be 1-D"
        assert len(self.flux) == len(self.uncertainty), \
            "flux and uncertainty lengths differ"

        if self.wavelength is not None:
            assert self.wavelength.ndim == 1, "wavelength must be 1-D"
            assert len(self.wavelength) == len(self.flux), \
                f"wavelength length ({len(self.wavelength)}) ≠ flux length ({len(self.flux)})"

        self._automask()
        assert self.ndof > 0, "no valid unmasked data points after masking"

    # ------------------------------------------------------------------
    def _automask(self):
        """
        AND the user-supplied mask with finite-flux and positive-uncertainty
        constraints.  Pure JAX — safe inside JIT-traced code.
        """
        if self.flux is None:
            return
        valid = (
            jnp.isfinite(self.flux) &
            jnp.isfinite(self.uncertainty) &
            (self.uncertainty > 0)
        )
        self.mask = self.mask & valid

    # ------------------------------------------------------------------
    @property
    def kind(self):
        return self._kind

    @property
    def ndof(self):
        return int(jnp.sum(self.mask))

    @property
    def ndata(self):
        return 0 if self.flux is None else len(self.flux)

    @property
    def wave_min(self):
        return None if self.wavelength is None else float(jnp.min(self.wavelength))

    @property
    def wave_max(self):
        return None if self.wavelength is None else float(jnp.max(self.wavelength))

    @property
    def metadata(self):
        meta = {m: getattr(self, m, None) for m in self._meta}
        if "filternames" in meta and meta["filternames"] is not None:
            meta["filters"] = ",".join(meta["filternames"])
        return meta

    @property
    def maggies_to_nJy(self):
        """Conversion factor: 1 maggie = maggies_to_nJy nJy."""
        return 1e9 * 3631.0

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def to_struct(self, data_dtype=np.float32):
        """Return a NumPy structured array suitable for writing to FITS/HDF5."""
        self._automask()
        cols = []
        for c in self._data:
            dat = getattr(self, c, None)
            if dat is None:
                continue
            dat_np = np.asarray(dat)
            if dat_np.ndim != 1 or len(dat_np) != self.ndata:
                continue
            cols.append((c, dat_np.dtype))

        dtype  = np.dtype(cols)
        struct = np.zeros(self.ndata, dtype=dtype)
        for c in dtype.names:
            struct[c] = np.asarray(getattr(self, c))
        return struct

    def to_fits(self, filename=""):
        from astropy.io import fits
        hdus = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(self.to_struct()),
        ])
        for hdu in hdus:
            hdu.header.update(self.metadata)
        if filename:
            hdus.writeto(filename, overwrite=True)
        return hdus

    def to_h5_dataset(self, handle):
        dset = handle.create_dataset(self.name, data=self.to_struct())
        dset.attrs.update(self.metadata)

    def to_json(self):
        obs     = {m: getattr(self, m, None) for m in self._meta + self._data}
        convert = {
            k: (np.asarray(v).tolist() if isinstance(v, jnp.ndarray) else v)
            for k, v in obs.items()
        }
        return json.dumps(convert)


# =====================================================================
# Photometry
# =====================================================================

class Photometry(Observation):
    """
    Broadband photometric observation in AB maggies.

    Flux and uncertainty are stored in *maggies* (linear AB flux units;
    1 maggie = 3631 Jy).  Filter information is held in a
    ``sedpy_jax.observate.FilterSet``.

    Parameters
    ----------
    filters : list of str or list of Filter objects
        Filters to include.  Strings are resolved to ``.par`` files in the
        sedpy_jax filter library.
    flux : array-like, shape (n_filters,), optional
        Observed maggies.
    uncertainty : array-like, shape (n_filters,), optional
        1-sigma uncertainties in maggies.
    mask : array-like of bool, shape (n_filters,), optional
        True for filters that should be included in the fit.

    Examples
    --------
    >>> phot = Photometry(
    ...     filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
    ...     flux=obs_maggies,
    ...     uncertainty=obs_maggies_unc,
    ... )
    >>> model_maggies = phot.get_maggies(model_wave, model_fnu)
    >>> chi2 = phot.chi_sq(model_maggies)
    """

    _kind = "photometry"
    alias = dict(
        maggies     = "flux",
        maggies_unc = "uncertainty",
        filters     = "filters",
        phot_mask   = "mask",
    )
    _meta = ("kind", "name", "filternames")

    def __init__(self, filters=[], name=None, **kwargs):
        self.set_filters(filters)
        super().__init__(name=name, **kwargs)

    # ------------------------------------------------------------------
    def set_filters(self, filters):
        """
        Set the filter list.  ``filters`` may be a list of filter-name
        strings or of sedpy_jax ``Filter`` objects.
        """
        if not filters:
            self.filters     = []
            self.filternames = []
            self.filterset   = None
            return

        try:
            self.filternames = [f.name for f in filters]
        except (AttributeError, TypeError):
            self.filternames = list(filters)

        self.filterset = FilterSet(self.filternames)
        self.filters   = list(self.filterset.filters)

    # ------------------------------------------------------------------
    @property
    def wavelength(self):
        """Effective wavelengths of the filters [Å], shape (n_filters,)."""
        if not self.filters:
            return jnp.array([], dtype=float)
        return jnp.asarray([f.wave_effective for f in self.filters])

    # ------------------------------------------------------------------
    def get_maggies(self, model_wave, model_fnu):
        """
        Project a model spectrum onto the filters and return synthetic maggies.

        The model spectrum is expected in **F_nu units** (e.g. L_sun Hz^{-1}
        M_sun^{-1} as returned by ``CSPBasis.get_spectrum``).  Internally the
        spectrum is converted to F_lambda before being projected through the
        AB-normalised FilterSet transmission matrix, so the output has the
        same relative normalisation as a standard AB photometric integral.

        Parameters
        ----------
        model_wave : array-like, shape (n_wave,)
            Wavelength grid [Å].
        model_fnu : array-like, shape (n_wave,)
            Model spectrum in F_nu units (L_sun/Hz/M_sun or erg/s/Hz/cm^2).

        Returns
        -------
        maggies : jnp.ndarray, shape (n_filters,)
            Synthetic photometry with the same relative normalisation as the
            input flux.

        Notes
        -----
        The AB normalisation constant in sedpy_jax cancels dimensionally when
        both the Ceridwen and FSPS spectra are expressed in the same units,
        making model/data comparisons unit-independent.
        """
        if self.filterset is None:
            raise ValueError("No FilterSet configured; call set_filters() first.")
        _c         = jnp.array(2.998e18)        # Å/s
        wave       = jnp.asarray(model_wave,  dtype=float)
        flux_flam  = jnp.asarray(model_fnu,   dtype=float) * _c / wave**2
        return self.filterset.get_sed_maggies(flux_flam, sourcewave=wave)

    # ------------------------------------------------------------------
    def chi_sq(self, model_maggies):
        """
        Chi-squared contribution from this photometric observation.

        Parameters
        ----------
        model_maggies : array-like, shape (n_filters,)
            Synthetic photometry on the same filter set.

        Returns
        -------
        chi2 : float
        """
        resid = (self.flux - jnp.asarray(model_maggies, dtype=float)) / self.uncertainty
        return float(jnp.sum(jnp.where(self.mask, resid**2, 0.0)))

    def residuals(self, model_maggies):
        """
        Per-filter (data − model) / sigma.  Masked filters are set to NaN.

        Returns
        -------
        res : jnp.ndarray, shape (n_filters,)
        """
        res = (self.flux - jnp.asarray(model_maggies, dtype=float)) / self.uncertainty
        return jnp.where(self.mask, res, jnp.nan)

    # ------------------------------------------------------------------
    def __str__(self):
        weff = [float(f.wave_effective) for f in self.filters] if self.filters else []
        w_range = (
            "no filters"
            if not weff
            else f"{min(weff):.0f} – {max(weff):.0f} Å"
        )
        fnames = self.filternames
        if len(fnames) > 6:
            fname_str = ", ".join(fnames[:5]) + f" … (+{len(fnames)-5} more)"
        else:
            fname_str = ", ".join(fnames)
        lines = [
            f"Photometry ({self.name})",
            f"  n_filters     : {len(self.filters)}",
            f"  ndof          : {self.ndof}",
            f"  wave_eff range: {w_range}",
            f"  filters       : {fname_str}",
            f"  masked filters: {len(self.filters) - self.ndof} / {len(self.filters)}",
        ]
        return "\n".join(lines)


# =====================================================================
# Spectrum
# =====================================================================

class Spectrum(Observation):
    """
    Spectroscopic observation.

    Stores a densely-sampled spectrum together with optional resolution and
    multiplicative flux-calibration arrays.  Provides helpers for masking
    spectral regions, computing chi-squared residuals, and projecting the
    spectrum onto broadband filters.

    Parameters
    ----------
    wavelength : array-like, shape (n_pix,)
        Wavelength grid [Å], vacuum rest-frame.
    flux : array-like, shape (n_pix,)
        Observed flux.  Units must be consistent with ``uncertainty`` and
        any model spectra passed to ``chi_sq`` / ``residuals``.
        Ceridwen model spectra are in L_sun Hz^{-1} M_sun^{-1}.
    uncertainty : array-like, shape (n_pix,)
        1-sigma uncertainty, same units as ``flux``.
    mask : array-like of bool or slice, shape (n_pix,)
        True for pixels that are *used* (not masked).
    resolution : float or array-like, optional
        Spectral resolution.  Scalar R = λ/Δλ, or per-pixel sigma [km/s].
        Stored for use by a fitter / line-spread-function model; not applied
        internally.
    calibration : array-like, shape (n_pix,), optional
        Multiplicative flux-calibration vector (model × calibration ≈ data).
        Stored for use by a fitter; not applied internally.
    logify_spectrum : bool
        If True, ``chi_sq`` and ``residuals`` operate in log-flux space
        (residuals are Δln f / σ_ln f).

    Examples
    --------
    >>> spec = Spectrum(
    ...     wavelength=wave_aa,
    ...     flux=obs_fnu,
    ...     uncertainty=obs_fnu_unc,
    ... )
    >>> spec.mask_lines([6563., 4861.], dv=500.)     # mask Hα, Hβ
    >>> chi2 = spec.chi_sq(model_fnu)
    >>> phot = spec.synthetic_photometry(filterset)
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

    def __init__(
        self,
        wavelength   = None,
        flux         = None,
        uncertainty  = None,
        mask         = slice(None),
        noise        = None,
        name         = None,
        resolution   = None,
        calibration  = None,
        logify_spectrum = False,
        **kwargs,
    ):
        # Store wavelength via the property setter so subclasses can override.
        self._wavelength    = (
            None if wavelength is None
            else jnp.asarray(wavelength, dtype=float)
        )
        self.resolution     = resolution
        self.calibration    = (
            None if calibration is None
            else jnp.asarray(calibration, dtype=float)
        )
        self.logify_spectrum = logify_spectrum

        super().__init__(
            flux        = flux,
            uncertainty = uncertainty,
            mask        = mask,
            noise       = noise,
            name        = name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        """Allow base-class ``rectify`` to set wavelength = None."""
        self._wavelength = (
            None if value is None
            else jnp.asarray(value, dtype=float)
        )

    # ------------------------------------------------------------------
    def synthetic_photometry(self, filterset):
        """
        Project this spectrum onto a FilterSet and return synthetic maggies.

        The spectrum is assumed to be in **F_nu units** (e.g. L_sun/Hz/M_sun)
        and is converted to F_lambda before the AB-normalised projection.

        Parameters
        ----------
        filterset : sedpy_jax.observate.FilterSet

        Returns
        -------
        maggies : jnp.ndarray, shape (n_filters,)
            Returns ``None`` if the spectrum has no data.
        """
        if self.flux is None or self._wavelength is None:
            return None
        _c        = jnp.array(2.998e18)   # Å/s
        flux_flam = self.flux * _c / self._wavelength**2
        return filterset.get_sed_maggies(flux_flam, sourcewave=self._wavelength)

    # ------------------------------------------------------------------
    def mask_wavelength_range(self, wave_min, wave_max):
        """
        Mask pixels with wavelengths in [wave_min, wave_max] Å (inclusive).

        Sets ``self.mask[i] = False`` for all pixels whose wavelength falls
        inside the specified range.

        Parameters
        ----------
        wave_min, wave_max : float
            Wavelength bounds [Å].
        """
        if self._wavelength is None:
            return
        in_range  = (self._wavelength >= wave_min) & (self._wavelength <= wave_max)
        self.mask = self.mask & ~in_range

    def mask_lines(self, line_waves, dv=1000.0):
        """
        Mask spectral lines by zeroing the mask within ±dv km/s of each line.

        Parameters
        ----------
        line_waves : array-like
            Rest-frame central wavelengths [Å].
        dv : float
            Half-width to mask on each side [km/s].  Default 1000 km/s.
        """
        if self._wavelength is None:
            return
        c_kms = 2.998e5   # km/s
        for lam0 in np.asarray(line_waves).ravel():
            dlam = lam0 * dv / c_kms
            self.mask_wavelength_range(float(lam0 - dlam), float(lam0 + dlam))

    # ------------------------------------------------------------------
    def chi_sq(self, model_flux):
        """
        Chi-squared contribution from this spectrum.

        If ``self.logify_spectrum`` is True, residuals are computed as
        Δln(f) / (σ/f), i.e. operating in log-flux space.

        Parameters
        ----------
        model_flux : array-like, shape (n_pix,)
            Model flux on the same wavelength grid.

        Returns
        -------
        chi2 : float
        """
        mf = jnp.asarray(model_flux, dtype=float)
        if self.logify_spectrum:
            resid = (jnp.log(self.flux) - jnp.log(mf)) / (self.uncertainty / self.flux)
        else:
            resid = (self.flux - mf) / self.uncertainty
        return float(jnp.sum(jnp.where(self.mask, resid**2, 0.0)))

    def residuals(self, model_flux):
        """
        Per-pixel (data − model) / sigma.  Masked pixels are set to NaN.

        Returns
        -------
        res : jnp.ndarray, shape (n_pix,)
        """
        mf = jnp.asarray(model_flux, dtype=float)
        if self.logify_spectrum:
            res = (jnp.log(self.flux) - jnp.log(mf)) / (self.uncertainty / self.flux)
        else:
            res = (self.flux - mf) / self.uncertainty
        return jnp.where(self.mask, res, jnp.nan)

    # ------------------------------------------------------------------
    def __str__(self):
        wmin = (
            "none" if self._wavelength is None
            else f"{float(jnp.min(self._wavelength)):.1f}"
        )
        wmax = (
            "none" if self._wavelength is None
            else f"{float(jnp.max(self._wavelength)):.1f}"
        )
        if self.resolution is None:
            res_str = "none"
        elif np.ndim(self.resolution) == 0:
            res_str = f"R = {float(self.resolution):.1f}"
        else:
            res_str = f"array, shape {np.shape(self.resolution)}"

        lines = [
            f"Spectrum ({self.name})",
            f"  ndata         : {self.ndata}",
            f"  ndof          : {self.ndof}",
            f"  wavelength    : {wmin} – {wmax} Å",
            f"  resolution    : {res_str}",
            f"  logify        : {self.logify_spectrum}",
            f"  calibration   : {'provided' if self.calibration is not None else 'none'}",
            f"  masked pixels : {self.ndata - self.ndof} / {self.ndata}",
        ]
        return "\n".join(lines)
