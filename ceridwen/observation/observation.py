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
from sedpy_jax.smoothing import (
    make_vel_smoother,
    make_wave_smoother,
    make_lsf_smoother,
)

_CKMS = 2.998e5   # speed of light [km/s]


# =====================================================================
# Gaussian Process noise model
# =====================================================================

class GaussianProcess:
    """
    Squared-exponential Gaussian Process noise model for spectral residuals.

    Adds correlated residual structure to the spectral likelihood beyond the
    usual pixel-independent Gaussian noise, accounting for systematic
    calibration residuals or continuum modelling errors.

    The GP log-likelihood contribution is:

    .. math::

        \\log p(\\mathbf{r} \\mid \\mathrm{GP}) =
            -\\tfrac{1}{2} \\mathbf{r}^\\top K^{-1} \\mathbf{r}
            -\\tfrac{1}{2} \\log |K|
            -\\tfrac{n}{2} \\log 2\\pi

    where :math:`\\mathbf{r}` is the vector of (unmasked) normalised residuals
    :math:`(d_i - m_i)/\\sigma_i` and :math:`K` is the correlation matrix:

    .. math::

        K_{ij} = a^2 \\exp\\!\\left[-\\tfrac{1}{2}
            \\left(\\frac{\\lambda_i - \\lambda_j}{\\ell}\\right)^2\\right]
            + \\delta_{ij}\\,\\varepsilon

    Parameters
    ----------
    amplitude : float
        GP kernel amplitude (dimensionless, in units of the per-pixel noise
        :math:`\\sigma`).  Typical values: 0.01–0.5.
    length_scale : float
        Correlation length [Å].  Residuals separated by more than
        ~3× the length scale are effectively uncorrelated.
    jitter : float, optional
        Diagonal white-noise jitter added to the kernel matrix for
        numerical stability.  Default 1e-6.

    Examples
    --------
    >>> gp = GaussianProcess(amplitude=0.1, length_scale=50.0)
    >>> spec = Spectrum(..., noise=gp)
    >>> ll  = spec.log_likelihood(model_flux)   # includes GP correction
    """

    def __init__(self, amplitude: float, length_scale: float,
                 jitter: float = 1e-6):
        self.amplitude    = float(amplitude)
        self.length_scale = float(length_scale)
        self.jitter       = float(jitter)

    def log_likelihood(self,
                       residuals: np.ndarray,
                       wavelength: np.ndarray,
                       mask: np.ndarray | None = None) -> float:
        """
        Compute the GP log-likelihood for normalised spectral residuals.

        Parameters
        ----------
        residuals : array-like, shape (n_pix,)
            Per-pixel normalised residuals ``(data - model) / sigma``.
        wavelength : array-like, shape (n_pix,)
            Wavelengths corresponding to ``residuals`` [Å].
        mask : array-like of bool, shape (n_pix,), optional
            If provided, only pixels with ``mask=True`` are included.

        Returns
        -------
        float
            GP log-likelihood contribution.  Add to the standard Gaussian
            log-likelihood to obtain the full marginal log-likelihood.
        """
        r   = np.asarray(residuals,  dtype=np.float64)
        wav = np.asarray(wavelength, dtype=np.float64)

        if mask is not None:
            m   = np.asarray(mask, dtype=bool)
            r   = r[m]
            wav = wav[m]

        n = len(r)
        if n == 0:
            return 0.0

        # Squared-exponential kernel matrix
        dlam = wav[:, None] - wav[None, :]                          # (n, n)
        K    = (self.amplitude ** 2
                * np.exp(-0.5 * (dlam / self.length_scale) ** 2))
        K   += self.jitter * np.eye(n)

        # Log-likelihood via Cholesky decomposition
        try:
            L      = np.linalg.cholesky(K)
            alpha  = np.linalg.solve(K, r)
            log_ll = (-0.5 * float(r @ alpha)
                      - float(np.sum(np.log(np.diag(L))))
                      - 0.5 * n * float(np.log(2.0 * np.pi)))
        except np.linalg.LinAlgError:
            log_ll = -np.inf

        return log_ll

    def __repr__(self):
        return (f"GaussianProcess(amplitude={self.amplitude}, "
                f"length_scale={self.length_scale}, jitter={self.jitter})")


# =====================================================================
# GPU / JIT-friendly projection protocol
# =====================================================================
#
# Every Observation subclass exposes two methods:
#
#   setup_for_model(wave_model)
#       Called ONCE (Python-level, not inside JIT) after the CSP wave grid is
#       known.  Precomputes any static matrices (interpolation matrix,
#       Gaussian weight matrix) that are needed by ``predict``.  Subclasses
#       that need no precomputation can ignore this method (the base-class
#       no-op is inherited automatically).
#
#   predict(spectrum, wave_model) -> jax.Array
#       Callable inside jax.jit with NO Python if/isinstance branches.
#       For Spectrum and Lines this is a single dense matrix–vector multiply
#       (the static matrix was computed once in ``setup_for_model``), giving
#       optimal throughput on both CPU and GPU.
#
# CSPBasis.predict(theta, observations) simply calls
#       { obs.name: obs.predict(spectrum, wave) for obs in observations }
# The Python loop is unrolled at trace-time; no dynamic dispatch reaches the
# XLA-compiled kernel.
# =====================================================================


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
    # GPU / JIT-compatible projection interface
    # ------------------------------------------------------------------

    def setup_for_model(self, wave_model):
        """
        Precompute any static projection matrices that depend on the CSP
        wavelength grid ``wave_model``.

        Call this **once** after the model is constructed, before JIT-compiling
        any function that calls ``predict``.  ``SedModel.__init__`` calls it
        automatically for every registered observation.

        The base-class implementation is a no-op (``Photometry`` does not need
        precomputation because its ``FilterSet`` already holds the transmission
        curves).  Override in subclasses that need a precomputed matrix
        (``Spectrum``, ``Lines``).

        Parameters
        ----------
        wave_model : array-like, shape (n_wave,)
            Model wavelength grid [Å], as stored in ``CSPBasis.wave``.
        """
        pass   # no-op by default; overridden in Spectrum and Lines

    def predict(self, spectrum, wave_model):
        """
        Project the model ``spectrum`` onto this observation type.

        This method is called inside ``CSPBasis.predict(theta, observations)``
        which is JIT-compiled.  Implementations MUST be pure JAX, with no
        Python ``if`` / ``isinstance`` branches over traced values.

        For ``Spectrum`` and ``Lines``, this reduces to a single dense
        matrix–vector multiply using a constant matrix precomputed in
        ``setup_for_model``.  For ``Photometry``, the ``FilterSet``
        convolution is used directly.

        Parameters
        ----------
        spectrum : jax.Array, shape (n_wave,)
            Model spectrum in F_nu units (e.g. L_sun Hz^{-1} M_sun^{-1}).
        wave_model : jax.Array, shape (n_wave,)
            Model wavelength grid [Å] (same grid used in ``setup_for_model``).

        Returns
        -------
        jax.Array
            Projection result; shape depends on subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict(). "
            "Subclasses must override this method."
        )

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
        self.wave_eff = [f.wave_effective for f in self.filters]

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
    # GPU / JIT projection interface
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def setup_for_model(self, wave_model, zred: float = 0.0):
        """
        Precompute a (n_filters, n_wave) projection matrix ``_T`` so that
        ``predict`` reduces to a single GEMV: ``maggies = _T @ F_nu``.

        The matrix folds together three operations that
        ``FilterSet.get_sed_maggies`` does per call:

        1. F_nu -> F_lambda conversion: ``F_lam = F_nu * c / lam^2``
        2. Interpolation from the model wavelength grid onto the
           FilterSet's internal grid (``interp_source``)
        3. Dot product with the precomputed ``FilterSet.trans`` matrix

        By composing these into a single static matrix ``_T`` of shape
        ``(n_filters, n_wave_model)``, all three steps collapse into one
        GEMV at predict time.

        Must be called once before ``predict`` (and before JIT compilation).
        ``SedModel.__init__`` calls this automatically.

        Parameters
        ----------
        wave_model : array-like
            Rest-frame wavelength grid of the model spectrum [Å].
        zred : float, optional
            Fixed redshift at which to precompute the filter projection.
            Defaults to 0 (rest-frame).  For a non-zero ``zred``, the grid
            is effectively taken in the observed frame
            (``wave_effective = (1 + zred) * wave_model``) before filter
            integration — this keeps the predict-time GEMV path unchanged
            and preserves sampling-hot-path speed.  Combine with
            :func:`ceridwen.cosmology.flux_factor_maggies` (applied inside
            ``CSPBasis.predict`` when ``theta['zred']`` is present) to get
            correctly calibrated observed-frame maggies.
        """
        wm_rest = np.asarray(wave_model, dtype=np.float64)   # (n_wave,)
        # Effective ("observed-frame") grid used for the maggies integral.
        # At zred = 0 this equals wm_rest and the method is bit-for-bit
        # identical to the pre-redshift implementation.
        opz = 1.0 + float(zred)
        wm = opz * wm_rest
        n_wave = len(wm)
        _c = 2.998e18  # speed of light [A/s]

        # F_nu -> F_lambda factor per model wavelength bin
        fnu_to_flam = _c / wm**2                         # (n_wave,)

        # Build interpolation matrix H: (n_lam_filter, n_wave_model)
        # such that  F_lam_filtergrid = H @ F_lam_modelgrid
        # This is the linear interpolation that interp_source does per call.
        lam_filt = np.asarray(self.filterset.lam, dtype=np.float64)  # (n_lam,)
        n_lam = len(lam_filt)

        # Construct sparse interpolation weights
        # For each point in lam_filt, find the bracketing indices in wm
        # and the interpolation fraction.
        idx = np.searchsorted(wm, lam_filt, side="right") - 1
        idx = np.clip(idx, 0, n_wave - 2)
        frac = (lam_filt - wm[idx]) / (wm[idx + 1] - wm[idx])
        frac = np.clip(frac, 0.0, 1.0)

        # Zero out entries outside the model wavelength range
        outside = (lam_filt < wm[0]) | (lam_filt > wm[-1])
        frac[outside] = 0.0

        # Build H as a dense matrix (n_lam, n_wave)
        H = np.zeros((n_lam, n_wave), dtype=np.float64)
        for j in range(n_lam):
            if outside[j]:
                continue
            H[j, idx[j]]     = (1.0 - frac[j])
            H[j, idx[j] + 1] = frac[j]

        # FilterSet.trans is (n_filters, n_lam): already includes
        # R * lam * dlam / ab_zero_counts normalisation.
        trans = np.asarray(self.filterset.trans, dtype=np.float64)  # (n_filt, n_lam)

        # Compose: _T = trans @ H @ diag(fnu_to_flam)
        #   maggies = trans @ (H @ (F_nu * fnu_to_flam))
        #           = (trans @ H @ diag(fnu_to_flam)) @ F_nu
        #           = _T @ F_nu
        TH = trans @ H                                   # (n_filt, n_wave)
        T  = TH * fnu_to_flam[None, :]                   # (n_filt, n_wave)

        self._T = jnp.array(T.astype(np.float32))
        self._has_precomputed_T = True

    # ------------------------------------------------------------------
    def predict(self, spectrum, wave_model):
        """
        Project a model F_nu spectrum onto the filters.

        If ``setup_for_model`` has been called, this is a single GEMV
        (``_T @ spectrum``).  Otherwise falls back to ``get_maggies``.

        Parameters
        ----------
        spectrum : jax.Array, shape (n_wave,)
            Model spectrum in F_nu units.
        wave_model : jax.Array, shape (n_wave,)
            Model wavelength grid [Å].

        Returns
        -------
        jax.Array, shape (n_filters,)
            Synthetic AB maggies.
        """
        if getattr(self, "_has_precomputed_T", False):
            return self._T @ spectrum
        return self.get_maggies(wave_model, spectrum)

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
        Instrumental smoothing width.  Interpretation depends on
        ``smoothtype``:

        * ``smoothtype=None`` — stored but never applied (backward-compat).
        * ``"vel"`` — scalar σ_v [km/s].
        * ``"R"``   — scalar resolving power R = λ/σ_λ.
        * ``"lambda"`` — scalar σ_λ [Å].
        * ``"lsf"`` — 1-D array of σ(λ) [Å] at each observed pixel.
    smoothtype : {"vel", "R", "lambda", "lsf"} or None
        Type of instrumental broadening to apply in ``predict``.  See the
        ``__init__`` docstring and ``setup_for_model`` for full details.
        Default ``None`` (no smoothing, backward-compatible).
    inres : float, optional
        Intrinsic resolution of the model library, subtracted in quadrature
        before applying the target smoothing.  Units match ``smoothtype``
        (km/s for vel/R, Å for lambda).  Default 0.0.
    calibration : array-like, shape (n_pix,), optional
        Multiplicative flux-calibration vector (model × calibration ≈ data).
        Stored for use by a fitter; not applied internally.
    logify_spectrum : bool
        If True, ``chi_sq`` and ``residuals`` operate in log-flux space
        (residuals are Δln f / σ_ln f).
    sky : array-like, shape (n_pix,), optional
        Observed sky background spectrum, same units and pixel grid as
        ``flux``.  When provided, the sky is subtracted from the data
        before computing chi-squared residuals:
        ``residual = (flux - sky - model) / sigma``.
        The sky vector is *not* propagated through ``predict``; it enters
        only in ``chi_sq``, ``residuals``, and ``log_likelihood``.
    noise_floor : float, optional
        Fractional uncertainty floor applied to the model flux.  The
        effective per-pixel sigma used in chi-squared becomes:

        .. math::

            \\sigma_{\\rm eff}^2 = \\sigma^2 + (f_{\\rm floor}\\,|m|)^2

        where :math:`f_{\\rm floor}` is ``noise_floor`` and :math:`m` is
        the model flux.  Prevents chi-squared from being dominated by
        pixels where the photon-noise uncertainty is smaller than
        calibration systematics.  Default 0.0 (disabled).
    sigma_losvd : float or None, optional
        Galaxy line-of-sight velocity dispersion [km/s].  When set, an
        additional velocity-broadening step is applied to the model
        spectrum *before* any instrumental smoothing specified by
        ``smoothtype``.  Useful when sigma_losvd is a free parameter of
        the SED fit.  Requires a call to ``setup_for_model`` whenever
        this value changes.  Default None (disabled).
    noise : GaussianProcess or None, optional
        If a ``GaussianProcess`` instance is provided, its log-likelihood
        contribution is added to the standard Gaussian log-likelihood in
        ``log_likelihood()``, accounting for correlated residual
        structure.  ``chi_sq`` is not affected by this attribute (it
        remains a simple diagonal chi-squared).

    Examples
    --------
    >>> spec = Spectrum(
    ...     wavelength=wave_aa,
    ...     flux=obs_fnu,
    ...     uncertainty=obs_fnu_unc,
    ...     resolution=100.0,
    ...     smoothtype="vel",   # 100 km/s instrumental broadening
    ...     noise_floor=0.01,   # 1% calibration floor
    ...     sigma_losvd=150.0,  # 150 km/s galaxy velocity dispersion
    ... )
    >>> spec.setup_for_model(model_wave)
    >>> predicted = spec.predict(model_spectrum, model_wave)
    >>> spec.mask_lines([6563., 4861.], dv=500.)     # mask Hα, Hβ
    >>> chi2 = spec.chi_sq(model_fnu)
    >>> coeffs, cal_model = spec.fit_polynomial_calibration(predicted, order=4)
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
        smoothtype   = None,
        inres        = 0.0,
        sky          = None,
        noise_floor  = 0.0,
        sigma_losvd  = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        smoothtype : {"vel", "R", "lambda", "lsf"} or None
            Which instrumental smoothing kernel to apply in ``predict``:

            ``"vel"``
                Constant velocity dispersion.  ``resolution`` is σ_v [km/s].
                Uses a log-λ FFT so the kernel is shift-invariant in velocity.
            ``"R"``
                Constant spectral resolving power R = λ/Δλ = c/σ_v.
                ``resolution`` is the scalar R value; converted internally to
                σ_v = c/R [km/s].
            ``"lambda"``
                Constant wavelength dispersion.  ``resolution`` is σ_λ [Å].
                Uses a linear-λ FFT.
            ``"lsf"``
                Wavelength-dependent line-spread function.  ``resolution``
                must be a 1-D array of σ(λ) [Å] evaluated at the *observed*
                pixel wavelengths (``self.wavelength``).  The kernel is
                interpolated to the model wavelength grid inside
                ``setup_for_model``.
            ``None`` (default)
                No smoothing applied; ``predict`` performs pure linear
                interpolation (``_H @ spectrum``).  Backward-compatible with
                the original behaviour when ``resolution`` was stored but
                unused.

        inres : float, optional
            Intrinsic (library) resolution of the input model spectrum,
            subtracted in quadrature before applying the target smoothing.
            Units must match ``smoothtype``: km/s for "vel"/"R", Å for
            "lambda".  Ignored for "lsf" mode.  Default 0.0.
        """
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
        self.smoothtype      = smoothtype
        self.inres           = float(inres)
        self.sky             = (None if sky is None
                                else jnp.asarray(sky, dtype=float))
        self.noise_floor     = float(noise_floor)
        self.sigma_losvd     = (None if sigma_losvd is None
                                else float(sigma_losvd))

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
    # GPU / JIT projection interface
    # ------------------------------------------------------------------

    def setup_for_model(self, wave_model, zred: float = 0.0):
        """
        Precompute projection matrices and/or smoothing kernels, then build
        ``_predict_fn`` — the single callable used by ``predict``.

        Must be called once (Python-level, outside JIT) after the model
        wavelength grid is known.  ``SedModel.__init__`` calls this
        automatically.

        Two behaviours depending on ``self.smoothtype``:

        **No smoothing** (``smoothtype=None``)
            Builds the dense (n_pix, n_wave) linear-interpolation matrix
            ``_H`` and sets ``_predict_fn(spec) = _H @ spec``.  Identical
            to the original behaviour.

        **With instrumental smoothing** (``smoothtype`` in
        ``{"vel", "R", "lambda", "lsf"}``)
            Uses a factory function from ``sedpy_jax.smoothing`` to
            precompute all FFT grid transforms.  The returned closure is
            fully JAX-JIT-compilable with respect to the spectrum.
            ``_predict_fn(spec)`` applies smoothing *and* interpolation to
            the observed pixel grid in one call.  ``_H`` is still built for
            backward compatibility.

        Parameters
        ----------
        wave_model : array-like, shape (n_wave,)
            Rest-frame model wavelength grid [Å], strictly increasing.
        zred : float, optional
            Fixed redshift at which to precompute the spectral projection.
            Default 0 (rest-frame).  For ``zred > 0`` the interpolation
            maps from the observed-frame grid
            ``(1 + zred) * wave_model`` onto ``self.wavelength`` (which is
            interpreted as observed-frame pixel wavelengths), preserving
            the predict-time GEMV fast path.  At ``zred = 0`` this method
            is bit-for-bit identical to the pre-redshift implementation.
        """
        wm_rest = np.asarray(wave_model, dtype=np.float64)
        opz = 1.0 + float(zred)
        wm = opz * wm_rest
        wo = np.asarray(self._wavelength, dtype=np.float64)
        n_wave = len(wm)
        n_pix  = len(wo)

        # ── Always build the dense interpolation matrix _H ────────────────
        # (kept for backward compatibility and the no-smoothing fast path)
        j_hi = np.searchsorted(wm, wo, side='right')
        j_hi = np.clip(j_hi, 1, n_wave - 1)
        j_lo = j_hi - 1

        dw    = wm[j_hi] - wm[j_lo]
        alpha = np.where(dw > 0, (wo - wm[j_lo]) / dw, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        H = np.zeros((n_pix, n_wave), dtype=np.float32)
        rows = np.arange(n_pix)
        H[rows, j_lo] += (1.0 - alpha).astype(np.float32)
        H[rows, j_hi] += alpha.astype(np.float32)
        self._H = jnp.array(H)

        # ── Build _predict_fn ──────────────────────────────────────────────
        st         = self.smoothtype
        has_instr  = (st is not None) and (self.resolution is not None)
        has_losvd  = self.sigma_losvd is not None

        if not has_instr and not has_losvd:
            # No smoothing: pure interpolation via _H.
            _H = self._H
            self._predict_fn = lambda spec: _H @ spec

        else:
            # ── Trim model grid to observed wavelength range ────────────
            # The factory functions build a uniform log/linear FFT grid
            # over the full span of ``wave_model``.  If that grid covers
            # 100–25000 Å but the observation covers only 3700–7200 Å,
            # the FFT pixel is ~800 km/s wide.  Nebular emission lines
            # have σ ≈ 2 Å ≈ 0.09 log-pixels and are completely aliased
            # — their flux is redistributed away and the line disappears.
            #
            # Fix: restrict the model grid to the observed spectral window
            # (plus a generous buffer for smoothing-kernel wings) so the
            # FFT grid pixel is small enough to resolve the line profiles.
            _buf     = max(500.0, 0.15 * float(wo.max() - wo.min()))
            _trim    = ((wm >= float(wo.min()) - _buf) &
                        (wm <= float(wo.max()) + _buf))
            _wm_trim = wm[_trim]

            # Integer index array for JIT-safe gather inside the closure:
            # spec[_idx] selects only the trimmed wavelength range.  The
            # shape of _idx is statically known at Python level, so XLA
            # can lower this to a static gather with no shape ambiguity.
            _idx = jnp.array(np.where(_trim)[0])

            # ── Optional LOSVD pre-smoothing stage ─────────────────────
            # If sigma_losvd is set, a velocity-broadening step (galaxy
            # line-of-sight velocity dispersion) is applied to the model
            # spectrum BEFORE instrumental smoothing.  The LOSVD smoother
            # maps _wm_trim → _wm_trim when chained with an instrumental
            # smoother, or _wm_trim → wo when it is the only smoothing
            # step.
            if has_losvd:
                _sv_losvd = float(self.sigma_losvd)
                if has_instr:
                    # Output stays on trimmed model grid for chaining with
                    # the instrumental smoother below.
                    _losvd_sm = make_vel_smoother(_wm_trim, _wm_trim, inres=0.0)
                    _sv       = _sv_losvd   # capture scalar before lambda
                    def _apply_losvd(spec_trim, _s=_losvd_sm, _v=_sv):
                        return _s(spec_trim, _v)
                else:
                    # LOSVD only: output goes directly to observed grid.
                    _losvd_sm = make_vel_smoother(_wm_trim, wo, inres=0.0)
                    _sv       = _sv_losvd
                    def _apply_losvd(spec_trim, _s=_losvd_sm, _v=_sv):
                        return _s(spec_trim, _v)

            if has_instr:
                if st == "vel":
                    # Constant velocity dispersion σ_v [km/s].
                    sigma_v   = float(self.resolution)
                    _instr_sm = make_vel_smoother(_wm_trim, wo, inres=self.inres)
                    if has_losvd:
                        _L = _apply_losvd
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _sv=sigma_v, _L=_L:
                                _sm(_L(spec[_idx]), _sv)
                        )
                    else:
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _sv=sigma_v:
                                _sm(spec[_idx], _sv)
                        )

                elif st == "R":
                    # Constant resolving power R = λ/σ_λ = c/σ_v → σ_v = c/R.
                    sigma_v   = float(_CKMS / self.resolution)
                    _instr_sm = make_vel_smoother(_wm_trim, wo, inres=self.inres)
                    if has_losvd:
                        _L = _apply_losvd
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _sv=sigma_v, _L=_L:
                                _sm(_L(spec[_idx]), _sv)
                        )
                    else:
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _sv=sigma_v:
                                _sm(spec[_idx], _sv)
                        )

                elif st == "lambda":
                    # Constant wavelength dispersion σ_λ [Å].
                    sigma_l   = float(self.resolution)
                    _instr_sm = make_wave_smoother(_wm_trim, wo, inres=self.inres)
                    if has_losvd:
                        _L = _apply_losvd
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _sl=sigma_l, _L=_L:
                                _sm(_L(spec[_idx]), _sl)
                        )
                    else:
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _sl=sigma_l:
                                _sm(spec[_idx], _sl)
                        )

                elif st == "lsf":
                    # Wavelength-dependent LSF: resolution is σ(λ) [Å] at
                    # the *observed* pixel grid.  Interpolate to trimmed grid.
                    res_obs        = np.asarray(self.resolution, dtype=np.float64)
                    sigma_lsf_trim = np.interp(_wm_trim, wo, res_obs)
                    _instr_sm      = make_lsf_smoother(_wm_trim, sigma_lsf_trim, wo)
                    if has_losvd:
                        _L = _apply_losvd
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm, _L=_L:
                                _sm(_L(spec[_idx]))
                        )
                    else:
                        self._predict_fn = (
                            lambda spec, _sm=_instr_sm:
                                _sm(spec[_idx])
                        )

                else:
                    raise ValueError(
                        f"Spectrum.smoothtype={st!r} is not recognised.  "
                        "Valid choices: None, 'vel', 'R', 'lambda', 'lsf'."
                    )

            else:
                # LOSVD only (no instrumental smoothing); _apply_losvd
                # already maps _wm_trim → wo (observed grid).
                _L = _apply_losvd
                self._predict_fn = lambda spec, _L=_L: _L(spec[_idx])

    def predict(self, spectrum, wave_model):
        """
        Project the model spectrum onto the observed pixel grid, applying
        instrumental smoothing if configured.

        Calls ``_predict_fn(spectrum)`` which was constructed by
        ``setup_for_model``.  Depending on ``self.smoothtype``:

        * ``None``  — pure linear interpolation (``_H @ spectrum``).
        * ``"vel"`` / ``"R"`` — constant-velocity FFT broadening then
          interpolation to observed pixels.
        * ``"lambda"`` — constant-wavelength FFT broadening then interpolation.
        * ``"lsf"`` — wavelength-dependent LSF broadening (CDF-transform FFT)
          then interpolation.

        In all smoothing cases, the full smooth→interpolate pipeline is a
        single closure that is JAX-JIT-compilable with respect to ``spectrum``.

        Must call ``setup_for_model(wave_model)`` before this method.

        Parameters
        ----------
        spectrum : jax.Array, shape (n_wave,)
            Model spectrum in F_nu units on the model wavelength grid.
        wave_model : jax.Array, shape (n_wave,)
            Model wavelength grid [Å] (accepted for interface consistency;
            the grid mapping was precomputed by ``setup_for_model``).

        Returns
        -------
        jax.Array, shape (n_pix,)
            Model F_nu (smoothed and) interpolated onto ``self.wavelength``.
        """
        return self._predict_fn(spectrum)

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
    def _sky_corrected_data(self):
        """Return sky-subtracted flux (or raw flux if no sky is set)."""
        if self.sky is not None:
            return self.flux - self.sky
        return self.flux

    def _effective_sigma(self, model_flux):
        """Effective per-pixel sigma including the noise-floor term."""
        if self.noise_floor > 0.0:
            return jnp.sqrt(
                self.uncertainty ** 2
                + (self.noise_floor * jnp.abs(model_flux)) ** 2
            )
        return self.uncertainty

    def _compute_residuals(self, model_flux):
        """
        Internal: (sky-subtracted data − model) / sigma_eff, per pixel.
        Accounts for sky subtraction and noise-floor inflation.
        """
        mf    = jnp.asarray(model_flux, dtype=float)
        data  = self._sky_corrected_data()
        sigma = self._effective_sigma(mf)
        if self.logify_spectrum:
            return (jnp.log(data) - jnp.log(mf)) / (sigma / data)
        return (data - mf) / sigma

    def chi_sq(self, model_flux):
        """
        Chi-squared contribution from this spectrum.

        Accounts for sky subtraction (``self.sky``) and a fractional noise
        floor (``self.noise_floor``).  If ``self.logify_spectrum`` is True,
        residuals are computed in log-flux space: Δln f / (σ_eff / f).

        Parameters
        ----------
        model_flux : array-like, shape (n_pix,)
            Model flux on the observed pixel grid (output of ``predict``).

        Returns
        -------
        chi2 : float
            Sum of squared normalised residuals over unmasked pixels.
        """
        resid = self._compute_residuals(model_flux)
        return float(jnp.sum(jnp.where(self.mask, resid ** 2, 0.0)))

    def residuals(self, model_flux):
        """
        Per-pixel (sky-corrected data − model) / sigma_eff.
        Masked pixels are set to NaN.

        Returns
        -------
        res : jnp.ndarray, shape (n_pix,)
        """
        resid = self._compute_residuals(model_flux)
        return jnp.where(self.mask, resid, jnp.nan)

    def log_likelihood(self, model_flux):
        """
        Full log-likelihood for this spectrum.

        Combines the standard pixel-independent Gaussian log-likelihood with
        an optional Gaussian Process (GP) correction for correlated residuals
        if ``self.noise`` is a ``GaussianProcess`` instance.

        .. math::

            \\log\\mathcal{L} =
                -\\tfrac{1}{2}\\sum_{i\\,\\in\\,\\rm mask} r_i^2
                + \\log p_{\\rm GP}(\\mathbf{r} \\mid \\mathrm{GP})

        where :math:`r_i = (d_i - m_i)/\\sigma_{\\rm eff,i}` and the GP term
        is zero if no noise model is set.

        Parameters
        ----------
        model_flux : array-like, shape (n_pix,)
            Model flux on the observed pixel grid.

        Returns
        -------
        float
            Log-likelihood (larger is better).
        """
        resid  = self._compute_residuals(model_flux)
        chi2   = float(jnp.sum(jnp.where(self.mask, resid ** 2, 0.0)))
        log_ll = -0.5 * chi2

        if self.noise is not None and self._wavelength is not None:
            log_ll += self.noise.log_likelihood(
                np.array(resid),
                np.array(self._wavelength),
                np.array(self.mask),
            )
        return float(log_ll)

    def fit_polynomial_calibration(self, model_flux, order: int = 3):
        """
        Analytically fit a Chebyshev multiplicative calibration polynomial
        P(λ) such that ``data ≈ P(λ) × model_flux``.

        The polynomial coefficients are solved at each call via weighted
        linear least squares, making this suitable for marginalising out the
        calibration at every likelihood evaluation without a parameter-space
        penalty.

        The polynomial is evaluated in a normalised wavelength coordinate
        :math:`x \\in [-1, 1]` using Chebyshev basis functions
        :math:`T_n(x)`, which are numerically stable for high orders.

        Parameters
        ----------
        model_flux : array-like, shape (n_pix,)
            Model flux on the observed pixel grid (output of ``predict``).
        order : int, optional
            Polynomial order.  0 = constant, 1 = linear, etc.  Default 3.

        Returns
        -------
        coeffs : np.ndarray, shape (order + 1,)
            Chebyshev polynomial coefficients.
        calibrated_flux : jnp.ndarray, shape (n_pix,)
            ``P(λ) × model_flux`` — the calibration-corrected model
            prediction to be compared with ``self.flux``.

        Notes
        -----
        Only unmasked pixels enter the least-squares fit.  The returned
        ``calibrated_flux`` is evaluated over the full pixel grid.
        """
        mf     = np.asarray(model_flux, dtype=np.float64)
        data   = np.asarray(self._sky_corrected_data(), dtype=np.float64)
        sigma  = np.asarray(self.uncertainty, dtype=np.float64)
        wav    = np.asarray(self._wavelength,  dtype=np.float64)
        mask   = np.asarray(self.mask,         dtype=bool)

        # Normalise wavelength axis to [-1, 1] for numerical stability
        wav_mid  = 0.5 * (wav.max() + wav.min())
        wav_half = 0.5 * (wav.max() - wav.min())
        x        = (wav - wav_mid) / (wav_half if wav_half > 0 else 1.0)

        # Chebyshev design matrix: A[i, n] = T_n(x_i)
        A = np.polynomial.chebyshev.chebvander(x, order)  # (n_pix, order+1)

        # Weight by model flux and 1/sigma so we minimise
        # sum_mask ((data_i - P(x_i) * model_i) / sigma_i)^2
        A_w = (A * mf[:, None]) / sigma[:, None]   # (n_pix, order+1)
        y_w = data / sigma                          # (n_pix,)

        # Apply mask
        A_wm = A_w[mask]
        y_wm = y_w[mask]

        # Solve linear least squares
        coeffs, _, _, _ = np.linalg.lstsq(A_wm, y_wm, rcond=None)

        # Evaluate calibration polynomial on the full pixel grid
        poly_vals       = A @ coeffs                      # (n_pix,)
        calibrated_flux = jnp.asarray(poly_vals * mf)

        return coeffs, calibrated_flux

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
            f"  smoothtype    : {self.smoothtype}",
            f"  sigma_losvd   : {self.sigma_losvd} km/s",
            f"  logify        : {self.logify_spectrum}",
            f"  calibration   : {'provided' if self.calibration is not None else 'none'}",
            f"  sky           : {'provided' if self.sky is not None else 'none'}",
            f"  noise_floor   : {self.noise_floor:.4f}",
            f"  noise model   : {repr(self.noise) if self.noise is not None else 'none'}",
            f"  masked pixels : {self.ndata - self.ndof} / {self.ndata}",
        ]
        return "\n".join(lines)


# =====================================================================
# Lines
# =====================================================================

class Lines(Observation):
    """
    Observed nebular emission-line fluxes.

    Stores a set of emission-line fluxes together with their FSPS line-array
    indices, vacuum rest-frame wavelengths, and per-line 1-sigma uncertainties.
    The interface is deliberately compatible with
    ``prospect.observation.Lines``: the ``line_ind`` attribute holds integer
    indices into the FSPS ``emline_luminosity`` array, and the ``alias``
    mapping exposes ``"line_inds"`` as an alias for ``line_ind`` so that
    existing Prospector model code can address this object without modification.

    Beyond Prospector, this class adds JAX-native ``chi_sq`` / ``residuals``
    (JIT-compilable through a fitter), and ``mask_by_name`` / ``select_by_name``
    helpers that operate on human-readable line names.

    Parameters
    ----------
    line_ind : array-like of int
        Indices of the observed lines in the FSPS emission-line array
        (``$SPS_HOME/data/emlines_info.dat``).  Required.
    line_names : list of str, optional
        Human-readable names, one per line (e.g. ``"Halpha"``,
        ``"[OIII]5007"``).  Required for ``mask_by_name`` and
        ``select_by_name``.
    wavelength : array-like of float, optional
        Vacuum rest-frame wavelengths [Å], length = ``len(line_ind)``.
    flux : array-like of float, optional
        Observed line fluxes.  Units should be consistent with any model
        prediction passed to ``chi_sq`` / ``residuals`` (typically
        erg s⁻¹ cm⁻²).
    uncertainty : array-like of float, optional
        1-sigma line-flux uncertainties, same units as ``flux``.
    mask : array-like of bool, optional
        True for lines to include in chi-squared.  Defaults to all-True.
    upper_limit : array-like of bool, shape (n_lines,), optional
        If True for a given line, that line is treated as a non-detection
        upper limit rather than a positive detection.  The chi-squared
        contribution for such lines is one-sided: a penalty is applied only
        when the model flux *exceeds* the observed value (i.e., the model
        predicts more emission than the upper limit allows):

        .. math::

            \\chi^2_{\\rm UL} =
            \\begin{cases}
                \\left(\\frac{d - m}{\\sigma}\\right)^2 & m > d \\\\
                0 & m \\leq d
            \\end{cases}

        where :math:`d` is the observed upper-limit flux and :math:`m` is the
        model prediction.  Physically this corresponds to integrating the
        likelihood over all undetected flux values below the upper limit.
        Default None (all lines treated as detections).

    Examples
    --------
    >>> lines = Lines(
    ...     line_ind   = [59, 63, 71],
    ...     line_names = ["Hbeta", "[OIII]5007", "Halpha"],
    ...     wavelength = [4861., 5007., 6563.],
    ...     flux       = obs_fluxes,
    ...     uncertainty= obs_unc,
    ...     upper_limit= [False, True, False],  # [OIII]5007 is a non-detection
    ... )
    >>> lines.mask_by_name(["[OIII]5007"])   # exclude one line
    >>> chi2 = lines.chi_sq(model_fluxes)
    >>> subset = lines.select_by_name(["Hbeta", "Halpha"])
    """

    _kind = "lines"
    alias = dict(
        spectrum   = "flux",
        unc        = "uncertainty",
        wavelength = "wavelength",
        mask       = "mask",
        line_inds  = "line_ind",     # Prospector-compatible alias
    )
    _meta = ("kind", "name")
    _data = ("wavelength", "flux", "uncertainty", "mask", "line_ind")

    def __init__(
        self,
        line_ind,
        line_names  = None,
        wavelength  = None,
        name        = None,
        upper_limit = None,
        **kwargs,
    ):
        if line_ind is None:
            raise ValueError(
                "line_ind is required: pass the indices of the observed lines "
                "in the FSPS emline_luminosity array."
            )
        self.line_ind   = jnp.asarray(np.atleast_1d(line_ind), dtype=int)
        self.line_names = list(line_names) if line_names is not None else None
        self._wavelength = (
            None if wavelength is None
            else jnp.asarray(np.atleast_1d(wavelength), dtype=float)
        )
        self.upper_limit = (
            None if upper_limit is None
            else jnp.asarray(np.atleast_1d(upper_limit), dtype=bool)
        )
        super().__init__(name=name, **kwargs)

    # ------------------------------------------------------------------
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        """Allow base-class ``rectify`` to set wavelength = None."""
        self._wavelength = (
            None if value is None
            else jnp.asarray(np.atleast_1d(value), dtype=float)
        )

    # ------------------------------------------------------------------
    # GPU / JIT projection interface
    # ------------------------------------------------------------------

    def setup_for_model(self, wave_model, sigma_v=200.0, zred: float = 0.0):
        """
        Precompute the (n_lines, n_wave) Gaussian-aperture weight matrix
        ``_W`` that extracts line fluxes from a model spectrum via a single
        matrix–vector multiply.

        Must be called once before ``predict`` (and before JIT-compiling any
        function containing ``predict``).  ``SedModel.__init__`` calls this
        automatically.

        Physical description
        --------------------
        For each emission line centred at wavelength :math:`\\lambda_k`, the
        integrated line flux is estimated as a Gaussian-weighted integral over
        the model spectrum:

        .. math::

            F_k = \\int w_k(\\lambda)\\, f_\\nu(\\lambda)\\, \\mathrm{d}\\lambda

        where

        .. math::

            w_k(\\lambda) = \\exp\\!\\left[-\\frac{1}{2}
            \\left(\\frac{\\lambda - \\lambda_k}{\\sigma_k}\\right)^2\\right],
            \\quad \\sigma_k = \\lambda_k\\, \\frac{\\sigma_v}{c}

        Discretised with the trapezoidal rule on the model wavelength grid,
        this becomes ``_W @ spectrum`` where
        ``_W[k, j] = w_k(wave_j) * dlambda_j`` and ``dlambda_j`` are the
        trapezoidal quadrature weights.

        Parameters
        ----------
        wave_model : array-like, shape (n_wave,)
            Model wavelength grid [Å], strictly increasing.
        sigma_v : float, optional
            1-sigma Gaussian aperture width [km/s].  Default 200 km/s.
            Sufficient to capture narrow nebular lines as generated by
            ``NebularGridModel`` while excluding continuum and neighbouring
            lines spaced by more than ~600 km/s.  The same aperture is applied
            to both model and data so the absolute calibration cancels in the
            likelihood.
        """
        wm_rest = np.asarray(wave_model,        dtype=np.float64)   # (n_wave,)
        lam0_rest = np.asarray(self._wavelength, dtype=np.float64)   # (n_lines,)
        c_kms  = 2.998e5  # km/s
        opz = 1.0 + float(zred)

        # Both the model grid and the line centres move together into the
        # observed frame by the (1 + zred) factor.  The Gaussian shape is
        # preserved because the velocity aperture sigma_v is defined in
        # velocity units — at higher redshift the wavelength sigma grows
        # proportionally with the line wavelength, so (lambda - lambda_0) /
        # sigma is invariant.
        wm   = opz * wm_rest
        lam0 = opz * lam0_rest

        # Gaussian weight matrix: W[k, j] = exp(-0.5 ((wm_j - lam0_k) / sigma_k)^2)
        sigma_aa = lam0 * (sigma_v / c_kms)           # (n_lines,)
        diff     = wm[None, :] - lam0[:, None]         # (n_lines, n_wave)
        W        = np.exp(-0.5 * (diff / sigma_aa[:, None]) ** 2)

        # Trapezoidal quadrature weights along the (observed-frame) model
        # wavelength axis.  At zred > 0 these pick up one factor of
        # (1 + zred) naturally — this is the dlambda_obs = (1+z) dlambda_rest
        # Jacobian — so integrated line fluxes scale with (1+z) as expected
        # for a redshift-preserving Gaussian aperture.
        dlam        = np.empty(len(wm), dtype=np.float64)
        dlam[1:-1]  = 0.5 * (wm[2:] - wm[:-2])
        dlam[0]     = 0.5 * (wm[1]  - wm[0])
        dlam[-1]    = 0.5 * (wm[-1] - wm[-2])

        # Absorb dlam into the weight matrix.
        W = (W * dlam[None, :]).astype(np.float32)     # (n_lines, n_wave)

        # Store as a JAX constant (XLA constant-folds at trace time).
        self._W = jnp.array(W)

    def predict(self, spectrum, wave_model):
        """
        Extract emission-line fluxes from the model spectrum via Gaussian-
        aperture integration.

        Computes ``_W @ spectrum`` where ``_W`` is the precomputed
        (n_lines, n_wave) weight matrix.  On GPU this is a single GEMV call.

        Must call ``setup_for_model(wave_model)`` before this method.

        Parameters
        ----------
        spectrum : jax.Array, shape (n_wave,)
            Model spectrum in F_nu units.
        wave_model : jax.Array, shape (n_wave,)
            Accepted for interface consistency; not used inside this method.

        Returns
        -------
        jax.Array, shape (n_lines,)
            Gaussian-aperture integrated flux for each line.
        """
        return self._W @ spectrum

    # ------------------------------------------------------------------
    def chi_sq(self, model_fluxes):
        """
        Chi-squared contribution from the observed line fluxes.

        For lines flagged as upper limits (``self.upper_limit[k] = True``),
        the contribution is one-sided: a penalty is applied only when the
        model flux exceeds the observed upper-limit value.

        Parameters
        ----------
        model_fluxes : array-like, shape (n_lines,)
            Predicted line fluxes, same units as ``self.flux``.

        Returns
        -------
        chi2 : float
        """
        mf    = jnp.asarray(model_fluxes, dtype=float)
        resid = (self.flux - mf) / self.uncertainty       # (data - model)/sigma

        if self.upper_limit is not None:
            # For upper-limit lines: only penalise when model > data,
            # i.e., when resid < 0  (model exceeded the observed limit).
            resid_sq = jnp.where(
                self.upper_limit,
                jnp.where(resid < 0.0, resid ** 2, 0.0),
                resid ** 2,
            )
        else:
            resid_sq = resid ** 2

        return float(jnp.sum(jnp.where(self.mask, resid_sq, 0.0)))

    def residuals(self, model_fluxes):
        """
        Per-line ``(data − model) / sigma``.  Masked lines are set to NaN.
        Upper-limit lines where the model does not exceed the limit are set
        to zero (no tension) rather than showing a negative residual.

        Returns
        -------
        res : jnp.ndarray, shape (n_lines,)
        """
        mf    = jnp.asarray(model_fluxes, dtype=float)
        resid = (self.flux - mf) / self.uncertainty

        if self.upper_limit is not None:
            # Show zero residual when model is safely below the upper limit
            resid = jnp.where(
                self.upper_limit & (resid >= 0.0),
                0.0,
                resid,
            )

        return jnp.where(self.mask, resid, jnp.nan)

    # ------------------------------------------------------------------
    def mask_by_name(self, names):
        """
        Exclude lines whose name appears in ``names`` from chi-squared.

        Sets ``self.mask[i] = False`` for all lines whose entry in
        ``self.line_names`` matches any element of ``names``.  A no-op if
        ``self.line_names`` is not set.

        Parameters
        ----------
        names : list of str
        """
        if self.line_names is None:
            return
        names_set = set(names)
        exclude = jnp.array(
            [n in names_set for n in self.line_names], dtype=bool
        )
        self.mask = self.mask & ~exclude

    def select_by_name(self, names):
        """
        Return a new ``Lines`` instance containing only the named lines.

        Parameters
        ----------
        names : list of str
            Must all be present in ``self.line_names``.

        Returns
        -------
        Lines

        Raises
        ------
        ValueError
            If ``self.line_names`` is not set.
        KeyError
            If any element of ``names`` is absent from ``self.line_names``.
        """
        if self.line_names is None:
            raise ValueError(
                "line_names not set on this Lines object; "
                "cannot select by name."
            )
        idx = []
        for n in names:
            if n not in self.line_names:
                raise KeyError(
                    f"Line '{n}' not found.  Available: {self.line_names}"
                )
            idx.append(self.line_names.index(n))
        idx = np.array(idx)

        def _pick(arr):
            return None if arr is None else np.array(arr)[idx]

        return Lines(
            line_ind    = np.array(self.line_ind)[idx],
            line_names  = [self.line_names[i] for i in idx],
            wavelength  = _pick(self._wavelength),
            flux        = _pick(self.flux),
            uncertainty = _pick(self.uncertainty),
            mask        = np.array(self.mask)[idx],
            upper_limit = _pick(self.upper_limit) if self.upper_limit is not None else None,
            name        = self.name + "_sel",
        )

    # ------------------------------------------------------------------
    def __str__(self):
        n = len(self.line_ind)
        if self.line_names is not None:
            names_str = ", ".join(self.line_names[:6])
            if n > 6:
                names_str += f" … (+{n - 6} more)"
        else:
            names_str = f"{n} lines (no names set)"

        if self._wavelength is not None:
            wmin = float(jnp.min(self._wavelength))
            wmax = float(jnp.max(self._wavelength))
            wave_str = f"{wmin:.1f} – {wmax:.1f} Å"
        else:
            wave_str = "none"

        text = [
            f"Lines ({self.name})",
            f"  n_lines       : {n}",
            f"  ndof          : {self.ndof}",
            f"  wavelength    : {wave_str}",
            f"  line_names    : {names_str}",
            f"  masked lines  : {n - self.ndof} / {n}",
        ]
        return "\n".join(text)
