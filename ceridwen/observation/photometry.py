"""
ceridwen/observation/photometry.py
==================================
Broadband photometric observation container.
"""

import json
import jax.numpy as jnp
import numpy as np
from sedpy_jax.observate import FilterSet
from sedpy_jax.smoothing import (
    make_vel_smoother,
    make_wave_smoother,
    make_lsf_smoother,
)
from .base import Observation


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

    def __init__(self, filters=[], name=None, upper_limit=None, **kwargs):
        """
        Parameters
        ----------
        upper_limit : array-like of bool, shape (n_filters,), optional
            Per-band non-detection flags.  If True for band ``i`` the
            photometric likelihood treats that band as an upper limit:
            a chi-squared penalty is applied *only* when the model flux
            exceeds the observed value, i.e.

            .. math::

                \\chi^2_{\\rm UL}
                    = \\left[\\max(m - d, 0) \\,/\\, \\sigma\\right]^2.

            This mirrors the convention already used in
            :class:`ceridwen.observation.Lines` and matches Prospector's
            recommended treatment of non-detections (the simple flux=0,
            sigma=1-sigma-limit approximation).  ``None`` (default) treats
            every band as a positive detection.
        """
        self.set_filters(filters)
        self.upper_limit = (
            None if upper_limit is None
            else jnp.asarray(np.atleast_1d(upper_limit), dtype=bool)
        )
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
        # At zred = 0 this equals wm_rest.
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
    def predict_at_redshift(self, spectrum_fnu_observed, wave_rest, zred):
        """
        Project an observer-frame F_nu spectrum through the filters when
        the redshift is a *traced* (sampled) JAX scalar.

        This is the free-redshift counterpart of :meth:`predict`.  The
        GEMV fast path baked by :meth:`setup_for_model` assumes a single
        Python-scalar ``zred`` was known at trace time and bakes the
        observed-frame wavelength grid into the projection matrix
        ``_T``; that path cannot be used for sampling.  Here the
        observed-frame wavelength grid is reconstructed per-sample as
        ``wave_obs = (1 + zred) * wave_rest`` and the spectrum is
        projected via :meth:`FilterSet.get_sed_maggies` with the
        traced ``sourcewave``.

        Pre-condition: ``spectrum_fnu_observed`` is the observer-frame
        F_nu, i.e. ``CSPBasis.predict`` has already multiplied by
        ``flux_factor_maggies(zred)`` and (optionally) by the IGM
        transmission.  This method only handles the wavelength-grid
        bookkeeping and the filter integral.

        Parameters
        ----------
        spectrum_fnu_observed : jax.Array, shape (n_wave,)
            Observer-frame F_nu on the rest-frame model wavelength grid
            (the standard ceridwen output of get_spectrum + mass +
            flux-factor + IGM).
        wave_rest : jax.Array, shape (n_wave,)
            Rest-frame model wavelength grid [Å] (typically ``csp.wave``).
        zred : jax.Array, scalar
            Sampled redshift.  May be a traced array; the entire path
            below is JIT-compatible and differentiable in ``zred``
            (sedpy_jax's ``interp_source`` uses ``jnp.interp`` which
            has a defined gradient w.r.t. its xp argument).

        Returns
        -------
        jax.Array, shape (n_filters,)
            Synthetic AB maggies in observer frame.

        Notes
        -----
        - Cost is one filter interpolation + one trans-matrix dot per
          sample, vs the single GEMV of the fixed-z path.  For a 14-d
          NUTS / 4000-particle NS / 20 000-step SVI run on a 40 GB A100
          this is ~10-20x slower than the GEMV but still saturates
          the GPU.
        - Numerically equivalent to ``setup_for_model(wave_rest, zred=z)
          + predict(spectrum, wave_rest)`` evaluated at the same z, to
          float32 precision.
        - Works regardless of whether ``setup_for_model`` has been
          called.  When both paths are wired (e.g. for compare-mode
          plots), prefer the GEMV path for any fixed-z observation
          and this method for any free-z observation.
        """
        if self.filterset is None:
            raise ValueError("No FilterSet configured; call set_filters() first.")
        _c = jnp.array(2.998e18, dtype=spectrum_fnu_observed.dtype)
        wave_obs = (jnp.float32(1.0) + zred.astype(spectrum_fnu_observed.dtype)) \
            * jnp.asarray(wave_rest, dtype=spectrum_fnu_observed.dtype)
        flux_flam = jnp.asarray(spectrum_fnu_observed,
                                 dtype=spectrum_fnu_observed.dtype) \
            * _c / (wave_obs * wave_obs)
        return self.filterset.get_sed_maggies(flux_flam, sourcewave=wave_obs)

    # ------------------------------------------------------------------
    def chi_sq(self, model_maggies):
        """
        Chi-squared contribution from this photometric observation.

        For bands flagged as upper limits (``self.upper_limit[i] = True``),
        the contribution is one-sided: a penalty is applied only when the
        model flux exceeds the observed upper-limit value.  Matches the
        convention used in :class:`Lines` and Prospector's recommended
        treatment of non-detections.

        Parameters
        ----------
        model_maggies : array-like, shape (n_filters,)
            Synthetic photometry on the same filter set.

        Returns
        -------
        chi2 : float
        """
        mf    = jnp.asarray(model_maggies, dtype=float)
        resid = (self.flux - mf) / self.uncertainty       # (data - model) / sigma

        if self.upper_limit is not None:
            # For upper-limit bands: only penalise when model > data,
            # i.e. when resid < 0.  Identical to the Lines convention.
            resid_sq = jnp.where(
                self.upper_limit,
                jnp.where(resid < 0.0, resid ** 2, 0.0),
                resid ** 2,
            )
        else:
            resid_sq = resid ** 2

        return float(jnp.sum(jnp.where(self.mask, resid_sq, 0.0)))

    def residuals(self, model_maggies):
        """
        Per-filter (data − model) / sigma.  Masked filters are set to NaN.

        For bands flagged as upper limits, residuals are clipped to 0 when
        the model is safely below the limit (positive residual), so the
        returned vector matches what enters ``chi_sq`` band-by-band.

        Returns
        -------
        res : jnp.ndarray, shape (n_filters,)
        """
        res = (self.flux - jnp.asarray(model_maggies, dtype=float)) / self.uncertainty

        if self.upper_limit is not None:
            res = jnp.where(
                self.upper_limit & (res >= 0.0),
                0.0,
                res,
            )
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

    # ------------------------------------------------------------------
    def _display_str(self, max_rows: int = 80) -> str:
        """Per-filter table: filter name, λ_eff, flux, σ, S/N, mask.

        Filter units follow the ``Photometry`` contract (maggies);
        ``wave_effective`` is pulled from each sedpy_jax ``Filter``."""
        header = str(self)
        flux = np.asarray(self.flux) if self.flux is not None else None
        unc  = np.asarray(self.uncertainty) if self.uncertainty is not None else None
        mask = np.asarray(self.mask)
        n = len(self.filters)

        if flux is None or n == 0:
            return header + "\n  (no flux vector)"

        ul = (np.asarray(self.upper_limit) if self.upper_limit is not None
              else np.zeros(n, dtype=bool))

        col = f"  {'#':<3}  {'filter':<28}  {'λ_eff [Å]':>12}  " \
              f"{'flux [mag]':>14}  {'σ':>12}  {'S/N':>7}  {'mask':>5}  {'UL':>3}"
        sep = "  " + "-" * (len(col) - 2)
        out = [header, "", col, sep]

        for i in range(n):
            fname = self.filternames[i] if i < len(self.filternames) else "?"
            try:
                weff = float(self.filters[i].wave_effective)
            except Exception:
                weff = float("nan")
            f = float(flux[i]) if i < len(flux) else float("nan")
            u = float(unc[i])  if (unc is not None and i < len(unc)) else float("nan")
            snr = (abs(f / u) if (u is not None and np.isfinite(u) and u > 0)
                   else float("inf"))
            m = bool(mask[i]) if i < len(mask) else False
            ulim = bool(ul[i]) if i < len(ul) else False
            ul_str = "UL" if ulim else "-"
            out.append(
                f"  {i:<3d}  {fname:<28}  {weff:>12.1f}  "
                f"{f:>14.4e}  {u:>12.4e}  {snr:>7.2f}  {str(m):>5}  {ul_str:>3}"
            )
        return "\n".join(out)


# =====================================================================
# Spectrum
# =====================================================================

