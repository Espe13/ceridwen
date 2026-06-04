"""
ceridwen/observation/lines.py
=============================
Split out of the former monolithic ``observation.py`` (review 2026-06-01).
Class body is byte-identical to the original.
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
        if wavelength is None:
            raise ValueError(
                "wavelength is required: pass the wavelengths of the observed lines."
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

            ``sigma_v`` is a construction-time hyperparameter; it is not part
            of ``theta`` and is not differentiable through ``predict``.
            Catalogued line fluxes are single scalars per line and carry no
            shape information, so HMC cannot constrain it.
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

        # Trapezoidal quadrature weights along the (observed-frame) model
        # wavelength axis.  At zred > 0 these pick up one factor of
        # (1 + zred) naturally — this is the dlambda_obs = (1+z) dlambda_rest
        # Jacobian — so integrated line fluxes scale with (1+z) as expected
        # for a redshift-preserving Gaussian aperture.
        dlam        = np.empty(len(wm), dtype=np.float64)
        dlam[1:-1]  = 0.5 * (wm[2:] - wm[:-2])
        dlam[0]     = 0.5 * (wm[1]  - wm[0])
        dlam[-1]    = 0.5 * (wm[-1] - wm[-2])

        # Absolute-flux normalisation.  Without this per-line factor,
        # ``_W @ F_nu`` returns ``F_line * lambda_obs**2 / c`` (units:
        # erg s^-1 cm^-2 Hz^-1 * A — a mixed-unit "aperture proxy"), NOT
        # the integrated line flux in erg s^-1 cm^-2.  The raw proxy is
        # fine if you feed *both* data and model through the same
        # aperture (the docstring's "calibration cancels" regime), but
        # catalogue emission-line tables almost always quote already-
        # reduced integrated line fluxes in erg s^-1 cm^-2 — so we
        # normalise once here and have ``_W @ spectrum`` return flux
        # in the catalogue's own unit system.
        #
        # Derivation: FSPS Cloudy lines are added to the spectrum with
        # a Gaussian of width sigma_v_model = nebular_smooth_init km/s
        # (floor of ~2 pixel widths).  This is typically narrower than
        # the sigma_v = 200 km/s aperture used here.  In the narrow-
        # model-line limit the aperture integral reduces to
        #   _W @ F_nu ≈ F_line * lambda_obs^2 / c .
        # Multiplying each row by c / lambda_obs^2 restores
        #   _W @ F_nu ≈ F_line [erg s^-1 cm^-2] ,
        # letting observed data in the same units be passed in directly
        # as ``Lines.flux``.
        c_aa_s = 2.998e18                          # speed of light [Å/s]
        norm = c_aa_s / (lam0 ** 2)                # (n_lines,)

        # Bake W into a static (n_lines, n_wave) JAX constant.  XLA
        # constant-folds at trace time.
        diff     = wm[None, :] - lam0[:, None]         # (n_lines, n_wave)
        sigma_aa = lam0 * (sigma_v / c_kms)            # (n_lines,)
        W = np.exp(-0.5 * (diff / sigma_aa[:, None]) ** 2)
        W = (W * dlam[None, :]).astype(np.float32)     # (n_lines, n_wave)
        W = (W * norm[:, None].astype(np.float32))
        self._W = jnp.array(W)

    def predict(self, spectrum, wave_model):
        """
        Extract emission-line fluxes from the model spectrum via Gaussian-
        aperture integration: computes ``_W @ spectrum`` where ``_W`` was
        precomputed once in ``setup_for_model``.  On GPU this is a single
        GEMV; XLA constant-folds ``_W`` into the compiled graph.

        Must call ``setup_for_model(wave_model, sigma_v=...)`` first.

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
        if not hasattr(self, "_W"):
            raise RuntimeError(
                "Lines.predict() called before setup_for_model(): the "
                "Gaussian aperture weight matrix has not been built. "
                "Call lines.setup_for_model(wave_model, sigma_v=...) "
                "once before the first predict / JIT trace."
            )
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

    # ------------------------------------------------------------------
    def _display_str(self, max_rows: int = 80) -> str:
        """Per-line table: #, line name, FSPS idx (1-based), λ, flux,
        σ, S/N, mask.  Used as a sanity check after building a Lines
        observation — all emission-line mysteries (missing lines,
        swapped FSPS indices, wrong units) show up here."""
        header = str(self)
        if self.flux is None or len(self.line_ind) == 0:
            return header + "\n  (no line vector)"

        inds  = np.asarray(self.line_ind)
        flux  = np.asarray(self.flux)
        unc   = (np.asarray(self.uncertainty)
                 if self.uncertainty is not None
                 else np.full_like(flux, np.nan))
        mask  = np.asarray(self.mask)
        waves = (np.asarray(self._wavelength)
                 if self._wavelength is not None
                 else np.full(len(inds), np.nan))
        names = (self.line_names if self.line_names is not None
                 else [f"line_{i}" for i in range(len(inds))])

        col = (f"  {'#':<3}  {'line_name':<28}  {'idx':>4}  "
               f"{'λ [Å]':>10}  {'flux':>14}  {'σ':>14}  "
               f"{'S/N':>7}  {'mask':>5}")
        sep = "  " + "-" * (len(col) - 2)
        out = [header, "", col, sep]

        for i in range(len(inds)):
            name = names[i] if i < len(names) else "?"
            idx  = int(inds[i])
            w    = float(waves[i]) if i < len(waves) else float("nan")
            f    = float(flux[i])
            u    = float(unc[i]) if i < len(unc) else float("nan")
            snr  = (abs(f / u) if np.isfinite(u) and u > 0
                    else float("inf"))
            m    = bool(mask[i]) if i < len(mask) else False
            out.append(
                f"  {i:<3d}  {name:<28}  {idx:>4d}  "
                f"{w:>10.2f}  {f:>14.4e}  {u:>14.4e}  "
                f"{snr:>7.2f}  {str(m):>5}"
            )
        return "\n".join(out)
