"""
ceridwen/observation/spectrum.py
================================
Spectroscopic observation container.
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
from .base import Observation, _CKMS

# FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian kernel.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # 1/2.3548
from .gp import GaussianProcess


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
        Wavelength grid [Å], vacuum, **observed frame** (as delivered by the
        instrument).  ``setup_for_model(wave_model, zred=...)`` redshifts the
        rest-frame *model* grid by (1 + zred) and interpolates onto these
        pixels; at ``zred = 0`` observed and rest frame coincide.
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

        * ``smoothtype=None`` — stored but never applied.
        * ``"vel"`` — scalar σ_v [km/s].
        * ``"R"``   — scalar resolving power R = λ/σ_λ.
        * ``"lambda"`` — scalar σ_λ [Å].
        * ``"lsf"`` — 1-D array of σ(λ) [Å] at each observed pixel.
    smoothtype : {"vel", "R", "lambda", "lsf"} or None
        Type of instrumental broadening to apply in ``predict``.  See the
        ``__init__`` docstring and ``setup_for_model`` for full details.
        Default ``None`` (no smoothing, backward-compatible).
    inres : "auto" or float, optional
        Intrinsic resolution of the model (SSP library), subtracted in
        quadrature before applying the target smoothing.  Default
        ``"auto"``: the wavelength-dependent library curve stored in the
        SSP grid (``SSPData.ssp_resolution``) is used, threaded in by
        ``SedModel`` via ``setup_for_model(..., lib_resolution=...)``;
        when smoothing is enabled and no curve is available this is a
        hard error.  An explicit float is an expert override (a constant
        width in the units of ``smoothtype``: km/s for vel/R, Å for
        lambda/lsf; ``0.0`` disables the subtraction entirely).
    calibration : array-like, shape (n_pix,), optional
        Multiplicative flux-calibration vector (model × calibration ≈ data).
        When set, it is applied as a per-pixel multiplicative correction to
        the model inside ``chi_sq``, ``residuals``, and ``log_likelihood``
        (and inside the noise-floor inflation in ``_effective_sigma``).
        For the alternative analytic-marginalisation path, see
        ``fit_polynomial_calibration``.
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
        res_convention = None,
        inres        = "auto",
        sky          = None,
        noise_floor  = 0.0,
        sigma_losvd  = None,
        fit_sigma_smooth = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        smoothtype : {"vel", "R", "lambda", "lsf"} or None
            Which instrumental smoothing kernel to apply in ``predict``:

            ``"vel"``
                Constant velocity dispersion.  ``resolution`` is a velocity
                width [km/s].  Uses a log-λ FFT so the kernel is
                shift-invariant in velocity.
            ``"R"``
                Constant spectral resolving power.  ``resolution`` is the
                scalar R value.  Because "R" is quoted in BOTH conventions
                in the literature (R = λ/FWHM on instrument datasheets,
                e.g. NIRSpec "R = 1000"; R = λ/σ_λ in the sedpy/Prospector
                tradition), ``res_convention`` is REQUIRED for this
                smoothtype — there is no safe default.
            ``"lambda"``
                Constant wavelength width.  ``resolution`` is in Å.
                Uses a linear-λ FFT.
            ``"lsf"``
                Wavelength-dependent line-spread function.  ``resolution``
                must be a 1-D array of the wavelength width [Å] evaluated
                at the *observed* pixel wavelengths (``self.wavelength``).
                The kernel is interpolated to the model wavelength grid
                inside ``setup_for_model``.
            ``None`` (default)
                No smoothing applied; ``predict`` performs pure linear
                interpolation (``_H @ spectrum``).  ``resolution`` is stored
                but unused in this mode.

        res_convention : {"sigma", "fwhm"} or None
            Whether ``resolution`` quotes a Gaussian dispersion σ or a
            FWHM (FWHM = 2.3548 σ).  Applies to every smoothtype:

            * ``"sigma"`` — ``resolution`` is σ_v ("vel"), R = λ/σ_λ
              ("R"), or σ_λ ("lambda"/"lsf").
            * ``"fwhm"``  — ``resolution`` is FWHM_v ("vel"),
              R = λ/FWHM_λ ("R", the usual instrument-datasheet number),
              or FWHM_λ ("lambda"/"lsf").

            Default: ``"sigma"`` for "vel"/"lambda"/"lsf" (the standard
            convention for those quantities).  For ``smoothtype="R"`` it
            MUST be given explicitly — passing a datasheet R = λ/FWHM
            where λ/σ is expected silently over-broadens the model by
            2.3548×, so the constructor raises rather than guess.

        inres : "auto" or float, optional
            Intrinsic (library) resolution of the input model spectrum,
            subtracted in quadrature before applying the target smoothing.
            ``"auto"`` (default) uses the wavelength-dependent curve stored
            in the schema-2.x SSP grid (see the class docstring) — leave it
            alone unless you know better.  An explicit float is a constant
            expert override, ALWAYS a Gaussian σ (never FWHM), in km/s for
            "vel"/"R" and Å for "lambda"/"lsf"; 0.0 disables the
            subtraction entirely.
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
        # ── Resolution convention (σ vs FWHM) ─────────────────────────────
        if smoothtype is None:
            if res_convention is not None:
                raise ValueError(
                    "Spectrum: res_convention was given but smoothtype is "
                    "None (no smoothing) — remove res_convention or set a "
                    "smoothtype."
                )
        else:
            if res_convention is None:
                if smoothtype == "R":
                    raise ValueError(
                        "Spectrum: smoothtype='R' requires "
                        "res_convention='fwhm' or res_convention='sigma', "
                        "because published R values use both conventions "
                        "and they differ by 2.3548x.  Use 'fwhm' if your R "
                        "is an instrument-datasheet resolving power "
                        "R = lambda/FWHM (e.g. NIRSpec 'R = 1000'); use "
                        "'sigma' if it is R = lambda/sigma_lambda (the "
                        "sedpy/Prospector convention)."
                    )
                res_convention = "sigma"
            if res_convention not in ("sigma", "fwhm"):
                raise ValueError(
                    f"Spectrum: res_convention={res_convention!r} is not "
                    f"recognised; use 'sigma' or 'fwhm'."
                )
        self.res_convention  = res_convention
        self.inres           = ("auto" if (isinstance(inres, str)
                                           and inres == "auto")
                                else float(inres))
        self.sky             = (None if sky is None
                                else jnp.asarray(sky, dtype=float))
        self.noise_floor     = float(noise_floor)
        # ── Galaxy LOSVD (sigma_smooth in Prospector convention) ──────────
        # When ``fit_sigma_smooth=False`` (default), ``sigma_losvd`` is
        # baked into ``_predict_fn`` at ``setup_for_model`` time as a
        # Python float.  When ``fit_sigma_smooth=True``, the closure
        # instead accepts a runtime ``sigma_smooth`` jnp scalar (km/s) and
        # the caller (CSPBasis.predict) passes ``theta["sigma_smooth"]``
        # through.  In that fittable mode ``sigma_losvd`` is only used as
        # the warmup / smoother-init value, so we default to 200 km/s
        # (Prospector ``TemplateLibrary["spectral_smoothing"]`` init) when
        # the user does not supply one.
        self.fit_sigma_smooth = bool(fit_sigma_smooth)
        if self.fit_sigma_smooth and sigma_losvd is None:
            sigma_losvd = 200.0
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
    def _resolution_as_sigma(self):
        """``self.resolution`` converted to the internal Gaussian-σ form.

        Returns σ_v [km/s] for smoothtype "vel"/"R", σ_λ [Å] (scalar) for
        "lambda", or the σ_λ(λ_obs) [Å] array for "lsf".  The
        ``res_convention`` chosen at construction is applied here — a
        FWHM-quoted resolution is divided by 2√(2 ln 2) ≈ 2.3548 (for
        "R", R = λ/FWHM implies σ_v = c/(R·2.3548), i.e. the same
        factor).  All downstream smoothing code is σ-based.
        """
        f = (_FWHM_TO_SIGMA if self.res_convention == "fwhm" else 1.0)
        st = self.smoothtype
        if st == "vel":
            return float(self.resolution) * f
        if st == "R":
            return float(_CKMS / self.resolution) * f
        if st == "lambda":
            return float(self.resolution) * f
        return np.asarray(self.resolution, dtype=np.float64) * f  # "lsf"

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

    def _warn_floor(self, target, lib, units, wave=None):
        """Setup-time diagnostics for the library-resolution subtraction.

        Warns (once, Python-level) when the quadrature floor engages —
        i.e. the library is coarser than the requested resolution, so the
        model CANNOT be sharpened to the target there — and when the
        library curve is unknown (NaN) over part of the fitted window, so
        nothing is subtracted there.
        """
        import warnings as _w
        target = np.asarray(target, dtype=np.float64)
        lib    = np.asarray(lib, dtype=np.float64)
        fin    = np.isfinite(lib)
        if fin.any():
            bad = fin & (lib >= target)
            if bad.any():
                frac = 100.0 * bad.sum() / bad.size
                rng = ("" if wave is None else
                       f" over [{np.asarray(wave)[bad].min():.0f}, "
                       f"{np.asarray(wave)[bad].max():.0f}] AA")
                _w.warn(
                    f"Spectrum {self.name!r}: the SSP library resolution "
                    f"meets or exceeds the target resolution on "
                    f"{frac:.0f}% of pixels{rng} ({units}); those pixels "
                    f"stay at LIBRARY resolution (no deconvolution is "
                    f"attempted).  The model cannot be as sharp as the "
                    f"data there — use a higher-resolution SSP grid.",
                    stacklevel=3)
        if (~fin).any():
            frac = 100.0 * (~fin).sum() / fin.size
            rng = ("" if wave is None else
                   f" (e.g. [{np.asarray(wave)[~fin].min():.0f}, "
                   f"{np.asarray(wave)[~fin].max():.0f}] AA)")
            _w.warn(
                f"Spectrum {self.name!r}: the library resolution is "
                f"unknown (NaN) on {frac:.0f}% of the fitted pixels{rng}; "
                f"no library subtraction is applied there and the model "
                f"is over-broadened by the (unknown) library width.",
                stacklevel=3)

    def setup_for_model(self, wave_model, zred: float = 0.0,
                        lib_resolution=None):
        """
        Precompute projection matrices and/or smoothing kernels, then build
        ``_predict_fn`` — the single callable used by ``predict``.

        Must be called once (Python-level, outside JIT) after the model
        wavelength grid is known.  ``SedModel.__init__`` calls this
        automatically.

        Two behaviours depending on ``self.smoothtype``:

        **No smoothing** (``smoothtype=None``)
            Builds the dense (n_pix, n_wave) linear-interpolation matrix
            ``_H`` and sets ``_predict_fn(spec) = _H @ spec``.

        **With instrumental smoothing** (``smoothtype`` in
        ``{"vel", "R", "lambda", "lsf"}``)
            Uses a factory function from ``sedpy_jax.smoothing`` to
            precompute all FFT grid transforms.  The returned closure is
            fully JAX-JIT-compilable with respect to the spectrum.
            ``_predict_fn(spec)`` applies smoothing *and* interpolation to
            the observed pixel grid in one call.  ``_H`` is also built (used
            only by the no-smoothing fast path).

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
            the predict-time GEMV fast path.
        lib_resolution : (wave_rest, sigma_v_kms) tuple, optional
            The SSP library resolution curve: rest-frame wavelengths [Å]
            and per-pixel Gaussian dispersion sigma_v [km/s] (NaN where
            unknown), as stored in ``SSPData.ssp_resolution``.
            ``SedModel`` threads this in automatically.  Consumed only
            when ``self.inres == "auto"`` and instrumental smoothing is
            enabled; sigma_v is frame-invariant, so the curve applies at
            observed pixel ``(1+z)·λ_rest`` unchanged.  Required in that
            case — a smoothing-enabled Spectrum with ``inres="auto"`` and
            no curve raises (pass an explicit ``inres`` float to opt out).
        """
        if self._wavelength is None:
            raise ValueError(
                "Spectrum.setup_for_model() needs the observed pixel "
                "wavelength grid, but this Spectrum has wavelength=None. "
                "Pass wavelength= (Å, vacuum, observed frame) at "
                "construction — flux/uncertainty may be added later (e.g. a "
                "predictive container for mock generation)."
            )
        wm_rest = np.asarray(wave_model, dtype=np.float64)
        opz = 1.0 + float(zred)
        wm = opz * wm_rest
        wo = np.asarray(self._wavelength, dtype=np.float64)
        n_wave = len(wm)
        n_pix  = len(wo)

        # ── Always build the dense interpolation matrix _H ────────────────
        # (used by the no-smoothing fast path)
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
        # When the galaxy LOSVD is a free parameter, the closure has a
        # runtime ``sigma_smooth`` argument; ``predict`` switches its
        # signature accordingly.  The constructor injects a 200 km/s
        # default when ``fit_sigma_smooth=True`` and ``sigma_losvd is
        # None``, so a valid smoother can always be built here.
        fit_lo     = self.fit_sigma_smooth and has_losvd

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
                    # Runtime-sigma variant: takes (spec_trim, sigma_v)
                    # and is differentiable w.r.t. sigma_v (sedpy_jax's
                    # make_vel_smoother already supports this).
                    def _apply_losvd_rt(spec_trim, sigma_v, _s=_losvd_sm):
                        return _s(spec_trim, sigma_v)
                else:
                    # LOSVD only: output goes directly to observed grid.
                    _losvd_sm = make_vel_smoother(_wm_trim, wo, inres=0.0)
                    _sv       = _sv_losvd
                    def _apply_losvd(spec_trim, _s=_losvd_sm, _v=_sv):
                        return _s(spec_trim, _v)
                    def _apply_losvd_rt(spec_trim, sigma_v, _s=_losvd_sm):
                        return _s(spec_trim, sigma_v)

            if has_instr:
                if st not in ("vel", "R", "lambda", "lsf"):
                    raise ValueError(
                        f"Spectrum.smoothtype={st!r} is not recognised.  "
                        "Valid choices: None, 'vel', 'R', 'lambda', 'lsf'."
                    )

                # ── Library (input) resolution on the trimmed model grid ──
                # sigma_v is frame-invariant, so the rest-frame curve is
                # evaluated at λ_rest = λ_obs/(1+z) and used unchanged.
                # NaN = "unknown here" → nothing subtracted at that pixel.
                auto = (self.inres == "auto")
                if auto and lib_resolution is None:
                    raise ValueError(
                        f"Spectrum {self.name!r}: smoothtype={st!r} needs "
                        "the SSP library resolution to subtract in "
                        "quadrature, but inres='auto' and no "
                        "lib_resolution curve was provided.  Load a "
                        "schema-2.0 SSP grid (SedModel threads its curve "
                        "in automatically), or pass an explicit inres "
                        "float to override."
                    )
                if auto:
                    _lw = np.asarray(lib_resolution[0], dtype=np.float64)
                    _ls = np.asarray(lib_resolution[1], dtype=np.float64)
                    if _lw.shape != _ls.shape:
                        raise ValueError(
                            "lib_resolution wave/sigma shape mismatch: "
                            f"{_lw.shape} vs {_ls.shape}")
                    # NaN-aware interpolation onto the trimmed grid: NaN
                    # segments stay NaN (np.interp would smear them).
                    _rest_trim = _wm_trim / opz
                    fin = np.isfinite(_ls)
                    sig_v_lib = np.full(_wm_trim.shape, np.nan)
                    if fin.any():
                        sig_v_lib = np.where(
                            (_rest_trim >= _lw[fin].min())
                            & (_rest_trim <= _lw[fin].max()),
                            np.interp(_rest_trim, _lw[fin], _ls[fin]),
                            np.nan)
                        # pixels inside coverage but adjacent to NaN gaps:
                        # mark as NaN when the nearest native pixel is NaN
                        near = np.clip(np.searchsorted(_lw, _rest_trim),
                                       0, _lw.size - 1)
                        sig_v_lib = np.where(np.isfinite(_ls[near]),
                                             sig_v_lib, np.nan)
                else:
                    sig_v_lib = None      # explicit scalar override path

                # ── Target width and the library curve, per smoothtype ────
                # Everything reduces to ONE of two smoother kinds:
                #   scalar-σ FFT (shift-invariant target, constant or no
                #   library curve) or the CDF-transform LSF smoother
                #   (any wavelength dependence, in target or library).
                def _flat(a):
                    f = a[np.isfinite(a)]
                    return f.size and (np.ptp(f) < 1e-6) and \
                        np.isfinite(a).all()

                _sig_lam = _wm_trim / _CKMS   # dλ/dv at each trimmed pixel

                # An all-NaN curve (resolution unknown everywhere) subtracts
                # nothing: take the scalar inres=0 fast path so the result
                # is BIT-identical to an explicit inres=0.0, and warn.
                all_nan = auto and not np.isfinite(sig_v_lib).any()
                if all_nan:
                    self._warn_floor(np.full_like(_wm_trim, np.inf),
                                     sig_v_lib, "km/s", _wm_trim)

                if st in ("vel", "R"):
                    sigma_v_t = self._resolution_as_sigma()
                    if not auto or all_nan:
                        _in = 0.0 if all_nan else float(self.inres)
                        _instr_sm = make_vel_smoother(
                            _wm_trim, wo, inres=_in)
                        _apply_instr = (lambda spec_trim, _sm=_instr_sm,
                                        _sv=sigma_v_t: _sm(spec_trim, _sv))
                        self._warn_floor(np.full(1, sigma_v_t),
                                         np.full(1, _in), "km/s")
                    elif _flat(sig_v_lib):
                        _in = float(sig_v_lib[np.isfinite(sig_v_lib)][0])
                        _instr_sm = make_vel_smoother(_wm_trim, wo, inres=_in)
                        _apply_instr = (lambda spec_trim, _sm=_instr_sm,
                                        _sv=sigma_v_t: _sm(spec_trim, _sv))
                        self._warn_floor(np.full(1, sigma_v_t),
                                         np.full(1, _in), "km/s")
                    else:
                        # wavelength-dependent library curve → LSF route in
                        # wavelength space: σ_λ = λ σ_v / c for both target
                        # and library.
                        tgt = sigma_v_t * _sig_lam
                        lib = sig_v_lib * _sig_lam
                        _instr_sm = make_lsf_smoother(
                            _wm_trim, tgt, wo, inres=lib)
                        _apply_instr = (lambda spec_trim, _sm=_instr_sm:
                                        _sm(spec_trim))
                        self._warn_floor(np.full_like(_wm_trim, sigma_v_t),
                                         sig_v_lib, "km/s", _wm_trim)

                elif st == "lambda":
                    sigma_l_t = self._resolution_as_sigma()
                    if not auto or all_nan:
                        _in = 0.0 if all_nan else float(self.inres)
                        _instr_sm = make_wave_smoother(
                            _wm_trim, wo, inres=_in)
                        _apply_instr = (lambda spec_trim, _sm=_instr_sm,
                                        _sl=sigma_l_t: _sm(spec_trim, _sl))
                        self._warn_floor(np.full(1, sigma_l_t),
                                         np.full(1, _in), "AA")
                    else:
                        # constant-λ target, velocity-defined library curve
                        # → always wavelength-dependent in λ: LSF route.
                        tgt = np.full_like(_wm_trim, sigma_l_t)
                        lib = sig_v_lib * _sig_lam
                        _instr_sm = make_lsf_smoother(
                            _wm_trim, tgt, wo, inres=lib)
                        _apply_instr = (lambda spec_trim, _sm=_instr_sm:
                                        _sm(spec_trim))
                        self._warn_floor(tgt, lib, "AA", _wm_trim)

                else:   # st == "lsf"
                    # Wavelength-dependent target LSF: resolution is σ(λ)
                    # [Å] at the *observed* pixel grid → trimmed model grid.
                    res_obs        = self._resolution_as_sigma()
                    sigma_lsf_trim = np.interp(_wm_trim, wo, res_obs)
                    if not auto or all_nan:
                        lib = 0.0 if all_nan else float(self.inres)
                        _instr_sm = make_lsf_smoother(
                            _wm_trim, sigma_lsf_trim, wo, inres=lib)
                        self._warn_floor(sigma_lsf_trim,
                                         np.full_like(sigma_lsf_trim, lib),
                                         "AA", _wm_trim)
                    else:
                        lib = sig_v_lib * _sig_lam
                        _instr_sm = make_lsf_smoother(
                            _wm_trim, sigma_lsf_trim, wo, inres=lib)
                        self._warn_floor(sigma_lsf_trim, lib, "AA", _wm_trim)
                    _apply_instr = (lambda spec_trim, _sm=_instr_sm:
                                    _sm(spec_trim))

                # ── One wrapper set for every smoothtype ──────────────────
                if fit_lo:
                    _Lrt = _apply_losvd_rt
                    self._predict_fn = (
                        lambda spec, sigma_lo, _A=_apply_instr, _L=_Lrt:
                            _A(_L(spec[_idx], sigma_lo))
                    )
                elif has_losvd:
                    _L = _apply_losvd
                    self._predict_fn = (
                        lambda spec, _A=_apply_instr, _L=_L:
                            _A(_L(spec[_idx]))
                    )
                else:
                    self._predict_fn = (
                        lambda spec, _A=_apply_instr: _A(spec[_idx])
                    )

            else:
                # LOSVD only (no instrumental smoothing); _apply_losvd
                # already maps _wm_trim → wo (observed grid).
                if fit_lo:
                    _Lrt = _apply_losvd_rt
                    self._predict_fn = (
                        lambda spec, sigma_lo, _L=_Lrt:
                            _L(spec[_idx], sigma_lo)
                    )
                else:
                    _L = _apply_losvd
                    self._predict_fn = lambda spec, _L=_L: _L(spec[_idx])

    def predict(self, spectrum, wave_model, sigma_smooth=None):
        """
        Project the model spectrum onto the observed pixel grid, applying
        instrumental smoothing if configured.

        Calls ``_predict_fn(spectrum[, sigma_smooth])`` which was constructed
        by ``setup_for_model``.  Depending on ``self.smoothtype``:

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
        sigma_smooth : jax.Array scalar, optional
            Runtime galaxy LOSVD [km/s] -- the Prospector ``sigma_smooth``
            parameter.  Only consulted when this Spectrum was constructed
            with ``fit_sigma_smooth=True``; ignored otherwise (the static
            ``sigma_losvd`` baked at ``setup_for_model`` time is used
            instead).  Passing the value from ``theta`` makes the LOSVD
            differentiable and fittable inside JIT/SVI/NUTS.

        Returns
        -------
        jax.Array, shape (n_pix,)
            Model F_nu (smoothed and) interpolated onto ``self.wavelength``.
        """
        # Clear error instead of a cryptic AttributeError on the projection
        # closure if setup was skipped.  ``hasattr`` is a static Python check,
        # so inside a jit trace it resolves at compile time (no hot-path cost).
        if not hasattr(self, "_predict_fn"):
            raise RuntimeError(
                "Spectrum.predict() called before setup_for_model(): the "
                "projection/smoothing closure has not been built. Call "
                "spec.setup_for_model(wave_model) once (before the first "
                "predict / JIT trace)."
            )
        if self.fit_sigma_smooth:
            if sigma_smooth is None:
                # Fall back to the constructor default; lets a caller
                # invoke ``predict(spec, wave)`` for warmup / debugging
                # even when the closure is the runtime-sigma variant.
                sigma_smooth = self.sigma_losvd
            # Pass sigma through without forcing a dtype -- JAX's
            # dtype-promotion rules with jax_enable_x64=True will
            # promote to float64 to match the cached smoother grids,
            # giving the same precision the static fast path achieved.
            # Forcing float32 here would round-trip-degrade the smoother
            # output even when the user is fitting in double precision.
            sv = jnp.asarray(sigma_smooth).reshape(())
            return self._predict_fn(spectrum, sv)
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

    def mask_lines(self, line_waves, dv=1000.0, zred=0.0):
        """
        Mask spectral lines by zeroing the mask within ±dv km/s of each line.

        Parameters
        ----------
        line_waves : array-like
            Rest-frame central wavelengths [Å]; redshifted internally by
            (1 + ``zred``) before masking against the observed-frame pixel
            grid.
        dv : float
            Half-width to mask on each side [km/s].  Default 1000 km/s.
        zred : float
            Redshift to apply to ``line_waves``.  Default 0.0 (preserves
            pre-fix behaviour for callers already passing observed-frame
            wavelengths).
        """
        if self._wavelength is None:
            return
        c_kms = 2.998e5   # km/s
        opz   = 1.0 + float(zred)
        for lam0_rest in np.asarray(line_waves).ravel():
            lam0_obs = opz * float(lam0_rest)
            dlam     = lam0_obs * dv / c_kms
            self.mask_wavelength_range(lam0_obs - dlam, lam0_obs + dlam)

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

    def _compute_residuals(self, model_flux, return_sigma=False):
        """
        Internal: (sky-subtracted data − calibration × model) / sigma_eff,
        per pixel.  Accounts for sky subtraction, multiplicative flux
        calibration, and noise-floor inflation (which is referenced to the
        *calibrated* model amplitude).
        """
        mf    = jnp.asarray(model_flux, dtype=float)
        # Static Python branch — same pattern as ``self.sky``; keeps the JIT
        # trace branch-free.
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
                -\\tfrac{1}{2}\\,\\mathbf{r}^\\top \\Sigma^{-1}\\mathbf{r}
                -\\tfrac{1}{2}\\ln|\\Sigma|
                -\\tfrac{1}{2}\\sum_{i\\,\\in\\,\\rm mask}\\ln\\!\\big(2\\pi\\sigma_{\\rm eff,i}^2\\big)

        where :math:`r_i = ((d_i - s_i) - c_i m_i)/\\sigma_{\\rm eff,i}`
        (sky :math:`s_i` and calibration :math:`c_i` are optional; the
        noise floor in :math:`\\sigma_{\\rm eff,i}` is scaled against the
        calibrated model :math:`c_i m_i` when ``calibration`` is set), and
        the GP term is zero if no noise model is set.

        Parameters
        ----------
        model_flux : array-like, shape (n_pix,)
            Model flux on the observed pixel grid.

        Returns
        -------
        float
            Log-likelihood (larger is better).
        """
        resid, sigma_r = self._compute_residuals(model_flux, return_sigma=True)
        mask = self.mask

        # Whitening Jacobian of r = (data - model) / sigma_r:
        #     -1/2 sum_{i in mask} ln(sigma_r_i^2)  ( = -sum ln sigma_r_i ).
        # The 2*pi factor / the -n/2 ln(2 pi) constant is already carried
        # by ``lnL_struct`` below (both the GP and the diagonal branch
        # include it), so only the log-variance Jacobian is added here.
        # It is retained -- not dropped to a bare chi^2 -- because sigma_r
        # can depend on the model through the noise floor.
        safe_sigma = jnp.where(mask, sigma_r, 1.0)
        lognorm    = float(
            -0.5 * jnp.sum(jnp.where(mask, jnp.log(safe_sigma ** 2), 0.0))
        )

        if self.noise is not None and self._wavelength is not None:
            # GP path: a SINGLE Gaussian on the normalised residuals with
            # covariance Sigma = I + a^2 SE + eps I.  GaussianProcess.
            # log_likelihood already folds in the white-noise identity and
            # the -n/2 ln(2 pi) constant, so we do NOT add a separate
            # -1/2 sum r^2 here (that would double-count the white noise).
            lnL_struct = float(self.noise.log_likelihood(
                np.array(resid),
                np.array(self._wavelength),
                np.array(mask),
            ))
        else:
            # Diagonal Gaussian on the normalised residuals: r ~ N(0, I).
            r = jnp.where(mask, resid, 0.0)
            n = float(jnp.sum(mask))
            lnL_struct = float(
                -0.5 * jnp.sum(r ** 2) - 0.5 * n * jnp.log(2.0 * jnp.pi)
            )

        return float(lnL_struct + lognorm)

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
            res_str = f"{float(self.resolution):.1f}"
        else:
            res_str = f"array, shape {np.shape(self.resolution)}"
        if self.resolution is not None and self.smoothtype is not None:
            res_str += f" ({self.res_convention} convention)"

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

    # ------------------------------------------------------------------
    def _display_str(self, max_rows: int = 20) -> str:
        """Per-pixel table: index, wavelength, flux, σ, mask.  Long
        spectra (more than ``max_rows`` pixels) are head/tail truncated
        with a ``...`` marker in the middle.  A summary row appended at
        the end reports min / median / max flux and the mean S/N over
        unmasked pixels — the row view is for spot-checking values;
        these aggregates give the first-pass sanity check."""
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
            idxs = head + [None] + tail  # None → ellipsis row

        for i in idxs:
            if i is None:
                out.append("  ...")
            else:
                out.append(_row(i))

        # Aggregate stats over unmasked pixels — quick sanity numbers.
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


# =====================================================================
# Lines
# =====================================================================

