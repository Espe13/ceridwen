"""Spectral broadening: the one place where model features get their width.

    continuum:  sigma_cont_i^2 = sigma_gal^2 + sigma_inst_i^2 - sigma_lib_i^2
    lines:      sigma_line_i^2 = sigma_gas^2 + sigma_inst_i^2

sigma_gal, sigma_gas  galaxy stars / gas [km/s], ``Kinematics``, fixed or theta keys
sigma_inst            instrument LSF at the observed pixel [km/s], ``Instrument``, times its
                      ``scale`` s (default 1; fixed, or a theta key) in BOTH lines above
sigma_lib             SSP library resolution at the rest pixel [km/s], ``SSPData.ssp_resolution``

All sigma are dispersions.  Kernels are Gaussians in ln(lambda) acting on f_nu.
sigma_inst is the LSF as measured on the detector (pixelisation included), so
profiles are sampled at pixel centres.

Runtime path (Spectrum):  spec_rest --gather+interp--> log grid (dv = finest model
pixel; read at the sampled redshift when z is free) --FFT sigma_gal--> --static banded
response (sigma_inst - sigma_lib, resampling to observed pixels)--> + analytic lines
(sigma_gas, sigma_inst).  With a sampled instrument scale the response weights are
recomputed per call on the static band (``ScaledResponse``); nothing else changes.
Runtime path (Photometry): the FFT stage only, scattered back into spec_rest.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import jax.numpy as jnp

__all__ = [
    "CKMS", "C_AA_S", "FWHM_TO_SIGMA", "BAND_NSIGMA",
    "Instrument", "Kinematics", "TIED", "DEFAULT_KINEMATICS",
    "LogGrid", "WindowSmoother", "make_gaussian_fft", "build_response", "apply_response",
    "ScaledResponse", "check_scale_range",
    "make_line_painter_free_z",
    "make_line_painter", "SpectralProjector", "PhotometricBroadener",
    "agn_line_profiles_on_grid", "make_agn_line_painter_prospector",
]

CKMS = 2.99792458e5
C_AA_S = 2.99792458e18
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
BAND_NSIGMA = 5.0
# Half-width of a line window in units of the line's largest sigma: exp(-x^2 / 2) is exactly
# 0 in float64 beyond |x| = 38.60 (37.64 on XLA CPU, which flushes subnormals to zero).
_LINE_WINDOW_NSIGMA = 40.0


def check_scale_range(rng, what="scale_range") -> tuple:
    """(lo, hi) of a sampled LSF scale as floats; finite, 0 < lo < hi, else ValueError."""
    try:
        lo, hi = (float(np.ravel(np.asarray(v, dtype=float))[0]) for v in rng)
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{what} must be a pair (lo, hi), got {rng!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{what} must be finite (a bounded prior), got ({lo}, {hi})")
    if not (0.0 < lo < hi):
        raise ValueError(f"{what} must satisfy 0 < lo < hi (the LSF scale multiplies a "
                         f"dispersion), got ({lo}, {hi})")
    return (lo, hi)


@dataclass(frozen=True, eq=False)
class Instrument:
    """Line-spread function of a spectrograph; unit and convention are in the constructor name.

        Instrument.R_fwhm(2700)                  R = lambda/FWHM   (datasheet)
        Instrument.R_sigma(6358)                 R = lambda/sigma  (same LSF as R_fwhm(2700))
        Instrument.fwhm_aa(2.5)                  FWHM [A], observed frame
        Instrument.sigma_aa(1.06)                sigma [A], observed frame
        Instrument.sigma_kms(47.0)               sigma [km/s]
        Instrument.fwhm_kms(111.0)               FWHM [km/s]
        Instrument.<any>(array, wave=obs_wave)   per-pixel value on observed wavelengths

    LSF scale (every constructor takes ``scale=`` and ``scale_range=``): the instrumental
    dispersion in force is ``s * sigma_inst(lambda)``, in the continuum kernel
    (sigma_cont^2 = sigma_gal^2 + s^2 sigma_inst^2 - sigma_lib^2) and in the line width
    (sigma_line^2 = sigma_gas^2 + s^2 sigma_inst^2) alike.

        scale=1.0 (default)     the nominal LSF; the code path is the one without a scale
        scale=1.1 (float > 0)   fixed: identical to building the Instrument with the width x 1.1
                                (folded into the static response at setup, no runtime cost)
        scale="lsf_scale"       sampled theta key (give it a prior and a free_param_init); the
                                continuum response weights are recomputed per call from the
                                static band geometry, exactly, for any wavelength dependence of
                                sigma_inst; values are clipped to the supported range
        scale_range=(lo, hi)    the range the compiled kernel supports for a sampled scale,
                                0 < lo < hi; default: the finite bounds of the key's prior.
                                Needed when the key is a transform.

    Prospector's ``resolution_jitter_parameter`` (``prospect/models/sedmodel.py:289-295``,
    a78d153) multiplies ``obs.resolution`` by the parameter too, but differs: (1) it rescales
    the continuum only: the line widths are cached from the unscaled ``obs.resolution`` just
    before (``cache_eline_parameters``, sedmodel.py:283, 575-583), so a jittered LSF broadens
    the absorption features but not the emission lines; here both.  (2) It overwrites
    ``obs.padded_resolution`` in place on every call (state that outlives the call); here the
    Instrument is immutable and the scale is a pure function argument (jit / vmap / grad).
    (3) No validation: a value <= 0, or one that takes the instrument below the library,
    fails an ``assert`` inside the likelihood call (``observation.py:454``); here a
    non-positive scale, an unbounded or non-positive prior, or a prior beyond ``scale_range``
    raises at construction, and instrument-below-library pixels follow the usual rule (the
    continuum stays at library resolution there, with the setup warning evaluated at the
    lower end of the range).
    """
    kind: str
    value: np.ndarray
    wave: Optional[np.ndarray] = None
    scale: Union[float, str] = 1.0
    scale_range: Optional[tuple] = None

    _KINDS = ("R_fwhm", "R_sigma", "fwhm_aa", "sigma_aa", "sigma_kms", "fwhm_kms")

    def __post_init__(self):
        if self.kind not in self._KINDS:
            raise ValueError(f"Instrument.kind must be one of {self._KINDS}")
        self._check_scale()
        v = np.asarray(self.value, dtype=np.float64)
        object.__setattr__(self, "value", v)
        if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
            raise ValueError("Instrument width must be finite and > 0 everywhere")
        if v.ndim > 1:
            raise ValueError("Instrument width must be a scalar or a 1-D array")
        if v.ndim == 1:
            if self.wave is None:
                raise ValueError("an array Instrument width needs wave= (observed "
                                 "frame, Angstrom) of the same length")
            w = np.asarray(self.wave, dtype=np.float64)
            if w.shape != v.shape:
                raise ValueError(f"wave shape {w.shape} != value shape {v.shape}")
            if np.any(np.diff(w) <= 0.0):
                raise ValueError("Instrument wave must be strictly increasing")
            object.__setattr__(self, "wave", w)
        elif self.wave is not None:
            raise ValueError("wave= is only meaningful with an array width")

    def _check_scale(self):
        sc, rng = self.scale, self.scale_range
        if isinstance(sc, bool):
            raise TypeError("Instrument scale must be a float (fixed) or a str (theta key)")
        if isinstance(sc, str):
            if not sc.strip():
                raise ValueError("Instrument scale: empty theta key")
        elif isinstance(sc, (int, float, np.floating, np.integer)):
            sc = float(sc)
            if not (np.isfinite(sc) and sc > 0.0):
                raise ValueError(f"Instrument scale must be finite and > 0, got {sc}")
            object.__setattr__(self, "scale", sc)
        else:
            raise TypeError("Instrument scale must be a float (fixed) or a str (theta key), "
                            f"got {type(sc).__name__}")
        if rng is not None:
            if not isinstance(sc, str):
                raise ValueError("scale_range= is only meaningful with a sampled scale "
                                 "(scale='<theta key>')")
            object.__setattr__(self, "scale_range", check_scale_range(rng, "scale_range"))

    @property
    def free_keys(self) -> tuple:
        """Theta keys this instrument reads (the sampled LSF scale), or ()."""
        return (self.scale,) if isinstance(self.scale, str) else ()

    @classmethod
    def R_fwhm(cls, R, wave=None, scale=1.0, scale_range=None):
        return cls("R_fwhm", R, wave, scale, scale_range)

    @classmethod
    def R_sigma(cls, R, wave=None, scale=1.0, scale_range=None):
        return cls("R_sigma", R, wave, scale, scale_range)

    @classmethod
    def fwhm_aa(cls, fwhm, wave=None, scale=1.0, scale_range=None):
        return cls("fwhm_aa", fwhm, wave, scale, scale_range)

    @classmethod
    def sigma_aa(cls, sigma, wave=None, scale=1.0, scale_range=None):
        return cls("sigma_aa", sigma, wave, scale, scale_range)

    @classmethod
    def sigma_kms(cls, sigma, wave=None, scale=1.0, scale_range=None):
        return cls("sigma_kms", sigma, wave, scale, scale_range)

    @classmethod
    def fwhm_kms(cls, fwhm, wave=None, scale=1.0, scale_range=None):
        return cls("fwhm_kms", fwhm, wave, scale, scale_range)

    def sigma_kms_at(self, wave_obs) -> np.ndarray:
        """sigma [km/s] at each observed wavelength (NumPy, setup only): the width in force
        for a fixed ``scale`` (nominal x scale), the nominal width (scale 1) for a sampled
        one, which the projector multiplies by ``theta[scale]`` per call."""
        v = self._nominal_sigma_kms_at(wave_obs)
        if isinstance(self.scale, str) or self.scale == 1.0:
            return v
        return v * self.scale

    def _nominal_sigma_kms_at(self, wave_obs) -> np.ndarray:
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        if self.value.ndim == 1:
            if wave_obs.min() < self.wave[0] or wave_obs.max() > self.wave[-1]:
                raise ValueError(
                    "Instrument width array does not cover the observed "
                    f"wavelengths [{wave_obs.min():.1f}, {wave_obs.max():.1f}] "
                    f"(covers [{self.wave[0]:.1f}, {self.wave[-1]:.1f}])")
            v = np.interp(wave_obs, self.wave, self.value)
        else:
            v = np.full_like(wave_obs, float(self.value))
        if self.kind == "R_fwhm":
            return CKMS * FWHM_TO_SIGMA / v
        if self.kind == "R_sigma":
            return CKMS / v
        if self.kind == "fwhm_aa":
            return CKMS * FWHM_TO_SIGMA * v / wave_obs
        if self.kind == "sigma_aa":
            return CKMS * v / wave_obs
        if self.kind == "fwhm_kms":
            return FWHM_TO_SIGMA * v
        return v


class _Tied:
    def __repr__(self):
        return "TIED"


TIED = _Tied()
Width = Union[float, str]


@dataclass(frozen=True)
class Kinematics:
    """Galaxy velocity dispersions [km/s]; set once on the model, shared by all observations.

    sigma_gal : float (fixed) or str (theta key of a free parameter).
                No default here; SedModel defaults to DEFAULT_KINEMATICS (300 km/s, stars and gas).
    sigma_gas : float, str, or TIED (= sigma_gal).
    sigma_max : largest value the compiled kernels support; free values are clipped to it.
    """
    sigma_gal: Width
    sigma_gas: Union[Width, _Tied] = TIED
    sigma_max: float = 2000.0

    def __post_init__(self):
        for name in ("sigma_gal", "sigma_gas"):
            v = getattr(self, name)
            if v is TIED:
                if name == "sigma_gal":
                    raise ValueError("sigma_gal cannot be TIED; give a float or a theta key")
                continue
            if isinstance(v, bool):
                raise TypeError(f"{name} must be a float (fixed) or a str (theta key)")
            if isinstance(v, str):
                if not v.strip():
                    raise ValueError(f"{name}: empty theta key")
            elif isinstance(v, (int, float, np.floating, np.integer)):
                v = float(v)
                if not np.isfinite(v) or v < 0.0:
                    raise ValueError(f"{name} must be finite and >= 0, got {v}")
                if v > self.sigma_max:
                    raise ValueError(f"{name}={v} exceeds sigma_max={self.sigma_max}")
                object.__setattr__(self, name, v)
            else:
                raise TypeError(f"{name} must be a float (fixed) or a str (theta key), "
                                f"got {type(v).__name__}")
        if self.sigma_max <= 0.0:
            raise ValueError("sigma_max must be > 0")

    @classmethod
    def none(cls) -> "Kinematics":
        return cls(sigma_gal=0.0, sigma_gas=0.0)

    @property
    def effective_sigma_gas(self) -> Width:
        return self.sigma_gal if self.sigma_gas is TIED else self.sigma_gas

    @property
    def free_keys(self) -> tuple:
        keys = [v for v in (self.sigma_gal, self.effective_sigma_gas) if isinstance(v, str)]
        return tuple(dict.fromkeys(keys))

    @property
    def is_static(self) -> bool:
        return not self.free_keys

    def validate_theta(self, theta, priors: Optional[dict] = None) -> None:
        """Free keys are in ``theta`` (a dict or a set of names); a bounded prior on a
        free key stays within sigma_max; fixed widths are not also in theta."""
        for k in self.free_keys:
            if k not in theta:
                raise KeyError(
                    f"Kinematics expects theta['{k}'] (free width) but it is not "
                    f"in theta; add it (with a prior) or give a float to fix it")
            if priors is not None and k in priors:
                b = getattr(priors[k], "bounds", None)
                if callable(b):
                    b = b()
                if b is not None:
                    hi = float(np.max(np.asarray(b[1], dtype=float)))
                    if np.isfinite(hi) and hi > self.sigma_max:
                        raise ValueError(
                            f"prior on '{k}' allows up to {hi} km/s > "
                            f"sigma_max={self.sigma_max}; raise sigma_max")
        for name, v in (("sigma_gal", self.sigma_gal),
                        ("sigma_gas", self.effective_sigma_gas)):
            if not isinstance(v, str) and name in theta:
                raise ValueError(
                    f"{name} is fixed to {v} km/s in Kinematics but theta also "
                    f"contains '{name}'; remove one of them")

    def resolve(self, theta: dict):
        """(sigma_gal, sigma_gas) as JAX scalars; branching on Python types only."""
        def _get(v):
            if isinstance(v, str):
                return jnp.clip(jnp.ravel(jnp.asarray(theta[v]))[0], 0.0, self.sigma_max)
            return jnp.asarray(float(v))
        return _get(self.sigma_gal), _get(self.effective_sigma_gas)


DEFAULT_KINEMATICS = Kinematics(sigma_gal=300.0)   # SedModel default; printed in every summary


@dataclass(frozen=True, eq=False)
class LogGrid:
    """Log-uniform grid over [wmin, wmax] with dv = finest model pixel in the window."""
    wave: np.ndarray
    lnw: np.ndarray
    dv: float
    idx: np.ndarray
    wave_model_window: np.ndarray
    n_pad: int

    @classmethod
    def build(cls, wave_model, wmin, wmax, sigma_max_kms, extend_to=None) -> "LogGrid":
        """``extend_to=(wmin_ext, wmax_ext)`` appends nodes at the same ``dv`` beyond
        [wmin, wmax] (the nodes inside are unchanged), staying on the model grid."""
        wave_model = np.asarray(wave_model, dtype=np.float64)
        if np.any(np.diff(wave_model) <= 0.0):
            raise ValueError("model wavelength grid must be strictly increasing")
        if wmin >= wmax:
            raise ValueError("wmin must be < wmax")
        if wmin < wave_model[0] or wmax > wave_model[-1]:
            raise ValueError(
                f"window [{wmin:.1f}, {wmax:.1f}] A is not covered by the model grid "
                f"[{wave_model[0]:.1f}, {wave_model[-1]:.1f}] A")
        lo = max(int(np.searchsorted(wave_model, wmin, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(wave_model, wmax, side="left")) + 1, wave_model.size)
        idx = np.arange(lo, hi)
        w_win = wave_model[idx]
        dv = CKMS * float(np.min(np.diff(np.log(w_win))))
        n = int(np.ceil((np.log(wmax) - np.log(wmin)) / (dv / CKMS))) + 1
        lnw = np.linspace(np.log(wmin), np.log(wmax), n)
        dv = CKMS * float(lnw[1] - lnw[0])
        wave = np.exp(lnw)
        wave[0], wave[-1] = wmin, wmax
        if extend_to is not None:
            dln = lnw[1] - lnw[0]
            lo_ext, hi_ext = float(extend_to[0]), float(extend_to[1])
            n_left = max(int(np.ceil((lnw[0] - np.log(lo_ext)) / dln)), 0)
            n_right = max(int(np.ceil((np.log(hi_ext) - lnw[-1]) / dln)), 0)
            while n_left and np.exp(lnw[0] - n_left * dln) < wave_model[0]:
                n_left -= 1
            while n_right and np.exp(lnw[-1] + n_right * dln) > wave_model[-1]:
                n_right -= 1
            lnw = np.concatenate([lnw[0] - dln * np.arange(n_left, 0, -1), lnw,
                                  lnw[-1] + dln * np.arange(1, n_right + 1)])
            wave = np.concatenate([np.exp(lnw[:n_left]), wave, np.exp(lnw[n_left + n:])])
            n = int(lnw.size)
            lo = max(int(np.searchsorted(wave_model, wave[0], side="right")) - 1, 0)
            hi = min(int(np.searchsorted(wave_model, wave[-1], side="left")) + 1, wave_model.size)
            idx = np.arange(lo, hi)
            w_win = wave_model[idx]
        need = n + int(np.ceil(10.0 * sigma_max_kms / dv)) + 2
        n_pad = int(2 ** np.ceil(np.log2(max(2 * n, need))))
        return cls(wave=wave, lnw=lnw, dv=dv, idx=idx,
                   wave_model_window=w_win, n_pad=n_pad)

    @property
    def n(self) -> int:
        return int(self.wave.size)


def make_gaussian_fft(grid: LogGrid) -> Callable:
    """f(spec_log, sigma_kms): Gaussian on the log grid; FFT buffer continues the edge values."""
    n, M, dv = grid.n, grid.n_pad, grid.dv
    n_right = (M - n) // 2
    n_left = M - n - n_right
    nu = jnp.asarray(np.fft.rfftfreq(M, d=1.0))

    def smooth(spec_log, sigma_kms):
        sigma_px = sigma_kms / dv
        pad_r = jnp.full((n_right,), 0.0, spec_log.dtype) + spec_log[-1]
        pad_l = jnp.full((n_left,), 0.0, spec_log.dtype) + spec_log[0]
        padded = jnp.concatenate([spec_log, pad_r, pad_l])
        taper = jnp.exp(-2.0 * jnp.pi ** 2 * sigma_px ** 2 * nu ** 2)
        out = jnp.fft.irfft(jnp.fft.rfft(padded) * taper, n=M)
        return out[:n].astype(spec_log.dtype)

    return smooth


class WindowSmoother:
    """to_log: gather window of spec_rest onto the log grid; smooth; from_log: back to model pixels."""

    def __init__(self, grid: LogGrid):
        self.grid = grid
        self.smooth = make_gaussian_fft(grid)
        self.idx = jnp.asarray(grid.idx)
        self._wave_win = jnp.asarray(grid.wave_model_window)
        self._wave_log = jnp.asarray(grid.wave)
        # model pixels strictly inside the log window (the window carries one
        # bracketing pixel on each side for the interpolation; those stay untouched)
        inside = np.flatnonzero((grid.wave_model_window >= grid.wave[0])
                                & (grid.wave_model_window <= grid.wave[-1]))
        self._inside = jnp.asarray(inside)
        self.idx_inside = jnp.asarray(grid.idx[inside])

    def to_log(self, spec_rest, scale=None):
        """Model window interpolated onto the log grid; ``scale`` (traced) reads it at
        ``wave_log * scale`` instead, i.e. the rest-frame model seen at another redshift."""
        x = self._wave_log if scale is None else self._wave_log * scale
        return jnp.interp(x, self._wave_win, spec_rest[self.idx])

    def from_log(self, spec_log):
        return jnp.interp(self._wave_win, self._wave_log, spec_log)


def build_response(grid: LogGrid, lnw_obs_rest, sigma_fix_kms, band_nsigma=BAND_NSIGMA):
    """Banded row-normalised Gaussian (in ln lambda) from the log grid onto observed pixels.

    Row i: centre lnw_obs_rest[i], sigma sigma_fix_kms[i]; rows narrower than
    half a log pixel are linear interpolation.  Returns gather indices J and
    weights W, both (n_obs, n_band).
    """
    lnw_obs_rest = np.asarray(lnw_obs_rest, dtype=np.float64)
    sigma_fix_kms = np.asarray(sigma_fix_kms, dtype=np.float64)
    n = grid.n
    dln = grid.dv / CKMS
    s_px = sigma_fix_kms / grid.dv
    if np.any(lnw_obs_rest < grid.lnw[0]) or np.any(lnw_obs_rest > grid.lnw[-1]):
        raise ValueError("observed pixels fall outside the LogGrid window")
    x = (lnw_obs_rest - grid.lnw[0]) / dln
    j0 = np.floor(x).astype(np.int64)
    h = int(np.ceil(band_nsigma * max(float(s_px.max()), 0.5))) + 1
    J = j0[:, None] + np.arange(-h, h + 2)[None, :]
    inside = (J >= 0) & (J < n)
    Jc = np.clip(J, 0, n - 1)
    d = Jc - x[:, None]
    gauss_rows = s_px >= 0.5
    sg = np.where(gauss_rows, s_px, 1.0)[:, None]
    W = np.where(gauss_rows[:, None], np.exp(-0.5 * (d / sg) ** 2),
                 np.clip(1.0 - np.abs(d), 0.0, 1.0))
    W = np.where(inside, W, 0.0)
    norm = W.sum(axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise RuntimeError("empty response row: observed pixel not covered")
    return Jc.astype(np.int32), W / norm


def apply_response(J, W, spec_log):
    return jnp.sum(W * spec_log[J], axis=1)


class ScaledResponse:
    """The continuum response of :func:`build_response` for a SAMPLED LSF scale ``s``.

    The gather indices ``J`` and the offsets ``D = J - x`` are static, with the band sized
    for ``scale_max``; the weights are recomputed per call with row width
    ``sigma_fix_i(s) = sqrt(max(s^2 sigma_inst_i^2 - sigma_lib_i^2, 0))`` by the same
    formula as ``build_response`` (Gaussian rows from half a log pixel up, linear
    interpolation below, row-normalised).  Exact for any wavelength dependence of
    sigma_inst.  At ``s = scale_max`` the band equals the one ``build_response`` builds for
    ``sigma_fix(scale_max)``; at smaller ``s`` it is wider, i.e. the Gaussian is truncated
    further out than in the fixed-scale response (the fixed one drops a tail of relative
    mass < erfc(5 / sqrt 2) = 5.7e-7 per row, this one less).
    """

    def __init__(self, grid: LogGrid, lnw_obs_rest, sigma_inst_kms, sigma_lib_kms,
                 scale_max, band_nsigma=BAND_NSIGMA):
        lnw_obs_rest = np.asarray(lnw_obs_rest, dtype=np.float64)
        s_inst = np.asarray(sigma_inst_kms, dtype=np.float64)
        s_lib = np.asarray(sigma_lib_kms, dtype=np.float64)
        n = grid.n
        dln = grid.dv / CKMS
        if np.any(lnw_obs_rest < grid.lnw[0]) or np.any(lnw_obs_rest > grid.lnw[-1]):
            raise ValueError("observed pixels fall outside the LogGrid window")
        s_px_max = np.sqrt(np.clip((float(scale_max) * s_inst) ** 2 - s_lib ** 2,
                                   0.0, None)) / grid.dv
        x = (lnw_obs_rest - grid.lnw[0]) / dln
        j0 = np.floor(x).astype(np.int64)
        h = int(np.ceil(band_nsigma * max(float(s_px_max.max()), 0.5))) + 1
        J = j0[:, None] + np.arange(-h, h + 2)[None, :]
        inside = (J >= 0) & (J < n)
        Jc = np.clip(J, 0, n - 1)
        d = Jc - x[:, None]
        lin = np.where(inside, np.clip(1.0 - np.abs(d), 0.0, 1.0), 0.0)
        if np.any(lin.sum(axis=1) <= 0.0):
            raise RuntimeError("empty response row: observed pixel not covered")
        self.dv = float(grid.dv)
        self.scale_max = float(scale_max)
        self.J_np, self.D_np, self.inside_np, self.lin_np = Jc.astype(np.int32), d, inside, lin
        self.sigma_inst_kms, self.sigma_lib_kms = s_inst, s_lib
        self.J = jnp.asarray(self.J_np)
        self._D = jnp.asarray(d)
        self._inside = jnp.asarray(inside)
        self._lin = jnp.asarray(lin)
        self._s_inst = jnp.asarray(s_inst)
        self._s_lib2 = jnp.asarray(s_lib ** 2)

    @property
    def n_band(self) -> int:
        return int(self.J_np.shape[1])

    def weights_np(self, scale) -> np.ndarray:
        """NumPy weights at a fixed ``scale`` (setup / reference)."""
        s_fix = np.sqrt(np.clip((self.sigma_inst_kms * float(scale)) ** 2
                                - self.sigma_lib_kms ** 2, 0.0, None))
        s_px = s_fix / self.dv
        gauss_rows = s_px >= 0.5
        sg = np.where(gauss_rows, s_px, 1.0)[:, None]
        W = np.where(gauss_rows[:, None], np.exp(-0.5 * (self.D_np / sg) ** 2), self.lin_np)
        W = np.where(self.inside_np, W, 0.0)
        return W / W.sum(axis=1, keepdims=True)

    def weights(self, scale):
        """(n_obs, n_band) weights at a traced scalar ``scale``; the square root is guarded
        so pixels with the instrument below the library (and the linear-interpolation rows)
        give a zero, not a NaN, gradient."""
        s_fix2 = (self._s_inst * scale) ** 2 - self._s_lib2
        pos = s_fix2 > 0.0
        s_fix = jnp.where(pos, jnp.sqrt(jnp.where(pos, s_fix2, 1.0)), 0.0)
        s_px = s_fix / self.dv
        gauss_rows = s_px >= 0.5
        sg = jnp.where(gauss_rows, s_px, 1.0)[:, None]
        W = jnp.where(gauss_rows[:, None], jnp.exp(-0.5 * (self._D / sg) ** 2), self._lin)
        W = jnp.where(self._inside, W, 0.0)
        return W / jnp.sum(W, axis=1, keepdims=True)

    def n_rows_switching(self, lo, hi) -> int:
        """Rows whose width crosses half a log pixel (linear <-> Gaussian) inside [lo, hi]:
        the continuum is discontinuous in s there (a step of ~1 % of the row weight)."""
        def px(s):
            return np.sqrt(np.clip((self.sigma_inst_kms * s) ** 2 - self.sigma_lib_kms ** 2,
                                   0.0, None)) / self.dv
        return int(np.sum((px(lo) < 0.5) & (px(hi) >= 0.5)))


def make_line_painter(wave_obs, line_wave_obs, sigma_inst_lines_kms) -> Callable:
    """paint(line_flux, sigma_gas_kms, inst_scale=None) -> f_nu on wave_obs.

    line_flux: observed-frame integrated flux [spectrum unit x Hz].
    f_nu = F phi(ln lambda) lambda / c_AA, phi unit-area in ln lambda with
    sigma = sqrt(sigma_gas^2 + (inst_scale sigma_inst)^2) / c (inst_scale None = 1).
    """
    wave_obs = np.asarray(wave_obs, dtype=np.float64)
    lnw = jnp.asarray(np.log(wave_obs))
    lnl = jnp.asarray(np.log(np.asarray(line_wave_obs, dtype=np.float64)))
    s_inst = jnp.asarray(np.asarray(sigma_inst_lines_kms, dtype=np.float64))
    lam_over_c = jnp.asarray(wave_obs / C_AA_S)

    def paint(line_flux, sigma_gas_kms, inst_scale=None):
        si = s_inst if inst_scale is None else s_inst * inst_scale
        s = jnp.sqrt(sigma_gas_kms ** 2 + si ** 2) / CKMS
        x = (lnw[:, None] - lnl[None, :]) / s[None, :]
        phi = jnp.exp(-0.5 * x * x) / (jnp.sqrt(2.0 * jnp.pi) * s[None, :])
        return (phi @ line_flux) * lam_over_c

    return paint


def make_line_painter_free_z(wave_obs, line_wave_rest, sigma_inst_table_kms) -> Callable:
    """paint(line_flux, sigma_gas_kms, opz, inst_scale=None) -> f_nu on wave_obs for a
    traced 1 + z: lines at line_wave_rest * opz, instrument width interpolated from the
    per-pixel table ``sigma_inst_table_kms`` (on wave_obs) at those positions, times
    ``inst_scale`` when given."""
    wave_obs = np.asarray(wave_obs, dtype=np.float64)
    lnw = jnp.asarray(np.log(wave_obs))
    wo = jnp.asarray(wave_obs)
    lnl_rest = jnp.asarray(np.log(np.asarray(line_wave_rest, dtype=np.float64)))
    lam_rest = jnp.asarray(np.asarray(line_wave_rest, dtype=np.float64))
    s_tab = jnp.asarray(np.asarray(sigma_inst_table_kms, dtype=np.float64))
    lam_over_c = jnp.asarray(wave_obs / C_AA_S)

    def paint(line_flux, sigma_gas_kms, opz, inst_scale=None):
        lnl = lnl_rest + jnp.log(opz)
        s_inst = jnp.interp(lam_rest * opz, wo, s_tab)
        if inst_scale is not None:
            s_inst = s_inst * inst_scale
        s = jnp.sqrt(sigma_gas_kms ** 2 + s_inst ** 2) / CKMS
        x = (lnw[:, None] - lnl[None, :]) / s[None, :]
        phi = jnp.exp(-0.5 * x * x) / (jnp.sqrt(2.0 * jnp.pi) * s[None, :])
        return (phi @ line_flux) * lam_over_c

    return paint


def _line_windows(wave_obs, line_wave_rest, sigma_inst_table_kms, sigma_gas_max_kms,
                 opz_lo, opz_hi, inst_scale_max=1.0, nsigma=_LINE_WINDOW_NSIGMA):
    """Static pixel windows holding every nonzero value of each line profile (NumPy, setup).

    For every line centre ``line_wave_rest * opz`` with ``opz`` in [opz_lo, opz_hi], every
    gas dispersion up to ``sigma_gas_max_kms`` and every instrument scale up to
    ``inst_scale_max``, the profile of :meth:`SpectralProjector.line_basis` is exactly 0 in
    float64 on the pixels more than ``nsigma`` of its largest sigma from the centre.  Returns
    ``(start, width)``: line ``j`` is nonzero only on pixels ``start[j] .. start[j] + width - 1``
    (one width for all lines, the largest, so the window array has a static shape), or None
    when that width covers the whole spectrum (nothing to gain).
    """
    wave_obs = np.asarray(wave_obs, dtype=np.float64)
    lam = np.asarray(line_wave_rest, dtype=np.float64)
    tab = np.asarray(sigma_inst_table_kms, dtype=np.float64)
    lo_c, hi_c = lam * opz_lo, lam * opz_hi
    # the largest instrument width the traced jnp.interp can return for a centre in
    # [lo_c, hi_c]: the larger end value, or a table node in between (piecewise linear)
    s_inst = np.maximum(np.interp(lo_c, wave_obs, tab), np.interp(hi_c, wave_obs, tab))
    for j in range(lam.size):
        inner = tab[(wave_obs > lo_c[j]) & (wave_obs < hi_c[j])]
        if inner.size:
            s_inst[j] = max(s_inst[j], float(inner.max()))
    s_max = np.sqrt(float(sigma_gas_max_kms) ** 2 + (float(inst_scale_max) * s_inst) ** 2) / CKMS
    lnw = np.log(wave_obs)
    lo = np.searchsorted(lnw, np.log(lo_c) - nsigma * s_max, side="left")
    hi = np.searchsorted(lnw, np.log(hi_c) + nsigma * s_max, side="right")
    width = int(np.max(hi - lo)) if lam.size else 0
    if width >= wave_obs.size:
        return None
    return np.clip(lo, 0, wave_obs.size - width).astype(np.int64), width


# --- AGN narrow lines (ceridwen.agn.lines) -----------------------------------------------
# Prospector's constants (prospect.sources.constants), used only by its painter below, so that
# the "prospector" mode reproduces AGNSpecModel to rounding.
PROSPECT_CKMS = 2.998e5
PROSPECT_LIGHTSPEED = 2.998e18
_PROSPECT_NU_NUMER = 3e18          # SpecModel.get_eline_gaussians: trapezoid(g, 3e18 / warr)


def agn_line_profiles_on_grid(wave, line_wave, sigma_kms, res_floor_factor=2.0):
    """(n_wave, n_lines) L_nu profiles of unit-luminosity lines on the rest model grid ``wave``
    for a traced width ``sigma_kms``: ``NebularModel.line_profiles(sigma_kms)`` with its default
    settings (a Gaussian in lambda of width sqrt(floor^2 + (pos sigma / c)^2), floor =
    ``res_floor_factor`` x the local pixel width, times pos^2 / c), the nebular convention, so the
    AGN lines reach the photometry exactly as the nebular lines do."""
    from .neb.NebularGridModel import CLIGHT_AA_S, SQRT_2PI
    lam = np.asarray(wave, dtype=np.float64)
    pos = np.asarray(line_wave, dtype=np.float64)
    idx = np.clip(np.searchsorted(lam, pos, side="right") - 1, 1, lam.size - 2)
    floor = jnp.asarray((lam[idx + 1] - lam[idx]) * float(res_floor_factor))
    posj = jnp.asarray(pos)
    dl = jnp.sqrt(floor ** 2 + (posj * sigma_kms / CLIGHT_AA_S * 1.0e13) ** 2)
    prof = jnp.exp(-0.5 * ((jnp.asarray(lam)[:, None] - posj[None, :]) / dl[None, :]) ** 2)
    return prof / (SQRT_2PI * dl[None, :]) * (posj[None, :] ** 2 / CLIGHT_AA_S)


def make_agn_line_painter_prospector(wave_obs, agn_line_wave_rest, sf_line_wave_rest,
                                     agn_sf_rows, pixel_mask=None) -> Callable:
    """paint(line_flux, sigma_agn_kms, sigma_sf_kms, eline_sigma_kms, opz) -> f_nu on wave_obs,
    Prospector's ``AGNSpecModel.predict_spec`` AGN-line term with static shapes.

    Prospector (``cache_eline_parameters(nsigma=...)`` then ``predict_aline_spec``): with
    ``nsigma = 5 agn_eline_sigma / eline_sigma`` a pixel is used when it lies within
    ``nsigma * lambda_obs / c * sigma_sf`` of a nebular line (``sigma_sf_kms`` (n_sf,): the
    nebular line widths, ``sf_line_wave_rest`` (n_sf,) their rest wavelengths) and is unmasked
    (``pixel_mask``, the Spectrum's mask; None = all); a nebular line with no such pixel is not
    valid. AGN line ``l`` (rest ``agn_line_wave_rest[l]``, the nebular line ``agn_sf_rows[l]``)
    is painted only when that nebular line is valid, as a Gaussian in velocity of dispersion
    ``sigma_agn_kms`` alone (no instrument), times dv/dnu, divided by minus its trapezoid in
    ``nu = 3e18 / lambda`` over the used pixels in order (gaps included, as ``np.trapezoid`` over
    the compressed array does), and only on the used pixels. ``line_flux`` (n_agn,) observed
    integrated fluxes; centres at ``rest * opz`` (``opz`` = 1 + zred + eline_delta_zred).
    Prospector's ``ckms`` / ``lightspeed`` (2.998e5, 2.998e18) are used here."""
    from jax import lax
    wo = np.asarray(wave_obs, dtype=np.float64)
    n = wo.size
    w = jnp.asarray(wo)
    nu = jnp.asarray(_PROSPECT_NU_NUMER / wo)
    lam_a = jnp.asarray(np.asarray(agn_line_wave_rest, dtype=np.float64))
    lam_sf = jnp.asarray(np.asarray(sf_line_wave_rest, dtype=np.float64))
    rows = np.asarray(agn_sf_rows, dtype=np.int64)
    if rows.shape != (int(lam_a.shape[0]),) or rows.min() < 0 or rows.max() >= lam_sf.shape[0]:
        raise ValueError("agn_sf_rows must give, for every AGN line, its row in sf_line_wave_rest")
    rows = jnp.asarray(rows)
    om = None if pixel_mask is None else jnp.asarray(np.asarray(pixel_mask, dtype=bool))
    if om is not None and om.shape != (n,):
        raise ValueError(f"pixel_mask must have one entry per pixel ({n}), got {om.shape}")
    idx = jnp.arange(n)

    def paint(line_flux, sigma_agn_kms, sigma_sf_kms, eline_sigma_kms, opz):
        ew_sf = lam_sf * opz
        nsigma = 5 * (sigma_agn_kms / eline_sigma_kms)
        linewidth = nsigma * ew_sf / PROSPECT_CKMS * sigma_sf_kms
        pm = jnp.abs(w[None, :] - ew_sf[:, None]) < linewidth[:, None]
        if om is not None:
            pm = pm & om[None, :]
        valid = pm.any(axis=1)
        emask = pm.any(axis=0)
        on = valid[rows]
        mu = (lam_a * opz)[None, :]
        sigma = sigma_agn_kms
        dv = PROSPECT_CKMS * (w[:, None] / mu - 1)
        dv_dnu = PROSPECT_CKMS * w[:, None] ** 2 / (PROSPECT_LIGHTSPEED * mu)
        g = 1. / (sigma * jnp.sqrt(jnp.pi * 2)) * jnp.exp(-dv ** 2 / (2 * sigma ** 2))
        g = g * dv_dnu
        # trapezoid over the used pixels in order: each used pixel pairs with the next used one
        nxt_ge = lax.cummin(jnp.where(emask, idx, n), reverse=True)
        nxt = jnp.concatenate([nxt_ge[1:], jnp.full((1,), n, dtype=nxt_ge.dtype)])
        has = emask & (nxt < n)
        j = jnp.minimum(nxt, n - 1)
        d = jnp.where(has, nu[j] - nu, 0.0)
        norm = -jnp.sum(d[:, None] * (g[j, :] + g) / 2.0, axis=0)
        use = on & (norm != 0.0)
        flux = jnp.where(use, line_flux, 0.0) / jnp.where(use, norm, 1.0)
        return jnp.where(emask, g @ flux, 0.0)

    return paint


@dataclass(eq=False)
class SpectralProjector:
    """Built once per Spectrum by setup_for_model; predict() is the per-likelihood call.

    With ``zred_range`` the projector serves a sampled redshift: the log grid is the one of
    the reference redshift extended at the same dv to cover every z in the range (so at
    ``opz_ref`` it reproduces the fixed-z projector), the response is built at ``opz_ref`` and the model
    is read at ``wave_log * opz_ref / opz`` per call (the sigma_gal kernel commutes with
    the shift); the lines are painted at ``line_wave_rest * opz``.  The library width in
    the fixed kernel is the one at ``opz_ref``.

    With a sampled instrument scale (``Instrument(scale="<key>")``, ``inst_scale_key``) the
    continuum weights come from ``scaled`` (a :class:`ScaledResponse`, band sized for the top
    of ``inst_scale_range``) per call, the lines are painted with ``s * sigma_inst``, and
    ``sigma_inst_kms`` / ``line_sigma_table_kms`` hold the NOMINAL (s = 1) widths; ``W`` and
    ``sigma_fix_kms`` are then the values at the reference scale (1 clipped into the range).
    A fixed scale is folded into ``sigma_inst_kms`` by ``Instrument.sigma_kms_at``.
    """
    kinematics: Kinematics
    instrument: Optional[Instrument]
    subtract_library: bool
    window: WindowSmoother
    J: jnp.ndarray
    W: jnp.ndarray
    paint: Optional[Callable]
    line_idx: np.ndarray
    sigma_inst_kms: np.ndarray
    sigma_lib_kms: np.ndarray
    sigma_fix_kms: np.ndarray
    zred_range: Optional[tuple] = None
    opz_ref: float = 1.0
    wave_obs: Optional[np.ndarray] = None
    line_wave_rest: Optional[np.ndarray] = None
    line_sigma_table_kms: Optional[np.ndarray] = None
    inst_scale_key: Optional[str] = None
    inst_scale_range: Optional[tuple] = None
    scaled: Optional[ScaledResponse] = None
    line_window: Optional[tuple] = None
    _line_idx_j: jnp.ndarray = field(init=False)

    def __post_init__(self):
        self._line_idx_j = jnp.asarray(self.line_idx.astype(np.int32))

    @property
    def grid(self) -> LogGrid:
        return self.window.grid

    @property
    def free_z(self) -> bool:
        return self.zred_range is not None

    @property
    def free_inst_scale(self) -> bool:
        return self.inst_scale_key is not None

    def inst_scale(self, theta):
        """The sampled LSF scale from ``theta`` clipped to ``inst_scale_range`` (a JAX
        scalar), or None when the scale is fixed (the static response is used)."""
        if self.inst_scale_key is None:
            return None
        k = self.inst_scale_key
        if k not in theta:
            raise KeyError(f"the Spectrum's Instrument samples its LSF scale as theta['{k}'], "
                           "which is not in theta")
        lo, hi = self.inst_scale_range
        return jnp.clip(jnp.ravel(jnp.asarray(theta[k]))[0], lo, hi)

    def enable_line_windows(self) -> Optional[tuple]:
        """Opt in to the windowed line basis in :meth:`predict_with_line_basis` (off by default:
        on CPU it measured slower than the dense one; it targets GPUs).  Each line profile is
        then evaluated on a static window of +-40 of its largest sigma only; the values are
        bitwise those of the dense :meth:`line_basis` (gradients agree to 1-2 ulp: the reverse
        pass sums over the window): the bounds are ``Kinematics.sigma_max`` for a sampled
        sigma_gas (``Kinematics.resolve`` clips to it), the top of ``inst_scale_range`` and the
        line centres over ``zred_range``.  A sampled redshift outside ``zred_range`` (beyond the
        projector's support) truncates the lines; ``eline_delta_zred`` in theta always takes the
        dense path.  Call before the first trace: the setting is read when jit traces; a rebuilt
        projector (``SedModel.setup_observations``, ``fitSED(observations=...)``) starts with it
        off again.  Sets and returns ``line_window`` ((start, width), or None when a window would
        cover the whole spectrum)."""
        if self.line_wave_rest is None or self.line_idx.size == 0:
            self.line_window = None
            return None
        b = self.kinematics
        gas = b.effective_sigma_gas
        opz_lo, opz_hi = ((self.opz_ref, self.opz_ref) if self.zred_range is None
                          else (1.0 + self.zred_range[0], 1.0 + self.zred_range[1]))
        self.line_window = _line_windows(
            self.wave_obs, self.line_wave_rest, self.line_sigma_table_kms,
            b.sigma_max if isinstance(gas, str) else float(gas), opz_lo, opz_hi,
            1.0 if self.inst_scale_range is None else self.inst_scale_range[1])
        return self.line_window

    @classmethod
    def build(cls, kinematics: Kinematics, instrument: Optional[Instrument],
              wave_model, wave_obs, zred, lib_sigma_kms=None,
              line_wave_rest=None, subtract_library=True, zred_range=None,
              inst_scale_range=None):
        """
        wave_model      rest-frame model grid [A]
        wave_obs        observed pixel centres [A]
        zred            fixed redshift, or the reference redshift inside ``zred_range``
        lib_sigma_kms   SSPData.ssp_resolution on wave_model, NaN = unknown (or None)
        line_wave_rest  rest wavelengths of all model lines, in predict_line_fluxes order (or None)
        zred_range      (z_min, z_max) of a SAMPLED redshift (None: fixed at ``zred``)
        inst_scale_range (lo, hi) of a SAMPLED instrument scale when the Instrument carries no
                        ``scale_range`` (SedModel passes the prior bounds); the kernel band
                        and window margins are sized for ``hi``
        """
        b = kinematics
        if instrument is not None and not isinstance(instrument, Instrument):
            raise TypeError("instrument must be an Instrument (Instrument.R_fwhm(...), "
                            "Instrument.fwhm_aa(...), ...) or None")
        wave_model = np.asarray(wave_model, dtype=np.float64)
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        if np.any(np.diff(wave_obs) <= 0.0):
            raise ValueError("observed wavelengths must be strictly increasing")
        opz = 1.0 + float(zred)
        if zred_range is not None:
            z_lo, z_hi = float(zred_range[0]), float(zred_range[1])
            if not (np.isfinite(z_lo) and np.isfinite(z_hi) and -1.0 < z_lo < z_hi):
                raise ValueError(f"zred_range must be finite with -1 < z_min < z_max, got {zred_range}")
            if not (z_lo <= float(zred) <= z_hi):
                raise ValueError(f"the reference redshift {float(zred):g} lies outside "
                                 f"zred_range = ({z_lo:g}, {z_hi:g})")
            zred_range = (z_lo, z_hi)
            opz_lo, opz_hi = 1.0 + z_lo, 1.0 + z_hi
        else:
            opz_lo = opz_hi = opz

        scale_key = scale_rng = None
        if instrument is not None and isinstance(instrument.scale, str):
            scale_key = instrument.scale
            scale_rng = instrument.scale_range
            if scale_rng is None:
                if inst_scale_range is None:
                    raise ValueError(
                        f"the Instrument samples its LSF scale as theta['{scale_key}'] but no "
                        "range is known for it: give its prior finite bounds (Uniform / "
                        "ClippedNormal, lower bound > 0) or pass Instrument(..., "
                        "scale_range=(lo, hi))")
                scale_rng = check_scale_range(inst_scale_range, f"prior on '{scale_key}'")
        if instrument is None:
            s_inst = np.zeros_like(wave_obs)
            subtract_library = False
        else:
            s_inst = instrument.sigma_kms_at(wave_obs)
        # the narrowest (diagnostics) and widest (kernel sizing) instrument in force
        s_inst_lo = s_inst if scale_key is None else scale_rng[0] * s_inst
        s_inst_hi = s_inst if scale_key is None else scale_rng[1] * s_inst
        if subtract_library and lib_sigma_kms is not None:
            lib = np.asarray(lib_sigma_kms, dtype=np.float64)
            if lib.shape != wave_model.shape:
                raise ValueError("lib_sigma_kms must be on the model grid")
            lib = np.nan_to_num(lib, nan=0.0)
            s_lib = np.interp(wave_obs / opz, wave_model, lib)
        else:
            lib = None
            s_lib = np.zeros_like(wave_obs)
        s_fix2 = s_inst_lo ** 2 - s_lib ** 2
        n_bad = int(np.sum(s_fix2 < 0.0))
        if n_bad:
            warnings.warn(
                ("" if scale_key is None else
                 f"at the lower end of the '{scale_key}' range ({scale_rng[0]:g}): ")
                + f"instrument narrower than the SSP library at {n_bad} of "
                f"{wave_obs.size} pixels ({100 * n_bad / wave_obs.size:.1f} %, "
                f"{wave_obs[s_fix2 < 0].min():.0f}-{wave_obs[s_fix2 < 0].max():.0f} A): "
                "the continuum is delivered at library resolution there, which is "
                "correct for a coarse grid; if the instrument is really finer, use a "
                "finer grid; if the number is wrong, check the Instrument unit")
        s_fix = np.sqrt(np.clip(s_fix2, 0.0, None))
        if zred_range is not None and lib is not None:
            worst = 0.0
            for o in (opz_lo, opz_hi):
                s_end = np.sqrt(np.clip(s_inst_lo ** 2 - np.interp(wave_obs / o, wave_model, lib) ** 2,
                                        0.0, None))
                worst = max(worst, float(np.max(np.abs(s_end - s_fix) / np.maximum(s_fix, 1e-3))))
            if worst > 0.1:
                warnings.warn(
                    f"the library width in the fixed kernel is taken at z = {opz - 1:g}; over "
                    f"zred_range it changes the kernel by up to {100 * worst:.0f} % at some "
                    "pixel. Narrow the redshift prior or accept the approximation")

        s_fix_hi = (s_fix if scale_key is None
                    else np.sqrt(np.clip(s_inst_hi ** 2 - s_lib ** 2, 0.0, None)))
        marg = BAND_NSIGMA * (b.sigma_max + float(s_fix_hi.max())) / CKMS
        wmin_ref = wave_obs[0] / opz * np.exp(-marg)
        wmax_ref = wave_obs[-1] / opz * np.exp(+marg)
        wmin = wave_obs[0] / opz_hi * np.exp(-marg)
        wmax = wave_obs[-1] / opz_lo * np.exp(+marg)
        if wmin < wave_model[0] or wmax > wave_model[-1]:
            raise ValueError(
                f"model grid [{wave_model[0]:.0f}, {wave_model[-1]:.0f}] A does not "
                f"cover the observed window plus kernel margin "
                f"[{wmin:.0f}, {wmax:.0f}] A (rest"
                + (f", zred in [{z_lo:g}, {z_hi:g}]" if zred_range else "")
                + "); trim the observation, narrow the redshift range or reduce sigma_max")
        window = WindowSmoother(LogGrid.build(
            wave_model, wmin_ref, wmax_ref, b.sigma_max,
            extend_to=None if zred_range is None else (wmin, wmax)))
        scaled = None
        if scale_key is None:
            J, W = build_response(window.grid, np.log(wave_obs) - np.log(opz), s_fix)
        else:
            scaled = ScaledResponse(window.grid, np.log(wave_obs) - np.log(opz), s_inst,
                                    s_lib, scale_rng[1])
            s_ref = float(np.clip(1.0, *scale_rng))
            J, W = scaled.J_np, scaled.weights_np(s_ref)
            s_fix = np.sqrt(np.clip((s_ref * s_inst) ** 2 - s_lib ** 2, 0.0, None))
            n_sw = scaled.n_rows_switching(*scale_rng)
            if n_sw:
                warnings.warn(
                    f"over the '{scale_key}' range {scale_rng} the continuum kernel of {n_sw} "
                    f"of {wave_obs.size} pixels crosses half a log-grid pixel "
                    f"({0.5 * window.grid.dv:.1f} km/s), where the response switches between "
                    "linear interpolation and a Gaussian: the prediction steps slightly in the "
                    "scale there (instrument close to the library resolution). Narrow the "
                    "range, or check the Instrument / subtract_library")

        paint, line_idx = None, np.zeros(0, dtype=np.int64)
        lw_kept = s_table = None
        if line_wave_rest is not None:
            lwr = np.asarray(line_wave_rest, dtype=np.float64)
            marg_l = BAND_NSIGMA * (b.sigma_max + float(s_inst_hi.max())) / CKMS
            keep = ((lwr * opz_hi > wave_obs[0] * np.exp(-marg_l))
                    & (lwr * opz_lo < wave_obs[-1] * np.exp(marg_l)))
            line_idx = np.flatnonzero(keep)
            if line_idx.size:
                if instrument is None:
                    # no LSF: a line cannot be narrower than the pixel it lands on
                    s_table = 0.5 * CKMS * np.gradient(np.log(wave_obs))
                else:
                    s_table = s_inst
                lw_kept = lwr[keep]
                if zred_range is None:
                    lw = lwr[keep] * opz
                    paint = make_line_painter(wave_obs, lw, np.interp(lw, wave_obs, s_table))
                else:
                    paint = make_line_painter_free_z(wave_obs, lwr[keep], s_table)

        return cls(kinematics=b, instrument=instrument,
                   subtract_library=bool(subtract_library), window=window,
                   J=jnp.asarray(J), W=jnp.asarray(W), paint=paint,
                   line_idx=line_idx, sigma_inst_kms=s_inst,
                   sigma_lib_kms=s_lib, sigma_fix_kms=s_fix,
                   zred_range=zred_range, opz_ref=opz, wave_obs=wave_obs,
                   line_wave_rest=lw_kept, line_sigma_table_kms=s_table,
                   inst_scale_key=scale_key, inst_scale_range=scale_rng, scaled=scaled)

    def continuum(self, spec_rest, sigma_gal_kms, opz=None, inst_scale=None, transmission=None):
        """Continuum on the observed pixels; ``opz`` = 1 + z (traced) when ``free_z``,
        ``inst_scale`` (traced) when the instrument scale is sampled.  ``transmission``: a
        callable ``T(rest wavelengths)`` (the IGM) applied after the sigma_gal kernel and before
        the instrument, on the log grid; ``spec_rest`` is then the spectrum before the IGM."""
        scale = None if opz is None else self.opz_ref / opz
        spec_log = self.window.smooth(self.window.to_log(spec_rest, scale), sigma_gal_kms)
        if transmission is not None:
            w_log = self.window._wave_log if scale is None else self.window._wave_log * scale
            spec_log = spec_log * transmission(w_log).astype(spec_log.dtype)
        if inst_scale is None:
            return apply_response(self.J, self.W, spec_log)
        return apply_response(self.scaled.J, self.scaled.weights(inst_scale), spec_log)

    def lines(self, line_flux_obs_all, sigma_gas_kms, opz=None, inst_scale=None):
        if self.paint is None:
            return 0.0
        flux = line_flux_obs_all[self._line_idx_j]
        if self.free_z:
            if inst_scale is None:
                return self.paint(flux, sigma_gas_kms, opz)
            return self.paint(flux, sigma_gas_kms, opz, inst_scale)
        if inst_scale is None:
            return self.paint(flux, sigma_gas_kms)
        return self.paint(flux, sigma_gas_kms, inst_scale)

    def predict(self, spec_rest, line_flux_obs_all, theta, transmission=None):
        s_gal, s_gas = self.kinematics.resolve(theta)
        opz = None
        if self.free_z:
            if "zred" not in theta:
                raise KeyError("projector built with zred_range needs theta['zred']")
            opz = 1.0 + jnp.ravel(jnp.asarray(theta["zred"]))[0]
        s_ins = self.inst_scale(theta)
        out = self.continuum(spec_rest, s_gal, opz, s_ins, transmission)
        if self.paint is not None and line_flux_obs_all is not None:
            out = out + self.lines(line_flux_obs_all, s_gas, opz, s_ins)
        return out

    def line_basis(self, sigma_gas_kms, opz_line, inst_scale=None):
        """(n_pix, n_kept) f_nu of UNIT-flux lines on the observed pixels, centred at
        ``line_wave_rest * opz_line`` (traced), width sqrt(sigma_gas^2 + sigma_inst^2) with the
        instrument width interpolated at the line (times ``inst_scale`` when given); the
        painter's profile, one column per line."""
        wo = np.asarray(self.wave_obs, dtype=np.float64)
        lam = jnp.asarray(np.asarray(self.line_wave_rest, dtype=np.float64))
        lnl = jnp.log(lam) + jnp.log(opz_line)
        s_inst = jnp.interp(lam * opz_line, jnp.asarray(wo),
                            jnp.asarray(np.asarray(self.line_sigma_table_kms, dtype=np.float64)))
        if inst_scale is not None:
            s_inst = s_inst * inst_scale
        s = jnp.sqrt(sigma_gas_kms ** 2 + s_inst ** 2) / CKMS
        x = (jnp.asarray(np.log(wo))[:, None] - lnl[None, :]) / s[None, :]
        phi = jnp.exp(-0.5 * x * x) / (jnp.sqrt(2.0 * jnp.pi) * s[None, :])
        return phi * jnp.asarray(wo / C_AA_S)[:, None]

    def _line_basis_windowed(self, sigma_gas_kms, opz_line, inst_scale=None):
        """:meth:`line_basis` (bitwise values) with the exponentials evaluated on the static windows
        of ``line_window`` only; valid for ``sigma_gas_kms`` and ``inst_scale`` within the
        bounds the windows were built for and ``opz_line`` in the projector's redshift range."""
        start, width = self.line_window
        wo = np.asarray(self.wave_obs, dtype=np.float64)
        rows = start[None, :] + np.arange(width)[:, None]            # (width, n_kept), static
        lam = jnp.asarray(np.asarray(self.line_wave_rest, dtype=np.float64))
        lnl = jnp.log(lam) + jnp.log(opz_line)
        s_inst = jnp.interp(lam * opz_line, jnp.asarray(wo),
                            jnp.asarray(np.asarray(self.line_sigma_table_kms, dtype=np.float64)))
        if inst_scale is not None:
            s_inst = s_inst * inst_scale
        s = jnp.sqrt(sigma_gas_kms ** 2 + s_inst ** 2) / CKMS
        x = (jnp.asarray(np.log(wo)[rows]) - lnl[None, :]) / s[None, :]
        phi = jnp.exp(-0.5 * x * x) / (jnp.sqrt(2.0 * jnp.pi) * s[None, :])
        win = phi * jnp.asarray((wo / C_AA_S)[rows])
        # each pixel takes its line's window value, or 0 outside the window (one gather + select)
        offset = np.arange(wo.size)[:, None] - start[None, :]
        inside = (offset >= 0) & (offset < width)
        basis = jnp.where(inside, win[np.clip(offset, 0, width - 1),
                                      np.arange(start.size)[None, :]], 0.0)
        return basis

    def predict_with_line_basis(self, spec_rest, line_flux_obs_all, theta, fit_pos=None,
                                basis=None, transmission=None):
        """``(prediction, A)``: continuum plus the lines of ``line_flux_obs_all`` painted with
        :meth:`line_basis` at ``(1 + zred + eline_delta_zred)`` (theta key, default 0), and the
        unit-flux columns ``A`` (n_pix, len(fit_pos)) of the kept lines at positions ``fit_pos``
        (None when ``fit_pos`` is None)."""
        s_gal, s_gas = self.kinematics.resolve(theta)
        opz = None
        if self.free_z:
            if "zred" not in theta:
                raise KeyError("projector built with zred_range needs theta['zred']")
            opz = 1.0 + jnp.ravel(jnp.asarray(theta["zred"]))[0]
        s_ins = self.inst_scale(theta)
        out = self.continuum(spec_rest, s_gal, opz, s_ins, transmission)
        if self.line_wave_rest is None or self.line_idx.size == 0:
            return out, None
        if basis is None:   # a precomputed basis is only passed for fixed z, widths, dz = 0
            opz_line = self.opz_ref if opz is None else opz
            if "eline_delta_zred" in theta:
                # unbounded shift: the static windows do not hold, the dense basis does
                opz_line = opz_line + jnp.ravel(jnp.asarray(theta["eline_delta_zred"]))[0]
                basis = self.line_basis(s_gas, opz_line, s_ins)
            elif self.line_window is not None:
                basis = self._line_basis_windowed(s_gas, opz_line, s_ins)
            else:
                basis = self.line_basis(s_gas, opz_line, s_ins)
        else:
            basis = jnp.asarray(basis)
        if line_flux_obs_all is not None:
            out = out + basis @ line_flux_obs_all[self._line_idx_j]
        A = None if fit_pos is None else basis[:, np.asarray(fit_pos, dtype=np.int64)]
        return out, A

    # AGN narrow lines (ceridwen.agn.lines): a separate static painter, built by
    # setup_agn_lines; None until then (the nebular path never reads it). Further AGN line sets
    # (AGN2: the broad lines) get their own painter in _agn_sets, keyed by the set's name.
    _agn = None
    _agn_sets = None

    def has_agn_lines(self, key="nlr") -> bool:
        """True when the painter of AGN line set ``key`` has been built."""
        if key == "nlr":
            return self._agn is not None
        return self._agn_sets is not None and key in self._agn_sets

    def setup_agn_lines(self, agn_line_wave_rest, *, mode, sf_line_wave_rest=None,
                        agn_sf_rows=None, pixel_mask=None, key="nlr"):
        """Build the AGN-line painter for this projector (setup time, NumPy).

        ``key``: the line set ("nlr", the default, is stored where it always was; any other
        set, e.g. "blr", only with ``mode="corrected"``, the nebular painter's profile with the
        set's own width).

        ``mode`` "corrected": the nebular painter's profile (Gaussian in ln lambda, unit area,
        width sqrt(agn_eline_sigma^2 + sigma_inst^2) with the instrument at the line, times the
        sampled instrument scale). "prospector" / "fsps": Prospector's ``AGNSpecModel`` painter
        (:func:`make_agn_line_painter_prospector`), which needs every nebular line
        (``sf_line_wave_rest``, rest A), the nebular row of each AGN line (``agn_sf_rows``) and
        the Spectrum's ``pixel_mask``."""
        lam = np.asarray(agn_line_wave_rest, dtype=np.float64)
        if self.instrument is None:
            s_table = 0.5 * CKMS * np.gradient(np.log(self.wave_obs))
        else:
            s_table = np.asarray(self.sigma_inst_kms, dtype=np.float64)
        if mode == "corrected":
            paint = make_line_painter_free_z(self.wave_obs, lam, s_table)
            if key == "nlr":
                self._agn = {"mode": mode, "paint": paint}
            else:
                if self._agn_sets is None:
                    self._agn_sets = {}
                self._agn_sets[key] = {"mode": mode, "paint": paint}
            return
        if key != "nlr":
            raise ValueError(f"setup_agn_lines(key={key!r}, mode={mode!r}): only the narrow "
                             "lines have Prospector's painter; other line sets use 'corrected'")
        if mode not in ("prospector", "fsps"):
            raise ValueError(f"setup_agn_lines(mode={mode!r}): 'corrected', 'prospector' or 'fsps'")
        if sf_line_wave_rest is None or agn_sf_rows is None:
            raise ValueError(
                "the Prospector AGN-line painter selects its pixels from the nebular lines "
                "(AGNSpecModel asserts 'must add agn and nebular lines'): pass "
                "sf_line_wave_rest and agn_sf_rows (a CSP with add_neb=True)")
        sf = np.asarray(sf_line_wave_rest, dtype=np.float64)
        # library width of each nebular line at its rest wavelength (Prospector interpolates the
        # library resolution at the rest line wavelength); sigma_lib_kms is on wave_obs at opz_ref
        lib = np.interp(sf * self.opz_ref, self.wave_obs,
                        np.asarray(self.sigma_lib_kms, dtype=np.float64))
        self._agn = {
            "mode": mode,
            "paint": make_agn_line_painter_prospector(self.wave_obs, lam, sf, agn_sf_rows,
                                                      pixel_mask),
            "sf_rest": jnp.asarray(sf), "lib": jnp.asarray(lib),
            "inst": None if self.instrument is None else jnp.asarray(s_table),
        }

    def agn_lines(self, line_flux_obs, sigma_agn_kms, theta, key="nlr"):
        """f_nu of the AGN lines on the observed pixels (add to :meth:`predict`):
        ``line_flux_obs`` (n_agn,) observed integrated fluxes, ``sigma_agn_kms`` the set's width
        (``agn_eline_sigma``, ``blr_sigma``); centres at rest x (1 + zred +
        ``eline_delta_zred``). ``key``: the line set (see :meth:`setup_agn_lines`)."""
        if not self.has_agn_lines(key):
            raise RuntimeError(f"SpectralProjector.agn_lines(key={key!r}) before setup_agn_lines")
        s_gal, s_gas = self.kinematics.resolve(theta)
        opz = self.opz_ref
        if self.free_z:
            if "zred" not in theta:
                raise KeyError("projector built with zred_range needs theta['zred']")
            opz = 1.0 + jnp.ravel(jnp.asarray(theta["zred"]))[0]
        if "eline_delta_zred" in theta:
            opz = opz + jnp.ravel(jnp.asarray(theta["eline_delta_zred"]))[0]
        s_ins = self.inst_scale(theta)
        a = self._agn if key == "nlr" else self._agn_sets[key]
        if a["mode"] == "corrected":
            return a["paint"](line_flux_obs, sigma_agn_kms, opz, s_ins)
        # Prospector's nebular widths (cache_eline_parameters): hypot(eline_sigma, lib), then
        # hypot with sqrt(inst^2 - lib^2) (0 where negative) when the spectrum has a resolution
        lib = a["lib"]
        e = jnp.hypot(s_gas, lib)
        if a["inst"] is not None:
            inst = jnp.interp(a["sf_rest"] * opz, jnp.asarray(self.wave_obs), a["inst"])
            if s_ins is not None:
                inst = inst * s_ins
            d2 = inst ** 2 - lib ** 2
            e = jnp.hypot(e, jnp.where(d2 > 0.0, jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0)), 0.0))
        return a["paint"](line_flux_obs, sigma_agn_kms, e, s_gas, opz)

    def summary(self) -> str:
        b = self.kinematics
        si, sl, sf = self.sigma_inst_kms, self.sigma_lib_kms, self.sigma_fix_kms
        return "\n".join([
            f"Kinematics: sigma_gal={b.sigma_gal!r}  sigma_gas={b.effective_sigma_gas!r}  "
            f"sigma_max={b.sigma_max} km/s",
            f"  instrument : {self.instrument.kind if self.instrument else 'none'}  "
            f"sigma_inst [{si.min():.1f}, {si.max():.1f}] km/s"
            + (f" x theta[{self.inst_scale_key!r}] in [{self.inst_scale_range[0]:g}, "
               f"{self.inst_scale_range[1]:g}]" if self.free_inst_scale else
               (f" (scale {self.instrument.scale:g} included)"
                if self.instrument is not None and self.instrument.scale != 1.0 else "")),
            f"  library    : subtract={self.subtract_library}  "
            f"sigma_lib [{sl.min():.1f}, {sl.max():.1f}] km/s",
            f"  continuum kernel (fixed part) [{sf.min():.1f}, {sf.max():.1f}] km/s",
            f"  redshift   : " + (f"sampled in [{self.zred_range[0]:g}, {self.zred_range[1]:g}], "
                                f"reference z = {self.opz_ref - 1:g}"
                                if self.free_z else f"fixed at z = {self.opz_ref - 1:g}"),
            f"  log grid   : {self.grid.n} px, dv={self.grid.dv:.2f} km/s, "
            f"FFT length {self.grid.n_pad}, band {self.W.shape[1]} px",
            f"  lines      : {self.line_idx.size} painted in window",
        ])


@dataclass(eq=False)
class PhotometricBroadener:
    """sigma_gal broadening of spec_rest over the rest range the filters cover; pixels outside pass through."""
    window: WindowSmoother

    @property
    def grid(self) -> LogGrid:
        return self.window.grid

    @classmethod
    def build(cls, kinematics: Kinematics, wave_model, wmin_rest, wmax_rest):
        wave_model = np.asarray(wave_model, dtype=np.float64)
        marg = BAND_NSIGMA * kinematics.sigma_max / CKMS
        wmin = max(float(wmin_rest) * np.exp(-marg), wave_model[0])
        wmax = min(float(wmax_rest) * np.exp(+marg), wave_model[-1])
        return cls(window=WindowSmoother(
            LogGrid.build(wave_model, wmin, wmax, kinematics.sigma_max)))

    def __call__(self, spec_rest, sigma_gal_kms):
        w = self.window
        back = w.from_log(w.smooth(w.to_log(spec_rest), sigma_gal_kms))
        return spec_rest.at[w.idx_inside].set(back[w._inside].astype(spec_rest.dtype))
