"""Library spectral-resolution curves for SSPData: Gaussian sigma_v(lambda) [km/s] on the
rest-frame ``ssp_wave``, built as max(2-pixel sampling floor of the grid, documented library LSF).
"""
from __future__ import annotations

import numpy as np

from ..constants import C_KMS as CKMS
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_v_from_segments(wave, segments, *, outside=np.nan):
    """sigma_v(lambda) [km/s] on ``wave`` [A] from segments ``(lam_lo, lam_hi, kind, value)``,
    kind in ``"R_fwhm"`` (resolving power, FWHM), ``"fwhm_AA"`` (wavelength FWHM [A]),
    ``"sigma_v_kms"``; later segments overwrite earlier ones; ``outside`` (default NaN =
    unknown, no library subtraction) where no segment applies."""
    wave = np.asarray(wave, dtype=np.float64)
    out = np.full(wave.shape, float(outside), dtype=np.float64)
    for lam_lo, lam_hi, kind, value in segments:
        sel = (wave >= float(lam_lo)) & (wave <= float(lam_hi))
        if not sel.any():
            continue
        if kind == "R_fwhm":
            out[sel] = CKMS / float(value) * FWHM_TO_SIGMA
        elif kind == "fwhm_AA":
            out[sel] = CKMS * float(value) * FWHM_TO_SIGMA / wave[sel]
        elif kind == "sigma_v_kms":
            out[sel] = float(value)
        else:
            raise ValueError(
                f"unknown segment kind {kind!r}; expected 'R_fwhm', "
                f"'fwhm_AA', or 'sigma_v_kms'")
    return out


def miles_segments():
    """MILES: FWHM = 2.54 A over 3525-7500 A rest (Falcon-Barroso et al. 2011); NaN outside."""
    return [(3525.0, 7500.0, "fwhm_AA", 2.54)]


PRESETS = {
    "miles": miles_segments,
}


def sigma_v_for_library(wave, library, extra_segments=None):
    """sigma_v(lambda) from the preset segments of ``library`` (a key of ``PRESETS``) plus extras."""
    lib = str(library).lower()
    if lib not in PRESETS:
        raise KeyError(
            f"no shipped resolution preset for library {lib!r} (have: "
            f"{sorted(PRESETS)}).  Build the curve explicitly with "
            f"sigma_v_from_segments() using the numbers for your grid.")
    segments = list(PRESETS[lib]())
    if extra_segments:
        segments += list(extra_segments)
    return sigma_v_from_segments(wave, segments)


def grid_two_pixel_sigma_v(wave):
    """Two-pixel velocity FULL width of the grid, c * 2 * dln(lambda) [km/s] (not a sigma)."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("wave must be a 1-D array with at least 2 pixels")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be strictly increasing")
    dln = np.gradient(np.log(wave))
    return CKMS * 2.0 * dln


def sampling_floor_sigma_v(wave):
    """Sampling floor of the grid as a Gaussian sigma_v(lambda) [km/s]: the two-pixel width taken as FWHM."""
    return FWHM_TO_SIGMA * grid_two_pixel_sigma_v(wave)


def combined_sigma_v(wave, segments=None):
    """Element-wise max of the sampling floor and the LSF from ``segments``; finite (n_wave,) sigma_v [km/s]."""
    floor = sampling_floor_sigma_v(wave)
    if segments is None:
        return floor
    lsf = sigma_v_from_segments(wave, segments)
    return np.fmax(floor, lsf)


def combined_source(segment_source=None):
    """Provenance string for :func:`combined_sigma_v`."""
    base = ("grid 2-pixel sampling floor derived from ssp_wave "
            "(sigma_v = 0.8493 c dlambda/lambda)")
    if segment_source:
        return f"{base}; element-wise max with library LSF: {segment_source}"
    return base
