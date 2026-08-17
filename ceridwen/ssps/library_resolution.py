"""Library spectral-resolution curves for SSPData (σ_v(λ) in km/s).

The stellar library's intrinsic resolution must be subtracted in quadrature
from the target (instrumental) resolution before the model is smoothed to
match data — otherwise the model is over-broadened by the library width.
This module provides the machinery to attach a wavelength-dependent
resolution curve to an :class:`~ceridwen.ssps.ssp_data.SSPData` grid, from
which the Spectrum projection resolves its ``inres`` automatically.

Conventions
-----------
* The curve is stored as a Gaussian dispersion in VELOCITY units,
  ``sigma_v(lambda)`` [km/s], sampled on the grid's native (rest-frame)
  ``ssp_wave``.  Velocity units are redshift-invariant, so the observation
  layer needs no frame bookkeeping beyond mapping observed pixels to rest
  wavelengths.
* Conversions: for a resolving power quoted as FWHM,
  ``sigma_v = c / (R_fwhm * 2 sqrt(2 ln 2))``; for a wavelength FWHM,
  ``sigma_v = c * fwhm_lambda / (lambda * 2 sqrt(2 ln 2))``.
* The curve every grid ships with is the element-wise MAXIMUM of two
  contributions (:func:`combined_sigma_v`):

  1. the grid's own 2-pixel SAMPLING FLOOR, derived from the stored
     ``ssp_wave`` itself (:func:`sampling_floor_sigma_v`) — features in
     the stored spectra can never be narrower than ~2 pixels of the grid
     they are tabulated on, whatever the parent library's LSF was; and
  2. optionally, a documented library LINE-SPREAD FUNCTION given as
     piecewise segments (:func:`sigma_v_from_segments`), e.g. the MILES
     2.54 Angstrom FWHM.

  The floor requires no external numbers (it is computed from the
  wavelength array of the library grid at hand), so release curves are
  finite at EVERY pixel; NaN ("resolution unknown, subtract nothing")
  remains supported by the observation layer for hand-built curves only.
  Where the target resolution is finer than the combined curve, the
  quadrature subtraction floors at zero and the model simply stays at
  grid/library resolution (the Spectrum projection warns).

Presets
-------
Only LSF segments with a solid literature source are shipped.  MILES:
FWHM = 2.54 Angstrom over the MILES range (Falcon-Barroso et al. 2011,
A&A 532, A95).  For libraries without a documented LSF broader than
their tabulation (e.g. BPASS at 1 Angstrom sampling), the sampling floor
alone is the honest resolution — pass no segments.
"""
from __future__ import annotations

import numpy as np

CKMS = 2.998e5
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # 1/2.3548


def sigma_v_from_segments(wave, segments, *, outside=np.nan):
    """Build sigma_v(lambda) [km/s] from piecewise resolution segments.

    Parameters
    ----------
    wave : (n_wave,) array
        Rest-frame wavelength grid [Angstrom] (``SSPData.ssp_wave``).
    segments : list of (lam_lo, lam_hi, kind, value)
        Piecewise-constant resolution specification, applied in order (later
        segments overwrite earlier ones where they overlap):

        * ``("R_fwhm", R)``        — resolving power, FWHM convention.
        * ``("fwhm_AA", f)``       — wavelength FWHM [Angstrom].
        * ``("sigma_v_kms", s)``   — Gaussian sigma [km/s] directly.
    outside : float, optional
        Value for pixels covered by no segment.  Default NaN — the
        observation layer treats NaN as "resolution unknown here" and
        applies NO library subtraction at those pixels (conservative:
        reproduces the pre-fix behaviour rather than inventing a width).

    Returns
    -------
    (n_wave,) float64 array of sigma_v [km/s], NaN where unspecified.
    """
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


# --------------------------------------------------------------------------- #
# Documented presets (extend only with sourced numbers).                       #
# --------------------------------------------------------------------------- #
def miles_segments():
    """MILES optical library: FWHM = 2.54 A over 3525-7500 A (rest).

    Source: Falcon-Barroso et al. 2011, A&A 532, A95.  Outside this range
    FSPS pads with other libraries whose resolution differs — those pixels
    are left unspecified (NaN) unless you add segments for them.
    """
    return [(3525.0, 7500.0, "fwhm_AA", 2.54)]


PRESETS = {
    "miles": miles_segments,
}


def sigma_v_for_library(wave, library, extra_segments=None):
    """Convenience: preset segments for ``library`` (+ optional extras).

    ``library`` must be a key of :data:`PRESETS`; unknown libraries raise —
    deliberately, so a grid never silently ships a guessed curve.
    """
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
    """Two-pixel velocity WIDTH of the native grid, c*2*dln(lambda) [km/s].

    Diagnostic quantity (a full width, not a Gaussian sigma).  For the
    sigma_v the release curves store, use :func:`sampling_floor_sigma_v`,
    which treats this two-pixel width as a FWHM and converts to sigma.
    """
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("wave must be a 1-D array with at least 2 pixels")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be strictly increasing")
    dln = np.gradient(np.log(wave))
    return CKMS * 2.0 * dln


def sampling_floor_sigma_v(wave):
    """Sampling floor of the stored grid as a Gaussian sigma_v(lambda) [km/s].

    The spectra of a grid are tabulated on ``wave``; no feature they carry
    can be narrower than about two pixels of that tabulation, whatever the
    parent library's LSF was.  This function treats the local two-pixel
    width as a Gaussian FWHM:

        sigma_v = c * 2 * dln(lambda) / (2 sqrt(2 ln 2))
                ~= 0.8493 * c * dlambda / lambda

    computed per pixel from the grid's OWN wavelength array (``np.gradient``
    of ``ln lambda``), so it needs no external resolution numbers and is
    finite everywhere.  It is the correct lower bound for the library
    resolution curve: on grids resampled by FSPS (e.g. 4-400 Angstrom
    steps outside the MILES range of the master grid) the stored sampling —
    not the parent library LSF — is the binding resolution.
    """
    return FWHM_TO_SIGMA * grid_two_pixel_sigma_v(wave)


def combined_sigma_v(wave, segments=None):
    """The release resolution curve: max(sampling floor, library LSF).

    Element-wise maximum of the grid's own 2-pixel sampling floor
    (:func:`sampling_floor_sigma_v`, always finite) and, when ``segments``
    is given, the documented library LSF from
    :func:`sigma_v_from_segments` (NaN outside its segments; ``np.fmax``
    ignores those pixels, so the floor applies there).  With
    ``segments=None`` the floor alone is returned — the honest choice for
    libraries whose only documented "resolution" is their tabulation step.

    Returns a finite (n_wave,) float64 sigma_v [km/s] at every pixel.
    """
    floor = sampling_floor_sigma_v(wave)
    if segments is None:
        return floor
    lsf = sigma_v_from_segments(wave, segments)
    return np.fmax(floor, lsf)          # fmax: NaN in lsf -> floor wins


def combined_source(segment_source=None):
    """Provenance string matching :func:`combined_sigma_v`'s construction."""
    base = ("grid 2-pixel sampling floor derived from ssp_wave "
            "(sigma_v = 0.8493 c dlambda/lambda)")
    if segment_source:
        return f"{base}; element-wise max with library LSF: {segment_source}"
    return base
