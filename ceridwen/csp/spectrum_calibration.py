"""
Multiplicative spectrophotometric calibration of ``Spectrum`` predictions from the
optional theta keys ``spectrum_scaling`` (grey level) and ``spectrum_calib`` (Legendre shape),
one pair per spectrum (``spectrum_scaling_<obs.name>`` with several spectra).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

__all__ = ["spectrum_calibration_factor", "legendre_design_matrix", "calibration_keys"]

_CACHE_ATTR = "_ceridwen_calib_design_cache"


def legendre_design_matrix(wavelength, order: int) -> np.ndarray:
    """``(n_pix, order)`` float64 matrix of Legendre P_1..P_order on the pixel wavelengths
    mapped affinely onto [-1, 1] (range over ALL pixels, masked or not; no P_0 term)."""
    lam = np.asarray(wavelength, dtype=np.float64)
    if lam.ndim != 1 or lam.size < 2:
        raise ValueError(
            "spectrum_calib needs a 1-D observed wavelength array with at "
            f"least 2 pixels; got shape {lam.shape}."
        )
    fin = np.isfinite(lam)
    lo, hi = float(lam[fin].min()), float(lam[fin].max())
    if not hi > lo:
        raise ValueError(
            "spectrum_calib: observed wavelength range is degenerate "
            f"(min = max = {lo})."
        )
    x = 2.0 * (lam - lo) / (hi - lo) - 1.0
    x = np.where(fin, x, 0.0)
    return np.polynomial.legendre.legvander(x, int(order))[:, 1:]


def _design(obs, order: int):
    cache = getattr(obs, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(obs, _CACHE_ATTR, cache)
    key = (int(order), int(np.asarray(obs.wavelength).shape[0]))
    if key not in cache:
        cache[key] = np.asarray(legendre_design_matrix(obs.wavelength, order))
    return jnp.asarray(cache[key])


def calibration_keys(obs, theta):
    """``(level key, shape key)`` of ``obs`` in ``theta`` (either may be None): the
    per-spectrum ``spectrum_scaling_<obs.name>`` / ``spectrum_calib_<obs.name>``
    before the plain ``spectrum_scaling`` / ``spectrum_calib``.  Which names a model may use
    (the plain one only with a single Spectrum) is checked by ``SedModel`` at construction."""
    name = getattr(obs, "name", None)
    out = []
    for root in ("spectrum_scaling", "spectrum_calib"):
        own = f"{root}_{name}"
        out.append(own if own in theta else (root if root in theta else None))
    return tuple(out)


def spectrum_calibration_factor(obs, theta, dtype=None):
    """Factor ``spectrum_scaling * (1 + spectrum_calib . P(x))`` multiplying the MODEL spectrum
    of ``obs`` (keys from :func:`calibration_keys`): a scalar (level only), an ``(n_pix,)``
    vector (shape term), or ``None`` if the spectrum has neither.
    """
    level_key, shape_key = calibration_keys(obs, theta)
    if level_key is None and shape_key is None:
        return None
    factor = None
    if level_key is not None:
        factor = jnp.ravel(theta[level_key])[0]
    if shape_key is not None:
        coeff = jnp.ravel(theta[shape_key])
        order = int(coeff.shape[0])
        if order < 1:
            raise ValueError(
                "spectrum_calib must have at least one coefficient "
                "(shape (order,), order >= 1); the level term is "
                "spectrum_scaling."
            )
        design = _design(obs, order)
        shape_term = 1.0 + design @ coeff
        factor = shape_term if factor is None else factor * shape_term
    if dtype is not None:
        factor = factor.astype(dtype)
    return factor
