"""Profiled polynomial calibration of a ``Spectrum`` (v1.0.7), the equivalent of Prospector's
``PolyOptCal`` (``prospect/observation/observation.py:579-658``, commit a78d153).

Instead of sampling calibration coefficients, the likelihood multiplies the model spectrum
``mu`` by the response ``1 + sum_{m=0}^{M} c_m T_m(x)`` (Chebyshev T_m, x the observed pixel
wavelength mapped onto [-1, 1] over the UNMASKED pixels, as ``prospect.observation.wave_to_x``)
with the coefficients that maximise the Gaussian likelihood at the current parameters:

    c = argmin_c  sum_i w_i (y_i - mu_i (1 + A_i c))^2 + sum_m reg_m^2 c_m^2,

a weighted linear least-squares problem solved in closed form inside the jitted likelihood,
so it adds no sampled dimensions and is differentiable.  The weights are the noise model's
``1 / sigma_eff^2`` at the uncalibrated model (Prospector: the raw ``1 / sigma^2``; the two are
equal when the noise model has no sampled or model-anchored terms).  ``order = 0`` means no
polynomial, as in Prospector (``polyopt = order > 0``).
"""
from __future__ import annotations

import hashlib

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["PolynomialCalibration", "chebyshev_design_matrix"]


def chebyshev_design_matrix(wavelength, mask, order: int) -> np.ndarray:
    """``(n_pix, order + 1)`` Chebyshev T_0..T_order on the pixel wavelengths mapped onto
    [-1, 1] over the unmasked pixels (masked pixels may fall outside), float64."""
    lam = np.asarray(wavelength, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(lam)
    if m.sum() <= order:
        raise ValueError(f"polynomial_order={order} needs more than {order} unmasked pixels, "
                         f"got {int(m.sum())}")
    x = lam - lam[m].min()
    x = 2.0 * (x / x[m].max()) - 1.0
    x = np.where(np.isfinite(x), x, 0.0)
    return np.polynomial.chebyshev.chebvander(x, int(order))


class PolynomialCalibration:
    """Static description of the profiled polynomial of one spectrum: the design matrix
    ``A`` (n_pix, order+1) and the regularisation (scalar or one per coefficient).  Hashable
    (by value), so it can sit in a likelihood's static pytree data."""

    def __init__(self, design, regularization=0.0):
        A = np.asarray(design, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] < 2:
            raise ValueError("the calibration design matrix must be (n_pix, order + 1), order >= 1")
        reg = np.broadcast_to(np.asarray(regularization, dtype=np.float64), (A.shape[1],)).copy()
        if np.any(~np.isfinite(reg)) or np.any(reg < 0.0):
            raise ValueError(f"polynomial_regularization must be finite and >= 0, got {reg}")
        self.A = A
        self.reg = reg
        self._key = hashlib.sha256(A.tobytes() + reg.tobytes()).hexdigest()

    @classmethod
    def for_spectrum(cls, obs):
        """From ``Spectrum(polynomial_order=..., polynomial_regularization=...)``; None when off."""
        order = int(getattr(obs, "polynomial_order", 0) or 0)
        if order <= 0:
            return None
        return cls(chebyshev_design_matrix(obs.wavelength, obs.mask, order),
                   getattr(obs, "polynomial_regularization", 0.0))

    @property
    def order(self) -> int:
        return self.A.shape[1] - 1

    def __eq__(self, other):
        return isinstance(other, PolynomialCalibration) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return (f"PolynomialCalibration(order={self.order}, n_pix={self.A.shape[0]}, "
                f"regularization={self.reg.tolist()})")

    def config(self) -> dict:
        return {"order": self.order, "regularization": self.reg.tolist(),
                "basis": "Chebyshev T_0..T_order over the unmasked wavelength range"}

    def solve(self, y, mu, inv_var, mask):
        """``(coefficients (order+1,), response (n_pix,))`` maximising the likelihood; a
        column-scaled normal-equation solve, differentiable in ``mu`` and ``inv_var``."""
        A = jnp.asarray(self.A, dtype=mu.dtype)
        w = jnp.where(mask, inv_var, 0.0)
        D = mu[:, None] * A
        r = jnp.where(mask, y - mu, 0.0)
        M = D.T @ (w[:, None] * D) + jnp.diag(jnp.asarray(self.reg, dtype=mu.dtype) ** 2)
        b = D.T @ (w * r)
        d = 1.0 / jnp.sqrt(jnp.maximum(jnp.diagonal(M), jnp.finfo(mu.dtype).tiny))
        c = d * jnp.linalg.solve(d[:, None] * M * d[None, :], d * b)
        return c, 1.0 + A @ c

    def calibrate(self, y, mu, sigma_obs, mask, params, noise_model):
        """``mu`` times the profiled response (the noise model's weights at ``mu``)."""
        out = noise_model.compute(sigma_obs, mu, mask, params, data=y)
        _c, response = self.solve(y, mu, out.inv_var, mask)
        return mu * response


def calibrated_prediction(lhood, y, mu, sigma_obs, mask, params):
    """``mu`` after the likelihood's profiled calibration (unchanged when it has none)."""
    pc = getattr(lhood, "poly_calibration", None)
    if pc is None:
        return mu
    return pc.calibrate(y, mu, sigma_obs, mask, params, lhood.noise_model)


jax.tree_util.register_static(PolynomialCalibration)
