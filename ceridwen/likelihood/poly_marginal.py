"""Analytic marginalisation of the calibration polynomial of a ``Spectrum``
(``Spectrum(polynomial_order=M, polynomial_mode="marginalize", polynomial_prior_sigma=s)``).

The model spectrum ``mu`` is multiplied by the response ``1 + A c``, ``A`` the Chebyshev
design matrix T_0..T_M of :func:`~ceridwen.likelihood.poly_calibration.chebyshev_design_matrix`
(the profiled mode's basis), and the coefficients have the Gaussian prior
``c ~ N(0, diag(s^2))``.  The residual ``r = y - mu = D c + n``, ``D = diag(mu) A``,
``n ~ N(0, C)`` with ``C^-1 = W`` the noise model's inverse variance at the UNCALIBRATED ``mu``
(as the profiled mode), is linear in ``c``, so ``c`` integrates out in closed form (Woodbury and
the matrix determinant lemma):

    ln L = ln L_diag(r) + 1/2 b^T M^-1 b - 1/2 ln|M| - 1/2 ln|Lambda|,
    M = D^T W D + Lambda^-1,   b = D^T W r,   Lambda = diag(s^2),

O(n k^2) per call (k = M + 1), no n x n matrix.  It is evaluated as
``P = S G S + I = S M S`` (``S = diag(s)``, ``G = D^T W D``; exact also for s_m = 0, a
coefficient pinned at 0), Jacobi-scaled to unit diagonal (``Q = E P E``, as the column scaling
of :meth:`PolynomialCalibration.solve`) before its Cholesky factorisation.  Given theta the
coefficients are Gaussian with mean ``M^-1 b`` and covariance ``M^-1``; the mean equals the
profiled solution with ``polynomial_regularization = 1/s``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from .likelihood import (LikelihoodBase, LikelihoodOutput, calibrated_mu,
                         single_observation_data)
from .noise_model import DiagonalNoiseModel
from .poly_calibration import chebyshev_design_matrix

__all__ = ["PolynomialMarginal", "PolyMarginalGaussianLikelihood", "poly_marginal_loglike",
           "poly_marginal_posterior"]

Array = jax.Array


def _factor(y, mu, inv_var, mask, A, s):
    """``(r, D, w, L, g, v, logdet_P)``: the masked residual, design and weights, the Cholesky
    factor ``L`` of the unit-diagonal ``Q = E P E``, ``g = s e`` (the combined column scale) and
    ``v = L^-1 g b`` (so that ``b^T M^-1 b = v.v``)."""
    w = jnp.where(mask, inv_var, 0.0)
    r = jnp.where(mask, y - mu, 0.0)
    D = jnp.where(mask[:, None], mu[:, None] * A, 0.0)
    G = D.T @ (w[:, None] * D)
    b = D.T @ (w * r)
    P = s[:, None] * G * s[None, :] + jnp.eye(s.shape[0], dtype=G.dtype)
    e = 1.0 / jnp.sqrt(jnp.diagonal(P))                 # P_jj >= 1
    L = jnp.linalg.cholesky(e[:, None] * P * e[None, :])
    g = s * e
    v = solve_triangular(L, g * b, lower=True)
    logdet_P = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L))) - 2.0 * jnp.sum(jnp.log(e))
    return r, D, w, L, g, v, logdet_P


def poly_marginal_loglike(y, mu, inv_var, log_det, mask, A, prior_sigma,
                          *, return_posterior=False):
    """Marginal ln-likelihood of the spectrum over the calibration coefficients.

    y, mu : (n,) data and uncalibrated model; inv_var, log_det : (n,) from the noise model
    (``1/sigma_eff^2`` and ``0.5 ln(2 pi sigma_eff^2)``); mask : (n,) bool, True = used;
    A : (n, k) design; prior_sigma : (k,) prior widths s (>= 0, finite).

    Returns ``(lnl_total, LikelihoodOutput)``, whose ``residuals`` / ``chi`` /
    ``lnl_pointwise`` are those of the conditional-mean calibrated model ``mu (1 + A c_hat)``;
    ``lnl_total = sum(lnl_pointwise) - 1/2 c_hat^T Lambda^-1 c_hat - 1/2 ln|P|`` (the prior and
    Occam terms are global, not per pixel).  With ``return_posterior`` also the conditional
    mean (k,) and covariance (k, k) of the coefficients.
    """
    A = jnp.asarray(A, dtype=mu.dtype)
    s = jnp.asarray(prior_sigma, dtype=mu.dtype)
    r, D, w, L, g, v, logdet_P = _factor(y, mu, inv_var, mask, A, s)
    lnorm = jnp.sum(jnp.where(mask, log_det, 0.0))
    lnl_total = -0.5 * jnp.sum(w * r * r) - lnorm + 0.5 * jnp.dot(v, v) - 0.5 * logdet_P

    c_hat = g * cho_solve((L, True), g * (D.T @ (w * r)))
    resid = jnp.where(mask, r - D @ c_hat, 0.0)
    chi = resid * jnp.sqrt(inv_var)
    lnl_i = jnp.where(mask, -0.5 * chi ** 2 - log_det, 0.0)
    aux = LikelihoodOutput(lnl_total=lnl_total, lnl_pointwise=lnl_i, residuals=resid,
                           chi=jnp.where(mask, chi, 0.0), ndof=jnp.sum(mask))
    if not return_posterior:
        return lnl_total, aux
    cov = g[:, None] * cho_solve((L, True), jnp.eye(g.shape[0], dtype=g.dtype)) * g[None, :]
    return lnl_total, aux, c_hat, cov


def poly_marginal_posterior(y, mu, inv_var, mask, A, prior_sigma):
    """``(mean (k,), cov (k, k))`` of the coefficients given the model ``mu``."""
    A = jnp.asarray(A, dtype=mu.dtype)
    s = jnp.asarray(prior_sigma, dtype=mu.dtype)
    r, D, w, L, g, _v, _ld = _factor(y, mu, inv_var, mask, A, s)
    mean = g * cho_solve((L, True), g * (D.T @ (w * r)))
    cov = g[:, None] * cho_solve((L, True), jnp.eye(g.shape[0], dtype=g.dtype)) * g[None, :]
    return mean, cov


class PolynomialMarginal:
    """Static description of the marginalised polynomial of one spectrum: the design matrix
    ``A`` (n_pix, order+1) and the prior widths ``sigma`` (order+1,).  Hashable (by value), so
    it can sit in a likelihood's static pytree data."""

    def __init__(self, design, prior_sigma):
        A = np.asarray(design, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] < 2:
            raise ValueError("the calibration design matrix must be (n_pix, order + 1), order >= 1")
        sig = np.broadcast_to(np.asarray(prior_sigma, dtype=np.float64), (A.shape[1],)).copy()
        if np.any(~np.isfinite(sig)) or np.any(sig < 0.0):
            raise ValueError(f"polynomial_prior_sigma must be finite and >= 0, got {sig}")
        self.A = A
        self.sigma = sig
        self._key = hashlib.sha256(A.tobytes() + sig.tobytes()).hexdigest()

    @classmethod
    def for_spectrum(cls, obs):
        """From ``Spectrum(polynomial_mode="marginalize", ...)``; None otherwise."""
        order = int(getattr(obs, "polynomial_order", 0) or 0)
        if order <= 0 or getattr(obs, "polynomial_mode", "profile") != "marginalize":
            return None
        return cls(chebyshev_design_matrix(obs.wavelength, obs.mask, order),
                   obs.polynomial_prior_sigma)

    @property
    def order(self) -> int:
        return self.A.shape[1] - 1

    def __eq__(self, other):
        return isinstance(other, PolynomialMarginal) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return (f"PolynomialMarginal(order={self.order}, n_pix={self.A.shape[0]}, "
                f"prior_sigma={self.sigma.tolist()})")

    def config(self) -> dict:
        return {"mode": "marginalize", "order": self.order, "prior_sigma": self.sigma.tolist(),
                "prior_mean": 0.0,
                "basis": "Chebyshev T_0..T_order over the unmasked wavelength range"}


@dataclass(frozen=True)
class PolyMarginalGaussianLikelihood(LikelihoodBase):
    """Diagonal Gaussian likelihood of a spectrum with its calibration polynomial marginalised
    analytically (:func:`poly_marginal_loglike`).  No outlier mixture and no upper limits
    (the marginal is exact only for a Gaussian likelihood)."""

    noise_model: DiagonalNoiseModel = field(default_factory=DiagonalNoiseModel)
    poly_marginal: Optional[PolynomialMarginal] = None

    def __post_init__(self):
        if self.poly_marginal is None:
            raise ValueError("PolyMarginalGaussianLikelihood needs a PolynomialMarginal")
        if getattr(self.noise_model, "use_outlier", False):
            raise NotImplementedError(
                "the outlier mixture cannot be combined with a marginalised calibration "
                "polynomial: the closed-form marginal needs a Gaussian likelihood")

    def __call__(self, y, mu, sigma_obs, mask, params=None):
        """Return ``(lnl_total, LikelihoodOutput)`` (see :func:`poly_marginal_loglike`)."""
        nout = self.noise_model.compute(sigma_obs, mu, mask, params, data=y)
        return poly_marginal_loglike(y, mu, nout.inv_var, nout.log_det, mask,
                                     self.poly_marginal.A, self.poly_marginal.sigma)

    def conditional(self, y, mu, sigma_obs, mask, params=None):
        """``(mean (k,), cov (k, k), response (n,))``: the conditional posterior of the
        coefficients at the model ``mu`` and the response ``1 + A mean``."""
        nout = self.noise_model.compute(sigma_obs, mu, mask, params, data=y)
        mean, cov = poly_marginal_posterior(y, mu, nout.inv_var, mask,
                                            self.poly_marginal.A, self.poly_marginal.sigma)
        return mean, cov, 1.0 + jnp.asarray(self.poly_marginal.A, dtype=mu.dtype) @ mean

    def make_lnprobfn(self, observations, model, prior) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior for one observation (``.flux``, ``.uncertainty``, ``.mask``)."""
        y, sigma_obs, mask, calib, _ = single_observation_data(observations, "PolyMarginalGaussianLikelihood")
        lhood = self

        @jax.jit
        def lnprobfn(theta):
            lnl, _ = lhood(y, calibrated_mu(model.predict(theta), calib), sigma_obs, mask,
                           theta)
            return lnl + prior.log_prob(theta)

        return lnprobfn

    def __repr__(self) -> str:
        return (f"PolyMarginalGaussianLikelihood(noise_model={self.noise_model!r}, "
                f"poly_marginal={self.poly_marginal!r})")


jax.tree_util.register_static(PolynomialMarginal)
jax.tree_util.register_pytree_node(
    PolyMarginalGaussianLikelihood,
    flatten_func=lambda lh: ([], (lh.noise_model, lh.poly_marginal)),
    unflatten_func=lambda aux, _: PolyMarginalGaussianLikelihood(
        noise_model=aux[0], poly_marginal=aux[1]),
)
