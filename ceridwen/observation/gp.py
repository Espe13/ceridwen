"""
ceridwen/observation/gp.py
==========================
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

