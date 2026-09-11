"""Gaussian Process noise model for spectral residuals."""

import jax.numpy as jnp
import numpy as np


class GaussianProcess:
    """Squared-exponential GP on normalised spectral residuals, with
    covariance K = I + a^2 SE(l) + jitter I.

    Parameters
    ----------
    amplitude : float, dimensionless -- kernel amplitude in units of the per-pixel sigma.
    length_scale : float, Å -- correlation length.
    jitter : float -- diagonal added for numerical stability.
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
        """Full Gaussian log-likelihood of the normalised residuals under K,
        including the white-noise identity term and the -n/2 ln(2 pi) constant;
        do NOT add a separate -1/2 sum r^2 term."""
        r   = np.asarray(residuals,  dtype=np.float64)
        wav = np.asarray(wavelength, dtype=np.float64)
        if mask is not None:
            m   = np.asarray(mask, dtype=bool)
            r   = r[m]
            wav = wav[m]
        n = len(r)
        if n == 0:
            return 0.0
        L, logdet = self._factor(wav)
        if L is None:
            return -np.inf
        from scipy.linalg import cho_solve
        alpha = cho_solve((L, True), r)
        return (-0.5 * float(r @ alpha) - logdet - 0.5 * n * float(np.log(2.0 * np.pi)))

    def _factor(self, wav):
        """Cached Cholesky factor and log|K|^(1/2) on ``wav``."""
        key = (wav.shape[0], float(wav[0]), float(wav[-1]), float(wav.sum()))
        cache = getattr(self, "_chol_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        dlam = wav[:, None] - wav[None, :]
        K = self.amplitude ** 2 * np.exp(-0.5 * (dlam / self.length_scale) ** 2)
        K[np.diag_indices_from(K)] += 1.0 + self.jitter
        try:
            L = np.linalg.cholesky(K)
            logdet = float(np.sum(np.log(np.diag(L))))
        except np.linalg.LinAlgError:
            L, logdet = None, np.inf
        self._chol_cache = (key, L, logdet)
        return L, logdet

    def __repr__(self):
        return (f"GaussianProcess(amplitude={self.amplitude}, "
                f"length_scale={self.length_scale}, jitter={self.jitter})")
