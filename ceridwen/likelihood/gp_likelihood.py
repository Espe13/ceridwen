"""Squared-exponential Gaussian-process likelihood for a spectrum, compiled (jit / grad /
vmap): the correlated-residual model of ``ceridwen.observation.GaussianProcess`` inside the
sampled likelihood.

The residuals are first whitened by the diagonal noise model, ``r_i = (y_i - mu_i) /
sigma_eff,i``, then

    ln L = -1/2 r^T K^-1 r - 1/2 ln|K| - sum_i ln sqrt(2 pi sigma_eff,i^2)
    K_ij = delta_ij + a^2 exp(-(lambda_i - lambda_j)^2 / (2 l^2)) + eps delta_ij

over the unmasked pixels, with lambda the observed-frame pixel wavelength [Angstrom], ``a``
dimensionless (in units of sigma_eff) and ``l`` in Angstrom.  The hyperparameters enter as
natural logs, ``ln a`` and ``ln l``.  The mask is fixed at setup: masked pixels get the
identity row and column of K and r = 0, so they add exactly 0 to the quadratic form and to
ln|K|.  A dense Cholesky: O(n^3) per call, O(n^2) memory per vmap lane.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .likelihood import LikelihoodBase, LikelihoodOutput, finite_data
from .noise_model import DiagonalNoiseModel, NoiseModelOutput

Array = jax.Array

#: diagonal jitter eps of K, the default of ``GaussianProcess(jitter=)``
GP_JITTER = 1e-6


def gp_sqdist(wavelength) -> Array:
    """``(n, n)`` float64 matrix of squared wavelength differences ``(l_i - l_j)^2``
    [Angstrom^2]; built once at setup."""
    w = np.asarray(wavelength, dtype=np.float64).ravel()
    if not np.all(np.isfinite(w)):
        raise ValueError("the GP needs a finite wavelength for every pixel")
    return jnp.asarray((w[:, None] - w[None, :]) ** 2)


def _gp_cholesky(mask, sqdist, log_amp, log_len, eps):
    """Lower Cholesky factor of K (masked rows and columns the identity)."""
    m = jnp.asarray(mask, dtype=bool)
    corr = jnp.exp(2.0 * log_amp) * jnp.exp(-0.5 * sqdist * jnp.exp(-2.0 * log_len))
    K = jnp.where(m[:, None] & m[None, :], corr, 0.0)
    K = K + jnp.diag(jnp.where(m, 1.0 + eps, 1.0))
    return jnp.linalg.cholesky(K)


def lnlike_gp_gaussian(
    y       : Array,
    mu      : Array,
    inv_var : Array,
    log_det : Array,
    mask    : Array,
    sqdist  : Array,
    log_amp : Array,
    log_len : Array,
    eps     : float = GP_JITTER,
) -> tuple[Array, LikelihoodOutput]:
    """Return ``(lnl_total, LikelihoodOutput)`` of the GP likelihood (module docstring) for
    the noise-model output ``inv_var`` = 1/sigma_eff^2 and ``log_det`` = ln sqrt(2 pi
    sigma_eff^2).  ``residuals`` and ``chi`` are those of :func:`lnlike_diag_gaussian`;
    ``lnl_pointwise_i = -1/2 z_i^2 - ln L_ii - log_det_i`` with ``z = L^-1 r`` (the
    Cholesky-ordered decomposition, summing exactly to ``lnl_total``); masked entries 0."""
    mask  = jnp.asarray(mask, dtype=bool)
    resid = jnp.where(mask, y - mu, 0.0)
    r     = resid * jnp.sqrt(inv_var)
    L     = _gp_cholesky(mask, sqdist, log_amp, log_len, eps)
    z     = solve_triangular(L, r, lower=True)
    lnl_i = -0.5 * z ** 2 - jnp.log(jnp.diagonal(L)) - log_det
    lnl_masked = jnp.where(mask, lnl_i, 0.0)
    lnl_total  = jnp.sum(lnl_masked)
    aux = LikelihoodOutput(
        lnl_total     = lnl_total,
        lnl_pointwise = lnl_masked,
        residuals     = resid,
        chi           = r,
        ndof          = jnp.sum(mask),
    )
    return lnl_total, aux


def gp_conditional_mean(
    y       : Array,
    mu      : Array,
    inv_var : Array,
    mask    : Array,
    sqdist  : Array,
    log_amp : Array,
    log_len : Array,
    eps     : float = GP_JITTER,
) -> Array:
    """Posterior mean of the correlated GP component given the residuals, in DATA units
    (``sigma_eff * a^2 E K^-1 r``), on every pixel (masked ones predicted from the unmasked);
    ``mu +`` this is the model plus the GP's reading of the correlated residuals.  A
    diagnostic for plots: the sampled likelihood never computes it."""
    mask = jnp.asarray(mask, dtype=bool)
    r    = jnp.where(mask, (y - mu) * jnp.sqrt(inv_var), 0.0)
    L    = _gp_cholesky(mask, sqdist, log_amp, log_len, eps)
    alpha = jax.scipy.linalg.cho_solve((L, True), r)
    corr = jnp.exp(2.0 * log_amp) * jnp.exp(-0.5 * sqdist * jnp.exp(-2.0 * log_len))
    corr = jnp.where(mask[None, :], corr, 0.0)
    return (corr @ alpha) / jnp.sqrt(inv_var)


class GPGaussianLikelihood(LikelihoodBase):
    """Gaussian likelihood of one spectrum with a squared-exponential GP on the whitened
    residuals (:func:`lnlike_gp_gaussian`).

    Parameters
    ----------
    noise_model : DiagonalNoiseModel -- gives sigma_eff (err_scale, jitter, f_calib, f_data,
        noise_floor); its outlier mixture must be off
    sqdist : (n_pix, n_pix) -- :func:`gp_sqdist` of the observed-frame pixel wavelengths
    log_amp, log_len : str or float -- ln a and ln l [ln Angstrom]: a theta key (sampled) or
        a fixed value
    eps : float -- diagonal jitter (default ``GP_JITTER`` = 1e-6)
    """

    def __init__(self, noise_model: Optional[DiagonalNoiseModel] = None, sqdist=None,
                 log_amp: float | str = None, log_len: float | str = None,
                 eps: float = GP_JITTER) -> None:
        noise_model = DiagonalNoiseModel() if noise_model is None else noise_model
        if getattr(noise_model, "use_outlier", False):
            raise ValueError("GPGaussianLikelihood: the outlier mixture cannot be combined "
                             "with the GP (the mixture is per pixel, the GP couples pixels)")
        if sqdist is None:
            raise ValueError("GPGaussianLikelihood needs sqdist = gp_sqdist(wavelength)")
        for what, v in (("log_amp", log_amp), ("log_len", log_len)):
            if isinstance(v, str):
                if not v:
                    raise ValueError(f"GPGaussianLikelihood: {what} theta key is empty")
            elif isinstance(v, bool) or not isinstance(v, (int, float, np.floating)) \
                    or not np.isfinite(v):
                raise TypeError(f"GPGaussianLikelihood: {what} must be a theta key or a "
                                f"finite float (natural log), got {v!r}")
        if not (isinstance(eps, (int, float)) and float(eps) >= 0.0):
            raise ValueError(f"GPGaussianLikelihood: eps must be a float >= 0, got {eps!r}")
        self.noise_model = noise_model
        self.sqdist = sqdist
        self.log_amp = log_amp if isinstance(log_amp, str) else float(log_amp)
        self.log_len = log_len if isinstance(log_len, str) else float(log_len)
        self.eps = float(eps)
        #: the profiled calibration is refused with a GP (see fit._gp_refusals)
        self.poly_calibration = None

    @classmethod
    def _unflatten(cls, aux, leaves):
        # no validation: JAX may unflatten with tracers or placeholder leaves
        obj = object.__new__(cls)
        obj.noise_model, obj.log_amp, obj.log_len, obj.eps = aux
        obj.sqdist = leaves[0]
        obj.poly_calibration = None
        return obj

    @property
    def n_pix(self) -> int:
        return int(np.shape(self.sqdist)[0])

    @property
    def gp_param_names(self) -> tuple[str, ...]:
        """theta keys of the sampled GP hyperparameters (empty when both are fixed)."""
        return tuple(v for v in (self.log_amp, self.log_len) if isinstance(v, str))

    def gp_params(self, params: Optional[dict[str, Array]]) -> tuple[Array, Array]:
        """``(ln a, ln l)``: the fixed values, or looked up in ``params``."""
        def get(v, what):
            if not isinstance(v, str):
                return v
            if params is None or v not in params:
                raise KeyError(f"the GP reads {what} from theta[{v!r}], which is missing: "
                               "sample it (with a prior) or give a fixed value")
            return jnp.reshape(params[v], ())
        return get(self.log_amp, "ln a"), get(self.log_len, "ln l")

    def __call__(
        self,
        y         : Array,
        mu        : Array,
        sigma_obs : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """Return ``(lnl_total, LikelihoodOutput)``."""
        noise_out: NoiseModelOutput = self.noise_model.compute(
            sigma_obs, mu, mask, params, data=y
        )
        log_amp, log_len = self.gp_params(params)
        return lnlike_gp_gaussian(y, mu, noise_out.inv_var, noise_out.log_det, mask,
                                  self.sqdist, log_amp, log_len, self.eps)

    def conditional_mean(self, y, mu, sigma_obs, mask, params=None) -> Array:
        """:func:`gp_conditional_mean` at ``params`` (data units, every pixel)."""
        noise_out = self.noise_model.compute(sigma_obs, mu, mask, params, data=y)
        log_amp, log_len = self.gp_params(params)
        return gp_conditional_mean(y, mu, noise_out.inv_var, mask, self.sqdist,
                                   log_amp, log_len, self.eps)

    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior for one Spectrum (``.flux``, ``.uncertainty``,
        ``.mask``)."""
        y, sigma_obs = finite_data(observations.flux, observations.uncertainty)
        mask = observations.mask
        lhood = self

        @jax.jit
        def lnprobfn_gp(theta: dict[str, Array]) -> Array:
            lnl, _ = lhood(y, model.predict(theta), sigma_obs, mask, theta)
            return lnl + prior.log_prob(theta)

        return lnprobfn_gp

    def config(self) -> dict:
        """JSON-able GP settings (stored in the result file's ``likelihood_json``)."""
        return {"kernel": "squared_exponential", "log_amp": self.log_amp,
                "log_len": self.log_len, "eps": self.eps, "n_pix": self.n_pix,
                "wavelength": "observed frame, Angstrom",
                "amplitude_units": "sigma_eff", "sampled_parameters": list(self.gp_param_names)}

    def __repr__(self) -> str:
        return (f"GPGaussianLikelihood(noise_model={self.noise_model!r}, "
                f"log_amp={self.log_amp!r}, log_len={self.log_len!r}, eps={self.eps!r}, "
                f"n_pix={self.n_pix})")


jax.tree_util.register_pytree_node(
    GPGaussianLikelihood,
    flatten_func   = lambda lh: ([lh.sqdist], (lh.noise_model, lh.log_amp, lh.log_len, lh.eps)),
    unflatten_func = lambda aux, leaves: GPGaussianLikelihood._unflatten(aux, leaves),
)
