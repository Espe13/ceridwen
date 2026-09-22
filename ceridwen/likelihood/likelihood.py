"""Gaussian log-likelihood kernels and classes."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .noise_model import DiagonalNoiseModel, NoiseModelBase, NoiseModelOutput

Array = jax.Array


class LikelihoodOutput:
    """Per-datum diagnostics returned as the ``aux`` of ``(lnl_total, aux)``; masked entries are 0."""

    __slots__ = ("lnl_total", "lnl_pointwise", "residuals", "chi", "ndof")

    def __init__(
        self,
        lnl_total     : Array,
        lnl_pointwise : Array,
        residuals     : Array,
        chi           : Array,
        ndof          : Array,
    ) -> None:
        self.lnl_total     = lnl_total
        self.lnl_pointwise = lnl_pointwise
        self.residuals     = residuals
        self.chi           = chi
        self.ndof          = ndof

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LikelihoodOutput("
            f"lnl={float(self.lnl_total):.4f}, "
            f"ndof={int(self.ndof)}, "
            f"chi2_red={float(jnp.sum(self.chi**2) / jnp.maximum(self.ndof, 1)):.3f})"
        )


jax.tree_util.register_pytree_node(
    LikelihoodOutput,
    flatten_func=lambda o: (
        [o.lnl_total, o.lnl_pointwise, o.residuals, o.chi, o.ndof],
        None,
    ),
    unflatten_func=lambda _, leaves: LikelihoodOutput(*leaves),
)


def lnlike_diag_gaussian(
    y       : Array,
    mu      : Array,
    inv_var : Array,
    log_det : Array,
    mask    : Array,
) -> tuple[Array, LikelihoodOutput]:
    """Return ``(lnl_total, LikelihoodOutput)`` with ``lnl_i = -0.5 chi_i^2 - log_det_i``;
    ``log_det = 0.5 log(2 pi sigma_eff^2)``, masked data contribute 0."""
    resid = y - mu
    chi   = resid * jnp.sqrt(inv_var)

    lnl_i = -0.5 * chi ** 2 - log_det

    lnl_masked   = jnp.where(mask, lnl_i,  0.0)
    resid_masked  = jnp.where(mask, resid,  0.0)
    chi_masked    = jnp.where(mask, chi,    0.0)

    lnl_total = jnp.sum(lnl_masked)
    ndof      = jnp.sum(mask)

    aux = LikelihoodOutput(
        lnl_total     = lnl_total,
        lnl_pointwise = lnl_masked,
        residuals     = resid_masked,
        chi           = chi_masked,
        ndof          = ndof,
    )
    return lnl_total, aux


def lnlike_diag_gaussian_with_upper_limits(
    y               : Array,
    mu              : Array,
    inv_var         : Array,
    log_det         : Array,
    mask            : Array,
    is_upper_limit  : Array,
) -> tuple[Array, LikelihoodOutput]:
    """Like :func:`lnlike_diag_gaussian`, but data flagged ``is_upper_limit`` use a one-sided
    penalty ``-0.5 max(mu - y, 0)^2 / sigma_eff^2`` (``y`` is the limit value); ``log_det``
    is kept on both branches and ``aux.chi`` stays signed."""
    resid     = y - mu
    sqrt_iv   = jnp.sqrt(inv_var)
    chi       = resid * sqrt_iv

    chi2_det  = chi ** 2

    chi2_ul   = jnp.maximum(-chi, 0.0) ** 2

    chi2_used = jnp.where(is_upper_limit, chi2_ul, chi2_det)
    lnl_i     = -0.5 * chi2_used - log_det

    lnl_masked   = jnp.where(mask, lnl_i,  0.0)
    resid_masked = jnp.where(mask, resid,  0.0)
    chi_masked   = jnp.where(mask, chi,    0.0)

    lnl_total = jnp.sum(lnl_masked)
    ndof      = jnp.sum(mask)

    aux = LikelihoodOutput(
        lnl_total     = lnl_total,
        lnl_pointwise = lnl_masked,
        residuals     = resid_masked,
        chi           = chi_masked,
        ndof          = ndof,
    )
    return lnl_total, aux


class LikelihoodBase(abc.ABC):
    """Abstract base for likelihood classes."""

    @abc.abstractmethod
    def __call__(
        self,
        y         : Array,
        mu        : Array,
        sigma_obs : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """Return ``(lnl_total, LikelihoodOutput)``; ``params`` carries noise nuisance parameters."""

    @abc.abstractmethod
    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted ``theta -> log-posterior`` closing over observations, model and prior."""


@dataclass(frozen=True)
class DiagonalGaussianLikelihood(LikelihoodBase):
    """Gaussian log-likelihood with an independent (diagonal) noise model."""

    noise_model: DiagonalNoiseModel = field(
        default_factory=DiagonalNoiseModel
    )

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
        return lnlike_diag_gaussian(
            y, mu, noise_out.inv_var, noise_out.log_det, mask
        )

    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior for one observation (needs ``.flux``, ``.uncertainty``, ``.mask``)."""
        y         : Array = observations.flux
        sigma_obs : Array = observations.uncertainty
        mask      : Array = observations.mask
        noise_model       = self.noise_model

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            mu = model.predict(theta)
            noise_out = noise_model.compute(sigma_obs, mu, mask, theta, data=y)
            lnl, _    = lnlike_diag_gaussian(
                y, mu, noise_out.inv_var, noise_out.log_det, mask
            )
            lnp = prior.log_prob(theta)
            return lnl + lnp

        return lnprobfn

    def __repr__(self) -> str:
        return f"DiagonalGaussianLikelihood(noise_model={self.noise_model!r})"


jax.tree_util.register_pytree_node(
    DiagonalGaussianLikelihood,
    flatten_func   = lambda lh: ([], (lh.noise_model,)),
    unflatten_func = lambda aux, _: DiagonalGaussianLikelihood(
        noise_model=aux[0]
    ),
)


@dataclass(frozen=True)
class DiagonalGaussianLikelihoodWithUpperLimits(LikelihoodBase):
    """Diagonal Gaussian log-likelihood honouring per-datum upper-limit flags
    (one-sided penalty; reduces to :class:`DiagonalGaussianLikelihood` when no flags are set)."""

    noise_model: DiagonalNoiseModel = field(
        default_factory=DiagonalNoiseModel
    )

    def __call__(
        self,
        y               : Array,
        mu              : Array,
        sigma_obs       : Array,
        mask            : Array,
        params          : Optional[dict[str, Array]] = None,
        is_upper_limit  : Optional[Array]            = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """Return ``(lnl_total, LikelihoodOutput)``; ``is_upper_limit=None`` means all detections."""
        noise_out: NoiseModelOutput = self.noise_model.compute(
            sigma_obs, mu, mask, params, data=y
        )
        if is_upper_limit is None:
            is_upper_limit = jnp.zeros_like(mask, dtype=bool)
        return lnlike_diag_gaussian_with_upper_limits(
            y, mu, noise_out.inv_var, noise_out.log_det, mask, is_upper_limit,
        )

    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior using ``observations.upper_limit`` (all-False if absent)."""
        y         : Array = observations.flux
        sigma_obs : Array = observations.uncertainty
        mask      : Array = observations.mask
        is_ul = getattr(observations, "upper_limit", None)
        if is_ul is None:
            is_ul = jnp.zeros_like(mask, dtype=bool)
        else:
            is_ul = jnp.asarray(is_ul, dtype=bool)
        noise_model = self.noise_model

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            mu = model.predict(theta)
            noise_out = noise_model.compute(sigma_obs, mu, mask, theta, data=y)
            lnl, _    = lnlike_diag_gaussian_with_upper_limits(
                y, mu, noise_out.inv_var, noise_out.log_det, mask, is_ul,
            )
            lnp = prior.log_prob(theta)
            return lnl + lnp

        return lnprobfn

    def __repr__(self) -> str:
        return (
            f"DiagonalGaussianLikelihoodWithUpperLimits("
            f"noise_model={self.noise_model!r})"
        )


jax.tree_util.register_pytree_node(
    DiagonalGaussianLikelihoodWithUpperLimits,
    flatten_func   = lambda lh: ([], (lh.noise_model,)),
    unflatten_func = lambda aux, _: DiagonalGaussianLikelihoodWithUpperLimits(
        noise_model=aux[0]
    ),
)


@dataclass(frozen=True)
class MultiObservationLikelihood(LikelihoodBase):
    """Sum of independent likelihoods over several observation keys.

    Parameters
    ----------
    keys : tuple of str -- observation keys, e.g. ``("phot", "spec", "lines")``
    likelihoods : tuple of LikelihoodBase -- one per key, same order
    """

    keys        : tuple[str, ...]          = field(default_factory=tuple)
    likelihoods : tuple[LikelihoodBase, ...] = field(default_factory=tuple)

    def __call__(
        self,
        y         : dict[str, Array],
        mu        : dict[str, Array],
        sigma_obs : dict[str, Array],
        mask      : dict[str, Array],
        params    : Optional[dict[str, Array]] = None,
        is_upper_limit : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, dict[str, LikelihoodOutput]]:
        """Return ``(lnl_total, {key: LikelihoodOutput})``; all inputs are dicts keyed like ``self.keys``
        (``is_upper_limit`` only needs the keys whose likelihood honours upper limits)."""
        lnl_total = jnp.zeros(())
        aux: dict[str, LikelihoodOutput] = {}
        for key, lhood in zip(self.keys, self.likelihoods):
            ul = None if is_upper_limit is None else is_upper_limit.get(key)
            if ul is not None:
                lnl_i, aux_i = lhood(y[key], mu[key], sigma_obs[key], mask[key], params,
                                     is_upper_limit=ul)
            else:
                lnl_i, aux_i = lhood(y[key], mu[key], sigma_obs[key], mask[key], params)
            lnl_total    = lnl_total + lnl_i
            aux[key]     = aux_i
        return lnl_total, aux

    def make_lnprobfn(
        self,
        observations : dict[str, Any],
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior; ``observations`` and ``model.predict(theta)`` are dicts keyed
        like ``self.keys``.  Each observation's ``sky``, ``calibration`` and ``upper_limit`` are honoured
        as in ``ceridwen.sampler.runner.run_sampler``."""
        static_data = {}
        for key in self.keys:
            obs = observations[key]
            y = obs.flux
            sky = getattr(obs, "sky", None)
            if sky is not None:
                y = y - sky
            ul = getattr(obs, "upper_limit", None)
            ul = None if ul is None or not bool(jnp.any(ul)) else jnp.asarray(ul, dtype=bool)
            static_data[key] = (y, obs.uncertainty, obs.mask, getattr(obs, "calibration", None), ul)
        keys        = self.keys
        likelihoods = self.likelihoods

        if getattr(model, "_eline_system", None) is not None:
            from .eline_marginal import joint_loglike

            @jax.jit
            def lnprobfn_elines(theta: dict[str, Array]) -> Array:
                return (joint_loglike(model, keys, likelihoods, static_data, theta)
                        + prior.log_prob(theta))

            return lnprobfn_elines

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            predictions: dict[str, Array] = model.predict(theta)
            lnl = jnp.zeros(())

            for key, lhood in zip(keys, likelihoods):
                y_k, sig_k, mask_k, calib_k, ul_k = static_data[key]
                mu_k = predictions[key]
                if calib_k is not None:
                    mu_k = mu_k * calib_k
                if ul_k is not None:
                    lnl_k, _ = lhood(y_k, mu_k, sig_k, mask_k, params=theta, is_upper_limit=ul_k)
                else:
                    lnl_k, _ = lhood(y_k, mu_k, sig_k, mask_k, params=theta)
                lnl = lnl + lnl_k

            lnp = prior.log_prob(theta)
            return lnl + lnp

        return lnprobfn

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{k!r}: {lh!r}" for k, lh in zip(self.keys, self.likelihoods)
        )
        return f"MultiObservationLikelihood({{{pairs}}})"


jax.tree_util.register_pytree_node(
    MultiObservationLikelihood,
    flatten_func=lambda ml: (
        [],
        (ml.keys, ml.likelihoods),
    ),
    unflatten_func=lambda aux, _: MultiObservationLikelihood(
        keys=aux[0], likelihoods=aux[1]
    ),
)


def make_lnprobfn(
    observations : Any,
    model        : Any,
    prior        : Any,
    likelihood   : LikelihoodBase,
) -> Callable[[dict[str, Array]], Array]:
    """Return ``likelihood.make_lnprobfn(observations, model, prior)``."""
    return likelihood.make_lnprobfn(observations, model, prior)
