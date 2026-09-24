"""Gaussian log-likelihood kernels and classes."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .noise_model import DiagonalNoiseModel, NoiseModelBase, NoiseModelOutput
from .poly_calibration import PolynomialCalibration

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


def finite_data(y: Array, sigma: Array) -> tuple[Array, Array]:
    """``y`` and ``sigma`` with non-finite entries replaced by 0 and 1.  Observations mask
    those entries at construction, but a NaN left in a masked slot still reaches the
    gradient as 0 * NaN; every sampled likelihood reads its data through this."""
    y = jnp.asarray(y)
    sigma = jnp.asarray(sigma)
    return (jnp.where(jnp.isfinite(y), y, 0.0),
            jnp.where(jnp.isfinite(sigma), sigma, 1.0))


def observation_data(obs) -> tuple:
    """``(y - sky, sigma, mask, calibration, upper_limit)`` of ``obs`` as the sampled
    likelihood uses them (``upper_limit`` None unless some datum is flagged), through
    :func:`finite_data`."""
    y = obs.flux
    sky = getattr(obs, "sky", None)
    if sky is not None:
        y = y - sky
    y, sigma = finite_data(y, obs.uncertainty)
    ul = getattr(obs, "upper_limit", None)
    ul = None if ul is None or not bool(jnp.any(ul)) else jnp.asarray(ul, dtype=bool)
    return (y, sigma, obs.mask, getattr(obs, "calibration", None), ul)


def single_observation_data(observations, what: str, upper_limits: bool = False) -> tuple:
    """:func:`observation_data` for a single-observation ``make_lnprobfn``: sky subtracted,
    finite data; returns ``(y, sigma, mask, calibration, upper_limit)``.  An observation that
    flags upper limits is refused unless the likelihood handles them (``upper_limits``)."""
    y, sigma, mask, calib, ul = observation_data(observations)
    if ul is not None and not upper_limits:
        raise ValueError(
            f"{what}.make_lnprobfn: the observation flags upper limits, which this likelihood "
            "would treat as detections.  Use DiagonalGaussianLikelihoodWithUpperLimits, or "
            "MultiObservationLikelihood / fitSED, which pick the one-sided kernel")
    return y, sigma, mask, calib, ul


def calibrated_mu(mu, calibration):
    """The prediction times the observation's fixed ``calibration`` vector (None: unchanged)."""
    return mu if calibration is None else mu * calibration


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


@jax.custom_jvp
def _mixture(f: Array, a: Array, b: Array) -> Array:
    """``ln[(1 - f) e^a + f e^b]`` evaluated as ``logaddexp(log1p(-f) + a, log(f) + b)``."""
    return jnp.logaddexp(jnp.log1p(-f) + a, jnp.log(f) + b)


@_mixture.defjvp
def _mixture_jvp(primals, tangents):
    # Exact derivative.  d/df = (e^b - e^a) / L is finite at f = 0, where differentiating
    # the log(f) above gives 0 * inf = NaN; the values are those of the primal.
    f, a, b = primals
    df, da, db = tangents
    out = _mixture(f, a, b)
    w_good = jnp.exp(jnp.log1p(-f) + a - out)
    w_bad  = jnp.exp(jnp.log(f) + b - out)
    d_f    = jnp.exp(b - out) - jnp.exp(a - out)
    return out, d_f * df + w_good * da + w_bad * db


def _outlier_terms(y, mu, inv_var, log_det, mask, f, nsigma):
    """``(resid, chi, lnl_i)`` of the mixture; the residual is zeroed on masked data before
    any arithmetic, so masked (even non-finite) data give finite gradients."""
    resid = jnp.where(mask, y - mu, 0.0)
    chi   = resid * jnp.sqrt(inv_var)
    chi2  = chi ** 2
    lnp_good = -0.5 * chi2 - log_det
    lnp_bad  = -0.5 * chi2 / nsigma ** 2 - log_det - jnp.log(nsigma)
    return resid, chi, _mixture(f, lnp_good, lnp_bad)


def lnlike_diag_outlier(
    y       : Array,
    mu      : Array,
    inv_var : Array,
    log_det : Array,
    mask    : Array,
    f       : Array,
    nsigma  : Array,
) -> tuple[Array, LikelihoodOutput]:
    """Outlier mixture (Hogg, Bovy & Lang 2010), Prospector's ``NoiseModel.lnlike`` branch
    ``f_outlier > 0``: per datum

        lnl_i = logaddexp(log(1 - f) + lnp_good_i, log(f) + lnp_bad_i)
        lnp_good_i = -0.5 chi_i^2 - log_det_i
        lnp_bad_i  = -0.5 chi_i^2 / nsigma^2 - log_det_i - log(nsigma)

    with ``sigma_eff`` (``inv_var``, ``log_det``) from the noise model after every term, as
    Prospector uses its noise model's ``Sigma``.  As f -> 0 it tends to
    :func:`lnlike_diag_gaussian`, NOT to Prospector's ``f_outlier == 0`` value, whose
    ``NoiseModel.lnlikelihood`` multiplies chi^2 by ln(2 pi) and drops n ln(2 pi).
    ``lnl_pointwise`` holds the mixture terms (summing to ``lnl_total``); ``chi`` and
    ``residuals`` are those of the inlier Gaussian.  Masked data contribute 0 with finite
    gradients; the derivative in f is the exact ``(e^b - e^a)/L``, finite at f = 0 unless a
    datum sits beyond ~38 sigma (nsigma = 50) (then +inf: the likelihood rises that steeply)."""
    resid, chi, lnl_i = _outlier_terms(y, mu, inv_var, log_det, mask, f, nsigma)

    lnl_masked = jnp.where(mask, lnl_i, 0.0)
    lnl_total  = jnp.sum(lnl_masked)

    aux = LikelihoodOutput(
        lnl_total     = lnl_total,
        lnl_pointwise = lnl_masked,
        residuals     = resid,
        chi           = chi,
        ndof          = jnp.sum(mask),
    )
    return lnl_total, aux


def lnlike_diag_outlier_with_upper_limits(
    y               : Array,
    mu              : Array,
    inv_var         : Array,
    log_det         : Array,
    mask            : Array,
    is_upper_limit  : Array,
    f               : Array,
    nsigma          : Array,
) -> tuple[Array, LikelihoodOutput]:
    """:func:`lnlike_diag_outlier` on the detections and the one-sided penalty of
    :func:`lnlike_diag_gaussian_with_upper_limits` (unchanged, no mixture) on data flagged
    ``is_upper_limit``.  With no flags it equals :func:`lnlike_diag_outlier`."""
    resid, chi, lnl_mix = _outlier_terms(y, mu, inv_var, log_det, mask, f, nsigma)
    lnl_ul = -0.5 * jnp.maximum(-chi, 0.0) ** 2 - log_det
    lnl_i  = jnp.where(is_upper_limit, lnl_ul, lnl_mix)

    lnl_masked = jnp.where(mask, lnl_i, 0.0)
    lnl_total  = jnp.sum(lnl_masked)

    aux = LikelihoodOutput(
        lnl_total     = lnl_total,
        lnl_pointwise = lnl_masked,
        residuals     = resid,
        chi           = chi,
        ndof          = jnp.sum(mask),
    )
    return lnl_total, aux


def outlier_probability(
    y       : Array,
    mu      : Array,
    inv_var : Array,
    log_det : Array,
    mask    : Array,
    f       : Array,
    nsigma  : Array,
) -> Array:
    """Per-datum posterior probability of belonging to the outlier component,
    ``exp(log f + lnp_bad_i - lnl_i)``; 0 for masked data.  A diagnostic: the sampled
    likelihood never computes it."""
    resid = jnp.where(mask, y - mu, 0.0)
    chi2  = (resid * jnp.sqrt(inv_var)) ** 2
    lnp_good = -0.5 * chi2 - log_det
    lnp_bad  = -0.5 * chi2 / nsigma ** 2 - log_det - jnp.log(nsigma)
    a = jnp.log1p(-f) + lnp_good
    b = jnp.log(f) + lnp_bad
    return jnp.where(mask, jnp.exp(b - jnp.logaddexp(a, b)), 0.0)


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
    """Gaussian log-likelihood with an independent (diagonal) noise model; with
    ``noise_model.f_outlier`` set, the outlier mixture :func:`lnlike_diag_outlier`."""

    noise_model: DiagonalNoiseModel = field(
        default_factory=DiagonalNoiseModel
    )
    #: profiled polynomial calibration (Spectrum(polynomial_order > 0)); None = off
    poly_calibration: Optional[PolynomialCalibration] = None

    def _calibrated(self, y, mu, sigma_obs, mask, params):
        """``mu`` times the profiled calibration response (unchanged when off)."""
        if self.poly_calibration is None:
            return mu
        return self.poly_calibration.calibrate(y, mu, sigma_obs, mask, params,
                                               self.noise_model)

    def __call__(
        self,
        y         : Array,
        mu        : Array,
        sigma_obs : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """Return ``(lnl_total, LikelihoodOutput)``."""
        mu = self._calibrated(y, mu, sigma_obs, mask, params)
        noise_out: NoiseModelOutput = self.noise_model.compute(
            sigma_obs, mu, mask, params, data=y
        )
        if getattr(self.noise_model, "use_outlier", False):
            f, nsigma = self.noise_model.outlier_params(params)
            return lnlike_diag_outlier(
                y, mu, noise_out.inv_var, noise_out.log_det, mask, f, nsigma
            )
        return lnlike_diag_gaussian(
            y, mu, noise_out.inv_var, noise_out.log_det, mask
        )

    def outlier_probability(
        self,
        y         : Array,
        mu        : Array,
        sigma_obs : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> Array:
        """Per-datum probability of being an outlier (see :func:`outlier_probability`);
        raises when the mixture is off."""
        if not getattr(self.noise_model, "use_outlier", False):
            raise ValueError("the outlier mixture is off (noise_model.f_outlier is None)")
        noise_out = self.noise_model.compute(sigma_obs, mu, mask, params, data=y)
        f, nsigma = self.noise_model.outlier_params(params)
        return outlier_probability(
            y, mu, noise_out.inv_var, noise_out.log_det, mask, f, nsigma
        )

    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior for one observation (needs ``.flux``, ``.uncertainty``,
        ``.mask``; honours ``sky`` and ``calibration`` as MultiObservationLikelihood does, and
        refuses flagged upper limits)."""
        y, sigma_obs, mask, calib, _ = single_observation_data(
            observations, "DiagonalGaussianLikelihood")
        noise_model       = self.noise_model
        calibrate         = self._calibrated

        if getattr(noise_model, "use_outlier", False):
            @jax.jit
            def lnprobfn_outlier(theta: dict[str, Array]) -> Array:
                mu = calibrate(y, calibrated_mu(model.predict(theta), calib), sigma_obs, mask,
                               theta)
                noise_out = noise_model.compute(sigma_obs, mu, mask, theta, data=y)
                f, nsigma = noise_model.outlier_params(theta)
                lnl, _    = lnlike_diag_outlier(
                    y, mu, noise_out.inv_var, noise_out.log_det, mask, f, nsigma
                )
                return lnl + prior.log_prob(theta)

            return lnprobfn_outlier

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            mu = calibrate(y, calibrated_mu(model.predict(theta), calib), sigma_obs, mask,
                               theta)
            noise_out = noise_model.compute(sigma_obs, mu, mask, theta, data=y)
            lnl, _    = lnlike_diag_gaussian(
                y, mu, noise_out.inv_var, noise_out.log_det, mask
            )
            lnp = prior.log_prob(theta)
            return lnl + lnp

        return lnprobfn

    def __repr__(self) -> str:
        pc = "" if self.poly_calibration is None else f", poly_calibration={self.poly_calibration!r}"
        return f"DiagonalGaussianLikelihood(noise_model={self.noise_model!r}{pc})"


jax.tree_util.register_pytree_node(
    DiagonalGaussianLikelihood,
    flatten_func   = lambda lh: ([], (lh.noise_model, lh.poly_calibration)),
    unflatten_func = lambda aux, _: DiagonalGaussianLikelihood(
        noise_model=aux[0], poly_calibration=aux[1]
    ),
)


@dataclass(frozen=True)
class DiagonalGaussianLikelihoodWithUpperLimits(LikelihoodBase):
    """Diagonal Gaussian log-likelihood honouring per-datum upper-limit flags
    (one-sided penalty; reduces to :class:`DiagonalGaussianLikelihood` when no flags are set).
    With ``noise_model.f_outlier`` set, the detections use the outlier mixture and the limits
    keep the one-sided penalty (:func:`lnlike_diag_outlier_with_upper_limits`)."""

    noise_model: DiagonalNoiseModel = field(
        default_factory=DiagonalNoiseModel
    )
    #: profiled polynomial calibration (Spectrum(polynomial_order > 0)); None = off
    poly_calibration: Optional[PolynomialCalibration] = None

    def _calibrated(self, y, mu, sigma_obs, mask, params):
        """``mu`` times the profiled calibration response (unchanged when off)."""
        if self.poly_calibration is None:
            return mu
        return self.poly_calibration.calibrate(y, mu, sigma_obs, mask, params,
                                               self.noise_model)

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
        mu = self._calibrated(y, mu, sigma_obs, mask, params)
        noise_out: NoiseModelOutput = self.noise_model.compute(
            sigma_obs, mu, mask, params, data=y
        )
        if is_upper_limit is None:
            is_upper_limit = jnp.zeros_like(mask, dtype=bool)
        if getattr(self.noise_model, "use_outlier", False):
            f, nsigma = self.noise_model.outlier_params(params)
            return lnlike_diag_outlier_with_upper_limits(
                y, mu, noise_out.inv_var, noise_out.log_det, mask, is_upper_limit, f, nsigma,
            )
        return lnlike_diag_gaussian_with_upper_limits(
            y, mu, noise_out.inv_var, noise_out.log_det, mask, is_upper_limit,
        )

    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """Return a jitted log-posterior using ``observations.upper_limit`` (all-False if absent);
        honours ``sky`` and ``calibration`` as MultiObservationLikelihood does."""
        y, sigma_obs, mask, calib, _ = single_observation_data(
            observations, "DiagonalGaussianLikelihoodWithUpperLimits", upper_limits=True)
        is_ul = getattr(observations, "upper_limit", None)
        if is_ul is None:
            is_ul = jnp.zeros_like(mask, dtype=bool)
        else:
            is_ul = jnp.asarray(is_ul, dtype=bool)
        noise_model = self.noise_model
        calibrate   = self._calibrated

        if getattr(noise_model, "use_outlier", False):
            @jax.jit
            def lnprobfn_outlier(theta: dict[str, Array]) -> Array:
                mu = calibrate(y, calibrated_mu(model.predict(theta), calib), sigma_obs, mask,
                               theta)
                noise_out = noise_model.compute(sigma_obs, mu, mask, theta, data=y)
                f, nsigma = noise_model.outlier_params(theta)
                lnl, _    = lnlike_diag_outlier_with_upper_limits(
                    y, mu, noise_out.inv_var, noise_out.log_det, mask, is_ul, f, nsigma,
                )
                return lnl + prior.log_prob(theta)

            return lnprobfn_outlier

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            mu = calibrate(y, calibrated_mu(model.predict(theta), calib), sigma_obs, mask,
                               theta)
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
            f"noise_model={self.noise_model!r}"
            + ("" if self.poly_calibration is None
               else f", poly_calibration={self.poly_calibration!r}") + ")"
        )


jax.tree_util.register_pytree_node(
    DiagonalGaussianLikelihoodWithUpperLimits,
    flatten_func   = lambda lh: ([], (lh.noise_model, lh.poly_calibration)),
    unflatten_func = lambda aux, _: DiagonalGaussianLikelihoodWithUpperLimits(
        noise_model=aux[0], poly_calibration=aux[1]
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
        static_data = {key: observation_data(observations[key]) for key in self.keys}
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
