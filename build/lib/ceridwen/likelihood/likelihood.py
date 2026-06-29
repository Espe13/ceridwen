"""
ceridwen/likelihood/likelihood.py
====================================
Gaussian log-likelihood classes for Ceridwen SED fitting.

Architecture overview
---------------------
The system has three layers:

1. **Pure kernel** ``lnlike_diag_gaussian``
   A single pure JAX function.  Takes pre-computed ``inv_var`` and ``log_det``
   arrays from a noise model and returns a scalar log-likelihood plus a
   ``LikelihoodOutput`` NamedTuple carrying diagnostics.  No class, no state,
   no side effects.  This is the hot path -- everything else is scaffolding
   around it.

2. **Likelihood class** ``DiagonalGaussianLikelihood``
   A frozen dataclass that holds a ``NoiseModelBase`` instance and sequences
   the call: noise model → kernel → output.  Its ``__call__`` method has the
   signature required by ``jax.value_and_grad(has_aux=True)``, returning
   ``(lnl_total, LikelihoodOutput)``.

3. **Multi-observation wrapper** ``MultiObservationLikelihood``
   Combines an arbitrary collection of ``(observation_key, likelihood)`` pairs
   into a single log-likelihood, summed across photometry, spectroscopy, lines,
   etc.  Provides the same ``make_lnprobfn`` factory method.

Key JAX design decisions
------------------------
**Frozen dataclasses + empty PyTree leaves**
    All configuration (noise model type, flags) is Python-level.  Classes are
    registered as JAX PyTrees with empty leaves, so they act as static compile-
    time constants when closed over in ``jax.jit``.  Only ``theta`` is traced.

**``(scalar, aux)`` return convention**
    ``DiagonalGaussianLikelihood.__call__`` returns ``(lnl_total, aux)``.
    This matches the ``jax.value_and_grad(fn, has_aux=True)`` pattern exactly::

        (lnl, aux), grad = jax.value_and_grad(
            lambda t: lhood(y, model(t), sigma, mask, t),
            has_aux=True
        )(theta)

    Gradients of ``lnl_total`` w.r.t. ``theta`` (including noise nuisance
    parameters) come out of one backward pass, with diagnostics in ``aux``.

**``jnp.where`` masking everywhere**
    Masks are applied with ``jnp.where`` rather than Python boolean indexing.
    This preserves array shapes across the JIT boundary and keeps gradients
    well-defined for all elements (masked elements simply contribute 0).

**No Python control flow over traced values**
    The Python ``for`` loop in ``MultiObservationLikelihood`` iterates over
    *static* keys -- it is unrolled at trace time.  No ``jax.lax.cond`` is
    needed here.

**``make_lnprobfn`` factory**
    Closes over the static objects (observations, model, prior, likelihood) and
    returns a ``@jax.jit``-compiled function of ``theta`` only.  This is the
    interface blackjax and other gradient-based samplers expect.

Comparison to Prospector
------------------------
Prospector's ``compute_lnlike`` dispatches through a mutable noise model
object attached to each observation, calls numpy, and returns a scalar.
The design here differs on every dimension relevant to JAX:

    +-------------------------+---------------------+----------------------------+
    | Aspect                  | Prospector          | Ceridwen                   |
    +=========================+=====================+============================+
    | Noise model state       | Mutable attributes  | Frozen dataclass           |
    | Array library           | numpy / scipy       | JAX (differentiable)       |
    | JIT-compilable          | No                  | Yes                        |
    | Returns gradient        | No                  | Yes (value_and_grad)       |
    | Masking                 | Boolean indexing    | jnp.where (shape-stable)   |
    | Multi-modality          | Separate functions  | MultiObservationLikelihood |
    | Nuisance parameters     | Stored in model     | Passed through theta       |
    +-------------------------+---------------------+----------------------------+

What is kept from Prospector's philosophy:
  - Separating noise model computation from the likelihood kernel.
  - Supporting per-datum contributions and normalised residuals.
  - Treating masking as a first-class concern.
  - Making the interface observation-centric (obs carries data + sigma + mask).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .noise_model import DiagonalNoiseModel, NoiseModelBase, NoiseModelOutput

# Type alias matching the ceridwen codebase convention.
Array = jax.Array


# ===========================================================================
# Diagnostic output container
# ===========================================================================

class LikelihoodOutput:
    """
    Per-datum diagnostics returned alongside the scalar log-likelihood.

    Returned as the ``aux`` element of ``(lnl_total, aux)`` so that
    ``jax.value_and_grad(fn, has_aux=True)`` delivers gradients *and*
    diagnostics from a single backward pass.

    Attributes
    ----------
    lnl_total : Array, scalar
        Total log-likelihood (same as the first return value).  Stored here
        for convenience when calling without ``value_and_grad``.
    lnl_pointwise : Array, shape (n_data,)
        Per-datum log-likelihood contributions.  Zero for masked data.
    residuals : Array, shape (n_data,)
        Raw residuals ``y - mu``.  Zero for masked data.
    chi : Array, shape (n_data,)
        Normalised residuals ``(y - mu) / sigma_eff``.  Zero for masked data.
        Useful for visualising fit quality; chi^2 / ndof = sum(chi**2) / ndof.
    ndof : Array, scalar int
        Number of unmasked (active) data points.

    Notes
    -----
    This class is registered as a JAX PyTree so it survives ``jax.jit`` and
    ``jax.vmap`` boundaries.  All array fields are JAX arrays; ``ndof`` is an
    integer-dtype JAX scalar.
    """

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


# ===========================================================================
# Pure kernel  --  the hot path
# ===========================================================================

def lnlike_diag_gaussian(
    y       : Array,
    mu      : Array,
    inv_var : Array,
    log_det : Array,
    mask    : Array,
) -> tuple[Array, LikelihoodOutput]:
    """
    Diagonal Gaussian log-likelihood kernel.

    This is a **pure JAX function** with no side effects.  It is safe inside
    ``jax.jit``, ``jax.grad``, and ``jax.vmap``.  Call it directly when you
    have already computed ``inv_var`` and ``log_det`` from a noise model.

    The per-datum log-likelihood is:

    .. math::

        \\ell_i = -\\tfrac{1}{2}\\chi_i^2 - \\texttt{log\\_det}_i

    where

    .. math::

        \\chi_i = (y_i - \\mu_i) \\sqrt{\\sigma_{\\rm eff,i}^{-2}}

    and ``log_det_i = 0.5 * log(2 pi sigma_eff_i^2)`` is supplied by the noise
    model.  Masked data points contribute 0 to every sum.

    Parameters
    ----------
    y : Array, shape (n_data,)
        Observed data vector.
    mu : Array, shape (n_data,)
        Model prediction.
    inv_var : Array, shape (n_data,)
        Effective inverse variance ``1 / sigma_eff^2``, from ``NoiseModelOutput``.
    log_det : Array, shape (n_data,)
        Per-datum log-normalisation ``0.5 * log(2 pi sigma_eff^2)``, from
        ``NoiseModelOutput``.
    mask : Array of bool, shape (n_data,)
        True for data points included in the likelihood.

    Returns
    -------
    lnl_total : Array, scalar
        Total log-likelihood (sum over unmasked data).
    aux : LikelihoodOutput
        Per-datum diagnostics (pointwise contributions, residuals, chi,
        ndof).  Compatible with ``jax.value_and_grad(has_aux=True)``.

    Notes
    -----
    Numerical stability: squaring the residual before multiplying by
    ``inv_var`` (rather than dividing by ``var``) avoids potential overflow
    when ``var`` is very small.  The ``jnp.sqrt(inv_var)`` path is safe
    because the noise model guarantees ``inv_var > 0`` for all elements.
    """
    # Residual and normalised residual (chi).
    # Writing chi = resid * sqrt(inv_var) keeps gradient paths symmetric and
    # avoids catastrophic cancellation that could occur with resid / sigma.
    resid = y - mu
    chi   = resid * jnp.sqrt(inv_var)    # (y - mu) / sigma_eff

    # Per-datum log-likelihood (unmasked).
    lnl_i = -0.5 * chi ** 2 - log_det

    # Zero out masked contributions.  Using jnp.where preserves array shapes
    # and keeps gradients well-defined (gradient through masked elements is 0).
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


# ===========================================================================
# Pure kernel  --  diagonal Gaussian with per-datum upper-limit support
# ===========================================================================

def lnlike_diag_gaussian_with_upper_limits(
    y               : Array,
    mu              : Array,
    inv_var         : Array,
    log_det         : Array,
    mask            : Array,
    is_upper_limit  : Array,
) -> tuple[Array, LikelihoodOutput]:
    r"""
    Diagonal Gaussian log-likelihood with per-datum upper-limit support.

    Photometric or line-flux data points flagged as upper limits do not
    constrain the model from below: only model predictions that *exceed*
    the observed upper-limit value pay a chi-squared penalty.  This matches
    Prospector's recommended treatment for non-detections (see
    :ref:`prospector FAQ on upper limits
    <https://prospect.readthedocs.io/en/latest/faq.html#what-do-i-do-about-upper-limits>`)
    and the existing convention used in
    :class:`ceridwen.observation.Lines` for emission-line non-detections.

    Per-datum log-likelihood
    ------------------------
    Let

    .. math::

        \chi_i = (y_i - \mu_i) \,/\, \sigma_{\mathrm{eff},i}.

    For a **detection** (``is_upper_limit[i] = False``):

    .. math::

        \ell_i = -\tfrac{1}{2}\chi_i^2 \;-\; \log\!\det_i

    For an **upper limit** (``is_upper_limit[i] = True``):

    .. math::

        \ell_i^{\mathrm{UL}}
            = -\tfrac{1}{2}\bigl[\max(\mu_i - y_i,\,0)\bigr]^2 \,
                \sigma_{\mathrm{eff},i}^{-2} \;-\; \log\!\det_i.

    The :math:`\max(\mu_i - y_i, 0)` clip means that any model prediction
    consistent with -- or below -- the observed upper limit incurs no penalty.
    The log-determinant is retained on both branches so the absolute
    log-likelihood remains a proper Gaussian normalisation (NS log Z stays
    interpretable); ``log_det`` does not depend on theta and contributes a
    fixed offset to model-comparison statistics.

    Parameters
    ----------
    y : Array, shape (n_data,)
        Observed data.  For detections this is the measured flux; for upper
        limits this is the upper-limit value (e.g. 1-sigma, 2-sigma, etc.;
        the convention is whatever the user used).
    mu : Array, shape (n_data,)
        Model prediction.
    inv_var : Array, shape (n_data,)
        Effective inverse variance ``1 / sigma_eff^2`` from the noise model.
    log_det : Array, shape (n_data,)
        Per-datum log-normalisation ``0.5 * log(2 pi sigma_eff^2)``.
    mask : Array of bool, shape (n_data,)
        ``True`` for data points (detections OR upper limits) that contribute
        to the likelihood.  ``False`` removes the datum entirely.
    is_upper_limit : Array of bool, shape (n_data,)
        ``True`` marks band ``i`` as an upper limit (one-sided chi-squared);
        ``False`` treats it as a normal detection (two-sided Gaussian).

    Returns
    -------
    lnl_total : Array, scalar
        Total log-likelihood (sum over unmasked data).
    aux : LikelihoodOutput
        Per-datum diagnostics.  ``aux.chi`` is set to the *signed* normalised
        residual on both branches; ``aux.lnl_pointwise`` reflects the
        one-sided clip on upper-limit bands so the chi^2 reported by
        downstream tooling matches the actual likelihood contribution.

    Notes
    -----
    The kernel is pure JAX, JIT-safe, vmap-safe, and differentiable in
    ``mu``, ``inv_var``, and ``log_det``.  The :math:`\max(\cdot, 0)` clip is
    implemented with :func:`jnp.maximum`, which has a defined sub-gradient
    at zero (it returns 0) so HMC / NUTS / SVI gradient paths remain valid.
    """
    resid     = y - mu                                # (y - mu)
    sqrt_iv   = jnp.sqrt(inv_var)
    chi       = resid * sqrt_iv                       # signed normalised residual

    # Detection branch: full chi^2.
    chi2_det  = chi ** 2

    # Upper-limit branch: penalty only when mu > y, i.e. resid < 0.
    # Equivalent to (max(mu - y, 0) / sigma_eff)^2 == max(-chi, 0)^2.
    chi2_ul   = jnp.maximum(-chi, 0.0) ** 2

    chi2_used = jnp.where(is_upper_limit, chi2_ul, chi2_det)
    lnl_i     = -0.5 * chi2_used - log_det

    # Zero-out masked contributions (preserves shapes for jnp.where).
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


# ===========================================================================
# Abstract base class
# ===========================================================================

class LikelihoodBase(abc.ABC):
    """
    Abstract base for all Ceridwen likelihood classes.

    Every subclass must implement ``__call__`` with the signature below.
    This ensures all likelihood classes are interchangeable in
    ``MultiObservationLikelihood`` and ``make_lnprobfn``.
    """

    @abc.abstractmethod
    def __call__(
        self,
        y         : Array,
        mu        : Array,
        sigma_obs : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """
        Evaluate the log-likelihood.

        Parameters
        ----------
        y : Array, shape (n_data,)
            Observed data vector.
        mu : Array, shape (n_data,)
            Model prediction (differentiable w.r.t. theta).
        sigma_obs : Array, shape (n_data,)
            Observational 1-sigma uncertainties.
        mask : Array of bool, shape (n_data,)
            True where data points are used.
        params : dict[str, Array], optional
            Sampled nuisance parameters from theta (e.g. jitter, f_calib).

        Returns
        -------
        lnl_total : Array, scalar
            Differentiable total log-likelihood.
        aux : LikelihoodOutput
        """

    @abc.abstractmethod
    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """
        Build a JIT-compiled log-posterior function for use with samplers.

        Parameters
        ----------
        observations : Observation or dict[str, Observation]
        model : object with a ``predict(theta)`` method
        prior : object with a ``log_prob(theta)`` method

        Returns
        -------
        Callable[[theta], Array]
            A ``@jax.jit``-compiled function mapping a parameter PyTree to the
            scalar log-posterior.  Suitable for blackjax NUTS/HMC/MCLMC.
        """


# ===========================================================================
# Diagonal Gaussian likelihood
# ===========================================================================

@dataclass(frozen=True)
class DiagonalGaussianLikelihood(LikelihoodBase):
    """
    Gaussian log-likelihood with an independent (diagonal) noise model.

    This class sequences three operations:

    1. Call ``noise_model.compute(sigma_obs, mu, mask, params)`` to obtain
       per-datum ``inv_var`` and ``log_det``.
    2. Call ``lnlike_diag_gaussian(y, mu, inv_var, log_det, mask)`` to get
       the scalar log-likelihood and diagnostics.

    Because both steps are pure JAX functions, the entire ``__call__`` is
    JIT-compilable and differentiable end-to-end.

    Parameters
    ----------
    noise_model : DiagonalNoiseModel
        Noise model instance.  Defaults to a plain observational-uncertainty-
        only model (no jitter, no fractional error).

    Examples
    --------
    Basic usage with no nuisance parameters::

        lhood = DiagonalGaussianLikelihood()
        lnl, aux = lhood(y, mu, sigma_obs, mask)
        # aux.chi      -- normalised residuals
        # aux.lnl_pointwise  -- per-datum contributions

    With jitter as a sampled parameter::

        lhood = DiagonalGaussianLikelihood(
            noise_model=DiagonalNoiseModel(use_jitter=True)
        )
        lnl, aux = lhood(y, mu, sigma_obs, mask,
                         params={"log_jitter": theta["log_jitter"]})

    Gradient of log-likelihood w.r.t. theta (for HMC/gradient-based samplers)::

        def lnl_fn(theta):
            mu = model.predict(theta)
            return lhood(y, mu, sigma_obs, mask, params=theta)[0]

        grad = jax.grad(lnl_fn)(theta)

    Using ``has_aux=True`` to get diagnostics and gradient in one pass::

        (lnl, aux), grad = jax.value_and_grad(
            lambda t: lhood(y, model.predict(t), sigma_obs, mask, t),
            has_aux=True,
        )(theta)
    """

    noise_model: DiagonalNoiseModel = field(
        default_factory=DiagonalNoiseModel
    )

    # ------------------------------------------------------------------
    def __call__(
        self,
        y         : Array,
        mu        : Array,
        sigma_obs : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """
        Evaluate log-likelihood and return diagnostics.

        This method is the intended hot path for sampling.  It is safe inside
        ``jax.jit``, ``jax.grad``, and ``jax.vmap``.

        Parameters
        ----------
        y : Array, shape (n_data,)
        mu : Array, shape (n_data,)
        sigma_obs : Array, shape (n_data,)
        mask : Array of bool, shape (n_data,)
        params : dict, optional
            Nuisance parameters expected by the noise model.

        Returns
        -------
        lnl_total : Array, scalar
        aux : LikelihoodOutput
        """
        noise_out: NoiseModelOutput = self.noise_model.compute(
            sigma_obs, mu, mask, params
        )
        return lnlike_diag_gaussian(
            y, mu, noise_out.inv_var, noise_out.log_det, mask
        )

    # ------------------------------------------------------------------
    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """
        Build a JIT-compiled log-posterior for a single observation type.

        ``observations`` must have attributes ``.flux``, ``.uncertainty``, and
        ``.mask`` (the interface defined by ``ceridwen.observation.Observation``).

        The returned function has the exact signature expected by blackjax::

            kernel = blackjax.nuts(lnprobfn, step_size)
            state  = kernel.init(initial_theta)

        Parameters
        ----------
        observations : Observation
            A single observation object (photometry *or* spectroscopy *or*
            lines).  For multiple modalities use ``MultiObservationLikelihood``.
        model : object
            Must implement ``predict(theta) -> Array``.
        prior : object
            Must implement ``log_prob(theta) -> Array``.

        Returns
        -------
        Callable[[theta], Array]
            JIT-compiled log-posterior.

        Notes
        -----
        ``observations``, ``model``, ``prior``, and ``self`` are all closed
        over at factory time and become compile-time constants.  Only ``theta``
        is traced.  This means the first call incurs a one-time XLA compilation
        cost; subsequent calls are fully compiled and have minimal Python
        overhead.
        """
        # Hoist static data out of the closure so JAX sees them as constants.
        y         : Array = observations.flux
        sigma_obs : Array = observations.uncertainty
        mask      : Array = observations.mask
        noise_model       = self.noise_model

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            mu = model.predict(theta)
            # Pass the full theta as params.  The noise model extracts only
            # the keys it needs (log_jitter, log_f_calib); all other keys are
            # ignored.  This avoids conditional dict construction inside the
            # JIT-compiled hot path and is safe for any noise model subclass.
            noise_out = noise_model.compute(sigma_obs, mu, mask, theta)
            lnl, _    = lnlike_diag_gaussian(
                y, mu, noise_out.inv_var, noise_out.log_det, mask
            )
            lnp = prior.log_prob(theta)
            return lnl + lnp

        return lnprobfn

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"DiagonalGaussianLikelihood(noise_model={self.noise_model!r})"


# PyTree: no dynamic leaves; the frozen dataclass config is all auxiliary.
jax.tree_util.register_pytree_node(
    DiagonalGaussianLikelihood,
    flatten_func   = lambda lh: ([], (lh.noise_model,)),
    unflatten_func = lambda aux, _: DiagonalGaussianLikelihood(
        noise_model=aux[0]
    ),
)


# ===========================================================================
# Diagonal Gaussian likelihood with per-datum upper-limit support
# ===========================================================================

@dataclass(frozen=True)
class DiagonalGaussianLikelihoodWithUpperLimits(LikelihoodBase):
    r"""
    Diagonal Gaussian log-likelihood that honours per-datum upper-limit flags.

    Identical in interface to :class:`DiagonalGaussianLikelihood`, except that
    ``__call__`` and ``make_lnprobfn`` additionally consume an
    ``is_upper_limit`` boolean array that marks each datum as either a normal
    detection (two-sided Gaussian) or an upper limit (one-sided chi-squared,
    no penalty when ``mu_i <= y_i``).

    The likelihood kernel used is
    :func:`lnlike_diag_gaussian_with_upper_limits` -- see that function for
    the per-datum mathematical form and gradient behaviour.

    When ``is_upper_limit`` is ``None`` or all-``False``, this class reduces
    bit-for-bit to :class:`DiagonalGaussianLikelihood`.  The intended use is
    therefore drop-in replacement: callers who do not have non-detections pay
    no performance penalty (the ``jnp.where`` on a constant mask is folded
    away at trace time).

    Parameters
    ----------
    noise_model : DiagonalNoiseModel
        Noise model instance (default: bare observational sigma, no nuisance
        parameters).  Identical role to :class:`DiagonalGaussianLikelihood`.

    Examples
    --------
    Construct an upper-limit-aware photometric likelihood and evaluate it:

    >>> from ceridwen.likelihood import (
    ...     DiagonalGaussianLikelihoodWithUpperLimits,
    ...     DiagonalNoiseModel,
    ... )
    >>> lhood = DiagonalGaussianLikelihoodWithUpperLimits()
    >>> # Three bands; the third is a 1-sigma upper limit at 5.0 nJy.
    >>> y          = jnp.array([12.3,  8.1,  5.0])
    >>> sigma_obs  = jnp.array([ 0.5,  0.4,  5.0])
    >>> mu         = jnp.array([12.1,  8.4,  3.2])   # model below the limit
    >>> mask       = jnp.array([True, True, True])
    >>> is_ul      = jnp.array([False, False, True])
    >>> lnl, aux   = lhood(y, mu, sigma_obs, mask, is_upper_limit=is_ul)

    The third datum contributes zero penalty because ``mu < y``; switch the
    third element of ``mu`` to ``7.0`` (exceeds the upper limit) and the
    contribution becomes
    ``-0.5 * ((7.0 - 5.0) / 5.0)^2 - log_det``.
    """

    noise_model: DiagonalNoiseModel = field(
        default_factory=DiagonalNoiseModel
    )

    # ------------------------------------------------------------------
    def __call__(
        self,
        y               : Array,
        mu              : Array,
        sigma_obs       : Array,
        mask            : Array,
        params          : Optional[dict[str, Array]] = None,
        is_upper_limit  : Optional[Array]            = None,
    ) -> tuple[Array, LikelihoodOutput]:
        """
        Evaluate the upper-limit-aware Gaussian log-likelihood.

        Parameters
        ----------
        y, mu, sigma_obs, mask : Array
            See :class:`DiagonalGaussianLikelihood.__call__`.
        params : dict, optional
            Noise-model nuisance parameters (passed through to
            ``noise_model.compute``).
        is_upper_limit : Array of bool, shape (n_data,), optional
            Per-datum upper-limit flags.  ``None`` is equivalent to an
            all-``False`` array (every datum is a detection).

        Returns
        -------
        lnl_total : Array, scalar
        aux : LikelihoodOutput
        """
        noise_out: NoiseModelOutput = self.noise_model.compute(
            sigma_obs, mu, mask, params
        )
        if is_upper_limit is None:
            is_upper_limit = jnp.zeros_like(mask, dtype=bool)
        return lnlike_diag_gaussian_with_upper_limits(
            y, mu, noise_out.inv_var, noise_out.log_det, mask, is_upper_limit,
        )

    # ------------------------------------------------------------------
    def make_lnprobfn(
        self,
        observations : Any,
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """
        Build a JIT-compiled log-posterior that consumes the observation's
        ``upper_limit`` array (if present) and applies one-sided chi-squared
        to flagged data points.

        Falls back to an all-``False`` array of the appropriate shape when
        ``observations.upper_limit`` is ``None`` or missing -- making this
        method a drop-in replacement for the standard
        :class:`DiagonalGaussianLikelihood.make_lnprobfn`.

        Parameters
        ----------
        observations : Observation
            Single observation with ``.flux``, ``.uncertainty``, ``.mask``,
            and optionally ``.upper_limit``.
        model, prior : object
            See :class:`DiagonalGaussianLikelihood.make_lnprobfn`.
        """
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
            noise_out = noise_model.compute(sigma_obs, mu, mask, theta)
            lnl, _    = lnlike_diag_gaussian_with_upper_limits(
                y, mu, noise_out.inv_var, noise_out.log_det, mask, is_ul,
            )
            lnp = prior.log_prob(theta)
            return lnl + lnp

        return lnprobfn

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DiagonalGaussianLikelihoodWithUpperLimits("
            f"noise_model={self.noise_model!r})"
        )


# PyTree: identical structure to DiagonalGaussianLikelihood.
jax.tree_util.register_pytree_node(
    DiagonalGaussianLikelihoodWithUpperLimits,
    flatten_func   = lambda lh: ([], (lh.noise_model,)),
    unflatten_func = lambda aux, _: DiagonalGaussianLikelihoodWithUpperLimits(
        noise_model=aux[0]
    ),
)


# ===========================================================================
# Multi-observation likelihood
# ===========================================================================

@dataclass(frozen=True)
class MultiObservationLikelihood(LikelihoodBase):
    """
    Combine log-likelihoods across multiple independent data modalities.

    In SED fitting, the posterior typically conditions on several observation
    types simultaneously -- broadband photometry (in maggies), a spectrum (in
    F_lambda), and emission-line fluxes.  Each has different units, different
    noise properties, and potentially different noise model nuisance parameters.

    This class owns a **static** mapping from string keys to likelihood
    objects.  At trace time the Python loop over keys is unrolled by XLA;
    there is no runtime dispatch overhead.

    Parameters
    ----------
    keys : tuple of str
        Ordered observation keys, e.g. ``("phot", "spec", "lines")``.
    likelihoods : tuple of LikelihoodBase
        One likelihood per key, in the same order.  Different elements may
        have different noise models (e.g. photometry with fractional error,
        spectroscopy with jitter).

    Examples
    --------
    Combine photometry and spectroscopy::

        multi = MultiObservationLikelihood(
            keys=("phot", "spec"),
            likelihoods=(
                DiagonalGaussianLikelihood(
                    DiagonalNoiseModel(use_fractional=True)
                ),
                DiagonalGaussianLikelihood(
                    DiagonalNoiseModel(use_jitter=True)
                ),
            ),
        )
        lnprobfn = multi.make_lnprobfn(observations, model, prior)

    where ``observations`` is a dict ``{"phot": phot_obs, "spec": spec_obs}``
    and ``model.predict(theta)`` returns a dict with the same keys.
    """

    keys        : tuple[str, ...]          = field(default_factory=tuple)
    likelihoods : tuple[LikelihoodBase, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    def __call__(
        self,
        y         : dict[str, Array],
        mu        : dict[str, Array],
        sigma_obs : dict[str, Array],
        mask      : dict[str, Array],
        params    : Optional[dict[str, Array]] = None,
    ) -> tuple[Array, dict[str, LikelihoodOutput]]:
        """
        Evaluate total log-likelihood across all modalities.

        Parameters
        ----------
        y, mu, sigma_obs, mask : dict[str, Array]
            Per-modality data arrays, keyed by the same strings as ``self.keys``.
        params : dict[str, Array], optional
            All nuisance parameters (shared across modalities).  Each
            likelihood/noise model extracts only the keys it needs.

        Returns
        -------
        lnl_total : Array, scalar
            Sum of log-likelihoods across all modalities.
        aux : dict[str, LikelihoodOutput]
            Per-modality diagnostics.
        """
        lnl_total = jnp.zeros(())
        aux: dict[str, LikelihoodOutput] = {}
        for key, lhood in zip(self.keys, self.likelihoods):
            lnl_i, aux_i = lhood(y[key], mu[key], sigma_obs[key], mask[key], params)
            lnl_total    = lnl_total + lnl_i
            aux[key]     = aux_i
        return lnl_total, aux

    # ------------------------------------------------------------------
    def make_lnprobfn(
        self,
        observations : dict[str, Any],
        model        : Any,
        prior        : Any,
    ) -> Callable[[dict[str, Array]], Array]:
        """
        Build a JIT-compiled log-posterior for multiple observation types.

        Parameters
        ----------
        observations : dict[str, Observation]
            Keyed by the same strings as ``self.keys``.  Each value must have
            ``.flux``, ``.uncertainty``, and ``.mask`` attributes.
        model : object
            Must implement ``predict(theta) -> dict[str, Array]``.
        prior : object
            Must implement ``log_prob(theta) -> Array``.

        Returns
        -------
        Callable[[theta], Array]
            JIT-compiled log-posterior, suitable for blackjax.
        """
        # Pre-extract static data arrays from observations to avoid attribute
        # lookups inside the JIT-compiled hot path.
        static_data: dict[str, tuple[Array, Array, Array]] = {
            key: (
                observations[key].flux,
                observations[key].uncertainty,
                observations[key].mask,
            )
            for key in self.keys
        }
        keys        = self.keys
        likelihoods = self.likelihoods

        @jax.jit
        def lnprobfn(theta: dict[str, Array]) -> Array:
            predictions: dict[str, Array] = model.predict(theta)
            lnl = jnp.zeros(())

            for key, lhood in zip(keys, likelihoods):
                y_k, sig_k, mask_k = static_data[key]
                mu_k = predictions[key]
                # Delegate entirely to each likelihood's __call__, which
                # handles noise model dispatch internally.  Pass the full
                # theta as params; each noise model silently ignores keys
                # it does not recognise.  This respects the LikelihoodBase
                # interface and works for any future subclass.
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


# PyTree: no leaves; keys and likelihoods go into auxiliary (static).
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


# ===========================================================================
# Standalone factory (convenience)
# ===========================================================================

def make_lnprobfn(
    observations : Any,
    model        : Any,
    prior        : Any,
    likelihood   : LikelihoodBase,
) -> Callable[[dict[str, Array]], Array]:
    """
    Convenience wrapper: build a JIT-compiled log-posterior function.

    This is a module-level function that delegates to ``likelihood.make_lnprobfn``.
    Prefer calling ``likelihood.make_lnprobfn(...)`` directly when you have a
    ``LikelihoodBase`` instance; use this function when you want a functional-
    style call without constructing the class explicitly.

    Parameters
    ----------
    observations : Observation or dict[str, Observation]
    model : object with ``predict(theta)``
    prior : object with ``log_prob(theta)``
    likelihood : LikelihoodBase

    Returns
    -------
    Callable[[theta], Array]

    Example
    -------
    ::

        lnprobfn = make_lnprobfn(obs, model, prior,
                                  DiagonalGaussianLikelihood())
        blackjax.nuts(lnprobfn, step_size).init(theta0)
    """
    return likelihood.make_lnprobfn(observations, model, prior)
