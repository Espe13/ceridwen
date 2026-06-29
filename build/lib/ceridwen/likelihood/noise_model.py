"""
ceridwen/likelihood/noise_model.py
====================================
Noise models for the Ceridwen likelihood system.

Design philosophy
-----------------
A noise model is a **pure, stateless transformation**:

    (sigma_obs, mu, mask, nuisance_params) --> NoiseModelOutput(inv_var, log_det)

``inv_var`` and ``log_det`` are the two per-datum quantities consumed by any
Gaussian likelihood kernel:

    lnl_i = -0.5 * chi_i^2 - log_det_i
          = -0.5 * (y_i - mu_i)^2 * inv_var_i  -  0.5 * log(2 pi sigma_eff_i^2)

Separating these two quantities from the likelihood kernel means:
  - The kernel is a fixed, simple dot-product regardless of the noise model.
  - Heteroscedastic, jitter-inflated, or correlated noise all plug in through
    the same interface without touching the kernel.
  - Both quantities are differentiable w.r.t. any nuisance parameter in theta,
    so gradient-based samplers (HMC/NUTS) see smooth gradients through the
    noise model without special casing.

JAX design rules applied here
------------------------------
1. **Frozen dataclasses**: all config is Python-level (bools, floats).  No JAX
   arrays are stored in the class, so the class is fully static from JAX's
   perspective and never triggers retracing when closed over in jax.jit.

2. **PyTree registration with empty leaves**: each class is registered as a
   JAX PyTree with no dynamic leaves.  This allows the noise model to appear
   inside JIT-compiled functions, vmap, and grad transformations safely.

3. **Nuisance parameters live in theta**: sampled noise parameters (jitter,
   calibration errors) are *never* stored in the noise model instance.  They
   are passed at call time via ``params``, which is typically a sub-dict of the
   full parameter dict ``theta``.  This is essential for HMC/NUTS, which must
   be able to differentiate through the noise model w.r.t. those parameters.

4. **Log-space nuisance parameters**: jitter and fractional errors are passed
   as ``log_jitter`` and ``log_f_calib`` in ``params`` and exponentiated
   inside ``compute``.  This guarantees positivity without requiring constrained
   samplers, and gives better geometry for gradient-based methods.

Extension pattern
-----------------
Subclass ``NoiseModelBase`` and override ``compute``.  The only contract is
that ``compute`` is a pure JAX function returning a ``NoiseModelOutput``.
Examples of future subclasses::

    CorrelatedNoiseModel     -- full covariance matrix via GP kernel
    HeteroscedasticNoiseModel -- per-datum variance from a parametric model
    MixtureNoiseModel        -- Gaussian + outlier mixture

Comparison to Prospector
------------------------
Prospector's ``NoiseModel`` is a **mutable object** that stores intermediate
quantities (``self.Sigma``, ``self.log_det``) as instance attributes and calls
``numpy`` and ``scipy``.  It is not JIT-compilable, not differentiable, and
cannot be used inside ``jax.vmap``.

Here we make the noise model a **frozen, stateless function object**:
no mutation, no side effects, pure JAX arithmetic throughout.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

# Type alias matching the rest of the ceridwen codebase.
Array = jax.Array

# ---------------------------------------------------------------------------
# Pre-computed constant
# ---------------------------------------------------------------------------

# 0.5 * log(2 pi)  -- the fixed part of the Gaussian log-normalisation.
# Computed once at module import time (scalar JAX device array).
_HALF_LOG_2PI: Array = 0.5 * jnp.log(2.0 * jnp.pi)


# ===========================================================================
# Output container
# ===========================================================================

class NoiseModelOutput:
    """
    Per-datum quantities consumed by a Gaussian likelihood kernel.

    Both arrays have shape ``(n_data,)`` and correspond to the *effective*
    noise after adding any nuisance contributions.

    Attributes
    ----------
    inv_var : Array, shape (n_data,)
        Inverse effective variance ``1 / sigma_eff^2`` per datum.
        Used to form the chi-squared: ``chi_i^2 = (y_i - mu_i)^2 * inv_var_i``.
    log_det : Array, shape (n_data,)
        Per-datum log-normalisation: ``0.5 * log(2 pi sigma_eff^2)``.
        Summed over unmasked data, this is the ``-0.5 * log|Sigma|`` term
        of the Gaussian log-likelihood.

    Notes
    -----
    This is a plain class rather than a NamedTuple so that it can be
    registered as a JAX PyTree with named fields accessible by attribute.
    """

    __slots__ = ("inv_var", "log_det")

    def __init__(self, inv_var: Array, log_det: Array) -> None:
        self.inv_var = inv_var
        self.log_det = log_det

    def __repr__(self) -> str:
        return (
            f"NoiseModelOutput("
            f"inv_var={self.inv_var}, "
            f"log_det={self.log_det})"
        )


# Register NoiseModelOutput as a JAX PyTree so it can flow through
# jax.jit / jax.grad / jax.vmap boundaries.
jax.tree_util.register_pytree_node(
    NoiseModelOutput,
    flatten_func=lambda o: ([o.inv_var, o.log_det], None),
    unflatten_func=lambda _, leaves: NoiseModelOutput(
        inv_var=leaves[0], log_det=leaves[1]
    ),
)


# ===========================================================================
# Abstract base class
# ===========================================================================

class NoiseModelBase(abc.ABC):
    """
    Abstract base class for all Ceridwen noise models.

    Subclasses implement a single method, ``compute``, which is a pure JAX
    function mapping observational uncertainties and (optionally) model
    predictions and nuisance parameters to a ``NoiseModelOutput``.

    The ``compute`` method **must**:
    - contain no Python side effects
    - be JIT-safe (no dynamic Python control flow over traced values)
    - return finite values for all inputs (guard against zeros internally)
    - be differentiable w.r.t. ``sigma_obs``, ``mu``, and all elements of
      ``params`` that are JAX arrays
    """

    @abc.abstractmethod
    def compute(
        self,
        sigma_obs: Array,
        mu: Array,
        mask: Array,
        params: Optional[dict[str, Array]] = None,
    ) -> NoiseModelOutput:
        """
        Compute effective inverse variance and log-normalisation per datum.

        Parameters
        ----------
        sigma_obs : Array, shape (n_data,)
            Observational 1-sigma uncertainties in the same units as the data.
        mu : Array, shape (n_data,)
            Current model prediction.  Required when the effective noise
            depends on the predicted flux (e.g. fractional calibration error).
        mask : Array of bool, shape (n_data,)
            True for data points that contribute to the likelihood.  Masked
            points receive a safe fill value for ``inv_var`` so that no NaN
            or infinity propagates into the kernel; the kernel itself zeros
            their contribution via ``jnp.where``.
        params : dict[str, Array], optional
            Sampled nuisance parameters extracted from ``theta``.  Keys and
            meanings are noise-model-specific.  Pass ``None`` or ``{}`` when
            the model has no nuisance parameters.

        Returns
        -------
        NoiseModelOutput
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ===========================================================================
# Diagonal (independent) Gaussian noise model
# ===========================================================================

@dataclass(frozen=True)
class DiagonalNoiseModel(NoiseModelBase):
    """
    Independent Gaussian noise model with optional nuisance variance terms.

    The effective variance per datum is:

    .. math::

        \\sigma_{\\rm eff,i}^2 = \\sigma_{\\rm obs,i}^2
            + [\\exp(\\log f_{\\rm calib}) \\cdot |\\mu_i|]^2   \\quad (\\text{if use\\_fractional})
            + \\exp(2 \\log j)                                    \\quad (\\text{if use\\_jitter})

    The fractional calibration term models a systematic uncertainty that scales
    with the model flux -- standard in photometric SED fitting where the flux
    calibration is uncertain at the percent level.

    The jitter (noise-floor) term represents unmodelled noise that is
    independent of flux -- common in spectroscopy where sky subtraction
    residuals or read noise exceed the formal photon-noise estimate.

    Both nuisance parameters are sampled in **log space** (see ``params``
    below).  Log-space sampling ensures positivity without hard constraints,
    giving gradient-based samplers a smooth, unconstrained parameter space.

    Parameters
    ----------
    use_jitter : bool, default False
        If True, add an additive noise floor ``exp(log_jitter)`` in
        quadrature.  Reads ``params["log_jitter"]`` at compute time.
    use_fractional : bool, default False
        If True, add a fractional calibration error ``exp(log_f_calib) * |mu|``
        in quadrature.  Reads ``params["log_f_calib"]`` at compute time.

    Notes
    -----
    This class has **no JAX array fields**.  All configuration is Python-level.
    It is therefore registered as a JAX PyTree with empty leaves, making it a
    fully static compile-time constant when closed over inside ``jax.jit``.
    No retracing occurs when only ``theta`` changes between calls.

    Usage
    -----
    >>> nm = DiagonalNoiseModel(use_jitter=True)
    >>> out = nm.compute(sigma_obs, mu, mask, params={"log_jitter": jnp.log(0.02)})
    >>> out.inv_var   # shape (n_data,)
    >>> out.log_det   # shape (n_data,)
    """

    use_jitter     : bool = False
    use_fractional : bool = False

    # ------------------------------------------------------------------
    def compute(
        self,
        sigma_obs : Array,
        mu        : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
    ) -> NoiseModelOutput:
        """
        Compute ``inv_var`` and ``log_det`` for the diagonal Gaussian model.

        Parameters
        ----------
        sigma_obs : Array, shape (n_data,)
        mu : Array, shape (n_data,)
        mask : Array of bool, shape (n_data,)
        params : dict, optional
            Expected keys (depending on configuration):

            ``"log_jitter"``
                Log of the additive noise floor (same units as ``sigma_obs``).
                Must be present when ``use_jitter=True``.
            ``"log_f_calib"``
                Log of the fractional calibration error (dimensionless).
                Must be present when ``use_fractional=True``.

        Returns
        -------
        NoiseModelOutput
        """
        if params is None:
            params = {}

        # Start with observational variance.
        var: Array = sigma_obs ** 2

        # Fractional calibration error: sigma_calib = f_calib * |mu|
        # Gradient flows through mu -> var -> inv_var -> lnl cleanly.
        if self.use_fractional:
            f_calib = jnp.exp(params["log_f_calib"])
            var = var + (f_calib * jnp.abs(mu)) ** 2

        # Additive noise floor (jitter).
        # Parameterised as log_jitter so sampling is unconstrained.
        if self.use_jitter:
            jitter = jnp.exp(params["log_jitter"])
            var = var + jitter ** 2

        # Safety: masked data points receive var = 1 so that inv_var stays
        # finite.  The likelihood kernel will zero their contribution via
        # jnp.where(mask, ...) anyway, but we must avoid NaN gradients.
        var = jnp.where(mask, var, jnp.ones_like(var))

        # Hard floor: prevent numerical underflow for extremely small variances.
        # Uses the smallest representable positive float64 as a floor.
        var = jnp.maximum(var, jnp.finfo(var.dtype).tiny)

        inv_var = 1.0 / var
        # 0.5 * log(2 pi sigma^2) = 0.5*log(sigma^2) + 0.5*log(2pi)
        log_det = 0.5 * jnp.log(var) + _HALF_LOG_2PI

        return NoiseModelOutput(inv_var=inv_var, log_det=log_det)

    # ------------------------------------------------------------------
    @property
    def nuisance_param_names(self) -> tuple[str, ...]:
        """Names of nuisance parameters expected in ``params`` at compute time."""
        names: list[str] = []
        if self.use_fractional:
            names.append("log_f_calib")
        if self.use_jitter:
            names.append("log_jitter")
        return tuple(names)

    def __repr__(self) -> str:
        return (
            f"DiagonalNoiseModel("
            f"use_jitter={self.use_jitter}, "
            f"use_fractional={self.use_fractional})"
        )


# ---------------------------------------------------------------------------
# PyTree registration for DiagonalNoiseModel
#
# Empty leaves [] means there are no JAX-traced arrays stored in the class.
# The config booleans go into auxiliary data (static, compile-time).
# JAX treats this object as a constant when it is closed over in jax.jit,
# so changing only theta between calls never triggers retracing.
# ---------------------------------------------------------------------------
jax.tree_util.register_pytree_node(
    DiagonalNoiseModel,
    flatten_func=lambda nm: (
        [],                                            # leaves  (none)
        (nm.use_jitter, nm.use_fractional),            # aux     (static)
    ),
    unflatten_func=lambda aux, _: DiagonalNoiseModel(
        use_jitter=aux[0], use_fractional=aux[1]
    ),
)
