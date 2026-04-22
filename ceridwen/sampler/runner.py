"""
ceridwen/sampler/runner.py
==========================
Sampler-agnostic protocol and dispatch layer for Ceridwen SED fitting.

Architecture
------------
The sampler layer sits between the Ceridwen ``SedModel`` (physics + priors)
and the external sampling library (BlackJAX, NumPyro, emcee, …).

Three components:

1. ``SamplingResult``
   A universal, backend-agnostic container for posterior samples,
   Bayesian evidence, and diagnostics.  Includes a ``to_anesthetic()``
   helper for corner plots and evidence estimates.

2. ``SamplerAdapter``
   An abstract base class (protocol) that every sampling backend must
   implement.  The interface is deliberately minimal::

       result = adapter.run(loglike_fn, logprior_fn, theta_init, rng_key)

   Adding a new backend requires subclassing and overriding one method.

3. ``run_sampler``
   The single user-facing entry point.  Builds two pure JAX callables
   (log-likelihood and log-prior, *kept separate*) and delegates to the
   adapter.

Separation of log-likelihood and log-prior
------------------------------------------
MCMC samplers (HMC, NUTS, MCLMC) combine them into a single log-posterior.
Nested sampling algorithms (BlackJAX NSS, MultiNest, PolyChord) *must* keep
them separate: the NS algorithm uses the likelihood for the dead-point
contour and the prior as the proposal distribution.

Both functions are pure JAX — JIT-compilable, differentiable, and free of
Python-level side effects.

Extending to new samplers
--------------------------
::

    class BlackJAXHMCAdapter(SamplerAdapter):
        def run(self, loglike_fn, logprior_fn, theta_init, rng_key):
            lnpost = jax.jit(lambda t: loglike_fn(t) + logprior_fn(t))
            # … configure and run blackjax.nuts(lnpost) …
            return SamplingResult(...)
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

Array = jax.Array


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class SamplingResult:
    """
    Universal container for the output of any Ceridwen sampling run.

    All fields are backend-agnostic.  The sampler-specific raw output
    (e.g. a BlackJAX ``dead`` pytree) is preserved in ``raw`` for
    downstream processing.

    Attributes
    ----------
    samples : dict[str, Array]
        Posterior samples keyed by free-parameter name.
        Shape ``(n_samples, *param_shape)``.  For a scalar parameter
        of shape ``(1,)``, the trailing dimension is squeezed to give
        shape ``(n_samples,)`` for convenience.  For vector parameters
        such as ``logsfr_ratios`` of shape ``(k,)``, shape is
        ``(n_samples, k)``.
    log_evidence : float
        Log Bayesian evidence, :math:`\\ln Z`.
        ``float("nan")`` for MCMC samplers where :math:`Z` is not
        computed.
    log_evidence_err : float
        1-:math:`\\sigma` uncertainty on :math:`\\ln Z` (nested sampling
        only).  ``float("nan")`` for MCMC.
    log_weights : Array, shape (n_samples,)
        Log importance weights.  All zeros for MCMC (equal weights).
        Standard nested-sampling :math:`\\ln w_i` for NS.
    log_likelihoods : Array, shape (n_samples,)
        Log-likelihood :math:`\\ln \\mathcal{L}` at each sample.
        Required by ``anesthetic`` for evidence estimation.
    log_likelihoods_birth : Array or None, shape (n_samples,)
        Log-likelihood at birth used by ``anesthetic`` for the
        nested-sampling compression curve.  ``None`` for MCMC.
    param_names : list[str]
        Ordered list of free-parameter names (same order as
        ``model.param_names``).
    n_likelihood_calls : int
        Approximate total number of likelihood evaluations.
    wall_time_s : float
        Wall-clock run time in seconds.
    sampler_name : str
        Identifier string for the backend (e.g. ``"blackjax.nss"``).
    raw : Any
        Raw sampler output preserved for advanced use.  For BlackJAX
        NSS this is the ``dead`` pytree; for HMC it would be the
        ``mcmc_state`` sequence.
    """

    samples               : dict[str, Array]
    log_evidence          : float
    log_evidence_err      : float
    log_weights           : Array
    log_likelihoods       : Array
    param_names           : list[str]
    n_likelihood_calls    : int
    wall_time_s           : float
    sampler_name          : str
    log_likelihoods_birth : Optional[Array] = None
    raw                   : Any = None

    # ------------------------------------------------------------------
    def to_anesthetic(self, labels: Optional[dict[str, str]] = None):
        """
        Convert to an ``anesthetic.NestedSamples`` object.

        Requires ``anesthetic``
        (``pip install git+https://github.com/handley-lab/anesthetic``).

        Parameters
        ----------
        labels : dict[str, str], optional
            Mapping from parameter name to LaTeX label string.

        Returns
        -------
        anesthetic.NestedSamples
        """
        try:
            from anesthetic import NestedSamples
        except ImportError as exc:
            raise ImportError(
                "anesthetic is required for to_anesthetic(). "
                "Install: pip install git+https://github.com/handley-lab/anesthetic"
            ) from exc

        import numpy as np

        columns, col_labels, data_cols = [], [], []

        for name in self.param_names:
            arr = np.asarray(self.samples[name])      # (n, *shape)
            if arr.ndim == 1:
                columns.append(name)
                col_labels.append(labels.get(name, name) if labels else name)
                data_cols.append(arr[:, None])
            else:
                for i in range(arr.shape[1]):
                    col = f"{name}[{i}]"
                    columns.append(col)
                    col_labels.append(
                        f"{labels.get(name, name)}$_{{{i}}}$"
                        if labels else col
                    )
                    data_cols.append(arr[:, i : i + 1])

        data = np.hstack(data_cols)

        return NestedSamples(
            data,
            logL       = np.asarray(self.log_likelihoods),
            logL_birth = (
                np.asarray(self.log_likelihoods_birth)
                if self.log_likelihoods_birth is not None
                else None
            ),
            columns    = columns,
            labels     = col_labels,
            logzero    = float("nan"),
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """One-line human-readable summary."""
        n = next(iter(self.samples.values())).shape[0]
        lz = (f"{self.log_evidence:.3f} ± {self.log_evidence_err:.3f}"
              if jnp.isfinite(self.log_evidence) else "n/a")
        return (
            f"{self.sampler_name}  |  n_samples={n}  "
            f"ln Z = {lz}  |  "
            f"n_like = {self.n_likelihood_calls}  |  "
            f"wall = {self.wall_time_s:.1f} s"
        )

    def __repr__(self) -> str:
        n = next(iter(self.samples.values())).shape[0]
        return (
            f"SamplingResult(sampler={self.sampler_name!r}, "
            f"n_samples={n}, ln_Z={self.log_evidence:.3f})"
        )


# ===========================================================================
# Abstract adapter (protocol)
# ===========================================================================

class SamplerAdapter(abc.ABC):
    """
    Abstract base class for Ceridwen sampling backends.

    Subclass this to wrap a new sampler.  The only required method is
    ``run``, which receives two pure JAX callables and returns a
    ``SamplingResult``.

    The deliberate separation of ``loglike_fn`` and ``logprior_fn``
    makes the adapter suitable for *both* MCMC and nested sampling
    without any additional glue code::

        # MCMC: combine into log-posterior
        lnpost = jax.jit(lambda t: loglike_fn(t) + logprior_fn(t))

        # Nested sampling: keep separate
        blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglike_fn)

    Notes
    -----
    ``theta_init`` carries the shapes and dtypes of all free parameters
    (as a ``dict[str, Array]``).  Use it to determine chain/live-point
    initialisation and to reconstruct the dict structure from any flat
    array representation required by the backend.
    """

    @abc.abstractmethod
    def run(
        self,
        loglike_fn  : Callable[[dict[str, Array]], Array],
        logprior_fn : Callable[[dict[str, Array]], Array],
        theta_init  : dict[str, Array],
        rng_key     : Array,
    ) -> SamplingResult:
        """
        Run the sampler and return posterior samples.

        Parameters
        ----------
        loglike_fn : callable
            Pure JAX log-likelihood function, signature
            ``loglike_fn(theta_dict) → scalar``.  No prior included.
        logprior_fn : callable
            Pure JAX log-prior function, signature
            ``logprior_fn(theta_dict) → scalar``.
        theta_init : dict[str, Array]
            Initial / reference parameter values with correct shapes.
        rng_key : Array
            JAX PRNGKey.

        Returns
        -------
        SamplingResult
        """


# ===========================================================================
# Main dispatch function
# ===========================================================================

def run_sampler(
    model      : Any,
    likelihood : Any,
    adapter    : SamplerAdapter,
    rng_key    : Array,
) -> SamplingResult:
    """
    Build JAX callables from a Ceridwen model and dispatch to a sampler.

    This is the **primary user-facing entry point** for posterior sampling
    in Ceridwen.  It is intentionally thin:

    1. Constructs a JIT-compiled ``loglike_fn`` that sums log-likelihoods
       over all registered observations (no prior contribution).
    2. Wraps ``model.ln_prior`` as a JIT-compiled ``logprior_fn``.
    3. Delegates to ``adapter.run(loglike_fn, logprior_fn, theta_init, key)``.

    Parameters
    ----------
    model : SedModel
        Initialised model with observations set up and priors registered.
        ``model.predict(theta)`` and ``model.ln_prior(theta)`` are called
        inside the JIT-compiled hot paths.
    likelihood : MultiObservationLikelihood
        Likelihood object for the registered observations.  Its ``.keys``
        and ``.likelihoods`` attributes are iterated statically at
        XLA trace time — they are Python constants, not traced values.
    adapter : SamplerAdapter
        Concrete backend (e.g. ``BlackJAXNestedSamplerAdapter``).
    rng_key : Array
        JAX PRNGKey for reproducibility.

    Returns
    -------
    SamplingResult
        Call ``.to_anesthetic()`` for corner plots and evidence estimates.

    Examples
    --------
    Nested sampling with BlackJAX NSS::

        from ceridwen.sampler import run_sampler
        from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter

        adapter = BlackJAXNestedSamplerAdapter(
            priors          = model.priors,
            num_live        = 500,
            num_inner_steps = len(model.param_names) * 5,
        )
        result = run_sampler(model, multi_likelihood, adapter,
                             jax.random.PRNGKey(42))

        ns = result.to_anesthetic(labels={"Z": r"$\\log Z/Z_\\odot$"})
        print(result.summary())

    Future HMC adapter (same call signature)::

        adapter = BlackJAXHMCAdapter(step_size=0.01, n_warmup=500)
        result  = run_sampler(model, multi_likelihood, adapter, key)
    """
    # ── Static data extracted once, before trace ──────────────────────────
    _obs_dict    = model.obs_dict
    _keys        = tuple(likelihood.keys)
    _likelihoods = tuple(likelihood.likelihoods)
    _static_data = {
        key: (
            _obs_dict[key].flux,
            _obs_dict[key].uncertainty,
            _obs_dict[key].mask,
        )
        for key in _keys
    }

    # ── Log-likelihood: sum over observations, no prior ───────────────────
    @jax.jit
    def loglike_fn(theta: dict[str, Array]) -> Array:
        predictions = model.predict(theta)
        lnl = jnp.zeros(())
        for key, lhood in zip(_keys, _likelihoods):
            y_k, sig_k, mask_k = _static_data[key]
            mu_k    = predictions[key]
            lnl_k, _ = lhood(y_k, mu_k, sig_k, mask_k, params=theta)
            lnl = lnl + lnl_k
        return lnl

    # ── Log-prior ─────────────────────────────────────────────────────────
    @jax.jit
    def logprior_fn(theta: dict[str, Array]) -> Array:
        return model.ln_prior(theta)

    # ── Delegate ──────────────────────────────────────────────────────────
    return adapter.run(loglike_fn, logprior_fn, model.theta_init, rng_key)
