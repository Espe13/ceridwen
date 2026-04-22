"""
ceridwen/sampler/nested.py
==========================
BlackJAX adaptive nested sampler (``blackjax.nss``) adapter for Ceridwen.

Algorithm background
--------------------
Nested sampling (Skilling 2006) integrates the Bayesian evidence by
compressing the prior volume along iso-likelihood contours.  At each
step the least-likely live point is replaced by a new point drawn from
the prior inside the current likelihood contour.

BlackJAX's ``nss`` (nested slice sampler) uses an adaptive slice-sampling
inner kernel.  The key parameters are:

``num_live``
    Resolution parameter.  Evidence uncertainty scales as
    :math:`1/\\sqrt{N_\\mathrm{live}}`.  Typical: 200 (fast) – 2000
    (publication quality).

``num_inner_steps``
    Reliability of the inner MCMC chain.  Best practice: ``n_dims * 5``.
    Check stability by halving/doubling.

``num_delete``
    Parallelisation: points removed per iteration.
    Default: ``num_live // 2``.

``logZ_tol``
    Convergence threshold :math:`\\ln(Z_\\mathrm{live}/Z)`.
    Iteration stops when the live contribution drops below this.
    Default ``-3`` (:math:`\\approx 5\\%` of total evidence remaining).

Installation
------------
::

    pip install git+https://github.com/handley-lab/blackjax
    pip install git+https://github.com/handley-lab/anesthetic  # optional

Usage
-----
::

    from ceridwen.sampler        import run_sampler
    from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter

    adapter = BlackJAXNestedSamplerAdapter(
        priors          = model.priors,
        num_live        = 500,
        num_inner_steps = len(model.param_names) * 5,
    )
    result = run_sampler(model, multi_likelihood, adapter,
                         jax.random.PRNGKey(42))

    print(result.summary())
    ns = result.to_anesthetic(labels={...})
    ns.plot_2d([...])
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .runner import SamplerAdapter, SamplingResult

Array = jax.Array


class BlackJAXNestedSamplerAdapter(SamplerAdapter):
    """
    Adapter wrapping ``blackjax.nss`` for use with any Ceridwen ``SedModel``.

    Handles three concerns that are specific to nested sampling in Ceridwen:

    1. **Live-point initialisation** — samples each free parameter
       independently from its registered prior using the ``Prior.sample``
       method from ``ceridwen.sampler.priors``.

    2. **Shape reconciliation** — the BlackJAX NSS live-point dict has
       shape ``{name: (num_live, *param_shape)}``.  The step function
       vmaps over axis 0, delivering single-particle slices of shape
       ``(*param_shape,)`` to ``loglike_fn`` / ``logprior_fn``.  This
       matches the Ceridwen dict-theta convention exactly.

    3. **Evidence extraction** — optionally uses ``anesthetic`` for a
       more accurate :math:`\\ln Z` estimate with uncertainty.

    Parameters
    ----------
    priors : dict[str, Prior]
        Mapping from free-parameter name to a Ceridwen ``Prior`` object.
        **Every free parameter must have a prior** — nested sampling
        requires a proper (normalisable) prior; an improper flat prior
        makes the evidence integral undefined.
    num_live : int, optional
        Number of live points.  Default 500.
    num_inner_steps : int, optional
        Inner MCMC steps per NS iteration.  Default ``n_dims * 5``
        where ``n_dims`` is the total scalar dimension count.
    num_delete : int, optional
        Live points discarded per iteration.  Default ``num_live // 2``.
    logZ_tol : float, optional
        Convergence threshold on :math:`\\ln(Z_\\mathrm{live}/Z)`.
        Default ``-3.0``.
    verbose : bool, optional
        Print a ``tqdm`` progress bar and convergence info.  Default True.
    """

    def __init__(
        self,
        priors          : dict,
        num_live        : int   = 500,
        num_inner_steps : Optional[int] = None,
        num_delete      : Optional[int] = None,
        logZ_tol        : float = -3.0,
        verbose         : bool  = True,
    ):
        self.priors          = dict(priors)
        self.num_live        = int(num_live)
        self._num_inner_steps = num_inner_steps   # None → auto
        self._num_delete      = num_delete         # None → num_live // 2
        self.logZ_tol        = float(logZ_tol)
        self.verbose         = bool(verbose)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_dims(self, theta_init: dict[str, Array]) -> int:
        """Total scalar degrees of freedom."""
        return sum(int(jnp.size(v)) for v in theta_init.values())

    def _sample_prior(
        self,
        theta_init : dict[str, Array],
        rng_key    : Array,
    ) -> dict[str, Array]:
        """
        Draw ``num_live`` initial live points from the registered priors.

        Each parameter is sampled independently.  The returned dict has
        shape ``{name: (num_live, *param_shape)}``.

        Raises
        ------
        ValueError
            If any free parameter has no registered prior.
        """
        particles = {}
        for name, init_val in theta_init.items():
            if name not in self.priors:
                raise ValueError(
                    f"Parameter '{name}' has no prior.  "
                    f"Nested sampling requires a proper prior for every "
                    f"free parameter.  Add '{name}' to the priors dict "
                    f"passed to BlackJAXNestedSamplerAdapter."
                )
            prior          = self.priors[name]
            rng_key, sub   = jax.random.split(rng_key)
            expected_shape = init_val.shape           # e.g. (1,) or (k,)

            # Pass the full target shape (num_live, *expected_shape) to
            # prior.sample so TFP broadcasts a scalar distribution over
            # all required dimensions in a single call.
            #
            # Examples:
            #   Normal(0, 1), expected_shape=(4,)
            #     → sample((500, 4)) → (500, 4)  ✓  iid per element
            #   Uniform(-2.5, 0.2), expected_shape=(1,)
            #     → sample((500, 1)) → (500, 1)  ✓
            #
            # This avoids the shape mismatch that arises when sampling
            # (num_live,) and then trying to reshape to (num_live, k).
            particles[name] = prior.sample(sub, shape=(self.num_live, *expected_shape))

        return particles

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        loglike_fn  : Callable[[dict[str, Array]], Array],
        logprior_fn : Callable[[dict[str, Array]], Array],
        theta_init  : dict[str, Array],
        rng_key     : Array,
    ) -> SamplingResult:
        """
        Run BlackJAX NSS and return a ``SamplingResult``.

        Parameters
        ----------
        loglike_fn : callable
            JIT-compiled log-likelihood (no prior).
        logprior_fn : callable
            JIT-compiled log-prior (must be proper).
        theta_init : dict[str, Array]
            Reference parameter dict (shapes / dtypes).
        rng_key : Array

        Returns
        -------
        SamplingResult
        """
        try:
            import blackjax
            import blackjax.ns.utils as ns_utils
        except ImportError as exc:
            raise ImportError(
                "BlackJAX is required for nested sampling.\n"
                "Install: pip install git+https://github.com/handley-lab/blackjax"
            ) from exc

        import tqdm

        n_dims          = self._n_dims(theta_init)
        num_inner_steps = (self._num_inner_steps
                           if self._num_inner_steps is not None
                           else n_dims * 5)
        num_delete      = (self._num_delete
                           if self._num_delete is not None
                           else self.num_live // 2)

        if self.verbose:
            print(
                f"BlackJAX NSS  |  n_dims={n_dims}  "
                f"num_live={self.num_live}  "
                f"num_inner_steps={num_inner_steps}  "
                f"num_delete={num_delete}"
            )

        # ── Initialise live points ────────────────────────────────────────
        rng_key, prior_key = jax.random.split(rng_key)
        particles = self._sample_prior(theta_init, prior_key)

        # ── Build NSS kernel ──────────────────────────────────────────────
        # loglike_fn / logprior_fn operate on a SINGLE particle (un-batched).
        # The NSS step_fn vmaps internally over the live-point ensemble.
        nested_sampler = blackjax.nss(
            logprior_fn      = logprior_fn,
            loglikelihood_fn = loglike_fn,
            num_delete       = num_delete,
            num_inner_steps  = num_inner_steps,
        )
        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)

        if self.verbose:
            print("  [timing] Calling init_fn (JIT compile + eval) ...",
                  flush=True)
            _t0 = time.perf_counter()

        live = init_fn(particles)

        # ── BlackJAX version compatibility ───────────────────────────────
        # Three known layouts for logZ / logZ_live:
        #   v1  NSState (direct):           state.logZ, state.logZ_live
        #   v2  AdaptiveNSState (wrapper):  state.sampler_state.logZ, ...
        #   v3  AdaptiveNSState (integrator): state.integrator.logZ, ...
        def _build_logZ_accessors(state):
            """Return (get_logZ, get_logZ_live) callables for *state*."""
            # v1 – direct NSState
            if hasattr(state, "logZ") and hasattr(state, "logZ_live"):
                return (lambda s: float(s.logZ),
                        lambda s: float(s.logZ_live))
            # v3 – newest: integrator sub-object
            if hasattr(state, "integrator"):
                ig = state.integrator
                if hasattr(ig, "logZ") and hasattr(ig, "logZ_live"):
                    return (lambda s: float(s.integrator.logZ),
                            lambda s: float(s.integrator.logZ_live))
            # v2 – older wrapper: sampler_state sub-object
            if hasattr(state, "sampler_state"):
                inner = state.sampler_state
                if hasattr(inner, "logZ") and hasattr(inner, "logZ_live"):
                    return (lambda s: float(s.sampler_state.logZ),
                            lambda s: float(s.sampler_state.logZ_live))
            # Unknown layout – raise with diagnostics
            _fields = [f for f in dir(state) if not f.startswith("_")]
            raise AttributeError(
                f"Cannot locate logZ/logZ_live on {type(state).__name__}.\n"
                f"  Top-level fields : {_fields}\n"
                + "Please check your BlackJAX version."
            )

        _get_logZ, _get_logZ_live = _build_logZ_accessors(live)

        if self.verbose:
            _t1 = time.perf_counter()
            print(f"  [timing] init_fn done    ({_t1 - _t0:.1f} s)  "
                  f"logZ={_get_logZ(live):.4f}  "
                  f"logZ_live={_get_logZ_live(live):.4f}", flush=True)
            print(f"  (state type: {type(live).__name__})")

        # ── NS run loop ───────────────────────────────────────────────────
        dead_list    = []
        n_like_calls = 0
        t_start      = time.perf_counter()

        desc = "NS  (starting)"
        try:
            desc = f"NS  logZ={_get_logZ(live):.1f}"
        except (AttributeError, NameError):
            pass

        with tqdm.tqdm(
            desc=desc,
            unit=" dead",
            disable=not self.verbose,
        ) as pbar:
            _iter = 0
            while float(_get_logZ_live(live) - _get_logZ(live)) >= self.logZ_tol:
                rng_key, subkey = jax.random.split(rng_key)
                if _iter == 0 and self.verbose:
                    print("  [step_fn] Compiling + running first iteration "
                          "(this may take 1-2 min on CPU) ...", flush=True)
                _t_iter = time.perf_counter()

                live, dead_info = step_fn(subkey, live)

                _dt_iter = time.perf_counter() - _t_iter
                _iter += 1
                if self.verbose:
                    _logZ = _get_logZ(live)
                    _dlogZ = _get_logZ_live(live) - _logZ
                    print(
                        f"  [iter {_iter:>4d}]  {_dt_iter:6.1f} s  "
                        f"logZ={_logZ:+.3f}  ΔlogZ={_dlogZ:.3f}  "
                        f"dead={num_delete * _iter}",
                        flush=True,
                    )
                dead_list.append(dead_info)
                n_like_calls += num_delete * num_inner_steps
                pbar.update(num_delete)
                try:
                    pbar.set_description(
                        f"NS  logZ={_get_logZ(live):.2f}  "
                        f"ΔlogZ={_get_logZ_live(live) - _get_logZ(live):.2f}"
                    )
                except AttributeError:
                    pass

        wall_time = time.perf_counter() - t_start
        if self.verbose:
            print(
                f"  Converged  logZ = {_get_logZ(live):.3f}  "
                f"({wall_time:.1f} s,  {n_like_calls:,} likelihood calls)"
            )

        # ── Merge live points into dead set ───────────────────────────────
        # update_info=False avoids tree_map over update metadata that
        # some BlackJAX versions do not populate on NSInfo.
        dead = ns_utils.finalise(live, dead_list, update_info=False)

        # ── Unpack dead-point structure ───────────────────────────────────
        # finalise() returns an object whose layout depends on version:
        #   older: dead.particles = dict,  dead.loglikelihood, ...
        #   v3:    dead.particles = StateWithLogLikelihood
        #             .position (dict), .loglikelihood, .loglikelihood_birth
        _dp = dead.particles
        if isinstance(_dp, dict):
            _dead_positions      = _dp
            _dead_logl           = dead.loglikelihood
            _dead_logl_birth     = dead.loglikelihood_birth
        elif hasattr(_dp, "position"):
            # v3 StateWithLogLikelihood
            _dead_positions      = _dp.position
            _dead_logl           = _dp.loglikelihood
            _dead_logl_birth     = _dp.loglikelihood_birth
        else:
            # last resort: assume dead itself has the fields
            _dead_positions      = dead.position if hasattr(dead, "position") else _dp
            _dead_logl           = dead.loglikelihood
            _dead_logl_birth     = dead.loglikelihood_birth

        # ── Evidence & importance weights (anesthetic preferred) ─────────
        log_Z        = float(_get_logZ(live))
        log_Z_err    = float("nan")
        log_weights  = None
        try:
            from anesthetic import NestedSamples
            import numpy as np

            _raw   = _dead_positions
            _names = [n for n in _raw if n in theta_init]
            _cols  = [
                np.asarray(_raw[n]).reshape(
                    len(np.asarray(_dead_logl)), -1
                )
                for n in _names
            ]
            _data  = np.hstack(_cols)
            _ns    = NestedSamples(
                _data,
                logL       = np.asarray(_dead_logl),
                logL_birth = np.asarray(_dead_logl_birth),
                logzero    = float("nan"),
            )
            log_Z     = float(_ns.logZ())
            log_Z_err = float(_ns.logZ(12).std())
            # Extract proper NS importance weights from anesthetic.
            # These encode the prior-volume compression at each dead point.
            log_weights = jnp.asarray(np.asarray(_ns.logw()))
        except Exception:
            pass  # fall back to live.logZ and manual weights below

        # Fallback: compute log-weights from prior volume shrinkage if
        # anesthetic is unavailable or failed.
        if log_weights is None:
            n_dead  = len(jnp.asarray(_dead_logl))
            n_live  = self.num_live
            # Standard NS trapezoid rule: log(X_{i-1} - X_{i+1}) / 2
            # where X_i = exp(-i / n_live) is the prior volume fraction.
            log_vols = -jnp.arange(n_dead, dtype=float) / n_live
            log_dvol = jnp.log(
                jnp.exp(jnp.roll(log_vols, 1) - log_vols)
                - jnp.exp(jnp.roll(log_vols, -1) - log_vols)
            ) + log_vols
            # Fix boundary: first and last points
            log_dvol = log_dvol.at[0].set(
                jnp.log1p(-jnp.exp(-1.0 / n_live))
            )
            log_dvol = log_dvol.at[-1].set(
                log_vols[-1] - jnp.log(n_live)
            )
            log_weights = log_dvol

        # ── Pack samples ──────────────────────────────────────────────────
        # Squeeze trailing size-1 axes so scalar params have shape (n_dead,)
        # rather than (n_dead, 1), matching typical user expectation.
        samples = {}
        for name in theta_init:
            arr = jnp.asarray(_dead_positions[name])   # (n_dead, *shape)
            # Squeeze only if the parameter was a scalar (shape (1,))
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = jnp.squeeze(arr, axis=-1)
            samples[name] = arr

        return SamplingResult(
            samples               = samples,
            log_evidence          = log_Z,
            log_evidence_err      = log_Z_err,
            log_weights           = log_weights,
            log_likelihoods       = jnp.asarray(_dead_logl),
            log_likelihoods_birth = jnp.asarray(_dead_logl_birth),
            param_names           = list(theta_init.keys()),
            n_likelihood_calls    = n_like_calls,
            wall_time_s           = wall_time,
            sampler_name          = "blackjax.nss",
            raw                   = dead,
        )
