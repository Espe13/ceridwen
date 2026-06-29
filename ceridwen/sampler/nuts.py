"""
ceridwen/sampler/nuts.py
=========================
BlackJAX NUTS (No-U-Turn Sampler) adapter for Ceridwen.

Algorithm background
--------------------
NUTS (Hoffman & Gelman 2014) is an adaptive variant of Hamiltonian Monte
Carlo that automatically tunes the trajectory length.  Combined with
dual-averaging step-size adaptation and an inverse mass matrix (diagonal
or dense), NUTS is the default gradient-based sampler in probabilistic
programming frameworks such as Stan, NumPyro, and PyMC.

Because the Ceridwen forward model is fully JAX-traceable and
differentiable, ``jax.grad`` of the log-posterior flows through the
CSP prediction, observation projection, and likelihood evaluation
end-to-end.  NUTS can therefore leverage exact gradients without
finite differences, giving efficient exploration even in moderately
high dimensions (n_dims ~ 10--50).

Reparameterisation
~~~~~~~~~~~~~~~~~~
Parameters with *bounded* (uniform) priors are automatically mapped to
an unconstrained space via sigmoid/logit transforms, exactly as Stan
and NumPyro do.  This eliminates the hard boundary walls that cause
divergent transitions with the leapfrog integrator.

The mapping for a Uniform(a, b) parameter is::

    constrained   = a + (b - a) * sigmoid(x)
    unconstrained = logit((theta - a) / (b - a))
    log-Jacobian  = log(b - a) - softplus(x) - softplus(-x)

Parameters with Gaussian or other unbounded priors are passed through
untransformed.

Key parameters
~~~~~~~~~~~~~~
``num_warmup``
    Number of adaptation (warmup) steps.  During warmup the step size
    and mass matrix are tuned via ``blackjax.window_adaptation``.
    Typical: 500--2000.

``num_samples``
    Number of post-warmup posterior draws.  Typical: 1000--5000.

``num_chains``
    Number of independent chains.  Multiple chains are used for
    convergence diagnostics (R-hat).  Default: 4.

``dense_mass``
    If True (default), adapt a full (dense) inverse mass matrix rather
    than a diagonal.  This is critical when the posterior has strong
    parameter correlations (e.g. logsfr_ratios -- logmass degeneracy).

``target_acceptance``
    Target acceptance probability.  Default 0.9.  Higher values (0.9--
    0.95) reduce divergent transitions in difficult geometries at the
    cost of smaller step sizes.

Installation
------------
::

    pip install git+https://github.com/blackjax-devs/blackjax

Usage
-----
::

    from ceridwen.sampler       import run_sampler
    from ceridwen.sampler.nuts  import BlackJAXNUTSAdapter

    adapter = BlackJAXNUTSAdapter(
        num_warmup  = 1500,
        num_samples = 2000,
        num_chains  = 4,
    )
    result = run_sampler(model, multi_likelihood, adapter,
                         jax.random.PRNGKey(0))

    print(result.summary())
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .runner import SamplerAdapter, SamplingResult

Array = jax.Array


# ======================================================================
# Bounded-parameter reparameterisation helpers
# ======================================================================

def _build_transforms(bounds: dict[str, tuple[float, float]],
                      theta_template: dict[str, Array]):
    """
    Build per-element sigmoid/logit transforms for the flattened vector.

    Parameters
    ----------
    bounds : dict
        Maps parameter names to (low, high) tuples.  Parameters not in
        this dict are treated as unconstrained.
    theta_template : dict
        Reference theta dict for parameter names, shapes, and ordering.

    Returns
    -------
    lo_safe, hi_safe : Array (n_dims,)
        Lower and upper bounds for every scalar element.  Unbounded
        elements get dummy bounds (0, 1) to avoid NaN in the unused
        branch of ``jnp.where`` (JAX evaluates both branches).
    is_bounded : Array (n_dims,) bool
        True for elements that need sigmoid transform.
    """
    lo_list, hi_list, bounded_list = [], [], []
    for name, template in theta_template.items():
        size = int(jnp.size(template))
        if name in bounds:
            a, b = bounds[name]
            lo_list.append(jnp.full(size, a))
            hi_list.append(jnp.full(size, b))
            bounded_list.append(jnp.ones(size, dtype=bool))
        else:
            # Dummy finite bounds (0, 1) so the sigmoid branch never
            # produces inf * 0 = nan.  The result is discarded by
            # jnp.where, but JAX traces both branches for gradients.
            lo_list.append(jnp.full(size, 0.0))
            hi_list.append(jnp.full(size, 1.0))
            bounded_list.append(jnp.zeros(size, dtype=bool))

    lo = jnp.concatenate(lo_list)
    hi = jnp.concatenate(hi_list)
    is_bounded = jnp.concatenate(bounded_list)
    return lo, hi, is_bounded


def _to_constrained(x: Array, lo: Array, hi: Array, is_bounded: Array) -> Array:
    """Unconstrained x -> constrained theta (flat).

    IMPORTANT: ``lo`` and ``hi`` must be finite for ALL elements (use
    dummy bounds for unbounded params) because ``jnp.where`` evaluates
    both branches and inf/nan poisons JAX's gradient tape.
    """
    sig = jax.nn.sigmoid(x)
    constrained = lo + (hi - lo) * sig
    return jnp.where(is_bounded, constrained, x)


def _to_unconstrained(theta_flat: Array, lo: Array, hi: Array, is_bounded: Array) -> Array:
    """Constrained theta (flat) -> unconstrained x.

    ``lo`` and ``hi`` must be finite for ALL elements (dummy bounds for
    unbounded params).
    """
    frac = (theta_flat - lo) / (hi - lo)
    frac = jnp.clip(frac, 1e-7, 1.0 - 1e-7)
    unconstrained = jnp.log(frac / (1.0 - frac))
    return jnp.where(is_bounded, unconstrained, theta_flat)


def _log_jacobian(x: Array, lo: Array, hi: Array, is_bounded: Array) -> Array:
    """
    Log |det J| for the sigmoid transform (unconstrained -> constrained).

    For each bounded element:  log(hi - lo) + log sigma(x) + log(1 - sigma(x))
                              = log(hi - lo) - softplus(x) - softplus(-x)

    Unbounded elements contribute 0.
    """
    log_jac_elem = jnp.where(
        is_bounded,
        jnp.log(hi - lo) - jax.nn.softplus(x) - jax.nn.softplus(-x),
        0.0,
    )
    return jnp.sum(log_jac_elem)


# ======================================================================
# NUTS adapter
# ======================================================================

class BlackJAXNUTSAdapter(SamplerAdapter):
    """
    Adapter wrapping ``blackjax.nuts`` with window adaptation for Ceridwen.

    Bounded (uniform-prior) parameters are automatically reparameterised
    onto an unconstrained space via sigmoid/logit, eliminating the hard
    boundary walls that cause divergent transitions.

    Optionally, a variational transport map (see :mod:`ceridwen.sampler.vi`)
    may be supplied via ``vi``.  When set, the adapter trains the map
    against the unconstrained posterior and then runs NUTS on the
    *whitened* target :math:`\\log p(f(z)) + \\log|\\partial f/\\partial z|`
    (Hoffman et al. 2019, arXiv:1903.03704).  In whitened space the
    target is approximately :math:`\\mathcal{N}(0, I)` so identity mass
    matrix and step size :math:`\\mathcal{O}(1)` are near-optimal,
    giving dramatically shorter warmup.

    Parameters
    ----------
    num_warmup : int, optional
        Number of warmup (adaptation) steps per chain.  Default 1500
        for native NUTS.  When ``vi`` is set, warmup defaults to 200;
        very short warmup lets the dual-averaging adapter overshoot and
        produce many divergences on a ~14-D SED posterior.
    num_samples : int, optional
        Number of post-warmup posterior draws per chain.  Default 2000.
    num_chains : int, optional
        Number of independent chains.  Default 4.
    initial_step_size : float, optional
        Starting step size for the leapfrog integrator before adaptation.
        Default 0.01 for native NUTS.  When ``vi`` is set, default is
        0.5 — z-space is pre-whitened so the optimal :math:`\\varepsilon`
        is :math:`\\mathcal{O}(1)`, and dual averaging is slow to shrink
        from 1.0.
    target_acceptance : float, optional
        Target acceptance probability for dual averaging.  Default 0.95;
        higher values reduce divergences in VI-preconditioned NUTS.
    max_num_doublings : int, optional
        Maximum tree depth (2^max_num_doublings leapfrog steps).
        Default 10.
    dense_mass : bool, optional
        Use a dense (full) inverse mass matrix.  Default True for
        native NUTS.  When ``vi`` is set, default is False — the VI
        map already orthogonalises the geometry, so an adapted diagonal
        mass matrix in z-space is sufficient and faster to fit.
    bounds : dict, optional
        Maps parameter names to (low, high) tuples for bounded params.
        If None (default), bounds are auto-detected from the model priors
        passed through ``run_sampler``.  You can also pass them explicitly::

            bounds={'Z': (-2.5, 0.2), 'logmass': (9.0, 12.0)}

    vi : None, str, VariationalMap, or TrainedMap
        Variational preconditioning mode.

        - ``None`` (default): run native NUTS with window adaptation.
        - ``'tril'``: train a full-rank Gaussian map, then whiten.
        - ``'iaf'``: train a stacked-IAF neural-transport map, then whiten.
        - :class:`VariationalMap` instance: train *this* map.
        - :class:`TrainedMap` instance: skip training, use as-is.

    vi_kwargs : dict, optional
        Forwarded either to the VI map constructor (when ``vi`` is a
        string) or to :func:`ceridwen.sampler.vi.train_vi` for training
        hyperparameters.  Recognised keys: ``num_steps`` (default 1500),
        ``batch_size`` (16), ``lr0`` (1e-2) plus map-specific kwargs
        (e.g. ``init_scale`` for 'tril', ``n_flows`` for 'iaf').

    verbose : bool, optional
        Print progress information.  Default True.
    """

    def __init__(
        self,
        num_warmup: int | None = None,
        num_samples: int = 2000,
        num_chains: int = 4,
        initial_step_size: float | None = None,
        target_acceptance: float = 0.95,
        max_num_doublings: int = 10,
        dense_mass: bool | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        vi: "str | Any | None" = None,
        vi_kwargs: dict | None = None,
        verbose: bool = True,
    ):
        # VI-aware defaults: in whitened space, target geometry is
        # approximately isotropic Gaussian, so shorter warmup / larger
        # initial step / diagonal mass matrix are all appropriate.
        _has_vi = vi is not None
        if num_warmup is None:
            num_warmup = 200 if _has_vi else 1500
        if initial_step_size is None:
            initial_step_size = 0.5 if _has_vi else 0.01
        if dense_mass is None:
            dense_mass = not _has_vi

        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.initial_step_size = float(initial_step_size)
        self.target_acceptance = float(target_acceptance)
        self.max_num_doublings = int(max_num_doublings)
        self.dense_mass = bool(dense_mass)
        self.bounds = dict(bounds) if bounds is not None else None
        self.vi = vi
        self.vi_kwargs = dict(vi_kwargs) if vi_kwargs is not None else {}
        self.verbose = bool(verbose)
        # Filled during .run() so the caller can inspect the trained map
        # after sampling (e.g. for plotting the learnt covariance).
        self.trained_map = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_dims(self, theta_init: dict[str, Array]) -> int:
        """Total scalar degrees of freedom."""
        return sum(int(jnp.size(v)) for v in theta_init.values())

    def _flatten(self, theta: dict[str, Array]) -> Array:
        """Flatten a parameter dict into a 1-D vector."""
        return jnp.concatenate([jnp.ravel(v) for v in theta.values()])

    def _unflatten(self, x: Array, theta_template: dict[str, Array]) -> dict[str, Array]:
        """Unflatten a 1-D vector back to the parameter dict structure."""
        out = {}
        idx = 0
        for name, template in theta_template.items():
            size = int(jnp.size(template))
            out[name] = x[idx:idx + size].reshape(template.shape)
            idx += size
        return out

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        loglike_fn: Callable[[dict[str, Array]], Array],
        logprior_fn: Callable[[dict[str, Array]], Array],
        theta_init: dict[str, Array],
        rng_key: Array,
    ) -> SamplingResult:
        """
        Run BlackJAX NUTS with window adaptation.

        Strategy (GPU-optimised):
          1. Run ONE warmup to adapt step size + mass matrix.
          2. Build a single NUTS kernel from the adapted parameters.
          3. vmap the sampling across all chains in parallel.
             This compiles ONE XLA program and runs all chains
             simultaneously, fully utilising GPU parallelism.

        Parameters
        ----------
        loglike_fn : callable
            JIT-compiled log-likelihood (no prior).
        logprior_fn : callable
            JIT-compiled log-prior.
        theta_init : dict[str, Array]
            Initial parameter values with correct shapes.
        rng_key : Array

        Returns
        -------
        SamplingResult
        """
        try:
            import blackjax
        except ImportError as exc:
            raise ImportError(
                "BlackJAX is required for NUTS sampling.\n"
                "Install: pip install git+https://github.com/blackjax-devs/blackjax"
            ) from exc

        # Dispatch: VI-preconditioned path is structurally different
        # (whitened target, chains-from-q init, no mass-matrix adapt in
        # z-space) so it lives in its own method.
        if self.vi is not None:
            return self._run_whitened(
                loglike_fn, logprior_fn, theta_init, rng_key,
            )

        n_dims = self._n_dims(theta_init)
        theta_template = theta_init
        bounds = self.bounds if self.bounds is not None else {}

        if self.verbose:
            print(
                f"BlackJAX NUTS  |  n_dims={n_dims}  "
                f"num_warmup={self.num_warmup}  "
                f"num_samples={self.num_samples}  "
                f"num_chains={self.num_chains}  "
                f"dense_mass={self.dense_mass}"
            )
            if bounds:
                for k, (a, b) in bounds.items():
                    print(f"  Bounded: {k} -> sigmoid({a}, {b})")
            else:
                print("  No bounded parameters (consider passing bounds=...)")

        # ── Build reparameterisation layer ─────────────────────────────
        lo, hi, is_bounded = _build_transforms(bounds, theta_template)
        n_bounded = int(jnp.sum(is_bounded))

        if self.verbose and n_bounded > 0:
            print(f"  Reparameterised {n_bounded}/{n_dims} "
                  f"bounded dimensions via sigmoid/logit")

        # ── Unconstrained log-posterior ────────────────────────────────
        @jax.jit
        def logposterior_flat(x):
            theta_flat = _to_constrained(x, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            lnl = loglike_fn(theta)
            lnp = logprior_fn(theta)
            lnj = _log_jacobian(x, lo, hi, is_bounded)
            return lnl + lnp + lnj

        @jax.jit
        def loglike_constrained(x):
            """Log-likelihood in unconstrained coords (for diagnostics)."""
            theta_flat = _to_constrained(x, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            return loglike_fn(theta)

        # ── Initial position in unconstrained space ────────────────────
        x_init_flat = self._flatten(theta_init)
        x_init = _to_unconstrained(x_init_flat, lo, hi, is_bounded)

        t_start = time.perf_counter()

        # ==============================================================
        #  Phase 1: Warmup (single chain, adapts step size + mass matrix)
        # ==============================================================
        if self.verbose:
            print(f"\n  Warmup ({self.num_warmup} steps, "
                  f"adapting step size + {'dense' if self.dense_mass else 'diagonal'} mass matrix)...")

        # ``max_num_doublings`` forwards through window_adaptation to the
        # wrapped blackjax.nuts kernel used during warmup; it caps the
        # per-step leapfrog count at 2**max_num_doublings.  Must be passed
        # explicitly — BlackJAX otherwise falls back to its default of 10
        # (→ 1024 leapfrog steps), and at small adapted step sizes one
        # NUTS iteration can take seconds.
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logposterior_flat,
            target_acceptance_rate=self.target_acceptance,
            initial_step_size=self.initial_step_size,
            progress_bar=self.verbose,
            is_mass_matrix_diagonal=not self.dense_mass,
            max_num_doublings=self.max_num_doublings,
        )

        warmup_key, sample_key = jax.random.split(rng_key)

        _t0_warmup = time.perf_counter()
        (warmup_state, parameters), _ = warmup.run(
            warmup_key,
            x_init,
            num_steps=self.num_warmup,
        )
        jax.block_until_ready(warmup_state.position)
        _t_warmup = time.perf_counter() - _t0_warmup

        step_size = parameters['step_size']
        if self.verbose:
            print(f"    Adapted step size: {float(step_size):.4f}")
            if self.dense_mass:
                im = np.asarray(parameters['inverse_mass_matrix'])
                cond = np.linalg.cond(im)
                print(f"    Mass matrix condition number: {cond:.1f}")
            print(f"    Warmup wall time: {_t_warmup:.1f} s  "
                  f"(includes XLA compilation)")

        # ==============================================================
        #  Phase 2: Sampling
        #
        #  Strategy depends on the number of available devices:
        #
        #  Multi-GPU (n_devices >= n_chains):
        #    Use jax.pmap to run one chain per GPU in true parallel.
        #    Unlike vmap, pmap places each chain on a *separate device*,
        #    so each NUTS while_loop (tree building) runs independently
        #    with its own adaptive tree depth — no padding to max depth.
        #    This gives near-linear speedup with number of GPUs.
        #
        #  Single-GPU fallback:
        #    Run chains sequentially with a cached XLA kernel.
        #    The lax.scan compiles once on Chain 1 and is reused.
        # ==============================================================
        # ``parameters`` from window_adaptation already carries
        # ``max_num_doublings``, so it must not be passed a second time
        # here (``TypeError: got multiple values for keyword argument``).
        nuts_kernel = blackjax.nuts(
            logposterior_flat,
            **parameters,
        ).step

        def _nuts_step(state, key):
            state, info = nuts_kernel(key, state)
            return state, (state, info)

        @jax.jit
        def _run_one_chain(init_state, chain_key):
            """Run num_samples NUTS steps from init_state."""
            keys = jax.random.split(chain_key, self.num_samples)
            final_state, (states, infos) = jax.lax.scan(
                _nuts_step, init_state, keys
            )
            return states, infos

        chain_keys = jax.random.split(sample_key, self.num_chains)

        n_devices = len(jax.devices())
        use_pmap = (n_devices >= self.num_chains) and (self.num_chains > 1)

        if self.verbose:
            print(f"\n  Available devices: {n_devices}  |  "
                  f"Chains: {self.num_chains}  |  "
                  f"Strategy: {'pmap (one chain per GPU)' if use_pmap else 'sequential'}")

        all_chain_positions = []  # (num_samples, n_dims) per chain
        all_loglikelihoods = []
        all_divergences = []
        all_infos = []
        _t_sample_chains = []
        _t_postproc_chains = []

        if use_pmap:
            # ── Multi-GPU parallel path ──────────────────────────────
            # Replicate the warmup state across devices and pmap the scan.
            if self.verbose:
                print(f"\n  Running {self.num_chains} chains in parallel "
                      f"across {self.num_chains} GPUs...", flush=True)

            # pmap expects a leading device axis.  Replicate warmup_state
            # across chains (each chain starts from the same adapted state).
            def _replicate_state(state, n):
                """Replicate a NUTS state across n devices."""
                return jax.tree.map(
                    lambda x: jnp.broadcast_to(x, (n,) + x.shape), state
                )

            pmap_init = _replicate_state(warmup_state, self.num_chains)

            # pmap the chain runner.  axis_name is used for potential
            # cross-device reductions (not needed here, but good practice).
            @jax.pmap
            def _run_chains_pmap(init_state, chain_key):
                keys = jax.random.split(chain_key, self.num_samples)
                final_state, (states, infos) = jax.lax.scan(
                    _nuts_step, init_state, keys
                )
                return states, infos

            _t0_sample = time.perf_counter()
            pmap_states, pmap_infos = _run_chains_pmap(
                pmap_init, chain_keys
            )
            # Block until all GPUs finish
            jax.block_until_ready(pmap_states.position)
            _t_sample_total = time.perf_counter() - _t0_sample

            if self.verbose:
                print(f"    All chains wall time: {_t_sample_total:.1f} s  "
                      f"(parallel across {self.num_chains} GPUs)")

            # ── Post-processing: unpack pmap results ─────────────────
            _t0_pp = time.perf_counter()
            for ci in range(self.num_chains):
                x_chain = pmap_states.position[ci]  # (num_samples, n_dims)
                theta_chain = jax.vmap(
                    lambda x: _to_constrained(x, lo, hi, is_bounded)
                )(x_chain)
                all_chain_positions.append(theta_chain)

                _chunk = min(50, self.num_samples)
                _lnl_parts = []
                for _i in range(0, self.num_samples, _chunk):
                    _lnl_parts.append(
                        jax.vmap(loglike_constrained)(x_chain[_i:_i + _chunk])
                    )
                chain_lnl = jnp.concatenate(_lnl_parts, axis=0)
                jax.block_until_ready(chain_lnl)
                all_loglikelihoods.append(chain_lnl)

                chain_infos = jax.tree.map(lambda x: x[ci], pmap_infos)
                all_infos.append(chain_infos)
                if hasattr(chain_infos, 'is_divergent'):
                    n_div = int(jnp.sum(chain_infos.is_divergent))
                    all_divergences.append(n_div)
                    if self.verbose and n_div > 0:
                        print(f"    Chain {ci+1}: {n_div} divergent transitions!")
                else:
                    all_divergences.append(0)

            _t_pp = time.perf_counter() - _t0_pp
            _t_sample_chains = [_t_sample_total / self.num_chains] * self.num_chains
            _t_postproc_chains = [_t_pp / self.num_chains] * self.num_chains

            if self.verbose:
                print(f"    Post-processing: {_t_pp:.1f} s")

        else:
            # ── Single-GPU sequential path ───────────────────────────
            for chain_idx in range(self.num_chains):
                if self.verbose:
                    print(f"\n  Chain {chain_idx + 1}/{self.num_chains}  "
                          f"({self.num_samples} draws)...", flush=True)

                _t0_sample = time.perf_counter()
                states, infos = _run_one_chain(warmup_state, chain_keys[chain_idx])
                jax.block_until_ready(states.position)
                _t_sample = time.perf_counter() - _t0_sample
                _t_sample_chains.append(_t_sample)

                if self.verbose:
                    print(f"    Sampling wall time: {_t_sample:.1f} s")

                _t0_pp = time.perf_counter()

                x_chain = states.position
                theta_chain = jax.vmap(
                    lambda x: _to_constrained(x, lo, hi, is_bounded)
                )(x_chain)
                all_chain_positions.append(theta_chain)

                _chunk = min(50, self.num_samples)
                _lnl_parts = []
                for _i in range(0, self.num_samples, _chunk):
                    _lnl_parts.append(
                        jax.vmap(loglike_constrained)(x_chain[_i:_i + _chunk])
                    )
                chain_lnl = jnp.concatenate(_lnl_parts, axis=0)
                jax.block_until_ready(chain_lnl)
                all_loglikelihoods.append(chain_lnl)

                all_infos.append(infos)
                if hasattr(infos, 'is_divergent'):
                    n_div = int(jnp.sum(infos.is_divergent))
                    all_divergences.append(n_div)
                    if self.verbose and n_div > 0:
                        print(f"    WARNING: {n_div} divergent transitions!")
                else:
                    all_divergences.append(0)

                _t_pp = time.perf_counter() - _t0_pp
                _t_postproc_chains.append(_t_pp)

                if self.verbose:
                    print(f"    Post-processing wall time: {_t_pp:.1f} s")

        n_like_calls = self.num_chains * (self.num_warmup + self.num_samples)
        wall_time = time.perf_counter() - t_start

        # ── Timing summary ────────────────────────────────────────────
        _t_merge_start = time.perf_counter()

        if self.verbose:
            _t_sample_total = sum(_t_sample_chains)
            _t_pp_total = sum(_t_postproc_chains)
            print("\n  " + "=" * 60)
            print("  NUTS timing breakdown")
            print("  " + "=" * 60)
            print(f"  {'Phase':<40s}  {'Time':>10s}")
            print("  " + "-" * 60)
            print(f"  {'Warmup (incl. XLA compilation)':<40s}  {_t_warmup:>9.1f}s")
            if use_pmap:
                print(f"  {'Sampling (pmap, all chains parallel)':<40s}  "
                      f"{_t_sample_total:>9.1f}s")
            else:
                for ci in range(self.num_chains):
                    print(f"  {'  Chain ' + str(ci+1) + ' sampling':<40s}  "
                          f"{_t_sample_chains[ci]:>9.1f}s")
                print(f"  {'Sampling total (sequential)':<40s}  {_t_sample_total:>9.1f}s")
            print(f"  {'Post-processing total':<40s}  {_t_pp_total:>9.1f}s")
            print("  " + "-" * 60)
            print(f"  {'TOTAL':<40s}  {wall_time:>9.1f}s")
            if use_pmap:
                seq_est = _t_sample_total * self.num_chains
                print(f"  {'Estimated sequential time':<40s}  {seq_est:>9.1f}s")
                print(f"  {'pmap speedup':<40s}  {seq_est / _t_sample_total:>9.1f}x")
            print("  " + "=" * 60)

        # ── Merge chains (constrained space) ──────────────────────────
        merged_flat = jnp.concatenate(all_chain_positions, axis=0)

        merged_samples = {}
        idx = 0
        for name, template in theta_template.items():
            size = int(jnp.size(template))
            arr = merged_flat[:, idx:idx + size]
            if arr.shape[-1] == 1:
                arr = jnp.squeeze(arr, axis=-1)
            merged_samples[name] = arr
            idx += size

        merged_lnl = jnp.concatenate(all_loglikelihoods, axis=0)

        total_divergences = sum(all_divergences)
        _t_merge = time.perf_counter() - _t_merge_start

        if self.verbose:
            print(
                f"\n  Done  ({wall_time:.1f} s total,  "
                f"merge {_t_merge:.1f} s,  "
                f"{n_like_calls:,} likelihood calls"
                f"{f', {total_divergences} divergences' if total_divergences else ''})"
            )

        # ── Convergence diagnostics ───────────────────────────────────
        if self.verbose and self.num_chains >= 2:
            self._print_diagnostics(all_chain_positions, theta_template)

        return SamplingResult(
            samples=merged_samples,
            log_evidence=float("nan"),
            log_evidence_err=float("nan"),
            log_weights=jnp.zeros_like(merged_lnl),
            log_likelihoods=merged_lnl,
            param_names=list(theta_init.keys()),
            n_likelihood_calls=n_like_calls,
            wall_time_s=wall_time,
            sampler_name="blackjax.nuts",
            raw={
                "num_chains": self.num_chains,
                "num_warmup": self.num_warmup,
                "num_samples": self.num_samples,
                "total_divergences": total_divergences,
                "dense_mass": self.dense_mass,
                "per_chain_constrained": all_chain_positions,
                "per_chain_infos": all_infos,
                "used_pmap": use_pmap,
                "n_devices": n_devices,
            },
        )

    # ------------------------------------------------------------------
    # VI-preconditioned path
    # ------------------------------------------------------------------

    def _resolve_vi(self, logpost_flat, x_init, rng_key):
        """Turn ``self.vi`` into a :class:`TrainedMap`.

        Accepts a string name, a :class:`VariationalMap` instance, or an
        already-:class:`TrainedMap`.  Strings / untrained maps are passed
        through :func:`train_vi` using ``self.vi_kwargs``.
        """
        from .vi import (
            VariationalMap, TrainedMap, make_vi_map, train_vi,
        )
        target = self.vi
        kwargs = dict(self.vi_kwargs)

        if isinstance(target, TrainedMap):
            return target

        if isinstance(target, str):
            # Partition kwargs: map constructor vs. train_vi
            train_keys = {"num_steps", "batch_size", "lr0"}
            ctor_kwargs = {k: v for k, v in kwargs.items()
                           if k not in train_keys}
            train_kwargs = {k: v for k, v in kwargs.items()
                            if k in train_keys}
            vi_map = make_vi_map(target, **ctor_kwargs)
        elif isinstance(target, VariationalMap):
            vi_map = target
            train_kwargs = {k: v for k, v in kwargs.items()
                            if k in {"num_steps", "batch_size", "lr0"}}
        else:
            raise TypeError(
                f"Unrecognised vi argument type {type(target).__name__}."
                " Expected None, str, VariationalMap, or TrainedMap."
            )

        return train_vi(
            vi_map, logpost_flat, x_init, rng_key,
            verbose=self.verbose, **train_kwargs,
        )

    def _run_whitened(
        self,
        loglike_fn: Callable[[dict[str, Array]], Array],
        logprior_fn: Callable[[dict[str, Array]], Array],
        theta_init: dict[str, Array],
        rng_key: Array,
    ) -> SamplingResult:
        """Run NUTS on a whitened target built from a VI transport map.

        The unconstrained target is
            log p_x(x)  with  x = f(z),  z ~ N(0, I).
        NUTS samples in z-space against
            log p_z(z) = log p_x(f(z)) + log|det df/dz|.
        """
        import blackjax

        n_dims = self._n_dims(theta_init)
        theta_template = theta_init
        bounds = self.bounds if self.bounds is not None else {}

        if self.verbose:
            print(
                f"BlackJAX NUTS (VI-preconditioned)  |  n_dims={n_dims}  "
                f"num_warmup={self.num_warmup}  "
                f"num_samples={self.num_samples}  "
                f"num_chains={self.num_chains}  "
                f"dense_mass={self.dense_mass}  "
                f"vi={getattr(self.vi, 'name', self.vi)!r}"
            )

        lo, hi, is_bounded = _build_transforms(bounds, theta_template)

        @jax.jit
        def logpost_flat(x):
            theta_flat = _to_constrained(x, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            return (loglike_fn(theta) + logprior_fn(theta)
                    + _log_jacobian(x, lo, hi, is_bounded))

        x_init = _to_unconstrained(
            self._flatten(theta_init), lo, hi, is_bounded,
        )

        vi_key, warm_key, init_key, sample_key = jax.random.split(rng_key, 4)

        # ── Train (or accept) the VI transport map ─────────────────────
        trained = self._resolve_vi(logpost_flat, x_init, vi_key)
        self.trained_map = trained
        vi_map = trained.vi_map
        params = trained.params
        aux = trained.aux

        @jax.jit
        def logpost_z(z):
            x, logdet = vi_map.forward(z, params, aux)
            return logpost_flat(x) + logdet

        # Sanity: log p_z(0) should be finite if the map was initialised
        # reasonably.  Fails loudly here before we burn warmup time on it.
        _lp0 = float(logpost_z(jnp.zeros(n_dims)))
        if self.verbose:
            print(f"  log p_z(0) = {_lp0:.3f}")

        # ── Warmup (step size + optional diagonal mass) ────────────────
        # Cap per-step leapfrog count via max_num_doublings; without it
        # blackjax defaults to 10 and a tiny adapted step size translates
        # into seconds per NUTS iteration.
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logpost_z,
            target_acceptance_rate=self.target_acceptance,
            initial_step_size=self.initial_step_size,
            progress_bar=self.verbose,
            is_mass_matrix_diagonal=not self.dense_mass,
            max_num_doublings=self.max_num_doublings,
        )

        t_start = time.perf_counter()
        _t0 = time.perf_counter()
        (warmup_state, parameters), _ = warmup.run(
            warm_key, jnp.zeros(n_dims), num_steps=self.num_warmup,
        )
        jax.block_until_ready(warmup_state.position)
        t_warmup = time.perf_counter() - _t0
        step_size = float(parameters['step_size'])
        if self.verbose:
            print(f"    Adapted step size: {step_size:.4f}")
            print(f"    Warmup wall time:  {t_warmup:.1f} s")

        # ── Per-chain initialisation from q(theta) ─────────────────────
        # Start each chain from an independent sample of the variational
        # distribution (paper Sec. 4.1.2): z_c ~ N(0, I) (which induces
        # independent q(x) draws through the map).
        init_zs = jax.random.normal(init_key, (self.num_chains, n_dims))
        # ``parameters`` already contains ``max_num_doublings`` — don't
        # double-pass it (would raise TypeError).
        nuts_full = blackjax.nuts(
            logpost_z,
            **parameters,
        )
        init_states = jax.vmap(nuts_full.init)(init_zs)
        kernel = nuts_full.step

        def _step(state, key):
            state, info = kernel(key, state)
            return state, (state, info)

        chain_keys = jax.random.split(sample_key, self.num_chains)
        n_devices = len(jax.devices())
        use_pmap = (n_devices >= self.num_chains) and (self.num_chains > 1)

        if self.verbose:
            print(f"\n  Available devices: {n_devices}  |  "
                  f"Chains: {self.num_chains}  |  "
                  f"Strategy: {'pmap' if use_pmap else 'sequential scan'}")

        if use_pmap:
            @jax.pmap
            def _run_chains(init_state, key):
                keys = jax.random.split(key, self.num_samples)
                _, (states, infos) = jax.lax.scan(_step, init_state, keys)
                return states, infos
            _t0 = time.perf_counter()
            states, infos = _run_chains(init_states, chain_keys)
            jax.block_until_ready(states.position)
            t_sample = time.perf_counter() - _t0
        else:
            @jax.jit
            def _run_one(init_state, key):
                keys = jax.random.split(key, self.num_samples)
                _, (st, inf) = jax.lax.scan(_step, init_state, keys)
                return st, inf
            all_st, all_inf = [], []
            _t0 = time.perf_counter()
            for ci in range(self.num_chains):
                init_ci = jax.tree.map(lambda a: a[ci], init_states)
                st, inf = _run_one(init_ci, chain_keys[ci])
                jax.block_until_ready(st.position)
                all_st.append(st); all_inf.append(inf)
            t_sample = time.perf_counter() - _t0
            states = jax.tree.map(lambda *a: jnp.stack(a, axis=0), *all_st)
            infos = jax.tree.map(lambda *a: jnp.stack(a, axis=0), *all_inf)

        # ── Push forward z -> x_unconstrained -> theta_constrained ─────
        # Both operations vmapped over (chain, sample).
        @jax.jit
        def _z_to_flat_constrained(z):
            x_unc, _ = vi_map.forward(z, params, aux)
            return _to_constrained(x_unc, lo, hi, is_bounded)

        z_all = states.position                             # (C, S, D)
        theta_flat_all = jax.vmap(jax.vmap(_z_to_flat_constrained))(z_all)

        # ── Per-draw log-likelihood in constrained space (for weights) ─
        @jax.jit
        def _loglike_at_z(z):
            x_unc, _ = vi_map.forward(z, params, aux)
            theta_flat = _to_constrained(x_unc, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            return loglike_fn(theta)

        # Compute in chunks per chain, then concat.
        chain_lnls = []
        chunk = min(50, self.num_samples)
        for ci in range(self.num_chains):
            parts = []
            for k0 in range(0, self.num_samples, chunk):
                zc = z_all[ci, k0:k0 + chunk]
                parts.append(jax.vmap(_loglike_at_z)(zc))
            lnl_ci = jnp.concatenate(parts, axis=0)
            jax.block_until_ready(lnl_ci)
            chain_lnls.append(lnl_ci)
        merged_lnl = jnp.concatenate(chain_lnls, axis=0)

        # ── Merge chains; split into per-parameter arrays ──────────────
        merged_flat = theta_flat_all.reshape(
            self.num_chains * self.num_samples, n_dims,
        )
        merged_samples = {}
        idx = 0
        for name, template in theta_template.items():
            size = int(jnp.size(template))
            arr = merged_flat[:, idx:idx + size]
            if arr.shape[-1] == 1:
                arr = jnp.squeeze(arr, axis=-1)
            merged_samples[name] = arr
            idx += size

        # ── Divergence count (across all chains) ──────────────────────
        total_divergences = 0
        per_chain_constrained = []
        per_chain_infos = []
        for ci in range(self.num_chains):
            inf_ci = jax.tree.map(lambda a: a[ci], infos)
            per_chain_infos.append(inf_ci)
            per_chain_constrained.append(theta_flat_all[ci])
            if hasattr(inf_ci, "is_divergent"):
                total_divergences += int(jnp.sum(inf_ci.is_divergent))

        wall_time = time.perf_counter() - t_start
        n_like_calls = self.num_chains * (self.num_warmup + self.num_samples)

        if self.verbose:
            print(f"\n  Whitened-NUTS timing:")
            print(f"    VI train: {trained.train_time_s:.2f} s  "
                  f"({len(trained.losses)} iters)")
            print(f"    Warmup:   {t_warmup:.2f} s  "
                  f"({self.num_warmup} steps, eps={step_size:.4f})")
            print(f"    Sample:   {t_sample:.2f} s  "
                  f"({self.num_chains} x {self.num_samples} draws)")
            if total_divergences:
                print(f"    Divergences: {total_divergences} "
                      f"(consider raising target_acceptance or num_warmup)")

        if self.verbose and self.num_chains >= 2:
            self._print_diagnostics(per_chain_constrained, theta_template)

        return SamplingResult(
            samples=merged_samples,
            log_evidence=float("nan"),
            log_evidence_err=float("nan"),
            log_weights=jnp.zeros_like(merged_lnl),
            log_likelihoods=merged_lnl,
            param_names=list(theta_init.keys()),
            n_likelihood_calls=n_like_calls,
            wall_time_s=wall_time,
            sampler_name="blackjax.nuts+vi",
            raw={
                "num_chains": self.num_chains,
                "num_warmup": self.num_warmup,
                "num_samples": self.num_samples,
                "total_divergences": total_divergences,
                "dense_mass": self.dense_mass,
                "per_chain_constrained": per_chain_constrained,
                "per_chain_infos": per_chain_infos,
                "used_pmap": use_pmap,
                "n_devices": n_devices,
                "vi_map_name": getattr(vi_map, "name", "custom"),
                "vi_train_time_s": trained.train_time_s,
                "vi_losses": np.asarray(trained.losses),
                "trained_map": trained,
                "step_size": step_size,
                "warmup_time_s": t_warmup,
                "sample_time_s": t_sample,
            },
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_diagnostics(self, all_chain_samples, theta_template):
        """Print basic convergence diagnostics (R-hat, ESS)."""
        print("\n  Convergence diagnostics:")
        print(f"  {'Parameter':<25s}  {'R-hat':>8s}  {'ESS':>8s}")
        print("  " + "-" * 45)

        idx = 0
        for name, template in theta_template.items():
            size = int(jnp.size(template))
            if size == 1:
                chains = [np.asarray(s[:, idx]).ravel() for s in all_chain_samples]
                rhat = self._rhat(chains)
                ess = self._ess(chains)
                print(f"  {name:<25s}  {rhat:>8.4f}  {ess:>8.0f}")
            else:
                for k in range(size):
                    chains = [np.asarray(s[:, idx + k]).ravel() for s in all_chain_samples]
                    rhat = self._rhat(chains)
                    ess = self._ess(chains)
                    label = f"{name}[{k}]"
                    print(f"  {label:<25s}  {rhat:>8.4f}  {ess:>8.0f}")
            idx += size

    @staticmethod
    def _rhat(chains: list[np.ndarray]) -> float:
        """Compute split R-hat (Gelman-Rubin diagnostic)."""
        split_chains = []
        for c in chains:
            mid = len(c) // 2
            split_chains.append(c[:mid])
            split_chains.append(c[mid:])

        m = len(split_chains)
        n = min(len(c) for c in split_chains)
        if n < 2 or m < 2:
            return float("nan")

        chain_means = np.array([np.mean(c[:n]) for c in split_chains])
        chain_vars = np.array([np.var(c[:n], ddof=1) for c in split_chains])

        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)

        if W < 1e-30:
            return float("nan")

        var_hat = (1 - 1 / n) * W + B / n
        return float(np.sqrt(var_hat / W))

    @staticmethod
    def _ess(chains: list[np.ndarray]) -> float:
        """Bulk effective sample size (simple autocorrelation estimate)."""
        combined = np.concatenate(chains)
        n = len(combined)
        if n < 4:
            return float(n)

        mean = np.mean(combined)
        var = np.var(combined)
        if var < 1e-30:
            return float(n)

        max_lag = min(n // 2, 1000)
        centered = combined - mean
        acf = np.correlate(centered, centered, mode='full')
        acf = acf[n - 1:n - 1 + max_lag + 1] / (n * var)

        # Geyer's initial monotone sequence estimator (simplified)
        tau = 1.0
        for lag in range(1, max_lag):
            rho = acf[lag]
            if rho < 0.05:
                break
            tau += 2.0 * rho

        return float(n / tau)
