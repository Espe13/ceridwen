"""NUTS sampler adapter with sigmoid/logit reparameterisation of bounded
parameters and optional variational preconditioning."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .runner import SamplerAdapter, SamplingResult

Array = jax.Array


def _build_transforms(bounds: dict[str, tuple[float, float]],
                      theta_template: dict[str, Array]):
    """Return (lo, hi, is_bounded) per flat element; unbounded elements get dummy finite bounds (0, 1)."""
    lo_list, hi_list, bounded_list = [], [], []
    for name, template in theta_template.items():
        size = int(jnp.size(template))
        if name in bounds:
            a, b = bounds[name]
            lo_list.append(jnp.full(size, a))
            hi_list.append(jnp.full(size, b))
            bounded_list.append(jnp.ones(size, dtype=bool))
        else:
            lo_list.append(jnp.full(size, 0.0))
            hi_list.append(jnp.full(size, 1.0))
            bounded_list.append(jnp.zeros(size, dtype=bool))

    lo = jnp.concatenate(lo_list)
    hi = jnp.concatenate(hi_list)
    is_bounded = jnp.concatenate(bounded_list)
    return lo, hi, is_bounded


def _to_constrained(x: Array, lo: Array, hi: Array, is_bounded: Array) -> Array:
    """Unconstrained x -> constrained flat theta; lo/hi must be finite for all elements."""
    sig = jax.nn.sigmoid(x)
    constrained = lo + (hi - lo) * sig
    return jnp.where(is_bounded, constrained, x)


def _to_unconstrained(theta_flat: Array, lo: Array, hi: Array, is_bounded: Array) -> Array:
    """Constrained flat theta -> unconstrained x; lo/hi must be finite for all elements."""
    frac = (theta_flat - lo) / (hi - lo)
    frac = jnp.clip(frac, 1e-7, 1.0 - 1e-7)
    unconstrained = jnp.log(frac / (1.0 - frac))
    return jnp.where(is_bounded, unconstrained, theta_flat)


def _log_jacobian(x: Array, lo: Array, hi: Array, is_bounded: Array) -> Array:
    """Summed log |det J| of the sigmoid transform (unbounded elements contribute 0)."""
    log_jac_elem = jnp.where(
        is_bounded,
        jnp.log(hi - lo) - jax.nn.softplus(x) - jax.nn.softplus(-x),
        0.0,
    )
    return jnp.sum(log_jac_elem)


class BlackJAXNUTSAdapter(SamplerAdapter):
    """NUTS adapter with window adaptation; bounded (uniform-prior) parameters
    are sampled in sigmoid/logit space, and an optional VI map (``vi``) whitens
    the target before sampling.

    Parameters
    ----------
    num_warmup : int -- adaptation steps per chain; default 1500 (200 with ``vi``)
    num_samples : int -- post-warmup draws per chain
    initial_step_size : float -- leapfrog step before adaptation; default 0.01 (0.5 with ``vi``)
    target_acceptance : float -- dual-averaging target, default 0.95
    max_num_doublings : int -- max tree depth (2**n leapfrog steps)
    dense_mass : bool -- full inverse mass matrix; default True (False with ``vi``)
    bounds : dict -- name -> (low, high); None auto-detects from model priors
    vi : None | 'tril' | 'iaf' | VariationalMap | TrainedMap -- variational preconditioning
    vi_kwargs : dict -- map constructor kwargs plus ``num_steps``, ``batch_size``, ``lr0`` for training
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
        self.trained_map = None

    def _n_dims(self, theta_init: dict[str, Array]) -> int:
        return sum(int(jnp.size(v)) for v in theta_init.values())

    def _flatten(self, theta: dict[str, Array]) -> Array:
        return jnp.concatenate([jnp.ravel(v) for v in theta.values()])

    def _unflatten(self, x: Array, theta_template: dict[str, Array]) -> dict[str, Array]:
        out = {}
        idx = 0
        for name, template in theta_template.items():
            size = int(jnp.size(template))
            out[name] = x[idx:idx + size].reshape(template.shape)
            idx += size
        return out

    def run(
        self,
        loglike_fn: Callable[[dict[str, Array]], Array],
        logprior_fn: Callable[[dict[str, Array]], Array],
        theta_init: dict[str, Array],
        rng_key: Array,
    ) -> SamplingResult:
        """Run NUTS (one warmup, then all chains) and return a ``SamplingResult`` in constrained space."""
        try:
            import blackjax
        except ImportError as exc:
            raise ImportError(
                "BlackJAX is required for NUTS sampling.\n"
                "Install: pip install git+https://github.com/blackjax-devs/blackjax"
            ) from exc

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

        lo, hi, is_bounded = _build_transforms(bounds, theta_template)
        n_bounded = int(jnp.sum(is_bounded))

        if self.verbose and n_bounded > 0:
            print(f"  Reparameterised {n_bounded}/{n_dims} "
                  f"bounded dimensions via sigmoid/logit")

        @jax.jit
        def logposterior_flat(x):
            theta_flat = _to_constrained(x, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            lnl = loglike_fn(theta)
            lnp = logprior_fn(theta)
            lnj = _log_jacobian(x, lo, hi, is_bounded)
            return lnl + lnp + lnj

        @jax.jit
        def _prior_and_jacobian(x):
            theta_flat = _to_constrained(x, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            return logprior_fn(theta) + _log_jacobian(x, lo, hi, is_bounded)

        def _loglike_of(states_):
            return states_.logdensity - jax.vmap(_prior_and_jacobian)(states_.position)

        x_init_flat = self._flatten(theta_init)
        x_init = _to_unconstrained(x_init_flat, lo, hi, is_bounded)

        t_start = time.perf_counter()

        if self.verbose:
            print(f"\n  Warmup ({self.num_warmup} steps, "
                  f"adapting step size + {'dense' if self.dense_mass else 'diagonal'} mass matrix)...")

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

        nuts_kernel = blackjax.nuts(
            logposterior_flat,
            **parameters,
        ).step

        def _nuts_step(state, key):
            state, info = nuts_kernel(key, state)
            return state, (state, info)

        @jax.jit
        def _run_one_chain(init_state, chain_key):
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

        all_chain_positions = []
        all_loglikelihoods = []
        all_divergences = []
        all_infos = []
        _t_sample_chains = []
        _t_postproc_chains = []

        if use_pmap:
            if self.verbose:
                print(f"\n  Running {self.num_chains} chains in parallel "
                      f"across {self.num_chains} GPUs...", flush=True)

            def _replicate_state(state, n):
                return jax.tree.map(
                    lambda x: jnp.broadcast_to(x, (n,) + x.shape), state
                )

            pmap_init = _replicate_state(warmup_state, self.num_chains)

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
            jax.block_until_ready(pmap_states.position)
            _t_sample_total = time.perf_counter() - _t0_sample

            if self.verbose:
                print(f"    All chains wall time: {_t_sample_total:.1f} s  "
                      f"(parallel across {self.num_chains} GPUs)")

            _t0_pp = time.perf_counter()
            for ci in range(self.num_chains):
                x_chain = pmap_states.position[ci]
                theta_chain = jax.vmap(
                    lambda x: _to_constrained(x, lo, hi, is_bounded)
                )(x_chain)
                all_chain_positions.append(theta_chain)
                all_loglikelihoods.append(
                    _loglike_of(jax.tree.map(lambda x: x[ci], pmap_states)))

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
                all_loglikelihoods.append(_loglike_of(states))

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

    def _resolve_vi(self, logpost_flat, x_init, rng_key):
        """Turn ``self.vi`` (name, untrained map, or TrainedMap) into a TrainedMap."""
        from .vi import (
            VariationalMap, TrainedMap, make_vi_map, train_vi,
        )
        target = self.vi
        kwargs = dict(self.vi_kwargs)

        if isinstance(target, TrainedMap):
            return target

        if isinstance(target, str):
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
        """Run NUTS in z-space on log p_x(f(z)) + log|det df/dz|, z ~ N(0, I), using the VI map f."""
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

        trained = self._resolve_vi(logpost_flat, x_init, vi_key)
        self.trained_map = trained
        vi_map = trained.vi_map
        params = trained.params
        aux = trained.aux

        @jax.jit
        def logpost_z(z):
            x, logdet = vi_map.forward(z, params, aux)
            return logpost_flat(x) + logdet

        if self.verbose:
            print(f"  log p_z(0) = {float(logpost_z(jnp.zeros(n_dims))):.3f}")

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

        init_zs = jax.random.normal(init_key, (self.num_chains, n_dims))
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

        @jax.jit
        def _z_to_flat_constrained(z):
            x_unc, _ = vi_map.forward(z, params, aux)
            return _to_constrained(x_unc, lo, hi, is_bounded)

        z_all = states.position                             # (n_chains, n_samples, n_dims)
        theta_flat_all = jax.vmap(jax.vmap(_z_to_flat_constrained))(z_all)

        @jax.jit
        def _nonlike_at_z(z):
            x_unc, logdet = vi_map.forward(z, params, aux)
            theta_flat = _to_constrained(x_unc, lo, hi, is_bounded)
            theta = self._unflatten(theta_flat, theta_template)
            return logprior_fn(theta) + _log_jacobian(x_unc, lo, hi, is_bounded) + logdet

        merged_lnl = (states.logdensity
                      - jax.vmap(jax.vmap(_nonlike_at_z))(z_all)).reshape(-1)

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

    def _print_diagnostics(self, all_chain_samples, theta_template):
        """Print split R-hat and ESS per scalar parameter."""
        print("\n  Convergence diagnostics:")
        print(f"  {'Parameter':<25s}  {'R-hat':>8s}  {'ESS':>8s}")
        print("  " + "-" * 45)

        idx = 0
        all_chain_samples = [np.asarray(s) for s in all_chain_samples]
        for name, template in theta_template.items():
            size = int(jnp.size(template))
            if size == 1:
                chains = [s[:, idx].ravel() for s in all_chain_samples]
                rhat = self._rhat(chains)
                ess = self._ess(chains)
                print(f"  {name:<25s}  {rhat:>8.4f}  {ess:>8.0f}")
            else:
                for k in range(size):
                    chains = [s[:, idx + k].ravel() for s in all_chain_samples]
                    rhat = self._rhat(chains)
                    ess = self._ess(chains)
                    label = f"{name}[{k}]"
                    print(f"  {label:<25s}  {rhat:>8.4f}  {ess:>8.0f}")
            idx += size

    @staticmethod
    def _rhat(chains: list[np.ndarray]) -> float:
        """Split R-hat."""
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
        """Bulk effective sample size from a truncated autocorrelation sum."""
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

        tau = 1.0
        for lag in range(1, max_lag):
            rho = acf[lag]
            if rho < 0.05:
                break
            tau += 2.0 * rho

        return float(n / tau)
