"""Nested sampling adapter: live-point initialisation from priors, the NS
loop with periodic checkpoints (resumable), and evidence/weight extraction."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .runner import SamplerAdapter, SamplingResult

Array = jax.Array


# relative tolerance of the resume check on the saved live points' ln L (see _load_resume)
RESUME_LOGL_RTOL = 1e-6


class BlackJAXNestedSamplerAdapter(SamplerAdapter):
    """Nested-sampling adapter driving the NSS kernel with a Ceridwen ``SedModel``.

    Parameters
    ----------
    priors : dict[str, Prior] -- every free parameter needs a proper prior
    num_inner_steps : int -- inner MCMC steps per iteration; default n_dims * 5
    num_delete : int -- live points removed per iteration; default max(1, num_live // 5), must be < num_live
    logZ_tol : float -- stop when ln(Z_live / Z) < logZ_tol; default -5
    checkpoint_interval_s : float -- seconds between checkpoints; <= 0 disables
    checkpoint_dir : str -- falls back to $CERIDWEN_CHECKPOINT_DIR, then $CERIDWEN_RESCUE_DIR, else off
    resume_from : str -- a periodic checkpoint (``ns_checkpoint_<pid>.pkl``) to continue from
        instead of starting afresh.  The run continues with the saved live points, dead points
        and rng key, so it reproduces the uninterrupted run at the same key.  The checkpoint
        must come from the same model: the settings, the parameter shapes and the
        log-likelihood of the saved live points are checked before sampling resumes.
    """

    #: format of the ``resume`` block written into periodic checkpoints
    RESUME_VERSION = 1

    def __init__(
        self,
        priors          : dict,
        num_live        : int   = 500,
        num_inner_steps : Optional[int] = None,
        num_delete      : Optional[int] = None,
        logZ_tol        : float = -5.0,
        verbose         : bool  = True,
        checkpoint_interval_s : float = 1200.0,
        checkpoint_dir        : Optional[str] = None,
        resume_from           : Optional[str] = None,
    ):
        self.priors          = dict(priors)
        self.num_live        = int(num_live)
        self._num_inner_steps = num_inner_steps
        self._num_delete      = num_delete
        self.logZ_tol        = float(logZ_tol)
        self.verbose         = bool(verbose)
        self.checkpoint_interval_s = float(checkpoint_interval_s)
        self._checkpoint_dir       = checkpoint_dir
        self.resume_from           = None if resume_from is None else str(resume_from)
        if self.resume_from is not None and not os.path.isfile(self.resume_from):
            raise FileNotFoundError(f"resume_from={self.resume_from!r}: no such checkpoint")

    def _n_dims(self, theta_init: dict[str, Array]) -> int:
        """Total scalar degrees of freedom."""
        return sum(int(jnp.size(v)) for v in theta_init.values())

    def _sample_prior(
        self,
        theta_init : dict[str, Array],
        rng_key    : Array,
    ) -> dict[str, Array]:
        """Draw ``num_live`` live points from the priors; returns {name: (num_live, *param_shape)}."""
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
            expected_shape = init_val.shape

            particles[name] = prior.sample(sub, shape=(self.num_live, *expected_shape))

        return particles

    @staticmethod
    def _finalise_dead(live, dead_list, ns_utils):
        """Merge live into dead points; returns (positions_dict, loglikelihood, loglikelihood_birth)."""
        import inspect as _inspect
        if "update_info" in _inspect.signature(ns_utils.finalise).parameters:
            dead = ns_utils.finalise(live, dead_list, update_info=False)
        else:
            dead = ns_utils.finalise(live, dead_list)
        _dp = dead.particles
        if isinstance(_dp, dict):
            return _dp, dead.loglikelihood, dead.loglikelihood_birth
        if hasattr(_dp, "position"):
            return _dp.position, _dp.loglikelihood, _dp.loglikelihood_birth
        positions = dead.position if hasattr(dead, "position") else _dp
        return positions, dead.loglikelihood, dead.loglikelihood_birth

    def _resolve_ckpt_dir(self):
        """Checkpoint dir: explicit arg -> $CERIDWEN_CHECKPOINT_DIR -> $CERIDWEN_RESCUE_DIR -> None."""
        return (self._checkpoint_dir
                or os.environ.get("CERIDWEN_CHECKPOINT_DIR")
                or os.environ.get("CERIDWEN_RESCUE_DIR"))

    def _dump_snapshot(self, ckpt_dir, live, dead_list, ns_utils, logZ,
                       *, tag, partial, finalised=None, resume=None):
        """Atomically pickle a finalised snapshot; returns the path or None (never raises).
        ``resume`` (periodic checkpoints) adds the raw sampler state needed by ``resume_from``."""
        try:
            import pickle as _pickle
            import numpy as _np
            pos, logl, logl_birth = (finalised if finalised is not None
                                     else self._finalise_dead(live, dead_list, ns_utils))
            os.makedirs(ckpt_dir, exist_ok=True)
            fname = (f"ns_raw_dead_{os.getpid()}.pkl" if tag == "rescue"
                     else f"ns_checkpoint_{os.getpid()}.pkl")
            path = os.path.join(ckpt_dir, fname)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                _pickle.dump({
                    "positions": {k: _np.asarray(v) for k, v in pos.items()},
                    "loglikelihood": _np.asarray(logl),
                    "loglikelihood_birth": _np.asarray(logl_birth),
                    "logZ": float(logZ),
                    "n_dead": int(_np.asarray(logl).shape[0]),
                    "partial": bool(partial),
                    **({"resume": resume} if resume is not None else {}),
                }, fh)
            os.replace(tmp, path)
            return path
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [{tag}] WARNING: snapshot failed: {exc}")
            return None

    def _run_config(self, theta_init, num_inner_steps, num_delete):
        """What must match between a checkpoint and the run that resumes it."""
        return {"num_live": self.num_live, "num_delete": int(num_delete),
                "num_inner_steps": int(num_inner_steps),
                "params": [(k, tuple(int(d) for d in jnp.shape(v)))
                           for k, v in theta_init.items()]}

    def _resume_block(self, live, dead_list, rng_key, n_iter, n_like_calls, elapsed, config):
        import numpy as _np
        return {"version": self.RESUME_VERSION,
                "live": jax.device_get(live),
                "dead": jax.device_get(list(dead_list)),
                "rng_key": _np.asarray(rng_key),
                "n_iter": int(n_iter),
                "n_like_calls": int(n_like_calls),
                "elapsed_s": float(elapsed),
                "config": config}

    def _load_resume(self, config, loglike_fn):
        """Read ``self.resume_from`` and check it belongs to this run; returns the resume block."""
        import numpy as _np
        ck = self.load_checkpoint(self.resume_from)
        res = ck.get("resume") if isinstance(ck, dict) else None
        if res is None:
            raise ValueError(
                f"{self.resume_from} holds no sampler state (a rescue pickle, or a checkpoint "
                "written before resuming was supported): it can be read with "
                "load_checkpoint() for its dead points, but a run cannot continue from it")
        if res.get("version") != self.RESUME_VERSION:
            raise ValueError(f"{self.resume_from}: resume format {res.get('version')!r}, "
                             f"this version reads {self.RESUME_VERSION}")
        diff = {k: (res["config"].get(k), v) for k, v in config.items()
                if res["config"].get(k) != v}
        if diff:
            raise ValueError(
                f"{self.resume_from} was written by a different run; differing settings "
                + ", ".join(f"{k}: checkpoint {a!r} vs now {b!r}" for k, (a, b) in diff.items()))
        # the same data and forward model: the saved live points must have the same ln L now,
        # to the forward model's float32-contraction precision (re-evaluated at another batch
        # width, ln L moves ~1e-8 relative; the T3 spectra tolerance is 1e-6)
        pts = res["live"].particles
        logl_saved = _np.asarray(pts.loglikelihood)
        logl_now = _np.asarray(jax.jit(jax.vmap(loglike_fn))(pts.position))
        bad = ~_np.isclose(logl_now, logl_saved, rtol=RESUME_LOGL_RTOL, atol=1e-6)
        if bad.any():
            i = int(_np.argmax(_np.abs(logl_now - logl_saved)))
            raise ValueError(
                f"{self.resume_from}: the log-likelihood of the saved live points differs "
                f"from this model's for {int(bad.sum())}/{bad.size} points (worst: "
                f"{logl_saved[i]!r} saved vs {logl_now[i]!r} now); the checkpoint belongs to "
                "another model or data set")
        return res

    @staticmethod
    def load_checkpoint(path):
        """Load a checkpoint/rescue pickle: {positions, loglikelihood, loglikelihood_birth, logZ, n_dead, partial}."""
        import pickle as _pickle
        with open(path, "rb") as fh:
            return _pickle.load(fh)

    def run(
        self,
        loglike_fn  : Callable[[dict[str, Array]], Array],
        logprior_fn : Callable[[dict[str, Array]], Array],
        theta_init  : dict[str, Array],
        rng_key     : Array,
    ) -> SamplingResult:
        """Run nested sampling and return a ``SamplingResult``; ``logprior_fn`` must be proper."""
        try:
            import blackjax
            import blackjax.ns.utils as ns_utils
        except ImportError as exc:
            raise ImportError(
                "BlackJAX with nested sampling (blackjax.ns) is required.\n"
                "Install: pip install -U 'blackjax>=1.6'"
            ) from exc

        import tqdm

        n_dims          = self._n_dims(theta_init)
        num_inner_steps = (self._num_inner_steps
                           if self._num_inner_steps is not None
                           else n_dims * 5)
        num_delete      = (self._num_delete
                           if self._num_delete is not None
                           else max(1, self.num_live // 5))

        if self.verbose:
            print(
                f"BlackJAX NSS  |  n_dims={n_dims}  "
                f"num_live={self.num_live}  "
                f"num_inner_steps={num_inner_steps}  "
                f"num_delete={num_delete}"
            )

        _config = self._run_config(theta_init, num_inner_steps, num_delete)
        _resume = (self._load_resume(_config, loglike_fn)
                   if self.resume_from is not None else None)
        if _resume is None:
            rng_key, prior_key = jax.random.split(rng_key)
            particles = self._sample_prior(theta_init, prior_key)

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

        if _resume is None:
            live = init_fn(particles)
        else:
            # the saved state replaces the prior draw, init and the first iterations
            live = jax.tree_util.tree_map(jnp.asarray, _resume["live"])
            rng_key = jnp.asarray(_resume["rng_key"])

        def _logz_fields(state):
            if hasattr(state, "logZ") and hasattr(state, "logZ_live"):
                return state.logZ, state.logZ_live
            ig = getattr(state, "integrator", None)
            if ig is not None and hasattr(ig, "logZ") and hasattr(ig, "logZ_live"):
                return ig.logZ, ig.logZ_live
            inner = getattr(state, "sampler_state", None)
            if inner is not None and hasattr(inner, "logZ") and hasattr(inner, "logZ_live"):
                return inner.logZ, inner.logZ_live
            raise AttributeError(
                f"Cannot locate logZ/logZ_live on {type(state).__name__} "
                f"(fields {[f for f in dir(state) if not f.startswith('_')]}); "
                "check your BlackJAX version.")

        def _logz(state):
            a, b = jax.device_get(_logz_fields(state))
            return float(a), float(b)

        _get_logZ = lambda s: _logz(s)[0]

        if self.verbose:
            _t1 = time.perf_counter()
            lz, lzl = _logz(live)
            print(f"  [timing] init_fn done    ({_t1 - _t0:.1f} s)  "
                  f"logZ={lz:.4f}  logZ_live={lzl:.4f}", flush=True)

        dead_list    = [] if _resume is None else list(_resume["dead"])
        n_like_calls = 0 if _resume is None else _resume["n_like_calls"]
        _elapsed0    = 0.0 if _resume is None else _resume["elapsed_s"]
        t_start      = time.perf_counter()
        if _resume is not None and self.verbose:
            print(f"  [resume] {self.resume_from}: iteration {_resume['n_iter']}, "
                  f"{len(dead_list) * num_delete} dead points", flush=True)

        _ckpt_dir   = self._resolve_ckpt_dir()
        _ckpt_on    = bool(_ckpt_dir) and self.checkpoint_interval_s > 0
        _last_ckpt  = t_start
        if _ckpt_on and self.verbose:
            print(f"  [checkpoint] every {self.checkpoint_interval_s:.0f} s "
                  f"-> {_ckpt_dir}", flush=True)
        elif self.verbose:
            print("  [checkpoint] off (set checkpoint_dir= or $CERIDWEN_CHECKPOINT_DIR "
                  "to write recoverable snapshots)", flush=True)

        logZ, logZ_live = _logz(live)
        with tqdm.tqdm(desc=f"NS  logZ={logZ:.1f}", unit=" dead",
                       disable=not self.verbose) as pbar:
            _iter = 0 if _resume is None else _resume["n_iter"]
            while logZ_live - logZ >= self.logZ_tol:
                rng_key, subkey = jax.random.split(rng_key)
                if _iter == (0 if _resume is None else _resume["n_iter"]) and self.verbose:
                    pbar.write("  [step_fn] compiling the step kernel (one-time JIT)")
                _t_iter = time.perf_counter()
                live, dead_info = step_fn(subkey, live)
                logZ, logZ_live = _logz(live)
                _dt_iter = time.perf_counter() - _t_iter
                _iter += 1
                dead_list.append(dead_info)
                n_like_calls += num_delete * num_inner_steps
                pbar.update(num_delete)
                pbar.set_description(f"NS  logZ={logZ:.2f}  dlogZ={logZ_live - logZ:.2f}",
                                     refresh=False)
                pbar.set_postfix({"s/iter": f"{_dt_iter:.1f}"}, refresh=False)

                if _ckpt_on and (time.perf_counter() - _last_ckpt
                                 >= self.checkpoint_interval_s):
                    _p = self._dump_snapshot(
                        _ckpt_dir, live, dead_list, ns_utils,
                        logZ, tag="checkpoint", partial=True,
                        resume=self._resume_block(
                            live, dead_list, rng_key, _iter, n_like_calls,
                            _elapsed0 + time.perf_counter() - t_start, _config))
                    _last_ckpt = time.perf_counter()
                    if _p and self.verbose:
                        pbar.write(f"  [checkpoint] iter {_iter}: {_p}")

        wall_time = _elapsed0 + time.perf_counter() - t_start
        if self.verbose:
            print(
                f"  Converged  logZ = {_get_logZ(live):.3f}  "
                f"({wall_time:.1f} s,  {n_like_calls:,} likelihood calls)"
            )

        _dead_positions, _dead_logl, _dead_logl_birth = self._finalise_dead(
            live, dead_list, ns_utils)

        _rescue_dir = self._resolve_ckpt_dir()
        if _rescue_dir:
            self._dump_snapshot(_rescue_dir, live, dead_list, ns_utils,
                                _get_logZ(live), tag="rescue", partial=False,
                                finalised=(_dead_positions, _dead_logl, _dead_logl_birth))

        # dead set is in deletion order, NOT sorted by logL; weights follow sample order
        import numpy as np
        from .ns_weights import nested_log_weights, log_evidence_from_weights
        _logl_np    = np.asarray(_dead_logl)
        _birth_np   = np.asarray(_dead_logl_birth)
        _lw_np      = nested_log_weights(_logl_np, _birth_np)
        log_weights = jnp.asarray(_lw_np)
        log_Z       = None
        log_Z_err   = float("nan")
        try:
            from anesthetic import NestedSamples

            _raw   = _dead_positions
            _names = [n for n in _raw if n in theta_init]
            _cols  = [np.asarray(_raw[n]).reshape(_logl_np.shape[0], -1) for n in _names]
            _data  = np.hstack(_cols)
            _ns    = NestedSamples(
                _data,
                logL       = _logl_np,
                logL_birth = _birth_np,
                logzero    = float("nan"),
            )
            log_Z     = float(_ns.logZ())
            log_Z_err = float(_ns.logZ(12).std())
        except Exception:
            pass
        if log_Z is None:
            log_Z = log_evidence_from_weights(_lw_np)

        samples = {}
        for name in theta_init:
            arr = jnp.asarray(_dead_positions[name])
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
            raw                   = {
                "positions": _dead_positions,
                "loglikelihood": _dead_logl,
                "loglikelihood_birth": _dead_logl_birth,
            },
        )
