"""Nested sampling adapter: live-point initialisation from priors, the NS
loop with periodic checkpoints (resumable), and evidence/weight extraction."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .runner import SamplerAdapter, SamplingResult
from .priors import sample_for_parameter

Array = jax.Array

#: inner slice kernels of :class:`BlackJAXNestedSamplerAdapter` (``slice_kernel=``)
SLICE_KERNELS = ("stock", "carry", "lanes")


def stepping_out_carry(rng_key, in_slice, width, max_expansions):
    """Neal (2003) Fig. 3 stepping-out with the slice test in the loop body.

    Drop-in for ``blackjax.mcmc.slice.stepping_out``: same random numbers, same sequence of
    ``in_slice`` evaluations, bitwise-equal bracket.  The stock loops call ``in_slice`` in
    their condition, and a vmapped ``lax.while_loop`` evaluates the condition a second time
    inside its body, so every expansion costs two batched likelihood evaluations.  Here the
    body evaluates the new edge once and carries the result.

    Ported from Liu Hao's CERIDWEN fork (potatoist314/ceridwen, commit f973c10).
    """
    u_key, jk_key = jax.random.split(rng_key)
    u = jax.random.uniform(u_key)
    left = -width * u
    right = left + width
    v = jax.random.uniform(jk_key)
    j = jnp.floor(max_expansions * v).astype(int)
    k = (max_expansions - 1) - j

    def cond(carry):
        _, n, inside = carry
        return inside & (n > 0)

    def step(sign):
        def body(carry):
            edge, n, _ = carry
            edge = edge + sign * width
            return edge, n - 1, in_slice(edge)
        return body

    left, jl, _ = jax.lax.while_loop(cond, step(-1), (left, j, in_slice(left)))
    right, kr, _ = jax.lax.while_loop(cond, step(+1), (right, k, in_slice(right)))
    return left, right, (j - jl) + (k - kr), lambda t: jnp.asarray(True)


_LEFT, _RIGHT, _SHRINK = 0, 1, 2


def lane_update(logprior_fn, loglikelihood_fn, num_inner_steps, num_delete,
                max_expansions=10, max_shrinkage=100, width=1.0, free_moves=8,
                lookahead=4):
    """Inner update of the NS kernel: every particle's slice chain as one lane.

    ``blackjax.nss`` replaces ``num_delete`` particles per iteration.  Each
    particle (a lane) runs ``num_inner_steps`` slice steps, and each slice
    step runs three ``lax.while_loop``s: two stepping-out loops and one
    shrink loop.  Under ``vmap`` a while loop runs until its slowest lane
    stops, so at every slice step every lane pays for the longest loop of
    any lane.

    Here the lanes share one loop.  Each round evaluates one batch of
    ``num_delete`` candidates, one per lane; the lane's phase (left edge,
    right edge, shrink) selects its candidate, and a lane that completes a
    slice step starts its next step in the next round.  This is the
    finite-state-machine vectorisation of Dance et al. (2025,
    arXiv:2503.17405).  Two further rules remove rounds:

    * A candidate below the slice level of the prior is outside the slice
      whatever its likelihood.  Each round a lane first moves past up to
      ``free_moves`` such candidates (and stored ones outside the slice,
      and stepping-out edges with no expansion left, whose result does not
      change the step) without a likelihood evaluation.  The lane's next
      candidates are worked out in one pass, each on the assumption that
      the ones before it are outside the slice, and their prior is
      evaluated as one batch.
    * Once some lanes have finished, their batch slots evaluate candidates
      that the running lanes need later if their current candidates are
      outside the slice: the next edge, then the shrink proposals (up to
      ``lookahead - 1`` per lane, nearest first, then lanes with the most
      slice steps left).  A lane takes a stored result when its candidate
      is the same slice step, the same point on the slice line and the
      same position.  When that result is inside the slice, the lane also
      evaluates its next candidate in the same round.

    The random numbers are drawn as BlackJAX draws them (the shrink
    proposals of a slice step in advance: their key moves once per
    proposal, whatever its outcome) and every candidate is evaluated with
    the same arithmetic in a batch of ``num_delete``, so the new particles
    and the per-step ``SliceInfo`` are bitwise equal to those of
    ``blackjax.nss``.

    Drop-in for ``blackjax.ns.from_mcmc.update_with_mcmc_take_last`` applied
    to ``blackjax.ns.nss.slice_constrained_step`` with the stepping-out
    interval and the covariance proposal.  Returns the ``num_delete`` new
    particles and a ``SliceInfo`` of shape ``(num_delete, num_inner_steps)``.

    Ported from Liu Hao's CERIDWEN fork (potatoist314/ceridwen: 16c7a90, a935a99, e8b7643,
    ea91453), unchanged.
    """
    from functools import partial
    from blackjax.mcmc.slice import SliceInfo
    from blackjax.ns.base import init_state_strategy
    from blackjax.ns.nss import sample_direction_from_covariance

    depth = free_moves + lookahead   # candidates looked at per lane and round
    slots = 2 * lookahead            # stored results per lane

    def update_function(rng_key, state, loglikelihood_0, cov):
        choice_key, sample_key = jax.random.split(rng_key)
        particles = state.particles
        evaluate = jax.vmap(partial(init_state_strategy, logprior_fn=logprior_fn,
                                    loglikelihood_fn=loglikelihood_fn,
                                    loglikelihood_birth=loglikelihood_0))

        # Start particles, selected as in update_with_mcmc_take_last.
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key, len(weights), shape=(num_delete,),
            p=weights / weights.sum(), replace=True)
        start = jax.tree.map(lambda x: x[start_idx], particles)
        _, unravel = ravel_pytree(jax.tree.map(lambda x: x[0], start.position))

        def lane_draws(rng_key, position):
            def draws(key):
                # The random numbers of one slice step: blackjax.mcmc.slice
                # build_kernel, _univariate_slice and stepping_out.
                prop_key, slice_key = jax.random.split(key)
                direction, _ = ravel_pytree(sample_direction_from_covariance(
                    prop_key, position, cov))
                level_key, interval_key, shrink_key = jax.random.split(slice_key, 3)
                log_u = jnp.log(jax.random.uniform(level_key))
                u_key, jk_key = jax.random.split(interval_key)
                left = -width * jax.random.uniform(u_key)
                right = left + width
                j = jnp.floor(max_expansions * jax.random.uniform(jk_key)).astype(int)
                k = (max_expansions - 1) - j
                return direction, jnp.stack([log_u, left, right]), jnp.stack([j, k]), shrink_key

            # A scan, as in BlackJAX, so the draws keep their batch structure.
            return jax.lax.map(draws, jax.random.split(rng_key, num_inner_steps))

        def shrink_draws(key):
            # blackjax.mcmc.slice.shrinkage: one split per proposal.
            def proposal(key, _):
                key, subkey = jax.random.split(key)
                return key, jax.random.uniform(subkey)
            return jax.lax.scan(proposal, key, None, length=max_shrinkage)[1]

        directions, reals, caps, shrink_keys = jax.vmap(lane_draws)(
            jax.random.split(sample_key, num_delete), start.position)
        shrink_u = jax.vmap(jax.vmap(shrink_draws))(shrink_keys)
        counter = caps.dtype.type
        lanes = jnp.arange(num_delete)

        # Per-lane functions, vmapped over the lane index l.
        def begin(l, i, logdensity):
            log_u, left, right = reals[l, i]
            return dict(phase=jnp.asarray(_LEFT), level=logdensity + log_u,
                        left=left, right=right,
                        n_left=caps[l, i, 0], n_right=caps[l, i, 1],
                        bracket_left=left, bracket_right=right,
                        n_expansions=counter(0), n_shrink=counter(0))

        def propose(l, i, s):
            u = shrink_u[l, i, jnp.minimum(s["n_shrink"], max_shrinkage - 1)]
            t = s["left"] + u * (s["right"] - s["left"])
            return jnp.where(s["phase"] == _LEFT, s["left"],
                             jnp.where(s["phase"] == _RIGHT, s["right"], t))

        def point(l, i, position, t):
            direction = unravel(directions[l, i])
            return jax.tree.map(lambda p, d: p + t * d, position, direction)

        def advance(l, active, i, lane, s, info, t, candidate):
            """One transition of the lane's state machine for an evaluated candidate."""
            in_left, in_right = s["phase"] == _LEFT, s["phase"] == _RIGHT
            in_shrink = s["phase"] == _SHRINK
            inside = ((candidate.logdensity >= s["level"])
                      & (candidate.loglikelihood > loglikelihood_0))

            # Stepping out (Neal 2003, Fig. 3).
            grow_left = in_left & inside & (s["n_left"] > 0)
            grow_right = in_right & inside & (s["n_right"] > 0)
            left = jnp.where(grow_left, s["left"] - width, s["left"])
            right = jnp.where(grow_right, s["right"] + width, s["right"])
            bracket_left = jnp.where(in_shrink, s["bracket_left"], left)
            bracket_right = jnp.where(in_shrink, s["bracket_right"], right)
            n_expansions = s["n_expansions"] + (grow_left | grow_right)
            phase = jnp.where(in_left & ~grow_left, _RIGHT,
                              jnp.where(in_right & ~grow_right, _SHRINK, s["phase"]))

            # Shrinkage (Neal 2003, Fig. 5).
            found = active & in_shrink & inside
            left = jnp.where(in_shrink & (t < 0.0), t, left)
            right = jnp.where(in_shrink & (t >= 0.0), t, right)
            n_shrink = s["n_shrink"] + in_shrink
            lane = jax.tree.map(
                lambda new, old: jnp.where(found, new, old), candidate, lane)
            done = active & in_shrink & (found | (n_shrink >= max_shrinkage))

            row = jnp.stack([found, n_expansions, n_shrink,
                             bracket_left, bracket_right]).astype(info.dtype)
            info = info.at[jnp.where(done, i, num_inner_steps)].set(row, mode="drop")
            running = dict(
                phase=phase, level=s["level"], left=left, right=right,
                n_left=s["n_left"] - grow_left, n_right=s["n_right"] - grow_right,
                bracket_left=bracket_left, bracket_right=bracket_right,
                n_expansions=n_expansions, n_shrink=n_shrink)
            running = jax.tree.map(lambda a, b: jnp.where(active, a, b), running, s)
            i = i + done
            fresh = begin(l, jnp.minimum(i, num_inner_steps - 1), lane.logdensity)
            s = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, running)
            return i, lane, s, info

        def no_expansion_left(s):
            # An edge evaluated with no expansion left does not change the step.
            return (((s["phase"] == _LEFT) & (s["n_left"] == 0))
                    | ((s["phase"] == _RIGHT) & (s["n_right"] == 0)))

        def path(l, i, lane, s, info):
            """The lane's next ``depth`` candidates, each on the assumption that
            the ones before it are outside the slice, the state before each,
            and whether an outside candidate ends the slice step."""
            outside = lane._replace(logdensity=jnp.full_like(lane.logdensity, -jnp.inf))
            k = jnp.minimum(i, num_inner_steps - 1)
            out = []
            for _ in range(depth):
                t = propose(l, k, s)
                i_next, _, s_next, _ = advance(l, True, i, lane, s, info, t, outside)
                out.append((s, t, point(l, k, lane.position, t), i_next > i))
                # Without the barrier XLA fuses the whole chain of states and
                # its compile time grows faster than the depth.
                s = jax.lax.optimization_barrier(s_next)
            return jax.tree.map(lambda *v: jnp.stack(v), *out)

        v_path = jax.vmap(path)
        v_advance = jax.vmap(advance)
        v_logprior = jax.vmap(jax.vmap(logprior_fn))

        def round_(carry):
            i, lane, s, info, stored = carry
            running = i < num_inner_steps
            states, t, x, ends = v_path(lanes, i, lane, s, info)
            logprior = v_logprior(x)
            level = s["level"][:, None]

            # A stored result for the same slice step, t and position.  The
            # position must match bitwise: XLA may contract p + t * d into an
            # FMA in one layout and not in another.
            match = ((stored["step"][:, None, :] == i[:, None, None])
                     & (stored["t"][:, None, :] == t[..., None]))
            for v, u in zip(jax.tree.leaves(stored["state"].position), jax.tree.leaves(x)):
                equal = v[:, None] == u[:, :, None]
                match &= jnp.all(equal.reshape(*match.shape, -1), axis=-1)
            hit = match.any(-1)
            found = jax.tree.map(lambda v: v[lanes[:, None], jnp.argmax(match, -1)],
                                 stored["state"])
            found_inside = (found.logdensity >= level) & (found.loglikelihood > loglikelihood_0)

            # Move past candidates below the prior level (outside the slice
            # whatever their likelihood), stored ones outside the slice and
            # edges with no expansion left (the step goes on whatever they
            # are), up to free_moves of them, within the slice step.
            spent = jax.vmap(jax.vmap(no_expansion_left))(states)
            free = (~(logprior >= level) | (hit & ~found_inside) | spent) & ~ends
            m = jnp.argmin(free & (jnp.arange(depth) < free_moves), axis=1)
            at = lambda tree: jax.tree.map(lambda v: v[lanes, m], tree)  # noqa: E731
            s, now_t, now_x, now_hit, now_found, now_inside = at(
                (states, t, x, hit, found, found_inside))

            # A stored result inside the slice: the lane takes it and
            # evaluates its next candidate in this round.
            hop = running & now_hit & now_inside
            step = i
            i, lane, s, info = v_advance(lanes, hop, i, lane, s, info, now_t, now_found)
            k = jnp.minimum(i, num_inner_steps - 1)
            hop_t = jax.vmap(propose)(lanes, k, s)
            hop_x = jax.vmap(point)(lanes, k, lane.position, hop_t)
            now_t = jnp.where(hop, hop_t, now_t)
            now_x = jax.tree.map(
                lambda a, b: jnp.where(jnp.expand_dims(hop, tuple(range(1, a.ndim))), a, b),
                hop_x, now_x)
            active = jnp.where(hop, i < num_inner_steps, running)
            now_hit = now_hit & ~hop
            evaluate_now = active & ~now_hit & ~(hop & jax.vmap(no_expansion_left)(s))

            # Idle slots: the next candidates of the running lanes on the same
            # path, nearest first, then lanes with the most slice steps left.
            ahead = jnp.arange(depth)[None, :] - m[:, None]
            before = jnp.cumsum(ends, axis=1) - ends
            same_step = before == before[lanes, m][:, None]
            wanted = ((running & ~hop)[:, None] & (ahead > 0) & (ahead < lookahead) & same_step
                      & (logprior >= level) & ~hit & ~spent)
            score = jnp.where(wanted, ahead * (num_inner_steps + 1) + step[:, None],
                              depth * (num_inner_steps + 1)).ravel()
            rank = jnp.empty_like(score).at[jnp.argsort(score)].set(
                jnp.arange(score.size)).reshape(wanted.shape)
            chosen = wanted & (rank < num_delete - evaluate_now.sum())
            idle = jnp.argsort(evaluate_now)     # slots without a candidate first
            slot = jnp.where(chosen, idle[jnp.minimum(rank, num_delete - 1)], num_delete)

            batch = jax.tree.map(
                lambda a, b: a.at[slot.ravel()].set(
                    b.reshape(-1, *b.shape[2:]), mode="drop"),
                now_x, x)
            evaluated = evaluate(batch)
            candidate = jax.tree.map(
                lambda a, b: jnp.where(jnp.expand_dims(now_hit, tuple(range(1, a.ndim))), a, b),
                now_found, evaluated)

            # Store every result under its lane, slice step and t (at most
            # lookahead per lane and round, so no two share a slot).
            take = jnp.concatenate([evaluate_now[:, None], chosen], axis=1)
            source = jnp.concatenate([lanes[:, None], slot], axis=1)
            where_to = (stored["next"][:, None] + jnp.cumsum(take, axis=1) - 1) % slots
            where_to = jnp.where(take, where_to, slots)
            put = lambda old, new: old.at[lanes[:, None], where_to].set(new, mode="drop")  # noqa: E731
            stored = dict(
                step=put(stored["step"], jnp.concatenate(
                    [i[:, None], jnp.broadcast_to(step[:, None], chosen.shape)], axis=1)),
                t=put(stored["t"], jnp.concatenate([now_t[:, None], t], axis=1)),
                state=jax.tree.map(
                    lambda old, new: put(old, new[jnp.minimum(source, num_delete - 1)]),
                    stored["state"], evaluated),
                next=stored["next"] + take.sum(1, dtype=stored["next"].dtype))

            i, lane, s, info = v_advance(lanes, active, i, lane, s, info, now_t, candidate)
            return i, lane, s, info, stored

        i = jnp.zeros(num_delete, dtype=jnp.int32)
        s = jax.vmap(begin)(lanes, i, start.logdensity)
        info = jnp.zeros((num_delete, num_inner_steps, 5), reals.dtype)
        stored = dict(
            step=jnp.full((num_delete, slots), -1, dtype=i.dtype),
            t=jnp.zeros((num_delete, slots), reals.dtype),
            state=jax.tree.map(
                lambda v: jnp.zeros((num_delete, slots, *v.shape[1:]), v.dtype), start),
            next=jnp.zeros(num_delete, dtype=i.dtype))
        _, lane, _, info, _ = jax.lax.while_loop(
            lambda c: jnp.any(c[0] < num_inner_steps), round_, (i, start, s, info, stored))
        return lane, SliceInfo(
            is_accepted=info[..., 0] != 0,
            num_expansions=info[..., 1].astype(caps.dtype),
            num_shrink=info[..., 2].astype(caps.dtype),
            bracket_left=info[..., 3], bracket_right=info[..., 4])

    return update_function


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
    slice_kernel : {"stock", "carry", "lanes"} -- inner slice kernel.  ``"stock"`` (default) is the
        unchanged ``blackjax.nss``.  ``"carry"`` assembles the same kernel with
        :func:`stepping_out_carry`, which evaluates each stepping-out edge once instead of
        twice under ``vmap``; same random numbers and evaluations, so on CPU the samples,
        evidence and call counts are bitwise equal to ``"stock"``.  ``"lanes"`` runs every
        particle's chain of slice steps as one lane of a shared loop (:func:`lane_update`;
        Dance et al. 2025): lanes no longer wait for the slowest one inside each slice step,
        candidates below the prior slice level skip the likelihood, and idle batch slots
        evaluate look-ahead candidates.  Bitwise equal to ``"stock"`` on CPU; on GPU the
        likelihood batches differ in composition, so float rounding can make the trajectories
        separate after many iterations (statistically equivalent, not bitwise).
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
        slice_kernel          : str = "stock",
    ):
        if slice_kernel not in SLICE_KERNELS:
            raise ValueError(f"slice_kernel={slice_kernel!r}: choose one of {SLICE_KERNELS}")
        self.slice_kernel    = slice_kernel
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

            try:
                particles[name] = sample_for_parameter(prior, sub, self.num_live,
                                                       expected_shape)
            except ValueError as exc:
                raise ValueError(f"prior of {name!r}: {exc}") from None

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
        """What must match between a checkpoint and the run that resumes it.  ``slice_kernel``
        is not part of it: the kernels make the same moves with the same random numbers, so a
        checkpoint may be continued with any of them (bitwise equal on CPU)."""
        return {"num_live": self.num_live, "num_delete": int(num_delete),
                "num_inner_steps": int(num_inner_steps),
                "params": [(k, tuple(int(d) for d in jnp.shape(v)))
                           for k, v in theta_init.items()]}

    def _build_nested_sampler(self, loglike_fn, logprior_fn, num_inner_steps, num_delete):
        """The NSS transition kernel for ``self.slice_kernel``: ``"stock"`` is
        ``blackjax.nss``; ``"carry"`` assembles the kernel as
        ``blackjax.ns.nss.as_top_level_api`` does, with :func:`stepping_out_carry` as the
        interval procedure; ``"lanes"`` replaces the inner update with :func:`lane_update`
        (blackjax >= 1.6, whose ``ns`` package is that of commit f73e12956)."""
        import blackjax
        if self.slice_kernel == "stock":
            return blackjax.nss(
                logprior_fn      = logprior_fn,
                loglikelihood_fn = loglike_fn,
                num_delete       = num_delete,
                num_inner_steps  = num_inner_steps,
            )
        from functools import partial
        from blackjax.mcmc.slice import build_kernel as build_slice_kernel
        from blackjax.ns import adaptive, base, from_mcmc, nss
        init_state_fn = partial(base.init_state_strategy,
                                logprior_fn=logprior_fn, loglikelihood_fn=loglike_fn)
        if self.slice_kernel == "lanes":
            kernel = adaptive.build_kernel(
                partial(base.delete_fn, num_delete=num_delete),
                lane_update(logprior_fn, loglike_fn, num_inner_steps, num_delete),
                update_inner_kernel_params_fn=nss.live_covariance)
        else:
            constrained_step = nss.slice_constrained_step(
                init_state_fn,
                build_slice_kernel(interval=stepping_out_carry,
                                   max_expansions=10, max_shrinkage=100),
                nss.covariance_proposal)
            kernel = from_mcmc.build_kernel(
                constrained_step, num_inner_steps, nss.live_covariance, num_delete)

        def init_fn(position, rng_key=None):
            return adaptive.init(
                position, init_state_fn=jax.vmap(init_state_fn),
                update_inner_kernel_params_fn=nss.live_covariance,
                rng_key=rng_key)

        return blackjax.SamplingAlgorithm(init_fn, kernel)

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
        import numpy as np
        # seed of the ln Z error estimate (anesthetic draws it from numpy's global RNG): derived
        # from rng_key, so the same key gives the same error, and the global RNG is restored
        _err_seed = int(np.asarray(jax.random.randint(
            jax.random.fold_in(rng_key, 0x5EED), (), 0, 2 ** 31 - 1)))
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
                f"BlackJAX NSS ({self.slice_kernel})  |  n_dims={n_dims}  "
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

        nested_sampler = self._build_nested_sampler(
            loglike_fn, logprior_fn, num_inner_steps, num_delete)
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
                f"  Converged  (live-point logZ estimate {_get_logZ(live):.3f}; the final ln Z follows)  "
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
            _state = np.random.get_state()
            try:
                np.random.seed(_err_seed)
                log_Z_err = float(_ns.logZ(12).std())
            finally:
                np.random.set_state(_state)
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
