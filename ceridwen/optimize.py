"""MAP optimisation of a ``SedModel``: L-BFGS on ``fitSED``'s own log-posterior from many starts.

``map_fit(model)`` maximises ``ln L + ln prior`` -- the same jitted function ``fitSED`` samples,
built from the same per-observation likelihoods (``fit._likelihood_for``) and
``MultiObservationLikelihood.make_lnprobfn`` -- from ``n_starts`` prior draws (plus, by
default, ``model.theta_init``), and returns a :class:`MAPResult` whose ``theta`` is a dict with
the shapes of ``model.theta_init``, ready for ``SedModel(..., free_param_init=res.theta)`` or
``fitSED(..., optimize=True)``.

* The optimiser works in the unconstrained coordinates of the NUTS adapter: bounded priors
  (``fit._detect_bounds``) go through its sigmoid/logit map (``sampler/nuts.py``), the others are
  left as they are.  The objective carries **no** log-Jacobian: the map is only a change of the
  optimiser's coordinates, so its maximum is the MAP of ``p(theta)`` itself (NUTS samples
  ``ln p + ln|J|``, whose maximum is not the MAP).
* Each start is one jitted ``lax.while_loop`` with ``optax.lbfgs`` (zoom line search); the
  same compiled solver runs every start in turn.  A start stops when
  ``|f_k - f_{k-1}| <= ftol (1 + |f_k|)`` or after ``max_steps`` iterations.  Where a trial
  step makes the forward model non-finite, the loss is ``+inf`` and its gradient is zeroed, so
  the line search backtracks instead of poisoning the L-BFGS memory.
* Deterministic: the starts come from ``rng_key`` only, and the solver has no other randomness.

Prospector's equivalent is ``fit_model(optimize=True)`` -> ``run_minimize``
(``prospect/fitting/fitting.py:223-310``, ``nmin`` starts from ``minimizer_ball``: the current
theta plus ``nmin - 1`` prior draws).  Differences: exact JAX gradients and L-BFGS instead of
Levenberg-Marquardt on the chi vector (which omits the prior) or Powell; bounds handled by the
logit map instead of the prior returning -inf.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["map_fit", "MAPResult", "build_lnprob", "laplace_sigma"]


def build_lnprob(model):
    """Jitted ``theta -> ln L + ln prior``, built exactly as ``fitSED`` builds its likelihood."""
    from .fit import _likelihood_for
    from .likelihood.likelihood import MultiObservationLikelihood, make_lnprobfn

    if not model.observations:
        raise ValueError("map_fit: the model has no observations")
    obs_dict = model.obs_dict
    keys = tuple(obs_dict)
    likelihoods = tuple(_likelihood_for(obs_dict[k], model.param_names, model=model)
                        for k in keys)
    if getattr(model, "_eline_system", None) is not None:
        from .likelihood.eline_marginal import refuse_outlier_with_elines
        refuse_outlier_with_elines(keys, likelihoods, model._eline_system.keys)
    multi = MultiObservationLikelihood(keys=keys, likelihoods=likelihoods)
    return make_lnprobfn(obs_dict, model, model, multi)


@dataclass
class MAPResult:
    """Result of :func:`map_fit`.

    theta       : dict -- best theta (constrained), shapes of ``model.theta_init`` (NumPy)
    lnp         : float -- ln posterior at ``theta``
    lnp_starts  : (n,) -- final ln posterior of every start
    theta_starts: dict -- the starting points, {name: (n, *shape)}
    n_steps     : (n,) -- L-BFGS iterations per start
    grad_norm   : (n,) -- final |grad| in the unconstrained space, per start
    best_start  : int  -- index of the winning start
    wall_time   : float -- seconds, compile included
    bounds      : dict -- the bounds used for the logit map
    """
    theta: dict
    lnp: float
    lnp_starts: np.ndarray
    theta_starts: dict
    n_steps: np.ndarray
    grad_norm: np.ndarray
    best_start: int
    wall_time: float
    bounds: dict = field(default_factory=dict)

    @property
    def free_param_init(self) -> dict:
        """``theta`` as JAX float64 arrays, for ``SedModel(..., free_param_init=...)``."""
        return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in self.theta.items()}

    def summary(self) -> str:
        ok = np.isfinite(self.lnp_starts)
        lines = [f"MAP: ln p = {self.lnp:.4f} (start {self.best_start} of "
                 f"{self.lnp_starts.size}; {int(ok.sum())} finite; {self.wall_time:.1f} s)"]
        for k, v in self.theta.items():
            lines.append(f"  {k:>22} = {np.array2string(np.asarray(v), precision=5)}")
        return "\n".join(lines)


def _prior_starts(model, n_starts, rng_key, include_init):
    """``n_starts`` prior draws per parameter, preceded by ``model.theta_init`` if asked."""
    starts = {}
    for name, init in model.theta_init.items():
        if name not in model.priors:
            raise ValueError(f"map_fit: parameter {name!r} has no prior; every free parameter "
                             "needs one to draw starting points")
        rng_key, sub = jax.random.split(rng_key)
        init = jnp.asarray(init, dtype=jnp.float64)
        draws = jnp.asarray(model.priors[name].sample(sub, shape=(n_starts, *init.shape)),
                            dtype=jnp.float64)
        starts[name] = jnp.concatenate([init[None], draws]) if include_init else draws
    return starts


def map_fit(model, n_starts: int = 16, rng_key=None, *, include_init: bool = True,
            max_steps: int = 3000, ftol: float = 1e-12, bounds: dict | None = None,
            lnprob=None) -> MAPResult:
    """Maximise ``fitSED``'s log-posterior of ``model`` with L-BFGS from many starts.

    Parameters
    ----------
    model : SedModel -- every free parameter needs a prior (starts are prior draws)
    n_starts : int -- number of prior draws to start from (>= 1)
    rng_key : PRNGKey -- draws the starts; default ``PRNGKey(0)``.  Same key, same result.
    include_init : bool -- also start from ``model.theta_init`` (as start 0), as Prospector's
        ``minimizer_ball`` does
    max_steps, ftol : L-BFGS stopping rule (see the module docstring)
    bounds : dict -- {name: (low, high)} for the logit map; default ``fit._detect_bounds(model)``
    lnprob : callable -- a prebuilt ``build_lnprob(model)``, to share its compilation

    Returns
    -------
    MAPResult -- ``.theta`` is usable as ``free_param_init``
    """
    import optax
    from .fit import _detect_bounds
    from .sampler.nuts import _build_transforms, _to_constrained, _to_unconstrained

    t0 = time.perf_counter()
    n_starts = int(n_starts)
    if n_starts < 1:
        raise ValueError(f"map_fit: n_starts must be >= 1, got {n_starts}")
    if int(max_steps) < 1:
        raise ValueError(f"map_fit: max_steps must be >= 1, got {max_steps}")
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    if lnprob is None:
        lnprob = build_lnprob(model)
    template = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in model.theta_init.items()}
    if bounds is None:
        bounds = _detect_bounds(model)
    else:
        unknown = sorted(set(bounds) - set(template))
        if unknown:
            raise ValueError(f"map_fit: bounds given for {unknown}, which are not free "
                             f"parameters ({list(template)})")
    lo, hi, isb = _build_transforms(bounds, template)

    def unflatten(flat):
        out, i = {}, 0
        for k, v in template.items():
            out[k] = flat[i:i + v.size].reshape(v.shape)
            i += v.size
        return out

    def theta_of(x):
        return unflatten(_to_constrained(x, lo, hi, isb))

    def _loss(x):
        v = -lnprob(theta_of(x))
        return jnp.where(jnp.isfinite(v), v, jnp.inf)

    @jax.custom_jvp
    def loss(x):
        return _loss(x)

    @loss.defjvp
    def _loss_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        v, g = jax.value_and_grad(_loss)(x)
        return v, jnp.where(jnp.isfinite(g), g, 0.0) @ dx

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(loss)

    @jax.jit
    def run_one(x0):
        def step(carry):
            x, state, f_now, _ = carry
            value, grad = value_and_grad(x, state=state)
            updates, state = opt.update(grad, state, x, value=value, grad=grad,
                                        value_fn=loss)
            return optax.apply_updates(x, updates), state, value, f_now

        def cont(carry):
            _, state, f_now, f_prev = carry
            it = optax.tree_utils.tree_get(state, "count")
            moving = jnp.abs(f_prev - f_now) > ftol * (1.0 + jnp.abs(f_now))
            return (it < 2) | ((it < max_steps) & moving)

        x, state, _, _ = jax.lax.while_loop(
            cont, step, (x0, opt.init(x0), jnp.inf, -jnp.inf))
        return (x, -loss(x), optax.tree_utils.tree_get(state, "count"),
                optax.tree_utils.tree_l2_norm(jax.grad(loss)(x)))

    # Starts run one after another through one compiled solver: under vmap a batched
    # while_loop runs every lane until the slowest start ends and evaluates both branches of
    # the zoom line search (the recipe measured 1018 s for 16 starts vs ~5 s per start).
    starts = _prior_starts(model, n_starts, rng_key, include_init)
    n_total = next(iter(starts.values())).shape[0]
    # flatten in template key order explicitly (never rely on a dict's order under jit/vmap)
    flat0 = jnp.concatenate([starts[k].reshape(n_total, -1) for k in template], axis=1)
    x0s = jax.vmap(lambda f: _to_unconstrained(f, lo, hi, isb))(flat0)
    out = [run_one(x0s[i]) for i in range(n_total)]
    lnps = np.array([float(o[1]) for o in out])
    good = np.isfinite(lnps)
    if not good.any():
        raise RuntimeError("map_fit: every start ended at a non-finite ln posterior")
    best = int(np.argmax(np.where(good, lnps, -np.inf)))
    theta = {k: np.asarray(v) for k, v in theta_of(out[best][0]).items()}
    return MAPResult(theta=theta, lnp=float(lnps[best]), lnp_starts=lnps,
                     theta_starts={k: np.asarray(v) for k, v in starts.items()},
                     n_steps=np.array([int(o[2]) for o in out]),
                     grad_norm=np.array([float(o[3]) for o in out]),
                     best_start=best, wall_time=time.perf_counter() - t0,
                     bounds=dict(bounds))


def laplace_sigma(lnprob, theta, template=None):
    """Laplace 1-sigma widths ``sqrt(diag(H^-1))``, ``H`` = Hessian of ``-lnprob`` in the
    constrained, flattened parameters at ``theta``: ``(widths shaped like theta, covariance)``.
    A local approximation, wrong near a prior edge and for non-Gaussian posteriors."""
    template = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in (template or theta).items()}

    def unflatten(flat):
        out, i = {}, 0
        for k, v in template.items():
            out[k] = flat[i:i + v.size].reshape(v.shape)
            i += v.size
        return out

    t0 = jnp.concatenate([jnp.ravel(jnp.asarray(theta[k], dtype=jnp.float64))
                          for k in template])
    H = jax.hessian(lambda t: -lnprob(unflatten(t)))(t0)
    cov = jnp.linalg.inv(H)
    sig = jnp.sqrt(jnp.diag(cov))
    return {k: np.asarray(v) for k, v in unflatten(sig).items()}, np.asarray(cov)
