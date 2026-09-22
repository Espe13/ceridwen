"""
map_fit.py -- MAP optimisation before sampling (Prospector's ``nmin`` + LM/Powell), no package code.

What it does
------------
``map_fit(model)`` maximises the log-posterior that ``fitSED`` samples, from ``n_starts``
starting points, with L-BFGS (``optax.lbfgs``, zoom line search), and returns the best
``theta`` dict.  The dict has the shapes of ``model.theta_init`` and is usable directly as
``SedModel(..., free_param_init=res.theta)``; ``SedModel`` copies it into ``theta_init``
(``ceridwen/model/model.py:112-117``), which is where the NUTS adapter starts its chains
(``ceridwen/sampler/nuts.py:197-198``).  Nested sampling draws its live points from the prior
(``ceridwen/sampler/nested.py:59-76``) and ignores ``theta_init``, so a MAP start helps NUTS
only.

* The log-posterior is not re-implemented.  ``build_lnprob`` makes the same per-observation
  likelihoods as ``fitSED`` (``fit._likelihood_for``, ``ceridwen/fit.py:321-354``, combined as
  in ``ceridwen/fit.py:78-87``) and asks ``MultiObservationLikelihood.make_lnprobfn``
  (``ceridwen/likelihood/likelihood.py:519-561``) for the jitted ``theta -> ln L + ln prior``.
  The prior term is ``SedModel.log_prob`` = ``ln_prior`` (``ceridwen/model/model.py:397-407``).
* The optimiser works in the unconstrained space of the NUTS adapter: bounds from
  ``fit._detect_bounds`` (``ceridwen/fit.py:394-408``: Uniform / TopHat / bounded
  ClippedNormal), the sigmoid/logit map ``_build_transforms`` / ``_to_constrained`` /
  ``_to_unconstrained`` (``ceridwen/sampler/nuts.py:18-52``) and the adapter's flattening order
  (``BlackJAXNUTSAdapter._flatten`` / ``_unflatten``, ``nuts.py:121-131``).
* **The objective has no log-Jacobian.**  NUTS samples ``ln p(theta(x)) + ln|J(x)|``
  (``nuts.py:179-186``); the maximum of that density in ``x`` is not the MAP.  Here ``x`` is only
  a change of coordinates of the optimiser, so the maximum is the MAP of ``p(theta)`` itself.
* Starts: start 0 is ``model.theta_init`` and starts 1..n-1 are prior draws, as Prospector's
  ``minimizer_ball`` (the current ``theta`` plus ``nmin - 1`` prior draws).  Every free
  parameter needs a prior (as for nested sampling).  Each start is one jitted
  ``lax.while_loop`` (the same compiled solver for every start).
* ``laplace_sigma`` gives the Laplace (inverse-Hessian) 1-sigma widths at the MAP in the
  constrained parameters.  They are a local approximation: wrong near a prior edge and for
  non-Gaussian posteriors.

Prospector equivalent (``bd-j/prospector`` at ``a78d153``)
-----------------------------------------------------------
``prospect/fitting/fitting.py:201-205`` (``fit_model(optimize=True)`` runs ``run_minimize``
and sets the model to the best result), ``:223-310`` (``run_minimize``: ``min_method='lm'`` ->
``scipy.optimize.least_squares`` on the chi vector, ``'powell'`` -> ``scipy.optimize.minimize``
on ``-lnprob``; ``nmin`` starts from ``minimizer_ball``, ``:293``; best = lowest chi^2 / loss,
``:304-309``), ``prospect/fitting/minimizer.py:34-50`` (``minimizer_ball``).
Differences: L-BFGS with exact JAX gradients instead of LM (finite-difference Jacobian) or
Powell (derivative-free); the objective is always the full ln-posterior (Prospector's LM
minimises the chi^2 vector, which omits the prior); bounded parameters are handled by the
logit map instead of being clipped by the prior returning -inf.

Literature
----------
Liu & Nocedal 1989, Math. Programming 45, 503 (L-BFGS); Nocedal & Wright 2006, Numerical
Optimization, 2nd ed., Alg. 3.5-3.6 (zoom line search, strong Wolfe); Leja et al. 2017,
ApJ 837, 170 (Prospector: optimisation before sampling); Johnson et al. 2021, ApJS 254, 22.

Public / package API used
-------------------------
``SedModel.obs_dict`` / ``theta_init`` / ``param_names`` / ``priors`` / ``log_prob``
(``ceridwen/model/model.py:100-117, 397-413``);
``ceridwen.likelihood.likelihood.MultiObservationLikelihood`` (``likelihood.py:483``),
``make_lnprobfn`` (``likelihood.py:582-589``);
``Prior.sample(key, shape)`` / ``logpdf`` (``ceridwen/sampler/priors.py:92-107``).
Private helpers reused so the numbers are those of fitSED and the NUTS adapter:
``ceridwen.fit._likelihood_for``, ``ceridwen.fit._detect_bounds``,
``ceridwen.sampler.nuts._build_transforms`` / ``_to_constrained`` / ``_to_unconstrained``,
``BlackJAXNUTSAdapter._flatten`` / ``_unflatten``.

Package change it prepares for: ``fitSED(..., optimize=True, n_starts=...)`` (or a
``sampler="map"`` adapter) that runs this before NUTS and records the MAP in the result file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

import ceridwen  # noqa: F401  (enables jax_enable_x64)
from ceridwen.fit import _detect_bounds, _likelihood_for
from ceridwen.likelihood.likelihood import MultiObservationLikelihood, make_lnprobfn
from ceridwen.sampler.nuts import (BlackJAXNUTSAdapter, _build_transforms,
                                   _to_constrained, _to_unconstrained)

jax.config.update("jax_enable_x64", True)

__all__ = ["build_lnprob", "map_fit", "laplace_sigma", "MAPResult"]


def build_lnprob(model):
    """Jitted ``theta -> ln L + ln prior`` built exactly as ``fitSED`` builds its likelihood
    (``ceridwen/fit.py:78-87``), through ``MultiObservationLikelihood.make_lnprobfn``."""
    obs_dict = model.obs_dict
    keys = tuple(obs_dict)
    likelihoods = tuple(_likelihood_for(obs_dict[k], model.param_names, model=model)
                        for k in keys)
    multi = MultiObservationLikelihood(keys=keys, likelihoods=likelihoods)
    if getattr(model, "_eline_system", None) is not None:
        from ceridwen.likelihood.eline_marginal import refuse_outlier_with_elines
        refuse_outlier_with_elines(keys, likelihoods, model._eline_system.keys)
    return make_lnprobfn(obs_dict, model, model, multi)


@dataclass
class MAPResult:
    theta: dict            # best theta (constrained), shapes of model.theta_init
    lnp: float             # ln posterior at theta
    lnp_starts: np.ndarray  # final ln posterior of every start (n_starts,)
    n_steps: np.ndarray    # L-BFGS iterations per start
    grad_norm: np.ndarray  # final |grad| in the unconstrained space, per start
    wall_time: float       # seconds, compile included
    bounds: dict           # the bounds used for the logit map


def _prior_starts(model, n_starts, rng_key):
    """Start 0 = model.theta_init, starts 1..n-1 = prior draws (Prospector's minimizer_ball)."""
    starts = {}
    for name, init in model.theta_init.items():
        if name not in model.priors:
            raise ValueError(f"map_fit: parameter {name!r} has no prior; every free parameter "
                             "needs one to draw starting points")
        rng_key, sub = jax.random.split(rng_key)
        init = jnp.asarray(init, dtype=float)
        draws = model.priors[name].sample(sub, shape=(n_starts - 1, *init.shape))
        starts[name] = jnp.concatenate([init[None], jnp.asarray(draws, dtype=float)], axis=0)
    return starts


def map_fit(model, n_starts=16, rng_key=None, max_steps=3000, ftol=1e-12, bounds=None,
            lnprob=None):
    """Maximise ``build_lnprob(model)`` with L-BFGS from ``n_starts`` starts; see module doc.

    A start stops when the relative change of ln p between two iterations is below ``ftol``
    (``|f_k - f_{k-1}| <= ftol (1 + |f_k|)``) or after ``max_steps`` iterations.  ``bounds``
    defaults to ``fit._detect_bounds(model)``, as in fitSED's NUTS adapter."""
    t0 = time.perf_counter()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    if lnprob is None:
        lnprob = build_lnprob(model)
    if bounds is None:
        bounds = _detect_bounds(model)
    template = {k: jnp.asarray(v, dtype=float) for k, v in model.theta_init.items()}
    flat = BlackJAXNUTSAdapter(verbose=False)
    lo, hi, isb = _build_transforms(bounds, template)

    def theta_of(x):
        return flat._unflatten(_to_constrained(x, lo, hi, isb), template)

    def _loss(x):
        v = -lnprob(theta_of(x))
        return jnp.where(jnp.isfinite(v), v, jnp.inf)

    # A trial step of the line search can land where the forward model is non-finite; the
    # value is then +inf (the line search backtracks) and the gradient is zeroed there
    # instead of propagating NaN into the L-BFGS memory.
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

    # One compiled solver, starts run one after another.  vmap over the starts is slower
    # here: a batched while_loop runs every lane until the slowest start ends and executes
    # both branches of the zoom line search (measured 1018 s for 16 starts vs ~5 s per start).
    starts = _prior_starts(model, n_starts, rng_key)
    # Flatten in the key order of ``template`` explicitly: vmap/jit hand dicts back with sorted
    # keys, and ``BlackJAXNUTSAdapter._flatten`` follows the dict's own order, so under vmap it
    # would pair values with the wrong bounds.
    x0s = jax.vmap(lambda th: _to_unconstrained(
        jnp.concatenate([jnp.ravel(th[k]) for k in template]), lo, hi, isb))(starts)
    out = [run_one(x0s[i]) for i in range(n_starts)]
    xs = [o[0] for o in out]
    lnps = np.array([float(o[1]) for o in out])
    good = np.isfinite(lnps)
    if not good.any():
        raise RuntimeError("map_fit: every start ended at a non-finite ln posterior")
    best = int(np.argmax(np.where(good, lnps, -np.inf)))
    theta = jax.tree_util.tree_map(np.asarray, theta_of(xs[best]))
    return MAPResult(theta=theta, lnp=float(lnps[best]), lnp_starts=lnps,
                     n_steps=np.array([int(o[2]) for o in out]),
                     grad_norm=np.array([float(o[3]) for o in out]),
                     wall_time=time.perf_counter() - t0, bounds=dict(bounds))


def laplace_sigma(lnprob, theta, template=None):
    """Laplace 1-sigma widths ``sqrt(diag(H^-1))``, ``H`` = Hessian of ``-lnprob`` in the
    constrained, flattened parameters at ``theta``; returned as a dict shaped like ``theta``.
    Also returns the covariance matrix (flat order of ``theta``)."""
    template = {k: jnp.asarray(v, dtype=float) for k, v in (template or theta).items()}
    flat = BlackJAXNUTSAdapter(verbose=False)
    t0 = flat._flatten({k: jnp.asarray(theta[k], dtype=float) for k in template})
    H = jax.hessian(lambda t: -lnprob(flat._unflatten(t, template)))(t0)
    cov = jnp.linalg.inv(H)
    sig = jnp.sqrt(jnp.diag(cov))
    return jax.tree_util.tree_map(np.asarray, flat._unflatten(sig, template)), np.asarray(cov)
