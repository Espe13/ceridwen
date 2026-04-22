r"""
ceridwen/sampler/vi.py
======================
Variational-inference transport maps for NeuTra-HMC preconditioning.

Implements the three-step neural-transport HMC procedure of
Hoffman, Sountsov, Dillon, Langmore, Tran & Vasudevan (2019),
"NeuTra-lizing Bad Geometry in HMC Using Neural Transport",
arXiv:1903.03704.

Concept
-------
Given an unconstrained log-posterior :math:`\log p(x)`, we learn a
bijective transport map :math:`x = f_\phi(z)` that minimises
:math:`\mathrm{KL}\bigl(q(x) \,\Vert\, p(x)\bigr)` with
:math:`q(x) = q_z(z) \,\lvert \partial f/\partial z\rvert^{-1}` and
:math:`q_z(z) = \mathcal{N}(0, I)`.  Equivalently we maximise the ELBO

.. math::
    \mathcal{L}(\phi) \;=\;
    \mathbb{E}_{z \sim \mathcal{N}(0,I)}
    \bigl[\log p(f_\phi(z)) + \log \lvert \partial f_\phi/\partial z\rvert\bigr].

NUTS then runs on the *whitened* target
:math:`\log p_z(z) = \log p(f_\phi(z)) + \log\lvert\partial f_\phi/\partial z\rvert`,
which is approximately :math:`\mathcal{N}(0, I)` when the map is
well-fitted, so identity mass matrix + step size :math:`\mathcal{O}(1)`
is near-optimal.

Map families
------------
:class:`TriLMap` — full-rank Gaussian, :math:`f(z) = \mu + L z` with
:math:`L` lower-triangular, positive diagonal.  This is the paper's
"TriL" baseline (= full-rank ADVI); sufficient for posteriors that are
well-approximated by a correlated Gaussian in unconstrained space.

:class:`IAFMap` — stacked inverse autoregressive flows (Kingma et al.
2016).  Three flows by default, each a MADE MLP with two hidden layers
of width :math:`D` and ELU activations, dimension-reversal between
flows, on a learnable affine base :math:`z \mapsto \mu_\mathrm{b} + \sigma_\mathrm{b}\,z`.
Strictly more expressive than TriL; required when the posterior has
funnel-like non-Gaussian structure.

Training
--------
:func:`train_vi` maximises the ELBO with Adam using the paper's
piecewise-constant learning-rate schedule (Sec. 4.1.1):
:math:`\mathrm{lr} = 10^{-2}` for the first 20% of steps, :math:`10^{-3}`
for 60%, :math:`10^{-4}` for the final 20%.  Uses ``optax`` if
available; a hand-rolled flat-lr Adam otherwise.

Everything is JAX-matrix-form and ``jax.jit``-friendly.
"""
from __future__ import annotations

import abc
import time as _time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


# ======================================================================
# Low-level numerical helpers
# ======================================================================

def _softplus(x: Array) -> Array:
    return jnp.logaddexp(x, 0.0)


def _pack_L(L_off: Array, d_raw: Array, tril_idx) -> Array:
    """Assemble lower-triangular L from strict-lower entries + raw diagonal.

    ``L_ii = softplus(d_raw_i) > 0`` keeps :math:`LL^\top` positive
    definite without eigendecomposition at each evaluation.
    """
    n = d_raw.shape[0]
    L = jnp.zeros((n, n), dtype=d_raw.dtype)
    L = L.at[tril_idx].set(L_off)
    return L + jnp.diag(_softplus(d_raw))


# ======================================================================
# MADE mask construction (Germain et al. 2015)
# ======================================================================

def _make_made_masks(D: int, H: int, n_hidden: int) -> list[Array]:
    """MADE masks for an MLP ``D -> H -> ... -> H -> 2D``.

    Degree scheme: input unit ``k`` carries degree ``k`` (0..D-1); each
    hidden unit ``j`` carries degree ``j mod (D-1)`` so degrees
    ``{0, ..., D-2}`` are covered when ``H >= D-1``; output units are
    arranged as ``[mu_0..mu_{D-1}, sigma_0..sigma_{D-1}]`` with degree
    ``0..D-1`` repeated twice.

    The hidden masks use ``>=``, the final output mask uses strict
    ``>`` — this enforces output ``i`` depends only on inputs
    ``0..i-1``, not on input ``i`` itself.  ``D-2`` as the upper bound
    of hidden degrees (rather than ``D-1``) is required for output 1 to
    reach input 0 through a degree-0 hidden unit.
    """
    m = [np.arange(D)]
    for _ in range(n_hidden):
        m.append(np.arange(H) % (D - 1))
    m.append(np.concatenate([np.arange(D), np.arange(D)]))

    masks = []
    for l in range(n_hidden + 1):
        prev_deg = m[l]
        cur_deg = m[l + 1]
        if l < n_hidden:
            mask = (cur_deg[:, None] >= prev_deg[None, :]).astype(np.float64)
        else:
            mask = (cur_deg[:, None] > prev_deg[None, :]).astype(np.float64)
        masks.append(jnp.asarray(mask))
    return masks


def _init_made(D: int, H: int, n_hidden: int, rng: Array) -> dict:
    """Initialise a single MADE MLP; output layer is zero so the IAF
    starts as the identity transform."""
    sizes_in = [D] + [H] * n_hidden
    sizes_out = [H] * n_hidden
    keys = jax.random.split(rng, n_hidden + 1)
    Ws, bs = [], []
    for din, dout, k in zip(sizes_in, sizes_out, keys[:-1]):
        W = jax.random.normal(k, (dout, din)) * (1.0 / np.sqrt(din))
        Ws.append(W)
        bs.append(jnp.zeros(dout))
    W_out = jnp.zeros((2 * D, H))
    b_out = jnp.zeros(2 * D)
    return {"W": Ws, "b": bs, "W_out": W_out, "b_out": b_out}


def _made_forward(z: Array, params: dict, masks: list[Array]):
    """Forward pass through a MADE MLP.  Returns (mu, log_sigma) per dim."""
    h = z
    for W, b, M in zip(params["W"], params["b"], masks[:-1]):
        h = (W * M) @ h + b
        h = jax.nn.elu(h)
    out = (params["W_out"] * masks[-1]) @ h + params["b_out"]
    D = z.shape[0]
    mu = out[:D]
    # Clip log_sigma for numerical stability; parameter ranges here are
    # never exotic enough to need the tanh trick from Kingma 2016.
    log_sigma = jnp.clip(out[D:], -5.0, 5.0)
    return mu, log_sigma


def _iaf_single_forward(z: Array, params: dict, masks: list[Array]):
    """One IAF step: :math:`\\theta_i = \\mu_i(z_{<i}) + \\sigma_i(z_{<i})\\,z_i`."""
    mu, log_sigma = _made_forward(z, params, masks)
    theta = z * jnp.exp(log_sigma) + mu
    logdet = jnp.sum(log_sigma)
    return theta, logdet


def _stacked_iaf_forward(z: Array, flows_params: list[dict],
                         masks_per_flow: list[list[Array]]):
    """Stack of IAFs with dimension reversal between consecutive flows."""
    x = z
    total_logdet = jnp.zeros((), dtype=z.dtype)
    K = len(flows_params)
    for k in range(K):
        x, ld = _iaf_single_forward(x, flows_params[k], masks_per_flow[k])
        total_logdet = total_logdet + ld
        if k < K - 1:
            x = x[::-1]
    return x, total_logdet


# ======================================================================
# VariationalMap interface
# ======================================================================

class VariationalMap(abc.ABC):
    """Abstract transport map :math:`z \\sim \\mathcal{N}(0, I) \\mapsto x \\in \\mathbb{R}^D`.

    Concrete subclasses supply ``init_params`` and ``forward``.  After
    :func:`train_vi` has found good parameters the map is wrapped in a
    :class:`TrainedMap` which exposes ``forward(z)`` with the parameters
    baked in.
    """

    name: str = "map"

    @abc.abstractmethod
    def init_params(self, rng: Array, x_init: Array) -> tuple[Any, Any]:
        """Return ``(params, aux)`` where ``params`` is the trainable pytree
        and ``aux`` is a dict of static auxiliary data (masks, indices, D)."""

    @abc.abstractmethod
    def forward(self, z: Array, params: Any, aux: Any) -> tuple[Array, Array]:
        """Map a single ``z`` (shape ``(D,)``) to ``(x, log|det df/dz|)``."""


class TriLMap(VariationalMap):
    """Full-rank Gaussian variational family
    :math:`q(x) = \\mathcal{N}(\\mu, L L^\\top)`.

    Parameters
    ----------
    init_scale : float, default 0.1
        Initial marginal standard deviation (uniform across dims).  The
        raw diagonal is parameterised via ``softplus`` so ``L_ii > 0``.
    """

    name = "tril"

    def __init__(self, init_scale: float = 0.1):
        self.init_scale = float(init_scale)

    def init_params(self, rng: Array, x_init: Array):
        n = int(x_init.shape[0])
        mu = x_init
        L_off = jnp.zeros(n * (n - 1) // 2, dtype=x_init.dtype)
        d_raw = jnp.full(
            (n,), jnp.log(jnp.expm1(self.init_scale)), dtype=x_init.dtype
        )
        params = (mu, L_off, d_raw)
        aux = {"D": n, "tril_idx": jnp.tril_indices(n, k=-1)}
        return params, aux

    def forward(self, z, params, aux):
        mu, L_off, d_raw = params
        L = _pack_L(L_off, d_raw, aux["tril_idx"])
        return mu + L @ z, jnp.sum(jnp.log(_softplus(d_raw)))

    def cholesky(self, params, aux) -> Array:
        """Return the lower-triangular Cholesky factor ``L`` so callers can
        inspect the learnt covariance :math:`L L^\\top` directly."""
        _, L_off, d_raw = params
        return _pack_L(L_off, d_raw, aux["tril_idx"])


class IAFMap(VariationalMap):
    """Stacked inverse autoregressive flows + learnable affine base
    (= paper's NeuTra map).

    Parameters
    ----------
    n_flows : int, default 3
        Number of stacked flows.  Paper uses 3.
    n_hidden : int, default 2
        Hidden layers per MADE MLP.  Paper uses 2.
    hidden_mult : int, default 1
        Hidden layer width = ``hidden_mult * D``.  Paper uses 1.
    """

    name = "iaf"

    def __init__(self, n_flows: int = 3, n_hidden: int = 2,
                 hidden_mult: int = 1):
        self.n_flows = int(n_flows)
        self.n_hidden = int(n_hidden)
        self.hidden_mult = int(hidden_mult)

    def init_params(self, rng, x_init):
        D = int(x_init.shape[0])
        H = self.hidden_mult * D
        masks_single = _make_made_masks(D, H, self.n_hidden)
        masks_per_flow = [masks_single for _ in range(self.n_flows)]
        flow_keys = jax.random.split(rng, self.n_flows)
        flows_params = [
            _init_made(D, H, self.n_hidden, k) for k in flow_keys
        ]
        params = {
            "flows": flows_params,
            "base_mu": x_init.astype(jnp.float64),
            "base_log_sigma": jnp.zeros(D, dtype=jnp.float64),
        }
        aux = {
            "D": D,
            "masks_per_flow": masks_per_flow,
            "n_flows": self.n_flows,
            "n_hidden": self.n_hidden,
            "H": H,
        }
        return params, aux

    def forward(self, z, params, aux):
        mu_b = params["base_mu"]
        log_sig_b = params["base_log_sigma"]
        y = mu_b + jnp.exp(log_sig_b) * z
        logdet_b = jnp.sum(log_sig_b)
        x, logdet_iaf = _stacked_iaf_forward(
            y, params["flows"], aux["masks_per_flow"]
        )
        return x, logdet_b + logdet_iaf


# ======================================================================
# Trained map bundle
# ======================================================================

@dataclass
class TrainedMap:
    """Result of :func:`train_vi`: a :class:`VariationalMap` with its
    trained parameters, static aux, training trace, and timing.

    Exposes convenience methods so the caller (e.g. NUTS adapter) does
    not need to know the map implementation details.
    """
    vi_map: VariationalMap
    params: Any
    aux: Any
    losses: np.ndarray
    train_time_s: float

    @property
    def D(self) -> int:
        return int(self.aux["D"])

    def forward(self, z: Array) -> tuple[Array, Array]:
        """Map one ``z`` -> ``(x, log|det df/dz|)``.  Not vmapped."""
        return self.vi_map.forward(z, self.params, self.aux)

    def sample_x(self, rng: Array, n: int) -> Array:
        """Draw ``n`` independent samples from ``q(x)``."""
        zs = jax.random.normal(rng, (n, self.D))
        xs, _ = jax.vmap(self.forward)(zs)
        return xs


# ======================================================================
# ELBO optimisation
# ======================================================================

def _decayed_lr(num_steps: int, lr0: float):
    """Piecewise-constant schedule: lr0, lr0/10, lr0/100 at 20%/80%.

    Returns a callable schedule if optax is available, else None.
    """
    try:
        import optax  # type: ignore
    except Exception:
        return None
    b1 = max(1, int(num_steps * 0.2))
    b2 = max(b1 + 1, int(num_steps * 0.8))
    return optax.piecewise_constant_schedule(
        init_value=lr0,
        boundaries_and_scales={b1: 0.1, b2: 0.1},
    )


def train_vi(
    vi_map: VariationalMap,
    logpost_fn: Callable[[Array], Array],
    x_init: Array,
    rng: Array,
    *,
    num_steps: int = 1500,
    batch_size: int = 16,
    lr0: float = 1e-2,
    verbose: bool = True,
) -> TrainedMap:
    """Maximise the ELBO of ``vi_map`` against ``logpost_fn``.

    Parameters
    ----------
    vi_map : VariationalMap
        Transport map to train (e.g. ``TriLMap()`` or ``IAFMap()``).
    logpost_fn : callable
        JIT-friendly unconstrained log-posterior, signature ``(D,) -> ()``.
    x_init : Array
        Starting point in unconstrained space; used to centre the base
        distribution or the TriL mean.  Should be ``jax.numpy.float64``
        on x64 platforms.
    rng : Array
        JAX PRNG key.
    num_steps : int, default 1500
        Adam iterations.
    batch_size : int, default 16
        MC samples per ELBO gradient.
    lr0 : float, default 1e-2
        Initial Adam learning rate.  Decayed by 10x at 20% and 80% of
        ``num_steps`` when ``optax`` is available.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    TrainedMap
    """
    init_key, train_key = jax.random.split(rng)
    params0, aux = vi_map.init_params(init_key, x_init)
    D = int(aux["D"])

    def neg_elbo(params, z_batch):
        def per_sample(z):
            x, logdet = vi_map.forward(z, params, aux)
            return logpost_fn(x) + logdet
        return -jnp.mean(jax.vmap(per_sample)(z_batch))

    loss_and_grad = jax.value_and_grad(neg_elbo)

    # Optimiser: optax Adam w/ decayed schedule if available,
    # else hand-rolled flat Adam.
    try:
        import optax  # type: ignore
        _have_optax = True
    except Exception:
        _have_optax = False

    if _have_optax:
        import optax
        sched = _decayed_lr(num_steps, lr0) or lr0
        opt = optax.adam(sched)
        opt_state = opt.init(params0)

        @jax.jit
        def step(params, opt_state, key):
            zb = jax.random.normal(key, (batch_size, D))
            loss, grads = loss_and_grad(params, zb)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
    else:
        if verbose:
            print("  [vi] optax not available; using hand-rolled flat Adam")
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        m0 = jax.tree_util.tree_map(jnp.zeros_like, params0)
        v0 = jax.tree_util.tree_map(jnp.zeros_like, params0)
        opt_state = (m0, v0, jnp.asarray(0, dtype=jnp.int32))

        @jax.jit
        def step(params, opt_state, key):
            m, v, t = opt_state
            zb = jax.random.normal(key, (batch_size, D))
            loss, grads = loss_and_grad(params, zb)
            t = t + 1
            m = jax.tree_util.tree_map(
                lambda a, g: b1 * a + (1 - b1) * g, m, grads)
            v = jax.tree_util.tree_map(
                lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
            mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
            vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
            params = jax.tree_util.tree_map(
                lambda p, mhh, vhh: p - lr0 * mhh / (jnp.sqrt(vhh) + eps_a),
                params, mh, vh,
            )
            return params, (m, v, t), loss

    params = params0
    losses = np.empty(num_steps)
    keys = jax.random.split(train_key, num_steps)

    if verbose:
        print(f"  [vi] training {vi_map.name}: D={D}  batch={batch_size}  "
              f"steps={num_steps}  lr0={lr0}  "
              f"optax={'yes' if _have_optax else 'no'}")

    t0 = _time.perf_counter()
    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state, keys[i])
        losses[i] = float(loss)
        if verbose and (i < 3 or (i + 1) % 250 == 0 or i == num_steps - 1):
            print(f"    iter {i+1:5d}/{num_steps}  -ELBO = {losses[i]:+.3e}")
    # Block on any single leaf for deterministic wall timing.
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    t_train = _time.perf_counter() - t0
    if verbose:
        print(f"  [vi] {vi_map.name} trained in {t_train:.2f} s "
              f"({num_steps / t_train:.0f} it/s)")

    return TrainedMap(
        vi_map=vi_map, params=params, aux=aux,
        losses=losses, train_time_s=t_train,
    )


# ======================================================================
# Factory
# ======================================================================

_MAP_REGISTRY = {
    "tril": TriLMap,
    "iaf":  IAFMap,
}


def make_vi_map(name: str, **kwargs) -> VariationalMap:
    """Instantiate a :class:`VariationalMap` by name.

    Parameters
    ----------
    name : {'tril', 'iaf'}
        Map family.
    **kwargs
        Passed to the constructor (e.g. ``init_scale`` for ``tril``,
        ``n_flows`` for ``iaf``).
    """
    name = name.lower().strip()
    if name not in _MAP_REGISTRY:
        raise ValueError(
            f"Unknown VI map '{name}'.  Available: {sorted(_MAP_REGISTRY)}"
        )
    return _MAP_REGISTRY[name](**kwargs)


__all__ = [
    "VariationalMap", "TriLMap", "IAFMap",
    "TrainedMap", "train_vi", "make_vi_map",
]
