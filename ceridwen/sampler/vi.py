"""Variational-inference transport maps (TriL, IAF) and ELBO training for
NeuTra-HMC preconditioning (Hoffman et al. 2019, arXiv:1903.03704)."""
from __future__ import annotations

import abc
import time as _time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array



def _softplus(x: Array) -> Array:
    return jnp.logaddexp(x, 0.0)


def _pack_L(L_off: Array, d_raw: Array, tril_idx) -> Array:
    """Assemble lower-triangular L; diagonal is softplus(d_raw) > 0."""
    n = d_raw.shape[0]
    L = jnp.zeros((n, n), dtype=d_raw.dtype)
    L = L.at[tril_idx].set(L_off)
    return L + jnp.diag(_softplus(d_raw))



def _make_made_masks(D: int, H: int, n_hidden: int) -> list[Array]:
    """MADE masks for an MLP D -> H -> ... -> H -> 2D (outputs [mu, log_sigma]);
    output i depends only on inputs < i."""
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
    """Initialise one MADE MLP (zero output layer: identity transform)."""
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
    """Return (mu, log_sigma) per dim."""
    h = z
    for W, b, M in zip(params["W"], params["b"], masks[:-1]):
        h = (W * M) @ h + b
        h = jax.nn.elu(h)
    out = (params["W_out"] * masks[-1]) @ h + params["b_out"]
    D = z.shape[0]
    mu = out[:D]
    log_sigma = jnp.clip(out[D:], -5.0, 5.0)
    return mu, log_sigma


def _iaf_single_forward(z: Array, params: dict, masks: list[Array]):
    """One IAF step; returns (theta, log|det|)."""
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



class VariationalMap(abc.ABC):
    """Abstract transport map from z ~ N(0, I) to x in R^D."""

    name: str = "map"

    @abc.abstractmethod
    def init_params(self, rng: Array, x_init: Array) -> tuple[Any, Any]:
        """Return ``(params, aux)``: trainable pytree and static dict (must contain "D")."""

    @abc.abstractmethod
    def forward(self, z: Array, params: Any, aux: Any) -> tuple[Array, Array]:
        """Map a single ``z`` (shape ``(D,)``) to ``(x, log|det df/dz|)``."""


class TriLMap(VariationalMap):
    """Full-rank Gaussian family q(x) = N(mu, L L^T).

    Parameters
    ----------
    init_scale : float -- initial marginal standard deviation
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
        """Return the lower-triangular Cholesky factor L."""
        _, L_off, d_raw = params
        return _pack_L(L_off, d_raw, aux["tril_idx"])


class IAFMap(VariationalMap):
    """Stacked inverse autoregressive flows on a learnable affine base.

    Parameters
    ----------
    hidden_mult : int -- hidden layer width = hidden_mult * D
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



@dataclass
class TrainedMap:
    """A :class:`VariationalMap` with trained parameters, aux, loss trace and timing."""
    vi_map: VariationalMap
    params: Any
    aux: Any
    losses: np.ndarray
    train_time_s: float

    @property
    def D(self) -> int:
        return int(self.aux["D"])

    def forward(self, z: Array) -> tuple[Array, Array]:
        """Map one ``z`` (D,) -> ``(x, log|det df/dz|)``; not vmapped."""
        return self.vi_map.forward(z, self.params, self.aux)

    def sample_x(self, rng: Array, n: int) -> Array:
        """Draw ``n`` independent samples from ``q(x)``."""
        zs = jax.random.normal(rng, (n, self.D))
        xs, _ = jax.vmap(self.forward)(zs)
        return xs



def _decayed_lr(num_steps: int, lr0: float):
    """optax schedule lr0 -> lr0/10 -> lr0/100 at 20%/80% of steps, or None without optax."""
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
    """Maximise the ELBO of ``vi_map`` against the unconstrained log-posterior
    ``logpost_fn`` ((D,) -> ()) with Adam, starting the map at ``x_init``."""
    init_key, train_key = jax.random.split(rng)
    params0, aux = vi_map.init_params(init_key, x_init)
    D = int(aux["D"])

    def neg_elbo(params, z_batch):
        def per_sample(z):
            x, logdet = vi_map.forward(z, params, aux)
            return logpost_fn(x) + logdet
        return -jnp.mean(jax.vmap(per_sample)(z_batch))

    loss_and_grad = jax.value_and_grad(neg_elbo)

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

    @jax.jit
    def _run(params, opt_state, keys):
        def body(carry, key):
            params, opt_state = carry
            params, opt_state, loss = step(params, opt_state, key)
            return (params, opt_state), loss
        (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), keys)
        return params, opt_state, losses

    t0 = _time.perf_counter()
    report = max(1, num_steps // 6)
    done = 0
    while done < num_steps:
        n = min(report, num_steps - done)
        params, opt_state, chunk = _run(params, opt_state, keys[done:done + n])
        losses[done:done + n] = np.asarray(chunk)
        done += n
        if verbose:
            print(f"    iter {done:5d}/{num_steps}  -ELBO = {losses[done - 1]:+.3e}")
    t_train = _time.perf_counter() - t0
    if verbose:
        print(f"  [vi] {vi_map.name} trained in {t_train:.2f} s "
              f"({num_steps / t_train:.0f} it/s)")

    return TrainedMap(
        vi_map=vi_map, params=params, aux=aux,
        losses=losses, train_time_s=t_train,
    )



_MAP_REGISTRY = {
    "tril": TriLMap,
    "iaf":  IAFMap,
}


def make_vi_map(name: str, **kwargs) -> VariationalMap:
    """Instantiate a :class:`VariationalMap` by name ('tril' or 'iaf'); kwargs go to the constructor."""
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
