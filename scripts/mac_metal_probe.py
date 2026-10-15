#!/usr/bin/env python3
"""Op probe for CERIDWEN on the Apple Metal PJRT plugin (jax-graft, JAX platform ``mtl``).

With float64 OFF, runs each JAX primitive the package uses (grep of ``ceridwen/`` for
``jnp.`` / ``jax.`` / ``lax.`` calls, 2026-10-14; sites quoted per row), at CERIDWEN-like shapes
(the BPASS test grid: 12 metallicities x 43 ages x 15000 pixels; a 600-pixel spectrum; 5 bands;
an order-10 calibration polynomial), on the CPU device and on the Metal device, and prints

    op | status (OK / REFUSED <message> / WRONG) | max|delta| vs cpu | us cpu | us metal

``WRONG`` means both ran but the Metal result differs from the CPU result by more than the
row's tolerance (default 1e-5 of max|cpu|; sorts and searches must be exact).  Timings are the
median of ``--reps`` calls after 3 warm-ups, ``block_until_ready`` included, so a Metal number
is dominated by the ~0.2 ms fixed dispatch cost for small ops (jax-graft docs/performance.md).

Usage (both platforms must be initialised; the plugin makes ``mtl`` JAX's default):

    JAX_PLATFORMS=mtl,cpu CERIDWEN_X64=0 python scripts/mac_metal_probe.py [--reps 20]
        [--with-model]      # also vmap(grad) of the real forward model (needs the test grid
                            # and $SPS_HOME; skipped with a message otherwise)

This is a diagnostic script, not part of the package; nothing here selects a platform for
CERIDWEN (that is the user's ``JAX_PLATFORMS``).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("CERIDWEN_X64", "0")
import jax                                                     # noqa: E402
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp                                        # noqa: E402
import numpy as np                                             # noqa: E402
import ceridwen  # noqa: E402,F401  (applies the tfp/jax compatibility shim of ceridwen/__init__.py)

N_Z, N_AGE, N_WAVE = 12, 43, 15000      # BPASS test grid (tests/fixtures), csp.py:275
N_PIX, N_BAND, N_LINE = 600, 5, 3       # scripts/bit_identity_check.py mock
N_TIME = 5                              # SFH nodes of the same mock
ORDER = 10                              # calibration polynomial order (prompt: poly_marginal)
N_LIVE, N_DELETE, N_INNER, N_DIM = 500, 100, 60, 12  # nested.py defaults for 12 dims


def _np(seed=0):
    return np.random.default_rng(seed)


def _f32(a):
    return np.asarray(a, dtype=np.float32)


# --------------------------------------------------------------------------------------------
# the op table: name -> (site, make_inputs(rng) -> tuple, fn(*inputs), tolerance or "exact")
# --------------------------------------------------------------------------------------------

def _ops():
    ops = []

    def add(name, site, make, fn, tol=1e-5):
        ops.append((name, site, make, fn, tol))

    # --- the big contractions of the forward model -------------------------------------------
    def mk_einsum(rng):
        W = _f32(rng.uniform(0, 1, (N_Z, N_AGE)))
        F = _f32(rng.uniform(0, 1, (N_Z, N_AGE, N_WAVE)))
        A = _f32(rng.uniform(0, 1, (N_AGE, N_WAVE)))
        return W, F, A
    add("einsum za,zaw,aw->w (12,43,15000)", "csp.py:1726 / :1665", mk_einsum,
        lambda W, F, A: jnp.einsum("za,zaw,aw->w", W, F, A))
    add("einsum ab,bw->aw (43x43 @ 43x15000)", "csp.py:1715 / :1646",
        lambda rng: (_f32(rng.uniform(0, 1, (N_AGE, N_AGE))), _f32(rng.uniform(0, 1, (N_AGE, N_WAVE)))),
        lambda M, a: jnp.einsum("ab,bw->aw", M, a))
    add("matvec T @ spec (5x15000)", "observation/photometry.py:160",
        lambda rng: (_f32(rng.uniform(0, 1, (N_BAND, N_WAVE))), _f32(rng.uniform(0, 1, N_WAVE))),
        lambda T, s: T @ s)

    # --- SSP weights: scatter-add with duplicate indices, searchsorted, cumsum ---------------
    def mk_scatter(rng):
        idx = rng.integers(1, N_Z, size=N_TIME - 1)          # duplicates by construction
        w = _f32(rng.uniform(0, 1, (N_TIME - 1,)))
        wt = _f32(rng.uniform(0, 1, (N_TIME - 1, N_AGE)))
        return idx, w, wt
    def scatter_add(idx, w, wt):
        tw = jnp.zeros((N_Z, N_AGE), dtype=wt.dtype)
        tw = tw.at[idx - 1].add((1 - w)[:, None] * wt)        # csp.py:1533
        tw = tw.at[idx].add(w[:, None] * wt)                  # csp.py:1534
        return tw
    add(".at[].add duplicate indices (12,43)", "csp.py:1533-1534, 1548-1549", mk_scatter, scatter_add)
    zmet = np.linspace(-2.3, 0.3, N_Z).astype(np.float32)
    add("searchsorted side=left", "csp.py:1525",
        lambda rng: (zmet, _f32(rng.uniform(-2.5, 0.5, 64))),
        lambda ax, x: jnp.searchsorted(ax, x, side="left"), tol="exact")
    add("searchsorted side=right", "neb/NebularGridModel.py:386, csp.py:1190",
        lambda rng: (np.linspace(900, 25000, N_WAVE).astype(np.float32), _f32(rng.uniform(1000, 20000, 128))),
        lambda ax, x: jnp.searchsorted(ax, x, side="right") - 1, tol="exact")
    add("cumsum (SFH ratios)", "model/transforms.py:23",
        lambda rng: (_f32(rng.normal(0, 1, N_TIME - 1)),),
        lambda r: jnp.concatenate([jnp.zeros(1, r.dtype), -jnp.cumsum(r)]))
    add("interp (15000 -> 600 pixels)", "broadening.py:416, csp.py:1032",
        lambda rng: (np.linspace(3800, 8800, N_PIX).astype(np.float32),
                     np.linspace(900, 25000, N_WAVE).astype(np.float32), _f32(rng.uniform(0, 1, N_WAVE))),
        lambda x, xp, fp: jnp.interp(x, xp, fp))
    add("trapezoid (filter integral)", "observation/filters.py:300, neb:379",
        lambda rng: (_f32(rng.uniform(0, 1, (N_BAND, N_WAVE))), np.linspace(900, 25000, N_WAVE).astype(np.float32)),
        lambda y, x: jnp.trapezoid(y, x, axis=-1))
    add("sort", "ns_weights / filters.py:170",
        lambda rng: (_f32(rng.normal(0, 1, 20000)),), lambda x: jnp.sort(x), tol="exact")
    add("argsort", "observation/filters.py:170",
        lambda rng: (_f32(rng.normal(0, 1, 20000)),), lambda x: jnp.argsort(x), tol="exact")
    add("rfft/irfft (continuum kernel, n=16384)", "broadening.py:390",
        lambda rng: (_f32(rng.normal(0, 1, 16384)), _f32(np.exp(-np.linspace(0, 5, 8193)))),
        lambda p, taper: jnp.fft.irfft(jnp.fft.rfft(p) * taper, n=16384), tol=1e-4)

    # --- likelihood pieces ---------------------------------------------------------------------
    def mk_chi(rng):
        y = _f32(rng.uniform(0.5, 1.5, N_PIX)); mu = _f32(rng.uniform(0.5, 1.5, N_PIX))
        sig = _f32(rng.uniform(0.05, 0.1, N_PIX)); mask = rng.uniform(size=N_PIX) > 0.1
        return y, mu, sig, mask
    def lnlike(y, mu, sig, mask):                             # likelihood.py:107-116, noise_model.py:173-201
        var = jnp.where(mask, sig ** 2, 1.0)
        var = jnp.maximum(var, jnp.finfo(var.dtype).tiny)
        inv_var = 1.0 / var
        log_det = 0.5 * jnp.log(var) + 0.5 * jnp.log(2.0 * jnp.pi)
        chi = (y - mu) * jnp.sqrt(inv_var)
        return jnp.sum(jnp.where(mask, -0.5 * chi ** 2 - log_det, 0.0))
    add("diagonal Gaussian ln L (600)", "likelihood.py:98-126", mk_chi, lnlike)
    add("logaddexp (outlier mixture)", "likelihood.py:132",
        lambda rng: (_f32(rng.normal(-5, 2, N_PIX)), _f32(rng.normal(-5, 2, N_PIX))),
        lambda a, b: jnp.logaddexp(a, b))
    add("jnp.power / log / exp / sqrt / clip chain", "dust/attenuation_laws.py, csp.py",
        lambda rng: (_f32(rng.uniform(0.1, 3, N_WAVE)), _f32(rng.uniform(-1, 0.4, 1))),
        lambda w, n: jnp.clip(jnp.exp(-jnp.power(w, n[0]) * jnp.sqrt(w)) * jnp.log(1 + w), 0.0, 10.0))

    # --- linear algebra at the package's sizes ------------------------------------------------
    def mk_spd(n):
        def mk(rng):
            A = rng.normal(0, 1, (n, n)); P = A @ A.T + n * np.eye(n)
            return _f32(P), _f32(rng.normal(0, 1, n))
        return mk
    from jax.scipy.linalg import cho_solve, solve_triangular
    def chol_solve(P, b):                                     # poly_marginal.py:55-57, eline_marginal.py:83-89
        e = 1.0 / jnp.sqrt(jnp.diagonal(P))
        L = jnp.linalg.cholesky(e[:, None] * P * e[None, :])
        v = solve_triangular(L, e * b, lower=True)
        x = cho_solve((L, True), e * b)
        return jnp.concatenate([v, x, jnp.log(jnp.diagonal(L))])
    add(f"cholesky+solve_triangular+cho_solve n={ORDER + 1}", "likelihood/poly_marginal.py:55-82",
        mk_spd(ORDER + 1), chol_solve, tol=1e-4)
    add("cholesky+solve_triangular+cho_solve n=20 (lines)", "likelihood/eline_marginal.py:83-89",
        mk_spd(20), chol_solve, tol=1e-4)
    add("cholesky+solve_triangular n=600 (GP spectrum)", "likelihood/gp_likelihood.py:52,75",
        mk_spd(N_PIX), lambda P, b: solve_triangular(jnp.linalg.cholesky(P), b, lower=True), tol=1e-3)
    add(f"jnp.linalg.solve n={ORDER + 1}", "likelihood/poly_calibration.py:97",
        mk_spd(ORDER + 1), lambda P, b: jnp.linalg.solve(P, b), tol=1e-4)
    add("jnp.linalg.inv n=12", "optimize.py (laplace_sigma)",
        mk_spd(N_DIM), lambda P, b: jnp.linalg.inv(P), tol=1e-3)

    # --- random numbers ------------------------------------------------------------------------
    def _split(k):
        return jax.random.split(jax.random.wrap_key_data(k))
    add("random.split/uniform/fold_in (bits)", "sampler/nested.py:250,323; priors.py:397",
        lambda rng: (np.array([0, 7], dtype=np.uint32),),
        lambda k: (lambda k1, k2: jnp.concatenate([jax.random.uniform(k1, (N_LIVE,)),
                                                   jax.random.uniform(jax.random.fold_in(k2, 3), (4,))]))
        (*_split(k)), tol="exact")
    add("random.normal (erf_inv of the same bits)", "sampler/nuts.py, vi.py",
        lambda rng: (np.array([0, 7], dtype=np.uint32),),
        lambda k: jax.random.normal(_split(k)[0], (N_LIVE,)), tol=1e-6)

    # --- tfp priors used by sampler/priors.py ---------------------------------------------------
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    dists = {
        "Uniform": lambda: tfd.Uniform(low=-2.0, high=0.5),                       # priors.py:157
        "Normal": lambda: tfd.Normal(loc=0.0, scale=0.3),                          # priors.py:188
        "TruncatedNormal": lambda: tfd.TruncatedNormal(loc=0.0, scale=1.0, low=-1.0, high=2.0),  # :280
        "LogNormal": lambda: tfd.LogNormal(loc=0.0, scale=0.5),                    # :311
        "StudentT": lambda: tfd.StudentT(df=2.0, loc=0.0, scale=1.0),              # :416
        "Transformed(Uniform,Exp)": lambda: tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=jnp.log(1e-3), high=jnp.log(0.15)), bijector=tfp.bijectors.Exp()),  # :369
        "MVNFullCovariance(12)": lambda: tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(N_DIM), covariance_matrix=jnp.eye(N_DIM) * 0.5 + 0.1),    # :218
    }
    for dname, mk_d in dists.items():
        def _mk(rng, _d=mk_d):
            d = _d()
            ev = tuple(int(s) for s in d.event_shape)
            x = _f32(rng.uniform(0.01, 0.1, (N_LIVE,) + ev))   # inside every support above
            return (x, np.array([0, 11], dtype=np.uint32))
        def _fn(x, k, _d=mk_d):
            d = _d()
            key = jax.random.wrap_key_data(k)
            return jnp.concatenate([jnp.ravel(d.log_prob(x)), jnp.ravel(d.sample(seed=key, sample_shape=(8,)))])
        add(f"tfp {dname} log_prob+sample", "sampler/priors.py", _mk, _fn, tol=1e-4)

    # --- control flow: the blackjax NSS step (while_loop + pytree carry) and a scan -------------
    def mk_nss(rng):
        return (_f32(rng.normal(0, 1, (N_LIVE, N_DIM))), np.array([0, 5], dtype=np.uint32))
    def nss_step(x0, k):
        import blackjax
        nss = blackjax.nss(logprior_fn=lambda t: jnp.zeros(()),
                           loglikelihood_fn=lambda t: -0.5 * jnp.sum(t["x"] ** 2),
                           num_delete=N_DELETE, num_inner_steps=N_INNER)
        live = nss.init({"x": x0})
        live, info = nss.step(jax.random.wrap_key_data(k), live)
        return jnp.concatenate([jnp.ravel(live.particles.loglikelihood),
                                jnp.ravel(live.integrator.logZ)[None][0][None] if False else jnp.ravel(live.integrator.logZ)])
    add(f"blackjax.nss init+step (live {N_LIVE}, delete {N_DELETE}, inner {N_INNER}, dim {N_DIM})",
        "sampler/nested.py:253-327", mk_nss, nss_step, tol="finite")
    add("lax.scan 200 steps (NUTS-style chain)", "sampler/nuts.py:258",
        lambda rng: (_f32(rng.normal(0, 1, N_DIM)),),
        lambda x0: jax.lax.scan(lambda c, _: (c - 0.01 * c ** 3 + 0.001, c), x0, None, length=200)[0])
    add("lax.while_loop (L-BFGS-style)", "optimize.py:212",
        lambda rng: (_f32(rng.normal(0, 1, N_DIM)),),
        lambda x0: jax.lax.while_loop(lambda c: jnp.sum(c[0] ** 2) > 1e-3,
                                      lambda c: (c[0] * 0.9, c[1] + 1), (x0, jnp.int32(0)))[0])

    # --- vmap(grad) of a stand-in forward model with the package's op mix ----------------------
    def mk_vg(rng):
        F = _f32(rng.uniform(0, 1, (N_Z, N_AGE, N_WAVE)))
        T = _f32(rng.uniform(0, 1, (N_BAND, N_WAVE)))
        y = _f32(rng.uniform(0, 1, N_BAND))
        th = _f32(rng.normal(0, 0.1, (N_DELETE, N_TIME + 1)))
        return F, T, y, th
    def vgrad(F, T, y, th):
        def model(t):
            w_age = jnp.exp(-jnp.cumsum(t[:N_TIME]))
            W = jnp.zeros((N_Z, N_AGE), dtype=F.dtype).at[jnp.arange(N_TIME) % N_Z, jnp.arange(N_TIME) * 7].add(w_age)
            spec = jnp.einsum("za,zaw->w", W, F) * jnp.exp(-jnp.abs(t[N_TIME]) * jnp.linspace(0, 1, N_WAVE))
            return -0.5 * jnp.sum(((T @ spec) * 1e-3 - y) ** 2)
        return jax.vmap(jax.grad(model))(th)
    add(f"vmap(grad(stand-in forward model)) W={N_DELETE}", "likelihood.make_lnprobfn / nested.py", mk_vg, vgrad, tol=1e-3)

    # --- pmap: nuts.py only counts devices; expected to run on one device or be refused --------
    return ops


def _pmap_row(cpu, mtl, reps):
    """nuts.py:265/581 count devices and use pmap only when n_devices >= num_chains > 1; with the
    plugin's single device that branch is never taken.  Here: pmap over the one device."""
    x = np.ones((1, 8), dtype=np.float32)
    out = []
    for dev in (cpu, mtl):
        try:
            f = jax.pmap(lambda v: v * 2.0, devices=[dev])
            r = f(x); jax.block_until_ready(r)
            out.append((np.asarray(r), _time(f, (x,), reps)))
        except Exception as exc:                                       # noqa: BLE001
            out.append((None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"))
    (rc, tc), (rm, tm) = out
    if rc is None:
        return ("pmap over 1 device", "sampler/nuts.py:265, 581, 590", f"CPU FAILED: {tc}", "", "", "")
    if rm is None:
        return ("pmap over 1 device", "sampler/nuts.py:265, 581, 590", f"REFUSED {tm}", "", f"{tc:.0f}", "")
    d = float(np.max(np.abs(rc - rm)))
    return ("pmap over 1 device", "sampler/nuts.py:265, 581, 590", "OK" if d == 0 else "WRONG", f"{d:.3g}", f"{tc:.0f}", f"{tm:.0f}")


# --------------------------------------------------------------------------------------------

def _time(fn, args, reps):
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(ts))


def _flat(x):
    return np.concatenate([np.ravel(np.asarray(a, dtype=np.float64)) for a in jax.tree_util.tree_leaves(x)])


def run(ops, reps, cpu, mtl):
    rows = []
    for name, site, make, fn, tol in ops:
        rng = _np(0)
        inputs = make(rng)
        jfn = jax.jit(fn)
        try:
            a_cpu = [jax.device_put(a, cpu) for a in inputs]
            out_cpu = jfn(*a_cpu)
            jax.block_until_ready(out_cpu)
            t_cpu = _time(jfn, a_cpu, reps)
        except Exception as exc:                                       # noqa: BLE001
            rows.append((name, site, f"CPU FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}", "", "", ""))
            continue
        try:
            a_mtl = [jax.device_put(a, mtl) for a in inputs]
            out_mtl = jfn(*a_mtl)
            jax.block_until_ready(out_mtl)
            t_mtl = _time(jfn, a_mtl, reps)
        except Exception as exc:                                       # noqa: BLE001
            msg = str(exc).splitlines()[0]
            rows.append((name, site, f"REFUSED {type(exc).__name__}: {msg[:110]}", "", f"{t_cpu:.0f}", ""))
            continue
        fc, fm = _flat(out_cpu), _flat(out_mtl)
        if fc.shape != fm.shape:
            status, dmax = "WRONG (shape)", float("nan")
        else:
            dmax = float(np.max(np.abs(fc - fm))) if fc.size else 0.0
            if tol == "exact":
                status = "OK" if dmax == 0.0 else "WRONG"
            elif tol == "finite":
                status = "OK" if np.all(np.isfinite(fm)) else "WRONG (non-finite)"
            else:
                scale = float(np.max(np.abs(fc))) if fc.size else 1.0
                status = "OK" if dmax <= tol * max(scale, 1e-30) else f"WRONG (> {tol:g} x max|cpu|={scale:.3g})"
        rows.append((name, site, status, f"{dmax:.3g}", f"{t_cpu:.0f}", f"{t_mtl:.0f}"))
        print(f"  done: {name}", file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--with-model", action="store_true",
                    help="also vmap(grad) of the real CERIDWEN forward model (needs the test grid)")
    args = ap.parse_args()
    print(f"jax {jax.__version__}  x64={jax.config.read('jax_enable_x64')}  devices={jax.devices()}")
    try:
        mtl = jax.devices("mtl")[0]
    except RuntimeError as exc:
        sys.exit(f"no Metal device: {exc}\n(run with JAX_PLATFORMS=mtl,cpu in an env with jax-graft)")
    cpu = jax.devices("cpu")[0]
    try:
        import jax_graft
        print(f"jax-graft {getattr(jax_graft, '__version__', '?')}")
    except ImportError:
        pass
    ops = _ops()
    if args.with_model:
        ops.append(_model_op())
    rows = run(ops, args.reps, cpu, mtl)
    rows.append(_pmap_row(cpu, mtl, args.reps))
    print(f"  devices seen by nuts.py (len(jax.devices())): {len(jax.devices())}", file=sys.stderr)
    print()
    print("| op | site | status | max|delta| vs cpu | us cpu | us metal |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    n_ref = sum(1 for r in rows if r[2].startswith("REFUSED"))
    n_wrong = sum(1 for r in rows if r[2].startswith("WRONG") or r[2].startswith("CPU FAILED"))
    print(f"\n{len(rows)} ops: {len(rows) - n_ref - n_wrong} OK, {n_ref} REFUSED, {n_wrong} WRONG/failed")


def _model_op():
    """vmap(grad) of fitSED's log-posterior for the bit_identity_check mock (needs the grid)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    from _gridfixture import find_test_grid
    from baselines._capture_f32 import build_mock, log_posterior            # tests/baselines/_capture_f32.py
    path = find_test_grid()
    if path is None:
        raise SystemExit("--with-model: no test grid (tests/_gridfixture.py)")
    model = build_mock(str(path))
    lnp, _offset = log_posterior(model)          # the float32 ln L offset is irrelevant to a gradient
    names = list(model.theta_init)
    def mk(rng):
        base = {k: np.asarray(model.theta_init[k], dtype=np.float32) for k in names}
        return tuple(base[k][None] + np.float32(0.02) * _f32(rng.normal(0, 1, (N_DELETE,) + base[k].shape)) for k in names)
    def fn(*arrs):
        th = dict(zip(names, arrs))
        return jax.vmap(jax.grad(lnp))(th)
    return (f"vmap(grad(fitSED log-posterior)) W={N_DELETE}, real model", "likelihood.py:593, nested.py", mk, fn, 1e-3)


if __name__ == "__main__":
    main()
