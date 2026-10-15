"""Nested-sampling fit of the float32-parity mock in the precision and backend of THIS
process, with the timing of the sampler's value-only ln L at its batch widths; and a
comparison of such runs (``docs/mac_gpu.md``).

The mock is ``tests/baselines/_capture_f32.build_mock`` (z = 0.8 galaxy: SDSS photometry,
a 600-pixel spectrum in cgs F_nu, three line fluxes; nine free parameters), fitted to the data
of a reference capture so every run sees the same observations.  Run it once per mode, e.g.::

    python tests/baselines/_capture_f32.py --no-goldens ref.npz        # float64 reference data
    JAX_PLATFORMS=cpu                  python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f64_cpu
    JAX_PLATFORMS=cpu CERIDWEN_X64=0   python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f32_cpu
    JAX_PLATFORMS=mtl,cpu              python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f32_mtl
    python scripts/bench_mac_metal.py --compare bench/f64_cpu/summary.json bench/f32_cpu/summary.json bench/f32_mtl/summary.json

Each run writes ``summary.json`` (device, precision, timings, ln Z, posterior means and
standard deviations) next to the ``ceridwen_result.h5`` of the fit.  Nothing here forces
``jax_enable_x64``: ``import ceridwen`` decides it from the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
sys.path.insert(0, str(HERE.parent / "tests" / "baselines"))


def _timeit(fn, args, reps):
    """(median, iqr) seconds per call over ``reps`` calls, after one warm-up (compile)."""
    import jax
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return float(compile_s), float(med), float(q3 - q1)


def run(args) -> dict:
    import jax
    import jax.numpy as jnp
    import ceridwen
    from ceridwen import fitSED
    from ceridwen.precision import precision_name
    from _gridfixture import find_test_grid
    from _capture_f32 import build_mock, THETA_TRUE

    grid = find_test_grid()
    if grid is None:
        raise SystemExit("no test grid (tests/_gridfixture.py)")
    data = None
    if args.mock_data:
        with np.load(args.mock_data) as ref:
            data = {k: ref[k] for k in ref.files if k.startswith(("data_", "sigma_"))}
    t0 = time.perf_counter()
    model = build_mock(str(grid), data)
    build_s = time.perf_counter() - t0
    info = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "precision": precision_name(), "jax": jax.__version__,
            "ceridwen": ceridwen.__version__, "machine": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "build_s": build_s,
            "num_live": args.num_live, "num_delete": args.num_delete,
            "num_inner_steps": args.inner, "mock_data": args.mock_data}

    # the sampler's own log-likelihood (no prior), value only, as run_sampler builds it
    from ceridwen.sampler.runner import run_sampler  # noqa: F401  (same construction below)
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood import MultiObservationLikelihood
    from ceridwen.likelihood.likelihood import observation_data, float32_rescaling, rescaled_params
    keys = tuple(model.obs_dict)
    lhs = tuple(_likelihood_for(model.obs_dict[k], model.param_names, model=model) for k in keys)
    static = {k: observation_data(model.obs_dict[k]) for k in keys}
    resc = {k: float32_rescaling(model.obs_dict[k], lh) for k, lh in zip(keys, lhs)}

    def loglike(theta):
        pred = model.predict(theta)
        lnl = jnp.zeros(())
        for k, lh in zip(keys, lhs):
            y, sig, mask, calib, ul = static[k]
            mu = pred[k] if calib is None else pred[k] * calib
            th = rescaled_params(theta, resc[k])
            lnl = lnl + (lh(y, mu, sig, mask, params=th, is_upper_limit=ul)[0] if ul is not None
                         else lh(y, mu, sig, mask, params=th)[0])
        return lnl

    names = list(model.theta_init)
    rng = np.random.default_rng(0)
    timings = {}
    for width in sorted({args.num_delete, args.num_live}):
        batch = {k: jnp.asarray(np.asarray(THETA_TRUE[k])[None] + 0.02 * rng.normal(size=(width,) + np.shape(THETA_TRUE[k])))
                 for k in names}
        fn = jax.jit(jax.vmap(loglike))
        compile_s, med, iqr = _timeit(fn, (batch,), args.reps)
        timings[f"W={width}"] = {"compile_s": compile_s, "median_s": med, "iqr_s": iqr,
                                 "per_point_us": 1e6 * med / width}
        print(f"  vmap ln L  W={width:4d}: {1e3 * med:9.3f} ms/call (IQR {1e3 * iqr:.3f}), "
              f"{1e6 * med / width:.1f} us/point, compile {compile_s:.1f} s", flush=True)
    info["loglike_timing"] = timings

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = fitSED(model, sampler="nested", rng_key=jax.random.PRNGKey(args.seed),
                    sampler_kwargs={"num_live": args.num_live, "num_delete": args.num_delete,
                                    "num_inner_steps": args.inner, "logZ_tol": args.logz_tol},
                    output_dir=str(out_dir), verbose=True)
    info["fit_wall_s"] = time.perf_counter() - t0
    info["n_likelihood_calls"] = int(result.n_likelihood_calls)
    info["logZ"] = float(result.log_evidence)
    info["logZ_err"] = float(result.log_evidence_err)
    lw = np.asarray(result.log_weights, dtype=np.float64)
    w = np.exp(lw - lw.max())
    w = w / w.sum()
    post = {}
    for k in names:
        s = np.asarray(result.samples[k], dtype=np.float64).reshape(w.size, -1)
        mean = (w[:, None] * s).sum(0)
        sd = np.sqrt((w[:, None] * (s - mean) ** 2).sum(0))
        post[k] = {"mean": mean.tolist(), "sd": sd.tolist(), "true": np.asarray(THETA_TRUE[k]).ravel().tolist()}
    info["posterior"] = post
    info["max_loglike"] = float(np.max(np.asarray(result.log_likelihoods)))
    (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
    print(f"\n{info['backend']} {info['precision']}: ln Z = {info['logZ']:.3f} +- {info['logZ_err']:.3f}, "
          f"{info['n_likelihood_calls']:,} calls, {info['fit_wall_s']:.0f} s  -> {out_dir / 'summary.json'}")
    return info


def compare(paths) -> None:
    runs = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    ref = runs[0]
    print(f"{'run':24s} {'lnZ':>10s} {'+-':>6s} {'dlnZ/err':>9s} {'calls':>8s} {'wall s':>7s} "
          f"{'us/pt@del':>10s} {'us/pt@live':>11s}")
    for r in runs:
        d = (r["logZ"] - ref["logZ"]) / np.hypot(r["logZ_err"], ref["logZ_err"])
        t = r["loglike_timing"]
        wd, wl = f"W={r['num_delete']}", f"W={r['num_live']}"
        print(f"{r['backend'] + ' ' + r['precision']:24s} {r['logZ']:10.3f} {r['logZ_err']:6.3f} {d:9.2f} "
              f"{r['n_likelihood_calls']:8,d} {r['fit_wall_s']:7.0f} {t[wd]['per_point_us']:10.1f} {t[wl]['per_point_us']:11.1f}")
    print("\nposterior means, difference from the first run in units of its sd (per component):")
    for k in ref["posterior"]:
        row = []
        for r in runs[1:]:
            a, b, sd = (np.array(r["posterior"][k]["mean"]), np.array(ref["posterior"][k]["mean"]),
                        np.array(ref["posterior"][k]["sd"]))
            row.append(" ".join(f"{x:+.2f}" for x in (a - b) / sd))
        print(f"  {k:20s} " + " | ".join(row))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="bench_mac_metal")
    ap.add_argument("--mock-data", default=None, help="ref.npz of tests/baselines/_capture_f32.py")
    ap.add_argument("--num-live", type=int, default=200)
    ap.add_argument("--num-delete", type=int, default=40)
    ap.add_argument("--inner", type=int, default=30)
    ap.add_argument("--logz-tol", type=float, default=-5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--compare", nargs="+", metavar="SUMMARY_JSON")
    args = ap.parse_args(argv)
    if args.compare:
        compare(args.compare)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
