#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_ceridwen.py — stage-resolved timing, roofline and memory audit for the
CERIDWEN forward model, with the nebular path broken out explicitly.

Answers three questions:

  1. WHERE does the ~21 ms per spectrum evaluation go?  (ablation timing:
     nebular grid interpolation vs dense array construction vs the SSP
     contraction vs dust, forward AND under jax.grad)
  2. Is each stage compute-bound or bandwidth-bound?  (XLA cost analysis ->
     arithmetic intensity vs the device roofline ridge point)
  3. WHY does it OOM?  (analytic byte ledger of every full-size intermediate,
     XLA temp-buffer report, and a vmap-width bisection to find the wall)

Usage
-----
    # basic
    python profile_ceridwen.py

    # point at a specific grid, and include the vmap scaling scan
    python profile_ceridwen.py --ssp /path/to/ssp_data.h5 --vmap-scan

    # dump a pprof heap profile for the OOM postmortem
    python profile_ceridwen.py --mem-profile mem.pprof

Read `build()` first — that is the only part you should need to edit.

Set these before launching on a GPU node, or the numbers lie:
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import sys
import time

if platform.system() == "Darwin":
    # jax-metal is the default backend on the Mac and fails on ceridwen's
    # float64 predict paths.  Must be set BEFORE any jax import.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Device roofline constants.  Override with --peak-flops / --peak-bw if you are
# not on one of these.  fp32 (non-tensor-core) TFLOP/s and HBM TB/s.
# ---------------------------------------------------------------------------
ROOFLINE = {
    "A100":   (19.5e12, 1.55e12),
    "A100-80": (19.5e12, 2.04e12),
    "V100":   (15.7e12, 0.90e12),
    "H100":   (67.0e12, 3.35e12),
    "cpu":    (0.20e12, 0.05e12),
}


# ===========================================================================
# 1.  BUILD — edit this to match how you actually construct the model
# ===========================================================================
MAC_JADES_ROOT = "/Users/amanda/Desktop/PhD/Tursa/Tursa/jades_full"


def find_jades_root(explicit=None):
    """Locate the directory that contains the ``tursa`` package.

    Search order: explicit argument, ``$JADES_ROOT``, ancestors of this script
    (plus their ``tests/jades_full`` and ``jades_full`` subdirectories), then
    the laptop default.  Keeps this file runnable unmodified on the Mac and on
    tursa.  Duplicated in verify_neb_refactor.py so each script stays
    standalone.
    """
    def ok(p):
        return bool(p) and os.path.isfile(
            os.path.join(p, "tursa", "common", "config.py"))

    cands = []
    if explicit:
        cands.append(os.path.abspath(explicit))
    if os.environ.get("JADES_ROOT"):
        cands.append(os.path.abspath(os.environ["JADES_ROOT"]))
    p = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        cands += [p,
                  os.path.join(p, "tests", "jades_full"),
                  os.path.join(p, "jades_full")]
        p = os.path.dirname(p)
    cands.append(MAC_JADES_ROOT)
    for c in dict.fromkeys(cands):
        if ok(c):
            return c
    raise SystemExit(
        "Could not locate the tursa package (tursa/common/config.py).\n"
        "Set $JADES_ROOT or pass --jades-root.  Tried:\n  "
        + "\n  ".join(dict.fromkeys(cands)))


def build(jades_root: str, zred: float = 7.0,
          frac_obrun: bool = True, n_bins_sfh: int | None = None):
    """Return (csp, theta) for the PRODUCTION JADES configuration.

    Uses ``tursa.common.predictor.build_csp_extra_young`` directly, so this is
    the same CSPBasis the fits used: BPASS SSPs, extra_young agebin grid,
    zh_const, step SFH, Charlot & Fall birth cloud on (-inf, -1.97],
    Kriek & Conroy diffuse law, cloudy_dust BPASS nebular grid, Madau 1995 IGM,
    no dust emission.
    """
    if jades_root not in sys.path:
        sys.path.insert(0, jades_root)

    from tursa.common.config import (resolve_ssp_file, resolve_sps_home,
                                     N_BINS_SFH)
    from tursa.common.predictor import build_csp_extra_young

    nb = int(n_bins_sfh or N_BINS_SFH)
    csp, agebins = build_csp_extra_young(
        zred=zred,
        ssp_file=resolve_ssp_file(),
        sps_home=resolve_sps_home(),
        nbins_sfh=nb,
        intrinsic=False,
    )

    # Assemble a representative theta from the component models' own defaults,
    # so we never hard-code parameter names that the dust laws might rename.
    theta = {}
    for attr in ("dust_attn", "diff_dust", "diffuse_dust", "neb", "dust_emi"):
        comp = getattr(csp, attr, None)
        if comp is not None and hasattr(comp, "get_default_params"):
            try:
                theta.update(comp.get_default_params())
            except Exception:                               # noqa: BLE001
                pass
    theta["lookback_time"] = jnp.asarray(agebins)
    theta["sfh"] = jnp.ones(nb)
    theta["Z"] = jnp.asarray([-2.0])          # log10 ABSOLUTE Z (ssp_lgmet)
    theta.setdefault("gas_logz", jnp.asarray(-2.0))
    theta.setdefault("gas_logu", jnp.asarray(-2.0))
    if frac_obrun:
        # Present in the fesc variant; it activates the kill_ion restoration
        # and the runaway attenuation mix, so include it by default.
        theta["frac_obrun"] = jnp.asarray([0.1])
    return csp, theta


# ===========================================================================
# 2.  Timing harness
# ===========================================================================
def _sync(x):
    jax.block_until_ready(x)
    return x


def bench(fn, *args, n=50, warmup=5, label=""):
    """Median wall-clock per call, in ms.  Returns (median_ms, iqr_ms)."""
    try:
        for _ in range(warmup):
            out = fn(*args)
        _sync(out)
    except Exception as e:                                  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"

    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        _sync(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    med = statistics.median(ts)
    iqr = ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]
    return med, iqr


def scalarise(f):
    """Wrap an array- or tuple-returning fn into a scalar so jax.grad applies.

    Sum-of-squares stands in for the log-likelihood: it has the same
    reverse-mode structure (one cotangent per wavelength) so the residual
    footprint and backward FLOP count are representative.
    """
    def g(theta):
        out = f(theta)
        leaves = [l for l in jax.tree_util.tree_leaves(out)
                  if jnp.issubdtype(jnp.asarray(l).dtype, jnp.inexact)]
        return sum(jnp.sum(jnp.square(l)) for l in leaves)
    return g


# ===========================================================================
# 3.  XLA cost / memory introspection
# ===========================================================================
def compiled_stats(fn, *args):
    """Return dict with flops, bytes, temp bytes for a jitted callable."""
    out = {}
    try:
        comp = jax.jit(fn).lower(*args).compile()
    except Exception as e:                                  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}

    try:
        ca = comp.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        out["flops"] = float(ca.get("flops", 0.0))
        out["bytes"] = float(ca.get("bytes accessed", 0.0))
    except Exception:                                       # noqa: BLE001
        out["flops"] = out["bytes"] = float("nan")

    try:
        ma = comp.memory_analysis()
        out["temp_bytes"] = float(getattr(ma, "temp_size_in_bytes", 0))
        out["arg_bytes"] = float(getattr(ma, "argument_size_in_bytes", 0))
        out["out_bytes"] = float(getattr(ma, "output_size_in_bytes", 0))
    except Exception:                                       # noqa: BLE001
        out["temp_bytes"] = out["arg_bytes"] = out["out_bytes"] = float("nan")

    out["_compiled"] = comp
    return out


def roofline_verdict(flops, byts, peak_flops, peak_bw):
    """Classify a kernel: arithmetic intensity vs the machine ridge point."""
    if not byts or not np.isfinite(byts) or byts == 0:
        return float("nan"), "n/a"
    ai = flops / byts
    ridge = peak_flops / peak_bw
    if ai < 0.1 * ridge:
        v = "BANDWIDTH-BOUND (hard)"
    elif ai < ridge:
        v = "bandwidth-bound"
    else:
        v = "compute-bound"
    t_bw = byts / peak_bw
    t_fl = flops / peak_flops
    return ai, f"{v}; roofline floor = {max(t_bw, t_fl)*1e3:.3f} ms"


def mem_stats():
    try:
        d = jax.devices()[0]
        s = d.memory_stats() or {}
        return {
            "in_use_MB": s.get("bytes_in_use", 0) / 2**20,
            "peak_MB": s.get("peak_bytes_in_use", 0) / 2**20,
            "limit_MB": s.get("bytes_limit", 0) / 2**20,
        }
    except Exception:                                       # noqa: BLE001
        return {}


# ===========================================================================
# 4.  Analytic byte ledger — this is what explains the OOM
# ===========================================================================
def byte_ledger(csp):
    """Every full-size intermediate the nebular path materialises, in MB.

    Compares against the irreducible working set so you can see the
    inflation factor directly.
    """
    f32 = 4
    n_z, n_age, n_wave = (int(s) for s in csp.flux.shape)
    cube_MB = n_z * n_age * n_wave * f32 / 2**20

    neb = getattr(csp, "neb", None)
    n_young = int(getattr(csp, "_neb_n_young", 0) or 0)
    n_lines = int(neb.nebem_line.shape[0]) if neb is not None else 0
    gauss_MB = (float(np.prod(neb.gaussnebarr.shape)) * f32 / 2**20
                if neb is not None else 0.0)

    rows = [
        ("SSP flux grid (persistent)", cube_MB, f"({n_z},{n_age},{n_wave})"),
        ("neb line_spec  (n_z,n_wave,n_young)",
         n_z * n_wave * n_young * f32 / 2**20, "einsum output"),
        ("neb cont_flux  (n_z,n_wave,n_young)",
         n_z * n_wave * n_young * f32 / 2**20, ""),
        ("neb_all dense zeros cube", cube_MB, "_build_neb_array"),
        ("stellar_fluxes (jnp.where on kill_ion)", cube_MB, "full copy"),
        ("combined_fluxes = stellar + neb", cube_MB, "full copy"),
        ("gaussnebarr (persistent)", gauss_MB, f"(n_wave,{n_lines})"),
    ]

    useful = n_z * n_wave * n_young * f32 / 2**20     # what actually carries neb info
    transient = sum(r[1] for r in rows[1:6])

    return {
        "rows": rows,
        "cube_MB": cube_MB,
        "n_z": n_z, "n_age": n_age, "n_wave": n_wave,
        "n_young": n_young, "n_lines": n_lines,
        "useful_MB": useful,
        "transient_MB": transient,
        "inflation": transient / useful if useful else float("nan"),
    }


# ===========================================================================
# 5.  vmap-width bisection — where exactly does it OOM?
# ===========================================================================
def vmap_scan(fn, theta, widths=(1, 2, 4, 8, 16, 32, 64, 128)):
    results = []
    for w in widths:
        batch = jax.tree_util.tree_map(
            lambda x, w=w: jnp.broadcast_to(jnp.asarray(x), (w,) + jnp.shape(x)),
            theta)
        vf = jax.jit(jax.vmap(fn))
        try:
            t, _ = bench(vf, batch, n=5, warmup=2)
            m = mem_stats()
            results.append((w, t, m.get("peak_MB", float("nan")), "ok"))
        except Exception as e:                              # noqa: BLE001
            results.append((w, None, mem_stats().get("peak_MB", float("nan")),
                            type(e).__name__))
            break
        finally:
            gc.collect()
    return results


# ===========================================================================
# 6.  Report
# ===========================================================================
def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def fmt(v, unit="ms", w=9):
    if v is None:
        return "  FAILED".rjust(w)
    if isinstance(v, str):
        return v[:w].rjust(w)
    return f"{v:{w}.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jades-root", default=None,
                    help="dir containing tursa/; auto-detected, or $JADES_ROOT")
    ap.add_argument("--zred", type=float, default=7.0,
                    help="redshift setting the extra_young agebin grid")
    ap.add_argument("--no-frac-obrun", action="store_true",
                    help="drop frac_obrun from theta (fiducial, not fesc)")
    ap.add_argument("--device", default="A100", choices=list(ROOFLINE),
                    help="roofline reference device")
    ap.add_argument("--peak-flops", type=float, default=None)
    ap.add_argument("--peak-bw", type=float, default=None)
    ap.add_argument("--n", type=int, default=50, help="timing repeats")
    ap.add_argument("--vmap-scan", action="store_true")
    ap.add_argument("--mem-profile", default=None, help="write pprof here")
    args = ap.parse_args()

    peak_flops, peak_bw = ROOFLINE[args.device]
    if args.peak_flops:
        peak_flops = args.peak_flops
    if args.peak_bw:
        peak_bw = args.peak_bw

    hr("ENVIRONMENT")
    print(f"  jax {jax.__version__}   backend: {jax.default_backend()}")
    for d in jax.devices():
        print(f"  device: {d}")
    print(f"  x64 enabled: {jax.config.jax_enable_x64}")
    print(f"  roofline ref: {args.device}  "
          f"{peak_flops/1e12:.1f} TFLOP/s fp32, {peak_bw/1e12:.2f} TB/s")
    print(f"  ridge point : {peak_flops/peak_bw:.1f} FLOP/byte")

    jades_root = find_jades_root(args.jades_root)
    print(f"  jades root : {jades_root}")
    csp, theta = build(jades_root, zred=args.zred,
                       frac_obrun=not args.no_frac_obrun)
    print(f"  config     : JADES production, zred={args.zred}, "
          f"frac_obrun={'yes' if not args.no_frac_obrun else 'no'}")

    # ---- grid geometry and the byte ledger --------------------------------
    L = byte_ledger(csp)
    hr("GRID GEOMETRY")
    print(f"  n_z = {L['n_z']}   n_age = {L['n_age']}   n_wave = {L['n_wave']}")
    print(f"  n_young = {L['n_young']}  ({100*L['n_young']/max(L['n_age'],1):.0f}%"
          f" of age bins carry nebular emission)")
    print(f"  n_lines = {L['n_lines']}")
    print(f"  one full (n_z,n_age,n_wave) fp32 cube = {L['cube_MB']:.1f} MB")

    hr("MEMORY LEDGER  — full-size intermediates per forward evaluation")
    print(f"  {'buffer':<44s} {'MB':>9s}   note")
    print("  " + "-" * 74)
    for name, mb, note in L["rows"]:
        print(f"  {name:<44s} {mb:9.1f}   {note}")
    print("  " + "-" * 74)
    print(f"  {'transient nebular-path total':<44s} {L['transient_MB']:9.1f}")
    print(f"  {'information-carrying subset':<44s} {L['useful_MB']:9.1f}")
    print(f"  {'inflation factor':<44s} {L['inflation']:9.1f}x")
    print("\n  Under jax.grad, XLA must keep the einsum operands alive as")
    print("  residuals for the backward pass, so multiply the transient total")
    print("  by ~2. Under vmap over chains, multiply again by n_chains.")

    # ---- stage ablation ---------------------------------------------------
    neb = csp.neb
    stages = {}

    stages["1. neb.evaluate_batch (Z,U,age trilinear + line painting)"] = (
        lambda th: csp.neb.evaluate_batch(
            th["gas_logz"], th["gas_logu"],
            csp._neb_ages_young, csp._neb_logqq_young,
            return_components=False))

    stages["2. neb.evaluate_batch_line_lum (no line painting)"] = (
        lambda th: csp.neb.evaluate_batch_line_lum(
            th["gas_logz"], th["gas_logu"],
            csp._neb_ages_young, csp._neb_logqq_young))

    stages["3. _build_neb_array (OLD: scatter into dense cube)"] = (
        lambda th: csp._build_neb_array(th, include_lines=True))

    stages["3b. evaluate_batch_factored (NEW: rank-1 in z)"] = (
        lambda th: csp.neb.evaluate_batch_factored(
            th["gas_logz"], th["gas_logu"],
            csp._neb_ages_young, csp._neb_logqq_young, include_lines=True))

    stages["3c. _neb_spectrum_term (NEW: contracted to n_wave)"] = (
        lambda th: csp._neb_spectrum_term(
            csp.calculate_ssp_weights(th).astype(jnp.float32), th,
            include_lines=True))

    stages["4. calculate_ssp_weights"] = csp.calculate_ssp_weights

    stages["5. attenuate_dust"] = (
        lambda th: csp.attenuate_dust(csp.wave, th))

    stages["6. spectrum, NO nebular"] = csp.get_spectrum_dattn_nodem_noneb
    stages["7. spectrum, WITH nebular"] = (
        lambda th: csp.get_spectrum_dattn_nodem_neb(th, include_lines=True))

    hr(f"STAGE TIMING  (median of {args.n}, ms)")
    print(f"  {'stage':<52s} {'fwd':>9s} {'grad':>9s} {'ratio':>7s}")
    print("  " + "-" * 80)

    fwd_times, grad_times = {}, {}
    for name, fn in stages.items():
        jf = jax.jit(fn)
        tf, _ = bench(jf, theta, n=args.n, label=name)
        jg = jax.jit(jax.grad(scalarise(fn)))
        tg, _ = bench(jg, theta, n=max(args.n // 2, 10), label=name)
        fwd_times[name], grad_times[name] = tf, tg
        ratio = (tg / tf) if (tf and tg) else None
        print(f"  {name:<52s} {fmt(tf)} {fmt(tg)} "
              f"{('   n/a' if ratio is None else f'{ratio:6.2f}x')}")

    t_neb_total = fwd_times.get("7. spectrum, WITH nebular")
    t_no_neb = fwd_times.get("6. spectrum, NO nebular")
    if t_neb_total and t_no_neb:
        hr("NEBULAR ATTRIBUTION")
        print(f"  spectrum with nebular : {t_neb_total:8.3f} ms")
        print(f"  spectrum without      : {t_no_neb:8.3f} ms")
        print(f"  nebular increment     : {t_neb_total - t_no_neb:8.3f} ms "
              f"({100*(t_neb_total-t_no_neb)/t_neb_total:.0f}% of the total)")
        t_eval = fwd_times.get(
            "1. neb.evaluate_batch (Z,U,age trilinear + line painting)")
        t_lum = fwd_times.get("2. neb.evaluate_batch_line_lum (no line painting)")
        t_build = fwd_times.get("3. _build_neb_array (scatter into dense cube)")
        if t_eval and t_lum:
            print(f"    of which line painting (einsum wl,zly->zwy): "
                  f"{t_eval - t_lum:8.3f} ms")
        if t_build and t_eval:
            print(f"    of which dense scatter into (n_z,n_age,n_wave): "
                  f"{t_build - t_eval:8.3f} ms")

    # ---- roofline ---------------------------------------------------------
    hr("ROOFLINE  (XLA cost analysis; whole-executable)")
    print(f"  {'stage':<52s} {'GFLOP':>8s} {'MB':>8s} {'AI':>7s}  verdict")
    print("  " + "-" * 100)
    for name, fn in stages.items():
        st = compiled_stats(fn, theta)
        if "error" in st:
            print(f"  {name:<52s}   {st['error'][:40]}")
            continue
        ai, verdict = roofline_verdict(st["flops"], st["bytes"],
                                       peak_flops, peak_bw)
        print(f"  {name:<52s} {st['flops']/1e9:8.2f} {st['bytes']/2**20:8.1f} "
              f"{ai:7.2f}  {verdict}")

    hr("XLA TEMPORARY BUFFERS  (peak scratch per call)")
    print(f"  {'stage':<52s} {'args MB':>9s} {'temp MB':>9s} {'out MB':>8s}")
    print("  " + "-" * 82)
    for name, fn in stages.items():
        st = compiled_stats(fn, theta)
        if "error" in st:
            continue
        print(f"  {name:<52s} {st['arg_bytes']/2**20:9.1f} "
              f"{st['temp_bytes']/2**20:9.1f} {st['out_bytes']/2**20:8.1f}")
    print("\n  Same table under jax.grad — this is the OOM-relevant one:")
    print(f"  {'stage':<52s} {'args MB':>9s} {'temp MB':>9s} {'out MB':>8s}")
    print("  " + "-" * 82)
    for name, fn in stages.items():
        st = compiled_stats(jax.grad(scalarise(fn)), theta)
        if "error" in st:
            continue
        print(f"  {name:<52s} {st['arg_bytes']/2**20:9.1f} "
              f"{st['temp_bytes']/2**20:9.1f} {st['out_bytes']/2**20:8.1f}")

    m = mem_stats()
    if m:
        hr("DEVICE MEMORY")
        print(f"  in use {m['in_use_MB']:.0f} MB   peak {m['peak_MB']:.0f} MB"
              f"   limit {m['limit_MB']:.0f} MB")

    # ---- vmap scaling -----------------------------------------------------
    if args.vmap_scan:
        hr("VMAP WIDTH SCAN  (gradient of the full nebular spectrum)")
        gfn = jax.grad(scalarise(
            lambda th: csp.get_spectrum_dattn_nodem_neb(th, include_lines=True)))
        print(f"  {'width':>6s} {'ms/call':>10s} {'ms/galaxy':>11s} "
              f"{'peak MB':>10s}  status")
        print("  " + "-" * 60)
        for w, t, pk, status in vmap_scan(gfn, theta):
            per = (t / w) if t else float("nan")
            print(f"  {w:6d} {fmt(t,w=10)} {per:11.3f} {pk:10.0f}  {status}")
        print("\n  If ms/galaxy keeps falling as width grows, you are")
        print("  latency-bound and batching is the single biggest available")
        print("  win. Where it stops falling is your throughput saturation;")
        print("  where it errors is your memory wall.")

    if args.mem_profile:
        jax.profiler.save_device_memory_profile(args.mem_profile)
        print(f"\n  wrote {args.mem_profile}  "
              f"(inspect: pprof --web {sys.executable} {args.mem_profile})")

    hr("HOW TO READ THIS")
    print("""
  * Compare stage 7 minus stage 6 against your 21 ms. That difference IS the
    nebular cost; everything above it is the stellar contraction and dust.
  * Stage 1 minus stage 2 isolates the line-painting einsum
    'wl,zly->zwy'. If that dominates, note that line_lum factorises as
    L[l,y] * Q[z,y] — log_line carries no z dependence, only logqq does — so
    the contraction should be done once at (n_wave, n_young) and then scaled
    by Q, not repeated n_z times. That is a free ~n_z-fold reduction in both
    FLOPs and bytes on what is likely the most expensive op in the model.
  * Stage 3 minus stage 1 isolates _build_neb_array's dense scatter. It
    allocates a full (n_z, n_age, n_wave) zeros cube to hold information that
    lives on only n_young age rows, and get_spectrum_*_neb then makes two more
    full-size copies (the kill_ion jnp.where, and stellar + neb). See the
    memory ledger for the inflation factor. Splitting the contraction
      sum_za W (S + N) A  ->  sum_za W S A  +  sum_z sum_{a in young} W N A
    removes all three temporaries and is algebraically identical.
  * In the roofline table, arithmetic intensity far below the ridge point
    means no kernel rewrite will help until you move fewer bytes.
  * The grad temp-buffer table is the one that explains OOM: XLA keeps the
    einsum operands alive as residuals, so every full-size temporary is
    charged roughly twice.
""")


if __name__ == "__main__":
    main()
