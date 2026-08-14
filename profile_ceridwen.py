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
    "auto":    (None, None),          # resolved from the live device
    "A100":    (19.5e12, 1.555e12),   # SXM4-40GB, HBM2
    "A100-80": (19.5e12, 2.039e12),   # SXM4/PCIe-80GB, HBM2e
    "V100":    (15.7e12, 0.900e12),
    "H100":    (67.0e12, 3.350e12),   # SXM5-80GB
    "cpu":     (0.20e12, 0.050e12),
}


def autodetect_roofline():
    """Resolve (peak_flops, peak_bw, label) from the live device.

    The A100 ships in 40 GB (HBM2, 1.555 TB/s) and 80 GB (HBM2e, 2.039 TB/s)
    variants with IDENTICAL fp32 throughput, so the memory capacity is what
    distinguishes them -- and getting it wrong by 31% silently shifts every
    bandwidth-bound verdict in the report.  Detect rather than trust a flag.
    """
    try:
        d = jax.devices()[0]
    except Exception:                                       # noqa: BLE001
        return ROOFLINE["cpu"] + ("cpu (no device)",)
    if d.platform == "cpu":
        return ROOFLINE["cpu"] + ("cpu",)
    kind = (getattr(d, "device_kind", "") or "").upper()
    gib = 0.0
    try:
        gib = (d.memory_stats() or {}).get("bytes_limit", 0) / 2**30
    except Exception:                                       # noqa: BLE001
        pass
    if "H100" in kind:
        return ROOFLINE["H100"] + (f"H100 ({gib:.0f} GiB)",)
    if "V100" in kind:
        return ROOFLINE["V100"] + (f"V100 ({gib:.0f} GiB)",)
    if "A100" in kind:
        # bytes_limit is ~90-95% of physical, so 40 GB reads as ~36 GiB.
        if gib > 50:
            return ROOFLINE["A100-80"] + (f"A100-80GB ({gib:.0f} GiB visible)",)
        return ROOFLINE["A100"] + (f"A100-40GB ({gib:.0f} GiB visible)",)
    return ROOFLINE["A100"] + (f"UNKNOWN '{kind}' -- assuming A100-40GB",)


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


def roofline_verdict(flops, byts, peak_flops, peak_bw, t_meas_ms=None):
    """Return ``(ai, floor_ms, ratio, verdict)``.

    Arithmetic intensity vs the ridge point tells you which resource *would*
    limit the kernel if it were running at hardware speed.  It does NOT tell
    you whether the kernel actually is limited by that resource -- for that you
    must compare the MEASURED time against the roofline floor.  A kernel far
    above its own floor is limited by neither bandwidth nor FLOPs but by
    per-launch overhead, and the only fix is a bigger batch, not a better
    kernel.  Reporting AI alone (as this function used to) labelled everything
    "BANDWIDTH-BOUND" even when it was 20x above the bandwidth floor.
    """
    if not byts or not np.isfinite(byts) or byts == 0:
        return float("nan"), float("nan"), float("nan"), "n/a"
    ai = flops / byts
    ridge = peak_flops / peak_bw
    floor_ms = max(byts / peak_bw, flops / peak_flops) * 1e3
    limiter = "bw" if ai < ridge else "flops"
    if t_meas_ms is None or not np.isfinite(t_meas_ms) or floor_ms <= 0:
        return ai, floor_ms, float("nan"), f"{limiter}-limited in principle"
    ratio = t_meas_ms / floor_ms
    if ratio < 2.0:
        v = f"AT ROOFLINE ({limiter}-bound)"
    elif ratio < 5.0:
        v = f"near roofline ({limiter})"
    else:
        v = "LATENCY-BOUND -> batch it"
    return ai, floor_ms, ratio, v


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
    """The full-size intermediates the OLD nebular path materialised, in MB.

    HISTORICAL as of the 2026-08-12 refactor: rows 2-6 are what
    ``_build_neb_array`` + the dense ``stellar + neb`` sum used to allocate on
    every forward call.  The current code forms NONE of them -- see the XLA
    temp-buffer table for what it actually allocates now.  The ledger is kept
    because it is the clearest statement of what the refactor removed, and
    because ``_build_neb_array`` is still live on the picket-fence branch.
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
    ap.add_argument("--device", default="auto", choices=list(ROOFLINE),
                    help="roofline reference device ('auto' reads the live GPU)")
    ap.add_argument("--peak-flops", type=float, default=None)
    ap.add_argument("--peak-bw", type=float, default=None)
    ap.add_argument("--n", type=int, default=50, help="timing repeats")
    ap.add_argument("--vmap-scan", action="store_true")
    ap.add_argument("--mem-profile", default=None, help="write pprof here")
    args = ap.parse_args()

    if args.device == "auto":
        peak_flops, peak_bw, dev_label = autodetect_roofline()
    else:
        peak_flops, peak_bw = ROOFLINE[args.device]
        dev_label = f"{args.device} (forced by --device)"
    if args.peak_flops:
        peak_flops, dev_label = args.peak_flops, dev_label + " +override"
    if args.peak_bw:
        peak_bw, dev_label = args.peak_bw, dev_label + " +override"

    hr("ENVIRONMENT")
    print(f"  jax {jax.__version__}   backend: {jax.default_backend()}")
    for d in jax.devices():
        print(f"  device: {d}")
    # NB: ceridwen enables x64 at import time, which happens inside build()
    # below -- so this reads False here and True in ceridwen's own banner.
    # Same process, no contradiction; the flag is flipped between the two lines.
    print(f"  x64 enabled: {jax.config.jax_enable_x64} (before ceridwen import)")
    print(f"  roofline ref: {dev_label}  "
          f"{peak_flops/1e12:.1f} TFLOP/s fp32, {peak_bw/1e12:.3f} TB/s")
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

    hr("MEMORY LEDGER  — what the OLD path allocated  [HISTORICAL]")
    print("  These buffers were REMOVED by the 2026-08-12 refactor.  The")
    print("  current code allocates none of them; see the XLA temp table below")
    print("  for what it actually allocates.  Shown to quantify the change.")
    print()
    print(f"  {'buffer (old path)':<44s} {'MB':>9s}   note")
    print("  " + "-" * 74)
    for name, mb, note in L["rows"]:
        print(f"  {name:<44s} {mb:9.1f}   {note}")
    print("  " + "-" * 74)
    print(f"  {'transient nebular-path total (OLD)':<44s} {L['transient_MB']:9.1f}")
    print(f"  {'information-carrying subset':<44s} {L['useful_MB']:9.1f}")
    print(f"  {'inflation factor (OLD)':<44s} {L['inflation']:9.1f}x")

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

    # Weights precomputed as a CONSTANT so this stage times the nebular term
    # alone.  Folding calculate_ssp_weights (stage 4) into it would double-count
    # ~0.19 ms and make the new path look no faster than the old one.
    _W_const = csp.calculate_ssp_weights(theta).astype(jnp.float32)
    stages["3c. _neb_spectrum_term (NEW: neb term only, W precomputed)"] = (
        lambda th: csp._neb_spectrum_term(_W_const, th, include_lines=True))

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
    hr("ROOFLINE  (XLA cost analysis vs MEASURED forward time)")
    print(f"  {'stage':<48s} {'GFLOP':>7s} {'MiB':>7s} {'AI':>6s} "
          f"{'floor ms':>9s} {'meas ms':>8s} {'x floor':>8s}  verdict")
    print("  " + "-" * 118)
    for name, fn in stages.items():
        st = compiled_stats(fn, theta)
        if "error" in st:
            print(f"  {name:<48s}   {st['error'][:40]}")
            continue
        tm = fwd_times.get(name)
        ai, floor_ms, ratio, verdict = roofline_verdict(
            st["flops"], st["bytes"], peak_flops, peak_bw, tm)

        def _f(x, w, p):
            return (f"{x:{w}.{p}f}" if isinstance(x, float) and np.isfinite(x)
                    else "n/a".rjust(w))
        print(f"  {name:<48s} {st['flops']/1e9:7.3f} {st['bytes']/2**20:7.1f} "
              f"{_f(ai,6,2)} {_f(floor_ms,9,4)} "
              f"{_f(tm if tm else float('nan'),8,3)} {_f(ratio,8,1)}  {verdict}")
    print("\n  'x floor' is measured time divided by the hardware floor. Values")
    print("  >>1 mean the kernel is limited by per-launch overhead, not by")
    print("  bandwidth or FLOPs -- batching is the only fix. AI vs the ridge")
    print("  point tells you which resource would bind at hardware speed; the")
    print("  ratio tells you whether you are anywhere near it.")

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
  * Stage 3 vs 3b/3c contrasts the OLD dense-cube path against the factored
    one.  Stage 3c has the SSP weights precomputed, so compare it to
    stage 3 minus nothing (both exclude calculate_ssp_weights, stage 4).
  * The MEMORY LEDGER is historical -- those buffers no longer exist.  The XLA
    temp-buffer tables are the live numbers.  The grad table is the
    OOM-relevant one: XLA keeps einsum operands alive as backward residuals.
  * In the ROOFLINE table look at 'x floor', not the AI column.  Everything in
    this model has AI well below the ridge point, so AI alone labels
    everything bandwidth-bound.  What matters is whether the measured time is
    NEAR the floor (then bytes are the lever) or far above it (then per-launch
    overhead is the lever, and only batching helps).
  * The VMAP WIDTH SCAN is therefore the most important table here.  Read
    ms/galaxy, not ms/call.  Where it stops falling is your throughput
    saturation; that asymptote is the per-galaxy cost you can actually reach.
    Compare it against the stage-7 'floor ms' -- if they agree, the batched
    kernel is running at hardware speed and no further code change will help.
  * Production consequence: tursa/common/ns_build.py evaluates the likelihood
    vmapped over ns_eval_chunk particles (default 8) via lax.scan.  With
    num_delete=25 that is 4 sequential launches per inner step.  The padding in
    _chunked_loglikelihood is discarded before the return, so ns_eval_chunk is
    a PURE performance knob -- it cannot change a single likelihood value.
    Setting it >= num_delete collapses those 4 launches into 1.  num_delete
    itself is NOT free: it changes the nested-sampling shells, so any increase
    needs logZ and the marginals re-checked.
""")


if __name__ == "__main__":
    main()
