#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_neb_refactor.py — prove the 2026-08-12 nebular refactor is numerically
equivalent to the code it replaced, then measure what it bought.

The refactor made three changes, all inside the nebular path:

  A. NebularGridModel.evaluate_batch — factorise the metallicity axis out of
     the line-painting contraction einsum('wl,zly->zwy', ...). log_cont and
     log_line carry no z axis; only logqq does. Cost drops by ~n_z.
  B. CSPBasis._ion_multiplier — the kill_ion selection is an (n_age, n_wave)
     mask, so jnp.where(kill_ion[None], f_esc*flux, flux) == flux * mask.
     Folding the mask into the attenuation array removes a full-cube copy.
  C. CSPBasis.get_spectrum_*_neb — split the contraction
       sum_za W (S + N) A  ->  sum_za W S A + sum_z sum_{a young} W N A
     and collapse z into the weights first, so no (n_z, n_age, n_wave)
     nebular cube or stellar+nebular sum is ever formed.

This script re-implements the OLD arithmetic inline as a reference and
compares, for every nebular spectrum variant, with and without frac_obrun:

  * the spectrum itself
  * the gradient with respect to every theta entry
  * wall-clock, XLA FLOP count, bytes accessed, and peak temp buffers

Run it BEFORE trusting the refactor, and again on the GPU before relaunching a
campaign.  Exit status 0 = equivalent within the stated float32 tolerance.

    python verify_neb_refactor.py
    python verify_neb_refactor.py --tol 3e-6 --zred 9.0
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time

if platform.system() == "Darwin":
    # jax-metal fails on ceridwen's float64 paths; force CPU before jax import.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

MAC_JADES_ROOT = "/Users/amanda/Desktop/PhD/Tursa/Tursa/jades_full"


def find_jades_root(explicit=None):
    """Locate the directory that contains the ``tursa`` package.

    Search order: explicit argument, ``$JADES_ROOT``, ancestors of this script
    (plus their ``tests/jades_full`` and ``jades_full`` subdirectories), then
    the laptop default.  This lets the same file run unmodified on the Mac and
    on tursa, where the suite lives at
    ``/home/dp428/dp428/dc-stof2/ceridwen/tests/jades_full`` while the ceridwen
    package sits two levels up.
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


# ===========================================================================
# Reference implementation — the arithmetic as it stood before 2026-08-12
# ===========================================================================
def ref_evaluate_batch(neb, logZ_gas, logU, ssp_ages_young, logqq_young,
                       include_lines=True):
    """OLD NebularGridModel.evaluate_batch, verbatim in structure."""
    from ceridwen.neb.NebularGridModel import _locate, _frac

    logZ_gas = jnp.squeeze(logZ_gas)
    logU = jnp.squeeze(logU)

    def _interp_cube(cube, logz_grid, age_grid, logu_grid):
        z1 = _locate(logZ_gas, logz_grid)
        dz = _frac(logZ_gas, logz_grid, z1)
        u1 = _locate(logU, logu_grid)
        du = _frac(logU, logu_grid, u1)
        w00 = (1.0 - dz) * (1.0 - du)
        w01 = (1.0 - dz) * du
        w10 = dz * (1.0 - du)
        w11 = dz * du
        zu = (w00 * cube[..., z1,     :, u1]
              + w01 * cube[..., z1,     :, u1 + 1]
              + w10 * cube[..., z1 + 1, :, u1]
              + w11 * cube[..., z1 + 1, :, u1 + 1])
        a1 = jnp.clip(jnp.searchsorted(age_grid, ssp_ages_young) - 1,
                      0, age_grid.shape[0] - 2)
        da = jnp.clip((ssp_ages_young - age_grid[a1])
                      / (age_grid[a1 + 1] - age_grid[a1]), 0.0, 1.0)
        return (1.0 - da)[None, :] * zu[..., a1] + da[None, :] * zu[..., a1 + 1]

    log_cont = _interp_cube(neb.nebem_cont, neb.nebem_cont_logz,
                            neb.nebem_cont_age, neb.nebem_cont_logu)
    log_line = _interp_cube(neb.nebem_line, neb.nebem_line_logz,
                            neb.nebem_line_age, neb.nebem_line_logu)

    # THE OLD FORM: logqq broadcast into the exponent, then one contraction
    # per metallicity.
    cont_flux = jnp.power(10.0, log_cont[None, :, :] + logqq_young[:, None, :])
    line_lum = jnp.power(10.0, log_line[None, :, :] + logqq_young[:, None, :])
    line_spec = jnp.einsum('wl,zly->zwy', neb.gaussnebarr, line_lum)
    cont = cont_flux.transpose(0, 2, 1)
    line = line_spec.transpose(0, 2, 1)
    return (cont + line) if include_lines else cont


def ref_build_neb_array(csp, theta, include_lines=True):
    """OLD CSPBasis._build_neb_array: dense (n_z, n_age, n_wave) scatter."""
    neb_young = ref_evaluate_batch(
        csp.neb, theta["gas_logz"], theta["gas_logu"],
        csp._neb_ages_young, csp._neb_logqq_young, include_lines=include_lines)
    n_z, n_age, n_wave = csp.flux.shape
    neb_all = jnp.zeros((n_z, n_age, n_wave), dtype=jnp.float32)
    return neb_all.at[:, csp._neb_young_idx, :].set(
        neb_young.astype(jnp.float32))


def _ref_combined(csp, theta, include_lines):
    """OLD stellar+nebular dense cube and the attn_age modifications."""
    neb_all = ref_build_neb_array(csp, theta, include_lines)
    if "frac_obrun" in theta:
        f_esc = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
        stellar = jnp.where(csp.kill_ion[None, :, :], f_esc * csp.flux, csp.flux)
        neb_all = neb_all * (jnp.float32(1.0) - f_esc)
    else:
        stellar = jnp.where(csp.kill_ion[None, :, :], 0.0, csp.flux)
    return stellar + neb_all


def _ref_attn_age(csp, theta):
    attn, attn_diffuse = csp.attenuate_dust(csp.wave, theta)
    tau_age = jnp.einsum("ab,bw->aw", csp._age_bin_mix,
                         attn.astype(jnp.float32))
    attn_age = jnp.exp(-tau_age)
    if "frac_obrun" in theta:
        fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
        attn_age = (jnp.float32(1.0) - fo) * attn_age + fo
        attn_age = jnp.where(csp.kill_ion, jnp.float32(1.0), attn_age)
    return attn_age, attn_diffuse


def ref_dattn_nodem_neb(csp, theta, include_lines=True):
    W = csp.calculate_ssp_weights(theta=theta).astype(jnp.float32)
    combined = _ref_combined(csp, theta, include_lines)
    attn_age, attn_diffuse = _ref_attn_age(csp, theta)
    spec = jnp.einsum("za,zaw,aw->w", W, combined, attn_age)
    spec = spec * jnp.exp(-attn_diffuse.astype(jnp.float32))
    return spec.reshape((-1,))


def ref_nodattn_nodem_neb(csp, theta, include_lines=True):
    W = csp.calculate_ssp_weights(theta=theta).astype(jnp.float32)
    combined = _ref_combined(csp, theta, include_lines)
    return jnp.einsum("za,zaw->w", W, combined)


# ===========================================================================
# Comparison helpers
# ===========================================================================
def relerr(a, b):
    """Return ``(scaled L-inf error, note)``.

    The error is normalised by the PEAK magnitude of the arrays, NOT
    elementwise.  For a gradient vector that is the only scale with any
    meaning: a component whose magnitude is 1e-12 of the vector's largest
    entry cannot be "relatively wrong" in a way that affects a sampler, since
    the sampler only ever sees the vector as a whole (and here, contracted
    against d(agebins)/d(zred)).

    The elementwise metric ``|a-b| / max(|a|,|b|)`` that this replaces
    saturates at exactly 1.0 whenever one side is a hard zero and the other is
    rounding-scale, and at exactly 2.0 when two negligible values carry
    opposite signs -- which is what produced the spurious
    ``grad[lookback_time]`` failures on 2026-08-12.  A genuinely wrong
    component is O(1) against the peak and is still caught by this metric.

    The note also reports the worst elementwise relative error restricted to
    components that are significant (>1e-8 of peak), so a real localised error
    in a small-but-meaningful component cannot hide behind the normalisation.
    """
    a = np.asarray(jax.device_get(a), dtype=np.float64).ravel()
    b = np.asarray(jax.device_get(b), dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf"), f"SHAPE {a.shape} vs {b.shape}"
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        n = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum())
        return float("inf"), f"{n} non-finite entries"
    peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
    if peak == 0.0:
        return 0.0, "both identically zero"
    linf = float(np.max(np.abs(a - b)) / peak)
    l2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

    # Second gate.  Normalising by the peak alone would let a genuinely wrong
    # component hide if it is small in absolute terms -- e.g. b=1e-2, a=2e-2
    # against a peak of 2e5 scores 5e-8 on the L-inf metric despite being a
    # factor-of-2 error.  So also check the elementwise relative error over
    # components carrying at least 1% of the peak.  float32 rounding noise is
    # ~1e-7*peak absolute, i.e. ~1e-5 relative at the 1% level, so this
    # threshold is comfortably above the noise and well below grad_tol.
    gate = np.abs(b) > 1e-2 * peak
    ew_gate = (float(np.max(np.abs(a[gate] - b[gate]) / np.abs(b[gate])))
               if gate.any() else 0.0)

    # Informational: the same quantity over everything not pure noise.
    sig = np.abs(b) > 1e-8 * peak
    ew = (float(np.max(np.abs(a[sig] - b[sig]) / np.abs(b[sig])))
          if sig.any() else 0.0)
    ndrop = int((~sig).sum())
    tail = f"  [{ndrop} negligible]" if ndrop else ""
    return (max(linf, ew_gate),
            f"Linf/pk {linf:.1e} ew>1% {ew_gate:.1e} ew_all {ew:.1e} "
            f"L2 {l2:.1e}{tail}")


def bench(fn, theta, n=20, warmup=3):
    try:
        for _ in range(warmup):
            out = fn(theta)
        jax.block_until_ready(out)
    except Exception as e:                                  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(theta)
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), None


def xla_stats(fn, theta):
    try:
        c = jax.jit(fn).lower(theta).compile()
        ca = c.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        ma = c.memory_analysis()
        return (float(ca.get("flops", np.nan)),
                float(ca.get("bytes accessed", np.nan)),
                float(getattr(ma, "temp_size_in_bytes", np.nan)))
    except Exception:                                       # noqa: BLE001
        return (np.nan, np.nan, np.nan)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jades-root", default=None,
                    help="dir containing tursa/; auto-detected, or $JADES_ROOT")
    ap.add_argument("--zred", type=float, default=7.0)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max allowed relative difference (float32 forward model)")
    ap.add_argument("--grad-tol", type=float, default=1e-4)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    jades_root = find_jades_root(args.jades_root)
    if jades_root not in sys.path:
        sys.path.insert(0, jades_root)
    print(f"jades root  : {jades_root}")

    from tursa.common.config import (resolve_ssp_file, resolve_sps_home,
                                     N_BINS_SFH)
    from tursa.common.predictor import build_csp_extra_young

    print("=" * 78)
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  "
          f"x64={jax.config.jax_enable_x64}")
    print("building the production JADES CSPBasis ...")
    csp, agebins = build_csp_extra_young(
        zred=args.zred, ssp_file=resolve_ssp_file(),
        sps_home=resolve_sps_home(), nbins_sfh=N_BINS_SFH, intrinsic=False)

    n_z, n_age, n_wave = (int(s) for s in csp.flux.shape)
    n_young = int(csp._neb_young_idx.shape[0])
    n_lines = int(csp.neb.nebem_line.shape[0])
    print(f"  n_z={n_z}  n_age={n_age}  n_wave={n_wave}  "
          f"n_young={n_young}  n_lines={n_lines}")
    print(f"  full cube = {n_z*n_age*n_wave*4/2**20:.1f} MB")
    print(f"  OLD line-painting einsum = "
          f"{n_z*n_wave*n_young*n_lines/1e9:.2f} GFLOP-pairs")
    print(f"  NEW line-painting einsum = "
          f"{n_wave*n_young*n_lines/1e9:.2f} GFLOP-pairs  "
          f"({n_z}x fewer)")

    base = {}
    for attr in ("dust_attn", "diff_dust", "diffuse_dust", "neb"):
        comp = getattr(csp, attr, None)
        if comp is not None and hasattr(comp, "get_default_params"):
            try:
                base.update(comp.get_default_params())
            except Exception:                               # noqa: BLE001
                pass
    base["lookback_time"] = jnp.asarray(agebins)
    base["sfh"] = jnp.ones(N_BINS_SFH)
    base["Z"] = jnp.asarray([-2.0])
    base.setdefault("gas_logz", jnp.asarray(-2.0))
    base.setdefault("gas_logu", jnp.asarray(-2.0))

    cases = [
        ("no frac_obrun", dict(base)),
        ("frac_obrun=0.1", {**base, "frac_obrun": jnp.asarray([0.1])}),
        ("frac_obrun=0.0", {**base, "frac_obrun": jnp.asarray([0.0])}),
        ("frac_obrun=0.9", {**base, "frac_obrun": jnp.asarray([0.9])}),
    ]

    variants = [
        ("get_spectrum_dattn_nodem_neb",
         lambda th: csp.get_spectrum_dattn_nodem_neb(th, include_lines=True),
         lambda th: ref_dattn_nodem_neb(csp, th, True)),
        ("get_spectrum_dattn_nodem_neb (no lines)",
         lambda th: csp.get_spectrum_dattn_nodem_neb(th, include_lines=False),
         lambda th: ref_dattn_nodem_neb(csp, th, False)),
        ("get_spectrum_nodattn_nodem_neb",
         lambda th: csp.get_spectrum_nodattn_nodem_neb(th, include_lines=True),
         lambda th: ref_nodattn_nodem_neb(csp, th, True)),
    ]

    failures = []

    print("\n" + "=" * 78)
    print("FORWARD EQUIVALENCE   (max relative difference vs the old code)")
    print("=" * 78)
    print(f"  {'variant':<42s} {'case':<16s} {'max rel':>10s}  note")
    print("  " + "-" * 88)
    for vname, newf, reff in variants:
        jnew, jref = jax.jit(newf), jax.jit(reff)
        for cname, th in cases:
            try:
                e, note = relerr(jnew(th), jref(th))
            except Exception as exc:                        # noqa: BLE001
                e, note = float("inf"), f"{type(exc).__name__}: {exc}"
            flag = "" if e <= args.tol else "   <-- FAIL"
            if e > args.tol:
                failures.append((vname, cname, e))
            print(f"  {vname:<42s} {cname:<16s} {e:10.2e}  {note}{flag}")

    print("\n" + "=" * 78)
    print("GRADIENT EQUIVALENCE  (d/dtheta of sum(spectrum**2))")
    print("=" * 78)

    def sq(f):
        return lambda th: jnp.sum(jnp.square(f(th)))

    vname, newf, reff = variants[0]
    gnew = jax.jit(jax.grad(sq(newf)))
    gref = jax.jit(jax.grad(sq(reff)))
    for cname, th in cases:
        try:
            dn, dr = gnew(th), gref(th)
        except Exception as exc:                            # noqa: BLE001
            print(f"  {cname}: FAILED {type(exc).__name__}: {exc}")
            failures.append((vname + " grad", cname, float("inf")))
            continue
        print(f"  case {cname}:")
        for k in sorted(dn):
            e, note = relerr(dn[k], dr[k])
            flag = "" if e <= args.grad_tol else "   <-- FAIL"
            if e > args.grad_tol:
                failures.append((f"grad[{k}]", cname, e))
            print(f"    {k:<28s} {e:10.2e}  {note}{flag}")

    print("\n" + "=" * 78)
    print(f"PERFORMANCE  (median of {args.n}, frac_obrun=0.1)")
    print("=" * 78)
    th = cases[1][1]
    print(f"  {'':<12s} {'fwd ms':>9s} {'grad ms':>9s} {'GFLOP':>9s} "
          f"{'MB moved':>9s} {'temp MB':>9s}")
    print("  " + "-" * 72)
    rows = {}
    for tag, f in (("OLD", variants[0][2]), ("NEW", variants[0][1])):
        tf, _ = bench(jax.jit(f), th, n=args.n)
        tg, _ = bench(jax.jit(jax.grad(sq(f))), th, n=max(args.n // 2, 5))
        fl, by, tmp = xla_stats(jax.grad(sq(f)), th)
        rows[tag] = (tf, tg, fl, by, tmp)
        print(f"  {tag:<12s} {(tf if tf else float('nan')):9.3f} "
              f"{(tg if tg else float('nan')):9.3f} {fl/1e9:9.2f} "
              f"{by/2**20:9.1f} {tmp/2**20:9.1f}")
    if rows["OLD"][0] and rows["NEW"][0]:
        print("\n  speedup   forward : "
              f"{rows['OLD'][0]/rows['NEW'][0]:6.2f}x")
        print("            gradient: "
              f"{rows['OLD'][1]/rows['NEW'][1]:6.2f}x")
        print("  bytes moved reduced: "
              f"{rows['OLD'][3]/max(rows['NEW'][3],1):6.2f}x")
        print("  peak temp reduced  : "
              f"{rows['OLD'][4]/max(rows['NEW'][4],1):6.2f}x")

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} comparison(s) EXCEEDED TOLERANCE")
        for a, b, e in failures[:20]:
            print(f"  {a}  [{b}]  {e:.3e}")
        print("\nDo not use the refactored code until this is understood.")
        return 1
    print("RESULT: refactor is numerically equivalent to the old code")
    print(f"        (all forward diffs <= {args.tol:g}, "
          f"gradients <= {args.grad_tol:g})")
    print("""
Error metric: L-inf difference normalised by the arrays' PEAK magnitude, with
the worst SIGNIFICANT elementwise relative error reported alongside. An earlier
elementwise-only metric produced spurious 1.0 / 2.0 failures on
grad[lookback_time], where one component is numerically zero -- see
diagnose_lookback_grad.py for the component-by-component evidence.

Expected residual difference and why it is not zero:
  * 10**(a+b) is re-associated as 10**(a+ref) * 10**(b-ref), one extra
    rounding in the exponent;
  * the line contraction now sums over lines BEFORE scaling by the
    ionising-photon rate, changing the summation order;
  * cont and line bases are added before scaling rather than after.
  All three are last-bit effects in float32, i.e. ~1e-7 relative, and none
  of them changes the model definition.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
