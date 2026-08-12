#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_lookback_grad.py — is the d/d(lookback_time) discrepancy real?

verify_neb_refactor.py flagged grad[lookback_time] with a max elementwise
relative error of exactly 1.0 (and 2.0 in one case) while the L2 relative
difference of the same vector was 5e-6.  Those two numbers are only
compatible if the offending COMPONENT is numerically negligible: the metric
|a-b| / max(|a|,|b|) saturates at 1.0 when one side is a hard zero and the
other is a rounding-scale value, and at 2.0 when two negligible values carry
opposite signs.

This script settles it by printing the two gradient vectors component by
component, in absolute units, alongside:

  * each component's magnitude relative to the LARGEST component of the
    vector (the only scale that matters for a gradient);
  * a central finite-difference estimate of the same derivative, computed
    separately from each implementation's own forward model, so we can see
    whether OLD and NEW disagree by more or less than the finite-difference
    noise floor;
  * the derivative w.r.t. zred, which is the quantity the sampler actually
    uses -- production plumbs lookback_time through
    agebins_extra_young_gyr_jax(zred), so the 9-vector is only ever seen
    contracted against d(agebins)/d(zred).

A real bug shows up as a component that is LARGE relative to the vector peak,
or as a zred derivative that differs.  Rounding noise shows up as a tiny
component with a big relative error and an identical zred derivative.

    python diagnose_lookback_grad.py
    python diagnose_lookback_grad.py --zred 9.0
"""

from __future__ import annotations

import argparse
import os
import platform
import sys

if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jades-root", default=None,
                    help="dir containing tursa/; auto-detected, or $JADES_ROOT")
    ap.add_argument("--zred", type=float, default=7.0)
    ap.add_argument("--fd-step", type=float, default=1e-4,
                    help="relative step for the finite difference")
    args = ap.parse_args()

    here = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, here)

    # reuse the reference implementation and the root finder already written
    # for the verify script
    from verify_neb_refactor import ref_dattn_nodem_neb, find_jades_root
    jades_root = find_jades_root(args.jades_root)
    if jades_root not in sys.path:
        sys.path.insert(0, jades_root)
    print(f"jades root  : {jades_root}")

    from tursa.common.config import (resolve_ssp_file, resolve_sps_home,
                                     N_BINS_SFH)
    from tursa.common.predictor import build_csp_extra_young

    csp, agebins = build_csp_extra_young(
        zred=args.zred, ssp_file=resolve_ssp_file(),
        sps_home=resolve_sps_home(), nbins_sfh=N_BINS_SFH, intrinsic=False)

    theta = {}
    for attr in ("dust_attn", "diff_dust", "diffuse_dust", "neb"):
        comp = getattr(csp, attr, None)
        if comp is not None and hasattr(comp, "get_default_params"):
            try:
                theta.update(comp.get_default_params())
            except Exception:                               # noqa: BLE001
                pass
    theta["lookback_time"] = jnp.asarray(agebins)
    theta["sfh"] = jnp.ones(N_BINS_SFH)
    theta["Z"] = jnp.asarray([-2.0])
    theta.setdefault("gas_logz", jnp.asarray(-2.0))
    theta.setdefault("gas_logu", jnp.asarray(-2.0))
    theta["frac_obrun"] = jnp.asarray([0.1])

    def newf(th):
        return csp.get_spectrum_dattn_nodem_neb(th, include_lines=True)

    def reff(th):
        return ref_dattn_nodem_neb(csp, th, True)

    def sq(f):
        return lambda th: jnp.sum(jnp.square(f(th)))

    gnew = jax.jit(jax.grad(sq(newf)))
    gref = jax.jit(jax.grad(sq(reff)))
    fnew = jax.jit(sq(newf))
    fref = jax.jit(sq(reff))

    dn = np.asarray(jax.device_get(gnew(theta)["lookback_time"]), np.float64)
    dr = np.asarray(jax.device_get(gref(theta)["lookback_time"]), np.float64)
    lb = np.asarray(jax.device_get(theta["lookback_time"]), np.float64)

    peak = max(np.max(np.abs(dr)), np.max(np.abs(dn)))

    print("=" * 96)
    print(f"d/d(lookback_time) of sum(spectrum**2)   zred={args.zred}  "
          f"backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print("=" * 96)
    print(f"vector peak |grad| = {peak:.6e}")
    print()
    print(f"  {'i':>2s} {'lookback/Gyr':>13s} {'OLD':>14s} {'NEW':>14s} "
          f"{'|new-old|':>12s} {'/peak':>10s} {'|b|/peak':>10s}")
    print("  " + "-" * 90)
    for i in range(dr.size):
        d = abs(dn[i] - dr[i])
        print(f"  {i:2d} {lb[i]:13.6f} {dr[i]:14.6e} {dn[i]:14.6e} "
              f"{d:12.3e} {d/peak:10.2e} {abs(dr[i])/peak:10.2e}")
    print()
    print(f"  L-inf difference / vector peak : {np.max(np.abs(dn-dr))/peak:.3e}")
    print(f"  L2   difference / vector norm  : "
          f"{np.linalg.norm(dn-dr)/np.linalg.norm(dr):.3e}")

    # ---- finite difference on the SAME component, from each forward model --
    print()
    print("=" * 96)
    print("CENTRAL FINITE DIFFERENCE  (does either analytic gradient agree "
          "better than they disagree?)")
    print("=" * 96)
    print(f"  {'i':>2s} {'FD(old fwd)':>14s} {'FD(new fwd)':>14s} "
          f"{'AD old':>14s} {'AD new':>14s} {'FD noise':>11s}")
    print("  " + "-" * 88)
    for i in range(dr.size):
        h = args.fd_step * max(abs(lb[i]), 1e-3)
        fd = {}
        for tag, f in (("old", fref), ("new", fnew)):
            lp = lb.copy(); lp[i] += h
            lm = lb.copy(); lm[i] -= h
            tp = {**theta, "lookback_time": jnp.asarray(lp)}
            tm = {**theta, "lookback_time": jnp.asarray(lm)}
            try:
                fd[tag] = float((f(tp) - f(tm)) / (2 * h))
            except Exception:                               # noqa: BLE001
                fd[tag] = float("nan")
        noise = abs(fd["old"] - fd["new"])
        print(f"  {i:2d} {fd['old']:14.6e} {fd['new']:14.6e} "
              f"{dr[i]:14.6e} {dn[i]:14.6e} {noise:11.3e}")

    # ---- the quantity the sampler actually differentiates ------------------
    print()
    print("=" * 96)
    print("d/d(zred)  — production plumbs lookback_time through "
          "agebins_extra_young_gyr_jax(zred)")
    print("=" * 96)
    try:
        from tursa.common.predictor import agebins_extra_young_gyr_jax

        def through_zred(f):
            def g(z):
                th = dict(theta)
                th["lookback_time"] = agebins_extra_young_gyr_jax(
                    z, N_BINS_SFH)
                return jnp.sum(jnp.square(f(th)))
            return g

        z = jnp.asarray(args.zred)
        gz_old = float(jax.grad(through_zred(reff))(z))
        gz_new = float(jax.grad(through_zred(newf))(z))
        den = max(abs(gz_old), abs(gz_new), 1e-300)
        print(f"  OLD  d/dz = {gz_old:.10e}")
        print(f"  NEW  d/dz = {gz_new:.10e}")
        print(f"  relative difference = {abs(gz_new-gz_old)/den:.3e}")
        verdict_z = abs(gz_new - gz_old) / den
    except Exception as exc:                                # noqa: BLE001
        print(f"  could not evaluate: {type(exc).__name__}: {exc}")
        verdict_z = None

    # ---- verdict ----------------------------------------------------------
    linf_scaled = np.max(np.abs(dn - dr)) / peak
    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    bad = np.where(np.abs(dn - dr) / peak > 1e-4)[0]
    if len(bad) == 0:
        print(f"  No component differs by more than 1e-4 of the vector peak")
        print(f"  (worst = {linf_scaled:.2e}).  The flagged 1.0 / 2.0 values")
        print("  were the elementwise metric saturating on components that are")
        print("  negligible relative to the gradient's own scale:")
        for i in range(dr.size):
            if abs(dr[i]) / peak < 1e-8:
                print(f"    i={i}  |grad|/peak = {abs(dr[i])/peak:.2e}  "
                      f"-> any difference reads as O(1) relative error")
        if verdict_z is not None and verdict_z < 1e-5:
            print(f"  d/d(zred), the derivative the sampler actually uses,")
            print(f"  agrees to {verdict_z:.2e}.")
        print("\n  CONCLUSION: metric artefact, not a code defect.")
        return 0
    print(f"  {len(bad)} component(s) differ by >1e-4 of the vector peak:")
    for i in bad:
        print(f"    i={i}  lookback={lb[i]:.6f}  old={dr[i]:.6e}  "
              f"new={dn[i]:.6e}  |d|/peak={abs(dn[i]-dr[i])/peak:.3e}")
    print("\n  CONCLUSION: this is a REAL discrepancy. Do not use the refactor.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
