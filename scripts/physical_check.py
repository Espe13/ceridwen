#!/usr/bin/env python
"""T5, the physical-output check: does a change keep every spectrum, photometry and line flux
within the physical standard of the float64 code before it?

    The standard (docs/development.md, "The checks"): no model-grid spectrum, projected
    spectrum, photometric maggies, Lines prediction or predicted line flux moves by more than
    1e-3 relative, per element where |value| > 1e-4 max|array|, and by more than 1e-3 of
    max|array| below that floor; at every configuration and theta tested.

Usage (as bit_identity_check.py: save on the old commit, compare on the new):

    python scripts/physical_check.py --save /tmp/phys_old.npz      # on the old commit
    python scripts/physical_check.py --compare /tmp/phys_old.npz   # on the new commit

Configurations: every one of scripts/bit_identity_check.py's (the same builders, read from its
``collect``), plus ``all_components`` (nebular, dust emission, the corrected AGN with its lines,
the IGM with a DLA, a sampled LSF scale). Theta: bit_identity_check's 16 draws around
theta_init plus four prior extremes: ``faint`` (logmass 8, old SFH, maximum dust, low
metallicity), ``bright`` (logmass 12, young SFH, no dust), ``low`` and ``high`` (every
parameter at its low / high bound). Needs the test SSP grid, $SPS_HOME for the nebular, dust and
AGN configurations and $CERIDWEN_AGN_DATA for the AGN tables; what was skipped is printed.

--compare prints, per configuration and output, the worst per-element relative error at the
floor, the worst |diff| / max|array|, and the theta that produced each; it exits 1 if any
output exceeds the standard. lnL and gradients are not judged here (they involve cancellation:
quote their shift against their own scale, and compare posteriors statistically).
"""
from __future__ import annotations

import argparse
import collections
import inspect
import os
import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "tests")]
warnings.simplefilter("ignore")

import jax                                                     # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

import bit_identity_check as b                                 # noqa: E402

RTOL = 1e-3          # the physical standard
FLOOR = 1e-4         # elements below FLOOR * max|array| are judged against max|array|
OUTPUTS = ("model_grid", "predict_spec", "predict_phot", "predict_lines", "line_fluxes")

LO = {"logmass": 8.0, "logzsol": -2.0, "logsfr_ratios": -2.0, "tau_pow": 2.0, "alpha_pow": -2.0,
      "diffuse_tau_kc": 2.0, "diffuse_dust_index": -1.0, "gas_logu": -4.0, "gas_logz": -2.0,
      "duste_qpah": 0.5, "duste_umin": 0.5, "duste_gamma": 0.0, "lsf_scale": 0.8,
      "sigma_gas": 30.0, "fagn": 1e-3, "agn_tau": 5.0, "fagn_bbb": 0.0, "dust4": 2.0,
      "dust4_index": -2.0, "agn_elum": 1e-6, "agn_eline_sigma": 50.0, "x_HI": 0.0,
      "logN_HI": 18.0}
HI = {"logmass": 12.0, "logzsol": 0.2, "logsfr_ratios": 2.0, "tau_pow": 0.0, "alpha_pow": 0.0,
      "diffuse_tau_kc": 0.0, "diffuse_dust_index": 0.4, "gas_logu": -1.0, "gas_logz": 0.5,
      "duste_qpah": 4.5, "duste_umin": 20.0, "duste_gamma": 0.5, "lsf_scale": 1.3,
      "sigma_gas": 400.0, "fagn": 2.0, "agn_tau": 150.0, "fagn_bbb": 2.0, "dust4": 0.0,
      "dust4_index": 0.0, "agn_elum": 1e-2, "agn_eline_sigma": 500.0, "x_HI": 1.0,
      "logN_HI": 22.0}
FAINT = dict(LO)
BRIGHT = dict(HI)
EXTREMES = (("faint", FAINT), ("bright", BRIGHT), ("low", LO), ("high", HI))


def configurations():
    """bit_identity_check.collect()'s configuration list (read from its source, so the two
    stay the same), plus all_components."""
    code = inspect.getsource(b.collect)
    start = code.rindex("\n", 0, code.index("configs = [")) + 1
    end = code.rindex("\n", 0, code.index("for tag, cfg in configs:")) + 1
    body = "\n".join(line[4:] for line in code[start:end].splitlines())
    g = dict(vars(b))
    g.update({"np": np, "os": os, "inspect": inspect})
    exec(body, g)
    configs = list(g["configs"])
    if os.environ.get("SPS_HOME"):
        configs.append(("all_components", dict(
            add_neb=True, add_igm=True, sigma_losvd=250.0, dust_emission=True, igm_dla=True,
            lsf="lsf_scale", agn={"agn_mode": "corrected", "dust4_type": 1, "lines": True})))
    return configs


def collect():
    import ceridwen
    print("package:", ceridwen.__file__)
    print("CERIDWEN_AGN_DATA:", os.environ.get("CERIDWEN_AGN_DATA"),
          sorted(os.listdir(os.environ["CERIDWEN_AGN_DATA"]))
          if os.environ.get("CERIDWEN_AGN_DATA") else "")
    out = {}
    rng = np.random.default_rng(b.SEED)
    configs = configurations()
    print("configurations:", [t for t, _ in configs], flush=True)
    for tag, cfg in configs:
        model = b._build(**cfg)
        draws = b._draws(model, 16, rng)
        names = list(model.param_names)
        csp = model.csp

        def outputs(th):
            th = {k: jnp.asarray(v) for k, v in th.items()}
            o = {f"predict_{k}": np.asarray(v, dtype=np.float64)
                 for k, v in model.predict(th).items()}
            t = model.apply_transforms(th)
            t["zred"] = jnp.array([model.zred])
            o["model_grid"] = np.asarray(csp.get_spectrum(t, include_lines=True),
                                         dtype=np.float64)
            if getattr(csp, "neb", None) is not None:
                o["line_fluxes"] = np.asarray(csp.predict_line_fluxes(t), dtype=np.float64)
            return o

        thetas = [(f"draw{i}", {n: draws[n][i] for n in names}) for i in range(16)]
        for lab, d in EXTREMES:
            thetas.append((lab, {n: (np.full(np.shape(model.theta_init[n]), d[n]) if n in d
                                     else np.asarray(model.theta_init[n], dtype=float))
                                 for n in names}))
        for lab, th in thetas:
            for k, v in outputs(th).items():
                out[f"{tag}/{lab}/{k}"] = v
        print(f"{tag:24s} {len(thetas)} theta", flush=True)
    return out


def compare(old, new):
    worst = collections.defaultdict(lambda: [0.0, "", 0.0, ""])
    missing = [k for k in old.files if k not in new]
    for k in old.files:
        if k not in new:
            continue
        tag, lab, kind = k.split("/")
        e = np.asarray(old[k], dtype=float).ravel()
        x = np.asarray(new[k], dtype=float).ravel()
        if e.shape != x.shape or not np.all(np.isfinite(x) == np.isfinite(e)):
            worst[(tag, kind)] = [np.inf, lab + " (shape/finiteness)", np.inf, lab]
            continue
        ok = np.isfinite(e)
        e, x = e[ok], x[ok]
        mx = float(np.max(np.abs(e))) if e.size else 0.0
        if mx == 0.0:
            continue
        d = np.abs(x - e)
        big = np.abs(e) > FLOOR * mx
        rel = max(float((d[big] / np.abs(e[big])).max()) if big.any() else 0.0,
                  float((d[~big] / mx).max()) if (~big).any() else 0.0)
        pk = float(d.max() / mx)
        w = worst[(tag, kind)]
        if rel > w[0]:
            w[0], w[1] = rel, lab
        if pk > w[2]:
            w[2], w[3] = pk, lab
    print(f"standard: {RTOL:g} relative, floor {FLOOR:g} x max|array|")
    print(f"{'configuration':24s} {'output':13s} {'relative (floor)':>24s} {'|diff| / max':>22s}")
    failed = []
    for (tag, kind), (rel, la, pk, lb) in sorted(worst.items()):
        flag = "  FAIL" if rel > RTOL else ""
        if flag:
            failed.append((tag, kind, rel, la))
        print(f"{tag:24s} {kind:13s} {rel:12.2e} ({la:9s}) {pk:12.2e} ({lb:7s}){flag}")
    for kind in OUTPUTS:
        vals = [(v[0], t, v[1]) for (t, k), v in worst.items() if k == kind]
        if vals:
            r, t, la = max(vals)
            print(f"worst {kind:13s} {r:9.2e}  {t} / {la}")
    if missing:
        print(f"MISSING in the new run: {len(missing)} arrays (e.g. {missing[0]})")
    print(f"compared {len(old.files) - len(missing)} arrays: {len(failed)} outputs above the "
          f"standard")
    return 1 if failed or missing else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="NPZ")
    g.add_argument("--compare", metavar="NPZ")
    a = ap.parse_args()
    out = collect()
    if a.save:
        np.savez(a.save, **out)
        print(f"saved {len(out)} arrays -> {a.save}")
        return 0
    return compare(np.load(a.compare), out)


if __name__ == "__main__":
    sys.exit(main())
