#!/usr/bin/env python
"""Reference arrays of CIGALE's ``skirtor2016`` module for the AGN2 parity tests (L1, L3).

Runs ONLY in the reference environment (pcigale with a built ``skirtor2016`` database):

    PYTHONPATH=~/Desktop/PhD/Tools/ceridwen_workspace/agn2_ref/src/cigale \
    ~/Desktop/PhD/Tools/ceridwen_workspace/agn2_ref/envs/cigale/bin/python \
        scripts/agn2_reference/cigale_skirtor_reference.py --out tests/reference/agn2/

(pcigale runs from its source tree, database built in place: agn2_ref/envs/README.md.)

For every parameter set it calls the module exactly as pcigale does
(``get_module("skirtor2016", **params)`` then ``module.process(sed)`` on an ``SED`` whose
``dust.luminosity`` is ``L_DUST``) and stores, on the module's own wavelength grid (nm, the
SKIRTOR grid of the database), the three contributions it adds (W/nm):
``agn.SKIRTOR2016_torus``, ``agn.SKIRTOR2016_polar_dust``, ``agn.SKIRTOR2016_disk``, and every
``agn.*`` entry of ``sed.info``. Nothing of CERIDWEN is imported.

Parameter sets (all at SKIRTOR grid nodes; CIGALE does not interpolate):
  ``node``   : 40 sets spread over every axis value of t, pl, q, oa, R, i (polar dust off,
               EBV = 0), disk_type 1 (CIGALE's default) and 0;
  ``cell``   : the 64 corners of one interior cell in (t, pl, q, oa, i) at R = 20, for the
               off-node tests (the test interpolates these with its own NumPy code);
  ``node``   : + PLAN_AGN2 5 L1: every oa x i in {0, 90-oa-10, 90-oa, 90}, t {3, 11},
               pl/q {0, 1.5}, R {10, 30};
  ``polar``  : PLAN_AGN2 5 L3: law {0,1,2} x EBV {0.03, 0.3, 1, 3} x T {50, 100, 300} K x
               i {30, 70} x oa {20, 40, 60}, emissivity 1.6 (+ 1.0, 2.2 at one setting).
Output: ``skirtor_cigale.npz`` (+ ``skirtor_cigale.json`` provenance: pcigale commit,
database file hashes, parameter lists, this script's sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import subprocess
import sys

import numpy as np

L_DUST = 1.0
FRAC_AGN = 0.5                 # agn_power = L_DUST * (1 / (1 - f) - 1) = 1
T = [3, 5, 7, 9, 11]
PL = [0.0, 0.5, 1.0, 1.5]
Q = [0.0, 0.5, 1.0, 1.5]
OA = [10, 20, 30, 40, 50, 60, 70, 80]
R = [10, 20, 30]
INC = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
BASE = dict(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=30, disk_type=1, delta=0.0,
            fracAGN=FRAC_AGN, lambda_fracAGN="0/0", law=0, EBV=0.0, temperature=100.0,
            emissivity=1.6)


def param_sets():
    sets = []
    rng = np.random.default_rng(20261003)
    for k in range(40):              # every axis value appears; the rest random nodes
        p = dict(BASE, t=T[k % 5], pl=PL[k % 4], q=Q[(k // 4) % 4], oa=OA[k % 8],
                 R=R[k % 3], i=INC[k % 10], disk_type=int(rng.integers(0, 2)))
        sets.append(("node", p))
    # PLAN_AGN2 5 L1: every oa x i in {0, 90-oa-10, 90-oa, 90} (both sides of the type-1/2
    # boundary), t in {3, 11}, pl/q in {0, 1.5}, R in {10, 30}
    for n, oa in enumerate(OA):
        for k, i in enumerate(sorted({0, max(90 - oa - 10, 0), 90 - oa, 90})):
            sets.append(("node", dict(BASE, oa=oa, i=i, t=(3, 11)[(n + k) % 2],
                                      pl=(0.0, 1.5)[k % 2], q=(1.5, 0.0)[(n + k) % 2],
                                      R=(10, 30)[n % 2])))
    for t, pl, q, oa, i in itertools.product([5, 7], [0.5, 1.0], [0.5, 1.0], [30, 40],
                                             [20, 30]):
        sets.append(("cell", dict(BASE, t=t, pl=pl, q=q, oa=oa, i=i)))
    # PLAN_AGN2 5 L3: law x EBV x T x i x oa, emissivity 1.6 (+ two emissivity variants)
    for law, ebv, temp, i, oa in itertools.product([0, 1, 2], [0.03, 0.3, 1.0, 3.0],
                                                   [50.0, 100.0, 300.0], [30, 70],
                                                   [20, 40, 60]):
        sets.append(("polar", dict(BASE, EBV=ebv, law=law, temperature=temp, i=i, oa=oa)))
    for emis in (1.0, 2.2):
        sets.append(("polar", dict(BASE, EBV=0.3, emissivity=emis)))
    return sets


def run_one(params):
    from pcigale.sed import SED
    from pcigale.sed_modules import get_module
    mod = get_module("skirtor2016", **params)
    sed = SED()
    sed.add_info("dust.luminosity", L_DUST, True, unit="W")
    mod.process(sed)
    wl = np.asarray(sed.wavelength_grid, dtype=np.float64)
    comps = {k: np.asarray(sed.luminosities[k], dtype=np.float64)
             for k in ("agn.SKIRTOR2016_torus", "agn.SKIRTOR2016_polar_dust",
                       "agn.SKIRTOR2016_disk")}
    info = {k: float(v) for k, v in sed.info.items() if k.startswith("agn.")}
    # PLAN_AGN2 5: the module-level quantities of _init_code
    for k in ("lumin_disk", "lumin_polar_dust", "lumin_intrin_disk", "l_agn_2500A"):
        info[f"module.{k}"] = float(getattr(mod, k))
    return wl, comps, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/reference/agn2")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import pcigale
    src = pathlib.Path(pcigale.__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        commit = "unknown"
    sets = param_sets()
    wl0 = None
    arrays = {"torus": [], "polar": [], "disk": []}
    infos, kinds, plist = [], [], []
    for kind, p in sets:
        wl, comps, info = run_one(p)
        if wl0 is None:
            wl0 = wl
        elif wl.shape != wl0.shape or np.any(wl != wl0):
            raise RuntimeError("the module's wavelength grid changed between parameter sets")
        arrays["torus"].append(comps["agn.SKIRTOR2016_torus"])
        arrays["polar"].append(comps["agn.SKIRTOR2016_polar_dust"])
        arrays["disk"].append(comps["agn.SKIRTOR2016_disk"])
        infos.append(info)
        kinds.append(kind)
        plist.append(p)
    keys = sorted(infos[0])
    pkeys = ["t", "pl", "q", "oa", "R", "Mcl", "i", "disk_type", "delta", "fracAGN", "law",
             "EBV", "temperature", "emissivity"]
    np.savez_compressed(
        out / "skirtor_cigale.npz", wl_nm=wl0,
        torus=np.array(arrays["torus"]), polar=np.array(arrays["polar"]),
        disk=np.array(arrays["disk"]),
        info=np.array([[d[k] for k in keys] for d in infos]), info_keys=np.array(keys),
        params=np.array([[float(p[k]) for k in pkeys] for p in plist]),
        param_keys=np.array(pkeys), kind=np.array(kinds), L_dust=L_DUST)
    db = src / "pcigale" / "data" / "skirtor2016"
    hashes = {str(f.relative_to(src)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(db.glob("*")) if f.is_file()} if db.exists() else {}
    prov = {"script": "scripts/agn2_reference/cigale_skirtor_reference.py",
            "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "pcigale_src": str(src), "pcigale_commit": commit, "python": sys.version,
            "database_files": len(hashes),
            "database_sha256_of_sorted_file_hashes": hashlib.sha256(
                "\n".join(f"{k} {v}" for k, v in sorted(hashes.items())).encode()).hexdigest(),
            "L_dust": L_DUST, "fracAGN": FRAC_AGN,
            "n_sets": len(sets), "units": "wl nm; contributions W/nm per L_dust = 1 W"}
    (out / "skirtor_cigale.json").write_text(json.dumps(prov, indent=1))
    print(f"wrote {len(sets)} sets on {wl0.size} wavelengths -> {out}")


if __name__ == "__main__":
    main()
