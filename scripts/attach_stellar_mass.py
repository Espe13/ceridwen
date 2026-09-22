#!/usr/bin/env python3
"""attach_stellar_mass.py — add the surviving stellar-mass table (SSP schema 3) to an
EXISTING grid file, without rebuilding its spectra.

The table ``ssp_stellar_mass`` holds, for every SSP of the grid (every metallicity and age
node, and every [alpha/Fe] plane of an alpha grid), FSPS's ``stellar_mass``: the mass in
stars + remnants per M_sun formed (FSPS ``sfh=0``, ``tage=0``).  ``PostProcess`` weights it
with each draw's SSP weights to give ``mfrac``, ``mass_surviving`` and ``ssfrW_surviving``.

Steps
-----
1. Load the grid through the strict loader (SSPData, or SSPDataAfe for a file with
   ``ssp_afe`` / a 4-D flux cube).
2. Build ``fsps.StellarPopulation(zcontinuous=0, sfh=0, <the grid's recorded IMF/library
   kwargs>)`` and refuse when its isochrones, age nodes or metallicity nodes differ from
   the grid's: a mass table from another isochrone set would be silently wrong.
3. Read ``stellar_mass`` for every metallicity (and alpha plane).  When the spectral
   library also matches, compare FSPS's spectra with the grid's flux cube as a check that
   this FSPS reproduces the grid (reported; ``--flux-rtol`` makes it fatal).
4. COPY the input to ``--out`` (default ``<name>_schema3.h5``), write the table and its
   provenance into the copy, and verify: strict reload, every original dataset
   bit-identical (sha256) to the input, the table round-trips exactly, chash unchanged.
   The input file is never opened for writing.

FSPS keeps global state (one StellarPopulation per process), so run one grid per call.
FSPS runs in a subprocess: this interpreter by default, or ``--fsps-python`` for an
environment whose python-fsps is compiled with the grid's isochrones (e.g. a BPASS build;
it needs only fsps + numpy).  ``$SPS_HOME`` must be set.

Examples
--------
python scripts/attach_stellar_mass.py ceridwen/data/test_data/ssp_data_mist_miles.h5
python scripts/attach_stellar_mass.py ssp_data_bpass.h5 --out ssp_data_bpass_schema3.h5 \
    --fsps-python ~/opt/anaconda3/envs/<bpass-fsps-env>/bin/python
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else (None if x is None else str(x))


# Runs where python-fsps lives (this interpreter, or --fsps-python in a subprocess): it
# needs only fsps + numpy, so an FSPS compiled with other isochrones (e.g. BPASS) can sit in
# another environment.  Writes the raw FSPS output; every check is made by the caller.
WORKER = r"""
import json, sys
import numpy as np
import fsps
kw, n_afe, out = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
sp = fsps.StellarPopulation(zcontinuous=0, sfh=0, **kw)
fsps_n_afe = int(getattr(sp, "n_afe", 1))
mass, flux = [], []
for ia in range(n_afe):
    if n_afe > 1:
        sp.params["afeindx"] = ia + 1
    m_rows, f_rows = [], []
    for iz in range(len(sp.zlegend)):
        print(f"  plane {ia + 1}/{n_afe}  metallicity {iz + 1}/{len(sp.zlegend)}", flush=True)
        _w, f = sp.get_spectrum(tage=0.0, zmet=iz + 1, peraa=False)
        m_rows.append(np.array(sp.stellar_mass, dtype=np.float64))
        f_rows.append(np.asarray(f, dtype=np.float64))
    mass.append(m_rows); flux.append(f_rows)
libs = [l.decode() if isinstance(l, bytes) else str(l) for l in sp.libraries]
np.savez(out, mass=np.array(mass), flux=np.array(flux), log_age=np.asarray(sp.log_age),
         zlegend=np.asarray(sp.zlegend), libraries=np.array(libs),
         version=np.array(str(getattr(fsps, "__version__", None))), n_afe=fsps_n_afe)
"""


def run_fsps(kw, n_afe, python=None):
    """Raw FSPS output (see WORKER) for ``kw``, ``n_afe`` planes, in this interpreter or ``python``."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:      # FSPS may leave fort.NN files in its cwd
        out = os.path.join(tmp, "fsps_out.npz")
        cmd = [python or sys.executable, "-c", WORKER, json.dumps(kw), str(n_afe), out]
        res = subprocess.run(cmd, cwd=tmp, env=os.environ.copy())
        if res.returncode or not os.path.exists(out):
            sys.exit(f"the FSPS worker failed ({' '.join(cmd[:1])}); see its output above")
        with np.load(out) as z:
            return {k: z[k] for k in z.files}


def fsps_mass_table(grid, is_afe, flux_rtol=None, python=None):
    """``(mass, report)``: FSPS stellar_mass on the grid's nodes, after checking that this
    FSPS build has the grid's isochrones, ages and metallicities."""
    from ceridwen.ssps.ssp_data import _validate_fsps_kwargs
    kw = dict(grid.fsps_kwargs or {})
    if grid.imf_type is not None:
        kw.setdefault("imf_type", int(grid.imf_type))
    kw = _validate_fsps_kwargs(kw)
    if grid.isoc_type is None:
        sys.exit("the grid records no isoc_type: cannot check that FSPS uses the same "
                 "isochrones, so no mass table is attached")
    n_afe = grid.n_afe if is_afe else 1
    raw = run_fsps(kw, n_afe, python)
    libs = [str(x) for x in raw["libraries"]]
    report = {"python_fsps": str(raw["version"]), "libraries": libs, "fsps_kwargs": kw}
    if libs[0] != grid.isoc_type:
        sys.exit(f"this python-fsps uses isochrones {libs[0]!r} but the grid was built with "
                 f"{grid.isoc_type!r}: its stellar masses would belong to another population. "
                 "Use an FSPS compiled with the grid's isochrones (--fsps-python).")
    if n_afe > 1 and int(raw["n_afe"]) != n_afe:
        sys.exit(f"the grid has {n_afe} [alpha/Fe] planes but this FSPS has {int(raw['n_afe'])}")
    ages = np.asarray(raw["log_age"], dtype=np.float64) - 9.0
    g_ages = np.asarray(grid.ssp_lg_age_gyr, dtype=np.float64)
    if ages.shape != g_ages.shape or np.max(np.abs(ages - g_ages)) > 1e-6:
        sys.exit(f"FSPS age nodes ({ages.size}) differ from the grid's ({g_ages.size})")
    lgz = np.log10(np.asarray(raw["zlegend"], dtype=np.float64))
    g_lgz = np.asarray(grid.ssp_lgmet, dtype=np.float64)
    if lgz.shape != g_lgz.shape or np.max(np.abs(lgz - g_lgz)) > 1e-6:
        sys.exit(f"FSPS metallicity nodes {np.round(lgz, 4)} differ from the grid's "
                 f"{np.round(g_lgz, 4)}")
    mass = raw["mass"] if is_afe else raw["mass"][0]
    if libs[1] == grid.spec_library:
        g = np.asarray(grid.ssp_flux, dtype=np.float64)
        f = raw["flux"] if is_afe else raw["flux"][0]
        if f.shape != g.shape:
            sys.exit(f"FSPS flux cube {f.shape} does not match the grid's {g.shape}")
        sig = g > 1e-30 * np.max(g, axis=-1, keepdims=True)
        worst = float(np.max(np.abs(f[sig] / g[sig] - 1.0)))
        report["flux_check"] = f"max |F_fsps / F_grid - 1| = {worst:.3e} over all SSPs"
        if flux_rtol is not None and worst > flux_rtol:
            sys.exit(f"FSPS spectra differ from the grid's by {worst:.3e} > --flux-rtol "
                     f"{flux_rtol:g}: this FSPS does not reproduce the grid")
    else:
        report["flux_check"] = (f"skipped (spectral library {libs[1]!r} != grid's "
                                f"{grid.spec_library!r}; the masses depend on isochrones + "
                                "IMF only)")
    return mass, report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grid", type=Path, help="existing SSPData / SSPDataAfe .h5 file")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default <name>_schema3.h5 next to the input)")
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    p.add_argument("--fsps-python", default=None,
                   help="python executable whose python-fsps has the grid's isochrones (run "
                        "as a subprocess; needs only fsps + numpy). Default: this interpreter")
    p.add_argument("--flux-rtol", type=float, default=None,
                   help="abort when FSPS's spectra differ from the grid's by more than this "
                        "(only when the spectral library matches); default: report only")
    args = p.parse_args()

    from ceridwen.ssps.ssp_data import SSPData, SSP_SCHEMA_VERSION, fsps_stellar_mass_source
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe, SSP_AFE_SCHEMA_VERSION

    src = args.grid
    if not src.is_file():
        sys.exit(f"{src}: not found")
    out = args.out if args.out is not None else src.with_name(src.stem + "_schema3.h5")
    if out.resolve() == src.resolve():
        sys.exit("--out is the input file: this script never writes the original grid")
    if out.exists() and not args.force:
        sys.exit(f"{out} exists — use --force to overwrite")
    if not os.environ.get("SPS_HOME"):
        sys.exit("$SPS_HOME is not set: FSPS is needed to compute the masses")

    with h5py.File(src, "r") as f:
        if "ssp_stellar_mass" in f:
            sys.exit(f"{src} already carries ssp_stellar_mass — nothing to do")
        is_afe = "ssp_afe" in f or f["ssp_flux"].ndim == 4
        names = [k for k in f if isinstance(f[k], h5py.Dataset)]
        in_sha = {k: sha(f[k][()]) for k in names}
    cls = SSPDataAfe if is_afe else SSPData
    grid = cls.load(str(src))
    print(f"{src.name}: {cls.__name__}  isoc={grid.isoc_type} spec={grid.spec_library} "
          f"imf_type={grid.imf_type}  flux {tuple(np.shape(grid.ssp_flux))}")

    mass, report = fsps_mass_table(grid, is_afe, args.flux_rtol, args.fsps_python)
    grid.with_stellar_mass(mass, source="validation")     # shape / finiteness / positivity
    source = fsps_stellar_mass_source(report["python_fsps"]) + (
        f"; libraries {report['libraries']}; attached by scripts/attach_stellar_mass.py")
    print(f"  python-fsps    : {report['python_fsps']}  libraries {report['libraries']}")
    print(f"  flux check     : {report['flux_check']}")
    print(f"  stellar mass   : shape {mass.shape}, [{mass.min():.6g}, {mass.max():.6g}] "
          "M_sun per M_sun formed")

    tmp = out.with_name(out.name + ".partial")
    shutil.copy2(src, tmp)
    with h5py.File(tmp, "r+") as f:
        f.create_dataset("ssp_stellar_mass", data=np.asarray(mass, dtype=np.float64))
        f.attrs["units_stellar_mass"] = ("M_sun surviving (stars + remnants) per M_sun "
                                         "formed, per SSP")
        f.attrs["stellar_mass_source"] = source
        f.attrs["stellar_mass_fsps_report_json"] = json.dumps(
            {k: v for k, v in report.items() if k != "fsps_kwargs"} | {
                "fsps_kwargs": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                for k, v in report["fsps_kwargs"].items()}})
        prev = _dec(f.attrs.get("schema_version"))
        f.attrs["schema_version_before_stellar_mass"] = str(prev)
        f.attrs["schema_version"] = SSP_AFE_SCHEMA_VERSION if is_afe else SSP_SCHEMA_VERSION

    # ---- verify ----------------------------------------------------------------------
    ok = True
    with h5py.File(tmp, "r") as f:
        for k in names:
            same = sha(f[k][()]) == in_sha[k]
            ok &= same
            print(f"  {k:17s} {'bit-identical to the input' if same else 'MISMATCH!'}")
    back = cls.load(str(tmp))
    rt = back.ssp_stellar_mass is not None and np.array_equal(back.ssp_stellar_mass, mass)
    ok &= rt
    print(f"  {'ssp_stellar_mass':17s} {'round-trips exactly' if rt else 'MISMATCH!'}")
    same_id = back.chash == grid.chash
    ok &= same_id
    print(f"  chash            {'unchanged' if same_id else 'CHANGED!'} ({back.chash})")
    print(f"  schema_version   {back.schema_version}")
    if not ok:
        tmp.unlink()
        sys.exit("VERIFICATION FAILED — nothing written")
    tmp.replace(out)
    print(f"-> {out}  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
