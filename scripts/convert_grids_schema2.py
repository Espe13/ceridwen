#!/usr/bin/env python3
"""convert_grids_schema2.py — upgrade SSP grids to schema 2.x WITHOUT FSPS.

Reads an old SSPData / SSPDataAfe HDF5 file directly (raw h5py — the
strict schema-2.x loaders would reject it), copies every dataset and
provenance attribute bit-for-bit, attaches the library resolution curve,
and writes the new file through the strict ``save()``.  No FSPS rebuild,
no reinstalls.

The resolution curve (sigma_v(lambda) [km/s]) is built automatically as
the element-wise MAXIMUM of

  1. the grid's own 2-pixel SAMPLING FLOOR, derived from the file's own
     ``ssp_wave`` (sigma_v ~= 0.8493 c dlambda/lambda) — always applied,
     needs no external numbers; and
  2. optionally, a documented library LSF: a shipped preset
     (``--library miles``) or explicit segments
     (``--segments '[[3525, 7500, "fwhm_AA", 2.54]]'``), cited via
     ``--source``.

Segment kinds: "R_fwhm" (resolving power, FWHM), "fwhm_AA" (wavelength
FWHM in Angstrom), "sigma_v_kms" (Gaussian sigma directly).  The stored
curve is finite at every pixel (the floor covers pixels outside all
segments).

Alpha-enhanced grids: files carrying an ``ssp_afe`` dataset or a 4-D
``ssp_flux`` are dispatched to :class:`SSPDataAfe` automatically
(``--afe`` forces this for a 3-D solar-scaled file, which is promoted to
n_afe = 1 at [alpha/Fe] = 0).

Examples
--------
# MIST+MILES Zenodo default grid (floor + MILES LSF):
python scripts/convert_grids_schema2.py examples/ssp_data.h5 \
    --library miles --source "MILES FWHM 2.54A (Falcon-Barroso+2011)"

# BPASS grid: the 1A tabulation IS the resolution — floor only:
python scripts/convert_grids_schema2.py examples/ssp_data_bpass.h5

# aMIST/C3K alpha grid (4-D, auto-dispatched; C3K LSF not separately
# documented for this build -> floor only):
python scripts/convert_grids_schema2.py \
    ceridwen/data/test_data/amist_c3k_lr_chab_afe.h5

By default writes <name>_schema2.h5 next to the input; --in-place
replaces the input after keeping <name>_schema1_backup.h5.

Verification (always run): the output is re-loaded through the strict
loader and every original dataset is compared BIT-FOR-BIT (sha256)
against the input; any mismatch aborts with a non-zero exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def sha(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grid", type=Path, help="old SSPData/SSPDataAfe .h5 file")
    p.add_argument("--library", default=None,
                   help="shipped LSF preset name (see "
                        "ceridwen.ssps.library_resolution.PRESETS); "
                        "max-combined with the sampling floor")
    p.add_argument("--segments", default=None,
                   help='JSON list of [lam_lo, lam_hi, kind, value] LSF '
                        'segments; max-combined with the sampling floor')
    p.add_argument("--source", default=None,
                   help="citation for --library/--segments numbers "
                        "(REQUIRED with either; forbidden without — the "
                        "floor's provenance is recorded automatically)")
    p.add_argument("--afe", action="store_true",
                   help="force SSPDataAfe handling for a 3-D solar-scaled "
                        "file (promoted to n_afe=1 at [alpha/Fe]=0); 4-D "
                        "or ssp_afe-carrying files are dispatched "
                        "automatically")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default <name>_schema2.h5)")
    p.add_argument("--in-place", action="store_true",
                   help="replace the input (schema-1 backup kept alongside)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    args = p.parse_args()

    if args.library is not None and args.segments is not None:
        sys.exit("specify at most one of --library or --segments")
    has_lsf = args.library is not None or args.segments is not None
    if has_lsf and args.source is None:
        sys.exit("--source is required with --library/--segments "
                 "(cite where the LSF numbers come from)")
    if args.source is not None and not has_lsf:
        sys.exit("--source given without --library/--segments; the "
                 "sampling-floor provenance is recorded automatically")

    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe
    from ceridwen.ssps.library_resolution import (
        PRESETS, combined_sigma_v, combined_source,
        sampling_floor_sigma_v)

    src = args.grid
    if not src.is_file():
        sys.exit(f"{src}: not found")
    out = (args.out if args.out is not None
           else src.with_name(src.stem + "_schema2.h5"))
    if args.in_place:
        out = src
    elif out.exists() and not args.force:
        sys.exit(f"{out} exists — use --force to overwrite")

    # ---- read the old file raw (strict loaders would reject it) ---------
    with h5py.File(src, "r") as f:
        if "ssp_resolution" in f:
            sys.exit(f"{src} already carries ssp_resolution — nothing to do")
        arrays = {k: np.array(f[k][:])
                  for k in ("ssp_lgmet", "ssp_lg_age_gyr",
                            "ssp_wave", "ssp_flux")}
        ssp_afe = np.array(f["ssp_afe"][:]) if "ssp_afe" in f else None
        attrs = {k: f.attrs[k] for k in f.attrs}

    is_afe = args.afe or ssp_afe is not None or arrays["ssp_flux"].ndim == 4
    if is_afe and ssp_afe is None:
        if arrays["ssp_flux"].ndim == 4:
            sys.exit(f"{src}: 4-D ssp_flux but no ssp_afe dataset — the "
                     f"[alpha/Fe] values of the planes are unknown; this "
                     f"file cannot be converted safely")
        # 3-D solar-scaled file forced onto the afe path: single plane.
        ssp_afe = np.zeros(1)
        arrays["ssp_flux"] = arrays["ssp_flux"][None, ...]
        print("  (--afe: promoted 3-D flux to n_afe=1 at [alpha/Fe]=0)")
    if not is_afe and arrays["ssp_flux"].ndim != 3:
        sys.exit(f"{src}: expected 3-D ssp_flux for SSPData, got "
                 f"{arrays['ssp_flux'].ndim}-D")

    # NB: promotion via [None, ...] does not change the underlying bytes,
    # so the sha comparison below still checks bit-identity vs the input.
    in_sha = {k: sha(v) for k, v in arrays.items()}
    if ssp_afe is not None:
        in_sha["ssp_afe"] = sha(ssp_afe)

    def _dec(v):
        return v.decode() if isinstance(v, (bytes, bytearray)) else v

    cls = SSPDataAfe if is_afe else SSPData
    from ceridwen.ssps.ssp_data import SSP_SCHEMA_VERSION
    from ceridwen.ssps.ssp_data_afe import SSP_AFE_SCHEMA_VERSION
    schema = SSP_AFE_SCHEMA_VERSION if is_afe else SSP_SCHEMA_VERSION

    meta = {
        "isoc_type":    _dec(attrs.get("isoc_type")),
        "spec_library": _dec(attrs.get("spec_library")),
        "imf_type":     (int(attrs["imf_type"])
                         if "imf_type" in attrs else None),
        "fsps_version": _dec(attrs.get("fsps_version")),
        "fsps_kwargs":  (json.loads(_dec(attrs["fsps_kwargs_json"]))
                         if "fsps_kwargs_json" in attrs else {}),
        "wave_min":     (float(attrs["wave_min"])
                         if "wave_min" in attrs else None),
        "wave_max":     (float(attrs["wave_max"])
                         if "wave_max" in attrs else None),
        "schema_version": schema,
    }
    meta = {k: v for k, v in meta.items() if v is not None or k == "fsps_kwargs"}

    # ---- resolution curve: floor (from the file's own wave) [+ LSF] -----
    wave = arrays["ssp_wave"]
    if args.library:
        lib = str(args.library).lower()
        if lib not in PRESETS:
            sys.exit(f"no shipped LSF preset {lib!r} (have: "
                     f"{sorted(PRESETS)}); use --segments")
        segments = list(PRESETS[lib]())
    elif args.segments:
        segments = [tuple(s) for s in json.loads(args.segments)]
    else:
        segments = None
    sigma_v = combined_sigma_v(wave, segments=segments)
    source = combined_source(args.source if segments else None)
    if not np.all(np.isfinite(sigma_v)):
        sys.exit("internal error: combined curve is not finite everywhere")

    if is_afe:
        data = SSPDataAfe(ssp_lgmet=arrays["ssp_lgmet"], ssp_afe=ssp_afe,
                          ssp_lg_age_gyr=arrays["ssp_lg_age_gyr"],
                          ssp_wave=arrays["ssp_wave"],
                          ssp_flux=arrays["ssp_flux"],
                          ssp_resolution=sigma_v,
                          resolution_source=source, **meta)
    else:
        data = SSPData(**arrays, ssp_resolution=sigma_v,
                       resolution_source=source, **meta)

    floor = sampling_floor_sigma_v(wave)
    print(f"{src.name}: {'SSPDataAfe' if is_afe else 'SSPData'}  "
          f"spec_library={meta.get('spec_library')} "
          f"isoc={meta.get('isoc_type')}"
          + (f"  n_afe={ssp_afe.size}" if is_afe else ""))
    print(f"  sampling floor : sigma_v [{floor.min():.1f}, "
          f"{floor.max():.1f}] km/s")
    print(f"  stored curve   : sigma_v [{sigma_v.min():.1f}, "
          f"{sigma_v.max():.1f}] km/s over "
          f"[{wave.min():.0f}, {wave.max():.0f}] AA (finite everywhere)"
          + ("" if segments is None else "  [floor max LSF segments]"))

    if args.in_place:
        backup = src.with_name(src.stem + "_schema1_backup.h5")
        if backup.exists() and not args.force:
            sys.exit(f"{backup} exists — use --force")
        shutil.copy2(src, backup)
        print(f"  schema-1 backup: {backup.name}")

    data.save(str(out))

    # ---- verify: strict reload + bit-identical arrays -------------------
    back = cls.load(str(out))
    ok = True
    for k in in_sha:
        h = sha(getattr(back, k))
        same = (h == in_sha[k])
        ok &= same
        print(f"  {k:15s} {'bit-identical' if same else 'MISMATCH!'}")
    res_ok = np.array_equal(np.asarray(back.ssp_resolution), sigma_v)
    ok &= res_ok
    print(f"  {'ssp_resolution':15s} "
          f"{'round-trips exactly' if res_ok else 'MISMATCH!'}")
    print(f"  schema_version   {back.schema_version}")
    print(f"  source           {back.resolution_source}")
    if not ok:
        sys.exit("VERIFICATION FAILED — output NOT trustworthy")
    print(f"-> {out}  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
