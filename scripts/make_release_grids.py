"""Build the release SSP grids for Zenodo (run TWICE, once per FSPS build).

The isochrone/spectral library is fixed at python-fsps COMPILE time, so this
script detects what is compiled and writes the matching grid:

    BPASS build -> examples/ssp_data_bpass.h5   (no IMF kwargs: BPASS SSP
                                                 parameters are fixed)
    MIST  build -> examples/ssp_data.h5         (imf_type=1, Chabrier; this
                                                 is the quickstart/Zenodo
                                                 default grid)

Procedure (see RELEASE_TODO.md section B0):

    # 1. with the current (BPASS) install:
    python scripts/make_release_grids.py

    # 2. reinstall as MIST+MILES (the default libraries):
    pip uninstall fsps
    python -m pip install fsps --no-binary fsps --no-cache-dir

    # 3. run again:
    python scripts/make_release_grids.py

Refuses to overwrite an existing output unless --force is given, and verifies
the written file by re-loading it and printing its provenance.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = {
    "bpss": REPO / "examples" / "ssp_data_bpass.h5",   # FSPS reports 'bpss'
    "bpass": REPO / "examples" / "ssp_data_bpass.h5",
    "mist": REPO / "examples" / "ssp_data.h5",
}
IMF_KWARGS = {  # per compiled isochrone library
    "bpss": {}, "bpass": {},          # fixed at compile time — pass nothing
    "mist": {"imf_type": 1},          # Chabrier (2003)
}


def compiled_libraries() -> tuple[str, str]:
    """Return (isochrone, spectral) library names of the compiled python-fsps."""
    import fsps
    sp = fsps.StellarPopulation(zcontinuous=0)   # ~30 s: loads the SSPs once
    libs = sp.libraries
    dec = lambda b: b.decode() if isinstance(b, bytes) else str(b)
    return dec(libs[0]).strip().lower(), dec(libs[1]).strip().lower()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    args = p.parse_args()

    import dataclasses

    import ceridwen
    from ceridwen.ssps.ssp_data import SSPData

    # Guard against a stale non-editable install: the imported ceridwen may
    # be an old site-packages snapshot whose SSPData writes NO provenance.
    print(f"ceridwen {ceridwen.__version__} from {ceridwen.__file__}")
    if "isoc_type" not in {f.name for f in dataclasses.fields(SSPData)}:
        sys.exit("the imported ceridwen has a pre-provenance SSPData — "
                 "refresh the install with `pip install .` in the repo root, "
                 "then re-run.")

    isoc, spec = compiled_libraries()
    print(f"compiled python-fsps libraries: isochrones={isoc!r}, spectra={spec!r}")

    if isoc not in OUT:
        sys.exit(f"unexpected isochrone library {isoc!r} — expected MIST or "
                 "BPASS. Reinstall python-fsps with the intended FFLAGS "
                 "(see RELEASE_TODO.md B0).")

    target = OUT[isoc]
    kwargs = IMF_KWARGS[isoc]
    if target.exists() and not args.force:
        sys.exit(f"{target} already exists — re-run with --force to overwrite.")

    # Library resolution curves (schema 2.0): from_fsps automatically
    # applies the grid's own 2-pixel sampling floor (derived from the
    # built ssp_wave); segments add a documented library LSF on top
    # (element-wise max), so the stored curve is finite everywhere.
    #   MIST/MILES: LSF FWHM 2.54 A over 3525-7500 A (Falcon-Barroso et
    #               al. 2011); outside that range the FSPS master grid's
    #               own coarse sampling is the binding resolution and the
    #               floor covers it.
    #   BPASS: no documented LSF broader than the ~1 A tabulation — the
    #          stored grid's sampling floor alone is the honest curve.
    RESOLUTION = {
        "mist": ([(3525.0, 7500.0, "fwhm_AA", 2.54)],
                 "MILES FWHM 2.54A (Falcon-Barroso et al. 2011)"),
        "bpss": (None, None),
    }
    RESOLUTION["bpass"] = RESOLUTION["bpss"]
    segments, source = RESOLUTION[isoc]

    print(f"building {isoc.upper()} grid -> {target}  (kwargs={kwargs})")
    SSPData.from_fsps(save_to=str(target), resolution_segments=segments,
                      resolution_source=source, **kwargs)

    # ---- verify: re-load and print provenance --------------------------
    data = SSPData.load(str(target))
    print("\nwritten and re-loaded OK:")
    print(f"  file            : {target} "
          f"({target.stat().st_size/1e6:.1f} MB)")
    print(f"  isoc_type       : {data.isoc_type}")
    print(f"  spec_library    : {data.spec_library}")
    print(f"  fsps_kwargs     : {data.fsps_kwargs}")
    print(f"  schema_version  : {data.schema_version}")
    print(f"  ssp_lgmet       : {len(data.ssp_lgmet)} points "
          f"[{float(data.ssp_lgmet.min()):.2f}, {float(data.ssp_lgmet.max()):.2f}]")
    print(f"  ssp_lg_age_gyr  : {len(data.ssp_lg_age_gyr)} points")
    print(f"  ssp_wave        : {len(data.ssp_wave)} points "
          f"[{float(data.ssp_wave.min()):.0f}, {float(data.ssp_wave.max()):.0f}] AA")
    import numpy as _np
    _res = _np.asarray(data.ssp_resolution)
    print(f"  ssp_resolution  : sigma_v [{_res.min():.1f}, {_res.max():.1f}] "
          f"km/s (finite everywhere: {bool(_np.all(_np.isfinite(_res)))})")
    print(f"  res. source     : {data.resolution_source}")

    if data.isoc_type is None:
        sys.exit("provenance missing after save — investigate before uploading.")
    print("\nready for Zenodo." if isoc == "mist" else
          "\nnow reinstall python-fsps as MIST (RELEASE_TODO.md B0 step 2) "
          "and run this script again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
