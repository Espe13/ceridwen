#!/usr/bin/env python3
"""Build the alpha-enhanced SSP grid (Phase A of AFE_MOCK_TEST_DESIGN.md).

Run this wherever python-fsps is compiled against FSPS v4.0:
  * with AFE_FLAG=1  -> the 4-D aMIST/C3K alpha grid   (--expect-nafe 5)
  * with AFE_FLAG=0  -> the C3K null grid, n_afe = 1    (--expect-nafe 1)

The grid build is CPU-only but memory-heavy with AFE_FLAG=1 (FSPS holds
the full (nspec, ntfull, nz, nafe) SSP set resident) — a login node with
>= 32 GB or a CPU batch node is fine.  Fits then run on the GPU nodes
against the saved HDF5; FSPS is NOT needed at fit time.

Usage:
    python build_afe_grid.py --out ssp_data_afe_c3k.h5 --expect-nafe 5
    python build_afe_grid.py --out ssp_data_c3k_null.h5 --expect-nafe 1
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output HDF5 path")
    ap.add_argument("--expect-nafe", type=int, default=5,
                    help="assert the compiled FSPS grid has this n_afe "
                         "(5 = AFE_FLAG=1 aMIST/C3K, 1 = null build)")
    ap.add_argument("--imf-type", type=int, default=1,
                    help="FSPS imf_type (default 1 = Chabrier)")
    args = ap.parse_args()

    import fsps
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe

    # ------------------------------------------------------------ smoke
    probe = fsps.StellarPopulation(zcontinuous=0, sfh=0)
    libs = tuple(x.decode() if isinstance(x, bytes) else str(x)
                 for x in probe.libraries)
    n_afe = int(getattr(probe, "n_afe", 1))
    print(f"python-fsps {fsps.__version__}")
    print(f"libraries   : {libs}")
    print(f"n_afe       : {n_afe}")
    print(f"nz          : {probe.zlegend.size}  "
          f"(log10 Z in [{np.log10(probe.zlegend).min():+.3f}, "
          f"{np.log10(probe.zlegend).max():+.3f}])")

    if n_afe != args.expect_nafe:
        sys.exit(
            f"FATAL: compiled grid has n_afe={n_afe}, expected "
            f"{args.expect_nafe}. Wrong AFE_FLAG at compile time, or an "
            f"old python-fsps (< 2026-08-02 alpha-MC merge)."
        )
    if "mist" not in libs[0].lower():
        sys.exit(f"FATAL: isochrones are {libs[0]!r}; alpha grids need MIST.")
    if n_afe > 1 and "c3k" not in libs[1].lower():
        sys.exit(f"FATAL: spectral library is {libs[1]!r}; alpha needs C3K.")
    del probe

    # ------------------------------------------------------------ build
    data = SSPDataAfe.from_fsps(save_to=args.out, imf_type=args.imf_type)
    data.display()

    # ------------------------------------------------------ verification
    # Reload and cross-check one plane against a fresh FSPS pull: catches
    # HDF5 corruption and the 1-based afeindx off-by-one in one shot.
    back = SSPDataAfe.load(args.out)
    assert back.ssp_flux.shape == data.ssp_flux.shape, "reload shape mismatch"

    check = fsps.StellarPopulation(zcontinuous=0, sfh=0,
                                   imf_type=args.imf_type)
    solar_plane = int(np.argmin(np.abs(np.asarray(back.ssp_afe))))
    if back.n_afe > 1:
        check.params["afeindx"] = solar_plane + 1          # FSPS 1-based
    zmid = back.ssp_lgmet.size // 2
    _w, fl = check.get_spectrum(tage=0.0, zmet=zmid + 1, peraa=False)
    stored = np.asarray(back.ssp_flux[solar_plane, zmid])
    rel = np.max(np.abs(stored - fl) / np.maximum(np.abs(fl), 1e-30))
    print(f"solar-plane spot check (afeindx={solar_plane + 1}, "
          f"zmet={zmid + 1}): max rel diff = {rel:.3e}")
    if rel > 1e-6:
        sys.exit("FATAL: stored plane does not match FSPS — indexing bug.")

    print(f"\nOK: wrote {args.out}")


if __name__ == "__main__":
    main()
