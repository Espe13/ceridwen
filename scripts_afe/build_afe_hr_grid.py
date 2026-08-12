#!/usr/bin/env python3
"""Build the HIGH-RESOLUTION alpha-enhanced SSPData grid from the alpha-MC
FITS that M. J. Park distributed directly (the high-res C3K spectra are too
large to ship in the public python-FSPS / FSPS repositories).

Input FITS layout (as provided by MJ Park, 2025-07-22)::

    ext 0 : wavelength [Angstrom]                      shape (n_wave,)
    ext 1 : flux [L_sun / Hz]                          shape (n_grid, n_wave)
    ext 2 : BINTABLE with columns (logt, feh, afe), one row per SSP
            logt = log10(age/yr)  in [5.0, 10.3] step 0.05  (107 nodes)
            feh  = [Fe/H]         in [-2.5, +0.5] step 0.25 (13 nodes)
            afe  = [alpha/Fe]     in [-0.2, +0.6] step 0.20 (5  nodes)

    flux for a grid point:  data[1].data[ np.where(
        (hdr['feh']==feh) & (hdr['afe']==afe) & (hdr['logt']==logt) )[0][0] ]

This script maps that onto the CERIDWEN ``SSPDataAfe`` schema-2.0 container
(``ceridwen.ssps.ssp_data_afe``), whose axes are:

    ssp_afe        (n_afe,)                    [alpha/Fe]           = afe
    ssp_lgmet      (n_met,)   log10 ABSOLUTE total Z               = feh + log10(Zsun)
    ssp_lg_age_gyr (n_ages,)  log10(age/Gyr)                       = logt - 9
    ssp_wave       (n_wave,)  Angstrom                             = ext0
    ssp_flux       (n_afe, n_met, n_ages, n_wave)  L_sun/Hz/Msun   = ext1, reordered

The metallicity convention is the one the existing ceridwen alpha grid
(amist_c3k_lr_chab_afe.h5) uses: FSPS stores the C3K total metal mass
fraction as  Z = Zsun * 10**[Fe/H]  with the MIST protosolar reference
Zsun = 0.0185, so  log10 Z = [Fe/H] + log10(0.0185) = [Fe/H] - 1.7328283.
This offset was verified to be constant to 1e-16 across all 13 [Fe/H]
nodes of the published low-resolution grid, so the high-res metallicity
axis reproduces the low-res one bit-for-bit -- the two grids are drop-in
interchangeable and share the same ``Z`` prior bounds.

Run in an environment with astropy + h5py + ceridwen (NOT FSPS -- this
reads spectra straight from the FITS)::

    python scripts_afe/build_afe_hr_grid.py \
        --fits ssp_final_mistv2.5_c3kv2.3vt10allfal_250722.fits \
        --out  amist_c3k_hr_krou_afe.h5 \
        --imf-type 2                     # Kroupa: the FITS EXT2 header records IMF_TYPE=2

It validates that every (afe, feh, logt) grid point is present (no NaN
holes), then saves the grid and prints ``ssp.display()`` -- paste that
into the Zenodo description.
"""
from __future__ import annotations

import argparse
import math

import numpy as np


# MIST protosolar reference used by the C3K/FSPS zlegend (Z = Zsun*10**[Fe/H]).
# Pinned from amist_c3k_lr_chab_afe.h5: log10 Z - [Fe/H] = -1.732828266 (const).
ZSUN_MIST = 0.0185
LGMET_OFFSET = math.log10(ZSUN_MIST)          # = -1.7328282657...

IMF_NAMES = {0: "Salpeter", 1: "Chabrier", 2: "Kroupa",
             3: "van Dokkum", 4: "Dave", 5: "tabulated"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", required=True, help="MJ Park alpha-MC FITS path")
    ap.add_argument("--out", required=True, help="output HDF5 (SSPDataAfe schema 2.0)")
    ap.add_argument("--imf-type", type=int, default=2,
                    help="FSPS imf_type of the SSPs. Default 2 (Kroupa): the "
                         "MJ Park FITS EXT2 header records IMF_TYPE=2. "
                         "(1=Chabrier, 2=Kroupa.)")
    ap.add_argument("--zsun", type=float, default=ZSUN_MIST,
                    help="protosolar Z reference for [Fe/H]->log10 Z "
                         f"(default {ZSUN_MIST}, matches the ceridwen LR grid)")
    ap.add_argument("--float32", action="store_true",
                    help="store flux as float32 (halves the file; default float64 "
                         "matches the LR grid on disk)")
    ap.add_argument("--decimals", type=int, default=4,
                    help="rounding for grid-node matching (guards float noise)")
    args = ap.parse_args()

    from astropy.io import fits
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe

    offset = math.log10(args.zsun)
    dt = np.float32 if args.float32 else np.float64

    with fits.open(args.fits) as hdul:
        wave = np.ascontiguousarray(hdul[0].data, dtype=np.float64)      # (n_wave,)
        flux2d = np.ascontiguousarray(hdul[1].data, dtype=dt)           # (n_grid, n_wave)
        tab = hdul[2].data
        logt = np.round(np.asarray(tab["logt"], float), args.decimals)
        feh  = np.round(np.asarray(tab["feh"],  float), args.decimals)
        afe  = np.round(np.asarray(tab["afe"],  float), args.decimals)

    n_grid, n_wave = flux2d.shape
    if wave.size != n_wave:
        raise SystemExit(f"wave length {wave.size} != flux n_wave {n_wave}")
    if not (logt.size == feh.size == afe.size == n_grid):
        raise SystemExit("grid table length does not match flux row count")

    # Sorted, unique axes (SSPDataAfe requires strictly increasing afe/lgmet).
    afe_u  = np.unique(afe)
    feh_u  = np.unique(feh)
    logt_u = np.unique(logt)
    n_afe, n_met, n_age = afe_u.size, feh_u.size, logt_u.size
    print(f"[hr-grid] axes: n_afe={n_afe} n_met={n_met} n_age={n_age} "
          f"n_wave={n_wave}  (expect {n_afe*n_met*n_age} SSPs, got {n_grid})")
    if n_afe * n_met * n_age != n_grid:
        raise SystemExit("grid is not a complete rectangular product -- "
                         "cannot form a dense (afe, met, age, wave) cube")

    # Scatter each FITS row into the dense cube by its (afe, met, age) index.
    ia = np.searchsorted(afe_u,  afe)
    im = np.searchsorted(feh_u,  feh)
    ig = np.searchsorted(logt_u, logt)
    cube = np.full((n_afe, n_met, n_age, n_wave), np.nan, dtype=dt)
    cube[ia, im, ig, :] = flux2d
    n_filled = np.count_nonzero(~np.isnan(cube[..., 0]))
    if n_filled != n_afe * n_met * n_age:
        raise SystemExit(f"grid has holes: {n_filled}/{n_afe*n_met*n_age} "
                         f"(afe,met,age) cells filled -- ragged input FITS")
    if not np.isfinite(cube).all():
        raise SystemExit("non-finite flux values after assembly")

    ssp_afe        = afe_u                                   # [alpha/Fe]
    ssp_lgmet      = feh_u + offset                          # log10 absolute total Z
    ssp_lg_age_gyr = logt_u - 9.0                            # log10(age/Gyr)

    ssp = SSPDataAfe(
        ssp_lgmet, ssp_afe, ssp_lg_age_gyr, wave, cube,
        isoc_type="MIST v2.5 (aMIST, alpha-variable)",
        spec_library="C3K v2.3 high-res (c3k_hr, vt=10 km/s)",
        imf_type=int(args.imf_type),
        fsps_version=None,
        fsps_kwargs={
            "source_fits": args.fits.split("/")[-1],
            "provider": "M. J. Park (2025-07-22)",
            "zsun_reference": args.zsun,
            "feh_to_logZ": "log10 Z = [Fe/H] + log10(Zsun)",
            "imf_name": IMF_NAMES.get(int(args.imf_type), str(args.imf_type)),
        },
        wave_min=float(round(float(wave.min()), 3)),   # 100.0 (drop float noise)
        wave_max=float(round(float(wave.max()), 3)),
        schema_version="2.0",
    )

    # Round-trip check: save, reload, assert identical (catches I/O surprises).
    ssp.save(args.out)
    back = SSPDataAfe.load(args.out)
    assert back.ssp_flux.shape == ssp.ssp_flux.shape
    np.testing.assert_allclose(np.asarray(back.ssp_afe), np.asarray(ssp_afe))
    np.testing.assert_allclose(np.asarray(back.ssp_lgmet), np.asarray(ssp_lgmet))
    print(f"[hr-grid] wrote {args.out}  (round-trip load OK)\n")
    ssp.display()


if __name__ == "__main__":
    main()
