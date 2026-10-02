"""L_bol for the AGN torus: CERIDWEN grid integral vs python-fsps ``log_lbol``.

FSPS scales its torus by ``10**lbol_csp`` (agn_dust.f90:29), an isochrone
luminosity sum (csp_gen.f90:280). CERIDWEN grids carry no lbol table, so the
plan (AGN_PLAN.md section 5) uses the dust-free stellar spectrum integral with
the DL07 trapezoid weights (``DustEmission._trap_w``, DustEmission.py:72-78).
This script quotes the fractional difference between the two.

Two interpreters, because python-fsps lives only in the reference env:

    $CERIDWEN_PYTHON scripts/agn_lbol_check.py [--grid PATH] [--refpy PATH]

runs CERIDWEN's side and calls itself with ``--fsps`` under ``--refpy``.

The grid defaults to the published MIST/MILES grid (python-fsps 0.5.0,
imf_type=1), whose isochrones match python-fsps's (log_lbol does not depend
on the spectral library). Rows:

- SSP at grid-node ages, solar node: grid integral vs FSPS log_lbol
  (and FSPS's own spectrum integral, on its own grid, vs its log_lbol).
- CSP, constant SFR over the last 1 Gyr: CERIDWEN side = numpy trapezoid in
  linear age of the per-node integrals (not CSPBasis); FSPS side = tabular
  sfh=3, log_lbol / formed mass.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

AGES_GYR = [0.001, 0.01, 0.1, 1.0, 10.0]
CLIGHT_AA_S = 2.99792458e18
DEFAULT_GRID = os.path.expanduser(
    "~/Desktop/PhD/Tools/ceridwen_workspace/data/grid_deposit/ssp_data_mist_miles.h5")
DEFAULT_REFPY = "/Users/amanda/opt/anaconda3/envs/prospector/bin/python"


def trap_w(wave):
    """DustEmission._trap_w, as written there (negative: nu falls with wave)."""
    nu = CLIGHT_AA_S / np.asarray(wave, np.float64)
    dnu = np.diff(nu)
    return np.concatenate([[0.5 * dnu[0]], 0.5 * (dnu[:-1] + dnu[1:]), [0.5 * dnu[-1]]])


def fsps_side(imf_type):
    import fsps
    out = {"python_fsps": fsps.__version__}
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, logzsol=0.0, imf_type=imf_type,
                                dust2=0.0, add_dust_emission=False, add_neb_emission=False,
                                add_igm_absorption=False)
    out["libraries"] = [x.decode() for x in sp.libraries]
    w = np.asarray(sp.wavelengths, np.float64)
    tw = trap_w(w)
    ssp = []
    for a in AGES_GYR:
        _, s = sp.get_spectrum(tage=a, peraa=False)
        ssp.append({"age_gyr": a, "log_lbol": float(sp.log_lbol),
                    "own_integral": float(-np.dot(s, tw))})
    out["ssp"] = ssp
    sp.params["sfh"] = 3
    t = np.linspace(0.0, 1.0, 1001)
    sp.set_tabular_sfh(t, np.ones_like(t))
    _, s = sp.get_spectrum(tage=1.0, peraa=False)
    out["csp"] = {"log_lbol": float(sp.log_lbol), "formed_mass": float(sp.formed_mass),
                  "own_integral": float(-np.dot(s, tw))}
    return out


def ceridwen_side(grid):
    from ceridwen.ssps.ssp_data import SSPData
    ssp = SSPData.load(grid)
    wave = np.asarray(ssp.ssp_wave, np.float64)
    lgmet = np.asarray(ssp.ssp_lgmet, np.float64)
    lgage = np.asarray(ssp.ssp_lg_age_gyr, np.float64)
    flux = np.asarray(ssp.ssp_flux, np.float64)
    lz = float(ssp.log10_zsun)  # the grid's resolved solar node (ssp_data.py:166)
    iz = int(np.argmin(np.abs(lgmet - lz)))
    assert lgmet[iz] == lz
    tw = trap_w(wave)
    lint = -flux[iz] @ tw  # (n_age,) L_sun per M_sun formed
    rows = []
    for a in AGES_GYR:
        ia = int(np.argmin(np.abs(lgage - np.log10(a))))
        rows.append({"age_gyr": a, "node_lg_age": float(lgage[ia]), "integral": float(lint[ia])})
    # constant SFR over [0, 1] Gyr, per M_sun formed, trapezoid in linear age
    age = 10 ** lgage
    m = age <= 1.0
    csp = float(np.trapezoid(lint[m], age[m]) / (age[m][-1] - 0.0))
    return {"grid": grid, "solar_node_lgmet": float(lgmet[iz]), "log10_zsun": lz,
            "wave_range": [float(wave[0]), float(wave[-1])], "n_wave": len(wave),
            "ssp": rows, "csp_integral": csp,
            "csp_note": f"trapezoid over {int(m.sum())} nodes from age {age[m][0]:.2e} Gyr "
                        "to 1 Gyr, divided by 1 Gyr (the [0, first node] sliver is dropped)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsps", action="store_true")
    ap.add_argument("--imf", type=int, default=1)
    ap.add_argument("--grid", default=DEFAULT_GRID)
    ap.add_argument("--refpy", default=DEFAULT_REFPY)
    args = ap.parse_args()
    if args.fsps:
        print(json.dumps(fsps_side(args.imf)))
        return
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    fs = json.loads(subprocess.check_output(
        [args.refpy, os.path.abspath(__file__), "--fsps", "--imf", str(args.imf)],
        env=env, text=True, stderr=subprocess.DEVNULL).strip().splitlines()[-1])
    ce = ceridwen_side(args.grid)
    print(f"grid: {ce['grid']}  (solar node lgmet {ce['solar_node_lgmet']:.4f}, "
          f"{ce['n_wave']} px, {ce['wave_range'][0]:g}-{ce['wave_range'][1]:g} A)")
    print(f"python-fsps {fs['python_fsps']} libraries {fs['libraries']}, imf_type={args.imf}")
    print(f"{'age/Gyr':>8} {'FSPS log_lbol':>14} {'CERIDWEN int':>13} "
          f"{'frac(CER/FSPS-1)':>17} {'FSPS own int/lbol-1':>20}")
    for f_, c_ in zip(fs["ssp"], ce["ssp"]):
        L = 10 ** f_["log_lbol"]
        print(f"{f_['age_gyr']:8.3f} {f_['log_lbol']:14.6f} {c_['integral']:13.6e} "
              f"{c_['integral'] / L - 1:+17.4e} {f_['own_integral'] / L - 1:+20.4e}")
    Lc = 10 ** fs["csp"]["log_lbol"] / fs["csp"]["formed_mass"]
    print(f"CSP const SFR 1 Gyr: FSPS L/Mformed {Lc:.6e}, CERIDWEN {ce['csp_integral']:.6e}, "
          f"frac {ce['csp_integral'] / Lc - 1:+.4e}; FSPS own int/lbol-1 "
          f"{fs['csp']['own_integral'] / 10 ** fs['csp']['log_lbol'] - 1:+.4e}")
    print("CSP note:", ce["csp_note"])


if __name__ == "__main__":
    main()
