"""Independent check of examples/recipes/reference_mfrac.json (BPASS part).

The BPASS SSP surviving mass is a table, $SPS_HOME/ISOCHRONES/BPASS/bpass.mass
(FSPS src/sps_setup.f90:185-192): column 0 = log10(age/yr), then one column
per metallicity of zlegend.dat. This script recomputes mfrac from that table
alone, without FSPS:

* burst:    m_ssp(T), read at the node log10(T/yr) (8, 9, 10 are nodes).
* constant: (1/T) int_0^T m_ssp(tau) dtau,
* rising:   int_0^T (T - tau) m_ssp(tau) dtau / (T^2/2)   (SFR ~ t = T - tau),
  with m_ssp linear in log10(tau) between nodes and 1 below the first node.

FSPS works in single precision (real(SP)), so the tolerance is 1e-7; the
measured agreement is <= 2.8e-9 for all nine numbers.
"""
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
ref = json.loads((HERE / "reference_mfrac.json").read_text())["libraries"]["bpass"]
bdir = pathlib.Path(os.environ["SPS_HOME"]) / "ISOCHRONES" / "BPASS"
tab = np.loadtxt(bdir / "bpass.mass")
zleg = np.loadtxt(bdir / "zlegend.dat")
iz = int(np.argmin(np.abs(zleg - 0.020)))          # BPASS Z_sun = 0.020
assert abs(zleg[iz] - 0.020) < 1e-12, zleg
logt, m = tab[:, 0], tab[:, 1 + iz]

ok = True
for T_str, row in ref["mfrac"].items():
    T = float(T_str) * 1e9
    tau = np.concatenate([np.linspace(0, 10 ** logt[0], 200)[:-1],
                          np.logspace(logt[0], np.log10(T), 20001)])
    ms = np.interp(np.log10(np.maximum(tau, 1.0)), logt, m, left=1.0)
    ind = {"burst": float(np.interp(np.log10(T), logt, m)),
           "constant": np.trapezoid(ms, tau) / T,
           "rising": np.trapezoid((T - tau) * ms, tau) / (T ** 2 / 2)}
    for k, tol in (("burst", 1e-7), ("constant", 1e-7), ("rising", 1e-7)):
        d = abs(row[k]["mfrac"] - ind[k])
        good = d <= tol
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} bpass T={T_str:>4} Gyr {k:8s} "
              f"fsps={row[k]['mfrac']:.5f} table={ind[k]:.5f} |d|={d:.1e} tol={tol:.0e}")
print("PASS check_reference_mfrac" if ok else "FAIL check_reference_mfrac")
sys.exit(0 if ok else 1)
