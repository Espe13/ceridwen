"""Reference surviving-mass fractions (mfrac) from python-fsps, for a future
CERIDWEN surviving-mass feature to reproduce.

mfrac = M_surviving(stars + remnants) / M_formed for three SFHs (constant,
rising SFR ~ t, instantaneous burst) at three population ages, on the
isochrones of the CERIDWEN test SSP grids:

* ``bpass``: the canonical test grid ``ceridwen/data/test_data/ssp_data_bpass.h5``
  (provenance: ``isoc_type='bpss'``, ``spec_library='bpass'``, ``imf_type=1``,
  ``fsps_version='0.5.0'``). For BPASS, FSPS does not use ``imf_type``: the
  SSP mass is read from ``$SPS_HOME/ISOCHRONES/BPASS/bpass.mass``
  (FSPS ``src/sps_setup.f90:185-192``, ``src/ssp_gen.f90:77-80``) and the IMF
  is BPASS's own (``bpass_v2.2_salpeter100``, ``sps_setup.f90:196``).
* ``mist``: the MIST test grids (``ssp_data_mist_c3k_lr*.h5``, ``imf_type=1``).
  mfrac depends on isochrones + IMF only, not on the spectral library, so the
  local MIST/MILES python-fsps build gives the MIST/Chabrier value.

FSPS keeps global Fortran state (one ``StellarPopulation`` per process,
GOTCHAS.md section 6), and the BPASS and MIST builds live in different
conda environments, so each library runs in its own subprocess.

Usage::

    python examples/recipes/make_reference_mfrac.py   # writes reference_mfrac.json

SFH definitions (all at logzsol = 0, i.e. the grid's solar node):

* ``constant``: FSPS ``sfh=3`` table, SFR = 1 on [0, T] (forward time).
* ``rising``: FSPS ``sfh=3`` table, SFR proportional to t on [0, T]. FSPS
  treats a table as piecewise-linear SFR in time (``src/csp_gen.f90:170-176``),
  so both tables are represented exactly.
* ``burst``: FSPS ``sfh=0`` SSP of age T.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ENVS = {
    # library tag -> python executable whose python-fsps was compiled with it
    "bpass": os.path.expanduser("~/opt/anaconda3/envs/phmc/bin/python"),
    "mist": os.path.expanduser("~/opt/anaconda3/envs/ceridwen311/bin/python"),
}
AGES_GYR = [0.1, 1.0, 10.0]

WORKER = r"""
import json, sys
import numpy as np
import fsps
ages = json.loads(sys.argv[1])
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=1, logzsol=0.0,
                            add_neb_emission=False, add_dust_emission=False)
out = {"python_fsps": fsps.__version__,
       "libraries": [l.decode() for l in sp.libraries], "mfrac": {}}
for T in ages:
    row = {}
    t = np.linspace(0.0, T, 201)
    for name, sfr in (("constant", np.ones_like(t)), ("rising", t / T)):
        sp.params["sfh"] = 3
        sp.set_tabular_sfh(t, sfr)
        sp.get_spectrum(tage=T)
        mform = getattr(np, "trapezoid", getattr(np, "trapz", None))(sfr, t * 1e9)   # exact: piecewise-linear SFR
        row[name] = {"stellar_mass": float(sp.stellar_mass),
                     "formed_mass_fsps": float(sp.formed_mass),
                     "formed_mass_trapz": float(mform),
                     "mfrac": float(sp.stellar_mass / sp.formed_mass)}
    sp.params["sfh"] = 0
    sp.get_spectrum(tage=T)
    row["burst"] = {"stellar_mass": float(sp.stellar_mass),
                    "mfrac": float(sp.stellar_mass)}   # SSP: 1 M_sun formed
    out["mfrac"][str(T)] = row
print("JSON" + json.dumps(out))
"""


def run(lib, py):
    # FSPS may leave fort.NN scratch files in its cwd: run it in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run([py, "-c", WORKER, json.dumps(AGES_GYR)], cwd=tmp,
                             capture_output=True, text=True, env=os.environ.copy())
    line = [l for l in res.stdout.splitlines() if l.startswith("JSON")]
    if res.returncode or not line:
        raise RuntimeError(f"{lib}: fsps worker failed\n{res.stderr[-2000:]}")
    return json.loads(line[-1][4:])


def main():
    if not os.environ.get("SPS_HOME"):
        sys.exit("SPS_HOME not set: FSPS unavailable, reference not made")
    sps_home = os.environ["SPS_HOME"]
    out = {
        "description": "Surviving-mass fraction mfrac = M_surviving/M_formed "
                       "(stars + remnants, FSPS stellar_mass / formed_mass).",
        "made_by": "examples/recipes/make_reference_mfrac.py",
        "SPS_HOME": sps_home,
        "ages_gyr": AGES_GYR,
        "logzsol": 0.0,
        "sfh_definitions": {
            "constant": "FSPS sfh=3 table SFR=1 on [0,T] forward time",
            "rising": "FSPS sfh=3 table SFR proportional to t on [0,T]",
            "burst": "FSPS sfh=0 SSP of age T"},
        "libraries": {},
    }
    for lib, py in ENVS.items():
        out["libraries"][lib] = run(lib, py)
    out["libraries"]["bpass"]["note"] = (
        "Matches ceridwen/data/test_data/ssp_data_bpass.h5 isochrones "
        "(isoc_type='bpss'). BPASS ignores imf_type; mass from "
        "$SPS_HOME/ISOCHRONES/BPASS/bpass.mass. python-fsps here is "
        "0.4.7 (the grid was built with 0.5.0); the mass table is read from "
        "SPS_HOME, not compiled in.")
    out["libraries"]["mist"]["note"] = (
        "Matches the MIST test grids (isoc_type='mist', imf_type=1). "
        "Spectral library differs (miles vs c3k_lr), which does not enter mfrac.")
    path = HERE / "reference_mfrac.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
