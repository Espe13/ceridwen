"""B1-024: the nebular default is the CLOUDY grid without dust in the H II region (ZAU_ND), as in
FSPS, python-fsps and Prospector.  A default CSPBasis reproduces python-fsps's default nebular
lines for one young population (same tolerances as the gas_tied parity test in
test_logzsol_convention.py: line ratios 1e-6, absolute luminosities 2e-3, Q differs ~5e-4)."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import TEST_DATA_DIR                                    # noqa: E402

from ceridwen import SSPData, CSPBasis, Cosmology                          # noqa: E402

GRIDS = {"mist": "ssp_data_mist_miles.h5", "bpss": "ssp_data_bpass.h5"}
LINES = {"Ly-alpha": 1215.67, "[O II]3727": 3727.1, "H-beta": 4862.71,
         "[O III]5007": 5008.24, "H-alpha": 6564.61, "[N II]6584": 6585.27}


def _val(arr, ww, x):
    return float(arr[int(np.argmin(np.abs(np.asarray(ww) - x)))])


def test_default_is_the_dust_free_grid():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set")
    path = TEST_DATA_DIR / "ssp_data_bpass.h5"
    if not path.is_file():
        pytest.skip("no BPASS test grid")
    csp = CSPBasis(SSPData.load(str(path)), lookback_time=jnp.linspace(0.0, 1.0, 3),
                   zh_const=True, verbose=False, cosmo=Cosmology.planck18())
    assert csp.neb.cloudy_dust is False
    assert csp.neb.line_file.name.startswith("ZAU_ND_")
    # an init_neb_params without cloudy_dust keeps the default (it used to need the key)
    csp2 = CSPBasis(SSPData.load(str(path)), lookback_time=jnp.linspace(0.0, 1.0, 3),
                    zh_const=True, verbose=False, cosmo=Cosmology.planck18(),
                    init_neb_params={"sigma_smooth": 0.0})
    assert csp2.neb.cloudy_dust is False


def test_default_lines_match_python_fsps_default():
    fsps = pytest.importorskip("fsps")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set")
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=1, add_neb_emission=True,
                                gas_logu=-2.0, gas_logz=0.0, logzsol=0.0)
    # python-fsps 0.5 has no cloudy_dust parameter; FSPS reads ZAU_ND unless compiled with
    # cloudy_dust = 1 (src/sps_vars.f90), so this is the FSPS default grid
    libs = [b.decode() if isinstance(b, bytes) else str(b) for b in sp.libraries]
    name = GRIDS.get(libs[0])
    if name is None or not (TEST_DATA_DIR / name).is_file():
        pytest.skip(f"no local grid for isochrones {libs[0]}")
    ssp = SSPData.load(str(TEST_DATA_DIR / name))
    if float(sp.solar_metallicity) != pytest.approx(ssp.zsun_nominal, rel=1e-6):
        pytest.skip("this FSPS build's zsol does not match the local grid")
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 0.05, 3), zh_const=True,
                   add_dust=False, add_diffuse_dust=False, verbose=False,
                   cosmo=Cosmology.planck18())                 # default init_neb_params
    ages = np.asarray(csp._neb_ages_young)
    iz = int(np.argmin(np.abs(np.asarray(csp.zmet))))          # logzsol = 0
    ia = int(np.argmin(np.abs(ages - 6.5)))                    # ~3 Myr
    row = np.asarray(csp.neb.evaluate_batch_line_lum(
        jnp.array(0.0), jnp.array(-2.0), csp._neb_ages_young, csp._neb_logqq_young))[iz, ia]
    wl = np.asarray(csp.neb.nebem_line_pos)
    sp.get_spectrum(tage=10.0 ** (ages[ia] - 9.0), peraa=False)
    w, f = np.asarray(sp.emline_wavelengths), np.asarray(sp.emline_luminosity)
    hb_c, hb_f = _val(row, wl, LINES["H-beta"]), _val(f, w, LINES["H-beta"])
    for tag, x in LINES.items():
        assert _val(row, wl, x) / hb_c == pytest.approx(_val(f, w, x) / hb_f, rel=1e-6), tag
        assert _val(row, wl, x) == pytest.approx(_val(f, w, x), rel=2e-3), tag
