"""One set of physical constants (B1-003, B1-020).

The L_sun/Hz -> F_nu unit constant is checked against FSPS itself: FSPS's magnitudes
(``getmags.f90``) are ``-2.5 log10(int L_nu band/lambda) - 48.6 - 2.5 mag2cgs`` with
``mag2cgs = log10(lsun / (4 pi (10 pc)^2))`` (``sps_vars.f90:422,426,432``), so replaying that
formula with CERIDWEN's constant must reproduce ``fsps.StellarPopulation.get_mags``.
"""
from __future__ import annotations

import importlib
import math
import pathlib

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from ceridwen import constants
from ceridwen.cosmology import flux_factor_cgs

from _gridfixture import FIXTURE_DIR, TEST_DATA_DIR

# FSPS src/sps_vars.f90:422,426 (single precision there)
FSPS_LSUN = 3.839e33
FSPS_PC2CM = 3.08568e18


def test_flux_factor_is_fsps_lsun_at_10pc():
    """The 10 pc factor is FSPS's lsun / (4 pi (10 pc)^2); the only difference is FSPS's
    rounded parsec (1.6e-6).  The former 3.1967965e-7 was 0.367 % low."""
    fsps_to_cgs = FSPS_LSUN / (4.0 * math.pi * (10.0 * FSPS_PC2CM) ** 2)
    ff = float(flux_factor_cgs(0.0))                    # z <= 0 is pinned to 10 pc
    assert abs(ff / fsps_to_cgs - 1.0) < 2e-6
    assert abs(ff / 3.1967965e-7 - 1.0) > 3e-3


def test_one_lsun_everywhere():
    """Flux factor, nebular Q and PostProcess luminosities all use the same L_sun."""
    NebularGridModel = importlib.import_module("ceridwen.neb.NebularGridModel")
    from ceridwen import postprocess
    assert NebularGridModel.LSUN_ERG_S == constants.LSUN_ERG_S == FSPS_LSUN
    assert postprocess._LSUN_ERG_S == constants.LSUN_ERG_S
    d = 10.0 * constants.PC_TO_CM
    assert math.isclose(constants.LSUN_HZ_TO_FNU_CGS_AT_10PC * 4.0 * math.pi * d * d,
                        constants.LSUN_ERG_S, rel_tol=1e-14)


def _mist_miles_grid():
    for d in (FIXTURE_DIR, TEST_DATA_DIR):
        p = pathlib.Path(d) / "ssp_data_mist_miles.h5"
        if p.is_file():
            return p
    pytest.skip("ssp_data_mist_miles.h5 test grid not found")


@pytest.mark.fsps
def test_ssp_magnitudes_match_fsps():
    """FSPS get_mags of grid nodes vs FSPS's own magnitude formula with CERIDWEN's constant.
    Agreement is limited by FSPS's single precision and rounded parsec (~3e-6)."""
    fsps = pytest.importorskip("fsps")
    from ceridwen.ssps.ssp_data import SSPData
    data = SSPData.load(str(_mist_miles_grid()))
    sp = fsps.StellarPopulation(zcontinuous=0, **data.fsps_kwargs)
    libs = [b.decode().strip().lower() if isinstance(b, bytes) else str(b).lower()
            for b in sp.libraries[:2]]
    if libs != ["mist", "miles"]:
        pytest.skip(f"python-fsps is compiled with {libs}, the grid is MIST/MILES")
    lam = np.asarray(sp.wavelengths, dtype=float)
    assert np.array_equal(lam, np.asarray(data.ssp_wave))

    def tsum(x, y):                                     # FSPS tsum.f90
        return np.sum(np.abs(np.diff(x)) * 0.5 * (y[1:] + y[:-1]))

    ff = float(flux_factor_cgs(0.0))
    bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_ks", "galex_nuv"]
    ratios = []
    for iz, ia in [(9, 90), (0, 20)]:
        mags = sp.get_mags(tage=0.0, zmet=iz + 1, bands=bands)[ia]
        lnu = np.asarray(data.ssp_flux)[iz, ia]
        assert np.array_equal(lnu, sp.get_spectrum(tage=0.0, zmet=iz + 1, peraa=False)[1][ia])
        for b, m in zip(bands, mags):
            fw, ft = (np.asarray(a, dtype=float) for a in fsps.get_filter(b).transmission)
            band = np.zeros_like(lam)                   # sps_setup.f90: interpolate, normalise
            i1 = max(np.searchsorted(lam, fw[0], side="right") - 1, 0)
            i2 = np.searchsorted(lam, fw[-1], side="right") - 1
            band[i1:i2 + 1] = np.interp(lam[i1:i2 + 1], fw, np.maximum(ft, 0.0))
            band /= tsum(lam, band / lam)
            fnu = tsum(lam, lnu * band / lam) * ff
            ratios.append(10 ** (-0.4 * m) / (fnu / 10 ** (-0.4 * 48.6)))
    dev = np.abs(np.array(ratios) - 1.0)
    # median ~2.9e-6 (FSPS float32 + parsec); the filter replay adds up to ~1.3e-5 at band
    # edges (sdss_u).  The former constant was off by 3.67e-3 in every band.
    assert np.median(dev) < 5e-6
    assert np.max(dev) < 5e-5


# ---- B1-020: one speed of light ---------------------------------------------------------

C_SI = 299792458.0                                      # m/s, exact (SI definition)


def test_every_module_uses_the_exact_speed_of_light():
    from ceridwen import broadening, igm
    DustEmission = importlib.import_module("ceridwen.dust.DustEmission")
    NebularGridModel = importlib.import_module("ceridwen.neb.NebularGridModel")
    from ceridwen.observation import filters
    from ceridwen.ssps import library_resolution
    assert constants.C_KMS == C_SI * 1e-3
    assert constants.C_CMS == C_SI * 1e2
    assert constants.C_AA_S == C_SI * 1e10
    for v in (broadening.C_AA_S, NebularGridModel.CLIGHT_AA_S, DustEmission.CLIGHT_AA_S,
              filters.lightspeed, filters.Filter.lightspeed):
        assert v == constants.C_AA_S
    for v in (broadening.CKMS, library_resolution.CKMS):
        assert v == constants.C_KMS
    assert igm._C_CMS == constants.C_CMS


def test_no_hard_coded_speed_of_light_in_the_package():
    """No numeric literal within 1e-3 of c (km/s, cm/s or A/s) outside ceridwen/constants.py."""
    import ast
    root = pathlib.Path(constants.__file__).resolve().parent
    targets = (C_SI * 1e-3, C_SI * 1e2, C_SI * 1e10)
    hits = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "constants.py" or "tests" in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool):
                if any(abs(node.value / t - 1.0) < 1e-3 for t in targets):
                    hits.append(f"{path.relative_to(root)}:{node.lineno} {node.value!r}")
    assert not hits, hits
