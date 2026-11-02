"""The nebular line-width floor is ``res_floor_fwhm_px`` model-grid pixels per FWHM, as in FSPS
(``sps_setup.f90:999-1001``: ``dlam = dlam*2/2.355*nebular_smooth_factor``, the sigma of the
Gaussians in ``gaussnebarr``); the pre-1.1.0 ``res_floor_factor``
(pixels per sigma) raises with the conversion."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import require_test_grid                                # noqa: E402

from ceridwen import SSPData, CSPBasis, Cosmology, NebularModel           # noqa: E402
from ceridwen.neb.NebularGridModel import CLIGHT_AA_S                      # noqa: E402


@pytest.fixture(scope="module")
def ssp():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    return SSPData.load(str(require_test_grid()))


def _neb(ssp, **kw):
    return NebularModel(sps_home=os.environ["SPS_HOME"], csp_lambda=np.asarray(ssp.ssp_wave),
                        isoc_type=ssp.isoc_type, **kw)


def test_floor_is_two_pixels_per_fwhm_as_fsps(ssp):
    neb = _neb(ssp)
    pix = np.asarray(neb.neb_res_min, dtype=np.float64)
    np.testing.assert_allclose(np.asarray(neb.dlam_lines), pix * 2.0 / 2.355, rtol=1e-12, atol=0)


def test_floor_keeps_the_integrated_line_flux(ssp):
    """The painted Gaussian (``gaussnebarr`` without its lambda_0^2 / c factor) integrates to 1
    in lambda: sigma = 0.85 px sampled on a uniform grid sums to 1 within
    2 exp(-2 pi^2 (sigma/px)^2) = 1.3e-6.  (In frequency the profile integrates to
    E[(lambda_0/lambda)^2] = 1 + 3 (sigma/lambda_0)^2 + ..., FSPS's lambda_0^2 convention.)"""
    neb = _neb(ssp)
    wave = np.asarray(neb.csp_lambda, dtype=np.float64)
    pos = np.asarray(neb.nebem_line_pos, dtype=np.float64)
    G = np.asarray(neb.gaussnebarr, dtype=np.float64)
    flux = np.trapezoid(G * CLIGHT_AA_S / pos[None, :] ** 2, wave, axis=0)
    # lines on uniform pixels: the same pixel width over +-8 old sigmas around the line
    idx = np.searchsorted(wave, pos, side="right") - 1
    ok = np.zeros(pos.size, dtype=bool)
    for j, i in enumerate(idx):
        lo, hi = i - 20, i + 21
        if lo >= 0 and hi < wave.size:
            d = np.diff(wave[lo:hi])
            ok[j] = np.ptp(d) <= 1e-9 * d.mean()
    assert ok.sum() > 50
    np.testing.assert_allclose(flux[ok], 1.0, rtol=2e-6, atol=0)


def test_res_floor_fwhm_px_sets_the_fwhm(ssp):
    neb = _neb(ssp, res_floor_fwhm_px=1.0)
    np.testing.assert_allclose(np.asarray(neb.dlam_lines),
                               np.asarray(neb.neb_res_min) * 1.0 / 2.355, rtol=1e-12, atol=0)
    # the pre-1.1.0 width (res_floor_factor pixels per sigma) is res_floor_fwhm_px = 2.355 x that
    old = _neb(ssp, res_floor_fwhm_px=2.0 * 2.355)
    np.testing.assert_allclose(np.asarray(old.dlam_lines), 2.0 * np.asarray(old.neb_res_min),
                               rtol=1e-12, atol=0)


def test_removed_res_floor_factor_raises_with_the_conversion(ssp):
    with pytest.raises(TypeError, match=r"res_floor_fwhm_px=2\.355"):
        _neb(ssp, res_floor_factor=1.0)
    with pytest.raises(TypeError, match="res_floor_fwhm_px"):
        CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 1.0, 3), zh_const=True, verbose=False,
                 cosmo=Cosmology.planck18(), init_neb_params={"res_floor_factor": 3.0})
    with pytest.raises(TypeError, match="unexpected keyword"):
        _neb(ssp, res_floor_fwhm=2.0)
