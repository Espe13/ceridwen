"""B1-006: the dust-emission energy balance counts the energy the dust absorbs from the nebular
lines on every path.  ``include_lines`` only decides whether the attenuated lines are in the
output, so the static line basis (fixed-z photometry) and the painted lines (free z, sampled
sigma_gas) predict the same infrared."""
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
from _gridfixture import find_test_grid                                   # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                # noqa: E402
from ceridwen.observation import Photometry                                # noqa: E402

CLIGHT_AA_S = 2.99792458e18


def _csp(dem=True, geometry="runaway_bc", cosmo=None):
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY / dust-emission grids)")
    cosmo = cosmo or Cosmology.planck18()
    return CSPBasis(SSPData.load(str(path)), lookback_time=np.linspace(0, float(cosmo.age(0.1)), 6),
                    zh_const=True, add_neb=True, add_dust=True, add_diffuse_dust=True,
                    add_dust_emission=dem, fesc_geometry=geometry, verbose=False, cosmo=cosmo)


def _theta(csp, tau_diff):
    th = dict(csp.theta_init)
    th.update(sfh=jnp.array([10., 3., 1., 1., 1., 1.]), logzsol=jnp.array([0.0]),
              tau_pow=jnp.array([1.0]), alpha_pow=jnp.array([-1.0]),
              diffuse_tau_kc=jnp.array([tau_diff]), diffuse_dust_index=jnp.array([0.0]),
              gas_logz=jnp.array([0.0]), gas_logu=jnp.array([-2.0]))
    return th


def _integral_nu(wave, f):
    """Trapezoid in frequency, positive for positive f (independent of DustEmission._trap_w)."""
    nu = CLIGHT_AA_S / np.asarray(wave, float)
    return float(-np.trapezoid(np.asarray(f, float), nu))


def _dense_runaway(csp, th, include_lines):
    """(dust-free, attenuated) spectra from the dense nebular cube (``_build_neb_array``), not
    the factored path the model uses."""
    W = np.asarray(csp.calculate_ssp_weights(th), float)
    flux = np.asarray(csp.flux, float)
    neb = np.asarray(csp._build_neb_array(th, include_lines=include_lines), float)
    attn, attn_d = csp.attenuate_dust(csp.wave, th)
    A = np.exp(-np.asarray(csp._age_bin_mix, float) @ np.asarray(attn, float))
    D = np.exp(-np.asarray(attn_d, float))
    ion = np.where(np.asarray(csp.kill_ion), 0.0, 1.0)
    free = np.einsum("za,zaw,aw->w", W, flux, ion) + np.einsum("za,zaw->w", W, neb)
    att = (np.einsum("za,zaw,aw->w", W, flux, ion * A) + np.einsum("za,zaw,aw->w", W, neb, A)) * D
    return free, att


def test_energy_balance_includes_lines():
    """No diffuse dust (so no self-absorption of the dust emission): the re-emitted luminosity
    equals the luminosity absorbed from continuum + lines, on both include_lines settings."""
    csp = _csp()
    th = _theta(csp, 0.0)
    w = np.asarray(csp.wave)
    free_l, att_l = _dense_runaway(csp, th, True)
    _, att_c = _dense_runaway(csp, th, False)
    L_abs = _integral_nu(w, free_l - att_l)
    L_abs_cont_only = _integral_nu(w, _dense_runaway(csp, th, False)[0] - att_c)
    assert L_abs / L_abs_cont_only - 1 > 1e-3        # the lines matter at this level
    for inc, att in ((True, att_l), (False, att_c)):
        E = np.asarray(csp.get_spectrum(th, include_lines=inc), float) - att
        assert abs(_integral_nu(w, E) / L_abs - 1) < 1e-6, inc
    full = np.asarray(csp.get_spectrum(th, include_lines=True), float)
    cont = np.asarray(csp.get_spectrum(th, include_lines=False), float)
    np.testing.assert_allclose(full - cont, att_l - att_c, rtol=1e-4,
                               atol=1e-6 * np.max(np.abs(att_l - att_c)))


@pytest.mark.parametrize("geometry", ["runaway_bc", "picket"])
def test_dust_emission_same_with_and_without_lines(geometry):
    """The dust emission (output minus the dust-emission-free model) is the same whether the
    lines are in the output or not; with lines off it used to miss the lines' energy."""
    dem, nodem = _csp(True, geometry), _csp(False, geometry)
    th = _theta(dem, 0.5)
    if geometry == "picket":
        th["frac_obrun"] = jnp.array([0.3])
    E = {inc: np.asarray(dem.get_spectrum(th, include_lines=inc), float)
         - np.asarray(nodem.get_spectrum(th, include_lines=inc), float) for inc in (False, True)}
    ir = np.asarray(dem.wave) > 3e4
    np.testing.assert_allclose(E[False][ir], E[True][ir], rtol=1e-5)


def test_static_and_painted_photometry_agree_with_dust_emission():
    cosmo = Cosmology.planck18()
    csp = _csp(True, cosmo=cosmo)
    th = {k: v for k, v in _theta(csp, 0.5).items() if not k.startswith("duste_")}
    th.update(duste_qpah=jnp.array([3.5]), duste_umin=jnp.array([1.0]),
              duste_gamma=jnp.array([0.01]))
    th["logmass"] = jnp.array([10.0])
    filt = ["galex_FUV", "sdss_r0", "spitzer_irac_ch4", "herschel_pacs_70",
            "herschel_pacs_160", "herschel_spire_250"]
    out = {}
    try:
        for paint in (False, True):
            phot = Photometry(filters=filt, flux=np.ones(len(filt)),
                              uncertainty=np.ones(len(filt)), name="p")
            m = SedModel(csp, [phot], zred=0.1, broaden_photometry=False)
            csp._force_paint_lines = paint
            out[paint] = np.asarray(m.predict(th)["p"], float)
    finally:
        csp._force_paint_lines = False
    r = out[True] / out[False] - 1
    assert np.all(np.abs(r[2:]) < 1e-6), r      # infrared bands (were 2-3%)
    # UV/optical: the float32 static-basis vs painted line projection, independent of the
    # energy balance (GALEX FUV 8.6e-6 here with dust emission on AND off, measured 2026-09-24)
    assert np.all(np.abs(r) < 2e-5), r
