"""THEMIS dust emission is selectable: ``CSPBasis(duste_model="THEMIS")`` (2026-09-22).

Before, ``CSPBasis`` always built ``DustEmission`` with its DL07 default, so the THEMIS branch
of ``DustEmission`` was unreachable from the model.  Checks:

1. the (qPAH, Umin) axes are FSPS's own THEMIS axes (``src/sps_vars.f90``, read from
   ``$SPS_HOME/src`` when present), and a template column equals a direct read of the
   ``$SPS_HOME/dust/dustem/THEMIS_MW3.1_*.dat`` file interpolated onto the model grid;
2. energy balance: with no self-absorption the emitted dust luminosity equals the absorbed
   stellar luminosity;
3. the CSP spectrum differs from DL07 only redward of 1 um (the templates start there), the
   default is DL07, and misuse raises at construction.
"""
import os
import pathlib
import re

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ceridwen import CSPBasis, SSPData, Cosmology

from _gridfixture import require_test_grid

_SSP = str(require_test_grid())
SPS_HOME = os.environ.get("SPS_HOME")
pytestmark = pytest.mark.skipif(not SPS_HOME, reason="SPS_HOME not set (dust-emission templates)")


def _csp(**kw):
    return CSPBasis(SSPData.load(_SSP), lookback_time=jnp.linspace(0.0, 13.0, 6),
                    cosmo=Cosmology.planck18(), zh_const=True, add_neb=False, add_dust=True,
                    add_diffuse_dust=True, verbose=False, sps_home=SPS_HOME, **kw)


@pytest.fixture(scope="module")
def themis():
    return _csp(add_dust_emission=True, duste_model="THEMIS")


def _fsps_axes(name):
    src = pathlib.Path(SPS_HOME) / "src" / "sps_vars.f90"
    if not src.is_file():
        pytest.skip(f"{src} not present")
    txt = src.read_text()
    blk = txt[txt.index(f"#elif ({name})") if name != "DL07" else txt.index("#if (DL07)"):]
    blk = blk[:blk.index("#e", 5)]

    def arr(var):
        m = re.search(var + r"\s*=\s*&?\s*\(/(.*?)/\)(/[\d.]+\*[\d.]+)?", blk, re.S)
        vals = np.array([float(v) for v in m.group(1).replace("&", "").split(",")])
        if m.group(2):
            a, b = m.group(2)[1:].split("*")
            vals = vals / float(a) * float(b)
        return vals
    return arr("qpaharr"), arr("uminarr")


def test_axes_match_fsps(themis):
    q, u = _fsps_axes("THEMIS")
    np.testing.assert_allclose(np.asarray(themis.dust_emi.qpaharr), q, rtol=1e-15)
    np.testing.assert_array_equal(np.asarray(themis.dust_emi.uminarr), u)
    assert themis.dust_emi.duste_model == "THEMIS"


def test_template_column_is_the_file(themis):
    k, j = 3, 10                                   # qPAH node 3, Umin column pair 5 (Umin)
    f = pathlib.Path(SPS_HOME) / "dust" / "dustem" / "THEMIS_MW3.1_30.dat"
    tab = np.loadtxt(f, skiprows=2, max_rows=576)
    wave = np.asarray(themis.wave)
    jj = int(np.searchsorted(wave / 1e4, 1, side="left"))
    ref = np.zeros_like(wave)
    ref[jj:] = np.interp(wave[jj:], tab[:, 0] * 1e4, tab[:, 1 + j])
    np.testing.assert_array_equal(np.asarray(themis.dust_emi.dustem2_dustem[:, k, j]), ref)


def test_energy_balance(themis):
    th = dict(themis.theta_init)
    spec_free = jnp.sum(themis.flux[0], axis=0)
    _, attn_diffuse = themis.attenuate_dust(themis.wave, dict(th, diffuse_tau_kc=jnp.array([0.5])))
    dc = jnp.exp(-jnp.ravel(attn_diffuse))                   # attenuate_dust returns tau
    spec_attn = spec_free * dc
    de = themis.dust_emi
    _, _, tduste = de.compute_dust_emission(spec_attn, spec_free, themis.wave,
                                            jnp.ones_like(dc),       # no self-absorption
                                            jnp.asarray(3.5), jnp.asarray(1.0), jnp.asarray(0.01))
    absorbed = float(jnp.dot(spec_free - spec_attn, de._trap_w))
    emitted = float(jnp.dot(tduste, de._trap_w))
    assert abs(absorbed) > 0                  # _trap_w runs over decreasing nu: both negative
    assert emitted == pytest.approx(absorbed, rel=1e-10)


def test_spectrum_differs_from_dl07_only_in_ir(themis):
    dl07 = _csp(add_dust_emission=True)
    assert dl07.dust_emi.duste_model == "DL07" and dl07.duste_model == "DL07"
    th = dict(themis.theta_init, diffuse_tau_kc=jnp.array([0.5]))
    a = np.asarray(themis.get_spectrum(th))
    b = np.asarray(dl07.get_spectrum(th))
    uv_opt = np.asarray(themis.wave) < 9000.0
    # blueward of 1 um both template sets are floored at 1e-70, so the models differ only there
    np.testing.assert_allclose(a[uv_opt], b[uv_opt], rtol=1e-12, atol=1e-80)
    ir = (np.asarray(themis.wave) > 5e4) & (np.asarray(themis.wave) < 1e6)
    assert np.max(np.abs(a[ir] / b[ir] - 1.0)) > 0.05
    g = jax.grad(lambda q: jnp.sum(themis.get_spectrum(dict(th, duste_qpah=q))))(jnp.array([5.0]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_misuse_raises():
    with pytest.raises(ValueError, match="'DL07' or 'THEMIS'"):
        _csp(add_dust_emission=True, duste_model="themis")
    with pytest.raises(ValueError, match="add_dust_emission=False"):
        _csp(duste_model="THEMIS")
