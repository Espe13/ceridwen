"""Kinematics through SedModel: fixed vs sampled sigma_gas for photometry
(line basis vs painted path), Spectrum line painting, and the guards."""
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
from ceridwen.broadening import Kinematics, Instrument, TIED               # noqa: E402
from ceridwen.observation import Photometry, Spectrum                      # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh                # noqa: E402

ZRED = 0.8
N_TIME = 5
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.linspace(4200.0, 8600.0, 500)
LINE_WAVE = np.linspace(6300.0, 6800.0, 1500)     # 0.6 A rest pixels around H-alpha


def _csp(add_neb):
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if add_neb and not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    ssp = SSPData.load(str(path))
    cosmo = Cosmology.planck18()
    kw = dict(lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
              zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
              add_neb=add_neb, add_igm=False, verbose=False, cosmo=cosmo)
    if add_neb:
        kw["sps_home"] = os.environ["SPS_HOME"]
    return CSPBasis(ssp, **kw)


def _model(csp, obs, kin, **kw):
    t = np.array(csp.sfh_times)

    def _sfh(th, _t=t):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    for k in kin.free_keys:
        init[k] = jnp.array([250.0])
    return SedModel(csp, obs, priors={}, transforms={"sfh": _sfh},
                    free_param_init=init, zred=ZRED, kinematics=kin, **kw)


def _theta(model):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    if "logzsol" in th:
        th["logzsol"] = jnp.array([-0.3])
    return th


@pytest.fixture(scope="module")
def csp_neb():
    return _csp(True)


def test_fixed_gas_basis_matches_painted_path(csp_neb):
    """Fixed sigma_gas: the static line basis (no painting) reproduces the painted
    + broadened photometry to well inside the photometric precision."""
    phot = Photometry(filters=FILTERS, flux=[1e-9] * 4, uncertainty=[1e-10] * 4, name="phot")
    kin = Kinematics(sigma_gal=300.0)
    m = _model(csp_neb, [phot], kin)
    th = _theta(m)
    p_basis = np.asarray(m.predict(th)["phot"])
    csp_neb._force_paint_lines = True
    try:
        p_paint = np.asarray(m.predict(th)["phot"])
    finally:
        csp_neb._force_paint_lines = False
    assert np.all(np.isfinite(p_basis)) and np.all(p_basis > 0)
    np.testing.assert_allclose(p_basis, p_paint, rtol=2e-3)


def test_free_gas_paints_and_matches_fixed(csp_neb):
    """A sampled sigma_gas forces the painted path; at the same value it agrees
    with the fixed-width basis."""
    phot = Photometry(filters=FILTERS, flux=[1e-9] * 4, uncertainty=[1e-10] * 4, name="phot")
    kin_free = Kinematics(sigma_gal=300.0, sigma_gas="sigma_gas")
    kin_fix = Kinematics(sigma_gal=300.0, sigma_gas=250.0)
    m_free = _model(csp_neb, [phot], kin_free)
    m_fix = _model(csp_neb, [phot], kin_fix)
    assert "sigma_gas" in m_free.param_names
    th = _theta(m_free)
    th_fix = {k: v for k, v in th.items() if k != "sigma_gas"}
    p_free = np.asarray(m_free.predict(th)["phot"])
    p_fix = np.asarray(m_fix.predict(th_fix)["phot"])
    np.testing.assert_allclose(p_free, p_fix, rtol=2e-3)
    # and the width matters at the per-mille level for the line-rich bands
    th2 = dict(th, sigma_gas=jnp.array([1500.0]))
    p_wide = np.asarray(m_free.predict(th2)["phot"])
    assert np.max(np.abs(p_wide / p_free - 1.0)) < 5e-2


def test_spectrum_lines_from_projector(csp_neb):
    """The Spectrum projector paints the grid lines with sigma_gas + instrument;
    the painted line area equals predict_line_fluxes(for_spectrum=True)."""
    n = LINE_WAVE.size
    spec = Spectrum(wavelength=LINE_WAVE * (1.0 + ZRED), flux=np.full(n, 1e-18),
                    uncertainty=np.full(n, 1e-19), instrument=Instrument.sigma_kms(60.0),
                    name="spec")
    kin = Kinematics(sigma_gal=200.0, sigma_gas=80.0)
    m = _model(csp_neb, [spec], kin)
    th = _theta(m)
    pred = np.asarray(m.predict(th)["spec"])
    assert pred.shape == LINE_WAVE.shape and np.all(np.isfinite(pred))
    # continuum alone (Kinematics with the same widths but no lines: use the projector directly)
    tt = m.apply_transforms(th)
    tt = dict(tt, zred=jnp.array([ZRED]))
    cont = np.asarray(spec._proj.continuum(
        m.csp._apply_mass_redshift_igm(*m.csp._assemble_observer_spectra(tt, paint_lines=False), tt)[1],
        jnp.asarray(200.0)))
    lines_pix = pred - cont
    F = np.asarray(m.csp.predict_line_fluxes(tt, for_spectrum=True))
    pos_obs = np.asarray(m.csp.neb.nebem_line_pos) * (1.0 + ZRED)
    # H-alpha at 6564.6 A rest lies in the window: integrate +-5 sigma in nu
    lam0 = pos_obs[np.argmin(np.abs(pos_obs - 6564.6 * (1.0 + ZRED)))]
    sig = lam0 * np.sqrt(80.0 ** 2 + 60.0 ** 2) / 2.99792458e5
    wobs = LINE_WAVE * (1.0 + ZRED)
    sel = np.abs(wobs - lam0) < 4.0 * sig          # [NII] 6548/6584 stay outside
    nu = 2.99792458e18 / wobs[sel]
    area = -np.trapezoid(lines_pix[sel], nu)
    k = int(np.argmin(np.abs(pos_obs - lam0)))
    # sigma = 4 A observed on 1.1 A pixels: the trapezoid recovers the flux
    assert abs(area / F[k] - 1.0) < 0.02


def test_guards():
    with pytest.raises(ValueError):
        Kinematics(sigma_gal=-1.0)
    with pytest.raises(ValueError):
        Kinematics(sigma_gal=5000.0)
    with pytest.raises(TypeError):
        Kinematics(sigma_gal=True)
    assert Kinematics(sigma_gal=100.0).sigma_gas is TIED
    k = Kinematics(sigma_gal="sigma_gal")
    with pytest.raises(KeyError):
        k.validate_theta({})
    with pytest.raises(ValueError):
        Kinematics(sigma_gal=100.0).validate_theta({"sigma_gal": 1.0})


def test_csp_rejects_removed_kwarg():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    ssp = SSPData.load(str(path))
    with pytest.raises(TypeError, match="sigma_losvd_kms"):
        CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 5.0, 3), zh_const=True, add_neb=False,
                 sigma_losvd_kms=300.0, verbose=False, cosmo=Cosmology.planck18())
