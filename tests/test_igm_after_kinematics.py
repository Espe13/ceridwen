"""B1-027: the IGM acts after the galaxy's kinematic broadening and before the instrument LSF
(galaxy kinematics, rest frame -> redshift -> IGM -> instrument), so the Ly-alpha break is as
sharp as the IGM model and the LSF make it, not smeared by sigma_gal.  Checked against a direct
construction with numpy (explicit Gaussian sums, not the FFT or the banded response)."""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid                                   # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology, Kinematics, Instrument  # noqa: E402
from ceridwen.observation import Spectrum, Photometry                      # noqa: E402

Z = 6.0
CKMS = 299792.458
# prism-like widths: the test grid (BPASS, 2 A pixels = 450 km/s at Ly-alpha, library sigma ~400
# km/s) resolves them; a 111 km/s instrument would be below the library there
SIG_GAL, SIG_INST = 1200.0, 1000.0


@pytest.fixture(scope="module")
def csp():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    cosmo = Cosmology.planck18()
    return CSPBasis(SSPData.load(str(path)), lookback_time=np.linspace(0, float(cosmo.age(Z)), 5),
                    zh_const=True, add_neb=False, add_dust=False, add_diffuse_dust=False,
                    add_igm=True, verbose=False, cosmo=cosmo)


def _theta(csp):
    return dict(csp.theta_init, logmass=jnp.array([9.0]), zred=jnp.array([Z]))


def _gauss_smooth(y, sigma_px):
    """Direct Gaussian sum on a uniform grid, edges continued with the edge values."""
    h = int(np.ceil(8 * sigma_px))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma_px) ** 2)
    k /= k.sum()
    yp = np.concatenate([np.full(h, y[0]), y, np.full(h, y[-1])])
    return np.convolve(yp, k, mode="valid")


def test_spectrum_break_is_igm_plus_lsf_only(csp):
    wave_obs = np.geomspace(1150.0 * (1 + Z), 1300.0 * (1 + Z), 300)
    spec = Spectrum(wavelength=wave_obs, instrument=Instrument.sigma_kms(SIG_INST), name="s")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SedModel(csp, [spec], zred=Z, kinematics=Kinematics(sigma_gal=SIG_GAL))
    th = _theta(csp)
    pred = np.asarray(m.predict(th)["s"], float)

    # direct construction on the projector's own log grid
    proj = spec._proj
    g = proj.window.grid
    S0 = np.asarray(csp.get_spectrum(th), float) * 10.0 ** 9.0 * float(csp._flux_factor(th))
    s_log = np.interp(g.wave, np.asarray(csp.wave), S0)
    # the IGM model on the model grid, read onto the log grid linearly (as the spectrum)
    T = np.interp(g.wave, np.asarray(csp.wave),
                  np.asarray(csp.igm.attenuation(csp.wave, Z, factor=1.0), float))
    dln = g.dv / CKMS
    x_obs = (np.log(wave_obs) - np.log(1 + Z) - g.lnw[0]) / dln
    s_fix = np.asarray(proj.sigma_fix_kms, float) / g.dv          # instrument (library removed)
    j = np.arange(g.n)
    R = np.exp(-0.5 * ((j[None, :] - x_obs[:, None]) / s_fix[:, None]) ** 2)
    R /= R.sum(axis=1, keepdims=True)
    physical = R @ (_gauss_smooth(s_log, SIG_GAL / g.dv) * T)     # kinematics -> IGM -> LSF
    smeared = R @ _gauss_smooth(s_log * T, SIG_GAL / g.dv)        # the old order
    lsf_only = R @ (s_log * T)                                    # no galaxy kernel at all

    top = np.max(physical)
    np.testing.assert_allclose(pred, physical, rtol=2e-4, atol=1e-6 * top)
    # the old order differs visibly near the break (7% of the peak at these widths) ...
    assert np.max(np.abs(smeared - physical)) > 0.03 * top
    # ... and the break is exactly as sharp as the IGM + LSF: across the break the model
    # rises over the same pixels as lsf_only (the galaxy kernel only smooths the continuum)
    blue = wave_obs < 1210.0 * (1 + Z)
    red = wave_obs > 1222.0 * (1 + Z)
    edge = ~blue & ~red
    jump = lambda y: np.max(np.abs(np.diff(y[edge])))              # noqa: E731
    assert jump(pred) == pytest.approx(jump(physical), rel=1e-3)
    print(f"steepest step: model {jump(pred):.4e}, physical {jump(physical):.4e}, "
          f"old order {jump(smeared):.4e}, IGM + LSF only {jump(lsf_only):.4e}")
    assert jump(pred) > 1.3 * jump(smeared)
    assert jump(pred) == pytest.approx(jump(lsf_only), rel=0.05)


def test_broadened_photometry_applies_igm_after_sigma_gal(csp):
    filt = ["jwst_f090w", "jwst_f115w"]
    phot = Photometry(filters=filt, flux=np.ones(2), uncertainty=np.ones(2), name="p")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SedModel(csp, [phot], zred=Z, kinematics=Kinematics(sigma_gal=SIG_GAL),
                     broaden_photometry=True)
    th = _theta(csp)
    got = np.asarray(m.predict(th)["p"], float)
    S0 = csp.get_spectrum(th) * 10.0 ** 9.0 * csp._flux_factor(th)
    T = csp._igm_transmission(jnp.asarray(Z), th)
    pb = phot._broadener
    want = np.asarray(phot.predict(pb(S0, SIG_GAL) * T, csp.wave), float)
    old = np.asarray(phot.predict(pb(S0 * T, SIG_GAL), csp.wave), float)
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert np.max(np.abs(old / want - 1)) > 1e-5         # the order is visible in F090W


def test_free_z_paths_match_fixed_z(csp):
    """Sampled zred: the free-z projector reads the model at wave_log x opz_ref / opz and applies
    the IGM there; the free-z photometry broadens before the IGM too.  At the reference redshift
    the Spectrum equals the fixed-z projector (verified above) exactly, elsewhere to the
    per-mille node-phase difference of the two log grids (as tests/test_spectrum_free_z.py)."""
    from ceridwen.sampler.priors import Uniform

    def obs():
        return [Spectrum(wavelength=np.geomspace(1150.0 * (1 + Z), 1300.0 * (1 + Z), 300),
                         instrument=Instrument.sigma_kms(SIG_INST), name="s"),
                Photometry(filters=["jwst_f090w", "jwst_f115w"], flux=np.ones(2),
                           uncertainty=np.ones(2), name="p")]

    kin = Kinematics(sigma_gal=SIG_GAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = SedModel(csp, obs(), priors={"zred": Uniform(low=5.8, high=6.2)},
                        free_param_init={"zred": jnp.array([Z]), "logmass": jnp.array([9.0])},
                        kinematics=kin, broaden_photometry=True)
    for z in (Z, 5.9):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = SedModel(csp, obs(), zred=z, kinematics=kin, broaden_photometry=True,
                             free_param_init={"logmass": jnp.array([9.0])})
        base = dict(csp.theta_init, logmass=jnp.array([9.0]))
        a = free.predict(dict(base, zred=jnp.array([z])))
        b = fixed.predict(base)
        for k in ("s", "p"):
            tol = 1e-9 if (z == Z and k == "s") else 1e-2
            ra, rb = np.asarray(a[k]), np.asarray(b[k])
            assert np.max(np.abs(ra - rb)) < tol * np.max(np.abs(rb)), (k, z)
