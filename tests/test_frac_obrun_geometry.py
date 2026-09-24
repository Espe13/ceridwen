"""The ``frac_obrun`` escape channel: ``frac_obrun = 0`` is the model without the key on every
spectrum variant (runaway_bc and picket, with and without nebular and dust emission, and
CSPBasis_afe), and the picket geometry does what it is defined to do."""
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
from _gridfixture import find_test_grid, TEST_DATA_DIR                    # noqa: E402

from ceridwen import SSPData, CSPBasis, Cosmology                          # noqa: E402


def _ssp():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    return SSPData.load(str(path))


def _needs_sps_home():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY / dust-emission grids)")


def _theta(csp, **extra):
    th = dict(csp.theta_init)
    for k in [k for k in th if k.startswith("tau_pow")]:
        th[k] = jnp.array([1.0])
    if "dust2" in th:
        th["dust2"] = jnp.array([0.3])
    th.update(extra)
    return th


VARIANTS = [   # (add_neb, add_dust_emission, add_diffuse_dust)
    (True, False, True), (True, True, True), (True, False, False),
    (False, False, True), (False, True, True),
]


@pytest.mark.parametrize("geometry,neb,dem,diffuse",
                         [("runaway_bc", *v) for v in VARIANTS]
                         + [("picket", *v) for v in VARIANTS if v[0]])   # picket needs add_neb
def test_frac_obrun_zero_equals_absent(geometry, neb, dem, diffuse):
    """B1-013: fo = 0 must be exactly the model without frac_obrun (the young nebular continuum
    below 912 A kept its birth-cloud attenuation only without the key).  runaway_bc runs the same
    float32 operations with and without the key (1e-12); the picket path is a different but
    algebraically equal contraction at fo = 0, so it agrees to float32 rounding."""
    ssp = _ssp()
    if neb or dem:
        _needs_sps_home()
    csp = CSPBasis(ssp, lookback_time=np.linspace(0, 1, 6), zh_const=True, add_neb=neb,
                   add_dust_emission=dem, add_diffuse_dust=diffuse, fesc_geometry=geometry,
                   verbose=False, cosmo=Cosmology.planck18())
    th = _theta(csp)
    exact = geometry == "runaway_bc" or not neb
    for inc in (False, True):
        a = np.asarray(csp.get_spectrum(th, include_lines=inc), float)
        b = np.asarray(csp.get_spectrum({**th, "frac_obrun": jnp.array([0.0])},
                                        include_lines=inc), float)
        if exact:
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=0.0)
        else:
            np.testing.assert_allclose(b, a, rtol=2e-5, atol=1e-12 * np.max(a))
    if neb:
        la = np.asarray(csp.predict_line_fluxes(th), float)
        lb = np.asarray(csp.predict_line_fluxes({**th, "frac_obrun": jnp.array([0.0])}), float)
        np.testing.assert_allclose(lb, la, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("dem", [False, True])
def test_frac_obrun_zero_equals_absent_afe(dem):
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe
    from ceridwen.csp.csp_afe import CSPBasis_afe
    path = TEST_DATA_DIR / "amist_c3k_lr_chab_afe.h5"
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    if dem:
        _needs_sps_home()
    csp = CSPBasis_afe(SSPDataAfe.load(str(path)), lookback_time=np.linspace(0, 1, 6),
                       zh_const=True, add_dust_emission=dem, verbose=False,
                       cosmo=Cosmology.planck18())
    th = _theta(csp)
    a = np.asarray(csp.get_spectrum(th), float)
    b = np.asarray(csp.get_spectrum({**th, "frac_obrun": jnp.array([0.0])}), float)
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=0.0)


def test_picket_needs_nebular_model():
    with pytest.raises(ValueError, match="picket.*add_neb=True"):
        CSPBasis(_ssp(), lookback_time=np.linspace(0, 1, 6), zh_const=True, add_neb=False,
                 fesc_geometry="picket", verbose=False, cosmo=Cosmology.planck18())


# --------------------------------------------------------------------------- picket definition
# P1-011: the fraction frac_obrun of the young light is not attenuated by any dust or gas.  A
# population formed in the last 3 Myr only (all its SSP ages are inside both the birth-cloud
# bin and the nebular age range) makes each point an exact linear statement.

FO = 0.3


def _young_picket(dem=False):
    _needs_sps_home()
    csp = CSPBasis(_ssp(), theta={"lookback_time": jnp.array([0.0, 0.003, 0.5, 1.0]),
                                  "sfh": jnp.array([1.0, 0.0, 0.0]),
                                  "logzsol": jnp.array([0.0])},
                   zh_const=True, add_neb=True, add_dust=True, add_diffuse_dust=True,
                   add_dust_emission=dem, fesc_geometry="picket", verbose=False,
                   cosmo=Cosmology.planck18())
    W = np.asarray(csp.calculate_ssp_weights(csp.theta_init))
    la = np.asarray(csp.ssp_ages_lgyr)
    bc = np.asarray(csp._age_bin_mix).sum(1) > 0
    used = W.sum(0) > 1e-12 * W.sum(0).max()     # the SFR floor (1e-30) leaves tiny old weights
    assert used.any() and np.all(bc[used]) and np.all(np.asarray(csp.young_mask)[used]), la[used]
    return csp


def _dust(th, tau_bc, tau_diff):
    th = dict(th)
    th["tau_pow"] = jnp.array([tau_bc])
    th["diffuse_tau_kc"] = jnp.array([tau_diff])
    return th


def _intrinsic_stellar(csp, th):
    W = csp.calculate_ssp_weights(th).astype(jnp.float32)
    return np.asarray(jnp.einsum("za,zaw->w", W, csp.flux), float)


@pytest.mark.parametrize("tau_bc,tau_diff", [(300.0, 0.0), (0.0, 300.0), (300.0, 300.0)])
def test_picket_clear_fraction_skips_birth_cloud_and_diffuse_dust(tau_bc, tau_diff):
    """Opaque birth cloud and/or opaque diffuse dust: only the clear channel is left, and it is
    frac_obrun times the intrinsic young stellar light (LyC included), unattenuated."""
    csp = _young_picket()
    th = _dust({**csp.theta_init, "frac_obrun": jnp.array([FO])}, tau_bc, tau_diff)
    w = np.asarray(csp.wave)
    opaque = w < 9000.0                   # both laws have tau > 50 here at tau = 300
    got = np.asarray(csp.get_spectrum(th, include_lines=True), float)
    want = FO * _intrinsic_stellar(csp, th)
    np.testing.assert_allclose(got[opaque], want[opaque], rtol=2e-5,
                               atol=1e-6 * np.max(want[opaque]))


def test_picket_nebular_scales_with_covered_fraction_and_lyc_escapes():
    """No dust: S(fo) - (1 - fo) S(0) = fo x intrinsic stellar light at every wavelength, i.e. the
    nebular emission scales with 1 - fo and a fraction f_esc = fo of the LyC escapes; with dust,
    the nebular lines alone scale with 1 - fo."""
    csp = _young_picket()
    th0 = _dust(csp.theta_init, 0.0, 0.0)
    s0 = np.asarray(csp.get_spectrum({**th0, "frac_obrun": jnp.array([0.0])}, include_lines=True), float)
    sf = np.asarray(csp.get_spectrum({**th0, "frac_obrun": jnp.array([FO])}, include_lines=True), float)
    C = _intrinsic_stellar(csp, th0)
    np.testing.assert_allclose(sf, (1 - FO) * s0 + FO * C, rtol=1e-5, atol=1e-9 * np.max(sf))
    lyc = np.asarray(csp.wave) < 912.0
    esc = (sf - (1 - FO) * s0)[lyc].sum() / C[lyc].sum()
    assert abs(esc - FO) < 1e-4
    thd = _dust(csp.theta_init, 1.0, 0.5)
    L = {f: np.asarray(csp.get_spectrum({**thd, "frac_obrun": jnp.array([f])}, include_lines=True)
                       - csp.get_spectrum({**thd, "frac_obrun": jnp.array([f])}, include_lines=False),
                       float) for f in (0.0, FO)}
    np.testing.assert_allclose(L[FO], (1 - FO) * L[0.0], rtol=1e-4, atol=1e-6 * np.max(np.abs(L[0.0])))
    lf = {f: np.asarray(csp.predict_line_fluxes({**thd, "frac_obrun": jnp.array([f])}), float)
          for f in (0.0, FO)}
    np.testing.assert_allclose(lf[FO], (1 - FO) * lf[0.0], rtol=1e-6)


def test_picket_energy_balance_excludes_clear_light():
    """The dust emission is powered by the covered channel only: for the young population the
    re-emitted spectrum at frac_obrun = fo is (1 - fo) times that at fo = 0."""
    dem = _young_picket(dem=True)
    nodem = _young_picket(dem=False)
    th = _dust(dem.theta_init, 1.0, 0.5)
    E = {}
    for f in (0.0, FO):
        t = {**th, "frac_obrun": jnp.array([f])}
        E[f] = (np.asarray(dem.get_spectrum(t, include_lines=True), float)
                - np.asarray(nodem.get_spectrum(t, include_lines=True), float))
    ir = np.asarray(dem.wave) > 5e4
    np.testing.assert_allclose(E[FO][ir], (1 - FO) * E[0.0][ir], rtol=1e-4)
