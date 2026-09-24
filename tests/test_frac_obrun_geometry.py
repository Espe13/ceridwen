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


@pytest.mark.parametrize("geometry", ["runaway_bc", "picket"])
@pytest.mark.parametrize("neb,dem,diffuse", VARIANTS)
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
