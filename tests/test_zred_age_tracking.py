"""Free-redshift SFH age-grid tracking (``CSPBasis(track_zred_age=True)``).

Proves the differentiable call path
    CSPBasis.predict -> calculate_ssp_weights -> _ssp_weights
        -> _lookback_from_zred(theta["zred"]) -> ceridwen.cosmology.age_gyr
so that when ``zred`` is a sampled parameter the SFH lookback grid (its oldest
node / effective age of the universe) is recomputed from the cosmological age
at that redshift inside the JIT-compiled forward pass -- not held fixed at a
single reference redshift.

The three checks the reviewer asked for:
  1. evaluating the model at two redshifts gives a DIFFERENT oldest-bin age,
     each equal to ``age_gyr(z)`` to tolerance;
  2. the rest-frame SSP-weighted spectrum (no flux factor, no IGM, no
     observed-frame projection) changes with ``zred`` -- i.e. the dependence
     is through the AGE GRID, not merely the flux factor;
  3. ``jax.grad`` of a grid-dependent scalar w.r.t. ``zred`` is finite and
     non-zero (the gradient flows through the age-grid construction).

It also checks that ``track_zred_age=False`` (the default) is bit-for-bit the
fixed-grid path.
"""
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.cosmology import age_gyr

from _gridfixture import require_test_grid

_SSP = str(require_test_grid())
_NB = 8


def _build_csp(track):
    ssp = SSPData.load(_SSP)
    lb_build = np.linspace(0.0, float(age_gyr(2.0)), _NB)        # Gyr, oldest~age(2)
    theta = {"lookback_time": jnp.asarray(lb_build),
             "sfh": jnp.ones(_NB), "Z": jnp.array([-2.0])}
    return CSPBasis(ssp, theta=theta, tuniv=13.8, zh_const=True,
                    sfh_interp="step", add_dust=False, add_diffuse_dust=False,
                    add_neb=False, add_igm=False, track_zred_age=track,
                    verbose=False)


def _rest_spectrum(csp, z):
    th = {"sfh": jnp.ones(_NB), "Z": jnp.array([-2.0]),
          "logmass": jnp.array([10.0]), "zred": jnp.array([float(z)])}
    return csp.get_spectrum(theta=th)


def test_oldest_bin_tracks_age_gyr():
    csp = _build_csp(track=True)
    ages = {}
    for z in (2.0, 10.0):
        eff_yr = np.asarray(csp._lookback_from_zred(jnp.array([z])))
        ages[z] = eff_yr[-1] / 1e9
        assert ages[z] == pytest.approx(float(age_gyr(z)), abs=1e-6)
    # The two redshifts must give a genuinely different age grid.
    assert abs(ages[2.0] - ages[10.0]) > 1.0          # Gyr (3.28 vs 0.47)


def test_grid_changes_restframe_spectrum():
    csp = _build_csp(track=True)
    s2 = np.asarray(_rest_spectrum(csp, 2.0))
    s10 = np.asarray(_rest_spectrum(csp, 10.0))
    # Rest-frame, grid-only dependence: no flux factor / IGM applied here.
    assert np.max(np.abs(s2 - s10)) > 0.0
    assert np.all(np.isfinite(s2)) and np.all(np.isfinite(s10))


def test_grad_flows_through_age_grid():
    csp = _build_csp(track=True)

    def scalar(z):
        th = {"sfh": jnp.ones(_NB), "Z": jnp.array([-2.0]),
              "logmass": jnp.array([10.0]), "zred": z}
        return jnp.sum(csp.get_spectrum(theta=th))     # zred enters ONLY via grid

    g = float(jax.grad(lambda z: scalar(jnp.array([z])))(7.0))
    assert np.isfinite(g) and abs(g) > 0.0


def test_fixed_z_path_unchanged():
    """track_zred_age=False must ignore zred and use the static grid."""
    csp = _build_csp(track=False)
    s2 = np.asarray(_rest_spectrum(csp, 2.0))
    s10 = np.asarray(_rest_spectrum(csp, 10.0))
    # No zred dependence through the grid -> identical rest-frame spectra.
    np.testing.assert_allclose(s2, s10, rtol=0, atol=0)


if __name__ == "__main__":
    test_oldest_bin_tracks_age_gyr()
    test_grid_changes_restframe_spectrum()
    test_grad_flows_through_age_grid()
    test_fixed_z_path_unchanged()
    print("all zred age-tracking tests passed")
