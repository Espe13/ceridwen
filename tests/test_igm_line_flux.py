"""B1-018: the IGM transmission of a line flux (Lines, and the lines the Spectrum projector paints)
is the transmission averaged over the line's painted profile, so it equals what the painted
path (lines on the model grid, then the IGM) gives.  At Ly-alpha, where Madau (1995) jumps
from absorbed to 1, the old line-centre interpolation depended on the grid's pixel phase."""
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

from ceridwen import SSPData, CSPBasis, Cosmology                          # noqa: E402

C = 2.99792458e18


@pytest.fixture(scope="module")
def csp():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    return CSPBasis(SSPData.load(str(path)), lookback_time=np.linspace(0, 0.5, 6), zh_const=True,
                    add_neb=True, add_igm=True, add_dust=False, add_diffuse_dust=False,
                    verbose=False, cosmo=Cosmology.planck18())


@pytest.mark.parametrize("z", [3.0, 6.0, 9.0])
def test_lyman_alpha_line_flux_matches_painted_path(csp, z):
    pos = np.asarray(csp.neb.nebem_line_pos)
    w = np.asarray(csp.wave)
    k = int(np.argmin(np.abs(pos - 1215.67)))
    sig = float(np.asarray(csp.neb.dlam_lines)[k])
    th = dict(csp.theta_init, zred=jnp.array([z]), logmass=jnp.array([9.0]))
    th0 = dict(th, igm_factor=jnp.array([0.0]))                  # no IGM, same everything else
    win = np.abs(w - pos[k]) < 5 * sig
    near = pos[np.abs(pos - pos[k]) < 10 * sig]                  # grid lines sharing the window
    nu = C / w
    ls, ls0 = (np.asarray(csp.get_line_spec(t), float) for t in (th, th0))
    painted = np.trapezoid(ls[win], nu[win]) / np.trapezoid(ls0[win], nu[win])
    rows = [int(np.argmin(np.abs(pos - p))) for p in near]
    for spec in (False, True):
        F = np.asarray(csp.predict_line_fluxes(th, for_spectrum=spec), float)[rows]
        F0 = np.asarray(csp.predict_line_fluxes(th0, for_spectrum=spec), float)[rows]
        # the painted window holds every line in `near`: compare the summed flux.  3e-6: the
        # painted lines are full - continuum of float32 spectra (measured 1.2e-6 at z = 9)
        assert F.sum() / F0.sum() == pytest.approx(painted, rel=3e-6, abs=0)
    # and Ly-alpha is really absorbed on its blue half: between the red limit 1 and the
    # blue-side transmission
    T = np.asarray(csp._igm_transmission(jnp.asarray(z), th), float)
    r = float(csp.predict_line_fluxes(th)[k] / csp.predict_line_fluxes(th0)[k])
    assert T[np.searchsorted(w, pos[k]) - 1] < r < 1.0
