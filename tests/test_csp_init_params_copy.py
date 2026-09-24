"""B1-015: CSPBasis does not mutate the caller's init_neb_params / init_dust_params."""
import os
import pathlib
import sys

import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid  # noqa: E402

GRID = find_test_grid()


@pytest.mark.skipif(GRID is None or not os.environ.get("SPS_HOME"),
                    reason="needs the test grid and $SPS_HOME (nebular model)")
def test_init_params_dicts_are_not_mutated():
    from ceridwen import CSPBasis, Cosmology, SSPData
    ssp = SSPData.load(str(GRID))
    neb = {"cloudy_dust": True}
    dust = {"bin_edges": [(-jnp.inf, -1.97)], "laws": ["powerlaw"]}
    for _ in range(2):
        CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 10.0, 5), zh_const=True, add_neb=True,
                 add_dust=True, add_diffuse_dust=False, verbose=False,
                 cosmo=Cosmology.planck18(), init_neb_params=neb, init_dust_params=dust)
    assert neb == {"cloudy_dust": True}
    assert list(dust) == ["bin_edges", "laws"]
