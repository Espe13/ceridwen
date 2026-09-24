"""B1-002: a sampled zred (track_zred_age=False) whose prior puts the whole redshift range
after the oldest SFH node warns (the fixed-zred case raises)."""
import pathlib
import sys
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid  # noqa: E402

GRID = find_test_grid()


@pytest.mark.skipif(GRID is None, reason="no test SSP grid")
def test_sampled_zred_grid_older_than_universe_warns():
    from ceridwen import CSPBasis, Cosmology, SedModel, SSPData
    from ceridwen.observation import Photometry
    from ceridwen.priors import Uniform
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(GRID)), lookback_time=np.linspace(0.0, float(cosmo.age(0.0)), 6),
                   zh_const=True, add_neb=False, add_dust=False, add_diffuse_dust=False,
                   verbose=False, cosmo=cosmo)

    def build(lo, hi):
        p = Photometry(filters=["sdss_g0", "sdss_r0"], flux=np.ones(2), uncertainty=np.ones(2),
                       name="p")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SedModel(csp, [p], free_param_init={"zred": jnp.array([0.5 * (lo + hi)])},
                     priors={"zred": Uniform(low=lo, high=hi)})
        return [str(x.message) for x in w if "predates the Universe" in str(x.message)]

    assert any("at every zred" in m for m in build(4.0, 6.0))
    assert build(0.0, 0.0001) == []
