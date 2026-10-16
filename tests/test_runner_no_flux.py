"""Fitting an observation built without ``flux=`` fails at setup with a message that says so,
not with a JAX error from inside the likelihood (``ceridwen/sampler/runner.py``)."""
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

import ceridwen  # noqa: F401  (sets the precision)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _gridfixture import find_test_grid  # noqa: E402

GRID = find_test_grid()


@pytest.mark.skipif(GRID is None, reason="no test SSP grid")
def test_fit_without_flux_names_the_observation(tmp_path):
    from ceridwen import CSPBasis, SedModel, SSPData, fitSED
    from ceridwen.cosmology import Cosmology
    from ceridwen.observation import Photometry
    from ceridwen.priors import Uniform

    ssp = SSPData.load(str(GRID))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 5.0, 3), zh_const=True,
                   sfh_interp="step", add_dust=False, add_neb=False, verbose=False,
                   cosmo=Cosmology.planck18())
    phot = Photometry(filters=["sdss_g0", "sdss_r0"], name="phot")        # no flux
    model = SedModel(csp, [phot],
                     priors={"logzsol": Uniform(low=-1.0, high=0.0),
                             "diffuse_tau_kc": Uniform(low=0.0, high=2.0),
                             "diffuse_dust_index": Uniform(low=-1.0, high=0.4)},
                     transforms={"sfh": lambda th: jnp.ones(3)},
                     zred=0.1)
    with pytest.raises(ValueError, match="'phot' has no flux"):
        fitSED(model, sampler="nested", rng_key=jax.random.PRNGKey(0),
               sampler_kwargs={"num_live": 10}, output_dir=str(tmp_path))
