"""logmass needs only a prior: SedModel samples it from the prior's median when free_param_init
does not give it a start value; a free_param_init value still wins."""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import require_test_grid                                # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                # noqa: E402
from ceridwen.observation import Photometry                                # noqa: E402
from ceridwen.priors import Uniform                                        # noqa: E402


@pytest.fixture(scope="module")
def csp():
    return CSPBasis(SSPData.load(str(require_test_grid())), lookback_time=jnp.linspace(0, 1, 4),
                    zh_const=True, add_neb=False, verbose=False, cosmo=Cosmology.planck18())


def _model(csp, **kw):
    phot = Photometry(filters=["sdss_g0"], flux=[1e-9], uncertainty=[1e-10], name="p")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, [phot], zred=0.5, **kw)


def test_a_prior_alone_samples_logmass_from_its_median(csp):
    m = _model(csp, priors={"logmass": Uniform(low=8.0, high=12.0)})
    assert "logmass" in m.param_names
    np.testing.assert_array_equal(np.asarray(m.theta_init["logmass"]), [10.0])


def test_free_param_init_still_wins(csp):
    m = _model(csp, priors={"logmass": Uniform(low=8.0, high=12.0)},
               free_param_init={"logmass": jnp.array([9.3])})
    np.testing.assert_array_equal(np.asarray(m.theta_init["logmass"]), [9.3])
    assert m.param_names.count("logmass") == 1


def test_no_prior_no_logmass(csp):
    assert "logmass" not in _model(csp).param_names
