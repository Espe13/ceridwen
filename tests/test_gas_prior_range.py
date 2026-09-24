"""T2-002 / T4-002: a gas_logz / gas_logu prior whose support leaves the CLOUDY axis of the grid
in use, or a fixed (constant-transform) value outside it, raises at SedModel construction, like
the logzsol guard; for both nebular grids (ZAU_WD and ZAU_ND)."""
from __future__ import annotations

import os
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

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                # noqa: E402
from ceridwen.observation import Photometry                                # noqa: E402
from ceridwen.sampler.priors import Uniform, Normal                        # noqa: E402


@pytest.fixture(scope="module", params=[True, False], ids=["ZAU_WD", "ZAU_ND"])
def csp(request):
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    c = CSPBasis(SSPData.load(str(path)), lookback_time=np.linspace(0.0, 8.0, 5), zh_const=True,
                 add_dust=False, verbose=False, cosmo=Cosmology.planck18(),
                 init_neb_params={"cloudy_dust": request.param})
    assert c.neb.line_file.name.startswith("ZAU_WD" if request.param else "ZAU_ND")
    return c


def _model(csp, priors=None, transforms=None):
    ph = Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1e-9, 1e-9],
                    uncertainty=[1e-10, 1e-10], name="p")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, [ph], priors=priors or {}, transforms=transforms or {}, zred=0.5)


def test_gas_logz_prior_outside_the_axis_raises(csp):
    with pytest.raises(ValueError, match=r"gas_logz.*\[-2\.000, \+0\.500\].*\[-1\.300, \+0\.300\]"):
        _model(csp, {"gas_logz": Uniform(low=-2.0, high=0.5)})


def test_gas_logz_prior_inside_the_axis_passes(csp):
    _model(csp, {"gas_logz": Uniform(low=-1.3, high=0.3)})


def test_gas_logu_prior_outside_the_axis_raises(csp):
    with pytest.raises(ValueError, match=r"gas_logu.*\[-4\.000, -1\.000\]"):
        _model(csp, {"gas_logu": Uniform(low=-4.5, high=-1.0)})
    _model(csp, {"gas_logu": Uniform(low=-4.0, high=-1.0)})


def test_fixed_gas_values_outside_the_axis_raise(csp):
    with pytest.raises(ValueError, match="transform for 'gas_logz'"):
        _model(csp, transforms={"gas_logz": lambda th: jnp.array([-1.5])})
    with pytest.raises(ValueError, match="transform for 'gas_logu'"):
        _model(csp, transforms={"gas_logu": lambda th: jnp.array([-0.5])})
    _model(csp, transforms={"gas_logz": lambda th: jnp.array([-1.3]),
                            "gas_logu": lambda th: jnp.array([-1.0])})


def test_unbounded_gas_prior_warns(csp):
    ph = Photometry(filters=["sdss_g0"], flux=[1e-9], uncertainty=[1e-10], name="p")
    with pytest.warns(UserWarning, match="prior on 'gas_logz' is unbounded"):
        SedModel(csp, [ph], priors={"gas_logz": Normal(mean=0.0, sigma=0.3)}, zred=0.5)
