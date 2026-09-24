"""B1-011: an afe prior / constant transform outside the [alpha/Fe] grid is refused, like
logzsol (it was accepted silently and the interpolation clamped)."""
import types
import warnings

import numpy as np
import pytest

from ceridwen.model.model import SedModel
from ceridwen.priors import Normal, Uniform

GRID = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])


def _fake(priors, fixed=None):
    return types.SimpleNamespace(csp=types.SimpleNamespace(afe_grid=GRID), priors=priors,
                                 _transform_value=lambda name: fixed)


def test_wide_bounded_prior_raises():
    with pytest.raises(ValueError, match=r"'afe' covers \[-0.800, \+0.400\]"):
        SedModel._check_afe_range(_fake({"afe": Uniform(low=-0.8, high=0.4)}))


def test_prior_on_the_grid_and_no_alpha_grid_pass():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SedModel._check_afe_range(_fake({"afe": Uniform(low=-0.2, high=0.6)}))
        SedModel._check_afe_range(types.SimpleNamespace(csp=types.SimpleNamespace(),
                                                        priors={}))


def test_unbounded_prior_warns_and_fixed_value_outside_raises():
    with pytest.warns(UserWarning, match="'afe' is unbounded"):
        SedModel._check_afe_range(_fake({"afe": Normal(mean=0.2, sigma=0.3)}))
    with pytest.raises(ValueError, match="transform for 'afe'"):
        SedModel._check_afe_range(_fake({}, fixed=np.array([0.9])))


def test_eline_scaling_without_lines_is_refused():
    """B1-014: eline_scaling scales only a Lines prediction; sampled with observations but
    no Lines it was a silent prior-volume nuisance."""
    def fake(kinds):
        return types.SimpleNamespace(
            param_names=["logmass", "eline_scaling"], transforms={},
            observations=[types.SimpleNamespace(kind=k, name=k) for k in kinds])
    with pytest.raises(ValueError, match="no observation is a Lines"):
        SedModel._check_calibration_names(fake(["photometry", "spectrum"]))
    SedModel._check_calibration_names(fake(["photometry", "lines"]))
    SedModel._check_calibration_names(fake([]))          # predict-only model


def test_tied_gas_prior_outside_the_cloudy_axis_warns():
    """B1-019: gas_tied with a logzsol prior reaching below the CLOUDY axis froze the lines
    silently."""
    def fake(prior):
        csp = types.SimpleNamespace(gas_tied=True,
                                    neb=types.SimpleNamespace(nebem_logz=np.linspace(-1.3, 0.3, 5)))
        return types.SimpleNamespace(csp=csp, priors={"logzsol": prior},
                                     _transform_value=lambda n: None)
    with pytest.warns(UserWarning, match="leaves the CLOUDY gas axis"):
        SedModel._check_tied_gas_range(fake(Uniform(low=-2.3, high=0.3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SedModel._check_tied_gas_range(fake(Uniform(low=-1.3, high=0.3)))
