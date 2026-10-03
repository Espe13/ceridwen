"""agn_eline_sigma as a free parameter (session SIGMA): setup validation
(``SedModel._check_agn_eline_sigma_range``, ``ceridwen/model/model.py``).

R1  a prior, a constant transform or a fixed value of agn_eline_sigma must lie in
    (0, AGN_ELINE_SIGMA_MAX = 2000] km/s (the Kinematics.sigma_max default); otherwise
    ValueError at setup (the upper-edge message names the missing BLR component).
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_emission_lines as T  # noqa: E402
import test_integration as TI  # noqa: E402

from ceridwen import SedModel  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics  # noqa: E402
from ceridwen.model.model import AGN_ELINE_SIGMA_MAX  # noqa: E402
from ceridwen.observation import Spectrum  # noqa: E402
from ceridwen.priors import Normal, Uniform  # noqa: E402

KEY = "agn_eline_sigma"
WARN = "agn_eline_sigma is sampled but the data cannot constrain it"


def _sed(csp, obs, priors=None, transforms=None, init=None):
    """SedModel on the emission-line test setup; returns (model, our warnings)."""
    t = np.array(csp.sfh_times)
    tr = {"sfh": lambda th, _t=t: T.logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)}
    tr.update(transforms or {})
    fi = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.0])}
    fi.update(init or {})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = SedModel(csp, list(obs), priors=priors or {}, transforms=tr, free_param_init=fi,
                     zred=T.ZRED, kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS))
    return m, [str(x.message) for x in w if WARN in str(x.message)]


def _prism():
    w = T.WAVE_OBS
    return Spectrum(wavelength=w, flux=np.ones_like(w), uncertainty=np.ones_like(w),
                    instrument=Instrument.R_fwhm(100.0), name="prism")


@pytest.fixture(scope="module", params=["corrected", "prospector"])
def csp(request):
    return TI._csp(TI._cfg(request.param))


# ------------------------------------------------------------------ R1
def test_limit_is_the_kinematics_default():
    assert AGN_ELINE_SIGMA_MAX == Kinematics(sigma_gal=1.0).sigma_max == 2000.0


@pytest.mark.parametrize("prior", [Uniform(low=0.0, high=500.0), Uniform(low=-10.0, high=500.0),
                                   Normal(mean=100.0, sigma=30.0)])
def test_R1_prior_lower_edge_refused(csp, prior):
    spec, phot, _ = T._obs()
    with pytest.raises(ValueError, match=r"must be > 0.*Uniform\(low=50.0, high=500.0\)"):
        _sed(csp, [spec, phot], priors={KEY: prior})


def test_R1_prior_upper_edge_refused(csp):
    spec, phot, _ = T._obs()
    with pytest.raises(ValueError, match=r"above 2000 km/s.*separate BLR component"):
        _sed(csp, [spec, phot], priors={KEY: Uniform(low=50.0, high=2000.5)})


@pytest.mark.parametrize("value", [0.0, -5.0, 2500.0, float("inf")])
def test_R1_fixed_value_refused(csp, value):
    spec, phot, _ = T._obs()
    with pytest.raises(ValueError, match="agn_eline_sigma"):          # constant transform
        _sed(csp, [spec, phot], transforms={KEY: lambda th, v=value: jnp.array([v])})
    if np.isfinite(value):
        with pytest.raises(ValueError, match="agn_eline_sigma"):      # initial / fixed value
            _sed(csp, [spec, phot], init={KEY: jnp.array([value])})


@pytest.mark.parametrize("prior", [Uniform(low=50.0, high=500.0), Uniform(low=1e-3, high=2000.0)])
def test_R1_accepted(csp, prior):
    spec, phot, _ = T._obs()
    m, w = _sed(csp, [spec, phot], priors={KEY: prior})
    assert KEY in m.param_names and not w
    m, w = _sed(csp, [spec, phot], transforms={KEY: lambda th: jnp.array([100.0])})
    assert KEY not in m.param_names and not w
    m, w = _sed(csp, [spec, phot])                                    # the default, 100 km/s
    assert float(m.theta_init[KEY][0]) == 100.0 and not w


def test_R1_without_agn_lines_nothing_checked():
    """AGNConfig(lines=False): agn_eline_sigma is not a parameter; the checks do not run."""
    from ceridwen.agn import AGNConfig
    c = TI._csp(AGNConfig(agn_mode="corrected", lines=False))
    _, phot, _ = T._obs()
    m, w = _sed(c, [phot])
    assert KEY not in m.param_names and not w
