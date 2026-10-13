"""agn_eline_sigma as a free parameter (session SIGMA): setup validation and the
unresolved-width warning (``SedModel._check_agn_eline_sigma_range`` /
``_check_agn_eline_sigma_resolved``, ``ceridwen/model/model.py``) and its gradients.

R1  a prior, a constant transform or a fixed value of agn_eline_sigma must lie in
    (0, AGN_ELINE_SIGMA_MAX = 2000] km/s (the Kinematics.sigma_max default); otherwise
    ValueError at setup (the upper-edge message names the missing BLR component).
W1  a sampled agn_eline_sigma (a prior on it) warns at setup when no Spectrum can resolve the
    AGN lines: no Spectrum, no unmasked pixel at any AGN line, or the instrumental sigma at
    every covered AGN line > 3 x the prior's upper edge.  No warning otherwise.
G1  jax.grad of a log-likelihood is finite near the lower edge (1e-3 .. 50 km/s) in both the
    corrected and the prospector mode, and matches central differences where it is nonzero.
    (Prospector's painter uses pixels within 5 agn_eline_sigma / sigma_gas nebular widths of a
    line, so below ~1 km/s it paints nothing there and the gradient is exactly 0.)
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
from _f32bounds import B8   # noqa: E402  (float32 bounds)

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


# ------------------------------------------------------------------ W1
def test_W1_no_spectrum_warns(csp):
    _, phot, lines = T._obs()
    for obs in ([phot], [phot, lines]):
        _, w = _sed(csp, obs, priors={KEY: Uniform(low=50.0, high=500.0)})
        assert len(w) == 1 and "no Spectrum" in w[0] and "docs/agn.md" in w[0]


def test_W1_spectrum_without_agn_lines_warns(csp):
    """Every pixel within 20 A of an AGN line masked (R = 1000: 2 sigma_inst ~ 5 A), the rest
    used: as good as no spectrum for the width."""
    spec, phot, _ = T._obs()
    obs_l = np.asarray(csp._agn.model.lines.line_wave) * (1 + T.ZRED)
    w = T.WAVE_OBS
    mask = np.all(np.abs(w[:, None] - obs_l[None, :]) > 20.0, axis=1)
    assert 0 < mask.sum() < w.size
    s = Spectrum(wavelength=w, flux=np.ones_like(w), uncertainty=np.ones_like(w),
                 instrument=Instrument.R_fwhm(T.R_FWHM), name="spec", mask=mask)
    _, got = _sed(csp, [s, phot], priors={KEY: Uniform(low=50.0, high=500.0)})
    assert len(got) == 1 and "no Spectrum has unmasked pixels at any AGN line" in got[0]
    _, got = _sed(csp, [spec, phot], priors={KEY: Uniform(low=50.0, high=500.0)})
    assert not got                                                    # unmasked: quiet


def test_W1_prism_threshold(csp):
    """R_FWHM = 100: sigma_inst = c / (2.3548 R) = 1273 km/s.  Unresolvable for an upper edge
    below 1273 / 3 = 424 km/s (warns), not for 500 (1273 < 1500: no warning)."""
    _, phot, _ = T._obs()
    _, w = _sed(csp, [_prism(), phot], priors={KEY: Uniform(low=50.0, high=300.0)})
    assert len(w) == 1 and "1273 km/s" in w[0] and "'prism'" in w[0]
    _, w = _sed(csp, [_prism(), phot], priors={KEY: Uniform(low=50.0, high=500.0)})
    assert not w


def test_W1_grating_and_fixed_quiet(csp):
    spec, phot, _ = T._obs()                                          # R_FWHM = 1000
    _, w = _sed(csp, [spec, phot], priors={KEY: Uniform(low=50.0, high=150.0)})
    assert not w
    _, w = _sed(csp, [phot], transforms={KEY: lambda th: jnp.array([100.0])})
    assert not w                                                      # fixed: not sampled


def test_W1_one_resolving_spectrum_is_enough(csp):
    spec, phot, _ = T._obs()
    _, w = _sed(csp, [_prism(), spec, phot], priors={KEY: Uniform(low=50.0, high=300.0)})
    assert not w


# ------------------------------------------------------------------ G1
@pytest.mark.parametrize("mode", ["corrected", "prospector"])
def test_G1_grad_finite_near_lower_edge(mode):
    model, _, _, _ = TI._model(TI._csp(TI._cfg(mode), igm=True))
    f, x0, pos = TI._lnlike_factory(model, mode)
    vg = jax.jit(jax.value_and_grad(f))
    i = pos[KEY]
    for s in (1e-3, 1.0, 5.0, 50.0):
        x = x0.copy()
        x[i] = s
        v, g = vg(jnp.asarray(x))
        g = np.asarray(g)
        assert np.isfinite(float(v)) and np.all(np.isfinite(g)), (mode, s)
        # h-scan on the float32 model (2026-10-13): corrected agrees to <= 1.1e-4 at h = 0.5;
        # prospector has a kink at s = 5 (the lines enter the 5-width window) that h >= 0.5
        # straddles, so h = 0.01 s there (4.6e-3 at s = 5, 5.8e-6 at s = 50). Was h = 1e-3 s.
        h = min(0.5, (0.5 if mode == "corrected" else 0.01) * s)
        e = np.zeros_like(x)
        e[i] = h
        fd = (float(f(jnp.asarray(x + e))) - float(f(jnp.asarray(x - e)))) / (2 * h)
        if mode == "corrected" or s >= 5.0:
            assert g[i] != 0.0
            # rel 2e-3 as before, plus the float32 noise floor of the gradient, B8 |lnL| (the
            # forward model rounds lnL at that level; measured |dg| from float32: 4.7e-5 at
            # s = 1e-3, where every central difference is 0, and 8.6e-5 at prospector s = 5)
            assert abs(g[i] - fd) <= 2e-3 * abs(fd) + B8 * abs(float(v)), (mode, s, g[i], fd)
        else:                     # prospector: no pixel within 5 s / sigma_gas widths
            assert abs(g[i]) <= 1e-10 * max(1.0, abs(float(v))) and abs(fd) <= 1e-6
