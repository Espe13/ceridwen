"""CSPBasis(fixed={name: value}): parameters held fixed at construction.  A fixed value is a
constant transform: the model with ``fixed=`` must equal, byte for byte, the model with
``transforms={name: lambda th: jnp.array([value])}``, and every misuse raises at construction."""
from __future__ import annotations

import os
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

GRID = require_test_grid()
SPS = os.environ.get("SPS_HOME")
needs_sps = pytest.mark.skipif(not SPS, reason="SPS_HOME not set (CLOUDY grids)")
AFE_GRID = (pathlib.Path(__file__).resolve().parents[1] / "ceridwen" / "data" / "test_data"
            / "amist_c3k_lr_chab_afe.h5")
needs_afe = pytest.mark.skipif(not AFE_GRID.is_file(), reason=f"{AFE_GRID.name} not found")


@pytest.fixture(scope="module")
def ssp():
    return SSPData.load(str(GRID))


def _csp(ssp, **kw):
    base = dict(lookback_time=jnp.linspace(0.0, 1.0, 4), zh_const=True, verbose=False,
                cosmo=Cosmology.planck18(), add_neb=False)
    base.update(kw)
    if base["add_neb"] or base.get("add_dust_emission") or base.get("agn") is not None:
        base["sps_home"] = SPS
    return CSPBasis(ssp, **base)


def _model(csp, **kw):
    phot = Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1e-9, 1e-9],
                      uncertainty=[1e-10, 1e-10], name="phot")
    with warnings.catch_warnings():             # no priors given: "no prior for sampled ..."
        warnings.simplefilter("ignore")
        return SedModel(csp, [phot], zred=0.5, **kw)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _pair(ssp, fix, **kw):
    """The model with fixed=fix and the one with the equivalent constant transforms."""
    a = _model(_csp(ssp, fixed=fix, **kw))
    b = _model(_csp(ssp, **kw),
               transforms={k: (lambda th, _v=float(v): jnp.array([_v])) for k, v in fix.items()})
    return a, b


# ---- the CSP side ----------------------------------------------------------------------

def test_fixed_value_is_in_theta_init_not_in_param_names(ssp):
    csp = _csp(ssp, fixed={"alpha_pow": -0.7})
    assert "alpha_pow" not in csp.param_names
    assert list(csp.fixed) == ["alpha_pow"]
    np.testing.assert_array_equal(np.asarray(csp.theta_init["alpha_pow"]), [-0.7])
    # direct CSP use starts from theta_init and gets the fixed value
    ref = _csp(ssp)
    th = dict(ref.theta_init, alpha_pow=jnp.array([-0.7]))
    assert _same(csp.get_spectrum(csp.theta_init), ref.get_spectrum(th))


def test_fixed_survives_initialize_model_structure(ssp):
    """The template-copy recipe rebuilds the structure; the fixed values stay."""
    csp = _csp(ssp, fixed={"alpha_pow": -0.7})
    th = dict(csp.theta_init, lookback_time=jnp.linspace(0.0, 1.0, 4))
    csp.initialize_model_structure({**th, "alpha_pow": jnp.array([0.0])})
    np.testing.assert_array_equal(np.asarray(csp.theta_init["alpha_pow"]), [-0.7])
    assert "alpha_pow" not in csp.param_names


@pytest.mark.parametrize("fix, kw, match", [
    ({"gas_logu": -2.5}, {}, "needs add_neb=True"),
    ({"duste_qpah": 2.0}, {}, "needs add_dust_emission=True"),
    ({"diffuse_R_v_cal00": 3.1}, {}, "diffuse_law='calzetti'"),
    ({"agn_tau": 20.0}, {}, r"agn=AGNConfig"),
    ({"afe": 0.2}, {}, "CSPBasis_afe"),
    ({"logzsol_bins": [0.0] * 4}, {}, "zh_const=False"),
    ({"logzsol_hist": [0.0] * 5}, {}, "logzsol_hist.*was removed"),   # deliberate misuse
    ({"igm_factor": 0.5}, {}, r"CSPBasis\(igm_factor=\.\.\.\)"),
    ({"logmass": 10.0}, {}, "constant transform on the SedModel"),
    ({"not_a_key": 1.0}, {}, "not a parameter of this CSP"),
])
def test_unknown_or_switched_off_key_raises(ssp, fix, kw, match):
    with pytest.raises(ValueError, match=match):
        _csp(ssp, fixed=fix, **kw)


def test_unknown_key_lists_the_valid_ones(ssp):
    with pytest.raises(ValueError, match=r"\['alpha_pow', 'diffuse_dust_index'"):
        _csp(ssp, fixed={"gas_logu": -2.5})


@pytest.mark.parametrize("fix, match", [
    ({"alpha_pow": [-0.7, -0.5]}, r"shape \(2,\), but 'alpha_pow' has shape \(1,\)"),
    ({"sfh": [1.0, 1.0]}, r"shape \(2,\), but 'sfh' has shape \(3,\)"),   # 3 step bins
    ({"alpha_pow": float("nan")}, "not finite"),
])
def test_bad_value_raises(ssp, fix, match):
    with pytest.raises(ValueError, match=match):
        _csp(ssp, fixed=fix)


def test_fixed_must_be_a_dict(ssp):
    with pytest.raises(TypeError, match="must be a dict"):
        _csp(ssp, fixed=[("alpha_pow", -0.7)])


@needs_sps
def test_gas_logz_with_gas_tied_raises(ssp):
    with pytest.raises(ValueError, match="gas_tied=True"):
        _csp(ssp, add_neb=True, gas_tied=True, fixed={"gas_logz": -0.3})


# ---- the SedModel side -----------------------------------------------------------------

@pytest.mark.parametrize("where", ["priors", "transforms", "free_param_init"])
def test_fixed_and_given_again_raises(ssp, where):
    csp = _csp(ssp, fixed={"alpha_pow": -0.7})
    given = {"priors": Uniform(low=-2.0, high=0.0),
             "transforms": lambda th: jnp.array([-0.5]),
             "free_param_init": jnp.array([-0.5])}[where]
    phot = Photometry(filters=["sdss_g0"], flux=[1e-9], uncertainty=[1e-10], name="phot")
    with pytest.raises(ValueError, match=f"'alpha_pow' is fixed on the CSP .* also in {where}"):
        SedModel(csp, [phot], zred=0.5, **{where: {"alpha_pow": given}})


def test_fixed_is_a_constant_transform_byte_for_byte(ssp):
    a, b = _pair(ssp, {"alpha_pow": -0.7, "diffuse_dust_index": -0.3})
    assert a.param_names == b.param_names and "alpha_pow" not in a.param_names
    th = {k: jnp.asarray(v) for k, v in a.theta_init.items()}
    pa, pb = a.predict(th), b.predict(th)
    assert all(_same(pa[k], pb[k]) for k in pb)
    assert _same(a.csp.calculate_ssp_weights(a.apply_transforms(th)),
                 b.csp.calculate_ssp_weights(b.apply_transforms(th)))


@needs_sps
@pytest.mark.parametrize("fix, kw", [
    ({"gas_logu": -2.5}, {"add_neb": True}),
    ({"gas_logz": -0.3}, {"add_neb": True}),
    ({"duste_qpah": 2.0, "duste_umin": 3.0}, {"add_dust_emission": True}),
])
def test_nebular_and_dust_emission_equal_the_transform(ssp, fix, kw):
    a, b = _pair(ssp, fix, **kw)
    th = {k: jnp.asarray(v) for k, v in a.theta_init.items()}
    pa, pb = a.predict(th), b.predict(th)
    assert all(_same(pa[k], pb[k]) for k in pb)
    ta, tb = a.apply_transforms(th), b.apply_transforms(th)
    for f in ("get_spectrum", "get_line_spec", "predict_line_fluxes"):
        if hasattr(a.csp, f) and getattr(a.csp, "neb", None) is not None:
            assert _same(getattr(a.csp, f)(ta), getattr(b.csp, f)(tb)), f


def test_describe_lists_a_fixed_parameter(ssp):
    from ceridwen.model.describe import model_structure
    m = _model(_csp(ssp, fixed={"alpha_pow": -0.7}))
    p = next(p for p in model_structure(m).params if p.name == "alpha_pow")
    assert p.kind == "fixed" and float(np.ravel(p.value)[0]) == -0.7


# ---- range checks: the same check as a constant transform ------------------------------

def _raises_like_the_transform(ssp, fix, match, **kw):
    with pytest.raises(ValueError, match=match) as by_fixed:
        _model(_csp(ssp, fixed=fix, **kw))
    with pytest.raises(ValueError, match=match):
        _model(_csp(ssp, **kw), transforms={k: (lambda th, _v=float(v): jnp.array([_v]))
                                             for k, v in fix.items()})
    assert "CSPBasis(fixed=" in str(by_fixed.value)


def test_logzsol_outside_the_grid_raises(ssp):
    with pytest.warns(UserWarning):              # the CSP's own check_param_ranges, at build
        _raises_like_the_transform(ssp, {"logzsol": 3.0}, "outside this grid's logzsol range")


@needs_sps
@pytest.mark.parametrize("fix", [{"gas_logu": -6.0}, {"gas_logz": -3.0}])
def test_cloudy_axis_raises(ssp, fix):
    with pytest.warns(UserWarning):
        _raises_like_the_transform(ssp, fix, "outside the nebular grid's", add_neb=True)


@needs_sps
def test_agn_line_width_raises(ssp):
    from ceridwen.agn import AGNConfig
    _raises_like_the_transform(ssp, {"agn_eline_sigma": 3000.0}, "above 2000 km/s",
                               add_neb=True, agn=AGNConfig())


@needs_afe
def test_afe_outside_the_grid_raises():
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe
    ssp = SSPDataAfe.load(str(AFE_GRID))
    kw = dict(lookback_time=jnp.linspace(0.0, 1.0, 4), zh_const=True, verbose=False,
              cosmo=Cosmology.planck18())
    phot = Photometry(filters=["sdss_g0"], flux=[1e-9], uncertainty=[1e-10], name="phot")
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match=r"CSPBasis\(fixed=.*alpha/Fe\] range"):
            SedModel(CSPBasis_afe(ssp, fixed={"afe": 2.0}, **kw), [phot], zred=0.5)
    csp = CSPBasis_afe(ssp, fixed={"afe": 0.2}, **kw)
    assert "afe" not in csp.param_names and float(csp.theta_init["afe"][0]) == 0.2


# ---- the result file -------------------------------------------------------------------

def test_result_file_round_trip(ssp, tmp_path):
    from ceridwen.fit import write_result_h5
    from ceridwen.resultfile import rebuild_model
    from ceridwen.sampler.runner import SamplingResult
    csp = _csp(ssp, fixed={"alpha_pow": -0.7})
    m = _model(csp)
    names = list(m.theta_init)
    res = SamplingResult(samples={k: jnp.asarray(m.theta_init[k])[None] for k in names},
                         log_evidence=float("nan"), log_evidence_err=float("nan"),
                         log_weights=jnp.zeros(1), log_likelihoods=jnp.zeros(1),
                         param_names=names, n_likelihood_calls=0, wall_time_s=0.0,
                         sampler_name="test")
    path = tmp_path / "r.h5"
    write_result_h5(path, m, res)
    with pytest.warns(UserWarning):
        back = rebuild_model(path, _csp(ssp, fixed={"alpha_pow": -0.7}), m.observations)
    assert "alpha_pow" in back.transforms and back.param_names == m.param_names
    with pytest.raises(ValueError, match=r"\['alpha_pow'\] fixed on the CSP"):
        rebuild_model(path, _csp(ssp), m.observations)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):       # a different fixed value is a different model
            rebuild_model(path, _csp(ssp, fixed={"alpha_pow": -0.5}), m.observations)
