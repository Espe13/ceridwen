"""Result files carry the metallicity convention (v1.0.5), and pre-v1.0.5 files are refused.

``write_result_h5`` records Z_sun, the axis and the grid identity; ``load_result_h5`` /
``read_result_h5`` refuse a file whose metallicity samples are absolute log10 Z; and
``convert_result`` converts one only against the grid the fit actually used.
"""
from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from _gridfixture import require_test_grid

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.cosmology import Cosmology
from ceridwen.model.model import SedModel
from ceridwen.observation import Photometry
from ceridwen.fit import (write_result_h5, load_result_h5, read_result_h5, convert_result,
                          METALLICITY_CONVENTION)
from ceridwen.sampler.priors import Uniform
from ceridwen.sampler.runner import SamplingResult

N = 12


@pytest.fixture(scope="module")
def model_and_result():
    ssp = SSPData.load(str(require_test_grid()))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 8.0, 4), zh_const=True,
                   add_dust=False, add_diffuse_dust=False, add_neb=False, add_igm=False,
                   verbose=False, cosmo=Cosmology.planck18())
    phot = Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1e-9, 1e-9],
                      uncertainty=[1e-10, 1e-10], name="p")
    model = SedModel(csp, [phot], priors={"logzsol": Uniform(low=-1.0, high=0.2),
                                          "logmass": Uniform(low=9.0, high=11.0),
                                          "sfh": Uniform(low=0.0, high=2.0)},
                     free_param_init={"logmass": jnp.array([10.0])}, zred=0.5)
    rng = np.random.default_rng(0)
    samples = {"logzsol": jnp.asarray(rng.uniform(-1.0, 0.2, N)),
               "logmass": jnp.asarray(rng.uniform(9.0, 11.0, N)),
               "sfh": jnp.asarray(rng.uniform(0.0, 2.0, (N, 4)))}
    result = SamplingResult(samples=samples, log_evidence=-1.0, log_evidence_err=0.1,
                            log_weights=jnp.zeros(N), log_likelihoods=jnp.zeros(N),
                            param_names=list(samples), n_likelihood_calls=N,
                            wall_time_s=0.0, sampler_name="fake",
                            log_likelihoods_birth=jnp.full((N,), -1.0))
    return model, result


def test_result_records_the_convention_and_the_grid(tmp_path, model_and_result):
    model, result = model_and_result
    path = tmp_path / "ceridwen_result.h5"
    write_result_h5(str(path), model, result, model.observations)
    with h5py.File(path) as f:
        a = dict(f["model"].attrs)
        assert a["metallicity_convention"] == METALLICITY_CONVENTION
        assert a["log10_zsun"] == model.csp.log10_zsun
        assert a["zsun_nominal"] == pytest.approx(model.csp.zsun_nominal)
        assert a["metallicity_axis_meaning"] == model.csp.axis_meaning
        assert a["grid_chash"] == model.csp.grid_chash
        assert json.loads(a["grid_file_sha256"]), "the published file sha256(s) of the grid"
        assert np.allclose(np.asarray(f["model"]["metallicity_axis_logzsol"]),
                           np.asarray(model.csp.zmet))
        assert np.allclose(np.asarray(f["model"]["metallicity_axis_native"]),
                           np.asarray(model.csp.zmet_native))
    # and it reads back
    back = load_result_h5(str(path))
    assert "logzsol" in back.samples


def _legacy_file(path, model, result, l0):
    """Rewrite a v1.0.5 result as a pre-v1.0.5 one: logzsol -> absolute Z."""
    write_result_h5(str(path), model, result, model.observations)
    with h5py.File(path, "r+") as f:
        mod, samp = f["model"], f["samples"]
        samp["Z"] = np.asarray(samp["logzsol"]) + l0
        del samp["logzsol"]
        mod["theta_init"]["Z"] = np.asarray(mod["theta_init"]["logzsol"]) + l0
        del mod["theta_init"]["logzsol"]
        pr = json.loads(mod["priors"].attrs["logzsol"])
        pr["low"] += l0
        pr["high"] += l0
        mod["priors"].attrs["Z"] = json.dumps(pr)
        del mod["priors"].attrs["logzsol"]
        names = [("Z" if n == "logzsol" else n) for n in list(mod["param_names"].asstr()[()])]
        del mod["param_names"]
        mod.create_dataset("param_names", data=np.array(names, dtype=object),
                           dtype=h5py.string_dtype())
        for k in ("metallicity_convention", "log10_zsun", "zsun_nominal",
                  "metallicity_axis_meaning", "grid_chash", "zsun_source", "grid_name",
                  "grid_file_sha256"):
            if k in mod.attrs:
                del mod.attrs[k]
    return path


def test_old_result_is_refused_by_both_readers(tmp_path, model_and_result):
    model, result = model_and_result
    old = _legacy_file(tmp_path / "old.h5", model, result, model.csp.log10_zsun)
    for reader in (load_result_h5, read_result_h5):
        with pytest.raises(ValueError, match="written before v1.0.5"):
            reader(str(old))


def test_convert_result_converts_against_the_right_grid(tmp_path, model_and_result):
    model, result = model_and_result
    l0 = model.csp.log10_zsun
    old = _legacy_file(tmp_path / "old.h5", model, result, l0)
    out = convert_result(str(old), ssp_grid=str(require_test_grid()))
    conv = load_result_h5(str(out))
    assert np.allclose(np.asarray(conv.samples["logzsol"]),
                       np.asarray(result.samples["logzsol"]), rtol=0, atol=1e-12)
    with h5py.File(out) as f:
        a = dict(f["model"].attrs)
        assert a["metallicity_convention"] == METALLICITY_CONVENTION
        assert a["log10_zsun"] == l0
        pr = json.loads(f["model"]["priors"].attrs["logzsol"])
        assert pr["low"] == pytest.approx(-1.0) and pr["high"] == pytest.approx(0.2)
        assert "converted_from" in a
    # the input file is untouched
    with h5py.File(old) as f:
        assert "Z" in f["samples"]


def test_convert_result_refuses_the_wrong_grid(tmp_path, model_and_result):
    """A grid of a different shape cannot be the grid of that fit, so its Z_sun would be wrong."""
    model, result = model_and_result
    old = _legacy_file(tmp_path / "old2.h5", model, result, model.csp.log10_zsun)
    other = pathlib.Path(__file__).resolve().parents[1] / "ceridwen/data/test_data/ssp_data_mist_miles.h5"
    if not other.is_file():
        pytest.skip("second grid not present")
    with pytest.raises(ValueError, match="not the grid of that fit"):
        convert_result(str(old), ssp_grid=str(other), out=str(tmp_path / "bad.h5"))


def test_convert_result_refuses_an_already_converted_file(tmp_path, model_and_result):
    model, result = model_and_result
    path = tmp_path / "new.h5"
    write_result_h5(str(path), model, result, model.observations)
    with pytest.raises(ValueError, match="already in the logzsol convention"):
        convert_result(str(path), ssp_grid=str(require_test_grid()),
                       out=str(tmp_path / "x.h5"))


def test_postprocess_refuses_a_result_from_another_zsun(tmp_path, model_and_result):
    """A result fitted on a grid with a different Z_sun must not be read with this model."""
    from ceridwen.postprocess import PostProcess
    model, result = model_and_result
    path = tmp_path / "other_zsun.h5"
    write_result_h5(str(path), model, result, model.observations)
    with h5py.File(path, "r+") as f:
        f["model"].attrs["log10_zsun"] = float(model.csp.log10_zsun) - 0.1
    with pytest.raises(ValueError, match="log10 Z_sun"):
        PostProcess(model, str(path), n_samples=4)
