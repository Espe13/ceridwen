"""Result-file provenance (v1.0.6): the CERIDWEN version and git state, the sampler settings
actually used, the rng key, the observation extras (sky, calibration, upper_limit,
noise_floor) and each observation's likelihood / noise-model configuration are written by
``write_result_h5`` / ``fitSED`` and read back by ``read_result_h5``."""
from __future__ import annotations

import pathlib
import subprocess
import warnings

import h5py
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from _gridfixture import require_test_grid

import ceridwen
from ceridwen import SSPData, CSPBasis, SedModel, Cosmology, fitSED, read_result_h5
from ceridwen.broadening import Instrument
from ceridwen.fit import (write_result_h5, read_provenance, adapter_settings, _likelihood_for)
from ceridwen.likelihood.likelihood import MultiObservationLikelihood
from ceridwen.observation import Photometry, Spectrum
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
from ceridwen.sampler.nuts import BlackJAXNUTSAdapter
from ceridwen.sampler.priors import Uniform
from ceridwen.sampler.runner import SamplingResult

N = 10
WAVE = np.linspace(5000.0, 8000.0, 40)
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0"]


@pytest.fixture(scope="module")
def model():
    ssp = SSPData.load(str(require_test_grid()))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 8.0, 4), zh_const=True,
                   add_dust=False, add_diffuse_dust=False, add_neb=False, add_igm=False,
                   verbose=False, cosmo=Cosmology.planck18())
    phot = Photometry(filters=FILTERS, flux=[1e-9, 1.2e-9, 1.1e-9],
                      uncertainty=[1e-10] * 3, upper_limit=[False, False, True], name="p")
    spec = Spectrum(wavelength=WAVE, flux=np.full(WAVE.size, 2e-29),
                    uncertainty=np.full(WAVE.size, 2e-30), instrument=Instrument.sigma_kms(80.0),
                    sky=np.full(WAVE.size, 1e-31), calibration=np.linspace(0.9, 1.1, WAVE.size),
                    noise_floor=0.02, name="s")
    return SedModel(csp, [phot, spec],
                    priors={"logzsol": Uniform(low=-0.5, high=0.2),
                            "logmass": Uniform(low=9.5, high=10.5),
                            "sfh": Uniform(low=0.1, high=2.0),
                            "log_jitter_spec": Uniform(low=-5.0, high=0.0)},
                    free_param_init={"logmass": jnp.array([10.0]),
                                     "log_jitter_spec": jnp.array([-2.0])}, zred=0.5)


def _result(model):
    rng = np.random.default_rng(0)
    samples = {n: jnp.asarray(np.broadcast_to(np.asarray(model.theta_init[n], float),
                                              (N,) + np.shape(model.theta_init[n])).squeeze(-1)
                              if np.shape(model.theta_init[n]) == (1,) else
                              np.broadcast_to(np.asarray(model.theta_init[n], float),
                                              (N,) + np.shape(model.theta_init[n])))
               for n in model.param_names}
    return SamplingResult(samples=samples, log_evidence=-1.0, log_evidence_err=0.1,
                          log_weights=jnp.zeros(N), log_likelihoods=jnp.asarray(rng.normal(size=N)),
                          param_names=list(model.param_names), n_likelihood_calls=N,
                          wall_time_s=0.0, sampler_name="fake")


def _likelihood(model):
    keys = tuple(model.obs_dict)
    return MultiObservationLikelihood(keys=keys, likelihoods=tuple(
        _likelihood_for(model.obs_dict[k], model.param_names, model=model) for k in keys))


def _git_head():
    root = pathlib.Path(ceridwen.__file__).resolve().parent
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                       text=True, env={"GIT_OPTIONAL_LOCKS": "0", "PATH": "/usr/bin:/bin"})
    return r.stdout.strip() if r.returncode == 0 else None


def test_write_read_roundtrip(model, tmp_path):
    adapter = BlackJAXNestedSamplerAdapter(priors=model.priors, num_live=40, num_delete=8,
                                           logZ_tol=-2.5, verbose=False)
    key = jax.random.PRNGKey(20260922)
    path = tmp_path / "r.h5"
    write_result_h5(path, model, _result(model), verbose=False, likelihood=_likelihood(model),
                    adapter=adapter, rng_key=key)
    r = read_result_h5(path)
    prov = r["provenance"]
    assert prov["ceridwen_version"] == ceridwen.__version__
    assert prov["ceridwen_githash"] == str(ceridwen.__githash__)
    head = _git_head()
    if head is not None:
        assert prov["git_head"] == head and isinstance(prov["git_dirty"], (bool, np.bool_))
    assert prov["jax_version"] == jax.__version__
    n_dims = sum(int(np.size(v)) for v in model.theta_init.values())
    assert prov["sampler"] == {"adapter": "BlackJAXNestedSamplerAdapter", "num_live": 40,
                               "num_inner_steps": 5 * n_dims, "num_delete": 8,
                               "logZ_tol": -2.5, "checkpoint_interval_s": 1200.0}
    np.testing.assert_array_equal(prov["rng_key"], np.asarray(key))
    assert read_provenance(path).keys() == prov.keys()

    s, p = r["obs"]["s"], r["obs"]["p"]
    np.testing.assert_array_equal(s["sky"], np.full(WAVE.size, 1e-31))
    np.testing.assert_array_equal(s["calibration"], np.linspace(0.9, 1.1, WAVE.size))
    assert s["noise_floor"] == 0.02 and p["noise_floor"] == 0.0
    np.testing.assert_array_equal(p["upper_limit"], [False, False, True])
    assert "sky" not in p and "calibration" not in p and "upper_limit" not in s
    assert p["likelihood"]["class"] == "DiagonalGaussianLikelihoodWithUpperLimits"
    assert s["likelihood"]["class"] == "DiagonalGaussianLikelihood"
    nm = s["likelihood"]["noise_model"]
    assert nm["class"] == "DiagonalNoiseModel" and nm["noise_floor"] == 0.02
    assert nm["use_jitter"] is True and nm["f_outlier"] is None
    assert nm["sampled_parameters"] == ["log_jitter_spec"] and nm["jitter_key"] == "log_jitter_spec"


def test_nuts_settings_and_old_files(model, tmp_path):
    ad = BlackJAXNUTSAdapter(num_warmup=123, num_samples=45, num_chains=3,
                             target_acceptance=0.9, bounds={"logmass": (9.5, 10.5)},
                             verbose=False)
    got = adapter_settings(ad)
    assert got["num_warmup"] == 123 and got["num_samples"] == 45 and got["num_chains"] == 3
    assert got["target_acceptance"] == 0.9 and got["bounds"] == {"logmass": [9.5, 10.5]}
    assert got["vi"] is None
    # a file written without adapter / rng key / likelihood still carries the version
    path = tmp_path / "bare.h5"
    write_result_h5(path, model, _result(model), verbose=False)
    r = read_result_h5(path)
    assert r["provenance"]["sampler"] == {} and "rng_key" not in r["provenance"]
    assert "likelihood" not in r["obs"]["s"]
    # and a file from before v1.0.6 (no /provenance group) reads with no 'provenance' key
    with h5py.File(path, "r+") as f:
        del f["provenance"]
    assert "provenance" not in read_result_h5(path) and read_provenance(path) == {}


def test_fitsed_records_what_it_ran(model, tmp_path):
    for name in model.param_names:
        model.priors.setdefault(name, Uniform(low=-1.0, high=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitSED(model, output_dir=tmp_path, rng_key=jax.random.PRNGKey(3), verbose=False,
               sampler_kwargs={"num_live": 20, "num_delete": 5, "num_inner_steps": 3,
                               "logZ_tol": -0.5})
    prov = read_result_h5(tmp_path / "ceridwen_result.h5")["provenance"]
    assert prov["sampler"]["num_live"] == 20 and prov["sampler"]["num_delete"] == 5
    assert prov["sampler"]["num_inner_steps"] == 3 and prov["sampler"]["logZ_tol"] == -0.5
    np.testing.assert_array_equal(prov["rng_key"], np.asarray(jax.random.PRNGKey(3)))


def test_versions_and_foreign_repo(model, tmp_path):
    """blackjax / numpy versions are recorded; a git repository that does not track the
    package (a pip install inside a user's project) is not reported as ceridwen's HEAD."""
    import subprocess
    import shutil
    from importlib.metadata import version
    from ceridwen.fit import _git_state
    path = tmp_path / "r.h5"
    write_result_h5(path, model, _result(model), verbose=False)
    prov = read_result_h5(path)["provenance"]
    assert prov["blackjax_version"] == version("blackjax")
    assert prov["numpy_version"] == version("numpy")
    if shutil.which("git") is None:
        pytest.skip("git not available")
    proj = tmp_path / "project"
    pkg = proj / ".venv" / "site-packages" / "ceridwen"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "GIT_OPTIONAL_LOCKS": "0",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q", str(proj)], check=True, env=env)
    (proj / "README").write_text("x")
    subprocess.run(["git", "-C", str(proj), "add", "README"], check=True, env=env)
    subprocess.run(["git", "-C", str(proj), "commit", "-qm", "x"], check=True, env=env)
    assert _git_state(pkg) == (None, None)
