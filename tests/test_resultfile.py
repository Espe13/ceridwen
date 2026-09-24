"""``ceridwen.resultfile``: rebuild a SedModel from a result file / check a model against one.

The result files here are written by ``fit.write_result_h5`` itself with a one-draw stub
``SamplingResult``, so no sampler runs; the model is the mock configuration of
``examples/make_mock_data.py`` on the test grid.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _gridfixture import find_test_grid  # noqa: E402

GRID = find_test_grid()
needs_grid = pytest.mark.skipif(GRID is None, reason="no test SSP grid")

FILTERS = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.linspace(4000.0, 8000.0, 300)


@pytest.fixture(scope="module")
def setup():
    from ceridwen import CSPBasis, Instrument, Kinematics, SSPData, SedModel
    from ceridwen.cosmology import Cosmology
    from ceridwen.model import logsfr_ratios_to_sfh
    from ceridwen.observation import Photometry, Spectrum
    from ceridwen.priors import ClippedNormal, StudentT, Uniform

    ssp = SSPData.load(str(GRID))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 10.0, 5), zh_const=True,
                   sfh_interp="step", add_dust=False, add_diffuse_dust=True, add_neb=False,
                   verbose=False, cosmo=Cosmology.planck18())
    t_yr = np.array(csp.sfh_times)

    def sfh_from_ratios(th, _t=t_yr):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)

    rng = np.random.default_rng(3)
    phot_flux = 1e-9 * (1.0 + 0.1 * rng.standard_normal(len(FILTERS)))
    spec_flux = 1e-9 * (1.0 + 0.1 * rng.standard_normal(SPEC_WAVE.size))

    def observations(spec_flux=spec_flux):
        return [Photometry(filters=FILTERS, flux=phot_flux, uncertainty=0.05 * phot_flux,
                           name="phot"),
                Spectrum(wavelength=SPEC_WAVE, flux=spec_flux, uncertainty=0.05 * spec_flux,
                         instrument=Instrument.sigma_kms(100.0), name="spec")]

    priors = {"logzsol": Uniform(low=-2.0, high=0.2),
              "logmass": Uniform(low=9.0, high=12.0),
              "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
              "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
              "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
              "sigma_gal": Uniform(low=50.0, high=400.0)}
    kin = Kinematics(sigma_gal="sigma_gal")

    def build(priors=priors, zred=0.1, kinematics=kin, obs=None, init=None, transforms=None):
        return SedModel(csp, observations=obs or observations(), priors=priors,
                        transforms=transforms or {"sfh": sfh_from_ratios},
                        free_param_init=init or {"logsfr_ratios": jnp.zeros(4),
                                                 "logmass": jnp.array([10.0]),
                                                 "sigma_gal": jnp.array([200.0])},
                        zred=zred, kinematics=kinematics)

    return dict(csp=csp, build=build, observations=observations, priors=priors,
                sfh=sfh_from_ratios)


def _write(model, path, with_likelihood=True):
    from ceridwen.fit import write_result_h5, _likelihood_for
    from ceridwen.likelihood import MultiObservationLikelihood
    from ceridwen.sampler.runner import SamplingResult
    names = list(model.theta_init)
    n = 3
    res = SamplingResult(
        samples={k: jnp.repeat(jnp.asarray(model.theta_init[k])[None], n, axis=0)
                 for k in names},
        log_evidence=-1.0, log_evidence_err=0.1, log_weights=jnp.zeros(n),
        log_likelihoods=jnp.zeros(n), param_names=names, n_likelihood_calls=1,
        wall_time_s=0.0, sampler_name="test")
    lh = None
    if with_likelihood:         # exactly as fitSED builds it (fit.py, fitSED)
        keys = tuple(model.obs_dict)
        lh = MultiObservationLikelihood(keys=keys, likelihoods=tuple(
            _likelihood_for(model.obs_dict[k], model.param_names, model=model) for k in keys))
    write_result_h5(path, model, res, verbose=False, likelihood=lh)
    return path


@needs_grid
def test_file_written_like_fitsed_rebuilds(setup, tmp_path):
    """B2-001: fitSED passes likelihood= to write_result_h5, so its files carry
    /obs/<name>@likelihood_json; the model side must record it too."""
    from ceridwen.resultfile import check_model_against_result, rebuild_model
    model = setup["build"]()
    path = _write(model, tmp_path / "fitsed.h5")
    cmp = check_model_against_result(model, path)
    assert cmp.ok, str(cmp)
    rebuild_model(path, setup["csp"], setup["observations"](),
                  transforms={"sfh": setup["sfh"]})
    # a file without likelihood_json (written before it was recorded) is a note, not a
    # difference
    old = _write(model, tmp_path / "old.h5", with_likelihood=False)
    cmp = check_model_against_result(model, old)
    assert cmp.ok, str(cmp)
    assert {n[0] for n in cmp.notes} == {"/obs/phot@likelihood_json",
                                         "/obs/spec@likelihood_json"}


@needs_grid
def test_same_model_matches_and_rebuild_round_trips(setup, tmp_path):
    from ceridwen.resultfile import check_model_against_result, rebuild_model
    model = setup["build"]()
    path = _write(model, tmp_path / "r.h5")
    cmp = check_model_against_result(model, path)
    assert cmp.ok, str(cmp)
    assert not cmp.notes

    rebuilt = rebuild_model(path, setup["csp"], setup["observations"](),
                            transforms={"sfh": setup["sfh"]})
    assert rebuilt.param_names == model.param_names
    assert list(rebuilt.theta_init) == list(model.theta_init)
    for k in model.theta_init:
        assert np.array_equal(np.asarray(rebuilt.theta_init[k]), np.asarray(model.theta_init[k]))
    assert {k: p.serialize() for k, p in rebuilt.priors.items()} == \
        {k: p.serialize() for k, p in model.priors.items()}
    assert rebuilt.kinematics == model.kinematics
    assert rebuilt.zred == model.zred
    # the rebuilt model predicts what the original predicts
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    a, b = model.predict(th), rebuilt.predict(th)
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


@needs_grid
def test_differences_are_named(setup, tmp_path):
    from ceridwen.priors import Uniform
    from ceridwen.resultfile import check_model_against_result
    path = _write(setup["build"](), tmp_path / "r.h5")

    def diff_keys(model):
        return {d[0] for d in check_model_against_result(model, path).differences}

    pri = dict(setup["priors"], logmass=Uniform(low=8.0, high=12.0))
    assert diff_keys(setup["build"](priors=pri)) == {"/model/priors@logmass"}
    assert "/model@zred" in diff_keys(setup["build"](zred=0.2))
    other_flux = setup["observations"]()[1].flux * 1.01
    assert diff_keys(setup["build"](obs=setup["observations"](spec_flux=other_flux))) \
        == {"/obs/spec/flux", "/obs/spec/uncertainty"}
    from ceridwen import Kinematics
    assert diff_keys(setup["build"](kinematics=Kinematics(sigma_gal=150.0), priors={
        k: v for k, v in setup["priors"].items() if k != "sigma_gal"},
        init={"logsfr_ratios": jnp.zeros(4), "logmass": jnp.array([10.0])})) >= {
        "/model@kinematics_sigma_gal", "/model/param_names", "/model/priors@sigma_gal"}

    def renamed(th):
        return setup["sfh"](th)
    assert diff_keys(setup["build"](transforms={"sfh": renamed})) == {"/model@transforms"}

    # a different starting point is a note, not a difference
    init = {"logsfr_ratios": jnp.zeros(4), "logmass": jnp.array([10.5]),
            "sigma_gal": jnp.array([200.0])}
    cmp = check_model_against_result(setup["build"](init=init), path)
    assert cmp.ok and [n[0] for n in cmp.notes] == ["/model/theta_init/logmass"]
    with pytest.raises(ValueError, match="priors@logmass"):
        check_model_against_result(setup["build"](priors=pri), path, raise_on_difference=True)


@needs_grid
def test_a_different_csp_is_named(setup, tmp_path):
    from ceridwen import CSPBasis, SSPData, SedModel
    from ceridwen.cosmology import Cosmology
    from ceridwen.resultfile import check_model_against_result
    path = _write(setup["build"](), tmp_path / "r.h5")
    csp2 = CSPBasis(SSPData.load(str(GRID)), lookback_time=jnp.linspace(0.0, 9.0, 5),
                    zh_const=True, sfh_interp="linear", add_dust=False, add_diffuse_dust=True,
                    add_neb=False, verbose=False, cosmo=Cosmology.planck18())
    m = setup["build"]()
    m2 = SedModel(csp2, setup["observations"](), priors=m.priors, transforms=m.transforms,
                  free_param_init={k: m.theta_init[k] for k in
                                   ("logsfr_ratios", "logmass", "sigma_gal")},
                  zred=m.zred, kinematics=m.kinematics)
    keys = {d[0] for d in check_model_against_result(m2, path).differences}
    assert keys == {"/model@csp_config", "/model/sfh_times_yr"}, keys

    # an older file without the CSP record: noted, not a difference
    import h5py
    with h5py.File(path, "a") as f:
        del f["model"].attrs["csp_config"]
        del f["model"]["sfh_times_yr"]
    cmp = check_model_against_result(m, path)
    assert cmp.ok and {n[0] for n in cmp.notes} == {"/model@csp_config", "/model/sfh_times_yr"}


@needs_grid
def test_rebuild_requires_the_transforms(setup, tmp_path):
    from ceridwen.resultfile import rebuild_model
    path = _write(setup["build"](), tmp_path / "r.h5")
    with pytest.raises(ValueError, match="sfh <- sfh_from_ratios"):
        rebuild_model(path, setup["csp"], setup["observations"]())
    # a different observation set is caught by the check
    obs = setup["observations"]()[:1]
    with pytest.raises(ValueError, match="/obs/spec"):
        rebuild_model(path, setup["csp"], obs, transforms={"sfh": setup["sfh"]})


def test_priors_round_trip_through_their_serialisation(tmp_path):
    """Every prior class the package exports is rebuilt exactly from its serialisation."""
    import h5py
    import json
    from ceridwen.sampler import priors as P
    from ceridwen.resultfile import priors_from_result

    made = {"u": P.Uniform(low=-1.0, high=2.0), "t": P.TopHat(low=0.0, high=1.0),
            "n": P.Normal(mean=0.5, sigma=2.0),
            "c": P.ClippedNormal(mean=0.0, sigma=1.0, low=-1.0, high=3.0),
            "l": P.LogNormal(mode=0.1, sigma=0.4), "s": P.StudentT(mean=0.0, scale=1.0, df=3.0),
            "m": P.MultivariateNormalPrior(mean=jnp.zeros(2), Sigma=jnp.eye(2))}
    if hasattr(P, "LogUniform"):
        made["lu"] = P.LogUniform(mini=0.01, maxi=3.0)
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("model")
        g.create_dataset("param_names", data=np.array(list(made), dtype=object),
                         dtype=h5py.string_dtype())
        g.create_group("theta_init")
        g.create_dataset("wave", data=np.arange(3.0))
        g.attrs["metallicity_convention"] = "logzsol"
        pg = g.create_group("priors")
        for k, p in made.items():
            pg.attrs[k] = json.dumps(p.serialize())
        f.create_group("obs")
        f.create_group("samples")
    back = priors_from_result(path)
    x = jnp.array([0.3])
    for k, p in made.items():
        assert back[k].serialize() == p.serialize(), k
        if k == "m":
            continue
        assert np.array_equal(np.asarray(back[k].logpdf(x)), np.asarray(p.logpdf(x))), k


@needs_grid
def test_vector_uniform_prior_round_trips(setup, tmp_path):
    """G2-001: a Uniform with per-element bounds on a vector parameter is written (it
    raised TypeError in Uniform.serialize, after sampling) and read back."""
    from ceridwen.priors import Uniform
    from ceridwen.resultfile import check_model_against_result, priors_from_result
    pri = dict(setup["priors"], logsfr_ratios=Uniform(low=-np.array([1.0, 2.0, 3.0, 4.0]),
                                                      high=np.array([1.0, 2.0, 3.0, 4.0])))
    model = setup["build"](priors=pri)
    path = _write(model, tmp_path / "vec.h5")
    back = priors_from_result(path)
    assert back["logsfr_ratios"].serialize() == pri["logsfr_ratios"].serialize()
    assert back["logsfr_ratios"].serialize()["low"] == [-1.0, -2.0, -3.0, -4.0]
    assert check_model_against_result(model, path).ok
    # scalar bounds serialise exactly as before (floats)
    assert Uniform(low=0.0, high=1.0).serialize() == {"type": "Uniform", "low": 0.0,
                                                      "high": 1.0, "name": ""}
