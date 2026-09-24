"""``ceridwen.optimize.map_fit``: MAP by L-BFGS from prior draws (Prospector's ``nmin``).

Mock: the configuration of ``examples/make_mock_data.py`` (FILTERS, SPEC_WAVE, SPEC_RES, SNR_*,
ZRED, N_TIME, T_OLDEST, TRUTH), regenerated in memory on the test grid with the same
``np.random.default_rng(42)`` draw order, and the priors of ``examples/demo_1_mock_test.py``.

Acceptance (night prompt, item 2):
  * ln p(MAP) >= the best ln p of 1000 prior draws, on the same jitted log-posterior;
  * a fixed ``rng_key`` gives a byte-identical result;
  * ``res.theta`` is usable as ``free_param_init`` (shapes, and the rebuilt model's
    ``theta_init`` evaluates to the same ln p).
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
sys.path.insert(0, str(HERE.parent / "examples"))
from _gridfixture import find_test_grid  # noqa: E402

GRID = find_test_grid()
pytestmark = pytest.mark.skipif(GRID is None, reason="no test SSP grid")

N_STARTS = 2
N_PRIOR = 1000


@pytest.fixture(scope="module")
def mock():
    import make_mock_data as mm
    from ceridwen import CSPBasis, Instrument, SSPData, SedModel
    from ceridwen.cosmology import Cosmology
    from ceridwen.model import logsfr_ratios_to_sfh
    from ceridwen.observation import Photometry, Spectrum
    from ceridwen.priors import ClippedNormal, StudentT, Uniform

    ssp = SSPData.load(str(GRID))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, mm.T_OLDEST, mm.N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False,
                   add_diffuse_dust=True, add_neb=False, verbose=False,
                   cosmo=Cosmology.planck18())
    t_yr = np.array(csp.sfh_times)
    inst = Instrument.sigma_kms(mm.SPEC_RES)

    def build(obs, priors=None, init=None):
        return SedModel(
            csp, observations=obs, priors=priors,
            transforms={"sfh": lambda th, _t=t_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
            free_param_init=init or {"logsfr_ratios": jnp.zeros(mm.N_TIME - 1),
                                     "logmass": jnp.array([10.0])},
            zred=mm.ZRED)

    gen = build([Photometry(filters=mm.FILTERS, name="phot"),
                 Spectrum(wavelength=mm.SPEC_WAVE, instrument=inst, name="spec")])
    noiseless = gen.predict(mm.TRUTH)
    rng = np.random.default_rng(42)
    mag = np.asarray(noiseless["phot"])
    mag_unc = mag / mm.SNR_PHOT
    mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)
    flux = np.asarray(noiseless["spec"])
    flux_unc = np.abs(flux) / mm.SNR_SPEC
    flux_obs = flux + flux_unc * rng.standard_normal(flux.shape)

    priors = {
        "logzsol": Uniform(low=-2.0, high=0.2),
        "logmass": Uniform(low=9.0, high=12.0),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
    }

    def observed():
        return [Photometry(filters=mm.FILTERS, flux=mag_obs, uncertainty=mag_unc, name="phot"),
                Spectrum(wavelength=mm.SPEC_WAVE, flux=flux_obs, uncertainty=flux_unc,
                         instrument=inst, name="spec")]

    model = build(observed(), priors)
    return dict(model=model, build=build, observed=observed, priors=priors, mm=mm)


@pytest.fixture(scope="module")
def fitted(mock):
    from ceridwen.optimize import build_lnprob, map_fit
    lnprob = build_lnprob(mock["model"])
    key = jax.random.PRNGKey(1)
    r1 = map_fit(mock["model"], n_starts=N_STARTS, rng_key=key, lnprob=lnprob)
    r2 = map_fit(mock["model"], n_starts=N_STARTS, rng_key=key, lnprob=lnprob)
    return lnprob, r1, r2


def test_map_beats_best_of_1000_prior_draws(mock, fitted):
    lnprob, res, _ = fitted
    model, priors = mock["model"], mock["priors"]
    key = jax.random.PRNGKey(123)
    draws = {}
    for name, init in model.theta_init.items():
        key, sub = jax.random.split(key)
        draws[name] = priors[name].sample(sub, shape=(N_PRIOR, *np.shape(init)))
    lnp_draws = np.asarray(jax.jit(jax.vmap(lnprob))(draws))
    best_draw = float(np.max(np.where(np.isfinite(lnp_draws), lnp_draws, -np.inf)))
    print(f"\nlnp_MAP = {res.lnp:.4f}, best of {N_PRIOR} prior draws = {best_draw:.4f}, "
          f"wall {res.wall_time:.1f} s, steps {res.n_steps.tolist()}, "
          f"lnp per start {np.array2string(res.lnp_starts, precision=3)}")
    assert res.lnp >= best_draw
    # the reported value is the log-posterior of the returned theta
    lnp_theta = float(lnprob({k: jnp.asarray(v) for k, v in res.theta.items()}))
    assert lnp_theta == pytest.approx(res.lnp, rel=1e-9, abs=0)


def test_map_is_deterministic(fitted):
    _, r1, r2 = fitted
    assert list(r1.theta) == list(r2.theta)
    for k in r1.theta:
        assert np.asarray(r1.theta[k]).tobytes() == np.asarray(r2.theta[k]).tobytes(), k
    assert r1.lnp_starts.tobytes() == r2.lnp_starts.tobytes()
    assert np.array_equal(r1.n_steps, r2.n_steps)
    assert r1.best_start == r2.best_start


def test_map_theta_is_a_free_param_init(mock, fitted):
    from ceridwen.optimize import build_lnprob
    _, res, _ = fitted
    model = mock["model"]
    assert set(res.theta) == set(model.theta_init)
    for k, v in model.theta_init.items():
        assert np.shape(res.theta[k]) == np.shape(v), k
    # the starts: theta_init first, then N_STARTS prior draws
    for k, v in res.theta_starts.items():
        assert v.shape == (N_STARTS + 1, *np.shape(model.theta_init[k]))
        assert np.array_equal(v[0], np.asarray(model.theta_init[k]))
    # bounded parameters stay inside their prior bounds
    for k, (lo, hi) in res.bounds.items():
        assert np.all((res.theta[k] >= lo) & (res.theta[k] <= hi)), k
    rebuilt = mock["build"](mock["observed"](), mock["priors"], init=res.theta)
    for k in model.theta_init:
        assert np.array_equal(np.asarray(rebuilt.theta_init[k]), np.asarray(res.theta[k])), k
    lnp = float(build_lnprob(rebuilt)(rebuilt.theta_init))
    assert lnp == pytest.approx(res.lnp, rel=1e-9, abs=0)


def test_map_setup_errors(mock):
    from ceridwen.optimize import map_fit
    with pytest.raises(ValueError, match="n_starts"):
        map_fit(mock["model"], n_starts=0)
    with pytest.raises(ValueError, match="not free"):
        map_fit(mock["model"], n_starts=1, bounds={"nope": (0.0, 1.0)})
    no_prior = mock["build"](mock["observed"](), {"logmass": Uniform_(9.0, 12.0)})
    with pytest.raises(ValueError, match="has no prior"):
        map_fit(no_prior, n_starts=1)


def Uniform_(lo, hi):
    from ceridwen.priors import Uniform
    return Uniform(low=lo, high=hi)


def test_fitsed_optimize_starts_nuts_at_the_map_and_records_it(mock, tmp_path, monkeypatch):
    """fitSED(optimize=True): the MAP is handed to the sampler as its start and stored in
    /map.  The adapter's run is replaced by a recorder, so no chain is run."""
    from ceridwen import fitSED, read_result_h5
    from ceridwen.optimize import map_fit
    from ceridwen.sampler.nuts import BlackJAXNUTSAdapter
    from ceridwen.sampler.runner import SamplingResult

    seen = {}

    def fake_run(self, loglike_fn, logprior_fn, theta_init, rng_key):
        seen["theta_init"] = {k: np.asarray(v) for k, v in theta_init.items()}
        seen["lnp"] = float(loglike_fn(theta_init) + logprior_fn(theta_init))
        names = list(theta_init)
        return SamplingResult(samples={k: jnp.asarray(theta_init[k])[None] for k in names},
                              log_evidence=float("nan"), log_evidence_err=float("nan"),
                              log_weights=jnp.zeros(1), log_likelihoods=jnp.zeros(1),
                              param_names=names, n_likelihood_calls=0, wall_time_s=0.0,
                              sampler_name="recorder")

    monkeypatch.setattr(BlackJAXNUTSAdapter, "run", fake_run)
    okw = dict(n_starts=1, max_steps=30)
    fitSED(mock["model"], output_dir=tmp_path, sampler="nuts", verbose=False,
           optimize=True, optimize_kwargs=okw, rng_key=jax.random.PRNGKey(5))
    ref = map_fit(mock["model"], rng_key=jax.random.fold_in(jax.random.PRNGKey(5), 1), **okw)
    for k, v in ref.theta.items():
        assert np.asarray(seen["theta_init"][k]).tobytes() == np.asarray(v).tobytes(), k
    assert seen["lnp"] == pytest.approx(ref.lnp, rel=1e-9, abs=0)
    rec = read_result_h5(tmp_path / "ceridwen_result.h5")["map"]
    assert rec["lnp"] == ref.lnp and rec["best_start"] == ref.best_start
    for k, v in ref.theta.items():
        assert np.array_equal(rec["theta"][k], v), k
    # B2-011: the file records what reproduces the MAP (settings incl. the key, bounds)
    import json
    st = json.loads(rec["settings_json"])
    assert st == ref.settings and st["n_starts"] == 1 and st["max_steps"] == 30
    assert st["rng_key"] == np.asarray(jax.random.fold_in(jax.random.PRNGKey(5), 1)).tolist()
    assert set(json.loads(rec["bounds_json"])) == set(ref.bounds)

    # without optimize the sampler starts at model.theta_init and no /map is written
    fitSED(mock["model"], output_dir=tmp_path / "plain", sampler="nuts", verbose=False)
    for k, v in mock["model"].theta_init.items():
        assert np.array_equal(seen["theta_init"][k], np.asarray(v)), k
    assert "map" not in read_result_h5(tmp_path / "plain" / "ceridwen_result.h5")
