"""PostProcess: shapes, conventions, consistency with the forward model, the
best fit, user-defined quantities, save/load.  Uses a synthetic
SamplingResult drawn around theta_init on the BPASS test grid (no sampler
run needed); skips without the grid."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid                                   # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                # noqa: E402
from ceridwen.postprocess import PostProcess, SpectrumSample, load_postprocess, DEFAULT_WINDOWS_MYR  # noqa: E402
from ceridwen.observation import Photometry, Spectrum                      # noqa: E402
from ceridwen.broadening import Instrument                                 # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh                # noqa: E402
from ceridwen.sampler.runner import SamplingResult                         # noqa: E402

ZRED = 0.5
N_TIME = 5
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0"]
SPEC_WAVE = np.linspace(5000.0, 9000.0, 80)


@pytest.fixture(scope="module")
def model():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    ssp = SSPData.load(str(path))
    cosmo = Cosmology.planck18()
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=False, add_igm=False, verbose=False, cosmo=cosmo)
    sfh_times_yr = np.array(csp.sfh_times)

    def _sfh(t, _t=sfh_times_yr):
        return logsfr_ratios_to_sfh(t["logsfr_ratios"], sfh_times_yr=_t)
    _sfh.__name__ = "logsfr_ratios_to_sfh"
    obs = [Photometry(filters=FILTERS, flux=[1e-9] * 3, uncertainty=[1e-10] * 3, name="phot"),
           Spectrum(wavelength=SPEC_WAVE, flux=np.ones_like(SPEC_WAVE) * 1e-30,
                    uncertainty=np.ones_like(SPEC_WAVE) * 1e-31,
                    instrument=Instrument.sigma_kms(100.0), name="spec")]
    return SedModel(csp, obs, priors={}, transforms={"sfh": _sfh},
                    free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                                     "logmass": jnp.array([10.0])}, zred=ZRED)


def _fake_result(model, n=40, weighted=True, seed=1, births=False):
    rng = np.random.default_rng(seed)
    samples = {}
    for name in model.param_names:
        base = np.asarray(model.theta_init[name], dtype=float)
        draws = base[None, :] + 0.05 * rng.standard_normal((n,) + base.shape)
        samples[name] = jnp.asarray(draws[:, 0] if base.shape == (1,) else draws)
    ll = rng.standard_normal(n)
    lw = rng.standard_normal(n) if weighted else np.zeros(n)
    llb = jnp.asarray(ll - rng.uniform(0.5, 2.0, n)) if births else None
    return SamplingResult(samples=samples, log_evidence=float("nan"), log_evidence_err=float("nan"),
                          log_weights=jnp.asarray(lw), log_likelihoods=jnp.asarray(ll),
                          param_names=list(model.param_names), n_likelihood_calls=n,
                          wall_time_s=0.0, sampler_name="fake", log_likelihoods_birth=llb)


def test_shapes_and_conventions(model, tmp_path):
    res = _fake_result(model)
    WIN = (3.0, 10.0, 100.0, 20000.0)
    pp = PostProcess(model, res, n_samples=12, seed=3, windows_myr=WIN,
                     derived={"A_V": lambda s: 2.5 * np.log10(s.dustfree[s.index_of(5500.0)] / s.full[s.index_of(5500.0)]),
                              "two": lambda s: np.array([s.zred, s.logmass])})
    out = pp.run()
    N = 12
    assert out["theta"]["logmass"].shape == (N,)
    assert out["theta"]["logsfr_ratios"].shape == (N, N_TIME - 1)
    assert out["draw_index"].shape == (N,) and out["meta"]["resampled"] is True
    sfh = out["extras"]["sfh"]
    assert sfh["lookback_gyr"].shape == (N, N_TIME) and sfh["sfr"].shape == (N, N_TIME)
    for w in WIN:
        assert sfh[f"sfr{w:g}"].shape == (N,) and sfh[f"ssfr{w:g}"].shape == (N,)
    # formed mass is 10**logmass under the unit-normalised transform
    np.testing.assert_allclose(sfh["mass_formed"], 10.0 ** out["theta"]["logmass"], rtol=1e-6)
    # the longest window covers the whole grid: mean SFR = mass / T_oldest
    T = sfh["lookback_gyr"][:, -1] * 1e9
    big = max(WIN) * 1e6
    assert big > T.max()
    np.testing.assert_allclose(sfh[f"sfr{max(WIN):g}"], sfh["mass_formed"] / big, rtol=1e-6)
    np.testing.assert_allclose(sfh["ssfr10"], sfh["sfr10"] / sfh["mass_formed"], rtol=1e-12)
    # step kernel: a 3 Myr window lies inside the youngest bin -> equals that bin's SFR
    assert sfh["lookback_gyr"][0, 1] * 1e9 > 3e6
    np.testing.assert_allclose(sfh["sfr3"], sfh["sfr_per_bin"][:, 0], rtol=1e-12)
    uv, ion = out["extras"]["uv"], out["extras"]["ionizing"]
    assert np.all(np.isfinite(uv["MUV"])) and np.all(uv["MUV_intrinsic"] <= uv["MUV"] + 1e-9)
    assert np.all(ion["nion"] > 0) and np.all(ion["xion"] > 0)
    np.testing.assert_allclose(ion["xion"], ion["nion"] / uv["LUV_intrinsic"])
    assert "fesc" not in ion                                       # model has no frac_obrun
    pred = out["prediction"]
    assert pred["photometry"]["phot"].shape == (N, 3) and pred["spectra"]["spec"].shape == (N, SPEC_WAVE.size)
    assert pred["lines"] == {}
    nw = pred["wave_rest"].size
    for k in ("spectra_model", "spectra_intrinsic", "spectra_dustfree", "spectra_observed"):
        assert pred[k].shape == (N, nw)
    np.testing.assert_allclose(pred["zred"], ZRED)
    assert np.all(pred["spectra_observed"] >= 0) and pred["spectra_observed"].max() < pred["spectra_model"].max()
    # no nebular, no birth-cloud dust: dustfree == intrinsic; diffuse dust attenuates the model
    np.testing.assert_array_equal(pred["spectra_dustfree"], pred["spectra_intrinsic"])
    opt = (pred["wave_rest"] > 4000) & (pred["wave_rest"] < 7000)
    assert np.all(pred["spectra_model"][:, opt] <= pred["spectra_intrinsic"][:, opt] * (1 + 1e-6))
    assert out["derived"]["A_V"].shape == (N,) and np.all(out["derived"]["A_V"] >= -1e-9)
    assert out["derived"]["two"].shape == (N, 2)
    np.testing.assert_allclose(out["derived"]["two"][:, 1], out["theta"]["logmass"])
    # predictions equal the model's own predict on the same draw
    i = 3
    th = {k: jnp.asarray(v[i]).reshape(model.theta_init[k].shape) for k, v in out["theta"].items()}
    direct = model.predict(th)
    np.testing.assert_allclose(pred["photometry"]["phot"][i], np.asarray(direct["phot"]), rtol=1e-6)
    # best fit is the max-likelihood raw sample, one draw, same tree
    b = out["bestfit"]
    ll = np.asarray(res.log_likelihoods)
    assert b["index"] == int(np.argmax(ll)) and b["log_likelihood"] == pytest.approx(float(ll.max()))
    np.testing.assert_allclose(b["theta"]["logmass"], float(np.asarray(res.samples["logmass"])[b["index"]]))
    assert b["extras"]["sfh"]["sfr"].shape == (N_TIME,) and np.ndim(b["extras"]["uv"]["MUV"]) == 0
    assert b["prediction"]["photometry"]["phot"].shape == (3,) and np.ndim(b["derived"]["A_V"]) == 0
    paths = pp.figures(tmp_path / "figs", title="test galaxy")
    assert all(paths[k].stat().st_size > 1000 for k in ("summary", "corner", "diagnostics"))
    # save / load
    p = pp.save(tmp_path / "post")
    back = load_postprocess(p)
    np.testing.assert_array_equal(back["extras"]["sfh"]["sfr10"], sfh["sfr10"])
    np.testing.assert_array_equal(back["bestfit"]["prediction"]["spectra_model"], b["prediction"]["spectra_model"])
    assert back["meta"]["cosmology"]["name"] == "Planck18" and back["meta"]["windows_myr"] == list(WIN)


def test_nested_weights_are_recomputed_from_birth_contours(model):
    from ceridwen.sampler.ns_weights import nested_log_weights
    res = _fake_result(model, n=30, births=True)          # stored log_weights are random noise
    with pytest.warns(UserWarning, match="stored log_weights differ"):
        pp = PostProcess(model, res, n_samples=5, predictions=False, uv=False, ionizing=False)
    expect = nested_log_weights(np.asarray(res.log_likelihoods), np.asarray(res.log_likelihoods_birth))
    np.testing.assert_array_equal(pp.log_weights, expect)
    out = pp.run()
    assert out["meta"]["weights"].startswith("recomputed") and out["meta"]["resampled"] is True
    assert np.all(np.isfinite(expect[out["draw_index"]]))


def test_uniform_weights_use_all_samples(model):
    res = _fake_result(model, n=7, weighted=False)
    out = PostProcess(model, res, predictions=False).run()
    assert out["theta"]["logmass"].shape == (7,) and out["meta"]["resampled"] is False
    assert out["meta"]["weights"] == "stored"
    np.testing.assert_array_equal(out["draw_index"], np.arange(7))
    assert out["prediction"]["photometry"] == {}


def test_refuses_wrong_model_and_bad_inputs(model):
    res = _fake_result(model, n=5)
    bad = SamplingResult(samples={**res.samples, "extra": jnp.zeros(5)}, log_evidence=0.0, log_evidence_err=0.0,
                         log_weights=res.log_weights, log_likelihoods=res.log_likelihoods,
                         param_names=res.param_names + ["extra"], n_likelihood_calls=5, wall_time_s=0.0,
                         sampler_name="fake")
    with pytest.raises(ValueError, match="only in result"):
        PostProcess(model, bad)
    with pytest.raises(ValueError, match="windows_myr"):
        PostProcess(model, res, windows_myr=(0.0,))
    with pytest.raises(ValueError, match="ssfr=True needs"):
        PostProcess(model, res, sfr=False)
    with pytest.raises(TypeError, match="not callable"):
        PostProcess(model, res, derived={"x": 1})
    with pytest.raises(ValueError, match="scalar or 1-D"):
        PostProcess(model, res, n_samples=2, derived={"m": lambda s: np.ones((2, 2))}).run()
