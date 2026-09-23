"""``fitSED`` writes ``/derived/mfrac`` (2026-09-23).

* A short nested fit of mock photometry (the quickstart's model: logsfr_ratios transform,
  diffuse dust) on the test grid with its FSPS mass table (``tests/reference``, by chash)
  writes one mfrac per stored sample, with the table / grid / SFH-scheme attributes.
* The stored values equal ``PostProcess``'s recomputation to 1e-12 and ``mfrac * 10**logmass``
  reproduces its ``mass_surviving``; ``PostProcess`` on the file reads the stored array (it
  does not need the table), and refuses one recorded on another grid or SFH scheme.
* A grid without the table: no ``/derived`` group, one log line, no warning from fitSED, and
  PostProcess's single warning afterwards.  ``mfrac=True`` refuses before sampling,
  ``mfrac=False`` skips.
"""
from __future__ import annotations

import logging
import pathlib
import sys
import warnings

import h5py
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid, REPO_ROOT                          # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology, fitSED, read_result_h5  # noqa: E402
from ceridwen.fit import read_derived_h5, load_result_h5                    # noqa: E402
from ceridwen.model import logsfr_ratios_to_sfh                             # noqa: E402
from ceridwen.observation import Photometry                                 # noqa: E402
from ceridwen.postprocess import PostProcess                                # noqa: E402
from ceridwen.priors import Uniform, StudentT                               # noqa: E402

REF_TABLES = REPO_ROOT / "tests" / "reference" / "ssp_stellar_mass.npz"
FILTERS = ["galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0", "twomass_Ks"]
N_TIME = 5
NS = {"num_live": 30, "num_delete": 6, "num_inner_steps": 6, "logZ_tol": -0.5}


def _grids():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    plain = SSPData.load(str(path))
    if not REF_TABLES.is_file():
        pytest.skip("tests/reference/ssp_stellar_mass.npz missing")
    with np.load(REF_TABLES) as z:
        tag = next((k[:-len("/chash")] for k in z.files
                    if k.endswith("/chash") and str(z[k]) == plain.chash), None)
        if tag is None:
            pytest.skip(f"no stored mass table for grid {plain.chash}")
        withm = plain.with_stellar_mass(np.array(z[f"{tag}/mass"]), source=str(z[f"{tag}/source"]))
    return plain, withm


def _model(grid, sfh_interp="step"):
    lookback = jnp.linspace(0.0, 13.0, N_TIME)
    csp = CSPBasis(grid, theta={"lookback_time": lookback, "sfh": jnp.ones(N_TIME),
                                "logzsol": jnp.array([-0.2])},
                   zh_const=True, sfh_interp=sfh_interp, add_dust=False, add_diffuse_dust=True,
                   add_neb=False, verbose=False, cosmo=Cosmology.planck18())
    t_yr = np.array(csp.sfh_times)
    sfh_true = logsfr_ratios_to_sfh(jnp.array([0.3, 0.2, -0.1, -0.5]), sfh_times_yr=t_yr)
    tmp = Photometry(filters=FILTERS, name="_tmp")
    tmp.setup_for_model(csp.wave)
    unit = csp.get_spectrum({"sfh": sfh_true, "logzsol": jnp.array([-0.2]),
                             "diffuse_tau_kc": jnp.array([0.5]),
                             "diffuse_dust_index": jnp.array([-0.7])})
    flux = np.array(tmp.predict(unit, csp.wave)) * 10.0 ** 10.5
    phot = Photometry(filters=FILTERS, flux=flux, uncertainty=flux / 10.0, name="phot")

    def to_sfh(th, _t=t_yr):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)

    return SedModel(csp, [phot], transforms={"sfh": to_sfh},
                    priors={"logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
                            "logzsol": Uniform(low=-1.0, high=0.2),
                            "logmass": Uniform(low=10.0, high=11.0),
                            "diffuse_tau_kc": Uniform(low=0.0, high=2.0),
                            "diffuse_dust_index": Uniform(low=-1.0, high=0.4)},
                    free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                                     "logmass": jnp.array([10.5])},
                    broaden_photometry=False)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    plain, withm = _grids()
    out = tmp_path_factory.mktemp("derived")
    model = _model(withm)
    result = fitSED(model, output_dir=out, rng_key=jax.random.PRNGKey(5), verbose=False,
                    sampler_kwargs=NS)
    return model, result, out / "ceridwen_result.h5", plain


def test_fit_writes_derived_mfrac(fitted):
    model, result, path, _plain = fitted
    n = int(np.asarray(result.log_likelihoods).shape[0])
    with h5py.File(path, "r") as f:
        assert list(f["derived"]) == ["mfrac"]
        d = f["derived/mfrac"]
        assert d.shape == (n,) and d.dtype == np.float64
        a = dict(d.attrs)
    assert a["grid_chash"] == model.csp.grid_chash
    assert a["sfh_interp"] == "step"
    assert a["stellar_mass_source"] == model.csp.stellar_mass_source
    assert a["stellar_mass_source"].startswith("FSPS StellarPopulation.stellar_mass")
    r = read_result_h5(path)
    np.testing.assert_array_equal(r["derived"]["mfrac"], read_derived_h5(path)["mfrac"])
    assert r["derived"]["mfrac_attrs"]["grid_chash"] == model.csp.grid_chash
    assert not hasattr(load_result_h5(path), "derived")        # load_result_h5 unchanged
    m = r["derived"]["mfrac"]
    assert np.all(np.isfinite(m)) and np.all((m > 0.4) & (m < 1.0))


def test_stored_mfrac_equals_postprocess(fitted):
    model, result, path, _plain = fitted
    stored = read_derived_h5(path)["mfrac"]
    # recomputed by PostProcess from the table (a SamplingResult, no file)
    out = PostProcess(model, result, n_samples=400, uv=False, ionizing=False,
                      predictions=False, windows_myr=(100.0,)).run()
    assert out["meta"]["mfrac"].startswith("computed")
    idx = out["draw_index"]
    assert np.unique(idx).size > 20
    blk = out["extras"]["sfh"]
    np.testing.assert_allclose(blk["mfrac"], stored[idx], rtol=1e-12, atol=0.0)
    logmass = np.asarray(result.samples["logmass"], dtype=float).reshape(-1)[idx]
    np.testing.assert_allclose(stored[idx] * 10.0 ** logmass, blk["mass_surviving"],
                               rtol=1e-12, atol=0.0)


def test_postprocess_reads_stored_array(fitted):
    model, _result, path, plain = fitted
    pp = PostProcess(model, path, n_samples=50, uv=False, ionizing=False, predictions=False)
    out = pp.run()
    assert out["meta"]["mfrac"] == "read from the result file's /derived/mfrac"
    stored = read_derived_h5(path)["mfrac"]
    np.testing.assert_array_equal(out["extras"]["sfh"]["mfrac"], stored[out["draw_index"]])
    np.testing.assert_array_equal(out["bestfit"]["extras"]["sfh"]["mfrac"],
                                  stored[out["bestfit"]["index"]])
    # the table is not needed once the file has the array: no warning with the plain grid
    plain_model = _model(plain)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o2 = PostProcess(plain_model, path, n_samples=50, uv=False, ionizing=False,
                         predictions=False).run()
    np.testing.assert_array_equal(o2["extras"]["sfh"]["mfrac"], out["extras"]["sfh"]["mfrac"])


def test_postprocess_refuses_foreign_stored_mfrac(fitted, tmp_path):
    model, _result, path, _plain = fitted
    for key, bad in (("grid_chash", "chash-v1:" + "0" * 64), ("sfh_interp", "linear")):
        p = tmp_path / f"{key}.h5"
        p.write_bytes(path.read_bytes())
        with h5py.File(p, "a") as f:
            f["derived/mfrac"].attrs[key] = bad
        with pytest.raises(ValueError, match=f"/derived/mfrac was computed with {key}"):
            PostProcess(model, p, predictions=False)
        PostProcess(model, p, predictions=False, mfrac=False)      # opting out still works


def test_grid_without_table(tmp_path, caplog):
    plain, _withm = _grids()
    model = _model(plain)
    with pytest.raises(ValueError, match="surviving stellar-mass table"):
        fitSED(model, output_dir=tmp_path / "x", mfrac=True, verbose=False, sampler_kwargs=NS)
    assert not (tmp_path / "x" / "ceridwen_result.h5").exists()     # refused before sampling
    with pytest.raises(TypeError, match="mfrac"):
        fitSED(model, output_dir=tmp_path / "x", mfrac="yes", verbose=False)

    caplog.set_level(logging.INFO, logger="ceridwen")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fitSED(model, output_dir=tmp_path, rng_key=jax.random.PRNGKey(5), verbose=False,
               sampler_kwargs=NS)
    lines = [r.getMessage() for r in caplog.records if "Derived mfrac" in r.getMessage()]
    assert len(lines) == 1 and "no surviving-mass table" in lines[0]
    path = tmp_path / "ceridwen_result.h5"
    with h5py.File(path, "r") as f:
        assert "derived" not in f
    assert "derived" not in read_result_h5(path) and read_derived_h5(path) == {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PostProcess(model, path, predictions=False)
    assert len([x for x in w if "surviving stellar-mass table" in str(x.message)]) == 1


def test_mfrac_false_skips(tmp_path):
    _plain, withm = _grids()
    fitSED(_model(withm), output_dir=tmp_path, rng_key=jax.random.PRNGKey(5), verbose=False,
           sampler_kwargs=NS, mfrac=False)
    with h5py.File(tmp_path / "ceridwen_result.h5", "r") as f:
        assert "derived" not in f
