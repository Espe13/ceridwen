"""Surviving stellar mass (SSP schema 3.0, v1.0.6).

* ``ssp_stellar_mass`` round-trips through save/load, is optional (schema-2 files load
  with None), is validated, stays out of the grid's content hash, and is promoted with a
  3-D file loaded as an alpha grid.
* ``CSPBasis.surviving_mass_fraction`` against an independent analytic integral for a
  constant SFH, in both SFH integration schemes.
* The FSPS reference ``examples/recipes/reference_mfrac.json``: the SSP table reproduces its
  single-burst values; the composite values differ by the documented SFH-integration
  difference between CERIDWEN and FSPS (bounded here at the measured size).
* ``PostProcess``: mfrac / mass_surviving / ssfrW_surviving, and the messages for a grid
  without the table: fetch the current copy of a published grid, rebuild one's own grid,
  never a repository script.

The FSPS masses of the test grids are stored in ``tests/reference/ssp_stellar_mass.npz``
(FSPS ``stellar_mass`` on the grids' nodes; each keyed by the grid's chash), so only the
last test needs FSPS.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import warnings

import h5py
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid, without_table, TEST_DATA_DIR, REPO_ROOT          # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                 # noqa: E402
from ceridwen.ssps import SSPDataAfe                                        # noqa: E402
from ceridwen.postprocess import PostProcess                                # noqa: E402
from ceridwen.sampler.runner import SamplingResult                          # noqa: E402

REF_TABLES = REPO_ROOT / "tests" / "reference" / "ssp_stellar_mass.npz"
REF_JSON = REPO_ROOT / "examples" / "recipes" / "reference_mfrac.json"
LN10 = math.log(10.0)


def _tiny(**kw):
    kw.setdefault("zsun", 0.01)
    return SSPData(jnp.array([-2.0, -1.0]), jnp.array([-1.0, 0.0, 1.0]),
                   jnp.linspace(1000.0, 10000.0, 5), jnp.ones((2, 3, 5)),
                   ssp_resolution=np.full(5, 50.0), **kw)


def _ref_table(grid):
    """The stored FSPS table for ``grid`` (matched by chash), or skip."""
    if not REF_TABLES.is_file():
        pytest.skip("tests/reference/ssp_stellar_mass.npz missing")
    with np.load(REF_TABLES) as z:
        for key in z.files:
            if key.endswith("/chash") and str(z[key]) == grid.chash:
                tag = key[:-len("/chash")]
                return tag, np.array(z[f"{tag}/mass"]), str(z[f"{tag}/source"])
    pytest.skip(f"no stored mass table for grid {grid.chash}")


# ------------------------------------------------------------------ schema / I/O -----
def test_schema3_roundtrip_and_chash(tmp_path):
    g = _tiny()
    m = np.array([[1.0, 0.8, 0.6], [1.0, 0.7, 0.5]])
    g3 = g.with_stellar_mass(m, source="unit test")
    assert g3.schema_version == "3.0"
    assert g3.chash == g.chash, "the mass table must not change the grid's identity"
    p = tmp_path / "g.h5"
    g3.save(p)
    back = SSPData.load(p)
    np.testing.assert_array_equal(back.ssp_stellar_mass, m)
    assert back.ssp_stellar_mass.dtype == np.float64
    assert back.stellar_mass_source == "unit test"
    assert back.schema_version == "3.0" and back.chash == g.chash
    np.testing.assert_array_equal(back.require_stellar_mass(), m)
    assert "surviving stellar mass" in back.display(return_str=True)


def test_schema2_file_loads_without_table(tmp_path):
    p = tmp_path / "old.h5"
    _tiny(schema_version="2.0").save(p)
    with h5py.File(p, "r") as f:
        assert "ssp_stellar_mass" not in f
    back = SSPData.load(p)
    assert back.ssp_stellar_mass is None and back.schema_version == "2.0"
    assert "not in this grid" in back.display(return_str=True)
    with pytest.raises(ValueError, match="from_fsps records the surviving-mass table") as e:
        back.require_stellar_mass()                   # not a published grid: rebuild it
    assert "scripts/" not in str(e.value)


@pytest.mark.parametrize("bad, match", [
    (np.ones((3, 2)), "shape"), (np.ones((2, 3, 1)), "shape"),
    (np.array([[1.0, np.nan, 0.5], [1.0, 0.7, 0.5]]), "finite and positive"),
    (np.array([[1.0, 0.0, 0.5], [1.0, 0.7, 0.5]]), "finite and positive")])
def test_bad_table_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        _tiny().with_stellar_mass(bad, source="x")
    with pytest.raises(ValueError, match=match):
        _tiny(ssp_stellar_mass=bad)


def test_table_above_formed_mass_rejected():
    """More mass cannot survive than was formed: a constructor refuses the table."""
    bad = np.array([[1.0, 4.62, 0.5], [1.0, 0.7, 0.5]])
    with pytest.raises(ValueError, match="more mass survives than was formed"):
        _tiny().with_stellar_mass(bad, source="x")
    with pytest.raises(ValueError, match="more mass survives than was formed"):
        _tiny(ssp_stellar_mass=bad)
    from ceridwen.ssps.ssp_data import STELLAR_MASS_MAX
    ok = np.array([[STELLAR_MASS_MAX, 0.9, 0.5], [1.0, 0.7, 0.5]])
    assert _tiny().with_stellar_mass(ok, source="x").ssp_stellar_mass.max() == STELLAR_MASS_MAX


def test_load_refuses_table_above_formed_mass(tmp_path):
    """A FILE whose table exceeds the bound (the first published mist_miles_chab: 4.62) loads
    without the table, warns once naming the file, and every mfrac path says why."""
    p = tmp_path / "g.h5"
    _tiny().with_stellar_mass(np.array([[1.0, 0.8, 0.6], [1.0, 0.7, 0.5]]), source="t").save(p)
    with h5py.File(p, "r+") as f:
        f["ssp_stellar_mass"][0, 0] = 4.62
    with pytest.warns(UserWarning, match="surviving stellar-mass table .* is refused") as rec:
        back = SSPData.load(p)
    msg = str(rec[0].message)
    assert str(p) in msg and "4.62" in msg and "mfrac=False" in msg
    assert "spectra and everything else are unaffected" in msg
    assert back.ssp_stellar_mass is None and back.stellar_mass_source is None
    assert "4.62" in back.stellar_mass_refused
    assert "REFUSED" in back.display(return_str=True)
    with pytest.raises(ValueError, match="refused when the grid was loaded"):
        back.require_stellar_mass()
    from ceridwen.ssps.ssp_data import missing_stellar_mass_message
    assert "refused" in missing_stellar_mass_message("x", refused=back.stellar_mass_refused)
    with pytest.warns(UserWarning, match="refused"):
        prom = SSPDataAfe.load(p, zsun=0.01)                 # the alpha loader shares _read_h5
    assert prom.ssp_stellar_mass is None and prom.stellar_mass_refused


def test_with_stellar_mass_needs_source():
    with pytest.raises(ValueError, match="source"):
        _tiny().with_stellar_mass(np.ones((2, 3)), source="")


def test_afe_grid_table_and_promotion(tmp_path):
    afe = SSPDataAfe(jnp.array([-2.0, -1.0]), jnp.array([-1.0, 0.0, 1.0]),
                     jnp.linspace(1000.0, 10000.0, 5), jnp.ones((2, 2, 3, 5)),
                     ssp_afe=jnp.array([0.0, 0.4]), ssp_resolution=np.full(5, 50.0),
                     zsun=0.01, axis_meaning="feh")
    m = np.linspace(0.5, 1.0, 12).reshape(2, 2, 3)
    a3 = afe.with_stellar_mass(m, source="t")
    assert a3.schema_version == "3.1"
    a3.save(tmp_path / "a.h5")
    np.testing.assert_array_equal(SSPDataAfe.load(tmp_path / "a.h5").ssp_stellar_mass, m)
    with pytest.raises(ValueError, match="shape"):
        afe.with_stellar_mass(m[0], source="t")
    # a 3-D (solar-scaled) file read as an alpha grid: table promoted with the flux
    m2 = np.array([[1.0, 0.8, 0.6], [1.0, 0.7, 0.5]])
    _tiny().with_stellar_mass(m2, source="t").save(tmp_path / "s.h5")
    prom = SSPDataAfe.load(tmp_path / "s.h5", zsun=0.01)
    np.testing.assert_array_equal(prom.ssp_stellar_mass, m2[None])


# ------------------------------------------------------------ the CSP mass fraction ---
def _grid_with_table():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    g = SSPData.load(str(path))
    tag, m, src = _ref_table(g)
    return g.with_stellar_mass(m, source=src), tag


def _csp(grid, lookback_gyr, sfh, interp):
    th = {"lookback_time": jnp.asarray(lookback_gyr), "sfh": jnp.asarray(sfh),
          "logzsol": jnp.array([0.0])}
    csp = CSPBasis(grid, theta=th, zh_const=True, sfh_interp=interp, add_neb=False,
                   add_dust=False, add_diffuse_dust=False, verbose=False,
                   cosmo=Cosmology.planck18())
    return csp, th


def _segment_integral(a, b, ma, mb, x0, x1):
    """int_{x0}^{x1} m(x) 10^x ln10 dx for m linear in x = log10 t between (a, ma), (b, mb)."""
    c1 = (mb - ma) / (b - a)
    c0 = ma - c1 * a
    F = lambda x: (c0 + c1 * x) * 10.0 ** x - c1 * 10.0 ** x / LN10   # noqa: E731
    return F(x1) - F(x0)


def _analytic_linear(logage_yr, m, t_edges_yr):
    """Constant SFR, CERIDWEN's piecewise-linear scheme written out independently: m is
    linear in log10 t between SSP nodes; each SFH bin keeps its own formed mass, spread over
    the part of the bin inside the node range; bins entirely outside it are dropped."""
    num = den = 0.0
    for lo, hi in zip(t_edges_yr[:-1], t_edges_yr[1:]):
        x_lo = math.log10(max(lo, 10.0 ** logage_yr[0]))
        x_hi = math.log10(min(hi, 10.0 ** logage_yr[-1]))
        if x_hi <= x_lo:
            continue
        integ = 0.0
        for j in range(len(logage_yr) - 1):
            a, b = logage_yr[j], logage_yr[j + 1]
            x0, x1 = max(a, x_lo), min(b, x_hi)
            if x1 > x0:
                integ += _segment_integral(a, b, m[j], m[j + 1], x0, x1)
        inside = 10.0 ** x_hi - 10.0 ** x_lo
        num += (hi - lo) * integ / inside
        den += hi - lo
    return num / den


def _analytic_step(logage_yr, m, T_yr):
    """Constant SFR on [0, T], piecewise-constant scheme: each SSP owns its Voronoi cell in
    linear age (first cell from 0, last cell extended by one spacing)."""
    t = 10.0 ** np.asarray(logage_yr)
    mid = 0.5 * (t[:-1] + t[1:])
    lo = np.concatenate([[0.0], mid])
    hi = np.concatenate([mid, [t[-1] + (t[-1] - t[-2])]])
    ov = np.clip(np.minimum(hi, T_yr) - np.maximum(lo, 0.0), 0.0, None)
    return float(np.sum(ov * m) / np.sum(ov))


@pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("interp", ["linear", "step"])
def test_constant_sfh_analytic(T, interp):
    grid, _ = _grid_with_table()
    lb = np.linspace(0.0, T, 41)
    csp, th = _csp(grid, lb, np.ones_like(lb), interp)
    iz = int(np.argmin(np.abs(grid.logzsol_axis)))
    m = grid.ssp_stellar_mass[iz]
    logage = np.asarray(grid.ssp_lg_age_gyr, dtype=np.float64) + 9.0
    got = float(csp.surviving_mass_fraction(th))
    want = (_analytic_linear(logage, m, lb * 1e9) if interp == "linear"
            else _analytic_step(logage, m, T * 1e9))
    assert got == pytest.approx(want, rel=1e-10, abs=0.0)


@pytest.mark.xfail(strict=True, reason=(
    "pre-existing defect of the 'linear' SFH scheme (csp._ssp_weights): jmax is clipped to "
    "n_ssp - 1 and the mask is j < jmax, so the OLDEST SSP node never gets weight; a bin "
    "reaching past the second-oldest node (BPASS 10^10.1 yr = 12.6 Gyr) loses it.  Measured "
    "mfrac -4.2e-4 relative at T = 13.8 Gyr, 10 nodes; 0.0 with jmax clipped to n_ssp.  A "
    "fix changes the forward model (all four tiers, Amanda's approval)."))
def test_linear_scheme_reaches_oldest_node():
    grid, _ = _grid_with_table()
    logage = np.asarray(grid.ssp_lg_age_gyr, dtype=np.float64) + 9.0
    T = 0.5 * (10.0 ** (logage[-2] - 9.0) + 10.0 ** (logage[-1] - 9.0))   # between the two oldest
    lb = np.linspace(0.0, T, 10)
    csp, th = _csp(grid, lb, np.ones_like(lb), "linear")
    iz = int(np.argmin(np.abs(grid.logzsol_axis)))
    want = _analytic_linear(logage, grid.ssp_stellar_mass[iz], lb * 1e9)
    assert float(csp.surviving_mass_fraction(th)) == pytest.approx(want, rel=1e-10, abs=0.0)


def test_mfrac_is_weighted_table_and_jit():
    grid, _ = _grid_with_table()
    lb = np.linspace(0.0, 5.0, 6)
    csp, th = _csp(grid, lb, np.array([3.0, 1.0, 0.5, 2.0, 1.0, 0.2]), "step")
    W = np.asarray(csp.calculate_ssp_weights(th), dtype=np.float64)
    iz = int(np.argmin(np.abs(grid.logzsol_axis)))
    want = float(np.sum(W[iz] * grid.ssp_stellar_mass[iz]) / np.sum(W))
    assert float(jax.jit(csp.surviving_mass_fraction)(th)) == pytest.approx(want, rel=1e-12,
                                                                            abs=0.0)
    g = jax.grad(lambda s: csp.surviving_mass_fraction(dict(th, sfh=s)))(th["sfh"])
    assert np.all(np.isfinite(np.asarray(g)))


def test_mfrac_without_table_raises():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    grid = without_table(SSPData.load(str(path)))
    csp, th = _csp(grid, np.linspace(0.0, 1.0, 3), np.ones(3), "step")
    from ceridwen.ssps.ssp_data import published_grid_name
    name = published_grid_name(grid.chash)
    want = (f"fetch_grid\\('{name}', force=True\\)" if name else "from_fsps records")
    with pytest.raises(ValueError, match=want) as e:
        csp.surviving_mass_fraction(th)
    assert "scripts/" not in str(e.value)


# --------------------------------------------------- the FSPS reference (recipe json) --
def _ref_json():
    return json.loads(REF_JSON.read_text())


# SSP level: FSPS's stellar_mass at the age nodes 10^8, 10^9, 10^10 yr.  The json's
# "burst" entries were evaluated by FSPS at tage = T (interpolated between nodes), the table
# at the nodes: identical for MIST, within 5.7e-10 relative for BPASS (json 0.77799999956
# vs node 0.778 at 1 Gyr, measured 2026-09-22).
BURST_RTOL = {"mist": 1e-10, "bpass": 1e-9}

# Composite level: CERIDWEN integrates the SFH over the SSP ages with its own schemes
# (csp._ssp_weights), not FSPS's csp_gen, and drops mass formed more recently than the
# youngest SSP node in the "linear" scheme.  Measured max |ceridwen/FSPS - 1| over
# T = 0.1, 1, 10 Gyr and the constant / rising SFHs (2026-09-22): linear 6.3e-3 (MIST,
# T = 0.1 Gyr, youngest node 10^5 yr with m = 2.63) / 1.8e-3 (BPASS); step 1.0e-4 (MIST) /
# 4.2e-4 (BPASS).  Bounded here at those sizes (x1.2) so a change is noticed.
CSP_RTOL = {("mist", "linear"): 7.6e-3, ("mist", "step"): 1.3e-4,
            ("bpass", "linear"): 2.2e-3, ("bpass", "step"): 5.1e-4}


def _tables_by_library():
    if not REF_TABLES.is_file():
        pytest.skip("tests/reference/ssp_stellar_mass.npz missing")
    out = {}
    with np.load(REF_TABLES) as z:
        for key in z.files:
            if key.endswith("/mass"):
                tag = key[:-len("/mass")]
                out[str(z[f"{tag}/library"])] = (tag, np.array(z[key]),
                                                 np.array(z[f"{tag}/log_age_gyr"]),
                                                 np.array(z[f"{tag}/logzsol"]))
    return out


@pytest.mark.parametrize("lib", ["mist", "bpass"])
def test_table_reproduces_reference_burst(lib):
    tabs = _tables_by_library()
    if lib not in tabs:
        pytest.skip(f"no stored {lib} table")
    _tag, m, lgage, lz = tabs[lib]
    iz = int(np.argmin(np.abs(lz)))
    assert abs(lz[iz]) < 1e-12
    ref = _ref_json()["libraries"][lib]["mfrac"]
    for T in ("0.1", "1.0", "10.0"):
        ia = int(np.argmin(np.abs(lgage - math.log10(float(T)))))
        assert abs(lgage[ia] - math.log10(float(T))) < 1e-9
        assert m[iz, ia] == pytest.approx(ref[T]["burst"]["mfrac"], rel=BURST_RTOL[lib], abs=0.0)


def test_mist_table_is_fsps_except_the_truncated_isochrones():
    """The stored MIST table (as published after the 2026-09-24 correction) equals FSPS's raw
    stellar_mass bit for bit at every age >= 10^6.45 yr; below, where FSPS reaches 4.62, it
    lies in [0.97, STELLAR_MASS_MAX] (nothing has died yet: ~1 minus wind losses)."""
    from ceridwen.ssps.ssp_data import STELLAR_MASS_MAX
    if not REF_TABLES.is_file():
        pytest.skip("tests/reference/ssp_stellar_mass.npz missing")
    with np.load(REF_TABLES) as z:
        if "mist_miles/mass_fsps" not in z.files:
            pytest.skip("no raw FSPS MIST table stored")
        fix, raw = np.array(z["mist_miles/mass"]), np.array(z["mist_miles/mass_fsps"])
        lgyr = np.array(z["mist_miles/log_age_gyr"]) + 9.0
    old = lgyr >= 6.45 - 1e-9
    np.testing.assert_array_equal(fix[:, old], raw[:, old])
    assert raw[:, ~old].max() > 4.6 and (raw > 1.0).sum() == 360
    young = fix[:, ~old]
    assert young.min() >= 0.97 and fix.max() <= STELLAR_MASS_MAX


@pytest.mark.parametrize("interp", ["linear", "step"])
def test_csp_matches_reference_within_scheme_difference(interp):
    """The json's constant and rising SFHs (201-node FSPS tables) through CSPBasis on the
    canonical test grid, at logzsol = 0."""
    grid, _tag = _grid_with_table()
    lib = {"bpss": "bpass", "mist": "mist"}.get(grid.isoc_type)
    ref = _ref_json()["libraries"].get(lib)
    if ref is None:
        pytest.skip(f"no json reference for isochrones {grid.isoc_type}")
    # FSPS's composite used FSPS's raw table; on MIST that exceeds 1 below 10^6.45 yr and the
    # grid's table is corrected there (ceridwen.ssps.stellar_mass), which lowers mfrac by
    # 0.9 % (constant) / 1.8 % (rising) at T = 0.1 Gyr.  This test checks the SFH integration
    # against FSPS, so it puts FSPS's raw table on the CSP (bypassing the grid's bound).
    with np.load(REF_TABLES) as z:
        raw = np.array(z[f"{_tag}/mass_fsps"]) if f"{_tag}/mass_fsps" in z.files else None
    worst = 0.0
    for T in (0.1, 1.0, 10.0):
        t = np.linspace(0.0, T, 201)
        for name, sfr in (("constant", np.ones_like(t)), ("rising", t / T)):
            lb, s = (T - t)[::-1], sfr[::-1]         # forward time -> lookback, index 0 = today
            csp, th = _csp(grid, lb, s, interp)
            if raw is not None:
                csp.ssp_stellar_mass = raw
            got = float(csp.surviving_mass_fraction(th))
            worst = max(worst, abs(got / ref["mfrac"][f"{T:.1f}"][name]["mfrac"] - 1.0))
    assert worst < CSP_RTOL[(lib, interp)], worst


# ------------------------------------------------------------------- PostProcess -------
@pytest.fixture(scope="module")
def pp_models():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    plain = without_table(SSPData.load(str(path)))
    _tag, m, src = _ref_table(plain)
    withm = plain.with_stellar_mass(m, source=src)
    cosmo = Cosmology.planck18()
    out = {}
    for key, g in (("plain", plain), ("mass", withm)):
        csp = CSPBasis(g, lookback_time=jnp.linspace(0.0, 5.0, 5), zh_const=True,
                       sfh_interp="step", add_neb=False, add_dust=False,
                       add_diffuse_dust=False, verbose=False, cosmo=cosmo)
        out[key] = SedModel(csp, [], priors={}, zred=0.0,
                            free_param_init={"logmass": jnp.array([10.0])})
    return out


def _result(model, n=16, seed=2):
    rng = np.random.default_rng(seed)
    samples = {}
    for name in model.param_names:
        base = np.asarray(model.theta_init[name], dtype=float)
        d = np.abs(base[None, :] + 0.3 * rng.standard_normal((n,) + base.shape))
        samples[name] = jnp.asarray(d[:, 0] if base.shape == (1,) else d)
    return SamplingResult(samples=samples, log_evidence=float("nan"),
                          log_evidence_err=float("nan"), log_weights=jnp.zeros(n),
                          log_likelihoods=jnp.asarray(rng.standard_normal(n)),
                          param_names=list(model.param_names), n_likelihood_calls=n,
                          wall_time_s=0.0, sampler_name="fake")


def test_postprocess_surviving_mass(pp_models):
    model = pp_models["mass"]
    res = _result(model)
    out = PostProcess(model, res, uv=False, ionizing=False, predictions=False,
                      windows_myr=(10.0, 100.0)).run()
    blk = out["extras"]["sfh"]
    n = blk["mass_formed"].shape[0]
    for i in range(n):
        th = {k: jnp.asarray(np.asarray(out["theta"][k])[i]).reshape(
            np.shape(model.theta_init[k])) for k in model.param_names}
        t = model.apply_transforms(th)
        want = float(model.csp.surviving_mass_fraction(t))
        assert blk["mfrac"][i] == pytest.approx(want, rel=1e-12, abs=0.0)
    assert np.all((blk["mfrac"] > 0.3) & (blk["mfrac"] < 1.0))
    np.testing.assert_array_equal(blk["mass_surviving"], blk["mfrac"] * blk["mass_formed"])
    for w in ("10", "100"):
        np.testing.assert_array_equal(blk[f"ssfr{w}_surviving"],
                                      blk[f"sfr{w}"] / blk["mass_surviving"])
        np.testing.assert_array_equal(blk[f"ssfr{w}"], blk[f"sfr{w}"] / blk["mass_formed"])
    assert out["meta"]["mfrac"].startswith("computed")


def test_postprocess_without_table(pp_models):
    model = pp_models["plain"]
    res = _result(model)
    with pytest.warns(UserWarning, match="mfrac unavailable: .* no surviving-mass table") as w:
        pp = PostProcess(model, res, uv=False, ionizing=False, predictions=False)
    assert all("scripts/" not in str(x.message) and "fetch_grid" not in str(x.message)
               for x in w)                           # one line, no command
    blk = pp.run()["extras"]["sfh"]
    assert "mfrac" not in blk and "mass_surviving" not in blk and "mass_formed" in blk
    with pytest.raises(ValueError, match="surviving stellar-mass table"):
        PostProcess(model, res, mfrac=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PostProcess(model, res, mfrac=False, uv=False, ionizing=False, predictions=False)
    with pytest.raises(TypeError, match="mfrac"):
        PostProcess(model, res, mfrac="yes")


# ------------------------------------------------------------------------- FSPS -------
@pytest.mark.fsps
def test_stored_table_is_fsps_stellar_mass():
    """One metallicity row of the stored table recomputed with this environment's FSPS
    (only for the library it is compiled with)."""
    fsps = pytest.importorskip("fsps")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set")
    sp = fsps.StellarPopulation(zcontinuous=0, sfh=0, imf_type=1)
    iso = sp.libraries[0].decode() if isinstance(sp.libraries[0], bytes) else sp.libraries[0]
    lib = {"bpss": "bpass"}.get(iso, iso)
    tabs = _tables_by_library()
    if lib not in tabs:
        pytest.skip(f"no stored table for this FSPS's isochrones {iso!r}")
    _tag, m, _lgage, lz = tabs[lib]
    iz = int(np.argmin(np.abs(lz)))
    sp.get_spectrum(tage=0.0, zmet=iz + 1, peraa=False)
    np.testing.assert_array_equal(np.asarray(sp.stellar_mass, dtype=np.float64), m[iz])
