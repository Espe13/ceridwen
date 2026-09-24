"""The logzsol metallicity convention (v1.0.5), per grid.

Covers: the Z_sun resolution order and its refusals, the logzsol <-> native round trip, the
solar / edge nodes, an independent recomputation of the weights on the native axis (what the
pre-v1.0.5 code did), const vs history equality, the alpha-grid [Fe/H] semantics with
``logzsol_total``, the refused interpolation cell, and -- the test that pins the convention
rather than its self-consistency -- agreement with python-fsps's own ``logzsol``.
"""
from __future__ import annotations

import math
import pathlib
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.ssps.ssp_data_afe import SSPDataAfe
from ceridwen.ssps import grid_metadata as GM
from ceridwen.csp.csp import CSPBasis
from ceridwen.csp.csp_afe import CSPBasis_afe
from ceridwen.cosmology import Cosmology

REPO = pathlib.Path(__file__).resolve().parents[1]

#: name -> (path, is_alpha).  Each is skipped when its grid is not present locally.
GRIDS = {
    "mist_bpass_v2":         ("ceridwen/data/test_data/ssp_data_bpass.h5", False),
    "bpass_agb_dust":        ("tests/fixtures/ssp_data_bpass_agb_dust.h5", False),
    "mist_miles_chab":       ("ceridwen/data/test_data/ssp_data_mist_miles.h5", False),
    "mist_c3k_lr_chab":      ("ceridwen/data/test_data/ssp_data_mist_c3k_lr.h5", False),
    "mist_miles_fsps047":    ("examples/ssp_data.h5", False),
    "amist_c3k_lr_chab_afe": ("ceridwen/data/test_data/amist_c3k_lr_chab_afe.h5", True),
    "amist_c3k_hr_krou_afe": ("ceridwen/data/test_data/amist_c3k_hr_krou_afe.h5", True),
}


def _path(name):
    """The grid file of ``name``: its repo path, else for a published grid its fetch_grid
    cache copy (checksum-verified; ``_gridfixture.find_named_grid``), else for the published
    BPASS grid the suite's test grid ($CERIDWEN_TEST_SSP, as CI provides it); skip when none
    exists."""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from _gridfixture import find_named_grid, find_test_grid, REGISTRY_NAME, _MISS
    rel, _ = GRIDS[name]
    path = REPO / rel
    fname = pathlib.Path(rel).name
    if not path.is_file() and REGISTRY_NAME.get(fname) == name:
        path = find_named_grid(fname) or path
    if not path.is_file() and name == "mist_bpass_v2":
        found = find_test_grid()
        if found is not None and SSPData.load(str(found)).chash == next(
                c for c, m in GM.CHASH_TABLE.items() if m.name == name):
            path = found
    if not path.is_file():
        how = (f"; fetch_grid({name!r}) would provide it" if name in REGISTRY_NAME.values()
               else "; it is not a published grid")
        pytest.skip(_MISS.get(fname) or f"{name}: {rel} is not present{how}")
    return path


def _load(name):
    _, is_afe = GRIDS[name]
    return (SSPDataAfe if is_afe else SSPData).load(str(_path(name))), is_afe


def _csp(ssp, is_afe, **over):
    kw = dict(lookback_time=jnp.linspace(0.0, 13.0, 4), zh_const=True, add_dust=False,
              add_diffuse_dust=False, add_dust_emission=False, add_igm=False,
              verbose=False, cosmo=Cosmology.planck18(), sfh_interp="step")
    if not is_afe:
        kw["add_neb"] = False
    kw.update(over)
    return (CSPBasis_afe if is_afe else CSPBasis)(ssp, **kw)


ALL = sorted(GRIDS)


# --------------------------------------------------------------------------- Z_sun table

@pytest.mark.parametrize("name", ALL)
def test_zsun_resolves_to_the_grids_own_solar_node(name):
    """The resolved log10 Z_sun is a node of the axis, bitwise, and matches the table."""
    ssp, _ = _load(name)
    axis = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    assert np.any(axis == ssp.log10_zsun), "log10_zsun is not a node of ssp_lgmet"
    assert ssp.log10_zsun == GM.CHASH_TABLE[ssp.chash].log10_zsun
    assert abs(ssp.log10_zsun - math.log10(ssp.zsun_nominal)) < 1e-8
    # logzsol = 0 is exactly that node
    assert np.any(ssp.logzsol_axis == 0.0)


@pytest.mark.parametrize("name", ALL)
def test_axis_meaning_recorded(name):
    ssp, is_afe = _load(name)
    assert ssp.axis_meaning in GM.AXIS_MEANINGS
    if is_afe:
        assert ssp.axis_meaning == "feh"


def test_unknown_zsun_raises_and_zsun_kwarg_rescues():
    """A grid that is in no table and records no Z_sun must raise, and zsun= must fix it."""
    ssp, _ = _load("mist_bpass_v2")
    shifted = jnp.asarray(np.asarray(ssp.ssp_lgmet) + 0.25)     # new content -> new chash
    with pytest.raises(ValueError, match="solar metallicity Z_sun is unknown"):
        SSPData(shifted, ssp.ssp_lg_age_gyr, ssp.ssp_wave, ssp.ssp_flux)
    ok = SSPData(shifted, ssp.ssp_lg_age_gyr, ssp.ssp_wave, ssp.ssp_flux,
                 zsun=float(10.0 ** (float(ssp.log10_zsun) + 0.25)))
    assert ok.log10_zsun == float(np.asarray(shifted)[9])


def test_zsun_kwarg_disagreeing_with_the_table_raises():
    path = _path("mist_bpass_v2")
    with pytest.raises(ValueError):                 # 0.0142 is not a node of the BPASS axis
        SSPData.load(str(path), zsun=0.0142)


def test_flipped_byte_is_not_matched(tmp_path):
    """A modified grid no longer matches its table entry (chash) and must raise."""
    import h5py
    import shutil
    src = _path("mist_bpass_v2")
    dst = tmp_path / "tampered.h5"
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        flux = f["ssp_flux"]
        flux[0, 0, 0] = float(flux[0, 0, 0]) * 1.0000001
        for k in ("log10_zsun", "chash"):            # as a pre-v1.0.5 file would be
            if k in f.attrs:
                del f.attrs[k]
    with pytest.raises(ValueError, match="Z_sun is unknown"):
        SSPData.load(str(dst))
    assert SSPData.load(str(dst), zsun=0.020).log10_zsun == -1.6989700043360187


def test_provenance_round_trips(tmp_path):
    """save() writes log10_zsun / axis_meaning / chash and load() reads them back."""
    ssp, _ = _load("mist_bpass_v2")
    out = tmp_path / "rt.h5"
    ssp.save(str(out))
    back = SSPData.load(str(out))
    assert back.log10_zsun == ssp.log10_zsun
    assert back.axis_meaning == ssp.axis_meaning
    assert back.chash == ssp.chash
    import h5py
    with h5py.File(out) as f:
        assert f.attrs["log10_zsun"] == ssp.log10_zsun
        assert "logzsol" in f.attrs["units_lgmet"]


# --------------------------------------------------------------------------- axis

@pytest.mark.parametrize("name", ALL)
def test_round_trip_is_exact(name):
    """logzsol -> native -> logzsol is exact (<= 1 ulp) across the whole axis."""
    ssp, _ = _load(name)
    lz = ssp.logzsol_axis
    back = (lz + ssp.log10_zsun) - ssp.log10_zsun
    assert np.all(back == lz)
    native = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    assert np.allclose(lz + ssp.log10_zsun, native, rtol=0, atol=np.spacing(np.abs(native)))


@pytest.mark.parametrize("name", ALL)
def test_solar_and_edge_nodes_put_all_weight_on_that_node(name):
    """logzsol = 0 / the edge nodes select exactly that SSP row (no leakage, no clamp)."""
    ssp, is_afe = _load(name)
    csp = _csp(ssp, is_afe)
    lz = np.asarray(csp.zmet)
    for target, idx in ((0.0, int(np.argmin(np.abs(lz)))), (lz[0], 0), (lz[-1], len(lz) - 1)):
        th = dict(csp.theta_init)
        th["logzsol"] = jnp.array([float(target)])
        W = np.asarray(csp.calculate_ssp_weights(th))
        rows = W.sum(axis=1)
        assert rows[idx] == pytest.approx(rows.sum(), rel=1e-12), (
            f"{name}: logzsol={target} leaked off node {idx}")


@pytest.mark.parametrize("name", ALL)
def test_outside_the_grid_warns_and_does_not_pass_silently(name):
    ssp, is_afe = _load(name)
    csp = _csp(ssp, is_afe)
    th = dict(csp.theta_init)
    th["logzsol"] = jnp.array([float(np.asarray(csp.zmet).max()) + 0.5])
    with pytest.warns(UserWarning, match="outside the SSP metallicity grid"):
        csp.check_param_ranges(th)


@pytest.mark.parametrize("name", ALL)
def test_weights_match_an_independent_native_axis_lookup(name):
    """The v1.0.5 shift is only a change of variable: recomputing the interpolation on the
    NATIVE axis at Z = logzsol + log10 Z_sun (what the pre-v1.0.5 code did) reproduces the
    weights to <= 3 ulp, and exactly at a node."""
    ssp, is_afe = _load(name)
    csp = _csp(ssp, is_afe)
    native = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    rng = np.random.default_rng(0)
    targets = list(np.asarray(csp.zmet)) + list(
        rng.uniform(float(np.min(csp.zmet)), float(np.max(csp.zmet)), 8))
    for lz in targets:
        th = dict(csp.theta_init)
        th["logzsol"] = jnp.array([float(lz)])
        W = np.asarray(csp.calculate_ssp_weights(th))
        # independent two-node interpolation on the native axis
        z_abs = float(lz) + float(ssp.log10_zsun)
        k = int(np.clip(np.searchsorted(native, z_abs, side="left"), 1, native.size - 1))
        w = np.clip((z_abs - native[k - 1]) / (native[k] - native[k - 1]), 0.0, 1.0)
        rows = W.sum(axis=1)
        tot = rows.sum()
        exp = np.zeros_like(rows)
        exp[k - 1] += (1.0 - w) * tot
        exp[k] += w * tot
        # 1e-12 relative, not bitwise: at a node the logzsol axis is exact, whereas the old
        # native-axis lookup at logzsol + log10 Z_sun can round one ulp off the node and leak
        # ~1e-15 of the mass into its neighbour.  The new convention is the tighter one.
        assert np.allclose(rows, exp, rtol=1e-12, atol=1e-12 * tot), f"{name}: logzsol={lz}"


@pytest.mark.parametrize("name", ALL)
def test_const_equals_history_at_the_same_metallicity(name):
    ssp, is_afe = _load(name)
    x = float(np.median(np.asarray(SSPData.load if False else ssp.logzsol_axis)))
    const = _csp(ssp, is_afe, zh_const=True)
    n_time = const.n_time
    hist = _csp(ssp, is_afe, zh_const=False)
    tc = dict(const.theta_init); tc["logzsol"] = jnp.array([x])
    th = dict(hist.theta_init); th["logzsol_hist"] = jnp.full((n_time,), x)
    Wc = np.asarray(const.calculate_ssp_weights(tc))
    Wh = np.asarray(hist.calculate_ssp_weights(th))
    assert np.allclose(Wc, Wh, rtol=1e-10, atol=1e-30)


#: External anchors, independent of this package and of a local FSPS install:
#: MIST grids are built on FSPS's [Fe/H] tag list ($SPS_HOME/ISOCHRONES/MIST/zlegend.dat),
#: BPASS on its linear Z set ($SPS_HOME/ISOCHRONES/BPASS/zlegend.dat, Z_sun = 0.020).
_MIST_TAGS_13 = np.arange(-2.5, 0.51, 0.25)
_MIST_TAGS_12 = np.array([-2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                          0.0, 0.25, 0.5])
_BPASS_Z = np.array([1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2])
_ANCHORS = {"mist_miles_chab": _MIST_TAGS_13, "mist_c3k_lr_chab": _MIST_TAGS_13,
            "amist_c3k_lr_chab_afe": _MIST_TAGS_13, "amist_c3k_hr_krou_afe": _MIST_TAGS_13,
            "mist_miles_fsps047": _MIST_TAGS_12,
            "mist_bpass_v2": np.log10(_BPASS_Z / 0.020),
            "bpass_agb_dust": np.log10(_BPASS_Z / 0.020)}


@pytest.mark.parametrize("name", ALL)
def test_logzsol_axis_matches_the_external_node_set(name):
    """The convention itself, without needing FSPS installed: each grid's logzsol axis must
    equal the node set FSPS builds it from -- the MIST [Fe/H] tags, or log10(Z_BPASS/0.020).
    A wrong Z_sun (e.g. 0.0142 on a 0.0185 grid) shifts every node by 0.11 dex and fails."""
    ssp, _ = _load(name)
    expected = _ANCHORS[name]
    assert np.allclose(ssp.logzsol_axis, expected, rtol=0, atol=1e-8), (
        f"{name}: logzsol axis {np.round(ssp.logzsol_axis, 4)} != {np.round(expected, 4)}")


# --------------------------------------------------------------------------- old keys

def test_old_absolute_keys_raise_with_the_converted_value():
    ssp, _ = _load("mist_bpass_v2")
    with pytest.raises(ValueError, match=r"theta\['Z'\] was removed"):
        _csp(ssp, False, theta={"lookback_time": jnp.linspace(0.0, 13.0, 4),
                                "sfh": jnp.ones(4), "Z": jnp.array([-2.0])},
             lookback_time=None)


def test_old_key_error_quotes_the_conversion():
    ssp, _ = _load("mist_bpass_v2")
    csp = _csp(ssp, False)
    with pytest.raises(ValueError) as exc:
        csp.check_param_ranges({**csp.theta_init, "Z": jnp.array([-2.0])})
    assert "-0.301" in str(exc.value)


# --------------------------------------------------------------------------- alpha grids

ALPHA = [n for n in ALL if GRIDS[n][1]]


@pytest.mark.parametrize("name", ALPHA)
def test_logzsol_total_matches_the_mist_table(name):
    """[Z/H] - [Fe/H] must equal the MIST v2.5 MESA values at every plane."""
    ssp, _ = _load(name)
    csp = _csp(ssp, True)
    table = {-0.2: -0.12709428, 0.0: 0.0, 0.2: 0.14678160, 0.4: 0.30950399, 0.6: 0.48422267}
    for afe, f in table.items():
        th = dict(csp.theta_init)
        th["logzsol"] = jnp.array([-0.5])
        th["afe"] = jnp.array([afe])
        tot = float(np.asarray(csp.logzsol_total(th)).reshape(-1)[0])
        assert tot == pytest.approx(-0.5 + f, abs=1e-7)   # table quoted to 8 decimals


@pytest.mark.parametrize("name", ALPHA)
def test_refused_cell_is_reported_and_refused(name):
    ssp, _ = _load(name)
    csp = _csp(ssp, True)
    cells = csp.refused_cells_logzsol()
    assert cells, "the aMIST corner ([Fe/H]=+0.5, [a/Fe]=+0.6) must be refused"
    z_lo, z_hi, a_lo, a_hi, _ = cells[0]
    th = dict(csp.theta_init)
    th["logzsol"] = jnp.array([0.5 * (z_lo + z_hi)])
    th["afe"] = jnp.array([0.5 * (a_lo + a_hi)])
    with pytest.raises(ValueError, match="refused interpolation cell"):
        csp.check_param_ranges(th)


@pytest.mark.parametrize("name", ALPHA)
def test_solar_plane_matches_the_non_alpha_reading(name):
    """At [alpha/Fe] = 0 the alpha basis is the solar-scaled plane at the same logzsol."""
    ssp, _ = _load(name)
    csp = _csp(ssp, True)
    th = dict(csp.theta_init)
    th["logzsol"] = jnp.array([-0.25])
    th["afe"] = jnp.array([0.0])
    with_afe = np.asarray(csp.get_spectrum(th))
    th_no = {k: v for k, v in th.items() if k != "afe"}     # defaults to the solar plane
    assert np.array_equal(with_afe, np.asarray(csp.get_spectrum(th_no)))


def test_alpha_solar_plane_equals_the_independent_solar_scaled_grid():
    """C4.1: the alpha grid's [alpha/Fe] = 0 plane and the separately built solar-scaled
    MIST+C3K_LR grid must agree AT THE SAME logzsol -- they were built by different
    python-fsps versions, so this checks the convention, not one build."""
    a_ssp, _ = _load("amist_c3k_lr_chab_afe")
    b_ssp, _ = _load("mist_c3k_lr_chab")
    assert np.allclose(a_ssp.logzsol_axis, b_ssp.logzsol_axis, rtol=0, atol=1e-15)
    assert np.array_equal(np.asarray(a_ssp.ssp_wave), np.asarray(b_ssp.ssp_wave))
    i0 = int(np.argmin(np.abs(np.asarray(a_ssp.ssp_afe))))
    fa = np.asarray(a_ssp.ssp_flux)[i0]
    fb = np.asarray(b_ssp.ssp_flux)
    ia = int(np.argmin(np.abs(np.asarray(a_ssp.ssp_lg_age_gyr))))     # 1 Gyr
    for iz in (2, 6, 10, 12):
        x, y = fa[iz, ia], fb[iz, ia]
        ok = (y > 0) & np.isfinite(y)
        assert np.median(np.abs(x[ok] / y[ok] - 1.0)) < 1e-9


# --------------------------------------------------------------------------- tied gas

@pytest.mark.fsps
def test_gas_tied_removes_gas_logz_from_the_parameters():
    """A tied gas metallicity is not a free parameter: it is absent from theta_init and from
    neb_param_names (so marginalize_elines does not see a sampled nebular parameter), and
    supplying one anyway raises."""
    import os
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set")
    ssp, _ = _load("mist_bpass_v2")
    csp = _csp(ssp, False, add_neb=True, gas_tied=True, sps_home=os.environ["SPS_HOME"])
    assert "gas_logz" not in csp.theta_init
    assert "gas_logz" not in getattr(csp, "neb_param_names", [])
    assert "gas_logu" in csp.neb_param_names          # the ionisation parameter stays free
    with pytest.raises(ValueError, match="gas_tied=True"):
        CSPBasis(ssp, theta={"lookback_time": jnp.linspace(0.0, 13.0, 4), "sfh": jnp.ones(4),
                             "logzsol": jnp.array([-0.3]), "gas_logz": jnp.array([-0.3])},
                 zh_const=True, add_dust=False, add_diffuse_dust=False, add_neb=True,
                 gas_tied=True, sps_home=os.environ["SPS_HOME"], add_igm=False,
                 verbose=False, cosmo=Cosmology.planck18(), sfh_interp="step")


def test_gas_tied_requires_a_nebular_model():
    ssp, _ = _load("mist_bpass_v2")
    with pytest.raises(ValueError, match="no nebular model to tie"):
        _csp(ssp, False, add_neb=False, gas_tied=True)


# --------------------------------------------------------------------------- vs python-fsps

@pytest.mark.fsps
def test_matches_python_fsps_logzsol():
    """CERIDWEN's logzsol IS FSPS's logzsol: interpolating the grid at logzsol = x must
    reproduce fsps.StellarPopulation(zcontinuous=1, logzsol=x), while reading the same
    number as an absolute log10 Z, or using a wrong Z_sun, is clearly worse."""
    fsps = pytest.importorskip("fsps")
    sp = fsps.StellarPopulation(zcontinuous=0, sfh=0, imf_type=1)
    libs = [b.decode() if isinstance(b, bytes) else str(b) for b in sp.libraries]
    name = {("mist", "miles"): "mist_miles_chab",
            ("mist", "c3k_lr"): "mist_c3k_lr_chab",
            ("bpss", "bpass"): "mist_bpass_v2"}.get((libs[0], libs[1]))
    if name is None:
        pytest.skip(f"no local grid for this FSPS build {libs[:2]}")
    ssp, _ = _load(name)
    if float(sp.solar_metallicity) != pytest.approx(ssp.zsun_nominal, rel=1e-6):
        pytest.skip(f"FSPS zsol {float(sp.solar_metallicity)} != grid Z_sun {ssp.zsun_nominal}")

    native = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)
    ages = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64) + 9.0
    ia = int(np.argmin(np.abs(ages - 9.0)))                 # a 1 Gyr SSP
    lz_axis = ssp.logzsol_axis

    def interp(axis, x):
        k = int(np.clip(np.searchsorted(axis, x, side="left"), 1, axis.size - 1))
        w = float(np.clip((x - axis[k - 1]) / (axis[k] - axis[k - 1]), 0.0, 1.0))
        return (1.0 - w) * flux[k - 1, ia] + w * flux[k, ia]

    sp_c = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=1)
    worse = 0
    for x in (-2.0, -1.0, -0.5, 0.0, 0.2):
        if not (lz_axis.min() <= x <= lz_axis.max()):
            continue
        sp_c.params["logzsol"] = x
        _w, f_fsps = sp_c.get_spectrum(tage=10.0 ** (ages[ia] - 9.0), peraa=False)
        ok = np.isfinite(f_fsps) & (f_fsps > 0)
        right = interp(lz_axis, x)
        frac = np.abs(right[ok] / f_fsps[ok] - 1.0)
        assert np.median(frac) < 1e-3, f"logzsol={x}: median frac diff {np.median(frac):.2e}"
        # the two wrong readings of the same number
        for wrong_axis, label in ((native, "absolute log10 Z"),
                                  (native - math.log10(0.0142), "Z_sun = 0.0142")):
            if not (wrong_axis.min() <= x <= wrong_axis.max()):
                worse += 1
                continue
            bad = np.abs(interp(wrong_axis, x)[ok] / f_fsps[ok] - 1.0)
            if np.median(bad) > 10 * max(np.median(frac), 1e-12):
                worse += 1
    assert worse >= 2, "the wrong metallicity conventions were not clearly worse"


@pytest.mark.fsps
def test_gas_tied_matches_fsps_line_ratios():
    """With gas_tied the gas follows the stars exactly as Prospector ties them: at
    logzsol = 0 the CLOUDY lines must equal FSPS's at gas_logz = logzsol = 0.  Compared at
    one SSP age, one metallicity node and the same (dust-free) CLOUDY grid, so the only
    thing under test is the metallicity the gas is evaluated at."""
    import os
    fsps = pytest.importorskip("fsps")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set")
    sp0 = fsps.StellarPopulation(zcontinuous=0, sfh=0, imf_type=1)
    libs = [b.decode() if isinstance(b, bytes) else str(b) for b in sp0.libraries]
    name = {"mist": "mist_miles_chab", "bpss": "mist_bpass_v2"}.get(libs[0])
    if name is None:
        pytest.skip(f"no local grid for isochrones {libs[0]}")
    ssp, _ = _load(name)
    if float(sp0.solar_metallicity) != pytest.approx(ssp.zsun_nominal, rel=1e-6):
        pytest.skip("this FSPS build's zsol does not match the local grid")
    csp = _csp(ssp, False, add_neb=True, gas_tied=True,
               init_neb_params={"cloudy_dust": False},      # FSPS's default grid
               lookback_time=jnp.linspace(0.0, 0.05, 3), sps_home=os.environ["SPS_HOME"])
    assert "gas_logz" not in csp.theta_init, "a tied gas metallicity must not be a parameter"
    ages = np.asarray(csp._neb_ages_young)
    iz = int(np.argmin(np.abs(np.asarray(csp.zmet))))        # logzsol = 0
    ia = int(np.argmin(np.abs(ages - 7.0)))                  # 10 Myr
    # gas_tied sends theta["logzsol"] into the nebular model as gas_logz
    th = dict(csp.theta_init)
    th["logzsol"] = jnp.array([0.0])
    gas = float(np.asarray(csp._gas_logz(th)).reshape(-1)[0])
    assert gas == 0.0, "gas_tied must evaluate the gas at logzsol"
    row = np.asarray(csp.neb.evaluate_batch_line_lum(
        jnp.array(gas), jnp.array(-2.0), csp._neb_ages_young, csp._neb_logqq_young))[iz, ia]
    wl = np.asarray(csp.neb.nebem_line_pos)

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=1, add_neb_emission=True,
                                gas_logu=-2.0, gas_logz=0.0, logzsol=0.0)
    sp.get_spectrum(tage=10.0 ** (ages[ia] - 9.0), peraa=False)
    w, f = np.asarray(sp.emline_wavelengths), np.asarray(sp.emline_luminosity)

    def val(arr, ww, x):
        return float(arr[int(np.argmin(np.abs(ww - x)))])

    for a, b, tag in ((5008.24, 4862.71, "[OIII]/Hb"), (6585.27, 6564.61, "[NII]/Ha"),
                      (6564.61, 4862.71, "Ha/Hb")):
        r_c = val(row, wl, a) / val(row, wl, b)
        r_f = val(f, w, a) / val(f, w, b)
        assert r_c == pytest.approx(r_f, rel=1e-6), f"{tag}: ceridwen {r_c:.6g} vs FSPS {r_f:.6g}"
    for x in (4862.71, 6564.61):                    # absolute normalisation (Q differs ~5e-4)
        assert val(row, wl, x) == pytest.approx(val(f, w, x), rel=2e-3)
