"""
Provenance + trap-proofing for :class:`ceridwen.ssps.SSPData`.

Covers (all runnable without FSPS unless marked):

* ``from_fsps`` rejects kwargs the CSP forward model owns (dust / SFH /
  nebular / IGM / redshift / metallicity / smoothing / internal).
* provenance metadata round-trips through ``save`` / ``load``.
* legacy metadata-less HDF5 fixtures still load, with metadata ``None`` / ``{}``.
* ``CSPBasis`` auto-picks ``isoc_type`` from the SSP grid's provenance.
* a conflicting ``isoc_type`` raises; a legacy grid with none warns.

The one test that actually builds a grid from FSPS is ``@pytest.mark.fsps``.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import jax.numpy as jnp
import pytest

# tests/ on sys.path so `_gridfixture` imports from any subdir (mirrors conftest).
sys.path.insert(0, os.path.dirname(__file__))
from _gridfixture import require_test_grid  # noqa: E402

from ceridwen.ssps import SSPData  # noqa: E402
from ceridwen.ssps.ssp_data import (  # noqa: E402
    LIBRARY_IMF_KWARGS,
    _validate_fsps_kwargs,
)
from ceridwen.csp.csp import CSPBasis, _resolve_isoc_type  # noqa: E402


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _tiny_ssp(**meta) -> SSPData:
    """A minimal, valid SSPData (2 Z, 3 ages, 5 wave) with optional metadata."""
    lgmet = jnp.array([-2.0, -1.0])
    lgage = jnp.array([-1.0, 0.0, 1.0])
    wave = jnp.linspace(1000.0, 10000.0, 5)
    flux = jnp.ones((2, 3, 5))
    return SSPData(lgmet, lgage, wave, flux, **meta)


# ----------------------------------------------------------------------
# 1. kwarg gate
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwarg, value, mechanism_word",
    [
        ("dust_type", 2, "dust"),
        ("dust2", 0.3, "dust"),
        ("agb_dust", 1.0, "dust"),          # AGB circumstellar dust -> dust
        ("sfh", 4, "star-formation history"),
        ("tage", 1.0, "star-formation history"),
        ("add_neb_emission", True, "nebular"),
        ("gas_logu", -2.0, "nebular"),
        ("add_igm_absorption", True, "IGM"),
        ("zred", 6.0, "redshift"),
        ("zmet", 3, "metallicity"),
        ("logzsol", 0.0, "metallicity"),
        ("sigma_smooth", 100.0, "smoothing"),
        ("zcontinuous", 1, "metallicity-grid mode"),
    ],
)
def test_from_fsps_rejects_csp_owned_kwarg(kwarg, value, mechanism_word):
    """Each CSP-owned kwarg is rejected, naming the kwarg and its mechanism."""
    with pytest.raises(ValueError) as exc:
        SSPData.from_fsps(**{kwarg: value})
    msg = str(exc.value)
    assert kwarg in msg
    assert mechanism_word in msg


def test_from_fsps_rejects_unknown_kwarg():
    with pytest.raises(ValueError, match="not_a_real_param"):
        SSPData.from_fsps(not_a_real_param=1)


def test_validate_accepts_library_imf_kwargs():
    """Whitelisted IMF / library kwargs pass validation untouched."""
    good = {"imf_type": 1, "imf1": 1.3, "tpagb_norm_type": 2, "use_wr_spectra": True}
    assert _validate_fsps_kwargs(dict(good)) == good
    # sanity: the documented core keys are actually whitelisted
    assert {"imf_type", "imf1", "imf2", "imf3"} <= LIBRARY_IMF_KWARGS


# ----------------------------------------------------------------------
# 2. metadata save -> load round trip (schema 2.0)
# ----------------------------------------------------------------------
def test_metadata_roundtrip(tmp_path):
    import numpy as np

    meta = dict(
        isoc_type="mist",
        spec_library="miles",
        imf_type=1,
        fsps_version="0.4.7",
        fsps_kwargs={"imf_type": 1, "tpagb_norm_type": 2},
        wave_min=1000.0,
        wave_max=10000.0,
        schema_version="2.0",
    )
    ssp = _tiny_ssp(**meta).with_resolution(
        segments=[(1000.0, 10000.0, "fwhm_AA", 2.54)],
        source="test segments")
    path = str(tmp_path / "grid.h5")
    ssp.save(path)

    loaded = SSPData.load(path)
    assert loaded.isoc_type == "mist"
    assert loaded.spec_library == "miles"
    assert loaded.imf_type == 1
    assert loaded.fsps_version == "0.4.7"
    assert loaded.fsps_kwargs == {"imf_type": 1, "tpagb_norm_type": 2}
    assert loaded.wave_min == 1000.0
    assert loaded.wave_max == 10000.0
    assert loaded.schema_version == "2.0"
    # arrays survive too, including the library resolution curve
    assert loaded.ssp_flux.shape == ssp.ssp_flux.shape
    assert np.array_equal(np.asarray(loaded.ssp_resolution),
                          np.asarray(ssp.ssp_resolution), equal_nan=True)
    assert loaded.resolution_source == "test segments"


# ----------------------------------------------------------------------
# 3. schema 2.0 strictness: no resolution -> no save, no load
# ----------------------------------------------------------------------
def _legacy_ssp():
    """An arrays-only SSPData (no provenance, no resolution) — legal in
    memory as an intermediate object, but not serialisable (schema 2.0)."""
    full = SSPData.load(str(require_test_grid()))
    return SSPData(full.ssp_lgmet, full.ssp_lg_age_gyr, full.ssp_wave, full.ssp_flux)


def test_save_without_resolution_raises(tmp_path):
    with pytest.raises(ValueError, match="with_resolution"):
        _legacy_ssp().save(str(tmp_path / "nope.h5"))


def test_load_schema1_file_raises_with_converter_pointer(tmp_path):
    import h5py
    import numpy as np

    path = str(tmp_path / "schema1.h5")
    tiny = _tiny_ssp()
    with h5py.File(path, "w") as f:            # a schema-1.x file by hand
        f.create_dataset("ssp_lgmet",      data=np.asarray(tiny.ssp_lgmet))
        f.create_dataset("ssp_lg_age_gyr", data=np.asarray(tiny.ssp_lg_age_gyr))
        f.create_dataset("ssp_wave",       data=np.asarray(tiny.ssp_wave))
        f.create_dataset("ssp_flux",       data=np.asarray(tiny.ssp_flux))
    with pytest.raises(ValueError, match="convert_grids_schema2"):
        SSPData.load(path)


def test_with_resolution_validates_shape():
    import numpy as np

    with pytest.raises(ValueError, match="shape"):
        _tiny_ssp().with_resolution(sigma_v=np.ones(3), source="bad shape")
    with pytest.raises(ValueError, match="exactly one"):
        _tiny_ssp().with_resolution(source="neither given")


# ----------------------------------------------------------------------
# 3b. resolution curve construction (pure numpy, no FSPS)
# ----------------------------------------------------------------------
def test_sampling_floor_and_combined_curve():
    import numpy as np
    from ceridwen.ssps.library_resolution import (
        CKMS, FWHM_TO_SIGMA, sampling_floor_sigma_v, combined_sigma_v,
        miles_segments, sigma_v_from_segments)

    # uniform log grid: the floor is analytic
    wave = np.exp(np.linspace(np.log(3000.0), np.log(9000.0), 4000))
    dln = np.log(wave[1]) - np.log(wave[0])
    floor = sampling_floor_sigma_v(wave)
    assert np.all(np.isfinite(floor)) and np.all(floor > 0)
    assert np.allclose(floor, FWHM_TO_SIGMA * CKMS * 2.0 * dln)

    # combined = element-wise max(floor, LSF); floor where LSF is NaN
    comb = combined_sigma_v(wave, segments=miles_segments())
    lsf = sigma_v_from_segments(wave, miles_segments())
    assert np.all(np.isfinite(comb))
    inside = np.isfinite(lsf)
    assert inside.any() and (~inside).any()
    assert np.allclose(comb[inside], np.maximum(floor[inside], lsf[inside]))
    assert np.allclose(comb[~inside], floor[~inside])

    # no segments -> floor alone
    assert np.array_equal(combined_sigma_v(wave), floor)

    # non-monotonic wave is rejected, not silently differentiated
    with pytest.raises(ValueError, match="strictly increasing"):
        sampling_floor_sigma_v(wave[::-1])


def test_from_fsps_source_without_segments_raises():
    with pytest.raises(ValueError, match="resolution_source"):
        SSPData.from_fsps(resolution_source="orphan citation")


def test_from_fsps_segments_without_source_raises():
    with pytest.raises(ValueError, match="resolution_source"):
        SSPData.from_fsps(
            resolution_segments=[(3525.0, 7500.0, "fwhm_AA", 2.54)])


# ----------------------------------------------------------------------
# 3c. SSPDataAfe schema 2.1 strictness (no FSPS)
# ----------------------------------------------------------------------
def _tiny_afe(**meta):
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe
    lgmet = jnp.array([-2.0, -1.0])
    afe = jnp.array([0.0])
    lgage = jnp.array([-1.0, 0.0, 1.0])
    wave = jnp.linspace(1000.0, 10000.0, 5)
    flux = jnp.ones((1, 2, 3, 5))
    return SSPDataAfe(lgmet, afe, lgage, wave, flux, **meta)


def test_afe_save_without_resolution_raises(tmp_path):
    with pytest.raises(ValueError, match="with_resolution"):
        _tiny_afe().save(str(tmp_path / "nope.h5"))


def test_afe_resolution_roundtrip(tmp_path):
    import numpy as np
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe

    ssp = _tiny_afe(schema_version="2.1").with_resolution(
        sigma_v=np.full(5, 50.0), source="test curve")
    path = str(tmp_path / "afe.h5")
    ssp.save(path)
    loaded = SSPDataAfe.load(path)
    assert np.array_equal(np.asarray(loaded.ssp_resolution), np.full(5, 50.0))
    assert loaded.resolution_source == "test curve"
    assert loaded.ssp_flux.shape == (1, 2, 3, 5)
    assert loaded.schema_version == "2.1"


def test_afe_load_without_resolution_raises_with_converter_pointer(tmp_path):
    import h5py
    import numpy as np
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe

    tiny = _tiny_afe()
    path = str(tmp_path / "afe_old.h5")
    with h5py.File(path, "w") as f:          # a schema-2.0 afe file by hand
        f.create_dataset("ssp_lgmet",      data=np.asarray(tiny.ssp_lgmet))
        f.create_dataset("ssp_afe",        data=np.asarray(tiny.ssp_afe))
        f.create_dataset("ssp_lg_age_gyr", data=np.asarray(tiny.ssp_lg_age_gyr))
        f.create_dataset("ssp_wave",       data=np.asarray(tiny.ssp_wave))
        f.create_dataset("ssp_flux",       data=np.asarray(tiny.ssp_flux))
    with pytest.raises(ValueError, match="convert_grids_schema2"):
        SSPDataAfe.load(path)


# ----------------------------------------------------------------------
# 4. CSP auto-propagates isoc_type from the grid
# ----------------------------------------------------------------------
def _minimal_theta(n=5):
    lb = jnp.linspace(0.0, 13.8, n)
    return {"lookback_time": lb, "sfh": jnp.ones(n), "Z": jnp.array([-1.85])}


def test_csp_reads_isoc_type_from_ssp_grid():
    base = SSPData.load(str(require_test_grid()))
    ssp = dataclasses.replace(base, isoc_type="mist", spec_library="miles")
    csp = CSPBasis(ssp, theta=_minimal_theta(), zh_const=True,
                   add_neb=False, add_igm=False, verbose=False)
    assert csp._ssp_isoc_type == "mist"
    assert csp._ssp_spec_library == "miles"


def test_csp_legacy_grid_has_none_isoc_type():
    csp = CSPBasis(_legacy_ssp(), theta=_minimal_theta(), zh_const=True,
                   add_neb=False, add_igm=False, verbose=False)
    assert csp._ssp_isoc_type is None


# ----------------------------------------------------------------------
# 5. isoc_type resolution: recorded / conflict / legacy
# ----------------------------------------------------------------------
def test_resolve_uses_recorded_when_unspecified():
    assert _resolve_isoc_type("pdva", None) == "pdva"
    assert _resolve_isoc_type("mist", "mist") == "mist"


def test_resolve_conflict_raises():
    with pytest.raises(ValueError, match="conflict"):
        _resolve_isoc_type("mist", "bpss")


def test_resolve_legacy_warns_and_falls_back():
    with pytest.warns(UserWarning, match="legacy"):
        assert _resolve_isoc_type(None, None) == "mist"


def test_resolve_legacy_explicit_is_honoured():
    # No recorded value + explicit user value: honour it, no warning/error.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert _resolve_isoc_type(None, "pdva") == "pdva"


# ----------------------------------------------------------------------
# FSPS-only: actually build a grid and check recorded provenance
# ----------------------------------------------------------------------
@pytest.mark.fsps
def test_from_fsps_records_provenance(tmp_path):
    path = str(tmp_path / "grid.h5")
    ssp = SSPData.from_fsps(imf_type=1, save_to=path)
    assert ssp.isoc_type is not None          # e.g. 'mist'
    assert ssp.spec_library is not None        # e.g. 'miles'
    assert ssp.imf_type == 1
    assert ssp.fsps_version is not None
    assert ssp.fsps_kwargs == {"imf_type": 1}
    assert ssp.schema_version is not None

    reloaded = SSPData.load(path)
    assert reloaded.isoc_type == ssp.isoc_type
    assert reloaded.imf_type == 1
    assert reloaded.fsps_kwargs == {"imf_type": 1}
