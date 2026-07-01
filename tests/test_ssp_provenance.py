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
# 2. metadata save -> load round trip
# ----------------------------------------------------------------------
def test_metadata_roundtrip(tmp_path):
    meta = dict(
        isoc_type="mist",
        spec_library="miles",
        imf_type=1,
        fsps_version="0.4.7",
        fsps_kwargs={"imf_type": 1, "tpagb_norm_type": 2},
        wave_min=1000.0,
        wave_max=10000.0,
        schema_version="1.0",
    )
    ssp = _tiny_ssp(**meta)
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
    assert loaded.schema_version == "1.0"
    # arrays survive too
    assert loaded.ssp_flux.shape == ssp.ssp_flux.shape


# ----------------------------------------------------------------------
# 3. backward compatibility: legacy (pre-provenance) metadata-less grid
# ----------------------------------------------------------------------
def _legacy_ssp():
    """A genuinely metadata-less SSPData (arrays only), emulating a grid built
    before provenance tracking. Reuses the committed fixture's arrays but drops
    all provenance, so the test does not depend on the fixture being legacy."""
    full = SSPData.load(str(require_test_grid()))
    return SSPData(full.ssp_lgmet, full.ssp_lg_age_gyr, full.ssp_wave, full.ssp_flux)


def test_legacy_grid_loads_without_metadata(tmp_path):
    out = str(tmp_path / "legacy.h5")
    _legacy_ssp().save(out)                 # save() omits None attrs -> truly legacy on disk
    ssp = SSPData.load(out)
    assert ssp.ssp_flux.ndim == 3           # grids present...
    assert ssp.isoc_type is None            # ...but no provenance
    assert ssp.spec_library is None
    assert ssp.imf_type is None
    assert ssp.fsps_version is None
    assert ssp.fsps_kwargs == {}
    assert ssp.schema_version is None


def test_resave_legacy_does_not_invent_metadata(tmp_path):
    out = str(tmp_path / "resaved.h5")
    _legacy_ssp().save(out)
    again = SSPData.load(out)
    assert again.isoc_type is None
    assert again.fsps_kwargs == {}


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
