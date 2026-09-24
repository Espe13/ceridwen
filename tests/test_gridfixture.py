"""The test-grid resolver finds grids that ``fetch_grid`` has downloaded (Q2-006).

A fake registry entry (a small file and its SHA-256) stands in for a published grid, so no
download and no real grid is needed.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

import _gridfixture as GF
from ceridwen.ssps import grid_fetch

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No $CERIDWEN_TEST_SSP, empty repo grid folders, a fresh fetch_grid cache, and a
    registry entry ``mist_bpass_v2`` whose checksum is that of a small stand-in file."""
    cache = tmp_path / "cache"
    cache.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("CERIDWEN_TEST_SSP", raising=False)
    monkeypatch.setenv("CERIDWEN_GRID_DIR", str(cache))
    monkeypatch.setattr(GF, "FIXTURE_DIR", empty)
    monkeypatch.setattr(GF, "TEST_DATA_DIR", empty)
    monkeypatch.setattr(GF, "_MISS", {})
    monkeypatch.setattr(GF, "_VERIFIED", {})
    body = b"stand-in grid bytes"
    entry = dict(grid_fetch.REGISTRY["mist_bpass_v2"], sha256=hashlib.sha256(body).hexdigest())
    monkeypatch.setitem(grid_fetch.REGISTRY, "mist_bpass_v2", entry)
    return cache, body


def test_nothing_anywhere_gives_none_and_says_fetch_grid(isolated):
    cache, _ = isolated
    assert GF.find_test_grid() is None
    assert "fetch_grid('mist_bpass_v2')" in GF.missing_reason()
    assert list(cache.iterdir()) == []                 # looking never downloads


def test_a_fetched_grid_is_found_by_its_file_name(isolated):
    cache, body = isolated
    (cache / "mist_bpass_v2.h5").write_bytes(body)
    assert GF.find_test_grid() == cache / "mist_bpass_v2.h5"
    assert GF.find_named_grid("ssp_data_bpass.h5") == cache / "mist_bpass_v2.h5"
    # a named fixture that is not a published grid falls back to the canonical one
    assert GF.find_test_grid("ssp_data_bpass_agb_dust.h5") == cache / "mist_bpass_v2.h5"
    assert GF.find_named_grid("ssp_data_bpass_agb_dust.h5") is None


def test_ceridwen_test_ssp_still_wins(isolated, tmp_path, monkeypatch):
    cache, body = isolated
    (cache / "mist_bpass_v2.h5").write_bytes(body)
    explicit = tmp_path / "explicit.h5"
    explicit.write_bytes(b"x")
    monkeypatch.setenv("CERIDWEN_TEST_SSP", str(explicit))
    assert GF.find_test_grid() == explicit


def test_a_cached_grid_failing_its_checksum_is_not_used_and_the_skip_says_why(isolated):
    cache, body = isolated
    (cache / "mist_bpass_v2.h5").write_bytes(body + b"corrupt")
    assert GF.find_test_grid() is None
    assert "fails its checksum" in GF.missing_reason()
    with pytest.raises(pytest.skip.Exception, match="fails its checksum"):
        GF.require_named_grid("ssp_data_bpass.h5")


def test_cached_grid_matches_fetch_grid(isolated):
    """``cached_grid`` is the no-download half of ``fetch_grid``: same path, same check."""
    cache, body = isolated
    assert grid_fetch.cached_grid("mist_bpass_v2") is None
    (cache / "mist_bpass_v2.h5").write_bytes(body)
    assert grid_fetch.cached_grid("mist_bpass_v2") == grid_fetch.fetch_grid("mist_bpass_v2")
    (cache / "mist_bpass_v2.h5").write_bytes(body + b"!")
    for f in (grid_fetch.cached_grid, grid_fetch.fetch_grid):
        with pytest.raises(RuntimeError, match="fails its checksum"):
            f("mist_bpass_v2")
