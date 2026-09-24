"""The test-grid resolver finds grids that ``fetch_grid`` has downloaded (Q2-006), and
``misuse_report.py`` says in one line how to get a grid when there is none (Q2-008).

A fake registry entry (a small file and its SHA-256) stands in for a published grid, so no
download and no real grid is needed.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

import pytest

import _gridfixture as GF
from ceridwen.ssps import grid_fetch

REPORT = pathlib.Path(__file__).resolve().parent / "regression" / "misuse_report.py"


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


_RUN_REPORT = """
import runpy, sys, pathlib
sys.path.insert(0, {tests!r})
import _gridfixture as GF
empty = pathlib.Path({empty!r})
GF.FIXTURE_DIR = GF.TEST_DATA_DIR = empty
runpy.run_path({report!r}, run_name="__main__")
"""


@pytest.mark.parametrize("cached", ["none", "corrupt"])
def test_misuse_report_without_a_grid_exits_with_one_line(tmp_path, cached):
    """Q2-008: run as a script with no grid, it exits 1 with one line naming fetch_grid (or
    the checksum failure), not a pytest ``Skipped`` traceback."""
    cache = tmp_path / "cache"
    cache.mkdir()
    if cached == "corrupt":
        (cache / "mist_bpass_v2.h5").write_bytes(b"not the published grid")
    env = {k: v for k, v in os.environ.items() if k != "CERIDWEN_TEST_SSP"}
    env["CERIDWEN_GRID_DIR"] = str(cache)
    code = _RUN_REPORT.format(tests=str(REPORT.parents[1]), empty=str(tmp_path),
                              report=str(REPORT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr and "Skipped" not in r.stderr
    lines = [ln for ln in r.stderr.splitlines() if ln.startswith("misuse_report:")]
    assert len(lines) == 1, r.stderr
    want = "fails its checksum" if cached == "corrupt" else "fetch_grid('mist_bpass_v2')"
    assert want in lines[0]
