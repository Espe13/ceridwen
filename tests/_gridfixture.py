"""Shared resolver for the test SSP grids.

The suite's canonical test grid is the published BPASS grid (registry name
``mist_bpass_v2``; local file name ``ssp_data_bpass.h5``).  A test grid is found, in
order, at:

1. ``$CERIDWEN_TEST_SSP`` (what CI sets; the canonical grid only);
2. ``tests/fixtures/<file>``;
3. ``ceridwen/data/test_data/<file>`` (NOT committed: ``data/`` is in .gitignore);
4. the :func:`ceridwen.ssps.fetch_grid` cache (``$CERIDWEN_GRID_DIR``, default
   ``~/.ceridwen/grids``), for the files that are published grids (:data:`REGISTRY_NAME`).
   The cached file is checksum-verified as ``fetch_grid`` does it; tests never download.

So ``python -c "from ceridwen.ssps import fetch_grid; fetch_grid('mist_bpass_v2')"`` is
enough for the grid-dependent tests to run.  When no grid is found, or a cached one fails
its checksum, those tests skip with the reason.  The published grids carry the
surviving-mass table; a test that needs a grid without one uses :func:`without_table`.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional


def _repo_root() -> pathlib.Path:
    p = pathlib.Path(__file__).resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: tests/ -> repo root
    return p.parents[1]


REPO_ROOT = _repo_root()
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
TEST_DATA_DIR = REPO_ROOT / "ceridwen" / "data" / "test_data"

#: The canonical (default) test grid.
DEFAULT_GRID = "ssp_data_bpass.h5"

#: Local test-grid file name -> the registry name of the same grid in ``fetch_grid``.
REGISTRY_NAME = {
    "ssp_data_bpass.h5": "mist_bpass_v2",
    "ssp_data_mist_miles.h5": "mist_miles_chab",
    "amist_c3k_hr_krou_afe.h5": "amist_c3k_hr_krou_afe",
}

#: Why the last lookup of each file failed (read by :func:`require_test_grid`).
_MISS: dict = {}
_VERIFIED: dict = {}


def _from_fetch_cache(name: str) -> Optional[pathlib.Path]:
    """The verified ``fetch_grid`` cache copy of the test grid file ``name``, or ``None``.

    Uses ``ceridwen.ssps.grid_fetch.cached_grid``, so the cache directory and the SHA-256
    check are those of ``fetch_grid``; never downloads.  A copy that fails its checksum is
    not used, and the reason is kept for the skip message."""
    reg = REGISTRY_NAME.get(name)
    if reg is None:
        return None
    from ceridwen.ssps import grid_fetch
    path = grid_fetch._grid_cache_root() / f"{reg}.h5"
    if not path.is_file():
        return None
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _VERIFIED:                  # hash each file once per process
        try:
            grid_fetch.cached_grid(reg)
            _VERIFIED[key] = None
        except RuntimeError as exc:
            _VERIFIED[key] = str(exc)
    if _VERIFIED[key] is not None:
        _MISS[name] = _VERIFIED[key]
        return None
    return path


def find_named_grid(name: str) -> Optional[pathlib.Path]:
    """The test grid file ``name`` itself (no substitution), or ``None``.

    Search order: ``tests/fixtures/<name>`` -> ``ceridwen/data/test_data/<name>`` -> the
    ``fetch_grid`` cache when ``name`` is a published grid (:data:`REGISTRY_NAME`)."""
    for c in (FIXTURE_DIR / name, TEST_DATA_DIR / name):
        if c.is_file():
            return c
    return _from_fetch_cache(name)


def find_test_grid(name: str = DEFAULT_GRID) -> Optional[pathlib.Path]:
    """Return a usable test SSP grid path, or ``None`` if unavailable.

    Search order: ``$CERIDWEN_TEST_SSP`` -> :func:`find_named_grid` of ``name`` -> the
    canonical grid (:data:`DEFAULT_GRID`) by the same search.
    """
    env = os.environ.get("CERIDWEN_TEST_SSP")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    for n in dict.fromkeys((name, DEFAULT_GRID)):
        g = find_named_grid(n)
        if g is not None:
            return g
    return None


def missing_reason(name: str = DEFAULT_GRID) -> str:
    """One line saying why :func:`find_test_grid` found nothing, and how to fix it."""
    bad = _MISS.get(name) or _MISS.get(DEFAULT_GRID)
    if bad:
        return f"test SSP grid {name!r}: the fetch_grid cache copy is unusable: {bad}"
    return (f"test SSP grid {name!r} not found; run "
            "python -c \"from ceridwen.ssps import fetch_grid; fetch_grid('mist_bpass_v2')\" "
            "(the tests look in the fetch_grid cache, $CERIDWEN_GRID_DIR or ~/.ceridwen/grids), "
            f"or set $CERIDWEN_TEST_SSP to a grid file, or place one at "
            f"ceridwen/data/test_data/{DEFAULT_GRID}")


def require_test_grid(name: str = DEFAULT_GRID) -> pathlib.Path:
    """Like :func:`find_test_grid`, but skip the whole test module if missing."""
    import pytest

    g = find_test_grid(name)
    if g is None:
        pytest.skip(missing_reason(name), allow_module_level=True)
    return g


def require_named_grid(name: str) -> pathlib.Path:
    """Like :func:`find_named_grid`, but skip the calling test if missing."""
    import pytest

    g = find_named_grid(name)
    if g is None:
        reg = REGISTRY_NAME.get(name)
        how = (f"fetch_grid({reg!r}) puts it in the fetch_grid cache" if reg
               else "it is not a published grid")
        pytest.skip(_MISS.get(name) or f"{name} is not present ({how})")
    return g


def without_table(grid):
    """``grid`` without its surviving-mass table: the published grids carry one, so a test
    of the no-table path must remove it rather than assume the test grid lacks it."""
    import dataclasses
    return dataclasses.replace(grid, ssp_stellar_mass=None, stellar_mass_source=None)
