"""Shared resolver for the test SSP grid.

The suite's canonical test grid is the local BPASS grid at
``ceridwen/data/test_data/ssp_data_bpass.h5`` (NOT committed — the
``data/`` rule in .gitignore excludes it).  On a machine without it,
either set ``$CERIDWEN_TEST_SSP`` to any schema-2.x SSP grid, or download
the BPASS release grid from Zenodo (doi:10.5281/zenodo.21221634; convert
with ``scripts/convert_grids_schema2.py`` if it predates schema 2.0) and
place it at that path.  When no grid is found, the grid-dependent tests
skip cleanly rather than erroring.

Named fixtures (e.g. ``ssp_data_bpass_agb_dust.h5``, used by the
BPASS+AGB regression test) are still resolved from ``tests/fixtures/``
first, then from ``ceridwen/data/test_data/``.
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


def find_test_grid(name: str = DEFAULT_GRID) -> Optional[pathlib.Path]:
    """Return a usable test SSP grid path, or ``None`` if unavailable.

    Search order: ``$CERIDWEN_TEST_SSP`` -> ``tests/fixtures/<name>`` ->
    ``ceridwen/data/test_data/<name>`` -> the canonical grid
    ``ceridwen/data/test_data/ssp_data_bpass.h5``.
    """
    candidates = []
    env = os.environ.get("CERIDWEN_TEST_SSP")
    if env:
        candidates.append(pathlib.Path(env))
    candidates.append(FIXTURE_DIR / name)
    candidates.append(TEST_DATA_DIR / name)
    candidates.append(TEST_DATA_DIR / DEFAULT_GRID)
    for c in candidates:
        if c.is_file():
            return c
    return None


def require_test_grid(name: str = DEFAULT_GRID) -> pathlib.Path:
    """Like :func:`find_test_grid`, but skip the whole test module if missing."""
    import pytest

    g = find_test_grid(name)
    if g is None:
        pytest.skip(
            f"test SSP grid {name!r} not found; place a schema-2.x grid at "
            f"ceridwen/data/test_data/{DEFAULT_GRID} (build with "
            "SSPData.from_fsps, or download from Zenodo "
            "doi:10.5281/zenodo.21221634 and convert with "
            "scripts/convert_grids_schema2.py), or set $CERIDWEN_TEST_SSP",
            allow_module_level=True,
        )
    return g
