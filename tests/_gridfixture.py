"""Shared resolver for the committed test SSP grid.

A small SSP grid used by the test suite is committed at
``tests/fixtures/ssp_data_test.h5`` (this directory is NOT covered by the
``data/`` rule in .gitignore, unlike ``ceridwen/data/test_data/``).

Build / refresh it with::

    python scripts/make_test_fixture_grid.py

The resolver also honours ``$CERIDWEN_TEST_SSP`` (an explicit path) and falls
back to a local full grid at ``ceridwen/data/test_data/ssp_data.h5`` if you
already have one, so existing developer checkouts keep working unchanged.
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


def find_test_grid(name: str = "ssp_data_test.h5") -> Optional[pathlib.Path]:
    """Return a usable test SSP grid path, or ``None`` if unavailable.

    Search order: ``$CERIDWEN_TEST_SSP`` -> committed fixture
    (``tests/fixtures/<name>``) -> legacy local grid
    (``ceridwen/data/test_data/<name or ssp_data.h5>``).
    """
    candidates = []
    env = os.environ.get("CERIDWEN_TEST_SSP")
    if env:
        candidates.append(pathlib.Path(env))
    candidates.append(FIXTURE_DIR / name)
    candidates.append(REPO_ROOT / "ceridwen" / "data" / "test_data" / name)
    candidates.append(REPO_ROOT / "ceridwen" / "data" / "test_data" / "ssp_data.h5")
    for c in candidates:
        if c.is_file():
            return c
    return None


def require_test_grid(name: str = "ssp_data_test.h5") -> pathlib.Path:
    """Like :func:`find_test_grid`, but skip the whole test module if missing."""
    import pytest

    g = find_test_grid(name)
    if g is None:
        pytest.skip(
            f"test SSP grid {name!r} not found; build it with "
            "`python scripts/make_test_fixture_grid.py` "
            "(writes tests/fixtures/ssp_data_test.h5), or set $CERIDWEN_TEST_SSP",
            allow_module_level=True,
        )
    return g
