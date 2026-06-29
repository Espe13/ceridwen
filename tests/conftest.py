"""Pytest configuration shared by the whole ``tests/`` tree.

* Makes ``_gridfixture`` importable from test modules in subdirectories
  (``tests/csp/``, ``tests/regression/`` ...) by putting ``tests/`` on
  ``sys.path`` -- this runs before any test module is imported.
* Exposes an ``ssp_grid_path`` session fixture that resolves the committed
  test SSP grid (and skips cleanly when it is not present, e.g. in CI without
  FSPS).
"""
from __future__ import annotations

import os
import sys

# tests/ on sys.path so `import _gridfixture` works from every subdirectory.
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

from _gridfixture import find_test_grid  # noqa: E402


@pytest.fixture(scope="session")
def ssp_grid_path():
    """Path (str) to a usable test SSP grid; skips the test if unavailable."""
    g = find_test_grid()
    if g is None:
        pytest.skip(
            "test SSP grid not found; build it with "
            "`python scripts/make_test_fixture_grid.py`"
        )
    return str(g)
