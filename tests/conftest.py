"""Pytest configuration shared by the whole ``tests/`` tree.

* Makes ``_gridfixture`` importable from test modules in subdirectories
  (``tests/csp/``, ``tests/regression/`` ...) by putting ``tests/`` on
  ``sys.path`` -- this runs before any test module is imported.
* Exposes an ``ssp_grid_path`` session fixture that resolves the test SSP
  grid (``_gridfixture``; the fetch_grid cache counts) and skips with the reason when
  it is not present.
"""
from __future__ import annotations

import os
import sys

# tests/ on sys.path so `import _gridfixture` works from every subdirectory.
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

from _gridfixture import find_test_grid, missing_reason  # noqa: E402


@pytest.fixture(scope="session")
def ssp_grid_path():
    """Path (str) to a usable test SSP grid; skips the test if unavailable."""
    g = find_test_grid()
    if g is None:
        pytest.skip(missing_reason())
    return str(g)
