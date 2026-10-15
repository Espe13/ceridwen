"""Pytest configuration shared by the whole ``tests/`` tree.

* Makes ``_gridfixture`` importable from test modules in subdirectories
  (``tests/csp/``, ``tests/regression/`` ...) by putting ``tests/`` on
  ``sys.path`` -- this runs before any test module is imported.
* Exposes an ``ssp_grid_path`` session fixture that resolves the test SSP
  grid (``_gridfixture``; the fetch_grid cache counts) and skips with the reason when
  it is not present.
* Exposes ``polar_data``: the directory holding CIGALE's extinction curves for the AGN
  polar dust (``ceridwen.agn.agn_data_dir()``, not shipped; ``python -m ceridwen.agn.fetch``);
  skips with the reason when they are absent.
* Switches the PyPI update check off (``CERIDWEN_NO_UPDATE_CHECK``).
"""
from __future__ import annotations

import os
import sys

# tests/ on sys.path so `import _gridfixture` works from every subdirectory.
sys.path.insert(0, os.path.dirname(__file__))

# No PyPI update check inside the tests: a newer release would otherwise add a warning to
# every fitSED call (and fail the tests that turn warnings into errors).
os.environ.setdefault("CERIDWEN_NO_UPDATE_CHECK", "1")

import pytest  # noqa: E402

from _gridfixture import find_test_grid, missing_reason  # noqa: E402


@pytest.fixture(scope="session")
def ssp_grid_path():
    """Path (str) to a usable test SSP grid; skips the test if unavailable."""
    g = find_test_grid()
    if g is None:
        pytest.skip(missing_reason())
    return str(g)


@pytest.fixture(scope="session")
def polar_data():
    """``agn_data_dir()`` when CIGALE's three extinction curves are there; skips otherwise."""
    from ceridwen.agn import agn_data_dir
    from ceridwen.agn.polar import EXTFUN_ALIASES, EXTFUN_FILES
    d = agn_data_dir()
    missing = [EXTFUN_FILES[k] for k in EXTFUN_FILES
               if not ((d / EXTFUN_FILES[k]).is_file() or (d / EXTFUN_ALIASES[k]).is_file())]
    if missing:
        pytest.skip(f"CIGALE's extinction curves {missing} not in {d} (not shipped; "
                    "python -m ceridwen.agn.fetch --download)")
    return d
