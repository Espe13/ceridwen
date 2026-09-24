"""Q2-013: without FSPS, ``from_fsps`` says how to get a published grid instead."""
from __future__ import annotations

import sys

import pytest

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.ssps.ssp_data_afe import SSPDataAfe


@pytest.fixture
def no_fsps(monkeypatch):
    monkeypatch.setitem(sys.modules, "fsps", None)        # makes `import fsps` raise ImportError


def test_ssp_data_from_fsps_points_to_fetch_grid(no_fsps):
    with pytest.raises(ImportError, match=r"SSPData\.load\(ceridwen\.ssps\.fetch_grid\("
                                          r"'mist_miles_chab'\)\)"):
        SSPData.from_fsps()


def test_alpha_from_fsps_points_to_the_alpha_grid(no_fsps):
    with pytest.raises(ImportError, match=r"SSPDataAfe\.load\(ceridwen\.ssps\.fetch_grid\("
                                          r"'amist_c3k_hr_krou_afe'\)\)"):
        SSPDataAfe.from_fsps()
