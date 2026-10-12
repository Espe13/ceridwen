"""``ceridwen.agn.fetch``: CIGALE's extinction curves are copied (or downloaded) into the AGN data
directory, sha256-checked, never written when they differ; ``PolarDust`` names the fetcher when
they are absent and refuses a changed table.

The copy tests that need CIGALE's real files take them from ``polar_data`` (skip without); the
download test runs only with ``CERIDWEN_TEST_NETWORK=1``.
"""
from __future__ import annotations

import hashlib
import os
import shutil

import numpy as np
import pytest

from ceridwen.agn import fetch_grids
from ceridwen.agn.fetch import FetchError
from ceridwen.agn.polar import EXTFUN_FILES, EXTFUN_SHA256, PolarDust, extfun_path

WAVE = np.geomspace(100.0, 1.0e7, 500)


def _fake_cigale(root, src=None, names=None, corrupt=False):
    curves = root / "pcigale" / "sed_modules" / "curves"
    curves.mkdir(parents=True)
    for law, name in EXTFUN_FILES.items():
        if names is not None and name not in names:
            continue
        if corrupt or src is None:
            (curves / name).write_text("# not CIGALE's table\n1.0 2.0 3.0 4.0 0.5\n")
        else:
            shutil.copy(extfun_path(law, src), curves / name)
    return root


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_from_cigale_copies_and_is_idempotent(polar_data, tmp_path, capsys):
    tree = _fake_cigale(tmp_path / "cigale", polar_data)
    out = tmp_path / "agn"
    got = fetch_grids(save=out, from_cigale=tree)
    assert sorted(got) == sorted(EXTFUN_FILES.values())
    for law, name in EXTFUN_FILES.items():
        assert _sha(out / name) == EXTFUN_SHA256[law]
    assert "CeCILL-2.0" in capsys.readouterr().out
    mtimes = {n: (out / n).stat().st_mtime_ns for n in EXTFUN_FILES.values()}
    fetch_grids(save=out, from_cigale=tree)
    assert "already present" in capsys.readouterr().out
    assert mtimes == {n: (out / n).stat().st_mtime_ns for n in EXTFUN_FILES.values()}


def test_from_cigale_with_one_file_missing(polar_data, tmp_path):
    names = list(EXTFUN_FILES.values())[:2]
    tree = _fake_cigale(tmp_path / "cigale", polar_data, names=names)
    with pytest.raises(FetchError, match="missing extFun_Gaskel04.dat"):
        fetch_grids(save=tmp_path / "agn", from_cigale=tree)


def test_from_cigale_refuses_a_changed_table(tmp_path):
    tree = _fake_cigale(tmp_path / "cigale", corrupt=True)
    with pytest.raises(FetchError, match="sha256 differs"):
        fetch_grids(save=tmp_path / "agn", from_cigale=tree)
    assert not any((tmp_path / "agn").iterdir())


def test_from_a_directory_that_is_not_cigale(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(FetchError, match="not a CIGALE checkout"):
        fetch_grids(save=tmp_path / "agn", from_cigale=tmp_path / "elsewhere")


def test_from_a_path_that_does_not_exist(tmp_path):
    with pytest.raises(FetchError, match="not a directory"):
        fetch_grids(save=tmp_path / "agn", from_cigale=tmp_path / "no_such_cigale")


def test_polar_dust_names_the_fetcher_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("CERIDWEN_AGN_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="python -m ceridwen.agn.fetch --download"):
        PolarDust(WAVE, mode="corrected", law="smc")


def test_polar_dust_refuses_a_changed_table(tmp_path, monkeypatch):
    for name in EXTFUN_FILES.values():
        (tmp_path / name).write_text("# not CIGALE's table\n1.0 2.0 3.0 4.0 0.5\n")
    monkeypatch.setenv("CERIDWEN_AGN_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="parity"):
        PolarDust(WAVE, mode="corrected", law="smc")


@pytest.mark.skipif(os.environ.get("CERIDWEN_TEST_NETWORK") != "1",
                    reason="downloads from gitlab.lam.fr; set CERIDWEN_TEST_NETWORK=1")
def test_download_from_cigale(tmp_path):
    fetch_grids(save=tmp_path, download=True)
    for law, name in EXTFUN_FILES.items():
        assert _sha(tmp_path / name) == EXTFUN_SHA256[law]
