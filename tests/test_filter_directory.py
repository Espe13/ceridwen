"""User filter directories: ``Photometry(filter_directory=)`` and ``$CERIDWEN_FILTER_DIR``.

Lookup order (``observation/filters.py:resolve_filter_file``): an explicit directory only;
otherwise ``$CERIDWEN_FILTER_DIR``, then the bundled library, with a warning when a name is
in both.  The independent check of the new path: a library ``.par`` copied under a new name
gives the library filter's maggies byte for byte.
"""
import os
import shutil
import warnings

import numpy as np
import pytest

import ceridwen  # noqa: F401  (sets the precision)
from ceridwen.observation import Photometry
from ceridwen.observation.filters import (FILTER_DIR_ENV, Filter, library_filter_dir,
                                          list_available_filters)
from ceridwen.observation.photometry import _FILTERSET_CACHE

LIB = ["jwst_f200w", "jwst_f444w"]
WAVE = np.geomspace(912.0, 3e4, 4000)                  # rest-frame model grid [A]
FNU = 1e-29 * (WAVE / 5000.0) ** 1.5 * (1.0 + 0.3 * np.sin(WAVE / 300.0))


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(FILTER_DIR_ENV, raising=False)


def _copy(tmp_path, pairs):
    for src, dst in pairs:
        shutil.copyfile(os.path.join(library_filter_dir(), f"{src}.par"), tmp_path / f"{dst}.par")
    return tmp_path


def _arrays(ph, zred=2.0):
    ph.setup_for_model(WAVE, zred=zred)
    return (np.asarray(ph._T), np.asarray(ph.predict(FNU, WAVE)),
            np.asarray(ph.get_maggies(WAVE, FNU)), np.asarray(ph.wavelength))


def _assert_bytes(a, b):
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def test_filter_directory_matches_the_library_byte_for_byte(tmp_path):
    d = _copy(tmp_path, [("jwst_f200w", "my_a"), ("jwst_f444w", "my_b")])
    ref = _arrays(Photometry(filters=LIB, name="lib"))
    new = Photometry(filters=["my_a", "my_b"], filter_directory=d, name="mine")
    _assert_bytes(_arrays(new), ref)
    assert new.filterfiles == [str(d / "my_a.par"), str(d / "my_b.par")]
    assert new.filter_directory == str(d)


def test_env_directory_matches_the_library_byte_for_byte(tmp_path, monkeypatch):
    d = _copy(tmp_path, [("jwst_f200w", "my_a"), ("jwst_f444w", "my_b")])
    ref = _arrays(Photometry(filters=LIB, name="lib"))
    monkeypatch.setenv(FILTER_DIR_ENV, str(d))
    new = Photometry(filters=["my_a", "my_b"], name="mine")
    _assert_bytes(_arrays(new), ref)
    assert new.filterfiles == [str(d / "my_a.par"), str(d / "my_b.par")]
    # library names still resolve while the variable is set
    assert Photometry(filters=["sdss_r0"], name="r").filterfiles[0].startswith(library_filter_dir())
    assert {"my_a", "my_b", "sdss_r0"} <= set(list_available_filters())
    assert "my_a" not in list_available_filters(directory=library_filter_dir())


def test_env_shadowing_a_library_name_warns_and_uses_the_env_file(tmp_path, monkeypatch):
    # sdss_r0 in the env directory holds the F444W curve, so the shadow is visible
    d = _copy(tmp_path, [("jwst_f444w", "sdss_r0")])
    monkeypatch.setenv(FILTER_DIR_ENV, str(d))
    with pytest.warns(UserWarning, match="in \\$CERIDWEN_FILTER_DIR and in the bundled library") as rec:
        ph = Photometry(filters=["sdss_r0"], name="p")
    msg = str(rec[0].message)
    assert str(d / "sdss_r0.par") in msg and os.path.join(library_filter_dir(), "sdss_r0.par") in msg
    assert len([w for w in rec if "bundled library" in str(w.message)]) == 1   # once per name
    assert ph.filterfiles == [str(d / "sdss_r0.par")]
    assert float(ph.wavelength[0]) == pytest.approx(float(Filter("jwst_f444w").wave_effective))


def test_explicit_directory_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv(FILTER_DIR_ENV, str(_copy(tmp_path, [("sdss_r0", "only_in_env")])))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FileNotFoundError, match="no other location is searched"):
        Photometry(filters=["sdss_r0"], filter_directory=other, name="p")
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # an explicit directory never warns
        Photometry(filters=["only_in_env"], filter_directory=tmp_path, name="p")


def test_missing_name_lists_every_location(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError) as e:
        Photometry(filters=["not_a_filter_xyz"], name="p")
    assert os.path.join(library_filter_dir(), "not_a_filter_xyz.par") in str(e.value)
    monkeypatch.setenv(FILTER_DIR_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError) as e:
        Photometry(filters=["not_a_filter_xyz"], name="p")
    msg = str(e.value)
    assert str(tmp_path / "not_a_filter_xyz.par") in msg
    assert os.path.join(library_filter_dir(), "not_a_filter_xyz.par") in msg


def test_same_names_from_two_directories_are_two_filter_sets(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _copy(tmp_path / "a", [("jwst_f200w", "band")])
    b = _copy(tmp_path / "b", [("jwst_f444w", "band")])
    pa = Photometry(filters=["band"], filter_directory=a, name="pa")
    pb = Photometry(filters=["band"], filter_directory=b, name="pb")
    assert pa.filterset is not pb.filterset
    assert float(pa.wavelength[0]) < 3e4 < float(pb.wavelength[0])
    # the same directory again shares its set; so do two spellings of one directory
    assert Photometry(filters=["band"], filter_directory=str(a) + os.sep + ".",
                      name="pa2").filterset is pa.filterset
    # an env directory is its own cache entry too
    monkeypatch.setenv(FILTER_DIR_ENV, str(b))
    pe = Photometry(filters=["band"], name="pe")
    assert pe.filterset is not pa.filterset and pe.filterset is not pb.filterset
    assert {k[1:] for k in _FILTERSET_CACHE if k[0] == ("band",)} >= {
        (str(a), None), (str(b), None), (None, str(b))}


def test_filter_objects_from_elsewhere_raise(tmp_path):
    d = _copy(tmp_path, [("jwst_f444w", "my_b")])
    f = Filter("jwst_f444w")
    with pytest.raises(ValueError, match="data= arrays"):
        Photometry(filters=[Filter("x", data=(np.asarray(f.wavelength), np.asarray(f.transmission)))],
                   name="p")
    with pytest.raises(ValueError, match="filter_directory="):
        Photometry(filters=[Filter("my_b", directory=d)], name="p")
    # a library Filter works as before, also mixed with names; a Filter from the
    # directory Photometry reads is accepted
    ph = Photometry(filters=[f, "sdss_r0"], name="p")
    assert ph.filternames == ["jwst_f444w", "sdss_r0"]
    assert Photometry(filters=[Filter("my_b", directory=d)], filter_directory=d,
                      name="p").filterfiles == [str(d / "my_b.par")]


def test_home_directory_is_expanded(tmp_path, monkeypatch):
    (tmp_path / "filt").mkdir()
    _copy(tmp_path / "filt", [("sdss_r0", "my_r")])
    monkeypatch.setenv("HOME", str(tmp_path))
    ph = Photometry(filters=["my_r"], filter_directory="~/filt", name="p")
    assert ph.filterfiles == [str(tmp_path / "filt" / "my_r.par")]
    monkeypatch.setenv(FILTER_DIR_ENV, "~/filt")
    assert Photometry(filters=["my_r"], name="p").filterfiles == ph.filterfiles
    assert "my_r" in list_available_filters()
