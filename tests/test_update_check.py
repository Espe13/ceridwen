"""``ceridwen.update_check``: a newer PyPI release is reported once, and nothing else is."""
import json
import time

import pytest

import ceridwen.update_check as uc


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, "CACHE", tmp_path / "update_check.json")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("CERIDWEN_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(uc, "_notified", False)
    return uc.CACHE


def _cached(cache, latest):
    cache.write_text(json.dumps({"checked": time.time(), "latest": latest}))


def test_parse():
    assert uc._parse("1.0.13") == (1, 0, 13) and uc._parse("1.0.10") > uc._parse("1.0.9")
    assert uc._parse("1.0.14rc1") is None and uc._parse("1.0.13+g1234") is None


def test_newer_release_warns_once(cache, monkeypatch):
    import ceridwen._version as v
    monkeypatch.setattr(v, "__version__", "1.0.12")
    _cached(cache, "1.0.13")
    with pytest.warns(UserWarning, match=r"1\.0\.13 is available \(you have 1\.0\.12\)"):
        assert uc.notify_if_outdated() == "1.0.13"
    assert uc.notify_if_outdated() is None                    # once per process


@pytest.mark.parametrize("latest", ["1.0.12", "1.0.11", "2.0.0rc1"])
def test_same_older_or_prerelease_is_silent(cache, monkeypatch, latest):
    import ceridwen._version as v
    monkeypatch.setattr(v, "__version__", "1.0.12")
    _cached(cache, latest)
    assert uc.newer_version() is None


def test_offline_is_silent_and_fast(cache, monkeypatch):
    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(uc.urllib.request, "urlopen", boom)
    assert uc.latest_version() is None and not cache.exists()


def test_switched_off(cache, monkeypatch):
    import ceridwen._version as v
    monkeypatch.setattr(v, "__version__", "1.0.0")
    _cached(cache, "9.9.9")
    for var in ("CERIDWEN_NO_UPDATE_CHECK", "CI"):
        monkeypatch.setenv(var, "1")
        assert uc.newer_version() is None
        monkeypatch.delenv(var)
    assert uc.newer_version() == "9.9.9"


def test_stale_cache_is_refreshed(cache, monkeypatch):
    cache.write_text(json.dumps({"checked": time.time() - 2 * uc.MAX_AGE_S, "latest": "1.0.0"}))

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"info": {"version": "1.0.99"}}'
    monkeypatch.setattr(uc.urllib.request, "urlopen", lambda *a, **k: R())
    assert uc.latest_version() == "1.0.99"
    assert json.loads(cache.read_text())["latest"] == "1.0.99"
