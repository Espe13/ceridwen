"""Tell the user when PyPI has a newer CERIDWEN than the one installed.

``fitSED`` and ``python -m ceridwen.check`` call :func:`notify_if_outdated`.  It asks PyPI's JSON
API for the latest release at most once a day (the answer is cached in
``~/.ceridwen/update_check.json``), waits at most two seconds, and stays silent on any failure:
an offline node or a firewall never slows a fit down or stops it.  It never runs on CI (``CI``
set) and is switched off with ``CERIDWEN_NO_UPDATE_CHECK=1``.
"""
import json
import os
import re
import time
import urllib.request
import warnings
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/ceridwen/json"
CACHE = Path.home() / ".ceridwen" / "update_check.json"
MAX_AGE_S = 24 * 3600
TIMEOUT_S = 2.0

_notified = False


def _parse(v: str):
    """``"1.0.13"`` -> ``(1, 0, 13)``; None for a pre-release or local version (never nag on those)."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v.strip())
    return tuple(int(x) for x in m.groups()) if m else None


def _disabled() -> bool:
    return bool(os.environ.get("CERIDWEN_NO_UPDATE_CHECK")) or bool(os.environ.get("CI"))


def latest_version(*, use_cache: bool = True) -> str | None:
    """The latest CERIDWEN release on PyPI, or None when it cannot be found out quickly."""
    if use_cache:
        try:
            c = json.loads(CACHE.read_text())
            if time.time() - float(c["checked"]) < MAX_AGE_S:
                return c["latest"]
        except Exception:                                      # noqa: BLE001  (no or bad cache)
            pass
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=TIMEOUT_S) as r:
            latest = str(json.load(r)["info"]["version"])
    except Exception:                                          # noqa: BLE001  (offline, firewall, ...)
        return None
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"checked": time.time(), "latest": latest}))
    except OSError:
        pass
    return latest


def newer_version() -> str | None:
    """The PyPI version when it is newer than the installed one, else None."""
    if _disabled():
        return None
    from ._version import __version__
    here, latest = _parse(__version__), latest_version()
    if here is None or latest is None or _parse(latest) is None:
        return None
    return latest if _parse(latest) > here else None


def notify_if_outdated() -> str | None:
    """Warn once per process when a newer CERIDWEN is on PyPI; returns that version or None."""
    global _notified
    if _notified:
        return None
    _notified = True
    new = newer_version()
    if new is not None:
        from ._version import __version__
        warnings.warn(f"CERIDWEN {new} is available (you have {__version__}): "
                      f"pip install --upgrade ceridwen.  Release notes: "
                      f"https://github.com/Espe13/ceridwen/blob/main/CHANGELOG.md  "
                      f"(silence with CERIDWEN_NO_UPDATE_CHECK=1)", stacklevel=3)
    return new
