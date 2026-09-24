"""Download-on-demand registry of published ceridwen SSP grids (cached in
$CERIDWEN_GRID_DIR, default ~/.ceridwen/grids; SHA-256 verified on every fetch).

``stellar_mass_table``: whether the published file carries the surviving-mass table
(``ssp_stellar_mass``) that ``mfrac`` needs."""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REGISTRY: dict[str, dict] = {
    "mist_miles_chab": {
        "stellar_mass_table": True,
        "url": "https://zenodo.org/records/22937956/files/"
               "ssp_data_mist_miles.h5?download=1",
        "sha256": "f2f40fe9d57b7fbdba5a60aa130c79ac"
                  "ad09c0d7f26e789d95f93092eeedf4fd",
        "size_mb": 67,
        "notes": "python-fsps 0.5.0, MIST + MILES, Chabrier IMF "
                 "(imf_type=1).  Schema 3.0: ssp_resolution = sampling "
                 "floor max MILES LSF (FWHM 2.54 A, Falcon-Barroso+2011); "
                 "surviving-mass table (FSPS stellar_mass; first isochrone bin "
                 "corrected for truncated young isochrones, Zenodo 22937956) "
                 "for mfrac.  Nebular-capable via CSPBasis.",
    },
    "mist_bpass_v2": {
        "stellar_mass_table": True,
        "url": "https://zenodo.org/records/22937956/files/"
               "ssp_data_bpass.h5?download=1",
        "sha256": "c119d19e6ade6f1de72ebf0a887e70e8"
                  "8330a1d36c6323b4d314d8baa952d151",
        "size_mb": 62,
        "notes": "python-fsps 0.5.0, BPASS v2 binary SSPs, Chabrier IMF. "
                 "Schema 3.0: ssp_resolution = grid sampling-floor curve "
                 "(no documented LSF broader than the tabulation); "
                 "surviving-mass table (FSPS stellar_mass) for mfrac.",
    },
    "amist_c3k_hr_krou_afe": {
        "stellar_mass_table": False,
        "url": "https://zenodo.org/records/22937956/files/"
               "amist_c3k_hr_krou_afe.h5?download=1",
        "sha256": "f6af03d813569f5982891d969f030d93"
                  "45278a60de907b90b2a910d56af32a16",
        "size_mb": 612,
        "notes": "MIST v2.5 (aMIST) + C3K v2.3 high-res, Kroupa IMF, "
                 "[alpha/Fe] = {-0.2, 0.0, +0.2, +0.4, +0.6}, "
                 "[Fe/H] in [-2.5, +0.5], log10(age/yr) in [5.0, 10.3]. "
                 "For CSPBasis_afe (no nebular; no FSPS needed at fit time). "
                 "No surviving-mass table yet: mfrac is not available on it. "
                 "Source: M. J. Park alpha-MC SSPs (2025-07-22).",
    },
}


def _grid_cache_root() -> Path:
    """$CERIDWEN_GRID_DIR or ~/.ceridwen/grids, without creating it."""
    root = os.environ.get("CERIDWEN_GRID_DIR")
    return Path(root) if root else Path.home() / ".ceridwen" / "grids"


def grid_cache_dir() -> Path:
    """Cache directory: $CERIDWEN_GRID_DIR or ~/.ceridwen/grids."""
    path = _grid_cache_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, path: Path, *, attempts: int = 8) -> None:
    """Download ``url`` to ``path`` completely.  A server may close the connection before
    ``Content-Length`` bytes arrive (Zenodo does, intermittently), and a chunked read then
    ends as if the file were complete; so the size is checked and a short transfer is
    resumed with an HTTP Range request, up to ``attempts`` times."""
    total = None
    with open(path, "wb"):
        pass
    for _ in range(attempts):
        have = path.stat().st_size
        if total is not None and have >= total:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:          # server ignored the range: start over
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab"
                if total is None or not have:
                    cl = r.headers.get("Content-Length")
                    total = (int(cl) + have) if cl is not None else None
                with open(path, mode) as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise                                 # wrong URL / access: retrying cannot help
            continue                                  # 5xx: retry
        except (OSError, http.client.HTTPException):
            continue                                  # retried from the bytes already written
        if total is None:
            break                                     # no length to check: the sha256 decides
    if total is not None and path.stat().st_size != total:
        raise RuntimeError(
            f"the download of {url} stopped at {path.stat().st_size} of {total} bytes after "
            f"{attempts} attempts; the connection keeps dropping.  Try again later.")


def _is_earlier_copy(name: str, sha: str) -> bool:
    """True when ``sha`` is a known earlier file of the registry grid ``name``."""
    from .grid_metadata import CHASH_TABLE, FILE_SHA_ALIASES
    ch = FILE_SHA_ALIASES.get(sha)
    return ch is not None and CHASH_TABLE[ch].name == name


def _verify_cached(name: str, dest: Path) -> None:
    """Raise RuntimeError unless the cached file ``dest`` has the registry checksum of ``name``."""
    entry = REGISTRY[name]
    got = _sha256(dest)
    if entry["sha256"] and got != entry["sha256"]:
        if _is_earlier_copy(name, got):
            raise RuntimeError(
                f"Cached grid {dest} is an earlier release of {name!r} (sha256 "
                f"{got[:12]}..., without the surviving-mass table); this version expects {entry['sha256'][:12]}....  "
                f"Refresh it with fetch_grid({name!r}, force=True)."
            )
        raise RuntimeError(
            f"Cached grid {dest} fails its checksum "
            f"(got {got[:12]}..., expected {entry['sha256'][:12]}...). "
            f"Delete it or call fetch_grid({name!r}, force=True)."
        )


def cached_grid(name: str) -> Path | None:
    """The checksum-verified cached copy of the registry grid ``name``, or ``None`` when
    :func:`fetch_grid` has not downloaded it.  Never downloads and never creates the cache
    directory; a cached file that fails its checksum raises, as in :func:`fetch_grid`."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown grid {name!r}. Available: {sorted(REGISTRY)}.")
    dest = _grid_cache_root() / f"{name}.h5"
    if not dest.is_file():
        return None
    _verify_cached(name, dest)
    return dest


def fetch_grid(name: str, *, force: bool = False, quiet: bool = False) -> Path:
    """Return a local, checksum-verified path to the registry grid ``name``,
    downloading into :func:`grid_cache_dir` on first use (``force`` re-downloads)."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown grid {name!r}. Available: {sorted(REGISTRY)}."
        )
    entry = REGISTRY[name]
    if entry["url"] is None:
        raise RuntimeError(
            f"Grid {name!r} is defined but not yet published (no URL in the "
            f"registry). Notes: {entry['notes']}"
        )

    dest = grid_cache_dir() / f"{name}.h5"

    if dest.exists() and not force:
        _verify_cached(name, dest)
        return dest

    if not quiet:
        size = f" (~{entry['size_mb']} MB)" if entry.get("size_mb") else ""
        print(f"[ceridwen] fetching grid {name!r}{size} -> {dest}",
              file=sys.stderr)

    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp)
    try:
        _download(entry["url"], tmp)
        got = _sha256(tmp)
        if entry["sha256"] and got != entry["sha256"]:
            raise RuntimeError(
                f"Downloaded grid {name!r} fails its checksum "
                f"(got {got[:12]}..., expected {entry['sha256'][:12]}...). "
                f"The remote file changed or the download was corrupted; "
                f"not installing it."
            )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()

    return dest


def available_grids(published_only: bool = False) -> dict[str, str]:
    """Map of grid name -> one-line description."""
    return {
        k: v["notes"] for k, v in REGISTRY.items()
        if v["url"] is not None or not published_only
    }
