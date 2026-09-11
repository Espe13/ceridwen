"""Download-on-demand registry of published ceridwen SSP grids (cached in
$CERIDWEN_GRID_DIR, default ~/.ceridwen/grids; SHA-256 verified on every fetch)."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REGISTRY: dict[str, dict] = {
    "mist_miles_chab": {
        "url": "https://zenodo.org/records/21977508/files/"
               "ssp_data_mist_miles.h5?download=1",
        "sha256": "d52f1940e4cfcf739a50e8afaea03898"
                  "71bec9404653a7e023faa53f86382f31",
        "size_mb": 67,
        "notes": "python-fsps 0.5.0, MIST + MILES, Chabrier IMF "
                 "(imf_type=1).  Schema 2.0: ssp_resolution = sampling "
                 "floor max MILES LSF (FWHM 2.54 A, Falcon-Barroso+2011). "
                 "Nebular-capable via CSPBasis.",
    },
    "mist_bpass_v2": {
        "url": "https://zenodo.org/records/21977508/files/"
               "ssp_data_bpass.h5?download=1",
        "sha256": "64c93ea751133cf3d34f4f33222af767"
                  "d029a69b9ac3549059568055a489b9e9",
        "size_mb": 62,
        "notes": "python-fsps 0.5.0, BPASS v2 binary SSPs, Chabrier IMF. "
                 "Schema 2.0: ssp_resolution = grid sampling-floor curve "
                 "(no documented LSF broader than the tabulation).",
    },
    "amist_c3k_lr_chab_afe": {
        "url": "https://zenodo.org/records/21794924/files/"
               "amist_c3k_lr_chab_afe.h5?download=1",
        "sha256": "0ae3ca192f1ba3a7825c83d77dd927ec069f4107"
                  "a52051b2e4484f80d5a47ef7",
        "size_mb": 108,
        "notes": "FSPS v4.0 alpha-MC (python-fsps >= 0.4.9.dev, AFE_FLAG=1), "
                 "aMIST + C3K_LR, Chabrier IMF, [alpha/Fe] = "
                 "{-0.2, 0.0, +0.2, +0.4, +0.6}. For CSPBasis_afe "
                 "(no nebular; no FSPS needed at fit time).  NB: this "
                 "published copy predates schema 2.x (no ssp_resolution); "
                 "after download, convert it once with "
                 "scripts/convert_grids_schema2.py, or use "
                 "'amist_c3k_hr_krou_afe' (schema 2.1, published in v5).",
    },
    "amist_c3k_hr_krou_afe": {
        "url": "https://zenodo.org/records/21977508/files/"
               "amist_c3k_hr_krou_afe.h5?download=1",
        "sha256": "f6af03d813569f5982891d969f030d93"
                  "45278a60de907b90b2a910d56af32a16",
        "size_mb": 612,
        "notes": "MIST v2.5 (aMIST) + C3K v2.3 high-res, Kroupa IMF, "
                 "[alpha/Fe] = {-0.2, 0.0, +0.2, +0.4, +0.6}, "
                 "[Fe/H] in [-2.5, +0.5], log10(age/yr) in [5.0, 10.3]. "
                 "High-resolution twin of amist_c3k_lr_chab_afe for "
                 "CSPBasis_afe (no nebular; no FSPS needed at fit time). "
                 "Source: M. J. Park alpha-MC SSPs (2025-07-22).",
    },
    "mist_c3k_lr_chab_null": {
        "url": None,
        "sha256": None,
        "size_mb": None,
        "notes": "FSPS v4.0 (AFE_FLAG=0), MIST + C3K_LR, Chabrier IMF, "
                 "n_afe=1. Null model separating C3K-library effects from "
                 "alpha effects.",
    },
}


def grid_cache_dir() -> Path:
    """Cache directory: $CERIDWEN_GRID_DIR or ~/.ceridwen/grids."""
    root = os.environ.get("CERIDWEN_GRID_DIR")
    path = Path(root) if root else Path.home() / ".ceridwen" / "grids"
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
            f"registry). Build it locally with scripts_afe/build_afe_grid.py, "
            f"or publish it with scripts_afe/publish_grid_zenodo.py and "
            f"paste the printed REGISTRY entry. Notes: {entry['notes']}"
        )

    dest = grid_cache_dir() / f"{name}.h5"

    if dest.exists() and not force:
        got = _sha256(dest)
        if entry["sha256"] and got != entry["sha256"]:
            raise RuntimeError(
                f"Cached grid {dest} fails its checksum "
                f"(got {got[:12]}..., expected {entry['sha256'][:12]}...). "
                f"Delete it or call fetch_grid({name!r}, force=True)."
            )
        return dest

    if not quiet:
        size = f" (~{entry['size_mb']} MB)" if entry.get("size_mb") else ""
        print(f"[ceridwen] fetching grid {name!r}{size} -> {dest}",
              file=sys.stderr)

    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp)
    try:
        with urllib.request.urlopen(entry["url"]) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
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
