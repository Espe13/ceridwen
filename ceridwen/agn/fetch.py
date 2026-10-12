"""Fetch CIGALE's three dust-model extinction curves (``extFun_SMC.dat``, ``extFun_MRN.dat``,
``extFun_Gaskel04.dat``) that the AGN polar dust (``polar.PolarDust``) reads.

They are CIGALE's files (CeCILL-2.0) and ceridwen does not redistribute them: the user copies
them from CIGALE (``from_cigale``: a checkout or an installed ``pcigale``, no network) or
downloads them from CIGALE's own repository (``download=True``, pinned to cigale 29cb909f; through
the system ``curl``: gitlab.lam.fr answers Python's HTTP clients with a bot check, HTTP 402).
Every file is checked against ``polar.EXTFUN_SHA256`` before it is written; nothing is ever
written into the package directory.

    python -m ceridwen.agn.fetch --from /path/to/cigale
    python -m ceridwen.agn.fetch --download
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import agn_data_dir

__all__ = ["fetch_grids", "CIGALE_COMMIT", "CIGALE_RAW_URL", "FetchError"]

CIGALE_COMMIT = "29cb909fe2636800b4acdb1dfc7129d8c8494a24"
CIGALE_RAW_URL = ("https://gitlab.lam.fr/cigale/cigale/-/raw/" + CIGALE_COMMIT
                  + "/pcigale/sed_modules/curves/{name}")
LICENCE_LINE = ("These are CIGALE's files (CeCILL-2.0), fetched from CIGALE, not redistributed "
                "by ceridwen.")


class FetchError(RuntimeError):
    """A fetch that could not complete; the message says what to do instead."""


def _tables():
    from .polar import EXTFUN_ALIASES, EXTFUN_FILES, EXTFUN_SHA256
    return [(EXTFUN_FILES[k], EXTFUN_ALIASES[k], EXTFUN_SHA256[k]) for k in sorted(EXTFUN_FILES)]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _from_cigale_hint() -> str:
    return ("copy them from CIGALE without network: python -m ceridwen.agn.fetch --from "
            "/path/to/cigale, where /path/to/cigale is a CIGALE checkout (git clone "
            "https://gitlab.lam.fr/cigale/cigale.git) or an installed pcigale (python -c "
            "\"import pcigale, os; print(os.path.dirname(pcigale.__file__))\").")


def _write(dest: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        Path(tmp).replace(dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _curl(url: str) -> bytes:
    """``url`` through the system curl (gitlab.lam.fr refuses Python's HTTP clients)."""
    exe = shutil.which("curl")
    if exe is None:
        raise OSError("curl not found on PATH")
    r = subprocess.run([exe, "-fsSL", "--max-time", "60", url], capture_output=True)
    if r.returncode != 0:
        raise OSError(f"curl exit {r.returncode}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def _search(root: Path, names) -> list:
    return [p for p in root.rglob("*") if p.name in names and p.is_file()]


def fetch_grids(save=None, from_cigale=None, download=False, overwrite=False) -> dict:
    """Put CIGALE's three extinction curves into ``save`` (default ``agn_data_dir()``: the
    directory ``PolarDust`` reads) and return ``{name: path}``.

    ``from_cigale``: a CIGALE checkout or installed ``pcigale`` directory, searched for the
    files under CIGALE's name (``extFun_SMC.dat``) or ceridwen's former one
    (``cigale_extFun_SMC.txt``). ``download``: fetch the files still missing from CIGALE's
    repository at cigale 29cb909f. Each file must match ``polar.EXTFUN_SHA256``; a mismatching
    file is never written. A correct file already in ``save`` is left alone unless
    ``overwrite``. Raises ``FetchError`` (with what to do instead) when a file cannot be got.
    """
    save = Path(save) if save is not None else agn_data_dir()
    root = Path(from_cigale).expanduser() if from_cigale is not None else None
    if root is not None and not root.is_dir():
        raise FetchError(f"--from {root}: not a directory. Instead, " + _from_cigale_hint())
    save.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    out, todo = {}, []
    for name, alias, sha in tables:
        have = [save / n for n in (name, alias) if (save / n).is_file()]
        good = [p for p in have if _sha(p.read_bytes()) == sha]
        if good and not overwrite:
            print(f"[ceridwen] {name}: already present, sha256 verified: {good[0]}")
            out[name] = good[0]
        elif have and not overwrite:
            raise FetchError(f"{have[0]} exists but is not CIGALE's {name} (sha256 differs); "
                             "re-run with --overwrite (overwrite=True) to replace it.")
        else:
            todo.append((name, alias, sha))
    if not todo:
        print(LICENCE_LINE)
        return out

    if root is not None:
        found = _search(root, {n for t in todo for n in t[:2]})
        if not found:
            raise FetchError(
                f"--from {root}: no extFun_*.dat found there, so this is not a CIGALE checkout "
                "or pcigale install (CIGALE keeps them in pcigale/sed_modules/curves/). Instead, "
                + _from_cigale_hint() + " Or download them: python -m ceridwen.agn.fetch "
                "--download")
        left = []
        for name, alias, sha in todo:
            cands = [p for p in found if p.name in (name, alias)]
            ok = [p for p in cands if _sha(p.read_bytes()) == sha]
            if ok:
                dest = save / name
                _write(dest, ok[0].read_bytes())
                print(f"[ceridwen] {name}: copied from {ok[0]} -> {dest} (sha256 verified)")
                out[name] = dest
            elif cands:
                raise FetchError(
                    f"{cands[0]} is not the {name} of cigale 29cb909f (sha256 differs); not "
                    "copied. Point --from at a CIGALE 2025.1 tree, or use --download.")
            else:
                left.append((name, alias, sha))
        todo = left

    if todo and download:
        left = []
        for name, alias, sha in todo:
            url = CIGALE_RAW_URL.format(name=name)
            try:
                data = _curl(url)
            except OSError as exc:
                print(f"[ceridwen] {name}: download failed ({exc}): {url}")
                left.append(name)
                continue
            if _sha(data) != sha:
                print(f"[ceridwen] {name}: the download from {url} is not cigale 29cb909f's "
                      "table (sha256 differs); not written.")
                left.append(name)
                continue
            dest = save / name
            _write(dest, data)
            print(f"[ceridwen] {name}: downloaded {url} -> {dest} (sha256 verified)")
            out[name] = dest
        if left:
            raise FetchError(f"could not download {', '.join(left)}. Instead, "
                             + _from_cigale_hint())
        todo = []

    if todo:
        raise FetchError(
            f"missing {', '.join(t[0] for t in todo)} in {save}. Download them from CIGALE: "
            "python -m ceridwen.agn.fetch --download. Or " + _from_cigale_hint())
    print(LICENCE_LINE)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ceridwen.agn.fetch",
        description="Put CIGALE's extinction curves (extFun_SMC.dat, extFun_MRN.dat, "
                    "extFun_Gaskel04.dat; CeCILL-2.0, not shipped with ceridwen) where the AGN "
                    "polar dust reads them: $CERIDWEN_AGN_DATA, default ~/.ceridwen/agn.")
    p.add_argument("--from", dest="from_cigale", metavar="CIGALE_DIR",
                   help="a CIGALE checkout or installed pcigale directory to copy them from "
                        "(no network)")
    p.add_argument("--download", action="store_true",
                   help="download them from CIGALE's repository (gitlab.lam.fr, cigale 29cb909f)")
    p.add_argument("--save", metavar="DIR",
                   help="target directory (default: $CERIDWEN_AGN_DATA or ~/.ceridwen/agn)")
    p.add_argument("--overwrite", action="store_true", help="replace files already present")
    a = p.parse_args(argv)
    try:
        fetch_grids(save=a.save, from_cigale=a.from_cigale, download=a.download,
                    overwrite=a.overwrite)
    except FetchError as e:
        print(f"[ceridwen] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
