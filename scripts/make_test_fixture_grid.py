#!/usr/bin/env python
"""
make_test_fixture_grid.py
=========================
Build the local test SSP grid the test suite resolves by default, writing it
to ``ceridwen/data/test_data/ssp_data_bpass.h5`` (NOT committed — the
``data/`` .gitignore rule excludes it; the committed LFS fixture
``ssp_data_test.h5`` was retired 2026-08-17).

Run this on a machine with a BPASS python-fsps build, or skip FSPS entirely:
download the BPASS grid from Zenodo (doi:10.5281/zenodo.21221634), convert it
with ``scripts/convert_grids_schema2.py`` if it predates schema 2.0, and place
it at that path.  Machines without any grid skip the grid-dependent tests
cleanly (see ``tests/_gridfixture.py``; ``$CERIDWEN_TEST_SSP`` overrides).

Extra FSPS settings can be passed as ``key=value`` tokens, e.g.::

    python scripts/make_test_fixture_grid.py imf_type=1

Platform note
-------------
On macOS the Apple Metal JAX backend lacks float64 (needed for the
metallicity-axis ``log10``), so we force the CPU backend unless ``--gpu`` is
passed.
"""
from __future__ import annotations

import os
import platform
import sys

# --- platform handling (must happen BEFORE jax / ceridwen import) -----------
_on_mac = platform.system() == "Darwin"
_want_gpu = "--gpu" in sys.argv
if _on_mac and not _want_gpu:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ---------------------------------------------------------------------------

import pathlib  # noqa: E402

OUTPUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "ceridwen" / "data" / "test_data" / "ssp_data_bpass.h5"
)


def _parse_kv(tok: str):
    k, v = tok.split("=", 1)
    for caster in (int, float):
        try:
            return k, caster(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    return k, v


def main() -> int:
    fsps_kwargs = dict(
        _parse_kv(tok)
        for tok in sys.argv[1:]
        if "=" in tok and not tok.startswith("--")
    )

    try:
        import fsps  # noqa: F401
    except ImportError:
        print(
            "ERROR: FSPS (python-fsps) is not importable. The fixture grid can "
            "only be built on a machine with FSPS installed. See the README "
            "Installation section and `pip install -e '.[grids]'`.",
            file=sys.stderr,
        )
        return 1

    from ceridwen.ssps.ssp_data import SSPData

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building local test SSP grid -> {OUTPUT}")
    if fsps_kwargs:
        print(f"  FSPS kwargs: {fsps_kwargs}")

    SSPData.from_fsps(save_to=str(OUTPUT), **fsps_kwargs)

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Done. {OUTPUT} ({size_mb:.1f} MB)")
    if size_mb > 50:
        print(
            "NOTE: fixture is >50 MB; consider tracking it with git-lfs "
            "(`git lfs track 'tests/fixtures/*.h5'`)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
