"""Q2-009: started from the folder that *contains* a clone named ``ceridwen``, Python finds
the clone first on sys.path.  The repository's top-level ``__init__.py`` then imports the
installed ceridwen instead of an empty namespace package, or, when none is installed,
raises one clear ImportError.  Checked in subprocesses on a stand-in clone (a copy of that
file next to a link to the real package)."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def parent(tmp_path):
    clone = tmp_path / "ceridwen"
    clone.mkdir()
    shutil.copy(REPO / "__init__.py", clone / "__init__.py")
    (clone / "ceridwen").symlink_to(REPO / "ceridwen", target_is_directory=True)
    return tmp_path


def _run(cwd, *args, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=300)


def test_the_installed_package_is_imported_not_the_clone_folder(parent):
    r = _run(parent, "-c", "import ceridwen, ceridwen.ssps; print(ceridwen.__file__); "
                           "print(ceridwen.__version__)", pythonpath=str(REPO))
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0] == str(REPO / "ceridwen" / "__init__.py")


def test_without_an_installed_package_the_error_says_what_happened(parent):
    r = _run(parent, "-S", "-c", "import ceridwen")          # -S: no site-packages at all
    assert r.returncode == 1
    last = r.stderr.strip().splitlines()[-1]
    assert last.startswith("ImportError:")
    assert "clone of the ceridwen repository" in last and "pip install" in last


def test_the_guard_is_inert_under_another_name(parent):
    r = _run(parent.parent, "-S", "-c", f"import {parent.name}.ceridwen as c; print('ok')")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
