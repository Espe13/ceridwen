"""The version string agrees in pyproject.toml, ceridwen/_version.py (``__version__`` and
``__version_tuple__``) and CITATION.cff, so a partial bump fails CI rather than being caught
by hand at release time (RELEASE_TODO.md, section 1). Reads the files; imports nothing from
the package."""
from __future__ import annotations

import ast
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _version_py() -> tuple[str, tuple]:
    tree = ast.parse((ROOT / "ceridwen" / "_version.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = ast.literal_eval(node.value)
            for t in node.targets:
                if isinstance(t, ast.Name):
                    found[t.id] = value
    return found["__version__"], found["__version_tuple__"]


def _citation() -> str:
    m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', (ROOT / "CITATION.cff").read_text(),
                  flags=re.M)
    assert m, "CITATION.cff has no top-level version: line"
    return m.group(1)


def test_version_sites_agree():
    py, (ver, tup) = _pyproject(), _version_py()
    cff = _citation()
    assert py == ver == cff, (f"version drift: pyproject.toml {py}, ceridwen/_version.py {ver}, "
                              f"CITATION.cff {cff}")
    assert tuple(int(x) for x in py.split(".")) == tuple(tup), (
        f"ceridwen/_version.py: __version_tuple__ {tup} does not match {py}")
