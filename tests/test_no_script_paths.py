"""No string in the installed package names a repository script.

``scripts/`` (and the maintainer's ``scripts_afe/``) are not in the wheel, so a message,
docstring or constant that tells a user to run one points at a file they do not have.
Every string literal in ``ceridwen/**/*.py`` is checked, docstrings and f-string parts
included (comments are not strings and are not seen).  The one exemption is provenance: the
``evidence=`` of a
``grid_metadata`` entry cites where a published grid's Z_sun was derived, which is a record,
not an instruction.
"""
from __future__ import annotations

import ast
import pathlib
import re

PKG = pathlib.Path(__file__).resolve().parents[1] / "ceridwen"
SCRIPT_PATH = re.compile(r"\bscripts(_\w+)?/")


def _provenance_strings(tree) -> set:
    """ids of the string constants passed as ``evidence=`` (grid provenance citations)."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "evidence":
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant):
                    ids.add(id(sub))
    return ids


def test_no_string_names_a_script():
    hits = []
    files = sorted(PKG.rglob("*.py"))
    assert len(files) > 40, "package not found"
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        exempt = _provenance_strings(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in exempt and SCRIPT_PATH.search(node.value)):
                hits.append(f"{path.relative_to(PKG.parent)}:{node.lineno}: "
                            f"{SCRIPT_PATH.search(node.value).group(0)}")
    assert not hits, "strings naming a repository script:\n" + "\n".join(hits)


def test_the_check_finds_a_planted_path(tmp_path):
    """The walker sees f-strings, docstrings and plain constants."""
    src = ('"""Run scripts/x.py."""\nA = "ok"\nB = f"see scripts_afe/y.py {A}"\n'
           'M = dict(evidence="(scripts_afe/z.py:1)")\n')
    tree = ast.parse(src)
    exempt = _provenance_strings(tree)
    found = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and id(n) not in exempt and SCRIPT_PATH.search(n.value)]
    assert found == ["Run scripts/x.py.", "see scripts_afe/y.py "]


def test_old_published_grid_is_told_to_fetch():
    """A published grid's content hash gets 'fetch the current copy'; any other grid
    'rebuild with from_fsps'; neither names a script."""
    import pytest
    from ceridwen.ssps.grid_metadata import CHASH_TABLE
    from ceridwen.ssps.ssp_data import missing_stellar_mass_message, published_grid_name
    chash = next(c for c, m in CHASH_TABLE.items() if m.name == "mist_bpass_v2")
    assert published_grid_name(chash) == "mist_bpass_v2"
    msg = missing_stellar_mass_message("g", chash=chash)
    assert "fetch_grid('mist_bpass_v2', force=True)" in msg and "scripts/" not in msg
    # a grid that is in the chash table but not published (no registry URL) is rebuilt
    unpub = next((c for c, m in CHASH_TABLE.items() if m.name == "bpass_agb_dust"), None)
    for c in [x for x in (unpub, "chash-v1:" + "0" * 64) if x is not None] + [None]:
        m = missing_stellar_mass_message("g", chash=c)
        assert "SSPData.from_fsps records the surviving-mass table" in m and "fetch_grid" not in m
    old = PKG / "data" / "test_data" / "ssp_data_bpass_schema1_backup.h5"
    if not old.is_file():
        pytest.skip("no local schema-1 copy of the published BPASS grid")
    from ceridwen.ssps.ssp_data import SSPData
    with pytest.raises(ValueError, match=r"fetch_grid\('mist_bpass_v2', force=True\)"):
        SSPData.load(str(old))
