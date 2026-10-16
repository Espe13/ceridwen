"""The dependency markers in ``pyproject.toml`` evaluate (no exception) on every platform.

pip evaluates every clause of a marker: ``platform_release >= '25'`` raised InvalidVersion on a
Linux kernel string ('6.17.0-1022-azure') with packaging 22-25, which failed ``pip install`` on
Linux (CI, 2026-10-15).  The Darwin release is therefore compared as a string.
"""
import pathlib
import re
import tomllib

import pytest

Requirement = pytest.importorskip("packaging.requirements").Requirement

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
ENVS = {
    "linux azure": dict(sys_platform="linux", platform_machine="x86_64", python_version="3.12",
                        platform_release="6.17.0-1022-azure"),
    "linux el8":   dict(sys_platform="linux", platform_machine="x86_64", python_version="3.11",
                        platform_release="4.18.0-553.el8_10.x86_64"),
    "windows":     dict(sys_platform="win32", platform_machine="AMD64", python_version="3.12",
                        platform_release="11"),
    "macos 15":    dict(sys_platform="darwin", platform_machine="arm64", python_version="3.12",
                        platform_release="24.6.0"),
    "macos 26":    dict(sys_platform="darwin", platform_machine="arm64", python_version="3.12",
                        platform_release="25.6.0"),
    "macos 26 py3.11": dict(sys_platform="darwin", platform_machine="arm64", python_version="3.11",
                            platform_release="25.6.0"),
}


def test_no_version_comparison_on_platform_release():
    for dep in DEPS:
        assert not re.search(r"platform_release\s*[<>=!~]", dep), dep


@pytest.mark.parametrize("env", ENVS)
def test_markers_evaluate(env):
    for dep in DEPS:
        r = Requirement(dep)
        if r.marker is not None:
            r.marker.evaluate(ENVS[env])


def test_jax_graft_only_on_a_new_apple_silicon_mac():
    graft = Requirement(next(d for d in DEPS if d.startswith("jax-graft")))
    on = {k for k, e in ENVS.items() if graft.marker.evaluate(e)}
    assert on == {"macos 26"}
