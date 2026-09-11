"""Static consistency check of the CERIDWEN API against its callers.

Collects the signatures of the public constructors and functions in
``ceridwen/`` and scans ``tests/``, ``examples/``, ``scripts/bit_identity_check.py`` and the fenced
Python blocks of ``README.md``, ``GOTCHAS.md`` and ``docs/*.md`` for calls that
pass a keyword argument the callee does not accept, or use a removed name.
Runs without JAX (pure ``ast``).

    python scripts/check_api_usage.py            # exit 1 on any finding
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "ceridwen"

REMOVED_KWARGS = {
    "sigma_losvd_kms", "sigma_losvd", "smoothtype", "res_convention", "fit_sigma_smooth",
    "tuniv", "inres", "resolution", "sigma_smooth", "nebular_smooth_init",
}
REMOVED_NAMES = {
    "_apply_losvd", "_setup_losvd_kernel", "NebularModelFSPSMatch", "SVDCSPBasis",
    "ThetaVector", "make_theta_vector_from_csp", "flux_factor_maggies",
}

# constructor / function names whose keyword arguments are checked
CHECKED = {
    "CSPBasis", "CSPBasis_afe", "SedModel", "Spectrum", "Photometry", "Lines", "Kinematics",
    "Instrument", "PostProcess", "fitSED", "NebularModel", "SSPData", "SSPDataAfe",
    "DiagonalNoiseModel", "GaussianProcess", "BlackJAXNestedSamplerAdapter", "BlackJAXNUTSAdapter",
    "Uniform", "Normal", "ClippedNormal", "LogNormal", "StudentT", "TopHat",
}


def _sig_of(fn: ast.FunctionDef):
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return set(names), a.kwarg is not None


def collect_signatures():
    sigs, bases = {}, {}
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in CHECKED | {"Observation"}:
                bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        sigs[node.name] = _sig_of(item)
            elif isinstance(node, ast.FunctionDef) and node.name in CHECKED:
                sigs[node.name] = _sig_of(node)
    # a constructor that forwards **kwargs accepts its base class's arguments too
    for name, (accepted, has_kwargs) in list(sigs.items()):
        if has_kwargs:
            for base in bases.get(name, []):
                if base in sigs:
                    accepted = accepted | sigs[base][0]
                    sigs[name] = (accepted, sigs[base][1])
    return sigs


def _call_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def check_source(src: str, label: str, sigs, findings):
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        findings.append(f"{label}: syntax error: {exc}")
        return
    # calls inside ``with pytest.raises(...)`` deliberately use removed arguments
    expected = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call) and _call_name(i.context_expr) == "raises"
                for i in node.items):
            expected.update(range(node.lineno, node.end_lineno + 1))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in REMOVED_NAMES:
            findings.append(f"{label}:{node.lineno}: removed name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in REMOVED_NAMES:
            findings.append(f"{label}:{node.lineno}: removed attribute {node.attr!r}")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in sigs:
                accepted, has_kwargs = sigs[name]
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    if kw.arg in REMOVED_KWARGS and kw.arg not in accepted:
                        if node.lineno not in expected:
                            findings.append(f"{label}:{node.lineno}: {name}({kw.arg}=...) removed")
                    elif kw.arg not in accepted and not has_kwargs:
                        findings.append(f"{label}:{node.lineno}: {name}() has no argument {kw.arg!r}")
                    elif kw.arg not in accepted and has_kwargs and name in ("CSPBasis", "CSPBasis_afe", "Spectrum", "Photometry", "Lines"):
                        findings.append(f"{label}:{node.lineno}: {name}() would reject {kw.arg!r}")


def markdown_blocks(path: pathlib.Path):
    text = path.read_text()
    for m in re.finditer(r"```(?:python|py)\n(.*?)```", text, flags=re.S):
        yield text[: m.start()].count("\n") + 2, m.group(1)


def main() -> int:
    sigs = collect_signatures()
    findings = []
    paths = [*(ROOT / "tests").rglob("*.py"), *(ROOT / "examples").rglob("*.py"),
             ROOT / "scripts" / "bit_identity_check.py"]
    for path in paths:
        if "_to_delete" in path.parts or "numpy_stub" in path.parts or not path.exists():
            continue
        check_source(path.read_text(), str(path.relative_to(ROOT)), sigs, findings)
    for path in [ROOT / "README.md", ROOT / "GOTCHAS.md", *sorted((ROOT / "docs").glob("*.md"))]:
        if not path.exists():
            continue
        for line0, block in markdown_blocks(path):
            block = "\n".join(l for l in block.splitlines() if not l.strip().startswith(("$", ">>>", "...")))
            try:
                ast.parse(block)
            except SyntaxError:
                continue
            check_source(block, f"{path.relative_to(ROOT)}:block@{line0}", sigs, findings)
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s); {len(sigs)} signatures checked")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
