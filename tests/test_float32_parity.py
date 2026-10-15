"""float32 mode (Apple GPU, ``docs/mac_gpu.md``): the forward model and the likelihood agree
with float64 to the tolerances predicted below, and the Metal backend agrees with the float32
CPU run.

``jax_enable_x64`` is process-global, so every mode runs in its own subprocess of
``tests/baselines/_capture_f32.py`` (float64 reference; ``CERIDWEN_X64=0`` on the CPU; float32
on the Apple GPU when jax-graft is installed, else those tests skip).  The goldens compared are
the six configs of ``tests/baselines/manifest.json`` (never re-captured here) and a mock galaxy
in physical units (cgs spectrum with sigma ~ 1e-31, maggies, erg/s/cm^2 lines) fitted by the
production likelihood.

Tolerances were PREDICTED before the first run (CLAUDE.md section 5); the first run
(2026-10-15, M3 Pro, jax 0.11.2, jax-graft 0.0.5) measured them, and the asserted values
below are the measured ones with a factor ~2 of room.  Predicted / measured / asserted:

* ``W`` (SSP weights): predicted ``1e-6 max|W|`` (float32 on the SFH nodes); measured 3.3e-6
  of the maximum for ``step`` and 5.2e-6 for ``linear`` (its kernel multiplies SFR by node
  spacings of ~1e9 yr in float32, 146 ulp at the largest weight); asserted ``1e-5 max|W|``.
* ``spec`` / ``lines``: predicted ``rtol 1e-5``; measured 5e-6 relative on the continuum but
  4e-4 relative on the pixels under a nebular line, 4e-5 of the spectrum's peak: the line
  profile's argument ``(lambda - lambda_0) / sigma`` is evaluated in float32, where lambda ~ 1e4
  Angstrom has a resolution of 1e-3 Angstrom, so the profile wings move by that fraction of
  a pixel.  Asserted: continuum pixels (``lines == 0``) ``rtol 2e-5``; all pixels
  ``|d| <= 1e-4 max + 1e-3 |ref|``.
* ``maggies`` and the mock's photometry / line fluxes: predicted ``rtol 1e-5``; measured
  2.3e-5 (the filter integrals in float32), 6e-6 and 4e-6; asserted ``rtol 5e-5``.
* the mock's observed spectrum: measured 3e-4 of its peak (the broadened lines, as above);
  asserted ``|d| <= 1e-3 max + 1e-3 |ref|``.
* mock ln posterior: predicted ``|dlnP| <= 1e-4 |lnP| + 0.05``; measured 4e-4 and 0.015 on
  values of 4e4 and 2e4 (after the float32 offset of 41839.75 was added back); asserted as
  predicted.
* mock gradient: predicted ``|dg| <= 1e-3 max|g| + 1e-3 |g|``; measured 5e-5 of the largest
  component (1.2 per cent of the smallest, ``gas_logz``); asserted as predicted.
* Metal vs float32 CPU: predicted ``rtol 1e-4``; measured 2e-7 for the spectra, 9e-7 for
  the mock's photometry but 3.4e-5 for the golden maggies (15000-pixel filter integrals at
  logmass = 0, a float32 sum reordered on the GPU), and ln P equal to 0.01 on 2e4; asserted
  ``rtol 1e-5`` (golden maggies ``1e-4``) and ``|dlnP| <= 1e-4 |lnP| + 0.05``.
* Metal run to run: predicted ``rtol 1e-6`` everywhere; measured: every forward-model array
  and both ln P byte-identical, only the gradients differ (1e-5 to 6e-4 relative over three runs: the transpose
  of the redshift interpolation is a scatter-add, atomic on the GPU; CLAUDE.md's GPU caveat).
  Asserted: forward model and ln P byte-identical, gradients ``rtol 1e-3``.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from _gridfixture import find_test_grid

pytestmark = pytest.mark.fsps

HERE = pathlib.Path(__file__).resolve().parent
CAPTURE = HERE / "baselines" / "_capture_f32.py"
TAGS = ["step_constZ_pernode", "step_constZ_perbin", "step_varZ_pernode", "step_varZ_perbin",
        "linear_constZ_pernode", "linear_varZ_pernode"]
MOCK_KEYS = ("phot", "spec", "lines")

_HAVE_METAL = (sys.platform == "darwin" and importlib.util.find_spec("jax_graft") is not None)


def _run_capture(out: pathlib.Path, env_extra: dict, mock_data: pathlib.Path | None):
    env = {k: v for k, v in os.environ.items() if k not in ("CERIDWEN_X64", "JAX_PLATFORMS")}
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HERE.parent), env.get("PYTHONPATH")) if p)
    cmd = [sys.executable, str(CAPTURE), str(out)]
    if mock_data is not None:
        cmd += ["--mock-data", str(mock_data)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"{' '.join(cmd)} failed ({proc.returncode}) with {env_extra}:\n"
                    f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return dict(np.load(out))


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    if find_test_grid() is None:
        pytest.skip("no test grid")
    d = tmp_path_factory.mktemp("f32parity")
    arrays = _run_capture(d / "ref_f64.npz", {"JAX_PLATFORMS": "cpu"}, None)
    assert str(arrays["precision"]) == "float64"
    np.savez(d / "ref_f64.npz", **arrays)
    arrays["_path"] = d / "ref_f64.npz"
    return arrays


@pytest.fixture(scope="module")
def f32_cpu(ref, tmp_path_factory):
    d = tmp_path_factory.mktemp("f32parity_cpu")
    arrays = _run_capture(d / "f32_cpu.npz", {"JAX_PLATFORMS": "cpu", "CERIDWEN_X64": "0"},
                          ref["_path"])
    assert str(arrays["precision"]) == "float32"
    return arrays


@pytest.fixture(scope="module")
def f32_metal(ref, tmp_path_factory):
    if not _HAVE_METAL:
        pytest.skip("jax-graft (Apple GPU) not installed")
    d = tmp_path_factory.mktemp("f32parity_metal")
    runs = [_run_capture(d / f"f32_metal_{i}.npz", {"JAX_PLATFORMS": "mtl,cpu"}, ref["_path"])
            for i in (1, 2)]
    for r in runs:
        assert str(r["precision"]) == "float32"
        assert str(r["backend"]) == "mtl"
    return runs


def _assert_close(got, want, *, rtol, atol_frac, what):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, what
    assert np.all(np.isfinite(got)), f"{what}: non-finite values"
    atol = atol_frac * float(np.max(np.abs(want)))
    diff = np.abs(got - want)
    worst = float(np.max(diff / (atol + rtol * np.abs(want))))
    assert worst <= 1.0, (f"{what}: max|diff| {float(diff.max()):.3e}, worst {worst:.2f} x the "
                          f"allowance (rtol {rtol:g}, atol {atol:.3e})")


# --------------------------------------------------------------------------- float32 vs float64

def test_float64_reference_matches_the_committed_goldens(ref):
    """The subprocess reference reproduces tests/baselines/*.npy (T3's tolerances), so the
    float32 comparison below is against the real goldens."""
    for tag in TAGS:
        base = HERE / "baselines"
        np.testing.assert_allclose(ref[f"W_{tag}"], np.load(base / f"W_{tag}.npy"), rtol=1e-12, atol=0)
        np.testing.assert_allclose(ref[f"spec_{tag}"], np.load(base / f"spec_{tag}.npy"), rtol=1e-6, atol=0)
        np.testing.assert_allclose(ref[f"maggies_{tag}"], np.load(base / f"maggies_{tag}.npy"), rtol=1e-6, atol=0)


@pytest.mark.parametrize("tag", TAGS)
def test_float32_goldens(ref, f32_cpu, tag):
    assert f32_cpu[f"spec_{tag}"].dtype == np.float32
    _assert_close(f32_cpu[f"W_{tag}"], ref[f"W_{tag}"], rtol=1e-4, atol_frac=1e-5, what=f"W {tag}")
    continuum = ref[f"lines_{tag}"] == 0
    _assert_close(f32_cpu[f"spec_{tag}"][continuum], ref[f"spec_{tag}"][continuum], rtol=2e-5,
                  atol_frac=1e-6, what=f"spec continuum {tag}")
    for name in ("spec", "lines"):
        _assert_close(f32_cpu[f"{name}_{tag}"], ref[f"{name}_{tag}"], rtol=1e-3, atol_frac=1e-4,
                      what=f"{name} {tag}")
    _assert_close(f32_cpu[f"maggies_{tag}"], ref[f"maggies_{tag}"], rtol=5e-5, atol_frac=1e-6,
                  what=f"maggies {tag}")


def test_float32_mock_predictions(ref, f32_cpu):
    for key in MOCK_KEYS:
        np.testing.assert_array_equal(f32_cpu[f"data_{key}"], ref[f"data_{key}"].astype(np.float32))
    _assert_close(f32_cpu["pred_phot"], ref["pred_phot"], rtol=5e-5, atol_frac=1e-6, what="mock phot")
    _assert_close(f32_cpu["pred_lines"], ref["pred_lines"], rtol=5e-5, atol_frac=1e-6, what="mock lines")
    _assert_close(f32_cpu["pred_spec"], ref["pred_spec"], rtol=1e-3, atol_frac=1e-3, what="mock spec")


def test_float32_mock_lnprob_and_grad(ref, f32_cpu):
    assert float(ref["lnprob_offset"]) == 0.0
    assert float(f32_cpu["lnprob_offset"]) != 0.0                 # the data were rescaled
    for which in ("lnprob_0", "lnprob_1"):
        want, got = float(ref[which]), float(f32_cpu[which])
        assert np.isfinite(got), which
        assert abs(got - want) <= 1e-4 * abs(want) + 0.05, f"{which}: {got} vs {want}"
    grads = sorted(k for k in ref if k.startswith("grad_"))
    gmax = max(float(np.max(np.abs(ref[k]))) for k in grads)
    for k in grads:
        g, w = np.asarray(f32_cpu[k], dtype=np.float64), np.asarray(ref[k])
        assert np.all(np.isfinite(g)), k
        assert np.all(np.abs(g - w) <= 1e-3 * gmax + 1e-3 * np.abs(w)), f"{k}: {g} vs {w}"


# --------------------------------------------------------------------------- Apple GPU

def test_metal_matches_float32_cpu(f32_cpu, f32_metal):
    m = f32_metal[0]
    for tag in TAGS:
        _assert_close(m[f"W_{tag}"], f32_cpu[f"W_{tag}"], rtol=1e-5, atol_frac=1e-6, what=f"metal W {tag}")
        for name in ("spec", "lines"):
            _assert_close(m[f"{name}_{tag}"], f32_cpu[f"{name}_{tag}"], rtol=1e-5, atol_frac=1e-6,
                          what=f"metal {name} {tag}")
        # the golden maggies integrate 15000-pixel spectra at logmass = 0 through the filters:
        # a float32 sum of that length reordered on the GPU; measured 3.4e-5 relative
        _assert_close(m[f"maggies_{tag}"], f32_cpu[f"maggies_{tag}"], rtol=1e-4, atol_frac=1e-6,
                      what=f"metal maggies {tag}")
    for key in MOCK_KEYS:
        _assert_close(m[f"pred_{key}"], f32_cpu[f"pred_{key}"], rtol=1e-5, atol_frac=1e-6,
                      what=f"metal mock prediction {key}")
    for which in ("lnprob_0", "lnprob_1"):
        want, got = float(f32_cpu[which]), float(m[which])
        assert abs(got - want) <= 1e-4 * abs(want) + 0.05, f"metal {which}: {got} vs {want}"


def test_metal_is_reproducible(f32_metal):
    a, b = f32_metal
    worst = 0.0
    for k in a:
        if k.startswith(("precision", "backend", "jax_version", "data_", "sigma_")):
            continue
        x, y = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        if not k.startswith("grad_"):
            np.testing.assert_array_equal(x, y, err_msg=f"{k}: Metal forward model / ln P not byte-identical run to run")
            continue
        scale = np.maximum(np.abs(x), np.abs(y))
        rel = float(np.max(np.where(scale > 0, np.abs(x - y) / np.where(scale > 0, scale, 1.0), 0.0)))
        worst = max(worst, rel)
        assert rel <= 1e-3, f"{k}: run-to-run relative difference {rel:.3e}"
    print(f"\nMetal run-to-run: forward model and ln P identical; largest gradient difference {worst:.3e}")
