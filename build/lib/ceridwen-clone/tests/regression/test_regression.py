"""
Regression test for the CERIDWEN review-driven refactor.

Re-runs every Step-0 baseline evaluation (via ``capture_baseline.compute_baselines``,
the single source of truth) and compares against the committed ``baselines/*.npz``
with ``numpy.testing.assert_allclose(atol=1e-10, rtol=1e-7)``.

This must pass identically before and after every refactor phase.  Run:

    SPS_HOME=/path/to/fsps python -m pytest tests/regression/test_regression.py -x -q
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from capture_baseline import compute_baselines, BASELINE_DIR

ATOL = 1e-10
RTOL = 1e-7


@pytest.fixture(scope="session")
def fresh():
    """Compute all baselines once for the whole test session."""
    return compute_baselines()


def _load(category: str) -> dict:
    path = BASELINE_DIR / f"{category}.npz"
    assert path.exists(), f"missing baseline {path} (run capture_baseline.py)"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


CATEGORIES = [
    "ssp_spectrum",
    "igm",
    "cosmology",
    "csp_components",
    "dust_attenuation",
    "dust_emission",
    "nebular",
    "csp_spectrum",
    "likelihood",
]


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_matches_baseline(category, fresh):
    expected = _load(category)
    actual = fresh[category]
    assert set(actual.keys()) == set(expected.keys()), (
        f"{category}: key mismatch actual={sorted(actual)} expected={sorted(expected)}"
    )
    for key in expected:
        exp = np.asarray(expected[key])
        act = np.asarray(actual[key])
        assert act.shape == exp.shape, (
            f"{category}/{key}: shape {act.shape} != baseline {exp.shape}"
        )
        # Guard against silently-degenerate (all-NaN) baselines, while
        # tolerating physically-legitimate infinities (e.g. flux_factor at z=0,
        # where the luminosity distance vanishes).
        assert np.isfinite(exp).any(), f"{category}/{key}: baseline is entirely non-finite"
        finite = np.isfinite(exp)
        # Non-finite positions (inf / nan) must match exactly in location & value.
        assert np.array_equal(act[~finite], exp[~finite], equal_nan=True), (
            f"{category}/{key}: non-finite entries changed"
        )
        np.testing.assert_allclose(
            act[finite], exp[finite], atol=ATOL, rtol=RTOL,
            err_msg=f"{category}/{key} changed beyond tolerance",
        )


def test_emit_visual_report(fresh):
    """Write the human-readable comparison figures (baseline vs current +
    residuals) to tests/regression/figures/ so the run can be eyeballed, and
    assert every category matches to <=1e-6 max relative residual."""
    from plot_regression import make_all_figures

    summary = make_all_figures(fresh)
    bad = {k: v for k, v in summary.items() if v > 1e-6}
    assert not bad, f"categories exceeding 1e-6 max relative residual: {bad}"
