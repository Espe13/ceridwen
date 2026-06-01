"""
Visual regression report for CERIDWEN.

For every Step-0 baseline category this re-runs the current code
(``capture_baseline.compute_baselines``) and, for each individual array in the
category, draws its own auto-scaled panel overlaying the committed **baseline**
(thick, faint) with the freshly-computed **current** result (thin dashed), plus
a residual strip underneath.  If the refactor preserved behaviour every
"current" curve sits exactly on its "baseline" and every residual strip is flat
at 0 (to ~1e-16 round-off).

Outputs (in ``tests/regression/figures/``):
  * ``<category>.png`` — one panel per array in that category
  * ``_summary.png``   — bar chart of the max relative residual per category
                          (everything left of the 1e-6 line = MATCH)

Run standalone:

    SPS_HOME=/path/to/fsps python tests/regression/plot_regression.py

It is also called by ``test_regression.py`` so ``pytest tests/regression``
both checks the numbers and leaves you these pictures to inspect.
"""
from __future__ import annotations

import os
import math
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

from capture_baseline import compute_baselines, BASELINE_DIR

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"

_COORD_KEYS = ("wave", "z")

CATEGORIES = [
    "ssp_spectrum", "igm", "cosmology", "csp_components",
    "dust_attenuation", "dust_emission", "nebular",
    "csp_spectrum", "likelihood",
]


def _load(category: str) -> dict:
    with np.load(BASELINE_DIR / f"{category}.npz", allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _rel_resid(act: np.ndarray, exp: np.ndarray) -> float:
    a = np.asarray(act, float).ravel()
    e = np.asarray(exp, float).ravel()
    m = np.isfinite(a) & np.isfinite(e)
    if not m.any():
        return 0.0
    a, e = a[m], e[m]
    denom = np.where(np.abs(e) > 0, np.abs(e), 1.0)
    return float(np.max(np.abs(a - e) / denom))


def _coord(arrays: dict):
    for c in _COORD_KEYS:
        if c in arrays:
            return c, np.asarray(arrays[c], float).ravel()
    return None, None


def _draw_panel(host, cat, key, e, a, cx, x):
    """Draw one array's overlay + residual into a 2-row sub-grid of `host`."""
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=host,
                                  height_ratios=[3, 1], hspace=0.05)
    fig = host.get_gridspec().figure
    axt = fig.add_subplot(sub[0])
    axb = fig.add_subplot(sub[1], sharex=axt)

    e = np.asarray(e, float).ravel()
    a = np.asarray(a, float).ravel()
    resid = _rel_resid(a, e)
    is_bar = e.size <= 6
    use_x = x if (x is not None and x.size == e.size and not is_bar) else None

    if is_bar:
        idx = np.arange(e.size)
        w = 0.4
        axt.bar(idx - w / 2, e, w, color="#1f77b4", alpha=0.5, label="baseline")
        axt.bar(idx + w / 2, a, w, facecolor="none", edgecolor="#d62728",
                linewidth=1.5, label="current")
        axb.bar(idx, a - e, 0.6, color="#555")
    else:
        xx = use_x if use_x is not None else np.arange(e.size)
        axt.plot(xx, e, color="#1f77b4", lw=2.6, alpha=0.45, label="baseline")
        axt.plot(xx, a, color="#d62728", lw=0.9, ls="--", label="current")
        axb.plot(xx, a - e, color="#555", lw=0.8)
        # wide-dynamic-range, strictly positive spectra read best on log-log
        finite = np.isfinite(e)
        if cx == "wave":
            axt.set_xscale("log"); axb.set_xscale("log")
            if finite.any() and np.nanmin(e[finite]) > 0:
                axt.set_yscale("log")

    axb.axhline(0.0, color="k", lw=0.5, ls=":")
    tag = "✓" if resid <= 1e-6 else "✗ MISMATCH"
    axt.set_title(f"{key}  [{tag} {resid:.1e}]", fontsize=8)
    axt.tick_params(labelbottom=False, labelsize=6)
    axb.tick_params(labelsize=6)
    axt.legend(fontsize=6, loc="best")
    xlabel = {"wave": "λ [Å]", "z": "redshift z"}.get(cx, "index")
    axb.set_xlabel(xlabel, fontsize=7)
    axb.set_ylabel("cur−base", fontsize=6)
    return resid


def make_all_figures(fresh: dict | None = None, outdir: pathlib.Path = FIG_DIR) -> dict:
    """Generate per-category figures (one panel per array) + a summary bar chart.

    Returns ``{category: max_relative_residual}``.
    """
    if fresh is None:
        fresh = compute_baselines()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cat in CATEGORIES:
        exp = _load(cat)
        act = fresh[cat]
        cx, x = _coord(exp)
        keys = [k for k in exp.keys() if k != cx and not k.endswith("_params")]

        ncol = 1 if len(keys) == 1 else 2
        nrow = math.ceil(len(keys) / ncol)
        fig = plt.figure(figsize=(6.0 * ncol, 3.2 * nrow))
        outer = fig.add_gridspec(nrow, ncol, hspace=0.45, wspace=0.28)

        worst = 0.0
        for i, key in enumerate(keys):
            host = outer[i // ncol, i % ncol]
            worst = max(worst, _draw_panel(host, cat, key, exp[key], act[key], cx, x))
        summary[cat] = worst

        status = "MATCH ✓" if worst <= 1e-6 else "MISMATCH ✗"
        fig.suptitle(f"{cat}   —   {status}  (max rel resid {worst:.2e})\n"
                     f"baseline = thick/faint blue, current = dashed red; "
                     f"residual strip should be flat at 0",
                     fontsize=10)
        fig.savefig(outdir / f"{cat}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Summary bar chart: max relative residual per category.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cats = list(summary.keys())
    vals = [max(summary[c], 1e-18) for c in cats]  # floor for log axis
    colors = ["#2ca02c" if summary[c] <= 1e-6 else "#d62728" for c in cats]
    ax.barh(cats, vals, color=colors)
    ax.axvline(1e-6, color="k", ls="--", lw=1, label="1e-6 tolerance")
    ax.set_xscale("log")
    ax.set_xlabel("max relative residual (current vs baseline)")
    ax.set_title("CERIDWEN regression summary — all bars green & left of the "
                 "dashed line ⇒ behaviour preserved")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"{'category':<20s}{'max rel resid':>16s}   status")
    print("-" * 60)
    for cat, w in summary.items():
        print(f"{cat:<20s}{w:>16.2e}   {'MATCH' if w <= 1e-6 else 'MISMATCH'}")
    print("=" * 60)
    print(f"Figures written to {outdir}")
    return summary


if __name__ == "__main__":
    make_all_figures()
