"""
Misuse report for CERIDWEN — exercises the common user-error scenarios from the
footgun analysis and records, for each, how the package now responds:

  ERROR     — raises a clear exception (loud, good)
  WARN      — emits a warnings.warn (caught, good)
  BY-DESIGN — intentional supported fallback (no error expected)
  SILENT    — returns a (possibly wrong) value with no error/warning (bad)

Produces ``figures/misuse_report.png`` (green = caught loudly/with a warning or
by design; red = still silent) and prints a table.  No FSPS needed
(add_neb / add_dust off), so it is fast.

Run:  python tests/regression/misuse_report.py
"""
from __future__ import annotations

import os
import sys
import warnings
import pathlib

# Make the shared test helper importable when run as a standalone script
# (under pytest, tests/conftest.py already puts tests/ on sys.path).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.observation.observation import Photometry, Spectrum, Lines

from _gridfixture import require_test_grid

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
REPO = HERE.parent.parent
SSP = str(require_test_grid())

_ssp = SSPData.load(SSP)
_T = 13.8
_lb = jnp.linspace(0.0, _T, 10)   # NEW convention: today @ idx 0
_sfr = jnp.exp(-0.5 * ((_lb - 0.05) / 0.03) ** 2) + 0.7 * jnp.exp(-0.5 * ((_lb - 11.) / 0.8) ** 2)


def _kw(**over):
    kw = dict(tuniv=_T, zh_const=True, add_dust=False, add_diffuse_dust=False,
              add_dust_emission=False, add_neb=False, add_igm=False,
              verbose=False, sfh_interp="linear")
    kw.update(over)
    return kw


def _run(fn):
    """Run fn, classify outcome as ERROR / WARN / SILENT."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            return ("WARN", str(w[-1].message)[:90]) if w else ("SILENT", "")
        except Exception as e:
            return ("ERROR", f"{type(e).__name__}: {str(e)[:80]}")


def _good_csp():
    return CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "Z": jnp.array([-1.85])},
                    **_kw())


def run_scenarios():
    csp = _good_csp()
    th = dict(csp.theta_init)

    scenarios = [
        # label, expected-good-kinds, fn
        ("zh_const=True, no 'Z'", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr}, **_kw())),
        ("zh_const=False, no 'zh'", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "Z": jnp.array([-1.85])},
                          **_kw(zh_const=False))),
        ("NaN in sfh", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr.at[0].set(jnp.nan),
                                       "Z": jnp.array([-1.85])}, **_kw())),
        ("negative SFR", {"WARN"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": -jnp.abs(_sfr),
                                       "Z": jnp.array([-1.85])}, **_kw())),
        ("missing 'sfh'", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "Z": jnp.array([-1.85])}, **_kw())),
        ("wrong 'Z' shape", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "Z": jnp.zeros(10)}, **_kw())),
        ("wrong 'sfh' length", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": jnp.ones(13),
                                       "Z": jnp.array([-1.85])}, **_kw())),
        ("sfh_interp typo", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "Z": jnp.array([-1.85])},
                          **_kw(sfh_interp="steppe"))),
        ("typo theta key (logmas)", {"WARN"},
         lambda: csp.get_spectrum_components({**th, "logmas": jnp.array([10.0])})),
        ("Z outside metallicity grid", {"WARN"},
         lambda: csp.check_param_ranges({**th, "Z": jnp.array([0.0])})),
        ("Spectrum.predict before setup", {"ERROR"},
         lambda: Spectrum(wavelength=jnp.linspace(4000, 7000, 40), flux=jnp.ones(40),
                          uncertainty=jnp.ones(40), name="s").predict(
                          csp.get_spectrum(th), csp.wave)),
        ("Lines.predict before setup", {"ERROR"},
         lambda: Lines(line_ind=[0, 1], line_names=["a", "b"], wavelength=[4861., 6563.],
                       name="l").predict(csp.get_spectrum(th), csp.wave)),
        ("unknown filter name", {"ERROR"},
         lambda: Photometry(filters=["not_a_filter_xyz"], name="p").setup_for_model(csp.wave)),
        ("all-zero uncertainty", {"ERROR"},
         lambda: (lambda sp: (sp.setup_for_model(csp.wave), sp.chi_sq(sp.predict(csp.get_spectrum(th), csp.wave)))) (
                  Spectrum(wavelength=jnp.linspace(4000, 7000, 40), flux=jnp.ones(40),
                           uncertainty=jnp.zeros(40), name="s"))),
        ("Photometry.predict before setup (fallback)", {"WARN", "SILENT"},
         lambda: Photometry(filters=["sdss_g0"], name="p").predict(csp.get_spectrum(th), csp.wave)),
    ]

    rows = []
    for label, good, fn in scenarios:
        kind, detail = _run(fn)
        ok = kind in good
        rows.append((label, kind, ok, detail, good))
    return rows


def make_misuse_figure(rows=None, outdir=FIG_DIR):
    if rows is None:
        rows = run_scenarios()
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in rows]
    kinds = [r[1] for r in rows]
    oks = [r[2] for r in rows]
    color = {"ERROR": "#2ca02c", "WARN": "#1f9e1f", "BY-DESIGN": "#7fbf7f", "SILENT": "#d62728"}
    # green if handled as expected, red otherwise
    bar_colors = ["#2ca02c" if ok else "#d62728" for ok in oks]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(labels) + 1.5))
    ax.barh(y, [1] * len(labels), color=bar_colors, alpha=0.85)
    for yi, (lab, kind, ok, detail, good) in zip(y, rows):
        ax.text(0.02, yi, f"{lab}", va="center", ha="left", fontsize=8,
                color="white", fontweight="bold")
        ax.text(0.98, yi, kind, va="center", ha="right", fontsize=8,
                color="white", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([])
    n_ok = sum(oks)
    ax.set_title(f"CERIDWEN misuse report — {n_ok}/{len(labels)} user mistakes "
                 f"now caught (green=loud error/warning or by-design, red=silent)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "misuse_report.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 78)
    print(f"{'scenario':<42s}{'outcome':<10s}{'handled?':<9s}")
    print("-" * 78)
    for lab, kind, ok, detail, good in rows:
        print(f"{lab:<42s}{kind:<10s}{'yes' if ok else 'NO':<9s} {detail}")
    print("=" * 78)
    print(f"Figure: {outdir / 'misuse_report.png'}")
    return rows


if __name__ == "__main__":
    make_misuse_figure()
