"""
Capture pre-flip baselines for the lookback-time convention refactor.

Run this BEFORE editing any source.  It builds a fresh CSPBasis with the
fiducial physical SFH in the current (OLD, decreasing) lookback-time
convention for every (sfh_interp, zh_const, sfh-shape) combination, then
stores the SSP-weight matrix ``W``, the integrated spectrum, the
line-only spectrum, and a small block of synthetic photometric maggies.

The post-flip regression test loads these arrays and rebuilds the same
*physical* SFH in the NEW (increasing-from-today) convention; the four
arrays must compare bit-for-bit (W) or to float64 rounding tolerance
(spec, lines, maggies).

Usage
-----
    python tests/baselines/_capture.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp     import CSPBasis
from ceridwen.observation.observation import Photometry

# Paths
THIS_DIR = pathlib.Path(__file__).resolve().parent
# Canonical test grid (see tests/_gridfixture.py; the old
# ceridwen/data/test_data/ssp_data.h5 was retired 2026-08-17).
SSP_FILE = str(THIS_DIR.parent.parent
               / "ceridwen/data/test_data/ssp_data_bpass.h5")


# Fiducial physical SFH on the NEW (post-2026-06-03 refactor) lookback
# convention: index 0 = today, last index ≈ T_univ.  The regression test
# loads these as-is and rebuilds CSPs without further reversal.  The
# variable names ``LB_OLD`` / ``PSI_OLD`` are kept for backwards-compatibility
# with the manifest.json schema; they now hold NEW-convention arrays.
T_UNIV  = 13.8
N_TIME  = 10
LB_OLD  = jnp.linspace(0.0, T_UNIV - 1e-2, N_TIME)  # NEW convention, ascending
PSI_OLD = jnp.exp(-LB_OLD / 1.0)   # SFR at each lookback node


def _build_theta(zh_const: bool, per_bin: bool):
    sfh = PSI_OLD
    if per_bin:
        sfh = 0.5 * (PSI_OLD[:-1] + PSI_OLD[1:])
    theta = {
        "lookback_time": LB_OLD,
        "sfh":           sfh,
    }
    if zh_const:
        theta["Z"]  = jnp.asarray([-2.0])
    else:
        theta["zh"] = jnp.asarray(np.full(N_TIME, -2.0))
    return theta


def _build_csp(ssp, theta, *, sfh_interp, zh_const):
    return CSPBasis(
        ssp,
        theta             = theta,
        tuniv             = T_UNIV,
        zh_const          = zh_const,
        add_dust          = False,
        add_diffuse_dust  = False,
        add_dust_emission = False,
        add_neb           = True,
        nebemlineinspec   = True,
        verbose           = False,
        sfh_interp        = sfh_interp,
    )


def _photometry(csp):
    """Five SDSS bands, placeholder fluxes — exercise the full
    csp.predict path (filter projection + IGM/redshift hooks)."""
    phot = Photometry(
        filters     = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
        flux        = jnp.zeros(5),
        uncertainty = jnp.ones(5) * 1e-12,
        mask        = jnp.ones(5, dtype=bool),
        name        = "phot",
    )
    return phot


def main():
    ssp = SSPData.load(SSP_FILE)

    configs = []
    for sfh_interp in ("step", "linear"):
        for zh_const in (True, False):
            for per_bin in (False, True):
                if sfh_interp == "linear" and per_bin:
                    # Linear scheme expects per-node SFR; per-bin input
                    # passes the constructor's shape check but does not
                    # produce a physical SFH through the linear kernel.
                    # Skip this combo from the regression matrix.
                    continue
                configs.append((sfh_interp, zh_const, per_bin))

    out_dir = THIS_DIR
    manifest = []
    for sfh_interp, zh_const, per_bin in configs:
        tag = (f"{sfh_interp}_"
               f"{'constZ' if zh_const else 'varZ'}_"
               f"{'perbin' if per_bin else 'pernode'}")

        theta = _build_theta(zh_const, per_bin)
        csp = _build_csp(ssp, theta, sfh_interp=sfh_interp, zh_const=zh_const)
        phot = _photometry(csp)
        csp.theta_init.setdefault("logmass", jnp.zeros(1))
        csp.theta_init.setdefault("zred",    jnp.zeros(1))

        theta_full = {k: jnp.asarray(v) for k, v in csp.theta_init.items()}
        theta_full["logmass"] = jnp.zeros(1)
        theta_full["zred"]    = jnp.zeros(1)
        theta_full["igm_factor"] = jnp.ones(1)
        theta_full["eline_scaling"] = jnp.ones(1)

        # Baseline arrays.
        W       = np.asarray(csp.calculate_ssp_weights(theta_full))
        spec    = np.asarray(csp.get_spectrum(theta_full))
        lines   = np.asarray(csp.get_line_spec(theta_full))
        maggies = np.asarray(csp.predict(theta_full, [phot])[phot.name])

        np.save(out_dir / f"W_{tag}.npy",       W)
        np.save(out_dir / f"spec_{tag}.npy",    spec)
        np.save(out_dir / f"lines_{tag}.npy",   lines)
        np.save(out_dir / f"maggies_{tag}.npy", maggies)
        manifest.append({
            "tag":         tag,
            "sfh_interp":  sfh_interp,
            "zh_const":    zh_const,
            "per_bin":     per_bin,
            "shapes": {
                "W":       W.shape,
                "spec":    spec.shape,
                "lines":   lines.shape,
                "maggies": maggies.shape,
            },
        })
        print(f"[capture] {tag}: W{W.shape} spec{spec.shape} lines{lines.shape} maggies{maggies.shape}")

    import json
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"convention": "OLD (decreasing)",
                   "n_time":      N_TIME,
                   "tuniv_gyr":   T_UNIV,
                   "psi_old":     [float(x) for x in np.asarray(PSI_OLD)],
                   "lb_old_gyr":  [float(x) for x in np.asarray(LB_OLD)],
                   "configs":     manifest},
                  f, indent=2)
    print(f"[capture] wrote {len(manifest)} baseline configs to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
