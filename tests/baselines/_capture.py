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
    python tests/baselines/_capture.py          # every pre-AGN config (never re-run: they are
                                                # the goldens; CLAUDE.md section 4)
    python tests/baselines/_capture.py --agn    # ONLY the AGN configs (AGN_CONFIGS): writes the
                                                # agn_* files and replaces just their manifest
                                                # entries; the existing entries are not touched
    python tests/baselines/_capture.py --agn2   # ONLY the AGN2 configs (AGN2_CONFIGS: SKIRTOR,
                                                # polar dust, Feltre+16 NLR, BLR; they need the
                                                # converted tables of $CERIDWEN_AGN_DATA)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp

import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.cosmology import Cosmology
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


# The goldens were captured at the native axis value log10 Z = -2.0 on the BPASS grid
# (Z = 0.01), which is grid node 7.  In the logzsol convention (v1.0.5) that is
# -2.0 - log10 Z_sun = -2.0 - (-1.6989700043360187), i.e. the same physical metallicity and
# the same exact node, so the stored arrays are unchanged (tested in _assert_baseline_node).
LOGZSOL_BASELINE = -0.30102999566398125


def _build_theta(zh_const: bool, per_bin: bool):
    sfh = PSI_OLD
    if per_bin:
        sfh = 0.5 * (PSI_OLD[:-1] + PSI_OLD[1:])
    theta = {
        "lookback_time": LB_OLD,
        "sfh":           sfh,
    }
    if zh_const:
        theta["logzsol"] = jnp.asarray([LOGZSOL_BASELINE])
    else:
        theta["logzsol_hist"] = jnp.asarray(np.full(N_TIME, LOGZSOL_BASELINE))
    return theta


def _build_csp(ssp, theta, *, sfh_interp, zh_const, agn=None, **over):
    kw = dict(add_dust=False, add_diffuse_dust=False, add_dust_emission=False)
    kw.update(over)
    return CSPBasis(
        ssp,
        theta             = theta,
        cosmo=Cosmology.planck18(),
        zh_const          = zh_const,
        add_neb           = True,
        init_neb_params   = {"cloudy_dust": True},   # the grid the goldens were captured with
        nebemlineinspec   = True,
        verbose           = False,
        sfh_interp        = sfh_interp,
        agn               = agn,
        **kw,
    )


# AGN golden configs (AGN_PLAN section 4 C): the step / constant-Z / per-node SFH of the
# pre-AGN goldens at zred = 2 with the IGM, plus the AGN of each mode.  "agn" holds the
# AGNConfig kwargs, "csp" CSPBasis overrides, "theta" the AGN (and dust) theta values.
AGN_CONFIGS = [
    {"tag": "agn_fsps", "agn": {"agn_mode": "fsps"},
     "csp": {"add_diffuse_dust": True, "add_igm": True},
     "theta": {"zred": 2.0, "fagn": 0.5, "agn_tau": 20.0, "diffuse_tau_kc": 0.6}},
    {"tag": "agn_prospector", "agn": {"agn_mode": "prospector", "dust4_type": 1},
     "csp": {"add_diffuse_dust": True, "add_igm": True},
     "theta": {"zred": 2.0, "fagn": 0.3, "agn_tau": 30.0, "fagn_bbb": 1.0, "dust4": 0.3,
               "dust4_index": -1.2, "agn_elum": 1e-3, "diffuse_tau_kc": 0.6}},
    {"tag": "agn_corrected_dustemi", "agn": {"agn_mode": "corrected", "dust4_type": 1},
     "csp": {"add_dust": True, "add_diffuse_dust": True, "add_dust_emission": True,
             "add_igm": True},
     "theta": {"zred": 2.0, "fagn": 0.3, "agn_tau": 30.0, "fagn_bbb": 1.0, "dust4": 0.3,
               "dust4_index": -1.2, "agn_elum": 1e-3, "diffuse_tau_kc": 0.6}},
]


# AGN2 golden configs (AGN2_BRIEF; session INT): the same SFH, the AGN2 components. "data"
# names the $CERIDWEN_AGN_DATA tables a config needs (the T3 test skips LOUDLY without them).
AGN2_CONFIGS = [
    {"tag": "agn2_skirtor_polar_cigale",
     "agn": {"agn_mode": "cigale", "torus_model": "skirtor16", "polar_dust": True,
             "lines": False},
     "csp": {"add_diffuse_dust": True, "add_igm": True}, "data": ["skirtor2016.h5"],
     "theta": {"zred": 2.0, "fagn": 0.3, "skirtor_t": 7.0, "skirtor_i": 30.0,
               "polar_ebv": 0.2, "diffuse_tau_kc": 0.6}},
    {"tag": "agn2_skirtor_polar_corrected_dustemi",
     "agn": {"agn_mode": "corrected", "torus_model": "skirtor16", "polar_dust": True,
             "lines": False},
     "csp": {"add_dust": True, "add_diffuse_dust": True, "add_dust_emission": True,
             "add_igm": True}, "data": ["skirtor2016.h5"],
     "theta": {"zred": 2.0, "fagn": 0.3, "skirtor_t": 6.0, "skirtor_i": 55.0,
               "skirtor_oa": 35.0, "polar_ebv": 0.2, "polar_temp": 150.0,
               "diffuse_tau_kc": 0.6}},
    {"tag": "agn2_nlr_blr_corrected",
     "agn": {"agn_mode": "corrected", "torus": False, "disk": False, "nlr_model": "feltre16",
             "blr": True},
     "csp": {"add_diffuse_dust": True, "add_igm": True}, "data": ["nlr_feltre16.h5"],
     "theta": {"zred": 2.0, "agn_elum": 1e-3, "nlr_logu": -2.2, "nlr_logz": -0.3,
               "blr_lum": 0.3, "blr_sigma": 1500.0, "diffuse_tau_kc": 0.6}},
]


def agn_arrays(ssp, cfg):
    """(W, spec, lines, maggies) of one AGN config, exactly as the T3 test computes them."""
    from ceridwen.agn import AGNConfig
    theta = _build_theta(True, False)
    csp = _build_csp(ssp, theta, sfh_interp="step", zh_const=True,
                     agn=AGNConfig(**cfg["agn"]), **cfg["csp"])
    phot = _photometry(csp)
    theta_full = {k: jnp.asarray(v) for k, v in csp.theta_init.items()}
    theta_full.setdefault("logmass", jnp.zeros(1))
    theta_full.setdefault("igm_factor", jnp.ones(1))
    theta_full.setdefault("eline_scaling", jnp.ones(1))
    theta_full.update({k: jnp.array([v]) for k, v in cfg["theta"].items()})
    W = np.asarray(csp.calculate_ssp_weights(theta_full))
    spec = np.asarray(csp.get_spectrum(theta_full))
    lines = np.asarray(csp.get_line_spec(theta_full))
    spec_phot = csp._apply_mass_redshift_igm(*csp._assemble_observer_spectra(theta_full),
                                             theta_full)[0]
    maggies = np.asarray(phot.get_maggies(csp.wave, spec_phot))
    return W, spec, lines, maggies


_AGN2_AMPLITUDES = ("fagn", "agn_elum", "blr_lum")


def check_agn2_zero(ssp, cfg):
    """Independent check before an AGN2 capture: with every AGN amplitude at 0 the four arrays
    equal those of the same CSP built without ``agn=`` byte for byte (the AGN2 term is additive
    and vanishes; its size is checked analytically in tests/regression/capture_baseline.py)."""
    zero = dict(cfg, theta={k: (0.0 if k in _AGN2_AMPLITUDES else v)
                            for k, v in cfg["theta"].items()})
    a = agn_arrays(ssp, zero)
    plain = {k: v for k, v in zero["theta"].items()
             if k in ("zred", "diffuse_tau_kc")}
    theta = _build_theta(True, False)
    csp = _build_csp(ssp, theta, sfh_interp="step", zh_const=True, **cfg["csp"])
    phot = _photometry(csp)
    th = {k: jnp.asarray(v) for k, v in csp.theta_init.items()}
    th.setdefault("logmass", jnp.zeros(1))
    th.setdefault("igm_factor", jnp.ones(1))
    th.setdefault("eline_scaling", jnp.ones(1))
    th.update({k: jnp.array([v]) for k, v in plain.items()})
    b = (np.asarray(csp.calculate_ssp_weights(th)), np.asarray(csp.get_spectrum(th)),
         np.asarray(csp.get_line_spec(th)),
         np.asarray(phot.get_maggies(csp.wave, csp._apply_mass_redshift_igm(
             *csp._assemble_observer_spectra(th), th)[0])))
    for name, x, y in zip(("W", "spec", "lines", "maggies"), a, b):
        assert x.tobytes() == y.tobytes(), f"{cfg['tag']}: {name} at zero AGN != no AGN"


def capture_agn(configs=None):
    """The AGN configs (default ``AGN_CONFIGS``) on the grid the T3 test loads
    (tests/_gridfixture.py); only their files and manifest entries are written."""
    configs = AGN_CONFIGS if configs is None else configs
    import json
    sys.path.insert(0, str(THIS_DIR.parent))
    from _gridfixture import require_test_grid
    grid = str(require_test_grid())
    print(f"[capture --agn] grid {grid}")
    ssp = SSPData.load(grid)
    with open(THIS_DIR / "manifest.json") as f:
        man = json.load(f)
    tags = {c["tag"] for c in configs}
    keep = [c for c in man["configs"] if c["tag"] not in tags]
    for cfg in configs:
        if "data" in cfg:
            check_agn2_zero(ssp, cfg)
        W, spec, lines, maggies = agn_arrays(ssp, cfg)
        for name, arr in (("W", W), ("spec", spec), ("lines", lines), ("maggies", maggies)):
            np.save(THIS_DIR / f"{name}_{cfg['tag']}.npy", arr)
        keep.append({"tag": cfg["tag"], "sfh_interp": "step", "zh_const": True,
                     "per_bin": False, "agn": cfg["agn"], "csp": cfg["csp"],
                     "theta": cfg["theta"],
                     **({"data": cfg["data"]} if "data" in cfg else {}),
                     "shapes": {"W": W.shape, "spec": spec.shape, "lines": lines.shape,
                                "maggies": maggies.shape}})
        print(f"[capture --agn] {cfg['tag']}: spec{spec.shape} maggies {maggies}")
    man["configs"] = keep
    with open(THIS_DIR / "manifest.json", "w") as f:
        json.dump(man, f, indent=2)


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
    if "--agn2" in sys.argv[1:]:
        sys.exit(capture_agn(AGN2_CONFIGS))
    sys.exit(capture_agn() if "--agn" in sys.argv[1:] else main())
