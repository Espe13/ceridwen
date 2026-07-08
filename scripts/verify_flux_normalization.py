#!/usr/bin/env python3
"""Verify the absolute photometric normalization after the zred-injection fix.

Traces one predicted band through every factor of

    F_nu,obs = (1+z) * M * L_nu / (4 pi D_L^2 * 3631 Jy)   (10 pc reference)

and checks a z = 0.1, logmass = 10.5 galaxy lands at ~1e-7 maggies (AB ~ 17-18).

Run:  conda activate ceridwen311 && python scripts/verify_flux_normalization.py
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.cosmology import flux_factor_maggies, luminosity_distance_mpc

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE.parent / "examples" / "ssp_data.h5"

ZRED, LOGMASS = 0.1, 10.5
FILTERS = ["galex_FUV", "galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0",
           "sdss_i0", "sdss_z0", "twomass_J", "twomass_H", "twomass_Ks",
           "wise_w1", "wise_w2"]
TRUTH = {
    "logsfr_ratios":      jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),
    "logmass":            jnp.array([LOGMASS]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}

# ── 1. cosmology factors in isolation ──────────────────────────────────────
dl_native = float(luminosity_distance_mpc(ZRED))
ff_native = float(flux_factor_maggies(ZRED))
print("── cosmology ─────────────────────────────────────────────")
print(f"D_L({ZRED}) native            = {dl_native:.2f} Mpc   (Planck18: 475.6)")
print(f"flux_factor_maggies({ZRED})   = {ff_native:.6e}  (expect 1.5532e-22)")
print(f"flux_factor_maggies(0.0)   = {float(flux_factor_maggies(0.0)):.6e}"
      f"  (expect 3.1968e-07, 10 pc convention)")
print(f"flux_factor_maggies(-0.05) = {float(flux_factor_maggies(-0.05)):.6e}"
      f"  (expect 3.1968e-07: z<=0 pinned, no silent clamp)")
assert np.isclose(ff_native, 1.553165e-22, rtol=1e-3), "ff(0.1) wrong!"

# ── 2. end-to-end model prediction ─────────────────────────────────────────
ssp = SSPData.load(str(SSP_FILE))
csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
               zh_const=True, sfh_interp="step",
               add_dust=False, add_diffuse_dust=True, add_neb=False,
               verbose=False)
sfh_times_yr = np.array(csp.sfh_times)
model = SedModel(
    csp, observations=[Photometry(filters=FILTERS, name="phot")],
    transforms={"sfh": lambda th, _t=sfh_times_yr:
                logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
    free_param_init={"logsfr_ratios": jnp.zeros(5),
                     "logmass": jnp.array([10.0])},
    zred=ZRED,
)

print("\n── model plumbing ────────────────────────────────────────")
print(f"zred in sampled theta_init : {'zred' in model.theta_init}"
      f"   (must be False — not a sampled dimension)")
print(f"model._zred_fixed          : {model._zred_fixed}"
      f"   (injected into csp theta at predict time)")
assert "zred" not in model.theta_init
assert model._zred_fixed is not None

maggies = np.asarray(model.predict(TRUTH)["phot"])   # TRUTH has NO zred key
ab = -2.5 * np.log10(np.clip(maggies, 1e-30, None))
print("\n── predicted photometry (TRUTH without an explicit zred) ─")
for f, m, a in zip(FILTERS, maggies, ab):
    print(f"  {f:12s}  {m:12.4e} maggies   AB = {a:6.2f}")
print(f"\nmin={maggies.min():.3e}  max={maggies.max():.3e}")

ok = 1e-9 < maggies.max() < 1e-5          # bright band ~1e-7, AB ~ 17-18
print("\nPASS: bright bands at ~1e-7 maggies (AB ~ 17-18)" if ok else
      "FAIL: normalization still off — paste this output back")
raise SystemExit(0 if ok else 1)
