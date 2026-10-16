#!/usr/bin/env python3
"""
agn_demo.py — an AGN in the forward model (``CSPBasis(agn=...)``, ``ceridwen.agn``).

Everything Prospector has for AGN, in three modes (``docs/agn.md``):

    AGNConfig()                          "corrected" (the default): Prospector's components with
                                         the defects fixed -- the torus carries exactly
                                         fagn x L_bol on any grid, agn_tau is clamped to the
                                         Nenkova grid, the AGN sits before the IGM like every
                                         other component, absorbed AGN light is re-emitted by
                                         DL07 when dust emission is on
    AGNConfig(agn_mode="fsps")           reproduces FSPS add_agn_dust (torus only)
    AGNConfig(agn_mode="prospector")     reproduces Prospector's AGNSSPBasis + AGNSpecModel

Components and their theta keys (defaults in brackets):
    Nenkova+08 clumpy torus   fagn [0] = L_torus / L_bol,*, agn_tau [10] (Nenkova grid 5-150)
    Temple+21 accretion disc  fagn_bbb [0] = F_nu,disc / F_nu,* at rest 5500 A
    AGN dust screen           dust4 [0], dust4_index [-1]  (AGNConfig(dust4_type=1))
    AGN narrow lines          agn_elum [0] = L_Hbeta [L_sun] per formed M_sun,
                              agn_eline_sigma [100 km/s]   (Richardson+14 'a42' ratios)

This demo makes a mock of a dusty star-forming galaxy with an obscured AGN at z = 2 in
JWST NIRCam + MIRI bands, prints how much of each band is AGN light in each mode, and
(with --fit) recovers fagn, agn_tau and fagn_bbb with nested sampling.  Sampling: fagn and
fagn_bbb span decades, so sample their logarithms (Prospector's priors are LogUniform(1e-5, 3)
and LogUniform(1e-5, 100)).

    export SPS_HOME=/path/to/fsps   # Nenkova templates ($SPS_HOME/dust), CLOUDY grids, DL07
    python examples/agn_demo.py           # the mock and the per-band AGN fraction
    python examples/agn_demo.py --fit     # plus a nested-sampling fit (3 min on an Apple GPU, up to 1 h on a CPU)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.agn import AGNConfig
from ceridwen.cosmology import Cosmology
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.observation import Photometry
from ceridwen.priors import ClippedNormal, LogUniform, StudentT, Uniform

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"
SEED = 7
ZRED = 2.0
N_TIME = 6
SNR = 15.0
FILTERS = ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w",
           "jwst_f356w", "jwst_f444w", "jwst_f770w", "jwst_f1000w", "jwst_f1500w",
           "jwst_f1800w", "jwst_f2100w"]

TRUTH = {
    "logsfr_ratios":      jnp.array([+0.2, +0.1, 0.0, -0.3, -0.5]),
    "logzsol":            jnp.array([-0.3]),
    "logmass":            jnp.array([10.6]),
    "diffuse_tau_kc":     jnp.array([0.8]),
    "diffuse_dust_index": jnp.array([-0.4]),
    "log_fagn":           jnp.array([np.log10(0.3)]),
    "agn_tau":            jnp.array([20.0]),
    "log_fagn_bbb":       jnp.array([np.log10(0.5)]),
    "agn_elum":           jnp.array([1e-4]),
}


def build(ssp, agn, observations):
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 3.0, N_TIME), zh_const=True,
                   sfh_interp="step", add_dust=True, add_diffuse_dust=True, add_neb=True,
                   add_dust_emission=True, add_igm=True, verbose=False,
                   cosmo=Cosmology.planck18(), agn=agn)
    t = np.array(csp.sfh_times)
    model_keys = set(csp.agn_param_names)              # the AGN keys this mode has
    priors = {
        "logzsol": Uniform(low=-2.0, high=0.2),
        "logmass": Uniform(low=9.0, high=12.0),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
        # the AGN: log-uniform amplitudes through transforms (docs/agn.md)
        "log_fagn": Uniform(low=-5.0, high=np.log10(3.0)),
        "agn_tau": LogUniform(mini=5.0, maxi=150.0),        # Prospector's prior
        "log_fagn_bbb": Uniform(low=-5.0, high=2.0),
        "agn_elum": Uniform(low=1e-6, high=1e-2),
    }
    transforms = {
        "sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t),
        "fagn": lambda th: 10.0 ** th["log_fagn"],
        "fagn_bbb": lambda th: 10.0 ** th["log_fagn_bbb"],
        "tau_pow": lambda th: th["diffuse_tau_kc"],          # birth clouds = diffuse
    }
    if "fagn_bbb" not in model_keys:                   # "fsps": torus only, no disc
        priors.pop("log_fagn_bbb"); transforms.pop("fagn_bbb")
    if "agn_elum" not in model_keys:                   # AGN lines off
        priors.pop("agn_elum")
    # the nebular, dust-emission and AGN-line-width keys are fixed at their defaults: nested
    # sampling needs a prior on every free parameter (docs/agn.md, "Sampling")
    for k in ("gas_logz", "gas_logu", "duste_qpah", "duste_umin", "duste_gamma", "alpha_pow",
              "agn_eline_sigma"):
        if k in csp.theta_init and k not in priors:
            transforms[k] = lambda th, _v=jnp.atleast_1d(csp.theta_init[k]): _v
    return SedModel(csp, observations=observations, priors=priors, transforms=transforms,
                    free_param_init={k: v for k, v in TRUTH.items() if k in priors},
                    zred=ZRED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="also run a nested-sampling fit")
    args = ap.parse_args()
    if not os.environ.get("SPS_HOME"):
        sys.exit("the AGN torus templates, CLOUDY grids and DL07 are read from $SPS_HOME, "
                 "which is unset. See the README, 'Installation'.")
    if SSP_FILE.is_file():
        ssp = SSPData.load(str(SSP_FILE))
    else:                                              # the published grid (Zenodo, cached)
        from ceridwen.ssps import fetch_grid
        ssp = SSPData.load(str(fetch_grid("mist_miles_chab")))

    # ── per-band AGN fraction in each mode ───────────────────────────────
    no_agn = {**TRUTH, "log_fagn": jnp.array([-30.0]), "log_fagn_bbb": jnp.array([-30.0]),
              "agn_elum": jnp.array([0.0])}
    for agn in (AGNConfig(), AGNConfig(agn_mode="fsps"), AGNConfig(agn_mode="prospector")):
        m = build(ssp, agn, [Photometry(filters=FILTERS, name="phot")])
        th = {**m.theta_init, **{k: v for k, v in TRUTH.items() if k in m.theta_init}}
        nth = {**th, **{k: v for k, v in no_agn.items() if k in m.theta_init}}
        tot = np.asarray(m.predict(th)["phot"])
        gal = np.asarray(m.predict(nth)["phot"])
        frac = " ".join(f"{f:.2f}" for f in 1.0 - gal / tot)
        print(f"{agn.agn_mode:>10}: AGN fraction per band (F090W .. F2100W): {frac}")

    if not args.fit:
        return
    # ── mock and fit (corrected mode) ─────────────────────────────────────
    rng = np.random.default_rng(SEED)
    gen = build(ssp, AGNConfig(), [Photometry(filters=FILTERS, name="phot")])
    flux = np.asarray(gen.predict({**gen.theta_init, **TRUTH})["phot"])
    unc = flux / SNR
    phot = Photometry(filters=FILTERS, flux=flux + unc * rng.standard_normal(flux.shape),
                      uncertainty=unc, name="phot")
    model = build(ssp, AGNConfig(), [phot])
    result = fitSED(model, sampler="ns", sampler_kwargs={"num_live": 200},
                    rng_key=jax.random.PRNGKey(SEED), output_dir="./agn_demo_output")
    for p in ("log_fagn", "agn_tau", "log_fagn_bbb", "logmass"):
        s = np.asarray(result.samples[p]).ravel()
        w = np.exp(np.asarray(result.log_weights) - np.max(result.log_weights))
        o = np.argsort(s)
        cdf = np.cumsum(w[o]) / np.sum(w)
        lo, med, hi = (s[o][np.searchsorted(cdf, q)] for q in (0.16, 0.5, 0.84))
        print(f"{p:>14}: true {float(TRUTH[p][0]):+7.3f}   posterior median {med:+7.3f}"
              f"  (16-84%: {lo:+7.3f} to {hi:+7.3f})")
    # 12 JWST bands constrain fagn_bbb and agn_tau weakly: expect wide ranges that contain the
    # truth, not a sharp recovery (fagn_bbb's range reaches the prior's lower edge)


if __name__ == "__main__":
    main()
