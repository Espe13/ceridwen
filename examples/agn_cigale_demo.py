#!/usr/bin/env python3
"""
agn_cigale_demo.py — the CIGALE-standard AGN (SKIRTOR + polar dust, Feltre+16 NLR, BLR).

``AGNConfig`` switches (all off by default; ``docs/agn.md``, "AGN2"):

    torus_model="skirtor16"   SKIRTOR 2016 clumpy torus + its disc (Stalevski+12, +16), as CIGALE's
                              skirtor2016 module (Schartmann disc, delta 0); theta skirtor_t, _pl,
                              _q, _oa, _i (inclination, 0 = face-on), amplitude fagn
    polar_dust=True           CIGALE polar dust: extinction of the disc for a type-1 view and a
                              grey body re-emitting the angle-averaged absorption;
                              polar_ebv, polar_temp, polar_emis; polar_law "smc" / "calzetti" /
                              "gaskell"
    nlr_model="feltre16"      Feltre+16 narrow-line grid (20 lines), interpolated in nlr_logu,
                              nlr_logz (log Z/Z_sun, Z_sun 0.01524), nlr_alpha, nlr_xid,
                              nlr_lognh; amplitude agn_elum = L_Hbeta,NLR per formed M_sun,
                              width agn_eline_sigma (sigma, km/s)
    blr=True                  broad H I lines (Case B), blr_lum = L_Halpha,BLR per formed M_sun,
                              blr_sigma = Gaussian SIGMA in km/s (FWHM = 2.3548 sigma)

Two modes: ``agn_mode="cigale"`` reproduces CIGALE (fagn IS CIGALE's fracAGN = P / (P + L_dust,gal),
the AGN after the galaxy's dust, before the IGM); ``agn_mode="corrected"`` (default) puts the same
components in CERIDWEN's framework (fagn = P / L_bol,*, the galaxy's diffuse dust on the AGN,
energy balance into DL07). Both report CIGALE's fracAGN as ``frac_agn_cigale`` in PostProcess.

The SKIRTOR and NLR tables are not shipped (SKIRTOR 227 MB): build them once with the converters
of the source repository (``scripts/agn2/build_skirtor_h5.py``, ``build_nlr_h5.py``) into
``$CERIDWEN_AGN_DATA`` (default ``~/.ceridwen/agn``).

    export SPS_HOME=/path/to/fsps CERIDWEN_AGN_DATA=/path/to/agn_tables
    python examples/agn_cigale_demo.py            # per-band AGN fraction, both modes
    python examples/agn_cigale_demo.py --fit      # plus a nested-sampling fit (corrected)
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
from ceridwen.priors import ClippedNormal, StudentT, Uniform

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"
SEED = 11
ZRED = 2.5
N_TIME = 6
SNR = 10.0
FILTERS = ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w",
           "jwst_f356w", "jwst_f444w", "jwst_f770w", "jwst_f1000w", "jwst_f1280w",
           "jwst_f1500w", "jwst_f1800w", "jwst_f2100w"]

TRUTH = {
    "logsfr_ratios":  jnp.array([+0.2, +0.1, 0.0, -0.3, -0.5]),
    "logzsol":        jnp.array([-0.3]),
    "logmass":        jnp.array([10.4]),
    "diffuse_tau_kc": jnp.array([0.5]),
    "fagn":           jnp.array([0.4]),        # corrected: P / L_bol,*; cigale: fracAGN (< 1)
    "skirtor_i":      jnp.array([70.0]),       # type 2 for oa = 40 (type 1 needs i <= 90 - oa)
    "polar_ebv":      jnp.array([0.2]),
    "agn_elum":       jnp.array([1e-3]),       # L_Hbeta,NLR per formed M_sun
    "nlr_logu":       jnp.array([-2.5]),
    "blr_lum":        jnp.array([3e-3]),       # L_Halpha,BLR per formed M_sun
    "blr_sigma":      jnp.array([1500.0]),     # sigma [km/s]
}
AMPLITUDES = ("fagn", "agn_elum", "blr_lum")


def agn_config(mode):
    return AGNConfig(agn_mode=mode, torus_model="skirtor16", polar_dust=True,
                     nlr_model="feltre16", blr=True)


def build(ssp, agn, observations, free=()):
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 2.4, N_TIME), zh_const=True,
                   sfh_interp="step", add_dust=True, add_diffuse_dust=True, add_neb=True,
                   add_dust_emission=True, add_igm=True, verbose=False,
                   cosmo=Cosmology.planck18(), agn=agn)
    t = np.array(csp.sfh_times)
    priors = {"logzsol": Uniform(low=-2.0, high=0.2), "logmass": Uniform(low=9.0, high=12.0),
              "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
              "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0)}
    transforms = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                                              sfh_times_yr=_t),
                  "tau_pow": lambda th: th["diffuse_tau_kc"]}
    agn_priors = {"fagn": Uniform(low=0.0, high=2.0), "skirtor_i": Uniform(low=0.0, high=90.0),
                  "polar_ebv": Uniform(low=0.0, high=1.0)}
    for k, v in TRUTH.items():                   # the AGN keys: sampled (free) or fixed
        if k in priors or k in transforms:
            continue
        if k in free:
            priors[k] = agn_priors[k]
        elif k not in ("logsfr_ratios", "logzsol", "logmass", "diffuse_tau_kc"):
            transforms[k] = lambda th, _v=v: _v
    for k, v in csp.theta_init.items():          # every other CSP key: fixed at its default
        if k not in priors and k not in transforms and k not in ("sfh", "zred", "logzsol_hist"):
            transforms[k] = lambda th, _v=jnp.atleast_1d(jnp.asarray(v)): _v
    return SedModel(csp, observations=observations, priors=priors, transforms=transforms,
                    free_param_init={k: v for k, v in TRUTH.items() if k in priors}, zred=ZRED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="also run a nested-sampling fit")
    ap.add_argument("--ssp", default=str(SSP_FILE), help="SSP grid (built with FSPS if absent)")
    args = ap.parse_args()
    if not os.environ.get("SPS_HOME"):
        sys.exit("CLOUDY grids and DL07 are read from $SPS_HOME, which is unset.")
    ssp = (SSPData.load(args.ssp) if pathlib.Path(args.ssp).is_file()
           else SSPData.from_fsps(imf_type=1, save_to=args.ssp))

    # ── per-band AGN fraction in both modes ──────────────────────────────
    for mode in ("cigale", "corrected"):
        m = build(ssp, agn_config(mode), [Photometry(filters=FILTERS, name="phot")])
        th = dict(m.theta_init)
        tot = np.asarray(m.predict(th)["phot"])
        gal = np.asarray(build_off(ssp, mode).predict(th)["phot"])
        frac = " ".join(f"{f:.2f}" for f in 1.0 - gal / tot)
        print(f"{mode:>10}: AGN fraction per band (F090W .. F2100W): {frac}")

    if not args.fit:
        return
    # ── mock and fit (corrected mode; fagn, skirtor_i, polar_ebv free) ────
    free = ("fagn", "skirtor_i", "polar_ebv")
    rng = np.random.default_rng(SEED)
    gen = build(ssp, agn_config("corrected"), [Photometry(filters=FILTERS, name="phot")], free)
    flux = np.asarray(gen.predict(dict(gen.theta_init))["phot"])
    unc = flux / SNR
    phot = Photometry(filters=FILTERS, flux=flux + unc * rng.standard_normal(flux.shape),
                      uncertainty=unc, name="phot")
    model = build(ssp, agn_config("corrected"), [phot], free)
    result = fitSED(model, sampler="ns",
                    sampler_kwargs={"num_live": 200, "slice_kernel": "lanes"},
                    rng_key=jax.random.PRNGKey(SEED), output_dir="./agn_cigale_demo_output")
    w = np.exp(np.asarray(result.log_weights) - np.max(result.log_weights))
    for p in free + ("logmass",):
        s = np.asarray(result.samples[p]).ravel()
        print(f"{p:>10}: true {float(TRUTH[p][0]):+8.3f}   posterior mean "
              f"{np.sum(w * s) / np.sum(w):+8.3f}")


def build_off(ssp, mode):
    """The same model with every AGN amplitude at 0 (the galaxy alone)."""
    off = build(ssp, agn_config(mode), [Photometry(filters=FILTERS, name="phot")])
    for k in AMPLITUDES:
        off.transforms[k] = lambda th: jnp.array([0.0])
    return off


if __name__ == "__main__":
    main()
