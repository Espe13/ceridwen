#!/usr/bin/env python3
"""AGN_PLAN section 4 D: end-to-end injection-recovery of the corrected-mode AGN on CPU.

Mock: JADES-like NIRCam + MIRI photometry at z = 3 from the test SSP grid, the corrected AGN
(torus + disc; fagn = 0.3, agn_tau = 20, fagn_bbb = 1), noiseless fluxes with 5 % uncertainties.
Fit: nested sampling, num_live 100, every stellar and AGN parameter free (log_fagn,
log_fagn_bbb uniform; agn_tau LogUniform(5, 150)), once with slice_kernel="lanes" and once with
"stock" at the same rng key.  Pass: every truth inside its 95 % interval, and the two runs
bitwise equal (CPU).

    SPS_HOME=... CERIDWEN_TEST_SSP=... PYTHONPATH=. python scripts/agn_injection_recovery.py

``--spec grating|prism`` (session SIGMA): the AGN line width ``agn_eline_sigma`` sampled.  Same
galaxy and photometry plus a Spectrum (rest 3600-6900 A at z = 3; R_FWHM 1000, sigma_inst
~ 127 km/s, or R_FWHM 100, ~ 1273 km/s; 2 pixels per FWHM; noiseless, continuum S/N 20 per
pixel), the CSP with nebular emission and the corrected AGN with its lines; truths
agn_elum = 3e-3, agn_eline_sigma = 250 km/s, sigma_gas = 60 km/s, all three free
(Uniform(1e-6, 1e-2), Uniform(50, 500), Uniform(10, 300)) next to the parameters above
(gas_logu, gas_logz fixed at the truth).  One nested-sampling run, slice_kernel="lanes",
num_live 100 (``--num-live``).  Prints posterior medians, 16/84 and 2.5/97.5 % and, for agn_eline_sigma, the
posterior / prior standard deviation; writes the weighted samples to
``<outdir>/sigma_<spec>[_nl<num_live>].npz`` and the summary to the same name ``.json``.  Pass for
``grating``: every truth inside its 2.5-97.5 % interval.

    ... python scripts/agn_injection_recovery.py --spec grating --outdir _scratch/sigma
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time
import warnings

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from _gridfixture import require_test_grid  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData, SedModel, fitSED  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.model import logsfr_ratios_to_sfh  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics  # noqa: E402
from ceridwen.observation import Photometry, Spectrum  # noqa: E402
from ceridwen.priors import ClippedNormal, LogUniform, StudentT, Uniform  # noqa: E402

ZRED = 3.0
N_TIME = 5
FILTERS = ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w", "jwst_f356w",
           "jwst_f410m", "jwst_f444w", "jwst_f770w", "jwst_f1000w", "jwst_f1800w"]
TRUTH = {"logsfr_ratios": jnp.array([0.2, 0.0, -0.2, -0.4]), "logzsol": jnp.array([-0.5]),
         "logmass": jnp.array([10.0]), "diffuse_tau_kc": jnp.array([0.4]),
         "log_fagn": jnp.array([np.log10(0.3)]), "agn_tau": jnp.array([20.0]),
         "log_fagn_bbb": jnp.array([0.0])}


def build(csp, obs):
    t = np.array(csp.sfh_times)
    priors = {"logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.5),
              "logzsol": Uniform(low=-2.0, high=0.2), "logmass": Uniform(low=8.5, high=11.5),
              "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=3.0),
              "log_fagn": Uniform(low=-5.0, high=float(np.log10(3.0))),
              "agn_tau": LogUniform(mini=5.0, maxi=150.0),
              "log_fagn_bbb": Uniform(low=-5.0, high=2.0)}
    transforms = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                                              sfh_times_yr=_t),
                  "fagn": lambda th: 10.0 ** th["log_fagn"],
                  "fagn_bbb": lambda th: 10.0 ** th["log_fagn_bbb"],
                  "diffuse_dust_index": lambda th: jnp.array([0.0])}       # fixed slope
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logzsol": jnp.array([-0.8]),
            "logmass": jnp.array([10.2]), "diffuse_tau_kc": jnp.array([0.3]),
            "log_fagn": jnp.array([-2.0]), "agn_tau": jnp.array([30.0]),
            "log_fagn_bbb": jnp.array([-1.0])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, [obs], priors=priors, transforms=transforms,
                        free_param_init=init, zred=ZRED)


def main():
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(require_test_grid())),
                   lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)) - 0.05, N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=False, add_igm=True, verbose=False, cosmo=cosmo,
                   agn=AGNConfig(agn_mode="corrected", lines=False))
    gen = build(csp, Photometry(filters=FILTERS, name="phot"))
    flux = np.asarray(gen.predict(TRUTH)["phot"], np.float64)
    obs = Photometry(filters=FILTERS, flux=flux, uncertainty=0.05 * flux, name="phot")
    model = build(csp, obs)
    gal = np.asarray(gen.predict({**TRUTH, "log_fagn": jnp.array([-30.0]),
                                  "log_fagn_bbb": jnp.array([-30.0])})["phot"])
    print("AGN fraction per band:", np.round(1 - gal / flux, 3).tolist(), flush=True)
    res = {}
    for kernel in ("lanes", "stock"):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            r = fitSED(model, sampler="nested",
                       sampler_kwargs=dict(num_live=100, num_delete=20, slice_kernel=kernel),
                       rng_key=jax.random.PRNGKey(0), output_dir=d, verbose=False)
        res[kernel] = r
        print(f"{kernel}: {time.perf_counter() - t0:.1f} s, lnZ {r.log_evidence:.4f} "
              f"+/- {r.log_evidence_err:.4f}, calls {r.n_likelihood_calls}", flush=True)
    a, b = res["lanes"], res["stock"]
    same = all(np.asarray(a.samples[k]).tobytes() == np.asarray(b.samples[k]).tobytes()
               for k in a.samples)
    same &= np.asarray(a.log_likelihoods).tobytes() == np.asarray(b.log_likelihoods).tobytes()
    same &= np.asarray(a.log_weights).tobytes() == np.asarray(b.log_weights).tobytes()
    print(f"lanes == stock bitwise (samples, logL, log_weights): {same}")
    lw = np.asarray(a.log_weights, np.float64)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    ok = True
    report = {}
    for k in ("log_fagn", "agn_tau", "log_fagn_bbb", "logmass", "logzsol", "diffuse_tau_kc"):
        s = np.asarray(a.samples[k], np.float64).reshape(len(w), -1)[:, 0]
        o = np.argsort(s)
        c = np.cumsum(w[o])
        q = [float(np.interp(x, c, s[o])) for x in (0.025, 0.5, 0.975)]
        tru = float(np.ravel(TRUTH[k])[0])
        inside = q[0] <= tru <= q[2]
        ok &= inside
        report[k] = {"truth": tru, "q025_50_975": q, "inside_95": inside}
        print(f"{k:>15}: truth {tru:+.3f}  2.5/50/97.5 % {q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}  "
              f"{'ok' if inside else 'OUTSIDE'}")
    print(json.dumps({"bitwise_lanes_stock": bool(same), "all_inside_95": bool(ok)}))
    return 0 if (same and ok) else 1


# ---------------------------------------------------------------- --spec (session SIGMA)
SIGMA_TRUTH = {"agn_elum": jnp.array([3e-3]), "agn_eline_sigma": jnp.array([250.0]),
               "sigma_gas": jnp.array([60.0])}
SIGMA_PRIORS = {"agn_elum": Uniform(low=1e-6, high=1e-2),
                "agn_eline_sigma": Uniform(low=50.0, high=500.0),
                "sigma_gas": Uniform(low=10.0, high=300.0)}
SIGMA_INIT = {"agn_elum": jnp.array([1e-3]), "agn_eline_sigma": jnp.array([150.0]),
              "sigma_gas": jnp.array([100.0])}
GAS = {"gas_logu": -2.5, "gas_logz": 0.0}
R_FWHM = {"grating": 1000.0, "prism": 100.0}
SIGMA_KEYS = ("agn_elum", "agn_eline_sigma", "sigma_gas", "log_fagn", "agn_tau",
              "log_fagn_bbb", "logmass", "logzsol", "diffuse_tau_kc")


def build_sigma(csp, obs):
    """``build`` plus the line parameters: agn_elum, agn_eline_sigma, sigma_gas free."""
    t = np.array(csp.sfh_times)
    m0 = build(csp, obs[0])
    priors = {**m0.priors, **SIGMA_PRIORS}
    transforms = {**m0.transforms, **{k: (lambda th, v=v: jnp.array([v])) for k, v in GAS.items()}}
    init = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logzsol": jnp.array([-0.8]),
            "logmass": jnp.array([10.2]), "diffuse_tau_kc": jnp.array([0.3]),
            "log_fagn": jnp.array([-2.0]), "agn_tau": jnp.array([30.0]),
            "log_fagn_bbb": jnp.array([-1.0]), **SIGMA_INIT}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = SedModel(csp, list(obs), priors=priors, transforms=transforms, free_param_init=init,
                     zred=ZRED, kinematics=Kinematics(sigma_gal=150.0, sigma_gas="sigma_gas"))
    for x in w:
        if "agn_eline_sigma" in str(x.message):
            print("setup warning:", x.message, flush=True)
    return m


def _wq(s, w, qs):
    o = np.argsort(s)
    c = np.cumsum(w[o])
    return [float(np.interp(x, c, s[o])) for x in qs]


def main_sigma(spec, outdir, num_live=100):
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(require_test_grid())),
                   lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)) - 0.05, N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=True, verbose=False, cosmo=cosmo,
                   sps_home=os.environ["SPS_HOME"], agn=AGNConfig(agn_mode="corrected"))
    R = R_FWHM[spec]
    wave = np.exp(np.arange(np.log(3600.0 * (1 + ZRED)), np.log(6900.0 * (1 + ZRED)),
                            1.0 / (2.0 * R)))
    truth = {**TRUTH, **SIGMA_TRUTH}
    gen = build_sigma(csp, [Photometry(filters=FILTERS, name="phot"),
                            Spectrum(wavelength=wave, instrument=Instrument.R_fwhm(R),
                                     name="spec")])
    pred = gen.predict(truth)
    flux = np.asarray(pred["phot"], np.float64)
    sp = np.asarray(pred["spec"], np.float64)
    off = gen.predict({**truth, "agn_elum": jnp.array([0.0])})
    print(f"{spec}: R_FWHM {R:g}, {wave.size} px, AGN lines peak / continuum "
          f"{np.max(sp - np.asarray(off['spec'])) / np.median(sp):.2f}", flush=True)
    obs = [Photometry(filters=FILTERS, flux=flux, uncertainty=0.05 * flux, name="phot"),
           Spectrum(wavelength=wave, flux=sp, uncertainty=np.full_like(sp, np.median(sp) / 20.0),
                    instrument=Instrument.R_fwhm(R), name="spec")]
    model = build_sigma(csp, obs)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        r = fitSED(model, sampler="nested",
                   sampler_kwargs=dict(num_live=num_live, num_delete=20, slice_kernel="lanes"),
                   rng_key=jax.random.PRNGKey(0), output_dir=d, verbose=False)
    print(f"lanes, num_live {num_live}: {time.perf_counter() - t0:.1f} s, lnZ {r.log_evidence:.4f} "
          f"+/- {r.log_evidence_err:.4f}, calls {r.n_likelihood_calls}", flush=True)
    lw = np.asarray(r.log_weights, np.float64)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    ok, report = True, {}
    for k in SIGMA_KEYS:
        s = np.asarray(r.samples[k], np.float64).reshape(len(w), -1)[:, 0]
        q = _wq(s, w, (0.025, 0.16, 0.5, 0.84, 0.975))
        tru = float(np.ravel(truth[k])[0])
        inside = q[0] <= tru <= q[4]
        ok &= inside
        mean = float(np.sum(w * s))
        report[k] = {"truth": tru, "q025_16_50_84_975": q, "inside_2sigma": bool(inside),
                     "post_std": float(np.sqrt(np.sum(w * (s - mean) ** 2)))}
        print(f"{k:>15}: truth {tru:+.4g}  median {q[2]:+.4g}  16/84 {q[1]:+.4g} {q[3]:+.4g}  "
              f"2.5/97.5 {q[0]:+.4g} {q[4]:+.4g}  {'ok' if inside else 'OUTSIDE'}")
    lo, hi = SIGMA_PRIORS["agn_eline_sigma"].bounds
    prior_std = (float(hi) - float(lo)) / np.sqrt(12.0)
    ratio = report["agn_eline_sigma"]["post_std"] / prior_std
    report["agn_eline_sigma"]["post_over_prior_std"] = ratio
    print(f"agn_eline_sigma posterior / prior std: {ratio:.3f}")
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"sigma_{spec}" + ("" if num_live == 100 else f"_nl{num_live}")
    np.savez(out / f"{tag}.npz", log_weights=lw, log_likelihoods=np.asarray(r.log_likelihoods),
             **{k: np.asarray(v) for k, v in r.samples.items()},
             truth_keys=np.array(list(truth)),
             truth_values=np.array([float(np.ravel(v)[0]) for v in truth.values()]))
    summary = {"spec": spec, "num_live": num_live, "R_fwhm": R, "n_pix": int(wave.size), "lnZ": float(r.log_evidence),
               "lnZ_err": float(r.log_evidence_err), "calls": int(r.n_likelihood_calls),
               "all_inside_2sigma": bool(ok), "params": report}
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"spec": spec, "all_inside_2sigma": bool(ok), "sigma_post_over_prior_std": ratio}))
    return 0 if (ok or spec == "prism") else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=("grating", "prism"), default=None)
    ap.add_argument("--outdir", default="_scratch/sigma_recovery")
    ap.add_argument("--num-live", type=int, default=100)
    a = ap.parse_args()
    sys.exit(main() if a.spec is None else main_sigma(a.spec, a.outdir, a.num_live))
