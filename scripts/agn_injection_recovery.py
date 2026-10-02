#!/usr/bin/env python3
"""AGN_PLAN section 4 D: end-to-end injection-recovery of the corrected-mode AGN on CPU.

Mock: JADES-like NIRCam + MIRI photometry at z = 3 from the test SSP grid, the corrected AGN
(torus + disc; fagn = 0.3, agn_tau = 20, fagn_bbb = 1), noiseless fluxes with 5 % uncertainties.
Fit: nested sampling, num_live 100, every stellar and AGN parameter free (log_fagn,
log_fagn_bbb uniform; agn_tau LogUniform(5, 150)), once with slice_kernel="lanes" and once with
"stock" at the same rng key.  Pass: every truth inside its 95 % interval, and the two runs
bitwise equal (CPU).

    SPS_HOME=... CERIDWEN_TEST_SSP=... PYTHONPATH=. python scripts/agn_injection_recovery.py
"""
from __future__ import annotations

import json
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
from ceridwen.observation import Photometry  # noqa: E402
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


if __name__ == "__main__":
    sys.exit(main())
