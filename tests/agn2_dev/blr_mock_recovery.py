#!/usr/bin/env python3
"""PLAN_AGN2 1.3: BLR mock injection-recovery on CPU (session BLR; not collected by pytest).

Mock: z = 4.5, the test SSP grid (constant SFH, logzsol = gas_logz = -0.5, gas_logu = -2.5, nebular on), the corrected-mode
BLR only (blr_lum = 0.1 L_sun per M_formed, i.e. L_Halpha,broad ~ 3e42 erg/s at logmass 9.5),
broad H alpha sigma in {500, 1500, 3000} km/s, in a prism-like (R_FWHM = 150) and a
grating-like (R_FWHM = 1000) Spectrum covering rest 4500-7000 A. Noiseless fluxes; the
uncertainty per pixel is 1/20 of the median continuum.
Fit: nested sampling (slice_kernel="lanes", num_live 100, PRNGKey(0)); free logmass,
diffuse_tau_kc, log_blr_lum, blr_sigma LogUniform(400, 5000). Pass: the truth of blr_sigma and
log_blr_lum inside the central 68 % (1 sigma) interval, as PLAN_AGN2 1.3 asks.

    SPS_HOME=... CERIDWEN_TEST_SSP=... PYTHONPATH=. python tests/agn2_dev/blr_mock_recovery.py OUT.json
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
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import require_test_grid  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData, SedModel, fitSED  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics  # noqa: E402
from ceridwen.observation import Spectrum  # noqa: E402
from ceridwen.priors import ClippedNormal, LogUniform, Uniform  # noqa: E402

ZRED = 4.5
N_TIME = 4
SIGMAS = (500.0, 1500.0, 3000.0)
RES = {"prism": 150.0, "grating": 1000.0}
TRUTH = {"logmass": 9.5, "diffuse_tau_kc": 0.3, "log_blr_lum": -1.0}


def wave_obs(R):
    lo, hi = 4500.0 * (1 + ZRED), 7000.0 * (1 + ZRED)
    return np.exp(np.arange(np.log(lo), np.log(hi), 1.0 / (2.3548 * R) / 2.0))


def build(csp, spec):
    priors = {"logmass": Uniform(low=8.0, high=11.0),
              "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=3.0),
              "log_blr_lum": Uniform(low=-4.0, high=1.0),
              "blr_sigma": LogUniform(mini=400.0, maxi=5000.0)}
    transforms = {"sfh": lambda th: jnp.ones(N_TIME),
                  "blr_lum": lambda th: 10.0 ** th["log_blr_lum"],
                  "logzsol": lambda th: jnp.array([-0.5]),
                  "gas_logz": lambda th: jnp.array([-0.5]),
                  "gas_logu": lambda th: jnp.array([-2.5]),
                  "diffuse_dust_index": lambda th: jnp.array([0.0])}
    init = {"logmass": jnp.array([9.8]), "diffuse_tau_kc": jnp.array([0.5]),
            "log_blr_lum": jnp.array([-1.5]), "blr_sigma": jnp.array([1200.0])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, [spec], priors=priors, transforms=transforms,
                        free_param_init=init, zred=ZRED,
                        kinematics=Kinematics(sigma_gal=100.0, sigma_gas=60.0))


def quantiles(r, k, qs):
    lw = np.asarray(r.log_weights, np.float64)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    s = np.asarray(r.samples[k], np.float64).reshape(len(w), -1)[:, 0]
    o = np.argsort(s)
    c = np.cumsum(w[o])
    return [float(np.interp(x, c, s[o])) for x in qs]


def main(out_path):
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(require_test_grid())),
                   lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)) - 0.05, N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=False, verbose=False, cosmo=cosmo,
                   agn=AGNConfig(agn_mode="corrected", torus=False, disk=False, lines=False,
                                 blr=True))
    report = {}
    ok_all = True
    for name, R in RES.items():
        wo = wave_obs(R)
        inst = Instrument.R_fwhm(R)
        gen = build(csp, Spectrum(wavelength=wo, instrument=inst, name="spec"))
        for sig in SIGMAS:
            truth = {k: jnp.asarray(v) for k, v in gen.theta_init.items()}
            truth.update({k: jnp.array([v]) for k, v in TRUTH.items()})
            truth["blr_sigma"] = jnp.array([sig])
            flux = np.asarray(gen.predict(truth)["spec"], np.float64)
            cont = np.asarray(gen.predict({**truth, "log_blr_lum": jnp.array([-30.0])})["spec"],
                              np.float64)
            unc = np.full_like(flux, np.median(cont) / 20.0)
            spec = Spectrum(wavelength=wo, flux=flux, uncertainty=unc, instrument=inst,
                            name="spec")
            model = build(csp, spec)
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory() as d:
                r = fitSED(model, sampler="nested",
                           sampler_kwargs=dict(num_live=100, num_delete=20, slice_kernel="lanes"),
                           rng_key=jax.random.PRNGKey(0), output_dir=d, verbose=False)
            dt = time.perf_counter() - t0
            row = {"R_fwhm": R, "seconds": round(dt, 1), "lnZ": float(r.log_evidence),
                   "line_peak_over_unc": float((flux - cont).max() / unc[0])}
            for k, tru in (("blr_sigma", sig), ("log_blr_lum", TRUTH["log_blr_lum"]),
                           ("logmass", TRUTH["logmass"]),
                           ("diffuse_tau_kc", TRUTH["diffuse_tau_kc"])):
                q = quantiles(r, k, (0.16, 0.5, 0.84))
                inside = q[0] <= tru <= q[2]
                row[k] = {"truth": tru, "q16_50_84": q, "inside_68": bool(inside)}
                if k in ("blr_sigma", "log_blr_lum"):
                    ok_all &= inside
            report[f"{name}_sigma{int(sig)}"] = row
            print(f"{name:7s} sigma {sig:6.0f}: blr_sigma {row['blr_sigma']['q16_50_84']} "
                  f"log_blr_lum {row['log_blr_lum']['q16_50_84']} ({dt:.0f} s)", flush=True)
    report["all_blr_inside_68"] = bool(ok_all)
    pathlib.Path(out_path).write_text(json.dumps(report, indent=1))
    print(json.dumps({"all_blr_inside_68": bool(ok_all)}))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "blr_mock_recovery.json"))
