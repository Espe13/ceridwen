#!/usr/bin/env python3
"""
demo_1_mock_test.py — how to set up a mock (injection-recovery) test.

The single most useful validation of any SED-fitting setup: push KNOWN truth
parameters through the forward model, add noise, fit, and check the truth
comes back. Because mock data and fit share the same forward model, any
systematic mismatch you see is a bug in your configuration, not in nature.

The recipe:
    1. one model builder used for BOTH mock generation and fitting,
    2. a fixed random seed, so the test is reproducible,
    3. a recovered-vs-true table with pulls ((fit - true)/sigma) at the end.

Needs an SSP grid (built via FSPS on first run) and ~10 min on a laptop CPU.

    conda activate <your-env>
    python examples/demo_1_mock_test.py
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"

# ---------------------------------------------------------------------------
# Configuration: everything that defines the mock lives at the top, so the
# test is one glance to audit and one edit to vary.
# ---------------------------------------------------------------------------
SEED = 42
ZRED = 0.1                              # fixed spectroscopic redshift
N_TIME = 6                              # SFH nodes -> 5 free logsfr_ratios
SNR = 20.0
FILTERS = ["galex_FUV", "galex_NUV",
           "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
           "twomass_J", "twomass_H", "twomass_Ks",
           "wise_w1", "wise_w2"]

TRUTH = {
    "logsfr_ratios":      jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),   # log10 ABSOLUTE Z (ssp_lgmet)
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── SSP grid: load the cache, or build it once with FSPS ─────────────
    if SSP_FILE.is_file():
        print(f"[grid] loading cached SSP grid: {SSP_FILE}")
        ssp = SSPData.load(str(SSP_FILE))
    else:
        print("[grid] no cache found, building with FSPS (a few minutes) ...")
        ssp = SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE))

    csp = CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, N_TIME),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=False,
        verbose=False,
    )
    sfh_times_yr = np.array(csp.sfh_times)

    # ── One builder for generator AND fit model: the heart of the test ───
    def build_model(observations):
        return SedModel(
            csp, observations=observations,
            priors={
                "Z": Uniform(low=-3.9, high=-1.45),
                "logmass": Uniform(low=9.0, high=12.0),
                "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0,
                                                low=0.0, high=4.0),
                "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
                "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
            },
            transforms={"sfh": lambda th, _t=sfh_times_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                             sfh_times_yr=_t)},
            free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                             "logmass": jnp.array([10.0])},
            zred=ZRED,
        )

    # ── Generate the mock: empty observations are enough to predict ──────
    gen = build_model([Photometry(filters=FILTERS, name="phot")])
    mag = np.asarray(gen.predict(TRUTH)["phot"])       # AB maggies
    print(f"[mock] photometry: {mag.min():.3e} .. {mag.max():.3e} maggies "
          f"(z=0.1, logmass=10.5 -> bright bands ~1e-7, AB ~ 17-18)")
    mag_unc = mag / SNR
    mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)

    # ── Fit the mock with the SAME builder ────────────────────────────────
    phot = Photometry(filters=FILTERS, flux=mag_obs, uncertainty=mag_unc,
                      name="phot")
    model = build_model([phot])

    result = fitSED(
        model,
        sampler="ns",
        sampler_kwargs={"num_live": 300, "num_delete": 100},
        rng_key=jax.random.PRNGKey(SEED),
        output_dir="./demo_1_output",
    )

    # ── Recovered vs true, with pulls ─────────────────────────────────────
    # Nested samples carry importance weights: resample to equal weight
    # before ANY summary statistic (unweighted medians drift to the prior).
    lw = np.asarray(result.log_weights)
    w = np.exp(lw - lw.max()); w /= w.sum()
    idx = rng.choice(w.size, size=2000, p=w)

    print("\nparameter             true      fit               pull")
    n_bad = 0
    for p in ("Z", "logmass", "diffuse_tau_kc", "diffuse_dust_index"):
        s = np.asarray(result.samples[p])[idx].ravel()
        med, sig = np.median(s), np.std(s)
        pull = (med - float(TRUTH[p][0])) / sig
        n_bad += abs(pull) > 3.0
        print(f"{p:>20}: {float(TRUTH[p][0]):+7.3f}   "
              f"{med:+7.3f} +/- {sig:5.3f}   {pull:+5.2f}")

    # A healthy mock test has |pull| < 3 for every parameter. If a pull is
    # large AND the corner looks tight, suspect a forward-model asymmetry
    # between generation and fitting (units, zred, SSP grid mismatch).
    print("\nPASS: all pulls < 3 sigma" if n_bad == 0 else
          f"FAIL: {n_bad} parameter(s) with |pull| > 3 -- investigate!")


if __name__ == "__main__":
    main()
