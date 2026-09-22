"""Emission-line marginalisation on a NIRSpec-like spectrum (docs/eline_marginalisation.md).

A z = 2.5 galaxy with a young-ish population (strong Balmer absorption) whose
emission lines do NOT follow the CLOUDY prediction the fit uses: [O III] is 3x,
[O II] 0.5x, the Balmer lines 1.5x and [Ne III] 2x the grid. The spectrum
(R_fwhm = 1000, rest 3650-6800 A, S/N ~ 20 per pixel) and NIRCam photometry are
fitted with the nebular model on and the line fluxes marginalised
(``Spectrum(marginalize_elines=True)``); the recovered line fluxes are compared
with the truth.

    conda activate <your-env>       # needs FSPS data files at $SPS_HOME
    python examples/demo_eline_marginalisation.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp

from ceridwen import SSPData, CSPBasis, SedModel, fitSED, read_result_h5, Kinematics, Instrument
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT
from ceridwen.cosmology import Cosmology

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"
OUT = HERE / "demo_eline_output"

SEED = 7
ZRED = 2.5
N_TIME = 6
R_FWHM = 1000.0
SNR = 20.0
FILTERS = ["jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w", "jwst_f356w", "jwst_f444w"]
GAS = {"gas_logu": -2.5, "gas_logz": -0.3}           # the nebular model is fixed in the fit
BOOST = {"[O III] 4959": 3.0, "[O III] 5007": 3.0, "[O II] 3726": 0.5, "[O II] 3729": 0.5,
         "[Ne III] 3869": 2.0, "Ba-alpha 6563": 1.5, "Ba-beta 4861": 1.5,
         "Ba-gamma 4341": 1.5, "Ba-delta 4101.76A": 1.5}
TRUTH = {
    "logsfr_ratios":      jnp.array([-0.6, -0.3, 0.2, 0.3, 0.3]),   # rising then quenching slowly
    "logzsol":            jnp.array([-0.2]),
    "logmass":            jnp.array([10.3]),
    "diffuse_tau_kc":     jnp.array([0.4]),
    "diffuse_dust_index": jnp.array([-0.3]),
}


def main() -> None:
    if not os.environ.get("SPS_HOME"):
        sys.exit("add_neb=True reads the CLOUDY grids from $SPS_HOME, which is unset.")
    rng = np.random.default_rng(SEED)
    ssp = (SSPData.load(str(SSP_FILE)) if SSP_FILE.is_file()
           else SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE)))
    cosmo = Cosmology.planck18()
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=True, verbose=False, cosmo=cosmo)
    t = np.array(csp.sfh_times)
    wave = np.exp(np.arange(np.log(3650 * (1 + ZRED)), np.log(6800 * (1 + ZRED)),
                            1.0 / (2.3548 * R_FWHM) / 2.0))

    def build(spec_flux=None, spec_unc=None, phot_flux=None, phot_unc=None):
        spec = Spectrum(wavelength=wave, flux=spec_flux, uncertainty=spec_unc, name="spec",
                        instrument=Instrument.R_fwhm(R_FWHM), marginalize_elines=True)
        phot = Photometry(filters=FILTERS, flux=phot_flux, uncertainty=phot_unc, name="phot")
        return SedModel(
            csp, [spec, phot], zred=ZRED,
            priors={"logzsol": Uniform(low=-2.0, high=0.2),
                    "logmass": Uniform(low=9.0, high=11.5),
                    "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=3.0),
                    "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
                    "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3)},
            transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t),
                        "gas_logu": lambda th: jnp.array([GAS["gas_logu"]]),
                        "gas_logz": lambda th: jnp.array([GAS["gas_logz"]])},
            free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])},
            kinematics=Kinematics(sigma_gal=150.0, sigma_gas=80.0))

    # -- mock: nebular model on, line fluxes rescaled away from CLOUDY --------------------
    gen = build()
    pred, aux = gen.predict_with_elines(TRUTH)
    es = gen._eline_system
    boost = np.array([BOOST.get(n, 1.0) for n in es.names])
    f_true = np.asarray(aux["prior_mean"]) * boost
    spec_true = np.asarray(pred["spec"]) + np.asarray(aux["cols"]["spec"]) @ f_true
    phot_true = np.asarray(pred["phot"], float) + np.asarray(aux["cols"]["phot"]) @ f_true
    spec_unc = np.full(wave.size, np.median(spec_true) / SNR)
    phot_unc = 0.05 * phot_true
    model = build(spec_true + spec_unc * rng.standard_normal(wave.size), spec_unc,
                  phot_true + phot_unc * rng.standard_normal(phot_true.size), phot_unc)
    print(model.summary())

    result = fitSED(model, output_dir=OUT, rng_key=jax.random.PRNGKey(SEED))
    el = read_result_h5(OUT / "ceridwen_result.h5")["elines"]
    w = np.exp(np.asarray(result.log_weights) - np.max(result.log_weights))
    w /= w.sum()
    mean = w @ el["mean"]
    sd = np.sqrt(w @ (el["sd"] ** 2 + (el["mean"] - mean) ** 2))
    print(f"\n{'line':22s} {'true':>11s} {'fit':>11s} {'sd':>10s} {'pull':>6s}")
    for j, n in enumerate(el["names"]):
        if boost[j] != 1.0 or f_true[j] > 5 * sd[j]:
            print(f"{n:22s} {f_true[j]:11.3e} {mean[j]:11.3e} {sd[j]:10.2e} {(mean[j] - f_true[j]) / sd[j]:+6.2f}")


if __name__ == "__main__":
    main()
