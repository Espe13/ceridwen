#!/usr/bin/env python3
"""
demo_2_photometry_lines.py — joint fit of broadband photometry + emission lines.

Emission-line fluxes constrain the ionized gas (metallicity, ionization
parameter) and recent star formation far better than broadband colors alone.
This demo shows the three ingredients specific to line fitting:

    1. ``add_neb=True`` on the CSPBasis: nebular continuum + lines from the
       CLOUDY grids shipped with FSPS (read from ``$SPS_HOME``, so FSPS's
       data files must be installed; the grid is matched automatically to
       the SSP isochrones recorded in your ssp_data.h5).
    2. a ``Lines`` observation: line identities are 1-BASED indices into
       ``$SPS_HOME/data/emlines_info.dat`` (``line_ind``); the names are just
       human-readable labels.
    3. the slit-loss nuisance parameter ``eline_scaling``: spectroscopic
       apertures lose flux, photometric apertures do not, so the lines seen
       by the Lines observation are scaled by this fraction while the
       photometry always sees the full line flux. Fit it, don't fix it.

Also demonstrated: an upper limit in the photometry (GALEX FUV treated as a
non-detection -> one-sided chi^2).

    conda activate <your-env>       # needs FSPS data files at $SPS_HOME
    python examples/demo_2_photometry_lines.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry, Lines
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"

SEED = 42
ZRED = 0.1
N_TIME = 6
SNR_PHOT, SNR_LINE = 20.0, 10.0
FILTERS = ["galex_FUV", "galex_NUV",
           "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
           "twomass_J", "twomass_H", "twomass_Ks"]

# Line identities: 1-based indices into $SPS_HOME/data/emlines_info.dat.
# The names are labels for you; the physics is keyed on line_ind.
LINE_IND   = [59, 62, 63, 71, 72]
LINE_NAMES = ["Hbeta", "[OIII]4959", "[OIII]5007", "Halpha", "[NII]6583"]
LINE_WAVE  = [4861.3, 4958.9, 5006.8, 6562.8, 6583.4]   # vacuum rest [A]

TRUTH = {
    "logsfr_ratios":      jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
    # nebular gas (added to theta by add_neb=True; these override defaults)
    "gas_logz":           jnp.array([-0.4]),    # log10 Z_gas / Z_sun
    "gas_logu":           jnp.array([-2.5]),    # log10 ionization parameter
    # slit-loss: the Lines observation sees 80% of the true line flux
    "eline_scaling":      jnp.array([0.8]),
}


def main() -> None:
    if not os.environ.get("SPS_HOME"):
        sys.exit("add_neb=True reads the CLOUDY grids from $SPS_HOME, "
                 "which is unset. See README, 'Installing FSPS'.")
    rng = np.random.default_rng(SEED)

    if SSP_FILE.is_file():
        ssp = SSPData.load(str(SSP_FILE))
    else:
        print("[grid] building SSP grid with FSPS (a few minutes) ...")
        ssp = SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE))

    # add_neb=True: nebular continuum in the SED and emission lines for the
    # Lines observation. The CLOUDY grid is selected to match the isochrone
    # library recorded in the SSP file -- never set isoc_type by hand.
    csp = CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, N_TIME),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=True,
        verbose=False,
    )
    sfh_times_yr = np.array(csp.sfh_times)

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
                # gas + slit-loss: the line-fitting-specific parameters
                "gas_logz":       Uniform(low=-2.0, high=0.5),
                "gas_logu":       Uniform(low=-4.0, high=-1.0),
                "eline_scaling":  Uniform(low=0.1, high=2.0),
            },
            transforms={"sfh": lambda th, _t=sfh_times_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                             sfh_times_yr=_t)},
            free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                             "logmass": jnp.array([10.0])},
            zred=ZRED,
        )

    # ── Mock: photometry + line fluxes from the same forward model ───────
    gen = build_model([
        Photometry(filters=FILTERS, name="phot"),
        Lines(line_ind=LINE_IND, line_names=LINE_NAMES,
              wavelength=jnp.array(LINE_WAVE), name="lines"),
    ])
    pred = gen.predict(TRUTH)

    mag = np.asarray(pred["phot"]); mag_unc = mag / SNR_PHOT
    mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)

    lin = np.asarray(pred["lines"]); lin_unc = np.abs(lin) / SNR_LINE
    lin_obs = lin + lin_unc * rng.standard_normal(lin.shape)

    # ── Observations to fit ───────────────────────────────────────────────
    # Treat GALEX FUV as a non-detection: report the 1-sigma limiting flux
    # in `flux` and flag it -- chi^2 then only penalises models ABOVE it.
    upper = np.zeros(len(FILTERS), dtype=bool); upper[0] = True
    mag_obs[0] = 3.0 * mag_unc[0]

    phot = Photometry(filters=FILTERS, flux=mag_obs, uncertainty=mag_unc,
                      upper_limit=upper, name="phot")
    lines = Lines(line_ind=LINE_IND, line_names=LINE_NAMES,
                  wavelength=jnp.array(LINE_WAVE),
                  flux=lin_obs, uncertainty=lin_unc, name="lines")

    model = build_model([phot, lines])
    result = fitSED(
        model,
        sampler="ns",
        sampler_kwargs={"num_live": 300},  # num_delete/logZ_tol: library defaults
        rng_key=jax.random.PRNGKey(SEED),
        output_dir="./demo_2_output",
    )

    # ── Recovered vs true ─────────────────────────────────────────────────
    lw = np.asarray(result.log_weights)
    w = np.exp(lw - lw.max()); w /= w.sum()
    idx = rng.choice(w.size, size=2000, p=w)
    for p in ("logmass", "Z", "gas_logz", "gas_logu", "eline_scaling"):
        s = np.asarray(result.samples[p])[idx].ravel()
        print(f"{p:>15}: true {float(TRUTH[p][0]):+7.3f}   "
              f"fit {np.median(s):+7.3f} +/- {np.std(s):.3f}")
    # Expect: gas_logz / gas_logu pinned by the [OIII]/Hbeta and [NII]/Halpha
    # ratios; eline_scaling recovered because the photometry anchors the
    # ABSOLUTE line luminosity while the Lines observation sees only 80%.


if __name__ == "__main__":
    main()
