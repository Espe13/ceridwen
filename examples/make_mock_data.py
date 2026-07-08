#!/usr/bin/env python3
"""
make_mock_data.py — generate a self-consistent mock galaxy for CERIDWEN.

Builds broadband photometry + an optical spectrum by pushing a known set of
TRUTH parameters through the SAME forward model you fit with, then adds Gaussian
noise. Because the data comes from the current model, its units match
``model.predict()`` exactly — which is what the ``Photometry`` / ``Spectrum``
objects expect — so a fit recovers TRUTH regardless of the absolute scale.

    conda activate <your-env>        # needs CERIDWEN + FSPS ($SPS_HOME) for the grid
    python examples/make_mock_data.py

Writes ``examples/mock_galaxy.npz`` and prints the photometry so you can see the
maggies scale for yourself.

UNITS. The photometry is in observed-frame AB maggies (1 maggie = 3631 Jy) and
is absolutely calibrated: ``SedModel.predict`` injects the fixed ZRED into the
forward model, which applies (1+z) * (10pc/D_L)^2 and the L_sun/Hz -> cgs
conversion (see ceridwen.cosmology.flux_factor_maggies). A z=0.1,
logmass~10.5 galaxy lands at ~1e-7 maggies (AB ~ 17-18) in the bright bands —
if the printed max is far from that, run scripts/verify_flux_normalization.py.
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # runs fine on any laptop

import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"
OUT = HERE / "mock_galaxy.npz"

# ---- mock configuration (all recorded in the output file) -----------------
ZRED = 0.1                              # fixed spectroscopic redshift
N_TIME = 6                              # SFH nodes -> 5 free logsfr_ratios
T_OLDEST = 12.0                         # Gyr, oldest SFH node (< age of universe at z)
FILTERS = ["galex_FUV", "galex_NUV",
           "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
           "twomass_J", "twomass_H", "twomass_Ks",
           "wise_w1", "wise_w2"]
SPEC_WAVE = np.linspace(4000.0, 8000.0, 600)     # AA, vacuum, OBSERVED frame
SPEC_RES = 150.0                        # km/s instrumental sigma (smoothtype="vel")
SNR_PHOT, SNR_SPEC = 20.0, 25.0

TRUTH = {
    "logsfr_ratios":      jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),      # log10 ABSOLUTE Z (ssp_lgmet units)
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}


def main() -> None:
    # Reuse examples/ssp_data.h5 if present; otherwise build it with the SAME
    # call the quickstart uses, so the mock matches your grid.
    if SSP_FILE.is_file():
        ssp = SSPData.load(str(SSP_FILE))
    else:
        print("no examples/ssp_data.h5 — building it with "
              "SSPData.from_fsps(imf_type=1) ...")
        ssp = SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE))

    csp = CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, T_OLDEST, N_TIME),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=False,
        verbose=False,
    )
    sfh_times_yr = np.array(csp.sfh_times)

    # Empty observations (filters / wavelengths only) are enough to predict.
    model = SedModel(
        csp,
        observations=[
            Photometry(filters=FILTERS, name="phot"),
            Spectrum(wavelength=SPEC_WAVE, resolution=SPEC_RES,
                     smoothtype="vel", name="spec"),
        ],
        transforms={"sfh": lambda th, _t=sfh_times_yr:
                    logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
        free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                         "logmass": jnp.array([10.0])},
        zred=ZRED,
    )

    noiseless = model.predict(TRUTH)          # dict keyed by obs.name
    rng = np.random.default_rng(42)

    maggies = np.asarray(noiseless["phot"])
    maggies_unc = maggies / SNR_PHOT
    maggies_obs = maggies + maggies_unc * rng.standard_normal(maggies.shape)

    flux = np.asarray(noiseless["spec"])
    flux_unc = np.abs(flux) / SNR_SPEC
    flux_obs = flux + flux_unc * rng.standard_normal(flux.shape)

    np.savez(
        OUT,
        # observations (photometry in AB maggies, model-consistent)
        filters=np.array(FILTERS),
        maggies=maggies_obs, maggies_unc=maggies_unc,
        spec_wave_obs=SPEC_WAVE, spec_flux=flux_obs, spec_unc=flux_unc,
        spec_resolution=SPEC_RES,
        # model setup needed to reproduce the fit
        zred=ZRED, lookback_time=np.linspace(0.0, T_OLDEST, N_TIME),
        # injected truth (for a recovered-vs-true table)
        **{f"true_{k}": np.asarray(v) for k, v in TRUTH.items()},
    )

    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} kB)")
    print(f"photometry [AB maggies]: min={maggies_obs.min():.3e}  "
          f"max={maggies_obs.max():.3e}  ({len(FILTERS)} bands)")
    print("load it into a Photometry object with:")
    print('    d = np.load("examples/mock_galaxy.npz")')
    print('    phot = Photometry(filters=[str(f) for f in d["filters"]],')
    print('                      flux=d["maggies"], uncertainty=d["maggies_unc"],')
    print('                      name="phot")')


if __name__ == "__main__":
    main()
