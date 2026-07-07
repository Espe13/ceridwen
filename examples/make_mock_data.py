"""Generate the mock dataset used by the README "fit a galaxy end-to-end" example.

Produces ``examples/mock_galaxy.npz``: joint UV-to-IR broadband photometry and
an optical spectrum of a mock galaxy at fixed z = 0.1, generated with the SAME
forward model used in fitting (``SedModel.predict``), with Gaussian noise
added at fixed SNR. The injected truth values are stored in the file so the
README script can print a recovered-vs-true table.

Needs an SSP grid but NOT FSPS (resolution order: $SSP_FILE ->
examples/ssp_data.h5 -> the git-lfs test fixture). Flux units are the
model's native units (L_sun Hz^-1 for the mass-scaled spectrum; AB maggies
for photometry) — the point of the mock is a self-consistent, runnable fit,
not an absolute flux calibration exercise.

Run:  python examples/make_mock_data.py
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # runs fine on any laptop

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "mock_galaxy.npz"

# ---- mock configuration (all recorded in the output file) -----------------
ZRED = 0.1                       # fixed spectroscopic redshift
N_TIME = 6                       # SFH nodes -> 5 free logsfr_ratios
T_OLDEST = 12.0                  # Gyr, oldest SFH node (< age of universe at z)
FILTERS = ["galex_FUV", "galex_NUV",
           "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
           "twomass_J", "twomass_H", "twomass_Ks",
           "wise_w1", "wise_w2"]
SPEC_WAVE_OBS = np.linspace(4000.0, 8000.0, 600)   # AA, vacuum, OBSERVED frame
SPEC_RESOLUTION = 150.0          # km/s instrumental sigma (smoothtype="vel")
SNR_PHOT, SNR_SPEC = 20.0, 25.0

TRUTH = {
    "logsfr_ratios": jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),   # log10 ABSOLUTE Z (ssp_lgmet units)
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}


def _default_ssp_file() -> str:
    env = os.environ.get("SSP_FILE")
    if env:
        return env
    local = HERE / "ssp_data.h5"
    if local.is_file():
        return str(local)
    fixture = HERE.parent / "tests" / "fixtures" / "ssp_data_test.h5"
    if fixture.is_file() and fixture.stat().st_size > 10_000:
        return str(fixture)
    raise SystemExit(
        "No SSP grid found: set $SSP_FILE, place a grid at examples/"
        "ssp_data.h5 (build with FSPS or download from Zenodo, see "
        "docs/installation.md), or `git lfs pull` for the test fixture."
    )


def main() -> None:
    ssp = SSPData.load(_default_ssp_file())
    csp = CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, T_OLDEST, N_TIME),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=False,
        verbose=False,
    )

    phot = Photometry(filters=FILTERS, name="phot")
    spec = Spectrum(wavelength=SPEC_WAVE_OBS,
                    resolution=SPEC_RESOLUTION, smoothtype="vel",
                    name="spec")

    sfh_times_yr = np.array(csp.sfh_times)
    model = SedModel(
        csp, observations=[phot, spec],
        transforms={"sfh": lambda th, _t=sfh_times_yr:
                    logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
        free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                         "logmass": TRUTH["logmass"]},
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
        # observations
        filters=np.array(FILTERS),
        maggies=maggies_obs, maggies_unc=maggies_unc,
        spec_wave_obs=SPEC_WAVE_OBS, spec_flux=flux_obs, spec_unc=flux_unc,
        spec_resolution=SPEC_RESOLUTION,
        # model setup needed to reproduce the fit
        zred=ZRED, lookback_time=np.linspace(0.0, T_OLDEST, N_TIME),
        # injected truth (for the recovered-vs-true table)
        **{f"true_{k}": np.asarray(v) for k, v in TRUTH.items()},
    )
    print(f"wrote {OUT} "
          f"({OUT.stat().st_size/1024:.0f} kB): "
          f"{len(FILTERS)} bands + {SPEC_WAVE_OBS.size}-pixel spectrum, "
          f"z = {ZRED}")


if __name__ == "__main__":
    main()
