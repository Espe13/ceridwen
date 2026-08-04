#!/usr/bin/env python3
"""
demo_3_spectrum_advanced.py — fitting a spectrum with the special machinery.

Spectra carry far more information than photometry, and correspondingly more
ways to go wrong. This demo exercises the spectrum-specific features:

    - instrumental smoothing:   ``smoothtype="vel"`` + ``resolution`` [km/s]
      (alternatives: "R" resolving power, "lambda" sigma in A, or "lsf" with
      a per-pixel sigma(lambda) array for real instrument LSFs)
    - fitted stellar velocity dispersion: ``fit_sigma_smooth=True`` promotes
      the galaxy LOSVD to a free theta parameter ``sigma_smooth`` [km/s]
    - pixel masking:            ``spec.mask_lines(...)`` to exclude emission
      line regions from a continuum-only fit
    - noise floor:              ``noise_floor`` adds a fractional error floor
      in quadrature (guards against overconfident pipeline uncertainties)
    - joint photometry anchor:  broadband fluxes constrain the continuum
      shape outside the spectral window

Division of labor for the LOSVD: the CSPBasis applies a SOURCE-side LOSVD
(``sigma_losvd_kms``, default 300 km/s) to the full SED before any
projection. When the Spectrum observation fits ``sigma_smooth`` at runtime,
set ``sigma_losvd_kms=0.0`` on the CSPBasis so the dispersion is applied
exactly once, by the observation that measures it.

Not shown but available: ``calibration=`` (per-pixel multiplicative vector),
``spec.fit_polynomial_calibration(model_flux, order)`` for post-hoc
calibration checks, and ``noise=GaussianProcess(amplitude, length_scale)``
for correlated-residual modeling (enters via ``Spectrum.log_likelihood`` in
custom likelihood pipelines; fitSED's default likelihood is diagonal).

    conda activate <your-env>
    python examples/demo_3_spectrum_advanced.py
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"

SEED = 42
ZRED = 0.1
N_TIME = 6
SNR_PHOT, SNR_SPEC = 20.0, 25.0
FILTERS = ["galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0",
           "sdss_z0", "twomass_J", "twomass_Ks", "wise_w1"]
SPEC_WAVE = np.linspace(3800.0, 7200.0, 800)    # OBSERVED-frame vacuum [A]
SPEC_RES = 150.0                                # instrument sigma_v [km/s]

TRUTH = {
    "logsfr_ratios":      jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
    "sigma_smooth":       jnp.array([180.0]),   # galaxy LOSVD [km/s], FITTED
}

# Strong optical lines to mask in this continuum-only fit (rest vacuum A).
MASK_LINES = [3727.1, 4861.3, 4958.9, 5006.8, 6562.8, 6583.4]


def main() -> None:
    rng = np.random.default_rng(SEED)

    if SSP_FILE.is_file():
        ssp = SSPData.load(str(SSP_FILE))
    else:
        print("[grid] building SSP grid with FSPS (a few minutes) ...")
        ssp = SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE))

    # sigma_losvd_kms=0: the Spectrum observation owns the LOSVD here
    # (fit_sigma_smooth=True below); leaving the CSP default of 300 km/s
    # would smooth the SED twice.
    csp = CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, N_TIME),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=False,
        sigma_losvd_kms=0.0,
        verbose=False,
    )
    sfh_times_yr = np.array(csp.sfh_times)

    def make_spectrum(flux=None, uncertainty=None):
        return Spectrum(
            wavelength=SPEC_WAVE,
            flux=flux, uncertainty=uncertainty,
            resolution=SPEC_RES, smoothtype="vel",   # instrument broadening
            # inres=<library sigma_v>  would deconvolve the SSP library
            # resolution in quadrature; 0 (default) treats it as exact.
            fit_sigma_smooth=True,                   # LOSVD from theta
            noise_floor=0.01,                        # 1% error floor
            name="spec",
        )

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
                # the fitted stellar velocity dispersion
                "sigma_smooth": Uniform(low=50.0, high=400.0),
            },
            transforms={"sfh": lambda th, _t=sfh_times_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                             sfh_times_yr=_t)},
            free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                             "logmass": jnp.array([10.0]),
                             "sigma_smooth": jnp.array([200.0])},
            zred=ZRED,
        )

    # ── Mock: photometry + spectrum through the same forward model ────────
    gen = build_model([Photometry(filters=FILTERS, name="phot"),
                       make_spectrum()])
    pred = gen.predict(TRUTH)

    mag = np.asarray(pred["phot"]); mag_unc = mag / SNR_PHOT
    mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)

    sfx = np.asarray(pred["spec"]); sfx_unc = np.abs(sfx) / SNR_SPEC
    sfx_obs = sfx + sfx_unc * rng.standard_normal(sfx.shape)

    # ── Observations to fit ────────────────────────────────────────────────
    phot = Photometry(filters=FILTERS, flux=mag_obs, uncertainty=mag_unc,
                      name="phot")
    spec = make_spectrum(flux=sfx_obs, uncertainty=sfx_unc)
    # Continuum-only fit: mask +/-800 km/s around each strong line. The
    # rest wavelengths are shifted by zred internally to match the
    # observed-frame pixel grid.
    spec.mask_lines(MASK_LINES, dv=800.0, zred=ZRED)

    model = build_model([phot, spec])

    # NUTS + VI: gradients shine when a spectrum adds hundreds of data
    # points; the VI transport map cuts warmup dramatically.
    result = fitSED(
        model,
        sampler="nuts",
        vi="tril",
        sampler_kwargs={"num_chains": 2, "num_samples": 1000},
        rng_key=jax.random.PRNGKey(SEED),
        output_dir="./demo_3_output",
    )

    # ── Recovered vs true (NUTS samples are unweighted) ───────────────────
    for p in ("logmass", "Z", "sigma_smooth",
              "diffuse_tau_kc", "diffuse_dust_index"):
        s = np.asarray(result.samples[p]).ravel()
        print(f"{p:>20}: true {float(TRUTH[p][0]):+8.3f}   "
              f"fit {np.median(s):+8.3f} +/- {np.std(s):.3f}")
    # sigma_smooth should recover ~180 km/s: the spectrum resolves the
    # absorption-line widths, which photometry cannot see at all. Z and the
    # dust parameters tighten dramatically compared to the photometry-only
    # fit of demo_1 -- that comparison is the whole argument for spectra.


if __name__ == "__main__":
    main()
