#!/usr/bin/env python3
"""
demo_3_spectrum_advanced.py — fitting a spectrum with the special machinery.

Spectra carry far more information than photometry, and correspondingly more
ways to go wrong. This demo exercises the spectrum-specific features:

    - instrumental line-spread function: ``Spectrum(instrument=Instrument.
      sigma_kms(150.0))``; the unit and the R convention are the constructor
      name (``R_fwhm``, ``R_sigma``, ``fwhm_aa``, ``sigma_aa``, ``sigma_kms``,
      ``fwhm_kms``, each also as a per-pixel array with ``wave=``)
    - fitted stellar velocity dispersion: ``Kinematics(sigma_gal="sigma_gal")``
      on the model names a free theta parameter ``sigma_gal`` [km/s]; the gas
      dispersion of the emission lines is tied to it unless set separately
    - pixel masking:            ``spec.mask_lines(...)`` to exclude emission
      line regions from a continuum-only fit
    - noise floor:              ``noise_floor`` adds a fractional (model-anchored)
      error floor in quadrature; fitSED honours it (and ``sky``, ``calibration``,
      ``upper_limit``) and records it in the fit log
    - joint photometry anchor:  broadband fluxes constrain the continuum
      shape outside the spectral window

Division of labour: the galaxy's dispersions live in ONE place, the
``Kinematics`` object on the model, and the instrument's LSF in ONE place,
the ``Instrument`` on the spectrum. The projection combines them in
quadrature with the SSP library resolution removed, so nothing is broadened
twice and the fitted ``sigma_gal`` is the dispersion itself, not a residual.

Not shown but available: ``calibration=`` (per-pixel multiplicative vector on
the model) and ``spec.fit_polynomial_calibration(model_flux, order)`` for
post-hoc calibration checks.  ``noise=GaussianProcess(...)`` is a diagnostic for
``Spectrum.log_likelihood`` only; fitSED refuses it.

    conda activate <your-env>
    python examples/demo_3_spectrum_advanced.py
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED, Kinematics, Instrument, PostProcess
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT
from ceridwen.cosmology import Cosmology

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = HERE / "ssp_data.h5"

SEED = 42
ZRED = 0.1
N_TIME = 6
SNR_PHOT, SNR_SPEC = 20.0, 25.0
FILTERS = ["galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0",
           "sdss_z0", "twomass_J", "twomass_Ks", "wise_w1"]
SPEC_WAVE = np.linspace(3800.0, 7200.0, 800)    # OBSERVED-frame vacuum [A]
SPEC_RES = 150.0                                # instrument LSF sigma [km/s]

TRUTH = {
    "logsfr_ratios":      jnp.array([+0.3, +0.2, -0.1, -0.4, -0.6]),
    "Z":                  jnp.array([-2.0]),
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
    "sigma_gal":          jnp.array([180.0]),   # stellar velocity dispersion [km/s], FITTED
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

    csp = CSPBasis(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, N_TIME),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True, add_neb=False,
        verbose=False,
        cosmo=Cosmology.planck18(),
    )
    sfh_times_yr = np.array(csp.sfh_times)

    # The galaxy's stellar dispersion is a free parameter named "sigma_gal";
    # the gas dispersion is tied to it (no lines in this fit anyway).
    kin = Kinematics(sigma_gal="sigma_gal")

    def make_spectrum(flux=None, uncertainty=None):
        return Spectrum(
            wavelength=SPEC_WAVE,
            flux=flux, uncertainty=uncertainty,
            instrument=Instrument.sigma_kms(SPEC_RES),   # LSF sigma [km/s]
            # Instrument.R_fwhm(2000) for a datasheet R = lambda/FWHM,
            # Instrument.R_sigma(...) for the sedpy/Prospector R = lambda/sigma,
            # Instrument.fwhm_aa(2.5) for a FWHM in Angstrom; any of them
            # with wave= for a per-pixel curve. The SSP library resolution
            # stored in the schema-2 grid is removed in quadrature
            # automatically (subtract_library=True).
            noise_floor=0.01,   # 1 % of the model flux added in quadrature
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
                # the fitted stellar velocity dispersion (upper bound must
                # stay below Kinematics.sigma_max, 2000 km/s by default)
                "sigma_gal": Uniform(low=50.0, high=400.0),
            },
            transforms={"sfh": lambda th, _t=sfh_times_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                             sfh_times_yr=_t)},
            free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                             "logmass": jnp.array([10.0]),
                             "sigma_gal": jnp.array([200.0])},
            zred=ZRED,
            kinematics=kin,                          # galaxy dispersions, once
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

    # ── Post-process (NUTS draws are uniform-weight; the diagnostics page
    #    shows the per-chain traces with split-R-hat and ESS) ───────────────
    params = ("logmass", "Z", "sigma_gal", "diffuse_tau_kc", "diffuse_dust_index")
    truths = {p: float(TRUTH[p][0]) for p in params}
    pp = PostProcess(model, result)
    out = pp.run()
    pp.figures("./demo_3_output/figures", title="demo 3: photometry + spectrum", truths=truths)
    for p in params:
        s = out["theta"][p]
        print(f"{p:>20}: true {truths[p]:+8.3f}   "
              f"fit {np.median(s):+8.3f} +/- {np.std(s):.3f}")
    # sigma_gal should recover ~180 km/s: the spectrum resolves the
    # absorption-line widths, which photometry cannot see at all. Z and the
    # dust parameters tighten dramatically compared to the photometry-only
    # fit of demo_1 -- that comparison is the whole argument for spectra.


if __name__ == "__main__":
    main()
