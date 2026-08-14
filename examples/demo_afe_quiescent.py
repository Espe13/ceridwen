#!/usr/bin/env python3
"""
demo_afe_quiescent.py -- [alpha/Fe] fitting of a quiescent galaxy from a
Legacy-Surveys-style spectrum + broadband photometry, with a fitted
spectrophotometric normalisation (``spectrum_scaling``).

What this demo shows
--------------------
Quiescent, early-quenched galaxies lock in supersolar [alpha/Fe]: the alpha
elements are released promptly by core-collapse supernovae while the bulk of
the iron arrives later from Type Ia supernovae, so a short, early burst leaves
an alpha-enhanced fossil record. Their continuum-dominated spectra are exactly
where [alpha/Fe] is both strongest and cleanest to measure. This demo fits:

    - a NON-parametric star-formation history: 10 bins with a continuity
      (Student-t on the log SFR ratios) prior
    - the total stellar metallicity ``Z`` (log10 ABSOLUTE metallicity)
    - the alpha-enhancement ``afe`` = [alpha/Fe] (the new axis)
    - diffuse-dust optical depth ``diffuse_tau_kc`` and slope
      ``diffuse_dust_index`` (Kriek & Conroy law; no birth-cloud dust)
    - a fixed redshift
    - ``spectrum_scaling``: a fitted spectrophotometric normalisation that rescales
      the model spectrum onto the photometric flux scale

The model is ``CSPBasis_afe`` -- the alpha-enhanced, NEBULAR-FREE forward
model (there are no alpha-enhanced CLOUDY grids, so nebular emission is
switched off; for a genuinely quiescent galaxy this is the physically correct
choice -- see ceridwen.csp.csp_afe). Because there is no nebular component,
``eline_scaling`` (the emission-LINE aperture correction) has nothing to act
on here. The whole-spectrum-to-photometry calibration is a SEPARATE nuisance,
``spectrum_scaling``, which is what a slit/fibre spectrum actually needs: photometry
captures the full galaxy and anchors the absolute flux, while ``spectrum_scaling``
absorbs the uncertain flux calibration of the spectrograph trace. ``spectrum_scaling``
and ``eline_scaling`` are independent by construction (see
CSPBasis._assemble_observer_spectra).

The example is self-contained: it builds a mock quiescent galaxy through the
forward model (injecting a known ``spectrum_scaling`` so the observed spectrum sits
at a different flux level than the photometry), perturbs it with noise, and
refits with the nested slice sampler. To fit REAL data, replace the mock
``phot``/``spec`` construction with your Legacy Surveys photometry and
spectrum (keep the spectrum on an OBSERVED-frame vacuum-Angstrom grid).

    conda activate <your-env>
    python examples/demo_afe_quiescent.py
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen import SedModel, fitSED
from ceridwen.csp import CSPBasis_afe
from ceridwen.ssps import SSPDataAfe
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

HERE = pathlib.Path(__file__).resolve().parent

# Alpha-enhanced 4-D SSP grid (aMIST isochrones + C3K spectra, n_afe=5 over
# [alpha/Fe] in {-0.2, 0.0, +0.2, +0.4, +0.6}). Prefer a local copy; else
# fall back to the published download helper (Zenodo).
LOCAL_GRIDS = [
    HERE / "amist_c3k_lr_chab_afe.h5",
    HERE.parent / "ceridwen" / "data" / "test_data" / "amist_c3k_lr_chab_afe.h5",
]

SEED = 7
ZRED = 0.10                         # fixed redshift (quiescent, Legacy-observable)
N_BINS = 10                         # SFH bins -> N_BINS-1 = 9 logsfr_ratios
GALAXY_LOSVD = 200.0                # stellar velocity dispersion [km/s], FIXED
SPEC_RES_KMS = 70.0                 # instrument resolution sigma_v [km/s]
SNR_PHOT, SNR_SPEC = 30.0, 30.0

# Legacy Surveys bands: DECam grz + WISE W1/W2 (decam_* / wise_* are valid
# sedpy_jax filter names). Swap in your own broadband set as needed.
FILTERS = ["decam_g", "decam_r", "decam_z", "wise_w1", "wise_w2"]

# Legacy-Surveys-style optical spectrum: observed-frame vacuum Angstrom.
# (DESI covers ~3600-9800 A; use your instrument's grid for real data.)
SPEC_WAVE = np.linspace(3600.0, 9800.0, 1500)

# ---- Truth for the mock quiescent galaxy --------------------------------
# Negative logsfr_ratios => SFR was higher in the PAST and declines toward the
# present (an old, early-quenched population); see logsfr_ratios_to_sfh.
LOGSFR_RATIOS_TRUE = -0.5 * jnp.ones(N_BINS - 1)
TRUTH = {
    "logsfr_ratios":      LOGSFR_RATIOS_TRUE,
    "Z":                  jnp.array([-2.3]),     # log10 absolute metallicity
    "afe":                jnp.array([+0.3]),     # supersolar [alpha/Fe]
    "logmass":            jnp.array([10.8]),
    "diffuse_tau_kc":     jnp.array([0.15]),     # low dust (quiescent)
    "diffuse_dust_index": jnp.array([-0.5]),
    # spectrophotometric normalisation: the observed spectrum sits at 85% of
    # the photometric flux scale (a deliberate slit/fibre miscalibration that
    # spectrum_scaling must recover; photometry anchors the true absolute scale).
    "spectrum_scaling":          jnp.array([0.85]),
}


def _load_grid() -> SSPDataAfe:
    for p in LOCAL_GRIDS:
        if p.is_file():
            return SSPDataAfe.load(str(p))
    # Published-grid fallback (cached download).
    from ceridwen.ssps import fetch_grid
    return SSPDataAfe.load(str(fetch_grid("amist_c3k_lr_chab_afe")))


def main() -> None:
    rng = np.random.default_rng(SEED)
    ssp = _load_grid()

    # Alpha-enhanced, nebular-free CSP. add_dust=False (no birth-cloud dust);
    # add_diffuse_dust=True gives the diffuse tau + slope we fit. The galaxy
    # LOSVD is applied source-side here (sigma_losvd_kms); the Spectrum then
    # adds only the INSTRUMENT broadening, so the dispersion is not double
    # counted. (We are NOT fitting sigma_smooth in this demo.)
    csp = CSPBasis_afe(
        ssp,
        lookback_time=jnp.linspace(0.0, 12.0, N_BINS),
        zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=True,
        sigma_losvd_kms=GALAXY_LOSVD,
        verbose=False,
    )
    sfh_times_yr = np.array(csp.sfh_times)

    def make_spectrum(flux=None, uncertainty=None):
        return Spectrum(
            wavelength=SPEC_WAVE,
            flux=flux, uncertainty=uncertainty,
            resolution=SPEC_RES_KMS, smoothtype="vel",   # instrument broadening
            fit_sigma_smooth=False,                       # LOSVD held fixed
            noise_floor=0.01,                             # 1% error floor
            name="spec",
        )

    def build_model(observations):
        return SedModel(
            csp, observations=observations,
            priors={
                # Stellar population.
                "Z":   ClippedNormal(mean=-2.0, sigma=0.6, low=-3.9, high=-1.45),
                "afe": Uniform(low=-0.2, high=0.6),        # over the grid support
                "logmass": Uniform(low=9.0, high=12.0),
                "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),  # continuity
                # Diffuse dust (optical depth + slope).
                "diffuse_tau_kc": ClippedNormal(mean=0.2, sigma=0.5,
                                                low=0.0, high=4.0),
                "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
                # Spectrophotometric normalisation of the spectrum -> photometry.
                "spectrum_scaling": ClippedNormal(mean=1.0, sigma=0.3,
                                           low=0.2, high=3.0),
            },
            transforms={"sfh": lambda th, _t=sfh_times_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"],
                                             sfh_times_yr=_t)},
            free_param_init={
                "logsfr_ratios": jnp.zeros(N_BINS - 1),
                "logmass":       jnp.array([10.0]),
                "afe":           jnp.array([0.0]),
                "spectrum_scaling":     jnp.array([1.0]),
            },
            zred=ZRED,                                     # redshift FIXED
        )

    # -- Mock: photometry + spectrum through the same forward model ---------
    # The TRUTH dict carries spectrum_scaling=0.85, so csp.predict scales the model
    # spectrum by 0.85 while leaving the photometry on the true scale.
    gen = build_model([Photometry(filters=FILTERS, name="phot"),
                       make_spectrum()])
    pred = gen.predict(TRUTH)

    mag = np.asarray(pred["phot"]); mag_unc = mag / SNR_PHOT
    mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)

    sfx = np.asarray(pred["spec"]); sfx_unc = np.abs(sfx) / SNR_SPEC
    sfx_obs = sfx + sfx_unc * rng.standard_normal(sfx.shape)

    # -- Observations to fit ------------------------------------------------
    phot = Photometry(filters=FILTERS, flux=mag_obs, uncertainty=mag_unc,
                      name="phot")
    spec = make_spectrum(flux=sfx_obs, uncertainty=sfx_unc)

    model = build_model([phot, spec])

    # Nested slice sampler (the CERIDWEN default engine).
    result = fitSED(
        model,
        sampler="ns",
        sampler_kwargs={"num_live": 400},  # num_delete/logZ_tol: library defaults
        rng_key=jax.random.PRNGKey(SEED),
        output_dir="./demo_afe_quiescent_output",
    )

    # -- Recovered vs true (weight NS samples by their importance weights) --
    lw = np.asarray(result.log_weights)
    w = np.exp(lw - lw.max()); w /= w.sum()
    idx = rng.choice(w.size, size=4000, p=w)
    print("\nparameter            truth      median      std")
    print("-" * 48)
    for p in ("afe", "Z", "logmass", "diffuse_tau_kc",
              "diffuse_dust_index", "spectrum_scaling"):
        s = np.asarray(result.samples[p])[idx].ravel()
        print(f"{p:>20} {float(TRUTH[p][0]):+8.3f} {np.median(s):+10.3f} "
              f"{np.std(s):8.3f}")
    # Expect: afe recovered near +0.3 from the Mg/Ca vs Fe absorption balance;
    # spectrum_scaling recovered near 0.85 because the photometry anchors the absolute
    # flux while the spectrum is free to slide onto it; Z/mass/dust as usual.


if __name__ == "__main__":
    main()
