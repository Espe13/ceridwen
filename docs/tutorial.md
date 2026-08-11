# Tutorial: fitting photometry, spectroscopy and lines jointly

CERIDWEN treats the three observation types uniformly: broadband **photometry**,
a resolved **spectrum**, and **emission-line** fluxes. Each is a small container
that knows how to project the model spectrum onto its own data space, and the
joint likelihood is the sum of their χ² contributions. You can fit any one of
them, or all three together, with no change to the model or sampler.

This tutorial builds one of each and fits them jointly. Replace the mock arrays
with your own data.

!!! tip "Runnable notebook"
    A notebook version of this tutorial is in the repository at
    [`examples/tutorial_joint_fit.ipynb`](https://github.com/Espe13/ceridwen/blob/main/examples/tutorial_joint_fit.ipynb).

!!! note "Before you start"
    Read **[Conventions & gotchas](conventions.md)**, especially that `Z` is
    log10 *absolute* metallicity and that `lookback_time` index 0 is *today*.
    Make sure FSPS and `$SPS_HOME` are set up ([Installation](installation.md));
    emission lines and nebular continuum need the CLOUDY grids from FSPS.

## 1. Build the forward model

Lines and nebular continuum require `add_neb=True`. We also enable diffuse and
birth-cloud dust here.

```python
import jax, jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry, Spectrum, Lines
from ceridwen.priors import Uniform, ClippedNormal, StudentT
from ceridwen.model import logsfr_ratios_to_sfh

ssp = SSPData.load("ssp_data.h5")          # built once via SSPData.from_fsps(...)
# The grid records its isochrone library; CSPBasis reads it automatically,
# so isoc_type never needs to be passed in init_neb_params.

ZRED = 0.5                                  # spectroscopic redshift of the galaxy

csp = CSPBasis(
    ssp,
    add_dust=True, add_diffuse_dust=True,   # birth-cloud + diffuse attenuation
    add_neb=True,                           # nebular continuum + lines (needs $SPS_HOME)
    add_igm=True,                           # Madau (1995), auto-scales with zred
    # sps_home defaults to $SPS_HOME
)
```

## 2. (a) Photometry

Broadband fluxes in **AB maggies**, with filter names resolved from the
`sedpy_jax` library. Photometry captures the full aperture, so it sees the
intrinsic (unscaled) line + continuum flux.

```python
# If your catalogue is in nJy (common for JWST), convert to maggies and apply a
# small error floor (here 5%), as the JADES pipeline does:
flux_nJy, unc_nJy = my_phot_nJy, my_phot_unc_nJy
unc_nJy = np.where(unc_nJy / flux_nJy > 0.05, unc_nJy, 0.05 * flux_nJy)

phot = Photometry(
    filters=["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w",
             "jwst_f277w", "jwst_f356w", "jwst_f444w"],
    flux=jnp.asarray(flux_nJy) * 1e-9 / 3631.0,        # nJy -> AB maggies
    uncertainty=jnp.asarray(unc_nJy) * 1e-9 / 3631.0,
    name="phot",                    # the key this observation is reported under
)
```

Optional: `mask=` (bool per band, True = used) and `upper_limit=` (bool per band;
non-detections enter as one-sided χ²).

## 3. (b) Spectrum

A densely-sampled spectrum. Pass the **observed-frame, vacuum** wavelength grid
in Å (the pixel wavelengths as delivered by the instrument, since the forward
model redshifts the model spectrum by `(1 + zred)` onto these pixels; at
`zred = 0` observed and rest frame coincide) and flux in `F_ν` (same system as
the model).
Set `resolution` + `smoothtype` to have the forward model apply instrumental
broadening.

```python
spec = Spectrum(
    wavelength=my_obs_wave_aa,      # Å, vacuum, OBSERVED frame, shape (n_pix,)
    flux=my_spec_fnu,               # F_nu per pixel
    uncertainty=my_spec_unc,
    resolution=120.0,               # see smoothtype for the unit
    smoothtype="vel",               # "vel" (km/s), "R", "lambda" (Å), or "lsf"
    inres=0.0,                      # model library intrinsic resolution to deconvolve
    noise_floor=0.01,               # 1% multiplicative calibration floor (optional)
    name="spec",
)

# Optional: mask known emission lines from the *continuum* spectrum fit so they
# do not double-count against the Lines object, redshifting line centres first.
spec.mask_lines([4861.3, 5006.8, 6562.8], dv=500.0, zred=ZRED)
```

Other optional knobs: `calibration=` (per-pixel multiplicative flux-calibration
vector), `sigma_losvd=` (galaxy velocity dispersion applied before instrumental
smoothing), and `mask=`.

## 4. (c) Emission lines

Integrated line fluxes. `line_ind` are **1-based** indices into FSPS's
`emlines_info.dat`; `wavelength` are the vacuum rest wavelengths in Å.

```python
# Catalogue line fluxes are often quoted in 1e-20 erg s^-1 cm^-2; scale to
# absolute CGS to match the model (adjust the factor to your catalogue).
LINE_UNIT = 1.0e-20

lines = Lines(
    line_ind=[59, 62, 63, 71, 72],                         # Hβ, [OIII]4959/5007, Hα, [NII]6583
    line_names=["Hbeta", "[OIII]4959", "[OIII]5007", "Halpha", "[NII]6583"],
    wavelength=[4861.3, 4958.9, 5006.8, 6562.8, 6583.4],   # Å, vacuum rest
    flux=np.asarray(my_line_flux) * LINE_UNIT,             # erg s^-1 cm^-2
    uncertainty=np.asarray(my_line_unc) * LINE_UNIT,
    name="lines",
)
```

!!! tip "Two independent calibrations: `eline_scaling` and `spectrum_scaling`"
    Photometry sees the full field of view, but slit/fibre spectroscopy and
    aperture-measured line fluxes lose (or miscalibrate) flux. CERIDWEN
    exposes **two separate, independent** nuisances for this:

    - `eline_scaling` — the fractional aperture correction applied to the
      emission-**LINE** component only (1.0 = no loss, 0.65 = lines at 65%).
      It drives the `Lines` observation and does **not** touch the spectrum.
    - `spectrum_scaling` — a multiplicative spectrophotometric normalisation applied
      to the whole **`Spectrum`** prediction (continuum + any lines), rescaling
      it onto the photometric flux scale. Photometry is left unscaled, so it
      anchors the absolute flux while `spectrum_scaling` absorbs the spectrum's
      uncertain flux calibration (the Prospector `spec_norm` convention).

    The two are decoupled by construction: `eline_scaling` scales lines,
    `spectrum_scaling` scales the spectrum, and neither affects the photometry. Add a
    prior on each nuisance you want to marginalise over (below).

## 5. Priors and the model

Collect the observations into a single list. Any subset is fine; use an empty
list for a type you are not fitting. Then define priors for every free parameter.

```python
observations = [phot, spec, lines]

priors = {
    # Stellar population. Z is log10 ABSOLUTE metallicity (grid ~[-4, -1.4]).
    "Z":                 ClippedNormal(mean=-2.0, sigma=0.5, low=-4.0, high=-1.4),
    "logmass":           Uniform(low=7.0, high=12.5),
    "logsfr_ratios":     StudentT(df=2.0, mean=0.0, scale=0.3),   # non-parametric SFH
    # Dust.
    "diffuse_tau_kc":    ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
    "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
    "tau_pow":           ClippedNormal(mean=0.3, sigma=0.5, low=0.0, high=4.0),
    "alpha_pow":         ClippedNormal(mean=-1.0, sigma=0.5, low=-2.5, high=0.5),
    # Nebular (required for the Lines / nebular continuum).
    "gas_logz":          Uniform(low=-2.0, high=0.5),
    "gas_logu":          Uniform(low=-4.0, high=-1.0),
    # Emission-line aperture correction (Lines observation only).
    "eline_scaling":     Uniform(low=0.1, high=2.0),
    # Spectrophotometric normalisation of the spectrum onto the photometry
    # (Spectrum observation only; independent of eline_scaling). Omit if the
    # spectrum is already flux-calibrated to the photometric system.
    "spectrum_scaling":         ClippedNormal(mean=1.0, sigma=0.3, low=0.2, high=3.0),
}

# The non-parametric SFH is sampled as logsfr_ratios and turned into the per-bin
# sfh by a REGISTERED transform; this step is required for logsfr_ratios to work.
sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

N_RATIOS = 4   # number of SFH bins - 1
model = SedModel(
    csp,
    observations=observations,
    priors=priors,
    transforms={"sfh": logsfr_to_sfh},        # REQUIRED for logsfr_ratios
    free_param_init={"logsfr_ratios": jnp.zeros(N_RATIOS),
                     "logmass": jnp.array([10.0])},
    zred=ZRED,                                # fixed spectroscopic redshift
)

# Fixed knob not sampled here: the birth-cloud (Charlot & Fall) slope.
model.theta_init["alpha_pow"] = jnp.array([-1.0])
```

To fit redshift instead of fixing it, omit `zred` and add a `"zred"` prior (and,
for a non-parametric SFH, register a `"lookback_time"` transform so the SFH
age-bin grid tracks the sampled redshift).

## 6. Fit

`fitSED` builds the joint likelihood automatically from `model.observations`
(one Gaussian likelihood per observation, keyed by `name`) and writes an HDF5
result.

```python
result = fitSED(
    model,
    observations,
    sampler="nested",               # or "nuts" (optionally vi="tril")
    rng_key=jax.random.PRNGKey(0),
    output_dir="./joint_fit",
)

print(f"ln Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")
```

## 7. Posterior and diagnostics

```python
ns = result.to_anesthetic()                      # nested-sampling posterior
post = ns.sample(2000, replace=True)
print("median logmass:", float(np.median(post["logmass"])))

# Predicted data for any posterior point (keyed by observation name):
theta = {k: jnp.asarray([float(np.median(post[k]))]) for k in
         ("Z", "logmass", "diffuse_tau_kc", "diffuse_dust_index",
          "tau_pow", "alpha_pow", "gas_logz", "gas_logu", "eline_scaling")}
theta["logsfr_ratios"] = jnp.asarray(
    [float(np.median(post[f"logsfr_ratios[{i}]"])) for i in range(4)])

pred = model.predict(theta)        # {"phot": maggies, "spec": F_nu, "lines": fluxes}
```

`result.samples` holds the posterior keyed by parameter name; the HDF5 file in
`output_dir` stores the observations, priors, samples, and (for nested sampling)
the log-evidence. See `examples/quickstart.py` for a complete photometry-only
script that also builds a corner plot and a model-vs-data figure.

## Consistency checklist for real joint fits

- **Flux systems must agree.** Photometry (maggies), spectrum (`F_ν`) and line
  fluxes (erg s⁻¹ cm⁻²) must be calibrated to the same physical normalisation
  the model produces at `zred`. Inconsistent absolute calibration between data
  sets is the most common cause of a "good χ² per set but bad joint fit".
- **Don't double-count lines.** If you fit both a spectrum *and* the line
  fluxes, mask the lines out of the continuum spectrum (`spec.mask_lines(...)`).
- **Aperture and flux calibration.** Use `eline_scaling` for the emission-line
  aperture loss (Lines) and `spectrum_scaling` for the spectrum's overall
  flux-calibration offset relative to the photometry (Spectrum). They are
  independent; fit whichever your data need. `noise_floor` (and a fixed
  per-pixel `calibration` vector) further absorb residual systematics.
- **Know your frames.** The spectrum's pixel grid is **observed-frame** vacuum
  Å (the model is redshifted onto it); line-list wavelengths and
  `mask_lines(...)` centres are **rest-frame** vacuum Å (redshifted internally
  by `(1 + zred)`).
