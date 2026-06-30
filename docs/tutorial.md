# Tutorial: fitting photometry, spectroscopy and lines jointly

CERIDWEN treats the three observation types — broadband **photometry**, a
resolved **spectrum**, and **emission-line** fluxes — uniformly. Each is a small
container that knows how to project the model spectrum onto its own data space,
and the joint likelihood is simply the sum of their χ² contributions. You can fit
any one of them, or all three together, with no change to the model or sampler.

This tutorial builds one of each and fits them jointly. Replace the mock arrays
with your own data.

!!! note "Before you start"
    Read **[Conventions & gotchas](conventions.md)** — especially that `Z` is
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

ssp = SSPData.load("ssp_data.h5")          # built once via SSPData.from_fsps(...)

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
phot = Photometry(
    filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
             "twomass_J", "twomass_H", "twomass_Ks"],
    flux=my_maggies,                # shape (n_filters,), AB maggies
    uncertainty=my_maggies_unc,     # 1-sigma, AB maggies
    name="phot",                    # the key this observation is reported under
)
```

Optional: `mask=` (bool per band, True = used) and `upper_limit=` (bool per band;
non-detections enter as one-sided χ²).

## 3. (b) Spectrum

A densely-sampled spectrum. Pass the **rest-frame, vacuum** wavelength grid in Å
and flux in `F_ν` (same system as the model). Set `resolution` + `smoothtype` to
have the forward model apply instrumental broadening.

```python
spec = Spectrum(
    wavelength=my_rest_wave_aa,     # Å, vacuum, rest-frame, shape (n_pix,)
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
lines = Lines(
    line_ind=[59, 62, 63, 71, 72],                         # Hβ, [OIII]4959/5007, Hα, [NII]6583
    line_names=["Hbeta", "[OIII]4959", "[OIII]5007", "Halpha", "[NII]6583"],
    wavelength=[4861.3, 4958.9, 5006.8, 6562.8, 6583.4],   # Å, vacuum rest
    flux=my_line_flux,              # erg s^-1 cm^-2 (consistent with the model)
    uncertainty=my_line_unc,
    name="lines",
)
```

!!! tip "Slit losses and `eline_scaling`"
    Photometry sees the full field of view, but slit spectroscopy and
    aperture-measured line fluxes lose flux. The model parameter `eline_scaling`
    is the fractional aperture correction applied to the **slit** spectra and
    lines (1.0 = no loss, 0.65 = lines at 65%). Add a prior on it (below) to
    marginalise over the aperture mismatch when fitting photometry together with
    a slit spectrum/lines.

## 5. Priors and the model

Collect the observations into a single list — any subset is fine; an empty list
for a type you are not fitting. Then define priors for every free parameter.

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
    # Aperture correction for the slit spectrum + lines.
    "eline_scaling":     Uniform(low=0.1, high=2.0),
}

model = SedModel(
    csp,
    observations=observations,
    priors=priors,
    zred=ZRED,                       # fixed spectroscopic redshift
)
```

To fit redshift instead of fixing it, omit `zred` and add a `"zred"` prior; see
the free-redshift example in `scripts/`.

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
- **Aperture.** Use `eline_scaling` (and, for the spectrum, `calibration`/
  `noise_floor`) to absorb slit-vs-photometry aperture and flux-calibration
  differences.
- **Wavelengths are rest-frame vacuum Å** for the spectrum and line lists; the
  forward model handles the redshifting to the observed frame.
