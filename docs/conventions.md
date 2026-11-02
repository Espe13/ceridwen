# Conventions and common errors

This page describes the units, parameter definitions and model settings
needed to interpret a fit. `GOTCHAS.md` in the repository covers additional
usage errors.

## Metallicity

Use `theta["logzsol"]` for constant metallicity or `theta["logzsol_hist"]`
for one value per lookback-time node. Values are log10(Z/Z_sun), using
the loaded grid's Z_sun; 0.0 is solar. Absolute-metallicity keys `Z` and
`zh` are rejected with an error that shows the converted value.

Typical ranges are [−2.30, +0.30] for BPASS (Z_sun = 0.020) and
[−2.50, +0.50] for MIST/aMIST (Z_sun = 0.0185, or 0.0142 for grids built
with python-fsps <= 0.4.7). Inspect your grid with
`print(csp.zmet.min(), csp.zmet.max(), csp.zsun_nominal)`.

Z_sun is resolved from an explicit `zsun=`, the file's `log10_zsun`
provenance or the content-hash table in `ceridwen.ssps.grid_metadata`.
It cannot be inferred reliably from `isoc_type`, since FSPS versions use
different solar values under the same isochrone name. Loading fails if
Z_sun cannot be determined.

For MIST and aMIST, `logzsol` represents [Fe/H]. Total metallicity [Z/H]
is derived as `logzsol_total`; see `grid_metadata.logzsol_total`.
`gas_logz` uses the CLOUDY grid's solar reference. `CSPBasis(gas_tied=True)`
sets `gas_logz := logzsol`, matching the FSPS and Prospector convention
even though the stellar and gas solar references differ.

An absolute log10 Z, such as −1.85, can be mistaken for `logzsol`.
On BPASS this is within the allowed axis but means 0.014 Z_sun, so the
package warns rather than rejecting it. Bounded priors extending beyond
the grid raise an error. `Uniform(low=-2.0, high=0.2)` fits within all
published grids. `csp.check_param_ranges(theta)` reports ranges in
`logzsol` and identifies the grid's Z_sun.

## Lookback-time ordering

`lookback_time` increases from the present at index 0 to the oldest node.
The `sfh` array uses the same ordering. Decreasing grids, including
`lookback = T_univ - t_grid` applied to an increasing `t_grid`, raise
`ValueError` at construction.

## Units

| Quantity | Unit |
|---|---|
| Model wavelength grid (`csp.wave`) | Å, vacuum, rest frame |
| `Spectrum` pixel wavelengths (data) | Å, vacuum, **observed frame** (model is redshifted onto them) |
| `Lines` wavelengths / `mask_lines` centres | Å, vacuum, **rest frame** (`Lines`: redshifted internally; `Spectrum.mask_lines(..., zred=z)`: by the `zred` you pass, default 0) |
| `csp.get_spectrum(theta)` (model grid) | rest-frame L_ν in L☉ Hz⁻¹ per M☉ (no mass, distance or IGM) |
| `Spectrum` predictions (`model.predict`) | observed-frame F_ν in erg s⁻¹ cm⁻² Hz⁻¹ (cgs; × 1e32 for nJy), only with a redshift or `lumdist_mpc` in force |
| Broadband fluxes | AB maggies |
| Emission-line fluxes | erg s⁻¹ cm⁻², observed frame; with an IGM, its transmission averaged over the line's profile on the model grid |
| Marginalised line fluxes (`marginalize_elines=True`: `/elines` in the result, `extras["elines"]`) | erg s⁻¹ cm⁻², observed frame, IGM-transmitted at the line, **not** divided by the spectrum calibration; lines named as in `$SPS_HOME/data/emlines_info.dat`, vacuum rest-frame Å |
| Stellar mass | `logmass` = log10(M⋆/M☉) |

The spectrum from `theta['sfh']` is multiplied by `10**logmass`. SSP weights
sum to the SFH integral on the construction grid (`csp.sfh_times`, or a
prediction-time `theta['lookback_time']`). With `track_zred_age=True`, the
grid stretches to `age(zred)` and SFR is scaled by `T_grid / age(zred)`
to preserve the integral. When that integral is 1 M☉, `logmass` is the
formed mass. `logsfr_ratios_to_sfh(..., sfh_times_yr=csp.sfh_times)` provides
this normalisation for fixed or sampled redshift. A directly sampled
`sfh` has its own normalisation.

## Precision

Importing CERIDWEN enables `jax_enable_x64` on devices that support float64.
Let the package set this option. Apple GPUs instead use float32 throughout;
see [Apple GPU](mac_gpu.md).

On float64-capable devices, large spectral, curve and projection arrays use
float32 to reduce memory and computation costs. Float64 is retained for
calculations requiring greater range or precision, and for small arrays.
Comments in the code explain each float64 calculation.

| float32 | |
|---|---|
| SSP cube and its contraction | `csp.flux`, the `W`-weighted einsums |
| attenuation curves | evaluated on `csp._wave32` with float32 parameters |
| nebular line/continuum arrays, once interpolated | grid interpolation and `10**x` use float64 for log values near 50; interpolated arrays and line-profile contractions use float32 |
| the projector's work | resampling, FFT smoothing, the instrument response and the painted line profiles of a `Spectrum` prediction |
| photometry matrices and maggies | `Photometry._T`, `Photometry` predictions |
| projector scaling | divide by a power of two before float32 projection, then restore the scale in float64 |
| line apertures and painted line profiles | `Lines` aperture matrices; the `(n_pix, n_lines)` profiles the projector paints |
| `PostProcess` model-grid spectra | `prediction/spectra_model`, `spectra_observed`, ... (its observation predictions are stored float64) |

| float64 | reason |
|---|---|
| `DustEmission`'s trapezoid dots and `labs0 = lboln - lbold` | absorbed luminosity is a difference of nearly equal numbers, requiring greater precision |
| the AGN `L_bol` and energy balance (`csp.py`, `agn/blr.py`) | frequency integrals and accumulators |
| `sigma`, `sigma**2`, chi² terms; the noise model, profiled and marginalised calibration | dynamic range: at cgs scale `sigma**2` ~ 1e-60 underflows float32. The likelihood promotes the float32 prediction to the data's float64 |
| the line-marginal basis `A` and its normal equations | the likelihood's linear algebra, conditioned up to `COND_MAX = 1e10` |
| likelihood sums, sampler scalars, cosmology and distances, wavelength grids | accumulation over many terms; positions where 1e-7 relative is a measurable shift |
| line fluxes (`predict_line_fluxes`) and `Lines` predictions | small arrays covering roughly 1e-51 to 1e2 L_sun, including summed AGN line contributions |
| returned `Spectrum` prediction | float64 multiplication restores the projector's scale before the likelihood gradient is propagated into float32 |
| `DustEmission` output spectrum | converting it to float32 did not reduce measured float64 work (8.5% of FLOPs in `neb_duste`) and changed two golden baselines |

Before float32 projection, the spectrum is divided by a power of two
near its largest absolute value. The scale is restored in float64.
This preserves binary mantissas and scales the likelihood gradient
before it enters float32. At cgs flux scales, an unscaled gradient
`|r| / sigma^2` can reach 1e35–1e41, exceeding the float32 maximum
of 3.4e38 and producing NaN.

With dust emission enabled, the model-grid spectrum is float64.
Observations store data in float64, both at construction and when assigning
`obs.flux` or `obs.uncertainty`. Input arrays may use float32.

## SFH mass normalisation

`logsfr_ratios_to_sfh` normalises the trapezoidal SFR integral to 1 M☉
on the supplied `sfh_times_yr=csp.sfh_times` construction grid, including
when redshift is free. `logmass` then sets the mass. Use this transform
to avoid normalisation errors that can shift `logmass` by several dex.

## Unrecognised `theta` keys

The model ignores keys it does not read. A typo such as `logmas` or `dust2`
therefore leaves `logmass` or `diffuse_tau_kc` at its default. `predict()`
and `get_spectrum_components()` warn about unrecognised keys during tracing,
without adding work to the sampling loop. Model-level parameters such as
`logsfr_ratios` are registered automatically.

## Spectral broadening

Broadening is modelled with Gaussian widths in ln λ. Projection combines
the relevant widths at each pixel:

```
continuum:  σ_cont² = σ_gal² + σ_inst²(λ) − σ_lib²(λ)
lines:      σ_line² = σ_gas² + σ_inst²(λ)
```

Set the widths through the following objects:

| Width | What it is | Where it is set |
|---|---|---|
| `σ_gal`, `σ_gas` | the galaxy's stellar and gas velocity dispersions [km/s] | `Kinematics(...)`, passed to `SedModel(kinematics=...)`; the same object serves every observation |
| `σ_inst(λ)` | the instrument's line-spread function at each observed pixel | `Instrument.<unit>(...)`, passed to each `Spectrum(instrument=...)` |
| `σ_lib(λ)` | the resolution already in the SSP library | stored in every schema-2 grid (`SSPData.ssp_resolution`); read automatically |

The projection applies galaxy kinematics in the rest frame, then redshift,
IGM transmission and the instrumental response. A `Spectrum` is smoothed by
σ_gal before IGM attenuation, then by the instrument/library contribution.
Broadened photometry also applies IGM transmission after galaxy broadening.
The Ly-α break therefore retains the sharpness set by the IGM model and
instrument. When σ_gal is fixed at zero, IGM transmission is applied on
the model grid.

`csp.get_spectrum` returns a model-grid spectrum without galaxy broadening.
Emission lines are projected directly from their luminosities onto spectral
pixels with width σ_line. `Lines` observations use integrated nebular fluxes
and do not depend on these widths.

### The galaxy: `Kinematics`

```python
from ceridwen import Kinematics, DEFAULT_KINEMATICS

Kinematics(sigma_gal="sigma_gal")                     # σ_gal free (theta key), σ_gas tied to it
Kinematics(sigma_gal=250.0, sigma_gas="sigma_gas")    # stars fixed, gas free
Kinematics(sigma_gal=300.0)                           # both fixed at 300 km/s
DEFAULT_KINEMATICS                                    # == Kinematics(sigma_gal=300.0); what SedModel uses if you pass nothing
Kinematics.none()                                     # no kinematic broadening at all
```

A numeric width is fixed; a string refers to a `theta` parameter that needs
a prior. `sigma_gas` defaults to `TIED`, sharing the value or key of
`sigma_gal`. `Kinematics(...)` requires `sigma_gal` explicitly. Omitting
kinematics from `SedModel` uses the named default of 300 km/s, shown in
the model summary.

`sigma_max`, default 2000 km/s, bounds the compiled kernels. Priors extending
above it are rejected at setup; sampled values are clipped to it.
Providing a fixed `sigma_gal` and also placing that key in `theta` raises
`ValueError`. A named key missing from `theta` raises `KeyError`, and a
sampled key without a prior produces a warning.

### The instrument: `Instrument`

Instrument resolutions are published in several units. Resolving power also
has two conventions: R = λ/FWHM, commonly used in datasheets, and R = λ/σ,
used by sedpy/Prospector. They differ by 2√(2 ln 2) ≈ 2.35. Choose the
`Instrument` constructor that specifies both the unit and convention:

```python
from ceridwen import Instrument

Instrument.R_fwhm(2700)                    # R = λ/FWHM  (datasheet, e.g. NIRSpec gratings)
Instrument.R_sigma(6358)                   # R = λ/σ     (sedpy / Prospector "R"); the same LSF as R_fwhm(2700)
Instrument.fwhm_aa(2.5)                    # FWHM in Å, observed frame
Instrument.sigma_aa(1.06)                  # σ in Å, observed frame
Instrument.sigma_kms(47.0)                 # σ in km/s
Instrument.fwhm_kms(111.0)                 # FWHM in km/s
Instrument.R_fwhm(R_array, wave=obs_wave)  # any of the above per pixel, on observed wavelengths
```

Widths must be finite and positive. Width arrays require a matching `wave=`
grid covering all spectral pixels. The LSF is interpreted as measured on
the detector, including pixel width. Model profiles are therefore evaluated
at pixel centres without a second integration over the pixel.

#### A free LSF scale: `Instrument(..., scale=...)`

Every constructor accepts `scale=` to model uncertainty in the LSF. The
effective width is s · σ_inst(λ), applied to both continuum and lines:
`σ_cont² = σ_gal² + s² σ_inst² − σ_lib²` and
`σ_line² = σ_gas² + s² σ_inst²`.

```python
from ceridwen import Instrument
from ceridwen.priors import Uniform

Instrument.R_fwhm(2700)                        # scale=1.0: the nominal LSF, the unscaled code path
Instrument.R_fwhm(2700, scale=1.1)             # fixed: identical to Instrument.R_fwhm(2700 / 1.1)
Instrument.R_fwhm(2700, scale="lsf_scale")     # sampled: needs free_param_init and a BOUNDED prior
priors = {"lsf_scale": Uniform(low=0.8, high=1.3)}
Instrument.R_fwhm(2700, scale="lsf_scale", scale_range=(0.8, 1.3))  # range given explicitly
```

A fixed scale is incorporated into the response at setup. A sampled scale
requires response weights to be recomputed at each call over a band sized
for the maximum scale. This supports wavelength-dependent σ_inst and
derivatives with respect to scale.

The scale range comes from finite prior bounds, or from `scale_range=`
when the key is supplied by a transform. It must satisfy `0 < lo < hi`.
A prior exceeding an explicit range raises an error; sampled values outside
the range are clipped. The library-resolution warning uses the minimum
scale. Another warning identifies pixels where the continuum width crosses
half a log-grid pixel: switching from linear interpolation to Gaussian
response introduces a small step in the prediction. Photometry and `Lines`
do not use the instrument scale.

Prospector's `resolution_jitter_parameter`
(`prospect/models/sedmodel.py:289–295`) scales only the continuum resolution.
Line widths are cached from the unscaled resolution at line 283. It also
overwrites `obs.padded_resolution` at each call without validation.

### Library resolution

The SSP library's resolution is subtracted in quadrature from the instrumental
width at each pixel's rest-frame wavelength. Velocity widths are independent
of frame. Where the instrument is finer than the library, the subtraction
is limited to zero and the continuum stays at library resolution. A warning
reports the affected pixels and wavelength range; this can also indicate
incorrect instrument units.

Use `Spectrum(..., subtract_library=False)` if the grid's stored resolution
curve is unreliable. With `instrument=None`, no subtraction is made and
the continuum retains library resolution.

### Photometric broadening

By default, photometry integrates the galaxy-broadened spectrum
(`SedModel(broaden_photometry=True)`), following Prospector. The effect on
broad bands is below 5e-4 mag at 300 km/s and 5e-3 mag at 1000 km/s, even
with 1000 Å emission lines and sharp breaks. For medium or narrow bands,
a line near a filter edge can change the flux by about a percent.

For broad-band-only fits, `broaden_photometry=False` saves one FFT per
likelihood call. It integrates the continuum at model-grid resolution and
adds the emission lines as delta lines, independent of σ_gas.
Use the same setting to generate and fit mocks.

The filter integral is exact for a spectrum linear between the model pixels
and a filter linear between its own points (`FilterSet.flux_weights`), so a
feature narrower than the filter grid keeps its flux. For a fixed redshift it
is the matrix `T`; for a sampled redshift it is evaluated on each filter's
window of model pixels.

Emission lines are not painted for photometry. Band predictions are
`T · continuum_broadened + E(σ_gas) · F_line`, with `F_line` the observed line
fluxes (dust, escape and the profile-averaged IGM applied, as for a `Spectrum`)
and `E` each filter's response to each line: `R(λ) λ / ab` at the line for a
delta line (`broaden_photometry=False`), the exact integral of a Gaussian of
σ_gas against the filter with `broaden_photometry=True`. At a fixed redshift
and σ_gas, `E` is one exact matrix; at a sampled redshift or σ_gas it is
interpolated from a table of the exact integral. Set
`csp._force_paint_lines = True` to paint the lines on the model grid instead
(a reference path).

## Cosmology

Pass a cosmology explicitly to `CSPBasis` or `CSPBasis_afe`. The CSP uses
it for the flux factor `(1+z) (10 pc / D_L)^2`
(`CSPBasis._flux_factor`) and for the universe age that sets the SFH grid
when `track_zred_age=True` (`CSPBasis._lookback_from_zred`).

All observation types follow sampled `zred`. Photometry is projected
through filters at each sample; `Lines` uses the corresponding flux
factor, Jacobian and IGM transmission. Spectra use a redshift-aware
projector covering the prior's support. Provide `Spectrum(zred_range=)`
if that range is not specified by bounded prior support.
```python
from ceridwen import Cosmology

cosmo = Cosmology.planck18()                # Planck 2018 VI, the usual choice
cosmo = Cosmology.planck15()
cosmo = Cosmology.wmap9()                   # what prospector hard-codes
cosmo = Cosmology.flat(H0=70.0, Om0=0.3)    # the two numbers a paper quotes
cosmo = Cosmology.from_name("Planck15")     # from a string (config file)
cosmo = Cosmology.from_astropy(astropy_cosmo)   # any flat astropy cosmology

csp = CSPBasis(ssp, lookback_time=..., cosmo=cosmo, ...)
```

`model.cosmo` is a read-only view of the CSP cosmology. `SedModel(cosmo=...)`
is accepted only if it equals `csp.cosmo`, and `csp.cosmo` cannot be
reassigned after construction. The cosmology appears in `repr(csp)`,
`model.summary()` and the `fitSED` log.

Result files store it as `/model` attributes `cosmo_H0`, `cosmo_Om0`,
`cosmo_Tcmb0`, `cosmo_Neff`, `cosmo_m_nu_ev_sum` and `cosmo_name`.
Reconstruct it with `ceridwen.result_cosmology(path)` or
`Cosmology.from_dict(attrs)`.

`cosmo.age(z)` and `csp.age_at(z)` return universe age in Gyr. The former
always returns a JAX array; the latter returns a Python float for scalar
input. `cosmo.luminosity_distance(z)` returns Mpc. Build an SFH grid from
the cosmology, for example `jnp.linspace(0.0, cosmo.age(z), n)`; there is
no `tuniv` argument. At fixed redshift, `SedModel` rejects an oldest node
more than 0.5% older than `csp.age_at(zred)`. This check is skipped if
`track_zred_age=True` rescales the grid or a `lookback_time` transform
replaces it. Age tracking also applies to fixed non-zero redshift,
which `SedModel` supplies to the CSP.

At `zred=0` without `lumdist_mpc`, there is no flux factor. The CSP scales
only when `theta` contains `zred`, and `SedModel` does not insert zero.
Predictions are then `L_sun/Hz × 10^logmass`. For a nearby object, supply
the distance, for example
`SedModel(csp, obs, priors, zred=0.0, lumdist_mpc=3.5)`. This uses the
given distance in place of D_L(zred), retaining the redshift's (1+z)
factor, as in Prospector's `lumdist`.

`model.summary()` reports the distance convention. `SedModel` warns when
observations have `zred=0` without flux scaling, or when `lumdist_mpc`
overrides the distance at a non-zero redshift. CSP constructors also
reject unknown keyword arguments.

`SedModel` builds observation projections at construction. After replacing
`model.observations`, call `model.setup_observations()`; passing observations
to `fitSED(model, observations)` does this automatically. A `Spectrum`
requires `wavelength=` at construction, even if flux is supplied later.

## Nebular grids

`CSPBasis` reads the Byler et al. (2017) CLOUDY grids from
`$SPS_HOME/nebular`. The default `ZAU_ND_<isochrones>` grids omit dust
inside H II regions, matching FSPS, python-fsps and Prospector.
Choose `init_neb_params={"cloudy_dust": True}` for `ZAU_WD_<isochrones>`.

The effect depends on the grid. For constant SFR over 0–1 Gyr, solar
metallicity and logU = −2.5, WD/ND flux ratios on MIST are 0.78 for Hα,
0.79 for Hβ, 0.17 for Ly-α and 1.13 for [O III] 5007. BPASS ratios are
0.95, 0.95, 0.40 and 0.99 respectively. The result file stores the choice
in `csp_config['cloudy_dust']`. Older files without this setting used
`cloudy_dust=True`; `rebuild_model` requires a matching CSP.

## Dust geometry and the escape fraction `frac_obrun`

Each SSP age row receives attenuation from its age bin, then diffuse dust
attenuation. Set bin edges with `init_dust_params['bin_edges']` in log10
age/Gyr. The default birth-cloud bin covers ages below 10^-1.97 Gyr,
or 10.7 Myr. `theta['frac_obrun']` specifies the fraction escaping some
of this attenuation, with geometry set by `fesc_geometry` on `CSPBasis`:

- **`'runaway_bc'` (default):** a fraction `frac_obrun` from every age row
  bypasses its age-bin dust but still receives diffuse attenuation.
  Nebular-age populations release that fraction of their ionising photons
  without age-bin attenuation, leaving `1 - frac_obrun` to power nebular
  emission. With the default bin, only ages below 10.7 Myr are affected.
  With several attenuated bins, older stars can also bypass their bin's
  dust. For an old-only population with two power-law bins, `tau_pow1=1`,
  `tau_pow2=0.5` and no SFR below 2.6 Gyr, the 5500 Å flux increases by
  factors of 1.19 at `frac_obrun=0.3` and 1.65 at `frac_obrun=1`.
- **`'picket'`:** a fraction `frac_obrun` of young light, including ionising
  photons, bypasses all dust and gas. It produces no nebular emission and
  adds no absorbed energy to dust emission. The remaining fraction powers
  nebular emission. This requires `add_neb=True`. Young ages are those in
  the CLOUDY grid, up to log age = 7.3 (20 Myr) in the published grids.
  Thus light at 11–20 Myr has no default birth-cloud attenuation, but its
  escaping fraction still bypasses diffuse dust.

In either geometry, `frac_obrun=0` reproduces the model without that key.

## FSPS at runtime

Nebular and dust-emission models read data from `$SPS_HOME` at runtime.
Install the FSPS data and set this path before using `add_neb=True` or
`add_dust_emission=True`; see [Installation](installation.md).
