# Tutorial: fitting photometry, spectroscopy and lines jointly

CERIDWEN can fit broadband photometry, resolved spectra and emission-line
fluxes separately or together. Each observation object projects the forward
model into the appropriate data space. The joint likelihood combines the
individual observation likelihoods.

This tutorial constructs one observation of each type and fits them jointly.
Replace the mock arrays with your data when using it for a real fit.

!!! tip "Example scripts"
    `examples/demo_2_photometry_lines.py` fits photometry and line fluxes.
    `examples/demo_3_spectrum_advanced.py` fits photometry and a spectrum
    with an `Instrument`, sampled `sigma_gal`, line masking and a noise floor.

!!! note "Requirements"
    See [Conventions and common errors](conventions.md) for metallicity
    (`logzsol = log10(Z/Z_sun)` for the loaded grid) and lookback-time
    ordering (index 0 is today). Set up FSPS and `$SPS_HOME` as described
    in [Installation](installation.md). Nebular continuum and lines require
    the FSPS CLOUDY grids.

## 1. Build the forward model

Set `add_neb=True` to include lines and nebular continuum. This example also
enables diffuse and birth-cloud dust attenuation.

```python
import jax, jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, fitSED, Kinematics, Instrument, Cosmology
from ceridwen.observation import Photometry, Spectrum, Lines
from ceridwen.priors import Uniform, ClippedNormal, StudentT
from ceridwen.model import logsfr_ratios_to_sfh

ssp = SSPData.load("ssp_data.h5")          # built once via SSPData.from_fsps(...)
# The grid records its isochrone library; CSPBasis reads it automatically,
# so isoc_type never needs to be passed in init_neb_params.

ZRED = 0.5                                  # spectroscopic redshift of the galaxy

# The cosmology is a property of the analysis: choose it once, here, and every
# distance and age in the fit comes from it.  Presets: Cosmology.planck18(),
# planck15(), wmap9() (prospector's); Cosmology.flat(H0, Om0) for the two
# numbers a paper quotes; Cosmology.from_astropy(...) for any flat astropy
# cosmology.  CSPBasis requires it; SedModel reads it from the CSP.
cosmo = Cosmology.planck18()

# lookback_time is the static SFH node grid (Gyr, increasing, index 0 = today,
# >= 2 nodes). Its oldest node must not exceed the age of the universe at ZRED,
# cosmo.age(0.5) = 8.6 Gyr under Planck18; SedModel refuses a grid that does.
# CSPBasis refuses to build without a grid (or an explicit theta).
lookback = jnp.linspace(0.0, 8.0, 6)        # 6 nodes -> 5 step bins -> 4 logsfr_ratios
csp = CSPBasis(
    ssp,
    lookback_time=lookback,
    cosmo=cosmo,
    zh_const=True, sfh_interp="step",       # one "logzsol"; zh_const=False samples "logzsol_bins"
                                            # (one per SFH bin)
    add_dust=True, add_diffuse_dust=True,   # birth-cloud + diffuse attenuation
    add_neb=True,                           # nebular continuum + lines (needs $SPS_HOME)
    add_igm=True,                           # Madau (1995), auto-scales with zred
    # sps_home defaults to $SPS_HOME
)
```

Set the galaxy's velocity dispersions on the model through `Kinematics`,
which is shared by all observations. Here the stellar dispersion is sampled
and the gas dispersion is tied to it:

```python
kin = Kinematics(sigma_gal="sigma_gal")     # free (theta key); sigma_gas tied to sigma_gal
# Kinematics(sigma_gal=250.0, sigma_gas="sigma_gas")   stars fixed, gas free
# Kinematics(sigma_gal=300.0)                          both fixed (the SedModel default)
```

With sampled `sigma_gas`, emission lines are placed on the model grid and
broadened at each likelihood call. This also applies when the gas width
is tied to sampled `sigma_gal`, as here. Fixed gas width, for example
`Kinematics(sigma_gal="sigma_gal", sigma_gas=150.0)`, allows photometry
to use a faster static line-to-band matrix. Sampled redshift requires
line profiles to be recomputed. See [Conventions](conventions.md).

## 2. Photometry

Supply broadband fluxes in AB maggies. Filter names refer to CERIDWEN's
bundled library of 296 filters; list them with
`python -c "from ceridwen.observation.filters import list_available_filters as f; print(f())"`.

To use your own curves, write each as `<name>.par` (two columns: wavelength in
Å, transmission) and either pass the directory, or set it once for every
session:

```python
phot = Photometry(filters=["my_nb816", "jwst_f444w"], filter_directory="~/filters", ...)
```

```bash
export CERIDWEN_FILTER_DIR=~/filters
```

The lookup order is: `filter_directory=` alone when it is given (no
fallback); otherwise `$CERIDWEN_FILTER_DIR`, then the bundled library. A name
found in both `$CERIDWEN_FILTER_DIR` and the library is read from
`$CERIDWEN_FILTER_DIR`, with a `UserWarning` naming both files. The result file
records the `.par` file of each filter (`/obs/<name>` attribute `filterfiles`).
The photometric prediction includes continuum and lines without the aperture
or spectral calibration factors discussed below.

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

Optional boolean arrays are `mask=` (True includes a band) and `upper_limit=`
(True marks a non-detection for the one-sided χ² penalty).

## 3. Spectrum

Supply observed-frame vacuum wavelengths in Å and observed-frame F_ν in
erg s⁻¹ cm⁻² Hz⁻¹. One nJy is 1e-32 in these units. The model is redshifted
onto the supplied pixels; at `zred=0`, rest and observed frames coincide.

Attach the line-spread function using `Instrument`. Its constructor specifies
both the unit and resolving-power convention: R = λ/FWHM in most datasheets,
or R = λ/σ in sedpy/Prospector. These differ by a factor of about 2.35,
so there is no constructor taking an unspecified R.

For schema-2 grids, library resolution is subtracted in quadrature from
the instrumental width at each pixel. Galaxy velocity dispersion comes
from the model's `Kinematics` and is combined with the instrument during
projection.

```python
spec = Spectrum(
    wavelength=my_obs_wave_aa,      # Å, vacuum, OBSERVED frame, shape (n_pix,)
    flux=my_spec_fnu,               # F_nu per pixel, erg s^-1 cm^-2 Hz^-1 (cgs)
    uncertainty=my_spec_unc,
    instrument=Instrument.sigma_kms(120.0),   # LSF: sigma in km/s
    # Instrument.R_fwhm(2700)                 # datasheet R = lambda/FWHM
    # Instrument.R_sigma(6358)                # sedpy / Prospector R = lambda/sigma (= R_fwhm(2700))
    # Instrument.fwhm_aa(2.5)                 # FWHM in Angstrom, observed frame
    # Instrument.R_fwhm(R_arr, wave=my_obs_wave_aa)   # per-pixel curve (prism)
    # Instrument.R_fwhm(2700, scale="lsf_scale")      # LSF width x a free scale (bounded
    #                                   prior + free_param_init; docs/conventions.md)
    # subtract_library=True (default): the SSP library resolution is removed
    # in quadrature; False only for a grid whose stored curve you distrust.
    noise_floor=0.01,               # 1% multiplicative calibration floor (optional)
    name="spec",
)

# Optional: mask known emission lines from the *continuum* spectrum fit so they
# do not double-count against the Lines object, redshifting line centres first.
spec.mask_lines([4862.76, 5008.31, 6564.72], dv=500.0, zred=ZRED)   # vacuum rest
```

You can also pass a fixed per-pixel multiplicative `calibration=` vector
and `mask=`. Where the instrument is finer than the SSP library, the
continuum remains at library resolution. A warning reports the affected
pixel count and range. If this affects the whole spectrum despite a
high-resolution grid, check the `Instrument` units.

## 4. Emission lines

Supply integrated line fluxes. `line_ind` contains indices into the FSPS
`emlines_info.dat` list; `wavelength` contains vacuum rest wavelengths in Å.
The model matches nebular lines by wavelength within 1 Å and raises an error
if no match exists. Indices are checked against that match, with a warning
on disagreement.

```python
# Catalogue line fluxes are often quoted in 1e-20 erg s^-1 cm^-2; scale to
# absolute CGS to match the model (adjust the factor to your catalogue).
LINE_UNIT = 1.0e-20

lines = Lines(
    line_ind=[59, 61, 62, 74, 75],                         # Hβ, [OIII]4959/5007, Hα, [NII]6584
    line_names=["Hbeta", "[OIII]4959", "[OIII]5007", "Halpha", "[NII]6583"],
    wavelength=[4862.76, 4960.37, 5008.31, 6564.72, 6585.37],  # Å, vacuum rest
    flux=np.asarray(my_line_flux) * LINE_UNIT,             # erg s^-1 cm^-2
    uncertainty=np.asarray(my_line_unc) * LINE_UNIT,
    name="lines",
)
```

A `Lines` prediction starts from the nebular grid luminosity and applies
dust attenuation, mass and distance scaling, and IGM transmission. IGM
transmission is averaged over the model-grid profile, which matters for
Ly-α because the Madau forest begins within the line. Integrated fluxes
do not depend on `Kinematics` or `Instrument`. Models with `add_neb=False`,
including `CSPBasis_afe`, reject `Lines` observations with `ValueError`.

!!! tip "Flux calibration: `eline_scaling` and `spectrum_scaling`"
    Aperture losses or calibration errors may put line fluxes and spectra
    on a different scale from the photometry. Two parameters handle this:

    - `eline_scaling` multiplies only `Lines` predictions. A value of 0.65
      gives 65% of the predicted line flux. It does not affect a `Spectrum`.
    - `spectrum_scaling` multiplies the entire `Spectrum` prediction,
      including lines, relative to the photometric flux scale. This follows
      Prospector's `spec_norm` convention.

    Neither changes photometry. Add a prior for each factor you want to fit.

!!! tip "Wavelength-dependent calibration: `spectrum_calib`"
    A single scale factor cannot describe wavelength-dependent calibration
    errors from throughput, differential refraction or aperture colour terms.
    `spectrum_calib` supplies Legendre coefficients `c_1 .. c_order` for
    the multiplicative response `1 + sum_k c_k P_k(x)`. Here x maps all
    observed pixel wavelengths to [-1, 1]. There is no c_0 term because
    `spectrum_scaling` sets the level. The combined response is
    `spectrum_scaling * (1 + sum_k c_k P_k(x))`.

    This resembles Prospector's `polyorder` and pPXF's `mdegree`, with
    coefficients sampled rather than solved analytically. Including
    photometry helps constrain the continuum shape. Orders 2–6 are typical;
    higher orders may fit real features such as the 4000 Å break or broad
    molecular bands. Inspect the fitted response. A single prior is
    broadcast over the coefficient vector:

    ```python
    priors["spectrum_calib"] = Uniform(low=-0.2, high=0.2)
    free_param_init["spectrum_calib"] = jnp.zeros(4)      # order 4
    ```

    Without `spectrum_calib` in `theta`, its factor is one. The implementation
    is in `ceridwen/csp/spectrum_calibration.py`, applied by
    `_project_observations` in both `CSPBasis` and `CSPBasis_afe`.

    For several spectra, use `spectrum_scaling_<obs.name>` and
    `spectrum_calib_<obs.name>`. Unsuffixed names are accepted only when
    there is exactly one spectrum; otherwise `SedModel` raises an error.

!!! tip "Profiled polynomial: `Spectrum(polynomial_order=M)`"
    `Spectrum(..., polynomial_order=3)` multiplies the model by
    `1 + sum_{m=0}^{3} c_m T_m(x)`, using Chebyshev polynomials over the
    unmasked wavelength range. A weighted least-squares solve finds the
    maximum-likelihood coefficients at every call, as in Prospector's
    `PolyOptCal`. The calculation is differentiable and adds no sampled
    parameters.

    Weights are `1/sigma_eff^2`, including noise terms. Prospector uses
    the quoted `1/sigma^2`, so results agree when no noise terms are active.
    T_0 sets the level; `spectrum_scaling` and `spectrum_calib` for that
    spectrum are therefore rejected. `polynomial_regularization=` adds a
    ridge term. The posterior is conditional on the fitted polynomial and
    omits its uncertainty. `PostProcess` returns each draw's response in
    `out["prediction"]["calibration"][name]` and applies it to predictions.

!!! tip "Marginalised calibration polynomial"
    The model depends linearly on polynomial coefficients. With Gaussian
    priors, these can be integrated out analytically:

    ```python
    spec = Spectrum(..., polynomial_order=6, polynomial_mode="marginalize",
                    polynomial_prior_sigma=0.1)     # float, or one width per T_0..T_6
    ```

    Using the same Chebyshev basis, response `1 + sum_m c_m T_m(x)` and
    priors `c_m ~ N(0, s_m^2)`, the marginal likelihood is

    `ln L = ln L_diag(y - mu) + 1/2 b^T M^-1 b - 1/2 ln|M| - 1/2 ln|Lambda|`,

    `M = D^T W D + Lambda^-1`, `b = D^T W (y - mu)`, `D = diag(mu) A`, `Lambda = diag(s^2)`.

    The calculation costs O(n k^2) per call and adds no sampled parameters.
    Calibration uncertainty is propagated into the posterior and evidence.
    The prior width s is in fractional-response units: 0.1 allows roughly
    10% calibration errors per term. Set `s_m=0` to fix a coefficient at
    zero. An infinite width is rejected because it leaves the marginal
    likelihood undefined.

    - T_0 sets the overall scale and is degenerate with mass unless
      photometry or a fixed calibration constrains the level. To sample
      `spectrum_scaling` instead, fix T_0 with
      `polynomial_prior_sigma=[0.0, s_1, ..., s_M]`. Other sampled
      calibration terms for that spectrum are rejected.
    - `polynomial_regularization` is available only in profiled mode.
      Here the prior width supplies the constraint; `reg_m = 1/s_m`
      gives the same best-fitting polynomial as the conditional mean.
    - The same spectrum cannot use an outlier mixture, upper limits,
      `logify_spectrum=True`, GP noise or `marginalize_elines=True`.
      These invalidate the assumed Gaussian linear model; jointly
      scaling line and polynomial coefficients makes it bilinear.
    - W uses the noise model's `1/sigma_eff^2` at the uncalibrated model,
      as in profiled mode. This is exact when the variance has no
      model-dependent terms.
    - `PostProcess` returns the conditional-mean response in
      `out["prediction"]["calibration"][name]` and applies it to `spectra[name]`.
      `out["extras"]["calibration"][name]` contains per-draw `mean`, `sd`,
      `cov` and `draws`. The last contains one draw from each conditional
      Gaussian, together sampling the marginal coefficient posterior.
      The result file records mode, order and prior widths in
      `read_result_h5(path)["obs"][name]["likelihood"]["poly_calibration"]`.

    Profiling and marginalisation both cost O(n k^2). Sampling is cheaper
    per likelihood call but adds M + 1 parameters. The following CPU
    measurements used float64 and `jit(vmap)` over W draws on a shared
    Apple machine, with about ±20% timing variation
    (`scripts/bench_poly_marginal.py`). Units are microseconds per batch:

    | | W | M | profile | marginalize | sampled |
    |---|---|---|---|---|---|
    | spectrum likelihood only (2000 px), value + grad | 32 | 1 | 916 | 905 | 177 |
    | | 32 | 3 | 900 | 943 | 294 |
    | | 32 | 6 | 2249 | 2298 | 288 |
    | | 32 | 10 | 3521 | 3878 | 297 |
    | | 32 | 20 | 5006 | 6328 | 356 |
    | full log-posterior (forward model, 1000 px), value + grad | 1 | 3 | 2372 | 2173 | 2591 |
    | | 1 | 20 | 2262 | 2616 | 2160 |
    | | 32 | 1 | 65090 | 65246 | 64855 |
    | | 32 | 3 | 67722 | 66507 | 69782 |
    | | 32 | 6 | 71060 | 70132 | 70526 |
    | | 32 | 10 | 70523 | 70375 | 81709 |
    | | 32 | 20 | 66412 | 65540 | 65974 |

    In the full log-posterior, the forward model dominated runtime and
    differences between calibration methods were within timing variation
    at every tested order. Sampling still adds dimensions as the order
    increases. GPU timings at production batch width W = 100 have not
    been measured. Code is in `ceridwen/likelihood/poly_marginal.py`.

## 5. Priors and the model

Collect the observations into one list. Include only the types you are
fitting, using empty lists for the others, and give each free parameter a prior.

```python
observations = [phot, spec, lines]

priors = {
    # Stellar population. logzsol = log10(Z/Z_sun); MIST grids span [-2.50, +0.50].
    # Birth-cloud dust ("tau_pow", "alpha_pow" for the powerlaw law) and every
    # other registered parameter needs a prior: print csp.param_names.
    "logzsol":           ClippedNormal(mean=-0.3, sigma=0.5, low=-2.0, high=0.2),
    "logmass":           Uniform(low=7.0, high=12.5),
    "logsfr_ratios":     StudentT(df=2.0, mean=0.0, scale=0.3),   # non-parametric SFH
    # Dust.
    "diffuse_tau_kc":    ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
    "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
    "tau_pow":           ClippedNormal(mean=0.3, sigma=0.5, low=0.0, high=4.0),
    "alpha_pow":         ClippedNormal(mean=-1.0, sigma=0.5, low=-2.5, high=0.5),
    # Nebular (required for the Lines / nebular continuum).
    "gas_logz":          Uniform(low=-1.3, high=0.2),   # inside the CLOUDY axis
    "gas_logu":          Uniform(low=-4.0, high=-1.0),
    # Emission-line aperture correction (Lines observation only).
    "eline_scaling":     Uniform(low=0.1, high=2.0),
    # Spectrophotometric normalisation of the spectrum onto the photometry
    # (Spectrum observation only; independent of eline_scaling). Omit if the
    # spectrum is already flux-calibrated to the photometric system.
    "spectrum_scaling":         ClippedNormal(mean=1.0, sigma=0.3, low=0.2, high=3.0),
    # Stellar velocity dispersion [km/s], the free key named in Kinematics above.
    # The upper bound must stay below Kinematics.sigma_max (2000 by default).
    "sigma_gal":         Uniform(low=20.0, high=600.0),
}

# The non-parametric SFH is sampled as logsfr_ratios and turned into the per-bin
# sfh by a REGISTERED transform; this step is required for logsfr_ratios to work.
sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

N_RATIOS = 4   # step SFH: number of bins - 1 = number of nodes - 2
model = SedModel(
    csp,
    observations=observations,
    priors=priors,
    transforms={"sfh": logsfr_to_sfh},        # REQUIRED for logsfr_ratios
    free_param_init={"logsfr_ratios": jnp.zeros(N_RATIOS),   # logmass needs only its prior
                     "sigma_gal": jnp.array([150.0]),
                     "eline_scaling": jnp.array([1.0]),      # sampled nuisances that are
                     "spectrum_scaling": jnp.array([1.0])},  # not CSP parameters need a start value
    zred=ZRED,                                # fixed spectroscopic redshift
    kinematics=kin,                           # galaxy dispersions (default: Kinematics(sigma_gal=300.0))
    broaden_photometry=True,                  # photometry sees the sigma_gal-broadened spectrum (default)
)
```

`SedModel` validates `Kinematics` before compilation. It rejects missing
free keys, fixed widths also present in `theta`, and bounded priors above
`sigma_max`. A sampled key without a prior produces a warning.
Omitting `kinematics=` uses `DEFAULT_KINEMATICS`, or
`Kinematics(sigma_gal=300.0)` for both stars and gas. The summary reports
these values.

To hold a CSP parameter fixed instead of fitting it, set it when you build
the CSP and give it no prior:

```python
csp = CSPBasis(ssp, lookback_time=lb, zh_const=True, add_neb=True, cosmo=cosmo,
               fixed={"gas_logu": -2.5, "duste_qpah": 2.0})   # duste_*: add_dust_emission=True
```

A fixed name is no longer a parameter: it is absent from `csp.param_names`
and `model.param_names`, and its value is in `csp.theta_init` and
`csp.fixed`. A name this CSP does not have (`gas_logu` with `add_neb=False`),
a value of the wrong shape, or a prior on a fixed name raises. Model-level
names (`logmass`, `zred`, `eline_scaling`, `spectrum_scaling`, the noise terms)
are fixed with a constant transform on the `SedModel`,
`transforms={"eline_scaling": lambda th: jnp.array([1.0])}`, which also works
for CSP parameters and gives the same model.

To fit redshift, add `zred` to `free_param_init` with a bounded prior such
as `Uniform` or `ClippedNormal`. For a non-parametric SFH, construct the
CSP with `track_zred_age=True`. It stretches the SFH grid to `age(zred)`
and scales SFR by `T_grid / age(zred)` to preserve mass. Keep the transform
`logsfr_ratios_to_sfh(..., sfh_times_yr=csp.sfh_times)`; `logmass` then
represents formed mass at every redshift. `PostProcess` reports each draw's
SFR on its own age grid with `mass_formed = 10**logmass`.

All observations follow sampled redshift. Flux scaling, IGM transmission
and line fluxes use `theta["zred"]`, and photometry is projected through
filters at that redshift. The spectral projector covers the full prior
range, reads the model at each redshift and places lines at
`lambda_rest (1 + z)`. Set `Spectrum(zred_range=(z_min, z_max))` if the
prior lacks finite bounds or a transform supplies redshift.

The projector extends the fixed-redshift log grid around the initial
`zred` value. At that value, free- and fixed-redshift projections agree
exactly. Elsewhere, interpolation occurs at a different grid phase,
causing differences at roughly the per-mille level for a MILES-resolution
grid. The library width in the fixed kernel remains that of the initial
redshift; `build` warns if it would vary by more than 10%.

The redshift derivative follows the linear interpolation and is piecewise
constant between log-grid nodes, on a scale of about 1e-8 in z, below the
scale NUTS resolves. Use a prior as narrow as the data justify. The
projector's window must fit inside the model grid throughout the range.

## 6. Fit

`fitSED` constructs a joint likelihood from `model.observations`, with one
Gaussian likelihood per observation name, and saves an HDF5 result.

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

`fitSED(model, observations)` replaces the model's observations and calls
`model.setup_observations()`. If you assign `model.observations` directly,
call setup yourself before prediction or fitting.

## 7. Posterior and diagnostics

```python
from ceridwen import PostProcess

pp  = PostProcess(model, result, n_samples=2000)   # equal-weight draws, nested weights recomputed
out = pp.run()
print("median logmass:", float(np.median(out["theta"]["logmass"])))

out["prediction"]["photometry"]["phot"]      # posterior-predictive maggies (draws x bands)
out["prediction"]["spectra"][spec.name]      # posterior-predictive spectrum (draws x pixels)
out["prediction"]["lines"][lines.name]       # posterior-predictive line fluxes
out["bestfit"]["theta"]                      # the maximum-likelihood sample
out["extras"]["sfh"]["sfr10"]                # derived quantities

# Summary (SED + chi, line residuals, SFH, marginals), corner and sampling
# diagnostics for this galaxy:
pp.figures("./joint_fit/figures", title="joint fit")
pp.save("./joint_fit/post.npz")
```

`result.samples` contains raw samples keyed by parameter name. The HDF5
file stores observations, priors, samples and nested-sampling evidence.
Pass the file to `PostProcess(model, "joint_fit/ceridwen_result.h5")` to
reload it. See [Post-processing](postprocessing.md) for its fields and
`examples/demo_3_spectrum_advanced.py` for a photometry-and-spectrum script.

## Checks before fitting data

- Put photometry in maggies, spectra in cgs F_ν and line fluxes in
  erg s⁻¹ cm⁻². Their calibrations must correspond to the same model
  normalisation at `zred`. Inconsistent calibrations can give acceptable
  separate fits but a poor joint fit.
- When fitting line fluxes measured from the same spectrum, mask those
  lines in the spectrum with `spec.mask_lines(...)` to avoid counting
  their measurements twice.
- Use `eline_scaling` for `Lines` aperture losses and `spectrum_scaling`
  for the spectral calibration relative to photometry. Fit whichever
  factors are needed. A fixed per-pixel `calibration` vector can correct
  known calibration differences; `noise_floor` accounts for additional
  uncertainty.
- Set galaxy dispersions through `Kinematics` and instrumental widths
  through each spectrum's `Instrument`. If `sigma_gal` reaches a prior
  boundary, check whether resolving power was supplied as R_FWHM or R_sigma;
  confusing them changes the width by a factor of 2.35.
- Spectral pixels use observed-frame vacuum Å. Line-list wavelengths and
  `mask_lines(..., zred=z)` centres use rest-frame vacuum Å. `mask_lines`
  shifts centres by the supplied (1 + zred); omitting redshift gives a
  warning and uses zero.
