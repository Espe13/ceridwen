# Tutorial: fitting photometry, spectroscopy and lines jointly

CERIDWEN treats the three observation types uniformly: broadband **photometry**,
a resolved **spectrum**, and **emission-line** fluxes. Each is a small container
that knows how to project the model spectrum onto its own data space, and the
joint likelihood is the sum of their χ² contributions. You can fit any one of
them, or all three together, with no change to the model or sampler.

This tutorial builds one of each and fits them jointly. Replace the mock arrays
with your own data.

!!! tip "Runnable scripts"
    The runnable counterparts of this tutorial are
    `examples/demo_2_photometry_lines.py` (photometry + lines) and
    `examples/demo_3_spectrum_advanced.py` (photometry + spectrum with an
    `Instrument`, a fitted `sigma_gal`, line masking and a noise floor).

!!! note "Before you start"
    Read **[Conventions & gotchas](conventions.md)**, especially that `Z` is
    `logzsol = log10(Z/Z_sun)` of the loaded grid and that `lookback_time` index 0 is *today*.
    Make sure FSPS and `$SPS_HOME` are set up ([Installation](installation.md));
    emission lines and nebular continuum need the CLOUDY grids from FSPS.

## 1. Build the forward model

Lines and nebular continuum require `add_neb=True`. We also enable diffuse and
birth-cloud dust here.

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
lookback = jnp.linspace(0.0, 8.0, 5)        # 5 nodes -> 4 free logsfr_ratios
csp = CSPBasis(
    ssp,
    lookback_time=lookback,
    cosmo=cosmo,
    zh_const=True, sfh_interp="step",       # one "logzsol"; zh_const=False samples "logzsol_hist"
    add_dust=True, add_diffuse_dust=True,   # birth-cloud + diffuse attenuation
    add_neb=True,                           # nebular continuum + lines (needs $SPS_HOME)
    add_igm=True,                           # Madau (1995), auto-scales with zred
    # sps_home defaults to $SPS_HOME
)
```

The galaxy's velocity dispersions are a property of the galaxy, not of any one
observation, so they are set once, on the model (section 5), through a
`Kinematics` object. Here we fit the stellar dispersion and tie the gas
dispersion of the emission lines to it:

```python
kin = Kinematics(sigma_gal="sigma_gal")     # free (theta key); sigma_gas tied to sigma_gal
# Kinematics(sigma_gal=250.0, sigma_gas="sigma_gas")   stars fixed, gas free
# Kinematics(sigma_gal=300.0)                          both fixed (the SedModel default)
```

For photometry this choice has a cost: with a *sampled* `sigma_gas` (here tied
to the sampled `sigma_gal`) the emission lines are painted onto the model grid
and broadened at run time for every likelihood call; with a *fixed* `sigma_gas`
(`Kinematics(sigma_gal="sigma_gal", sigma_gas=150.0)`, or both fixed) they enter
the photometry through a static line-to-band basis, which is cheaper. Free
redshift always uses the painted path. See [Conventions](conventions.md).

## 2. (a) Photometry

Broadband fluxes in **AB maggies**, with filter names resolved from the
filter library bundled with CERIDWEN (`python -c "from
ceridwen.observation.filters import list_available_filters as f; print(f())"`
lists all 293). Photometry captures the full aperture, so it sees the
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
`zred = 0` observed and rest frame coincide) and flux in observed-frame `F_ν`
in erg s⁻¹ cm⁻² Hz⁻¹ (cgs, the model's unit; 1 nJy = 1e-32).
The instrument's line-spread function is attached to the spectrum as an
`Instrument`. Its unit **and** its convention are the name of the constructor,
because published resolutions come in several units and in two resolving-power
conventions that differ by 2.35x (datasheets quote `R = λ/FWHM`, the
sedpy/Prospector tradition `R = λ/σ`); there is deliberately no plain `R`.
The SSP library's own resolution (stored in every schema-2 grid) is subtracted
in quadrature from the instrumental width automatically, pixel by pixel; you do
not need to set anything for that. The galaxy's own dispersion is **not** set
here: it comes from the model-level `Kinematics` (section 1) and is combined
with the instrument in quadrature at projection time, so nothing is broadened
twice.

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

Other optional knobs: `calibration=` (per-pixel multiplicative flux-calibration
vector) and `mask=`. If the instrument is *finer* than the library at some
pixels the continuum stays at library resolution there (the right model for a
coarse grid) and you are warned with the pixel count and range; a warning over
the whole spectrum with a high-resolution grid means a wrong unit on the
`Instrument`.

## 4. (c) Emission lines

Integrated line fluxes. `line_ind` are indices into FSPS's
`emlines_info.dat`; `wavelength` are the vacuum rest wavelengths in Å. The
forward model matches each line to the nebular grid **by rest wavelength**
(and raises if no grid line lies within 1 Å), so the wavelengths are what must
be right — the indices are bookkeeping, checked against the wavelength match
and warned about on disagreement.

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

The model prediction for a `Lines` observation is the line luminosity read
from the nebular grid and carried through dust, mass, distance and IGM like the
continuum (the IGM transmission averaged over the line's profile on the model grid,
which matters at Ly-α, where the Madau forest starts inside the line); no other
width enters, so `Kinematics` and `Instrument` play no role here. A basis built with `add_neb=False` (and `CSPBasis_afe`,
which has no nebular model) refuses a `Lines` observation with a `ValueError`
rather than predicting zeros for it.

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

!!! tip "Wavelength-dependent calibration: `spectrum_calib`"
    A grey factor cannot absorb a *shape* error in the spectrophotometric
    calibration (relative throughput, differential refraction, aperture colour
    terms; typical for slit spectra flux-calibrated against broad-band
    photometry). `spectrum_calib` is a vector of Legendre coefficients
    `c_1 .. c_order` multiplying the `Spectrum` prediction by
    `1 + sum_k c_k P_k(x)`, with `x` the observed pixel wavelength mapped
    affinely onto [-1, 1] (the Prospector `polyorder` / pPXF `mdegree`
    idea, but sampled, not solved analytically). There is no `c_0`: the
    level is `spectrum_scaling`, and the total factor is
    `spectrum_scaling * (1 + sum_k c_k P_k(x))`. Photometry is untouched, so
    keep photometry in the fit: it is what pins the continuum shape while the
    polynomial soaks up the spectrum-vs-photometry mismatch. Orders 2-6 are
    typical; higher orders start to eat real features (the 4000 Å break, wide
    molecular bands), so watch the recovered curve. Give it one prior, which is
    broadcast over the vector exactly like `logsfr_ratios`:

    ```python
    priors["spectrum_calib"] = Uniform(low=-0.2, high=0.2)
    free_param_init["spectrum_calib"] = jnp.zeros(4)      # order 4
    ```

    Absent from `theta`, the factor is exactly 1 (existing fits are unchanged).
    Implementation: `ceridwen/csp/spectrum_calibration.py`, applied in
    `_project_observations` of both `CSPBasis` and `CSPBasis_afe`.

    **Several spectra (v1.0.7).** Each spectrum has its own calibration:
    `spectrum_scaling_<obs.name>` / `spectrum_calib_<obs.name>`. The plain names are
    accepted only when the model has exactly one `Spectrum`; with two they are
    ambiguous and `SedModel` raises (before v1.0.7 one value silently scaled both).

!!! tip "Profiled calibration polynomial: `Spectrum(polynomial_order=M)`"
    Instead of sampling the calibration, let the likelihood solve it:
    `Spectrum(..., polynomial_order=3)` multiplies the model by
    `1 + sum_{m=0}^{3} c_m T_m(x)` (Chebyshev, `x` over the unmasked wavelength range)
    with the coefficients that maximise the likelihood at every call, a weighted linear
    least-squares solve inside the jitted likelihood (Prospector's `PolyOptCal`). It adds
    no sampled dimensions and is differentiable. The weights are the noise model's
    `1/sigma_eff^2` (Prospector uses the raw `1/sigma^2`; the two agree unless noise terms
    are on). `T_0` is the level, so `spectrum_scaling` / `spectrum_calib` of the same
    spectrum are refused. It is a *profile*, not a marginalisation: the posterior is
    conditional on the best polynomial. `polynomial_regularization=` adds a ridge term.
    `PostProcess` predictions include each draw's response
    (`out["prediction"]["calibration"][name]`).

!!! tip "Marginalised calibration polynomial: `polynomial_mode=\"marginalize\"`"
    The likelihood is linear in the polynomial coefficients, so under a Gaussian prior
    they can be integrated out in closed form instead of being profiled or sampled:

    ```python
    spec = Spectrum(..., polynomial_order=6, polynomial_mode="marginalize",
                    polynomial_prior_sigma=0.1)     # float, or one width per T_0..T_6
    ```

    With the prior `c_m ~ N(0, s_m^2)` and the response `1 + sum_m c_m T_m(x)` (the same
    Chebyshev basis as the profiled mode), the likelihood is the exact marginal

    `ln L = ln L_diag(y - mu) + 1/2 b^T M^-1 b - 1/2 ln|M| - 1/2 ln|Lambda|`,

    `M = D^T W D + Lambda^-1`, `b = D^T W (y - mu)`, `D = diag(mu) A`, `Lambda = diag(s^2)`:
    O(n k^2) per call, no sampled dimension, and the calibration uncertainty is
    propagated into the posterior and the evidence (the profiled mode conditions on the
    best polynomial). Units of `s`: the fractional response, so `s = 0.1` allows ~10 %
    calibration errors per term. `s_m = 0` pins a coefficient at 0; `s = inf` (a flat prior)
    is refused, because the marginal likelihood would be undefined.

    - `T_0` is a grey scale with prior width `s_0`: degenerate with the mass unless the
      photometry (or a fixed flux scale) anchors the level. To sample the level instead,
      pin `T_0` (`polynomial_prior_sigma=[0.0, s_1, ..., s_M]`) and sample
      `spectrum_scaling`; any other sampled calibration of that spectrum is refused.
    - `polynomial_regularization` belongs to the profiled mode and is refused here: the
      prior width is its analogue (`reg_m = 1/s_m` gives the same best polynomial, which is
      the conditional mean below).
    - Refused (they make the likelihood non-Gaussian or non-linear in `c`): the outlier
      mixture on this spectrum, upper limits, `logify_spectrum=True`, a Gaussian-process
      `noise`, and `marginalize_elines=True` on the same spectrum (the polynomial also
      scales the lines, so the model is bilinear in the two sets of coefficients).
    - The noise weights `W` are the noise model's `1/sigma_eff^2` at the uncalibrated
      model, as in the profiled mode (exact when no noise term depends on the model).
    - `PostProcess` applies each draw's conditional-mean response
      (`out["prediction"]["calibration"][name]`, included in `spectra[name]`) and returns
      the coefficients in `out["extras"]["calibration"][name]`: `mean`, `sd`, `cov` per draw
      and `draws` (one draw from each conditional Gaussian: together a sample of the
      marginal posterior of `c`). The result file records mode, order and prior widths
      (`read_result_h5(path)["obs"][name]["likelihood"]["poly_calibration"]`).

    **Cost.** Per call the marginal costs the same as the profile (both O(n k^2)); the sampled
    route is cheaper per call but adds M + 1 dimensions to the sampler. Measured on CPU
    (Apple, shared machine, +-20 %; `scripts/bench_poly_marginal.py`), jitted, float64, a batch
    of W draws under `vmap`, microseconds per batch:

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

    The forward model dominates the log-posterior: the three modes cost the same there at
    every order, so a high order costs nothing per call in the profiled or marginalised mode,
    while in the sampled mode it costs sampler dimensions. GPU timing at the production width
    W = 100 is not measured yet.

    Derivation: `docs/dev/poly_marginalisation_design.md`; code:
    `ceridwen/likelihood/poly_marginal.py`.

## 5. Priors and the model

Collect the observations into a single list. Any subset is fine; use an empty
list for a type you are not fitting. Then define priors for every free parameter.

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

N_RATIOS = 4   # number of SFH nodes - 1
model = SedModel(
    csp,
    observations=observations,
    priors=priors,
    transforms={"sfh": logsfr_to_sfh},        # REQUIRED for logsfr_ratios
    free_param_init={"logsfr_ratios": jnp.zeros(N_RATIOS),
                     "logmass": jnp.array([10.0]),
                     "sigma_gal": jnp.array([150.0]),
                     "eline_scaling": jnp.array([1.0]),      # sampled nuisances that are
                     "spectrum_scaling": jnp.array([1.0])},  # not CSP parameters need a start value
    zred=ZRED,                                # fixed spectroscopic redshift
    kinematics=kin,                           # galaxy dispersions (default: Kinematics(sigma_gal=300.0))
    broaden_photometry=True,                  # photometry sees the sigma_gal-broadened spectrum (default)
)
```

`SedModel` checks the `Kinematics` against `theta` and the priors at setup:
a free key that is missing from `theta`, a fixed width that also appears in
`theta`, or a bounded prior reaching above `sigma_max` all raise before
anything is compiled (a free key without a prior gets the usual
"no prior for sampled parameter" warning). Leaving `kinematics=` out uses `DEFAULT_KINEMATICS`,
`Kinematics(sigma_gal=300.0)` for stars and gas, which the model summary
prints so that it is never a hidden number.

To fit redshift instead of fixing it, add `"zred"` to `free_param_init` with a
bounded prior (`Uniform` / `ClippedNormal`) and, for a non-parametric SFH,
build the CSP with `track_zred_age=True` so the SFH age-bin grid tracks the
sampled redshift: the grid is stretched so its oldest node is `age(zred)`, and the SFR
is scaled by `T_grid / age(zred)` so the SFH still forms the mass it forms on the
construction grid. Keep the transform above,
`logsfr_ratios_to_sfh(..., sfh_times_yr=csp.sfh_times)`: `logmass` is then the formed mass
at every sampled redshift, and `PostProcess` reports each draw's SFR on its own
`age(zred)` grid with `mass_formed = 10**logmass`. Every observation type follows the sampled value: the flux
factor, the IGM and the line fluxes are evaluated at `theta["zred"]`;
`Photometry` is projected through the filters per sample; a `Spectrum` gets
a redshift-aware projector whose log-wavelength window covers the prior's
support: the model is read at the sampled redshift on every call and the
lines are painted at `lambda_rest (1 + z)` (pass `Spectrum(zred_range=(z_min,
z_max))` when the prior has no finite bounds or `zred` comes from a
transform). The log grid is the fixed-z grid of the reference redshift (the
`zred` start value) extended over the range, so at that redshift the free-z
projection equals the fixed-z one exactly; elsewhere the model is read at a
different node phase (per-mille level for a MILES-resolution grid), and the
library width in the fixed kernel stays the one at the reference redshift
(`build` warns when it would change by more than 10 %). The gradient with
respect to `zred` is that of the linear interpolation: exact for the current
node configuration, piecewise constant on the scale of one log-grid node
(1e-8 in z), which NUTS never resolves. Keep the prior as tight as the data
allow; the window must fit on the model grid for the whole range.

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

`fitSED(model, observations)` replaces `model.observations` and re-runs
`model.setup_observations()`; if you assign `model.observations = [...]`
yourself, call `model.setup_observations()` before predicting or fitting.

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

`result.samples` holds the raw posterior keyed by parameter name; the HDF5 file in
`output_dir` stores the observations, priors, samples, and (for nested sampling)
the log-evidence, and `PostProcess(model, "joint_fit/ceridwen_result.h5")`
reloads it. See [Post-processing](postprocessing.md) for the output layout and
`examples/demo_3_spectrum_advanced.py` for a runnable photometry + spectrum fit.

## Consistency checklist for real joint fits

- **Flux systems must agree.** Photometry (maggies), spectrum (`F_ν` in cgs, erg s⁻¹ cm⁻² Hz⁻¹) and line
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
- **One width per source.** The galaxy's `sigma_gal`/`sigma_gas` are set once
  in `Kinematics`; each `Spectrum` carries only its `Instrument`. If a fitted
  `sigma_gal` comes out at the edge of its prior, check the `Instrument` unit
  first (a datasheet `R` passed as `R_sigma`, or the reverse, is a factor 2.35
  in width).
- **Know your frames.** The spectrum's pixel grid is **observed-frame** vacuum
  Å (the model is redshifted onto it); line-list wavelengths and
  `mask_lines(..., zred=z)` centres are **rest-frame** vacuum Å (redshifted by the
  `(1 + zred)` you pass; without `zred` it warns and uses 0).
