# Emission-line marginalisation

Emission-line marginalisation fits the stellar spectrum while integrating
over the line fluxes analytically. The fluxes can vary from the CLOUDY
predictions, leaving the underlying stellar absorption available to constrain
the population. This can retain information from features such as the Balmer
series, Ca II H+K and Mg b that would be removed by `Spectrum.mask_lines`.

```python
import jax.numpy as jnp
from ceridwen import CSPBasis, SedModel, Kinematics, Instrument
from ceridwen.observation import Spectrum

spec = Spectrum(wavelength=w_obs, flux=f, uncertainty=e,
                instrument=Instrument.R_fwhm(1000.0),
                marginalize_elines=True)               # flat prior on every covered line

csp = CSPBasis(ssp, lookback_time=lb, zh_const=True, add_neb=True, cosmo=cosmo,
               fixed={"gas_logu": -2.5, "gas_logz": -0.3})   # nebular model fixed
model = SedModel(csp, [spec, phot], priors=priors, zred=3.2,
                 transforms={"sfh": sfh_transform},
                 kinematics=Kinematics(sigma_gal=150.0, sigma_gas=80.0))
print(model.summary())                                 # lists the marginalised lines
```

Constant transforms on the `SedModel` fix them the same way:
`transforms={"sfh": sfh_transform, "gas_logu": lambda th: jnp.array([-2.5]),
"gas_logz": lambda th: jnp.array([-0.3])}`.

## What is computed

Line fluxes α are observed-frame, IGM-transmitted fluxes in erg s⁻¹ cm⁻².
They enter the observations linearly:

- The marginalising `Spectrum` uses calibrated line profiles with width
  √(σ_gas² + σ_inst²).
- Each `Photometry` observation uses a static line-to-band matrix.
- Each `Lines` observation uses its blend matrix.

With independent Gaussian noise, integration over α has a closed-form solution.
Each likelihood call requires a weighted least-squares solve and Cholesky
factorisation of an n_line × n_line matrix. The supported priors are flat
(`eline_prior_width=0`, the default) or Gaussian about the CLOUDY flux.
For the latter, `eline_prior_width=0.2` means a fractional width of 20%,
following Prospector's convention. Ly-α always uses a flat prior because
resonant scattering makes its grid flux unreliable.

## Options

| `Spectrum(...)` argument | Meaning |
|---|---|
| `marginalize_elines` | enable marginalisation; default `False` keeps CLOUDY line fluxes |
| `eline_prior_width` | 0 = flat; > 0 = Gaussian, fractional width about the CLOUDY flux |
| `elines_to_fit` | FSPS names (`$SPS_HOME/data/emlines_info.dat`, e.g. `"[O III] 5007"`); default: every line the spectrum covers |
| `elines_to_fix` | lines kept at their CLOUDY flux |
| `elines_to_ignore` | lines removed from every observation |

`theta["eline_delta_zred"]`, supplied as a sampled parameter or transform,
shifts line centres to (1 + zred + eline_delta_zred) λ_rest, as in Prospector.

For a line to be fitted, its centre must stay more than 3σ inside the spectrum
over the projector's full redshift range, with at least three unmasked pixels
within 2σ. Lines that fail these conditions keep their CLOUDY fluxes and are
listed by `model.summary()`. Explicitly requesting such a line in
`elines_to_fit` raises an error.

## Requirements

Unsupported configurations raise an error before sampling.

- With `add_neb=True`, fix `gas_logu` and `gas_logz` (`CSPBasis(fixed=...)` or
  constant transforms) or tie them to other parameters. They cannot be independently sampled when
  the line fluxes are free.
- Supply an `Instrument` for the spectrum so the line width includes the LSF.
- Use at most one marginalising spectrum per model.
- The joint marginalisation does not support `logify_spectrum`, a
  [GP likelihood](gp_likelihood.md), upper limits or an
  [outlier mixture](outlier_model.md). An observation outside that system,
  such as a second spectrum, can use a separate noise model.
- Photometry fitted jointly takes the marginalised lines through its own line
  term (each line's flux through each filter), also with a sampled `zred` or a
  sampled `sigma_gas` with `broaden_photometry=True`; the photometric columns
  are then evaluated at each call's redshift and gas width.
- Set the gas velocity dispersion in km/s with `Kinematics(sigma_gas=...)`.
  There is no `eline_sigma` parameter.

## Without a nebular model (`add_neb=False`, `CSPBasis_afe`)

A flat line-flux prior does not require a nebular grid. Line names and rest
wavelengths come from `$SPS_HOME/data/emlines_info.dat`; widths remain
√(σ_gas² + σ_inst²) and are independent of metallicity and [α/Fe].
For `CSPBasis(..., add_neb=False)` and `CSPBasis_afe`:

- Covered lines use the default flat prior. `eline_prior_width > 0` and
  `elines_to_fix` require CLOUDY fluxes and raise an error.
- The model has no nebular continuum or grid lines, so there are no fixed lines.
- Set `$SPS_HOME`, or `sps_home=` on the basis, to locate `emlines_info.dat`.
- Photometry is marginalised jointly. `Lines` observations are unsupported
  because they require grid fluxes.
- Output `cloudy` entries are NaN.

## Choosing the prior

**Flat prior (`eline_prior_width=0`).** Line strengths are independent of the
stellar parameters. This is useful when emission may come from an AGN,
shocks or evolved stars, including in quiescent galaxies.

**Gaussian prior about CLOUDY (`eline_prior_width=0.2`).** Each fitted line
has prior N(F_j(θ), (0.2 F_j(θ))²). Both its centre and width follow the
star-forming nebular model, so the lines affect the inferred stellar
population. Stronger-than-predicted emission can favour more recent star
formation. When F_j approaches zero, the prior width also approaches zero,
restricting the line flux even if emission is present in the data.

Use this prior when the lines arise from star formation and the CLOUDY grid
describes them to roughly 20%. In an injection test with fluxes scaled by
0.5–3 relative to CLOUDY, this prior shifted recovered fluxes towards CLOUDY
by up to 3σ. In 2 of 50 realisations, the optimiser chose an SFH with no
recent star formation.

## Evidence

A flat prior has an undefined normalisation for each fitted line. Its ln Z
cannot be compared across different line sets, masked fits or ordinary fits.
Posterior inference is unaffected. Comparisons require the same fitted lines
and prior convention, or a proper prior (`eline_prior_width > 0`).

## Speed

The line profiles and design matrices can be computed once at setup when
`zred`, `sigma_gas` and instrument scale are fixed, with no `eline_delta_zred`,
sampled calibration (`spectrum_scaling*`, `spectrum_calib*`, `eline_scaling`),
sampled noise terms or `noise_floor`. For a flat prior, the factorisation can
also be cached. On an A100 with batch width 100, the cached flat-prior
likelihood took 0.99–1.00 times the ordinary likelihood's runtime; a 20% prior
took 1.11 times. Recomputing per call took about 1.2–1.25 times. When caching
is unavailable, `model._eline_system.static` is `None`.

Without caching, each call builds a dense `(n_pix, n_line)` Gaussian matrix.
In float64, profiles are exactly zero beyond about 38.6σ. An optional
windowed calculation restricts each line to ±40 of its largest possible σ.
The window uses `Kinematics.sigma_max` for sampled `sigma_gas`, the maximum
instrument scale and the line-centre range over `zred_range`.

Enable it after constructing the model and before evaluating the likelihood:
`model.obs_dict["spec"]._proj.enable_line_windows()`. Rebuilding the projector,
including through `fitSED(observations=...)`, disables it again. CPU values
are bitwise identical to the dense calculation; gradients differ by 1–2 ulp
from summation order. GPU fusion may change the last bits of the log-likelihood.

The option is disabled by default. CPU timings were 0.99–1.14 times the dense
calculation for 400- and 4000-pixel spectra at batch widths 64 and 100, with
matrix filling dominating the cost. On an A100, timings were 0.77–0.96 times
the dense calculation for a 4000-pixel grating-like spectrum and 0.97–0.99
times for a 400-pixel prism-like spectrum. Sampled `eline_delta_zred` always
uses the dense calculation. Sampled redshifts outside `zred_range` truncate
the windowed profiles.

## Noise terms

The solve uses the likelihood's inverse variances. Marginalisation remains
exact with `log_err_scale_*`, `log_jitter_*` and `log_f_data_*`.
Model-dependent terms (`noise_floor` and `log_f_calib_*`) are evaluated on
the model with the fitted lines removed, so they do not depend on the
unknown line amplitudes.

## Outputs

- `fitSED` saves `/elines` in `ceridwen_result.h5`, containing line
  `names`, `wave_rest`, and each draw's posterior `mean`, `sd` and
  `cloudy` prediction. Read these with `read_result_h5(path)["elines"]`.
- `PostProcess` returns the same values in `out["extras"]["elines"]`.
  Its predicted spectra, photometry and line fluxes use each draw's
  posterior-mean line fluxes.
- `ceridwen.likelihood.eline_marginal.eline_line_fluxes(model, theta,
  likelihood)` returns the line posterior for one parameter vector.

`model.predict(theta)` still returns the forward model's CLOUDY lines.
Use the outputs above for the marginalised line posterior.

## Masking versus marginalising

Masking excludes line pixels and any stellar absorption in them.
Marginalisation keeps those pixels in the fit and assigns emission to a
Gaussian with the specified position and width. When emission overlaps
absorption of a similar width, the two are partly degenerate. The resulting
line-flux uncertainties and stellar posterior are wider.

## Differences from Prospector

With a Gaussian prior, the algebra agrees with Prospector 2.0a2.dev42 to
1e-10 on identical inputs. The implementations differ in these respects:

- CERIDWEN's flat-prior marginal term is +½(n ln 2π + ln|Σ̂|).
  Prospector uses the opposite sign for its no-prior term.
- CERIDWEN uses unit-area analytic profiles and the exact speed of light
  on all unmasked pixels. Prospector numerically renormalises with c = 3e18
  within 5σ windows, causing jumps when pixels enter a window.
- CERIDWEN marginalises photometry and `Lines` jointly with the spectrum.
  Prospector substitutes fitted spectral fluxes into photometry, provided
  the spectrum is listed first.
- CERIDWEN converts amplitudes with the current calibration. Prospector
  uses the calibration from the previous likelihood call.
