# Noise and calibration

By default, the likelihood is Gaussian and uses the uncertainties supplied
with each observation. You can add error floors, rescale uncertainties, mark
upper limits or fit a spectral calibration. The [outlier mixture](outlier_model.md)
and [GP likelihood](gp_likelihood.md) are described separately. Section 5 of
[`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md)
lists unsupported combinations.

## Fixed terms, set on the observation

- `Spectrum(noise_floor=f)`: a fractional floor, sigma_eff² = sigma² + (f |model|)².
- `Photometry(upper_limit=mask)`: bands where `mask` is True get a one-sided penalty (only a
  model above the data is penalised).

## Sampled noise terms

To sample any of the following variance terms, add a prior and a
`free_param_init` entry in `SedModel`:

| parameter | variance becomes |
|---|---|
| `log_err_scale_<kind>` | sigma² exp(2 log_err_scale): a common rescaling of the quoted errors |
| `log_jitter_<kind>` | + exp(log_jitter)², in data units |
| `log_f_calib_<kind>` | + (exp(log_f_calib) \|model\|)² |
| `log_f_data_<kind>` | + (exp(log_f_data) \|data\|)² |

Replace `<kind>` with `phot`, `spec` or `lines`. For several observations of
the same type, use `<root>_<kind>_<obs.name>`, such as
`log_jitter_spec_nirspec`; the unsuffixed name is then ambiguous and raises
an error. Use a constant transform to fix a term. `CSPBasis(fixed=...)` takes only CSP parameters; these names belong to the observations, so they are fixed with a constant transform.

`fitSED` reads these names automatically. With `run_sampler`, construct the
`DiagonalNoiseModel` yourself. `log_jitter` is the natural log of a flux in
the observation's units: maggies, cgs F_nu or erg s⁻¹ cm⁻². Choose its prior
relative to the supplied uncertainties.

## Calibrating a spectrum

A spectrum and photometry may have different flux calibrations. Including
photometry constrains the absolute flux scale and helps separate continuum
calibration errors from dust and age. Choose one calibration method per spectrum:

- **Sampled calibration.** `spectrum_scaling` multiplies the entire `Spectrum`
  prediction. `spectrum_calib`, a vector of shape `(order,)`, adds a factor
  `1 + sum_k c_k P_k(x)`. The `P_k` are Legendre polynomials, with observed
  wavelengths mapped to [-1, 1] over all pixels. There is no P_0 term because
  `spectrum_scaling` sets the level. For several spectra, use
  `spectrum_scaling_<obs.name>` and `spectrum_calib_<obs.name>`.
- **Profiled polynomial.** `Spectrum(polynomial_order=M)` solves for Chebyshev
  coefficients c_0..c_M at each likelihood call, as in Prospector's `PolyOptCal`.
  The posterior is conditional on the best-fitting polynomial and does not
  include its uncertainty.
- **Marginalised polynomial.** `Spectrum(polynomial_order=M,
  polynomial_mode="marginalize", polynomial_prior_sigma=s)` integrates out the
  coefficients under a Gaussian prior. The width `s` is in fractional-response
  units. This propagates calibration uncertainty into the posterior and
  evidence without adding sampled parameters.

`eline_scaling` applies only to emission-line fluxes in a `Lines` observation.
It does not scale a `Spectrum` prediction.
