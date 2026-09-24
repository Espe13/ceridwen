# Noise and calibration

The default likelihood is Gaussian with the uncertainties you pass to each observation. This
page lists the terms that change it: error floors and inflated errors, upper limits, and the
calibration of a spectrum. The outlier mixture and the Gaussian-process likelihood have their
own pages ([Outlier model](outlier_model.md), [GP likelihood](gp_likelihood.md)).
[`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md) (section 5) lists
the combinations that are refused.

## Fixed terms, set on the observation

- `Spectrum(noise_floor=f)`: a fractional floor, sigma_eff² = sigma² + (f |model|)².
- `Photometry(upper_limit=mask)`: bands where `mask` is True get a one-sided penalty (only a
  model above the data is penalised).

## Sampled noise terms

Four terms inflate an observation's variance. Each is switched on by giving its parameter a
prior (and a `free_param_init`) in `SedModel`:

| parameter | variance becomes |
|---|---|
| `log_err_scale_<kind>` | sigma² exp(2 log_err_scale): a common rescaling of the quoted errors |
| `log_jitter_<kind>` | + exp(log_jitter)², in data units |
| `log_f_calib_<kind>` | + (exp(log_f_calib) \|model\|)² |
| `log_f_data_<kind>` | + (exp(log_f_data) \|data\|)² |

`<kind>` is `phot`, `spec` or `lines`. With several observations of one kind, each has its
own term, `<root>_<kind>_<obs.name>` (for example `log_jitter_spec_nirspec`); the plain name
then raises. A constant transform fixes a term at a value. These names are read by `fitSED`;
with `run_sampler` you build the `DiagonalNoiseModel` yourself. `log_jitter` is the natural
log of a flux in the observation's units (maggies, cgs F_nu or erg s⁻¹ cm⁻²), so set its prior
from the size of your uncertainties.

## Calibrating a spectrum

A spectrum's flux calibration rarely matches the photometry. Keep the photometry in the fit:
it anchors the absolute level, and without it the continuum shape is degenerate with dust and
age. Three ways to model the calibration, one per spectrum:

- **Sampled level and shape.** `spectrum_scaling` multiplies the whole `Spectrum` prediction.
  `spectrum_calib`, shape `(order,)`, multiplies it by `1 + sum_k c_k P_k(x)`, Legendre
  polynomials of the observed wavelength mapped onto [-1, 1] over all pixels (no P_0 term:
  the level is `spectrum_scaling`). With several spectra, use `spectrum_scaling_<obs.name>`
  and `spectrum_calib_<obs.name>`.
- **Profiled polynomial.** `Spectrum(polynomial_order=M)` solves for the Chebyshev coefficients
  c_0..c_M inside the likelihood at every step, as Prospector's `PolyOptCal` does. The
  posterior is conditional on the best-fit polynomial: its uncertainty is not propagated.
- **Marginalised polynomial.** `Spectrum(polynomial_order=M, polynomial_mode="marginalize",
  polynomial_prior_sigma=s)` integrates the coefficients out analytically under a Gaussian
  prior of width `s` (in units of the fractional response). The calibration uncertainty then
  reaches the posterior and the evidence, with no extra sampled dimension.

`eline_scaling` is a different parameter: it scales the model emission lines of a `Lines`
observation only, not the spectrum.
