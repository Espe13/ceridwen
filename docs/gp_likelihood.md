# Gaussian-process likelihood for correlated spectral residuals

Resampling, combining dithers and sky-subtraction errors can leave correlated
residuals across neighbouring spectral pixels. A diagonal likelihood treats
these pixels as independent and can underestimate posterior uncertainty.
The GP likelihood models a squared-exponential covariance in wavelength.
Its amplitude and length scale can be fixed or sampled with nested sampling,
NUTS, VI-preconditioned NUTS or `map_fit`.

The GP is disabled by default. Without GP parameters, the spectrum uses the
ordinary diagonal Gaussian likelihood and the compiled program is unchanged.

## The likelihood

Residuals are divided by the effective diagonal uncertainty σ_eff,i, including
`noise_floor`, `log_err_scale_spec`, `log_jitter_spec`, `log_f_calib_spec`
and `log_f_data_spec`. Thus r_i = (y_i − μ_i)/σ_eff,i, where y is the
sky-subtracted flux and μ includes any model calibration. For n unmasked pixels:

```
K_ij = δ_ij + a² exp(−(λ_i − λ_j)² / 2ℓ²) + ε δ_ij
ln L = −½ rᵀ K⁻¹ r − ½ ln|K| − Σ_i ½ ln(2π σ_eff,i²)
```

- λ is the observed-frame wavelength in Å (`Spectrum.wavelength`).
- a is dimensionless and measured relative to σ_eff. At a = 1, the correlated
  and white-noise variances are equal at each pixel.
- ℓ is the length scale in observed-frame Å.
- ε is a fixed diagonal jitter, default 1e-6 (`GP_JITTER`, or
  `GaussianProcess(jitter=)`).

The identity term accounts for white noise. As a → 0, the log-likelihood
approaches the diagonal Gaussian plus
`½ Σ r_i² ε/(1+ε) − ½ n ln(1+ε)`, about 5e-7 per pixel. The mask is fixed
at setup. Masked pixels receive identity rows and columns in K and r = 0,
so their contribution is zero.

`ceridwen.likelihood.lnlike_gp_gaussian` uses dense float64 Cholesky
factorisation and supports derivatives in a, ℓ and all model parameters.
`tests/likelihood/test_gp_likelihood.py` compares it with the independent
NumPy implementation `GaussianProcess.log_likelihood` at rtol 1e-10,
checks gradients with `jax.test_util.check_grads` and compares `jit`/`vmap`
results with a Python loop.

## Using it with `fitSED`

Each spectrum has two parameters, both expressed as natural logarithms:

| parameter | meaning | unit |
|---|---|---|
| `log_gp_amp_spec` | ln a | a in units of σ_eff |
| `log_gp_length_spec` | ln ℓ | ℓ in observed-frame Å |

For several spectra, use `log_gp_amp_spec_<obs.name>` and
`log_gp_length_spec_<obs.name>`. Unsuffixed names require exactly one
spectrum. A name that matches no spectrum, such as `log_gp_amp_phot`, raises
an error.

**Sampled parameters.** Give each a prior and a `free_param_init` entry:

```python
from ceridwen.priors import Uniform

model = SedModel(csp, [phot, spec],
                 priors={**priors,
                         "log_gp_amp_spec": Uniform(low=-3.0, high=1.5),     # a in [0.05, 4.5]
                         "log_gp_length_spec": Uniform(low=1.0, high=5.5)},  # l in [2.7, 245] A
                 free_param_init={**init, "log_gp_amp_spec": jnp.array([0.0]),
                                  "log_gp_length_spec": jnp.array([3.0])},
                 transforms=transforms, zred=zred)
result = fitSED(model, output_dir="out")
```

The `fitSED` log names the GP:
`spec: GPGaussianLikelihood, outlier mixture off, GP (squared exponential, 600 pixels)
a = exp(theta['log_gp_amp_spec']) (sampled), l = exp(theta['log_gp_length_spec']) (sampled)`.

**Fixed parameters.** Use constant transforms, such as
`transforms={"log_gp_amp_spec": lambda th: jnp.array([0.0]), "log_gp_length_spec":
lambda th: jnp.array([np.log(30.0)])}`, or pass a `GaussianProcess` to the spectrum:

```python
from ceridwen.observation import GaussianProcess, Spectrum

spec = Spectrum(wavelength=w, flux=f, uncertainty=e, noise=GaussianProcess(1.0, 30.0))
```

`GaussianProcess(amplitude, length_scale)` takes a and ℓ directly, without
logarithms. Its `jitter` argument sets ε.

Supply both named parameters together. For a given spectrum, choose either
the `GaussianProcess` object or the named parameters; providing both raises
an error. Named parameters may be sampled or fixed through transforms.

Bounded priors are useful, though the package does not require them. A length
scale below the pixel spacing acts as another white-noise term and is
degenerate with `log_err_scale_spec`. A very long length scale acts as a
smooth offset and is degenerate with calibration.

`examples/recipes/gp_likelihood.py` fits a mock spectrum with sampled GP
parameters and compares it with a diagonal-Gaussian fit.

## What it can be combined with

| combined with | status |
|---|---|
| the noise terms `noise_floor`, `log_err_scale_spec`, `log_jitter_spec`, `log_f_calib_spec`, `log_f_data_spec` | **yes**: they set σ_eff before the whitening |
| `sky`, a fixed `calibration` vector | **yes** |
| the sampled calibration `spectrum_scaling` / `spectrum_calib` | **yes**: it acts on the model before the residuals are formed (not tested together yet) |
| other observations (photometry, lines, other spectra) | **yes**: each keeps its own likelihood; the GP is per spectrum |
| a free redshift, a free LSF scale, emission lines painted from the grid | **yes** in principle: they change the model, the GP acts on the residuals (not tested together yet) |
| the outlier mixture `f_outlier_spec` on the same spectrum | **refused** at setup |
| upper limits on the spectrum | **refused** at setup |
| `marginalize_elines=True` on the same spectrum | **refused** (at construction for `noise=`, at setup for the names) |
| `logify_spectrum=True` | **refused** (not available in any sampled likelihood) |
| the profiled calibration polynomial (`polynomial_order > 0`) | **refused** at setup |

These combinations are unsupported by the implementation. The outlier and
upper-limit terms operate per pixel, while the line-marginalisation and
profiled-polynomial solvers assume independent pixels. Supporting a full
covariance would require changes to those calculations.

## After the fit

- The spectrum's `likelihood_json` records the kernel, `log_amp`, `log_len`,
  `eps`, `n_pix` and sampled names. Read it with
  `read_result_h5(path)["obs"][name]["likelihood"]["gp"]`. Amplitude and length
  are stored as theta keys when sampled and fixed log values otherwise.
  Sampled GP parameters also appear in `/samples`.
- For fixed `noise=GaussianProcess(...)` parameters,
  `Spectrum.log_likelihood(model_flux)` returns the same value as the compiled
  likelihood. `Spectrum.chi_sq` always returns diagonal χ².
- The summary figure reports diagonal χ²/ν using the quoted uncertainties,
  labelled "diagonal, no GP" for spectra fitted with a GP.
- `GPGaussianLikelihood.conditional_mean(y, mu, sigma, mask, theta)`, also
  available as `ceridwen.likelihood.gp_conditional_mean`, estimates the
  correlated residual in data units. Use it to plot model + GP against the
  data. It is not computed during sampling.

## Cost

Dense Cholesky factorisation costs O(n³) per likelihood call. Each batch lane
needs an n × n float64 covariance matrix (8n² bytes). The CPU measurements
below use `jit(vmap)` over W parameter vectors, exclude compilation and
time only the GP kernel, without the forward model:

| n_pix | W | value [ms] | value + grad [ms] | value + grad per lane [ms] | diagonal value + grad [ms] | K per lane [MB] |
|---|---|---|---|---|---|---|
| 200 | 1 | 0.22 | 1.20 | 1.20 | 0.013 | 0.3 |
| 200 | 32 | 2.88 | 20.1 | 0.63 | 0.026 | 0.3 |
| 500 | 1 | 1.93 | 6.50 | 6.50 | 0.014 | 2.0 |
| 500 | 32 | 28.0 | 161 | 5.0 | 0.044 | 2.0 |
| 1000 | 1 | 4.74 | 31.0 | 31.0 | 0.020 | 8.0 |
| 1000 | 32 | 115 | 844 | 26.4 | 0.153 | 8.0 |
| 2000 | 1 | 26.5 | 215 | 215 | 0.023 | 32.0 |
| 2000 | 32 | 716 | 5835 | 182 | 0.686 | 32.0 |

These measurements used an Apple M3 Pro CPU in float64 while another
CPU-intensive job was running. The gradient calculation took 5–8 times as
long as the value alone. Doubling the pixel count from 1000 to 2000 increased
the cost by about sevenfold, consistent with O(n³) scaling.

At 1000 pixels, the GP adds about 26 ms per lane for value and gradient.
For a NUTS run with roughly 2.5e5 gradient calls (4 chains × 2000 draws ×
about 32 leapfrog steps), this is about two CPU hours; at 2000 pixels it is
about thirteen hours. GPU timings have not yet been measured.

`fitSED` warns above `GP_WARN_NPIX=1000` pixels. For longer spectra, consider
fitting a wavelength window or binning the data.
