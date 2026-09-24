# Gaussian-process likelihood for correlated spectral residuals

Neighbouring pixels of a reduced spectrum are often not independent: resampling onto a
common grid, combining dithers, or imperfect sky subtraction leave residuals that are
correlated over a few to tens of pixels. A diagonal likelihood then counts each
correlated stretch as many independent measurements, and the posterior comes out too
narrow. The GP likelihood gives the spectrum a squared-exponential covariance in
wavelength. It is part of the compiled likelihood that nested sampling, NUTS, VI and
`map_fit` use, and its two hyperparameters can be fixed or sampled.

!!! note "Off by default."
    A spectrum has a GP only when you switch one on (below). Without one, the likelihood
    is the diagonal Gaussian, compiled to the same program as a model without a GP.

## The likelihood

The residuals are first whitened by the diagonal noise model. σ_eff,i is the
uncertainty after **every** noise term (`noise_floor`, `log_err_scale_spec`,
`log_jitter_spec`, `log_f_calib_spec`, `log_f_data_spec`):
r_i = (y_i − μ_i)/σ_eff,i, with y the sky-subtracted data and μ the model (times any
calibration). Over the n unmasked pixels:

```
K_ij = δ_ij + a² exp(−(λ_i − λ_j)² / 2ℓ²) + ε δ_ij
ln L = −½ rᵀ K⁻¹ r − ½ ln|K| − Σ_i ½ ln(2π σ_eff,i²)
```

- λ is the **observed-frame** pixel wavelength in Å (`Spectrum.wavelength`).
- a is dimensionless, in units of σ_eff: a = 1 puts as much correlated variance into
  each pixel as white variance.
- ℓ is in observed-frame Å.
- ε = 1e-6 is a fixed diagonal jitter (`GP_JITTER`, the default of
  `GaussianProcess(jitter=)`).

The white noise enters once, as the identity. As a → 0 the value tends to the diagonal
Gaussian, apart from the ε term, which is exactly
`½ Σ r_i² ε/(1+ε) − ½ n ln(1+ε)` (about 5e-7 per pixel). The mask is fixed at setup.
Masked pixels get the identity row and column of K and r = 0, so they add exactly 0.

The kernel is `ceridwen.likelihood.lnlike_gp_gaussian`. It uses a dense Cholesky
factorisation in float64 and is differentiable in a, ℓ and every model parameter.
`tests/likelihood/test_gp_likelihood.py` checks it against the independent NumPy
implementation `GaussianProcess.log_likelihood` (rtol 1e-10), checks its gradients with
`jax.test_util.check_grads`, and checks `jit` / `vmap` against a Python loop.

## Using it with `fitSED`

Two parameters per spectrum, named like the other per-observation noise terms. Both are
natural logs:

| parameter | meaning | unit |
|---|---|---|
| `log_gp_amp_spec` | ln a | a in units of σ_eff |
| `log_gp_length_spec` | ln ℓ | ℓ in observed-frame Å |

With several spectra, each takes its own `log_gp_amp_spec_<obs.name>` /
`log_gp_length_spec_<obs.name>`. The plain names are accepted only when the model has
exactly one spectrum. A GP name that matches no spectrum is an error (e.g.
`log_gp_amp_phot`: only spectra take a GP).

**Sampled.** Give both a prior and a `free_param_init`:

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

**Fixed.** Either use a constant transform for each name,
`transforms={"log_gp_amp_spec": lambda th: jnp.array([0.0]), "log_gp_length_spec":
lambda th: jnp.array([np.log(30.0)])}`, or give the spectrum a `GaussianProcess`:

```python
from ceridwen.observation import GaussianProcess, Spectrum

spec = Spectrum(wavelength=w, flux=f, uncertainty=e, noise=GaussianProcess(1.0, 30.0))
```

`GaussianProcess(amplitude, length_scale)` takes a and ℓ themselves, not their logs. Its
`jitter` becomes ε.

**Both are needed.** Setting only one of the two names is an error. So is giving a
`GaussianProcess` *and* the names for the same spectrum: choose fixed (the object) or
named (sampled, or fixed by transforms).

**Priors.** Nothing is enforced; bounded priors are advisable. A very small ℓ (below
the pixel spacing) makes the GP a second white-noise term, degenerate with
`log_err_scale_spec`. A very large ℓ makes it a smooth offset, degenerate with the
calibration.

The mock-data example `examples/recipes/gp_likelihood.py` fits a spectrum with sampled
hyperparameters and compares the result with the diagonal fit.

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

These refusals are current limitations, not physics. The mixture and the upper-limit
penalty are per pixel, and the line marginal and the profiled polynomial solve assume
independent pixels. Each could be generalised to a full covariance.

## After the fit

- The result file records the GP in the spectrum's `likelihood_json`
  (`read_result_h5(path)["obs"][name]["likelihood"]["gp"]`): the kernel, `log_amp` and
  `log_len` (a theta key when sampled, the fixed ln value otherwise), `eps`, `n_pix`,
  and the sampled names. The GP parameters are ordinary samples in `/samples`.
- `Spectrum.log_likelihood(model_flux)` (host-side, fixed hyperparameters from
  `noise=GaussianProcess(...)`) gives the same value as the compiled likelihood.
  `Spectrum.chi_sq` is always the **diagonal** χ².
- The summary figure's χ²/ν is diagonal with the quoted uncertainties. When a spectrum
  was fitted with a GP it says so ("diagonal, no GP").
- `GPGaussianLikelihood.conditional_mean(y, mu, sigma, mask, theta)` (or
  `ceridwen.likelihood.gp_conditional_mean`) returns the GP's estimate of the correlated
  residual in data units, for plotting model + GP against the data. The sampled
  likelihood never computes it.

## Cost

A dense Cholesky: O(n³) per likelihood call and an n × n float64 matrix (8n² bytes) per
batch lane. The kernel alone was measured on CPU (value, and value plus gradient as NUTS
and the gradient-based steps use it), with `jit(vmap)` over W parameter vectors and
compile time excluded:

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

(Apple M3 Pro laptop CPU, float64, with another CPU-heavy job running on the
machine, so treat these as indicative. The GP kernel only; the forward model comes on top.)
The gradient costs 5-8 times the value, and from 1000 to 2000 pixels the cost grows by a
factor of about 7 (close to n³). At 1000 pixels the GP adds ~26 ms of value and gradient per
lane, about 2 h of CPU for the ~2.5e5 gradient calls of a NUTS run (4 chains × 2000 draws
× ~32 leapfrog steps); at 2000 pixels, ~13 h.

`fitSED` warns at setup above `GP_WARN_NPIX` = 1000 pixels. For longer spectra,
fit a wavelength window or bin the spectrum. A GPU timing is still to be done.
