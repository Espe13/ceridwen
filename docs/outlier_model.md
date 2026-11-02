# Outlier mixture likelihood

Bad pixels, sky residuals and calibration errors can strongly affect a Gaussian
fit: a 20σ residual contributes −200 to ln L. The outlier mixture models
each datum with either the ordinary noise distribution or, with probability
`f`, a Gaussian `nsigma` times broader (Hogg, Bovy & Lang 2010,
arXiv:1008.4686). It reproduces Prospector's `f_outlier > 0` likelihood
(`prospect.likelihood.NoiseModel`) to machine precision.

All `f_outlier_*` values default to zero. Enable the mixture by sampling a
fraction or fixing it above zero. With the fraction absent or fixed at zero,
the compiled likelihood is the ordinary Gaussian.

## The likelihood

For each unmasked datum i, σ_eff,i includes all configured noise terms:
`noise_floor`, `log_err_scale_*`, `log_jitter_*`, `log_f_calib_*` and
`log_f_data_*`. Define χ_i = (y_i − μ_i)/σ_eff,i and
`log_det_i = ½ ln(2π σ_eff,i²)`:

```
ln p_good,i = −½ χ_i²          − log_det_i
ln p_bad,i  = −½ χ_i² / n²     − log_det_i − ln n
ln L_i      = logaddexp( ln(1 − f) + ln p_good,i ,  ln f + ln p_bad,i )
ln L        = Σ_i ln L_i          (masked data contribute 0)
```

Prospector sets `var_bad = var n²`, giving a broad-component normalisation
of `−½ ln(2π var) − ln n`. Its diagonal `Sigma` is equivalent to
CERIDWEN's σ_eff², including any noise kernels. `tests/test_outlier_model.py`
compares all noise-term combinations with a NumPy transcription at rtol
1e-12 and with Prospector outputs in `tests/reference/prospector_outlier.npz`.

At exactly `f_outlier == 0`, Prospector's `NoiseModel.lnlikelihood`
(`noise_model.py:90`, commit a78d153) instead computes
`−½[ln(2π) χ² + Σ ln σ²]`. This multiplies χ² by ln 2π and omits n ln 2π.
CERIDWEN uses the ordinary Gaussian at zero, which is also the limit of
its mixture as f → 0. The two implementations therefore differ at f = 0.

## Using it with `fitSED`

Configure the fraction by observation name, as with other noise terms.
Each observation has a separate fraction.

| parameter | observation | CERIDWEN default | Prospector template (reference only, `TemplateLibrary["outlier_model"]`) |
|---|---|---|---|
| `f_outlier_spec` | the `Spectrum` | **0 (off)** | free, init 0.01, `TopHat(1e-5, 0.5)` |
| `nsigma_outlier_spec` | the `Spectrum` | **50** | fixed at 50 |
| `f_outlier_phot` | the `Photometry` | **0 (off)** | fixed at 0; `TopHat(0.0, 0.5)` when freed |
| `nsigma_outlier_phot` | the `Photometry` | **50** | fixed at 50 |
| `f_outlier_lines` | the `Lines` | **0 (off)** | no template (see below) |
| `nsigma_outlier_lines` | the `Lines` | **50** | — |

These defaults also apply to names with observation suffixes.

For small photometric sets, leaving `f_outlier_phot=0` usually gives a better
SFR constraint unless a bad band is suspected. In an A100 recovery test with
eight clean NIRCam bands, sampling `f_outlier_phot` with `TopHat(0.0, 0.5)`
gave an SFR median and [16, 84%] interval of 0.592 [−0.344, 0.753], compared
with 0.770 [0.580, 0.965] for the Gaussian fit. Treating a rest-UV band as a
possible outlier weakened the constraint. More bands are needed to distinguish
bad measurements from changes in the SED.

For several observations of one type, include their names. Two spectra named
`"G140M"` and `"G395M"` use `f_outlier_spec_G140M` and
`f_outlier_spec_G395M`, with matching `nsigma_outlier_spec_<name>` keys.
The plain `f_outlier_spec` requires exactly one spectrum. Providing both
plain and suffixed names for one observation, or a name matching no
observation, raises an error.

To sample a fraction, give it a prior and `free_param_init` entry. This
example enables the mixture for the spectrum while keeping photometry Gaussian:

```python
from ceridwen.priors import TopHat

model = SedModel(csp, [phot, spec],
                 priors={**priors, "f_outlier_spec": TopHat(low=0.0, high=0.5)},
                 free_param_init={**init, "f_outlier_spec": jnp.array([0.01])},
                 transforms=transforms, zred=zred)
result = fitSED(model, output_dir="out")
```

The `fitSED` log reports the sampled fraction and fixed width as
`spec: DiagonalGaussianLikelihood, outlier mixture f = f_outlier_spec (sampled),
nsigma = 50 (fixed)`. Other observations report `outlier mixture off`.

- Fix a fraction with a constant transform, for example
  `transforms={"f_outlier_phot": lambda th: jnp.array([0.05])}`. Setup checks
  that the transform is constant. A value of 0 disables the mixture. `CSPBasis(fixed=...)` takes only CSP parameters; these names belong to the observations, so they are fixed with a constant transform.
- `nsigma_outlier_*` defaults to 50. It can be fixed with a transform or
  sampled with a prior bounded above zero. It requires a matching fraction.
- Priors on `f` must be bounded within [0, 1], such as `Uniform`, `TopHat`
  or `ClippedNormal`.
- With `run_sampler`, construct the noise model explicitly:
  `DiagonalNoiseModel(f_outlier="f_outlier_spec", nsigma_outlier=50.0)`.
  Strings refer to theta keys; numbers are fixed values.

### Line fluxes (`Lines`)

Prospector applies the same noise-model likelihood to `Lines`, though it has
no template parameter for the fraction. CERIDWEN calls it `f_outlier_lines`.
With few line fluxes, the fraction is usually prior-dominated. The mixture
can still reduce the effect of a mismeasured line, for example from blending
or aperture errors. A small fixed fraction, such as 0.05, is often sufficient.

### Upper limits

Upper limits retain the one-sided penalty
`−½ max(μ − y, 0)²/σ² − log_det`. Only detections use the mixture
(`lnlike_diag_outlier_with_upper_limits`). Prospector has no equivalent
upper-limit kernel; its outlier branch is the reference for detections only.

### Emission-line marginalisation

`Spectrum(marginalize_elines=True)` jointly integrates over line fluxes in
that spectrum and all `Photometry` and `Lines` observations. The analytic
solution requires a Gaussian likelihood. An outlier mixture within this
system raises `NotImplementedError`. A second spectrum outside the
marginalisation can use the mixture.

Prospector first fits line amplitudes with the quoted uncertainties
(`sedmodel.py:324,686`, `spec_unc=None`), ignoring noise kernels. It then
adds a Gaussian Occam term and applies the mixture to the spectrum with
maximum-likelihood lines. CERIDWEN does not implement this approximation
to integration over the line fluxes.

### The prior at f = 0 and its gradient

The mixture becomes Gaussian at f = 0, so a prior starting at zero includes
the no-outlier case. ln L is concave in f and finite at zero. Its derivative is
`d ln L/df = Σ_i (p_bad,i − p_good,i)/L_i`, reducing to
`Σ_i (p_bad,i/p_good,i − 1)` at f = 0. For `nsigma=50`, this overflows
float64 only for residuals beyond roughly 38σ.

CERIDWEN evaluates the derivative directly through the custom JVP
`_mixture_jvp` in `ceridwen/likelihood/likelihood.py`. Differentiating through
ln f at zero would instead produce NaN from 0·∞. The test
`test_gradient_in_f_is_exact_and_finite_at_zero` checks this boundary.
Prospector's lower bound of 1e-5 is therefore unnecessary for numerical
stability in CERIDWEN. A prior upper bound of 0.5 keeps the ordinary-noise
component as the majority.

## Diagnostics

With the mixture enabled, `LikelihoodOutput.lnl_pointwise` contains each
datum's ln L_i and sums to `lnl_total`. `chi` and `residuals` still use the
ordinary Gaussian residual (y − μ)/σ_eff. To compute each datum's posterior
outlier probability, `p_i = exp(ln f + ln p_bad,i − ln L_i)`, use:

```python
p = likelihood.outlier_probability(obs.flux, pred, obs.uncertainty, obs.mask, theta)
```

This is available through `DiagonalGaussianLikelihood.outlier_probability`
or `ceridwen.likelihood.outlier_probability`. It is computed only when requested.

## Not supported

- The mixture assumes diagonal noise, as does Prospector's implementation
  (`assert Sigma.ndim == 1`). Combining it with a
  [GP likelihood](gp_likelihood.md) on the same spectrum raises an error at setup.
- Observations within a joint emission-line marginalisation cannot use the
  mixture, as described above.

## Non-finite data

At construction, observations mask non-finite fluxes and uncertainties that
are non-finite or non-positive. They warn if those points were not already
masked. The likelihood substitutes finite placeholders for masked non-finite
values so they cannot contaminate values or gradients through 0·NaN.
The original `obs.flux` array is retained.

## Cost

For each datum, the mixture evaluates two Gaussians and a `logaddexp` instead
of one Gaussian. An A100 benchmark used a 3065-pixel spectrum, eight bands,
nebular emission, sampled fractions for both observation types and 240
interleaved calls. At the nested-sampling batch width W = 100, runtime
increased by 0.6% for both value-only and value-and-gradient calculations.
Changes at W = 1 and 500 ranged from −0.1% to +0.8%. At W = 4, corresponding
to four NUTS chains, value and gradient cost 1.9% more: 49 µs on 2.59 ms.

With the fraction absent or fixed at zero, the lowered StableHLO for
`jit(value_and_grad(lnprob))` is identical to the likelihood before this
feature in all four `scripts/bit_identity_check.py` configurations.
