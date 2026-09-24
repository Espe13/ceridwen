# Outlier mixture likelihood

A few bad pixels (cosmic rays, detector artefacts, sky residuals) or a mis-calibrated band
can drag a Gaussian fit a long way, because a 20σ residual costs 200 in ln L. The outlier
mixture (Hogg, Bovy & Lang 2010, arXiv:1008.4686) lets each datum come either from the
ordinary noise model or, with probability `f`, from a Gaussian `nsigma` times broader. It
is Prospector's outlier model (`prospect.likelihood.NoiseModel`, branch `f_outlier > 0`),
reproduced to machine precision.

!!! note "All outlier fractions default to 0; switch the mixture on explicitly."
    Every `f_outlier_*` is 0 (off) unless your model samples it or fixes it to a non-zero
    value. With no outlier parameter, or one fixed at 0, the likelihood is the ordinary
    Gaussian, compiled to the same program as a model without the mixture.

## The likelihood

Per unmasked datum *i*, with σ_eff,i the uncertainty after **every** noise term
(`noise_floor`, `log_err_scale_*`, `log_jitter_*`, `log_f_calib_*`, `log_f_data_*`, named
per observation like the outlier fractions), χ_i =
(y_i − μ_i)/σ_eff,i and `log_det_i = ½ ln(2π σ_eff,i²)`:

```
ln p_good,i = −½ χ_i²          − log_det_i
ln p_bad,i  = −½ χ_i² / n²     − log_det_i − ln n
ln L_i      = logaddexp( ln(1 − f) + ln p_good,i ,  ln f + ln p_bad,i )
ln L        = Σ_i ln L_i          (masked data contribute 0)
```

This is Prospector's expression term by term: its `var_bad = var n²` gives
`−½ ln(2π var n²) = −½ ln(2π var) − ln n`, and its `var` is the noise model's diagonal
`Sigma` (σ² for the default `NoiseModel`, the kernel sum when kernels are attached), which
is CERIDWEN's σ_eff². `tests/test_outlier_model.py` checks equality with a NumPy
transcription of Prospector's code for every noise-term combination (rtol 1e-12) and with
values written by Prospector itself (`tests/reference/prospector_outlier.npz`).

**One deliberate difference.** Prospector's plain branch (`f_outlier == 0`,
`NoiseModel.lnlikelihood`, `noise_model.py:90` at commit a78d153) computes
`−½[ln(2π) χ² + Σ ln σ²]`: the χ² is multiplied by ln 2π and `n ln 2π` is missing. Its
outlier branch is correct. CERIDWEN follows the outlier branch for f > 0 and, at f = 0,
uses its ordinary (correct) Gaussian. Prospector's value at exactly `f = 0` is therefore
**not** a reference, and as f → 0 CERIDWEN tends to the correct Gaussian, not to
Prospector's number.

## Using it with `fitSED`

The mixture is switched on per observation by **name**, like the other noise terms. Each
observation gets its own fraction; nothing is shared between observations.

| parameter | observation | CERIDWEN default | Prospector template (reference only, `TemplateLibrary["outlier_model"]`) |
|---|---|---|---|
| `f_outlier_spec` | the `Spectrum` | **0 (off)** | free, init 0.01, `TopHat(1e-5, 0.5)` |
| `nsigma_outlier_spec` | the `Spectrum` | **50** | fixed at 50 |
| `f_outlier_phot` | the `Photometry` | **0 (off)** | fixed at 0; `TopHat(0.0, 0.5)` when freed |
| `nsigma_outlier_phot` | the `Photometry` | **50** | fixed at 50 |
| `f_outlier_lines` | the `Lines` | **0 (off)** | no template (see below) |
| `nsigma_outlier_lines` | the `Lines` | **50** | — |

The per-observation names below have the same defaults: 0 (off) and 50.

**Small photometric sets: keep `f_outlier_phot` at 0.** On a clean mock with 8 NIRCam bands
and `f_outlier_phot` free (`TopHat(0.0, 0.5)`), the current SFR came out
0.592 [−0.344, 0.753] (median [16, 84 %]) against 0.770 [0.580, 0.965] for the Gaussian
fit: with few bands the fit can treat a rest-UV band as a possible outlier and the SFR
constraint loosens (A100 recovery test). Free it only when a bad band is
suspected and the photometry has enough bands to tell.

**Several observations of one kind** (two spectra, say `"G140M"` and `"G395M"`): each takes
its own `f_outlier_spec_G140M`, `f_outlier_spec_G395M` (and `nsigma_outlier_spec_<name>`),
`<name>` being the observation's `name`. The plain `f_outlier_spec` is accepted only when the
model has exactly one spectrum; with two it is an error, as is setting both the plain and the
per-observation name of one observation. An `f_outlier_*` / `nsigma_outlier_*` name that
matches no observation is an error (it would be sampled without entering the likelihood).

To opt in, sample a fraction by giving it a prior and a `free_param_init` (this example
switches the mixture on for the spectrum only; the photometry stays Gaussian):

```python
from ceridwen.priors import TopHat

model = SedModel(csp, [phot, spec],
                 priors={**priors, "f_outlier_spec": TopHat(low=0.0, high=0.5)},
                 free_param_init={**init, "f_outlier_spec": jnp.array([0.01])},
                 transforms=transforms, zred=zred)
result = fitSED(model, output_dir="out")
```

The `fitSED` log names it: `spec: DiagonalGaussianLikelihood, outlier mixture f =
f_outlier_spec (sampled), nsigma = 50 (fixed)`; every observation without it reads
`outlier mixture off`.

- **Fixed values** use the usual constant transform:
  `transforms={"f_outlier_phot": lambda th: jnp.array([0.05])}`. It must be a constant
  (checked at setup); a fixed `0.0` leaves the mixture off, so the likelihood is the
  ordinary Gaussian, graph unchanged.
- **`nsigma_outlier_*`** defaults to 50 (Prospector's). Sample it with a prior whose lower
  bound is positive, or fix it with a transform. Setting it without the matching `f` is an
  error.
- **Priors on `f`** must be bounded inside [0, 1] (`Uniform`, `TopHat`, `ClippedNormal`).
- **With `run_sampler`**, build the noise model yourself:
  `DiagonalNoiseModel(f_outlier="f_outlier_spec", nsigma_outlier=50.0)` (a string is a
  theta key, a float a fixed value).

### Line fluxes (`Lines`)

Prospector's `Lines` is an ordinary `Observation` (`observation.py:495`), so it carries a
noise model and `lnprobfn` sends it through the same `NoiseModel.lnlike`: the mixture works
for line fluxes there, it just has no template name. CERIDWEN names it `f_outlier_lines`.
With a handful of fluxes the data barely constrain f, so a fitted fraction mostly returns its
prior; the mixture is then a heavy-tailed likelihood that stops one mismeasured line
(blend, aperture, calibration) from dragging the fit, which is its main use here. A fixed
small value (for example 0.05) is often the sensible choice.

### Upper limits

Data flagged as upper limits keep their one-sided penalty `−½ max(μ − y, 0)²/σ² − log_det`;
the mixture applies to the detections only (`lnlike_diag_outlier_with_upper_limits`).
Prospector has no one-sided kernel, so there is no reference for the limits; the detections
are Prospector's outlier branch exactly.

### Emission-line marginalisation

`Spectrum(marginalize_elines=True)` marginalises the line fluxes jointly over the
marginalising spectrum, **every** `Photometry` and **every** `Lines` observation of the
model; that closed-form marginal assumes a Gaussian likelihood. The mixture is therefore
allowed only on observations outside that system — in practice a second spectrum that does
not marginalise its lines — and refused (`NotImplementedError`) inside it.

Prospector does it differently: with both on it fits the line amplitudes with the plain
uncertainties (`sedmodel.py:324`, `spec_unc = None` → `obs["unc"]**2` at :686, ignoring even
its noise kernels), adds a Gaussian Occam penalty, and then applies the mixture to the
spectrum with the maximum-likelihood lines. That is not the marginal of the mixture over the
line fluxes, and is not reproduced.

### The prior at f = 0 and its gradient

f is a fraction and the Gaussian is the mixture at f = 0, so the natural support is
[0, 1), and a lower bound of 0 keeps "no outliers" inside the model. ln L is concave in f
(a sum of logs of functions linear in f) and finite at 0, with the exact derivative
`d ln L/df = Σ_i (p_bad,i − p_good,i)/L_i`, which at f = 0 is `Σ_i (p_bad,i/p_good,i − 1)`.
That is finite unless a datum lies beyond ~38σ (for nsigma = 50) (then +∞ in float64: the likelihood really
rises that steeply). CERIDWEN evaluates this derivative directly (a custom JVP,
`ceridwen/likelihood/likelihood.py` `_mixture_jvp`), so a sampled f with lower bound 0 has a
finite gradient at f = 0 (`test_gradient_in_f_is_exact_and_finite_at_zero`); computing it
through `ln f` would give 0·∞ = NaN. Prospector's lower bound of 1e-5 for
the spectrum is therefore not needed for numerical reasons; either bound works. The upper
bound 0.5 keeps the inlier component the majority, which is what makes "inlier" and
"outlier" identifiable (above 0.5 the two labels can swap).

## Diagnostics

`LikelihoodOutput` with the mixture on: `lnl_pointwise` holds the per-datum mixture terms
ln L_i (they sum to `lnl_total`); `chi` and `residuals` are those of the inlier Gaussian,
(y − μ)/σ_eff. The posterior probability that a datum is an outlier,
`p_i = exp(ln f + ln p_bad,i − ln L_i)`, is available off the sampling path:

```python
p = likelihood.outlier_probability(obs.flux, pred, obs.uncertainty, obs.mask, theta)
```

(`DiagonalGaussianLikelihood.outlier_probability`, or the function
`ceridwen.likelihood.outlier_probability`). The sampled likelihood never computes it.

## Not supported

- **Correlated noise.** The mixture is diagonal only, as Prospector's
  (`assert Sigma.ndim == 1`). On a spectrum with the GP likelihood
  (`docs/gp_likelihood.md`) the mixture is refused at setup.
- **Inside the emission-line marginalisation** (above).

## Non-finite data

Observations mask non-finite flux and non-finite or non-positive uncertainties at
construction and **warn** when those points were not already in your mask. The sampled
likelihood reads the data with the masked non-finite values replaced by finite
placeholders, so they reach neither the value nor the gradient (a NaN left in a masked
slot would otherwise give 0·NaN = NaN in the gradient, for the Gaussian likelihood as well).
`obs.flux` itself is kept as given.

## Cost

The mixture replaces one `−½χ² − log_det` per datum by two Gaussians and a `logaddexp`.
Measured on one A100 (production spectrum of 3065 pixels + 8 bands, nebular on, fractions
sampled for spectrum and photometry, 240 interleaved calls): +0.6 % at the nested-sampling
batch width W = 100 (value and gradient, and value only), −0.1 % to +0.8 % at W = 1 and
500, and +1.9 % (+49 µs of 2.59 ms) for value and gradient at W = 4 (NUTS with four
chains), the one case above the 1 % target.
With it off (no outlier parameter, or `f` fixed at 0), the compiled likelihood is byte for
byte the one before the feature: the lowered StableHLO of `jit(value_and_grad(lnprob))`
is identical in all four `scripts/bit_identity_check.py` configurations.
