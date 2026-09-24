# Emission-line marginalisation

Fit a spectrum with the nebular model on, but with the **fluxes of the emission
lines set free**: they are integrated out analytically instead of being fixed at
the CLOUDY prediction, so the stellar absorption underneath (Balmer series,
Ca II H+K, Mg b, the 4000 Å break) still constrains the stellar population while
the lines can take whatever strength the data demand. This is the alternative to
`Spectrum.mask_lines`, which throws the line pixels (and the absorption in them)
away.

```python
import jax.numpy as jnp
from ceridwen import SedModel, Kinematics, Instrument
from ceridwen.observation import Spectrum

spec = Spectrum(wavelength=w_obs, flux=f, uncertainty=e,
                instrument=Instrument.R_fwhm(1000.0),
                marginalize_elines=True)               # flat prior on every covered line

model = SedModel(csp, [spec, phot], priors=priors, zred=3.2,
                 transforms={"sfh": sfh_transform,
                             "gas_logu": lambda th: jnp.array([-2.5]),   # nebular model fixed
                             "gas_logz": lambda th: jnp.array([-0.3])},
                 kinematics=Kinematics(sigma_gal=150.0, sigma_gas=80.0))
print(model.summary())                                 # lists the marginalised lines
```

## What is computed

The line fluxes α (erg s⁻¹ cm⁻², observed frame, IGM-transmitted at the line)
enter every observation linearly: the marginalising `Spectrum` through the
line profiles (the same painter profile as every other line, width
√(σ_gas² + σ_inst²), times the spectrum calibration), every `Photometry`
through the static line-to-band basis, and every `Lines` observation through
its blend matrix. With independent Gaussian noise the likelihood integrated
over α is a closed form: one weighted least-squares solve and one Cholesky
factorisation of an n_line × n_line matrix per likelihood call. It is the exact
marginal likelihood, for

- a **flat prior** (default, `eline_prior_width=0`), or
- a **Gaussian prior** centred on the CLOUDY flux with a fractional width,
  `eline_prior_width=0.2` meaning 20 % (Prospector's convention). Ly-α always
  has a flat prior (resonant scattering makes its grid flux unreliable).

The derivation, the proof that it is exact and the comparison with
Prospector's `fit_mle_elines` are in `docs/dev/eline_marginalisation_design.md`.

## Options

| `Spectrum(...)` argument | Meaning |
|---|---|
| `marginalize_elines` | switch it on (default `False`: every line keeps its CLOUDY flux, nothing changes) |
| `eline_prior_width` | 0 = flat; > 0 = Gaussian, fractional width about the CLOUDY flux |
| `elines_to_fit` | FSPS names (`$SPS_HOME/data/emlines_info.dat`, e.g. `"[O III] 5007"`); default: every line the spectrum covers |
| `elines_to_fix` | lines kept at their CLOUDY flux |
| `elines_to_ignore` | lines removed from every observation |

`theta["eline_delta_zred"]` (a sampled parameter or a transform) shifts every
painted line to (1 + zred + eline_delta_zred) λ_rest, as in Prospector.

A line is fitted when its centre lies inside the spectrum at every redshift the
projector serves, more than 3σ from the edges, with at least 3 unmasked pixels
within 2σ. Covered lines that miss this stay at their CLOUDY flux and are listed
by `model.summary()`; a line you name in `elines_to_fit` that misses it is an
error.

## Requirements (each refused before sampling, with the fix in the message)

- with a nebular model (`add_neb=True`), nebular parameters that are **not sampled**: with free line fluxes the lines cannot constrain `gas_logu` /
  `gas_logz`. Fix them with constant transforms, as above, or tie them to
  another parameter with a transform;
- an `Instrument` on the spectrum (the line width needs the LSF);
- one marginalising spectrum per model;
- no `logify_spectrum`, no GP likelihood (`noise=GaussianProcess(...)` or `log_gp_*`,
  [GP likelihood](gp_likelihood.md)), no upper limits and no outlier mixture
  (`f_outlier_*`, [outlier model](outlier_model.md)) in the jointly fitted observations
  (a spectrum outside the system may carry one);
- `Photometry` together with a sampled `zred`, or with a sampled `sigma_gas`
  and `broaden_photometry=True`, is not supported yet (`NotImplementedError`);
  a sampled `zred` works for a spectrum-only fit;
- there is no `eline_sigma`: the width is `Kinematics(sigma_gas=...)`, a
  velocity **dispersion** in km/s.

## Without a nebular model (`add_neb=False`, `CSPBasis_afe`)

The marginalisation does not need a nebular grid when the prior is flat: the line
list, the rest wavelengths and the names are FSPS's `$SPS_HOME/data/emlines_info.dat`,
and the widths are √(σ_gas² + σ_inst²) as always; none of them depends on the
metallicity or [α/Fe]. So with `CSPBasis(..., add_neb=False)` and with
`CSPBasis_afe` (which has no nebular grid: there are no α-enhanced CLOUDY grids)

- the lines the spectrum covers are fitted with a **flat prior** (the default);
  `eline_prior_width > 0` and `elines_to_fix` are refused, because both need the
  CLOUDY fluxes;
- the model has no nebular continuum and no grid lines, so there are no fixed lines;
- `$SPS_HOME` (or `sps_home=...` on the basis) must point at the FSPS data, for
  `emlines_info.dat`;
- `Photometry` is marginalised jointly as usual; `Lines` observations are refused
  (they need the grid's line fluxes);
- the `cloudy` entries of the outputs are NaN.

## Choosing the prior

- **Flat (`eline_prior_width=0`, the default).** Nothing about the lines is assumed:
  they carry no information about the stellar population, whatever produced them
  (star formation, an AGN, shocks, evolved stars). The safe choice whenever the lines
  may not come from young stars, e.g. in quiescent galaxies.
- **Gaussian about the CLOUDY flux (`eline_prior_width=0.2`, Prospector's usual value).**
  Each fitted line gets a prior N(F_j(θ), (0.2 F_j(θ))²): the data may move a line by
  about ±20 % from what the star-forming nebular model predicts for the current
  stellar parameters, more only at a price. Because the centre and the width both
  follow F_j(θ), the lines now inform the stellar fit: a line much stronger than
  predicted pulls the SFH towards more recent star formation, and where the model
  predicts no line (F_j → 0, e.g. an old population) the width shrinks to zero and
  the line is pinned at ~0 even if the data show one. Use it only when the lines are
  known to come from star formation and the CLOUDY grid describes them to ~20 %. In
  the injection test (lines rescaled by ×0.5 to ×3 from CLOUDY) it biased the line
  fluxes towards CLOUDY by up to 3σ, and in 2 of 50 realisations the optimiser moved
  to an SFH with no recent star formation.

## Evidence

With a flat prior the evidence is defined only up to the (infinite) prior volume of
every fitted line, so **ln Z of a flat-prior marginalised fit must not be compared
with that of another model** (another line set, a masked fit, a standard fit). Posteriors
are unaffected. Compare evidences only between fits with the same fitted lines and the
same prior, or use `eline_prior_width > 0` (a proper prior) when evidences matter.

## Speed

When nothing the line solve depends on can change between likelihood calls (fixed
`zred` and `sigma_gas`, no `eline_delta_zred`, no sampled `spectrum_scaling*`,
`spectrum_calib*` or `eline_scaling`, and no sampled noise terms or `noise_floor`), the
line profiles, the design matrices and, for a flat prior, the whole factorisation are
computed once at setup. On an A100 at the nested sampler's batch width (100) a
flat-prior marginalised likelihood then costs 0.99-1.00 times the ordinary one, and
the 20 per cent prior 1.11 times; otherwise the per-call path costs about 1.2-1.25
times. `model._eline_system.static` is `None` when the per-call path is in use.

## Noise terms

The weights of the line solve are the likelihood's own inverse variances, so the
result is the exact marginal also with `log_err_scale_*`, `log_jitter_*` and
`log_f_data_*`. `noise_floor` and `log_f_calib_*` scale with the model, which would
include the lines being solved for; they are evaluated on the model **without**
the fitted lines.

## Outputs

- `fitSED` writes `/elines` into `ceridwen_result.h5`: `names`, `wave_rest`,
  and per posterior draw the line-flux posterior `mean`, `sd` and the `cloudy`
  prediction; `read_result_h5(path)["elines"]` returns them.
- `PostProcess` returns `out["extras"]["elines"]` (the same per draw), and its
  posterior-predictive spectra, photometry and line fluxes carry each draw's
  posterior-mean lines.
- `ceridwen.likelihood.eline_marginal.eline_line_fluxes(model, theta,
  likelihood)` evaluates the line posterior at one `theta`.

`model.predict(theta)` is unchanged and still predicts the CLOUDY lines; it is
the forward model, not the fit.

## Masking versus marginalising

Masking removes the information in the line pixels, including the stellar
absorption the lines sit on. Marginalising keeps it: the continuum model must
fit those pixels, and only the part a Gaussian of the right width and position
can absorb is attributed to the line. Where emission fills in an absorption
line of similar width the two are partly degenerate, and the marginalisation
reports that honestly as a larger line-flux uncertainty and a wider stellar
posterior, where masking would simply lose the absorption.

## Differences from Prospector

The algebra with a prior is the same (tested to 1e-10 against Prospector
2.0a2.dev42 on identical inputs). Different by design:

- the flat-prior penalty is the correct marginal, +½(n ln 2π + ln|Σ̂|);
  Prospector's no-prior penalty has the opposite sign;
- the profile is the analytic one every other line uses (unit area, exact c),
  on all unmasked pixels; Prospector renormalises numerically with c = 3e18
  inside 5σ windows, which makes its likelihood jump when a pixel enters a
  window;
- the photometry and `Lines` are marginalised jointly with the spectrum;
  Prospector plugs the spectrum's fitted fluxes into the photometry, and only
  when the spectrum is listed first;
- the calibration used to convert amplitudes is the current one (Prospector
  uses the previous likelihood call's).
