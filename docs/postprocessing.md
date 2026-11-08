# Post-processing a fit

`ceridwen.PostProcess` evaluates posterior draws using the fitted forward
model. It returns sampled parameters, star formation histories, time-averaged
SFRs, UV and ionising-photon quantities, and predictions for each observation.
It also provides full, stellar-intrinsic and dust-free model-grid spectra,
the highest-likelihood sample and any quantities defined by your own functions.

```python
from ceridwen import PostProcess, load_postprocess

pp  = PostProcess(model, result)      # the SedModel of the fit; SamplingResult or the .h5 path
out = pp.run()                        # nested dict of numpy arrays
pp.save("post.npz")                   # flat .npz; load_postprocess("post.npz") rebuilds the dict
```

Use the same CSP, observations, transforms and `zred` as in the fit.
`PostProcess` checks parameter names and shapes against the result and
reports both lists if they differ. When loading an `.h5` file, it also
checks the saved fixed redshift, luminosity distance, cosmology and
kinematics, including `broaden_photometry`.

`run()` evaluates draws with `model.predict_vmap`. If you have reassigned
`model.observations`, call `model.setup_observations()` first.

## Output fields

| Key | Shape | Meaning |
|---|---|---|
| `out['theta'][name]` | `(N,)` or `(N, k)` | equal-weight posterior draws of every sampled parameter |
| `out['log_likelihood']`, `out['draw_index']` | `(N,)` | log-likelihood of each draw and its index in the raw samples |
| `out['extras']['sfh']['lookback_gyr']` | `(N, n_time)` | lookback grid of each draw (follows a sampled `zred` when `track_zred_age`; the SFR is scaled with it so the formed mass is kept) |
| `out['extras']['sfh']['sfr']` | `(N, n_time-1)` or `(N, n_time)` | SFR in M⊙/yr in the CSP's convention (per bin for `sfh_interp="step"` or `sfh_per_bin`, per node for `"linear"`) |
| `out['extras']['sfh']['sfr_per_bin']` | `(N, n_time-1)` | the per-bin SFR the step kernel integrates |
| `out['extras']['sfh']['mass_formed']` | `(N,)` | ∫ SFR dt, M⊙ |
| `out['extras']['sfh']['sfr10']`, `sfr100`, ... | `(N,)` | mean SFR over the last 10, 100, ... Myr (`windows_myr`, default 3, 5, 10, 20, 50, 100, 500) |
| `out['extras']['sfh']['ssfr10']`, ... | `(N,)` | `sfrW / mass_formed`, 1/yr |
| `out['extras']['sfh']['mfrac']` | `(N,)` | surviving / formed mass (stars + remnants); only with a grid that carries a mass table (below) |
| `out['extras']['sfh']['mass_surviving']` | `(N,)` | `mfrac * mass_formed`, M⊙ |
| `out['extras']['sfh']['ssfr10_surviving']`, ... | `(N,)` | `sfrW / mass_surviving`, 1/yr |
| `out['extras']['uv']['MUV']`, `LUV` | `(N,)` | absolute AB magnitude at 1500 Å and mean L_ν over 1450–1550 Å (erg/s/Hz) of the full (dust-attenuated) model |
| `out['extras']['uv']['MUV_intrinsic']`, `LUV_intrinsic` | `(N,)` | the same from the stellar-intrinsic spectrum |
| `out['extras']['ionizing']['nion']` | `(N,)` | Q(H) in s⁻¹ from the intrinsic spectrum, with the nebular model's constants |
| `out['extras']['ionizing']['xion']` | `(N,)` | `nion / LUV_intrinsic`, Hz/erg |
| `out['extras']['ionizing']['fesc']` | `(N,)` | `frac_obrun`, present only when the model has it |
| `out['prediction']['photometry'][obs]` | `(N, n_bands)` | maggies, one entry per `Photometry` observation |
| `out['prediction']['spectra'][obs]` | `(N, n_pix)` | as the `Spectrum` observation is fitted |
| `out['prediction']['lines'][obs]` | `(N, n_lines)` | integrated line fluxes |
| `out['prediction']['wave_rest']` | `(n_wave,)` | the model grid, Å |
| `out['prediction']['spectra_observed']`, `zred` | `(N, n_wave)`, `(N,)` | observed-frame f_ν in erg s⁻¹ cm⁻² Hz⁻¹ (cgs; × 1e32 for nJy) at `(1+z) wave_rest` (flux factor and IGM applied; not broadened), and the redshift of each draw; present only when the model has a fixed or sampled `zred` or a `lumdist_mpc` (absent for a `zred = 0` fit, which has no flux factor) |
| `out['prediction']['spectra_model']` | `(N, n_wave)` | rest-frame L_ν [L⊙/Hz] of the fitted model (dust, nebular, dust emission; at model resolution, no kinematic broadening), × 10^logmass |
| `out['prediction']['spectra_intrinsic']` | `(N, n_wave)` | stellar continuum only: no dust, no nebular, ionising continuum included |
| `out['prediction']['spectra_dustfree']` | `(N, n_wave)` | stars + nebular continuum + lines, no dust |
| `out['derived'][name]` | `(N,)` or `(N, k)` | your functions (below) |
| `out['bestfit']` | same tree, one draw | the highest-likelihood raw sample, with all of the above |
| `out['meta']` | | `n_samples`, `n_raw`, `seed`, `resampled`, `windows_myr`, `param_names`, `zred_fixed`, `cosmology`, `sampler`, `weights` (source), `log_evidence`, `log_evidence_err`, `observations`, `sfh_interp`, `sfh_per_bin`, `metallicity` (convention and grid Z_sun), `mfrac` (read from the file's `/derived/mfrac`, computed from the grid's table, or not computed) |

Disable unneeded outputs with `PostProcess(..., sfr=False, ssfr=False,
uv=False, ionizing=False, predictions=False, mfrac=False)`.

## Conventions

**Posterior draws.** Nested-sampling weights are recomputed from dead-point
log-likelihoods and birth contours, using exact live-point counts in sample
order (`ceridwen.sampler.ns_weights.nested_log_weights`). If birth contours
are unavailable, stored `log_weights` are used. A warning reports any
difference between stored and recomputed weights; the recomputed values
take precedence.

Draws are resampled to equal weight, so ordinary histograms and quantiles
describe the posterior. `n_samples` defaults to 2000, with a fixed `seed`.
NUTS draws already have equal weights; `n_samples` selects a subset.
`bestfit` is the raw sample with the highest log-likelihood.

**Star formation.** After transforms, `theta['sfh']` contains the SFH shape
in the CSP's convention. With `logmass`, physical SFR is that shape ×
10^logmass. The CSP applies mass scaling to the spectrum. Thus `logmass`
represents formed mass when the SFH integrates to one solar mass on the
construction grid, as ensured by
`logsfr_ratios_to_sfh(..., sfh_times_yr=csp.sfh_times)`. A directly sampled
`sfh` has its own normalisation.

With `track_zred_age=True`, each draw's grid ends at `age(zred)` and its SFR
is scaled by `T_grid / age(zred)` to preserve formed mass. `sfrW` averages
the same SFH used by the weight kernel over the last W Myr: constant per bin
for `sfh_interp="step"` and linear between nodes for `"linear"`. SFR is zero
beyond the oldest node. `ssfrW` divides this average by formed mass, without
a stellar return correction.

**Surviving mass.** `mfrac` is the fraction of formed mass remaining in stars
and remnants. It weights the SSP grid's `ssp_stellar_mass` table by the same
SSP weights used for the spectrum. The table stores FSPS's `stellar_mass`
per SSP under schema 3. For MIST ages below about 2.5 Myr, FSPS's lowest IMF
bin is counted by mass rather than number; see GOTCHAS section 14.
`mass_surviving = mfrac * mass_formed` and
`ssfrW_surviving = sfrW / mass_surviving` give the Prospector-style mass and sSFR.

The published `mist_miles_chab` and `mist_bpass_v2` grids include this table,
and `SSPData.from_fsps` saves it in new grids. The α-enhanced
`amist_c3k_hr_krou_afe` grid does not include it; use `mfrac=False`.
By default, a missing table produces one warning and only formed mass is
reported. `mfrac=True` instead raises an error; update a published grid with
`fetch_grid(<name>, force=True)` or rebuild a custom grid with `from_fsps`.
`mfrac=False` skips the calculation without a warning.

A table with surviving masses above 1 M_sun per M_sun formed is rejected
with a warning when loaded; the grid remains usable without it. Composite
`mfrac` uses CERIDWEN's SFH integration rather than FSPS's `csp_gen`.
For constant and rising SFHs over 0.1–10 Gyr, agreement with python-fsps is
4.2e-4 for step interpolation and 6.3e-3 for linear interpolation. The
larger difference occurs for young MIST populations, whose youngest node
is 10^5 yr. See GOTCHAS section 14.

**Stored `mfrac`.** When a table is available, `fitSED` computes `mfrac` for
every stored sample without evaluating spectra. This takes well under a
second for thousands of samples. Values are saved in `/derived/mfrac`,
aligned with `/samples`. Attributes `stellar_mass_source`, `grid_chash`
and `sfh_interp` identify the table and integration scheme. You can read
stellar masses directly:

```python
from ceridwen import read_result_h5
r = read_result_h5("out/ceridwen_result.h5")
mass_surviving = r["derived"]["mfrac"] * 10 ** r["samples"]["logmass"].ravel()   # per sample, M_sun
```

Apply the weights in `r["samples"]["log_weights"]`, or use `PostProcess`
to resample. `PostProcess(model, "out/ceridwen_result.h5")` uses stored
`mfrac` without requiring the mass table, but checks that `grid_chash`
and `sfh_interp` match the model. Given a `SamplingResult`, or a file without
`/derived`, it computes `mfrac` from the grid table.

`fitSED(mfrac=False)` skips `/derived`; `mfrac=True` requires a table before
sampling. By default, the array is saved when possible, with one log message
otherwise. `/derived` is reserved for inexpensive, reproducible functions
of theta and the model. Spectra, SFR averages, UV properties and ionising
quantities are computed by `PostProcess`.

**Model-grid spectra.** These are rest-frame luminosity densities, without
distance scaling, (1+z) or IGM absorption. Use per-observation predictions
when comparing with data.

**UV and ionising photons.** `LUV` is the trapezoidal mean of rest-frame L_ν
over 1450–1550 Å; `MUV = −2.5 log10(LUV / (4π (10 pc)²) / 3631 Jy)`.
`nion = (L⊙/h) ∫_{λ<912 Å} L_ν/λ dλ` uses the intrinsic spectrum and the
constants from `NebularGridModel.compute_log_qq`. It matches the ionising
photon rate used by the fit. `xion = nion / LUV_intrinsic`, and `fesc` is
the fixed or sampled `frac_obrun`.

## Custom quantities

Pass functions in `derived` to compute additional quantities. Each receives
one `SpectrumSample` and returns a float or 1-D array. Functions run in a
Python loop for each draw and the best-fit sample, so avoid expensive work:

```python
import numpy as np

def A_V(s):                                   # V-band attenuation from the two spectra
    i = s.index_of(5500.0)
    return 2.5 * np.log10(s.dustfree[i] / s.full[i])

def L_optical(s):                             # erg/s in 4000-7000 A rest, dust-attenuated
    return s.luminosity(s.full, 4000.0, 7000.0)

def beta_UV(s):                               # UV slope of the full model
    m = (s.wave_rest > 1300) & (s.wave_rest < 2600)
    flam = s.full[m] / s.wave_rest[m] ** 2
    return np.polyfit(np.log10(s.wave_rest[m]), np.log10(flam), 1)[0]

pp = PostProcess(model, result, derived={"A_V": A_V, "L_opt": L_optical, "beta": beta_UV})
out = pp.run()
out["derived"]["A_V"]                         # (N,)
out["bestfit"]["derived"]["A_V"]              # float
```

`SpectrumSample` provides `wave_rest`, `full`, `intrinsic` and `dustfree`
(spectra in L⊙/Hz × 10^logmass), transformed NumPy `theta`, `zred`, `logmass`,
`sfr`, `lookback_gyr` and `cosmo`. Helpers include `index_of(λ)`,
`mean_lnu(spec, lo, hi)` and `luminosity(spec, lo, hi)`.

## Figures

```python
pp.figures("post_figs/", title="ID 1009077")      # summary.pdf, corner.pdf, diagnostics.pdf
```

`ceridwen.plotting` creates three matplotlib figures:

- **Summary:** observed-frame SED, data, posterior photometry, 16–84% model
  band and χ residuals, plus line residuals and S/N when lines are fitted.
  Without lines, the SED spans the page width. The SFH panel shows per-bin
  SFR, posterior median and 16–84% interval, the prior interval at median
  mass and the input SFH when supplied in `truths`. Uniform time grids use
  a linear axis; other grids use a log axis. Parameter panels show medians,
  16–84% intervals, best-fit values and prior intervals. The header reports
  z, log M, log SFR10, best-fit χ²/ν and ln Z.
- **Corner:** all fitted parameters, with median dashed lines and the
  maximum-likelihood sample in red.
- **Diagnostics:** nested-sampling dead points in deletion order, coloured
  by posterior weight, with log-likelihood, cumulative posterior weight,
  ln Z and effective sample size. MCMC results show per-chain traces,
  split-R̂ and ESS. Without VI, NUTS chains share one warmup end state,
  so R̂ measures mixing within that run only.

`summary_figure(out, model)`, `corner_figure(out)` and
`diagnostic_figure(result, out)` return figures you can edit. The default
blue palette is defined in `plotting.COLORS`.

## Saving

`pp.save("post.npz")` writes a flat `.npz` file with slash-separated keys,
such as `extras/sfh/sfr10` and `bestfit/theta/logmass`. Strings and lists in
`meta` are JSON-encoded. `load_postprocess(path)` restores the nested dictionary.
