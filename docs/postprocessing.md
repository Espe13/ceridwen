# Post-processing a fit

`ceridwen.PostProcess` turns a sampling result into posterior distributions
of everything you report: the sampled parameters, the star-formation history
and its time averages, UV and ionising-photon properties, the
posterior-predictive photometry, spectra and line fluxes, three model-grid
spectra (full, stellar-intrinsic, dust-free), the best-fit point, and any
quantity you define yourself from those spectra. Every number comes from the
same forward model that was fitted, so it is consistent with the fit by
construction.

```python
from ceridwen import PostProcess, load_postprocess

pp  = PostProcess(model, result)      # the SedModel of the fit; SamplingResult or the .h5 path
out = pp.run()                        # nested dict of numpy arrays
pp.save("post.npz")                   # flat .npz; load_postprocess("post.npz") rebuilds the dict
```

The model must be the one the fit used (same CSP, observations, transforms,
`zred`): the parameter names and shapes of the result are checked against
it and a mismatch is refused with the two lists printed. When `result` is
the `.h5` path, the fixed redshift, luminosity distance, cosmology and
kinematics (`Kinematics` widths, `broaden_photometry`) recorded in the file
are also checked against the model. `run()` pushes every draw through
`model.predict_vmap`, so the model's observations must be set up (they are,
unless you reassigned `model.observations` without calling
`model.setup_observations()`).

## What you get

| Key | Shape | Meaning |
|---|---|---|
| `out['theta'][name]` | `(N,)` or `(N, k)` | equal-weight posterior draws of every sampled parameter |
| `out['log_likelihood']`, `out['draw_index']` | `(N,)` | log-likelihood of each draw and its index in the raw samples |
| `out['extras']['sfh']['lookback_gyr']` | `(N, n_time)` | lookback grid of each draw (follows a sampled `zred` when `track_zred_age`) |
| `out['extras']['sfh']['sfr']` | `(N, n_time)` or `(N, n_time-1)` | SFR in M⊙/yr in the CSP's convention (per node, or per bin for `sfh_per_bin`) |
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

Blocks can be switched off: `PostProcess(..., sfr=False, ssfr=False,
uv=False, ionizing=False, predictions=False, mfrac=False)`.

## Conventions you should know

**Draws.** A nested-sampling result carries importance weights. They are
recomputed from the dead points' log-likelihoods and birth contours
(`ceridwen.sampler.ns_weights.nested_log_weights`, exact n_live counting,
in the order of the samples) rather than read from the file: the
`log_weights` stored by fits before 2026-09-03 were sorted by likelihood
and misaligned with the sample arrays for the final live points (a warning
tells you when the stored weights differ). The draws are then resampled to
equal weight (`n_samples`, default 2000, `seed` fixed), so a histogram or
a median of any array is the posterior. A NUTS result has uniform weights
and is used as it is (`n_samples` subsamples it). `bestfit` is the raw
sample with the highest log-likelihood, not a draw.

**Star formation.** After the transforms `theta['sfh']` is the SFH shape in
the CSP's convention. The physical SFR is that shape × 10^logmass when
`logmass` is a parameter: the CSP scales the spectrum, not the SFH, so
`logmass` is the formed mass when the shape integrates to one solar mass, as
under `logsfr_ratios_to_sfh(..., sfh_times_yr=t)` (a directly sampled `sfh`
carries its own normalisation). `sfrW` is the mean SFR over the last W Myr of
the same piecewise function the weight kernel integrates: constant per bin
for `sfh_interp="step"`, linear between nodes for `"linear"`; beyond the
oldest node the SFR is zero. `ssfrW` divides by the formed mass, with no
stellar return fraction, and keeps that meaning.

**Surviving mass (v1.0.6).** `mfrac` is the fraction of the formed mass still in
stars and remnants: the SSP grid's surviving-mass table (`ssp_stellar_mass`, FSPS's
`stellar_mass` per SSP, SSP schema 3; on MIST below ~2.5 Myr with FSPS's lowest IMF bin
counted by mass instead of by number, GOTCHAS section 14) weighted by the draw's SSP weights, the same
weights that make its spectrum. `mass_surviving = mfrac * mass_formed` and
`ssfrW_surviving = sfrW / mass_surviving` are the Prospector-style stellar mass and
sSFR. The published `mist_miles_chab` and `mist_bpass_v2` grids carry the table (the α grid
`amist_c3k_hr_krou_afe` not yet: use `mfrac=False` there) and `SSPData.from_fsps` records it
in every grid it builds. An older copy has none: `PostProcess` then warns once that `mfrac` is
unavailable and reports `mass_formed` only (`mfrac=True` makes the missing table an error
that says how to get a current grid: `fetch_grid(<name>, force=True)` for a published grid,
a rebuild with `from_fsps` for your own; `mfrac=False` skips the block silently). A table
with values above 1 M_sun per M_sun formed is refused when the grid is loaded (one warning;
the grid works without it). The composite `mfrac` follows CERIDWEN's
SFH integration, not FSPS's `csp_gen`: against python-fsps for constant and rising
SFHs of 0.1-10 Gyr it agrees to 4.2e-4 (step) and 6.3e-3 (linear, young populations on
MIST, whose youngest node is 10^5 yr); GOTCHAS section 14.

**`mfrac` in the result file (2026-09-23).** When the grid has the table, `fitSED` evaluates
`mfrac` for every stored sample (SFH weights times the table, no spectrum; well under a
second for thousands of samples) and writes it as `/derived/mfrac`, aligned with
`/samples`, with the attributes `stellar_mass_source`, `grid_chash` and `sfh_interp` that
say which table and SFH scheme produced it. So a stellar mass needs no post-processing:

```python
from ceridwen import read_result_h5
r = read_result_h5("out/ceridwen_result.h5")
mass_surviving = r["derived"]["mfrac"] * 10 ** r["samples"]["logmass"].ravel()   # per sample, M_sun
```

(weight the samples with `r["samples"]["log_weights"]`, or use `PostProcess`, which does the
resampling). `PostProcess(model, "out/ceridwen_result.h5")` reads the stored array instead
of recomputing it, so it no longer needs the table, and raises when the recorded
`grid_chash` or `sfh_interp` differ from the model's. Without a file (a `SamplingResult`),
or for a file without `/derived`, it computes `mfrac` from the table as before.
`fitSED(mfrac=False)` skips the group; `mfrac=True` refuses a grid without a table before
sampling; the default writes it when it can and otherwise logs one line.

The rule for `/derived`: it holds only quantities that are a pure function of theta and the
model, cheap for every sample, and exactly reproducible from the file plus the model.
`mfrac` qualifies. Spectra, SFR windows and UV / ionising quantities do not, and stay in
`PostProcess`.

**Model-grid spectra** are rest-frame luminosity densities: no distance, no
(1+z), no IGM. To compare with an observation use the per-observation
predictions, which are what the likelihood saw.

**UV and ionising photons.** `LUV` is the trapezoid mean of L_ν over
1450–1550 Å rest; `MUV = −2.5 log10(LUV / (4π (10 pc)²) / 3631 Jy)`. `nion`
is `(L⊙/h) ∫_{λ<912 Å} L_ν/λ dλ` of the intrinsic spectrum with the constants
of `NebularGridModel.compute_log_qq`, so it is the Q the fit itself used;
`xion = nion / LUV_intrinsic`. `fesc` is `frac_obrun` as sampled (or fixed).

## Your own quantities

A `derived` function receives one `SpectrumSample` and returns a float or a
1-D array; it is evaluated per draw (a Python loop, so keep it cheap) and
for the best fit:

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

`SpectrumSample` carries `wave_rest`, `full`, `intrinsic`, `dustfree` (all
L⊙/Hz × 10^logmass), `theta` (this draw after the transforms, numpy), `zred`,
`logmass`, `sfr`, `lookback_gyr`, `cosmo`, and the helpers `index_of(λ)`,
`mean_lnu(spec, lo, hi)` and `luminosity(spec, lo, hi)`.

## Figures

```python
pp.figures("post_figs/", title="ID 1009077")      # summary.pdf, corner.pdf, diagnostics.pdf
```

`ceridwen.plotting` makes three per-galaxy figures from the output (plain
matplotlib; `plotting.COLORS` holds the blue default palette and is the only
styling): a **summary** page (observed-frame SED with the 16–84 % model band,
data, posterior photometry and χ residuals, across the full width when no lines
were fitted; emission-line residuals with S/N; the SFH as per-bin SFR against
lookback time, with the posterior 16–84 % and median, the prior 16–84 % at the
median mass and the injected SFH when `truths` gives it, on a linear time axis
for a uniform node grid and a log axis otherwise; 1-D marginals with
median, 16–84 %, best fit and prior 16–84 %; header with z, log M, log SFR10, χ²/ν of the best
fit and ln Z), a **corner** plot of all fitted parameters (median dashed,
maximum-likelihood sample in red), and a **diagnostics** page (nested sampling:
dead points in deletion order coloured by posterior weight, the log-likelihood
run and the cumulative posterior weight with ln Z and the effective sample size;
MCMC: per-chain traces with split-R̂ and ESS; without `vi` all NUTS chains start from one
warmup end state, so R̂ there measures within-run mixing only). The functions
`summary_figure(out, model)`, `corner_figure(out)` and
`diagnostic_figure(result, out)` return the matplotlib figures for further
editing.

## Saving

`pp.save("post.npz")` writes a flat `.npz` with `/`-joined keys
(`extras/sfh/sfr10`, `bestfit/theta/logmass`, ...); strings and lists in
`meta` are stored as JSON. `load_postprocess(path)` returns the nested dict
again.
