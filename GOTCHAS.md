# CERIDWEN — gotchas & user-error guide

A practical guide to the ways CERIDWEN can be *mis-used* and what now happens
when you do. The dominant historical hazard was **silent wrong results**: the
dict-valued `theta` plus JAX's clamp/NaN semantics let bad input through and
returned a plausible-looking number. The hardening pass (see
`tests/regression/misuse_report.py` for a live status figure) turns these into
**loud errors or warnings**. Every guard lives in non-jitted construction/setup
code or runs only at JIT *trace* time, so the compiled hot path is unchanged and
just as fast.

Run `python tests/regression/misuse_report.py` to regenerate
`tests/regression/figures/misuse_report.png` (a green/red status board).

---

## 1. Metallicity is SOLAR-RELATIVE (v1.0.5) — and Z_sun is per grid

`theta["logzsol"]` (constant) and `theta["logzsol_hist"]` (time-varying) are
**`log10(Z / Z_sun)`**, with the Z_sun **of the SSP grid you loaded**. `0.0` is solar on
every grid. Before v1.0.5 the keys were `theta["Z"]` / `theta["zh"]` and held `log10` of the
*absolute* metallicity; both now raise a `ValueError` that prints the converted value.

- Z_sun is never guessed from `isoc_type`: FSPS changed MIST's `zsol` from 0.0142 to 0.0191
  to 0.0185 (commits `0498750`, `c8752a1`, `1c9d876`) under the same name, so two grids
  labelled `mist` can differ by 0.11 dex. It is resolved at load, in this order:
  an explicit `SSPData.load(..., zsun=)`, the file's own `log10_zsun` provenance, or
  `ceridwen.ssps.grid_metadata.CHASH_TABLE` keyed by the grid's content hash. If none
  applies, loading **raises**; sources that disagree also raise.
- `log10 Z_sun` is the grid's own solar node, so `logzsol = 0` is exactly a grid point.
  `csp.zmet` is the axis in logzsol, `csp.zmet_native` the native axis, and
  `csp.zsun_nominal` / `csp.zsun_source` say what was resolved and from where.
- **Grid ranges** (logzsol): BPASS `[-2.30, +0.30]` (Z_sun = 0.020), MIST/aMIST
  `[-2.50, +0.50]` (Z_sun = 0.0185; 0.0142 for grids built with python-fsps <= 0.4.7).
- **On MIST and aMIST grids `logzsol` is `[Fe/H]`**, not the total metallicity: FSPS loads
  each node from `isoc_feh_<tag>_afe_<a>`, so every `[alpha/Fe]` plane shares one `[Fe/H]`
  axis. The *native* values there are FSPS labels `log10(Z_sun 10^[Fe/H])`, not MIST's
  physical initial Z (which is 0.0164 at `[Fe/H] = 0`, Dotter+2026 Table 1). On an alpha
  grid the total metallicity is the derived `logzsol_total`
  = `[Z/H]` = `logzsol + log10(1 - x + x 10^[alpha/Fe])`, `x = 0.687490`
  (`ceridwen.ssps.grid_metadata.logzsol_total`).
- The **gas** metallicity `theta["gas_logz"]` was already solar-relative and is unchanged:
  `log10(Z_gas / Z_sun,neb)` on the CLOUDY axis (Byler+2017), which spans `[-1.98, +0.20]`
  for the MIST/Padova/PARSEC grids and `[-1.3, +0.3]` for BPASS. `CSPBasis(gas_tied=True)`
  ties it to the stars, `gas_logz := logzsol` (Prospector's convention); `gas_logz` is then
  not a parameter, and giving one anyway raises. The tie sets the same *number* on two axes
  with different solar references (the SSP grid's Z_sun, the CLOUDY grid's); that is the
  FSPS/Prospector meaning of "gas metallicity = stellar metallicity", not equal absolute Z.
- **Guards:** out-of-grid values warn at construction and name the grid's Z_sun; a value or
  prior that lies entirely below `logzsol = -1.2` warns that it looks like an old absolute
  `log10 Z`; a *bounded* prior wider than the grid raises, and so does a constant transform
  outside it. The one route that cannot be checked at construction is a transform whose value
  depends on sampled parameters: it is warned about, and the forward model still clamps at the
  grid edge (`csp.check_param_ranges(theta)` on your own draws is the check).
- The FSPS manual (`$SPS_HOME/doc/sps.tex`) still prints `Z_sun = 0.0191` for MIST, which
  contradicts its own source (0.0185). CERIDWEN follows the source and the grid axis.

## 2. `theta` is a dict — typos are silently ignored

A mistyped key (`logmas` for `logmass`, `gas_logU` for `gas_logu`, `dust2` for
`diffuse_tau_kc`, …) is simply not read, so the parameter silently takes its
default. **Guard:** `predict()` / `get_spectrum_components()` now emit a
`warning` listing unrecognized keys (computed from the dict *keys* at trace
time → zero hot-path cost). Direct `get_spectrum(theta)` calls bypass this —
prefer `predict`/`get_spectrum_components`.

## 3. Metallicity-mode ↔ key mismatch

- `zh_const=True` needs `theta["logzsol"]` (shape `(1,)`); `zh_const=False` needs
  `theta["logzsol_hist"]` (shape `(n_time,)`, index 0 = today).
- A mismatch raises a clear `ValueError` at construction naming the fix, and supplying
  *both* keys now raises as well (it warned before v1.0.5).
- `theta["Z"]` / `theta["zh"]` raise wherever they appear — theta, `priors`,
  `free_param_init`, `transforms` — with the converted value in the message.

## 4. SFH pitfalls

- **NaN/Inf SFH** → was a silent NaN spectrum; **now a `ValueError` at
  construction.**
- **Negative SFR** (e.g. from an unconstrained sampler — sample `log SFR`!) is
  clipped to ≥0 internally → ~zero flux. **Now warned at construction.**
- **`sfh` length** is overloaded: `(n_time,)` = node-based, `(n_time-1,)` =
  per-bin/FastStepBasis. Wrong lengths raise a clear `AssertionError`; the two
  valid lengths switch interpretation silently — make sure you know which you
  mean.
- **Metallicity & SFR units are now identical across all four
  `calculate_ssp_weights_*` calculations.** Metallicity (`theta["logzsol"]` for
  constant-Z, `theta["logzsol_hist"]` for time-varying) is `log10(Z/Z_sun)` on the
  `self.zmet` logzsol axis in **every** variant; the
  SFR history `theta["sfh"]` is a linear rate floored identically at `1e-30`
  everywhere. Previously `var_zh` floored SFR with `self.tiny_logt = -70` (a
  log10-time constant), so it weighted the same SFR history differently from
  `const_zh` and returned **NaN** for (near-)zero SFR nodes. That is fixed:
  `const_zh` output is unchanged (it already used `1e-30`), `var_zh` now shares
  the floor, and `const_zh == var_zh` for a constant metallicity history
  (regression-tested in `test_misuse.py`).
- `lookback_time` fixes the number of nodes and the bin structure at
  construction and is not a free parameter, but a `theta['lookback_time']`
  supplied at predict time (a registered `lookback_time` transform, in Gyr)
  overrides the construction grid for that evaluation, and with
  `track_zred_age=True` the grid is rescaled to `age(zred)`. Its *length* can
  never change after construction.

## 5. Observations must be set up before `predict`

- Call `obs.setup_for_model(wave_model[, …])` once before the first
  `predict`/JIT trace. **Guard:** `Spectrum.predict` / `Lines.predict` now raise
  a clear `RuntimeError` instead of a cryptic `AttributeError`.
- `Photometry.predict` before `setup_for_model` raises `RuntimeError` (no
  rest-frame fallback).
- **All-zero uncertainties** → `ValueError` at construction.
- **Non-finite flux or non-finite / non-positive uncertainties** are masked at
  construction, with a warning when your mask did not already exclude them; the
  likelihood never sees them (not in the value, not in the gradient).
- **Unknown filter name** → `FileNotFoundError` naming the missing `.par` file.
- `SedModel` calls `setup_for_model` for every observation at construction, so
  you never call it yourself there. If you replace `model.observations`
  afterwards, call `model.setup_observations()` (it also drops the cached
  jitted predictors); `fitSED(model, observations)` does this automatically. A
  `Spectrum` needs `wavelength=` at construction for this (flux may come later).
- `eline_scaling` is a **fraction / direct multiplier** on the model emission
  lines: `1.0` = no aperture loss, `0.65` = lines at 65%, `2.0` = lines doubled.
  (It was previously a percentage where 100 = no loss; changed to 1.0 for
  intuitiveness.)
- `eline_scaling` vs `spectrum_scaling` — **two different calibrations, do not
  conflate.** `eline_scaling` scales the emission-**LINE** component only
  (the `Lines` observation). `spectrum_scaling` scales the whole **`Spectrum`**
  prediction onto the photometric flux scale. Neither touches the photometry,
  and the two are independent. Before 2026-08 the spectrum's lines were also
  scaled by `eline_scaling`; now the spectrum is governed solely by
  `spectrum_scaling`, so a joint Spectrum + Lines fit that previously leaned on
  `eline_scaling` to set the spectrum level must add a `spectrum_scaling` prior.
- `spectrum_scaling` acts on `Spectrum` **only** (a no-op for pure photometry/line
  fits). In the nebular-free `CSPBasis_afe` it is the *only* spectrum
  level knob (there are no lines, so `eline_scaling` is inert there).
- `spectrum_calib` (2026-08-31) is the wavelength-dependent companion of
  `spectrum_scaling`: Legendre coefficients `c_1..c_order` (shape `(order,)`)
  multiplying the `Spectrum` prediction by `1 + sum_k c_k P_k(x)`, `x` = observed
  pixel wavelength mapped onto [-1, 1] over **all** pixels of `obs.wavelength`
  (mask-independent, so the coefficients keep their meaning across runs). No
  `c_0` term: the level is `spectrum_scaling`. Photometry untouched. Fitting it
  without photometry in the likelihood leaves the continuum shape degenerate
  with dust and age — always keep the photometry in. Order too high eats real
  features; look at the recovered curve (`legendre_design_matrix` rebuilds it
  from the posterior).

## 6. Environment & data consistency (not auto-guarded — check yourself)

- **`SPS_HOME`** must point at an FSPS data checkout for the nebular model and
  dust emission (`CSPBasis`/`CSPBasis_afe` raise a `ValueError` at construction
  when `add_neb=True` or `add_dust_emission=True` and neither `sps_home=` nor
  `$SPS_HOME` is set), and at a full python-fsps install for
  `SSPBasis`/`FastStepBasis`.
- **SSP grid ↔ nebular library — now auto-enforced (was a silent trap).**
  CERIDWEN records the isochrone library in the SSP grid's provenance, and
  `CSPBasis` picks the matching CLOUDY nebular grid automatically; a conflicting
  `isoc_type` passed in `init_neb_params` now **raises** instead of silently
  loading the wrong grid. (Grids built before provenance tracking warn and fall
  back to `'mist'`.) You still must build the SSP grid from the *same* FSPS you
  directly compare against — a MILES grid (`ssp_data.h5`, 5994 λ-points) and a
  BPASS install (15000 points) differ in shape, so a raw ceridwen-vs-FSPS
  comparison would mismatch.
- **Only one `fsps.StellarPopulation` per process.** FSPS keeps global Fortran
  state; constructing a second `StellarPopulation` corrupts the first. (CERIDWEN
  itself makes none — the nebular model reads grid *files*.)
- **Apple Metal / GPU float32:** x64 is unsupported on Metal. The package enables
  `jax_enable_x64=True`; on Metal force `JAX_PLATFORMS=cpu` for float64 parity.
- Pin **`jax>=0.4.30`** (uses `jnp.trapezoid`, modern tree-util/PRNG).

## 7. Broadening: where the widths come from (2026-09-03)

- There are exactly three widths and each is set once: the galaxy's
  `sigma_gal` / `sigma_gas` in `Kinematics` (on `SedModel`), the instrument's
  LSF in `Instrument` (on each `Spectrum`), the library resolution in the SSP
  grid (read automatically). Nothing else broadens anything: `get_spectrum` is
  no longer velocity-broadened, and the old `sigma_losvd_kms` (CSP),
  `sigma_losvd` / `fit_sigma_smooth` / `resolution` / `smoothtype` /
  `res_convention` / `inres` (Spectrum) and `theta["sigma_smooth"]` are gone.
  Before this pass a `Spectrum(sigma_losvd=...)` was applied **on top of** the
  CSP's hidden 300 km/s, so a fitted width was a residual, not the dispersion.
- `Kinematics(sigma_gal=...)` has no default; `SedModel` defaults to
  `DEFAULT_KINEMATICS = Kinematics(sigma_gal=300.0)` (stars and gas) and prints
  it. A float is fixed, a string is a theta key. **Guards:** a key missing from
  `theta` → `KeyError` (no prior → the usual warning); a fixed width that is
  *also* in `theta` → `ValueError`; a bounded prior reaching above `sigma_max`
  (2000 km/s) → `ValueError`; a sampled value above `sigma_max` is clipped
  (zero gradient there, so keep the prior inside).
- `Instrument` puts the unit *and* the R convention in the constructor name
  (`R_fwhm`, `R_sigma`, `fwhm_aa`, `sigma_aa`, `sigma_kms`, `fwhm_kms`); there
  is no plain `R`. Non-positive or non-finite widths, an array without `wave=`,
  or an array not covering the spectrum's pixels → `ValueError`.
- Library subtraction: instrument finer than the library at some pixels →
  warning with the pixel count and range; the continuum stays at library
  resolution there, which is the correct model for a grid coarser than the
  instrument (`C3K_lr` at 70 km/s, say). A whole-spectrum warning with a
  high-resolution grid means a wrong `Instrument` unit. `instrument=None`
  subtracts nothing.
- Photometry is computed from the `sigma_gal`-broadened spectrum by default
  (`SedModel(broaden_photometry=True)`); < 5e-4 mag for broad bands at
  300 km/s, per-cent level for a narrow band with a line on its edge. With
  `broaden_photometry=False` nothing in the photometry is broadened: the
  continuum enters the filters at model-grid resolution and the lines at the
  grid's pixel-floor width, so `sigma_gas` plays no role there. A mock made
  with one setting must be fitted with the same setting
  (`examples/quickstart.py` sets it False for that reason).
- Lines in the photometry: with a **fixed** `sigma_gas` they enter through a
  static line-to-band basis built at that width (no painting on the model
  grid, the fast path); with a **sampled** `sigma_gas` they are painted onto
  the grid and broadened at runtime (slower, chosen automatically). Free-z
  photometry always paints. `csp._force_paint_lines = True` forces painting.
- `add_neb=False` + a `Lines` observation is refused at `predict` in every
  basis (`ValueError`), not answered with zeros.
- `Lines` (integrated fluxes) never see a width: they are read directly from
  the grid (blends summed).

## 8. `FastStepBasis` (FSPS-backed)

- It wraps FSPS Fortran and is **not** JIT-compatible (do not `jax.jit` it).
- `convert_sfh(agebins, mformed)` builds the FSPS tabular SFH from
  `agebins` (log10 yr) without validating the bin spacing; use increasing,
  non-overlapping bins or FSPS rejects the table ("Ages must be increasing").

## 9. Cosmology and distances (2026-09-03)

- `CSPBasis(cosmo=...)` is **required** (`TypeError` otherwise). Use the
  presets `Cosmology.planck18()`, `planck15()`, `wmap9()`, or
  `Cosmology.flat(H0, Om0)`, `from_name`, `from_astropy`. `tuniv` is gone;
  passing it is a `TypeError`. Use `csp.age_at(z)` / `cosmo.age(z)` instead.
- `SedModel` refuses a fixed-`zred` fit whose oldest SFH node is more than
  0.5 % older than the universe at that redshift; the message gives both
  numbers. `linspace(0, 13.8, n)` still passes at `z = 0` (13.787 Gyr under
  Planck18) but not at `z = 0.5` (8.59 Gyr). Not checked when the CSP
  rescales the grid itself (`track_zred_age=True`, which acts on a fixed
  non-zero `zred` as well) or a `lookback_time` transform supplies it.
- `SedModel(cosmo=...)` is accepted only when equal to `csp.cosmo`;
  different values raise, because the CSP is what evaluates distances and
  ages.
- `zred = 0` means **no flux factor** (predictions in `L_sun/Hz x
  10^logmass`), not "a source at 10 pc" and not physical maggies. For a
  nearby object pass `lumdist_mpc=` to `SedModel`. The summary and the
  `fitSED` log print which case is in force; check them. `lumdist_mpc`
  makes `SedModel` inject `zred` (even 0) into theta, so with
  `track_zred_age=True` the SFH grid is rescaled to `age(zred)` exactly as
  for any fixed non-zero redshift.
- The cosmology is written to the HDF5 result (`cosmo_*` attrs).
  `ceridwen.result_cosmology(path)` reads it back; files written before
  2026-09-03 carry none (KeyError).
- `csp.cosmo` is read-only: assigning it after construction raises, because
  compiled predictions would keep the old one. Build a new basis.
- `CSPBasis`/`CSPBasis_afe` now refuse unknown keyword arguments
  (`TypeError`). Before, `**kwargs` swallowed anything — `tuniv=`, a
  misspelt `cosmolgy=`, or `add_neb=True` on `CSPBasis_afe`, which has no
  nebular module — without a word.
- Two warnings you will see and should read: `SedModel(zred=0)` with
  observations ("NO flux factor"), and `lumdist_mpc` together with
  `zred > 0` ("replaces D_L(zred)"). `Cosmology.flat(0.3, 0.7)` is a
  `ValueError` (H0 outside 10-1000 km/s/Mpc: the order is H0, Om0).
- With a redshift (or `lumdist_mpc`) in force the predictions are physical:
  `Photometry` in AB maggies, `Spectrum` in observed-frame F_nu
  [erg s^-1 cm^-2 Hz^-1] (cgs), `Lines` in erg s^-1 cm^-2. The flux factor is
  `ceridwen.cosmology.flux_factor_cgs` (`flux_factor_maggies` is a
  backwards-compatible alias of the same function; it does not return maggies).

## 10. Nested-sampling weights are aligned with the samples (2026-09-03)

- `result.log_weights` of a BlackJAX NSS fit are now computed by
  `ceridwen.sampler.ns_weights.nested_log_weights(logL, logL_birth)` in the
  order of `result.samples`. Before, they came from anesthetic's `logw()`,
  which sorts by likelihood (and drops `logL <= logL_birth`), so the stored
  weights of older `.h5` files are misaligned with the sample arrays for the
  final live points. Do not zip `samples[p]` with `log_weights` from an old
  file; `PostProcess` recomputes the weights from `log_likelihoods_birth`
  (stored in every NSS file) and warns when the stored ones differ.
- `result.log_evidence` and its error bar come from anesthetic's
  `NestedSamples.logZ()` when anesthetic is installed (it is, with
  `pip install .`); without it, `log_evidence = logsumexp(log_weights)` and the
  error is NaN. Either way `log_weights` are the aligned per-point weights.

## 11. Behaviour changes of the 2026-09-04 optimisation pass

- A sampled `zred` now routes `Photometry` through the per-sample filter
  projection automatically (before, the projection stayed at the fixed setup
  redshift while the flux factor moved: silently wrong). A `Spectrum` with a
  sampled `zred` uses a redshift-aware projector built for the prior's support
  (`Spectrum(zred_range=...)` when the prior is unbounded or `zred` is a
  transform): the model is read at `theta["zred"]` per call and the lines are
  painted at `lambda_rest (1+z)`; the library width of the fixed kernel is the
  one at the reference redshift (a warning above 10 % change over the range).
  Every parameter can therefore be sampled with every observation type.
- `Photometry.predict` before `setup_for_model` raises (no rest-frame fallback).
- `Lines(sigma_v=)` is a constructor argument.
- `fitSED` honours `noise_floor`, `sky`, `calibration` and `upper_limit` and logs
  them; `logify_spectrum` and a `GaussianProcess` noise model are refused
  (`NotImplementedError`) instead of ignored.
- Sampled noise terms are switched on by NAME: when the model samples
  `log_err_scale` (sigma^2 x exp(2 log_err_scale), a common rescaling of the
  quoted errors), `log_jitter` (+ exp(log_jitter)^2, data units), `log_f_calib`
  (+ (exp(log_f_calib) |model|)^2) or `log_f_data` (+ (exp(log_f_data) |data|)^2),
  `fitSED` builds every observation's `DiagonalNoiseModel` with that term, one
  value shared by all observations, and logs it. Give them a prior and a
  `free_param_init`. This is a `fitSED` feature: with `run_sampler` you build the
  `DiagonalNoiseModel(use_error_scale=True, ...)` yourself, otherwise the
  parameter is sampled from its prior and never enters the likelihood, with no
  warning. The outlier mixture (`f_outlier_spec` / `f_outlier_phot`, section 13) is
  switched on the same way but is **per observation kind**, never shared.
- `SedModel` raises for a prior on a name that is not sampled and warns for
  sampled parameters without a prior; `free_param_init` is applied without
  transforms too; prior constructors reject unknown/missing arguments.
- `CSPBasis(verbose=False)` is silent; `SSPData.load(..., flux_dtype="float32")`
  halves the grid memory.
- `pp.figures(dir)` writes the summary / corner / diagnostics figures
  (`ceridwen.plotting`).

## 12. Emission-line marginalisation (2026-09-21)

`Spectrum(marginalize_elines=True)` integrates the line fluxes out analytically
(`docs/eline_marginalisation.md`).

- **Masking vs marginalising.** `mask_lines` throws away the line pixels and the
  stellar absorption under them. Marginalising keeps the pixels: the continuum
  must still fit them, and only what a line profile of the right width and
  position can absorb goes into the line. Prefer it when the lines sit on
  Balmer, Ca II or Mg b absorption.
- **Emission filling absorption.** A Balmer line of similar width to its
  absorption trough is partly degenerate with it. The marginalisation shows that
  as a wide line-flux posterior and a wider age posterior; that is the honest
  answer, not a bug. A Gaussian prior (`eline_prior_width=0.2`) narrows it by
  assuming the CLOUDY prediction is roughly right.
- **`gas_tied=True` is allowed** with `marginalize_elines`: a tied gas metallicity is not a
  free parameter, it follows `logzsol`, which the stellar continuum constrains. `gas_logu`
  still has to be fixed.
- **The nebular model must not be sampled** (`gas_logu`, `gas_logz`): with free
  line fluxes the lines cannot constrain it. CERIDWEN samples every CSP key
  unless a transform derives it, so fix them with constant transforms
  (`transforms={"gas_logu": lambda th: jnp.array([-2.5]), ...}`); a sampled one
  raises at construction.
- **Line names** are FSPS's (`"[O III] 5007"`, `"Ba-alpha 6563"`); a near miss
  (`"[OIII] 5007"`) raises with the correct name suggested.
- **The width is `sigma_gas`**, a dispersion in km/s; `Spectrum(eline_sigma=...)`
  raises (there is no second line width).
- **Flat prior and many faint lines.** By default every covered grid line is
  fitted, including ones CLOUDY predicts at ~0. They cost nothing in bias but
  each adds an Occam factor to the evidence; restrict with `elines_to_fit` when
  you compare evidences between models.
- **Ly-α** always has a flat prior.
- **Evidence with a flat prior** is defined only up to the prior volume of every
  fitted line: never compare ln Z of a flat-prior marginalised fit with another model.
- **The 20 % prior assumes star-forming lines.** Its centre and width are the CLOUDY
  flux F_j(θ), so lines inform the SFH, and where the model predicts ~no line (old
  populations) the line is pinned at ~0. In quiescent galaxies use the flat prior.
- **Without a nebular model** (`add_neb=False`, or `CSPBasis_afe`) the lines come
  from FSPS's `emlines_info.dat` and the prior must be flat; `eline_prior_width > 0`,
  `elines_to_fix` and `Lines` observations are refused, `$SPS_HOME` must be set.
- `model.predict(theta)` still predicts the CLOUDY lines (the forward model);
  the fitted lines are in `/elines`, `PostProcess` `extras["elines"]` and its
  predictions.

## 13. Outlier mixture likelihood (2026-09-22)

`f_outlier_spec` / `f_outlier_phot` / `f_outlier_lines` (with `nsigma_outlier_*`, default 50)
switch on Prospector's outlier mixture for the `Spectrum` / `Photometry` / `Lines`
(`docs/outlier_model.md`).

- **All outlier fractions default to 0; switch the mixture on explicitly** (sample the
  fraction, or fix it non-zero). Prospector's template (spectrum free, init 0.01) is a
  reference, not CERIDWEN's default.
- **Keep `f_outlier_phot` at 0 for small photometric sets.** On a clean 8-band mock with it
  free, SFR_now widened to 0.592 [-0.344, 0.753] from 0.770 [0.580, 0.965] (Gaussian).
- **A lower prior bound of 0 is safe for gradients:** d ln L/df is computed exactly (custom
  JVP) and is finite at f = 0.

- **One fraction per observation.** With two spectra the names are
  `f_outlier_spec_<obs.name>`; the plain `f_outlier_spec` is then an error (ambiguous).
- **Prospector's f = 0 value is not a reference.** Its plain branch multiplies chi^2 by
  ln 2 pi and drops n ln 2 pi (`noise_model.py:90`); CERIDWEN equals Prospector's outlier
  branch for f > 0 and the correct Gaussian at f = 0.
- **Fixed means a constant transform**, `transforms={"f_outlier_phot": lambda th:
  jnp.array([0.05])}`; a transform that depends on sampled parameters raises. Fixed at
  0.0 = off (the unchanged Gaussian).
- **Upper limits** keep the one-sided penalty; the mixture acts on the detections.
- **With `marginalize_elines`** the mixture is refused on the marginalised system (the
  marginalising spectrum, all photometry, all line fluxes) and allowed outside it.
- **Refused, loudly:** `nsigma_outlier_*` without its `f_outlier_*`; a name matching no
  observation; an unbounded prior on `f` or one outside [0, 1].
- `lnl_pointwise` holds the mixture terms; `chi` stays the inlier (y - mu)/sigma_eff. The
  per-datum outlier probability comes from `likelihood.outlier_probability(...)`.

## 14. Priors ported from Prospector: `LogNormal` and `LogUniform` (2026-09-22)

- **`LogNormal(mode, sigma)` does not mean what Prospector's does.** Same class name, same
  argument names, different distribution for the same numbers:
  - CERIDWEN (`ceridwen/sampler/priors.py`, class `LogNormal`) builds
    `tfd.LogNormal(loc=mode, scale=sigma)`: `ln x ~ N(mode, sigma)`, so `mode` is the
    **mean (= median) of ln x**; the pdf in x peaks at `exp(mode - sigma**2)`.
  - Prospector (a78d153, `prospect/models/priors.py:442-480`) is
    `scipy.stats.lognorm(sigma, loc=0, scale=exp(mode + sigma**2))`:
    `ln x ~ N(mode + sigma**2, sigma)`, so its `mode` is **ln of the peak** of the pdf in x.
  - Conversion, same `sigma`: `mode_ceridwen = mode_prospector + sigma**2`
    (and `mode_prospector = mode_ceridwen - sigma**2`).
  - Checked (22 Sep 2026, CPU): Prospector `LogNormal(mode=ln 2, sigma=0.5)` vs CERIDWEN
    `LogNormal(mode=ln 2 + 0.25, sigma=0.5)` on 20001 points in [0.05, 50]: max |d ln p| =
    1.07e-14; without the conversion 3.81. Prospector's pdf peaks at x = 2.000 = exp(mode);
    CERIDWEN's `LogNormal(mode=ln 2)` peaks at 1.558 = exp(mode - sigma**2).
  - `LogNormal.scale` still returns Prospector's `exp(mode + sigma**2)`, which is not the
    scale of the distribution CERIDWEN samples; nothing in the package reads it.
- **`LogUniform(mini, maxi)`** is Prospector's `LogUniform` (`scipy.stats.reciprocal`):
  pdf `1 / (x ln(maxi/mini))` on `[mini, maxi]`, uniform in `log x`. It needs
  `0 < mini < maxi < inf` (raises at construction otherwise). The parameter `x` itself is
  sampled and stored in the result file (not `log10 x`, unlike the recipe
  `examples/recipes/loguniform_prior.py`); NUTS maps it to `(mini, maxi)` with the logit
  that `fitSED` builds from `_detect_bounds`, nested sampling draws it by its inverse CDF.
  Its log-density is `-inf` outside the support.

```python
import math
from ceridwen.priors import LogNormal, LogUniform

priors = {"diffuse_tau_kc": LogUniform(mini=1e-2, maxi=3.0)}
m_prosp, sigma = math.log(2.0), 0.5          # Prospector LogNormal(mode=ln 2, sigma=0.5)
same_as_prospector = LogNormal(mode=m_prosp + sigma**2, sigma=sigma)
```

---

## 15. Instrumental LSF scale (2026-09-22)

`Instrument.<unit>(..., scale=...)` multiplies the instrumental dispersion by `s`, in the
continuum kernel and in the line widths (`docs/conventions.md`).

- **`scale=1.0` is the default and changes nothing** (the unscaled code path, byte-identical).
  A fixed float is the same model as the Instrument built with the width times `s`.
- **Degenerate with `sigma_gal` in the continuum.** The continuum sees only
  `sigma_gal^2 + s^2 sigma_inst^2`, so with both free and no lines to separate them, `s` and
  `sigma_gal` trade along that circle: expect a curved, correlated posterior, and a
  `sigma_gal` that is only as good as the prior on `s`. The lines (`sigma_gas^2 +
  s^2 sigma_inst^2`) break it only when `sigma_gas` is fixed or resolved differently;
  with `sigma_gas` TIED the degeneracy is the same in both. Keep the prior on `s` as tight as
  your LSF calibration allows.
- **A sampled scale needs a finite range**: a bounded prior (`Uniform`, `ClippedNormal`,
  `LogUniform`) with a lower bound `> 0`, or `Instrument(..., scale_range=(lo, hi))`
  (required when the key is a transform). An unbounded prior, a bound `<= 0`, a prior reaching
  beyond an explicit `scale_range`, a missing key, or a non-positive fixed scale raise at
  construction. Sampled values outside the range are clipped (zero gradient there).
- **A sampled scale is not bitwise the fixed one.** The log grid of the projector is sized for
  the top of the range, so `theta["lsf_scale"] = 1.1` and a fixed `scale=1.1` sample the model
  on slightly different grids; on a coarse grid with pixels wider than the LSF this is a
  per-cent-level difference (the same happens between two fixed models whose `sigma_max`
  differ). Compare sampled with sampled.
- **Warning to read:** "the continuum kernel of N pixels crosses half a log-grid pixel":
  the instrument is close to the library resolution there, and the response switches between
  linear interpolation and a Gaussian inside the range, a small step in `s`. Harmless for
  nested sampling; for NUTS narrow the range or use a finer grid.
- Photometry and `Lines` never see the instrument; with `marginalize_elines` a sampled scale
  switches off the static precomputation (the per-call path is used).

### What is *not* guarded (and why)

Runtime, per-sample value checks (e.g. "this drawn `Z` is out of grid") are
deliberately **not** placed in the jitted hot path — doing so would either break
JIT or slow every evaluation. Use the non-jitted `csp.check_param_ranges(theta)`
on your priors/bounds once before sampling instead.

## 16. MAP optimisation (`map_fit`, `fitSED(optimize=True)`) (2026-09-22)

- **The MAP is the maximum of `ln L + ln prior` in the parameters you sample**, not of the
  density NUTS explores: NUTS adds the log-Jacobian of its logit map for bounded priors, whose
  maximum is elsewhere. `map_fit` optimises in the logit coordinates only as a change of
  variables, without the Jacobian. It is also not the maximum-likelihood point (the prior is
  included), and it depends on the parametrisation (a `LogUniform` on `x` and a `Uniform` on
  `log10 x` have different MAPs).
- **Nested sampling ignores it**: live points are prior draws. `fitSED(optimize=True,
  sampler="nested")` records `/map` and says so in the log.
- **Every free parameter needs a prior** (the starts are prior draws); a parameter at a bound
  of a `Uniform` stays strictly inside it (the logit map never reaches the edge).
- **Deterministic for a fixed `rng_key`** (checked byte for byte on CPU). The default key in
  `fitSED` is `fold_in(rng_key, 1)`, so switching `optimize` on does not change the sampler's
  own key. Several starts reaching the same `ln p` is the sign of a well-defined optimum; a
  spread in `MAPResult.lnp_starts` means local optima.

## 17. Result files: resuming nested sampling, rebuilding the model (2026-09-22)

- **`resume_from=` needs a periodic checkpoint written by this version**
  (`ns_checkpoint_<pid>.pkl`, which now carries the live state, dead list, rng key and
  iteration). A rescue pickle (`ns_raw_dead_*`) or an older checkpoint holds only the finalised
  dead points: it still loads with `load_checkpoint`, but resuming from it raises.
- **Resume with the same model, settings and `rng_key`.** `num_live`, `num_delete`,
  `num_inner_steps` and the parameter names/shapes must match, and the live points' saved
  `ln L` must equal this model's (rtol 1e-9); anything else raises before sampling. A resumed
  CPU run is byte-identical to the uninterrupted one. `logZ_tol` may differ (it is only the
  stopping rule). The checkpoint file is named after the PID, so the resumed run writes a new one.
- **A result file stores transforms by name only** (`"sfh <- my_function"`), and not the CSP or
  the observation objects. `rebuild_model` therefore needs them from you, and checks what the
  file does record (priors, free parameters and shapes, transform names, zred, kinematics,
  cosmology, grid provenance, `csp_config`, `sfh_times_yr`, every observation's data and
  instrument). It cannot see a transform whose body changed under the same name, CSP options
  outside `csp_config`, or observation options not stored (noise floor, upper limits,
  calibration, sky): `ceridwen.resultfile.NOT_RECORDED` lists them.
