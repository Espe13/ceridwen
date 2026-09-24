# ceridwen — changes (2026‑09‑07 → 2026‑09‑24)

Changes made to the **ceridwen** package during the MIST/MILES JADES refit work,
with rationale and verification. Two themes: (1) **speed** — undo a forward
regression and then make the forward faster; (2) **correctness** — fix an SED
flux-unit mislabel and two JAX tracer-leak bugs. The changes of 2026‑09‑07 → 09‑09
preserve the public API and give bit-comparable science (identical/near-identical logZ);
later entries (e.g. the v1.0.5 `logzsol` metallicity keys, the v1.0.7 per-observation
noise names) change the API and say so in their own entry.

> Scope note: this documents the edits from *this* work. The repo working tree
> also carries other uncommitted changes that predate this session; `git diff`
> is the authoritative full record. Line numbers drift — references are by
> file/function.

---

## 1. Performance

### 1.1 Revert the nebular-forward refactor  (`csp/csp.py`, `neb/NebularGridModel.py`)
**What:** restored the *factored* nebular evaluation path
(`NebularModel.evaluate_batch_factored` + the `_neb_weights_and_base` /
`_neb_spectrum_term` helpers in `CSPBasis`), i.e. `git checkout HEAD -- ceridwen/csp ceridwen/neb`,
undoing the prior week's rewrite to `evaluate_young`/`_neb_terms`.

**Why:** the rewrite materialized a full `(n_Z, n_age, n_wave)` nebular cube per
call. Neutral/faster for a single CPU call, but ~1.4× slower per call and ~1.8×
over a whole fit when the forward is **vmapped over particles on GPU** (the
sampler hot path) — extra device-memory traffic. Invisible to single-call CPU
timing.

**Impact / verification:** Tursa A100, galaxy 142397: 601–603 s → ~340 s (~2×),
**logZ identical (1052.0)**. Proven by bisection (reverting only `csp/`+`neb/`
recovered the speed).

### 1.2 Line-projection optimization — skip painting for fixed‑z photometry  (`csp/csp.py`)
**What:** for fixed‑redshift photometry, the emission lines are no longer painted
as Gaussians across the full 5994‑wavelength spectrum. Instead the broadband line
contribution is added exactly as `G @ line_fluxes`, where
`G = obs._T @ neb.gaussnebarr` (constant → XLA constant-folds it), with the IGM
transmission folded per‑wavelength into `G`.
- `predict()` computes a static `paint_lines` flag; it paints **only** when a
  `Spectrum` observation or a free‑z `Photometry` needs lines on the grid.
- `_assemble_observer_spectra(theta, *, paint_lines=…)` skips the Gaussian paint
  when `paint_lines=False` (returns the line‑free continuum).
- `_project_observations(…, *, paint_lines=…)` adds the `G @ line_fluxes` term for
  fixed‑z photometry when not painting.
- `predict_line_fluxes(theta, *, for_photometry=False)` — new flag that drops the
  two Lines‑only tail factors (the `1/(1+z)` integrated‑flux Jacobian and the
  `eline_scaling` aperture correction) so the per‑line fluxes are the f_ν
  amplitudes the broadband sees.
- Debug/bench hook: `csp._force_paint_lines = True` forces the legacy painted path
  for A/B testing.

**Why:** painting 166 lines onto the full grid cost ~2.1 ms of a ~5.2 ms vmapped
forward, purely so filters "see" the lines — but `Lines` observations already come
from the CLOUDY grid directly, so for photometry(+lines) fits the paint is wasted.
The re-association `T @ (basis @ amp) = (T @ basis) @ amp` is exact; the broadened
Gaussian is baked into `G` (a line spilling across a filter edge is split
correctly — not a delta), and per‑λ IGM in `G` keeps lines near the Lyman break
exact.

**Impact / verification:** forward 5.2 → 3.3 ms (~1.6×), full fit ~1.5×; evidence &
posteriors unchanged; photometry matches the painted path to <10⁻³ of a 1σ error
bar (median 4e‑5). Package‑wide (any fixed‑z photometry+nebular fit). Confirmed
across 4 galaxies on Tursa and over the full 3191‑galaxy campaign (matched
fit‑only wall: **1.47× faster** than the old BPASS campaign, from per‑iteration
compute, same #iterations).

---

## 2. Correctness

### 2.1 SED flux-unit rename + plotting fix  (`cosmology.py`, `plotting.py`, `postprocess.py`)
**What:**
- `cosmology.py`: renamed `flux_factor_maggies` → **`flux_factor_cgs`** (it returns
  a **cgs f_ν** factor `(1+z)(10pc/D_L)² · L_sun/Hz→cgs`, *not* maggies); renamed
  the fiducial-distance constant `_MAGGIES_D_FID_PC` → **`_D_FID_10PC`**; kept
  **`flux_factor_maggies = flux_factor_cgs` as a back‑compat alias** (used by the
  old jades_full predictor and internal callers).
- `plotting.py`: the summary‑figure model spectrum is now converted to nJy with
  the correct **`_CGS_FNU_TO_NJY = 1e32`** (cgs f_ν → nJy), and photometry with
  `_MAGGIE_TO_NJY = 3.631e12`.
- `postprocess.py`: corrected docstrings/comments that mislabeled
  `spectra_observed` as maggies — it is observed‑frame cgs f_ν.

**Why:** the misnamed factor led the SED plot to scale the model spectrum with a
maggies factor while it was actually cgs f_ν, so the **photometry sat far above
the model spectrum**. The rename + correct nJy conversion fixes the display; the
alias keeps every existing caller working.

**Impact:** SED summary figures now show photometry and model spectrum on the same
scale; no change to fitted values.

### 2.2 JAX tracer-leak fixes — lazily-cached per-observation arrays  (`csp/csp.py`)
**What:** two per‑observation caches, built lazily on the first `predict()` call,
now store **NumPy** constants instead of `jnp.asarray(...)`:
- `_neb_blend_matrix_for`: `obs._neb_blend_matrix = jnp.asarray(B)` →
  `np.asarray(B, dtype=np.float32)` (unresolved‑doublet summation matrix).
- `_neb_cube_rows_for`: `obs._neb_cube_rows = jnp.asarray(idx)` →
  `np.asarray(idx, dtype=np.int32)` (wavelength‑matched cube‑row gather indices;
  the un‑blended‑lines path `_line_fluxes[self._neb_cube_rows_for(obs)]`).

**Why:** if the first call lands inside a `jax.jit`/`vmap` trace, a cached **jax**
array is a *tracer* that escapes into the next trace →
`UnexpectedTracerError`. `fitSED` warms up so it never bit normal fitting, but
**cold batch‑tracing** (A/B benchmarks; the 3191‑galaxy figure‑extract
conversion, where 121 all‑un‑blended‑line galaxies failed) tripped it. A NumPy
constant is trace‑independent and indexes/multiplies a jax array cleanly.

**Rule going forward:** *any array cached lazily inside `predict()` must be a
NumPy constant, never `jnp.asarray`.* Audited all of `csp.py`: only these two are
lazy‑in‑`predict`; every other cache (`_age_bin_mix`, `_losvd_idx`,
`_losvd_kernel_fft`, the `self._*` grid arrays) is built in `__init__`/setup
(`_init_age_bin_operator`, `_setup_losvd_kernel`) — concrete, outside any trace,
and safe as jax arrays. *(Later superseded: the LOSVD helpers named here no longer
exist; the broadening moved to `ceridwen/broadening.py`.)*

**Impact / verification:** cold two‑trace `jit(vmap(predict))` test → no leak;
`_neb_cube_rows` is an `ndarray`; vmap forward finite; line/predictive/misuse
tests pass. Identical numerical results (only the cache backend changed).

---

## 3. API / configuration

### 3.1 Removed vestigial `tuniv` argument  (`csp/csp.py`)
**What:** dropped the `tuniv` constructor parameter (and `self.tuniv`); `describe()`
(now `__repr__`) reports `self.age_at(0.0)`. The forward already used `age_gyr(z, self.cosmo)`.
**Why:** unused after the revert; keeps the constructor clean. `CSPBasis` still
accepts `cosmo`, `track_zred_age`, `fesc_geometry`, etc. — API otherwise unchanged.

### 3.2 Import-time JAX config knobs  (`__init__.py`)
**What:**
- Force x64 by default, overridable: `jax.config.update("jax_enable_x64",
  os.environ.get("CERIDWEN_X64","1") != "0")`.
- Opt‑in lower‑precision matmuls: if `CERIDWEN_MATMUL_PRECISION` is set (e.g.
  `high` for TF32 on Ampere+), apply `jax_default_matmul_precision`.
**Why:** x64 is needed for numerical stability (nebular Q(H)~1e46–1e53 overflow
float32; igm hardcodes float64). TF32 is a knob for matmul‑heavy downstream use —
**off by default** because it gives no measurable speedup for this
bandwidth/elementwise‑bound forward (logZ agrees within the evidence error).

---

## Verification summary
- **Speed:** vmapped‑64 forward 5.2→3.3 ms (line‑projection); full campaign
  1.47× faster than old BPASS on the matched fit‑only wall (per‑iteration compute,
  same #iterations, both packed 4/GPU‑node).
- **Accuracy:** logZ unchanged by the revert; photometry from the projected‑line
  path matches the painted path to <10⁻³ σ; posteriors indistinguishable.
- **Robustness:** both tracer leaks fixed and audited; cold two‑trace test clean.
- **Tests:** `test_lines_static`, `test_observation_predictive`,
  `test_misuse`, `test_line_projection_continuum` pass. The remaining suite
  failures (`test_neb_cube_cache`, `test_cosmology_configurable`,
  `test_csp_construction`, `test_spectrum_calib`) are pre‑existing on committed
  HEAD and unrelated to these changes.

## Not changed / intentionally deferred
- TF32 left disabled by default (no speedup here).
- Free‑redshift photometry and `Spectrum` observations still use the painted path
  (they need lines on the wavelength grid); the projection optimization applies to
  fixed‑z photometry.
- The `_neb_cube_rows_for` / `_neb_blend_matrix_for` fixes are the package‑level
  cure; the figure‑extract converter's eager pre‑cache workaround is no longer
  needed once this ceridwen build is used.

---

## 2026-09-11 — sampled redshift with a `Spectrum`

**What:** `SpectralProjector.build(..., zred_range=)` (`broadening.py`),
`Spectrum(zred_range=)`, `SedModel` derives the interval from the `zred` prior
bounds and builds the spectral projector for it instead of refusing; `fitSED`
refusal removed. Per call the rest-frame model is read at
`wave_log·(1+z_ref)/(1+z)` and the lines are painted at `λ_rest(1+z)`; the
static response and the FFT stage are unchanged. Library width frozen at
`z_ref` (warning above 10 % drift). Details: `FREE_Z_SPECTRUM_2026-09-11.md`.

**Bit-identity:** the fixed-`zred` path is byte-identical to before
(`np.array_equal` new vs old module). New tests in `tests/test_broadening.py`
and `tests/test_spectrum_free_z.py`.

---

## 2026-09-21 — emission-line marginalisation

**What:** `Spectrum(marginalize_elines=True, eline_prior_width=, elines_to_fit=,
elines_to_fix=, elines_to_ignore=)` and `theta["eline_delta_zred"]`. The fluxes
of the covered nebular lines are integrated out analytically and jointly over the
spectrum, every `Photometry` and every `Lines` observation
(`ceridwen/likelihood/eline_marginal.py`; wiring in `csp.predict(eline_system=)`,
`SedModel.predict_with_elines`, `run_sampler`, `MultiObservationLikelihood`,
`SpectralProjector.line_basis`). Flat prior by default (the correct marginal,
not Prospector's sign), fractional Gaussian prior about the CLOUDY flux as an
option, Ly-α always flat. Every refusal at construction (no nebular model,
sampled nebular parameters, no `Instrument`, more than one marginalising
spectrum, `logify_spectrum`, GP noise, upper limits, photometry with painted
lines, near-degenerate lines, unknown names). `fitSED` stores `/elines`;
`read_result_h5` returns it; `PostProcess` predictions carry the posterior lines
and `extras["elines"]`. `NebularModel.nebem_line_names` (from
`emlines_info.dat`, wavelength-matched). Design record:
`docs/dev/eline_marginalisation_design.md`; user page:
`docs/eline_marginalisation.md`.

**Verification:** `tests/test_eline_marginalisation.py` — exact against numerical
quadrature (1 and 2 lines, flat / Gaussian / mixed priors), against Prospector's
`fit_mle_elines` on identical inputs to 1e-10 (8 cases; reference dump
`tests/reference/`), width → 0 equals the ordinary likelihood, width → ∞ equals
flat plus the prior volume, jit / vmap at W = 100 / gradients, end-to-end
`fitSED` → HDF5 → `PostProcess`. New regression category `eline_marginal`. With
`marginalize_elines=False` the forward model is byte-identical to before
(`scripts/bit_identity_check.py`, 845 arrays; the two items that differ also
differ between two runs of the old code).

**Speed (2026-09-21):** a static fast path precomputes the line profiles, design
matrices, weights and (flat prior) the factorisation when zred, sigma_gas, noise and
calibration are fixed. Isolated A100, W = 100: flat prior 0.99x (R=1000) / 1.00x
(R=2700) the ordinary likelihood, 20 % prior 1.11x, per-call path 1.22x / 1.25x; in
whole fits the marginalised runs took 332-349 s against 358 s for masking. Equal to
the per-call path to 1e-11 in ln L. Without a nebular model (`CSPBasis(add_neb=False)`
or `CSPBasis_afe`) `marginalize_elines=True` works with a flat prior (2026-09-22): the line list, rest wavelengths and names come
from `$SPS_HOME/data/emlines_info.dat` and the photometric columns from
`line_profiles_on_grid` (equal to `NebularModel.line_profiles` to 1e-12);
`eline_prior_width > 0`, `elines_to_fix` and `Lines` observations are refused there.

**Also:** `Lines` without `line_names` now gets the teaching `ValueError` on a
wavelength mismatch instead of a `TypeError` (`csp.py` `_neb_cube_rows_for`).
`scripts/check_api_usage.py` skips `tests/reference/` (scripts for other codes)
and honours `# deliberate misuse` markers; `capture_baseline.py --only CAT`
writes only the named categories.

---

## 2026-09-22 — v1.0.3 (CI fix)

v1.0.2's CI failed: the test of line marginalisation without a nebular model read
`$SPS_HOME/data/emlines_info.dat`, and CI's FSPS-free job has no `$SPS_HOME`. The test now
skips without it, like the other `$SPS_HOME` tests. Without a nebular grid the refusals
of `eline_prior_width > 0`, `elines_to_fix` and `Lines` observations now run before the
line list is read, so a missing `$SPS_HOME` no longer masks them
(`eline_marginal.refuse_without_grid`, called from `SedModel.setup_observations`).
`mkdocs.yml` excludes `docs/dev/` from the site. No change to any likelihood or prediction.

The citation title is now "CERIDWEN: Fast and Flexible GPU-Accelerated Stellar Population
Inference" (`CITATION.cff`, and the BibTeX entries in `README.md` and `docs/index.md`).

---

## 2026-09-22 — v1.0.4: outlier mixture likelihood

**What:** Prospector's per-datum outlier mixture (Hogg, Bovy & Lang 2010) in the sampled
likelihood. `DiagonalNoiseModel(f_outlier=..., nsigma_outlier=...)` (None/float/theta key;
nsigma default 50), applied by `likelihood.lnlike_diag_outlier` (and
`lnlike_diag_outlier_with_upper_limits`) through a static branch in the likelihood classes,
so `run_sampler`, `MultiObservationLikelihood.make_lnprobfn` and `fitSED` all use it.
`fitSED` switches it on per observation from `f_outlier_spec` / `f_outlier_phot` /
`f_outlier_lines` (plain name for a single observation of a kind, `f_outlier_<kind>_<name>`
for each of several; + `nsigma_outlier_*`), sampled or fixed by a constant transform
(`fit._outlier_terms`, `fit._check_outlier_setup`); never shared between observations;
`_describe_likelihood` logs it. Upper limits keep the one-sided penalty, the detections get
the mixture. With `marginalize_elines` the mixture is refused inside the marginalised system
(the marginalising spectrum, all Photometry, all Lines) and applied outside it
(`eline_marginal.refuse_outlier_with_elines`). The derivative in f is the exact
`(p_bad - p_good)/L` (custom JVP), finite at f = 0. Diagnostic `outlier_probability` (off the
sampling path). `_likelihood_for` gains `model=` (postprocess passes it).

**Defaults:** every outlier fraction defaults to 0 (off); the mixture is on only when the
model samples a fraction or fixes it non-zero (Prospector's template, spectrum free with init
0.01, is a reference only). The fitSED log reports `outlier mixture off` otherwise. For small
photometric sets keep `f_outlier_phot` at 0: on a clean 8-band mock with it free, SFR_now
widened to 0.592 [-0.344, 0.753] from 0.770 [0.580, 0.965].

**Deliberate difference from Prospector:** equal to its outlier branch (f > 0); at f = 0
the correct Gaussian, not Prospector's plain branch, which multiplies chi^2 by ln 2 pi and
drops n ln 2 pi (`prospect/likelihood/noise_model.py:90`, commit a78d153).

**Non-finite data (all likelihoods):** observations now warn at construction when non-finite
flux or non-finite / non-positive uncertainties were not in the user's mask (they were, and
are, masked). The sampled likelihood reads the data through `likelihood.observation_data` /
`finite_data`, which replace masked non-finite values by 0 (flux) and 1 (uncertainty):
before, a NaN in a masked slot made every gradient NaN (0 * NaN) — NUTS/VI fits of data with
NaNs; nested sampling (no gradients) was unaffected. `obs.flux` itself is unchanged. The
three copies of the static-data code (`runner.py`, `MultiObservationLikelihood`,
`eline_marginal._static_data`) now share `observation_data`.

**Verification:** equal to a NumPy transcription of Prospector's branch for every noise
term (rtol 1e-12) and to values written by Prospector 2.0a2.dev42 (rtol 1e-12,
`tests/reference/prospector_outlier.npz`); nsigma = 1 is the Gaussian; gradients against
finite differences and the analytic f = 0 value; `tests/test_outlier_model.py`. New
regression category `outlier_likelihood`. With the mixture off and finite data the change
is bit-identical (T4). A100 (Tursa): +0.6 % at W = 100; value and gradient at W = 4 (NUTS,
4 chains) +1.9 % (+49 µs), over the 1 % target, not optimised; recovery test in
OUTLIER_MODEL_2026-09-22.md section 12 (`scripts/outlier/tursa/`). A100 full suite: the 16
failures (17 with an upload artefact) fail identically on 9b21fd6 on the same node, i.e.
pre-existing GPU float32 / FSPS-library (C3K_lr on Tursa vs MILES locally) differences
against CPU contracts; none new (section 13). Docs: `docs/outlier_model.md`, GOTCHAS sections 5 and 13.

---

## 2026-09-22 — v1.0.5: metallicity is solar-relative (`logzsol`)

**What.** Every user-facing stellar metallicity is now `logzsol = log10(Z / Z_sun)`, with
`Z_sun` the solar metallicity **of the grid in use**. `theta["Z"]` -> `theta["logzsol"]`
(shape `(1,)`) and `theta["zh"]` -> `theta["logzsol_hist"]` (shape `(n_time,)`); the old keys
raise wherever they appear (theta, priors, `free_param_init`, transforms, result files) with
the converted value in the message. `gas_logz` was already solar-relative and is unchanged;
`CSPBasis(gas_tied=True)` now ties it to the stars (`gas_logz := logzsol`) as Prospector does.

**Why.** `Z_sun` is not a package constant: FSPS changed MIST's `zsol` 0.0142 -> 0.0191 ->
0.0185 (`0498750`, `c8752a1`, `1c9d876`) under the same `isoc_type`, so absolute `log10 Z`
values were only meaningful together with the grid that produced them, and priors, figures and
cross-code comparisons silently mixed conventions (see the Phase 0/A audit in
`Claude outputs/metallicity_phase0_2026-09-22/`).

**How `Z_sun` is resolved** (`ceridwen/ssps/grid_metadata.py`, one table, code not data):
`zsun=` keyword -> the file's `log10_zsun` provenance -> the `chash-v1` content-hash table ->
otherwise a teaching error. Sources that are both present must agree bitwise. `log10 Z_sun` is
the grid's own solar node, so `logzsol = 0` is exactly a grid point, and
`self.zmet = ssp_lgmet - log10_zsun` is computed once at CSP construction (no per-call cost).
The content hash is over the arrays only, so a re-saved or converted copy of a grid still
matches (a file sha256 does not; it is kept as an alias).

**Per-grid table** (`log10_zsun` is the exact solar node; evidence in the module):

| grid | native axis | Z_sun | logzsol means |
|---|---|---|---|
| `mist_miles_chab`, `mist_c3k_lr_chab` (python-fsps 0.5.0) | `log10(Z_sun 10^[Fe/H])` | 0.0185 | `[Fe/H]`, `[-2.5, +0.5]` |
| `mist_bpass_v2`, `bpass_agb_dust` | `log10 Z` (BPASS zlegend) | 0.020 | `log10(Z/Z_sun)`, `[-2.301, +0.301]` |
| `examples/ssp_data.h5` (python-fsps 0.4.7) | `log10(Z_sun 10^[Fe/H])` | 0.0142 | `[Fe/H]`, `[-2.5, +0.5]` |
| `amist_c3k_lr_chab_afe`, `amist_c3k_hr_krou_afe` | `[Fe/H] + log10 Z_sun`, same on every plane | 0.0185 | `[Fe/H]`; `[Z/H]` = `logzsol_total` |
| CLOUDY nebular (`gas_logz`) | already `log10(Z_gas/Z_sun,neb)` | 0.019 (Padova nodes) / 0.020 (BPASS) | unchanged |

**Alpha grids.** `logzsol` is `[Fe/H]` (FSPS loads `isoc_feh_<tag>_afe_<a>` per node, so the
planes share one `[Fe/H]` axis; confirmed on the grid itself: from `[alpha/Fe]` 0 to +0.4 at a
fixed node Fe5270 moves -0.11 A while Mgb moves +2.3 A). The total metallicity is the derived
`logzsol_total` = `[Z/H]` = `logzsol + log10(1 - x + x 10^[alpha/Fe])`, `x = 0.687490`, from the
MESA `input_XYZ` of all 85 MIST v2.5 compositions (reproduced to 1.7e-11 dex). The
`([Fe/H] = +0.5, [alpha/Fe] = +0.6)` cell is **refused**: FSPS ships a byte-identical copy of the
`+0.4` isochrone there (MIST v2.5 excluded that model), and the grid's own L_bol confirms it.
`SSPDataAfe.from_fsps` now detects such duplicates at build time.

**Gas tie and marginalised lines.** `marginalize_elines=True` normally refuses a sampled
nebular model ("the lines no longer constrain it"). `gas_tied=True` is a deliberate exception:
the gas metallicity is not free, it follows `logzsol`, which the stellar continuum constrains.
It warns at construction naming the sampled key. The tie sets the same *number* on two axes
whose solar references differ (the SSP grid's Z_sun, the CLOUDY grid's), which is what
FSPS/Prospector mean by tying the gas to the stars.

**Result files.** `ceridwen_result.h5` records `metallicity_convention`, `log10_zsun`,
`zsun_nominal`, the axis meaning, both axes and the grid's `chash`. Pre-v1.0.5 files with
`Z`/`zh` samples are refused by `load_result_h5` / `read_result_h5`;
`ceridwen.fit.convert_result(path, ssp_grid=...)` converts one after checking the grid, into a
new file.

**Also.** `display()` shows both axes, `Z_sun` and where it came from, and states that MIST /
aMIST native values are FSPS labels, not MIST's physical initial Z (0.0164 at `[Fe/H] = 0`).
The FSPS manual still prints `Z_sun = 0.0191` for MIST, contradicting its own source (0.0185);
CERIDWEN follows the source and the grid.

**Verification.** T1/T2/T3/T4 in the commit message. New: `tests/test_logzsol_convention.py`
(63 tests, incl. agreement with python-fsps's own `logzsol` to 1.1e-16 median fractional
difference, where reading the same number as absolute `log10 Z` is 0.231 and a wrong
`Z_sun = 0.0142` is 3e-2 to 9e-2 off), `tests/test_result_metallicity.py`, three new
regression categories `logzsol_bpass` / `logzsol_mist` / `logzsol_afe`, and 12 new
`misuse_report` rows (37 rows, 0 SILENT).

---

## 2026-09-22 — v1.0.6: surviving stellar mass; provenance in the result file

**Surviving mass (SSP schema 3.0).** `SSPData` gains the optional table
`ssp_stellar_mass` (grid shape without the wavelength axis; `SSPDataAfe` schema 3.1:
`(n_afe, n_met, n_age)`), FSPS's `stellar_mass` (stars + remnants per M_sun formed) of every
SSP, with `stellar_mass_source`. It is written and read by `save` / `load`, validated
(shape, finite, positive), and kept out of the content hash (`chash`), so a grid keeps its
identity when the table is attached. Grids without it load as before. `from_fsps` now records
it; `scripts/attach_stellar_mass.py` adds it to an EXISTING grid without regenerating the
spectra: it runs FSPS (this interpreter or `--fsps-python`), refuses an FSPS whose isochrones,
ages or metallicities differ from the grid's, compares FSPS's spectra with the grid's flux
cube, writes into a copy (`<name>_schema3.h5`) and verifies it (all original datasets
sha256-identical, table round-trips, chash unchanged). No grid was regenerated and the
Zenodo registry is untouched.

`CSPBasis.surviving_mass_fraction(theta)` = sum(W m) / sum(W) with the spectrum's own SSP
weights (post-processing only; the sampled likelihood is unchanged, T4). `PostProcess` adds
`extras["sfh"]["mfrac"]`, `mass_surviving = mfrac * mass_formed` and
`ssfr<W>_surviving = sfr<W> / mass_surviving`; `mass_formed` and `ssfr<W>` keep their names
and meaning. `PostProcess(mfrac=None|True|False)`: auto (warn and skip without a table),
required, off.

**Verification.** On the BPASS and MIST/MILES test grids FSPS's spectra were bit-identical to
the grid (attach run), and the tables (stored in `tests/reference/ssp_stellar_mass.npz`)
reproduce `examples/recipes/reference_mfrac.json`'s single-burst values exactly (MIST) and to
5.7e-10 (BPASS: the json was evaluated at `tage = T`, interpolated by FSPS). Constant SFH:
equal to an independent analytic integral of the table to 1e-10 in both SFH schemes. The
composite values differ from FSPS's `csp_gen` by <= 4.2e-4 (step) / 6.3e-3 (linear) for
0.1-10 Gyr, CERIDWEN's own SFH integration (GOTCHAS 14). New regression category
`stellar_mass`; `tests/test_stellar_mass.py`. Found on the way and **not fixed** (forward
model, needs approval): the `"linear"` scheme never weights the oldest SSP node
(`jmax` clipped to `n_ssp - 1` with `j < jmax`), a strict xfail test documents it.

**Provenance in `ceridwen_result.h5`.** `/provenance`: `ceridwen_version`,
`ceridwen_githash` (build stamp, stale in an editable install) plus the live `git_head` /
`git_dirty` of the package checkout, `jax_version`, `written_utc`, `sampler_json` (the
settings the adapter ran with: nested `num_live`, `num_inner_steps`, `num_delete`, `logZ_tol`;
NUTS `num_warmup`, `num_samples`, `num_chains`, `target_acceptance`, step size, bounds, VI)
and the `rng_key`. Per observation: `sky`, `calibration`, `upper_limit` datasets, the
`noise_floor`, and `likelihood_json` (kernel class and every `DiagonalNoiseModel` field).
`read_result_h5` returns them (`["provenance"]`, `["obs"][name]["likelihood"]`);
`ceridwen.fit.read_provenance(path)`. Older files read as before.

---

## 2026-09-22 — v1.0.7: noise terms and calibration per observation; profiled polynomial

**Per-observation noise terms (renamed; old names refused).** `log_err_scale`,
`log_jitter`, `log_f_calib` and `log_f_data` were one value shared by every observation
(`fitSED`); an additive jitter shared between maggies and cgs F_nu was dimensionally
meaningless. They now follow the outlier-mixture naming (`ceridwen/model/obs_params.py`, one
helper for both): `log_jitter_<kind>` for the single observation of a kind, or
`log_jitter_<kind>_<obs.name>` for each of several (kind = phot / spec / lines); the plain
kind name is ambiguous with several and raises, as do both spellings for one observation and
names matching no observation. The old shared names raise in `fitSED` with the names to use
for that model. `DiagonalNoiseModel` gains `err_scale_key` / `jitter_key` / `f_calib_key` /
`f_data_key` (theta key or fixed float; defaults are the historical names, so a hand-built
noise model reads the same keys as before). A noise term fixed by a constant transform is now
used; before, `fitSED` silently ignored it.

**Per-spectrum calibration.** `spectrum_scaling_<obs.name>` / `spectrum_calib_<obs.name>`
calibrate one spectrum each; the plain names remain for a model with a single `Spectrum` and
raise in `SedModel` with several (they used to apply one value to every spectrum).

**Profiled polynomial calibration** (`Spectrum(polynomial_order=M,
polynomial_regularization=...)`, `ceridwen/likelihood/poly_calibration.py`): Prospector's
`PolyOptCal` inside the sampled likelihood. The response `1 + sum_{m<=M} c_m T_m(x)`
(Chebyshev over the unmasked wavelength range) is solved by weighted least squares at every
call, JIT-safe and differentiable, with the noise model's weights (Prospector: raw
`1/sigma^2`). Opt-in per spectrum (`order = 0` = off, as in Prospector); refused together with
a sampled calibration of the same spectrum (degenerate) and on the `marginalize_elines`
spectrum. Applied in `DiagonalGaussianLikelihood[WithUpperLimits].__call__` and their
`make_lnprobfn`, so `run_sampler`, `MultiObservationLikelihood` and `fitSED` all use it;
`PostProcess` predictions carry each draw's response (`prediction["calibration"]`); the
result file records the order and regularisation (`likelihood_json`).

**Verification.** Equal to Prospector's own `compute_response` (prospect 2.0a2.dev42, code
identical to a78d153) to 1.1e-15 on three reference cases
(`tests/reference/prospector_polyopt.npz`, `run_prospector_polyopt.py`); recovers a known
polynomial to 1e-10 noise-free and within 4 sigma with noise; profiled ln L equals the maximum
over the sampled `spectrum_scaling` / `spectrum_calib` to 1e-9; exact first-order gradients
(`check_grads`), finite full-model gradients with masked NaN data, vmap equals loop;
`tests/test_noise_calibration.py`. New regression category `calibration_likelihood`; new T4
variant `polycal`. With the feature off the change is bit-identical (T4).

## 2026-09-22 — IGM damping wing and DLA (`igm.py`, `csp/csp.py`)

**What.** `ceridwen.igm.MadauDampingDLA` (registry name `"madau1995_damping_dla"`), promoted
from `examples/recipes/igm_damping_dla.py`: Madau (1995) × IGM damping wing × damped Ly-α
absorber, `exp(-(igm_factor tau_Madau + tau_damp(x_HI) + tau_DLA(logN_HI, z_dla)))`. The
kernels are Prospector's (`sedmodel.py` @ a78d153, `:1192-1245` and `:1261-1333`), made
JIT-safe (masked `jnp.where`, `x = 0` singularity of the Voigt approximation moved to
`|x| = 1e-3`).

`x_HI`, `logN_HI`, `z_dla` are **theta keys**, like `igm_factor`: fixed by the constructor,
by theta, or sampled through `SedModel(free_param_init=..., priors=...)`. Mechanism:
`IGMModel.param_names` (new class attribute, empty by default) lists a model's keys;
`CSPBasis._igm_transmission` (new, used at all four IGM call sites, previously four copies of
the same code) passes `attenuation(..., params={key: scalar})` only to models that declare
keys, so a user subclass with the three-argument `attenuation` still works, and adds the keys
to the known theta keys. A model with `bind_cosmology` gets the CSP's cosmology (`h`, `Om0`);
`Ob0` is a required constructor argument for the wing (the `Cosmology` has none).
`igm_factor` is **not** reused as `x_HI` (Prospector does, `sedmodel.py:824`).

**Behaviour change.** None for existing models: `Madau1995` and `NoIGM` are untouched and the
refactored call sites are byte-identical (T4). `MadauDampingDLA()` with no damping/DLA
switched on, `x_HI = 0` and `logN_HI = -inf` all equal `Madau1995` byte for byte.

**Verification.** `examples/recipes/tests/check_igm_damping_dla.py` now runs against the
package code: 63/63 (damping wing vs Prospector ≤ 1.0e-12 in transmission, DLA 1.1e-16).
`tests/test_igm_damping_dla.py` (theta = constructor bit for bit, Madau limits bit for bit
through `csp.predict` and `SedModel.predict`, finite non-zero gradients in all three keys,
guards). New regression category `igm_damping_dla` (each row ≤ 9.4e-14 from Prospector's
own `tau_damping` / `voigt_profile`); new `igm_dla` configuration in
`scripts/bit_identity_check.py` (sampled `x_HI`, `logN_HI`).

**Not done.** `fit.py` does not record the IGM model or its fixed arguments in
`ceridwen_result.h5` (it records no IGM information at all, for any model); sampled keys are
stored like any other parameter. *(Later superseded: `/model@csp_config` records the IGM
model and `igm_factor`.)*

## 2026-09-22 — Gordon+03 SMC bar and Reddy+15 attenuation laws (`dust/attenuation_laws.py`)

**What.** Two new registered laws, promoted from `examples/recipes/extra_dust_laws.py`:
`gordon03_smcbar` (parameter `tau_g03smc`, FSPS dust_type=5, tau(5500 Å) = `tau_g03smc`
exactly) and `reddy15` (parameter `tau_reddy`, FSPS dust_type=6 / Prospector `fake_fsps`,
tau(5500 Å) = 0.997113 `tau_reddy`, i.e. FSPS `dust2`). Registry parameter names equal the
signature names. `examples/recipes/extra_dust_laws.py` is now a re-export, and its `register()`
is kept for old scripts.

**Verification.** `check_extra_dust_laws.py` against the package: 14 passed, 0 failed,
0 skipped (Gordon vs the FSPS table and Fortran 0.0 / 1.1e-16; Reddy vs Prospector 1.1e-15 on
2901 node-aligned pixels, and vs the FSPS Fortran with its single-precision literals 4.4e-16).
New regression category `dust_laws`.

## 2026-09-22 — attenuation-law registry bugs (`dust/attenuation_laws.py`)

`Dust` passes a law only the parameters that are both in its signature and in its registry
`params` (`DustModel.py:107-113`). Three registry entries were wrong. **Each fix changes
results for anyone who used that law as described below.** `tests/test_dust_laws.py` fails on
`main` for each of them (6 failures) and passes after (22/22); it also checks every built-in
law for signature/registry agreement and that every registered parameter reaches the curve.

- **`noll`: bump strength never reached an age-bin `Dust`.** Registry `params` said
  `E_bump`, the signature and `defaults` say `Ebump`. In an age-bin `Dust` (and each renamed
  copy `Ebump1`, `Ebump2`, ... when the law is used in several bins) the bump was dropped, so
  the curve was always the `Ebump = 0` curve; `get_param_names()` advertised `E_bump`, which
  was then warned as unknown. Measured: at 2175 Å with `tau_noll = 1`, `Ebump = 3`, the curve
  was 2.0946 and is now 2.8351 (= a direct `noll(...)` call). `DiffuseDust("noll")` read
  `diffuse_Ebump` correctly before and is unchanged. **Who is affected:** fits with `noll` in
  `init_dust_params["laws"]` and a non-zero `Ebump`: their bump was ignored; results move.
- **`drude`: registered with a function that takes inverse microns.** `Dust` feeds Å, so
  `Dust(laws=["drude"])` returned 1.7e-7 at 2175 Å instead of 0.9997. The registry now points
  at `drude_law(wave, x0, gamma)`, which evaluates `drude(1e4 / wave)` (peak 1 at
  x0 = 4.59 µm⁻¹, 2178.6 Å). `drude` itself (used by `noll`) is unchanged. **Who is affected:**
  any use of the `drude` law (it attenuated nothing before). It still has no amplitude
  parameter: as a bin law it is a fixed bump of peak optical depth 1.
- **`smc` / `lmc`: registry text.** The entries said "Optical depth at 1500 Å" and credited
  Gordon et al. (2003); the functions are Pei (1992) curves normalised at 5500 Å
  (tau(5500 Å) = `tau_smc` exactly, measured). Text only, the numbers are unchanged. **Who is
  affected:** anyone who read `tau_smc` / `tau_lmc` as a 1500 Å optical depth: it is the
  5500 Å depth, and the 1500 Å depth is 4.59× (SMC) / 3.45× (LMC) the parameter (measured).
  Also anyone who cited Gordon et al. (2003) for these laws. The Gordon SMC bar curve is now
  `gordon03_smcbar`.

Golden coverage: regression category `dust_laws` holds the age-bin and diffuse curves of all
six laws (noll with `Ebump = 2`, drude) and two CSP spectra with multi-bin
(`noll`/`gordon03_smcbar`/`lmc` + diffuse `reddy15`; `drude`/`smc` + diffuse `noll`)
configurations: any of the three bugs would have moved them.

## 2026-09-22 — THEMIS dust emission is selectable (`csp/csp.py`, `csp/csp_afe.py`)

**What.** `CSPBasis(..., duste_model="THEMIS")` (and `CSPBasis_afe`). `DustEmission` has read
the THEMIS templates (Jones et al. 2013, 2017; `$SPS_HOME/dust/dustem/THEMIS_MW3.1_*.dat`) all
along, but `CSPBasis` always built it with its `"DL07"` default (`csp.py:716` before this
change), so THEMIS was unreachable. Default unchanged (`"DL07"`, byte-identical, T4
`neb_duste`). A value other than `"DL07"`/`"THEMIS"`, or `"THEMIS"` without
`add_dust_emission=True`, raises at construction. Note the THEMIS `duste_qpah` axis is FSPS's
mass-fraction nodes × 100/2.2, i.e. 0.91-18.2, against 0.47-4.58 for DL07; the same
`duste_qpah` number therefore means a different PAH abundance in the two models.

**Verification.** FSPS selects THEMIS only at compile time (`src/sps_vars.f90:551-560`), and
the installed python-fsps is built with DL07, so there is no FSPS THEMIS spectrum to compare
with. Instead, `tests/test_dust_emission_themis.py`: the (qPAH, Umin) axes equal FSPS's own
(parsed from `$SPS_HOME/src/sps_vars.f90`); a template column equals a direct read of the file;
energy balance (emitted = absorbed, no self-absorption) to 1e-10; the CSP spectrum equals DL07
blueward of 0.9 µm and differs by > 5 % in the mid-IR; `jax.grad` in `duste_qpah` finite.
New regression category `dust_emission_themis`.

## 2026-09-22 — `LogUniform` prior

**What.** `ceridwen.priors.LogUniform(mini, maxi)` (also `ceridwen.sampler`): Prospector's
`LogUniform` (`scipy.stats.reciprocal`), pdf `1 / (x ln(maxi/mini))` on `[mini, maxi]`.
Analytic `jnp` logpdf (`-inf` outside), CDF, ppf and sampling; `tfp_dist()` is `Exp` of a
Uniform in `ln x`. Raises unless `0 < mini < maxi < inf`. `fit._detect_bounds` gives it
`(mini, maxi)`, so NUTS samples it through the logit map; `SedModel.display()` labels it.
`scripts/check_api_usage.py` now checks prior keyword arguments against `prior_params`.
`GOTCHAS.md` section 16 documents it and the `LogNormal` `mode` difference from Prospector
(`mode_ceridwen = mode_prospector + sigma**2`). No forward-model change.

**Verification.** `tests/test_loguniform_prior.py` (values vs a Prospector golden table and
scipy, KS, round trips, jit/vmap/grad, `_detect_bounds`, nested-sampling evidence vs the
analytic value, NUTS, fitSED log-posterior gradient vs finite differences).

## 2026-09-22 — free instrumental LSF scale

**What.** `Instrument.<unit>(..., scale=1.0 | float | "<theta key>", scale_range=None)`
multiplies the instrumental dispersion by `s` in the continuum kernel
(`sigma_gal^2 + s^2 sigma_inst^2 - sigma_lib^2`) **and** in the line widths
(`sigma_gas^2 + s^2 sigma_inst^2`). `scale=1.0` (default) takes the unchanged code path; a fixed
float is folded into `sigma_kms_at` (the same model as the Instrument built with the width times
`s`); a sampled key uses static band geometry sized for the top of its range and exact per-call
Gaussian weights (`broadening.ScaledResponse`), so it is exact for any wavelength dependence of
the LSF. The range comes from the key's bounded prior or `scale_range`; a missing key, an
unbounded prior, a bound `<= 0` or a non-positive fixed scale raise at construction.
Result files record `instrument_scale` / `instrument_scale_range` per spectrum.
Unlike Prospector (`sedmodel.py:289-295`), the lines are scaled too, nothing is mutated in
place, and bad values raise at setup rather than failing an `assert` mid-sampling. GOTCHAS 15,
`docs/conventions.md`.

**Verification.** T4 against pre-change `main`: 2703 arrays compared, the only 2 that differ also
differ between `main` and its own capture (a 8e-45 denormal in a post-processed spectrum, and
`[logZ, logZ_err]`, whose error comes from anesthetic's random `logZ(12)` draws); 1178 new arrays from the two new configurations. New regression category `lsf_scale`
(fixed `s = 1.2` equals `Instrument.R_fwhm(1500/1.2)` at rtol 1e-12; sampled equals fixed at the
top of the range at 1e-12). `tests/test_lsf_scale.py` (15 tests incl. gradients vs finite
differences, jit/vmap). CPU cost of a sampled scale ~10 % of the value-only log-posterior
(W=100 and 500); GPU timing pending (`scripts/bench_lsf_scale.py`).

## 2026-09-22 — MAP optimisation: `ceridwen.optimize.map_fit`, `fitSED(optimize=True)`

**What.** `examples/recipes/map_fit.py` promoted into the package. `map_fit(model, n_starts=16,
rng_key=...)` maximises fitSED's own log-posterior (`fit._likelihood_for` +
`MultiObservationLikelihood.make_lnprobfn`) with `optax.lbfgs` from `model.theta_init` plus
`n_starts` prior draws (Prospector's `nmin`, `fitting.py:223-310`), in the NUTS adapter's logit
coordinates without a Jacobian; returns `MAPResult` (`.theta` is a `free_param_init`,
`.lnp_starts`, `.summary()`). `fitSED(optimize=True, optimize_kwargs=...)` runs it, starts NUTS
(and VI) there through the new `run_sampler(..., theta_init=)`, and writes `/map`
(read back by `read_result_h5(...)["map"]`). Default off; with it off every array is unchanged.

**Verification.** `tests/test_map_fit.py` on the `examples/make_mock_data.py` mock (test grid):
ln p(MAP) = 38346.50 vs 36914.37 for the best of 1000 prior draws; two runs at the same key
byte-identical; `.theta` rebuilds the model at the same ln p; fitSED hands the MAP to the
sampler and stores it.

## 2026-09-22 — result files: resumable nested sampling, model rebuild and check

**What.** `BlackJAXNestedSamplerAdapter(..., resume_from=<checkpoint>)`: periodic checkpoints now
also carry the raw sampler state (live `AdaptiveNSState`, dead list, rng key, iteration, calls,
elapsed time, settings); a run started from one continues where the killed run stopped and is
byte-identical to the uninterrupted run at the same key (CPU). Settings, parameter shapes and
the live points' `ln L` are checked first. Works through `fitSED(sampler_kwargs=
{"resume_from": ...})`. New `ceridwen.resultfile`: `priors_from_result`,
`kinematics_from_result`, `rebuild_model(path, csp, observations, transforms)` and
`check_model_against_result(model, path)`, which writes the model's record through
`write_result_h5` and names every differing attribute/dataset. `write_result_h5` now also
records `/model@csp_config` (CSP class, spectrum model, SFH/metallicity/IGM options, SSP library)
and `/model/sfh_times_yr`. A full rebuild without the user's CSP and transform callables is not
possible (they are not stored).

**Verification.** `tests/test_ns_checkpoint.py` (+2: kill after 3 iterations and resume ->
samples, ln L, birth ln L, weights, ln Z and call count byte-identical; foreign / old
checkpoints refused), `tests/test_resultfile.py` (round trip; each kind of difference named;
priors of every class rebuilt exactly).

## 2026-09-23 — v1.0.8: pip-installable, blackjax from PyPI (`pyproject.toml`, `sampler/nuts.py`)

**What.** The direct git reference `blackjax @ git+...@f73e12956` (which PyPI refuses to
publish) is replaced by `blackjax>=1.6`, the first release with nested sampling. Its
`blackjax/ns/` is byte-identical to the pinned commit; `blackjax.nss` takes the same arguments,
`ns.utils.finalise` the same `update_info`. blackjax 1.6 forces the other floors: `jax`/`jaxlib`
>= 0.9.0 and `optax` >= 0.2.3, and jax 0.9 itself forces `numpy` >= 2.0 and `scipy` >= 1.13.
Because numpy 2 is now mandatory, the floors that predate it were raised to the versions of the
validated environment (`astropy` >= 8.0, `h5py` >= 3.16, `tensorflow-probability` >= 0.25,
`anesthetic` >= 2.14, `matplotlib` >= 3.9); only jax and blackjax were actually exercised AT
their floor, the rest are declared as known-good-at-or-above. `fastprogress`
is dropped: no blackjax release imports it. blackjax >= 1.6 also removed
`window_adaptation(progress_bar=)`, which NUTS passed, so NUTS and NUTS+VI raised `TypeError`
on every released blackjax; the argument is no longer passed (the warmup progress bar is gone;
timing prints unchanged). Install hints in `check.py`, `nested.py`, `nuts.py`, `runner.py` and
the install docs now point at PyPI. CI gains a `wheel` job: build, `twine check --strict`, no
direct-URL `Requires-Dist`, install the wheel into a clean venv, `ceridwen.check`.

Also fixed: `observation.filters.Lbol` called `jnp.trapz`, which jax removed, and raised
`AttributeError`; it now uses `jnp.trapezoid` (same integral).

**Verification** (CPU, macOS arm64, against HEAD `f813417` on the pinned stack, jax 0.10.2):
T4 on blackjax 1.6.2 + jax 0.10.2: 6048 arrays (6059 with `--ns`) byte-identical except the two
items that also differ old-vs-old (1-ulp `postprocess/.../spec`, the random `ns/logZ` error
estimate). NUTS toy (dense, diagonal, VI; 9 arrays) byte-identical on blackjax 1.6 and 1.6.2.
T3 7 passed, T2 21 passed, `check_api_usage` 0 findings, misuse report unchanged (0 SILENT).
The built wheel, installed into clean venvs, passes the FSPS-free suite with jax and blackjax at
their floors (jax 0.9.0, blackjax 1.6; every other package at its current release) and at
jax 0.10.2 + blackjax 1.6.2, and T3 from the wheel at both.
At jax 0.9.0 NUTS samples differ from jax 0.10.2 by <= 3.2e-14 (jax, not blackjax): the
validated environment stays jax 0.10.2.

## 2026-09-23 — `fitSED` writes `/derived/mfrac` (`fit.py`, `postprocess.py`)

**What.** When the SSP grid carries the surviving-mass table, `fitSED` evaluates `mfrac`
(M_surviving / M_formed) for every stored sample after the sampler returns: the CSP's
`surviving_mass_fraction` at the transformed theta, SFH weights times the table, no spectrum,
under `jax.vmap` in chunks (`postprocess.surviving_mass_fractions`, shared with `PostProcess`
through `postprocess.model_theta`). It is written as `/derived/mfrac` (aligned with
`/samples`) with attrs `stellar_mass_source`, `grid_chash`, `sfh_interp`, `units`, timed and
logged like the other stages. `read_result_h5` returns it under `'derived'`
(`read_derived_h5`); `load_result_h5` is unchanged. `fitSED(mfrac=None|True|False)` is the
same switch as `PostProcess(mfrac=)`: default writes it when the table exists and otherwise
logs one line; `True` refuses a grid without a table before sampling; `False` skips it.
`PostProcess` given a result file uses the stored array (so it no longer needs the table)
and raises when the recorded `grid_chash` or `sfh_interp` differ from the model's; given a
`SamplingResult`, or a file without `/derived`, it recomputes from the table as before.
`CSPBasis` / `CSPBasis_afe` now keep the grid's `stellar_mass_source`.

The rule for `/derived`, in the `write_result_h5` docstring and `docs/postprocessing.md`:
only quantities that are a pure function of theta and the model, cheap for every sample and
exactly reproducible. mfrac qualifies; spectra, SFR windows and UV quantities stay in
`PostProcess`.

**Verification.** `tests/test_derived_mfrac.py` (6 tests; each fails under a planted bug:
scaled mfrac, stored array ignored, chash check off, wrong draw index): a nested fit of the
quickstart-style mock on the BPASS test grid with its FSPS table writes 258 values, equal to
`PostProcess`'s recomputation to 1e-12, `mfrac * 10**logmass` = `mass_surviving` to 1e-12;
0.108 s for the mfrac stage including compilation. T4 against `53c5f6e` (clean archive,
`--ns`): 6058/6059 arrays byte-identical; the one difference is `plain/ns/logZ` element 1, the
random anesthetic error estimate (element 0, the evidence, byte-equal). T1 596 passed,
1 failed (`test_picket_factored`, fails identically on `53c5f6e` with `$SPS_HOME` set),
2 skipped, 1 xfailed; T2 21 passed; T3 7 passed; `check_api_usage` 0 findings; misuse report
0 SILENT, two new rows (`fitSED(mfrac=True)` without a table, `mfrac='yes'`).

## 2026-09-23 — no repository script on the user path; no local paths in published files

**What.** `scripts/attach_stellar_mass.py` and `scripts/convert_grids_schema2.py` are
maintainer tools (not in the wheel) and are no longer named by any message or user doc.
`STELLAR_MASS_SCRIPT` is deleted. The messages now say how to get a current grid:
`ssp_data.current_grid_advice(chash, what)` recognises a published grid by its content hash
(`published_grid_name`: `grid_metadata.CHASH_TABLE` name with a `grid_fetch.REGISTRY` URL)
and says `fetch_grid(<name>, force=True)`; any other grid is told that `SSPData.from_fsps`
records the table (or the resolution curve) and to rebuild. Used by
`missing_stellar_mass_message` (now takes `chash=`; `require_stellar_mass`,
`CSPBasis.surviving_mass_fraction`, `fitSED(mfrac=True)`, `PostProcess(mfrac=True)`) and by
the schema-1 refusal in `SSPData._read_h5`. `PostProcess` without a table warns in one line
that mfrac is unavailable and why, with no command. `grid_fetch`: the
`amist_c3k_lr_chab_afe` note no longer describes a conversion (the published copy predates
schema 2.0 and does not load; use `amist_c3k_hr_krou_afe`), and an unpublished registry name
no longer names `scripts_afe/` scripts. Docs updated: `docs/postprocessing.md`,
`docs/installation.md`, `examples/README.md`, `README.md`, `AGENTS.md`, `GOTCHAS.md` (section
14). `tests/test_no_script_paths.py` fails on any string literal in `ceridwen/**/*.py` naming
`scripts/` or `scripts_<x>/` (the `evidence=` provenance of a `grid_metadata` entry exempt).
The quickstart prints `log mass_formed`, `mfrac` and `log mass_surviving`.

No local path in anything published: `fsps_stellar_mass_source` no longer records
`SPS_HOME=<path>` (so `from_fsps` builds are clean). The attach script writes path-free
provenance, records the grid's resolved `log10_zsun`, `zsun_nominal`, `axis_meaning`, `chash`
and `units_lgmet` so the file reads correctly without the package's chash table, and refuses
to keep a file whose attributes contain the home directory, `SPS_HOME` or `scripts/`. The
same path was scrubbed from tracked files: `tests/reference/ssp_stellar_mass.npz` (the two
`source` strings only; the other 13 arrays byte-identical), `examples/recipes/
reference_mfrac.json` and its generator, `tests/reference/run_prospector_{elines,outlier}.py`
usage lines, `docs/dev/eline_marginalisation_design.md`. (Git history keeps the old text.)

Transitional until the next Zenodo deposit (registry patch): the registry still points at
the table-less copies, so `fetch_grid(<name>, force=True)` returns the same old file for now.

**Verification.** T1 599 passed, 1 failed (`test_picket_factored`, pre-existing with
`$SPS_HOME`), 2 skipped, 1 xfailed; the changed test files without `$SPS_HOME`
(`-m "not fsps and not gpu"`): 63 passed, 1 xfailed. T2 21 passed, T3 7 passed,
`check_api_usage` 0 findings, misuse report 0 SILENT. Wheel (`python -m build`) installed in
a fresh venv: `python -m ceridwen.check` all required components present; 56 `.py` files,
none named attach; the only `scripts*/` string in the installed package is the exempt
`grid_metadata` provenance citation; no `/Users` path anywhere in it.

## 2026-09-23 — one alpha grid; alpha grid without a mass table; PyPI polish

**Grids.** Only `amist_c3k_hr_krou_afe` is the published alpha grid: the registry entries
`amist_c3k_lr_chab_afe` (Zenodo 21794924, schema 1, did not load) and the unpublished
`mist_c3k_lr_chab_null` are removed, and so is every user-facing mention (README,
`docs/installation.md`, `examples/README.md`; `examples/demo_afe_quiescent.py` now uses the HR
grid). The LR file stays a developer-local test grid (tests, `grid_metadata` entry for its
Z_sun, regression baselines unchanged). Each registry entry records `stellar_mass_table`;
the HR grid has none until its masses exist, so `missing_stellar_mass_message` tells a user of
that published grid to fit with `mfrac=False` instead of re-fetching (which would return the
same file). `fitSED`'s default already skips `/derived` there with one log line.

**PyPI.** README links are absolute GitHub URLs (the PyPI page hosts no repo files).
`pyproject.toml` uses the PEP 639 `license = "MIT"` + `license-files` (build requires
setuptools >= 77; the `License ::` classifier is dropped), so the PyPI sidebar shows "MIT"
instead of the whole LICENSE text. `MANIFEST.in` prunes `tests/` from the sdist (39 test files
shipped without their fixtures and could not run). `/provenance` records `blackjax_version`
and `numpy_version`; `git_head` is only read from a repository that tracks the package's
`__init__.py`, so a pip install inside a user's project no longer records that project's
HEAD. Four parameter descriptions in `dust/attenuation_laws.py` used an invalid escape
`\AA` (SyntaxWarning on Python 3.12); written `\\AA`, identical strings at runtime.
`ceridwen/AFE_MOCK_TEST_DESIGN.md` moved to `docs/dev/`.

**Verification.** `check_api_usage` 0 findings; misuse report 0 SILENT;
`test_no_script_paths`, `test_dust_laws`, `test_ssp_provenance`, `test_logzsol_convention`
132 passed; `test_result_provenance` 4 passed (new: versions recorded; a foreign repository
gives no git_head); every attenuation-law module value identical before/after the escape fix.

## 2026-09-23 — v1.0.9: grids with the surviving-mass table on Zenodo

**What.** The registry points at Zenodo record 22921057
(doi:10.5281/zenodo.22921057): `mist_miles_chab` and `mist_bpass_v2` are new files with the
surviving-mass table (SSP schema 3.0; sha256 `2f6777a8…`, `c119d19e…`; same chash as before,
so every earlier fit's grid identity holds), `amist_c3k_hr_krou_afe` is the same file (no
table yet). The new file hashes are added to `grid_metadata`'s aliases; the old ones stay so
an old local copy is still recognised (and told to fetch the current one). README,
`docs/installation.md`, `examples/README.md` and the CI grid cache key follow. Version 1.0.9
(pyproject, `_version.py`, CITATION.cff, README / docs BibTeX and quickstart URL).

**Verification.** Zenodo API md5 of all three files equals the local files. Each registry
grid fetched into an empty `$CERIDWEN_GRID_DIR`: checksum verified, loads; MILES (13, 107)
and BPASS (12, 43) tables, HR schema 2.1 without a table.

## 2026-09-23 — v1.0.10: complete grid downloads; CI with the new grids; honest `check`

**Download.** Zenodo intermittently closes the connection before `Content-Length` bytes are
sent (reproduced: 62,427,980 of 66,823,560 bytes), and a chunked read then ends as if the file
were complete, so `fetch_grid` stored a truncated file and refused it on checksum: a new user
of 1.0.9 could not get a grid. `grid_fetch._download` now checks the size and resumes a short
transfer with an HTTP Range request (Zenodo answers 206), restarts if a server ignores the
range, retries transient errors (5xx, dropped connections) up to 8 times, fails at once on a
4xx, and raises a clear error if the file never completes. The sha256 check is unchanged.
`tests/test_grid_fetch_resume.py` (local server that cuts the first 0/1/3 responses short, a
server ignoring Range, give-up, 404); three rounds of real fetches of both new grids into
empty caches all complete.

**CI.** The published BPASS grid now carries the mass table, so the four tests of the no-table
path (`test_stellar_mass` x2, `test_derived_mfrac::test_grid_without_table`, the two misuse
rows) remove it explicitly (`_gridfixture.without_table`) instead of assuming the test grid
lacks one. `test_logzsol_convention` resolves the BPASS grid through `$CERIDWEN_TEST_SSP`
(it hard-coded a local path and failed in CI since before 1.0.8). The CI-equivalent run
locally (`env -u SPS_HOME`, `$CERIDWEN_TEST_SSP` = the new BPASS grid,
`-m "not fsps and not gpu"`): 494 passed, 77 skipped, 0 failed.

**`python -m ceridwen.check`.** It said python-fsps was needed "with nebular / dust emission at
runtime" and ended "All required components present" without FSPS. Nebular and dust emission
read only FSPS's data files (`$SPS_HOME`, a git clone, no compiling); python-fsps only builds
grids. The messages now say so, and without `$SPS_HOME` the summary is "Core installation OK
... NOT yet available: nebular emission and dust emission (need $SPS_HOME)". Exit code
unchanged.

## 2026-09-23 — v1.0.11: summary figure units and SFH; visible diagnostics; demo notice

(1.0.10 was tagged but never uploaded to PyPI; 1.0.11 contains it.)

**Summary figure (`plotting.py`).** At `zred = 0` the SED panel drew the model spectrum in
model units (L_sun/Hz x 10^logmass) but the photometry as raw maggies, which `Photometry`
makes by dividing the band-averaged F_nu by 3631 Jy even in model units: the points sat
~2.75e19 above the spectrum (verified: predicted maggies / band-averaged `spectra_model` =
1/3.631e-20 to 1e-6). The photometry is now converted back (`_MAGGIE_TO_CGS`) and the axis
reads L_nu [L_sun Hz^-1]. The SED x- and y-range cover only the observed wavelengths (filter
transmission > 1 %, spectrum pixels, padded 25 % in log). The SFH panel follows the paper's
figures: per-bin log10 SFR against lookback time [Gyr] on a log axis, posterior 16-84 % per
bin and median, first bin from t1/2; the injected SFH is drawn when `truths` gives the SFH
parameters (`logsfr_ratios`); the prior band, best-fit line and SFR10/SFR100 note are gone.
Log axes use plain ticks at 1, 2, 3, 5 x 10^k. **Diagnostics:** the weight colour map is
Blues from 30 % (its white end made the lowest-weight points invisible).

**Quickstart.** Prints a boxed "DEMO SETTINGS -- THIS IS NOT A CONVERGED FIT" notice (why,
and the settings a science fit needs) at the start, a reminder after the fit, and the figure
title says so; passes `logsfr_ratios` as a truth so the injected SFH is drawn.

**Verification.** `tests/test_plotting.py`: new test that at zred = 0 the plotted photometry
lies within x2 of the plotted spectrum (fails with the old unit: checked), the axes cover the
data only, and the SFH axes are log-Gyr / log10 SFR. CI mirror in a clean clone
(`-m "not fsps and not gpu"`, no `$SPS_HOME`, the new BPASS grid): 441 passed, 0 failed;
the changed tests on the final tree 40 passed; `check_api_usage` 0; misuse 0 SILENT. The
quickstart rerun: photometry on the spectrum, chi^2/nu = 0.36.

## 2026-09-23 — GP likelihood for correlated spectral residuals (`likelihood/gp_likelihood.py`, `fit.py`)

**What.** The squared-exponential Gaussian process of `ceridwen.observation.GaussianProcess`
is now part of the compiled likelihood every sampler uses (nested sampling, NUTS, VI,
`map_fit`), no longer only `Spectrum.log_likelihood` after a fit. The residuals are whitened
by the diagonal noise model (every noise term enters σ_eff), then
`ln L = −½ rᵀK⁻¹r − ½ ln|K| − Σ ln √(2π σ_eff²)` with
`K = I + a² exp(−Δλ²/2ℓ²) + 1e-6 I` over the unmasked pixels, λ the observed-frame pixel
wavelength. New per-spectrum parameters `log_gp_amp_spec[_<obs.name>]` (ln a, a in units of
σ_eff) and `log_gp_length_spec[_<obs.name>]` (ln ℓ, observed-frame Å), sampled or fixed by
constant transforms; `Spectrum(noise=GaussianProcess(a, ℓ))` now fixes them in `fitSED`
(before, `fitSED` refused it). Off by default.

**Refused at setup** (current limitations): the GP together with the outlier mixture, upper
limits, `marginalize_elines`, `logify_spectrum` or `polynomial_order > 0` on the same
spectrum; a `GaussianProcess` object together with the names; one of the two names alone; a
GP on photometry or lines. `fitSED` warns above 1000 pixels (`fit.GP_WARN_NPIX`).

**Records.** The result file's `likelihood_json` has a `gp` block (kernel, `log_amp`,
`log_len`, `eps`, `n_pix`, sampled names). `Spectrum.chi_sq` stays the diagonal χ²; the summary
figure's χ²/ν says "diagonal, no GP" for a fit with a GP. `docs/gp_likelihood.md`,
`examples/recipes/gp_likelihood.py`.

**Verification.** `tests/likelihood/test_gp_likelihood.py` (39 passed, also without `$SPS_HOME`): the compiled value equals the independent numpy `GaussianProcess.log_likelihood` plus the normalisation to rtol 1e-10; ln a = −30 equals the diagonal Gaussian plus the analytic ε term; `check_grads`; `jit`/`vmap` = loop; every refusal; a mock with GP noise whose profile peaks at the truth and a short nested fit with a result-file round trip. New T2 category `gp_likelihood` (values = numpy to ≤ 2.2e-16 relative, gradients = finite differences to ~1e-9). GP off: T4 against 0590b54 byte-identical except the two known run-to-run items (random ln Z error estimate, a 1-ulp postprocess value); all 24 StableHLO digests of the compiled log-posterior equal. Full suite: 580 passed, 1 failed (the pre-existing `test_picket_frac_obrun_zero_equals_full_covering`, fails identically on 0590b54). `check_api_usage` 0 findings; misuse report 0 SILENT (7 new GP rows).

## 2026-09-24 — calibration polynomial marginalised analytically (`likelihood/poly_marginal.py`)

**What.** A third calibration mode for a `Spectrum`, next to the sampled
(`spectrum_scaling` / `spectrum_calib`) and the profiled (`polynomial_order > 0`) ones:
`Spectrum(polynomial_order=M, polynomial_mode="marginalize", polynomial_prior_sigma=s)`. The
coefficients of the response `1 + sum_m c_m T_m` (the profiled mode's Chebyshev basis) get the
prior `c ~ N(0, diag(s^2))` and are integrated out in closed form:
`ln L = ln L_diag(y - mu) + 1/2 b^T M^-1 b - 1/2 ln|M| - 1/2 ln|Lambda|`, `M = D^T W D +
Lambda^-1`, `b = D^T W (y - mu)`, `D = diag(mu) A` (Woodbury + determinant lemma; derivation in
`docs/dev/poly_marginalisation_design.md`). O(n k^2), no sampled dimension, and the
calibration uncertainty enters the posterior and the evidence (the profile conditions on the
best polynomial). `PolyMarginalGaussianLikelihood` (`ceridwen.likelihood`) is a separate class
built by `fit._likelihood_for`, so `fitSED`, `run_sampler`, `MultiObservationLikelihood` and
`map_fit` use it unchanged.

**Default unchanged.** `polynomial_mode="profile"` is the default; the profile and
no-polynomial paths are byte-identical to 0590b54 (T4). The profile mode's `likelihood_json`
gains `"mode": "profile"` (additive key).

**Refused, with teaching errors.** At construction: no or infinite `polynomial_prior_sigma`
(flat prior: undefined marginal), negative/NaN widths, wrong length,
`polynomial_regularization` in marginalize mode (`s = 1/reg` is its analogue),
`polynomial_prior_sigma` in profile mode (would be ignored), `polynomial_order=0`,
`logify_spectrum=True`, a GP `noise`, `marginalize_elines=True` on the same spectrum (bilinear
in the two coefficient sets). At setup (`fit._poly_marginal_for`): the outlier mixture and
upper limits on the spectrum, a sampled `spectrum_calib`, and a sampled `spectrum_scaling`
unless `T_0` is pinned (`s_0 = 0`: then the scaling is the grey level and the polynomial the
shape; with `s_0 > 0` the two enter only as a product).

**Outputs.** `PostProcess` applies each draw's conditional-mean response
(`prediction["calibration"][name]`, included in `spectra[name]`) and returns
`extras["calibration"][name]` = `mean`, `sd`, `cov` per draw and `draws` (one draw from each
conditional Gaussian). `calibrated_prediction` handles the mode. The result file records mode,
order and prior widths in `likelihood_json` (`read_result_h5(...)["obs"][name]["likelihood"]`).

**Verification.** `tests/likelihood/test_poly_marginal.py`: the closed form equals the dense
`N(y - mu; 0, C + D Lambda D^T)` (numpy slogdet/solve) to rel 1e-10 on three random masked
problems and with pinned/loose widths, conditional mean and covariance equal the dense
posterior; k = 1 equals a Gauss-Hermite integral to 1e-8; s = 0 equals the uncalibrated
likelihood (rel 1e-14); the conditional mean equals the profiled solve with `reg = 1/s`;
`check_grads` order 1 (fwd, rev) with a sampled jitter and NaN under the mask; jit/vmap equal
the loop; every refusal; a mock with an injected order-3 calibration recovered by a short nested
fit, consistent with the sampled route; result-file round trip. New regression category
`poly_marginal` (asserts the dense brute force at capture); new T4 variant `polymarg`.

## 2026-09-24 — audit fixes (night/fixes; finding IDs from the night audit)

**Behaviour changes (numbers).**
- Nested-sampling weights: BlackJAX's NaN birth likelihood of the prior-drawn initial live
  points is a birth at −∞ (`ns_weights.nested_log_weights`), so those points are weighted
  and counted; `logsumexp(log_weights)` now equals the stored ln Z (B2-016).
- Prior draws for vector parameters (per-element `Uniform`, `MultivariateNormalPrior`) have
  shape `(n, *param_shape)` in nested sampling, `map_fit` and the figures (B2-002); scalar
  priors draw exactly as before.
- `chevallard` has a finite gradient at `tau_chev = 0` (bit-identical for τ > 0) (B1-004).

**Now refused or warned (were silent).** `SedModel`: an `afe` prior outside the α grid
(B1-011); sampled `eline_scaling` with no `Lines`, `igm_factor` without IGM, `frac_obrun`
with no nebular or dust model (B1-014, B1-021); `zred < 0` without `lumdist_mpc` (B1-026);
warnings for a sampled `zred` whose prior lies after the oldest SFH node (B1-002) and for
`gas_tied` beyond the CLOUDY axis (B1-019). `upper_limit` of the wrong length (B2-006); the
plain `DiagonalGaussianLikelihood.make_lnprobfn` on flagged upper limits (B2-003);
`Spectrum.mask_lines` without `zred` warns (B1-009); a dust-law key missing from theta warns
(B1-007); the eline fast path refuses a mask changed after setup (B2-015); the import-time
switches `CERIDWEN_X64=0` / `CERIDWEN_MATMUL_PRECISION` warn (B1-023).

**Fixed.** `rebuild_model` / `check_model_against_result` accept the files `fitSED` writes
(B2-001); `resume_from` accepts its own checkpoint (ln L check at rtol 1e-6) (Q1-006);
single-observation `make_lnprobfn` honours sky and calibration (B2-003); vector `Uniform`
serialises (G2-001) and `_detect_bounds` keeps per-element bounds (B2-005); `fitSED` no
longer mutates `sampler_kwargs` (B2-004), `CSPBasis` no longer mutates `init_*_params`
(B1-015); `/map` records its settings, key and bounds (B2-011); NS diagnostic titles are
weighted quantiles (B3-001); `fetch_grid` names a pre-re-deposit cache (B3-007);
`ceridwen.check` verifies the FSPS data files (B3-002). Docs, README, tutorial, GOTCHAS,
AGENTS and examples corrected per the B3/Q1/P findings (see `git log main..night/fixes`).

## 2026-09-24 — surviving mass on young MIST SSPs; α-grid messages (Q2-003, Q2-004)

**Behaviour change (numbers).** FSPS's `stellar_mass` on MIST exceeds the formed mass below
10^6.45 yr (4.62 M_sun per M_sun formed at 10^5 yr in `mist_miles_chab`, 360 of 1391 SSPs
above 1): FSPS `IMF_WEIGHT` counts every star from 0.08 M_sun up to the first isochrone point
at that point's mass, and the young pre-main-sequence isochrones start at up to 2.68 M_sun.
`ceridwen.ssps.stellar_mass` counts that bin by its IMF mass (times the point's mact/mini) in
SSPs whose isochrone is truncated; `SSPData.from_fsps`, `SSPDataAfe.from_fsps` and
`scripts/attach_stellar_mass.py` (new `--replace`) use it. On `mist_miles_chab` 326 SSPs change,
to [0.971, 1.0036]; every SSP at >= 10^6.45 yr keeps FSPS's value bit for bit. BPASS is unchanged
(its masses come from `bpass.mass`, <= 1). Spectra are not touched.

**Now refused (was silent).** A surviving-mass table above `STELLAR_MASS_MAX` = 1.01: the
constructor and `with_stellar_mass` raise; `load` keeps the grid, drops the table, warns
once (naming the grid and `fetch_grid(..., force=True)`) and every mfrac error says why. The
first Zenodo copy of `mist_miles_chab` (sha256 `2f6777a8…`) is refused this way until its
corrected copy is deposited.

**Messages.** `display()` of a grid without a table says so without calling it old: for the
α grid, "the published grid 'amist_c3k_hr_krou_afe' has none yet: mfrac and the surviving
mass are unavailable; everything else works". The α-grid metallicity note no longer refers
to earlier versions.
