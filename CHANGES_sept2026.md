# ceridwen — changes (2026‑09‑07 → 2026‑09‑09)

Changes made to the **ceridwen** package during the MIST/MILES JADES refit work,
with rationale and verification. Two themes: (1) **speed** — undo a forward
regression and then make the forward faster; (2) **correctness** — fix an SED
flux-unit mislabel and two JAX tracer-leak bugs. All changes preserve the public
API and give bit-comparable science (identical/near-identical logZ).

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
and safe as jax arrays.

**Impact / verification:** cold two‑trace `jit(vmap(predict))` test → no leak;
`_neb_cube_rows` is an `ndarray`; vmap forward finite; line/predictive/misuse
tests pass. Identical numerical results (only the cache backend changed).

---

## 3. API / configuration

### 3.1 Removed vestigial `tuniv` argument  (`csp/csp.py`)
**What:** dropped the `tuniv` constructor parameter (and `self.tuniv`); `describe()`
now reports `self.age_at(0.0)`. The forward already used `age_gyr(z, self.cosmo)`.
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
stored like any other parameter.

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
