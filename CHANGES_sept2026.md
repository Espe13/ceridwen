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
