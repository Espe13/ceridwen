<!-- Audience: ceridwen developers. Design record for DEV_ROADMAP.md section 8
     (emission-line amplitude marginalisation). Phase 0 of the implementation:
     no code has been written against this document yet. -->

# Emission-line marginalisation — design (phase 0)

*2026-09-21. Status: decided (section 5.1) and implemented in
`ceridwen/likelihood/eline_marginal.py`. Sections 1-4 are the phase-0 analysis
of Prospector and stay as the record; where they propose something that 5.1
overrides (Prospector profile, 5-sigma windows, a Prospector penalty mode), 5.1
is what was built.*

Reference implementation: Prospector `2.0a2.dev42+gff8d5e4`, installed at
`/Users/amanda/opt/anaconda3/envs/prospector/lib/python3.10/site-packages/prospect`
(conda env `prospector`, Python 3.10). All Prospector line numbers below refer
to that install.

**Provenance of the install (checked against the wheel's `RECORD` hashes).**
`models/sedmodel.py`, `fitting/fitting.py` and `sources/constants.py` are
unmodified. `observation/observation.py` **is** locally modified, but only in
`Observation.rectify` (`:104-121`, handling of `flux=None`). `fitting/nested.py`,
`io/write_results.py` and `models/parameters.py` are also modified. There is a
`fitting/fitting_mine.py` and a sibling `prospect_tweaked_version/` package. None
of the modifications touches the emission-line code path.

Every numerical statement below was computed this session. The probe scripts are
in the session scratchpad and are not committed, and each result is quoted where
it is used.

---

## 1. What Prospector does, re-verified

"Confirmed" means read in the function body and, where marked, also run.

| # | Claim in the task brief | Verdict | Evidence |
|---|---|---|---|
| 1 | Analytic lines only when `add_neb_emission` and not `nebemlineinspec` | Confirmed. The `nebemlineinspec` default is **True** (`params.get("nebemlineinspec", True)`), so it must be set to False explicitly. | `sedmodel.py:454-460`, used at `:549`, `:643-644` |
| 2 | `marginalize_elines` fits every line in `emlines_info.dat` unless `elines_to_fit`/`elines_to_fix` say otherwise | Confirmed. `_fit_eline = isin(all, to_fit) & ~isin(all, to_fix)`, `_fix_eline = ~_fit_eline`. Run: all 166 lines are fitted by default. | `:593-604` |
| 3 | `elines_to_ignore` drops lines entirely | Confirmed, and it goes further: ignored lines leave the **photometry** too (`nebline_photometry` uses `_use_eline`). | `:606-610`, `:573`, `:486-490` |
| 4 | Fixed lines are added before the calibration step | Confirmed. Fixed lines are painted **only inside the union of their own 5σ windows** (`_fix_eline_pixelmask`), with the same trapz-renormalised profile as fitted lines. | `:292-299`, `:578`, `:715-733` |
| 5 | `use_eline_prior(s)` is dead; the prior is on iff `eline_prior_width > 0` | Confirmed. `use_eline_priors` appears only in `_available_parameters`, and `use_eline_prior` only in `templates.py:297,329`. Neither is ever read. The branch test is `np.any(sigma_alpha_breve > 0)`. | `:56`, `:144-145`, `:682` |
| 6 | Covariance `diag((w · eline_lum)^2)`, mean = CLOUDY luminosity | Confirmed. **Consequence:** a line whose CLOUDY luminosity is 0 gets zero prior width, so with the prior on it is pinned at 0. | `:144-145`, `:660`, `:680`, `:687` |
| 7 | Centres `(1 + zred + eline_delta_zred) λ` | Confirmed. The shift is **additive in z**, not `(1+z)(1+dz)`. | `:138-139` |
| 8 | Widths `hypot(eline_sigma, σ_inst)`, default 100 km/s | Confirmed. With `obs.resolution is None` there is **no** instrumental term. | `:557-564` |
| 9 | Lines not passed through the instrumental kernel; painted after smoothing | Confirmed | `:289-320` |
| 10 | `obs.resolution` is σ in km/s | Confirmed from code, not only from the docstring: it enters `Kdelta = sqrt(res² − lib²)`, converted with `/CKMS*λ`. | `observation.py:447-452` (docstring `:336-338`) |
| 11 | A line is valid if any unmasked pixel lies within 5σ; `emask` = union of fitted windows; decided per call | Confirmed. The window is **linear in λ**: `|λ − λ0| < 5 λ0 σ/c`. | `:570-580` |
| 12 | Profiles: velocity Gaussian × dv/dν, trapz-renormalised on the `emask` sub-grid | Confirmed, with two details the brief did not give (below). | `:735-771` |
| 13 | `delta = obs.flux − calibrated_spec` on `emask`; weights `1/unc²` only | Confirmed. The noise model is not used (`spec_unc = None`, FIXME at `:314-318`). | `:653`, `:664-670` |
| 14 | `Σ̂ = pinv(AᵀN⁻¹A)`, `α̂ = Σ̂ AᵀN⁻¹δ` | Confirmed | `:674-675` |
| 15 | `linecal = flux_norm/(1+z) · response(λ_line)` | Confirmed, but the response used is **stale** (below). | `:656-659` |
| 16 | K with and without prior | Confirmed. `ln_mvn` uses `slogdet` and `pinv(rcond=1e-12)`. | `:680-704`, `:1124-1143` |
| 17 | `lnp_eline` added in `lnprobfn` | Confirmed: `lnp = lnp_prior + Σ lnp_data + lnp_eline`. | `fitting.py:107-113` |
| 18 | χ² uses the ML amplitudes even with a prior | Confirmed: it returns `alpha_hat * eline_gaussians`. | `:713` |
| 19 | Stored `_eline_lum` (ᾱ), `_eline_lum_var` (Σ̄), `_eline_lum_mle` (α̂) | Confirmed, **but `_eline_lum_var` is never initialised anywhere in the package** (below). | `:707-712` |
| 20 | Optimised calibration is fit only on the near-line pixels | **Confirmed by running it**: `PolyOptCal.compute_response(spec=1, extra_mask=near_line)` on data = 1 away from a line and 3 on it returns a median response of **3.0**. Excluding the line pixels would give 1.0. | `sedmodel.py:302-307` → `observation.py:601` (`mask = self.mask & extra_mask`); docstring `:582-591` says the opposite |

### 1.1 Findings the brief did not have

**P1. `fit_mle_elines` cannot run on a freshly built `SpecModel` as installed.**
This was run with the attributes set by hand (probe `probe_prosp.py`):

- no `_speccal` → `AttributeError: 'SpecModel' object has no attribute '_speccal'`.
  It is read at `:658` but first assigned at `:324`, *after* the fit.
- `_speccal = 1.0` → `TypeError: 'float' object is not subscriptable`. A plain
  `Spectrum.compute_response` returns the **scalar** `1.0` when `response is None`
  (`observation.py:462-466`). So "use a plain Spectrum (response = 1)" in the
  brief crashes: it needs `response=np.ones(n_pix)`.
- `_speccal = ones` → `AttributeError: ... '_eline_lum_var'`. It is written at
  `:709` and never created.
- with both set → it runs (`K = −193.98` on the probe).

**P2. `calib_factor` uses the previous call's response.** `_speccal` is updated
at `:324`, after `fit_mle_elines` has used it at `:658`. For a fixed response
array this is harmless. With `PolyOptCal` the prior mean and the reported
physical fluxes use the polynomial from the previous θ.

**P3. Profile constants.** `lightspeed = 2.998e18`, `ckms = 2.998e5`
(`sources/constants.py:17-18`), and the renormalisation integrates over
`3e18/λ` (`:769`). Measured with probe `probe_num.py`, which uses 2 px per σ_inst
and a 5σ window:

| R | σ_line [km/s] | ∫ profile dν (true c), trapz-normalised | ∫ raw profile dν (true c) | max \|Prospector − CERIDWEN painter\| / peak |
|---|---|---|---|---|
| 1000 | 162.0 | 0.999308 | 0.999975 | 7.99e-4 |
| 2700 | 110.6 | 0.999308 | 0.999975 | 7.42e-4 |
| 100 | 1277.9 | 0.999308 | 0.999976 | 2.23e-3 |

So Prospector's fitted amplitudes are 0.069 % above the true integrated flux
(2.99792458/3). The raw profile is a Gaussian in **linear** velocity `c(λ/μ−1)`
(`:761`), not in ln λ. The shape difference against CERIDWEN's ln λ painter
(`broadening.py:374-393`) is 7e-4 to 2e-3 of the peak. A `1e-10` equivalence
therefore needs a Prospector-exact profile in CERIDWEN (its constants,
linear-velocity Gaussian, trapz over `3e18/λ`). It cannot use the existing
painter.

**P4. The profile is evaluated on the whole `emask` union.** `get_eline_gaussians`
is called with `wave = _outwave[emask]` for all fitted lines (`:649-650`). Each
column is therefore non-zero on every pixel of the union, not only in its own
window, and is normalised over the union. The trapz runs across masked gaps and
across gaps between separate windows.

**P5. Photometry and `Lines` see the fitted fluxes, depending on order.**
`_eline_lum[idx]` is overwritten with **ᾱ** (`:707`). `predict_phot` →
`nebline_photometry` (`:486-490`) and `predict_lines` (`:378`) read
`_eline_lum`. The observations are predicted in list order (`:110`), and the
comment at `:108-109` says spectra must come first. With the photometry listed
first, it sees the CLOUDY fluxes. With the prior off, ᾱ = α̂.

**P6. Several spectra.** K accumulates (`:704`). A second spectrum uses the
first one's ᾱ as its prior mean (`_eline_lum` was overwritten), but the
**original** covariance (`_eline_lum_covar` is not updated).

**P7. Other Prospector defects seen on the way (not on our path):**
- `SplineOptCal.compute_response` always returns ones: `~splineopt` with
  `splineopt=True` is `-2`, which is truthy (`observation.py:684-687`).
  `__init__` sets `(wlo, whi) = w.min(), w.min()` (`:659`), so it divides by zero.
- `PolyFitCal.__init__` calls `super(SplineOptCal, self).__init(...)` (`:729`).
- `predict_intrinsic` has `if emask.any() (~continuum_only):` (`sedmodel.py:216`),
  which calls a bool and raises `TypeError`.

**P8. FSPS does not initialise in the `prospector` env with
`SPS_HOME=/Users/amanda/Prospector/fsps`.** `CSPSpecBasis(zcontinuous=1)` aborts
with `Fortran runtime error: End of file` at `sps_setup.f90:265` (unit 91).
python-fsps is 0.4.7, and the FSPS checkout is at `cbcd0ee`. The end-to-end
comparison (test 2b) cannot run until this is fixed (**D12**).

**P9. Conditioning of the default line set.** He I 3888.63 and H8 3889 are
32.4 km/s apart, and [O II] 3867 / [Ne III] 3869 are 136 km/s apart. These are
the only pairs closer than 150 km/s between 3000 and 7000 Å. At R = 1000, z = 3,
fitting the 3820–3920 Å rest window (5 lines) gives cond(AᵀN⁻¹A) = 2.0e2 and a
finite, stable K. Cholesky can handle that, but at lower R it will get worse.

---

## 2. The with-prior penalty is the exact Gaussian marginal

Take the pixels in `emask`, with diagonal noise `N` and a linear model
`m(α) = c + Aα`, where `c` is the calibrated continuum plus fixed lines. With
`δ = y − c`:

    χ²(α) = (δ − Aα)ᵀ N⁻¹ (δ − Aα) = χ²(α̂) + (α − α̂)ᵀ Σ̂⁻¹ (α − α̂)

where `Σ̂ = (AᵀN⁻¹A)⁻¹` and `α̂ = Σ̂ AᵀN⁻¹δ`. The expansion is exact because χ²
is quadratic and its gradient vanishes at α̂. Pixels outside `emask` do not
depend on α, since the fitted lines are added only on `emask` (`:320`). Their χ²
factors out.

**With a Gaussian prior** `p(α) = N(α; ᾰ, Σ̆)`:

    ∫ e^{−χ²(α)/2} N(α; ᾰ, Σ̆) dα
      = e^{−χ²(α̂)/2} ∫ e^{−(α−α̂)ᵀΣ̂⁻¹(α−α̂)/2} N(α; ᾰ, Σ̆) dα
      = e^{−χ²(α̂)/2} · (2π)^{n/2}|Σ̂|^{1/2} · ∫ N(α; α̂, Σ̂) N(α; ᾰ, Σ̆) dα
      = e^{−χ²(α̂)/2} · (2π)^{n/2}|Σ̂|^{1/2} · N(α̂; ᾰ, Σ̂ + Σ̆)

The last step is the Gaussian product integral. Since
`ln N(α̂; α̂, Σ̂) = −½(n ln 2π + ln|Σ̂|)`, the log of the marginal is

    −½ χ²(α̂) + [ ln N(α̂; ᾰ, Σ̂+Σ̆) − ln N(α̂; α̂, Σ̂) ]  =  −½ χ²(α̂) + K_prior

This is exactly Prospector's `:695-696` together with χ² at α̂ (`:713`). The
Gaussian normalisation `−½ Σ ln(2π σ_i²)` is common to both sides and is kept in
the CERIDWEN kernel as usual. The identity requires Σ̂ to exist (A of full
column rank on the valid lines) and holds for any PSD Σ̆. The posterior is
`N(ᾱ, Σ̄)` with `Σ̄ = Σ̂(Σ̂+Σ̆)⁻¹Σ̆` and `ᾱ = Σ̆(Σ̂+Σ̆)⁻¹α̂ + Σ̂(Σ̂+Σ̆)⁻¹ᾰ`,
matching `:690-693`.

**Without a prior (flat on ℝⁿ, improper):**
`∫ e^{−χ²/2} dα = e^{−χ²(α̂)/2} (2π)^{n/2} |Σ̂|^{1/2}`, so the correct term is
`K_flat = +½(n ln 2π + ln|Σ̂|)`, up to the constant prior volume. Prospector
uses `K_prosp = −½(n ln 2π + ln|Σ̂|)` (`:701`), which has the opposite sign.

**How much it matters (measured, probe `probe_penalty.py`).** R = 1000, z = 3,
8 fitted lines (Hδ, Hγ, Hβ, [O III]4959/5007, [O II]3726/3729, [Ne III]),
σ_pix = 0.05 on a unit continuum:

- `zred` ∈ [2.99, 3.01], `eline_sigma` = 100: K_prosp spans −187.094 … −187.054,
  so the two modes differ by at most **0.078 nats** over the interval. This is
  harmless for a sampled redshift at this width.
- `eline_sigma` = 30, 50, 100, 200, 400 km/s at z = 3:
  K_prosp = −186.12, −186.32, −187.07, −188.85, −191.62, and K_flat is the
  negative of each. Their difference grows from 372.25 to 383.25, i.e.
  **11.0 nats** over the range. With a sampled `eline_sigma`/`sigma_gas`,
  Prospector's term pushes the width down and the correct term pushes it up,
  by about one nat per line per e-fold in σ.
- When the number of valid lines changes (lines entering or leaving the window
  as `zred` moves), both terms jump by `n ln 2π`, with opposite signs. The flat
  prior's volume constant is then ill-defined as well.

**Limits for test 3.** (b) As `eline_prior_width → 0⁺`, the **marginal
ln-likelihood** → χ² evaluated at ᾰ, i.e. lines fixed at the grid prediction.
The *painted spectrum* does **not** tend to the grid lines, because Prospector
paints α̂ at every width. Test 3(b) must therefore compare ln L and ᾱ, not the
spectrum. (c) As `Σ̆ → ∞`, `K_prior − K_flat → −(n/2) ln 2π − ½ ln|Σ̆|`, which is
the log of the height of the Gaussian prior at its peak, i.e. the prior volume.
And `K_prior − K_prosp → (n/2) ln 2π + ln|Σ̂| − ½ ln|Σ̆|`.

---

## 3. Prospector → CERIDWEN term map

| Prospector | CERIDWEN | Where |
|---|---|---|
| `obs.wavelength` / `_outwave` (vacuum Å, observed) | `Spectrum.wavelength` (vacuum Å, observed) | `spectrum.py:113-122`; `docs/conventions.md` units table |
| `obs.flux` | `obs.flux − obs.sky` | `runner.py:137-141` |
| `obs.unc` | `obs.uncertainty` | `runner.py:144` |
| `obs.mask` (True = use) | `obs.mask` (True = use) | `spectrum.py:19` |
| `obs.resolution` (σ, km/s, per pixel) | `Instrument.sigma_kms(arr, wave=obs_wave)`, the value returned by `sigma_kms_at`. **`res_convention` no longer exists**; `Spectrum(res_convention=...)` raises `TypeError`. | `broadening.py:107-129`; `spectrum.py:50-57` |
| `sps.get_galaxy_elines()` wavelengths (rest, vacuum) | `csp.neb.nebem_line_pos`, read from the `.lines` cube and cross-checked against `emlines_info.dat` within 1 Å | `NebularGridModel.py:303-349`, `:211-244` |
| line names (`emline_info['name']`) | not loaded by CERIDWEN. Must be read from `emlines_info.dat` and matched **by wavelength** to cube rows, as `_neb_cube_rows_for` does. | `csp.py:824-853` |
| `_ewave_obs = (1+z+dz) λ` | painter centre `ln λ_rest + ln(opz)`. `eline_delta_zred` would be new (additive, to mirror Prospector). | `broadening.py:408-411` |
| `eline_sigma` (km/s, σ) | `Kinematics.sigma_gas` (fixed float or theta key; `TIED` to `sigma_gal` by default) — **D2** | `broadening.py:142-231` |
| `σ_line = hypot(eline_sigma, σ_inst(λ_line))` | `sqrt(σ_gas² + s_inst²)`, with `s_inst = interp(λ_line, wave_obs, σ_inst table)`. With no `Instrument`, CERIDWEN uses a half-pixel floor where Prospector uses 0. | `broadening.py:384-389`, `:551-555` |
| `ckms = 2.998e5`, `lightspeed = 2.998e18`, `3e18` | `CKMS = 2.99792458e5`, `C_AA_S = 2.99792458e18` | `broadening.py:36-37` |
| unit profile (lin-velocity Gaussian × dv/dν, trapz-renormalised on `emask`) | `phi(ln λ)·λ/c`, analytic unit area in ln λ → **needs a Prospector-exact mode** | `broadening.py:374-393` |
| `eline_lum · flux_norm/(1+z)` (maggies·Hz) | `csp.predict_line_fluxes(theta, for_spectrum=True)`: observed-frame integrated flux in erg s⁻¹ cm⁻² (spectrum unit × Hz), including dust, `frac_obrun`, IGM at the line, mass and distance, and **no** `eline_scaling` | `csp.py:880-945` |
| `response` / `_speccal` | `spectrum_calibration_factor(obs, theta)` (`spectrum_scaling · (1+Σ c_k P_k)`), applied in `_project_observations`, times the fixed `obs.calibration` vector applied in the runner | `spectrum_calibration.py:47-72`; `csp.py:785-790`; `runner.py:150-152` |
| `calibrated_spec` (continuum + fixed lines, × response) | `obs.predict(spectrum_slit, wave, line_flux_with_fitted_zeroed, theta) · calib · obs.calibration` | `csp.py:785-790`, `runner.py:150-152` |
| continuum passed to `fit_mle_elines` has no fitted lines (`nebemlineinspec=False`) | Already true for every CERIDWEN `Spectrum`: the projector receives `spectrum_slit` = the `include_lines=False` continuum, and lines are painted from fluxes on the observed pixels. Fitted lines only need to be dropped from `line_flux`. | `csp.py:690-697`, `:700-706`, `:785` |
| noise in the α solve: `unc²` only | `DiagonalNoiseModel.compute` (`noise_floor`·\|μ\|, `log_err_scale`, `log_jitter`, `log_f_calib`, `log_f_data`) — **D5** | `noise_model.py:89-133`; `fit.py:202-226` |
| `lnp_eline` added to ln L | must be added inside `run_sampler.loglike_fn` and `MultiObservationLikelihood.make_lnprobfn` | `runner.py:146-160`; `likelihood.py:342-359` |
| `_eline_lum` (ᾱ), `_eline_lum_var` (Σ̄), `_eline_lum_mle` (α̂) | new derived quantities: physical (uncalibrated) fluxes `α/c(λ_line)`, their diagonal variances, and the ML fluxes, per draw → `postprocess.py`, `fit.py` HDF5 | new; neither file has any eline code today |

### 3.1 Corrections to the task brief (CERIDWEN side)

- **C1.** `Spectrum(res_convention=...)` and `resolution=` were removed in the
  2026-09-03 broadening pass (`spectrum.py:50-57`). Widths come from
  `Instrument.<unit>` (`broadening.py:44-129`), where the unit and the σ/FWHM
  convention are the constructor name.
- **C2.** `csp/spectrum_calibration.py` has **no** analytic Legendre solve.
  `spectrum_calib` coefficients are sampled theta keys (`:47-72`). The only
  analytic calibration solve is `Spectrum.fit_polynomial_calibration`
  (`spectrum.py:280-307`), a post-hoc NumPy Chebyshev `lstsq`. It is not
  traceable.
- **C3.** `Spectrum.log_likelihood` / `chi_sq` (`spectrum.py:242-278`) cast to
  `float()`, so they are not traceable and are not the sampled likelihood. The
  sampled path is `run_sampler.loglike_fn` (`runner.py:146-160`):
  `model.predict` → `× obs.calibration` → `DiagonalNoiseModel` →
  `lnlike_diag_gaussian`.
- **C4.** The Lines-without-`add_neb` check that the brief cites
  (`csp.py:755-760`) runs at **trace time**, inside `_project_observations`, not
  at construction. The new guard should go earlier, in
  `SedModel.setup_observations` (`model.py:178-204`), which runs at construction.
- **C5.** CERIDWEN's `nebemlineinspec` (`csp.py:100`, `:239`) only sets the
  default of `get_spectrum(include_lines=...)`. The prediction path always asks
  for `include_lines=False` for the spectrum (§3 table). There is no
  double-counting to prevent in the spectrum, and no user switch to demand.
- **C6.** Test 3(b) must compare ln L and ᾱ, not the painted spectrum (§2).
- **C7.** Test 2(a) with a "plain Spectrum" must pass `response=np.ones(n)`, and
  the reference script must set `_speccal` and `_eline_lum_var` itself (P1).

---

## 4. Proposed CERIDWEN design (pending the DEFERRED answers)

**Static shapes, exact per-call selection.** At setup, the *candidate* fitted
set is: the user's selection, ∩ lines the projector keeps for some z in range at
`sigma_max` (`broadening.py:544-549`), ∩ lines with at least one unmasked pixel
within 5σ for some allowed (z, σ). Everything else is dropped at setup. Per
call, Prospector's dynamic selection is reproduced **as values, not shapes**:

- window weights `w_ij = [|λ_i − μ_j| < 5 μ_j σ_j / c] ∧ mask_i`, traced;
- line validity `v_j = any_i w_ij`;
- pixel set `e_i = any_j (v_j ∧ w_ij)`;
- invalid lines get a unit diagonal and zero right-hand side in AᵀN⁻¹A, and are
  excluded from `n` and from `ln|Σ̂|`;
- Prospector's trapz over the non-contiguous sub-grid is computed on the full
  grid with a "next selected pixel" index from a reverse cumulative minimum
  (`lax.cummin`), so the shape stays static.

This is Prospector-exact at every θ, with no static-window approximation.
(n_pix × n_cand) is small: n_cand is around 10–20 in the rest optical.

**Solve.** `jax.scipy.linalg.cho_factor/cho_solve` on the n_cand × n_cand
system, with `ln|Σ̂| = −2 Σ ln diag(L)`. No explicit inverse and no `pinv`.

**Profiles.** `eline_profile="prospector"` gives the linear-velocity Gaussian,
Prospector's constants and trapz on the sub-grid. The option `"analytic"` uses
the existing ln λ painter's unit-area profile over all unmasked pixels. The
default is `"prospector"`, as the brief requires. **D8** is about gradients.

**Units.** α is in calibrated data units × Hz, as in Prospector. The prior mean is
`ᾰ_j = F_grid,j · c(λ_j)`, where `c = spectrum_calibration_factor ·
obs.calibration`, interpolated at the line centre. The reported physical flux is
`α_j / c(λ_j)` in erg s⁻¹ cm⁻², the `Lines` unit. The calibration is taken from
the **current** θ, which fixes P2 on our side; see D7.

**Where the kernel lives: D4.** The recommendation is the likelihood layer,
where y, σ, the mask, `sky`, `obs.calibration` and the noise model already
meet. `SedModel` would expose the per-spectrum line inputs (the grid fluxes of
the candidate lines and the traced (z, σ_line) that set A) through an auxiliary
channel next to `predict`. The runner and `MultiObservationLikelihood` add K.

**Guards (construction/setup, never in a kernel).**
- `marginalize_elines=True` with `csp.neb is None` raises a `ValueError` in
  `SedModel.setup_observations` naming `add_neb=True`, and gets a misuse-report
  row.
- Unknown line names, and names in both `elines_to_fit` and `elines_to_fix`,
  raise.
- The combinations with `logify_spectrum=True`, a GP `noise`, `upper_limit`
  flags and (pending D5) `noise_floor` raise.
- An empty candidate set raises, with the observed window in the message.
- `eline_sigma` units: see D2.

**Default off = bit-identical.** With `marginalize_elines=False`, no new code
runs on the prediction path: no new theta keys are read, and the painter,
projector and runner are unchanged. T4 against `1dd128f` will verify this.

---

## 5. DEFERRED — decisions needed from Amanda

Each item has a recommendation, but none will be implemented until you answer.

- **D1. No-prior penalty default.** `"prospector"` (−½(n ln 2π + ln|Σ̂|), the
  sign error) or `"marginal"` (+½(…), the correct flat-prior marginal). Both
  will be implemented; the question is only the default. Measured: the two
  differ by ≤ 0.08 nats over a ±0.01 z range, and by 11 nats over
  `eline_sigma` 30–400 km/s with 8 lines. *Recommend `"marginal"`, with
  `"prospector"` available for reproduction and the equivalence tests.*
- **D2. `eline_sigma` vs `Kinematics.sigma_gas`.** Prospector has a separate
  `eline_sigma`. In CERIDWEN the line width is `sigma_gas`, and GOTCHAS §7 /
  `conventions.md` promise "three widths, each set once". Options:
  (a) no `eline_sigma`: the fitted lines use `sigma_gas`, and `eline_sigma=`
  raises a teaching error pointing to `Kinematics(sigma_gas=...)`;
  (b) `eline_sigma` as a separate width for fitted lines only. *Recommend (a).*
- **D3. `eline_delta_zred`.** Should it shift the fitted lines only, or also the
  fixed lines (Prospector shifts all lines, `:139`)? Additive
  `(1+z+dz)` as in Prospector? *Recommend all painted lines, additive; only when
  the key is present, so the default is unchanged.*
- **D4. Where the kernel lives.** (i) Inside `_project_observations`, which
  returns the spectrum with ML lines, with K passed through a reserved
  prediction key; or (ii) in the likelihood layer, with an auxiliary channel from
  `SedModel`. *Recommend (ii).* Both touch `runner.py`,
  `likelihood.py:MultiObservationLikelihood.make_lnprobfn`, `postprocess.py` and
  `fit.py`.
- **D5. Noise in the α solve.** Prospector uses `unc²` only. For K + χ² to be
  the exact marginal, CERIDWEN's weights must equal the likelihood's `inv_var`.
  `log_err_scale`, `log_jitter` and `log_f_data` do not depend on μ, so they are
  exact. `noise_floor` and `log_f_calib` use |μ|, which includes the lines, so
  they are circular. Options:
  (a) Prospector-exact: `unc²` in the solve, whatever the χ² uses (then not a
  true marginal when noise terms are on);
  (b) `inv_var` from the noise model evaluated at the line-free model
  (exact for the μ-independent terms, and a documented approximation for the
  floor);
  (c) refuse `noise_floor`/`log_f_calib` with marginalisation.
  *Recommend (b), with (a) selected automatically in `"prospector"` penalty
  mode.*
- **D6. Photometry and `Lines` after marginalisation.** Prospector feeds ᾱ to
  photometry and Lines, but only when the Spectrum is listed first (P5).
  Options: (a) photometry and `Lines` keep the grid fluxes (order-independent);
  (b) the fitted fluxes replace the grid fluxes there, Prospector-style,
  order-independent in CERIDWEN. *Recommend (a) as the default, with (b) as an
  option, because (b) makes a joint `Lines` + `Spectrum` fit use the spectrum's
  lines twice.*
- **D7. The calibration inside `linecal`.** Use the current θ's
  `spectrum_scaling`/`spectrum_calib`/`obs.calibration` (correct), not the
  previous call's (Prospector P2). *Recommend current θ.* This does not change
  the equivalence tests, which use a fixed response.
- **D8. Discontinuities.** Window membership and the trapz renormalisation are
  piecewise constant in `zred` and `sigma_gas`, so ln L has small jumps. Nested
  sampling doesn't care, but NUTS gradients see only the smooth part.
  Options: keep `"prospector"` profile/windows as the default (as the brief
  says), and document that `"analytic"` (all unmasked pixels, analytic
  normalisation, smooth in θ) is the choice for NUTS with sampled z or σ.
  *Recommend that.*
- **D9. Several marginalising spectra.** Prospector chains them (P6).
  *Recommend: each spectrum is marginalised independently against the grid
  prior, with K summed, and the difference documented. Or refuse more than one
  in v1. Which?*
- **D10. `Instrument=None`.** Prospector puts no instrumental term in σ_line.
  CERIDWEN's painter uses a half-pixel floor (`broadening.py:551-553`).
  *Recommend keeping CERIDWEN's floor, and requiring an `Instrument` for the
  equivalence tests.*
- **D11. Near-degenerate lines.** Cholesky on, e.g., He I 3889 + H8
  (cond ≈ 2e2 at R = 1000; worse at lower R). *Recommend a setup-time condition
  check at the reference θ that raises above 1e10 and suggests `elines_to_fix`
  for one of the pair, with no silent jitter.*
- **D12. Reference environment.** FSPS aborts in the `prospector` env with
  `SPS_HOME=/Users/amanda/Prospector/fsps` (P8). Test 2(a) needs no FSPS and can
  run now. Test 2(b) needs either a working `SPS_HOME` for python-fsps 0.4.7
  (which path?), or the continuum taken from somewhere else. Also: may the
  reference script **monkeypatch** Prospector at runtime (initialise `_speccal`
  and `_eline_lum_var`) without editing the install? Without that, 2(b) cannot
  run at all (P1).
- **D13. Production batch width.** CLAUDE.md §5 requires the timing at
  production W, and the brief points to "the RAC19 timing notes". CLAUDE.md §6
  says to ask before touching `RAC19/`. May I *read* those notes, or will you
  give me W?
- **D14. Physics that differs from FSPS and matters only for the prior mean /
  fixed lines** (dust on lines, IGM on lines: CERIDWEN applies IGM at the line,
  `csp.py:934-941`, while FSPS `get_galaxy_elines` does not). *Recommend:
  accept, and state it in the docs. It does not enter test 2(a).*

### 5.1 Decisions (Amanda, 2026-09-21)

Guiding principle: implement the **correct** solution, not a reproduction of
Prospector. Prospector is a reference for the with-prior algebra only.

| # | Decision |
|---|---|
| D1 | Correct flat-prior marginal `+½(n ln 2π + ln|Σ̂|)`. **No** Prospector-sign mode. The equivalence test compares α̂, Σ̂, ᾱ, Σ̄ and the with-prior K (which is exact in both codes); for the no-prior K it asserts `K_ceridwen = −K_prospector` analytically. |
| D2 | Width = `Kinematics.sigma_gas`; no `eline_sigma`. |
| D3 | As Prospector: centres `(1 + z + eline_delta_zred) λ`, additive, applied to every painted line. |
| D4 | Likelihood layer, optimised for GPU; timed on Tursa at production W. |
| D5 | (b): `inv_var` of the noise model evaluated on the line-free model. |
| D6 | Joint marginalisation: one set of line fluxes shared by the marginalising Spectrum, every Photometry (static line-to-band basis, `(1+z) T @ profiles`) and every Lines observation (blend matrix x `eline_scaling`). Only allowed when the nebular parameters are **not sampled** (constant or derived transforms are fine): with free line fluxes the lines cannot constrain them. Fluxes are physical (erg s^-1 cm^-2, observed, IGM-transmitted at the line), so the prior mean is the CLOUDY flux itself. Photometry through painted lines (sampled zred, or sampled sigma_gas with broaden_photometry) is refused (NotImplementedError) for now. |
| D6b (2026-09-22) | `CSPBasis_afe` (no nebular grid) is supported with a **flat prior only**: the line list, rest wavelengths and names come from `$SPS_HOME/data/emlines_info.dat` and the widths from σ_gas and the instrument, none of which depends on [α/Fe] or Z; the photometric columns use `line_profiles_on_grid` (== `NebularModel.line_profiles`). `eline_prior_width > 0`, `elines_to_fix` and `Lines` observations are refused there. Extended the same day (Amanda) to every basis without a nebular model, including `CSPBasis(add_neb=False)`: this supersedes D6's "nebular model required". |
| D7 | Current θ's calibration. |
| D8 | Superseded (5.2): the painter's analytic profile on all unmasked pixels, no windows, no trapz renormalisation; lines are fitted when their centre lies inside the spectrum (at every z of the projector) with >= 3 unmasked pixels within 2 sigma and > 3 sigma from the edges. |
| D9 | At most one marginalising Spectrum (ValueError otherwise). |
| D10 | `Instrument` required (setup-time `ValueError`). |
| D11 | Setup-time condition check; raise above 1e10. |
| D12 | Done: python-fsps 0.5.0 built from source for x86_64 in a throwaway env (conda-forge gfortran), Fortran runtime bundled with `delocate`, installed into `prospector` with `--no-deps`; `pip freeze` changes in that one line. Default spectral library of this build: `c3k_lr`. Diagnosis: python-fsps 0.4.7 bundles FSPS `82a8735`, whose compiled grid sizes predate checkout commit `1c9d876` (MILES/BaSeL logT/logg grids changed) → EOF reading `SPECTRA/BaSeL3.1/basel_logt.dat` (`sps_setup.f90:265` of the bundled source). The env is x86_64 (Rosetta); python-fsps 0.5.0, which reads this data (verified in `ceridwen311`), has no macOS x86_64 wheel. |
| D13 | W from code, not notes: BlackJAX NSS vmaps the MCMC kernel over `num_delete` chains (`blackjax/ns/from_mcmc.py:106-107`); `num_delete = num_live // 5` (`nested.py:158-160`), `fitSED` default `num_live = 500` (`fit.py:163-167`) → **W = 100** per step, 500 once at initialisation (`nss.py:439`). |
| D14 | Keep CERIDWEN's grid fluxes (attenuated, IGM at the line); Ly-alpha always has a flat prior. Facts: FSPS line luminosities are dust-attenuated (`add_dust.f90:109-110`, from `csp_gen.f90:285`) and carry no IGM (IGM only on `spec_csp`, `compsp.f90:144-146`). CERIDWEN's grid fluxes are attenuated (age-dependent + diffuse, `csp.py:907-926`) and **do** carry IGM at the line (`csp.py:934-941`). |

### 5.2 D1 vs D8

D8 (keep Prospector's 5σ windows and trapz renormalisation as the default) was
recommended only so that the default could reproduce Prospector. With D1 that
reason is gone, and both features are approximations of the right answer: the
windows make ln L discontinuous in `zred`/`sigma_gas`, and the renormalisation
uses `3e18` for c. The correct choice is the analytic profile of the existing
painter (unit area, true c, the same profile the fixed lines use) on all
unmasked pixels inside a static ±Nσ_max band, which is smooth in θ and
consistent with the rest of the spectrum. Confirmed by Amanda and implemented.

---

## 6. Phase plan (after the answers)

1. **core**: kernel module (`ceridwen/likelihood/eline_marginal.py`); `Spectrum`
   options and validation; line-name map; wiring in `SedModel`, the runner and
   `MultiObservationLikelihood`; guards.
2. **tests**: brute-force quadrature (1); Prospector kernel equivalence
   (2a now, 2b after D12); limits (3); jit/vmap/grad (4); guard (5).
3. **baselines**: a new regression category with the prior on and off (§4 of
   CLAUDE.md), captured after tests 1–2a pass, justified by them. No existing
   baseline is touched.
4. **figures/docs/example**, then the injection–recovery experiment. The GPU
   question goes to Amanda before anything is submitted.
