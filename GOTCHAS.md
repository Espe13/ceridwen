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

## 1. The metallicity units trap (most dangerous)

`theta["Z"]` (constant-Z mode) and `theta["zh"]` (time-varying mode) are
searched **directly** into `SSPData.ssp_lgmet`, i.e. they must be in the **same
units as the SSP grid: `log10` of the absolute metallicity** (`ssp_lgmet`),
**not** `log10(Z/Zsun)`.

- The grid for `ssp_data.h5` is roughly `[-4.35, -1.35]`; solar is ~`-1.85`
  (`log10(0.0142)`), **not `0.0`**.
- Passing `Z = 0.0` (a natural "solar" guess if you read it as `Z/Zsun`) is
  **outside the grid** and is **silently clamped** to the maximum grid node.
- **Guard:** `CSPBasis.__init__` now calls `check_param_ranges(theta_init)` and
  warns at construction; call `csp.check_param_ranges(theta)` yourself on sampled
  `theta`/bounds before a fit. (The class docstring's "log10 Z/Zsun" comment is
  misleading and should be read as "grid log10-metallicity units".)

## 2. `theta` is a dict — typos are silently ignored

A mistyped key (`logmas` for `logmass`, `gas_logU` for `gas_logu`, `dust2` for
`diffuse_tau_kc`, …) is simply not read, so the parameter silently takes its
default. **Guard:** `predict()` / `get_spectrum_components()` now emit a
`warning` listing unrecognized keys (computed from the dict *keys* at trace
time → zero hot-path cost). Direct `get_spectrum(theta)` calls bypass this —
prefer `predict`/`get_spectrum_components`.

## 3. Metallicity-mode ↔ key mismatch

- `zh_const=True` needs `theta["Z"]` (shape `(1,)`); `zh_const=False` needs
  `theta["zh"]` (shape `(n_time,)`).
- Previously a mismatch constructed fine and only failed later with a cryptic
  `KeyError` inside a JIT trace. **Guard:** construction now raises a clear
  `ValueError` naming the fix, and warns if you supply *both* keys.

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
  `calculate_ssp_weights_*` calculations.** Metallicity (`theta["Z"]` for
  constant-Z, `theta["zh"]` for time-varying) is `log10` of the *absolute*
  metallicity on the `self.zmet` / `ssp_lgmet` grid in **every** variant; the
  SFR history `theta["sfh"]` is a linear rate floored identically at `1e-30`
  everywhere. Previously `var_zh` floored SFR with `self.tiny_logt = -70` (a
  log10-time constant), so it weighted the same SFR history differently from
  `const_zh` and returned **NaN** for (near-)zero SFR nodes. That is fixed:
  `const_zh` output is unchanged (it already used `1e-30`), `var_zh` now shares
  the floor, and `const_zh == var_zh` for a constant metallicity history
  (regression-tested in `test_misuse.py`).
- `lookback_time` is **static** (baked at construction). Changing it in `theta`
  at predict time has no effect.

## 5. Observations must be set up before `predict`

- Call `obs.setup_for_model(wave_model[, …])` once before the first
  `predict`/JIT trace. **Guard:** `Spectrum.predict` / `Lines.predict` now raise
  a clear `RuntimeError` instead of a cryptic `AttributeError`.
- `Photometry.predict` *intentionally* falls back to `get_maggies` if setup was
  skipped (supported, but slower and not the GEMV fast path).
- **All-zero uncertainties** → `AssertionError: no valid unmasked data points`
  (clear) at setup.
- **Unknown filter name** → `FileNotFoundError` naming the missing `.par` file.
- `eline_scaling` is a **fraction / direct multiplier** on the model emission
  lines: `1.0` = no aperture loss, `0.65` = lines at 65%, `2.0` = lines doubled.
  (It was previously a percentage where 100 = no loss; changed to 1.0 for
  intuitiveness.)

## 6. Environment & data consistency (not auto-guarded — check yourself)

- **`SPS_HOME`** must point at an FSPS install for the nebular model, dust
  emission, and `SSPBasis`/`FastStepBasis`. Unset → `RuntimeError` at import/use.
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

## 7. `FastStepBasis` (FSPS-backed)

- It wraps FSPS Fortran and is **not** JIT-compatible (do not `jax.jit` it).
- `convert_sfh` requires age-bin spacing ≥ 1 Myr (validated) and currently has a
  pre-existing ordering sensitivity in how it builds the FSPS tabular SFH for
  arbitrary `agebins` (FSPS may reject as "Ages must be increasing"). Use the
  prospector agebin convention.

---

### What is *not* guarded (and why)

Runtime, per-sample value checks (e.g. "this drawn `Z` is out of grid") are
deliberately **not** placed in the jitted hot path — doing so would either break
JIT or slow every evaluation. Use the non-jitted `csp.check_param_ranges(theta)`
on your priors/bounds once before sampling instead.
