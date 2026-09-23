# AGENTS.md — guidance for AI assistants working with CERIDWEN

This file is for an AI coding assistant (Claude, etc.) that a user has pointed at
CERIDWEN to help them *use or extend* the package. Read this before writing code
against the API. It exists to prevent the handful of mistakes that are easy to
make and expensive to debug.

CERIDWEN is a JAX-native, GPU-capable SED-fitting package: a differentiable
forward model (SSP → composite stellar population → dust → nebular → IGM →
observed-frame projection) fit with nested sampling or VI-preconditioned NUTS.

## Conventions that are easy to get wrong — do not assume the common defaults

1. **Metallicity is SOLAR-RELATIVE: `logzsol = log10(Z/Z_sun)` (v1.0.5).** The parameters
   are `theta["logzsol"]` (constant, shape `(1,)`) and `theta["logzsol_hist"]`
   (shape `(n_time,)`); `0.0` is solar on every grid. Z_sun is the **grid's own** solar node,
   resolved at load from `zsun=`, the file's `log10_zsun` provenance, or the content-hash
   table in `ceridwen/ssps/grid_metadata.py` — never from `isoc_type` (FSPS changed MIST's
   `zsol` 0.0142 -> 0.0191 -> 0.0185 under the same name). A grid whose Z_sun is unknown
   raises. Typical ranges: BPASS `[-2.30, +0.30]` (Z_sun 0.020), MIST/aMIST `[-2.50, +0.50]`
   (Z_sun 0.0185). `Uniform(low=-2.0, high=0.2)` is safe on every shipped grid; a bounded
   prior wider than the grid raises. The old absolute keys `Z` / `zh` raise everywhere with
   the converted value. On MIST/aMIST grids `logzsol` **is `[Fe/H]`** and the total
   metallicity is the derived `logzsol_total` = `[Z/H]`. `gas_logz` was already
   solar-relative (CLOUDY axis) and is unchanged; `CSPBasis(gas_tied=True)` sets
   `gas_logz := logzsol`.

2. **`lookback_time` INCREASES with index; index 0 = today.** Element 0 is the
   present, the last element is the oldest bin (≈ age of the universe). The SFH
   array `sfh` is indexed the same way. The OLD decreasing convention
   (`lookback = T_univ - t_grid`) is rejected at construction with a `ValueError`
   — do not reintroduce it, and do not "helpfully" reverse arrays.

   A construction-time grid is REQUIRED (either `CSPBasis(ssp,
   lookback_time=...)` or a full `theta=` dict): it is where monotonicity and
   range are validated — per-call grids are traced under `jit` and cannot be
   value-checked. A per-call `theta["lookback_time"]` passed to
   `predict`/`get_spectrum` then overrides it verbatim (used by
   transform-derived grids from a sampled `zred`).

3. **Units and frames.** All wavelengths are Å, vacuum. The model grid
   (`csp.wave`) and `Lines` wavelengths (and `mask_lines` centres) are
   REST-frame; the `Spectrum` data pixel grid is OBSERVED-frame — the model is
   redshifted by (1 + zred) onto the data pixels, never the other way. Model
   spectra are `F_nu` (per unit frequency). Broadband fluxes are AB maggies.
   Emission-line fluxes are erg s⁻¹ cm⁻². Stellar mass is supplied as
   `logmass` = log10(M⋆/M_sun); the forward model is evaluated at unit mass
   and scaled by `10**logmass`.

4. **SFH mass normalisation is mass-weighted (trapezoidal), not mean-SFR.** See
   `model/transforms.py` (`logsfr_ratios_to_sfh`). Getting this wrong biases
   `logmass` by many dex. Use the provided transform; don't hand-roll it.

## Hard requirements (the model will not run otherwise)

- **float64 must be on.** `import ceridwen` already calls
  `jax.config.update("jax_enable_x64", True)`. Do not disable it; evidence
  estimates and gradients depend on it.
- **FSPS + `$SPS_HOME` are required at runtime,** not just to build the SSP cache.
  The CLOUDY nebular grids and Draine & Li dust-emission templates are read from
  `$SPS_HOME` whenever `add_neb=True` or `add_dust_emission=True`. Building the
  SSP cache (`SSPData.from_fsps`) also needs FSPS.
- **Filters and attenuation curves are internal** since v1.0.0:
  `ceridwen/observation/filters.py` and `ceridwen/dust/attenuation_laws.py`,
  vendored from `sedpy-jax` (MIT), with the 293 filter `.par` files and the two
  reference spectra under `ceridwen/data/`. There is no `sedpy` dependency; do
  not reintroduce one.
- **Nested sampling needs `blackjax.nss`/`blackjax.ns`**, now merged into the
  official blackjax (blackjax-devs) but not yet in a tagged PyPI release.
  `pyproject.toml` pins blackjax's `main` branch via direct git reference, so
  `pip install` pulls it automatically. Consequence: ceridwen is git/source-
  installed, not PyPI, until a blackjax release ships NSS — then swap the pin for
  `blackjax>=X.Y` (requires-python stays >=3.11). Don't "simplify" this to a
  stock released `blackjax` pin yet; the released versions still lack NSS.

## Staying JAX-correct when editing the package

The forward model is a single XLA graph; the sampling hot path has zero Python
branches. When modifying it:

- Use `jax.numpy`, not `numpy`, inside anything that gets traced/jitted.
- No data-dependent Python `if`/`for` over array *values* in the hot path; use
  `jnp.where`, `lax.select`, `lax.switch`, `lax.cond`, masks.
- Keep functions pure and traceable; parameters flow as a dict-based `theta`
  PyTree. Don't capture Python state as a hidden side-channel (no `@jit` on bound
  methods that close over `self` as a static value).
- Preserve dtype coherence; precompute constants outside the hot path.
- Validation that must raise on bad input belongs in `__init__`/setup shims
  (trace-time), not inside jitted kernels.

## Public API (import these; treat everything else as internal)

```python
# Primary model builders are available straight from the top level:
from ceridwen import (SSPData, CSPBasis, SedModel,
                      DustModel, DustEmission, NebularModel, fitSED, read_result_h5)
# Observation containers, priors and samplers live in clear sub-namespaces:
from ceridwen.observation import Photometry, Spectrum, Lines
from ceridwen.priors import (Prior, Uniform, TopHat, Normal, ClippedNormal,
                             LogNormal, StudentT)
from ceridwen.sampler import run_sampler
from ceridwen.likelihood import DiagonalGaussianLikelihood, MultiObservationLikelihood
```

Equivalent namespaced paths also work: `ceridwen.ssps.SSPData`,
`ceridwen.csp.CSPBasis`, `ceridwen.model.SedModel`.

- The only nebular class is `NebularModel`. The old `NebularModelFSPSMatch`
  (bug-for-bug FSPS reproduction) has been removed: upstream FSPS was fixed and
  now matches the strict `NebularModel`. `init_neb_params={"match_fsps": True}`
  is obsolete and ignored with a warning, and `CSPBasis(..., match_fsps=True)`
  raises a `TypeError` like any unknown keyword. Do not reintroduce either.
- `ssps/ssp_data.py` no longer stores `log_qq`; the nebular model derives the
  ionising-photon rate internally. Old HDF5 caches with a `log_qq` dataset still
  load (the field is ignored).

## Canonical workflow

Step 0 build/load the SSP cache → build `CSPBasis` → wrap observations
(`Photometry`/`Spectrum`/`Lines`) → `SedModel(csp, observations, priors=...)` →
`fitSED(...)` or `run_sampler(...)`. The end-to-end, runnable reference is
[`examples/quickstart.py`](examples/quickstart.py); the joint photometry +
spectroscopy + lines workflow is in [`docs/tutorial.md`](docs/tutorial.md) and
`examples/demo_2_photometry_lines.py` / `examples/demo_3_spectrum_advanced.py`.

## Package architecture / module map

The forward model is assembled bottom-up: SSP library → composite stellar
population → dust attenuation/emission → nebular → IGM → observed-frame
projection → likelihood → sampler.

- `ssps/` — `ssp_data.py`: `SSPData` (frozen dataclass, HDF5 I/O, ionising-photon
  rate derived internally; HDF5 caches the SSP spectral grid; schema 3.0 adds the optional
  surviving-mass table `ssp_stellar_mass`, added to an existing file by
  `scripts/attach_stellar_mass.py`, used only by `PostProcess` for `mfrac`). `ssp_basis.py`:
  `SSPBasis`, `FastStepBasis` (thin FSPS wrappers + tabular SFH binning).
- `csp/` — `csp.py`: `CSPBasis`, the core forward model. Holds the `get_spectrum_*`
  variants (stellar ± dust attenuation ± dust emission ± nebular), step/linear
  SFH interpolation, constant or time-varying metallicity, dict-based `theta`,
  fully JAX-traceable; `predict` projects onto the observations (static
  line-to-band basis for fixed-z photometry, painted lines otherwise).
  `csp_afe.py`: `CSPBasis_afe`, the [alpha/Fe] variant without a nebular model.
  `spectrum_calibration.py`: `spectrum_scaling` / `spectrum_calib` factor (one per
  spectrum, `spectrum_scaling_<obs.name>` with several). The profiled alternative is
  `Spectrum(polynomial_order=M)`, solved in the likelihood (`likelihood/poly_calibration.py`).
- `broadening.py` — `Kinematics` (sigma_gal / sigma_gas, fixed or theta keys),
  `Instrument` (LSF), `SpectralProjector` (continuum FFT kernel + banded
  instrument response + analytic line painting on the observed pixels),
  `PhotometricBroadener`. The only place spectral widths are set.
- `dust/` — `DustModel.py`: `Dust`/`DiffuseDust`, age-binned attenuation with
  multiple switchable laws per bin via `lax.switch` (params are plain dicts).
  `DustEmission.py`: DL07 + THEMIS grids, bilinear interp in (qPAH, Umin), dust
  mass. `AGBDustShell.py`: optional AGB circumstellar dust.
- `neb/` — `NebularGridModel.py`: `NebularModel` (CLOUDY grids, each cube
  interpolated against its own gas_logz/gas_logu/age axes, line profiles at
  the pixel floor, `line_profiles(sigma)` for the photometric line basis).
- `observation/` — `base.py` (ABC), `photometry.py` (filter convolution via
  `filters.FilterSet` → matrix-vector projection `_T`, optional
  `PhotometricBroadener` and static line basis), `spectrum.py` (`Instrument`
  + `SpectralProjector` built by `setup_for_model`), `lines.py` (line fluxes
  read from the nebular grid by `CSPBasis`; Gaussian aperture matrix `W` for
  `Lines.predict` alone), `gp.py` (GP residuals). `observation.py` is a re-export shim so
  `from ceridwen.observation.observation import Photometry, Spectrum, Lines`
  still works.
- `model/` — `model.py`: `SedModel` (`predict`, `apply_transforms`, `ln_prior`,
  `log_prob`; free vs. derived params, `logmass` amplitude scaling).
  `transforms.py`: `logsfr_ratios_to_sfh` and inverse (mass-weighted trapezoidal
  normalisation).
- `likelihood/` — `likelihood.py`: `DiagonalGaussianLikelihood`,
  `MultiObservationLikelihood`, pure-JAX `lnlike_diag_gaussian`, masking,
  `make_lnprobfn()` (the jitted log-posterior factory). `noise_model.py`:
  `DiagonalNoiseModel` (noise floor, optional error scale / jitter / calibration error,
  named per observation `log_jitter_<kind>[_<obs.name>]` in `fitSED` since v1.0.7, optional
  Prospector-style outlier mixture `f_outlier` / `nsigma_outlier` per observation
  (`f_outlier_spec/_phot/_lines[_<obs.name>]` in `fitSED`, all default 0 = off: switch on
  explicitly), applied by the likelihood kernels `lnlike_diag_outlier[_with_upper_limits]`;
  `docs/outlier_model.md`).
- `sampler/` — `priors.py` (TFP-JAX priors with logpdf/sample/unit_transform),
  `nested.py` (BlackJAX nested sampling), `nuts.py` (NUTS, VI-preconditioned),
  `vi.py` (VI transport maps: TriL, IAF/NeuTra), `runner.py` (`SamplerAdapter`
  protocol, `SamplingResult`, `run_sampler`, `to_anesthetic`).
- `cosmology.py` — JAX-native flat ΛCDM (Planck 18) with an astropy fallback.
- `igm.py` — IGM attenuation (`Madau1995`), extensible via the `IGMModel` ABC.
- `fit.py` — `fitSED` (top-level convenience wrapper) + `read_result_h5` /
  `load_result_h5` / `result_cosmology`; writes `<output_dir>/ceridwen_result.h5`
  (obs incl. sky / calibration / upper limits and each likelihood's noise model,
  model/priors as JSON, kinematics, cosmology, samples, log-weights, log-evidence,
  `/provenance`: version, git state, sampler settings, rng key; `read_provenance`).
- `postprocess.py` — `PostProcess` (equal-weight draws, SFH averages, formed and
  surviving mass (`mfrac`, grids with a mass table), UV and ionising properties,
  posterior predictions); `plotting.py` — summary, corner
  and diagnostic figures.

Design patterns to follow when extending: frozen dataclasses for immutable data
(`SSPData`); HDF5 for large spectral grids; modular dust laws (multiple per age
bin, automatic parameter renaming when a law is reused); `@jit`/`vmap` throughout;
`display()` methods on model components for visualisation.

## Building grids and installing for development

- `pip install -e .` for a dev environment — everything (incl. VI, nested-sampling
  plotting, and the test runner) is core, no extras to choose. FSPS is the one
  exception: install it separately (`pip install fsps`) with `$SPS_HOME` set.
- Build the SSP cache once: `SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")`.
  `from_fsps` accepts ONLY stellar-library / IMF kwargs — dust, SFH, nebular, IGM,
  redshift, and fixed-metallicity kwargs are rejected (the CSP owns those) — and
  records provenance (isochrone library, `imf_type`, FSPS version, build kwargs).
  `CSPBasis` reads the isochrone library from that provenance automatically, so
  `isoc_type` is never set by hand and the nebular grid always matches; a
  conflicting user `isoc_type` raises. Reload with `SSPData.load(...)`.
- Some nebular models need CLOUDY grid files present in the FSPS data directory.

## Repository pointers

- `README.md` — install + quick start for humans.
- `GOTCHAS.md` — the misuse/user-error guide (metallicity conventions, silent
  `theta` typos, …). Read it before constructing models; it expands on the
  conventions above.
- `examples/quickstart.py` — minimal runnable fit (mock photometry).
- Tests live in `tests/`; they resolve an SSP grid via `tests/_gridfixture.py`
  (`$CERIDWEN_TEST_SSP` → `tests/fixtures/<name>` → `ceridwen/data/test_data/`).
  The main test grid is NOT committed: it is the published BPASS grid, fetched
  with `fetch_grid("mist_bpass_v2")` (CI does this). Without a grid, or without
  `$SPS_HOME` for the nebular tests, tests *skip*, and a run with skips is not a
  pass: run `pytest -m "not fsps and not gpu" -ra` for the FSPS-free subset and
  read the skip summary. See "Verification" in `README.md`.

## Don't

- Don't recombine the split `__version__` / `__githash__` imports in
  `ceridwen/__init__.py` (the split fixes a real import-masking bug).
- Don't put research/diagnostic scripts inside the `ceridwen/` package; only the
  importable library ships. User-facing runnable examples live in `examples/`.
- Don't add new runtime dependencies without updating `pyproject.toml`.
