# AGENTS.md — guidance for AI assistants working with CERIDWEN

This file is for an AI coding assistant (Claude, etc.) that a user has pointed at
CERIDWEN to help them *use or extend* the package. Read this before writing code
against the API. It exists to prevent the handful of mistakes that are easy to
make and expensive to debug.

CERIDWEN is a JAX-native, GPU-capable SED-fitting package: a differentiable
forward model (SSP → composite stellar population → dust → nebular → IGM →
observed-frame projection) fit with nested sampling or VI-preconditioned NUTS.

## Conventions that are easy to get wrong — do not assume the common defaults

1. **Metallicity is log10 of ABSOLUTE Z, not Z/Z_sun.** The parameter `Z`
   (and `ssp_lgmet`) is `log10(Z)` in absolute units. A solar value is roughly
   `-1.85` (Z_sun ≈ 0.014), *not* `0.0`. Priors like `Uniform(low=-2.5, high=0.2)`
   are correct; do not "fix" them to be centred on 0.

2. **`lookback_time` INCREASES with index; index 0 = today.** Element 0 is the
   present, the last element is the oldest bin (≈ age of the universe). The SFH
   array `sfh` is indexed the same way. The OLD decreasing convention
   (`lookback = T_univ - t_grid`) is rejected at construction with a `ValueError`
   — do not reintroduce it, and do not "helpfully" reverse arrays.

3. **Units.** Wavelengths are Å, vacuum, rest-frame on input. Model spectra are
   `F_nu` (per unit frequency). Broadband fluxes are AB maggies. Emission-line
   fluxes are erg s⁻¹ cm⁻². Stellar mass is supplied as `logmass` = log10(M⋆/M_sun);
   the forward model is evaluated at unit mass and scaled by `10**logmass`.

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
- **`sedpy_jax`** (PyPI: `sedpy-jax`) provides filter convolutions and smoothing;
  it is imported at package import time.

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
from ceridwen import DustModel, DustEmission, NebularModel, fitSED, read_result_h5
from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.observation.observation import Photometry, Spectrum, Lines
from ceridwen.model.model import SedModel
from ceridwen.sampler import (Uniform, TopHat, Normal, ClippedNormal,
                              LogNormal, StudentT, run_sampler)
```

- The canonical nebular class is `NebularModel`. `NebularModelFSPSMatch`
  (bug-for-bug FSPS reproduction) is intentionally internal — only use it via
  `CSPBasis(..., match_fsps=True)` if you specifically need FSPS reproducibility.
- `ssps/ssp_data.py` no longer stores `log_qq`; the nebular model derives the
  ionising-photon rate internally. Old HDF5 caches with a `log_qq` dataset still
  load (the field is ignored).

## Canonical workflow

Step 0 build/load the SSP cache → build `CSPBasis` → wrap observations
(`Photometry`/`Spectrum`/`Lines`) → `SedModel(csp, observations, priors=...)` →
`fitSED(...)` or `run_sampler(...)`. The end-to-end, runnable reference is
[`examples/quickstart.py`](examples/quickstart.py); larger drivers
(spectroscopy, lines, free redshift, VI-NUTS) are in `scripts/`.

## Package architecture / module map

The forward model is assembled bottom-up: SSP library → composite stellar
population → dust attenuation/emission → nebular → IGM → observed-frame
projection → likelihood → sampler.

- `ssps/` — `ssp_data.py`: `SSPData` (frozen dataclass, HDF5 I/O, ionising-photon
  rate derived internally; HDF5 caches the SSP spectral grid). `ssp_basis.py`:
  `SSPBasis`, `FastStepBasis` (thin FSPS wrappers + tabular SFH binning).
- `csp/` — `csp.py`: `CSPBasis`, the core forward model. Holds the `get_spectrum_*`
  variants (stellar ± dust attenuation ± dust emission ± nebular), step/linear
  SFH interpolation, constant or time-varying metallicity, dict-based `theta`,
  fully JAX-traceable. `csp_svd.py`: SVD-accelerated variant.
- `dust/` — `DustModel.py`: `Dust`/`DiffuseDust`, age-binned attenuation with
  multiple switchable laws per bin via `lax.switch` (params are plain dicts).
  `DustEmission.py`: DL07 + THEMIS grids, bilinear interp in (qPAH, Umin), dust
  mass. `AGBDustShell.py`: optional AGB circumstellar dust.
- `neb/` — `NebularGridModel.py`: `NebularModel` (physically strict; CLOUDY grids,
  trilinear interp in gas_logz/gas_logu/age, precomputed line profiles, velocity
  broadening). `NebularGridModelSVD.py`: SVD-accelerated subclass.
  `NebularGridModel_fsps_match.py`: internal FSPS-reproducing variant (via
  `match_fsps=True` only).
- `observation/` — `base.py` (ABC), `photometry.py` (filter convolution via
  `sedpy_jax.FilterSet` → matrix-vector projection), `spectrum.py` (interpolation
  matrix `H`, resolution smoothing, single GEMV), `lines.py` (Gaussian weight
  matrix `W`), `gp.py` (GP residuals). `observation.py` is a re-export shim so
  `from ceridwen.observation.observation import Photometry, Spectrum, Lines`
  still works.
- `model/` — `model.py`: `SedModel` (`predict`, `apply_transforms`, `ln_prior`,
  `log_prob`; free vs. derived params, `logmass` amplitude scaling).
  `transforms.py`: `logsfr_ratios_to_sfh` and inverse (mass-weighted trapezoidal
  normalisation).
- `likelihood/` — `likelihood.py`: `DiagonalGaussianLikelihood`,
  `MultiObservationLikelihood`, pure-JAX `lnlike_diag_gaussian`, masking,
  `make_lnprobfn()` (the jitted log-posterior factory). `noise_model.py`:
  `DiagonalNoiseModel` (optional jitter + calibration error). `theta.py`:
  `ThetaVector`, a registered JAX PyTree giving both flat-array and named access.
- `sampler/` — `priors.py` (TFP-JAX priors with logpdf/sample/unit_transform),
  `nested.py` (BlackJAX nested sampling), `nuts.py` (NUTS, VI-preconditioned),
  `vi.py` (VI transport maps: TriL, IAF/NeuTra), `runner.py` (`SamplerAdapter`
  protocol, `SamplingResult`, `run_sampler`, `to_anesthetic`).
- `cosmology.py` — JAX-native flat ΛCDM (Planck 18) with an astropy fallback.
- `igm.py` — IGM attenuation (`Madau1995`), extensible via the `IGMModel` ABC.
- `fit.py` — `fitSED` (top-level convenience wrapper) + `read_result_h5`; writes
  `<output_dir>/ceridwen_result.h5` (obs, model/priors as JSON, samples,
  log-weights, log-evidence).

Design patterns to follow when extending: frozen dataclasses for immutable data
(`SSPData`); HDF5 for large spectral grids; modular dust laws (multiple per age
bin, automatic parameter renaming when a law is reused); `@jit`/`vmap` throughout;
`display()` methods on model components for visualisation.

## Building grids and installing for development

- `pip install -e ".[all]"` for a full dev environment (core + vi + nested +
  grids + test). FSPS must be installed separately and `$SPS_HOME` set.
- Build the SSP cache once: `SSPData.from_fsps(save_to="ssp_data.h5", ...)` (any
  FSPS kwargs pass through), then reload with `SSPData.load(...)`.
- Some nebular models need CLOUDY grid files present in the FSPS data directory.

## Repository pointers

- `README.md` — install + quick start for humans.
- `GOTCHAS.md` — the misuse/user-error guide (metallicity-units trap, silent
  `theta` typos, …). Read it before constructing models; it expands on the
  conventions above.
- `examples/quickstart.py` — minimal runnable fit (mock photometry).
- Tests live in `tests/`; they resolve a committed SSP grid via
  `tests/_gridfixture.py` and skip cleanly if it (or FSPS) is absent. Run
  `pytest -m "not fsps and not gpu"` for the FSPS-free subset.
  `python scripts/make_test_fixture_grid.py` (re)builds the committed fixture.

## Don't

- Don't recombine the split `__version__` / `__githash__` imports in
  `ceridwen/__init__.py` (the split fixes a real import-masking bug).
- Don't ship research/diagnostic scripts inside the `ceridwen/` namespace; they
  belong in `scripts/`.
- Don't add new runtime dependencies without updating `pyproject.toml`.
