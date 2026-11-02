# Development and verification

## Setting up

```bash
git clone --depth 1 https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install -e .
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
```

The tests use the published BPASS SSP grid. Tests that need a missing grid are
skipped; check the `-ra` summary to see which tests ran. Avoid running scripts
from the directory containing the clone, where Python may import the clone's
`ceridwen` folder instead of the installed package.

## Checks

The following checks cover installation, numerical output and input validation.

| Check | Command | Coverage | CI |
|---|---|---|---|
| Environment | `python -m ceridwen.check` | Python, dependencies, filters, attenuation laws, float64, nested sampling and FSPS data | wheel job |
| Tests | `pytest -m "not fsps and not gpu" -q -ra` | units, conventions, broadening, likelihoods, samplers, post-processing and input validation | each push and pull request |
| Regression baselines | `pytest tests/regression/test_regression.py -q` | 22 output categories, 4 optional grid-dependent categories and 3 requiring AGN tables; stored arrays compared at `rtol=2e-6` (`agn_lines` 2e-5; float32 summation order differs between jax versions), with absolute tolerance 1e-10 times each array's maximum; the first 9 also require maximum relative residual 1e-6; figures saved in `tests/regression/figures/` | requires local FSPS data |
| Golden spectra | `pytest tests/csp/test_lookback_flip_invariant.py -q` | 13 configurations: 6 SFH/metallicity, 3 AGN, 3 AGN2 (requiring AGN tables) and 1 with a fixed nebular block; SSP weights at `rtol=1e-12`, spectra, line fluxes and photometry (through `csp.predict`, as in a fit) at `rtol=1e-6` to allow float32 contractions | requires local FSPS data |
| Misuse report | `python tests/regression/misuse_report.py` | checks for an exception or warning for each known input error; `SILENT` indicates a missing check | assertions included in tests |
| Static API | `python scripts/check_api_usage.py` | checks keyword arguments and removed names in tests, examples and Python blocks in `README.md`, `GOTCHAS.md` and `docs/` | no |
| Byte identity | `python scripts/bit_identity_check.py --save old.npz`, then `--compare old.npz` on another commit | compares output arrays using `tobytes()`; `--ns` adds nested sampling with the same RNG key | no |
| Physical standard | `python scripts/physical_check.py --save old.npz`, then `--compare old.npz` on another commit | limits spectral, photometric and line-flux changes to 1e-3; covers the byte-identity configurations plus all components, at 16 draws and four prior extremes (faintest, brightest, all-low, all-high); reports the worst output and theta, exiting 1 on failure | no |

The physical-standard check measures relative differences where values exceed
1e-4 of an array's maximum. Below that threshold, differences are expressed
relative to the maximum.

CI has no FSPS data. Nebular tests, regression baselines and golden-spectrum
tests therefore need to run locally with `$SPS_HOME` set. Byte identity is
checked on the CPU. On a GPU, emission-line scatter-adds can differ in the
last bit between runs.

Run the byte-identity check on an idle machine, separately from the test suites.
One comparison on a busy machine reported a difference of about 3 ulp in a
float64 `PostProcess` spectrum. The same code was byte-identical across all
23,232 arrays when rerun on an idle machine. The cause was not identified.
Repeat an isolated difference under those conditions before attributing it to
a code change.

## Development practices

CERIDWEN has one author and maintainer. AI coding assistants are used for
drafting, refactoring and review. Changes are checked using the tests above
and the following requirements.

- [`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md)
  documents conventions, requirements and the module structure, including
  solar-relative `logzsol`, lookback-time ordering, units and frames.
  [`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md)
  describes common usage errors.
- Refactors and optimisations must either preserve CPU output byte for byte
  or pass the physical-standard check above. For changes to output, predict
  and document the size before measuring it. The tolerance reflects the
  model's float32 error: faint photometric bands at extreme parameters can
  differ from float64 by about 1e-4. Assess log-likelihoods and gradients on
  their own scales, and compare posteriors statistically.
- Changes to the physics need a predicted effect size. New features need a
  golden test configuration. Before updating an affected baseline, list the
  changed arrays, quantify the differences and explain them.
- Validate inputs during construction, setup or tracing, so checks do not
  add work to the compiled sampling loop.
- Use independent numerical references where available. Broadening tests
  compare with direct quadrature and, when installed, sedpy's smoothing
  routines (`tests/test_broadening.py`).
