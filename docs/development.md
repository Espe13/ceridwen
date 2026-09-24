# Development and verification

## Setting up

```bash
git clone --depth 1 https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install -e .
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
```

The suite's main SSP grid is the published BPASS grid. A grid-dependent test skips when it
finds no grid, so read the `-ra` summary at the end of a run: a skip is not a pass. Do not run
scripts from the folder that contains the clone: a folder named `ceridwen` in the working
directory shadows the installed package.

## The checks

Each layer answers a different question.

| layer | command | what it asserts | on CI |
|---|---|---|---|
| Environment | `python -m ceridwen.check` | Python version, dependencies, filter curves and attenuation laws, float64, nested sampling, `$SPS_HOME` and its data files | yes, in the wheel job |
| Tests | `pytest -m "not fsps and not gpu" -q -ra` | units, conventions, broadening, likelihoods, samplers, post-processing, the misuse guards | every push and pull request |
| Regression baselines | `pytest tests/regression/test_regression.py -q` | 18 categories of forward-model and likelihood output (plus 4 optional ones, skipped where their grid is absent) against stored arrays at `atol=1e-10, rtol=1e-7`, and a maximum relative residual of 1e-6; writes comparison figures to `tests/regression/figures/` | no, needs `$SPS_HOME` |
| Golden spectra | `pytest tests/csp/test_lookback_flip_invariant.py -q` | 6 SFH and metallicity configurations: SSP weights at `rtol=1e-12`; spectra, line fluxes and photometry at `rtol=1e-6` (they pass through float32 contractions) | no, needs `$SPS_HOME` |
| Misuse report | `python tests/regression/misuse_report.py` | each known user error ends in an exception or a warning; a `SILENT` row is a bug | its assertions run in the tests |
| Static API check | `python scripts/check_api_usage.py` | every call in `tests/`, `examples/` and the Python blocks of `README.md`, `GOTCHAS.md` and `docs/` uses only keyword arguments that exist and no removed name | no |
| Byte identity | `python scripts/bit_identity_check.py --save old.npz` on one commit, `--compare old.npz` on the next | a refactor changes nothing: every output array equal under `tobytes()`; `--ns` adds a nested-sampling run at the same RNG key | no |

CI has no FSPS data, so the nebular tests, the regression baselines and the golden spectra run
only on a machine with `$SPS_HOME` set. Byte identity is a CPU statement: on a GPU, the
scatter-add that paints emission lines can differ in the last bit from run to run.

## How this code was built

CERIDWEN is written and maintained by one person, with AI coding assistants used for drafting,
refactoring and review. The safeguard is not trust in the tool but the checks above, and a
few rules that apply to every change, whoever or whatever proposes it:

- **Conventions are written down where a tool reads them first.**
  [`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md) holds the conventions
  that are easy to get wrong (solar-relative `logzsol`, lookback-time ordering, units and
  frames), the hard requirements and the module map;
  [`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md) is the misuse guide.
- **A change has to say what it is.** A refactor or optimisation must be byte-identical on
  CPU. An intended change to the physics must come with its predicted size, and a new feature
  must add a golden configuration of its own. Existing baselines are not re-captured to make
  a test pass.
- **Wrong input fails loudly.** Every guard sits in construction or setup code, or runs once
  at trace time, so the compiled sampling path is unchanged.
- **Independent references where they exist.** The spectral broadening is tested against
  direct quadrature and, when `sedpy` is installed, against its smoothing routines
  (`tests/test_broadening.py`).
