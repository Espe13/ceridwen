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
scripts from the folder that contains the clone: Python then finds the clone folder
`ceridwen` before the installed package.

## The checks

Each layer answers a different question.

| layer | command | what it asserts | on CI |
|---|---|---|---|
| Environment | `python -m ceridwen.check` | Python version, dependencies, filter curves and attenuation laws, float64, nested sampling, `$SPS_HOME` and its data files | yes, in the wheel job |
| Tests | `pytest -m "not fsps and not gpu" -q -ra` | units, conventions, broadening, likelihoods, samplers, post-processing, the misuse guards | every push and pull request |
| Regression baselines | `pytest tests/regression/test_regression.py -q` | 22 categories of forward-model and likelihood output (plus 4 optional ones, skipped where their grid is absent, and 3 that need the AGN tables) against stored arrays at `rtol=1e-7` with an absolute tolerance of 1e-10 times each array's maximum, and, for the first 9, a maximum relative residual of 1e-6; writes comparison figures to `tests/regression/figures/` | no, needs `$SPS_HOME` |
| Golden spectra | `pytest tests/csp/test_lookback_flip_invariant.py -q` | 6 SFH and metallicity configurations: SSP weights at `rtol=1e-12`; spectra, line fluxes and photometry at `rtol=1e-6` (they pass through float32 contractions) | no, needs `$SPS_HOME` |
| Misuse report | `python tests/regression/misuse_report.py` | each known user error ends in an exception or a warning; a `SILENT` row is a bug | its assertions run in the tests |
| Static API check | `python scripts/check_api_usage.py` | every call in `tests/`, `examples/` and the Python blocks of `README.md`, `GOTCHAS.md` and `docs/` uses only keyword arguments that exist and no removed name | no |
| Byte identity | `python scripts/bit_identity_check.py --save old.npz` on one commit, `--compare old.npz` on the next | the footprint of a change: which output arrays differ at all under `tobytes()`; `--ns` adds a nested-sampling run at the same RNG key | no |
| Physical standard | `python scripts/physical_check.py --save old.npz` on one commit, `--compare old.npz` on the next | no spectrum (model grid or projected), photometry, `Lines` prediction or line flux moves by more than 1e-3 relative, per element where its value exceeds 1e-4 of the array's maximum and in units of that maximum below it, over every configuration of the byte-identity check plus one with every component on, at 16 draws and four prior extremes (faintest, brightest, all-low, all-high); prints the worst case per output with its theta and exits 1 above the standard | no |

CI has no FSPS data, so the nebular tests, the regression baselines and the golden spectra run
only on a machine with `$SPS_HOME` set. Byte identity is a CPU statement: on a GPU, the
scatter-add that paints emission lines can differ in the last bit from run to run.

Run the byte-identity check on an idle machine, never alongside the test suites. A loaded
`--compare` once reported a single differing array (a float64 `PostProcess` spectrum, about 3
ulp) that did not recur on a quiet machine, where the same code was byte-identical across all
23,232 arrays. A lone differing array from a loaded run is not evidence of a code change:
re-run it quietly first. The cause is unknown and was not reproduced
([`NOTE_T4_LOAD_2026-10-09.md`](https://github.com/Espe13/ceridwen/blob/main/NOTE_T4_LOAD_2026-10-09.md)).

## How this code was built

CERIDWEN is written and maintained by one person, with AI coding assistants used for drafting,
refactoring and review. The safeguard is not trust in the tool but the checks above, and a
few rules that apply to every change, whoever or whatever proposes it:

- **Conventions are written down where a tool reads them first.**
  [`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md) holds the conventions
  that are easy to get wrong (solar-relative `logzsol`, lookback-time ordering, units and
  frames), the hard requirements and the module map;
  [`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md) is the misuse guide.
- **A change has to say what it is.** A refactor or optimisation is either byte-identical on
  CPU, or it changes output arrays within the physical standard above (the physical-standard
  check passes, with the size of the change predicted before it was measured); the standard
  is set at the level of the code's own float32 error (the float32 SSP cube is good to ~1e-4
  in faint photometric bands at extreme parameters, against float64). Log-likelihoods and
  gradients are judged on their own scale, and posteriors are compared statistically, not
  draw by draw. An intended change to the physics must come with its predicted size, and a new
  feature must add a golden configuration of its own. Existing baselines are not re-captured
  to make a test pass: when a change within the standard moves one, the moved arrays are
  listed with their sizes and justified before it is re-captured.
- **Wrong input fails loudly.** Every guard sits in construction or setup code, or runs once
  at trace time, so the compiled sampling path is unchanged.
- **Independent references where they exist.** The spectral broadening is tested against
  direct quadrature and, when `sedpy` is installed, against its smoothing routines
  (`tests/test_broadening.py`).
