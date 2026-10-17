## 1.0.13 (2026-10-15)

### Apple GPU: float32 mode
- `pip install ceridwen` on an Apple Silicon Mac with macOS 26 and Python >= 3.12 also installs
  jax-graft (an environment marker in `pyproject.toml`), which makes the GPU (`mtl`) JAX's
  default device. Apple GPUs have no float64, so `import ceridwen` then runs the process in
  float32 and says so once; `fitSED` and `python -m ceridwen.check` report the device and the
  precision. `JAX_PLATFORMS=cpu` keeps float64 on the CPU, `CERIDWEN_X64=0` forces float32
  anywhere, and `CERIDWEN_X64=1` with the Apple GPU as default is refused at import. New page
  `docs/mac_gpu.md`, with the benchmark and the advice that a float32 nested-sampling evidence
  needs twice the default inner slice steps.
- Nothing forces `jax_enable_x64` any more: examples, scripts and tests `import ceridwen`, which
  sets the precision. `ceridwen.precision` (new) gives the working dtype (`is_x64`,
  `float_dtype`, `FLOAT`, `as_float`, `safe_tiny`, `precision_name`); under float64 nothing
  changes.
- In float32 mode each observation's data are rescaled by a power of two before the kernels and
  the constant is added back to ln Z, ln L and the MAP in float64
  (`ceridwen.likelihood.likelihood`: `Float32Rescaling`, `observation_scale`, ...). A fixed
  numeric jitter and `Spectrum(marginalize_elines=True)` are refused in float32 mode.
- The result file records `config/precision` and `config/platform`; `read_result_h5` returns
  them under `"config"`, and reads ln L and the weights as float64 in either mode.
- `tests/test_float32_parity.py` compares the golden configurations and a mock galaxy between
  float64, float32 on the CPU and float32 on the Apple GPU.
- Notebooks: `examples/apple_gpu_mock_fit.ipynb` and `examples/mock_fit_1..4_*.ipynb`
  (photometry; photometry with line fluxes; a spectrum; a spectrum calibrated against
  photometry), each a mock galaxy fitted and its parameters recovered.
- `examples/quickstart.py` downloads the published grid when none is found.

### Update notice
- `fitSED` and `python -m ceridwen.check` say when PyPI has a newer CERIDWEN
  (`ceridwen/update_check.py`: once a day at most, two-second timeout, silent offline;
  `CERIDWEN_NO_UPDATE_CHECK=1` switches it off, and it never runs on CI). `docs/installation.md`,
  "Updates". The README and `docs/installation.md` say how to get an email for each release
  (GitHub: Watch, Custom, Releases).

### Describing a model
- `SedModel.latex()` returns a LaTeX table (booktabs) of the free, derived and fixed parameters
  with their priors, and the model settings.
- `SedModel.display()` is new: the model as a diagram (free parameters, derived parameters with
  the inputs traced from each transform, the forward-model stages in the order they are applied
  with their fixed values, the observations with their nuisance parameters) and, below it, the
  prior-predictive 68 and 95 % band against the data. `ceridwen/model/describe.py`. It uses
  smplotlib's Hershey fonts when smplotlib is installed (without importing it) and a serif face
  otherwise. In a script, `model.display(return_fig=True)` returns the figure; a bare
  `display()` opens a window and waits (README, quick start).
- The `latex()` columns wrap (ragged right), so the table fits a two-column page.

### From a newcomer test (round 1)
- `docs/agn.md`, "Sampling": the snippet adds `free_param_init` for `log_fagn` and
  `log_fagn_bbb` (without it `SedModel` refuses their priors) and names the nebular and
  dust-emission keys that need a prior or a constant transform.
- `examples/agn_demo.py --fit` runs: it fixes those keys at their defaults and loads the
  published grid with `fetch_grid` instead of building one with python-fsps.
- Clearer errors: `SedModel` says a new name a transform reads must be in `free_param_init`;
  the nested sampler's "no prior" error names `SedModel(priors=...)` and the constant
  transform, not an internal class.
- README: `python=3.12` in the conda command (the Apple GPU needs it), the bash line for
  `$SPS_HOME`, `pip install -e ".[cuda12]"` for a source install, and the quick-start error
  bars at 100 live points (too narrow; seeds and float32 differ by 0.1 dex in logmass).
- `docs/mac_gpu.md`: the device prints as `mtl:0`; the float32 warnings from JAX and
  tensorflow-probability are listed; the progress bar shows ln Z in the sampler's rescaled
  units.

### From a newcomer test (round 2)
- The demo scripts (`demo_1..3`, `demo_eline_marginalisation`, `demo_postprocess`) fall back to
  the published grid `mist_miles_chab` instead of building one with python-fsps;
  `demo_postprocess.py` defaults to `demo_1_output/ceridwen_result.h5`.
- `demo_2_photometry_lines.py`: the FUV upper limit lay 17 sigma below the mock's own true
  flux, which biased the fit; it now lies 2 sigma above it. The docstring says 0-based line
  indices, as the code uses.
- `examples/README.md`: the grid fallback, measured CPU and Apple GPU run times, the
  notebooks (and that they need Jupyter), `agn_demo.py`. `mock_fit_1_photometry.ipynb` explains
  the metallicity-dust degeneracy behind its 2-3 sigma misses.
- README quick start: the figures mark the SFH truth too; the SFH ratios are prior-limited.
- The quick start leads with the Apple GPU, the default on an Apple Silicon Mac with macOS 26
  and Python 3.12: 3.7 min there, 8.4 min on a CPU (both measured on an M3 Pro).
- The README, `docs/quickstart.md`, the example scripts, the notebooks and the tests give the
  SFH ratios `StudentT(df=2, scale=0.3)`, as `docs/tutorial.md` already did (was `scale=1.0`).
  On the Apple GPU demo_1 then recovers logmass 10.26 +/- 0.11 against the truth 10.50 (pull
  -2.3; -3.6 with `scale=1.0`). The quick start's quoted times and results are re-measured.
- The README quick start samples with `num_inner_steps=60`: in an Apple GPU (float32) settings
  study its logmass posterior came out about 30 % narrower with 15 or 30 inner steps than with
  60 or 120. The example scripts pass `num_inner_steps=30` explicitly (the `fitSED` default;
  `quickstart.py` used `max(8, 2 x n_dims)`); for demo_1, demo_2 and agn_demo the posteriors
  changed little between 30 and 120 steps.
- The dust-emission setup error names `add_dust=True, add_diffuse_dust=True`; `docs/agn.md`'s
  first block passes `add_dust=True` explicitly.
- `docs/agn.md`: `tau_pow` among the keys to fix, the two dust screens dust emission needs,
  and "What photometry recovers" (torus vs DL07 re-emission under energy balance).
  `mock_fit_1_photometry.ipynb` and `examples/README.md`: at 200 live points the
  metallicity-dust ridge is under-sampled (500 recovers the inputs), not "the data";
  `demo_1`/`demo_2` note the logmass-SFH degeneracy; `demo_postprocess.py` writes to
  `postprocess/` and keeps demo_1's figures.
- Posterior figures print enough decimals for a small error (`agn_elum` read
  0.00^{+0.00}_{-0.00}).
- **Apple GPU (float32): a fit could stop at once with a meaningless ln Z.** About 0.2 % of
  `StudentT(df=2)` draws of five `logsfr_ratios` push log10 SFR above 38.5, where `10**x`
  overflows float32; `logsfr_ratios_to_sfh` returned NaN, the live points' ln L was NaN,
  `logZ_live` was NaN and nested sampling "converged" after 0 likelihood calls (exit code 0).
  At 300 live points one initial draw is that extreme in 43 % of runs (`demo_1_mock_test.py`
  hit it). Another 0.16 % of draws (log10 SFR between 28.5 and 38.5) overflowed the mass
  integral instead and gave a silent all-zero SFH. The float32 path now shifts the peak log SFR to 0 before exponentiating (the unit-mass
  normalisation divides it out; float64 keeps its bit-identical path); the nested sampler
  treats a NaN ln L as -inf (outside the support), labels its float32 live-point ln Z as
  rescaled, and warns when a run ends with 0 likelihood calls.
  `tests/test_sfh_float32_overflow.py`.
- Posterior figures label every parameter with the symbol `latex()` and `display()` use
  (`describe.symbol_of`): dust, AGN, dust-emission and `log_<name>` keys, and the SFH ratios as
  log(SFR_i/SFR_{i+1}); they were the raw names ("diffuse tau kc", "log fagn bbb").

### float32 forward model, cheap GPUs, blackjax 1.7
- The large arrays of the forward model are float32: the SSP contraction, attenuation curves,
  the nebular profile contraction, the projector (resampling, FFT smoothing, instrument
  response, painted line profiles) and the photometry. A consumer GPU with slow float64 (1:64)
  does its work in float32; `jax_enable_x64` stays on, and float64 is kept where float32 would
  be wrong (dust-emission energy balance, AGN `L_bol`, sigma and chi², the likelihood's linear
  algebra, line fluxes, wavelengths). Every site is listed in `docs/conventions.md`,
  "Precision". A fit at the defaults peaked at 2.75-2.91 GB of device memory (A100), so an
  8 GB card runs it (`docs/installation.md`).
- The float32 projection is evaluated on `x / s`, `s` an exact power of two near max|x|, and
  multiplied back in float64: the same values, and the likelihood's cotangent (|r| / sigma² ~
  1e35-1e41 at cgs scale) no longer overflows float32 into a NaN gradient.
- `Observation` stores flux and uncertainty as float64 whatever dtype is passed (a float32 FITS
  spectrum is fine), also when assigned later; the likelihood promotes the float32 prediction to
  the data's float64 before squaring (sigma² ~ 1e-60 underflowed float32).
- Install: `pip install ceridwen` runs on the CPU, and on an Apple Silicon Mac also on the GPU
  (below); CUDA is the extra `ceridwen[cuda12]` or `ceridwen[cuda13]` (RTX 50-series needs
  CUDA >= 12.8: `cuda13`, or a fresh `cuda12`).
- blackjax 1.6 and 1.7 both work (floor 1.6, no cap). With 1.7 the lane kernel takes the
  Cholesky factor that `blackjax.nss` now passes (the same directions up to rounding), and on
  every version a replacement particle is born at the contour it replaced, including chains
  whose moves were all rejected: blackjax 1.7's fix (on a two-level plateau toy, the ln Z bias
  fell from +0.34 to +0.11; on a Gaussian nothing changes). Checkpoints resume across kernels as before.
- Euclid NISP Y, J, H passbands (`euclid_nisp_y`, `_j`, `_h`; ESA NISP-PHOTO-PASSBANDS V1.0,
  total throughput).
- Verification: a new gate tier, T5 (`scripts/physical_check.py`, `docs/development.md`): no
  spectrum, photometry, `Lines` prediction or line flux may move by more than 1e-3 relative
  (per element above 1e-4 of the array maximum), at 16 draws and four prior extremes.
- Numerical (CPU; the physical standard of `docs/development.md`, worst per element above 1e-4
  of the array maximum, `|diff| / max` below it), against the code before the float32
  projector, painters and nebular contraction: model-grid
  spectra <= 3.3e-6, projected spectra <= 2.5e-5, photometry <= 4.2e-4 (a faint band at the
  faint prior extreme), `Lines` predictions <= 3.3e-7, line fluxes <= 1.2e-6,
  over 20 configurations at 16 draws and four prior extremes. ln L moves by <= 6.2e-6 relative,
  gradients by <= 4.3e-5. Stored baselines that moved and were re-captured: the regression
  categories `agn_blr`, `agn_disk`, `agn_lines`, `agn_nlr`, `agn_polar`, `agn_skirtor`,
  `agn_torus` (<= 8.7e-6), `csp_components` (<= 1.3e-4: the model-grid line profiles' far
  wings), `csp_spectrum`, `dust_emission_themis`, `dust_laws`, `eline_marginal`, `lsf_scale`
  (<= 6.9e-7); the AGN golden spectra `agn_fsps`, `agn_corrected_dustemi`,
  `agn2_skirtor_polar_cigale`, `agn2_skirtor_polar_corrected_dustemi`, `agn2_nlr_blr_corrected`
  (spectra <= 1.2e-6, model-grid line spectra <= 5.2e-4, photometry <= 2.8e-8); the BLR golden
  (<= 1.5e-6). The SSP weights `W` do not move.

### AGN2: the CIGALE-standard AGN
- `AGNConfig(torus_model="skirtor16")`: the SKIRTOR 2016 torus + disc as CIGALE's `skirtor2016`
  (theta `skirtor_t`, `_pl`, `_q`, `_oa`, `_i`; static `skirtor_R`); `polar_dust=True` (+
  `polar_law`): CIGALE's polar dust (`polar_ebv`, `polar_temp`, `polar_emis`);
  `nlr_model="feltre16"`: the Feltre+16 narrow-line grid (`nlr_logu`, `nlr_logz`, `nlr_alpha`,
  `nlr_xid`, `nlr_lognh`; amplitude `agn_elum`); `blr=True`: broad H I lines (`blr_lum`,
  `blr_sigma` = sigma in km/s; Case-B ratios). All off by default: `AGNConfig()` is unchanged
  (byte-identical). Parity with pcigale 29cb909f: 1.6e-14 (SKIRTOR), 1.7e-12 (polar dust);
  Feltre+16 grid nodes 4.4e-16 dex (`docs/agn.md`, "AGN2").
- `agn_mode="cigale"`: reproduces CIGALE (fagn = CIGALE's fracAGN, no galaxy dust or energy
  balance on the AGN); the components are refused in `"fsps"` / `"prospector"` modes.
- The SKIRTOR and NLR tables are built by `scripts/agn2/build_skirtor_h5.py` /
  `build_nlr_h5.py` into `$CERIDWEN_AGN_DATA` (not shipped). CIGALE's three extinction curves
  of the polar dust are not shipped either (CeCILL-2.0): `python -m ceridwen.agn.fetch --from
  <cigale>` or `--download` puts them there (`ceridwen.agn.fetch_grids`), sha256-checked.
- `PostProcess`: with SKIRTOR, `extras["agn"]` adds `frac_agn_cigale` (CIGALE's fracAGN, both
  modes), CIGALE's `agn.*` infos (`cigale_<name>`), `L_dust_gal` and the SKIRTOR luminosities.
- `SedModel` refuses a prior that leaves the SKIRTOR or Feltre+16 grid (as for `gas_logz`), and a
  `fagn` prior reaching 1 in `"cigale"` mode.
- Tests: `tests/agn2/` (validation suite), `tests/agn2_dev/`; T2 categories `agn_skirtor`,
  `agn_polar`, `agn_nlr` (need `$CERIDWEN_AGN_DATA`), `agn_blr`; T3 goldens `agn2_*`.

### AGN
- AGN in the forward model: `CSPBasis(agn=AGNConfig(...))` / `CSPBasis_afe(agn=...)`, package
  `ceridwen.agn` (`docs/agn.md`). Everything Prospector has: the Nenkova+08 clumpy torus
  (`fagn`, `agn_tau`), the Temple+21 accretion disc (`fagn_bbb`), the AGN dust screen
  (`dust4_type=1`: `dust4`, `dust4_index`) and the AGN narrow lines (`agn_elum`,
  `agn_eline_sigma`; Richardson+14 'a42' ratios by FSPS line name). Modes: `"fsps"` and
  `"prospector"` reproduce FSPS and Prospector (python-fsps 0.5.0: 1.7e-8, FSPS's own floor;
  prospect 91f8e13: <= 1.7e-14 per component), `"corrected"` (the default of `AGNConfig()`;
  `agn=True` is short for it): exact `fagn * L_bol` normalisation on any grid, `agn_tau`
  clamped, the AGN before the IGM, absorbed AGN light re-emitted by DL07 with dust emission.
  Without `agn=` every prediction is byte-identical to before. Prospector's double-counted
  torus (python-fsps's own plus `fake_fsps.agn_torus`) is not reproduced (GOTCHAS 25).
- `PostProcess`: `prediction["spectra_agn"]` (+ `spectra_agn_observed`) and `extras["agn"]`
  (`L_bol_star`, `L_absorbed`, `mdust` with the energy balance); the summary figure draws the
  AGN. The result file records the AGN set-up in `csp_config["agn"]` (template hashes included).
- `marginalize_elines=True` with AGN lines is refused at setup (as in Prospector).
- `agn_eline_sigma` is a supported free parameter: `SedModel` refuses a prior / fixed value
  outside (0, 2000] km/s at setup, and warns when it is sampled but no Spectrum can resolve the
  AGN lines (no Spectrum, no unmasked pixel at an AGN line, or sigma_inst > 3x the prior's upper
  edge at every line). `scripts/agn_injection_recovery.py --spec grating|prism` fits it free;
  `docs/agn.md`, "Fitting the AGN line width".
- `scripts/check_api_usage.py` also checks the keyword arguments of base-less dataclasses
  (`AGNConfig`).

### Speedups (adapted from work by Liu Hao, @potatoist314)
- Nested sampling is faster: `slice_kernel` option on `BlackJAXNestedSamplerAdapter` (and
  `fitSED(sampler_kwargs=...)`), default `"lanes"` (every particle's slice chain as one lane of a
  shared loop, Dance et al. 2025); `"stock"` is the unchanged `blackjax.nss`, `"carry"` evaluates
  each stepping-out edge once. All use the same random numbers and evaluations as `blackjax.nss`:
  bitwise equal on CPU (`"lanes"` on Apple silicon; to the last bits on x86), statistically
  equal on GPU, where `"lanes"` sampled 3.5-3.8x faster on an
  A100 for a 19-dimensional JADES model (`docs/samplers.md`).
- Nested sampling: `log_evidence_err` is drawn with a seed derived from `rng_key` (it varied
  from run to run at the same key); numpy's global RNG is restored afterwards.
- `CSPBasis(wavelength_support=True)` / `CSPBasis_afe(...)`: opt-in, off by default. `predict`
  synthesises only the model pixels the observations read (`Observation.model_support`) and
  zero-fills the rest after the scaling; the whole grid for dust emission, painted lines and
  free-z photometry. Not byte-identical (last float32 bit); 2.5-3x faster log-likelihood on
  CPU at W = 64-100 for a JWST photometry (+ grating) model on the BPASS grid (GOTCHAS 24).
- `SSPData.ssp_flux` / `SSPDataAfe.ssp_flux` are read-only host NumPy arrays (a JAX array
  passed in is copied to the host). The CSP casts to float32 on the device from a transient
  copy, so a fit keeps only the float32 cube on the device (CPU, grid alive: test BPASS grid
  88.8 -> 29.8 MiB, aMIST hr afe grid 1166.7 -> 583.4 MiB); `csp.flux` is byte-identical.

## 1.0.12 (2026-09-24)
First release accompanying Stoffers et al. (2026).
