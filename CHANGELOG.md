## 1.0.13 (2026-10-14)

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
- Install: `pip install ceridwen` is CPU-only everywhere; CUDA is the extra `ceridwen[cuda12]`
  or `ceridwen[cuda13]` (RTX 50-series needs CUDA >= 12.8: `cuda13`, or a fresh `cuda12`).
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
