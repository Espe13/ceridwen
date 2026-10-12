## Unreleased (AGN2: the CIGALE-standard AGN)
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

## Unreleased (AGN)
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

## Unreleased (speedups from Liu Hao's fork, potatoist314/ceridwen)
- Nested sampling: `slice_kernel` option on `BlackJAXNestedSamplerAdapter` (and
  `fitSED(sampler_kwargs=...)`): `"stock"` (default, unchanged), `"carry"` (each stepping-out
  edge evaluated once) and `"lanes"` (every particle's slice chain as one lane of a shared loop,
  Dance et al. 2025). Both use the same random numbers and evaluations as `blackjax.nss`: bitwise
  equal on CPU, statistically equal on GPU (`docs/samplers.md`).
- Nested sampling: `log_evidence_err` is drawn with a seed derived from `rng_key` (it varied
  from run to run at the same key); numpy's global RNG is restored afterwards.
- Emission-line marginalisation: FSPS's `"[O II] 3867"` (zero CLOUDY flux in the MIST,
  Padova and Parsec cubes, negligible in BPASS; 1.8 A from [Ne III] 3869) is no longer a
  default fitted line (it stays at its CLOUDY flux) and is refused in `elines_to_fit`.
- `CSPBasis(wavelength_support=True)` / `CSPBasis_afe(...)`: opt-in, off by default. `predict`
  synthesises only the model pixels the observations read (`Observation.model_support`) and
  zero-fills the rest after the scaling; the whole grid for dust emission, painted lines and
  free-z photometry. Not byte-identical (last float32 bit); 2.5-3x faster log-likelihood on
  CPU at W = 64-100 for a JWST photometry (+ grating) model on the BPASS grid (GOTCHAS 24).
- `SedModel._hoist_flux_factor` (private, off): the fixed-redshift flux factor computed once at
  construction and passed to `CSPBasis.predict(flux_factor=...)` instead of per call; byte-identical,
  no CPU gain beyond noise, kept for a GPU benchmark (`scripts/gpu_bench_E2.py`).
- `SSPData.ssp_flux` / `SSPDataAfe.ssp_flux` are read-only host NumPy arrays (a JAX array
  passed in is copied to the host). The CSP casts to float32 on the device from a transient
  copy, so a fit keeps only the float32 cube on the device (CPU, grid alive: test BPASS grid
  88.8 -> 29.8 MiB, aMIST hr afe grid 1166.7 -> 583.4 MiB); `csp.flux` is byte-identical.
- These mechanisms come from Liu Hao's speedups work on his fork of CERIDWEN.

## 1.0.12 (2026-09-24)
First release accompanying Stoffers et al. (2026).
