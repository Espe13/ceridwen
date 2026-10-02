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
