## 1.0.13 (2026-10-10)
- Nested sampling is faster: the inner slice kernel of `BlackJAXNestedSamplerAdapter` is
  `slice_kernel="lanes"` by default (every replacement's slice chain runs as one lane of a
  shared loop). On CPU it gives exactly the samples of `blackjax.nss` (`slice_kernel="stock"`);
  on GPU it agrees statistically and sampled 3.5-3.8x faster on an A100 for a 19-dimensional
  JADES model. `slice_kernel="carry"` is also available (`docs/samplers.md`).
- `log_evidence_err` is reproducible: the same `rng_key` gives the same error bar.
- Emission-line marginalisation: FSPS's `"[O II] 3867"` (zero or negligible CLOUDY flux, 1.8 A
  from [Ne III] 3869) is not a default fitted line; it stays at its CLOUDY flux.
- `CSPBasis(wavelength_support=True)` (opt-in): synthesises only the model pixels the
  observations read; 1.3-1.5x faster log-likelihood on an A100 (GOTCHAS section 24).
- Opt-in windowed emission-line basis for marginalised lines with a sampled line width
  (`docs/eline_marginalisation.md`).
- A fit keeps only the float32 SSP cube on the device (`SSPData.ssp_flux` is host memory):
  583 MiB instead of 1167 MiB for the high-resolution [alpha/Fe] grid.
- The speed-ups in this release are adapted from work by Liu Hao (@potatoist314).

## 1.0.12 (2026-09-24)
First release accompanying Stoffers et al. (2026).
