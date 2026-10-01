## Unreleased (speedups from Liu Hao's fork, potatoist314/ceridwen)
- Nested sampling: `slice_kernel` option on `BlackJAXNestedSamplerAdapter` (and
  `fitSED(sampler_kwargs=...)`): `"stock"` (default, unchanged), `"carry"` (each stepping-out
  edge evaluated once) and `"lanes"` (every particle's slice chain as one lane of a shared loop,
  Dance et al. 2025). Both use the same random numbers and evaluations as `blackjax.nss`: bitwise
  equal on CPU, statistically equal on GPU (`docs/samplers.md`).
- Nested sampling: the step kernel compiles in a thread while the live points initialise.
- Nested sampling: `log_evidence_err` is drawn with a seed derived from `rng_key` (it varied
  from run to run at the same key); numpy's global RNG is restored afterwards.
- Emission-line marginalisation: FSPS's phantom `"[O II] 3867"` (zero CLOUDY flux, no NIST
  line, 1.8 A from [Ne III] 3869) is no longer a default fitted line and is refused in
  `elines_to_fit`.
- These mechanisms come from Liu Hao's speedups work on his fork of CERIDWEN.

## 1.0.12 (2026-09-24)
First release accompanying Stoffers et al. (2026).
