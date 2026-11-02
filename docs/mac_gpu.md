# Apple GPU (float32 mode)

CERIDWEN uses [jax-graft](https://github.com/dfm/jax-graft) to run on Apple
Silicon GPUs through JAX's `mtl` platform. Apple GPUs do not support float64,
so this backend uses float32. The sections below describe installation,
numerical differences, performance and unsupported features.

## Install and run

`examples/apple_gpu_mock_fit.ipynb` generates mock photometry and a spectrum,
fits them with nested sampling on the GPU and compares recovered parameters
with their inputs and priors. It takes about three and a half minutes on
an M3 Pro. Four other notebooks cover specific observation types:

- `mock_fit_1_photometry.ipynb`: photometry only.
- `mock_fit_2_photometry_lines.ipynb`: photometry and line fluxes, including gas parameters.
- `mock_fit_3_spectrum.ipynb`: a spectrum with per-pixel resolving power, sampled
  velocity dispersion and emission lines.
- `mock_fit_4_spectrum_photometry.ipynb`: a spectrum with a sampled flux scale,
  constrained by photometry.

On Apple Silicon with macOS 26 and Python 3.12 or newer, `pip install ceridwen`
includes jax-graft. Older Macs and Python 3.11 use the CPU build, as specified
by the dependency marker in `pyproject.toml`. Jax-graft selects the GPU as
JAX's default device. Importing CERIDWEN then switches the process to float32
and issues a warning. JAX and tensorflow-probability also issue warnings;
see [Known limits](#known-limits).

```
UserWarning: float32 mode: JAX's default backend is the Apple GPU (mtl), which has no float64.
Evidences, gradients and the sampler state are float32 (docs/mac_gpu.md); for a float64 fit
set JAX_PLATFORMS=cpu
```

`python -m ceridwen.check` reports:

```
  [ok  ] jax float64  -- off: float32 mode, chosen because the Apple GPU is JAX's default backend ...
  [ok  ] jax backend  -- mtl (mtl:0): Apple GPU via jax-graft, float32
```

Each `fitSED` header reports `Device : MTL (mtl:0)` and
`Precision : float32`. Result files save these as `config/platform` (`"mtl"`)
and `config/precision` (`"float32"`).

Environment variables control the device and precision:

| Environment | Precision | Device |
|---|---|---|
| no variables set; jax-graft installed | float32 | Apple GPU |
| `JAX_PLATFORMS=cpu` | float64 | CPU |
| `CERIDWEN_X64=0` (any machine) | float32 | whatever JAX's default device is |
| `CERIDWEN_X64=1` with the GPU as default | refused at import | set `JAX_PLATFORMS=cpu` instead |

`CERIDWEN_X64=0` also enables float32 on CPU or CUDA devices. This is useful
for comparing with Mac results. Use float64 when the device supports it
unless you need that comparison.

## Implementation of float32 mode

The forward model, likelihood and samplers use float32 with the adjustments
below. These adjustments have no effect in float64. Compared with version
1.0.13, the float64 compiled programs were byte-identical across 21,445 arrays
in the T4 check. All 1680 output arrays in `scripts/physical_check.py` were
also identical.

**Rescaling observations.** A cgs F_nu spectrum can have uncertainties near
1e-31. Squaring these underflows float32, whose smallest normal value is
1.2e-38. Before evaluating sigma², inverse variance or A^T W A,
`ceridwen.likelihood.observation_data` scales each observation's data,
uncertainties and calibration vector by a power of two, bringing the median
uncertainty close to one. Power-of-two scaling preserves the mantissa.
The sampled log-jitter is shifted by ln(scale).

The scaled likelihood differs from the physical-unit likelihood by
`−ndof ln(scale)`, constant for each observation. Sampling retains this
offset: adding a constant of about 1e6 inside a float32 kernel would reduce
ln L resolution to about 0.1. `run_sampler` and `map_fit` restore the constant
on the host in float64 for reported ln Z, ln L and MAP log-posterior.
The values are available as `likelihood.float32_rescaling` and
`likelihood.loglike_offset`.

**Array types.** Device arrays use `ceridwen.precision.float_dtype()`.
Host-side NumPy calculations retain float64, including SSP axes, Z_sun
resolution, provenance and result-file data.

**Numerical range.** The ionising-photon table uses a constant L_sun / h =
5.8e59. It is computed on the host in float64 before storage in the working
type. The metallicity and dust-emission kernels replace their 1e-70 floors
with the smallest normal float32 value.

Float32 mode does not support `Spectrum(marginalize_elines=True)`, whose
normal equations use flux units. It also rejects fixed numeric jitter,
which cannot be rescaled with the data; use sampled jitter instead.

## Numerical comparisons

`tests/test_float32_parity.py` compares six golden configurations and a
nine-parameter mock galaxy with SDSS photometry, a 600-pixel cgs spectrum
(sigma ~ 1e-31) and three line fluxes. Separate processes evaluate float64
on the CPU, float32 on the CPU and float32 on the GPU. These measurements
used an M3 Pro, JAX 0.11.2 and jax-graft 0.0.5 on 15 October 2026:

| Quantity | float32 vs float64 | Metal vs float32 CPU | Metal run to run |
|---|---|---|---|
| SSP weights `W` | 3e-6 (step) / 5e-6 (linear) of the largest weight | 2e-7 | identical |
| spectrum, continuum pixels | 5e-6 relative | 2e-7 | identical |
| spectrum, pixels under a nebular line | up to 4e-4 relative (4e-5 of the peak) | 2e-7 | identical |
| photometry (maggies) | 2e-5 relative | 9e-7 (mock), 3e-5 (golden 15000-pixel integrals) | identical |
| line fluxes | 4e-6 relative | 2e-7 | identical |
| ln posterior (true units) | 4e-4 and 0.015 on values of 4e4 and 2e4 | 0.01 | identical |
| gradient | 5e-5 of the largest component | | up to 6e-4 relative (three runs: 1e-5, 8e-5, 6e-4) |

Line-pixel differences arise from evaluating `(lambda - lambda_0) / sigma`
in float32. Near 1e4 Å, float32 resolves about 1e-3 Å, which shifts the
profile wings slightly. CPU float32 shows the same effect.

GPU gradients vary between runs because the transpose of redshift
interpolation uses atomic scatter-adds. The forward predictions and ln L
were byte-identical between runs in these checks.

The tested float32 posteriors agreed with float64 within sampling variation.
Evidence agreement required twice as many inner steps; see [Speed](#speed).
Float32 and float64 outputs differ numerically, and GPU nested-sampling
runs with the same `rng_key` need not be identical.

## Speed

`scripts/bench_mac_metal.py` fits the comparison mock in each precision mode
and times value-only likelihood calls at the sampler's batch widths:

```bash
python tests/baselines/_capture_f32.py --no-goldens ref.npz
JAX_PLATFORMS=cpu                python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f64_cpu
JAX_PLATFORMS=cpu CERIDWEN_X64=0 python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f32_cpu
                                 python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f32_mtl
python scripts/bench_mac_metal.py --compare bench/*/summary.json
```

The benchmark used an M3 Pro, JAX 0.11.2 and jax-graft 0.0.5 on 15 October
2026, with nine free parameters, `num_live=200` and `num_delete=40`.
`us/point` is the median value-only ln L runtime under `vmap` over 200 calls,
excluding compilation:

| run | inner steps | ln Z | calls | wall | us/point W=40 | us/point W=200 |
|---|---|---|---|---|---|---|
| CPU float64, seed 0 | 30 | 40933.16 +- 0.36 | 168 000 | 1731 s | 1550 | 1335 |
| CPU float64, seed 1 | 30 | 40933.21 +- 0.32 | 169 200 | 1879 s | | |
| CPU float32, seed 0 | 30 | 40930.96 +- 0.40 | 180 000 | 1017 s | 765 | 750 |
| Metal float32, seed 0 | 30 | 40931.88 +- 0.39 | 176 400 | 421 s | 174 | 131 |
| Metal float32, seed 1 | 30 | 40932.04 +- 0.46 | 174 000 | 418 s | | |
| Metal float32, seed 0 | 60 | 40933.77 +- 0.38 | 328 800 | 908 s | | |

In this benchmark, GPU likelihood evaluations were 9–10 times faster than
float64 CPU evaluations and 4–6 times faster than float32 CPU evaluations.
The 30-step GPU fit took about a quarter of the float64 CPU runtime.
The sampler's control loop still runs on the CPU.

The evidence estimates were sensitive to inner-step count in float32.
The two float64 seeds differed by 0.05 in ln Z. With 30 inner slice steps,
float32 estimates on both devices were 1.1–2.2 nats lower, or 3–6 times
their quoted errors. With 60 steps, the GPU estimate was 40933.77 ± 0.38,
consistent with float64. The float32 slice sampler appears to mix less
efficiently per step; the cause is unknown. Peak ln L agreed within 0.3
across runs, while incomplete mixing in nested sampling can bias Z downward.

Posterior means and widths showed similar variation in all six runs.
The two float64 runs differed by up to 1.8 standard deviations in their
means, with width ratios of 0.3–1.7 at 200 live points. Float32 results fell
within this variation.

When comparing evidences, use matching precision, `num_live`, `num_delete`
and `num_inner_steps`, and check mixing. In this test, changing precision
or doubling the inner steps shifted ln Z by 1–2 nats. `fitSED` records
precision, platform and sampler settings in the result file. The default
remains 30 inner steps in both precision modes.

`scripts/mac_metal_probe.py` compares the values and runtimes of 34 JAX
operations on CPU and GPU. All 34 agreed in the comparison; the script
prints the results.

## Known limits

- Jax-graft 0.0.5 is experimental. Unsupported operations fail during
  compilation with `UNIMPLEMENTED: Metal: ...`. Include the `fitSED` header
  and `python -m ceridwen.check` output when reporting a failure.
- Tensorflow-probability requests float64 internally. On the first prior
  draw, JAX warns that it is truncating those requests to float32, with
  messages from tfp's `random_generators.py`, `numpy_array.py` and `ops.py`,
  and JAX's `array_methods.py`. JAX also prints `Platform 'mtl' is experimental ...`.
  These warnings are expected. XLA may report slow constant folding for
  large models; this indicates compilation time rather than a failed fit.
- The nested-sampling progress bar reports ln Z in scaled units. It may
  differ from the final value by `ndof ln(scale)`, which can be thousands
  of nats for a spectrum. The final report and result file use physical units.
- The nested sampler dispatches its inner loop step by step. Small GPU
  batches can be slower than CPU batches. Wider batches and gradient-based
  samplers make better use of the GPU.
