# Apple GPU (float32 mode)

CERIDWEN runs on the GPU of an Apple Silicon Mac through
[jax-graft](https://github.com/dfm/jax-graft), which registers JAX's `mtl` platform. Apple
GPUs have no float64, so on that backend CERIDWEN runs in **float32 mode**. This page says
what to install, what the mode changes, what it guarantees and what it refuses.

## Install and run

`pip install ceridwen` installs jax-graft on an Apple Silicon Mac with macOS 26 and
Python 3.12 or newer (the only platform jax-graft ships wheels for; the marker in
`pyproject.toml` keeps older Macs and Python 3.11 on the CPU build). Nothing else is needed:
jax-graft makes the GPU JAX's default backend, and `import ceridwen` sees that and switches
the process to float32, with one warning:

```
UserWarning: float32 mode: JAX's default backend is the Apple GPU (mtl), which has no float64.
Evidences, gradients and the sampler state are float32 (docs/mac_gpu.md); for a float64 fit
set JAX_PLATFORMS=cpu
```

`python -m ceridwen.check` reports the two lines

```
  [ok  ] jax float64  -- off: float32 mode, chosen because the Apple GPU is JAX's default backend ...
  [ok  ] jax backend  -- mtl (MtlDevice(id=0)): Apple GPU via jax-graft, float32
```

and every `fitSED` header prints `Device : MTL (MtlDevice(id=0))` and
`Precision : float32`. The result file records both: `config/precision` (`"float32"`) and
`config/platform` (`"mtl"`).

The environment switches:

| Environment | Precision | Device |
|---|---|---|
| nothing set (jax-graft installed) | float32 | Apple GPU |
| `JAX_PLATFORMS=cpu` | float64 | CPU, the same numbers as any other CPU |
| `CERIDWEN_X64=0` (any machine) | float32 | whatever JAX's default device is |
| `CERIDWEN_X64=1` with the GPU as default | refused at import | set `JAX_PLATFORMS=cpu` instead |

`CERIDWEN_X64=0` on a CPU or CUDA machine is the same float32 mode without the GPU: useful
to reproduce a Mac result elsewhere, not a precision to choose when float64 is available.

## What float32 mode changes inside the package

The forward model, the likelihood and the samplers run unchanged in float32. Three things
are done differently so that they survive it; all three are no-ops under float64, and the
float64 compiled programs are byte-identical to 1.0.13 (T4: 21 445 arrays compared, none
differ; the physical standard, `scripts/physical_check.py`: 1680 arrays, every output
identical).

**Data rescaling at the observation layer.** A spectrum in cgs F_nu has sigma ~ 1e-31, whose
square underflows float32 (smallest normal number 1.2e-38), so every kernel that forms
sigma^2, 1/sigma^2 or A^T W A would be wrong. `ceridwen.likelihood.observation_data` multiplies
each observation's data, uncertainties and calibration vector by one power of two that brings
its median uncertainty to order unity (exact: a power of two changes no mantissa bit), the
noise model's jitter parameter (data units) is shifted by ln(scale), and the kernels run in
those units. Their ln L is the true ln L minus `ndof ln(scale)`, a constant of the
observation; the sampler works in those units (adding ~1e6 inside a float32 kernel would
quantise ln L to 0.1), and `run_sampler` / `map_fit` add the constant back, in float64 on the
host, to the reported ln Z, ln L and MAP ln posterior. `likelihood.float32_rescaling` and
`likelihood.loglike_offset` expose the numbers.

**No literal float64 on the device.** `ceridwen.precision.float_dtype()` is the working dtype;
every device array whose dtype the package chooses goes through it. Host-side NumPy stays
float64 everywhere: the SSP grid axes, Z_sun resolution, provenance and the result file.

**Constants outside the float32 range.** The ionising-photon table (`L_sun / h` = 5.8e59)
is computed on the host in float64 and stored in the working dtype, and the 1e-70 floors
of the metallicity and dust-emission kernels become the smallest normal float32.

Refused in float32 mode, with the error saying so: `Spectrum(marginalize_elines=True)` (the
line marginal forms A^T W A in flux units) and a noise model with a *fixed numeric* jitter
(it cannot be rescaled with the data; sample it instead).

## What the mode guarantees

`tests/test_float32_parity.py` runs the six golden configurations and a mock galaxy in
physical units (SDSS photometry, a 600-pixel cgs spectrum with sigma ~ 1e-31, three line
fluxes; nine free parameters) in float64, in float32 on the CPU and in float32 on the GPU, in
separate processes. Measured on an M3 Pro (jax 0.11.2, jax-graft 0.0.5, 2026-10-15):

| Quantity | float32 vs float64 | Metal vs float32 CPU | Metal run to run |
|---|---|---|---|
| SSP weights `W` | 3e-6 (step) / 5e-6 (linear) of the largest weight | 2e-7 | identical |
| spectrum, continuum pixels | 5e-6 relative | 2e-7 | identical |
| spectrum, pixels under a nebular line | up to 4e-4 relative (4e-5 of the peak) | 2e-7 | identical |
| photometry (maggies) | 2e-5 relative | 9e-7 (mock), 3e-5 (golden 15000-pixel integrals) | identical |
| line fluxes | 4e-6 relative | 2e-7 | identical |
| ln posterior (true units) | 4e-4 and 0.015 on values of 4e4 and 2e4 | 0.01 | identical |
| gradient | 5e-5 of the largest component | | up to 6e-4 relative (three runs: 1e-5, 8e-5, 6e-4) |

The line-pixel error comes from evaluating the profile argument `(lambda - lambda_0) /
sigma` in float32: at 1e4 Angstrom a float32 resolves 1e-3 Angstrom, so the profile wings move
by that fraction of a pixel. It is a property of the mode, not of the GPU (the CPU float32
run shows the same numbers). The gradient varies run to run on the GPU because the transpose
of the redshift interpolation is a scatter-add, which is atomic there; the forward model and
ln L are byte-identical run to run.

So: a float32 fit gives the same posterior as a float64 fit to well within the sampling
error, and the same ln Z to better than the nested-sampling error, but **its numbers are not
comparable bit for bit** with a float64 fit, and two nested-sampling runs with the same
`rng_key` are only identical on the CPU.

## Speed

`scripts/bench_mac_metal.py` fits the parity mock with nested sampling in each mode and times
the sampler's value-only ln L at its batch widths:

```bash
python tests/baselines/_capture_f32.py --no-goldens ref.npz
JAX_PLATFORMS=cpu                python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f64_cpu
JAX_PLATFORMS=cpu CERIDWEN_X64=0 python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f32_cpu
                                 python scripts/bench_mac_metal.py --mock-data ref.npz --out bench/f32_mtl
python scripts/bench_mac_metal.py --compare bench/*/summary.json
```

BENCH_TABLE_PLACEHOLDER

`scripts/mac_metal_probe.py` times the 34 JAX operations CERIDWEN uses one by one on the CPU
and the GPU and checks their values agree (all 34 do; the table is in its output).

## Known limits

- jax-graft is at version 0.0.5 ("expect some sharp edges", its README). An operation it
  cannot compile fails at compile time with `UNIMPLEMENTED: Metal: ...`; report it with the
  `fitSED` header and `python -m ceridwen.check`.
- tensorflow-probability's `StudentT` sampler asks for float64 internally and is truncated to
  float32 with a `UserWarning` the first time a prior is drawn. It is harmless.
- The nested sampler's inner loop is dispatched step by step; at small batch widths the GPU is
  slower than the CPU (see the benchmark), so the GPU pays off for wide batches and
  gradient-based samplers.
