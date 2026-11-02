# Fitting [α/Fe]

The α-enhanced grid uses FSPS v4.0, aMIST isochrones and C3K spectra, with
[α/Fe] ∈ {−0.2, 0.0, +0.2, +0.4, +0.6}. It lets you fit [α/Fe] alongside
metallicity. [α/Fe] can help constrain the formation timescale, although it is
degenerate with total metallicity.

`SSPDataAfe` and `CSPBasis_afe` extend `SSPData` and `CSPBasis` with an [α/Fe]
axis and interpolation. They omit the nebular arguments and inherit the SFH
weights, dust, projection, flux scaling and cosmology.

## The grid

Download the grid by name. The 612 MB file is cached and verified with SHA-256.
To build it yourself, compile python-fsps from source with `AFE_FLAG=1` against
the FSPS v4.0 data tree. Check both settings before building.

```python
import jax.numpy as jnp
from ceridwen.ssps import fetch_grid, SSPDataAfe
from ceridwen.csp import CSPBasis_afe
from ceridwen import Cosmology

ssp = SSPDataAfe.load(fetch_grid("amist_c3k_hr_krou_afe"))
ssp.display()                       # (n_afe, n_Z, n_age, n_wave) = (5, 13, 107, 10992)

csp = CSPBasis_afe(ssp, lookback_time=jnp.linspace(0.0, 12.0, 9),
                   cosmo=Cosmology.planck18(), zh_const=True, verbose=False)

theta = {
    "lookback_time":      jnp.linspace(0.0, 12.0, 9),
    "sfh":                jnp.exp(-jnp.linspace(0.0, 12.0, 9) / 1.0),
    "logzsol":            jnp.array([-0.3]),   # = [Fe/H] on an aMIST grid
    "afe":                jnp.array([0.4]),    # [α/Fe]: re-partitions that Z
    "tau_pow":            jnp.array([0.3]),
    "alpha_pow":          jnp.array([-1.0]),
    "diffuse_tau_kc":     jnp.array([0.2]),
    "diffuse_dust_index": jnp.array([0.0]),
}
wave, fnu = csp.wave, csp.get_spectrum(theta)   # rest-frame Lsun/Hz per Msun
print(csp)                                      # grids, [Fe/H] axis, Z_sun, parameters
i = int(jnp.argmin(jnp.abs(wave - 5500.0)))
print(f"{fnu.shape[0]} pixels, L_nu(5500 A) = {float(fnu[i]):.3e} Lsun/Hz per Msun")
```

`amist_c3k_hr_krou_afe` uses M. J. Park's alpha-MC C3K high-resolution SSPs
(MIST v2.5 + C3K v2.3) and a Kroupa IMF. Between 3000 and 9000 Å, the spectra
are resampled at λ/Δλ = 6000 per pixel. The two-pixel resolution limit is
therefore R ≈ 3000; `ssp.display()` reports it. The native FSPS metallicity
axis is log10 Z = [Fe/H] + log10(0.0185), so `logzsol` represents [Fe/H].

## What differs from `CSPBasis`

- `theta["afe"]` is interpolated linearly between adjacent grid planes. It supports
  `jit`, `grad`, `vmap` and all samplers.
- All [α/Fe] planes share the FSPS `AFE_FLAG=1` [Fe/H] axis. Total metallicity
  is derived as `logzsol_total = logzsol + log10(1 - x + x 10^[α/Fe])`, with
  x = 0.687490 for MIST v2.5, GS98.
- There is no nebular model because α-enhanced CLOUDY grids are unavailable.
  `Lines` observations raise an error. The basis reads from `$SPS_HOME` only
  when `add_dust_emission=True`.
- `CSPBasis_afe` requires an α-aware 4-D grid. Passing a solar-scaled 3-D grid
  raises a `TypeError` directing you to `CSPBasis`. When constructing a grid
  by hand, pass `ssp_afe=` as a keyword.
- A prior that includes both `logzsol > +0.25` and [α/Fe] > +0.4 raises an error
  at construction. FSPS supplies the same isochrone file at [Fe/H] = +0.5 for
  [α/Fe] = +0.4 and +0.6. `display()` identifies this cell.
- The grid has no surviving-mass table. Use `fitSED(..., mfrac=False)` and
  report formed mass.
- Sampling `afe` uses interpolation weights in the [α/Fe] contraction rather
  than constructing a flux plane for each sample. Vectorised evaluation uses
  roughly the same memory as evaluation without `afe`. For
  `jit(vmap(csp.get_spectrum))` on the CPU, XLA's compile-time `memory_analysis`
  reports temporary memory of 0.005 GB at W = 1, 0.002 GB at W = 80 and
  0.011 GB at W = 400 for this grid. The basis also stores two float32 copies
  of the flux cube, each 306 MB.

`examples/demo_afe_quiescent.py` fits a quiescent galaxy with [α/Fe] and a
sampled spectrophotometric normalisation.
