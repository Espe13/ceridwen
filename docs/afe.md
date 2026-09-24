# Fitting [α/Fe]

The α-enhanced grid (FSPS v4.0, aMIST isochrones + C3K spectra,
[α/Fe] ∈ {−0.2, 0.0, +0.2, +0.4, +0.6}) adds α-element enhancement as a sampled stellar axis:
a chemical clock for the formation timescale, measured jointly with, and degenerate with, the
total metallicity. `SSPDataAfe` and `CSPBasis_afe` are subclasses of `SSPData` and
`CSPBasis`. The [α/Fe] axis and its interpolation are the only additions, the nebular
arguments are dropped, and everything else (SFH weights, dust, projection, flux factor,
cosmology) is inherited.

## The grid

Download it by name (612 MB, cached and SHA-256 verified like every published grid).
Building it yourself needs python-fsps compiled from source with `AFE_FLAG=1` against the
FSPS v4.0 data tree, which is easy to get silently wrong.

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

`amist_c3k_hr_krou_afe` is built from the alpha-MC C3K high-resolution SSPs (MIST v2.5 +
C3K v2.3) of M. J. Park, with a Kroupa IMF. The C3K spectra are resampled onto pixels of
λ/Δλ = 6000 between 3000 and 9000 Å, so the file resolves R ≈ 3000 at the two-pixel floor
there; `ssp.display()` reports the floor. Its native axis is the FSPS label
log10 Z = [Fe/H] + log10(0.0185), i.e. `logzsol` = [Fe/H].

## What differs from `CSPBasis`

- `theta["afe"]` is interpolated linearly between the two bracketing grid planes, so it works
  under `jit`, `grad` and `vmap` and in every sampler.
- `logzsol` is [Fe/H]: every [α/Fe] plane shares one [Fe/H] axis (FSPS `AFE_FLAG=1`). The
  total metallicity [Z/H] is the derived `logzsol_total`
  = `logzsol + log10(1 - x + x 10^[α/Fe])`, x = 0.687490 (MIST v2.5, GS98).
- There is no nebular model (no α-enhanced CLOUDY grids exist): `Lines` observations are
  refused, and the basis reads nothing from `$SPS_HOME` unless `add_dust_emission=True`.
- `CSPBasis_afe` accepts only α-aware 4-D grids; a solar-scaled 3-D grid raises a `TypeError`
  that points to `CSPBasis`. `ssp_afe=` is keyword-only when you construct a grid by hand.
- `logzsol` above +0.25 together with [α/Fe] above +0.4 is refused: FSPS ships the same
  isochrone file for [α/Fe] = +0.4 and +0.6 at [Fe/H] = +0.5, and `display()` lists the cell.
  A prior that reaches it raises at construction.
- The grid does not carry a surviving-mass table yet, so fit it with
  `fitSED(..., mfrac=False)` and quote the formed mass.
- Memory: a sampled `afe` enters the contraction as interpolation weights over the [α/Fe]
  axis, so no interpolated flux plane is built per sample and vectorised evaluation
  (`jax.vmap`, as nested sampling does over its live points) costs about as much memory as
  without `afe`. XLA's temporary memory for `jit(vmap(csp.get_spectrum))` on
  `amist_c3k_hr_krou_afe` (CPU, compile-time `memory_analysis`): 0.005 GB at W = 1, 0.002 GB
  at W = 80, 0.011 GB at W = 400. The basis keeps two copies of the flux cube (2 × 306 MB for
  this grid, float32).

A full fit of a quiescent galaxy with [α/Fe], including a fitted spectrophotometric
normalisation, is `examples/demo_afe_quiescent.py`.
