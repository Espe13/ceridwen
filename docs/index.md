# CERIDWEN

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven
**WE**ight calculatio**N**s. A JAX-native, GPU-capable spectral energy
distribution (SED) fitting package for galaxies, with variational-inference
preconditioned Hamiltonian Monte Carlo, nested sampling, and native redshift
support.

The forward model is a single differentiable XLA graph. It builds a galaxy
spectrum from stellar populations, applies dust attenuation and emission, adds
nebular continuum and line emission, attenuates the intergalactic medium, and
projects the result to the observed frame at the galaxy's redshift. Inference
runs on CPU or GPU.

## Features

- Star formation history (non-parametric continuity or parametric)
- Metallicity history (constant or time-varying)
- [α/Fe] as a sampled stellar axis (`CSPBasis_afe`, FSPS v4.0 aMIST + C3K grids; continuum-only, downloadable grid — no FSPS install needed)
- Dust attenuation (Kriek & Conroy diffuse, power-law birth-cloud, multi-component age-dependent)
- Dust emission (Draine & Li grids)
- Nebular continuum and emission lines (CLOUDY grids)
- Broadband photometry, spectra, and emission-line fluxes, fit on their own or jointly
- One broadening kernel: galaxy stellar and gas velocity dispersions (`Kinematics`, fixed or sampled) combined in quadrature with the instrument's line-spread function (`Instrument`, unit and convention in the constructor name) and the SSP library resolution, which is removed automatically
- Redshift-aware forward model with cosmological flux normalisation
- IGM attenuation (Madau 1995), extensible through an `IGMModel` base class
- NUTS, nested sampling, and VI-preconditioned NUTS
- Post-processing (`PostProcess`): derived quantities, posterior-predictive data, and per-galaxy summary, corner and sampling-diagnostic figures

## Where to next

- [Installation](installation.md): Python 3.11, dependencies, and the FSPS / `$SPS_HOME` setup.
- [Quick start](quickstart.md): build a model and fit it end to end.
- [Tutorial: joint fit](tutorial.md): photometry, a spectrum, and emission-line fluxes fitted together.
- [Post-processing](postprocessing.md): derived quantities, predictions, and the figures.
- [Conventions & gotchas](conventions.md): the unit and indexing conventions that catch people out, and where the three spectral widths are set. Read this before fitting real data.
- [API reference](api.md): the public classes and functions.

## Citing

If you use CERIDWEN in your research, please cite it:

```bibtex
@misc{stoffers2026ceridwen,
  author       = {Stoffers, Amanda},
  title        = {{CERIDWEN}: Fast and Flexible {GPU}-Accelerated Stellar Population Inference},
  year         = {2026},
  note         = {Version 1.0.5},
  howpublished = {\url{https://github.com/Espe13/ceridwen}}
}
```
