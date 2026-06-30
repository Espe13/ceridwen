# CERIDWEN

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven
**WE**ight calculatio**N**s — a JAX-native, GPU-capable spectral energy
distribution (SED) fitting package for galaxies, with variational-inference
preconditioned Hamiltonian Monte Carlo, nested sampling, and native redshift
support.

The forward model is a single differentiable XLA graph: stellar populations →
dust attenuation and emission → nebular continuum and line emission → IGM
attenuation → redshift-aware observed-frame projection. Inference runs on CPU or
GPU.

## Features

- Star formation history (non-parametric continuity + parametric)
- Metallicity history (constant or time-varying)
- Dust attenuation (Kriek & Conroy diffuse, power-law birth-cloud, multi-component age-dependent)
- Dust emission (Draine & Li grids)
- Nebular continuum + emission lines (CLOUDY grids)
- Observations: broadband photometry, spectra, emission-line fluxes — fit singly or jointly
- Redshift-aware forward model with cosmological flux normalisation
- IGM attenuation (Madau 1995), extensible via an `IGMModel` ABC
- NUTS / nested sampling / VI-preconditioned NUTS

## Where to next

- **[Installation](installation.md)** — Python 3.11+, dependencies, and the FSPS / `$SPS_HOME` setup.
- **[Quick start](quickstart.md)** — build a model and fit it end to end.
- **[Conventions & gotchas](conventions.md)** — the unit and indexing conventions that trip people up (read this before fitting real data).
- **[API reference](api.md)** — the public classes and functions.

## Citing

If you use CERIDWEN, please cite it — see `CITATION.cff` in the repository (the
methods paper reference will be added on publication).
