# CERIDWEN

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven
**WE**ight calculatio**N**s. CERIDWEN fits the spectral energy distributions of galaxies
(photometry, spectra and emission-line fluxes, alone or jointly) to infer their stellar
populations, dust, nebular emission and star formation histories. The forward model is a
single differentiable JAX graph: it builds a galaxy spectrum from stellar populations, applies
dust attenuation and emission, adds nebular continuum and lines, attenuates it by the
intergalactic medium, and projects it to the observed frame at the galaxy's redshift.
Inference runs on a CPU or a GPU.

## Features

- Star formation history on a lookback-time grid: non-parametric continuity (`logsfr_ratios`) or any parametric form written as a transform
- Constant metallicity or a metallicity history (`zh_const`)
- [α/Fe] as a sampled stellar axis (`CSPBasis_afe`, aMIST + C3K grid)
- Dust attenuation: Kriek & Conroy diffuse dust, power-law birth clouds, 12 registered laws (Calzetti, SMC, LMC, Gordon+03 SMC bar, Reddy+15, ...)
- Dust emission from Draine & Li (2007) or THEMIS templates
- Nebular continuum and emission lines from the FSPS CLOUDY grids
- IGM attenuation (Madau 1995), with an optional damping wing and damped Ly-α absorber (`MadauDampingDLA`)
- Photometry, spectra and emission-line fluxes, fitted alone or jointly
- One broadening kernel: stellar and gas velocity dispersions, instrument line-spread function, library resolution removed automatically
- Analytic marginalisation over emission-line fluxes (`Spectrum(marginalize_elines=True)`)
- Noise and calibration terms: jitter, calibration polynomials (sampled, profiled or marginalised), an outlier mixture, a Gaussian-process likelihood for correlated residuals
- Samplers: nested sampling (with the evidence), NUTS, VI-preconditioned NUTS, and a MAP optimiser
- Post-processing (`PostProcess`): SFRs, M_UV, ionising photon rates, posterior-predictive data, and summary, corner and diagnostic figures

## Where to next

- [Installation](installation.md): the package, FSPS and `$SPS_HOME`, SSP grids.
- [Quick start](quickstart.md): one fit end to end, on a laptop CPU.
- [Tutorial: joint fit](tutorial.md): photometry, a spectrum and emission-line fluxes fitted together.
- [Samplers](samplers.md): nested sampling, NUTS, VI-preconditioned NUTS, the MAP.
- [Post-processing](postprocessing.md): derived quantities, predictions and figures.
- [Noise and calibration](noise_calibration.md), [GP likelihood](gp_likelihood.md), [Outlier model](outlier_model.md), [Emission-line marginalisation](eline_marginalisation.md).
- [α/Fe](afe.md): the α-enhanced grid and `CSPBasis_afe`.
- [Conventions & gotchas](conventions.md): units, metallicity, lookback time, broadening, cosmology. Read this before fitting real data.
- [Troubleshooting](troubleshooting.md) and the [API reference](api.md).
- [Development and verification](development.md): the test layers and how the code is built.

## Citing

If you use CERIDWEN in your research, please cite it:

```bibtex
@misc{stoffers2026ceridwen,
  author       = {Stoffers, Amanda},
  title        = {{CERIDWEN}: Fast and Flexible {GPU}-Accelerated Stellar Population Inference},
  year         = {2026},
  note         = {Version 1.0.11},
  howpublished = {\url{https://github.com/Espe13/ceridwen}}
}
```

## References

- **Hoffman et al. 2019**, *NeuTra-lizing Bad Geometry in HMC Using Neural Transport*, [arXiv:1903.03704](https://arxiv.org/abs/1903.03704) (VI-preconditioned NUTS, `ceridwen.sampler.vi`)
- **Madau 1995**, ApJ 441, 18 (IGM transmission, `ceridwen.igm.Madau1995`)
- **Miralda-Escudé 1998** and **Totani et al. 2006** (IGM damping wing), **Tepper-García 2006, 2007** (Voigt profile of the DLA), as cited by and ported from Prospector (`ceridwen.igm.MadauDampingDLA`)
- **Planck Collaboration 2020**, A&A 641, A6 (the Planck18 cosmology, `ceridwen.cosmology`)
- **Kriek & Conroy 2013**, ApJ 775, L16 (diffuse dust attenuation shape, `ceridwen.dust`)
- **Conroy, Gunn & White 2009** (FSPS, the SSP and data-file provider)

## Related projects

[sedpy](https://github.com/bd-j/sedpy) by Benjamin D. Johnson is the origin of the filter-convolution and attenuation-curve code that CERIDWEN carries internally (via [sedpy_jax](https://github.com/Espe13/sedpy_jax), the JAX rewrite), and of the AB photon-counting conventions it follows. CERIDWEN depends on neither package.

Maintainer: [Amanda Stoffers](https://www.amanda-stoffers.de), Kavli Institute for Cosmology, University of Cambridge, `aas208@cam.ac.uk`
