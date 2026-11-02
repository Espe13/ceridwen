# CERIDWEN

CERIDWEN fits galaxy spectral energy distributions to infer stellar populations,
dust properties, nebular emission and star formation histories. It accepts
photometry, spectra and emission-line fluxes, separately or in a joint fit.

The forward model uses JAX to combine stellar populations, apply dust attenuation
and emission, add nebular continuum and lines, and account for intergalactic
absorption. It then projects the spectrum into the observed frame at the galaxy's
redshift. The model is differentiable, and inference can run on a CPU or GPU.

## Model components

- Star formation histories on a lookback-time grid, using non-parametric
  continuity (`logsfr_ratios`) or parametric forms supplied as transforms.
- Constant metallicity or a metallicity history (`zh_const`).
- Sampled [α/Fe] with `CSPBasis_afe` and the aMIST + C3K grid.
- Kriek & Conroy diffuse dust, power-law birth clouds and 12 attenuation laws,
  including Calzetti, SMC, LMC, Gordon+03 SMC bar and Reddy+15.
- Dust emission from Draine & Li (2007) or THEMIS templates.
- Nebular continuum and emission lines from FSPS CLOUDY grids.
- Madau (1995) IGM attenuation, with optional damping wing and damped Ly-α
  absorption through `MadauDampingDLA`.
- Broadening from stellar and gas velocity dispersions and the instrumental
  LSF, accounting for the SSP library resolution.

Fits can include noise rescaling, jitter, calibration polynomials, an outlier
mixture and a GP for correlated spectral residuals. Emission-line fluxes and
calibration coefficients can be marginalised analytically. Inference options
are nested sampling, NUTS, VI-preconditioned NUTS and MAP optimisation.
`PostProcess` computes derived quantities, posterior predictions and figures.

## Documentation

- [Installation](installation.md): dependencies, FSPS data and SSP grids.
- [Quick start](quickstart.md): fit a mock galaxy on a laptop.
- [Joint-fit tutorial](tutorial.md): photometry, a spectrum and line fluxes.
- [Samplers](samplers.md): settings, checkpoints and MAP initialisation.
- [Post-processing](postprocessing.md): SFRs, UV and ionising-photon quantities,
  predictions and figures.
- [Noise and calibration](noise_calibration.md), [GP likelihood](gp_likelihood.md),
  [Outlier model](outlier_model.md) and [Emission-line marginalisation](eline_marginalisation.md).
- [α/Fe](afe.md): the α-enhanced grid and `CSPBasis_afe`.
- [Conventions and common errors](conventions.md): units, metallicity, lookback
  time, broadening and cosmology.
- [Troubleshooting](troubleshooting.md) and [API reference](api.md).
- [Development and verification](development.md): tests and development practices.

## Citing

If you use CERIDWEN in your research, please cite it:

```bibtex
@misc{stoffers2026ceridwen,
  author       = {Stoffers, Amanda},
  title        = {{CERIDWEN}: Fast and Flexible {GPU}-Accelerated Stellar Population Inference},
  year         = {2026},
  note         = {Version 1.0.13},
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

CERIDWEN's internal filter-convolution and attenuation-curve code comes from
Benjamin D. Johnson's [sedpy](https://github.com/bd-j/sedpy), through the JAX
rewrite [sedpy_jax](https://github.com/Espe13/sedpy_jax). It also follows
sedpy's AB photon-counting conventions. Neither package is a dependency.

Maintainer: [Amanda Stoffers](https://www.amanda-stoffers.de), Kavli Institute for Cosmology, University of Cambridge, `aas208@cam.ac.uk`
