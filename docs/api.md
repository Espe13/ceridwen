# API reference

Auto-generated from the source docstrings. The public API is the symbols below;
treat everything else as internal.

## Stellar populations

::: ceridwen.ssps.SSPData

::: ceridwen.ssps.SSPDataAfe

::: ceridwen.ssps.fetch_grid

::: ceridwen.ssps.available_grids

## Composite stellar population (forward model)

::: ceridwen.csp.CSPBasis

::: ceridwen.csp.CSPBasis_afe

## Broadening

::: ceridwen.broadening.Kinematics

::: ceridwen.broadening.Instrument

::: ceridwen.broadening.DEFAULT_KINEMATICS

::: ceridwen.broadening.SpectralProjector

::: ceridwen.broadening.PhotometricBroadener

## Observations

::: ceridwen.observation.Photometry

::: ceridwen.observation.Spectrum

::: ceridwen.observation.Lines

## Model

::: ceridwen.model.SedModel

::: ceridwen.model.transforms.logsfr_ratios_to_sfh

## Priors

::: ceridwen.priors

## Likelihood

::: ceridwen.likelihood.DiagonalGaussianLikelihood

::: ceridwen.likelihood.DiagonalGaussianLikelihoodWithUpperLimits

::: ceridwen.likelihood.MultiObservationLikelihood

## Fitting

::: ceridwen.fit.fitSED

::: ceridwen.fit.load_result_h5

::: ceridwen.fit.read_result_h5

::: ceridwen.fit.result_cosmology

::: ceridwen.sampler.run_sampler

::: ceridwen.sampler.nested.BlackJAXNestedSamplerAdapter

::: ceridwen.sampler.nuts.BlackJAXNUTSAdapter

## Post-processing

::: ceridwen.postprocess.PostProcess

::: ceridwen.postprocess.SpectrumSample

::: ceridwen.postprocess.load_postprocess

## Figures

::: ceridwen.plotting.summary_figure

::: ceridwen.plotting.corner_figure

::: ceridwen.plotting.diagnostic_figure

::: ceridwen.plotting.make_figures

## Dust, nebular, IGM, cosmology

::: ceridwen.dust.Dust

::: ceridwen.dust.DiffuseDust

::: ceridwen.dust.DustEmission

::: ceridwen.neb.NebularModel

::: ceridwen.igm

::: ceridwen.cosmology

## Spectrophotometric calibration

::: ceridwen.csp.spectrum_calibration.spectrum_calibration_factor

::: ceridwen.csp.spectrum_calibration.legendre_design_matrix
