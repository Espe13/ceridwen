# API reference

Auto-generated from the source docstrings. The public API is the symbols below;
treat everything else as internal.

## Stellar populations

::: ceridwen.ssps.SSPData

::: ceridwen.ssps.SSPDataAfe

::: ceridwen.ssps.fetch_grid

::: ceridwen.ssps.available_grids

### Grid metallicity metadata

::: ceridwen.ssps.grid_metadata.logzsol_total

::: ceridwen.ssps.grid_metadata.f_alpha

::: ceridwen.ssps.grid_metadata.chash_arrays

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

The outlier mixture of `DiagonalNoiseModel` is off by default (`f_outlier=None`, i.e. 0);
all outlier fractions default to 0, switch the mixture on explicitly
([outlier model](outlier_model.md)).

::: ceridwen.likelihood.DiagonalNoiseModel

::: ceridwen.likelihood.lnlike_diag_outlier

::: ceridwen.likelihood.lnlike_diag_outlier_with_upper_limits

::: ceridwen.likelihood.outlier_probability

::: ceridwen.likelihood.poly_calibration.PolynomialCalibration

::: ceridwen.model.obs_params.check_names

## Fitting

::: ceridwen.fit.fitSED

::: ceridwen.fit.load_result_h5

::: ceridwen.fit.read_result_h5

::: ceridwen.fit.read_provenance

::: ceridwen.fit.convert_result

::: ceridwen.fit.result_cosmology

::: ceridwen.resultfile.rebuild_model

::: ceridwen.resultfile.check_model_against_result

::: ceridwen.resultfile.priors_from_result

::: ceridwen.optimize.map_fit

::: ceridwen.optimize.MAPResult

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
