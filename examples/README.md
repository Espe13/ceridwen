# Examples

## `quickstart.py`

A complete, self-contained fit: loads an SSP grid, makes
mock UV-to-IR photometry (GALEX+SDSS+2MASS+WISE) from known parameters, recovers them with BlackJAX nested
sampling, post-processes the fit with `PostProcess`, prints true-vs-posterior
values, and writes the summary, corner and sampling-diagnostic figures (truth
marked in green) plus `quickstart_post.npz` to `quickstart_figures/`.

Install CERIDWEN and FSPS as in the [README](../README.md#installation), then run it with the
published grid:

```bash
SSP_FILE=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_miles_chab'))") \
    python examples/quickstart.py
```

The script resolves the grid in this order: `$SSP_FILE`, then `examples/ssp_data.h5`, then
the local developer grid `ceridwen/data/test_data/ssp_data_bpass.h5` (not in the
repository); with none of them it builds `examples/ssp_data.h5` with `python-fsps`
([building your own SSP grid](../docs/installation.md#building-your-own-ssp-grid)).

It uses demo settings (150 live points, `logZ_tol=-2`) and is not a converged fit. Broadband
photometry alone constrains `logmass` well but `logzsol` and the dust only weakly: their
posteriors are broad and can miss the truth by about 2 sigma (one run recovered
`logzsol` -1.25 (-0.43/+0.48) against a truth of -0.20, with a bimodal posterior). It writes
`quickstart_post.npz` (~145 MB) next to the figures.

For a spectrum and photometry fitted jointly, see the [Quick start](../README.md#quick-start);
for emission lines, free redshift and the joint fit see [`docs/tutorial.md`](../docs/tutorial.md)
and `demo_2_photometry_lines.py` / `demo_3_spectrum_advanced.py` below.

## Grids and run times

Every script uses `examples/ssp_data.h5` when you have built one, and otherwise the published
grid `mist_miles_chab` (`fetch_grid`, downloaded once to `~/.ceridwen/grids`, about 67 MB);
python-fsps is not needed. The scripts that use nebular emission, dust emission or an AGN
need `$SPS_HOME`.

Wall times measured on an 11-core Apple M3 Pro shared with other jobs (so upper bounds):

| example | CPU (float64) | Apple GPU (float32) |
|---|---|---|
| `quickstart.py` | 2 min | 2 min |
| `demo_1_mock_test.py` | 11-14 min | |
| `demo_2_photometry_lines.py` | 26-55 min | |
| `demo_3_spectrum_advanced.py` (NUTS, prints no progress while it samples) | over 45 min | |
| `demo_eline_marginalisation.py` (500 live points) | over 45 min | |
| `agn_demo.py --fit` | 56 min | 6 min |
| `mock_fit_1_photometry.ipynb` | 11-19 min | 5-6 min |
| `mock_fit_2_photometry_lines.ipynb` | over 45 min | 14-17 min |
| `mock_fit_3_spectrum.ipynb`, `mock_fit_4_spectrum_photometry.ipynb` | | 8 and 6 min |
| `apple_gpu_mock_fit.ipynb` | | 7 min |

## Notebooks

`apple_gpu_mock_fit.ipynb` and `mock_fit_1..4_*.ipynb` build a mock galaxy, fit it and compare
the posterior with the truth: photometry alone, photometry with line fluxes, a spectrum alone,
and a spectrum calibrated against photometry. They need Jupyter in your environment
(`pip install jupyterlab`); to run one headless,
`jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 <notebook>`.
With photometry alone (`mock_fit_1`) the metallicity and the dust are degenerate, and
`logzsol` can pile up at the prior edge. At the notebook's 200 live points that ridge is
under-sampled: the dust ranges come out too narrow and missed the input by more than 3 sigma in
a CPU run; with `num_live=500` the inputs fell inside the 95 % ranges, with the same ln Z.

## `demo_1_mock_test.py`, `demo_2_photometry_lines.py`, `demo_3_spectrum_advanced.py`

Mock fits of increasing complexity: photometry only; photometry plus
emission-line fluxes (`add_neb=True`, needs `$SPS_HOME`); photometry plus a
spectrum with an instrumental LSF, a fitted stellar velocity dispersion, line
masking and a noise floor. Each writes its fit to
`demo_N_output/`, post-processes it with `PostProcess`, and writes the three
figures to `demo_N_output/figures/` with the injected truths marked. In `demo_1` and `demo_2`
`logmass` trades against the SFH shape, as in the README quick start: it came out about
0.3 dex low (2.5-2.9 sigma) in newcomer runs, with the truth's ln L inside the posterior's
range, so the miss is the data's.

## `agn_demo.py`

An obscured AGN at z = 2 in NIRCam and MIRI bands: prints the AGN share of each band in the
three AGN modes and, with `--fit`, fits the torus fraction, the torus optical depth and the
disc with nested sampling ([AGN](../docs/agn.md); needs `$SPS_HOME`). In this mock the torus
fraction and the stellar mass are recovered and the disc and `agn_tau` only weakly; that does
not hold for every mock (`docs/agn.md`, "What photometry recovers").

## `demo_postprocess.py`

Post-processes the saved `demo_1_output/ceridwen_result.h5` (the default; pass another
result file as the argument) from a rebuilt model (no refit), adds a user-defined derived quantity, and writes `post.npz`
and the figures to `postprocess/` next to the result file (demo_1's own figures are kept).

## `demo_afe_quiescent.py`

[alpha/Fe] fitting of a quiescent galaxy from a Legacy-Surveys-style spectrum
(DECam `grz` + WISE photometry) with the alpha-enhanced, nebular-free
`CSPBasis_afe`. Fits a 10-bin continuity SFH, metallicity `logzsol` (= [Fe/H] on the alpha grid),
alpha-enhancement `afe`, diffuse-dust optical depth and slope, at fixed
redshift, plus a fitted spectrophotometric normalisation `spectrum_scaling` that
rescales the observed spectrum onto the photometric flux scale (photometry
anchors the absolute level). Self-contained: it injects a known `spectrum_scaling`
into a mock and recovers it alongside `afe` with the nested slice sampler.
Needs the alpha-enhanced grid `amist_c3k_hr_krou_afe`: `amist_c3k_hr_krou_afe.h5` next to
the script, or the script fetches it with `fetch_grid("amist_c3k_hr_krou_afe")` (612 MB,
cached).

```bash
python examples/demo_afe_quiescent.py
```

Note the two independent spectrum calibrations: `spectrum_scaling` scales the whole
`Spectrum`, `eline_scaling` scales only emission **lines** (inert here, since
the alpha-enhanced model carries no nebular emission). See
[`docs/tutorial.md`](../docs/tutorial.md) and `GOTCHAS.md`.

## `demo_eline_marginalisation.py`

Emission-line marginalisation on a NIRSpec-like spectrum at z = 2.5 whose lines do not
follow the CLOUDY prediction ([O III] 3x, [O II] 0.5x, Balmer 1.5x, [Ne III] 2x): the line
fluxes are marginalised analytically (`Spectrum(marginalize_elines=True)`) and the recovered
fluxes are compared with the truth. Needs `$SPS_HOME` (nebular grids). See
`docs/eline_marginalisation.md`.

## `make_mock_data.py`

Writes a self-consistent mock galaxy (photometry + an optical spectrum) by pushing known
TRUTH parameters through the same forward model and adding Gaussian noise
(`examples/mock_galaxy.npz`). Builds the SSP grid with FSPS only when `ssp_data.h5` is not
next to it.

## `recipes/`

Prospector features written as small add-ons, each with a check script against Prospector;
several have since landed in the package. See `recipes/README.md`.
