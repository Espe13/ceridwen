# Examples

## `quickstart.py`

A complete, self-contained fit: builds (or loads) the FSPS SSP cache, makes
mock UV-to-IR photometry (GALEX+SDSS+2MASS+WISE) from known parameters, recovers them with BlackJAX nested
sampling, post-processes the fit with `PostProcess`, prints true-vs-posterior
values, and writes the summary, corner and sampling-diagnostic figures (truth
marked in green) plus `quickstart_post.npz` to `quickstart_figures/`.

Recommended route — build the SSP grid yourself with FSPS (you control
isochrones, spectral library, and IMF; see the README for the FSPS install):

```bash
pip install .                      # everything except FSPS
pip install "fsps>=0.4.4"          # needs gfortran + $SPS_HOME; see README
export SPS_HOME=/path/to/fsps
python examples/quickstart.py      # builds examples/ssp_data.h5 on first run
```

No FSPS? A pre-built grid always works too, by name (downloaded once into
`~/.ceridwen/grids`, sha256-verified):

```python
from ceridwen.ssps import fetch_grid, SSPData
ssp = SSPData.load(fetch_grid("mist_miles_chab"))
```

or by hand to the quickstart location:

```bash
curl -L -o examples/ssp_data.h5 \
    "https://zenodo.org/records/21977508/files/ssp_data_mist_miles.h5?download=1"
```

The script resolves the grid in this order: `$SSP_FILE` →
`examples/ssp_data.h5` → the local developer grid
`ceridwen/data/test_data/ssp_data_bpass.h5` (not shipped in the repository).

`add_neb=True` additionally needs `$SPS_HOME` at runtime (CLOUDY nebular
data), i.e. an FSPS data checkout even with a downloaded grid.

For spectroscopy and VI-preconditioned NUTS see the end-to-end template in the
[README](../README.md#step-1-fit-a-galaxy-end-to-end); for emission lines, free
redshift and the joint fit see [`docs/tutorial.md`](../docs/tutorial.md) and
`demo_2_photometry_lines.py` / `demo_3_spectrum_advanced.py` below.

## `demo_1_mock_test.py`, `demo_2_photometry_lines.py`, `demo_3_spectrum_advanced.py`

Mock fits of increasing complexity: photometry only; photometry plus
emission-line fluxes (`add_neb=True`, needs `$SPS_HOME`); photometry plus a
spectrum with an instrumental LSF, a fitted stellar velocity dispersion, line
masking and a noise floor. Each writes its fit to
`demo_N_output/`, post-processes it with `PostProcess`, and writes the three
figures to `demo_N_output/figures/` with the injected truths marked.

## `demo_postprocess.py`

Post-processes the saved `demo_1_output/ceridwen_result.h5` from a rebuilt
model (no refit), adds a user-defined derived quantity, and writes `post.npz`
and the figures next to the result file.

## `demo_afe_quiescent.py`

[alpha/Fe] fitting of a quiescent galaxy from a Legacy-Surveys-style spectrum
(DECam `grz` + WISE photometry) with the alpha-enhanced, nebular-free
`CSPBasis_afe`. Fits a 10-bin continuity SFH, total metallicity `Z`,
alpha-enhancement `afe`, diffuse-dust optical depth and slope, at fixed
redshift, plus a fitted spectrophotometric normalisation `spectrum_scaling` that
rescales the observed spectrum onto the photometric flux scale (photometry
anchors the absolute level). Self-contained: it injects a known `spectrum_scaling`
into a mock and recovers it alongside `afe` with the nested slice sampler.
Needs an alpha-enhanced grid: `amist_c3k_lr_chab_afe.h5` next to the script or
under `ceridwen/data/test_data/` (not shipped), or the script falls back to
`fetch_grid("amist_c3k_lr_chab_afe")`; that published copy predates schema 2,
so convert it once with `python scripts/convert_grids_schema2.py
~/.ceridwen/grids/amist_c3k_lr_chab_afe.h5` and use the `_schema2.h5` it writes
(or use the schema-2.1 `amist_c3k_hr_krou_afe` grid).

```bash
python examples/demo_afe_quiescent.py
```

Note the two independent spectrum calibrations: `spectrum_scaling` scales the whole
`Spectrum`, `eline_scaling` scales only emission **lines** (inert here, since
the alpha-enhanced model carries no nebular emission). See
[`docs/tutorial.md`](../docs/tutorial.md) and `GOTCHAS.md`.
