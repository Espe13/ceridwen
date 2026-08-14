# Examples

## `quickstart.py`

A complete, self-contained fit: builds (or loads) the FSPS SSP cache, makes
mock UV-to-IR photometry (GALEX+SDSS+2MASS+WISE) from known parameters, recovers them with BlackJAX nested
sampling, and prints true-vs-posterior values plus two figures: a corner plot
(`quickstart_corner.png`) and a model-vs-data SED with a chi residual strip
(`quickstart_sed.png`).

Recommended route — build the SSP grid yourself with FSPS (you control
isochrones, spectral library, and IMF; see the README for the FSPS install):

```bash
pip install .                      # everything except FSPS
pip install "fsps>=0.4.4"          # needs gfortran + $SPS_HOME; see README
export SPS_HOME=/path/to/fsps
python examples/quickstart.py      # builds examples/ssp_data.h5 on first run
```

No FSPS? A pre-built grid always works too — either download from Zenodo
([doi:10.5281/zenodo.21221634](https://doi.org/10.5281/zenodo.21221634)):

```bash
curl -L -o examples/ssp_data.h5 \
    "https://zenodo.org/records/21221634/files/ssp_data.h5?download=1"
```

or use the bundled LFS test grid (`git lfs install && git lfs pull`). The
script resolves the grid in this order: `$SSP_FILE` →
`examples/ssp_data.h5` → the LFS fixture `tests/fixtures/ssp_data_test.h5`.

`add_neb=True` additionally needs `$SPS_HOME` at runtime (CLOUDY nebular
data), i.e. an FSPS data checkout even with a downloaded grid.

For spectroscopy, emission lines, free redshift, and VI-preconditioned NUTS, see
the end-to-end template in the
[README](../README.md#step-1-fit-a-galaxy-end-to-end) and the joint-fit walk-through
in [`docs/tutorial.md`](../docs/tutorial.md).

## `demo_afe_quiescent.py`

[alpha/Fe] fitting of a quiescent galaxy from a Legacy-Surveys-style spectrum
(DECam `grz` + WISE photometry) with the alpha-enhanced, nebular-free
`CSPBasis_afe`. Fits a 10-bin continuity SFH, total metallicity `Z`,
alpha-enhancement `afe`, diffuse-dust optical depth and slope, at fixed
redshift, plus a fitted spectrophotometric normalisation `spectrum_scaling` that
rescales the observed spectrum onto the photometric flux scale (photometry
anchors the absolute level). Self-contained: it injects a known `spectrum_scaling`
into a mock and recovers it alongside `afe` with the nested slice sampler.
Needs the alpha-enhanced grid `amist_c3k_lr_chab_afe.h5` (bundled under
`ceridwen/data/test_data/`, or `fetch_grid("amist_c3k_lr_chab_afe")`).

```bash
python examples/demo_afe_quiescent.py
```

Note the two independent spectrum calibrations: `spectrum_scaling` scales the whole
`Spectrum`, `eline_scaling` scales only emission **lines** (inert here, since
the alpha-enhanced model carries no nebular emission). See
[`docs/tutorial.md`](../docs/tutorial.md) and `GOTCHAS.md`.
