# Examples

## `quickstart.py`

A complete, self-contained fit: builds (or loads) the FSPS SSP cache, makes
mock UV-to-IR photometry (GALEX+SDSS+2MASS+WISE) from known parameters, recovers them with BlackJAX nested
sampling, and prints true-vs-posterior values plus two figures: a corner plot
(`quickstart_corner.png`) and a model-vs-data SED with a chi residual strip
(`quickstart_sed.png`).

No FSPS needed — the quickstart runs off a bundled SSP grid:

```bash
git lfs install && git lfs pull    # fetch the bundled test grid (~120 MB)
pip install .                      # everything except FSPS
python examples/quickstart.py
```

The script resolves the grid in this order: `$SSP_FILE` →
`examples/ssp_data.h5` → the LFS fixture `tests/fixtures/ssp_data_test.h5`.
No git-lfs? Download the grid from Zenodo to `examples/ssp_data.h5` instead
(see `docs/installation.md`, "Getting the SSP grid").

FSPS is only needed to *build your own* grid (then it is written to
`examples/ssp_data.h5` and re-used), and for `add_neb=True`, which reads CLOUDY
nebular emission from `$SPS_HOME` at runtime — see the README for the FSPS
install.

For spectroscopy, emission lines, free redshift, and VI-preconditioned NUTS, see
the end-to-end template in the
[README](../README.md#step-1--fit-a-galaxy-end-to-end) and the joint-fit walk-through
in [`docs/tutorial.md`](../docs/tutorial.md).
