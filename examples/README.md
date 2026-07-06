# Examples

## `quickstart.py`

A complete, self-contained fit: builds (or loads) the FSPS SSP cache, makes
mock UV-to-IR photometry (GALEX+SDSS+2MASS+WISE) from known parameters, recovers them with BlackJAX nested
sampling, and prints true-vs-posterior values plus two figures: a corner plot
(`quickstart_corner.png`) and a model-vs-data SED with a chi residual strip
(`quickstart_sed.png`).

```bash
pip install .                      # everything except FSPS
pip install "fsps>=0.4.4"          # FSPS wrapper (needs gfortran + $SPS_HOME; see README)
export SPS_HOME=/path/to/fsps      # FSPS root directory (contains nebular/, ...)
python examples/quickstart.py
```

The SSP cache is written to `examples/ssp_data.h5` on first run and re-used
afterwards. Override with `SSP_FILE=/some/path.h5`.

FSPS is required even though this demo has `add_neb=False`: Step 0 builds the
SSP grid from FSPS. Flip `add_neb=True` in the script to add CLOUDY nebular
emission (read from `$SPS_HOME` at runtime).

For spectroscopy, emission lines, free redshift, and VI-preconditioned NUTS, see
the end-to-end template in the
[README](../README.md#step-1--fit-a-galaxy-end-to-end) and the joint-fit walk-through
in [`docs/tutorial.md`](../docs/tutorial.md).
