# Examples

## `quickstart.py`

A complete, self-contained fit: builds (or loads) the FSPS SSP cache, makes
mock SDSS photometry from known parameters, recovers them with BlackJAX nested
sampling, and prints true-vs-posterior values plus an optional corner plot.

```bash
pip install -e ".[grids,nested]"
export SPS_HOME=/path/to/fsps      # FSPS root directory (contains nebular/, ...)
python examples/quickstart.py
```

The SSP cache is written to `examples/ssp_data.h5` on first run and re-used
afterwards. Override with `SSP_FILE=/some/path.h5`.

FSPS is required even though this demo has `add_neb=False`: Step 0 builds the
SSP grid from FSPS. Flip `add_neb=True` in the script to add CLOUDY nebular
emission (read from `$SPS_HOME` at runtime).

Larger end-to-end drivers (spectroscopy, emission lines, free redshift, VI-
preconditioned NUTS) live in `../scripts/`.
