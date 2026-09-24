# Troubleshooting

Start with the environment check. It reports the Python version, every dependency, the
bundled filter curves and attenuation laws, float64, nested sampling and `$SPS_HOME`, each
problem with its fix:

```bash
python -m ceridwen.check
```

**`$SPS_HOME` is unset or incomplete.** `CSPBasis(add_neb=True)` or
`add_dust_emission=True` raises a `ValueError` when neither `sps_home=` nor `$SPS_HOME` is
given. Clone FSPS and set the variable in your shell startup file
([Installation](installation.md)). Open a new terminal and check that `echo $SPS_HOME` prints
the path; if it is blank, the line went into the wrong file (`echo $SHELL` tells you which
shell you use).

**`python-fsps` does not import.** You need it only to build your own SSP grid
(`SSPData.from_fsps`). It compiles Fortran against `$SPS_HOME`, so set `$SPS_HOME` first and
install a Fortran compiler ([Installation](installation.md#building-your-own-ssp-grid)).

**`fetch_grid` says the cached grid fails its checksum.** The file in `~/.ceridwen/grids` (or
`$CERIDWEN_GRID_DIR`) is incomplete or an earlier release. `fetch_grid(name, force=True)`
downloads it again.

**An import picks up the wrong `ceridwen`.** Started from the folder that contains a clone,
Python finds the clone folder `ceridwen` before the installed package. Run scripts from
inside the clone or from any other folder.

**NUTS prints nothing for a long time.** Warmup and sampling are each one compiled loop, so
there is no progress output, and Ctrl-C does not stop them; use `kill`. On a CPU, NUTS takes
hours for a typical fit; use nested sampling there, or a GPU ([Samplers](samplers.md)).

**A test skips.** A grid-dependent test skips when it finds no SSP grid; a skip is not a pass.
Point the tests at the published BPASS grid:

```bash
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
```

**A number looks plausible but wrong.** The common causes are a metallicity in the wrong
units (`logzsol` is log10(Z/Z_sun) of the grid), a lookback-time grid in the wrong order
(index 0 is today), and a misspelt `theta` key. See [Conventions](conventions.md) and
[`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md). If an AI assistant
helps you use CERIDWEN, point it at
[`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md).
