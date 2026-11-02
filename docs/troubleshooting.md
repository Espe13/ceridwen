# Troubleshooting

Run the environment check first. It checks Python, dependencies, filter curves,
attenuation laws, float64, nested sampling and `$SPS_HOME`, and suggests fixes
for any problems:

```bash
python -m ceridwen.check
```

**pip reports "No matching distribution found for jaxlib>=0.9.0" on a Mac.**
This usually means Python is an Intel (`x86_64`) build. JAX has no Intel-Mac
wheels after 0.4.38, while CERIDWEN requires JAX >= 0.9. On Apple Silicon,
an older Intel Anaconda installation running under Rosetta can cause this.
Check Python's architecture with:

```bash
python -c "import platform; print(platform.machine())"     # must print arm64
```

If it prints `x86_64`, install [Miniforge](https://github.com/conda-forge/miniforge) for
arm64 and create the environment with its `conda`. Intel Macs cannot run CERIDWEN.

**`$SPS_HOME` is unset or incomplete.** `CSPBasis(add_neb=True)` and
`add_dust_emission=True` require either `sps_home=` or `$SPS_HOME`. Clone FSPS
and add the path to your shell startup file; see [Installation](installation.md).
Open a new terminal and run `echo $SPS_HOME`. If it is blank, check that you
edited the startup file for your shell (`echo $SHELL`).

**`python-fsps` fails to import.** The wrapper is needed only for
`SSPData.from_fsps`, which builds SSP grids. Set `$SPS_HOME` and install a
Fortran compiler before installing it. See
[Building a grid](installation.md#building-your-own-ssp-grid).

**A cached grid fails its checksum.** The file in `~/.ceridwen/grids` or
`$CERIDWEN_GRID_DIR` may be incomplete or from an earlier release. Download
a replacement with `fetch_grid(name, force=True)`.

**Python imports the wrong `ceridwen`.** Running from the directory containing
a clone can cause Python to find that folder before the installed package.
Run the script from inside the clone or another directory.

**NUTS produces no output for a long time.** Warmup and sampling each run as a
compiled loop without progress output. Ctrl-C does not interrupt these loops;
use `kill` to stop the process. Typical CPU fits take hours. Use nested
sampling on the CPU or run NUTS on a GPU; see [Samplers](samplers.md).

**Grid-dependent tests are skipped.** These tests cannot run without an SSP
grid. Point them at the published BPASS grid:

```bash
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
```

**Results have unexpected values.** Check metallicity units (`logzsol` is
log10(Z/Z_sun) for the loaded grid), lookback-time ordering (index 0 is today)
and spelling of `theta` keys. See [Conventions](conventions.md) and
[`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md).
If you use an AI assistant, give it the repository's
[`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md) instructions.
