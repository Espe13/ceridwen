# Installation

CERIDWEN needs Python 3.11 or newer (`jax` >= 0.9 and `blackjax` >= 1.6 require it).

## 1. The package

```bash
conda create -n ceridwen python=3.11 -y && conda activate ceridwen
pip install ceridwen
```

This installs JAX, the samplers, the filter curves and attenuation laws, posterior plotting
and the test runner. On Linux it installs the CUDA 12 JAX wheels, with the CUDA libraries
bundled: JAX uses an NVIDIA GPU when one is present and needs only a recent NVIDIA driver
(see [JAX's install page](https://docs.jax.dev/en/latest/installation.html) for the minimum).
On macOS, and on Linux without a GPU, JAX runs on the CPU with the same results. On Windows,
use WSL2. Every `fitSED` call prints the device it runs on:

```
ceridwen.fitSED
  Device      : GPU  (CudaDevice(id=0))
```

## 2. FSPS and `$SPS_HOME`

CERIDWEN reads the nebular CLOUDY grids (`nebular/`), the dust-emission templates
(`dust/dustem/`) and the FSPS emission-line list (`data/emlines_info.dat`) from the
[FSPS](https://github.com/cconroy20/fsps) data files at `$SPS_HOME`. FSPS itself is not run
during a fit, so this step is a clone (3.4 GB), not a compile:

```bash
export SPS_HOME="$HOME/fsps"
git clone --depth 1 https://github.com/cconroy20/fsps.git "$SPS_HOME"
```

`$SPS_HOME` can be any path. Make it permanent in your shell startup file, with the same path:

```bash
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc    # zsh, the macOS default
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc   # bash, most Linux
```

Open a new terminal and check that `echo $SPS_HOME` prints the path.

## 3. Check the installation

```bash
python -m ceridwen.check
```

It prints an `ok` / `warn` / `FAIL` line per component (dependencies, filters and attenuation
laws, float64, nested sampling, python-fsps, `$SPS_HOME` and the data files in it), with the
fix for anything missing. The `python-fsps` line is a warning until you compile it (below),
and that is expected.

## SSP grids

Fitting needs an SSP grid (an HDF5 file). The published grids are on Zenodo
([doi:10.5281/zenodo.22937956](https://doi.org/10.5281/zenodo.22937956)) and registered by
name. `fetch_grid` downloads a grid once into `~/.ceridwen/grids` (or `$CERIDWEN_GRID_DIR`),
checks its SHA-256 on every call, and returns the path:

```python
from ceridwen import SSPData
from ceridwen.ssps import fetch_grid, available_grids

print(available_grids(published_only=True))        # name -> description
ssp = SSPData.load(fetch_grid("mist_miles_chab"))
```

| name | contents | size |
|---|---|---|
| `mist_miles_chab` | MIST isochrones + MILES spectra, Chabrier IMF | 67 MB |
| `mist_bpass_v2` | BPASS v2 binary populations, Chabrier IMF | 62 MB |
| `amist_c3k_hr_krou_afe` | aMIST + C3K, Kroupa IMF, five [α/Fe] planes, for `CSPBasis_afe` ([α/Fe](afe.md)) | 612 MB |

The two solar-scaled grids carry the surviving-mass table that `fitSED` and `PostProcess` use
for `mfrac`; the α grid does not yet, so fit it with `mfrac=False`. A copy fetched before the
2026-09 re-deposit fails the checksum: `fetch_grid(name, force=True)` replaces it.

## Building your own SSP grid

To use a different IMF, isochrone set or spectral library, build a grid with FSPS. This is
the one step that needs the compiled [`python-fsps`](https://dfm.io/python-fsps) wrapper,
which compiles Fortran against `$SPS_HOME`:

```bash
brew install gcc                       # macOS (Homebrew); or:
sudo apt-get install gfortran          # Debian/Ubuntu; or:
conda install -c conda-forge gfortran  # any OS, inside your conda env

python -m pip install "fsps>=0.4.4"
```

```python
from ceridwen import SSPData

ssp = SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")
# later: ssp = SSPData.load("ssp_data.h5")
```

`from_fsps` accepts only the keyword arguments that define the stellar library and IMF
(`imf_type` and its parameters, isochrone-phase switches); dust, SFH, nebular emission, IGM,
redshift and a fixed metallicity are rejected, because the forward model applies them. The
file records its provenance (isochrones, spectral library, `imf_type`, FSPS version, build
arguments), `CSPBasis` reads the isochrone library from it, and `ssp.display()` prints it.

## Building this documentation

```bash
pip install ".[docs]"
mkdocs build
```
