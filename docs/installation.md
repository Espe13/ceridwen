# Installation

CERIDWEN needs Python 3.11 or newer (`jax` >= 0.9 and `blackjax` >= 1.6 require it).

## 1. The package

```bash
conda create -n ceridwen python=3.12 -y && conda activate ceridwen
pip install ceridwen
```

This installs JAX, the samplers, the filter curves and attenuation laws, posterior plotting
and the test runner, and runs on the CPU on any platform (a laptop is fine; importing prints
nothing).

**NVIDIA GPU** (Linux, or Windows through WSL2): install the CUDA build instead,

```bash
pip install "ceridwen[cuda12]"      # NVIDIA driver >= 525
pip install "ceridwen[cuda13]"      # NVIDIA driver >= 580
```

The CUDA libraries come as pip wheels; the driver is the only system requirement (minimums
from [JAX's install page](https://docs.jax.dev/en/latest/installation.html)). RTX 50-series
cards (Blackwell) need CUDA >= 12.8: `cuda13` always provides it, and a fresh `cuda12` install
resolves CUDA 12.9. Do not install a CUDA extra on a machine without an NVIDIA GPU: JAX's CUDA
plugin then prints a traceback at every import (it still falls back to the CPU).

**Apple Silicon GPU** (macOS 26, Python 3.12 or newer): the same `pip install ceridwen` also
installs [jax-graft](https://github.com/dfm/jax-graft), and JAX uses the Apple GPU in float32
(see [Apple GPU](mac_gpu.md): Apple GPUs have no float64, so `import ceridwen` switches the
precision for the process and says so; `JAX_PLATFORMS=cpu` keeps float64 on the CPU). Older
Macs and Python 3.11 get the CPU build.

Every `fitSED` call prints the device and precision it runs with:

```
ceridwen.fitSED
  Device      : GPU  (CudaDevice(id=0))
  Precision   : float64
```

**GPU memory.** The forward model's large arrays are float32 (see
[Conventions](conventions.md), "Precision"), so a consumer card with slow float64 does its
work in float32. A nested-sampling fit at the defaults (`num_live=500`, `num_delete=100`,
the `lanes` slice kernel) peaked at these device memories (A100, JAX 0.11.2; photometry plus a
600-pixel spectrum, measured with `peak_bytes_in_use`):

| model | parameters | peak device memory |
|---|---|---|
| stellar only | 7 | 2.75 GB |
| + nebular emission, IGM | 9 | 2.87 GB |
| + dust emission | 12 | 2.91 GB |

An 8 GB card runs these defaults. Memory grows linearly with the number of parameter vectors
evaluated at once: the compiled log-likelihood and gradient of the nebular model take 0.81 GB
for a batch of 100 and 4.3 GB for 500 (0.94 and 4.8 GB with dust emission).

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

## Updates

`fitSED` and `python -m ceridwen.check` tell you when PyPI has a newer CERIDWEN than the one
installed; upgrade with `pip install --upgrade ceridwen`. They ask PyPI at most once a day
(the answer is cached in `~/.ceridwen/update_check.json`), wait at most two seconds and say
nothing when there is no network. `CERIDWEN_NO_UPDATE_CHECK=1` switches the check off; it never
runs on CI (`$CI` set). What changed is in the
[changelog](https://github.com/Espe13/ceridwen/blob/main/CHANGELOG.md).

To get an email for each new release, open the
[GitHub page](https://github.com/Espe13/ceridwen), click **Watch**, then **Custom**, and tick
**Releases** (a free GitHub account is needed).

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
