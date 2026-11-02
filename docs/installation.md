# Installation

CERIDWEN requires Python 3.11 or newer, as do `jax >= 0.9` and `blackjax >= 1.6`.

## 1. The package

```bash
conda create -n ceridwen python=3.12 -y && conda activate ceridwen
pip install ceridwen
```

This installs the CPU version of JAX, the samplers, filter curves, attenuation
laws, plotting dependencies and test runner.

**NVIDIA GPU** (Linux, or Windows through WSL2): install the CUDA build:

```bash
pip install "ceridwen[cuda12]"      # NVIDIA driver >= 525
pip install "ceridwen[cuda13]"      # NVIDIA driver >= 580
```

The CUDA libraries are installed as pip wheels. You also need an NVIDIA driver;
see [JAX's installation guide](https://docs.jax.dev/en/latest/installation.html)
for the minimum version. RTX 50-series (Blackwell) cards require CUDA >= 12.8.
`cuda13` meets that requirement, and a fresh `cuda12` installation resolves to
CUDA 12.9. Install a CUDA extra only on a machine with an NVIDIA GPU. Otherwise,
the CUDA plugin prints a traceback on import before JAX falls back to the CPU.

**Apple Silicon GPU** (macOS 26, Python 3.12 or newer): `pip install ceridwen`
also installs [jax-graft](https://github.com/dfm/jax-graft), which makes the Apple
GPU the default device. Apple GPUs lack float64, so importing CERIDWEN switches
the process to float32 and prints a warning. Set `JAX_PLATFORMS=cpu` to use
float64 on the CPU. Older Macs and Python 3.11 use the CPU build. Python must
be a native arm64 build; an Intel conda installation under Rosetta will not
work. See [Apple GPU](mac_gpu.md) and [Troubleshooting](troubleshooting.md).

Every `fitSED` call prints the device and precision it runs with:

```
ceridwen.fitSED
  Device      : GPU  (CudaDevice(id=0))
  Precision   : float64
```

**GPU memory.** Large forward-model arrays use float32, including on consumer
GPUs with slow float64 performance. See [Precision](conventions.md#precision).
The following peak device-memory measurements used an A100 with JAX 0.11.2,
photometry and a 600-pixel spectrum. Nested-sampling settings were
`num_live=500`, `num_delete=100` and the `lanes` slice kernel. Memory was
measured with `peak_bytes_in_use`.

| model | parameters | peak device memory |
|---|---|---|
| stellar only | 7 | 2.75 GB |
| + nebular emission, IGM | 9 | 2.87 GB |
| + dust emission | 12 | 2.91 GB |

These configurations fit on an 8 GB card. Memory use grows approximately
linearly with batch size. The compiled nebular log-likelihood and gradient use
0.81 GB for 100 parameter vectors and 4.3 GB for 500; adding dust emission
increases those figures to 0.94 and 4.8 GB.

## 2. FSPS and `$SPS_HOME`

CERIDWEN reads the CLOUDY grids (`nebular/`), dust-emission templates
(`dust/dustem/`) and emission-line list (`data/emlines_info.dat`) from the
[FSPS](https://github.com/cconroy20/fsps) data tree at `$SPS_HOME`. A fit needs
these files but does not run FSPS, so cloning the 3.4 GB repository is sufficient;
you do not need to compile it.

```bash
export SPS_HOME="$HOME/fsps"
git clone --depth 1 https://github.com/cconroy20/fsps.git "$SPS_HOME"
```

Choose any location for `$SPS_HOME`. Add that path to your shell startup file:

```bash
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc    # zsh, the macOS default
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc   # bash, most Linux
```

Open a new terminal and check that `echo $SPS_HOME` prints the path.

## 3. Check the installation

```bash
python -m ceridwen.check
```

The check reports `ok`, `warn` or `FAIL` for each component and suggests fixes
for missing dependencies or data. It checks filters, attenuation laws, float64,
nested sampling, python-fsps and `$SPS_HOME`. The python-fsps warning is expected
unless you have compiled the wrapper, which is needed only to build a grid.

## Updates

`fitSED` and `python -m ceridwen.check` check PyPI for newer releases at most
once a day. Results are cached in `~/.ceridwen/update_check.json`. The check
times out after two seconds and produces no message when the network is
unavailable. Set `CERIDWEN_NO_UPDATE_CHECK=1` to disable it; it is also disabled
when `$CI` is set. Upgrade with `pip install --upgrade ceridwen` and see the
[changelog](https://github.com/Espe13/ceridwen/blob/main/CHANGELOG.md) for changes.

To get an email for each new release, open the
[GitHub page](https://github.com/Espe13/ceridwen), click **Watch**, then **Custom**, and tick
**Releases** (a free GitHub account is needed).

## SSP grids

A fit requires an SSP grid in HDF5 format. Published grids are registered by
name and hosted on Zenodo at
[doi:10.5281/zenodo.22937956](https://doi.org/10.5281/zenodo.22937956).
`fetch_grid` downloads a grid to `~/.ceridwen/grids` (or `$CERIDWEN_GRID_DIR`),
verifies its SHA-256 on each call and returns its path:

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

The two solar-scaled grids include the surviving-mass table used by `fitSED`
and `PostProcess` for `mfrac`. The α-enhanced grid has no such table; use
`mfrac=False`. Grids downloaded before the September 2026 re-deposit fail
the checksum. Replace them with `fetch_grid(name, force=True)`.

## Building your own SSP grid

To use another IMF, isochrone set or spectral library, build a grid with FSPS.
This requires the compiled [`python-fsps`](https://dfm.io/python-fsps) wrapper
and a Fortran compiler. Set `$SPS_HOME` before installing the wrapper:

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

`from_fsps` accepts arguments defining the stellar library and IMF, including
`imf_type`, its parameters and isochrone-phase switches. It rejects dust, SFH,
nebular, IGM, redshift and fixed-metallicity arguments because the forward
model applies those separately. The file records its isochrones, spectral
library, IMF, FSPS version and build arguments. `CSPBasis` reads the isochrone
library from this metadata; `ssp.display()` shows it.

## Building this documentation

```bash
pip install ".[docs]"
mkdocs build
```
