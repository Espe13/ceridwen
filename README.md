# Ceridwen

[![CI](https://github.com/Espe13/ceridwen/actions/workflows/ci.yml/badge.svg)](https://github.com/Espe13/ceridwen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Espe13/ceridwen/blob/main/LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://github.com/Espe13/ceridwen/blob/main/pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-amanda--stoffers.de%2Fceridwen-blue.svg)](https://www.amanda-stoffers.de/ceridwen/)

CERIDWEN is a Python package for Bayesian modelling of galaxy spectral energy distributions (SEDs). It fits photometry, spectra and emission-line fluxes, separately or together, to infer stellar populations, dust properties, nebular emission and star formation histories.

The forward model is written in JAX and runs on CPUs and GPUs. It is differentiable, allowing gradient-based sampling with Hamiltonian Monte Carlo. CERIDWEN also supports Bayesian evidence calculations.

The name stands for **C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven **WE**ight calculatio**N**s.

See the [documentation](https://www.amanda-stoffers.de/ceridwen/) to get started, or try the [Ingredient Explorer](https://www.amanda-stoffers.de/ceridwen-explorer/#explorer) to see how each model ingredient changes a galaxy spectrum in your browser, and to build your first script in a more intuitive way.

## Features

### Stellar populations and star formation

- Star formation histories defined on a lookback-time grid, using non-parametric continuity (`logsfr_ratios`) or a parametric form supplied as a transform.
- Constant metallicity or a metallicity history (`zh_const`).
- Sampling of stellar [α/Fe] with `CSPBasis_afe` and the aMIST + C3K grid.

### Dust, gas and IGM

- Dust attenuation with Kriek & Conroy diffuse dust, power-law birth clouds, and 12 registered laws, including Calzetti, SMC, LMC, Gordon+03 SMC bar and Reddy+15.
- Dust emission using Draine & Li (2007) or THEMIS templates.
- Nebular continuum and emission lines from the FSPS CLOUDY grids.
- Madau (1995) IGM attenuation, with an optional damping wing and damped Ly-α absorber (`MadauDampingDLA`).

### AGN

Enable AGN components with `CSPBasis(agn=AGNConfig())`:

- Nenkova+08 clumpy torus, Temple+21 accretion disc, an AGN dust screen and narrow emission lines.
- Modes that reproduce FSPS and Prospector, alongside a corrected default with exact `fagn * L_bol` normalisation, AGN emission applied before IGM attenuation, and energy balance with dust emission.
- CIGALE-style models with a SKIRTOR torus, polar dust, Feltre+16 narrow lines and a broad-line region, including a mode that reproduces CIGALE.

### Data and fitting

- Fit photometry, spectra and emission-line fluxes separately or together.
- Use the 296 bundled filter curves, or your own `.par` files with `Photometry(filter_directory=...)` or `$CERIDWEN_FILTER_DIR`.
- Combine stellar and gas velocity dispersions with the instrumental line-spread function in one broadening kernel, automatically accounting for the stellar library’s resolution.
- Analytically marginalise over emission-line fluxes with `Spectrum(marginalize_elines=True)`.
- Model noise and calibration uncertainties with jitter, calibration polynomials, an outlier mixture and a Gaussian-process likelihood for correlated residuals. Calibration polynomials can be sampled, profiled or marginalised.
- Use nested sampling to calculate the Bayesian evidence, sample with NUTS or VI-preconditioned NUTS, or find a maximum a posteriori (MAP) solution.

### Results and diagnostics

- Use `PostProcess` to calculate star formation rates, M_UV, ionising photon rates and posterior-predictive data, and produce summary, corner and diagnostic plots.

## Installation

Use **Python 3.12 or newer** for the installation below. Follow the instructions for your operating system, then download the FSPS data files and check your installation.

### Apple Silicon Mac

#### 1. Check your Mac and Conda installation

Open **Terminal** (Cmd + Space, type “Terminal”, press Enter), then run:

```bash
sw_vers -productVersion
conda info | grep platform
```

- **macOS:** Apple GPU support requires **26 or newer**. To update, open **System Settings → General → Software Update**. Older versions can use the CPU.
- **Conda:** `osx-arm64` means you have the correct build. If you see `osx-64` or `command not found: conda`, install Miniforge below.

The Conda platform describes your software, not your Mac’s hardware. To check the hardware, open **Apple menu → About This Mac**: **Chip: Apple M…** means Apple Silicon; **Processor: Intel…** means an Intel Mac. Intel Macs are not supported by the JAX version CERIDWEN requires.

<details>
<summary>Install Miniforge for Apple Silicon</summary>

Run these commands in Terminal:

```bash
curl -L -o Miniforge3-MacOSX-arm64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Follow the prompts: accept the licence and default installation location, and answer **yes** when asked to initialise Conda. Quit Terminal with **Cmd + Q**, reopen it, and check:

```bash
conda info | grep platform
```

It should show `osx-arm64`. If it still shows `osx-64`, run the following, then quit and reopen Terminal again:

```bash
~/miniforge3/bin/conda init zsh
```

</details>

#### 2. Create an environment and install CERIDWEN

These commands create an environment with Python 3.12:

```bash
conda create -n ceridwen python=3.12 -y
conda activate ceridwen
pip install ceridwen
```

On an Apple Silicon Mac with **macOS 26 or newer** and native **arm64 Python 3.12 or newer**, this also installs [jax-graft](https://github.com/dfm/jax-graft) to enable Apple GPU support. Older macOS versions use the CPU.

No additional GPU configuration is needed. When you import CERIDWEN, it detects the GPU, switches to **float32 precision**, and prints a message once. When you run `fitSED`, you should see:

```text
Device : MTL
Precision : float32
```

Apple GPUs do not support float64. Evidence values, gradients, and samples are therefore computed in float32 and will not match a float64 fit bit for bit. See the [Apple GPU guide](https://www.amanda-stoffers.de/ceridwen/mac_gpu/) (`docs/mac_gpu.md`) for the guarantees and limitations of this mode.

To run a float64 fit on the CPU instead, set this variable **before starting Python or Jupyter**:

```bash
export JAX_PLATFORMS=cpu
```

Continue with **Download the FSPS data files** below.

### Linux

#### 1. Create an environment

With Conda installed, open a terminal and run:

```bash
conda create -n ceridwen python=3.12 -y
conda activate ceridwen
```

#### 2. Install CERIDWEN

Choose **one** installation option.

**CPU:**

```bash
pip install ceridwen
```

**NVIDIA GPU with CUDA 12** — requires NVIDIA driver **525 or newer**:

```bash
pip install "ceridwen[cuda12]"
```

**NVIDIA GPU with CUDA 13** — requires NVIDIA driver **580 or newer**:

```bash
pip install "ceridwen[cuda13]"
```

To check your NVIDIA driver version, run:

```bash
nvidia-smi
```

The CUDA libraries are installed automatically through pip; you still need a compatible NVIDIA GPU and driver. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for compatibility details. These NVIDIA installation options also apply to Linux running under **WSL2 on Windows**.

### Download the FSPS data files — all users

CERIDWEN reads its nebular CLOUDY grids and dust-emission templates from the FSPS data files. Download FSPS and set `SPS_HOME` to its location. Allow approximately **3.4 GB** for the download.

```bash
export SPS_HOME="$HOME/fsps"
git clone --depth 1 https://github.com/cconroy20/fsps.git "$SPS_HOME"
```

To save `SPS_HOME` for future terminal sessions, run **one** of the following commands.

For **zsh**, the default shell on macOS:

```bash
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc
```

For **bash**, commonly used on Linux:

```bash
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc
```

You only need to compile the `python-fsps` wrapper if you want to build your own SSP grid. To fit data, the published grids and the FSPS data files are sufficient. See the [full installation guide](https://www.amanda-stoffers.de/ceridwen/installation/) for details.

### Check the installation — all users

With the `ceridwen` environment active, run:

```bash
python -m ceridwen.check
```

This prints one line per component and explains how to fix anything missing. A warning about `python-fsps` is expected if you have not compiled the wrapper; you can ignore it when using the published SSP grids.

### Try an example

The notebooks in `examples/` walk through fitting mock data:

- `apple_gpu_mock_fit.ipynb` creates a mock galaxy, fits it on the Apple GPU, and shows the recovered parameters.
- The `mock_fit_*.ipynb` notebooks cover photometry alone, photometry with line fluxes, a spectrum alone, and a spectrum calibrated against photometry.

### Stay up to date

`fitSED` and `ceridwen.check` notify you when a newer version is available on PyPI. To disable these checks, set:

```bash
export CERIDWEN_NO_UPDATE_CHECK=1
```

To receive release notifications through GitHub, select **Watch → Custom → Releases** at the top of this repository. Email delivery depends on your GitHub notification settings.

## Verification

The test suite checks units and conventions, the broadening against direct quadrature, the
likelihoods, the samplers, post-processing and the misuse guards. Two layers compare the
forward model with stored arrays: 22 regression categories (plus optional ones that need a grid or the AGN tables) at
`rtol=2e-6` (AGN lines 2e-5) with an absolute tolerance of 1e-10 times each array's maximum
(`tests/regression/test_regression.py`), and 13 golden configurations (SFH and
metallicity, AGN, and a fixed nebular block; `tests/csp/test_lookback_flip_invariant.py`) with
SSP weights at `rtol=1e-12` and spectra, line fluxes and photometry at `rtol=1e-6`. See
[For developers](#for-developers) to run them.

## Documentation

- [Tutorial](https://www.amanda-stoffers.de/ceridwen/tutorial/): photometry, a spectrum and emission lines fitted jointly
- [Samplers](https://www.amanda-stoffers.de/ceridwen/samplers/): nested sampling, NUTS, VI-preconditioned NUTS, the MAP, resuming a run
- [Post-processing](https://www.amanda-stoffers.de/ceridwen/postprocessing/): derived quantities, predictions, figures
- [Noise and calibration](https://www.amanda-stoffers.de/ceridwen/noise_calibration/): jitter, error floors, calibration polynomials
- [GP likelihood](https://www.amanda-stoffers.de/ceridwen/gp_likelihood/): correlated spectral residuals
- [Outlier model](https://www.amanda-stoffers.de/ceridwen/outlier_model/): a mixture likelihood for bad data points
- [Emission-line marginalisation](https://www.amanda-stoffers.de/ceridwen/eline_marginalisation/)
- [α/Fe](https://www.amanda-stoffers.de/ceridwen/afe/): the α-enhanced grid and `CSPBasis_afe`
- [AGN](https://www.amanda-stoffers.de/ceridwen/agn/): torus, disc, AGN screen and lines; the modes and their parity with FSPS, Prospector and CIGALE (SKIRTOR, polar dust, Feltre+16 NLR, BLR)
- [Conventions](https://www.amanda-stoffers.de/ceridwen/conventions/): units, metallicity, lookback time, broadening, cosmology
- [Troubleshooting](https://www.amanda-stoffers.de/ceridwen/troubleshooting/)
- [Development and verification](https://www.amanda-stoffers.de/ceridwen/development/): every check, and how the code is built
- [API reference](https://www.amanda-stoffers.de/ceridwen/api/)
- [`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md): the misuse guide; [`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md): conventions for AI assistants

## For developers

Clone the repository and install CERIDWEN in editable mode so that changes to the code take effect without reinstalling:

```bash
git clone --depth 1 https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install -e .
```

`--depth 1` downloads only the latest version, reducing the download from 244 MB to 41 MB. For an NVIDIA GPU, replace the last command with `pip install -e ".[cuda12]"`.

### Running the tests

First, download the published BPASS grid and tell the tests where to find it:

```bash
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
```

Run the same tests as CI:

```bash
pytest -m "not fsps and not gpu" -q -ra
```

As a guide, this took about 33 minutes on an 11-core laptop CPU while other jobs were running.

To also compare the forward model against the stored reference results, run the following with `SPS_HOME` set:

```bash
pytest tests/regression/test_regression.py tests/csp/test_lookback_flip_invariant.py -q
```

Check the skip summary: tests that need a grid are skipped if the grid is missing. See the [development guide](https://www.amanda-stoffers.de/ceridwen/development/) for the full list of checks and details of the code.

Run scripts from inside the repository or another working directory. Avoid running them from the directory immediately above the clone, where Python may find the `ceridwen` folder instead of the installed package.
## Citing

If you use CERIDWEN in your research, please cite it:

```bibtex
@misc{stoffers2026ceridwen,
  author       = {Stoffers, Amanda},
  title        = {{CERIDWEN}: Fast and Flexible {GPU}-Accelerated Stellar Population Inference},
  year         = {2026},
  note         = {Version 1.0.15},
  howpublished = {\url{https://github.com/Espe13/ceridwen}}
}
```

## License

MIT; see [LICENSE](https://github.com/Espe13/ceridwen/blob/main/LICENSE).
