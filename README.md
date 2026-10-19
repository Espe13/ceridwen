# Ceridwen

[![CI](https://github.com/Espe13/ceridwen/actions/workflows/ci.yml/badge.svg)](https://github.com/Espe13/ceridwen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Espe13/ceridwen/blob/main/LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://github.com/Espe13/ceridwen/blob/main/pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-amanda--stoffers.de%2Fceridwen-blue.svg)](https://www.amanda-stoffers.de/ceridwen/)

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven **WE**ight
calculatio**N**s. CERIDWEN fits the spectral energy distributions of galaxies (photometry,
spectra and emission-line fluxes, alone or jointly) to infer their stellar populations, dust,
nebular emission and star formation histories. The forward model is written in JAX, so the
same code runs on a laptop CPU or a GPU, and is differentiable for gradient-based samplers.
It is for astronomers who want a Bayesian SED fit with the evidence or with Hamiltonian Monte
Carlo. Documentation: [www.amanda-stoffers.de/ceridwen](https://www.amanda-stoffers.de/ceridwen/).
The [Ingredient Explorer](https://www.amanda-stoffers.de/ceridwen-explorer/#explorer) shows,
in the browser, how each model ingredient changes a spectrum.

## Features

- Star formation history on a lookback-time grid: non-parametric continuity (`logsfr_ratios`) or any parametric form written as a transform
- Constant metallicity or a metallicity history (`zh_const`)
- [α/Fe] as a sampled stellar axis (`CSPBasis_afe`, aMIST + C3K grid)
- Dust attenuation: Kriek & Conroy diffuse dust, power-law birth clouds, 12 registered laws (Calzetti, SMC, LMC, Gordon+03 SMC bar, Reddy+15, ...)
- Dust emission from Draine & Li (2007) or THEMIS templates
- Nebular continuum and emission lines from the FSPS CLOUDY grids
- IGM attenuation (Madau 1995), with an optional damping wing and damped Ly-α absorber (`MadauDampingDLA`)
- AGN (`CSPBasis(agn=AGNConfig())`): Nenkova+08 clumpy torus, Temple+21 accretion disc, an AGN dust screen and AGN narrow lines, everything Prospector has; modes that reproduce FSPS and Prospector exactly, and a corrected default (exact `fagn * L_bol` normalisation, AGN before the IGM, energy balance with dust emission); and the CIGALE standard: SKIRTOR torus + polar dust, Feltre+16 narrow lines and a broad-line region, with a mode that reproduces CIGALE
- Photometry, spectra and emission-line fluxes, fitted alone or jointly
- One broadening kernel: stellar and gas velocity dispersions, instrument line-spread function, library resolution removed automatically
- Analytic marginalisation over emission-line fluxes (`Spectrum(marginalize_elines=True)`)
- Noise and calibration terms: jitter, calibration polynomials (sampled, profiled or marginalised), an outlier mixture, a Gaussian-process likelihood for correlated residuals
- Samplers: nested sampling (with the evidence), NUTS, VI-preconditioned NUTS, and a MAP optimiser
- Post-processing (`PostProcess`): SFRs, M_UV, ionising photon rates, posterior-predictive data, and summary, corner and diagnostic figures

## Installation

CERIDWEN needs Python 3.11 or newer. Create an environment and install the package:

```bash
conda create -n ceridwen python=3.12 -y && conda activate ceridwen
pip install ceridwen
```

This runs on the CPU, on any platform. For an NVIDIA GPU (Linux, or WSL2 on Windows), install
the CUDA build instead: `pip install "ceridwen[cuda12]"` (NVIDIA driver >= 525) or
`pip install "ceridwen[cuda13]"` (driver >= 580); the CUDA libraries come as pip wheels, see
[JAX's install page](https://docs.jax.dev/en/latest/installation.html).

### Apple Silicon GPU (float32)

On an Apple Silicon Mac with macOS 26 and Python 3.12 or newer, the same `pip install ceridwen`
also installs [jax-graft](https://github.com/dfm/jax-graft), which makes the Apple GPU JAX's
default backend. Nothing to configure: `import ceridwen` sees the GPU, switches to float32
(Apple GPUs have no float64) and says so once; `fitSED` prints `Device : MTL` and
`Precision : float32`. Evidences, gradients and samples are then float32, so they are not
comparable bit for bit with a float64 fit; for a float64 fit on the CPU set
`JAX_PLATFORMS=cpu`. Older macOS versions and Python 3.11 get the CPU build. The Python must
be native: `python -c "import platform; print(platform.machine())"` prints `arm64`. An Intel
(x86_64) conda, such as an old Anaconda install, gives an x86 Python, for which there is no
jax >= 0.9, and pip stops with "No matching distribution found for jaxlib>=0.9.0"; install
[Miniforge](https://github.com/conda-forge/miniforge) for arm64 instead. Intel Macs are not
supported for the same reason. What float32
mode guarantees, and what it refuses, is in
[Apple GPU](https://www.amanda-stoffers.de/ceridwen/mac_gpu/) (`docs/mac_gpu.md`).
`examples/apple_gpu_mock_fit.ipynb` builds a mock galaxy, fits it on the GPU and shows the
recovered parameters; `examples/mock_fit_*.ipynb` do the same for photometry alone, photometry
with line fluxes, a spectrum alone, and a spectrum calibrated against photometry.

CERIDWEN reads the nebular CLOUDY grids and the dust-emission templates from the FSPS data
files. Clone FSPS into `$SPS_HOME` (3.4 GB) and make the variable permanent:

```bash
export SPS_HOME="$HOME/fsps"
git clone --depth 1 https://github.com/cconroy20/fsps.git "$SPS_HOME"
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc    # zsh, the macOS default
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc   # bash, most Linux
```

Compiling the `python-fsps` wrapper is needed only to build your own SSP grid; the published
grids and the FSPS data files are enough to fit (see
[Installation](https://www.amanda-stoffers.de/ceridwen/installation/)).

Check the installation:

```bash
python -m ceridwen.check
```

It prints one line per component, with the fix for anything missing. The `python-fsps` line
is a warning until you compile it, and that is expected.

`fitSED` and `ceridwen.check` say when PyPI has a newer release (`CERIDWEN_NO_UPDATE_CHECK=1`
switches this off). For an email at each release, click **Watch** at the top of this page,
then **Custom**, and tick **Releases**.

## Quick start

This fits a mock galaxy (twelve photometric bands from GALEX to WISE and an optical spectrum)
made with the same forward model, so you can compare the fit with the truth. Run the blocks
in order in one Python session.

**1. Get an SSP grid.** The published MIST + MILES grid (Chabrier IMF) is downloaded once
(67 MB) into `~/.ceridwen/grids` (or `$CERIDWEN_GRID_DIR`), and its SHA-256 is checked on
every call. `available_grids()` lists the others. All published grids are archived on
Zenodo ([doi:10.5281/zenodo.22937956](https://doi.org/10.5281/zenodo.22937956)); this README
and `fetch_grid` always point at the current record.

```python
from ceridwen import SSPData
from ceridwen.ssps import fetch_grid

ssp = SSPData.load(fetch_grid("mist_miles_chab"))
ssp.display()                                      # library, IMF, grid coverage
```

**2. Build the model.** `lookback_time` holds the SFH nodes in Gyr, increasing from 0
(today). The SFH is sampled as `logsfr_ratios` and turned into a star formation rate per node
by a transform. `logzsol` is log10(Z/Z_sun) of the grid; on the MIST grids used here this is
[Fe/H], and the figures label it that way.

```python
import jax, jax.numpy as jnp
import numpy as np
from ceridwen import CSPBasis, SedModel, Instrument, Cosmology
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

ZRED = 0.1                                         # fixed spectroscopic redshift
FILTERS = ["galex_FUV", "galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0",
           "sdss_i0", "sdss_z0", "twomass_J", "twomass_H", "twomass_Ks",
           "wise_w1", "wise_w2"]
SPEC_WAVE = np.linspace(4000.0, 8000.0, 600)       # observed-frame vacuum Angstrom
LSF = Instrument.sigma_kms(150.0)                  # instrumental line-spread function

csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
               cosmo=Cosmology.planck18(),
               zh_const=True, sfh_interp="step",
               add_dust=False, add_diffuse_dust=True, add_neb=False, verbose=False)

sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=sfh_times_yr)

def build_model(observations):
    return SedModel(
        csp, observations=observations,
        priors={
            "logzsol": Uniform(low=-2.0, high=0.2),
            "logmass": Uniform(low=9.0, high=12.0),
            "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
            "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
            "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
        },
        transforms={"sfh": logsfr_to_sfh},
        free_param_init={"logsfr_ratios": jnp.zeros(5), "logmass": jnp.array([10.0])},
        zred=ZRED,
    )
```

`SedModel` also applies a galaxy velocity dispersion of 300 km/s by default. Pass
`kinematics=Kinematics(sigma_gal=...)` to change it, or `Kinematics(sigma_gal="sigma_gal")`
with a prior to fit it (see [Conventions](https://www.amanda-stoffers.de/ceridwen/conventions/)).

**3. Make the mock data.** Push known parameters through the model and add noise
(signal-to-noise 20 in the photometry, 25 in the spectrum).

```python
TRUTH = {
    "logsfr_ratios":      jnp.array([0.3, 0.2, -0.1, -0.4, -0.6]),
    "logzsol":            jnp.array([-0.2]),
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}
gen = build_model([Photometry(filters=FILTERS, name="phot"),
                   Spectrum(wavelength=SPEC_WAVE, instrument=LSF, name="spec")])
truth_pred = gen.predict(TRUTH)                    # AB maggies (phot), F_nu in cgs (spec)

rng = np.random.default_rng(42)
mag = np.asarray(truth_pred["phot"]); mag_unc = mag / 20.0
sfx = np.asarray(truth_pred["spec"]); sfx_unc = np.abs(sfx) / 25.0
phot = Photometry(filters=FILTERS, name="phot", uncertainty=mag_unc,
                  flux=mag + mag_unc * rng.standard_normal(mag.shape))
spec = Spectrum(wavelength=SPEC_WAVE, instrument=LSF, name="spec", uncertainty=sfx_unc,
                flux=sfx + sfx_unc * rng.standard_normal(sfx.shape))
model = build_model([phot, spec])
```

Look at the model before fitting it. `model.display()` draws it: each free parameter with its
prior, the transform or stage of the forward model it enters, the observations, and the range
of photometry and spectra the priors allow against the data. `model.latex()` returns a LaTeX
table of every free, derived and fixed parameter with its prior or value, ready for a paper
(it needs `\usepackage{booktabs,amsmath}`).

```python
fig, ax = model.display(return_fig=True)     # in a notebook, model.display() alone shows it
fig.savefig("model.png", dpi=150)
print(model.latex())
```

In a script, a bare `model.display()` opens a window and waits until you close it.

**4. Fit with nested sampling.** These settings are sized for a laptop. On an Apple Silicon
Mac (macOS 26, Python 3.12) the fit runs on the GPU in float32 by default and takes
**around 5 min**; on the CPU in float64 (`JAX_PLATFORMS=cpu`, or any other laptop) it takes a
little longer. It uses 60 inner slice steps: with 30 (the `fitSED` default) the GPU's float32
fit returns a logmass posterior about 30 % too narrow. It
prints its progress and writes `./my_fit/ceridwen_result.h5`. For production, raise `num_live`
(fitSED's default is 500); for NUTS, see [Samplers](https://www.amanda-stoffers.de/ceridwen/samplers/).

```python
from ceridwen import fitSED

result = fitSED(
    model,
    sampler="nested",
    sampler_kwargs={"num_live": 100, "num_delete": 20, "num_inner_steps": 60,
                    "logZ_tol": -5.0},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
print(f"log Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")
```

**5. Inspect and post-process.** `PostProcess` resamples the draws to equal weight, pushes
them through the fitted model, and computes derived quantities and predictions.

```python
from ceridwen import PostProcess

pp  = PostProcess(model, result, n_samples=1000)
out = pp.run()

truth = {p: float(TRUTH[p][0]) for p in ("logzsol", "logmass", "diffuse_tau_kc", "diffuse_dust_index")}
for p, t in truth.items():
    lo, med, hi = np.percentile(out["theta"][p], [16, 50, 84])
    print(f"{p:>20}: true {t:+7.3f}   fit {med:+7.3f}  (-{med - lo:.3f}/+{hi - med:.3f})")

print(np.percentile(out["extras"]["sfh"]["sfr100"], [16, 50, 84]))   # SFR over 100 Myr [Msun/yr]
pp.figures("./my_fit/figures", title="mock galaxy",
           truths={**truth, "logsfr_ratios": np.asarray(TRUTH["logsfr_ratios"])})
pp.save("./my_fit/post.npz")
```

The table compares the posterior median and 16-84% range with the truth. At these settings
the recovered values can sit 2-3 sigma from the truth: with a five-ratio SFH, `logmass`
trades against the SFH shape. The truth's oldest bins hold extra mass that barely shows in the
light, and a flat SFH at logmass 10.44 fits the data as well (within 1 in ln L, even without
noise). The SFH prior favours the flat history, so the MAP of this mock lies at 10.44, not at
10.50: the prior and the degeneracy, not the sampler or the noise (on noise-free data with
flat priors the MAP is 10.498). With 100 live points the quoted ranges are too
narrow. On the Apple GPU (float32) this seed gives logmass 10.43 (-0.03/+0.03) and seed 0
gives 10.44 (-0.03/+0.03); on the CPU (float64) this seed gives 10.44 (-0.02/+0.03). Raise
`num_live` before trusting an error bar. The five SFH ratios are mostly set by their prior:
one or two of them can lie outside their 95 % range. `figures` writes `summary.pdf` (SED with
residuals, SFH, marginals), `corner.pdf` and `diagnostics.pdf` (the sampler's diagnostics).
`post.npz` holds every draw's predictions and is about 100 MB; `load_postprocess` reads it back.
Everything `PostProcess` returns is described in
[Post-processing](https://www.amanda-stoffers.de/ceridwen/postprocessing/).

## Verification

The test suite checks units and conventions, the broadening against direct quadrature, the
likelihoods, the samplers, post-processing and the misuse guards. Two layers compare the
forward model with stored arrays: 22 regression categories (plus optional ones that need a grid or the AGN tables) at
`rtol=1e-7` with an absolute tolerance of 1e-10 times each array's maximum
(`tests/regression/test_regression.py`), and six SFH and
metallicity configurations (`tests/csp/test_lookback_flip_invariant.py`) with SSP weights at
`rtol=1e-12` and spectra, line fluxes and photometry at `rtol=1e-6`. See
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

Install from a clone (`--depth 1` skips the history: 41 MB instead of 244 MB):

```bash
git clone --depth 1 https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install -e .                 # or pip install -e ".[cuda12]" for an NVIDIA GPU
```

The test suite needs the published BPASS grid. The first command below runs what CI runs; it
took 33 min on an 11-core laptop CPU shared with other jobs. The second compares the forward
model with the stored baselines and needs `$SPS_HOME`.

```bash
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
pytest -m "not fsps and not gpu" -q -ra
pytest tests/regression/test_regression.py tests/csp/test_lookback_flip_invariant.py -q
```

A grid-dependent test skips when it finds no grid, so read the skip summary: a skip is not a
pass. [Development](https://www.amanda-stoffers.de/ceridwen/development/) lists every check and
how the code is built. Do not run scripts from the folder that contains the clone: Python then finds the clone
folder `ceridwen` before the installed package.

## Citing

If you use CERIDWEN in your research, please cite it:

```bibtex
@misc{stoffers2026ceridwen,
  author       = {Stoffers, Amanda},
  title        = {{CERIDWEN}: Fast and Flexible {GPU}-Accelerated Stellar Population Inference},
  year         = {2026},
  note         = {Version 1.0.13},
  howpublished = {\url{https://github.com/Espe13/ceridwen}}
}
```

## License

MIT; see [LICENSE](https://github.com/Espe13/ceridwen/blob/main/LICENSE).
