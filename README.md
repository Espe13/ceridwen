

# Ceridwen

[![CI](https://github.com/Espe13/ceridwen/actions/workflows/ci.yml/badge.svg)](https://github.com/Espe13/ceridwen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Espe13/ceridwen/blob/main/LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://github.com/Espe13/ceridwen/blob/main/pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-amanda--stoffers.de%2Fceridwen-blue.svg)](https://www.amanda-stoffers.de/ceridwen/)

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven **WE**ight calculatio**N**s: a JAX-native, GPU-capable Bayesian spectral energy distribution (SED) fitting package with nested sampling, variational-inference preconditioned Hamiltonian Monte Carlo and native redshift support.

Documentation: [www.amanda-stoffers.de/ceridwen](https://www.amanda-stoffers.de/ceridwen/)

**Try it in four commands** (Python 3.11+, no clone and no FSPS needed for
this path):

```bash
pip install ceridwen             # GPU wheels on Linux, CPU elsewhere
python -m ceridwen.check         # environment self-check, each problem with its fix
curl -LO https://raw.githubusercontent.com/Espe13/ceridwen/v1.0.11/examples/quickstart.py
SSP_FILE=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_miles_chab'))") \
    python quickstart.py         # demo mock fit (not converged); recovered-vs-true table
```

The last line downloads the published MIST + MILES grid once (67 MB,
SHA-256 verified, cached in `~/.ceridwen/grids`) and fits mock photometry with
it. That path never touches FSPS. **Fitting real galaxies with nebular
emission does need the FSPS data files** — a `git clone`, no compiler; see
[Installation](#installation). The [quick start](#quick-start) and
[how the code is verified](#verification) follow.

---

## Features

- [x] Star formation history (non-parametric continuity + parametric)
- [x] Metallicity history
- [x] Dust attenuation (Kriek & Conroy diffuse, power-law birth-cloud, multi-component age-dependent; 12 registered laws incl. Calzetti, Noll, Pei SMC/LMC, Gordon+03 SMC bar `gordon03_smcbar`, Reddy+15 `reddy15`)
- [x] Dust emission (Draine & Li 2007 grids, or THEMIS with `CSPBasis(duste_model="THEMIS")`)
- [x] Nebular continuum + emission lines (CLOUDY grids)
- [x] Analytic marginalisation over emission-line fluxes, jointly across spectrum, photometry and line fluxes (`Spectrum(marginalize_elines=True)`, [docs](https://github.com/Espe13/ceridwen/blob/main/docs/eline_marginalisation.md))
- [x] Observation input (broadband photometry, emission-line fluxes, spectra)
- [x] Correlated spectral residuals: a squared-exponential Gaussian-process likelihood per spectrum, its amplitude and length scale fixed or sampled (`log_gp_amp_spec`, `log_gp_length_spec`; [docs](https://github.com/Espe13/ceridwen/blob/main/docs/gp_likelihood.md))
- [x] Spectrophotometric calibration polynomial: sampled (`spectrum_calib`), profiled (`Spectrum(polynomial_order=M)`), or marginalised analytically under a Gaussian prior (`polynomial_mode="marginalize"`, no sampled dimension)
- [x] Redshift-aware forward model with cosmological flux normalisation
- [x] IGM attenuation (Madau 1995), optionally with an IGM damping wing (`x_HI`) and a damped Ly-α absorber (`logN_HI`, `z_dla`) as fixed or sampled parameters (`MadauDampingDLA`); extensible via `IGMModel` ABC
- [x] NUTS / nested sampling / variational-inference preconditioned NUTS
- [x] Post-processing (`PostProcess`): SFR averages, M_UV, Q(H), xi_ion, posterior-predictive photometry/spectra/lines, intrinsic and dust-free spectra, best fit, user-defined quantities
- [x] Per-galaxy figures (`ceridwen.plotting`): summary, corner, sampling diagnostics
- [x] α-enhanced SSPs (`CSPBasis_afe`, FSPS v4.0 aMIST + C3K, [α/Fe] sampled as a free parameter)

Everything is written against `jax.numpy` with `@jit` and `vmap`/`pmap` in mind: the forward model is a single XLA graph, the sampler runs on GPU, and the sampling hot path contains zero Python branches.

---

## Installation

**Requires Python 3.11 or newer** (`blackjax` >= 1.6 and `jax` >= 0.9 insist
on it). There are three steps, and only the first is always needed.

### 1. The package

```bash
conda create -n ceridwen python=3.11 -y
conda activate ceridwen

pip install ceridwen
```

That is the whole Python install: JAX, the samplers, the vendored filter curves
and attenuation laws, posterior plotting and the test runner. There are no
extras to choose.

**GPU is the default.** On Linux this installs the CUDA 12 JAX wheels (the
CUDA libraries are bundled; the only system requirement is a recent enough
NVIDIA driver -- see [jax's install page](https://docs.jax.dev/en/latest/installation.html)
for the current minimum -- no toolkit install), and JAX uses the GPU automatically. No flags,
no separate install. A machine without a usable NVIDIA GPU gets the same
install and falls back to CPU at import time: one warning, identical
results, more patience. macOS and native Windows have no CUDA wheels and
get the CPU build automatically; on a Windows machine with an NVIDIA GPU,
install inside WSL2. Every `fitSED` call prints which backend it landed on:

```
ceridwen.fitSED
  Device      : GPU  (CudaDevice(id=0))
```

### 2. The FSPS data files — needed for nebular and dust emission

CERIDWEN reads the CLOUDY nebular grids and the Draine & Li / THEMIS
dust-emission templates **directly out of an FSPS data directory** whenever
`add_neb=True` or `add_dust_emission=True`, which is to say in essentially
every fit to a real galaxy. FSPS itself is never run at fit time — it only
supplies the tables — so this step is a download, **not a compile**:

```bash
export SPS_HOME="$HOME/fsps"        # <- any path you like: $HOME, a data disk, scratch
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"
```

`git clone` writes to that absolute path, so it does not matter which directory
you run it from. Make the variable permanent (see
[Installing FSPS](#installing-fsps-and-setting-sps_home) below), then
`python -m ceridwen.check` will report `$SPS_HOME` as OK.

### 3. An SSP grid

The forward model consumes an HDF5 cache of SSP spectra. Download a published
one — one line, no FSPS:

```python
from ceridwen import SSPData
from ceridwen.ssps import fetch_grid
ssp = SSPData.load(fetch_grid("mist_miles_chab"))   # 67 MB, cached, SHA-256 verified
```

Building your own (a different IMF, isochrones or spectral library) is the only
thing that needs the compiled `python-fsps` wrapper and therefore a Fortran
compiler; see [Step 0](#step-0-the-ssp-grid) and
[Installing FSPS](#installing-fsps-and-setting-sps_home).

### From a clone

To run the bundled examples or the test suite, or to develop:

```bash
git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install -e .                 # or `pip install .`
```

**Check the whole setup at any time** with `python -m ceridwen.check`: it
reports each dependency, whether `$SPS_HOME` is set and holds its `nebular/`
tables, whether float64 is on, and whether nested sampling is available — each
problem with its fix.

### Installing FSPS and setting `$SPS_HOME`

CERIDWEN uses [FSPS](https://github.com/cconroy20/fsps) in two separate ways,
and they have very different costs.

**The data files (step 2 above) — needed for nebular and dust emission.** The
FSPS repository ships the CLOUDY nebular lookup tables (`nebular/`) and the
Draine & Li dust-emission spectra (`dust/`) as data. When `add_neb=True` or
`add_dust_emission=True`, **CERIDWEN reads those files directly from
`$SPS_HOME`** — FSPS is never executed at a fit. So this is a `git clone` and
an environment variable. **No Fortran compiler, no `pip install fsps`.**

```bash
export SPS_HOME="$HOME/fsps"        # <- any path: $HOME, a data disk, scratch
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"
```

**The `python-fsps` wrapper — needed only to build your own SSP grid.**
[`python-fsps`](https://dfm.io/python-fsps) compiles Fortran against
`$SPS_HOME`, so it cannot be a pip dependency of CERIDWEN. You need it only for
`SSPData.from_fsps(...)`, i.e. when the [published grids](#step-0-the-ssp-grid)
do not have the IMF, isochrones or spectral library you want:

```bash
# A Fortran compiler (pick one for your system):
brew install gcc                       # macOS (Homebrew)
sudo apt-get install gfortran          # Debian/Ubuntu
conda install -c conda-forge gfortran  # any OS, inside your conda env

python -m pip install "fsps>=0.4.4"    # compiles against $SPS_HOME
```

**Make `$SPS_HOME` permanent.** The `export` above only sets it for the current
terminal, and CERIDWEN needs it in *every* session that uses nebular or dust
emission. Add it to your shell startup file so it persists (use the **same path
you chose above**; the `$HOME/fsps` below is just the example default):

```bash
# zsh (the macOS default shell):
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc
source ~/.zshrc

# bash (most Linux):
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc
source ~/.bashrc
```

Open a **new** terminal and run `echo $SPS_HOME`; it should print the path. If
it's blank, the line went into the wrong file (check which shell you use with
`echo $SHELL`). Then confirm the whole setup with:

```bash
python -m ceridwen.check
```

It prints one line per dependency and flags an unset `$SPS_HOME`, or one whose
`nebular/` subdirectory is missing. Next stop:
[`examples/quickstart.py`](https://github.com/Espe13/ceridwen/blob/main/examples/quickstart.py), a complete runnable fit.

---

## Quick start

### Step 0: the SSP grid

CERIDWEN's forward model consumes an HDF5 cache of SSP spectra precomputed with
FSPS. You have two ways to get one.

**Download a published grid (no FSPS).** These are the grids CERIDWEN is tested
and released with, on Zenodo
([doi:10.5281/zenodo.22921057](https://doi.org/10.5281/zenodo.22921057)) and
registered by name. The file is downloaded once into `~/.ceridwen/grids` (or
`$CERIDWEN_GRID_DIR`) and its SHA-256 is checked on every call:

```python
from ceridwen import SSPData
from ceridwen.ssps import fetch_grid, available_grids

print(available_grids(published_only=True))          # names and what each grid is
ssp = SSPData.load(fetch_grid("mist_miles_chab"))    # MIST + MILES, Chabrier, 67 MB
ssp.display()                                        # library / IMF / grid coverage
```

**Or build your own**, so you control the isochrones, spectral library and IMF.
This is the one step that needs the compiled `python-fsps` and `$SPS_HOME`. It
takes a few minutes on CPU (about one coffee) and then serves every fit that
shares those choices.

This block is safe to rerun: it loads the cached grid if one exists at
`SSP_FILE` and only builds when it doesn't:

```python
import pathlib
from ceridwen import SSPData

SSP_FILE = pathlib.Path("ssp_data.h5")

if SSP_FILE.is_file():
    ssp = SSPData.load(str(SSP_FILE))
else:
    # from_fsps accepts ONLY the kwargs that define the stellar
    # library / IMF (imf_type and its parameters, isochrone-phase knobs like
    # tpagb_norm_type). Anything the forward model applies itself (dust, SFH,
    # nebular emission, IGM, redshift, or a fixed metallicity) is rejected.
    ssp = SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE))

ssp.display()    # confirm library / IMF / grid coverage before fitting
```

If you change the FSPS configuration (a different IMF, say), point `SSP_FILE`
at a new filename: the cache is keyed by nothing but its path, so an old file
with new intentions silently gives you the old grid. `display()` prints the
provenance, so a glance catches it.

The grid records that **provenance** (isochrone/spectral library, `imf_type`,
FSPS version, build kwargs) in the HDF5 file, and `CSPBasis` reads the
isochrone library back automatically: **you never set `isoc_type` by hand**,
and the nebular CLOUDY grid always matches the SSP isochrones.

**What each route needs.** With a fetched grid and `add_neb=False`,
`add_dust_emission=False` (as in Step 1 below), nothing is read from
`$SPS_HOME` and FSPS need not be installed at all. Switch nebular or dust
emission on and the FSPS **data files** are required — the clone, not the
compiler. Only `SSPData.from_fsps` needs `python-fsps` itself. See
[Installing FSPS](#installing-fsps-and-setting-sps_home).

### Step 1: fit a galaxy end-to-end

A self-contained, copy-paste-runnable joint fit. It makes a mock galaxy from
known truth **with the same forward model it then fits**, so the data is always
consistent with the SSP grid you built and the fit recovers the truth. No data
files needed.

```python
import jax, jax.numpy as jnp
import numpy as np
from ceridwen import SSPData, CSPBasis, SedModel, fitSED, Instrument, Cosmology
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

rng = np.random.default_rng(42)
ZRED = 0.1                                         # fixed spectroscopic redshift
FILTERS = ["galex_FUV", "galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0",
           "sdss_i0", "sdss_z0", "twomass_J", "twomass_H", "twomass_Ks",
           "wise_w1", "wise_w2"]
SPEC_WAVE = np.linspace(4000.0, 8000.0, 600)       # observed-frame vacuum Angstrom
TRUTH = {                                          # parameters to inject and recover
    "logsfr_ratios":      jnp.array([0.3, 0.2, -0.1, -0.4, -0.6]),
    "logzsol":            jnp.array([-0.2]),       # log10(Z/Z_sun) of the SSP grid
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}

# SSP grid: load the Step 0 cache, or build it on first run (needs FSPS).
# lookback_time is the static SFH node grid (Gyr, increasing, index 0 = today,
# >= 2 nodes; oldest node < age of universe at ZRED, which SedModel checks).
import pathlib
SSP_FILE = pathlib.Path("ssp_data.h5")
if SSP_FILE.is_file():
    print(f"[grid] loading cached SSP grid: {SSP_FILE}")
    ssp = SSPData.load(str(SSP_FILE))
else:
    print(f"[grid] no cache found, building with FSPS (a few minutes) ...")
    ssp = SSPData.from_fsps(imf_type=1, save_to=str(SSP_FILE))
ssp.display()                                      # grid summary + provenance

print("[csp] building the composite-stellar-population basis ...")
# The cosmology is set here, once, on the object that computes distances and
# ages (Cosmology.planck18() / planck15() / wmap9() / flat(H0, Om0) /
# from_astropy(...)); SedModel reads it from the CSP and fitSED records it.
csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
               cosmo=Cosmology.planck18(),
               zh_const=True, sfh_interp="step",
               add_dust=False, add_diffuse_dust=True, add_neb=False, verbose=False)
print("[csp] done")

# SFH is sampled as logsfr_ratios (Prospector convention) -> per-node SFR.
sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

# One SedModel builder for BOTH the mock and the fit: this is what keeps them
# consistent. Observations carry flux/uncertainty; empty ones (filters and
# wavelengths only) are enough to generate the mock.
def build_model(observations):
    return SedModel(
        csp, observations=observations,
        priors={
            # logzsol = log10(Z/Z_sun) of the loaded grid; 0.0 is solar.  Keep the
            # prior inside the grid axis (BPASS [-2.30, +0.30], MIST [-2.50, +0.50]
            # for the MIST grids); print the exact range with
            #     print(float(csp.zmet.min()), float(csp.zmet.max()))
            # and csp.check_param_ranges() warns about out-of-grid values.
            "logzsol": Uniform(low=-2.0, high=0.2),
            "logmass": Uniform(low=9.0, high=12.0),
            "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
            "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
            "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
        },
        transforms={"sfh": logsfr_to_sfh},
        free_param_init={"logsfr_ratios": jnp.zeros(5), "logmass": jnp.array([10.0])},
        zred=ZRED,
    )
```

Next, generate the mock: push TRUTH through the forward model and add Gaussian
noise. The generator model carries *empty* observations (filters and
wavelengths only), which is all `predict` needs.

```python
print("[mock] building the generator model (empty observations) ...")
gen = build_model([
    Photometry(filters=FILTERS, name="phot"),
    Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.sigma_kms(150.0),  # LSF sigma [km/s]
             name="spec"),
])
print("[mock] predicting TRUTH through the forward model ...")
truth_pred = gen.predict(TRUTH)                    # AB maggies (phot), F_nu (spec)
# Sanity check: the photometry is absolutely calibrated. The fixed ZRED is
# injected into the forward model by SedModel.predict, so a z=0.1,
# logmass=10.5 galaxy lands at ~1e-7 maggies (AB ~ 17-18) in the bright bands.
mag = np.asarray(truth_pred["phot"]); mag_unc = mag / 20.0
sfx = np.asarray(truth_pred["spec"]); sfx_unc = np.abs(sfx) / 25.0
print(f"[mock] photometry: {mag.min():.3e} .. {mag.max():.3e} maggies "
      f"(expect bright bands ~1e-7, AB ~ 17-18)")
mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)
sfx_obs = sfx + sfx_unc * rng.standard_normal(sfx.shape)
print("[mock] noise added (SNR 20 phot / 25 spec)")
```

Finally, wrap the noisy mock in observation containers and build the model to
fit: same `build_model`, now with data attached.

```python
print("[obs] building phot ...")
phot = Photometry(filters=FILTERS, flux=mag_obs, uncertainty=mag_unc, name="phot")
print("[obs] building spec ...")
spec = Spectrum(wavelength=SPEC_WAVE, flux=sfx_obs, uncertainty=sfx_unc,
                instrument=Instrument.sigma_kms(150.0), name="spec")  # LSF sigma [km/s]
phot.display(); spec.display()                     # sanity-check the observations
print("[model] building the fit model ...")
model = build_model([phot, spec])
print("[model] ready, handing over to fitSED")
```

Now pick a sampler. Both fill the same `result` object, so everything after the
fit (reporting, plotting) is identical.

#### Option A: nested sampling

Gradient-free, and also returns the Bayesian evidence (log Z) for model
comparison.

```python
result = fitSED(
    model,
    sampler="ns",
    sampler_kwargs={"num_live": 400, "num_delete": 80, "logZ_tol": -5.0},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
print(f"log Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")
```

#### Option B: VI-preconditioned NUTS

VI learns a full-rank Gaussian transport map; NUTS then samples in the whitened
space (Hoffman et al. 2019).

```python
result = fitSED(
    model,
    sampler="nuts",
    vi="tril",                               # "iaf" for NeuTra neural transport
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)

# VI convergence: -ELBO should drop and then plateau.
import matplotlib.pyplot as plt
plt.figure(); plt.plot(result.raw["vi_losses"]); plt.yscale("log")   # -ELBO
plt.xlabel("VI iteration"); plt.ylabel(r"$-\mathrm{ELBO}$")
```

#### Starting NUTS at the MAP

`fitSED(..., optimize=True)` first maximises the same log-posterior with L-BFGS
(`ceridwen.optimize.map_fit`, from `model.theta_init` plus `n_starts` prior draws, Prospector's
`nmin`), starts NUTS (and its VI map) there, and stores the MAP under `/map` in the result file.
Nested sampling draws its live points from the prior, so there the MAP is only recorded.
`map_fit(model)` on its own returns a `MAPResult` whose `.theta` is a ready `free_param_init`.

```python
from ceridwen.optimize import map_fit

best = map_fit(model, n_starts=16, rng_key=jax.random.PRNGKey(1))
result = fitSED(model, sampler="nuts", optimize=True,
                optimize_kwargs={"n_starts": 16}, output_dir="./my_fit")
```

#### Resuming a nested-sampling run

A nested-sampling run with checkpoints (`checkpoint_dir=` or `$CERIDWEN_CHECKPOINT_DIR`) can be
continued after a kill with `resume_from=`: it reproduces the uninterrupted run at the same
`rng_key`, and refuses a checkpoint whose settings, parameter shapes or live-point
log-likelihoods do not match the model.

```python
result = fitSED(model, sampler="nested", output_dir="./my_fit",
                sampler_kwargs={"checkpoint_dir": "./ckpt",
                                "resume_from": "./ckpt/ns_checkpoint_12345.pkl"})
```

#### Inspecting the results (identical for both samplers)

Nested samples carry importance weights; NUTS draws do not. `PostProcess`
handles both: it resamples to equal weight (recomputed from the birth
contours, so it is exact for the stored point order), pushes the draws
through the same forward model that was fitted, and writes the figures.
Nothing below needs `result.log_weights` by hand.

This block also works on a reloaded fit from an earlier session:
`result = load_result_h5("my_fit/ceridwen_result.h5")` (importable from
`ceridwen`) returns the same result object; only the model from Step 1 has
to be rebuilt, with the same `ZRED` (recorded in the file's `/model` attrs).
`ceridwen.resultfile.rebuild_model(path, csp, observations, transforms=...)` takes the priors,
free parameters, zred and kinematics from the file and raises if the rebuilt model differs from
the one recorded; `check_model_against_result(model, path)` lists every difference. The CSP,
the observations and the transform callables are not stored and must be supplied.

```python
from ceridwen import PostProcess

pp  = PostProcess(model, result, n_samples=2000)
out = pp.run()

# Recovered vs injected truth (equal-weight draws, so plain percentiles).
truth = {p: float(TRUTH[p][0]) for p in ("logzsol", "logmass", "diffuse_tau_kc", "diffuse_dust_index")}
for p, t in truth.items():
    lo, med, hi = np.percentile(out["theta"][p], [16, 50, 84])
    print(f"{p:>20}: true {t:+7.3f}   fit {med:+7.3f}  (-{med - lo:.3f}/+{hi - med:.3f})")

# Three figures per galaxy: summary (SED + chi, SFH, marginals), corner
# (median dashed, maximum-likelihood red, truth green) and sampling
# diagnostics (dead points / log-weights / ESS for nested sampling,
# traces / R-hat / ESS for NUTS).
paths = pp.figures("./my_fit/figures", title="mock galaxy", truths=truth)
pp.save("./my_fit/post.npz")           # load_postprocess() rebuilds the dict
```

The posterior-predictive photometry lives in
`out["prediction"]["photometry"]["phot"]` (draws x bands, maggies) and the
observer-frame spectra in `out["prediction"]["spectra_observed"]`, so a
custom SED plot is a matter of `np.percentile(..., [16, 50, 84], axis=0)`;
the individual figure functions in `ceridwen.plotting` accept a
`savepath=None` and return the `Figure` for further styling.

The `result` object has posterior samples keyed by parameter name, plus per-phase
wall-clock timings (and, for NUTS+VI, the VI trace) in `result.raw`; everything
is also written to `./my_fit/ceridwen_result.h5`.

Units: `Photometry` predictions are AB maggies, `Spectrum` predictions are
observed-frame F_nu in erg s^-1 cm^-2 Hz^-1 (cgs; multiply by 1e32 for nJy) and
`Lines` predictions are integrated fluxes in erg s^-1 cm^-2. All three need a
redshift (`zred=`) or `lumdist_mpc=`: at `zred = 0` without `lumdist_mpc` no flux
factor is applied and predictions are `L_sun/Hz x 10^logmass`. If you replace
`model.observations` after construction, call `model.setup_observations()`
before predicting (`fitSED(model, observations)` does this for you).

For **nebular emission lines** (a `Lines` container, `add_neb=True`, which needs
the CLOUDY grids at `$SPS_HOME`), see the [tutorial](https://github.com/Espe13/ceridwen/blob/main/docs/tutorial.md).


### Run a bundled example

The fastest way to confirm your whole setup works end to end. It follows the
same steps as above but drives the nested sampler directly through
`run_sampler` with relaxed demo settings (150 live points, `logZ_tol=-2`), so
it writes no HDF5 result: it loads the SSP grid (building it from FSPS only if
none is found), generates mock UV-to-IR photometry in the unitless `zred = 0`
convention (the `SedModel(zred=0) applies NO flux factor` warning is expected
here), fits it, prints recovered-vs-true parameters and writes the summary,
corner and sampling-diagnostic figures to `examples/quickstart_figures/`:

```bash
python examples/quickstart.py
```

Without a clone, fetch that one file and run it anywhere — it imports only from
the installed package:

```bash
curl -LO https://raw.githubusercontent.com/Espe13/ceridwen/v1.0.11/examples/quickstart.py
SSP_FILE=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_miles_chab'))") \
    python quickstart.py
```

If it prints a recovered-vs-true table, your setup works. `logmass` recovers the
injected truth; `Z` and the dust parameters are weakly constrained by broadband
photometry alone, so their posteriors are broad (add spectroscopy or emission
lines to pin them down).

### Post-process a fit

`PostProcess` turns the samples into posterior distributions of everything you
report, all computed by the same forward model that was fitted:

```python
from ceridwen import PostProcess

pp  = PostProcess(model, result)           # SamplingResult or the .h5 file fitSED wrote
out = pp.run()
out["theta"]["logmass"]                    # (N,) equal-weight draws
out["extras"]["sfh"]["sfr10"]              # mean SFR over the last 10 Myr; ssfr10, sfr100, ...
out["extras"]["sfh"]["mass_surviving"]     # mfrac x mass_formed; mfrac, ssfr10_surviving (grids with a mass table)
out["extras"]["uv"]["MUV"]                 # M_UV at 1500 A; LUV, MUV_intrinsic
out["extras"]["ionizing"]["nion"]          # Q(H); xion; fesc when the model has frac_obrun
out["prediction"]["photometry"]["phot"]    # posterior-predictive maggies; spectra, lines
out["prediction"]["spectra_intrinsic"]     # stellar continuum on the model grid; spectra_dustfree, spectra_model
out["bestfit"]["theta"]["logmass"]         # the highest-likelihood sample, with all of the above
pp.figures("./figures", title="my galaxy")  # summary.pdf, corner.pdf, diagnostics.pdf
pp.save("post.npz")                        # load_postprocess("post.npz") rebuilds the dict
```

`pp.figures()` writes three plain-matplotlib figures per galaxy (no style
library; ceridwen's blue palette, `ceridwen.plotting.COLORS`): a summary
(observed SED with the 16-84% posterior band and chi panel, emission-line
residuals when lines were fitted, the SFH with prior and posterior bands, and
the 1-D marginals), a corner of all fitted parameters with the median and the
maximum-likelihood sample marked, and a sampling diagnostic (nested: dead
points, log-likelihood run, cumulative weight, ESS; NUTS: traces, split
R-hat, ESS). `truths={name: value}` adds the injected values of a mock test;
`ceridwen.plotting.summary_figure/corner_figure/diagnostic_figure` return the
`Figure` when called directly.

Your own quantities are one function of a draw, `derived={"A_V": lambda s:
2.5 * np.log10(s.dustfree[s.index_of(5500)] / s.full[s.index_of(5500)])}`.
See `docs/postprocessing.md` for the conventions (equal-weight resampling,
formed mass, rest-frame luminosities) and `examples/demo_postprocess.py`.

---

## Verification

The checks are layered, and each layer answers a different question. What a
layer asserts, the command that runs it, and whether it runs on GitHub:

| layer | command | what it asserts | on CI |
|---|---|---|---|
| Environment | `python -m ceridwen.check` | Python version, dependencies, bundled filter curves and attenuation laws, float64, nested sampling, `$SPS_HOME`; each problem with its fix | no |
| Tests, FSPS-free | `pytest -m "not fsps and not gpu" -q -ra` | units, conventions, broadening, likelihood, samplers, post-processing, the misuse guards (25 test files) | every push and pull request |
| Regression baselines | `pytest tests/regression/test_regression.py -q` | 9 blocks (SSP spectrum, IGM, cosmology, CSP components, dust attenuation, dust emission, nebular, CSP spectrum, likelihood) against stored arrays at `atol=1e-10, rtol=1e-7`, and a maximum relative residual of 1e-6; writes comparison figures to `tests/regression/figures/` | no, needs `$SPS_HOME` |
| Golden spectra | `pytest tests/csp/test_lookback_flip_invariant.py -q` | 6 SFH / metallicity configurations against committed arrays: SSP weights at `rtol=1e-12`; spectra, line fluxes and photometry at `rtol=1e-6` (they pass through float32 contractions) | no, needs `$SPS_HOME` |
| Misuse report | `python tests/regression/misuse_report.py` | each known user error ends in an exception or a warning; a `SILENT` row is a bug | its assertions run in the tests above |
| Static API check | `python scripts/check_api_usage.py` | every call in `tests/`, `examples/` and the Python blocks of this README, `GOTCHAS.md` and `docs/` passes only keyword arguments that exist, and uses no removed name | no |
| Byte identity | `python scripts/bit_identity_check.py --save old.npz` on one commit, `--compare old.npz` on the next | a refactor changes nothing: every output array equal under `tobytes()` (dtype, shape, every bit); `--ns` adds a nested-sampling run at the same RNG key | no |
| End-to-end recovery | `python examples/quickstart.py` | mock photometry from known parameters, fitted back, recovered-vs-true table and figures | no |

**Skips are not passes.** The suite's main SSP grid is too large for the
repository, and a grid-dependent test *skips* when it cannot find one, so read
the `-ra` summary at the end of a run. The grid is the published BPASS grid;
fetch it once and point the tests at it:

```bash
export CERIDWEN_TEST_SSP=$(python -c "from ceridwen.ssps import fetch_grid; print(fetch_grid('mist_bpass_v2'))")
export SSP_FILE=$CERIDWEN_TEST_SSP
pytest -m "not fsps and not gpu" -q -ra
```

CI does the same, so its green tick covers the grid-dependent tests. What CI
cannot cover is everything that reads the CLOUDY and dust-emission tables: the
runners have no FSPS data, so the nebular tests, the regression baselines and
the golden spectra run only on a machine with `$SPS_HOME` set. Byte identity is
a CPU statement; on GPU the scatter-add that paints emission lines can differ
in the last bit from run to run.

## How this code was built

ceridwen is written and maintained by one person, with AI coding assistants
used for drafting, refactoring and review. The safeguard is not trust in the
tool but the checks above, and a few rules that apply to every change,
whoever or whatever proposes it:

- **Conventions are written down where a tool reads them first.**
  [`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md) holds the conventions that are easy to get wrong
  (solar-relative `logzsol` metallicity, lookback-time ordering, units and frames),
  the hard requirements and the module map; [`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md) is the
  misuse guide. Both are checked against the code by the static API check.
- **A change has to say what it is.** A refactor or optimisation must be
  byte-identical on CPU. An intended change to the physics must come with its
  predicted size, and a new feature must add a golden configuration of its
  own. Existing baselines are not re-captured to make a test pass: the
  re-capture script for the golden spectra first asserts that the SSP weights
  still match the stored ones at `rtol=1e-12`.
- **Wrong input fails loudly.** The historical hazard was a plausible number
  from a mistyped `theta` key or a metallicity in the wrong units. Every guard
  sits in construction or setup code, or runs once at trace time, so the
  compiled sampling path is unchanged.
- **Independent references where they exist.** The spectral broadening is
  tested against direct quadrature and, when `sedpy` is installed, against
  its smoothing routines (`tests/test_broadening.py`).

---

## Fitting [α/Fe] — no FSPS required

The α-enhanced grids (FSPS v4.0, aMIST isochrones + C3K spectra,
[α/Fe] ∈ {−0.2, 0.0, +0.2, +0.4, +0.6}) add α-element enhancement as a
sampled stellar axis: a chemical clock for the formation timescale,
measured jointly with — and physically degenerate with — the total
metallicity. Building these grids yourself requires python-fsps compiled
from source with `AFE_FLAG=1`, so the canonical grid is published on
Zenodo and fetched by name. `SSPDataAfe` and `CSPBasis_afe` are subclasses
of `SSPData` and `CSPBasis`: the [α/Fe] axis and its interpolation are the
only additions, the nebular arguments are dropped, and everything else (SFH
weights, dust, projection, flux factor, cosmology) is inherited (`ssp_afe=`
is keyword-only when constructing a grid by hand).
Because the α-enhanced basis carries **no nebular model** (no α-enhanced CLOUDY tables exist), nothing is read from
`$SPS_HOME` at fit time either: **the downloaded grid is the complete
stellar input, and no FSPS install is needed at all.**

```python
import jax.numpy as jnp
from ceridwen.ssps import fetch_grid, SSPDataAfe
from ceridwen.csp import CSPBasis_afe
from ceridwen import Cosmology

# One call: downloads once into ~/.ceridwen/grids, sha256-verified.
ssp = SSPDataAfe.load(fetch_grid("amist_c3k_hr_krou_afe"))
ssp.display()                       # (n_afe, n_Z, n_age, n_wave) = (5, 13, 107, 10992)

csp = CSPBasis_afe(ssp, lookback_time=jnp.linspace(0.0, 12.0, 9),
                   cosmo=Cosmology.planck18(), zh_const=True, verbose=False)

theta = {
    "lookback_time": jnp.linspace(0.0, 12.0, 9),
    "sfh":           jnp.exp(-jnp.linspace(0.0, 12.0, 9) / 1.0),
    "logzsol":       jnp.array([-0.3]),   # = [Fe/H] on an aMIST grid
    "afe":           jnp.array([0.4]),    # [α/Fe]: re-partitions that Z
    "tau_pow":           jnp.array([0.3]),
    "diffuse_tau_kc":    jnp.array([0.2]),
    "diffuse_dust_index": jnp.array([0.0]),
}
wave, fnu = csp.wave, csp.get_spectrum(theta)   # rest-frame Lsun/Hz per Msun
```

Notes: `theta["afe"]` is interpolated differentiably between the two
bracketing grid planes, so it works under `jit`/`grad`/`vmap` and in every
sampler; `logzsol` is `[Fe/H]` here (every `[alpha/Fe]` plane shares one `[Fe/H]` axis,
FSPS AFE_FLAG=1), and the total metallicity `[Z/H]` is the derived `logzsol_total`
= `logzsol + log10(1 - x + x 10^[alpha/Fe])`, `x = 0.687490` (MIST v2.5 / GS98); `CSPBasis_afe` accepts **only** α-aware 4-D grids — passing a
legacy 3-D grid raises a `TypeError` telling you to use `CSPBasis`;
emission-line observations are rejected (continuum and photometry only)
until α-enhanced photoionisation grids exist.

The grid used above, `amist_c3k_hr_krou_afe`, is the α grid of the Zenodo deposit:
**high-resolution** C3K (10992 λ points, R up to ~65000 in the optical, Kroupa IMF), built
from the alpha-MC C3K high-res SSPs (MIST v2.5 + C3K v2.3) of M. J. Park, which are too large
to ship inside FSPS/python-FSPS. Its native axis is the FSPS label
log10 Z = [Fe/H] + log10(0.0185), i.e. logzsol = [Fe/H]. It does not carry a surviving-mass
table yet, so fit it with `fitSED(..., mfrac=False)` and quote the formed mass.

## Troubleshooting

- **Run `python -m ceridwen.check` first.** It reports missing dependencies,
  whether the bundled filter curves and attenuation laws are present, an unset
  or wrong `$SPS_HOME`, whether float64 is enabled, and whether nested sampling is available, each with the fix.
- **Install needs Python 3.11+** (see Installation); `blackjax` >= 1.6
  and `jax` >= 0.9 require it.
- **Common scientific pitfalls** (metallicity conventions, silently-ignored
  `theta` typos, the lookback-time convention) are documented in
  [`GOTCHAS.md`](https://github.com/Espe13/ceridwen/blob/main/GOTCHAS.md). If you're letting an AI assistant help you use
  ceridwen, point it at [`AGENTS.md`](https://github.com/Espe13/ceridwen/blob/main/AGENTS.md).

---

## Modules

| module | purpose |
|---|---|
| `ceridwen.ssps`         | SSP tables, HDF5 I/O |
| `ceridwen.csp`          | composite stellar populations, forward model |
| `ceridwen.dust`         | dust attenuation + emission |
| `ceridwen.neb`          | nebular continuum + emission lines |
| `ceridwen.observation`  | `Photometry`, `Spectrum`, `Lines` data containers + projection matrices |
| `ceridwen.broadening`   | `Kinematics` (galaxy sigma_gal / sigma_gas), `Instrument` (LSF), `DEFAULT_KINEMATICS`: the one place spectral widths are set |
| `ceridwen.priors`       | `Uniform`, `Normal`, `ClippedNormal`, `LogNormal`, `LogUniform`, `StudentT` |
| `ceridwen.likelihood`  ; `PolyMarginalGaussianLikelihood` (calibration polynomial marginalised analytically under a Gaussian prior, `Spectrum(polynomial_mode="marginalize")`) | `DiagonalGaussianLikelihood`, `MultiObservationLikelihood` (honours `sky`, `calibration`, `upper_limit`, `noise_floor`; optional per-observation outlier mixture `f_outlier_spec` / `f_outlier_phot` / `f_outlier_lines`, default 0 = off, see `docs/outlier_model.md`); `GPGaussianLikelihood` (squared-exponential GP on a spectrum's whitened residuals, `log_gp_amp_spec` / `log_gp_length_spec`, off by default, see `docs/gp_likelihood.md`); `PolyMarginalGaussianLikelihood` (calibration polynomial marginalised analytically under a Gaussian prior, `Spectrum(polynomial_mode="marginalize")`) |
| `ceridwen.model`        | `SedModel` parameter + prediction layer |
| `ceridwen.sampler`      | priors, nested sampling, NUTS, VI transport maps |
| `ceridwen.cosmology`    | `Cosmology` (Planck18/Planck15/WMAP9 presets, `flat`, `from_astropy`), JAX-native distances and ages |
| `ceridwen.igm`          | IGM attenuation models (Madau 1995 by default; `MadauDampingDLA` adds the damping wing and a DLA) |
| `ceridwen.fit`          | `fitSED` top-level convenience wrapper |
| `ceridwen.postprocess`  | `PostProcess`: posterior distributions of derived quantities and predictions |
| `ceridwen.plotting`     | summary, corner and sampling-diagnostic figures (`PostProcess.figures`) |

---

## References

- **Hoffman et al. 2019**, *NeuTra-lizing Bad Geometry in HMC Using Neural Transport*, [arXiv:1903.03704](https://arxiv.org/abs/1903.03704) (VI-preconditioned NUTS, `ceridwen.sampler.vi`)
- **Madau 1995**, ApJ 441, 18 (IGM transmission, `ceridwen.igm.Madau1995`)
- **Miralda-Escudé 1998** and **Totani et al. 2006** (IGM damping wing), **Tepper-García 2006, 2007** (Voigt profile of the DLA), as cited by and ported from Prospector (`ceridwen.igm.MadauDampingDLA`)
- **Planck Collaboration 2020**, A&A 641, A6 (default cosmology, `ceridwen.cosmology`)
- **Kriek & Conroy 2013**, ApJ 775, L16 (diffuse dust attenuation shape, `ceridwen.dust`)
- **Conroy, Gunn & White 2009** (FSPS, upstream SSP provider)

---

## Citing

If you use ceridwen in your research, please cite it:

```bibtex
@misc{stoffers2026ceridwen,
  author       = {Stoffers, Amanda},
  title        = {{CERIDWEN}: Fast and Flexible {GPU}-Accelerated Stellar Population Inference},
  year         = {2026},
  note         = {Version 1.0.11},
  howpublished = {\url{https://github.com/Espe13/ceridwen}}
}
```

---

## Related projects

- [sedpy](https://github.com/bd-j/sedpy) by Benjamin D. Johnson: the origin of the filter-convolution and attenuation-curve code that ceridwen now carries internally (via [sedpy_jax](https://github.com/Espe13/sedpy_jax), the JAX rewrite), and of the AB photon-counting conventions it follows. Ceridwen no longer depends on either package. Spectral broadening is done inside ceridwen (`ceridwen.broadening`) and reproduces sedpy's direct convolutions to better than 1e-3.

---

Maintainer: [Amanda Stoffers](https://www.amanda-stoffers.de), Kavli Institute for Cosmology, University of Cambridge, `aas208@cam.ac.uk`
