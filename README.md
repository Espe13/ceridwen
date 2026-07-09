

# Ceridwen

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven **WE**ight calculatio**N**s — a JAX-native, GPU-capable spectral energy distribution (SED) fitting package with variational-inference preconditioned Hamiltonian Monte Carlo and native redshift support.

---

## Features

- [x] Star formation history (non-parametric continuity + parametric)
- [x] Metallicity history
- [x] Dust attenuation (Kriek & Conroy diffuse, power-law birth-cloud, multi-component age-dependent)
- [x] Dust emission (Draine & Li grids)
- [x] Nebular continuum + emission lines (CLOUDY grids)
- [x] Observation input (broadband photometry, emission-line fluxes, spectra)
- [x] Redshift-aware forward model with cosmological flux normalisation
- [x] IGM attenuation (Madau 1995), extensible via `IGMModel` ABC
- [x] NUTS / nested sampling / variational-inference preconditioned NUTS
- [ ] α-enhanced SSPs

Everything is written against `jax.numpy` with `@jit` and `vmap`/`pmap` in mind: the forward model is a single XLA graph, the sampler runs on GPU, and the sampling hot path contains zero Python branches.

---

## Installation

**Requires Python 3.11.** Nested sampling depends on the official `blackjax`
(its merged NSS), which needs Python 3.11. Create the environment with exactly
that version:

```bash
conda create -n ceridwen python=3.11 -y
conda activate ceridwen

git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install .
```

**GPU is the default.** On Linux this installs the CUDA 12 JAX wheels (the
CUDA libraries are bundled — an NVIDIA driver >= 525 is the only system
requirement, no toolkit install), and JAX uses the GPU automatically. No
flags, no separate install. If the machine has no usable NVIDIA GPU, the
same install falls back to CPU at import time — one warning, identical
results, just slower. macOS and native Windows have no CUDA wheels and get
the CPU build automatically; on a Windows machine with an NVIDIA GPU,
install inside WSL2. Every `fitSED` call prints which backend it is on:

```
ceridwen.fitSED
  Device      : GPU  (CudaDevice(id=0))
```

The one thing you install separately is **FSPS** (it compiles
Fortran, so it can't be a pip dependency); see
[Installing FSPS](#installing-fsps-and-setting-sps_home) below.

> **blackjax** is pinned to a fixed commit (`f73e12956`) because its nested
> sampler (`blackjax.nss`) is not in a PyPI release yet, so CERIDWEN installs it
> from GitHub. This becomes a normal version pin once that release ships.

### Installing FSPS and setting `$SPS_HOME`

CERIDWEN uses [FSPS](https://github.com/cconroy20/fsps) in two ways. The
[`python-fsps`](https://dfm.io/python-fsps) wrapper builds the SSP cache (Step
0). Separately, the FSPS **data files** supply the CLOUDY nebular grids and
Draine & Li dust-emission templates: when `add_neb=True` or
`add_dust_emission=True`, **CERIDWEN reads those files directly from
`$SPS_HOME`** — FSPS itself is not run at fit time, it just provides the data.

FSPS is **not** a pure-Python wheel: it needs a Fortran compiler and a clone of
the FSPS data files, and `python-fsps` reads the `$SPS_HOME` environment
variable (it fails to import if that is unset or wrong).

```bash
# 1. A Fortran compiler (pick one for your system):
brew install gcc                       # macOS (Homebrew)
sudo apt-get install gfortran          # Debian/Ubuntu
conda install -c conda-forge gfortran  # any OS, inside your conda env

# 2. Pick where the FSPS data should live and point $SPS_HOME at it. This can
#    be ANY path -- $HOME, a data disk, cluster scratch, etc. `git clone` writes
#    to that absolute path, so it does NOT matter which directory you run it from
#    (no `cd` needed). Just change the path below to wherever you want it.
export SPS_HOME="$HOME/fsps"        # <- edit this to your chosen location
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

# 3. Install the Python wrapper (it compiles against $SPS_HOME):
python -m pip install "fsps>=0.4.4"
```

**Make `$SPS_HOME` permanent.** Step 2 above only sets it for the current
terminal; `python-fsps` needs it in *every* session. Add it to your shell
startup file so it persists (use the **same path you chose in step 2** — the
`$HOME/fsps` below is just the example default):

```bash
# zsh (the macOS default shell):
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.zshrc
source ~/.zshrc

# bash (most Linux):
echo 'export SPS_HOME="$HOME/fsps"' >> ~/.bashrc
source ~/.bashrc
```

Open a **new** terminal and run `echo $SPS_HOME`, it should print the path. If
it's blank, the line went into the wrong file (check which shell you use with
`echo $SHELL`). Then confirm the whole setup with:

```bash
python -m ceridwen.check
```

FSPS builds your SSP grid (you control the isochrones, spectral library, and
IMF) and supplies the nebular / dust-emission data. With it set up, next stop:
[`examples/quickstart.py`](examples/quickstart.py), a complete runnable fit.

---

## Quick start

### Step 0 — build the SSP grid (once per FSPS configuration)

Ceridwen's forward model consumes an HDF5 cache of SSP spectra precomputed
with FSPS. You build it yourself (you control the isochrones, spectral library,
and IMF); it takes a few minutes on CPU and only has to be done once. You can use it
for all fits that you want to base on the same IMF and libraries.

```python
from ceridwen import SSPData

# Generate + cache. from_fsps accepts ONLY the kwargs that define the stellar
# library / IMF (imf_type and its parameters, isochrone-phase knobs like
# tpagb_norm_type). Anything the forward model applies itself (dust, SFH,
# nebular emission, IGM, redshift, or a fixed metallicity) is rejected.
ssp = SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")

# Subsequent runs just reload the cache:
# ssp = SSPData.load("ssp_data.h5")
```

The grid records its **provenance** (isochrone/spectral library, `imf_type`,
FSPS version, build kwargs) into the HDF5 file, and `CSPBasis` reads the
isochrone library back automatically — **you never set `isoc_type` by hand**,
and the nebular CLOUDY grid always matches the SSP isochrones.

### Step 1 — fit a galaxy end-to-end

A self-contained, copy-paste-runnable joint fit. It makes a mock galaxy from
known truth **with the same forward model it then fits**, so the data is always
consistent with the SSP grid you built and the fit recovers the truth — no data
files needed.

```python
import jax, jax.numpy as jnp
import numpy as np
from ceridwen import SSPData, CSPBasis, SedModel, fitSED
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
    "Z":                  jnp.array([-2.0]),       # log10 ABSOLUTE Z (ssp_lgmet)
    "logmass":            jnp.array([10.5]),
    "diffuse_tau_kc":     jnp.array([0.5]),
    "diffuse_dust_index": jnp.array([-0.7]),
}

# SSP grid from Step 0. lookback_time is the static SFH node grid (Gyr,
# increasing, index 0 = today, >= 2 nodes; oldest node < age of universe at ZRED).
ssp = SSPData.load("ssp_data.h5")
ssp.display()                                      # grid summary + provenance
csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
               zh_const=True, sfh_interp="step",
               add_dust=False, add_diffuse_dust=True, add_neb=False, verbose=False)

# SFH is sampled as logsfr_ratios (Prospector convention) -> per-node SFR.
sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

# One SedModel builder for BOTH the mock and the fit — this is what keeps them
# consistent. Observations carry flux/uncertainty; empty ones (filters and
# wavelengths only) are enough to generate the mock.
def build_model(observations):
    return SedModel(
        csp, observations=observations,
        priors={
            # Z is log10 ABSOLUTE metallicity (= ssp_lgmet), NOT log10(Z/Zsun);
            # solar ~ -1.85. Keep priors inside the grid — print the range with
            #     print(float(csp.zmet.min()), float(csp.zmet.max()))
            # and csp.check_param_ranges() warns about out-of-grid values.
            "Z": Uniform(low=-3.9, high=-1.45),
            "logmass": Uniform(low=9.0, high=12.0),
            "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
            "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
            "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
        },
        transforms={"sfh": logsfr_to_sfh},
        free_param_init={"logsfr_ratios": jnp.zeros(5), "logmass": jnp.array([10.0])},
        zred=ZRED,
    )

# (1) Make the mock: predict TRUTH through the model, add Gaussian noise.
gen = build_model([
    Photometry(filters=FILTERS, name="phot"),
    Spectrum(wavelength=SPEC_WAVE, resolution=150.0, smoothtype="vel", name="spec"),
])
truth_pred = gen.predict(TRUTH)                    # AB maggies (phot), F_nu (spec)
# Sanity check: the photometry is absolutely calibrated — the fixed ZRED is
# injected into the forward model by SedModel.predict, so a z=0.1,
# logmass=10.5 galaxy lands at ~1e-7 maggies (AB ~ 17-18) in the bright bands.
mag = np.asarray(truth_pred["phot"]); mag_unc = mag / 20.0
sfx = np.asarray(truth_pred["spec"]); sfx_unc = np.abs(sfx) / 25.0
mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)
sfx_obs = sfx + sfx_unc * rng.standard_normal(sfx.shape)

# (2) Observations to FIT (now carrying the mock data), then the fit model.
phot = Photometry(filters=FILTERS, flux=mag_obs, uncertainty=mag_unc, name="phot")
spec = Spectrum(wavelength=SPEC_WAVE, flux=sfx_obs, uncertainty=sfx_unc,
                resolution=150.0, smoothtype="vel", name="spec")
phot.display(); spec.display()                     # sanity-check the observations
model = build_model([phot, spec])
```

Now pick a sampler. Both fill the same `result` object, so everything after the
fit (reporting, plotting) is identical.

#### Option A — VI-preconditioned NUTS

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

# Recovered vs injected truth.
for p in ("Z", "logmass", "diffuse_tau_kc", "diffuse_dust_index"):
    samples = np.asarray(result.samples[p]).ravel()
    print(f"{p:>20}: true {float(TRUTH[p][0]):+7.3f}   "
          f"fit {np.median(samples):+7.3f} +/- {np.std(samples):.3f}")

# Quick-look plots — VI loss, a corner subset, and data vs model.
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples

plt.figure(); plt.plot(result.raw["vi_losses"]); plt.yscale("log")   # -ELBO
plt.xlabel("VI iteration"); plt.ylabel(r"$-\mathrm{ELBO}$")

subset = ["logmass", "Z", "diffuse_tau_kc"]
truth  = {p: float(TRUTH[p][0]) for p in subset}
data = np.column_stack([np.asarray(result.samples[p]).ravel() for p in subset])
axes = MCMCSamples(data=data, columns=subset).plot_2d(subset)
for yp in subset:                     # overlay injected truth as red dashed lines
    for xp in subset:
        ax = axes.loc[yp, xp]
        if ax is None:
            continue
        ax.axvline(truth[xp], color="red", ls="--", lw=1)
        if yp != xp:
            ax.axhline(truth[yp], color="red", ls="--", lw=1)

theta_med = {p: jnp.atleast_1d(jnp.median(jnp.asarray(v), axis=0))
             for p, v in result.samples.items()}
pred = model.predict(theta_med)                        # AB maggies, keyed by obs name
plt.figure()
plt.errorbar(phot.wave_eff, phot.flux, yerr=phot.uncertainty, fmt="o", label="data")
plt.plot(phot.wave_eff, np.asarray(pred["phot"]), "s", label="model")
plt.xlabel(r"$\lambda_{\rm eff}$ [Å]"); plt.ylabel("flux [maggies]")
plt.yscale("log"); plt.legend(); plt.show()
```

#### Option B — nested sampling

Gradient-free, and also returns the Bayesian evidence (log Z) for model
comparison. The truth table and the data-vs-model plot from Option A work
unchanged — only the sampler call and the (importance-weighted) corner differ.

```python
result = fitSED(
    model,
    sampler="ns",
    sampler_kwargs={"num_live": 400, "num_delete": 100},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
print(f"log Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")

# Nested samples carry importance weights — anesthetic applies them.
# plot_2d returns the same axes grid, so overlay the truth exactly as in Option A.
axes = result.to_anesthetic().plot_2d(["logmass", "Z", "diffuse_tau_kc"])
```

The `result` object has posterior samples keyed by parameter name, plus per-phase
wall-clock timings (and, for NUTS+VI, the VI trace) in `result.raw`; everything
is also written to `./my_fit/ceridwen_result.h5`.

For **nebular emission lines** (a `Lines` container, `add_neb=True`, which needs
the CLOUDY grids at `$SPS_HOME`), see the [tutorial](docs/tutorial.md).


### Run a bundled example 

The fastest way to confirm your whole setup works end to end. It builds an SSP
grid from FSPS on first run (and reuses it afterwards), generates mock UV-to-IR
photometry, fits it, and prints recovered-vs-true parameters plus a corner plot
(`examples/quickstart_corner.png`):

```bash
python examples/quickstart.py
```

If it prints a recovered-vs-true table, your setup works. `logmass` recovers the
injected truth; `Z` and the dust parameters are weakly constrained by broadband
photometry alone, so their posteriors are broad (add spectroscopy or emission
lines to pin them down). The two steps below are what the example does
internally, shown so you can adapt them to your own data.

---

## Troubleshooting

- **Run `python -m ceridwen.check` first.** It reports missing dependencies, an
  unset or wrong `$SPS_HOME`, a too-old `sedpy-jax`, and whether nested sampling
  is available — each with the fix.
- **Install needs Python 3.11** (see Installation); nested sampling pins the
  official `blackjax` (its merged NSS), which requires Python 3.11.
- **Common scientific pitfalls** — the metallicity-units trap, silently-ignored
  `theta` typos, the lookback-time convention — are documented in
  [`GOTCHAS.md`](GOTCHAS.md). If you're letting an AI assistant help you use
  ceridwen, point it at [`AGENTS.md`](AGENTS.md).

---

## Modules

| module | purpose |
|---|---|
| `ceridwen.ssps`         | SSP tables, HDF5 I/O |
| `ceridwen.csp`          | composite stellar populations, forward model |
| `ceridwen.dust`         | dust attenuation + emission |
| `ceridwen.neb`          | nebular continuum + emission lines |
| `ceridwen.observation`  | `Photometry`, `Spectrum`, `Lines` data containers + projection matrices |
| `ceridwen.model`        | `SedModel` parameter + prediction layer |
| `ceridwen.sampler`      | priors, nested sampling, NUTS, VI transport maps |
| `ceridwen.cosmology`    | JAX-native flat ΛCDM (Planck 18) + astropy fallback |
| `ceridwen.igm`          | IGM attenuation models (Madau 1995 by default) |
| `ceridwen.fit`          | `fitSED` top-level convenience wrapper |

---

## References

- **Hoffman et al. 2019**, *NeuTra-lizing Bad Geometry in HMC Using Neural Transport*, [arXiv:1903.03704](https://arxiv.org/abs/1903.03704) — VI-preconditioned NUTS (`ceridwen.sampler.vi`)
- **Madau 1995**, ApJ 441, 18 — IGM transmission (`ceridwen.igm.Madau1995`)
- **Planck Collaboration 2020**, A&A 641, A6 — default cosmology (`ceridwen.cosmology`)
- **Kriek & Conroy 2013**, ApJ 775, L16 — diffuse dust attenuation shape (`ceridwen.dust`)
- **Conroy, Gunn & White 2009** — FSPS, upstream SSP provider

---

## Related projects

- [sedpy_jax](https://github.com/Espe13/sedpy_jax) — JAX-compatible rewrite of [sedpy](https://github.com/bd-j/sedpy) by Benjamin D. Johnson; used by ceridwen for filter convolutions and smoothing.

---

Maintainer: [Amanda Stoffers](https://www.amanda-stoffers.de), Kavli Institute for Cosmology, University of Cambridge — `aas208@cam.ac.uk`

