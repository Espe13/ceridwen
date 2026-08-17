

# Ceridwen

**C**omprehensive **S**ED **E**stimation **R**outine **I**nvolving **D**ata-driven **WE**ight calculatio**N**s: a JAX-native, GPU-capable spectral energy distribution (SED) fitting package with variational-inference preconditioned Hamiltonian Monte Carlo and native redshift support.

Documentation: [www.amanda-stoffers.de/ceridwen](https://www.amanda-stoffers.de/ceridwen/)

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
- [x] α-enhanced SSPs (`CSPBasis_afe`, FSPS v4.0 aMIST + C3K, [α/Fe] sampled as a free parameter)

Everything is written against `jax.numpy` with `@jit` and `vmap`/`pmap` in mind: the forward model is a single XLA graph, the sampler runs on GPU, and the sampling hot path contains zero Python branches.

---

## Installation

**Requires Python 3.11.** The pinned `blackjax` (nested sampling) insists on
it. Create the environment with exactly that version:

```bash
conda create -n ceridwen python=3.11 -y
conda activate ceridwen

git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install .
```

**GPU is the default.** On Linux this installs the CUDA 12 JAX wheels (the
CUDA libraries are bundled; the only system requirement is an NVIDIA driver
\>= 525, no toolkit install), and JAX uses the GPU automatically. No flags,
no separate install. A machine without a usable NVIDIA GPU gets the same
install and falls back to CPU at import time: one warning, identical
results, more patience. macOS and native Windows have no CUDA wheels and
get the CPU build automatically; on a Windows machine with an NVIDIA GPU,
install inside WSL2. Every `fitSED` call prints which backend it landed on:

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
`$SPS_HOME`**; FSPS itself is not run at fit time, it just provides the data.

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
startup file so it persists (use the **same path you chose in step 2**; the
`$HOME/fsps` below is just the example default):

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

With FSPS set up, next stop:
[`examples/quickstart.py`](examples/quickstart.py), a complete runnable fit.

---

## Quick start

### Step 0: build the SSP grid (once per FSPS configuration)

Ceridwen's forward model consumes an HDF5 cache of SSP spectra precomputed
with FSPS. You build it yourself, so you control the isochrones, spectral
library, and IMF. It takes a few minutes on CPU (about one coffee) and then
serves every fit that shares those choices.

This block is safe to rerun: it loads the cached grid if one exists at
`SSP_FILE` and only builds (which needs FSPS and `$SPS_HOME`) when it doesn't:

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

### Step 1: fit a galaxy end-to-end

A self-contained, copy-paste-runnable joint fit. It makes a mock galaxy from
known truth **with the same forward model it then fits**, so the data is always
consistent with the SSP grid you built and the fit recovers the truth. No data
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

# SSP grid: load the Step 0 cache, or build it on first run (needs FSPS).
# lookback_time is the static SFH node grid (Gyr, increasing, index 0 = today,
# >= 2 nodes; oldest node < age of universe at ZRED).
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
csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
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
            # Z is log10 ABSOLUTE metallicity (= ssp_lgmet), NOT log10(Z/Zsun);
            # solar ~ -1.85. Keep priors inside the grid; print the range with
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
```

Next, generate the mock: push TRUTH through the forward model and add Gaussian
noise. The generator model carries *empty* observations (filters and
wavelengths only), which is all `predict` needs.

```python
print("[mock] building the generator model (empty observations) ...")
gen = build_model([
    Photometry(filters=FILTERS, name="phot"),
    Spectrum(wavelength=SPEC_WAVE, resolution=150.0, smoothtype="vel",  # sigma_v [km/s]
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
                resolution=150.0, smoothtype="vel", name="spec")  # sigma_v [km/s]
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

#### Inspecting the results (identical for both samplers)

Everything below is sampler-agnostic. The one thing to respect: nested
samples carry importance weights (`result.log_weights`), and ANY summary
statistic must use them or it is biased toward the prior (the corner would
look fine while medians and predicted fluxes drift). Resampling to equal
weight once makes everything downstream a plain median/percentile. NUTS
fills `log_weights` with zeros, so the same code runs unchanged there.

This block also works on a reloaded fit from an earlier session:
`result = load_result_h5("my_fit/ceridwen_result.h5")` (importable from
`ceridwen`) returns the same result object; only the forward-model parts
(`model.predict_vmap`) need the Step 1 setup re-run, with the same `ZRED`
(recorded in the file's `/model` attrs).

```python
import matplotlib.pyplot as plt

# Equal-weight posterior draws.
lw  = np.asarray(result.log_weights)
w   = np.exp(lw - lw.max()); w /= w.sum()
idx = rng.choice(w.size, size=1000, p=w)

# Recovered vs injected truth.
for p in ("Z", "logmass", "diffuse_tau_kc", "diffuse_dust_index"):
    s = np.asarray(result.samples[p])[idx].ravel()
    print(f"{p:>20}: true {float(TRUTH[p][0]):+7.3f}   "
          f"fit {np.median(s):+7.3f} +/- {np.std(s):.3f}")

# Corner plot: to_anesthetic() carries the weights, nothing to do by hand.
subset = ["logmass", "Z", "diffuse_tau_kc"]
truth  = {p: float(TRUTH[p][0]) for p in subset}
axes = result.to_anesthetic().plot_2d(subset)
for yp in subset:                     # overlay injected truth as red dashed lines
    for xp in subset:
        ax = axes.loc[yp, xp]
        if ax is None:
            continue
        ax.axvline(truth[xp], color="red", ls="--", lw=1)
        if yp != xp:
            ax.axhline(truth[yp], color="red", ls="--", lw=1)

# Data vs model WITH model uncertainty: push the equal-weight draws
# through the forward model in one vmapped call and plot the 16-84% band.
# The reshape matters: result.samples stores scalar parameters squeezed to
# (n_samples,), but predict_vmap expects per-sample shape (1,) -- i.e.
# batches of (N, 1) for scalars and (N, k) for vector parameters.
theta_draws = {p: jnp.asarray(np.asarray(v)[idx].reshape(len(idx), -1))
               for p, v in result.samples.items()}
pred_draws = np.asarray(model.predict_vmap(theta_draws)["phot"])  # (1000, n_bands)
lo, med, hi = np.percentile(pred_draws, [16, 50, 84], axis=0)

# Truth spectrum for the background, scaled to observed-frame flux exactly
# as csp.predict does (mass x flux factor); red like the truth lines in the
# corner plot.
from ceridwen.cosmology import flux_factor_maggies
wave_obs    = (1.0 + ZRED) * np.asarray(model.csp.wave)    # observed frame [A]
theta_truth = {k: jnp.asarray(v) for k, v in TRUTH.items()}
truth_fnu   = (np.asarray(model.csp.get_spectrum(model.apply_transforms(theta_truth)))
               * 10.0 ** float(TRUTH["logmass"][0]) * float(flux_factor_maggies(ZRED)))

# Both data and model photometry are AB maggies (F_nu-like); convert
# everything to F_lambda [erg s^-1 cm^-2 A^-1] for the classic SED plot.
AB_ZERO_FNU = 3.631e-20                    # 3631 Jy in erg s^-1 cm^-2 Hz^-1
C_AAS       = 2.998e18                     # speed of light [A/s]
wave       = np.asarray(phot.wave_eff)
to_flam    = AB_ZERO_FNU * C_AAS / wave**2  # per-band maggies -> F_lambda
truth_flam = truth_fnu * C_AAS / wave_obs**2
truth_flam = np.where(truth_flam > 0, truth_flam, np.nan)  # log-axis safety

fig, (ax, axc) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 6),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

# Main panel: truth spectrum behind the data and the model bands.
ax.plot(wave_obs, truth_flam, color="red", lw=0.5, alpha=0.6, zorder=2,
        label="truth")
ax.errorbar(wave, phot.flux * to_flam, yerr=phot.uncertainty * to_flam,
            fmt="o", color="red", markeredgecolor="black", markeredgewidth=1.5,
            zorder=3, label="data")
ax.errorbar(wave, med * to_flam,
            yerr=[(med - lo) * to_flam, (hi - med) * to_flam],
            fmt="s", color="None", markeredgecolor="blue",
            markeredgewidth=1, zorder=4, label="model (16-84%)")
ax.set_xlim(0.5 * wave.min(), 1.2 * wave.max())
ax.set_ylim(1e-18, 5e-16)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.legend(loc="upper right")

# Filter transmission curves, shaded, along the bottom of the main panel.
axf = ax.twinx()
for f in phot.filters:
    fw = np.asarray(f.wavelength)
    ft = np.asarray(f.transmission)
    axf.fill_between(fw, 0.0, ft / ft.max(), alpha=0.25, lw=0)
axf.set_ylim(0, 4); axf.set_yticks([])     # curves fill the lower quarter

# Chi panel: (data - model) / sigma. Unitless, so maggies are fine as is.
chi = (np.asarray(phot.flux) - med) / np.asarray(phot.uncertainty)
axc.axhspan(-1.0, 1.0, color="0.92", zorder=0)
axc.axhline(0.0, color="0.5", lw=0.8)
axc.scatter(wave, chi, s=14, zorder=3)
axc.set_ylabel(r"$\chi$")
axc.set_xlabel(r"$\lambda_{\rm eff}$ [Å]")
plt.show()
```

The `result` object has posterior samples keyed by parameter name, plus per-phase
wall-clock timings (and, for NUTS+VI, the VI trace) in `result.raw`; everything
is also written to `./my_fit/ceridwen_result.h5`.

For **nebular emission lines** (a `Lines` container, `add_neb=True`, which needs
the CLOUDY grids at `$SPS_HOME`), see the [tutorial](docs/tutorial.md).


### Run a bundled example

The fastest way to confirm your whole setup works end to end. It runs the two
steps above as one script: builds an SSP grid from FSPS on first run (and
reuses it afterwards), generates mock UV-to-IR photometry, fits it, and prints
recovered-vs-true parameters plus a corner plot
(`examples/quickstart_corner.png`):

```bash
python examples/quickstart.py
```

If it prints a recovered-vs-true table, your setup works. `logmass` recovers the
injected truth; `Z` and the dust parameters are weakly constrained by broadband
photometry alone, so their posteriors are broad (add spectroscopy or emission
lines to pin them down).

---

## Fitting [α/Fe] — no FSPS required

The α-enhanced grids (FSPS v4.0, aMIST isochrones + C3K spectra,
[α/Fe] ∈ {−0.2, 0.0, +0.2, +0.4, +0.6}) add α-element enhancement as a
sampled stellar axis: a chemical clock for the formation timescale,
measured jointly with — and physically degenerate with — the total
metallicity. Building these grids yourself requires python-fsps compiled
from source with `AFE_FLAG=1`, so the canonical grid is published on
Zenodo and fetched by name. Because the α-enhanced variant carries **no
nebular model** (no α-enhanced CLOUDY tables exist), nothing is read from
`$SPS_HOME` at fit time either: **the downloaded grid is the complete
stellar input, and no FSPS install is needed at all.**

```python
import jax.numpy as jnp
from ceridwen.ssps import fetch_grid, SSPDataAfe
from ceridwen.csp import CSPBasis_afe

# One call: downloads once into ~/.ceridwen/grids, sha256-verified.
ssp = SSPDataAfe.load(fetch_grid("amist_c3k_lr_chab_afe"))
ssp.display()                       # (n_afe, n_Z, n_age, n_wave) = (5, 13, 107, 1936)

csp = CSPBasis_afe(ssp, lookback_time=jnp.linspace(0.0, 12.0, 9),
                   zh_const=True, verbose=False)

theta = {
    "lookback_time": jnp.linspace(0.0, 12.0, 9),
    "sfh":           jnp.exp(-jnp.linspace(0.0, 12.0, 9) / 1.0),
    "Z":             jnp.array([-1.9]),   # log10 TOTAL Z (absolute) — unchanged
    "afe":           jnp.array([0.4]),    # [α/Fe]: re-partitions that Z
    "tau_pow":           jnp.array([0.3]),
    "diffuse_tau_kc":    jnp.array([0.2]),
    "diffuse_dust_index": jnp.array([0.0]),
}
wave, fnu = csp.wave, csp.get_spectrum(theta)   # rest-frame Lsun/Hz per Msun
```

Notes: `theta["afe"]` is interpolated differentiably between the two
bracketing grid planes, so it works under `jit`/`grad`/`vmap` and in every
sampler; `Z` stays the total metal mass fraction ([Fe/H] becomes a derived
quantity); `CSPBasis_afe` accepts **only** α-aware 4-D grids — passing a
legacy 3-D grid raises a `TypeError` telling you to use `CSPBasis`;
emission-line observations are rejected (continuum and photometry only)
until α-enhanced photoionisation grids exist.

Two α-enhanced grids are available. `amist_c3k_lr_chab_afe` is the
**low-resolution** C3K grid (1936 λ points, Chabrier IMF) built from
`AFE_FLAG=1` python-fsps and used for the method-paper mock suite.
`amist_c3k_hr_krou_afe` is the **high-resolution** twin (10992 λ points,
R up to ~65000 in the optical, Kroupa IMF), built from the alpha-MC C3K
high-res SSPs (MIST v2.5 + C3K v2.3) that are too large to ship inside
FSPS/python-FSPS. Both share the *same* `(afe, [Fe/H], age)` node grid and
the same `log10 Z` axis (Z = 0.0185·10^[Fe/H]), so they are drop-in
interchangeable — only the spectral resolution and the IMF differ (mind the
Chabrier↔Kroupa mass-normalisation offset when comparing masses across the
two). The high-res grid is rebuilt from the provider's FITS with
[`scripts_afe/build_afe_hr_grid.py`](scripts_afe/build_afe_hr_grid.py) and
published on Zenodo; fetch it by name exactly as above with
`fetch_grid("amist_c3k_hr_krou_afe")`.

## Troubleshooting

- **Run `python -m ceridwen.check` first.** It reports missing dependencies, an
  unset or wrong `$SPS_HOME`, a too-old `sedpy-jax`, and whether nested sampling
  is available, each with the fix.
- **Install needs Python 3.11** (see Installation); the pinned `blackjax`
  requires it.
- **Common scientific pitfalls** (the metallicity-units trap, silently-ignored
  `theta` typos, the lookback-time convention) are documented in
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

- **Hoffman et al. 2019**, *NeuTra-lizing Bad Geometry in HMC Using Neural Transport*, [arXiv:1903.03704](https://arxiv.org/abs/1903.03704) (VI-preconditioned NUTS, `ceridwen.sampler.vi`)
- **Madau 1995**, ApJ 441, 18 (IGM transmission, `ceridwen.igm.Madau1995`)
- **Planck Collaboration 2020**, A&A 641, A6 (default cosmology, `ceridwen.cosmology`)
- **Kriek & Conroy 2013**, ApJ 775, L16 (diffuse dust attenuation shape, `ceridwen.dust`)
- **Conroy, Gunn & White 2009** (FSPS, upstream SSP provider)

---

## Related projects

- [sedpy_jax](https://github.com/Espe13/sedpy_jax): a JAX-compatible rewrite of [sedpy](https://github.com/bd-j/sedpy) by Benjamin D. Johnson; used by ceridwen for filter convolutions and smoothing.

---

Maintainer: [Amanda Stoffers](https://www.amanda-stoffers.de), Kavli Institute for Cosmology, University of Cambridge, `aas208@cam.ac.uk`
