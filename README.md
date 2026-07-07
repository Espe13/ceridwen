

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

**Use Python 3.11 or newer.** Nested sampling depends on the official `blackjax`
(its merged NSS), which requires Python ≥ 3.11; `pip install` will refuse 3.10.
A fresh conda env is the easy route:

```bash
conda create -n ceridwen python=3.11 -y
conda activate ceridwen

git lfs install    # needs git-lfs: brew install git-lfs / apt-get install git-lfs
git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
git lfs pull       # fetches the bundled SSP test grid (~120 MB; used by the
                   # quickstart fallback and the test suite — or download a
                   # grid from Zenodo instead, see docs/installation.md)
pip install .
```

Use a plain `pip install .` (not `-e`): an editable install tracks your working
copy, so your version would change as the repo moves. To pin a specific release
instead, install a tag directly (no manual clone needed):

```bash
pip install "git+https://github.com/Espe13/ceridwen.git@v0.1.0"
```

(If you intend to modify ceridwen's source, then `pip install -e .` from a clone
is the developer workflow.)

After installing, verify your setup (deps, FSPS, `$SPS_HOME`, nested-sampling
support) before your first fit:

```bash
python -m ceridwen.check
```

It prints an `ok` / `warn` / `FAIL` line per component with the exact fix for
anything missing.

This installs everything needed to `import ceridwen`, build the forward model,
and run NUTS / VI / nested sampling **including posterior plotting** — `jax`,
`jaxlib`, `numpy`, `scipy`, `matplotlib`, `h5py`, `astropy`, `sedpy-jax`,
`tensorflow-probability`, `blackjax`, `tqdm`, `optax` (VI), and `anesthetic`
(nested-sampling posteriors + corner plots) are all pulled in automatically.
The only thing not installed for you is FSPS (it can't be — see below).

> **Note on `blackjax`.** Nested sampling uses `blackjax.nss`, which has been
> merged into the [official blackjax](https://github.com/blackjax-devs/blackjax)
> but is not yet in a tagged PyPI release. ceridwen therefore pins a fixed
> blackjax commit (`f73e12956`), so every install gets the same validated
> state. Because this is a direct git dependency, ceridwen is installed from
> source/GitHub rather than PyPI; once a blackjax release ships NSS, this
> becomes a normal `blackjax>=X.Y` pin and ceridwen installs straight from
> PyPI. (Python stays ≥ 3.11 either way.)

There are **no extras to choose** — `pip install .` gives you everything to
import, fit, plot, and test CERIDWEN. The only thing installed separately is
**FSPS**, which can't be a normal Python dependency (it compiles Fortran); see
[Installing FSPS](#installing-fsps-and-setting-sps_home) below.

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

Open a **new** terminal and run `echo $SPS_HOME` — it should print the path. If
it's blank, the line went into the wrong file (check which shell you use with
`echo $SHELL`). Then confirm the whole setup with:

```bash
python -m ceridwen.check
```

FSPS is the recommended route to an SSP grid (you control isochrones, spectral
library, and IMF) and is required for nebular / dust emission — but the bundled
example runs without it (see below). Either way, next stop:
[`examples/quickstart.py`](examples/quickstart.py), a complete runnable fit.

---

## Quick start

### Run the bundled example first

The fastest way to confirm your whole setup works end to end. It loads an SSP
grid — building it from FSPS only if none is found (`$SSP_FILE` →
`examples/ssp_data.h5` → the LFS test fixture; see
[`docs/installation.md`](docs/installation.md), "Getting the SSP grid") —
generates mock UV-to-IR photometry, fits it with nested sampling,
and prints recovered-vs-true parameters plus a corner plot
(`examples/quickstart_corner.png`):

```bash
python examples/quickstart.py
```

If that runs and prints a recovered-vs-true table, your setup works and you're
ready to fit real data — read on. Expect `logmass` to land near the injected
truth; `Z` and the dust parameters are only weakly constrained by broadband
photometry alone, so their posteriors are broad and can sit ~1 dex off truth.
That is expected, not a broken install — add spectroscopy or emission lines to
pin them down. The two steps below are what the example does internally, shown
so you can adapt them to your own observations.

### Step 0 — build the SSP grid (once per FSPS configuration)

Ceridwen's forward model consumes an HDF5 cache of SSP spectra precomputed
with FSPS. Building your own is the recommended route (you control the
isochrones, spectral library, and IMF); it takes a few minutes on CPU and only
has to be done once. No FSPS? Download a pre-built grid instead
(Zenodo/git-LFS options in [`docs/installation.md`](docs/installation.md)).

```python
from ceridwen import SSPData

# Generate + cache. from_fsps accepts ONLY the kwargs that define the stellar
# library / IMF (imf_type and its parameters, isochrone-phase knobs like
# tpagb_norm_type). Anything the forward model applies itself — dust, SFH,
# nebular emission, IGM, redshift, or a fixed metallicity — is rejected with a
# clear error, so the grid can never be silently double-processed.
ssp = SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")

# Subsequent runs just reload the cache:
# ssp = SSPData.load("ssp_data.h5")
```

The grid records its own **provenance** (isochrone/spectral library, `imf_type`,
FSPS version, the exact build kwargs, wavelength range) into the HDF5 file, and
`CSPBasis` reads the isochrone library back automatically — **you never set
`isoc_type` by hand**, and the nebular CLOUDY grid is guaranteed to match the
SSP isochrone set. (Old grids built before provenance tracking still load; for
those, `CSPBasis` warns that the isochrone type is unknown and falls back to
`'mist'`.)

`from_fsps(**fsps_kwargs)` is a thin classmethod wrapper around
`ceridwen.ssps.ssp_data.collect_ssp_data_wrapper` — either call works.
FSPS must be installed and importable; see the Installation section.

### Step 1 — fit a galaxy end-to-end

This is a real, copy-paste-runnable joint fit of broadband photometry plus an
optical spectrum. The data live in `examples/mock_galaxy.npz` — a mock galaxy
at fixed z = 0.1 generated with the same forward model
(`examples/make_mock_data.py` shows exactly how, and regenerates it). The
injected truth is stored in the file, so you can check the fit recovers it.

```python
import pathlib
import jax, jax.numpy as jnp
import numpy as np
from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry, Spectrum
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

# Load the mock observations (and the injected truth) shipped with the repo.
d = np.load(pathlib.Path("examples") / "mock_galaxy.npz")
ZRED = float(d["zred"])                      # fixed spectroscopic redshift

# Load the SSP grid from Step 0 (or the Zenodo download — see Installation).
ssp = SSPData.load("examples/ssp_data.h5")

# Composite-stellar-population forward model. lookback_time is the static SFH
# node grid (Gyr, increasing, index 0 = today, >= 2 nodes); it comes with the
# mock file so model and data match. (For control over the initial parameter
# values, pass a full theta= dict instead — see the CSPBasis docstring.)
lookback = jnp.asarray(d["lookback_time"])   # 6 nodes -> 5 free logsfr_ratios
csp = CSPBasis(
    ssp,
    lookback_time=lookback,
    zh_const=True, sfh_interp="step",
    add_dust=False, add_diffuse_dust=True,
    add_neb=False,        # nebular emission needs FSPS data ($SPS_HOME); see below
    verbose=False,
)

# Observations. Any combination of Photometry / Spectrum / Lines containers
# is fit jointly; the likelihood is a sum over their contributions.

# (a) Broadband photometry: fluxes in AB maggies (1 maggie = 3631 Jy).
phot = Photometry(
    filters=[str(f) for f in d["filters"]],
    flux=d["maggies"], uncertainty=d["maggies_unc"],
    name="phot",
)

# (b) Spectroscopy. The wavelength grid is vacuum Å in the OBSERVED frame
# (as delivered by the instrument): the forward model redshifts the model
# spectrum by (1 + zred) and projects it onto these pixels. ``resolution`` +
# ``smoothtype`` apply instrumental broadening ("vel": sigma in km/s;
# also "R", "lambda", "lsf"). Flux units must match the model spectrum
# (L_sun Hz^-1 after mass scaling) up to a calibration you supply.
spec = Spectrum(
    wavelength=d["spec_wave_obs"],
    flux=d["spec_flux"], uncertainty=d["spec_unc"],
    resolution=float(d["spec_resolution"]), smoothtype="vel",
    name="spec",
)

# The SFH is sampled as logsfr_ratios (Prospector convention) and transformed
# to per-node SFR; logmass then sets the absolute amplitude.
sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

model = SedModel(
    csp, observations=[phot, spec],
    priors={
        # Z is log10 of ABSOLUTE metallicity (= ssp_lgmet), NOT log10(Z/Zsun);
        # solar is ~ -1.85. Keep it inside your SSP grid — values outside are
        # silently clamped. Run `csp.check_param_ranges(...)` for the bounds.
        "Z": Uniform(low=-3.9, high=-1.45),
        "logmass": Uniform(low=9.0, high=12.0),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
    },
    transforms={"sfh": logsfr_to_sfh},
    free_param_init={"logsfr_ratios": jnp.zeros(lookback.size - 1),
                     "logmass": jnp.array([10.0])},
    zred=ZRED,                               # fixed spec-z
)

# VI-preconditioned NUTS: VI learns a full-rank Gaussian transport map,
# NUTS then samples in the whitened space (Hoffman et al. 2019).
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
    print(f"{p:>20}: true {float(d['true_' + p][0]):+7.3f}   "
          f"fit {np.median(samples):+7.3f} "
          f"+/- {np.std(samples):.3f}")
```

The `result` object has posterior samples keyed by parameter name, plus the VI
trace and per-phase wall-clock timings in `result.raw`; everything is also
written to `./my_fit/ceridwen_result.h5`.

To add **nebular emission lines** to the fit (a `Lines` container, joint with
photometry and spectroscopy, `add_neb=True`), you need the FSPS data files
(`$SPS_HOME`) for the CLOUDY grids — the walk-through in
[`docs/tutorial.md`](docs/tutorial.md) covers exactly that, including line
selection via FSPS's `emlines_info.dat` (rest-frame vacuum Å — unlike the
spectrum's pixel grid, line wavelengths are given in the REST frame and
redshifted internally).

---

## Troubleshooting

- **Run `python -m ceridwen.check` first.** It reports missing dependencies, an
  unset or wrong `$SPS_HOME`, a too-old `sedpy-jax`, and whether nested sampling
  is available — each with the fix.
- **Install needs Python 3.11 or newer** (see Installation); 3.10 is refused
  because nested sampling pins the official `blackjax` (its merged NSS), which
  requires Python ≥ 3.11.
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

