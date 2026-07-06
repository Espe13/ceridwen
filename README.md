

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

git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
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
> but is not yet in a tagged PyPI release. ceridwen therefore pins blackjax's
> `main` branch for now. Because this is a direct git dependency, ceridwen is
> installed from source/GitHub rather than PyPI; once a blackjax release ships
> NSS, this becomes a normal `blackjax>=X.Y` pin and ceridwen installs straight
> from PyPI. (Python stays ≥ 3.11 either way.)

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

Once FSPS is set up, see [`examples/quickstart.py`](examples/quickstart.py) for a
complete, runnable fit (mock UV-to-IR photometry, end to end).

---

## Quick start

### Run the bundled example first

The fastest way to confirm your whole setup works end to end. It builds the SSP
cache from FSPS, generates mock UV-to-IR photometry, fits it with nested sampling,
and prints recovered-vs-true parameters plus a corner plot
(`examples/quickstart_corner.png`):

```bash
export SPS_HOME=/path/to/fsps        # your FSPS data directory
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
with FSPS.  On first use you need to generate it.  This takes a few
minutes on CPU and only has to be done once:

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

This is a **template**, not a copy-paste-runnable script: the flux/uncertainty
arrays (and `my_observed_flux` / `my_observed_sigma`) are placeholders for your
own data. For a script that runs as-is, use `examples/quickstart.py` above.

```python
import jax, jax.numpy as jnp
from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry, Spectrum, Lines
from ceridwen.priors import Uniform, ClippedNormal, StudentT

# Load the cached grid produced in Step 0.
ssp = SSPData.load("ssp_data.h5")

# Composite-stellar-population forward model.
# sps_home defaults to the $SPS_HOME environment variable, so you can omit it
# if that is set (recommended); pass it explicitly to override.
csp = CSPBasis(
    ssp,
    add_dust=True, add_diffuse_dust=True,
    add_neb=True, add_igm=True,            # IGM (Madau 1995) auto-scales with zred
)

# Observations.  Any combination of the three container classes can be
# fit jointly; the likelihood is a simple sum over their chi-squared
# contributions.

# (a) Broadband photometry, e.g. JWST/NIRCam.  Fluxes in AB maggies.
phot = Photometry(
    filters=["jwst_f115w", "jwst_f200w", "jwst_f444w"],
    flux=[1.2e-8, 2.7e-8, 3.1e-8],
    uncertainty=[6e-10, 1.4e-9, 1.5e-9],
    name="my_phot",
)

# (b) Spectroscopy, e.g. a NIRSpec medium-resolution spectrum.  Pass the
# rest-frame vacuum wavelength grid; set ``resolution`` (km/s) and
# ``smoothtype`` if you want the forward model to apply instrumental
# broadening.  Optional: ``response`` = per-pixel multiplicative
# flux-calibration vector.
import numpy as np   # Spectrum already imported above

wave = np.linspace(3600.0, 9000.0, 1024)     # Å, vacuum rest-frame
spec = Spectrum(
    wavelength=wave,
    flux=my_observed_flux,                   # F_nu per pixel, same units as model
    uncertainty=my_observed_sigma,
    resolution=150.0,                        # km/s
    smoothtype="vel",                        # or "R", "lambda", "lsf"
    name="my_spec",
)

# (c) Nebular emission-line fluxes.
# ``line_ind`` are 1-based indices into FSPS's ``emlines_info.dat``;
# ``wavelength`` is the vacuum rest wavelength in Å.  (Lines imported above.)

LINE_IND   = [59, 62, 63, 71, 72]                       # Hβ, [OIII]4959, [OIII]5007, Hα, [NII]6583
LINE_NAMES = ["Hbeta", "[OIII]4959", "[OIII]5007", "Halpha", "[NII]6583"]
LINE_WAVE  = [4861.3, 4958.9, 5006.8, 6562.8, 6583.4]

lines = Lines(
    line_ind=LINE_IND,
    line_names=LINE_NAMES,
    wavelength=LINE_WAVE,
    flux=[1.4e-18, 1.1e-18, 3.2e-18, 4.2e-18, 1.3e-18],  # erg s^-1 cm^-2
    uncertainty=[1.4e-19, 1.1e-19, 3.2e-19, 4.2e-19, 1.3e-19],
    name="my_lines",
)

# Collect whichever observations you have into a single list; an empty
# list is fine for any type that is not being fit.
observations = [phot, spec, lines]

# SedModel at fixed spectroscopic redshift.
model = SedModel(
    csp, observations=observations,
    priors={
        # Z is log10 of ABSOLUTE metallicity (= ssp_lgmet), NOT log10(Z/Zsun).
        # Keep this inside the FSPS grid (≈ [-4, -1.4]; solar ≈ -1.85) — values
        # outside it are silently clamped. Run `csp.check_param_ranges(...)` or
        # `python -m ceridwen.check` if unsure of your grid bounds.
        "Z": ClippedNormal(mean=-2.0, sigma=0.5, low=-4.0, high=-1.4),
        "logmass": Uniform(low=6.0, high=12.5),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "tau_pow": ClippedNormal(mean=0.3, sigma=0.5, low=0.0, high=4.0),
        "alpha_pow": ClippedNormal(mean=-1.0, sigma=0.5, low=-2.5, high=0.5),
        "gas_logz": Uniform(low=-2.0, high=0.5),
        "gas_logu": Uniform(low=-4.0, high=-1.0),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
    },
    zred=6.5,                                # fixed spec-z
)

# VI-preconditioned NUTS.  VI learns a full-rank Gaussian approximation
# (~30 s); NUTS samples in the whitened space with target_acceptance = 0.95
# and a ~200-step step-size-only warmup.
result = fitSED(
    model, observations=observations,
    sampler="nuts",
    vi="tril",                               # "iaf" for NeuTra neural transport
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
```

The `result` object has posterior samples keyed by parameter name, plus the VI trace and per-phase wall-clock timings in `result.raw`.

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

