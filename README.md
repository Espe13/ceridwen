![Ceridwen Logo](CeridwenLogo.png)

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

```bash
git clone https://github.com/Espe13/ceridwen.git
cd ceridwen
pip install -e .
```

Required: Python ≥ 3.10, `jax`, `jaxlib`, `numpy`, `scipy`, `matplotlib`. Strongly recommended: `h5py`, `astropy`, `blackjax` (for NUTS / nested sampling), `optax` (for VI training).

External dependencies outside PyPI:
- [FSPS](https://github.com/cconroy20/fsps) with the `python-fsps` wrapper, for building SSP grids
- [sedpy_jax](https://github.com/Espe13/sedpy_jax) for filter convolutions

---

## Quick start

### Step 0 — build the SSP grid (once per FSPS configuration)

Ceridwen's forward model consumes an HDF5 cache of SSP spectra precomputed
with FSPS.  On first use you need to generate it.  This takes a few
minutes on CPU and only has to be done once:

```python
from ceridwen.ssps.ssp_data import SSPData

# Generate + cache.  Any FSPS params (imf_type, dust_type, nebular grid
# choices, ...) can be passed as kwargs.  The example below matches the
# defaults used throughout the quick-start below.
ssp = SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")

# Subsequent runs just reload the cache:
# ssp = SSPData.load("ssp_data.h5")
```

`from_fsps(**fsps_kwargs)` is a thin classmethod wrapper around
`ceridwen.ssps.ssp_data.collect_ssp_data_wrapper` — either call works.
FSPS must be installed and importable; see the Installation section.

### Step 1 — fit a galaxy end-to-end

```python
import jax, jax.numpy as jnp
from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.observation.observation import Photometry
from ceridwen.model.model import SedModel
from ceridwen.sampler import Uniform, ClippedNormal, StudentT
from ceridwen.fit import fitSED

# Load the cached grid produced in Step 0.
ssp = SSPData.load("ssp_data.h5")

# Composite-stellar-population forward model.
csp = CSPBasis(
    ssp,
    add_dust=True, add_diffuse_dust=True,
    add_neb=True, add_igm=True,            # IGM (Madau 1995) auto-scales with zred
    sps_home="/path/to/fsps",
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
from ceridwen.observation.observation import Spectrum
import numpy as np

wave = np.linspace(3600.0, 9000.0, 1024)     # Å, vacuum rest-frame
spec = Spectrum(
    wavelength=wave,
    flux=my_observed_flux,                   # F_nu per pixel, same units as model
    uncertainty=my_observed_sigma,
    resolution=150.0,                        # km/s
    smoothtype="vel",                        # or "R", "lambda", "lsf"
    name="my_spec",
)

# (c) Nebular emission-line fluxes (used by fit_lines.py drivers).
# ``line_ind`` are 1-based indices into FSPS's ``emlines_info.dat``;
# ``wavelength`` is the vacuum rest wavelength in Å.
from ceridwen.observation.observation import Lines

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
        "Z": ClippedNormal(mean=-1.0, sigma=0.3, low=-2.0, high=0.19),
        "logmass": Uniform(low=6.0, high=12.5),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "tau_pow": ClippedNormal(mean=0.3, sigma=0.5, low=0.0, high=4.0),
        "alpha": ClippedNormal(mean=-1.0, sigma=0.5, low=-2.5, high=0.5),
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

Maintainer: Amanda Stoffers, Institute of Astronomy, University of Cambridge — `aas208@cam.ac.uk`
