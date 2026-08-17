# Quick start

!!! tip "Run the bundled example first"
    The fastest way to confirm your whole setup works end to end. It loads an
    SSP grid (building it from FSPS only if none is found; see
    [Installation: Getting the SSP grid](installation.md#getting-the-ssp-grid)),
    generates mock UV-to-IR photometry, fits it with nested sampling, and
    writes a corner plot and a model-vs-data SED figure:

    ```bash
    python examples/quickstart.py
    ```

    `logmass` should land near the injected truth. `Z` and the dust parameters
    are only weakly constrained by broadband photometry alone, so their
    posteriors are broad and can sit ~1 dex off truth. That is expected, not a
    broken install; add spectroscopy or emission lines to pin them down.

    The two steps below are what the example does internally, shown so you can
    adapt them to your own observations.

## Step 0: build the SSP grid (once per FSPS configuration)

CERIDWEN's forward model consumes an HDF5 cache of SSP spectra precomputed with
FSPS. Build it once (a few minutes on CPU); subsequent runs reload it.

```python
from ceridwen import SSPData

ssp = SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")
# later: ssp = SSPData.load("ssp_data.h5")
```

`from_fsps` accepts only stellar-library / IMF kwargs (`imf_type` and friends);
dust, SFH, nebular, IGM, redshift, or a fixed metallicity are rejected, because
the forward model owns those. The grid records its provenance (isochrone/spectral
library, `imf_type`, FSPS version, build kwargs) and `CSPBasis` picks up the
isochrone library automatically, so **`isoc_type` never has to be set by hand**
and the nebular grid always matches the SSP isochrones.

!!! tip "No FSPS? Download a published grid"
    The canonical grids are on Zenodo and registered in
    `ceridwen.ssps.grid_fetch`:
    `ssp = SSPData.load(fetch_grid("mist_miles_chab"))`. For
    [α/Fe] fitting with `CSPBasis_afe` the download is the *recommended*
    route — the α grids need a custom FSPS v4.0 build to generate, but
    none at all to fit, since the α variant has no nebular model. See
    [Installation: α-enhanced grids](installation.md#α-enhanced-grids-download-dont-build).

## Step 1: build a model and fit

A minimal photometry-only fit. For a full runnable joint
photometry + spectroscopy example that generates its own self-consistent mock,
see the README's "fit a galaxy end-to-end" section; for lines and nebular
emission see the [tutorial](tutorial.md).

```python
import jax, jax.numpy as jnp
import numpy as np
from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

ssp = SSPData.load("ssp_data.h5")

# Composite-stellar-population forward model. lookback_time is the static SFH
# node grid (Gyr, increasing, index 0 = today-at-z, >= 2 nodes); the oldest
# node must not exceed the age of the universe at the fit redshift (~0.85 Gyr
# at z = 6.5). sps_home defaults to $SPS_HOME (needed because add_neb=True).
lookback = jnp.linspace(0.0, 0.8, 6)         # 6 nodes -> 5 free logsfr_ratios
csp = CSPBasis(
    ssp,
    lookback_time=lookback,
    zh_const=True, sfh_interp="step",
    add_dust=True, add_diffuse_dust=True, add_neb=True, add_igm=True,
)

# Observations (any combination of Photometry / Spectrum / Lines, fit jointly).
phot = Photometry(
    filters=["jwst_f115w", "jwst_f200w", "jwst_f444w"],
    flux=[1.2e-8, 2.7e-8, 3.1e-8],          # AB maggies
    uncertainty=[6e-10, 1.4e-9, 1.5e-9],
    name="phot",
)
phot.display()   # sanity-check the photometry you just built

# The SFH is sampled as logsfr_ratios and transformed to per-node SFR.
sfh_times_yr = np.array(csp.sfh_times)
def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
    return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

model = SedModel(
    csp, observations=[phot],
    priors={
        # Z is log10 ABSOLUTE metallicity (solar ~ -1.85); keep inside your
        # SSP grid. Print the allowed range with
        #     print(float(csp.zmet.min()), float(csp.zmet.max()))
        # and call csp.check_param_ranges() to warn about out-of-grid values.
        "Z": ClippedNormal(mean=-2.0, sigma=0.5, low=-4.0, high=-1.4),
        "logmass": Uniform(low=6.0, high=12.5),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "gas_logz": Uniform(low=-2.0, high=0.5),
        "gas_logu": Uniform(low=-4.0, high=-1.0),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
    },
    transforms={"sfh": logsfr_to_sfh},
    free_param_init={"logsfr_ratios": jnp.zeros(5),
                     "logmass": jnp.array([10.0])},
    zred=6.5,                                # fixed spec-z
)

# Pick ONE sampler. Option A, VI-preconditioned NUTS:
result = fitSED(
    model,
    sampler="nuts", vi="tril",
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)

# Option B, nested sampling (gradient-free; also returns the evidence log Z):
# result = fitSED(
#     model,
#     sampler="ns",
#     sampler_kwargs={"num_live": 400, "num_delete": 80, "logZ_tol": -5.0},
#     rng_key=jax.random.PRNGKey(42),
#     output_dir="./my_fit",
# )
```

`result` carries posterior samples keyed by parameter name, plus the VI trace and
per-phase timings. For nested sampling, `result.to_anesthetic()` gives an
[anesthetic](https://anesthetic.readthedocs.io) `NestedSamples` object for
evidence, corner plots, and posterior summaries.

!!! warning "Read the conventions first"
    The metallicity units and the lookback-time indexing are the two things most
    likely to bite. See **[Conventions & gotchas](conventions.md)** before
    fitting real data.

See `examples/quickstart.py` for a complete, runnable script (it also produces a
truth-overlaid corner plot and a model-vs-data SED with a χ residual panel).
