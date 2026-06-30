# Quick start

!!! tip "Run the bundled example first"
    The fastest way to confirm your whole setup works end to end. It builds the
    SSP cache from FSPS, generates mock UV-to-IR photometry, fits it with nested
    sampling, and writes a corner plot and a model-vs-data SED figure:

    ```bash
    export SPS_HOME=/path/to/fsps
    python examples/quickstart.py
    ```

    The two steps below are what the example does internally, shown so you can
    adapt them to your own observations.

## Step 0 — build the SSP grid (once per FSPS configuration)

CERIDWEN's forward model consumes an HDF5 cache of SSP spectra precomputed with
FSPS. Build it once (a few minutes on CPU); subsequent runs reload it.

```python
from ceridwen import SSPData

ssp = SSPData.from_fsps(imf_type=1, save_to="ssp_data.h5")
# later: ssp = SSPData.load("ssp_data.h5")
```

## Step 1 — build a model and fit

```python
import jax, jax.numpy as jnp
from ceridwen import SSPData, CSPBasis, SedModel, fitSED
from ceridwen.observation import Photometry
from ceridwen.priors import Uniform, ClippedNormal, StudentT

ssp = SSPData.load("ssp_data.h5")

# Composite-stellar-population forward model. sps_home defaults to $SPS_HOME.
csp = CSPBasis(ssp, add_dust=True, add_diffuse_dust=True, add_neb=True, add_igm=True)

# Observations (any combination of Photometry / Spectrum / Lines can be fit jointly).
phot = Photometry(
    filters=["jwst_f115w", "jwst_f200w", "jwst_f444w"],
    flux=[1.2e-8, 2.7e-8, 3.1e-8],          # AB maggies
    uncertainty=[6e-10, 1.4e-9, 1.5e-9],
    name="phot",
)

model = SedModel(
    csp, observations=[phot],
    priors={
        # Z is log10 ABSOLUTE metallicity — keep inside the FSPS grid (~[-4, -1.4]).
        "Z": ClippedNormal(mean=-2.0, sigma=0.5, low=-4.0, high=-1.4),
        "logmass": Uniform(low=6.0, high=12.5),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
    },
    zred=6.5,                                # fixed spec-z
)

result = fitSED(
    model, observations=[phot],
    sampler="nuts", vi="tril",
    sampler_kwargs={"num_chains": 4, "num_samples": 2000},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
```

`result` carries posterior samples keyed by parameter name, plus the VI trace and
per-phase timings. For nested sampling, `result.to_anesthetic()` gives an
[anesthetic](https://anesthetic.readthedocs.io) `NestedSamples` object for
evidence, corner plots, and posterior summaries.

!!! warning "Read the conventions first"
    The metallicity units and the lookback-time indexing are the two things most
    likely to bite — see **[Conventions & gotchas](conventions.md)** before
    fitting real data.

See `examples/quickstart.py` for a complete, runnable script (it also produces a
truth-overlaid corner plot and a model-vs-data SED with a χ residual panel).
