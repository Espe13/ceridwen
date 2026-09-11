# Quick start

!!! tip "Run the bundled example first"
    The fastest way to confirm your whole setup works end to end. It loads an
    SSP grid (building it from FSPS only if none is found; see
    [Installation: Getting the SSP grid](installation.md#getting-the-ssp-grid)),
    generates mock UV-to-IR photometry, fits it with nested sampling, and
    post-processes the fit with `PostProcess`, writing the summary, corner and
    sampling-diagnostic figures to `examples/quickstart_figures/`:

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
from ceridwen import SSPData, CSPBasis, SedModel, fitSED, Kinematics, Cosmology
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT

ssp = SSPData.load("ssp_data.h5")

# Cosmology: set once, here, on the object that computes distances and ages.
# Cosmology.planck18() / planck15() / wmap9() / flat(H0, Om0) / from_name(...)
# / from_astropy(...).  It is printed by every summary and stored in the result.
cosmo = Cosmology.planck18()

# Composite-stellar-population forward model. lookback_time is the static SFH
# node grid (Gyr, increasing, index 0 = today-at-z, >= 2 nodes); its oldest
# node must not exceed the age of the universe at the fit redshift,
# cosmo.age(6.5) = 0.84 Gyr here (SedModel refuses a grid that does).
# sps_home defaults to $SPS_HOME (needed because add_neb=True).
lookback = jnp.linspace(0.0, 0.8, 6)         # 6 nodes -> 5 free logsfr_ratios
csp = CSPBasis(
    ssp,
    lookback_time=lookback,
    cosmo=cosmo,
    zh_const=True, sfh_interp="step",
    add_dust=False, add_diffuse_dust=True, add_neb=True, add_igm=True,
)   # add_dust=True adds the birth-cloud parameters (tau_pow, alpha_pow): give them priors too

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
    # kinematics=Kinematics(sigma_gal=300.0) is the default: the galaxy's
    # velocity dispersion, applied to the spectrum the filters integrate. For
    # broad bands any value changes the photometry by < 5e-4 mag; pass
    # Kinematics(sigma_gal="sigma_gal") plus a prior to sample it instead.
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
per-phase timings (nested sampling also returns `result.log_evidence`).

Predictions and data are in physical units because `zred=6.5` is fixed: AB
maggies for photometry, cgs F_nu (erg s^-1 cm^-2 Hz^-1) for spectra,
erg s^-1 cm^-2 for lines. Omitting `zred` (and `lumdist_mpc`) leaves the model
at `zred = 0`, where no flux factor is applied and `SedModel` warns; see
[Conventions](conventions.md).

## Step 2: post-process

`PostProcess` resamples the draws to equal weight (nested-sampling weights are
recomputed from the birth contours), pushes them through the fitted forward
model, and writes three figures per galaxy:

```python
from ceridwen import PostProcess

pp  = PostProcess(model, result, n_samples=2000)
out = pp.run()
print(np.percentile(out["theta"]["logmass"], [16, 50, 84]))
out["extras"]["sfh"]["sfr10"]              # derived quantities, see postprocessing.md
pp.figures("./my_fit/figures", title="my galaxy")   # summary.pdf, corner.pdf, diagnostics.pdf
pp.save("./my_fit/post.npz")
```

`truths={name: value}` marks injected values in the summary and corner
figures of a mock test. The full output layout and the individual figure
functions are described in [Post-processing](postprocessing.md).

!!! warning "Read the conventions first"
    The metallicity units and the lookback-time indexing are the two things most
    likely to bite. See **[Conventions & gotchas](conventions.md)** before
    fitting real data.

See `examples/quickstart.py` for a complete, runnable script (it post-processes
the fit and writes the truth-overlaid summary, corner and diagnostic figures).
