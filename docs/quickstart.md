# Quick start

This fits a mock galaxy (twelve photometric bands from GALEX to WISE and an optical spectrum)
made with the same forward model, so you can compare the fit with the truth. Run the blocks
in order in one Python session, after the [installation](installation.md).

**1. Get an SSP grid.** The published MIST + MILES grid (Chabrier IMF) is downloaded once
(67 MB) into `~/.ceridwen/grids` (or `$CERIDWEN_GRID_DIR`), and its SHA-256 is checked on
every call. `available_grids()` lists the others.

```python
from ceridwen import SSPData
from ceridwen.ssps import fetch_grid

ssp = SSPData.load(fetch_grid("mist_miles_chab"))
ssp.display()                                      # library, IMF, grid coverage
```

**2. Build the model.** `lookback_time` holds the SFH nodes in Gyr, increasing from 0
(today). The SFH is sampled as `logsfr_ratios` and turned into a star formation rate per node
by a transform. `logzsol` is log10(Z/Z_sun) of the grid.

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
            "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
        },
        transforms={"sfh": logsfr_to_sfh},
        free_param_init={"logsfr_ratios": jnp.zeros(5), "logmass": jnp.array([10.0])},
        zred=ZRED,
    )
```

`SedModel` also applies a galaxy velocity dispersion of 300 km/s by default. Pass
`kinematics=Kinematics(sigma_gal=...)` to change it, or `Kinematics(sigma_gal="sigma_gal")`
with a prior to fit it (see [Conventions](conventions.md)).

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

**4. Fit with nested sampling.** These settings are sized for a laptop CPU: the fit took
**21-29 min** (67 800 likelihood calls) in two runs on an 11-core Apple M3 Pro shared with
other jobs. It prints its progress and writes `./my_fit/ceridwen_result.h5`. For production, raise `num_live` (fitSED's default
is 500) and lower `logZ_tol`; for NUTS, see [Samplers](samplers.md).

```python
from ceridwen import fitSED

result = fitSED(
    model,
    sampler="nested",
    sampler_kwargs={"num_live": 100, "num_delete": 20, "logZ_tol": -2.0},
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
pp.figures("./my_fit/figures", title="mock galaxy", truths=truth)
pp.save("./my_fit/post.npz")
```

The table compares the posterior median and 16-84% range with the truth. At these settings
the recovered values can sit 2-3 sigma from the truth: with a five-ratio SFH, `logmass`
trades against the SFH shape, and the MAP of this mock lies at logmass 10.44, not at 10.50
(the data and the degeneracy, not the sampler). `figures` writes `summary.pdf` (SED with
residuals, SFH, marginals), `corner.pdf` and `diagnostics.pdf` (the sampler's diagnostics).
`post.npz` holds every draw's predictions and is about 100 MB; `load_postprocess` reads it back.
Everything `PostProcess` returns is described in
[Post-processing](postprocessing.md).


## Reloading a fit

`load_result_h5("my_fit/ceridwen_result.h5")` (from `ceridwen`) returns the same result object
in a later session. The model is not stored as code: rebuild it with the same grid, CSP,
observations and transforms. `ceridwen.resultfile.rebuild_model(path, csp, observations,
transforms=...)` takes the priors, free parameters, redshift and kinematics from the file and
raises if the rebuilt model differs from the one recorded; `check_model_against_result(model,
path)` lists every difference. `PostProcess(model, "my_fit/ceridwen_result.h5")` also accepts
the file directly.

## Units

`Photometry` predictions are AB maggies, `Spectrum` predictions are observed-frame F_nu in
erg s⁻¹ cm⁻² Hz⁻¹ (multiply by 1e32 for nJy), and `Lines` predictions are integrated fluxes in
erg s⁻¹ cm⁻². All three need `zred=` (or `lumdist_mpc=`) on the `SedModel`: at `zred = 0`
without `lumdist_mpc` no flux factor is applied, and `SedModel` warns. See
[Conventions](conventions.md) before fitting real data, and the [Tutorial](tutorial.md) for
emission lines and nebular emission (`add_neb=True`).
