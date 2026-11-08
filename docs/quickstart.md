# Quick start

This example fits a mock galaxy observed in twelve bands from GALEX to WISE
and an optical spectrum. The mock uses the same forward model as the fit,
so the fitted parameters can be compared with their input values. After
[installation](installation.md), run the blocks in order in one Python session.

**1. Download an SSP grid.** The MIST + MILES grid uses a Chabrier IMF and is
67 MB. `fetch_grid` caches it in `~/.ceridwen/grids` (or `$CERIDWEN_GRID_DIR`)
and checks its SHA-256 on every call. Use `available_grids()` to list other grids.

```python
from ceridwen import SSPData
from ceridwen.ssps import fetch_grid

ssp = SSPData.load(fetch_grid("mist_miles_chab"))
ssp.display()                                      # library, IMF, grid coverage
```

**2. Build the model.** `lookback_time` contains SFH nodes in Gyr, increasing
from 0 at the present; with `sfh_interp="step"` the SFR is constant between
adjacent nodes, one value per bin. A transform converts the sampled
`logsfr_ratios` (log SFR ratios of adjacent bins, Prospector's continuity
SFH) into those bin SFRs: 7 nodes give 6 bins and 5 ratios. `logzsol` is log10(Z/Z_sun) for the loaded grid.

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

csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 7),
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
            "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
        },
        transforms={"sfh": logsfr_to_sfh},
        free_param_init={"logsfr_ratios": jnp.zeros(5), "logmass": jnp.array([10.0])},
        zred=ZRED,
    )
```

By default, `SedModel` uses a galaxy velocity dispersion of 300 km/s. Change it
with `kinematics=Kinematics(sigma_gal=...)`, or fit it by using
`Kinematics(sigma_gal="sigma_gal")` and adding a prior. See
[Conventions](conventions.md).

**3. Generate mock data.** Evaluate the model at known parameters and add noise,
with signal-to-noise ratios of 20 for photometry and 25 for the spectrum.

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

Before fitting, use `model.display()` to inspect the parameters, priors,
transforms and observations. The figure also compares the data with the range
of model predictions allowed by the priors. `model.latex()` returns a LaTeX
table of free, derived and fixed parameters, with their priors or values.
The table requires `\usepackage{booktabs,amsmath}`.

```python
fig, ax = model.display(return_fig=True)     # in a notebook, model.display() alone shows it
fig.savefig("model.png", dpi=150)
print(model.latex())
```

In a script, a bare `model.display()` opens a window and waits until you close it.

**4. Fit with nested sampling.** These settings are intended for a laptop.
On an Apple Silicon Mac with macOS 26 and Python 3.12, the fit uses the GPU
in float32 and takes about five minutes. Float64 CPU runs take somewhat longer;
set `JAX_PLATFORMS=cpu` to use that mode on a Mac.

The example uses 60 inner slice steps. With the `fitSED` default of 30, the
float32 GPU run gave a `logmass` posterior about 30% too narrow. The fit prints
progress and saves `./my_fit/ceridwen_result.h5`. For scientific use, increase
`num_live` from this example's value; the `fitSED` default is 500.
See [Samplers](samplers.md) for NUTS.

```python
from ceridwen import fitSED

result = fitSED(
    model,
    sampler="nested",
    sampler_kwargs={"num_live": 100, "num_delete": 20, "num_inner_steps": 60,
                    "logZ_tol": -5.0},
    rng_key=jax.random.PRNGKey(42),
    output_dir="./my_fit",
)
print(f"log Z = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")
```

**5. Inspect the result.** `PostProcess` resamples the posterior to equal weight,
evaluates the fitted model for those draws and computes derived quantities.

```python
from ceridwen import PostProcess

pp  = PostProcess(model, result, n_samples=1000)
out = pp.run()

truth = {p: float(TRUTH[p][0]) for p in ("logzsol", "logmass", "diffuse_tau_kc", "diffuse_dust_index")}
for p, t in truth.items():
    lo, med, hi = np.percentile(out["theta"][p], [16, 50, 84])
    print(f"{p:>20}: true {t:+7.3f}   fit {med:+7.3f}  (-{med - lo:.3f}/+{hi - med:.3f})")

print(np.percentile(out["extras"]["sfh"]["sfr100"], [16, 50, 84]))   # SFR over 100 Myr [Msun/yr]
pp.figures("./my_fit/figures", title="mock galaxy",
           truths={**truth, "logsfr_ratios": np.asarray(TRUTH["logsfr_ratios"])})
pp.save("./my_fit/post.npz")
```

The table compares each posterior median and 16–84% interval with the input.
Some recovered values differ from the input by 2–3 sigma. In this example,
`logmass` is degenerate with the five-ratio SFH: mass in the oldest bins adds
little light. A flat SFH with `logmass=10.44` fits within 1 in ln L of the
input, even without noise. The SFH prior favours that solution, giving a MAP
of 10.44 rather than the input 10.50. On noise-free data with flat priors,
the MAP is 10.498.

The 100-live-point runs also underestimate uncertainty. On the Apple GPU,
this seed gives `logmass=10.43` (−0.03/+0.03), and seed 0 gives 10.44
(−0.03/+0.03). The float64 CPU run gives 10.44 (−0.02/+0.03). Increase
`num_live` before using these intervals. The five SFH ratios are mostly
prior-dominated, and one or two may fall outside their 95% intervals.

`figures` writes `summary.pdf` (SED, residuals, SFH and marginals), `corner.pdf`
and `diagnostics.pdf`. `post.npz` stores predictions for every draw and is
about 100 MB; reload it with `load_postprocess`. See
[Post-processing](postprocessing.md) for the output fields.


## Reloading a fit

`load_result_h5("my_fit/ceridwen_result.h5")`, imported from `ceridwen`, reloads
the result in a later session. Reconstruct the model with the same grid, CSP,
observations and transforms. `ceridwen.resultfile.rebuild_model(path, csp,
observations, transforms=...)` reads the priors, free parameters, redshift and
kinematics from the file and checks the reconstructed model against the saved
configuration. `check_model_against_result(model, path)` lists differences.
You can also pass the file path directly to
`PostProcess(model, "my_fit/ceridwen_result.h5")`.

## Units

`Photometry` predictions use AB maggies. `Spectrum` predictions use observed-frame
F_nu in erg s⁻¹ cm⁻² Hz⁻¹; multiply by 1e32 to convert to nJy. `Lines`
predictions are integrated fluxes in erg s⁻¹ cm⁻². Set `zred=` or
`lumdist_mpc=` on `SedModel` to obtain these fluxes. At `zred=0` without a
luminosity distance, the model applies no flux factor and issues a warning.
See [Conventions](conventions.md) for details and the [Tutorial](tutorial.md)
for emission lines and nebular emission (`add_neb=True`).
