# Conventions & gotchas

A few conventions cause silent mistakes if you get them wrong. Read them before
fitting real data. The repository also ships a fuller misuse guide in
`GOTCHAS.md`.

## Metallicity is log10 of ABSOLUTE Z

The parameter `Z` (and `ssp_lgmet`) is `log10(Z)` in **absolute** units, not
`log10(Z/Z☉)`. Solar is roughly `-1.85` (Z☉ ≈ 0.014), not `0.0`.

The MIST grids span about `[-4.35, -1.35]` (`log10(0.0142) + logzsol`; print
`csp.zmet` for your grid). Values outside it are silently
clamped to the nearest grid edge, so a "solar" guess of `Z = 0.0` is off the
grid and gets clamped.

!!! danger "Common mistake"
    A prior like `Uniform(low=-2.5, high=0.2)` puts most of its mass off the
    grid. Use something like `ClippedNormal(mean=-2.0, sigma=0.5, low=-4.0, high=-1.4)`. If
    you are unsure of your grid bounds, print them with
    `print(float(csp.zmet.min()), float(csp.zmet.max()))`, or call
    `csp.check_param_ranges(theta)` to warn about out-of-grid values.

## Lookback time increases with index (index 0 = today)

`lookback_time` element 0 is the present; the last element is the oldest bin,
near the age of the universe. The `sfh` array is indexed the same way. The old
decreasing convention (`lookback = T_univ - t_grid`) is rejected at construction
with a `ValueError`, so do not reintroduce it, and do not reverse arrays "to be
safe".

## Units

| Quantity | Unit |
|---|---|
| Model wavelength grid (`csp.wave`) | Å, vacuum, rest frame |
| `Spectrum` pixel wavelengths (data) | Å, vacuum, **observed frame** (model is redshifted onto them) |
| `Lines` wavelengths / `mask_lines` centres | Å, vacuum, **rest frame** (redshifted internally) |
| `csp.get_spectrum(theta)` (model grid) | rest-frame L_ν in L☉ Hz⁻¹ per M☉ (no mass, distance or IGM) |
| `Spectrum` predictions (`model.predict`) | observed-frame F_ν in erg s⁻¹ cm⁻² Hz⁻¹ (cgs; × 1e32 for nJy), only with a redshift or `lumdist_mpc` in force |
| Broadband fluxes | AB maggies |
| Emission-line fluxes | erg s⁻¹ cm⁻² |
| Stellar mass | `logmass` = log10(M⋆/M☉) |

The forward model is evaluated at unit mass and scaled by `10**logmass`.

## SFH mass normalisation

The `logsfr_ratios_to_sfh` transform normalises the SFH so the trapezoidal
integral of SFR over the lookback grid equals 1 M☉, and `logmass` sets the
amplitude. Use the provided transform rather than hand-rolling it. Getting this
wrong biases `logmass` by many dex.

## `theta` is a dict, so typos are silently ignored

A mistyped key (`logmas` for `logmass`, `dust2` for `diffuse_tau_kc`, and so on)
is simply not read, so the parameter takes its default. `predict()` and
`get_spectrum_components()` emit a warning listing unrecognised keys at trace
time, at no cost to the hot path. Model-level free parameters like
`logsfr_ratios` are registered automatically and do not warn.

## Broadening: one kernel, three widths

<!-- 2026-09-03 broadening pass: describes ceridwen/broadening.py (Kinematics,
     Instrument, SpectralProjector, PhotometricBroadener) and the SedModel /
     Spectrum wiring of BROADENING_DESIGN.md section 7. -->

Every width a spectral feature acquires is a Gaussian in `ln λ`, so the
projection of the model onto a spectrum applies a single kernel per pixel:

```
continuum:  σ_cont² = σ_gal² + σ_inst²(λ) − σ_lib²(λ)
lines:      σ_line² = σ_gas² + σ_inst²(λ)
```

The three widths live in three different places, and each is set exactly once:

| Width | What it is | Where it is set |
|---|---|---|
| `σ_gal`, `σ_gas` | the galaxy's stellar and gas velocity dispersions [km/s] | `Kinematics(...)`, passed to `SedModel(kinematics=...)`; the same object serves every observation |
| `σ_inst(λ)` | the instrument's line-spread function at each observed pixel | `Instrument.<unit>(...)`, passed to each `Spectrum(instrument=...)` |
| `σ_lib(λ)` | the resolution already in the SSP library | stored in every schema-2 grid (`SSPData.ssp_resolution`); read automatically |

There is no other place where a width enters. In particular the model spectrum
(`csp.get_spectrum`) is **not** broadened by the galaxy any more, and emission
lines are painted onto a spectrum's pixels from their luminosities with
`σ_line`, never resampled from the model grid. `Lines` observations (integrated
fluxes) are read straight from the nebular grid and never see any width.

### The galaxy: `Kinematics`

```python
from ceridwen import Kinematics, DEFAULT_KINEMATICS

Kinematics(sigma_gal="sigma_gal")                     # σ_gal free (theta key), σ_gas tied to it
Kinematics(sigma_gal=250.0, sigma_gas="sigma_gas")    # stars fixed, gas free
Kinematics(sigma_gal=300.0)                           # both fixed at 300 km/s
DEFAULT_KINEMATICS                                    # == Kinematics(sigma_gal=300.0); what SedModel uses if you pass nothing
Kinematics.none()                                     # no kinematic broadening at all
```

A **float is fixed** (folded into the compiled graph), a **string is the name
of a free parameter** in `theta` and must carry a prior. `sigma_gas` defaults
to `TIED`, i.e. the same value or key as `sigma_gal`. `Kinematics(...)` itself
has no default for `sigma_gal`; the 300 km/s lives in one named constant so
that it is never hidden, and the model summary prints whichever values are in
force. `sigma_max` (default 2000 km/s) is the largest dispersion the compiled
kernels support: a prior whose upper bound exceeds it is rejected at setup,
and a sampled value is clipped to it.

!!! danger "Two sources of truth"
    Fixing `sigma_gal=250.0` *and* putting `"sigma_gal"` into `theta` raises a
    `ValueError` at setup. Naming a key that is not in `theta` raises a
    `KeyError`; a key without a prior gets the usual unpriored-parameter
    warning. There is no silent fallback.

### The instrument: `Instrument`

Published resolutions come in several units and in two resolving-power
conventions that differ by `2√(2 ln 2) ≈ 2.35`: instrument datasheets quote
`R = λ/FWHM`, the sedpy/Prospector tradition uses `R = λ/σ`. CERIDWEN will not
guess: the unit **and** the convention are the name of the constructor, so
there is nothing to declare and nothing to forget.

```python
from ceridwen import Instrument

Instrument.R_fwhm(2700)                    # R = λ/FWHM  (datasheet, e.g. NIRSpec gratings)
Instrument.R_sigma(6358)                   # R = λ/σ     (sedpy / Prospector "R"); the same LSF as R_fwhm(2700)
Instrument.fwhm_aa(2.5)                    # FWHM in Å, observed frame
Instrument.sigma_aa(1.06)                  # σ in Å, observed frame
Instrument.sigma_kms(47.0)                 # σ in km/s
Instrument.fwhm_kms(111.0)                 # FWHM in km/s
Instrument.R_fwhm(R_array, wave=obs_wave)  # any of the above per pixel, on observed wavelengths
```

Widths must be finite and positive; an array needs `wave=` on the same grid
and must cover the spectrum's pixels. The LSF is taken as *measured on the
detector* (from arc lines), so it already contains the pixel width and the
model profiles are sampled at pixel centres, not integrated over the pixel a
second time.

### The library: subtracted automatically

The SSP library's own resolution curve is subtracted in quadrature from the
instrumental width, pixel by pixel, at the rest-frame wavelength of each
observed pixel (velocity widths are frame-invariant). Where the instrument is
*finer* than the library the subtraction floors at zero and the continuum is
delivered at library resolution, which is the correct model for a grid coarser
than the instrument (`C3K_lr`, `BaSeL`), and you get a warning with the pixel
count and range so that a wrong unit on the `Instrument` cannot pass unseen.
`Spectrum(..., subtract_library=False)` turns the
subtraction off for a grid whose stored curve you do not trust; with
`instrument=None` there is nothing to subtract from and the continuum is
delivered as the library has it.

### Photometry is broadened too

Filters integrate the spectrum *as observed*, so by default the photometry is
computed from the `σ_gal`-broadened spectrum (`SedModel(broaden_photometry=True)`,
the Prospector behaviour). For broad bands this changes nothing measurable
(below `5e-4` mag at 300 km/s, `5e-3` at 1000 km/s, even with 1000 Å lines and
hard breaks); for a medium or narrow band with a line on its edge it reaches
the per-cent level, which is why the switch exists rather than the effect
being silently dropped. Turn it off only for broad-band-only fits where you
want to save the one FFT per likelihood call. With `broaden_photometry=False`
nothing in the photometry is broadened: the continuum is integrated at
model-grid resolution and any emission lines enter at the grid's pixel-floor
width, independent of `σ_gas`. Generate mocks and fit them with the same
setting.

How the lines reach the filters depends on whether `σ_gas` is fixed or
sampled. With a **fixed** `σ_gas` the emission lines are not painted onto the
model grid at all: each `Photometry` holds a static line-to-band basis
`G = T · Γ(σ_gas)` (the line profiles at `σ_gas` in quadrature with the
two-pixel grid floor, times the IGM transmission), and the band fluxes are
`T · continuum_broadened + G · F_line` — exact for the same profile, and cheap.
With a **sampled** `σ_gas` the width is a runtime quantity, so the lines are
painted onto the grid and broadened with the continuum-independent gas width
(`σ_gas` untied) or together with the continuum (`σ_gas` tied to `σ_gal`);
this is the slower path, chosen automatically. `csp._force_paint_lines = True`
forces the painted path for A/B checks.

## Cosmology: one object, chosen on the CSP, never silent

The cosmology enters the forward model in two places, both evaluated by the
CSP: the flux factor `(1+z) (10 pc / D_L)^2` that turns the rest-frame
luminosity density into observed-frame F_ν in cgs (`CSPBasis._flux_factor`, applied
to spectra, photometry and line fluxes), and the age of the universe that
rescales the SFH grid when `zred` is sampled with `track_zred_age=True`
(`CSPBasis._lookback_from_zred`). A sampled `zred` reaches every observation:
`Photometry` is projected through the filters per sample, `Lines` fluxes carry
the per-sample flux factor, Jacobian and IGM, and a `Spectrum` is projected by
a redshift-aware projector built for the prior's support (`Spectrum(zred_range=)`
when that support is not given by a bounded prior). So the cosmology is a required argument of
`CSPBasis` / `CSPBasis_afe`, there is no default and no global to patch:

```python
from ceridwen import Cosmology

cosmo = Cosmology.planck18()                # Planck 2018 VI, the usual choice
cosmo = Cosmology.planck15()
cosmo = Cosmology.wmap9()                   # what prospector hard-codes
cosmo = Cosmology.flat(H0=70.0, Om0=0.3)    # the two numbers a paper quotes
cosmo = Cosmology.from_name("Planck15")     # from a string (config file)
cosmo = Cosmology.from_astropy(astropy_cosmo)   # any flat astropy cosmology

csp = CSPBasis(ssp, lookback_time=..., cosmo=cosmo, ...)
```

Everything downstream reads it from the CSP: `model.cosmo` is a read-only
view, `SedModel(cosmo=...)` is accepted only when it equals `csp.cosmo`
and refused otherwise, `repr(csp)`, `model.summary()` and the `fitSED`
log print it, and `fitSED` writes it into the HDF5 result (`/model` attrs
`cosmo_H0`, `cosmo_Om0`, `cosmo_Tcmb0`, `cosmo_Neff`, `cosmo_m_nu_ev_sum`,
`cosmo_name`), from which `ceridwen.result_cosmology(path)` or
`Cosmology.from_dict(attrs)` rebuilds the same object.

Ages and distances come from the same object: `cosmo.age(z)` and
`csp.age_at(z)` give the age of the universe in Gyr (JAX-differentiable for
array input, a float for a scalar), `cosmo.luminosity_distance(z)` the
luminosity distance in Mpc. There is no `tuniv` argument any more; size the
SFH grid from the cosmology, `jnp.linspace(0.0, cosmo.age(z), n)`, and
`SedModel` refuses (with both numbers in the message) a grid whose oldest
node lies more than 0.5 % beyond `csp.age_at(zred)` at a fixed redshift.
The check is skipped when the CSP rescales the grid itself
(`track_zred_age=True`, which also acts on a fixed non-zero `zred`
because `SedModel` injects it) or a `lookback_time` transform replaces it.

`zred = 0` applies **no** flux factor: the CSP scales only when `theta`
carries `zred`, and `SedModel` does not inject a zero. Predictions are then
`L_sun/Hz x 10^logmass`, not maggies. For a nearby object give the distance
explicitly, `SedModel(csp, obs, priors, zred=0.0, lumdist_mpc=3.5)`: the
flux factor is then evaluated with that distance in place of `D_L(zred)`
(and `(1+zred)` still from `zred`), which is prospector's `lumdist`.
`model.summary()` says which of the three cases is in force, and
`SedModel` warns at construction when `zred = 0` is combined with
observations (no flux factor) and when `lumdist_mpc` overrides a non-zero
`zred`'s D_L. `csp.cosmo` cannot be reassigned after construction, and the
CSP constructors refuse unknown keyword arguments instead of swallowing
them.

`SedModel` builds every observation's projection at construction. Replacing
`model.observations` later requires `model.setup_observations()` (called for
you by `fitSED(model, observations)`); a `Spectrum` therefore needs its
`wavelength=` at construction even if the flux is attached later.

## FSPS at runtime

With `add_neb=True` or `add_dust_emission=True`, the forward model reads CLOUDY
nebular grids and Draine & Li templates from `$SPS_HOME`. FSPS must be installed
and `$SPS_HOME` set. See [Installation](installation.md).
