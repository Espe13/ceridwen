# Conventions & gotchas

These are the conventions that most often cause silent mistakes. Read them before
fitting real data. (The package also ships a fuller misuse guide in `GOTCHAS.md`.)

## Metallicity is log10 of ABSOLUTE Z

The parameter `Z` (and `ssp_lgmet`) is `log10(Z)` in **absolute** units, **not**
`log10(Z/Z☉)`. Solar is roughly `-1.85` (Z☉ ≈ 0.014), **not** `0.0`.

The FSPS grid spans roughly `[-4.0, -1.4]`. Values outside it are **silently
clamped** to the nearest grid edge, so a "solar" guess of `Z = 0.0` is off the
grid and will be clamped.

!!! danger "Common mistake"
    Setting a prior like `Uniform(low=-2.5, high=0.2)` puts most of the mass off
    the grid. Use something like `ClippedNormal(mean=-2.0, low=-4.0, high=-1.4)`.
    Call `csp.check_param_ranges(theta)` or `python -m ceridwen.check` if unsure
    of your grid bounds.

## Lookback time increases with index (index 0 = today)

`lookback_time` element 0 is the present; the last element is the oldest bin
(≈ the age of the universe). The `sfh` array is indexed the same way. The old
*decreasing* convention (`lookback = T_univ - t_grid`) is rejected at
construction with a `ValueError` — don't reintroduce it, and don't reverse
arrays "to be safe".

## Units

| Quantity | Unit |
|---|---|
| Wavelength (input) | Å, vacuum, rest-frame |
| Model spectra | `F_ν` (per unit frequency) |
| Broadband fluxes | AB maggies |
| Emission-line fluxes | erg s⁻¹ cm⁻² |
| Stellar mass | `logmass` = log10(M⋆/M☉) |

The forward model is evaluated at unit mass and scaled by `10**logmass`.

## SFH mass normalisation

The `logsfr_ratios_to_sfh` transform normalises the SFH so the trapezoidal
integral of SFR over the lookback grid equals 1 M☉, and `logmass` sets the
amplitude. Use the provided transform rather than hand-rolling it — getting this
wrong biases `logmass` by many dex.

## `theta` is a dict — typos are silently ignored

A mistyped key (`logmas` for `logmass`, `dust2` for `diffuse_tau_kc`, …) is
simply not read, so the parameter takes its default. `predict()` /
`get_spectrum_components()` emit a warning listing unrecognized keys at trace
time (zero hot-path cost). Model-level free parameters like `logsfr_ratios` are
registered automatically and do not warn.

## FSPS at runtime

With `add_neb=True` or `add_dust_emission=True`, the forward model reads CLOUDY
nebular grids and Draine & Li templates from `$SPS_HOME`. FSPS must be installed
and `$SPS_HOME` set; see [Installation](installation.md).
