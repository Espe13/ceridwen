# Conventions & gotchas

A few conventions cause silent mistakes if you get them wrong. Read them before
fitting real data. The repository also ships a fuller misuse guide in
`GOTCHAS.md`.

## Metallicity is log10 of ABSOLUTE Z

The parameter `Z` (and `ssp_lgmet`) is `log10(Z)` in **absolute** units, not
`log10(Z/Z☉)`. Solar is roughly `-1.85` (Z☉ ≈ 0.014), not `0.0`.

The FSPS grid spans roughly `[-4.0, -1.4]`. Values outside it are silently
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
| Model spectra | `F_ν` (per unit frequency) |
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

## Resolving powers: σ or FWHM?

Published resolutions come in two conventions that differ by
`2√(2 ln 2) ≈ 2.35`: instrument datasheets quote `R = λ/FWHM`, the
sedpy/Prospector tradition uses `R = λ/σ`. CERIDWEN will not guess: a
`Spectrum` with `smoothtype="R"` refuses to build until you declare the
convention (`res_convention="fwhm"` for datasheet numbers, `"sigma"` for the
Prospector one). For the other smooth types, widths are Gaussian sigmas unless
you pass `res_convention="fwhm"`. The SSP library's own resolution is stored in
every schema-2 grid and subtracted in quadrature automatically (`inres="auto"`,
the default); setting `inres=0.0` turns that off and over-broadens the model —
don't.

## FSPS at runtime

With `add_neb=True` or `add_dust_emission=True`, the forward model reads CLOUDY
nebular grids and Draine & Li templates from `$SPS_HOME`. FSPS must be installed
and `$SPS_HOME` set. See [Installation](installation.md).
