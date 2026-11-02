# AGN models

`CSPBasis(..., agn=AGNConfig(...))` adds AGN emission. The default
`agn_mode="corrected"` normalises and attenuates the components as described
below. Modes `"fsps"` and `"prospector"` reproduce the corresponding
implementations, subject to the differences listed in the comparison sections.
Mode `"cigale"` follows CIGALE's SKIRTOR and polar-dust conventions.

Omitting `agn=` preserves existing predictions byte for byte, as checked by
`scripts/bit_identity_check.py`. The additional AGN2 components are disabled
by default and do not change `AGNConfig()` predictions until enabled.

```python
from ceridwen import CSPBasis, Cosmology
from ceridwen.agn import AGNConfig

csp = CSPBasis(ssp, lookback_time=lb, zh_const=True, add_dust=True, add_neb=True,
               add_dust_emission=True, cosmo=Cosmology.planck18(),
               agn=AGNConfig())                       # corrected mode, torus + disc + lines
csp = CSPBasis(ssp, ..., agn=AGNConfig(agn_mode="fsps"))         # FSPS add_agn_dust
csp = CSPBasis(ssp, ..., agn=AGNConfig(agn_mode="prospector", dust4_type=1))
```

`agn=True` means `AGNConfig()`. A string such as `agn="fsps"` sets
`AGNConfig(agn_mode="fsps")`. `CSPBasis_afe` accepts the same argument.
See `examples/agn_demo.py` for an example.

## Components and parameters

| component | theta keys (default) | meaning | source |
|---|---|---|---|
| Nenkova+08 clumpy torus | `fagn` (0), `agn_tau` (10) | L_torus / L_bol,\* ; torus optical depth, grid 5-150 | FSPS `agn_dust.f90`, `$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat` |
| Temple+21 accretion disc (Wang+24) | `fagn_bbb` (0) | F_nu,disc / F_nu,\* at rest 5500 A (dust-free young + old stellar spectrum) | `ceridwen/agn/data/temple21.txt` (Prospector's `temple.txt`) |
| AGN dust screen | `dust4` (0), `dust4_index` (-1.0) | extra screen on the AGN: `exp(-dust4 (lam/5500)^dust4_index)` times the galaxy's diffuse curve; `AGNConfig(dust4_type=1)` | Prospector `fake_fsps.attenuate` |
| AGN narrow lines | `agn_elum` (0), `agn_eline_sigma` (100) | L_Hbeta [L_sun] per formed M_sun; Gaussian width [km/s]; 23 lines with Richardson+14 'a42' ratios by FSPS name | Prospector `AGNSpecModel` |

L_bol,* is the dust-free stellar luminosity integrated over frequency on
the model grid using DL07 trapezoidal weights, per formed mass. AGN
components scale with formed mass in the same way as stellar emission.

`AGNConfig` has the following fixed options:

- `agn_mode` selects the implementation.
- `torus`, `disk` and `lines` enable components. `None` uses the mode's
  default: torus only for `fsps`, all three for the other modes.
- `dust4_type` accepts 0 or 1; 2 is unsupported, as in Prospector.
- `tau_interp` is `"linear"` for FSPS interpolation. Corrected mode also
  accepts `"log"`.
- `agn_energy_balance` is available in corrected mode. `None` enables it
  when dust emission is active.
- `torus_file` specifies a template file.

**Sampling amplitudes.** `fagn` and `fagn_bbb` span several orders of
magnitude. Sample their logarithms and transform them:

```python
priors = {"log_fagn": Uniform(low=-5.0, high=np.log10(3.0)),       # Prospector: LogUniform(1e-5, 3)
          "agn_tau": LogUniform(mini=5.0, maxi=150.0),
          "log_fagn_bbb": Uniform(low=-5.0, high=2.0),               # LogUniform(1e-5, 100)
          "dust4_index": TopHat(low=-1.8, high=-0.8),
          "agn_elum": Uniform(low=1e-6, high=1e-2),
          "agn_eline_sigma": Uniform(low=50.0, high=500.0)}
transforms = {"fagn": lambda th: 10.0 ** th["log_fagn"],
              "fagn_bbb": lambda th: 10.0 ** th["log_fagn_bbb"]}
free_param_init = {"log_fagn": jnp.array([-1.0]), "log_fagn_bbb": jnp.array([-1.0])}
```

Add `log_fagn` and `log_fagn_bbb` to `free_param_init` as well as `priors`
and `transforms`. `SedModel` rejects priors for names missing from the
initial parameters. With `add_neb=True` and `add_dust_emission=True`, provide
priors or constant transforms for `gas_logz`, `gas_logu`, `duste_qpah`,
`duste_umin`, `duste_gamma`, `tau_pow` and `alpha_pow`. AGN lines also add
`agn_eline_sigma`. Nested sampling requires a prior for every free parameter.
Dust emission requires both `add_dust=True` and `add_diffuse_dust=True`,
which are enabled by default. Run `examples/agn_demo.py --fit` for a full fit.

**Photometric constraints.** Stellar dust emission and the torus both
contribute in the mid-infrared. With energy balance enabled, `log_fagn`
is anti-correlated with diffuse optical depth. Mass is also degenerate
with SFH shape.

In `agn_demo.py --fit`, using UV-to-24 µm photometry at S/N 20 on an Apple
GPU, the 16–84% intervals for `log_fagn` and `logmass` contained the input
values. The `agn_tau` interval, 21–76, was just above its input of 20.
With an earlier `StudentT(scale=1.0)` SFH prior, a mock fit underestimated
`log_fagn` by 0.25 dex and `logmass` by 0.3 dex on CPU and GPU. Both inputs
fell outside the 95% intervals even though the maximum-likelihood sample
was near the input. Broadband data weakly constrain `log_fagn_bbb`.
Inspect the corner plot when interpreting photometric AGN fractions.

## Fitting the AGN line width

`agn_eline_sigma` is the Gaussian dispersion in km/s for the 23 AGN narrow
lines. Corrected mode combines it with the instrumental width at each line
as sqrt(agn_eline_sigma² + sigma_inst²). Profiles are evaluated on all
pixels by `SpectralProjector.setup_agn_lines`. Gradients remain finite
down to 1e-3 km/s (`tests/agn/test_eline_sigma.py`).

Fixed values, transforms and prior support must lie in (0, 2000] km/s,
using the default `Kinematics.sigma_max`. Invalid values raise an error
at setup. These lines use Richardson+14 narrow-line ratios. For broad-line
emission, use `AGNConfig(blr=True)` and `blr_sigma`, described below.

Only a spectrum resolving AGN lines provides a useful width constraint.
Integrated `Lines` fluxes are independent of width; photometry depends
on it mainly near filter edges. `SedModel` warns if a width prior is
supplied without a spectrum, without unmasked AGN-line pixels, or when
instrumental sigma exceeds three times the prior's upper bound at every
AGN line. Fix the width in these cases:

```python
transforms = {"agn_eline_sigma": lambda th: jnp.array([100.0])}   # and no prior on it
```

The instrumental sigma is sigma_inst = c / (2.3548 R_FWHM):

| R_FWHM | sigma_inst [km/s] | resolves (sigma_inst <= 3 x prior upper edge) for an upper edge >= |
|---|---|---|
| 30 | 4244 | 1415 km/s |
| 100 | 1273 | 424 km/s |
| 300 | 424 | 141 km/s |
| 1000 | 127 | 42 km/s |
| 2700 | 47 | 16 km/s |

Passing this check does not establish that the width is measured.
A prism at R ≈ 100 with `Uniform(50, 500)` passes because
1273 < 3 × 500, yet the posterior remains prior-dominated. In the tests
below, the grating constrained the width and the prism did not.

The CPU recovery test used
`scripts/agn_injection_recovery.py --spec grating|prism` at z = 3 with
JADES-like photometry and a noiseless rest-frame 3600–6900 Å spectrum,
assigned continuum S/N 20 per pixel. Corrected mode included nebular
emission. Input values were `agn_elum=3e-3`, `agn_eline_sigma=250` and
`sigma_gas=60`. These, stellar parameters and continuum AGN parameters
were fitted jointly with nested sampling and `slice_kernel="lanes"`:

| parameter | truth | grating, R 1000 (num_live 400): median, 16-84 % | in 2.5-97.5 % | prism, R 100 (num_live 100): median, 16-84 % | in 2.5-97.5 % |
|---|---|---|---|---|---|
| `agn_elum` | 0.003 | 0.0033 (0.00312 to 0.00348) | yes | 0.00368 (0.00306 to 0.00435) | yes |
| `agn_eline_sigma` | 250 | 252 (248 to 255) | yes | 267 (141 to 378) | yes |
| `sigma_gas` | 60 | 60.4 (58.1 to 62.8) | yes | 139 (62.6 to 224) | yes |
| `log_fagn` | -0.523 | -0.417 (-0.61 to -0.231) | yes | -0.517 (-0.654 to -0.249) | yes |
| `agn_tau` | 20 | 28 (14.8 to 46.2) | yes | 21.4 (11.9 to 46) | yes |
| `log_fagn_bbb` | 0 | -0.0036 (-0.0178 to 0.0124) | yes | 0.0397 (-0.025 to 0.107) | yes |
| `logmass` | 10 | 9.956 (9.934 to 9.979) | NO (97.5 %: 9.999) | 9.931 (9.9 to 9.974) | yes |
| `logzsol` | -0.5 | -0.543 (-0.585 to -0.493) | yes | -0.558 (-0.659 to -0.478) | yes |
| `diffuse_tau_kc` | 0.4 | 0.4 (0.387 to 0.417) | yes | 0.439 (0.377 to 0.502) | yes |

The ratio of posterior to prior standard deviation for `agn_eline_sigma`
was 0.029 with the grating and 0.85 with the prism. Grating line parameters
were recovered to a few percent. `logmass` was just outside its 95% interval:
the 97.5% quantile was 9.999 with 400 live points and 9.991 with 100.
The input has χ² = 0, and the best 100-live-point sample had χ² = 0.9.
The mass offset reflects marginalisation over SFH directions rather than
a missed likelihood peak. Since `agn_elum` is per formed mass, it is
strongly anti-correlated with `logmass` (−0.92). The script writes
posteriors to `sigma_<spec>[_nl400].npz`.

**Gas-width degeneracy.** AGN and star-forming emission share wavelengths
such as Hβ, [O III], Hα, [N II] and [S II]. Each observed profile combines
Gaussians of widths sqrt(sigma_gas² + sigma_inst²) and
sqrt(agn_eline_sigma² + sigma_inst²). Separating them depends on their
relative fluxes and line ratios.

For the prism, both widths were weakly constrained: `sigma_gas` was
139 (+85/−76) km/s for an input of 60, and the AGN posterior/prior width
ratio was 0.85. The grating recovered `agn_eline_sigma=252` (+3/−4) and
`sigma_gas=60.4` (+2.4/−2.3) km/s, with correlation −0.17. Different
line ratios helped distinguish the two contributions.

The fixed Richardson+14 'a42' ratios and gas-dependent CLOUDY ratios
also constrain `agn_elum` when widths are unresolved. The prism fit gave
3.68e-3 [3.06e-3, 4.35e-3] for an input of 3e-3. Interpret AGN emission
using the line ratios as well as the profiles.

## FSPS, Prospector and corrected modes

| | `"fsps"` | `"prospector"` | `"corrected"` (default) |
|---|---|---|---|
| reproduces | FSPS `add_agn_dust` | Prospector `AGNSSPBasis` + `AGNSpecModel` | — |
| torus resampling | `sps_setup.f90`: log-log with FSPS's float32 `tiny30`, `locate` edges | `fake_fsps.agn_torus`: argmin edges, log-log | positive rows only, zero outside, renormalised so that the torus carries exactly `fagn * L_bol` on any grid |
| `agn_tau` outside [5, 150] | clamped (flat likelihood) | extrapolated linearly (can go negative) | clamped (flat likelihood) |
| torus screen | galaxy diffuse curve | none | galaxy diffuse curve |
| disc screen | (no disc) | diffuse curve x dust4 screen | diffuse curve x dust4 screen |
| IGM on the AGN | none (FSPS adds the torus after `igm_absorb`) | Prospector's `fake_fsps.add_igm` (17 Lyman lines, no metal term) | the CSP's IGM, like every other component |
| AGN lines | painted with Prospector's painter (no instrument, no dust, no IGM); photometry as delta functions | same as fsps | nebular painter with `agn_eline_sigma` (instrument included), the galaxy's diffuse dust and the IGM at each line; photometry through the model grid like the nebular lines |
| energy balance | no | no | absorbed AGN light (continuum and lines) re-emitted by DL07 when `add_dust_emission=True` |

Without dust emission, absorbed AGN light is removed from the output,
as is absorbed stellar light. FSPS and Prospector modes do not support
AGN energy balance and reject `agn_energy_balance=True`. In those modes,
AGN lines require `add_neb=True` because the Prospector profiles are placed
only near nebular lines. Corrected mode does not require a nebular model
to add AGN lines.

## Comparison with FSPS and Prospector

Reference files use python-fsps 0.5.0 (libfsps `05b5e55`) and prospect
`91f8e13`. `scripts/agn_reference/make_refs.py` writes them to
`tests/reference/agn/*.npz`; comparisons are in `tests/agn/`.

| test | what | result |
|---|---|---|
| A1 | fsps-mode torus vs python-fsps `spec(fagn) - spec(0)`: fagn {1e-3, 0.1, 1, 3} x agn_tau {5, 7.5, 10, 40, 150} x dust2 {0, 0.5, 1.5} x z {0, 2, 7} | max rel 1.7e-8, inside FSPS's own cancellation floor (REF's rule); a float64 numpy port of FSPS agrees to the same floor |
| A2 | prospector-mode torus vs `fake_fsps.agn_torus`, agn_tau 1-300 (extrapolation) | max rel 3.1e-14 |
| A3 | disc + dust4 screen vs `agnssp_basis.get_spectrum` (140 configs, dust_type 0/2/4/6, dust4_type 0/1, z 0/6) | max rel 5.7e-14 |
| A4 | AGN lines vs `AGNSpecModel` (agn_eline_sigma 50, 300) | luminosities exact; spectrum 1.2e-13 of peak; Lines 8.9e-18; photometry 1.7e-16 with sedpy's curves (6e-5 with CERIDWEN's vendored NIRCam curves, a filter-data difference) |
| A5 | whole model vs python-fsps + `AGNPolySpecModel` (z = 3, 15 NIRCam + MIRI bands) | torus spectrum 1.7e-14 (Prospector's L_bol fed in); disc spectrum 1.7e-14 in shape; full-CSP AGN-only photometry = Prospector's x the galaxy scalars to 1.1e-3 (the filter-curve term, the same on the galaxy) |

The whole-model A5 comparison depends on differences in stellar L_bol,
the 5500 Å disc normalisation and filter curves. For its BPASS SFH,
CERIDWEN/Prospector ratios were 0.8515 for L_bol, 0.8300 for disc
normalisation and 0.99447 for flux normalisation from cosmology.
Accounting for these factors leaves the filter-curve difference.

In the referenced Prospector version, `AGNSSPBasis` passes `fagn` and
`agn_tau` to python-fsps, whose Fortran default `add_agn_dust=1` adds a
torus. Python-fsps cannot disable that default. `AGNSSPBasis` then adds
`fake_fsps.agn_torus`, producing 1.39–1.90 times the torus-only photometry.
CERIDWEN's Prospector mode reproduces the single torus term. The test
`tests/agn/test_a5_whole_model.py` records this difference.

## Corrected-mode checks

| test | result |
|---|---|
| B1 resampled torus integrates to 1 on any grid | `abs(sum(T w) - 1) <= 2e-15` for every tau node |
| B2 node exactness, clamping, no negative flux, no NaN for tau in [0.1, 1e4] | pass |
| B3 fagn = fagn_bbb = agn_elum = 0 equals AGN off | byte-identical, all modes, all eight kernels, a/Fe, `wavelength_support=True`, Photometry + Spectrum + Lines |
| B4 linear in fagn, fagn_bbb, agn_elum | exact in the float64 AGN term; to float32 rounding in the predictions |
| B5 L_torus = fagn x L_bol (screen off) | rtol 1e-10 |
| B6 energy balance | absorbed AGN light (continuum + lines) re-emitted; conserved to DL07's own self-absorption loss, 4.6e-7 of the absorbed light, the same as the stellar path |
| B7 IGM order (corrected vs fsps at z = 8) | the difference is the AGN light blueward of Ly-alpha times the IGM: 0.36 % (tau 5) to 1.33 % (tau 150) of L_AGN |
| B8 jit, vmap (W = 64, 100), grad | finite gradients in every AGN key; vmap = loop to float32 reordering |
| B9 `marginalize_elines` with AGN lines | refused at setup |

On a dense grid, integrating the log-log interpolated Nenkova+08 SEDs
gives 0.989–0.996 times the trapezoidal integral of the original nodes.
Corrected mode renormalises to `fagn * L_bol`; FSPS mode retains the
FSPS normalisation. The resulting difference is 0.4–1.1%.

## AGN2 components: SKIRTOR, polar dust, NLR and BLR

These four `AGNConfig` switches are disabled by default:

```python
from ceridwen.agn import AGNConfig
agn = AGNConfig(agn_mode="corrected",            # or "cigale"
                torus_model="skirtor16",         # SKIRTOR torus + its disc (replaces Nenkova+08)
                polar_dust=True, polar_law="smc",  # CIGALE polar dust ("smc", "calzetti", "gaskell")
                nlr_model="feltre16",            # Feltre+16 narrow lines (replace the a42 ratios)
                blr=True)                        # broad H I lines
```

| component | theta keys (default) | meaning | source |
|---|---|---|---|
| SKIRTOR 2016 torus + disc | `fagn` (0), `skirtor_t` (7), `skirtor_pl` (1), `skirtor_q` (1), `skirtor_oa` (40), `skirtor_i` (30) | amplitude (below); tau_9.7 {3..11}, p, q {0..1.5}, opening angle [deg, equator to torus edge] {10..80}, inclination [deg, 0 = face-on] {0..90}; `AGNConfig(skirtor_R=10/20/30)` static (default 20) | Stalevski+12, +16 templates as CIGALE's `skirtor2016` uses them (Schartmann 2005 disc, delta 0; cigale 29cb909f) |
| polar dust | `polar_ebv` (0.03), `polar_temp` (100 K), `polar_emis` (1.6) | E(B-V) of the polar screen; grey body T, beta (lambda_0 = 200 um) | CIGALE `skirtor2016.py` (Yang+20, +22) |
| Feltre+16 NLR | `agn_elum` (0), `agn_eline_sigma` (100), `nlr_logu` (-2.5), `nlr_logz` (0), `nlr_alpha` (-1.7), `nlr_xid` (0.3), `nlr_lognh` (3) | L_Hbeta,NLR [L_sun] per formed M_sun; sigma [km/s]; log U_S [-5, -1]; log Z/Z_sun [-2.18, +0.66] (Z_sun = 0.01524); alpha [-2, -1.2]; xi_d [0.1, 0.5]; log n_H [2, 4] | public NEOGAL grid (`AGN_NLR_nebular_feltre16.tar.gz`), 17 of its 20 lines (N V 1240, [Si III] 1883, Si III] 1888 are not in the FSPS line list); [O II] 3726+3729 sits on `[O II] 3726` as in the a42 table |
| BLR | `blr_lum` (0), `blr_sigma` (1500) | L_Halpha,BLR [L_sun] per formed M_sun, before attenuation; Gaussian **sigma** [km/s] (FWHM = 2.35482 sigma; support [400, 5000]) | 8 H I lines (H alpha-delta, Pa alpha-delta), Case B (Storey & Hummer 1995 via PyNeb), T = 1e4 K, n_e = 1e4: H alpha / H beta = 2.847 (`ceridwen/agn/data/blr_caseB_T1e4_ne1e4.csv`) |

Interpolation is multilinear in the tabulated variables, using log line
ratios for the NLR. Values are clamped to grid edges. Fixed values outside
a grid produce a setup warning; priors extending outside it raise an error.

SKIRTOR (227 MB) and NLR tables are supplied separately. The repository
scripts `scripts/agn2/build_skirtor_h5.py` and `build_nlr_h5.py` write
`skirtor2016.h5` and `nlr_feltre16.h5` to `$CERIDWEN_AGN_DATA`, which
defaults to `~/.ceridwen/agn/`. Missing files raise an error at construction
with setup instructions. Dependent tests skip with "MISSING ...: did NOT run".
The SKIRTOR and Feltre+16 redistribution licences have not been checked.

### CIGALE extinction curves

Below 100 nm, polar dust uses CIGALE's `extFun_SMC.dat`, `extFun_MRN.dat`
and `extFun_Gaskel04.dat` from `pcigale/sed_modules/curves/` at commit
29cb909f. These CeCILL-2.0 files are not bundled with CERIDWEN.
Install them in `$CERIDWEN_AGN_DATA` (default `~/.ceridwen/agn/`):

```bash
python -m ceridwen.agn.fetch --from /path/to/cigale   # a CIGALE checkout or installed pcigale; no network
python -m ceridwen.agn.fetch --download               # from gitlab.lam.fr, through the system curl
```

From Python, use `ceridwen.agn.fetch_grids(from_cigale=...)` or
`fetch_grids(download=True)`. Each SHA-256 must match the reference table
used by the comparison tests; differing files are not written. Manual
copies may use the CIGALE names or the older `cigale_extFun_*.txt` names.
Missing or changed curves raise a construction error when `polar_dust=True`.
Tests requiring missing curves are skipped.

The following combinations are unsupported:

- AGN2 switches in `"fsps"` or `"prospector"` mode.
- SKIRTOR with the Temple disc or `dust4_type=1`, since SKIRTOR includes
  its own disc and uses polar dust for its nuclear screen.
- `polar_dust=True` without SKIRTOR.
- In `"cigale"` mode, the Nenkova torus, Temple disc, dust4 screen or
  a42 lines. With SKIRTOR, galaxy dust is required because `fagn` is defined
  relative to its luminosity. This mode also rejects `fesc_geometry="picket"`,
  since that luminosity is calculated using screen geometry. A `fagn`
  prior must remain below 1.
- `marginalize_elines=True` with narrow or broad AGN lines.

### `agn_mode="cigale"` and `"corrected"`

| | `"cigale"` | `"corrected"` |
|---|---|---|
| `fagn` (SKIRTOR) | CIGALE's `fracAGN` = P / (P + L_dust,gal): P = torus + polar-dust luminosity, L_dust,gal what the galaxy's dust absorbs from stars and nebular emission (CIGALE's `dust.luminosity`); must be < 1 | P / L_bol,\* (as the Nenkova torus); CIGALE's fracAGN is reported as `frac_agn_cigale` |
| galaxy dust on the AGN | none (CIGALE adds the AGN after its dust modules) | the diffuse curve on the disc, torus, polar re-emission, NLR and BLR |
| energy balance | none | what the galaxy's dust absorbs from the AGN goes to DL07 (with `add_dust_emission=True`) |
| polar dust | CIGALE: the disc extinguished only for a type-1 view (i <= 90 - oa); the re-emitted luminosity is the angle-averaged absorption of the face-on disc; disc, torus and grey body renormalised together | the same 4 pi accounting; CIGALE's Calzetti k (negative above ~3.1 um) clipped at 0 |
| resampling onto the model grid | CIGALE's `np.interp` of L_lambda, 0 outside the template grid | log-log between positive template points, scaled to the template's own integral (pixels independent of the model grid's extent) |
| NLR attenuation | none (the Feltre NLR is not a CIGALE component; CIGALE's unreleased `nebular_AGN` would apply the polar screen) | galaxy diffuse curve at each line |
| BLR | `blr_lum` with the Case-B table, polar-dust transmission at each line (CIGALE's `nebular_AGN` convention); CIGALE's own BLR scaling (ionising photons x covering factor, Cloudy tables with H alpha / H beta = 46 at its defaults) is not reproduced | polar (or dust4) transmission x galaxy curve at each line; what the polar screen removes is not re-emitted (CIGALE re-emits only disc light) |
| line profile | Gaussian in ln lambda (CERIDWEN's painter); CIGALE's AGN lines are Gaussian in lambda: they differ at order sigma/c (4.7e-4 of the peak at FWHM 800 km/s, 4.1e-3 at 7000), same area | same |
| IGM | after the AGN (CIGALE's redshifting module acts on every contribution) | the same |

With SKIRTOR, `PostProcess` returns `extras["agn"]["frac_agn_cigale"]`
in both modes; it equals `fagn` in `"cigale"` mode. CIGALE's `agn.*`
information is stored as `cigale_<name>`, including accretion power,
intrinsic L_nu at 2500 Å viewed at 30°, L_6um, L_12um and disc, torus
and polar luminosities. It also returns `L_dust_gal`, `L_skirtor_disk`,
`L_skirtor_dust` and `L_polar_ir`.

### Comparison with CIGALE and reference tables (`tests/agn2/`)

| test | what | result |
|---|---|---|
| L1 | `"cigale"` SKIRTOR (disc, torus separately) vs pcigale `skirtor2016` (CIGALE 29cb909f), 71 node sets incl. both sides of the type-1/2 boundary, CIGALE's 948-point grid | max rel 1.6e-14 where flux > 1e-6 x peak |
| L2 | the template files (Stalevski+16): torus-dust shape at 40 nodes; the disc anisotropy cos i (2 cos i + 1) | 5.1e-15; 1.5e-9 |
| L3 | with polar dust: 218 sets (3 laws, E(B-V) 0.03-3, T 50-300 K, i 30/70, oa 20/40/60) vs pcigale | max rel 1.7e-12 |
| L4 | Feltre+16 NLR through the CSP vs the public grid files, all nodes; BPT (their Fig. 2) | 4.4e-16 dex |
| L5 | BLR: line ratios vs the Case-B table; the profile vs CIGALE's AGN-line Gaussian | ratios exact; profile as in the table above |
| corrected SKIRTOR energy | torus + polar IR luminosity vs `fagn` x L_bol,\* on the 15000-point test grid | 0.9e-6 to 5.5e-6 (the model grid's quadrature) |

### Priors

The following choices are used in the AGN2 configuration, with NLR priors
following BEAGLE-AGN (Vidal-Garcia+24):

- In `"cigale"` mode, use `fagn ~ Uniform(0, 0.99)`. In corrected mode,
  use `log_fagn ~ Uniform(-5, log10 3)`.
- Without mid-IR coverage, fix `skirtor_t=7`, `skirtor_pl=1`,
  `skirtor_q=1` and `skirtor_oa=40`. Use `skirtor_i ~ Uniform(0, 90)`.
- Use `polar_ebv ~ Uniform(0, 1)`.
- For the NLR, use `nlr_logu ~ Uniform(-4, -1.5)` and
  `nlr_logz ~ Uniform(-2.18, 0.24)`. Fix `nlr_xid=0.3`, `nlr_alpha=-1.7`
  and `nlr_lognh=3`, which are degenerate with log U and amplitude.
- Use `blr_sigma ~ LogUniform(400, 3000)` km/s. This corresponds to
  the cited JWST broad-Hα FWHM range of 890–6279 km/s. Only a resolved
  spectrum constrains this width; integrated line fluxes do not.

## Model assumptions and limitations

### Line amplitudes are independent of the AGN continuum

`agn_elum` for Richardson+14 or Feltre+16 narrow lines and `blr_lum` for
broad lines are luminosities per formed solar mass. They do not depend
on `fagn`, `fagn_bbb` or the continuum. `FeltreNLR.line_luminosity` returns
`agn_elum * R_line(theta)`, `AGNLines` returns `agn_elum * ratio` and
`BroadLines` returns `blr_lum * ratio_halpha`. The CSP multiplies these
by the SFH's formed mass `sum(W)` and `10**logmass`.

This allows broad or narrow AGN lines with `fagn=fagn_bbb=0`. Use priors
to restrict this if required by your model. Without energy balance,
amplitudes are constrained by line emission and only weakly by photometry
that does not resolve it. With `agn_energy_balance=True`, line light
absorbed by galaxy dust is re-emitted by DL07, adding an IR constraint.

For scale, the AGN2 notes quote galaxy nebular Hβ luminosities of roughly
1e-3–1 L_sun per formed M_sun. This range has not been recomputed here.
An `agn_elum` of 1 can therefore be comparable to the galaxy's entire
nebular Hβ luminosity.

Tying line amplitudes to the AGN continuum would require an ionising-photon
integral, a Case-B or CLOUDY conversion and a covering factor. It would
also change parameter registration, the linear-amplitude tests and the
golden configuration.

### NLR ratios use a separate Feltre+16 grid

The NLR uses the five Feltre, Charlot & Gutkin (2016) axes: `nlr_logu`,
`nlr_logz`, `nlr_xid`, `nlr_lognh` and `nlr_alpha` (`nlr.py:54–60`).
`nlr_alpha` is the UV slope used to generate that grid. It is independent
of the fitted disc: SKIRTOR has no slope parameter and Temple+21 has
only amplitude `fagn_bbb`. NLR ratios therefore do not respond to the
fitted continuum's luminosity or shape.

A continuum-coupled NLR would require a disc model with a slope matching
the grid axes and the photon-rate normalisation described above. A
transform could then tie `nlr_alpha` to that slope.

### The BLR does not depend on inclination

`BroadLines` has parameters `blr_lum` and `blr_sigma` (`blr.py:112`).
Its attenuation includes the nuclear screen, either polar dust or
the dust4 power law, and in corrected mode the galaxy's diffuse curve.
`PolarDust.transmission` depends on `polar_ebv`, without an inclination
term, as in CIGALE.

The SKIRTOR disc receives polar attenuation only at type-1 inclinations,
`skirtor_i <= 90 - skirtor_oa`, while broad lines receive it at every
inclination. The model does not hide the BLR automatically for a type-2
view. If broad lines should be absent, constrain `blr_lum` accordingly.

Inclination-dependent BLR obscuration would require a covering or
attenuation function, a decision about re-emitting hidden light and
a new golden configuration. It would also change the interpretation
of `blr_lum` as observed broad-Hα luminosity.

### There is no X-ray or corona component

The model has no X-ray component or `alpha_ox` normalisation. It cannot
predict an X-ray flux or use X-ray measurements to constrain the AGN.

The disc templates do extend below the Lyman limit:

- The Temple+21 template ends at 91 Å. Corrected mode sets shorter-wavelength
  flux to zero rather than extrapolating a constant F_nu (`disk.py:98–100`).
- The SKIRTOR parametric disc starts at 8 Å (`skirtor_disk` /
  `schartmann2005_disk`, `skirtor.py:113–116`) and includes EUV emission
  on its template grid.

The SSP model grid's shortest wavelength and IGM transmission determine
which of this emission reaches a prediction.

An X-ray extension would need a component and normalisation convention,
such as CIGALE's `alpha_ox` relation to 2500 Å disc luminosity. It would
also require extending the model grid, IGM and filter projection below
the SSP wavelength limit, with a new golden configuration.

## Outputs and compatibility

- In `PostProcess`, `spectra_model` and `spectra_observed` include AGN continuum;
  `spectra_intrinsic` and `spectra_dustfree` exclude it. Consequently,
  `A_V = 2.5 log10(dustfree/full)` includes the AGN contribution.
  `spectra_agn` contains AGN continuum alone. `extras["agn"]` returns
  `L_bol_star`, `L_absorbed` and, with energy balance, `mdust` for dust
  heated by the AGN. Stellar dust mass is not reported. `LUV` and `MUV`
  include AGN emission. AGN narrow lines appear only in per-observation
  predictions.
- Result files record mode, components, energy balance and template/table
  hashes in `/model@csp_config["agn"]`. `check_model_against_result`
  reports differences in AGN configuration.
- Line marginalisation rejects models with AGN lines because the free
  fluxes would absorb those lines. Disable them with `AGNConfig(lines=False)`
  to marginalise, also leaving the BLR disabled when applicable.
- Supplying `fagn` without `agn=` produces an unknown-key warning and
  adds no AGN. Enable the component when constructing the CSP.
- AGN calculations use dense operations without scatter-adds, and the
  CPU comparison tolerances also apply on GPU.
