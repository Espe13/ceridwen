# AGN

`CSPBasis(..., agn=AGNConfig(...))` adds an active nucleus to the forward model: everything
Prospector has for AGN, built so that it can **reproduce** FSPS and Prospector exactly
(`agn_mode="fsps"`, `"prospector"`), and then **correctly** (`agn_mode="corrected"`, the
default). The CIGALE-standard components (SKIRTOR torus, polar dust, Feltre+16 narrow-line
region, broad-line region; section "AGN2") come with a fourth mode, `agn_mode="cigale"`, that
reproduces CIGALE. Without `agn=` nothing changes: every existing prediction is byte-identical
(`scripts/bit_identity_check.py`), and so does `AGNConfig()` with the AGN2 switches off.

```python
from ceridwen import CSPBasis, Cosmology
from ceridwen.agn import AGNConfig

csp = CSPBasis(ssp, lookback_time=lb, zh_const=True, add_dust=True, add_neb=True,
               add_dust_emission=True, cosmo=Cosmology.planck18(),
               agn=AGNConfig())                       # corrected mode, torus + disc + lines
csp = CSPBasis(ssp, ..., agn=AGNConfig(agn_mode="fsps"))         # FSPS add_agn_dust
csp = CSPBasis(ssp, ..., agn=AGNConfig(agn_mode="prospector", dust4_type=1))
```

`agn=True` is short for `AGNConfig()` and `agn="fsps"` (etc.) for `AGNConfig(agn_mode="fsps")`.
`CSPBasis_afe` takes the same argument. A worked example is `examples/agn_demo.py`.

## Components and parameters

| component | theta keys (default) | meaning | source |
|---|---|---|---|
| Nenkova+08 clumpy torus | `fagn` (0), `agn_tau` (10) | L_torus / L_bol,\* ; torus optical depth, grid 5-150 | FSPS `agn_dust.f90`, `$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat` |
| Temple+21 accretion disc (Wang+24) | `fagn_bbb` (0) | F_nu,disc / F_nu,\* at rest 5500 A (dust-free young + old stellar spectrum) | `ceridwen/agn/data/temple21.txt` (Prospector's `temple.txt`) |
| AGN dust screen | `dust4` (0), `dust4_index` (-1.0) | extra screen on the AGN: `exp(-dust4 (lam/5500)^dust4_index)` times the galaxy's diffuse curve; `AGNConfig(dust4_type=1)` | Prospector `fake_fsps.attenuate` |
| AGN narrow lines | `agn_elum` (0), `agn_eline_sigma` (100) | L_Hbeta [L_sun] per formed M_sun; Gaussian width [km/s]; 23 lines with Richardson+14 'a42' ratios by FSPS name | Prospector `AGNSpecModel` |

L_bol,\* is the dust-free stellar spectrum integrated over frequency on the model grid (the
DL07 trapezoid weights), per formed mass, so every AGN component scales with the SFH's formed
mass like the stars do. Static options of `AGNConfig`: `agn_mode`, `torus`/`disk`/`lines`
(`None` = the mode's default: torus only in `fsps`, all three otherwise), `dust4_type` (0 or
1; 2 is refused as in Prospector), `tau_interp` (`"linear"`, FSPS; `"log"` in corrected mode
only), `agn_energy_balance` (corrected mode; `None` = on with dust emission), `torus_file`.

**Sampling.** `fagn` and `fagn_bbb` span decades: sample their logarithms with transforms,

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

`log_fagn` and `log_fagn_bbb` are new names, so they also go into `free_param_init`
(`SedModel(..., priors=priors, transforms=transforms, free_param_init=free_param_init)`);
without it `SedModel` refuses their priors. With `add_neb=True` and `add_dust_emission=True`
the model also samples `gas_logz`, `gas_logu`, `duste_qpah`, `duste_umin`, `duste_gamma`,
the birth-cloud dust `tau_pow` and `alpha_pow` (and, with the AGN lines on,
`agn_eline_sigma`): give each a prior or fix it with a constant transform, because nested
sampling refuses a free parameter without a prior. Dust emission needs both dust screens:
`add_dust=True` (the birth clouds) and `add_diffuse_dust=True` (both are on by default).
`examples/agn_demo.py --fit` is a complete fit.

**What photometry recovers.** With dust emission and energy balance on, the DL07 re-emission
of the stellar dust competes with the torus in the mid-infrared: `log_fagn` is anti-correlated
with the diffuse optical depth, and `logmass` trades against the SFH shape. In newcomer tests
with UV-to-24 µm photometry at S/N 20, `agn_tau` was recovered within 1 sigma; `log_fagn`
and `logmass` were recovered in one mock but came out 0.25 and 0.3 dex low (outside the 95 %
range) in another, on both the GPU and the CPU, while the maximum-likelihood sample sat at the
truth. The disc (`log_fagn_bbb`) is weakly constrained by broad bands. Check the corner plot
before quoting an AGN fraction from photometry alone.

## Fitting the AGN line width

`agn_eline_sigma` is the Gaussian dispersion [km/s] of the 23 AGN narrow lines. In the
corrected mode they are painted with width sqrt(agn_eline_sigma^2 + sigma_inst^2), the
instrument at each line (`SpectralProjector.setup_agn_lines`), on every pixel: no window
truncates them, and the gradient in `agn_eline_sigma` is finite down to 1e-3 km/s
(`tests/agn/test_eline_sigma.py`).

**Allowed values.** A prior, a constant transform or a fixed value must lie in (0, 2000] km/s
(2000 = the `Kinematics.sigma_max` default); `SedModel` refuses anything else at setup. These
are narrow-line-region lines with Richardson+14 narrow-line ratios: broad-line-region widths
(thousands of km/s) are the BLR component's (`AGNConfig(blr=True)`, `blr_sigma`, below).

**When to sample it, when to fix it.** Only a Spectrum that resolves the AGN lines constrains
the width: Lines fluxes do not depend on it, and Photometry only through line light near a
filter edge. `SedModel` warns at setup when it has a prior but no Spectrum, no unmasked
pixel at an AGN line, or an instrumental sigma above 3x the prior's upper edge at every AGN
line. Then fix it:

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

The warning is a floor, not a guarantee: a prism at R ~ 100 with `Uniform(50, 500)` passes it
(1273 < 3 x 500) and the width is still prior-dominated (below). Sample the width with
a grating, fix it with a prism.

**Recovery on CPU** (`scripts/agn_injection_recovery.py --spec grating|prism`; z = 3, the
JADES-like photometry plus a noiseless spectrum of rest 3600-6900 A at continuum S/N 20 per
pixel; corrected mode with nebular emission; truths agn_elum = 3e-3, agn_eline_sigma = 250,
sigma_gas = 60; all free with every stellar and continuum-AGN parameter; nested sampling,
`slice_kernel="lanes"`):

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

`agn_eline_sigma` posterior / prior standard deviation: 0.029 with the
grating, 0.85 with the prism (prior-dominated). With the grating the line
parameters are recovered to a few per cent; `logmass` sits just outside its 95 % interval
(97.5 % at 9.999; 9.991 with num_live 100) although the truth has chi^2 = 0 (the best
sample of the num_live 100 run has chi^2 = 0.9): the SFH directions projected onto the mass
in the marginal, not a missed likelihood peak. `agn_elum` is per formed solar mass, so it trades with `logmass` (correlation -0.92).
Posteriors: `sigma_<spec>[_nl400].npz` written by the script.

**Degeneracy with sigma_gas.** The AGN lines sit on the same wavelengths as the star-forming
lines (H beta, [O III], H alpha, [N II], [S II], ...), so a line is the sum of two Gaussians,
sqrt(sigma_gas^2 + sigma_inst^2) and sqrt(agn_eline_sigma^2 + sigma_inst^2) wide. The data
fix the blended profile; how it splits into the two widths depends on how much of each line
is AGN. With the prism the two widths are both unconstrained (sigma_gas 139 +85/-76 km/s
against a truth of 60, agn_eline_sigma posterior / prior width 0.85); with the grating
the two widths are measured separately (agn_eline_sigma 252 +3/-4, sigma_gas 60.4 +2.4/-2.3
km/s) with a posterior correlation of only -0.17: the AGN and the nebular lines carry
different ratios, so different lines constrain different mixtures of the two widths.

**What separates AGN from star formation is the line RATIOS, not the widths.** The AGN lines
carry fixed Richardson+14 'a42' ratios (strong [O III]/H beta, [N II]/H alpha, [S II], [O I]),
the nebular lines the CLOUDY ratios of the gas parameters, so `agn_elum` is set by the
ratios: in the prism fit above the width is prior-dominated while `agn_elum` is recovered
(3.68e-3, 16/84 % 3.06e-3 to 4.35e-3, truth 3e-3). Do not read an AGN detection off a
width.

## The three modes

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

Without dust emission the corrected mode loses the absorbed AGN light exactly as absorbed
starlight is lost. FSPS and Prospector have no AGN energy balance, so `agn_energy_balance=True`
is refused in the reproduction modes. AGN lines need `add_neb=True` in the reproduction
modes (Prospector paints them only near nebular lines); the corrected mode paints them
without the nebular model.

## Parity with FSPS and Prospector (AGN_PLAN §4 A)

References: python-fsps 0.5.0 (libfsps `05b5e55`) and prospect `91f8e13`, written by
`scripts/agn_reference/make_refs.py` into `tests/reference/agn/*.npz`; tests in `tests/agn/`.

| test | what | result |
|---|---|---|
| A1 | fsps-mode torus vs python-fsps `spec(fagn) - spec(0)`: fagn {1e-3, 0.1, 1, 3} x agn_tau {5, 7.5, 10, 40, 150} x dust2 {0, 0.5, 1.5} x z {0, 2, 7} | max rel 1.7e-8, inside FSPS's own cancellation floor (REF's rule); a float64 numpy port of FSPS agrees to the same floor |
| A2 | prospector-mode torus vs `fake_fsps.agn_torus`, agn_tau 1-300 (extrapolation) | max rel 3.1e-14 |
| A3 | disc + dust4 screen vs `agnssp_basis.get_spectrum` (140 configs, dust_type 0/2/4/6, dust4_type 0/1, z 0/6) | max rel 5.7e-14 |
| A4 | AGN lines vs `AGNSpecModel` (agn_eline_sigma 50, 300) | luminosities exact; spectrum 1.2e-13 of peak; Lines 8.9e-18; photometry 1.7e-16 with sedpy's curves (6e-5 with CERIDWEN's vendored NIRCam curves, a filter-data difference) |
| A5 | whole model vs python-fsps + `AGNPolySpecModel` (z = 3, 15 NIRCam + MIRI bands) | torus spectrum 1.7e-14 (Prospector's L_bol fed in); disc spectrum 1.7e-14 in shape; full-CSP AGN-only photometry = Prospector's x the galaxy scalars to 1.1e-3 (the filter-curve term, the same on the galaxy) |

A5 cannot be one number: the AGN-only difference depends on the galaxy through two scalars
that two SPS codes compute differently by design — L_bol (the torus) and the galaxy at
5500 A (the disc normalisation) — and on the filter curves. On the A5 SFH with the BPASS test
grid: CERIDWEN / Prospector L_bol = 0.8515, disc normalisation 0.8300, flux normalisation
0.99447 (cosmology implementation). With those applied the whole CSP reproduces Prospector to
the filter term.

**Prospector counts the torus twice.** `AGNSSPBasis` passes `fagn` / `agn_tau` to python-fsps,
whose libfsps always adds its own torus (`add_agn_dust = 1` is a Fortran default python-fsps
cannot switch off), and then adds `fake_fsps.agn_torus` on top: the AGN-only torus photometry
of real Prospector is 1.39-1.90x the torus term alone. The prospector mode reproduces the
torus term alone (Prospector's evident intent); `tests/agn/test_a5_whole_model.py` records the
size of the double count.

## Corrected-mode physics (AGN_PLAN §4 B)

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

The Nenkova+08 SEDs integrate to 0.989-0.996 of the file's own trapezoid on any dense grid
(the log-log SED is not the trapezoid of its nodes); the corrected mode renormalises to
exactly `fagn * L_bol`, the fsps mode keeps FSPS's normalisation (a 0.4-1.1 % difference).

## AGN2: the CIGALE standard (SKIRTOR, polar dust, Feltre+16 NLR, BLR)

Four static switches of `AGNConfig`, all off by default (`AGNConfig()` is unchanged, byte for byte):

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

Interpolation is multilinear in the tabulated variables (log line ratios for the NLR), clamped at
the grid edges; a fixed value outside a grid warns at setup and a **prior** reaching outside it
raises (`SedModel`, as for `gas_logz`). The SKIRTOR (227 MB) and NLR tables are not shipped: the
converters `scripts/agn2/build_skirtor_h5.py` and `build_nlr_h5.py` of the source repository write
`$CERIDWEN_AGN_DATA/skirtor2016.h5` and `nlr_feltre16.h5` (default `~/.ceridwen/agn/`); a missing
file is a teaching error at construction. The tests that need them skip with "MISSING ...: did NOT
run". Data licences (SKIRTOR, Feltre+16) have not been checked for redistribution.

### CIGALE extinction curves

The polar dust reads CIGALE's three dust-model curves below 100 nm (`extFun_SMC.dat`,
`extFun_MRN.dat`, `extFun_Gaskel04.dat`, from `pcigale/sed_modules/curves/` at cigale 29cb909f).
They are CIGALE's files under CeCILL-2.0, so ceridwen does not ship them. Put them into
`$CERIDWEN_AGN_DATA` (default `~/.ceridwen/agn/`) once:

```bash
python -m ceridwen.agn.fetch --from /path/to/cigale   # a CIGALE checkout or installed pcigale; no network
python -m ceridwen.agn.fetch --download               # from gitlab.lam.fr, through the system curl
```

or from Python, `ceridwen.agn.fetch_grids(from_cigale=...)` / `fetch_grids(download=True)`. Every
file is checked against the sha256 of the table the parity tests were built against, and a file
that differs is never written. Copying them by hand works too: either CIGALE's name or the former
`cigale_extFun_*.txt` is read. Without them `polar_dust=True` raises at construction with these
commands; with a changed table it raises naming the file. The tests that need them skip.

**What is refused.** The AGN2 switches in `"fsps"` / `"prospector"` modes; the Temple disc with
SKIRTOR (SKIRTOR carries its own disc); `dust4_type=1` with SKIRTOR (its nuclear screen is the
polar dust); `polar_dust=True` without SKIRTOR; in `"cigale"` mode the Nenkova torus, the Temple
disc, the dust4 screen and the a42 lines (CIGALE has none of them), a CSP without any galaxy dust
screen with SKIRTOR (CIGALE's `fagn` is relative to the galaxy's dust luminosity) or with
`fesc_geometry="picket"` (that luminosity is computed with the screen geometry), and a `fagn`
prior reaching 1. `marginalize_elines=True` with AGN lines (narrow or broad) is refused as before.

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

`PostProcess` reports, with SKIRTOR, `extras["agn"]["frac_agn_cigale"]` (CIGALE's fracAGN, both
modes; equal to `fagn` in `"cigale"`), CIGALE's `agn.*` infos as `cigale_<name>` (accretion
power, intrinsic L_nu at 2500 A at 30 deg, L_6um, L_12um, disc / torus / polar luminosities),
`L_dust_gal`, `L_skirtor_disk`, `L_skirtor_dust` and `L_polar_ir`.

### Parity with CIGALE and the literature (`tests/agn2/`)

| test | what | result |
|---|---|---|
| L1 | `"cigale"` SKIRTOR (disc, torus separately) vs pcigale `skirtor2016` (CIGALE 29cb909f), 71 node sets incl. both sides of the type-1/2 boundary, CIGALE's 948-point grid | max rel 1.6e-14 where flux > 1e-6 x peak |
| L2 | the template files (Stalevski+16): torus-dust shape at 40 nodes; the disc anisotropy cos i (2 cos i + 1) | 5.1e-15; 1.5e-9 |
| L3 | with polar dust: 218 sets (3 laws, E(B-V) 0.03-3, T 50-300 K, i 30/70, oa 20/40/60) vs pcigale | max rel 1.7e-12 |
| L4 | Feltre+16 NLR through the CSP vs the public grid files, all nodes; BPT (their Fig. 2) | 4.4e-16 dex |
| L5 | BLR: line ratios vs the Case-B table; the profile vs CIGALE's AGN-line Gaussian | ratios exact; profile as in the table above |
| corrected SKIRTOR energy | torus + polar IR luminosity vs `fagn` x L_bol,\* on the 15000-point test grid | 0.9e-6 to 5.5e-6 (the model grid's quadrature) |

**Recommended priors** (PLAN_AGN2; BEAGLE-AGN, Vidal-Garcia+24, for the NLR): `fagn` Uniform(0,
0.99) in `"cigale"`, `log_fagn` Uniform(-5, log10 3) in `"corrected"`; fix `skirtor_t` 7,
`skirtor_pl` 1, `skirtor_q` 1, `skirtor_oa` 40 without mid-IR coverage, `skirtor_i` Uniform(0, 90);
`polar_ebv` Uniform(0, 1); `nlr_logu` Uniform(-4, -1.5), `nlr_logz` Uniform(-2.18, 0.24), with
`nlr_xid` 0.3, `nlr_alpha` -1.7 and `nlr_lognh` 3 fixed (they are degenerate with log U and the
amplitude); `blr_sigma` LogUniform(400, 3000) km/s (JWST broad H alpha FWHM 890-6279 km/s).
Only a resolved spectrum constrains `blr_sigma`; line fluxes alone do not.

## Fixed by design — four things the AGN model does not do

These are deliberate, not omissions in progress. Each says what the code does, and what an
option would have to add.

### The line luminosities are absolute, and `fagn` does not scale them

`agn_elum` (narrow lines, both the Richardson+14 `a42` set and the Feltre+16 NLR) and `blr_lum`
(broad lines) are **luminosities per formed solar mass**, not fractions:
`FeltreNLR.line_luminosity` returns `agn_elum * R_line(theta)` (`ceridwen/agn/nlr.py:283`),
`AGNLines` returns `agn_elum * ratio` (`lines.py:154`) and `BroadLines` returns
`blr_lum * ratio_halpha` (`blr.py:160`). None of the three reads `fagn`, `fagn_bbb` or any
continuum quantity; the CSP multiplies them by the SFH's formed mass `sum(W)` and then by
`10**logmass`, exactly like the stellar light.

Consequences:

* **Broad or narrow AGN lines without an AGN continuum are a legal model** (`fagn = 0`,
  `fagn_bbb = 0`, `blr_lum > 0`). Nothing couples the two, so a fit can put a BLR on a galaxy
  whose continuum shows no AGN at all. That is a modelling choice you make with the priors, not
  something the model forbids.
* Without the energy balance, `agn_elum` and `blr_lum` are constrained only by the lines, so
  photometry with no resolved line fixes them barely at all. **With `agn_energy_balance=True`
  they also reach the broadband IR**: the line light the galaxy's dust absorbs joins the DL07
  reservoir and is re-emitted (`hook.py:293-304` for the narrow lines, `hook.py:337-349`
  `blr_absorbed` for the broad ones), so IR photometry constrains them too.
* Typical scale to judge a prior against: the galaxy's own nebular H beta is of order
  `1e-3` to `1` L_sun per M_sun formed, so `agn_elum = 1` is an NLR H beta comparable with the
  whole galaxy's. (That range is the one quoted in the AGN2 brief; it is not recomputed here.)

An option (`agn_elum` tied to the AGN continuum, as CIGALE ties its AGN lines to the ionising
photon rate) would need: an ionising-photon integral of the nuclear SED on the model grid, a
Case-B (or Cloudy-table) conversion to H beta, a covering factor as a new theta key, a config
switch that removes `agn_elum` from `param_names`, and a new golden configuration — the lines
would then stop being linear in a single amplitude, which several AGN2 tests assert.

### The NLR is a fixed Feltre+16 grid, not the model's own disc

The narrow-line ratios come from the tabulated Feltre, Charlot & Gutkin (2016) grid at its own
five axes (`nlr_logu`, `nlr_logz`, `nlr_xid`, `nlr_lognh`, `nlr_alpha`; `nlr.py:54-60`).
`nlr_alpha` is the **grid's** accretion-disc UV slope, a free theta key of the NLR — it is not
read from the disc component: the SKIRTOR torus has no slope parameter at all
(`AGN2_THETA_KEYS["skirtor16"]`, `ceridwen/agn/__init__.py:155`) and the Temple+21 disc's only
parameter is its amplitude `fagn_bbb` (`disk.py:80`). The NLR therefore never sees the
luminosity or the shape of the continuum the same fit puts in.

An option would need an ionising continuum predicted on the Feltre grid's own axes — i.e. a
disc model parameterised by `alpha` — plus the photon-rate normalisation of the point above;
with the tabulated grid the most one can do cheaply is tie `nlr_alpha` to a disc slope through
a `transforms` entry once a disc with a slope exists.

### The BLR does not depend on inclination

`BroadLines` has exactly two parameters, `blr_lum` and `blr_sigma` (`blr.py:112`), and the
transmission the hook gives it is the nuclear screen (polar dust, or the `dust4` power law) times
the galaxy's diffuse curve in `"corrected"` (`hook.py:315-327`). `PolarDust.transmission` depends
only on `polar_ebv` (`polar.py:195-199`): `skirtor_i` does not enter. This matches CIGALE, whose
line module has no inclination term either.

Note the asymmetry it creates, which is also CIGALE's: the SKIRTOR **disc** receives the polar
screen only for a type-1 view (`skirtor_i <= 90 - skirtor_oa`, `skirtor.py:405-406` and
`SkirtorTorus.nuclear_parts`), while the **broad lines** receive it at every inclination. A
type-2 geometry in which the broad lines should be hidden altogether is not represented: put
that in the prior on `blr_lum`, not in `skirtor_i`.

An option would need a BLR covering/obscuration factor as a function of `skirtor_i` (CIGALE has
none to copy, so it would be a new model, e.g. the lines x a step or smooth function of
`skirtor_i - (90 - skirtor_oa)`), a decision about whether the hidden light is re-emitted, and
a golden configuration; `blr_lum` would stop being the observed broad H alpha.

### There is no X-ray or corona component

No component is tied to an X-ray luminosity: there is nothing like CIGALE's `xray` module and
nothing normalised through `alpha_ox`, so an AGN fit neither predicts an X-ray flux nor can be
constrained by one.

What the components cover blueward of the Lyman limit differs, and is not "nothing":

* the **Temple+21 disc** template ends at 91 A, and `"corrected"` sets the disc to exactly zero
  blueward of it rather than extrapolating a flat `F_nu` (`disk.py:98-100`);
* the **SKIRTOR parametric disc** is defined from **8 A** (`skirtor_disk` /
  `schartmann2005_disk`, `skirtor.py:113-116`: the broken power law's first limit is 8 A), so it
  does carry flux in the EUV on the templates' own grid.

What survives into a prediction is then set by the model grid's blue end (the SSP grid's first
wavelength) and by the IGM, not by the AGN model.

An option (a Corona/X-ray power law, as CIGALE's `xray` module has) would need the component
itself, its own amplitude convention (CIGALE uses `alpha_ox`, which couples it to the 2500 A
disc flux — so it would also need the disc's monochromatic luminosity), an extension of the
model grid blueward of the SSP grid's first wavelength (so the grid, the IGM and the filter
projection all come in), and a golden configuration.

## What the AGN changes elsewhere

* **PostProcess**: `spectra_model` and `spectra_observed` include the AGN continuum;
  `spectra_intrinsic` and `spectra_dustfree` do not (so `A_V = 2.5 log10(dustfree/full)`
  contains the AGN). New: `spectra_agn` (the AGN continuum alone), `extras["agn"]` with
  `L_bol_star`, `L_absorbed` and, with the energy balance, `mdust` (the dust mass the AGN
  heats; the stellar dust mass is not reported). `LUV`/`MUV` include the AGN. The AGN narrow
  lines reach only the per-observation predictions.
* **Result files** record the AGN in `/model@csp_config["agn"]` (mode, components, energy
  balance, template and table hashes); a model with a different AGN set-up differs in
  `check_model_against_result`.
* **Line marginalisation** (`marginalize_elines=True`) with AGN lines is refused: the
  marginalised fluxes would absorb the AGN lines. Switch the lines off
  (`AGNConfig(lines=False)`) to marginalise.
* `fagn` without `agn=` is an unknown theta key (a warning, and no AGN): build the CSP with
  `agn=`.
* GPU: the AGN paths are dense (no scatter-add), so CPU tolerances hold.
