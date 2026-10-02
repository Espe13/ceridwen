# AGN

`CSPBasis(..., agn=AGNConfig(...))` adds an active nucleus to the forward model: everything
Prospector has for AGN, built so that it can **reproduce** FSPS and Prospector exactly
(`agn_mode="fsps"`, `"prospector"`), and then **correctly** (`agn_mode="corrected"`, the
default). Without `agn=` nothing changes: every existing prediction is byte-identical
(`scripts/bit_identity_check.py`).

```python
from ceridwen import CSPBasis, Cosmology
from ceridwen.agn import AGNConfig

csp = CSPBasis(ssp, lookback_time=lb, zh_const=True, add_neb=True, add_dust_emission=True,
               cosmo=Cosmology.planck18(),
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
```

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
* GPU: the AGN paths are dense (no scatter-add), so CPU tolerances hold; nested-sampling
  timing on GPU is in `REPORT_INT.md`.
