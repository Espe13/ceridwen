# Recipes: Prospector features without package changes

Small, JIT-safe, float64 add-ons written against the post-rebuild working-tree API (`logzsol`, `logzsol_hist`, ...). None of them changes `ceridwen/`. Each one prepares a package feature: when that feature lands, it must reproduce the numbers below exactly, and the recipe's check script stays as its acceptance test. Four have landed: `igm_damping_dla.py` and `extra_dust_laws.py` (2026-09-22) now re-export the package code, and their checks run against it; `map_fit.py` is `ceridwen.optimize.map_fit` / `fitSED(optimize=True)`, and `loguniform_prior.py` is `ceridwen.priors.LogUniform`. Checks run on CPU in minutes.

Setup for the checks:

```bash
git clone https://github.com/bd-j/prospector && git -C prospector checkout a78d153
git clone https://github.com/cconroy20/fsps fsps_src           # tested at bd187a0
export PROSPECTOR_DIR=$PWD/prospector FSPS_SRC=$PWD/fsps_src SPS_HOME=/path/to/fsps
JAX_PLATFORMS=cpu python examples/recipes/tests/check_<name>.py
```

A missing `PROSPECTOR_DIR` or `FSPS_SRC` prints FAIL, never a silent skip. Import a recipe with `sys.path.insert(0, "examples/recipes")`; there is no package `__init__`.

| Recipe | What it gives you | Check (22 Sep 2026, CPU) | Package change it prepares |
|---|---|---|---|
| `derived_quantities.py` | `make_derived(csp, filters=, colors=)` returns a `PostProcess(derived=...)` dict: mass-weighted age, t50, t90, rest-frame absolute AB magnitudes and colours. The SFH integrals reuse `PostProcess`'s own helpers (`postprocess._per_bin_and_nodes`, `_mean_sfr_window`, `_formed_mass`). | **PASS**. Cumulative mass = `mass_formed` to 2.2e-16; constant SFH gives mwa = t50 = T/2 exactly; mwa/t50/t90 = Prospector formulas to 4.4e-16; magnitudes vs Prospector `absolute_rest_maggies` 1.7e-11 mag (10 bands). | Built-in derived quantities in `PostProcess`, with the private SFH helpers made public. |
| `map_fit.py` | `map_fit(model, n_starts=)`: L-BFGS (optax) on `fitSED`'s own log-posterior, in the NUTS adapter's unconstrained space. The result `.theta` is ready to use as `free_param_init`. `laplace_sigma` gives inverse-Hessian widths. | **FAIL** (1 of 2). The MAP ln p of 38190.27 beats the best of 1000 prior draws (37094.58). The "all within 1σ of truth" check fails: 8/9 within 1σ, logzsol pull +1.0013, 9/9 within 3σ. Wall time 81 s for 16 starts, compile included. | Done: `ceridwen.optimize.map_fit` and `fitSED(optimize=True)` (the MAP starts NUTS and is stored under `/map`). |
| `igm_damping_dla.py` | **Promoted to the package** as `ceridwen.igm.MadauDampingDLA` (2026-09-22); the recipe re-exports it. Madau (1995) × IGM damping wing × DLA Voigt profile, with `x_HI`, `logN_HI`, `z_dla` as theta keys (fixed or sampled). | **PASS** 63/63 against the package code. Damping wing vs Prospector `tau_damping` ≤ 1.0e-12; DLA vs `voigt_profile` 1.1e-16; `x_HI=0` and `N_HI→0` give Madau byte-identically; `jax.grad` through the CSP is finite. | Done, except serialising the IGM model in `ceridwen_result.h5` (`fit.py` records no IGM information for any model). |
| `extra_dust_laws.py` | **Promoted to the package** (2026-09-22): `gordon03_smcbar` (FSPS dust_type 5) and `reddy15` (FSPS dust_type 6) are registered in `ATTENUATION_LAWS`; the recipe re-exports them and keeps `register()` for old scripts. | **PASS** 14/14 (0 skipped) against the package code. Both accepted by `Dust` and `DiffuseDust`; Gordon+03 vs the FSPS table and Fortran is 0.0 / 1.1e-16; Reddy+15 vs Prospector is 1.1e-15. The normalisation at 5500 Å is 1 (Gordon) and 0.997113 (Reddy, = FSPS `dust2`). | Done, with the `noll` / `drude` / `smc` / `lmc` registry fixes (`CHANGES_sept2026.md`). |
| `loguniform_prior.py` | `loguniform(name, lo, hi)` / `loguniform_setup(...)`: a `Uniform` on `log10_<name>` plus a transform `name = 10**log10_<name>`. Also the LogNormal `mode` conversion to Prospector's. | **PASS** 5/5. KS vs Prospector `LogUniform`, 1e5 draws: D = 0.0057, p = 0.074; `_detect_bounds` sees the bounds; LogNormal log-pdfs match after conversion to 7e-15. | Done: native `ceridwen.priors.LogUniform(mini, maxi)` (samples `x` itself, bounds seen by `_detect_bounds`; `tests/test_loguniform_prior.py`), and the `LogNormal` parametrisation documented in `GOTCHAS.md` section 16. The recipe still works when you want the posterior in `log10 x`. |
| `make_reference_mfrac.py` → `reference_mfrac.json` | FSPS surviving-mass fractions for constant / rising / burst SFHs at 0.1, 1 and 10 Gyr, on BPASS (the test grid) and MIST. | **PASS** 9/9 (`check_reference_mfrac.py`). BPASS values recomputed from the raw `bpass.mass` table agree to ≤ 2.8e-9 (tolerance 1e-7). | Surviving stellar mass in `PostProcess` (stellar mass vs formed mass). |
| `gp_likelihood.py` | A package feature, not a Prospector port: a mock spectrum with GP-correlated noise (a = 1, l = 30 A), fitted with `log_gp_amp_spec` / `log_gp_length_spec` sampled, then with the diagonal likelihood for comparison (`docs/gp_likelihood.md`). | Covered by `tests/likelihood/test_gp_likelihood.py` (profile peaks at the truth, short nested fit); the recipe itself has no check script. | Done: `GPGaussianLikelihood` in `fitSED`. |
| `poly_marginal_calibration.py` | Usage of a package feature (not a recipe to promote): one mock spectrum with a known order-3 Chebyshev calibration, evaluated with the profiled (`polynomial_order=3`), the marginalised (`polynomial_mode="marginalize"`, `polynomial_prior_sigma=0.1`) and no polynomial, with the conditional coefficients. | Runs (23 Sep 2026, CPU, test BPASS grid): ln L 17587.124 / 17572.082 / 17080.367; conditional c = (+0.0284, +0.0614, -0.0362, +0.0263) +- (0.0019, 0.0033, 0.0029, 0.0028) vs injected (+0.030, +0.060, -0.040, +0.025). | Done: `ceridwen/likelihood/poly_marginal.py`. |
| `PAPER_CORRECTIONS.md` | 16 items where `Method_Paper/mnras_template.tex` disagrees with the working-tree code, including which paper numbers depend on Prospector's ln 2π likelihood bug. | n/a | Paper revision. |

## Findings for the package

- Fixed 2026-09-22 (`CHANGES_sept2026.md`): the `noll` registry name `E_bump` vs `Ebump`; the `drude` law fed Å where it expects inverse microns (it still has no amplitude parameter); the `smc` / `lmc` registry text (Pei 1992, normalised at 5500 Å).

Reported, not fixed:

- `ceridwen.priors.LogNormal.scale` (`sampler/priors.py:297-299`) returns Prospector's `exp(mode+σ²)`, which is not the scale of the `tfd.LogNormal(loc=mode)` that is actually sampled.
- Photometry: the gridded `FilterSet` projection used for fit predictions differs from sedpy / `Filter.ab_mag` by up to 1.5e-3 mag (sdss_u0 on a BPASS SSP). `Filter.ab_mag` itself equals sedpy exactly.
- L_sun: CERIDWEN and FSPS use 3.839e33 (`postprocess._LSUN_ERG_S`, FSPS `sps_vars.f90:422`); Prospector uses 3.846e33 (`prospect/sources/constants.py:15`). The same L_sun/Hz array therefore gives magnitudes 1.98 mmag apart.
- `BlackJAXNUTSAdapter._flatten` (`sampler/nuts.py:121-131`) concatenates in dict order. `jax.vmap` returns dicts with sorted keys, so a future caller flattening inside `vmap` would pair values with the wrong bounds. It is harmless today, because NUTS flattens outside any transform.
- Prospector at a78d153: `add_dla` converts to the absorber frame upside down (`sedmodel.py:816`: `wave_rest*(1+dla_z)/(1+zred)` should be `*(1+zred)/(1+dla_z)`), so a foreground DLA lands redward of Lyα. Its plain `NoiseModel` multiplies χ² by ln 2π (`noise_model.py:90-91`).
