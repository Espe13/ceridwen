#!/usr/bin/env python
"""
Ceridwen quick-start: fit mock UV-to-IR broadband photometry end-to-end.
==========================================================

This is a self-contained, runnable demo:

  Step 0  load an SSP grid (built from FSPS only if none is found)
  Step 1  build the CSP forward model
  Step 2  generate MOCK photometry from known "true" parameters
  Step 3  fit it back with BlackJAX nested sampling
  Step 4  post-process with PostProcess: recovered vs. true, and the summary,
          corner and sampling-diagnostic figures in quickstart_figures/

Requirements
------------
This demo reads no FSPS data files (``add_neb=False``: no CLOUDY data is
read). An SSP grid is resolved in this order: ``$SSP_FILE`` ->
``examples/ssp_data.h5`` (build it with FSPS or download it from Zenodo, see
docs/installation.md "SSP grids") -> the local
developer grid ``ceridwen/data/test_data/ssp_data_bpass.h5`` (not shipped in
the repository). Only if no grid is found
does the script fall back to building one, which then does need FSPS +
``$SPS_HOME``. Flipping ``add_neb=True`` (CLOUDY nebular emission) also needs
``$SPS_HOME`` at runtime.

The fit takes a few minutes on an unloaded laptop CPU (much faster on GPU;
longer on a busy machine).  It uses DEMO settings and is NOT a converged fit: it checks
that the installation works, and its posteriors are wider and noisier than a real fit's
(see DEMO_NOTICE for the settings a science fit needs).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

# 64-bit floats are required for accurate Bayesian evidence estimates.
jax.config.update("jax_enable_x64", True)

from ceridwen import SSPData, CSPBasis, SedModel, PostProcess
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT
from ceridwen.likelihood import DiagonalGaussianLikelihood, MultiObservationLikelihood
from ceridwen.sampler import run_sampler
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
from ceridwen.cosmology import Cosmology

HERE = pathlib.Path(__file__).resolve().parent


def _default_ssp_file() -> str:
    """Resolve a usable SSP grid without requiring FSPS.

    Order: $SSP_FILE -> examples/ssp_data.h5 -> the local developer grid
    ceridwen/data/test_data/ssp_data_bpass.h5 (not shipped in the
    repository; download the BPASS grid from Zenodo or build it with
    FSPS if you want this fallback).
    """
    env = os.environ.get("SSP_FILE")
    if env:
        return env
    local = HERE / "ssp_data.h5"
    if local.is_file():
        return str(local)
    dev_grid = (HERE.parent / "ceridwen" / "data" / "test_data"
                / "ssp_data_bpass.h5")
    if dev_grid.is_file():
        return str(dev_grid)
    return str(local)   # absent: step0 falls through to the FSPS build


SSP_FILE = _default_ssp_file()
SPS_HOME = os.environ.get("SPS_HOME")
RNG = jax.random.PRNGKey(42)

# A broad UV->IR set (GALEX + SDSS + 2MASS + WISE) so the mock is well
# constrained and the corner plot is informative. All ship with ceridwen
# (ceridwen/data/filters).
FILTERS = [
    "galex_FUV", "galex_NUV",
    "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
    "twomass_J", "twomass_H", "twomass_Ks",
    "wise_w1", "wise_w2",
]
N_FILTERS = len(FILTERS)


def step0_load_or_build_grid() -> SSPData:
    """Load the cached SSP grid, building it from FSPS on first run."""
    if pathlib.Path(SSP_FILE).is_file():
        print(f"[Step 0] loading cached SSP grid: {SSP_FILE}")
        return SSPData.load(SSP_FILE)

    try:
        import fsps  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "No SSP grid found and FSPS (python-fsps) is not importable.\n"
            "Easiest fix (no FSPS needed): fetch the published grid and point "
            "$SSP_FILE at it,\n"
            "  SSP_FILE=$(python -c \"from ceridwen.ssps import fetch_grid; "
            "print(fetch_grid('mist_miles_chab'))\") python examples/quickstart.py\n"
            "or download it from Zenodo to examples/ssp_data.h5 — see "
            "docs/installation.md ('Getting the SSP grid').\n"
            "Alternatively install FSPS with `pip install 'fsps>=0.4.4'` and "
            "set $SPS_HOME to build the grid locally. See the README."
        ) from exc

    print(f"[Step 0] building SSP grid from FSPS -> {SSP_FILE} (one-off, ~minutes)")
    return SSPData.from_fsps(save_to=SSP_FILE, imf_type=1)


DEMO_NOTICE = """
======================================================================
 DEMO SETTINGS -- THIS IS NOT A CONVERGED FIT.
 The sampler runs with deliberately small settings (num_live=150,
 logZ_tol=-2) so the demo finishes in minutes on a laptop CPU.  It checks
 that the installation works; the posteriors are wider and noisier than a
 real fit's.  For science use num_live >= 500, num_inner_steps >= 5 x the
 number of parameters and the default logZ_tol (-5): much longer on a CPU,
 fast on a GPU.
======================================================================
"""


def main() -> int:
    print(DEMO_NOTICE)
    ssp_data = step0_load_or_build_grid()

    # ---- Step 1: forward model -------------------------------------------
    T_UNIV = 13.8          # age of the universe [Gyr]
    N_TIME = 5             # SFH bins -> N_TIME - 1 = 4 free logsfr_ratios
    #                        (must match the length of TRUE_LOGSFR_RATIOS below)
    lookback = jnp.linspace(0.0, T_UNIV, N_TIME)   # today @ index 0

    csp = CSPBasis(
        ssp_data,
        theta={"lookback_time": lookback, "sfh": jnp.ones(N_TIME),
               # logzsol = log10(Z / Z_sun) with the SSP grid's OWN Z_sun (0 = solar).
               # Every shipped grid covers at least [-2.3, +0.3]; csp.zmet prints the axis
               # and csp.zsun_nominal the Z_sun it was resolved to.
               "logzsol": jnp.array([-0.2])},
        zh_const=True,
        sfh_interp="step",
        add_dust=False,
        add_diffuse_dust=True,
        add_neb=False,        # set True to include CLOUDY nebular emission
        sps_home=SPS_HOME,
        verbose=False,
        cosmo=Cosmology.planck18(),
    )

    # ---- Step 2: mock photometry from known truth ------------------------
    TRUE_LOGSFR_RATIOS = jnp.array([+0.3, +0.2, -0.1, -0.5])
    TRUE_Z = jnp.array([-0.2])      # logzsol = log10(Z/Z_sun): 0.63 Z_sun
    TRUE_LOGMASS = jnp.array([10.5])
    TRUE_DIFFDUST = jnp.array([0.5])
    TRUE_DUST_INDEX = jnp.array([-0.7])
    SNR = 10.0

    sfh_true = logsfr_ratios_to_sfh(
        TRUE_LOGSFR_RATIOS, sfh_times_yr=np.array(csp.sfh_times)
    )
    dummy_phot = Photometry(filters=FILTERS, name="_tmp")
    dummy_phot.setup_for_model(csp.wave)

    spec_unit = csp.get_spectrum(
        {"sfh": sfh_true, "logzsol": TRUE_Z,
         "diffuse_tau_kc": TRUE_DIFFDUST, "diffuse_dust_index": TRUE_DUST_INDEX}
    )
    maggies_unit = np.array(dummy_phot.predict(spec_unit, csp.wave))
    maggies_true = maggies_unit * float(10.0 ** TRUE_LOGMASS[0])
    sigma = maggies_true / SNR
    maggies_obs = maggies_true + np.array(sigma) * np.array(
        jax.random.normal(RNG, (N_FILTERS,))
    )

    phot_obs = Photometry(
        filters=FILTERS, flux=maggies_obs, uncertainty=sigma,
        name="phot",
    )

    # ---- Step 3: model + nested sampling ---------------------------------
    sfh_times_yr = np.array(csp.sfh_times)

    def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
        return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

    priors = {
        "logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
        "logzsol": Uniform(low=-2.0, high=0.2),   # inside every shipped grid's logzsol axis
        "logmass": Uniform(low=9.0, high=12.0),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
    }

    model = SedModel(
        csp,
        observations=[phot_obs],
        priors=priors,
        transforms={"sfh": logsfr_to_sfh},
        free_param_init={"logsfr_ratios": jnp.zeros(N_TIME - 1),
                         "logmass": TRUE_LOGMASS},
        broaden_photometry=False,   # the mock above was made without kinematic broadening
    )

    n_dims = sum(int(jnp.size(v)) for v in model.theta_init.values())
    # Demo settings tuned for a reasonable CPU runtime (~10 min unloaded; longer
    # on a busy machine) and a good-looking corner. Two dials:
    #   * num_inner_steps -- dominates the one-time JIT *compile* time of the
    #     step kernel (it is the unrolled inner MCMC chain) AND per-step cost.
    #   * num_live -- runtime<->quality (more = smoother contours, more steps).
    # For publication use num_live >= 500, num_inner_steps >= n_dims*5, and the
    # library-default logZ_tol (-5.0, ~0.7% of the evidence left in the live
    # set at termination); expect much longer on CPU.  The relaxed -2.0 below
    # is a demo-runtime compromise only.
    adapter = BlackJAXNestedSamplerAdapter(
        priors=model.priors, num_live=150, num_inner_steps=max(8, n_dims * 2),
        logZ_tol=-2.0, verbose=True,
    )
    likelihood = MultiObservationLikelihood(
        keys=("phot",), likelihoods=(DiagonalGaussianLikelihood(),)
    )

    result = run_sampler(model, likelihood, adapter, RNG)
    print(f"\nln Z = {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}")
    print("(demo settings: not a converged fit -- see the notice at the start)")

    # ---- Step 4: post-process --------------------------------------------
    # PostProcess resamples the nested-sampling draws to equal weight, pushes
    # them through the forward model and writes three figures per galaxy.
    TRUTH = {
        "logzsol": float(TRUE_Z[0]),
        "logmass": float(TRUE_LOGMASS[0]),
        "diffuse_tau_kc": float(TRUE_DIFFDUST[0]),
        "diffuse_dust_index": float(TRUE_DUST_INDEX[0]),
    }
    pp = PostProcess(model, result, n_samples=2000)
    out = pp.run()

    print("\nparameter             true     posterior median (16-84%)")
    for p, t in TRUTH.items():
        lo, med, hi = np.percentile(out["theta"][p], [16, 50, 84])
        print(f"  {p:<20}{t:+7.3f}   {med:+7.3f}  (-{med - lo:.3f} / +{hi - med:.3f})")

    # logmass is the mass FORMED; the stellar mass (stars + remnants) is mfrac times it,
    # from the grid's surviving-mass table (published grids carry it)
    sfh = out["extras"]["sfh"]
    rows = [("log mass_formed", np.log10(sfh["mass_formed"]))]
    if "mfrac" in sfh:
        rows += [("mfrac", sfh["mfrac"]), ("log mass_surviving", np.log10(sfh["mass_surviving"]))]
    for name, v in rows:
        lo, med, hi = np.percentile(v, [16, 50, 84])
        print(f"  {name:<20}{'':7s}   {med:+7.3f}  (-{med - lo:.3f} / +{hi - med:.3f})")

    figdir = HERE / "quickstart_figures"
    # the figures also draw the injected SFH, so they get the SFH parameters too
    fig_truth = {**TRUTH, "logsfr_ratios": np.asarray(TRUE_LOGSFR_RATIOS)}
    paths = pp.figures(figdir, title="CERIDWEN quickstart -- demo settings, NOT a converged "
                                     "fit (green = injected truth)",
                       truths=fig_truth)
    for name, path in paths.items():
        print(f"{name:<12} -> {path}")
    pp.save(figdir / "quickstart_post.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
