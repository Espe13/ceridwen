#!/usr/bin/env python
"""
Ceridwen quick-start: fit mock SDSS photometry end-to-end.
==========================================================

This is a self-contained, runnable demo:

  Step 0  build (or load) the FSPS SSP grid cache  -> needs FSPS + $SPS_HOME
  Step 1  build the CSP forward model
  Step 2  generate MOCK photometry from known "true" parameters
  Step 3  fit it back with BlackJAX nested sampling
  Step 4  report recovered vs. true parameters (+ optional corner plot)

Requirements
------------
Ceridwen needs FSPS at *runtime*, not just to build the cache: the CLOUDY
nebular grids and the dust-emission templates are read from ``$SPS_HOME``
(the FSPS data directory).  So FSPS must be installed and ``SPS_HOME`` set,
e.g.::

    pip install -e ".[grids,nested]"      # python-fsps + anesthetic
    export SPS_HOME=/path/to/fsps         # the FSPS root (contains nebular/, ...)
    python examples/quickstart.py

The SSP cache is written to ``examples/ssp_data.h5`` the first time and
re-used on subsequent runs.  Set ``$SSP_FILE`` to point elsewhere.

Runs on CPU in a couple of minutes; no GPU required.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

# 64-bit floats are required for accurate Bayesian evidence estimates.
jax.config.update("jax_enable_x64", True)

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.observation.observation import Photometry
from ceridwen.model.model import SedModel
from ceridwen.model.transforms import logsfr_ratios_to_sfh
from ceridwen.likelihood.likelihood import (
    DiagonalGaussianLikelihood,
    MultiObservationLikelihood,
)
from ceridwen.sampler import Uniform, ClippedNormal, StudentT, run_sampler
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter

HERE = pathlib.Path(__file__).resolve().parent
SSP_FILE = os.environ.get("SSP_FILE", str(HERE / "ssp_data.h5"))
SPS_HOME = os.environ.get("SPS_HOME")
RNG = jax.random.PRNGKey(42)

SDSS_FILTERS = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
N_FILTERS = len(SDSS_FILTERS)


def step0_load_or_build_grid() -> SSPData:
    """Load the cached SSP grid, building it from FSPS on first run."""
    if pathlib.Path(SSP_FILE).is_file():
        print(f"[Step 0] loading cached SSP grid: {SSP_FILE}")
        return SSPData.load(SSP_FILE)

    try:
        import fsps  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "FSPS (python-fsps) is required to build the SSP grid and is not "
            "importable. Install it with `pip install -e '.[grids]'` and a "
            "working FSPS build, then set $SPS_HOME. See the README."
        ) from exc

    print(f"[Step 0] building SSP grid from FSPS -> {SSP_FILE} (one-off, ~minutes)")
    return SSPData.from_fsps(save_to=SSP_FILE, imf_type=1)


def main() -> int:
    ssp_data = step0_load_or_build_grid()

    # ---- Step 1: forward model -------------------------------------------
    T_UNIV = 13.8          # age of the universe [Gyr]
    N_TIME = 5             # SFH bins -> N_TIME - 1 = 4 free logsfr_ratios
    #                        (must match the length of TRUE_LOGSFR_RATIOS below)
    lookback = jnp.linspace(0.0, T_UNIV, N_TIME)   # today @ index 0

    csp = CSPBasis(
        ssp_data,
        theta={"lookback_time": lookback, "sfh": jnp.ones(N_TIME),
               # Z is log10 of ABSOLUTE metallicity (= ssp_lgmet), NOT log10(Z/Zsun).
               # This FSPS grid spans roughly [-4.0, -1.4]; solar ~ -1.85.
               "Z": jnp.array([-2.0])},
        tuniv=T_UNIV,
        zh_const=True,
        sfh_interp="step",
        add_dust=False,
        add_diffuse_dust=True,
        add_neb=False,        # set True to include CLOUDY nebular emission
        sps_home=SPS_HOME,
        verbose=False,
    )

    # ---- Step 2: mock photometry from known truth ------------------------
    TRUE_LOGSFR_RATIOS = jnp.array([+0.3, +0.2, -0.1, -0.5])
    TRUE_Z = jnp.array([-2.0])      # log10 absolute Z, inside the grid (~Z/2 Zsun)
    TRUE_LOGMASS = jnp.array([10.5])
    TRUE_DIFFDUST = jnp.array([0.5])
    TRUE_DUST_INDEX = jnp.array([-0.7])
    SNR = 10.0

    sfh_true = logsfr_ratios_to_sfh(
        TRUE_LOGSFR_RATIOS, sfh_times_yr=np.array(csp.sfh_times)
    )
    dummy_phot = Photometry(filters=SDSS_FILTERS, name="_tmp")
    dummy_phot.setup_for_model(csp.wave)

    spec_unit = csp.get_spectrum(
        {"sfh": sfh_true, "Z": TRUE_Z,
         "diffuse_tau_kc": TRUE_DIFFDUST, "diffuse_dust_index": TRUE_DUST_INDEX}
    )
    maggies_unit = np.array(dummy_phot.predict(spec_unit, csp.wave))
    maggies_true = maggies_unit * float(10.0 ** TRUE_LOGMASS[0])
    sigma = maggies_true / SNR
    maggies_obs = maggies_true + np.array(sigma) * np.array(
        jax.random.normal(RNG, (N_FILTERS,))
    )

    phot_obs = Photometry(
        filters=SDSS_FILTERS, flux=maggies_obs, uncertainty=sigma,
        name="sdss_phot",
    )

    # ---- Step 3: model + nested sampling ---------------------------------
    sfh_times_yr = np.array(csp.sfh_times)

    def logsfr_to_sfh(free_theta, _t=sfh_times_yr):
        return logsfr_ratios_to_sfh(free_theta["logsfr_ratios"], sfh_times_yr=_t)

    priors = {
        "logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
        "Z": Uniform(low=-3.9, high=-1.45),   # stay within the FSPS metallicity grid
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
    )

    n_dims = sum(int(jnp.size(v)) for v in model.theta_init.values())
    # FAST DEMO settings so this finishes in ~1-2 min on a CPU. For science use
    # num_live >= 500, num_inner_steps >= n_dims*5, and a stricter logZ_tol
    # (e.g. -3.0) -- expect minutes-to-hours on CPU, seconds-to-minutes on GPU.
    adapter = BlackJAXNestedSamplerAdapter(
        priors=model.priors, num_live=80, num_inner_steps=max(8, n_dims * 2),
        logZ_tol=-0.5, verbose=True,
    )
    likelihood = MultiObservationLikelihood(
        keys=("sdss_phot",), likelihoods=(DiagonalGaussianLikelihood(),)
    )

    result = run_sampler(model, likelihood, adapter, RNG)
    print(f"\nln Z = {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}")

    # ---- Step 4: recovered vs. true --------------------------------------
    try:
        ns = result.to_anesthetic()
        # Weighted posterior draw, with replacement (the nested run yields fewer
        # weighted points than 2000, so replace=False would error).
        post = ns.sample(2000, replace=True)
        z_med = float(np.median(post["Z"]))
        m_med = float(np.median(post["logmass"]))
        print("\nparameter      true      posterior median")
        print(f"  Z          {float(TRUE_Z[0]):+.3f}     {z_med:+.3f}")
        print(f"  logmass    {float(TRUE_LOGMASS[0]):.3f}      {m_med:.3f}")

        out = HERE / "quickstart_corner.png"
        axes = ns.plot_2d(["Z", "logmass", "diffuse_tau_kc"])
        # anesthetic returns an AxesDataFrame; grab the Figure from any cell.
        axes.iloc[0, 0].figure.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\ncorner plot -> {out}")
    except ImportError:
        print("\n(install anesthetic for posterior summaries: pip install -e '.[nested]')")
    except Exception as exc:
        # The fit already succeeded (ln Z printed above); don't let a
        # plotting/summary hiccup crash the demo.
        print(f"\n(fit succeeded; posterior summary/plot skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
