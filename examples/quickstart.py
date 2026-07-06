#!/usr/bin/env python
"""
Ceridwen quick-start: fit mock UV-to-IR broadband photometry end-to-end.
==========================================================

This is a self-contained, runnable demo:

  Step 0  build (or load) the FSPS SSP grid cache  -> needs FSPS + $SPS_HOME
  Step 1  build the CSP forward model
  Step 2  generate MOCK photometry from known "true" parameters
  Step 3  fit it back with BlackJAX nested sampling
  Step 4  report recovered vs. true; write a corner plot (quickstart_corner.png)
          and a model-vs-data SED plot with a chi residual strip
          (quickstart_sed.png)

Requirements
------------
Ceridwen needs FSPS at *runtime*, not just to build the cache: the CLOUDY
nebular grids and the dust-emission templates are read from ``$SPS_HOME``
(the FSPS data directory).  So FSPS must be installed and ``SPS_HOME`` set,
e.g.::

    pip install .                         # everything except FSPS
    pip install "fsps>=0.4.4"             # FSPS wrapper (needs gfortran + $SPS_HOME)
    export SPS_HOME=/path/to/fsps         # the FSPS root (contains nebular/, ...)
    python examples/quickstart.py

The SSP cache is written to ``examples/ssp_data.h5`` the first time and
re-used on subsequent runs.  Set ``$SSP_FILE`` to point elsewhere.

The fit takes ~10 min on an unloaded CPU (much faster on GPU; longer on a busy
machine); lower ``num_live`` in the
adapter below for a quicker, rougher run.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 64-bit floats are required for accurate Bayesian evidence estimates.
jax.config.update("jax_enable_x64", True)

from ceridwen import SSPData, CSPBasis, SedModel
from ceridwen.observation import Photometry
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT
from ceridwen.likelihood import DiagonalGaussianLikelihood, MultiObservationLikelihood
from ceridwen.sampler import run_sampler
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter

HERE = pathlib.Path(__file__).resolve().parent


def _default_ssp_file() -> str:
    """Resolve a usable SSP grid without requiring FSPS.

    Order: $SSP_FILE -> examples/ssp_data.h5 -> the git-lfs test
    fixture tests/fixtures/ssp_data_test.h5 (present in every clone
    where `git lfs install && git lfs pull` has been run).  A tiny
    file at the fixture path is a git-lfs POINTER, not the data —
    detect it and explain, because the resulting HDF5 error is
    otherwise cryptic in a tutorial setting.
    """
    env = os.environ.get("SSP_FILE")
    if env:
        return env
    local = HERE / "ssp_data.h5"
    if local.is_file():
        return str(local)
    fixture = HERE.parent / "tests" / "fixtures" / "ssp_data_test.h5"
    if fixture.is_file():
        if fixture.stat().st_size < 10_000:
            raise SystemExit(
                f"{fixture} is a git-lfs pointer, not the actual grid.\n"
                "Run `git lfs install && git lfs pull` in the repository, "
                "then re-run this script."
            )
        return str(fixture)
    return str(local)   # absent: step0 falls through to the FSPS build


SSP_FILE = _default_ssp_file()
SPS_HOME = os.environ.get("SPS_HOME")
RNG = jax.random.PRNGKey(42)

# A broad UV->IR set (GALEX + SDSS + 2MASS + WISE) so the mock is well
# constrained and the corner plot is informative. All ship with sedpy_jax.
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
            "FSPS (python-fsps) is required to build the SSP grid and is not "
            "importable. Install it with `pip install 'fsps>=0.4.4'` and a "
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
    dummy_phot = Photometry(filters=FILTERS, name="_tmp")
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
        filters=FILTERS, flux=maggies_obs, uncertainty=sigma,
        name="phot",
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
    # Demo settings tuned for a reasonable CPU runtime (~10 min unloaded; longer
    # on a busy machine) and a good-looking corner. Two dials:
    #   * num_inner_steps -- dominates the one-time JIT *compile* time of the
    #     step kernel (it is the unrolled inner MCMC chain) AND per-step cost.
    #   * num_live -- runtime<->quality (more = smoother contours, more steps).
    # For publication use num_live >= 500, num_inner_steps >= n_dims*5, and a
    # stricter logZ_tol (e.g. -3.0); expect much longer on CPU.
    adapter = BlackJAXNestedSamplerAdapter(
        priors=model.priors, num_live=150, num_inner_steps=max(8, n_dims * 2),
        logZ_tol=-2.0, verbose=True,
    )
    likelihood = MultiObservationLikelihood(
        keys=("phot",), likelihoods=(DiagonalGaussianLikelihood(),)
    )

    result = run_sampler(model, likelihood, adapter, RNG)
    print(f"\nln Z = {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}")

    # ---- Step 4: recovered vs. true --------------------------------------
    TRUTH = {
        "Z": float(TRUE_Z[0]),
        "logmass": float(TRUE_LOGMASS[0]),
        "diffuse_tau_kc": float(TRUE_DIFFDUST[0]),
        "diffuse_dust_index": float(TRUE_DUST_INDEX[0]),
    }
    LABELS = {
        "Z": r"$\log_{10} Z$",
        "logmass": r"$\log_{10} M_\star$",
        "diffuse_tau_kc": r"$\hat{\tau}_V$",
        "diffuse_dust_index": r"$\delta_{\rm dust}$",
    }
    try:
        ns = result.to_anesthetic(labels=LABELS)
        post = ns.sample(4000, replace=True)   # weighted draw, with replacement
    except Exception as exc:
        ns, post = None, None
        print(f"\n(posterior post-processing unavailable: {exc})")

    if post is not None:
        params = list(TRUTH)

        # Posterior medians vs. injected truth.
        print("\nparameter             true     posterior median")
        for p in params:
            print(f"  {p:<20}{TRUTH[p]:+7.3f}   {float(np.median(post[p])):+7.3f}")

        # (a) Corner plot with the truth overlaid as red dashed lines.
        try:
            out = HERE / "quickstart_corner.png"
            axes = ns.plot_2d(params)
            for yp in params:
                for xp in params:
                    try:
                        ax = axes.loc[yp, xp]
                    except Exception:
                        ax = None
                    if ax is None:
                        continue
                    ax.axvline(TRUTH[xp], color="red", lw=1.1, ls="--")
                    if yp != xp:
                        ax.axhline(TRUTH[yp], color="red", lw=1.1, ls="--")
            fig = axes.iloc[0, 0].figure
            fig.suptitle("CERIDWEN quickstart — red = injected truth", fontsize=11)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"corner plot (truth overlaid) -> {out}")
        except Exception as exc:
            print(f"(corner plot skipped: {exc})")

        # (b) Model-vs-data SED with a chi residual strip. The "model" is the
        #     posterior-median fit: predicted photometry over the observed
        #     points, with the predicted spectrum behind them.
        try:
            ratio_cols = [f"logsfr_ratios[{i}]" for i in range(N_TIME - 1)]
            logmass_med = float(np.median(post["logmass"]))
            theta_med = {
                "logsfr_ratios": jnp.asarray(
                    [float(np.median(post[c])) for c in ratio_cols]),
                "Z": jnp.asarray([float(np.median(post["Z"]))]),
                "logmass": jnp.asarray([logmass_med]),
                "diffuse_tau_kc": jnp.asarray([float(np.median(post["diffuse_tau_kc"]))]),
                "diffuse_dust_index": jnp.asarray(
                    [float(np.median(post["diffuse_dust_index"]))]),
            }
            pred_maggies = np.asarray(model.predict(theta_med)["phot"])
            model_theta = model.apply_transforms(theta_med)
            spec = np.asarray(csp.get_spectrum(model_theta)) * 10.0 ** logmass_med
            wave_model = np.asarray(csp.wave)
            wave_eff = np.asarray(phot_obs.wave_eff)

            # Put the model spectrum on the photometry (maggies) scale by matching
            # it to the predicted points at the filter effective wavelengths.
            spec_at_eff = np.interp(wave_eff, wave_model, spec)
            good = spec_at_eff > 0
            scale = float(np.median(pred_maggies[good] / spec_at_eff[good])) if good.any() else 1.0
            spec_maggies = spec * scale

            # The injected (true) spectrum, on the same maggies scale.
            true_spec = np.asarray(spec_unit) * 10.0 ** float(TRUE_LOGMASS[0])
            true_maggies = true_spec * scale

            chi = (maggies_obs - pred_maggies) / sigma

            fig, (axsed, axchi) = plt.subplots(
                2, 1, sharex=True, figsize=(7.5, 5.2),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
            )
            axsed.plot(wave_model, true_maggies, color="C0", lw=1.0, alpha=0.9,
                       zorder=1, label="true spectrum")
            axsed.plot(wave_model, spec_maggies, color="0.5", lw=1.0, ls="--",
                       zorder=2, label="model spectrum (median)")
            axsed.errorbar(wave_eff, maggies_obs, yerr=sigma, fmt="o", color="k",
                           ms=5, capsize=2, zorder=3, label="observed")
            axsed.scatter(wave_eff, pred_maggies, marker="s", facecolors="none",
                          edgecolors="red", s=70, zorder=4, label="model (median)")
            axsed.set_xscale("log")
            axsed.set_yscale("log")
            axsed.set_ylabel("flux  [maggies]")
            axsed.legend(frameon=False, fontsize=9)
            axsed.set_title("CERIDWEN quickstart — model vs data")
            _lo = float(min(maggies_obs.min(), pred_maggies.min()))
            _hi = float(max(maggies_obs.max(), pred_maggies.max()))
            axsed.set_ylim(_lo * 0.3, _hi * 3.0)
            axsed.set_xlim(float(wave_eff.min()) * 0.7, float(wave_eff.max()) * 1.4)

            axchi.axhline(0.0, color="0.5", lw=0.8)
            for s in (-1.0, 1.0):
                axchi.axhline(s, color="0.8", lw=0.6, ls="--")
            axchi.scatter(wave_eff, chi, color="red", s=30, zorder=3)
            axchi.set_ylabel(r"$\chi$")
            axchi.set_xlabel(r"wavelength  [$\mathrm{\AA}$]  (rest frame)")
            _c = max(3.5, float(np.abs(chi).max()) * 1.2)
            axchi.set_ylim(-_c, _c)

            sed_out = HERE / "quickstart_sed.png"
            fig.savefig(sed_out, dpi=150, bbox_inches="tight")
            print(f"model-vs-data plot -> {sed_out}")
        except Exception as exc:
            print(f"(model-vs-data plot skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
