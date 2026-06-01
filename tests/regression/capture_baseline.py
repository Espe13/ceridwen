"""
Regression baseline capture for CERIDWEN (review-driven refactor, 2026-06-01).

This module is the SINGLE SOURCE OF TRUTH for the regression evaluations.
Both this script (run directly to *save* baselines) and
``tests/regression/test_regression.py`` (run under pytest to *compare*) import
``compute_baselines()`` so the pre- and post-refactor evaluations are
guaranteed identical in construction.

Run to (re)generate baselines:

    SPS_HOME=/path/to/fsps python tests/regression/capture_baseline.py

The model is the ``big_comparison.py`` CSP recipe (MIST SSP grid, Kriek-Conroy
diffuse dust, birth-cloud power-law, nebular grid, dust emission) plus IGM
(Madau 1995) and redshift z=2.0, matching the Step-0 specification:

    tau_v (diffuse_tau_kc) = 0.5, delta (diffuse_dust_index) = -0.1,
    log_u (gas_logu) = -2.5, log_zsol (gas_logz) = 0.0, z = 2.0.

Documented deviations from the literal Step-0 wording (the underlying package
API differs):

  * ``cosmology.py`` exposes NO ``lookback_time`` function (lookback time in
    this package is the user-supplied SFH ``lookback_time`` grid, not a
    cosmological quantity).  We instead baseline the distance / flux-factor
    functions that DO exist: ``comoving_distance_mpc``,
    ``luminosity_distance_mpc``, ``flux_factor``, ``flux_factor_maggies``,
    ``E_of_z``.  These take a *scalar* z, so we ``vmap`` over the z grid.
  * IGM exposes ``IGMModel.attenuation(wave, zred, factor)`` rather than a
    ``transmission`` function; ``attenuation`` returns exp(-tau*factor), i.e.
    the transmission.  We baseline that.
  * ``SSPBasis.get_galaxy_spectrum`` (FSPS-backed) is baselined for the SSP
    spectrum.  ``FastStepBasis.get_galaxy_spectrum`` is NOT baselined: in the
    pre-refactor code it is ``@jit`` on a bound method and is currently
    uncallable (``self`` cannot be traced as an array) — there is no "before"
    output to compare.  Phase 3 fixes this; its correctness is verified there
    via ``jax.make_jaxpr`` and a physical sanity check (per CHANGES.md).
  * Only ONE ``fsps.StellarPopulation`` is constructed per process (SSPBasis);
    FSPS keeps global Fortran state, so constructing a second one corrupts it.
"""
from __future__ import annotations

import os
import pathlib
import pickle

# Force CPU + float64 determinism for the reference comparison.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
BASELINE_DIR = HERE / "baselines"
REPO_ROOT = HERE.parent.parent

SSP_FILE = str(REPO_ROOT / "ceridwen" / "data" / "test_data" / "ssp_data.h5")
SPS_HOME = os.environ.get("SPS_HOME", str(pathlib.Path.home() / "Prospector" / "fsps"))


# --------------------------------------------------------------------------- #
#  Fixed parameter definitions (also pickled for model rebuilds)
# --------------------------------------------------------------------------- #
def fixed_params() -> dict:
    """All scalar / array constants that define the baseline evaluations."""
    T_UNIV = 13.8
    N_TIME = 10
    t_grid = jnp.linspace(1e-2, T_UNIV, N_TIME)
    lookback = T_UNIV - t_grid

    def gaussian_burst(tau, center, width, amp=1.0):
        return amp * jnp.exp(-0.5 * ((tau - center) / width) ** 2)

    sfh = (gaussian_burst(lookback, 0.05, 0.03, 1.0)
           + gaussian_burst(lookback, 11.0, 0.8, 0.7))

    return dict(
        T_UNIV=T_UNIV,
        lookback=lookback,
        sfh=sfh,
        # CSP physics theta (fixed)
        diffuse_tau_kc=0.5,      # tau_v
        diffuse_dust_index=-0.1,  # delta
        tau_pow=0.3,
        alpha=-1.0,
        gas_logu=-2.5,           # log_u
        gas_logz=0.0,            # log_zsol
        zred=2.0,
        logmass=10.0,
        # nebular standalone evaluation point
        neb_logZ=0.0, neb_logU=-2.5, neb_logage=6.8, neb_logQ=47.0,
        # dust emission params
        duste_qpah=3.5, duste_umin=1.0, duste_gamma=0.01,
        # SSP point
        ssp_tage=5.0,
        # filters / spec grid
        filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
        spec_n=200,
        spec_lo=3000.0, spec_hi=8000.0,
        z_grid=list(np.linspace(0.0, 10.0, 11)),
        noise_seed=0,
    )


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #
def compute_baselines() -> dict[str, dict[str, np.ndarray]]:
    """Return {category: {array_name: np.ndarray}} for every baseline.

    Deterministic: fixed grids and ``jax.random.PRNGKey(0)`` for noise.
    """
    p = fixed_params()
    out: dict[str, dict] = {}

    # ----- SSP (FSPS-backed) : construct the ONLY StellarPopulation first ---
    from ceridwen.ssps.ssp_basis import SSPBasis
    ssp = SSPBasis(zcontinuous=0, zmet=1)
    w_ssp, s_ssp, m_ssp = ssp.get_galaxy_spectrum(tage=p["ssp_tage"])
    out["ssp_spectrum"] = {
        "wave": np.asarray(w_ssp),
        "spectrum": np.asarray(s_ssp),
        "mass_fraction": np.asarray(m_ssp, dtype=np.float64),
    }
    del ssp  # release FSPS StellarPopulation

    # ----- IGM --------------------------------------------------------------
    from ceridwen.igm import make_igm_model
    igm = make_igm_model("madau1995")
    wave_igm = jnp.linspace(800.0, 9000.0, 1000)
    trans = igm.attenuation(wave_igm, jnp.asarray(p["zred"]), factor=1.0)
    out["igm"] = {"wave": np.asarray(wave_igm), "transmission": np.asarray(trans)}

    # ----- cosmology (scalar functions, vmap over z grid) -------------------
    from ceridwen import cosmology as cosmo
    zg = jnp.asarray(p["z_grid"])
    out["cosmology"] = {
        "z": np.asarray(zg),
        "E_of_z": np.asarray(jax.vmap(cosmo.E_of_z)(zg)),
        "comoving_distance_mpc": np.asarray(jax.vmap(cosmo.comoving_distance_mpc)(zg)),
        "luminosity_distance_mpc": np.asarray(jax.vmap(cosmo.luminosity_distance_mpc)(zg)),
        "flux_factor": np.asarray(jax.vmap(cosmo.flux_factor)(zg)),
        "flux_factor_maggies": np.asarray(jax.vmap(cosmo.flux_factor_maggies)(zg)),
    }

    # ----- CSP (full pipeline) ---------------------------------------------
    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.observation.observation import Photometry, Spectrum

    ssp_data = SSPData.load(SSP_FILE)
    base_theta = {
        "lookback_time": p["lookback"],
        "sfh": p["sfh"],
        "Z": jnp.array([0.0]),
    }
    csp = CSPBasis(
        ssp_data,
        theta=base_theta,
        tuniv=p["T_UNIV"],
        tiny_logt=-70,
        zh_const=True,
        add_dust=True,
        add_diffuse_dust=True,
        add_dust_emission=True,
        add_neb=True,
        add_igm=True,
        igm_model="madau1995",
        init_neb_params={"isoc_type": "mist", "cloudy_dust": False},
        init_dust_params={"bin_edges": [(-jnp.inf, -1.97)], "laws": ["powerlaw"]},
        diffuse_law="kriek_conroy",
        sps_home=SPS_HOME,
        verbose=False,
        sfh_interp="linear",
    )

    theta = dict(csp.theta_init)
    theta["diffuse_tau_kc"] = jnp.array([p["diffuse_tau_kc"]])
    theta["diffuse_dust_index"] = jnp.array([p["diffuse_dust_index"]])
    theta["tau_pow"] = jnp.array([p["tau_pow"]])
    theta["alpha"] = jnp.array([p["alpha"]])
    theta["gas_logz"] = jnp.array([p["gas_logz"]])
    theta["gas_logu"] = jnp.array([p["gas_logu"]])
    theta["duste_qpah"] = jnp.array([p["duste_qpah"]])
    theta["duste_umin"] = jnp.array([p["duste_umin"]])
    theta["duste_gamma"] = jnp.array([p["duste_gamma"]])
    theta["zred"] = jnp.array([p["zred"]])
    theta["logmass"] = jnp.array([p["logmass"]])

    # get_spectrum + components
    spec_full = csp.get_spectrum(theta, include_lines=True)
    spec_cont = csp.get_spectrum(theta, include_lines=False)
    continuum, lines = csp.get_spectrum_components(theta)
    out["csp_components"] = {
        "wave": np.asarray(csp.wave),
        "get_spectrum_full": np.asarray(spec_full),
        "get_spectrum_continuum": np.asarray(spec_cont),
        "components_continuum": np.asarray(continuum),
        "components_lines": np.asarray(lines),
    }

    # dust attenuation curves (the package's own Dust / DiffuseDust instances)
    attn, attn_diffuse = csp.attenuate_dust(csp.wave, theta)
    out["dust_attenuation"] = {
        "wave": np.asarray(csp.wave),
        "attn_binwise": np.asarray(attn),
        "attn_diffuse": np.asarray(attn_diffuse),
    }

    # dust emission (csp.dust_emi.compute_dust_emission) with deterministic
    # inputs derived from the SSP grid + the diffuse attenuation curve.
    spec_dustfree = jnp.sum(csp.flux[0], axis=0)            # (n_wave,)
    diffuse_curve = jnp.ravel(jnp.asarray(attn_diffuse))    # (n_wave,)
    spec_attn = spec_dustfree * diffuse_curve
    specdust, mdust, tduste = csp.dust_emi.compute_dust_emission(
        spec_attn, spec_dustfree, csp.wave, diffuse_curve,
        jnp.asarray(p["duste_qpah"]), jnp.asarray(p["duste_umin"]),
        jnp.asarray(p["duste_gamma"]),
    )
    out["dust_emission"] = {
        "wave": np.asarray(csp.wave),
        "specdust": np.asarray(specdust),
        "mdust": np.asarray(mdust, dtype=np.float64),
        "tduste": np.asarray(tduste, dtype=np.float64),
    }

    # nebular (NebularModel.evaluate at fixed logZ, logU, logage, logQ)
    neb_cont, neb_lines = csp.neb.evaluate(
        jnp.asarray(p["neb_logZ"]), jnp.asarray(p["neb_logU"]),
        jnp.asarray(p["neb_logage"]), jnp.asarray(p["neb_logQ"]),
    )
    out["nebular"] = {
        "continuum": np.asarray(neb_cont),
        "lines": np.asarray(neb_lines),
    }

    # ----- Observations + predict ------------------------------------------
    phot = Photometry(filters=p["filters"], name="phot")
    phot.setup_for_model(csp.wave)
    wave_spec = jnp.asarray(np.geomspace(p["spec_lo"], p["spec_hi"], p["spec_n"]))
    spec_obs = Spectrum(
        wavelength=wave_spec,
        flux=jnp.ones(p["spec_n"]),
        uncertainty=jnp.ones(p["spec_n"]),
        name="spec",
        smoothtype="vel",
        resolution=300.0,
    )
    spec_obs.setup_for_model(csp.wave)
    observations = [phot, spec_obs]

    predictions = csp.predict(theta, observations)
    out["csp_spectrum"] = {
        "phot": np.asarray(predictions["phot"]),
        "spec": np.asarray(predictions["spec"]),
        "get_spectrum_full": np.asarray(spec_full),
    }

    # ----- Likelihood (synthetic Photometry + Spectrum vs prediction) ------
    from ceridwen.likelihood.noise_model import DiagonalNoiseModel
    from ceridwen.likelihood.likelihood import DiagonalGaussianLikelihood

    key = jax.random.PRNGKey(p["noise_seed"])
    k1, k2 = jax.random.split(key)
    mu_phot = predictions["phot"]
    mu_spec = predictions["spec"]
    sig_phot = 0.05 * jnp.abs(mu_phot) + 1e-12
    sig_spec = 0.05 * jnp.abs(mu_spec) + 1e-12
    y_phot = mu_phot + sig_phot * jax.random.normal(k1, mu_phot.shape)
    y_spec = mu_spec + sig_spec * jax.random.normal(k2, mu_spec.shape)
    mask_phot = jnp.ones(mu_phot.shape, dtype=bool)
    mask_spec = jnp.ones(mu_spec.shape, dtype=bool)

    noise_model = DiagonalNoiseModel(use_jitter=False, use_fractional=False)
    lik = DiagonalGaussianLikelihood(noise_model=noise_model)
    lnl_phot, _ = lik(y_phot, mu_phot, sig_phot, mask_phot)
    lnl_spec, _ = lik(y_spec, mu_spec, sig_spec, mask_spec)
    out["likelihood"] = {
        "lnl_phot": np.asarray(lnl_phot, dtype=np.float64),
        "lnl_spec": np.asarray(lnl_spec, dtype=np.float64),
        "lnl_total": np.asarray(lnl_phot + lnl_spec, dtype=np.float64),
    }

    return out


# --------------------------------------------------------------------------- #
#  Save (run directly)
# --------------------------------------------------------------------------- #
def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baselines = compute_baselines()
    for category, arrays in baselines.items():
        np.savez(BASELINE_DIR / f"{category}.npz", **arrays)
        print(f"  wrote {category}.npz  ({', '.join(arrays.keys())})")
    with open(BASELINE_DIR / "params.pkl", "wb") as f:
        pickle.dump(fixed_params(), f)
    print(f"  wrote params.pkl")
    print(f"\nBaselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
