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
    ``luminosity_distance_mpc``, ``flux_factor``, ``flux_factor_maggies`` (= ``flux_factor_cgs``),
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

import json
import os
import pathlib
import sys

# Make the shared test helper importable when this file is run as a standalone
# script (under pytest, tests/conftest.py already puts tests/ on sys.path).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Force CPU + float64 determinism for the reference comparison.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from ceridwen.broadening import Instrument, Kinematics

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
BASELINE_DIR = HERE / "baselines"
REPO_ROOT = HERE.parent.parent

from _gridfixture import require_test_grid, find_named_grid

SSP_FILE = str(require_test_grid())
SPS_HOME = os.environ.get("SPS_HOME", str(pathlib.Path.home() / "Prospector" / "fsps"))


# --------------------------------------------------------------------------- #
#  Fixed parameter definitions (also written to params.json as a record)
# --------------------------------------------------------------------------- #
def fixed_params() -> dict:
    """All scalar / array constants that define the baseline evaluations."""
    T_UNIV = 13.8
    N_TIME = 10
    t_grid = jnp.linspace(1e-2, T_UNIV, N_TIME)
    lookback = jnp.linspace(0.0, T_UNIV, N_TIME)   # NEW convention

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
        # instrumental LSF scale category
        lsf_range=[0.8, 1.3], lsf_values=[0.85, 1.0, 1.22], lsf_fixed=1.2,
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
        "flux_factor_maggies": np.asarray(jax.vmap(cosmo.flux_factor_cgs)(zg)),
    }

    # ----- CSP (full pipeline) ---------------------------------------------
    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.cosmology import Cosmology
    from ceridwen.observation.observation import Photometry, Spectrum

    ssp_data = SSPData.load(SSP_FILE)
    # These baselines were captured with the pre-v1.0.5 key "Z" = 0.0, an ABSOLUTE log10 Z
    # above the BPASS grid (max -1.39794), so the metallicity interpolation clamped every
    # weight onto the top node.  In logzsol the same physical point is that top node, which
    # gives the identical (clipped) weights -- the stored arrays are unchanged.
    top_node_logzsol = float(np.asarray(ssp_data.ssp_lgmet)[-1] - ssp_data.log10_zsun)
    base_theta = {
        "lookback_time": p["lookback"],
        "sfh": p["sfh"],
        "logzsol": jnp.array([top_node_logzsol]),
    }
    csp = CSPBasis(
        ssp_data,
        theta=base_theta,
        cosmo=Cosmology.planck18(),   # the pre-2026-09-03 default, so the baselines still apply
        tiny_logt=-70,
        zh_const=True,
        add_dust=True,
        add_diffuse_dust=True,
        add_dust_emission=True,
        add_neb=True,
        add_igm=True,
        igm_model="madau1995",
        init_neb_params={"cloudy_dust": False},  # isoc_type auto from grid provenance
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

    # THEMIS dust emission (CSPBasis(duste_model="THEMIS"), 2026-09-22): the same inputs as
    # "dust_emission" through the THEMIS templates, plus a THEMIS CSP spectrum.  Justified by
    # tests/test_dust_emission_themis.py: FSPS's own THEMIS (qPAH, Umin) axes
    # (src/sps_vars.f90), template columns = the $SPS_HOME files, energy balance to 1e-10.
    out["dust_emission_themis"] = _dust_emission_themis_baseline(
        p, ssp_data, spec_attn, spec_dustfree, diffuse_curve)

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
        instrument=Instrument.sigma_kms(300.0),
    )
    spec_obs.setup_for_model(
        csp.wave, kinematics=Kinematics.none(),
        lib_resolution=getattr(csp, "lib_resolution", None))
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

    # ----- Outlier mixture (Prospector's NoiseModel.lnlike, f_outlier > 0) --------------
    # Same data as "likelihood" with deterministic +/-20 sigma outliers (every 37th spectral
    # pixel, alternating sign; the third band +10 sigma).  Validated against a NumPy
    # transcription of Prospector's outlier branch and the real Prospector
    # (tests/test_outlier_model.py).
    idx = jnp.arange(mu_spec.size)
    kick = jnp.where(idx % 37 == 0, jnp.where((idx // 37) % 2 == 0, 20.0, -20.0), 0.0)
    y_spec_o = y_spec + kick * sig_spec
    y_phot_o = y_phot.at[2].add(10.0 * sig_phot[2])
    lik_s = DiagonalGaussianLikelihood(noise_model=DiagonalNoiseModel(f_outlier=0.01))
    lik_p = DiagonalGaussianLikelihood(noise_model=DiagonalNoiseModel(
        f_outlier=0.05, nsigma_outlier=20.0, noise_floor=0.02))
    lnl_s, aux_s = lik_s(y_spec_o, mu_spec, sig_spec, mask_spec)
    lnl_p, aux_p = lik_p(y_phot_o, mu_phot, sig_phot, mask_phot)
    lnl_s_gauss, _ = lik(y_spec_o, mu_spec, sig_spec, mask_spec)
    out["outlier_likelihood"] = {
        "lnl_spec": np.asarray(lnl_s, dtype=np.float64),
        "lnl_phot": np.asarray(lnl_p, dtype=np.float64),
        "lnl_spec_gaussian": np.asarray(lnl_s_gauss, dtype=np.float64),
        "lnl_pointwise_spec": np.asarray(aux_s.lnl_pointwise, dtype=np.float64),
        "p_outlier_spec": np.asarray(lik_s.outlier_probability(
            y_spec_o, mu_spec, sig_spec, mask_spec), dtype=np.float64),
        "p_outlier_phot": np.asarray(lik_p.outlier_probability(
            y_phot_o, mu_phot, sig_phot, mask_phot), dtype=np.float64),
    }
    # upper limit on the last band: mixture on the detections, one-sided penalty on the limit
    from ceridwen.likelihood.likelihood import DiagonalGaussianLikelihoodWithUpperLimits
    ul_p = jnp.zeros(mu_phot.shape, dtype=bool).at[-1].set(True)
    lnl_p_ul, _ = DiagonalGaussianLikelihoodWithUpperLimits(noise_model=lik_p.noise_model)(
        y_phot_o, mu_phot, sig_phot, mask_phot, is_upper_limit=ul_p)
    out["outlier_likelihood"]["lnl_phot_upper_limit"] = np.asarray(lnl_p_ul, dtype=np.float64)

    out["calibration_likelihood"] = _calibration_likelihood_baseline(
        np.asarray(wave_spec), mu_spec, mask_spec, jax.random.PRNGKey(p["noise_seed"]))

    out["gp_likelihood"] = _gp_likelihood_baseline(
        np.asarray(wave_spec), mu_spec, jax.random.PRNGKey(p["noise_seed"] + 1))
    out["poly_marginal"] = _poly_marginal_baseline(
        np.asarray(wave_spec), mu_spec, mask_spec,
        jax.random.fold_in(jax.random.PRNGKey(p["noise_seed"]), 11))

    out["igm_damping_dla"] = _igm_damping_dla_baseline()
    out["dust_laws"] = _dust_laws_baseline(p, ssp_data)
    out.update(_logzsol_baselines(p))
    out.update(_stellar_mass_baseline(p))
    out["eline_marginal"] = _eline_marginal_baseline(p, ssp_data)
    out["lsf_scale"] = _lsf_scale_baseline(p, ssp_data)
    return out


def _igm_damping_dla_baseline() -> dict[str, np.ndarray]:
    """``MadauDampingDLA`` transmission on a rest-frame grid through Ly-alpha (WMAP9 with
    Ob0 = 0.04628, Prospector's cosmology).  Justification: the damping-wing and DLA components
    equal Prospector's ``tau_damping`` / ``voigt_profile`` (@ a78d153) to <= 1e-12 in
    transmission (``examples/recipes/tests/check_igm_damping_dla.py``, 63/63), and the
    ``x_HI = 0`` / ``logN_HI = -inf`` rows equal ``Madau1995`` byte for byte.  ``*_theta`` rows
    go through the ``params`` (theta) path; ``fg_dla`` is a foreground absorber, trough at rest
    1215.67 (1 + z_dla) / (1 + zred)."""
    from ceridwen.igm import MadauDampingDLA
    from ceridwen.cosmology import Cosmology
    c = Cosmology.wmap9()
    wave = jnp.asarray(1000.0 + 0.25 * np.arange(2400))      # 1000-1600 A rest
    z = jnp.asarray(7.0)
    base = MadauDampingDLA(Ob0=0.04628, cosmo=c)
    th = lambda **kw: {k: jnp.asarray([v]) for k, v in kw.items()}
    return {
        "wave": np.asarray(wave),
        "madau": np.asarray(base.attenuation(wave, z)),
        "damp_x03": np.asarray(jnp.exp(-base.tau_damp(wave, z, th(x_HI=0.3)))),
        "damp_x10": np.asarray(jnp.exp(-base.tau_damp(wave, z, th(x_HI=1.0)))),
        "dla_215_z3": np.asarray(jnp.exp(-base.tau_dla(wave, jnp.asarray(3.0),
                                                       th(logN_HI=21.5)))),
        "total_fixed": np.asarray(MadauDampingDLA(
            Ob0=0.04628, cosmo=c, x_HI=0.8, logN_HI=21.0).attenuation(wave, z, factor=0.9)),
        "total_theta": np.asarray(base.attenuation(
            wave, z, factor=0.9, params=th(x_HI=0.8, logN_HI=21.0))),
        "fg_dla_theta": np.asarray(base.attenuation(
            wave, z, params=th(logN_HI=21.0, z_dla=6.5))),
        "off_theta": np.asarray(base.attenuation(
            wave, z, params=th(x_HI=0.0, logN_HI=-np.inf))),
    }


def _dust_emission_themis_baseline(p, ssp_data, spec_attn, spec_dustfree, diffuse_curve):
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.cosmology import Cosmology
    csp = CSPBasis(ssp_data, theta={"lookback_time": p["lookback"], "sfh": p["sfh"],
                                    "logzsol": jnp.array([-0.4])},
                   cosmo=Cosmology.planck18(), zh_const=True, add_neb=False, add_dust=True,
                   add_diffuse_dust=True, add_dust_emission=True, duste_model="THEMIS",
                   sps_home=SPS_HOME, verbose=False, sfh_interp="step")
    specdust, mdust, tduste = csp.dust_emi.compute_dust_emission(
        spec_attn, spec_dustfree, csp.wave, diffuse_curve,
        jnp.asarray(p["duste_qpah"]), jnp.asarray(p["duste_umin"]),
        jnp.asarray(p["duste_gamma"]))
    th = dict(csp.theta_init, diffuse_tau_kc=jnp.array([p["diffuse_tau_kc"]]),
              duste_qpah=jnp.array([8.0]), duste_umin=jnp.array([2.2]),
              duste_gamma=jnp.array([0.05]))
    return {
        "wave": np.asarray(csp.wave),
        "specdust": np.asarray(specdust),
        "mdust": np.asarray(mdust, dtype=np.float64),
        "tduste": np.asarray(tduste, dtype=np.float64),
        "csp_spectrum": np.asarray(csp.get_spectrum(th, include_lines=False)),
    }


def _dust_laws_baseline(p, ssp_data) -> dict[str, np.ndarray]:
    """Laws added or fixed 2026-09-22, through the package's own ``Dust`` / ``DiffuseDust`` and a
    CSP.  Curves: ``gordon03_smcbar`` (= FSPS dust_type=5 table + interpolation, and the FSPS
    Fortran, to 1e-16) and ``reddy15`` (= Prospector ``fake_fsps`` to 1e-15), per
    ``examples/recipes/tests/check_extra_dust_laws.py``; ``noll`` with ``Ebump = 2`` in an age
    bin (= a direct ``noll(...)`` call, which before the fix it was not: the bump was dropped);
    ``drude`` (= ``drude(1e4 / wave)``, peak 1 at 2178.6 A; before the fix ~1e-7); ``smc`` /
    ``lmc`` (tau(5500) = amplitude).  Spectra: the stellar CSP with three age bins
    (noll / gordon03_smcbar / lmc) and a ``reddy15`` diffuse screen, and with
    (drude / smc) bins and a ``noll`` diffuse screen."""
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.cosmology import Cosmology
    from ceridwen.dust.DustModel import Dust, DiffuseDust
    import contextlib
    import io

    wave = jnp.asarray(np.geomspace(912.0, 3e4, 500))
    out = {"wave": np.asarray(wave)}
    for law, pars in (("gordon03_smcbar", {"tau_g03smc": 0.7}), ("reddy15", {"tau_reddy": 0.7}),
                      ("noll", {"tau_noll": 0.7, "delta": -0.2, "c_r": 0.1, "Ebump": 2.0}),
                      ("drude", {"x0": 4.59, "gamma": 0.9}), ("smc", {"tau_smc": 0.7}),
                      ("lmc", {"tau_lmc": 0.7})):
        th = {k: jnp.asarray(v) for k, v in pars.items()}
        out[f"bin_{law}"] = np.asarray(
            Dust(bin_edges=[(-jnp.inf, jnp.inf)], laws=[law]).compute_attenuation(wave, th)[0])
        out[f"diffuse_{law}"] = np.asarray(DiffuseDust(law).compute_attenuation(
            wave, {f"diffuse_{k}": v for k, v in th.items()}))

    configs = {
        "a": ({"bin_edges": [(-jnp.inf, -2.0), (-2.0, -1.0), (-1.0, jnp.inf)],
               "laws": ["noll", "gordon03_smcbar", "lmc"]}, "reddy15",
              {"tau_noll": 0.6, "delta": -0.2, "c_r": 0.0, "Ebump": 2.0, "tau_g03smc": 0.4,
               "tau_lmc": 0.2, "diffuse_tau_reddy": 0.3}),
        "b": ({"bin_edges": [(-jnp.inf, -2.0), (-2.0, jnp.inf)], "laws": ["drude", "smc"]},
              "noll", {"x0": 4.59, "gamma": 0.9, "tau_smc": 0.3, "diffuse_tau_noll": 0.3,
                       "diffuse_delta": -0.1, "diffuse_c_r": 0.0, "diffuse_Ebump": 1.5}),
    }
    for tag, (bins, diffuse, vals) in configs.items():
        with contextlib.redirect_stdout(io.StringIO()):
            csp = CSPBasis(ssp_data, theta={"lookback_time": p["lookback"], "sfh": p["sfh"],
                                            "logzsol": jnp.array([-0.4])},
                           cosmo=Cosmology.planck18(), zh_const=True, add_neb=False,
                           add_dust=True, add_diffuse_dust=True, init_dust_params=bins,
                           diffuse_law=diffuse, verbose=False, sfh_interp="step")
        th = dict(csp.theta_init, **{k: jnp.atleast_1d(jnp.asarray(v)) for k, v in vals.items()})
        attn, attn_diffuse = csp.attenuate_dust(csp.wave, th)
        out[f"csp_{tag}_attn_binwise"] = np.asarray(attn)
        out[f"csp_{tag}_attn_diffuse"] = np.asarray(attn_diffuse)
        out[f"csp_{tag}_spectrum"] = np.asarray(csp.get_spectrum(th, include_lines=False))
    return out


# ---------------------------------------------------------------------------- #
#  logzsol categories (v1.0.5): one per grid family, at an INTERIOR metallicity
# ---------------------------------------------------------------------------- #
LOGZSOL_INTERIOR = -0.4      # interior on every shipped grid, and not a node on any of them
AFE_INTERIOR = 0.3           # between the +0.2 and +0.4 planes, clear of the refused cell

LOGZSOL_GRIDS = {
    # the canonical test grid, resolved like every other test grid ($CERIDWEN_TEST_SSP ->
    # tests/fixtures -> ceridwen/data/test_data -> fetch_grid cache), so CI runs this
    # category too; the other two by file name (repo copy, else the fetch_grid cache)
    "logzsol_bpass": (SSP_FILE, False),
    "logzsol_mist":  (find_named_grid("ssp_data_mist_miles.h5"), False),
    "logzsol_afe":   (find_named_grid("amist_c3k_hr_krou_afe.h5"), True),
}


def _logzsol_baselines(p) -> dict:
    """Stellar weights and spectrum at a fixed logzsol on each available grid family.

    These are the v1.0.5 metallicity-convention baselines: the inputs are logzsol =
    log10(Z/Z_sun) with the grid's own Z_sun, at an interior (non-node) value, so they pin
    the conversion itself and not just a grid point.  Their numbers are justified in the
    commit message: each equals the pre-v1.0.5 code evaluated at the converted absolute
    metallicity (logzsol + log10 Z_sun) to the stated tolerance, and the same convention is
    checked against python-fsps directly in tests/test_logzsol_convention.py.
    """
    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.csp.csp_afe import CSPBasis_afe
    from ceridwen.cosmology import Cosmology

    out = {}
    for cat, (rel, is_afe) in LOGZSOL_GRIDS.items():
        if rel is None:
            continue
        path = pathlib.Path(rel) if pathlib.Path(rel).is_absolute() else REPO_ROOT / rel
        if not path.is_file():
            continue
        cls, basis = (SSPDataAfe, CSPBasis_afe) if is_afe else (SSPData, CSPBasis)
        ssp = cls.load(str(path))
        theta = {"lookback_time": p["lookback"], "sfh": p["sfh"],
                 "logzsol": jnp.array([LOGZSOL_INTERIOR])}
        if is_afe:
            theta["afe"] = jnp.array([AFE_INTERIOR])
        csp = basis(ssp, theta=theta, cosmo=Cosmology.planck18(), zh_const=True,
                    add_dust=False, add_diffuse_dust=False, add_dust_emission=False,
                    add_igm=False, verbose=False, sfh_interp="step",
                    **({} if is_afe else {"add_neb": False}))
        th = dict(csp.theta_init)
        blk = {
            "logzsol_axis": np.asarray(csp.zmet, dtype=np.float64),
            "log10_zsun": np.asarray([csp.log10_zsun], dtype=np.float64),
            "weights": np.asarray(csp.calculate_ssp_weights(th), dtype=np.float64),
            "spectrum": np.asarray(csp.get_spectrum(th), dtype=np.float64),
        }
        if is_afe:
            blk["logzsol_total"] = np.asarray(csp.logzsol_total(th), dtype=np.float64)
        out[cat] = blk
    return out


def _calibration_likelihood_baseline(wave, mu, mask, key) -> dict:
    """v1.0.7: the profiled polynomial calibration (Spectrum(polynomial_order > 0), Prospector's
    PolyOptCal) and a per-observation noise key.  Data = the "csp_spectrum" prediction times a
    known response 1 + 0.05 T_1 - 0.03 T_2 + 0.01 T_3, with 5 % Gaussian noise (sigma =
    0.05 |mu|; the "likelihood" category's 1e-12 error floor would swamp cgs fluxes).  Justified in the commit message: the
    solve equals Prospector's compute_response on its own reference dump to 1e-15
    (tests/test_noise_calibration.py), and the profiled ln L equals the maximum over the sampled
    spectrum_scaling / spectrum_calib (rel 1e-9)."""
    from ceridwen.likelihood.noise_model import DiagonalNoiseModel
    from ceridwen.likelihood.likelihood import DiagonalGaussianLikelihood
    from ceridwen.likelihood.poly_calibration import (PolynomialCalibration,
                                                      chebyshev_design_matrix)
    A = chebyshev_design_matrix(wave, np.asarray(mask), 3)
    sig = 0.05 * jnp.abs(mu)
    y_cal = (mu * (1.0 + jnp.asarray(A) @ jnp.array([0.0, 0.05, -0.03, 0.01]))
             + sig * jax.random.normal(key, mu.shape))
    pc = PolynomialCalibration(A)
    pc_reg = PolynomialCalibration(A, regularization=[0.0, 30.0, 60.0, 100.0])
    out = {}
    for tag, pcal in (("", pc), ("_reg", pc_reg)):
        c, resp = pcal.solve(y_cal, mu, 1.0 / sig ** 2, mask)
        lnl, _ = DiagonalGaussianLikelihood(poly_calibration=pcal)(y_cal, mu, sig, mask)
        out[f"coeffs{tag}"] = np.asarray(c, dtype=np.float64)
        out[f"response{tag}"] = np.asarray(resp, dtype=np.float64)
        out[f"lnl{tag}"] = np.asarray(lnl, dtype=np.float64)
    jit = 0.05 * float(jnp.median(jnp.abs(mu)))
    th = {"log_jitter_spec": jnp.array([np.log(jit)])}
    nm = DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec")
    lnl_j, _ = DiagonalGaussianLikelihood(noise_model=nm, poly_calibration=pc)(
        y_cal, mu, sig, mask, th)
    out["lnl_jitter_spec"] = np.asarray(lnl_j, dtype=np.float64)
    return out


def _gp_likelihood_baseline(wave, mu, key) -> dict:
    """The compiled GP likelihood of a spectrum (GPGaussianLikelihood).  Data = the
    "csp_spectrum" prediction plus noise drawn from the GP covariance itself, sigma = 0.05 |mu|
    times L z with L L^T = I + a^2 SE(l) (a = 1, l = 100 A: ~4 pixels of this grid), and a
    mask (every 13th pixel and pixels 60-79).  Justified in the commit message: the fixed-
    hyperparameter values equal the independent numpy GaussianProcess.log_likelihood plus the
    sigma_eff normalisation to rel 1e-10 (tests/likelihood/test_gp_likelihood.py), and the
    ln a = -30 value differs from the diagonal Gaussian by the analytic eps term."""
    from ceridwen.likelihood.noise_model import DiagonalNoiseModel
    from ceridwen.likelihood.gp_likelihood import (GPGaussianLikelihood, gp_sqdist,
                                                   GP_JITTER)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    n = mu.size
    sig = 0.05 * jnp.abs(mu)
    D = gp_sqdist(wave)
    C = np.exp(-0.5 * np.asarray(D) / 100.0 ** 2) + (1.0 + GP_JITTER) * np.eye(n)
    y = mu + sig * (jnp.asarray(np.linalg.cholesky(C)) @ jax.random.normal(key, (n,)))
    idx = np.arange(n)
    mask = jnp.asarray((idx % 13 != 0) & ~((idx >= 60) & (idx < 80)))
    full = jnp.ones(n, dtype=bool)
    fixed = GPGaussianLikelihood(DiagonalNoiseModel(), D, log_amp=0.0, log_len=np.log(100.0))
    nm = DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec")
    sampled = GPGaussianLikelihood(nm, D, log_amp="log_gp_amp_spec",
                                   log_len="log_gp_length_spec")
    th = {"log_gp_amp_spec": jnp.array([np.log(0.8)]),
          "log_gp_length_spec": jnp.array([np.log(70.0)]),
          "log_jitter_spec": jnp.array([np.log(0.02 * float(jnp.median(jnp.abs(mu))))])}
    small = GPGaussianLikelihood(DiagonalNoiseModel(), D, log_amp=-30.0, log_len=np.log(100.0))
    lnl_f, aux_f = fixed(y, mu, sig, full)
    lnl_m, aux_m = fixed(y, mu, sig, mask)
    lnl_s, _ = sampled(y, mu, sig, mask, th)

    def f(t):
        return sampled(y, mu, sig, mask, t)[0]
    g = jax.grad(f)(th)
    return {
        "lnl_fixed": np.asarray(lnl_f, dtype=np.float64),
        "lnl_pointwise_fixed": np.asarray(aux_f.lnl_pointwise, dtype=np.float64),
        "lnl_masked": np.asarray(lnl_m, dtype=np.float64),
        "lnl_pointwise_masked": np.asarray(aux_m.lnl_pointwise, dtype=np.float64),
        "lnl_sampled": np.asarray(lnl_s, dtype=np.float64),
        "grad_sampled": np.array([float(g[k][0]) for k in sorted(g)], dtype=np.float64),
        "lnl_small_amp": np.asarray(small(y, mu, sig, mask)[0], dtype=np.float64),
        "gp_mean_fixed": np.asarray(fixed.conditional_mean(y, mu, sig, mask), dtype=np.float64)
                         / float(jnp.median(sig)),
    }


def _poly_marginal_baseline(wave, mu, mask, key) -> dict:
    """The analytically marginalised calibration polynomial
    (Spectrum(polynomial_mode="marginalize"), ceridwen/likelihood/poly_marginal.py).  Data = the
    "csp_spectrum" prediction times 1 + 0.02 T_0 + 0.05 T_1 - 0.03 T_2 + 0.01 T_3 with 5 %
    Gaussian noise.  Three prior settings (order 3): widths (0.1, 0.1, 0.05, 0.05), T_0 pinned
    (0, 0.1, 0.1, 0.1), and the first with a sampled jitter.  Justified by the brute force:
    before anything is returned, every marginal ln L is checked here against the dense
    ln N(y - mu; 0, C + D Lambda D^T) (numpy slogdet / solve) to rel 1e-10, and every
    conditional mean against the dense posterior mean (tests/likelihood/test_poly_marginal.py
    does the same on random problems)."""
    from ceridwen.likelihood.noise_model import DiagonalNoiseModel
    from ceridwen.likelihood.poly_calibration import chebyshev_design_matrix
    from ceridwen.likelihood.poly_marginal import (PolyMarginalGaussianLikelihood,
                                                   PolynomialMarginal)
    A = chebyshev_design_matrix(wave, np.asarray(mask), 3)
    sig = 0.05 * jnp.abs(mu)
    y = (mu * (1.0 + jnp.asarray(A) @ jnp.array([0.02, 0.05, -0.03, 0.01]))
         + sig * jax.random.normal(key, mu.shape))
    jit = 0.05 * float(jnp.median(jnp.abs(mu)))
    th = {"log_jitter_spec": jnp.array([np.log(jit)])}
    m = np.asarray(mask, dtype=bool)
    out = {}
    for tag, s_prior, nm, extra in (
            ("", [0.1, 0.1, 0.05, 0.05], DiagonalNoiseModel(), 0.0),
            ("_pinned", [0.0, 0.1, 0.1, 0.1], DiagonalNoiseModel(), 0.0),
            ("_jitter", [0.1, 0.1, 0.05, 0.05],
             DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec"), jit)):
        lh = PolyMarginalGaussianLikelihood(noise_model=nm,
                                            poly_marginal=PolynomialMarginal(A, s_prior))
        lnl, aux = lh(y, mu, sig, mask, th)
        mean, cov, resp = lh.conditional(y, mu, sig, mask, th)
        # brute force (float64 numpy, n x n)
        D = (np.asarray(mu)[:, None] * A)[m]
        S = (np.diag(np.asarray(sig)[m] ** 2 + extra ** 2)
             + D @ np.diag(np.asarray(s_prior) ** 2) @ D.T)
        r = np.asarray(y - mu)[m]
        ref = -0.5 * r @ np.linalg.solve(S, r) - 0.5 * np.linalg.slogdet(2 * np.pi * S)[1]
        ref_mean = np.diag(np.asarray(s_prior) ** 2) @ D.T @ np.linalg.solve(S, r)
        assert abs(float(lnl) / ref - 1.0) < 1e-10, (tag, float(lnl), ref)
        assert np.allclose(np.asarray(mean), ref_mean, rtol=1e-8, atol=1e-12), tag
        out[f"lnl{tag}"] = np.asarray(lnl, dtype=np.float64)
        out[f"lnl_pointwise{tag}"] = np.asarray(aux.lnl_pointwise, dtype=np.float64)
        out[f"coeff_mean{tag}"] = np.asarray(mean, dtype=np.float64)
        out[f"coeff_cov{tag}"] = np.asarray(cov, dtype=np.float64)
        out[f"response{tag}"] = np.asarray(resp, dtype=np.float64)
    return out


STELLAR_MASS_TABLES = REPO_ROOT / "tests" / "reference" / "ssp_stellar_mass.npz"


def _stellar_mass_baseline(p) -> dict:
    """Surviving-mass fraction (v1.0.6, SSP schema 3) on the canonical test grid with the
    FSPS mass table stored in tests/reference/ssp_stellar_mass.npz (matched by chash; the
    category is absent when the grid is another one).  mfrac of a two-burst SFH and of a
    constant SFH on a 0-10 Gyr grid, in both SFH schemes, at logzsol = -0.4 (constant) and
    for a metallicity history.  Justified in the commit message: the constant-SFH values
    equal an independent analytic integral of the table to 1e-10 (tests/test_stellar_mass.py),
    and the table equals FSPS's stellar_mass at the nodes (examples/recipes/reference_mfrac.json
    burst values)."""
    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.cosmology import Cosmology
    if not STELLAR_MASS_TABLES.is_file():
        return {}
    grid = SSPData.load(SSP_FILE)
    with np.load(STELLAR_MASS_TABLES) as z:
        tag = next((k[:-len("/chash")] for k in z.files
                    if k.endswith("/chash") and str(z[k]) == grid.chash), None)
        if tag is None:
            return {}
        grid = grid.with_stellar_mass(np.array(z[f"{tag}/mass"]), source=str(z[f"{tag}/source"]))
    # 0-10 Gyr: every bin stays below the grid's second-oldest SSP age (BPASS 10^10.1 yr), where
    # the "linear" scheme lost node n-2's share of the top SSP interval until B1-016 (2026-09-24)
    n = 10
    lb = jnp.linspace(0.0, 10.0, n)
    sfh = (jnp.exp(-0.5 * ((lb - 0.05) / 0.03) ** 2)
           + 0.7 * jnp.exp(-0.5 * ((lb - 8.0) / 0.8) ** 2))
    out = {}
    for interp in ("step", "linear"):
        for zh in ("const", "var"):
            theta = {"lookback_time": lb, "sfh": sfh}
            if zh == "const":
                theta["logzsol"] = jnp.array([LOGZSOL_INTERIOR])
            else:
                theta["logzsol_hist"] = jnp.linspace(LOGZSOL_INTERIOR, -1.2, n)
            csp = CSPBasis(grid, theta=theta, cosmo=Cosmology.planck18(),
                           zh_const=(zh == "const"), add_neb=False, add_dust=False,
                           add_diffuse_dust=False, add_igm=False, verbose=False,
                           sfh_interp=interp)
            th = dict(csp.theta_init)
            vals = [csp.surviving_mass_fraction(th),
                    csp.surviving_mass_fraction(dict(th, sfh=jnp.ones(n)))]
            out[f"mfrac_{interp}_{zh}zh"] = np.asarray(vals, dtype=np.float64)
    return {"stellar_mass": out}


def _eline_marginal_baseline(p, ssp_data) -> dict[str, np.ndarray]:
    """Emission-line marginalisation (Spectrum(marginalize_elines=True)) at z = 2: Spectrum
    (rest 4700-6800 A, R_fwhm = 1000) + Photometry + Lines, nebular parameters fixed; mock =
    the grid model with [O III] 5007 x3 and Ha x0.6 plus PRNGKey(noise_seed) noise.  Joint
    marginal ln L and line-flux posterior for a flat prior and eline_prior_width = 0.2.
    Validated by tests/test_eline_marginalisation.py (exact vs quadrature, vs Prospector's
    fit_mle_elines, width -> 0 equals the ordinary likelihood)."""
    from ceridwen import SedModel, Cosmology
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.observation import Photometry, Spectrum, Lines
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood.likelihood import MultiObservationLikelihood
    from ceridwen.likelihood.eline_marginal import eline_line_fluxes

    z = p["zred"]
    cosmo = Cosmology.planck18()
    csp = CSPBasis(ssp_data, lookback_time=jnp.linspace(0.0, float(cosmo.age(z)), 6),
                   cosmo=cosmo, zh_const=True, sfh_interp="step", add_dust=False,
                   add_diffuse_dust=True, add_neb=True, add_igm=True, sps_home=SPS_HOME,
                   verbose=False)
    fixed = {k: jnp.atleast_1d(jnp.asarray(v)) for k, v in csp.theta_init.items()}
    # captured at the native axis value log10 Z = -2.0; in logzsol that is the same
    # physical metallicity, -2.0 - log10 Z_sun (BPASS: -0.30102999566398125)
    fixed.update(sfh=jnp.array([1.0, 1.0, 0.6, 0.3, 0.2, 0.1]),
                 logzsol=jnp.array([-2.0 - float(ssp_data.log10_zsun)]),
                 diffuse_tau_kc=jnp.array([p["diffuse_tau_kc"]]),
                 diffuse_dust_index=jnp.array([p["diffuse_dust_index"]]),
                 gas_logu=jnp.array([-2.3]), gas_logz=jnp.array([-0.3]))
    transforms = {k: (lambda th, v=v: v) for k, v in fixed.items()}
    wave = np.exp(np.arange(np.log(4700 * (1 + z)), np.log(6800 * (1 + z)), 1 / (2.3548 * 1000) / 2.5))
    lines_w = [4862.763, 5008.314, 6564.723, 6585.369]
    rows = [int(np.argmin(np.abs(np.asarray(csp.neb.nebem_line_pos) - w))) for w in lines_w]
    kin = Kinematics(sigma_gal=150.0, sigma_gas=90.0)

    def build(width, flux=None, phot_y=None, lines_y=None):
        spec = Spectrum(wavelength=wave, flux=flux, uncertainty=None if flux is None else unc,
                        instrument=Instrument.R_fwhm(1000.0), name="spec",
                        marginalize_elines=True, eline_prior_width=width)
        phot = Photometry(filters=["twomass_J", "twomass_H", "twomass_Ks"], flux=phot_y,
                          uncertainty=None if phot_y is None else 0.05 * np.abs(phot_y), name="phot")
        lin = Lines(line_ind=rows, wavelength=lines_w, flux=lines_y,
                    uncertainty=None if lines_y is None else 0.1 * np.abs(lines_y), name="lines")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SedModel(csp, [spec, phot, lin], transforms=transforms, zred=z,
                            free_param_init={"logmass": jnp.array([p["logmass"]])}, kinematics=kin)

    unc = None
    m0 = build(0.0)
    th = {"logmass": jnp.array([p["logmass"]])}
    pred, aux = m0.predict_with_elines(th)
    es = m0._eline_system
    boost = np.ones(es.m)
    boost[list(es.names).index("[O III] 5007")] = 3.0
    boost[list(es.names).index("Ba-alpha 6563")] = 0.6
    F = np.asarray(aux["prior_mean"]) * boost
    ys = np.asarray(pred["spec"]) + np.asarray(aux["cols"]["spec"]) @ F
    yp = np.asarray(pred["phot"], float) + np.asarray(aux["cols"]["phot"]) @ F
    yl = np.asarray(pred["lines"]) + np.asarray(aux["cols"]["lines"]) @ F
    unc = np.full(wave.size, 0.03 * float(np.median(ys)))
    ys = ys + unc * np.asarray(jax.random.normal(jax.random.PRNGKey(p["noise_seed"]), ys.shape))

    res = {"wave_rest": np.asarray(es.wave_rest), "cloudy": np.asarray(aux["prior_mean"])}
    for tag, width in (("flat", 0.0), ("prior", 0.2)):
        m = build(width, ys, yp, yl)
        lh = MultiObservationLikelihood(
            keys=tuple(m.obs_dict), likelihoods=tuple(_likelihood_for(o, m.param_names)
                                                      for o in m.observations))

        class _P:
            def log_prob(self, t):
                return 0.0
        res[f"lnl_{tag}"] = np.asarray(lh.make_lnprobfn(m.obs_dict, m, _P())(th), dtype=np.float64)
        post = eline_line_fluxes(m, th, lh)
        res[f"mean_{tag}"] = np.asarray(post["mean"])
        res[f"sd_{tag}"] = np.asarray(post["sd"])
    return res


def _lsf_scale_baseline(p, ssp_data) -> dict[str, np.ndarray]:
    """Instrumental LSF scale (``Instrument(..., scale=...)``, sigma_inst -> s sigma_inst in
    the continuum kernel and the line widths) at z = 2: a Spectrum (rest 4700-6800 A,
    R_fwhm = 1500, nebular lines painted, sigma_gal 150 / sigma_gas 90 km/s) with the scale
    SAMPLED in ``p["lsf_range"]`` and evaluated at each of ``p["lsf_values"]``, and FIXED at
    ``p["lsf_fixed"]``.

    Justified independently (not "whatever the code printed"): the fixed-scale spectrum equals
    the one of ``Instrument.R_fwhm(1500 / s)`` built directly (asserted here, rtol 1e-12), and
    at the top of the range the sampled spectrum equals the fixed-scale one (asserted here,
    same log grid and band there, rtol 1e-12); elsewhere tests/test_lsf_scale.py checks it on
    a matched grid to 2 x erfc(5/sqrt 2) (measured < 1e-8)."""
    import warnings
    from ceridwen import SedModel, Cosmology
    from ceridwen.csp.csp import CSPBasis
    from ceridwen.observation import Spectrum

    z = p["zred"]
    cosmo = Cosmology.planck18()
    csp = CSPBasis(ssp_data, lookback_time=jnp.linspace(0.0, float(cosmo.age(z)), 6),
                   cosmo=cosmo, zh_const=True, sfh_interp="step", add_dust=False,
                   add_diffuse_dust=True, add_neb=True, add_igm=False, sps_home=SPS_HOME,
                   verbose=False)
    fixed = {k: jnp.atleast_1d(jnp.asarray(v)) for k, v in csp.theta_init.items()}
    fixed.update(sfh=jnp.array([1.0, 1.0, 0.6, 0.3, 0.2, 0.1]),
                 logzsol=jnp.array([-0.3]),
                 diffuse_tau_kc=jnp.array([p["diffuse_tau_kc"]]),
                 diffuse_dust_index=jnp.array([p["diffuse_dust_index"]]),
                 gas_logu=jnp.array([-2.3]), gas_logz=jnp.array([-0.3]))
    transforms = {k: (lambda th, v=v: v) for k, v in fixed.items()}
    wave = np.exp(np.arange(np.log(4700 * (1 + z)), np.log(6800 * (1 + z)),
                            1 / (2.3548 * 1500) / 2.5))
    kin = Kinematics(sigma_gal=150.0, sigma_gas=90.0)
    lo, hi = p["lsf_range"]

    def build(ins, extra=None):
        init = {"logmass": jnp.array([p["logmass"]])}
        init.update(extra or {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SedModel(csp, [Spectrum(wavelength=wave, instrument=ins, name="spec")],
                            transforms=transforms, zred=z, free_param_init=init, kinematics=kin)

    def spec(m, **kw):
        th = {"logmass": jnp.array([p["logmass"]])}
        th.update({k: jnp.array([v]) for k, v in kw.items()})
        return np.asarray(m.predict(th)["spec"], dtype=np.float64)

    ms = build(Instrument.R_fwhm(1500.0, scale="lsf_scale", scale_range=(lo, hi)),
               {"lsf_scale": jnp.array([1.0])})
    res = {f"spec_sampled_{i}": spec(ms, lsf_scale=v) for i, v in enumerate(p["lsf_values"])}
    s = p["lsf_fixed"]
    res["spec_fixed"] = spec(build(Instrument.R_fwhm(1500.0, scale=s)))
    np.testing.assert_allclose(res["spec_fixed"], spec(build(Instrument.R_fwhm(1500.0 / s))),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(spec(ms, lsf_scale=hi), spec(build(Instrument.R_fwhm(1500.0, scale=hi))),
                               rtol=1e-12, atol=0)
    return res


# --------------------------------------------------------------------------- #
#  Save (run directly)
# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="write only these categories (never overwrite the others)")
    args = ap.parse_args()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baselines = compute_baselines()
    if args.only is not None:
        unknown = sorted(set(args.only) - set(baselines))
        if unknown:
            raise SystemExit(f"unknown categories {unknown}; known: {sorted(baselines)}")
        baselines = {k: v for k, v in baselines.items() if k in args.only}
    for category, arrays in baselines.items():
        np.savez(BASELINE_DIR / f"{category}.npz", **arrays)
        print(f"  wrote {category}.npz  ({', '.join(arrays.keys())})")
    # Record the fixed parameters in a transparent, version-stable format.
    # Nothing reads this back at test time (fixed_params() is called directly);
    # it is a human-readable provenance record, so JSON beats a pickle.
    def _jsonable(v):
        if isinstance(v, (jnp.ndarray, np.ndarray)):
            return np.asarray(v).tolist()
        return v

    params_json = {k: _jsonable(v) for k, v in fixed_params().items()}
    with open(BASELINE_DIR / "params.json", "w") as f:
        json.dump(params_json, f, indent=2)
    print(f"  wrote params.json")
    print(f"\nBaselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
