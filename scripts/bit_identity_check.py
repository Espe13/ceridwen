#!/usr/bin/env python3
"""Byte-identity check of the CERIDWEN forward model, post-processing and (optionally)
a short nested-sampling run between two versions of the package.

    git stash / checkout <old>;  python scripts/bit_identity_check.py --save  /tmp/bits_old.npz
    git checkout <new>;          python scripts/bit_identity_check.py --compare /tmp/bits_old.npz

Every array is compared with ``tobytes()`` (dtype, shape and every bit).  Add ``--ns`` to
include a small nested-sampling run (same rng key; samples, log_weights, logL, logZ) with the
package's default slice kernel (keys ``plain/ns/...``) and, when the adapter has
``slice_kernel=``, the same run with every kernel named explicitly (keys ``plain/ns_<kernel>/...``),
which on CPU must be byte-identical to the default run of the same version (checked in every
mode; exit 1 if not).
Needs the test grid (see tests/_gridfixture.py); the nebular block runs only with $SPS_HOME.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

import jax                                                     # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

from _gridfixture import find_test_grid                        # noqa: E402
from ceridwen import SSPData, CSPBasis, SedModel, Cosmology    # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines   # noqa: E402
from ceridwen.broadening import Kinematics, Instrument         # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh     # noqa: E402
from ceridwen.postprocess import PostProcess                   # noqa: E402
from ceridwen.sampler.runner import SamplingResult             # noqa: E402

ZRED = 0.8
N_TIME = 5
FILTERS = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.linspace(3800.0, 8800.0, 600)
LINES = {"Hbeta": 4862.71, "OIII5007": 5008.24, "Halpha": 6564.61}
N_THETA = 16
SEED = 20260906


def _flat(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flat(f"{prefix}/{i}", v, out)
    elif obj is None or isinstance(obj, str):
        return
    else:
        out[prefix] = np.asarray(obj)


def _build(add_neb, add_igm, sigma_losvd, dust_emission, gas_tied=False, igm_dla=False, lsf=None,
           interp="step", track=False, zred=ZRED, spec_wave=SPEC_WAVE, filters=FILTERS,
           marg=False, line_windows=False, support=False, agn=None):
    path = find_test_grid()
    if path is None:
        sys.exit("no test SSP grid found (tests/_gridfixture.py)")
    ssp = SSPData.load(str(path))
    cosmo = Cosmology.planck18()
    kw = dict(lookback_time=jnp.linspace(0.0, float(cosmo.age(0.0 if track else zred)), N_TIME),
              zh_const=True, sfh_interp=interp, add_dust=True, add_diffuse_dust=True,
              add_neb=add_neb, add_igm=add_igm, add_dust_emission=dust_emission,
              verbose=False, cosmo=cosmo)
    if track:                          # grid built to age(0), rescaled to age(ZRED) in the forward pass
        kw["track_zred_age"] = True
    if gas_tied:
        kw["gas_tied"] = True          # v1.0.5 path: gas_logz := logzsol
    if support:                        # only the model pixels the observations read
        kw["wavelength_support"] = True
    if igm_dla:                        # IGM damping wing + DLA with sampled theta keys
        from ceridwen.igm import MadauDampingDLA
        kw["igm_model"] = MadauDampingDLA(Ob0=0.04897)
    if add_neb or dust_emission or agn is not None:
        kw["sps_home"] = os.environ["SPS_HOME"]
    if agn is not None:                # ceridwen.agn: AGNConfig kwargs
        from ceridwen.agn import AGNConfig
        kw["agn"] = AGNConfig(**agn)
    csp = CSPBasis(ssp, **kw)
    sfh_times_yr = np.array(csp.sfh_times)

    def _sfh(t, _t=sfh_times_yr):
        return logsfr_ratios_to_sfh(t["logsfr_ratios"], sfh_times_yr=_t)
    _sfh.__name__ = "logsfr_ratios_to_sfh"

    obs = [Photometry(filters=filters, flux=[2e-9] * len(filters),
                      uncertainty=[2e-10] * len(filters), name="phot"),
           Spectrum(wavelength=spec_wave * (1.0 + zred), flux=np.full(spec_wave.size, 1e-9),
                    uncertainty=np.full(spec_wave.size, 1e-10),
                    instrument=(Instrument.sigma_kms(150.0) if lsf is None else
                                # LSF scale: fixed float, or sampled ("lsf_scale") in (0.8, 1.3)
                                Instrument.sigma_kms(150.0, scale=lsf, scale_range=(
                                    (0.8, 1.3) if isinstance(lsf, str) else None))),
                    name="spec", **({"marginalize_elines": True} if marg else {}))]
    if add_neb:
        obs.append(Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                         wavelength=np.array(list(LINES.values())),
                         flux=np.full(len(LINES), 1e-17), uncertainty=np.full(len(LINES), 1e-18),
                         name="lines"))
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    if igm_dla:
        init.update(x_HI=jnp.array([0.5]), logN_HI=jnp.array([20.5]))
    if isinstance(lsf, str):
        init[lsf] = jnp.array([1.1])
    if agn is not None:                # every AGN key of this configuration is sampled
        vals = {"fagn": 0.3, "agn_tau": 20.0, "fagn_bbb": 0.5, "dust4": 0.3,
                "dust4_index": -1.2, "agn_elum": 1e-3, "agn_eline_sigma": 200.0}
        init.update({k: jnp.array([vals[k]]) for k in csp._agn.param_names})
    transforms = {"sfh": _sfh}
    kin = Kinematics(sigma_gal=sigma_losvd)
    if marg:
        # marginalised line fluxes (the nebular parameters fixed, as eline_marginal requires)
        # with a sampled sigma_gas: the line basis is rebuilt at every call
        transforms.update(gas_logu=lambda th: jnp.array([-2.5]),
                          gas_logz=lambda th: jnp.array([-0.3]))
        init["sigma_gas"] = jnp.array([120.0])
        kin = Kinematics(sigma_gal=sigma_losvd, sigma_gas="sigma_gas", sigma_max=600.0)
    model = SedModel(csp, obs, priors={}, transforms=transforms,
                     free_param_init=init, zred=zred, kinematics=kin,
                     # a sampled sigma_gas in the joint marginal needs unbroadened photometry
                     **({"broaden_photometry": False} if marg else {}))
    proj = model.obs_dict["spec"]._proj
    if line_windows and hasattr(proj, "enable_line_windows"):
        # the windowed line basis (opt-in); a package without it runs the dense one, which is
        # what the windowed one must equal bit for bit
        if proj.enable_line_windows() is None:
            sys.exit("bit_identity_check: no line window for the neb_marg_windows configuration")
    return model


def _draws(model, n, rng):
    theta = {}
    for name in model.param_names:
        base = np.asarray(model.theta_init[name], dtype=float)
        theta[name] = base[None, :] + 0.05 * rng.standard_normal((n,) + base.shape)
    return theta


def _forward(model, theta_np, tag, out):
    csp = model.csp
    n = next(iter(theta_np.values())).shape[0]
    for i in range(n):
        one = {k: jnp.asarray(v[i]) for k, v in theta_np.items()}
        pred = model._predict_jit_fn(one)
        _flat(f"{tag}/predict/{i}", pred, out)
        t = model.apply_transforms(one)
        t["zred"] = jnp.array([model.zred])
        cont, lines = csp.get_spectrum_components(t)
        out[f"{tag}/components/{i}/cont"] = np.asarray(cont)
        out[f"{tag}/components/{i}/lines"] = np.asarray(lines)
        out[f"{tag}/full/{i}"] = np.asarray(csp.get_spectrum(t, include_lines=True))
        out[f"{tag}/intrinsic/{i}"] = np.asarray(csp.get_spectrum_nodattn_nodem_noneb(t))
        if getattr(csp, "neb", None) is not None:
            out[f"{tag}/dustfree/{i}"] = np.asarray(csp.get_spectrum_nodattn_nodem_neb(t, include_lines=True))
            out[f"{tag}/line_fluxes/{i}"] = np.asarray(csp.predict_line_fluxes(t))


def _postprocess(model, theta_np, tag, out):
    n = next(iter(theta_np.values())).shape[0]
    rng = np.random.default_rng(SEED + 1)
    samples = {k: jnp.asarray(v[:, 0] if v.shape[1:] == (1,) else v) for k, v in theta_np.items()}
    ll = rng.standard_normal(n)
    llb = ll - rng.uniform(0.5, 2.0, n)
    res = SamplingResult(samples=samples, log_evidence=float("nan"), log_evidence_err=float("nan"),
                         log_weights=jnp.zeros(n), log_likelihoods=jnp.asarray(ll),
                         param_names=list(model.param_names), n_likelihood_calls=n,
                         wall_time_s=0.0, sampler_name="fake", log_likelihoods_birth=jnp.asarray(llb))
    pp = PostProcess(model, res, n_samples=n, seed=0, batch_size=7,
                     derived={"f5500": lambda s: s.full[s.index_of(5500.0)]})
    o = pp.run()
    o = {k: v for k, v in o.items() if k != "meta"}
    _flat(f"{tag}/postprocess", o, out)
    _mfrac(model, theta_np, tag, out)


def _mfrac(model, theta_np, tag, out):
    """v1.0.6 surviving-mass fraction per draw, with the FSPS table stored for the test grid
    (tests/reference/ssp_stellar_mass.npz, matched by chash); skipped on older packages."""
    csp = model.csp
    ref = HERE.parent / "tests" / "reference" / "ssp_stellar_mass.npz"
    if not hasattr(csp, "surviving_mass_fraction") or not ref.is_file():
        return
    with np.load(ref) as z:
        tag_m = next((k[:-6] for k in z.files if k.endswith("/chash")
                      and str(z[k]) == getattr(csp, "grid_chash", None)), None)
        if tag_m is None:
            return
        mass = np.array(z[f"{tag_m}/mass"])
    saved = csp.ssp_stellar_mass
    csp.ssp_stellar_mass = mass
    try:
        f = jax.jit(csp.surviving_mass_fraction)
        n = next(iter(theta_np.values())).shape[0]
        for i in range(n):
            t = model.apply_transforms({k: jnp.asarray(v[i]) for k, v in theta_np.items()})
            out[f"{tag}/mfrac/{i}"] = np.asarray(f(t))
    finally:
        csp.ssp_stellar_mass = saved


def _likelihood(model, theta_np, tag, out, gauss_only=False):
    """fitSED's sampled log-posterior (MultiObservationLikelihood.make_lnprobfn): value and
    gradient at every draw, plus the SHA-256 of the lowered StableHLO of jit(value_and_grad).  Then
    the outlier-mixture variant (f_outlier_spec sampled, f_outlier_phot fixed), the profiled
    calibration and the spectrum GP likelihood (ln a, ln l sampled), each when the package has
    it."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood import DiagonalGaussianLikelihood, MultiObservationLikelihood
    rng = np.random.default_rng(SEED + 2)
    obs = model.obs_dict
    keys = tuple(obs)
    y_fake = {k: np.asarray(model.predict({n: jnp.asarray(v[0]) for n, v in theta_np.items()})[k])
              for k in keys}
    saved = {k: (obs[k].flux, obs[k].uncertainty) for k in keys}
    for k in keys:                      # data = draw 0's prediction + 10 % noise, 3 % outliers
        o = obs[k]
        noise = 0.1 * np.abs(y_fake[k]) * rng.standard_normal(y_fake[k].shape)
        bad = rng.uniform(size=y_fake[k].shape) < 0.03
        o.flux = jnp.asarray(y_fake[k] + noise + bad * 20.0 * 0.1 * np.abs(y_fake[k]))
        o.uncertainty = jnp.asarray(0.1 * np.abs(y_fake[k]) + 1e-30)
    variants = [("gauss", tuple(_likelihood_for(obs[k], model.param_names) for k in keys))]
    try:
        from ceridwen.likelihood import DiagonalNoiseModel
        nm = {"phot": DiagonalNoiseModel(f_outlier=0.05, nsigma_outlier=20.0),
              "spec": DiagonalNoiseModel(f_outlier="f_outlier_spec"),
              "lines": DiagonalNoiseModel()}
        variants.append(("outlier", tuple(DiagonalGaussianLikelihood(nm[k]) for k in keys)))
    except TypeError:
        print("  (package has no outlier model: variant skipped)")
    try:        # v1.0.7: per-observation noise key + profiled calibration polynomial
        from ceridwen.likelihood.poly_calibration import (PolynomialCalibration,
                                                          chebyshev_design_matrix)
        sp = obs["spec"]
        pc = PolynomialCalibration(chebyshev_design_matrix(sp.wavelength, sp.mask, 3))
        nm = {"phot": DiagonalNoiseModel(use_error_scale=True,
                                         err_scale_key="log_err_scale_phot"),
              "spec": DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec"),
              "lines": DiagonalNoiseModel()}
        variants.append(("polycal", tuple(
            DiagonalGaussianLikelihood(nm[k], poly_calibration=pc if k == "spec" else None)
            for k in keys)))
    except ImportError:
        print("  (package has no profiled calibration: variant skipped)")
    try:        # GP likelihood of the spectrum, hyperparameters sampled
        from ceridwen.likelihood.gp_likelihood import GPGaussianLikelihood, gp_sqdist
        sp = obs["spec"]
        gp_lh = GPGaussianLikelihood(DiagonalNoiseModel(), gp_sqdist(sp.wavelength),
                                     log_amp="log_gp_amp_spec", log_len="log_gp_length_spec")
        variants.append(("gp", tuple(gp_lh if k == "spec" else DiagonalGaussianLikelihood()
                                     for k in keys)))
    except ImportError:
        print("  (package has no GP likelihood: variant skipped)")
    try:        # the calibration polynomial marginalised analytically
        from ceridwen.likelihood.poly_marginal import (PolyMarginalGaussianLikelihood,
                                                       PolynomialMarginal)
        sp = obs["spec"]
        pm = PolynomialMarginal(chebyshev_design_matrix(sp.wavelength, sp.mask, 3),
                                [0.1, 0.1, 0.05, 0.05])
        nm = {"phot": DiagonalNoiseModel(), "lines": DiagonalNoiseModel(),
              "spec": DiagonalNoiseModel(use_jitter=True, jitter_key="log_jitter_spec")}
        variants.append(("polymarg", tuple(
            PolyMarginalGaussianLikelihood(nm[k], poly_marginal=pm) if k == "spec"
            else DiagonalGaussianLikelihood(nm[k]) for k in keys)))
    except ImportError:
        print("  (package has no marginalised calibration: variant skipped)")
    if gauss_only:          # marginalised lines: the other likelihoods are refused with them
        variants = variants[:1]
    n = next(iter(theta_np.values())).shape[0]
    f_draws = rng.uniform(1e-4, 0.3, (n, 1))
    for vname, lhs in variants:
        lnprob = MultiObservationLikelihood(keys=keys, likelihoods=lhs).make_lnprobfn(
            obs, model, model)
        vg = jax.jit(jax.value_and_grad(lnprob))
        for i in range(n):
            one = {k: jnp.asarray(v[i]) for k, v in theta_np.items()}
            if vname == "outlier":
                one["f_outlier_spec"] = jnp.asarray(f_draws[i])
            if vname in ("polycal", "polymarg"):
                one["log_jitter_spec"] = jnp.log(0.05 * jnp.median(jnp.abs(obs["spec"].flux))
                                                 * (1.0 + f_draws[i]))
                one["log_err_scale_phot"] = jnp.asarray(f_draws[i])
            if vname == "gp":
                one["log_gp_amp_spec"] = jnp.log(jnp.asarray(f_draws[i]) * 3.0)
                one["log_gp_length_spec"] = jnp.log(30.0 + 300.0 * jnp.asarray(f_draws[i]))
            val, grad = vg(one)
            out[f"{tag}/lnprob/{vname}/{i}/value"] = np.asarray(val)
            _flat(f"{tag}/lnprob/{vname}/{i}/grad", grad, out)
        hlo = vg.lower(one).as_text().encode()            # 0.1-0.4 GB of text: keep a digest
        out[f"{tag}/lnprob/{vname}/stablehlo_sha256"] = np.frombuffer(
            hashlib.sha256(hlo).digest(), dtype=np.uint8)
    for k, (flux, unc) in saved.items():
        obs[k].flux, obs[k].uncertainty = flux, unc


def _nested(model, tag, out, kernel=None):
    """One short nested run; ``kernel=None`` passes no slice_kernel (the package default)."""
    from ceridwen import fitSED
    from ceridwen.sampler.priors import Uniform, StudentT
    model.priors = {"logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
                    "logmass": Uniform(low=9.0, high=11.0),
                    # the pre-v1.0.5 prior Uniform(-3.5, -1.5) on the absolute log10 Z, in
                    # logzsol: the SAME physical prior, so T4 compares like with like
                    "logzsol": Uniform(low=-3.5 - model.csp.log10_zsun,
                                       high=-1.5 - model.csp.log10_zsun),
                    "tau_pow": Uniform(low=0.0, high=2.0),
                    "alpha_pow": Uniform(low=-2.0, high=0.0),
                    "diffuse_tau_kc": Uniform(low=0.0, high=2.0),
                    "diffuse_dust_index": Uniform(low=-1.0, high=0.4)}
    missing = [k for k in model.param_names if k not in model.priors]
    if missing:
        sys.exit(f"bit_identity_check: no prior defined for {missing}")
    model.priors = {k: v for k, v in model.priors.items() if k in model.param_names}
    skw = {"num_live": 40, "num_delete": 8, "num_inner_steps": 10, "logZ_tol": -1.0}
    ns = "ns"
    if kernel is not None:          # the default run keeps the sampler_kwargs (and keys) of old
        skw["slice_kernel"] = kernel
        ns = f"ns_{kernel}"
    result = fitSED(model, sampler="nested", rng_key=jax.random.PRNGKey(7),
                    sampler_kwargs=skw, output_dir="/tmp/ceridwen_bits_ns", verbose=False)
    _flat(f"{tag}/{ns}/samples", result.samples, out)
    out[f"{tag}/{ns}/log_weights"] = np.asarray(result.log_weights)
    out[f"{tag}/{ns}/log_likelihoods"] = np.asarray(result.log_likelihoods)
    out[f"{tag}/{ns}/log_likelihoods_birth"] = np.asarray(result.log_likelihoods_birth)
    out[f"{tag}/{ns}/logZ"] = np.asarray([result.log_evidence, result.log_evidence_err])


def _explicit_kernels():
    """Every slice kernel, to be run by name (none in versions without ``slice_kernel=``)."""
    import inspect
    from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
    if "slice_kernel" not in inspect.signature(BlackJAXNestedSamplerAdapter).parameters:
        return []
    from ceridwen.sampler.nested import SLICE_KERNELS
    return list(SLICE_KERNELS)


def collect(args):
    import ceridwen
    print("package:", ceridwen.__file__)
    out = {}
    rng = np.random.default_rng(SEED)
    configs = [("plain", dict(add_neb=False, add_igm=False, sigma_losvd=0.0, dust_emission=False)),
               ("igm_losvd", dict(add_neb=False, add_igm=True, sigma_losvd=250.0, dust_emission=False))]
    if os.environ.get("SPS_HOME"):
        configs.append(("neb", dict(add_neb=True, add_igm=True, sigma_losvd=250.0, dust_emission=False)))
        configs.append(("neb_duste", dict(add_neb=True, add_igm=True, sigma_losvd=250.0, dust_emission=True)))
        configs.append(("neb_tied", dict(add_neb=True, add_igm=True, sigma_losvd=250.0,
                                         dust_emission=False, gas_tied=True)))
        # instrumental LSF scale (Instrument(scale=...)): fixed != 1 and sampled; appended
        # last so the draws of the configurations above are unchanged
        configs.append(("neb_lsf_fixed", dict(add_neb=True, add_igm=True, sigma_losvd=250.0,
                                              dust_emission=False, lsf=1.15)))
        configs.append(("neb_lsf_sampled", dict(add_neb=True, add_igm=True, sigma_losvd=250.0,
                                                dust_emission=False, lsf="lsf_scale")))
    else:
        print("SPS_HOME not set: nebular configurations skipped")
    # last, so the shared rng gives the earlier configurations the same draws as before
    try:
        from ceridwen.igm import MadauDampingDLA  # noqa: F401
        configs.append(("igm_dla", dict(add_neb=False, add_igm=True, sigma_losvd=250.0,
                                        dust_emission=False, igm_dla=True)))
    except ImportError:
        print("  (package has no MadauDampingDLA: igm_dla configuration skipped)")
    # the zred-tracked SFH grid (built to age(0), rescaled to age(ZRED) with its formed mass
    # kept) under the "linear" scheme; last, for the same reason
    configs.append(("tracked_linear", dict(add_neb=False, add_igm=True, sigma_losvd=250.0,
                                           dust_emission=False, interp="linear", track=True)))
    # the Ly-alpha break in the spectrum (z = 6): the IGM on the model grid (sigma_gal = 0) and
    # after the galaxy kernel (sigma_gal = 300); appended last, earlier draws unchanged
    brk = dict(add_neb=False, add_igm=True, dust_emission=False, zred=6.0,
               spec_wave=np.linspace(1100.0, 1400.0, 300), filters=["jwst_f090w", "jwst_f115w"])
    configs.append(("igm_break_s0", dict(brk, sigma_losvd=0.0)))
    configs.append(("igm_break_s300", dict(brk, sigma_losvd=300.0)))
    if os.environ.get("SPS_HOME"):
        # marginalised emission lines with sampled sigma_gas and LSF scale: the per-call line
        # basis (SpectralProjector.predict_with_line_basis), dense and with the static line
        # windows; appended last, earlier draws unchanged
        marg = dict(add_neb=True, add_igm=True, sigma_losvd=250.0, dust_emission=False,
                    lsf="lsf_scale", marg=True)
        configs.append(("neb_marg_dense", marg))
        configs.append(("neb_marg_windows", dict(marg, line_windows=True)))
    # CSPBasis(wavelength_support=True): the opt-in path that synthesises only the model
    # pixels the observations read (not byte-identical to the whole grid, so its own
    # configurations); appended last, earlier draws unchanged
    import inspect
    if "wavelength_support" in inspect.signature(CSPBasis.__init__).parameters:
        configs.append(("support_plain", dict(add_neb=False, add_igm=True, sigma_losvd=250.0,
                                              dust_emission=False, support=True)))
        if os.environ.get("SPS_HOME"):
            configs.append(("support_neb", dict(add_neb=True, add_igm=True, sigma_losvd=250.0,
                                                dust_emission=False, support=True)))
    else:
        print("  (package has no wavelength_support: support configurations skipped)")
    # ceridwen.agn (CSPBasis(agn=...)): the torus of each mode, the disc + dust4 screen + lines,
    # the energy balance and the C1 support path with the AGN on; appended last, earlier draws
    # unchanged
    try:
        import ceridwen.agn  # noqa: F401
        has_agn = "agn" in inspect.signature(CSPBasis.__init__).parameters
    except ImportError:
        has_agn = False
    if has_agn and os.environ.get("SPS_HOME"):
        base = dict(add_neb=True, add_igm=True, sigma_losvd=250.0, dust_emission=False)
        configs.append(("agn_fsps", dict(base, agn={"agn_mode": "fsps", "lines": True})))
        configs.append(("agn_prospector", dict(base, agn={"agn_mode": "prospector",
                                                          "dust4_type": 1})))
        configs.append(("agn_corrected_duste", dict(base, dust_emission=True,
                                                    agn={"agn_mode": "corrected",
                                                         "dust4_type": 1})))
        configs.append(("agn_corrected_support", dict(base, add_neb=False, support=True,
                                                      agn={"agn_mode": "corrected",
                                                           "dust4_type": 1})))
    elif not has_agn:
        print("  (package has no ceridwen.agn: AGN configurations skipped)")
    for tag, cfg in configs:
        t0 = time.perf_counter()
        model = _build(**cfg)
        theta = _draws(model, N_THETA, rng)
        _forward(model, theta, tag, out)
        _postprocess(model, theta, tag, out)
        _likelihood(model, theta, tag, out, gauss_only=cfg.get("marg", False))
        if args.ns and tag == "plain":
            _nested(model, tag, out)
            for kernel in _explicit_kernels():
                _nested(model, tag, out, kernel)
        print(f"{tag:<12} {len(out)} arrays so far  ({time.perf_counter() - t0:.1f} s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--ns", action="store_true")
    args = ap.parse_args()
    if not (args.save or args.compare):
        ap.error("--save PATH or --compare PATH")
    out = collect(args)
    # every named slice kernel against the default-kernel nested run of this same version
    kernel_bad, n_kernel = [], 0
    for k in out:
        parts = k.split("/")
        if len(parts) > 2 and parts[1].startswith("ns_"):
            n_kernel += 1
            a, b = out["/".join([parts[0], "ns", *parts[2:]])], out[k]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                kernel_bad.append(k)
    if n_kernel:
        print(f"named slice kernels vs this version's default-kernel run: {n_kernel} arrays, "
              f"{len(kernel_bad)} differ")
        for k in kernel_bad:
            print(f"  DIFF {k}")
    if args.save:
        np.savez(args.save, **out)
        print(f"saved {len(out)} arrays -> {args.save}")
        if kernel_bad and not args.compare:
            sys.exit(1)
    if args.compare:
        ref = np.load(args.compare)
        bad, missing = [], []
        for k in ref.files:
            if k not in out:
                missing.append(k); continue
            a, b = ref[k], out[k]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                bad.append(k)
        extra = [k for k in out if k not in ref.files]
        print(f"compared {len(ref.files)} arrays: {len(bad)} differ, {len(missing)} missing, {len(extra)} new")
        for k in bad[:40]:
            a, b = ref[k], out[k]
            d = (np.max(np.abs(a.astype(float) - b.astype(float))) if a.shape == b.shape
                 and a.dtype == b.dtype and a.dtype.kind in "fc" else "shape/dtype")
            print(f"  DIFF {k}: max|old-new| = {d}")
        for k in missing[:20]:
            print(f"  MISSING {k}")
        sys.exit(1 if (bad or missing or kernel_bad) else 0)


if __name__ == "__main__":
    main()
