#!/usr/bin/env python3
"""Byte-identity check of the CERIDWEN forward model, post-processing and (optionally)
a short nested-sampling run between two versions of the package.

    git stash / checkout <old>;  python scripts/bit_identity_check.py --save  /tmp/bits_old.npz
    git checkout <new>;          python scripts/bit_identity_check.py --compare /tmp/bits_old.npz

Every array is compared with ``tobytes()`` (dtype, shape and every bit).  Add ``--ns`` to
include a small nested-sampling run (same rng key; samples, log_weights, logL, logZ).
Needs the test grid (see tests/_gridfixture.py); the nebular block runs only with $SPS_HOME.
"""
from __future__ import annotations

import argparse
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


def _build(add_neb, add_igm, sigma_losvd, dust_emission):
    path = find_test_grid()
    if path is None:
        sys.exit("no test SSP grid found (tests/_gridfixture.py)")
    ssp = SSPData.load(str(path))
    cosmo = Cosmology.planck18()
    kw = dict(lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
              zh_const=True, sfh_interp="step", add_dust=True, add_diffuse_dust=True,
              add_neb=add_neb, add_igm=add_igm, add_dust_emission=dust_emission,
              verbose=False, cosmo=cosmo)
    if add_neb or dust_emission:
        kw["sps_home"] = os.environ["SPS_HOME"]
    csp = CSPBasis(ssp, **kw)
    sfh_times_yr = np.array(csp.sfh_times)

    def _sfh(t, _t=sfh_times_yr):
        return logsfr_ratios_to_sfh(t["logsfr_ratios"], sfh_times_yr=_t)
    _sfh.__name__ = "logsfr_ratios_to_sfh"

    obs = [Photometry(filters=FILTERS, flux=[2e-9] * len(FILTERS),
                      uncertainty=[2e-10] * len(FILTERS), name="phot"),
           Spectrum(wavelength=SPEC_WAVE * (1.0 + ZRED), flux=np.full(SPEC_WAVE.size, 1e-9),
                    uncertainty=np.full(SPEC_WAVE.size, 1e-10),
                    instrument=Instrument.sigma_kms(150.0), name="spec")]
    if add_neb:
        obs.append(Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                         wavelength=np.array(list(LINES.values())),
                         flux=np.full(len(LINES), 1e-17), uncertainty=np.full(len(LINES), 1e-18),
                         name="lines"))
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    model = SedModel(csp, obs, priors={}, transforms={"sfh": _sfh},
                     free_param_init=init, zred=ZRED,
                     kinematics=Kinematics(sigma_gal=sigma_losvd))
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
        t["zred"] = jnp.array([ZRED])
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


def _nested(model, tag, out):
    from ceridwen import fitSED
    from ceridwen.sampler.priors import Uniform, StudentT
    model.priors = {"logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
                    "logmass": Uniform(low=9.0, high=11.0),
                    "Z": Uniform(low=-3.5, high=-1.5),
                    "tau_pow": Uniform(low=0.0, high=2.0),
                    "alpha_pow": Uniform(low=-2.0, high=0.0),
                    "diffuse_tau_kc": Uniform(low=0.0, high=2.0),
                    "diffuse_dust_index": Uniform(low=-1.0, high=0.4)}
    missing = [k for k in model.param_names if k not in model.priors]
    if missing:
        sys.exit(f"bit_identity_check: no prior defined for {missing}")
    model.priors = {k: v for k, v in model.priors.items() if k in model.param_names}
    result = fitSED(model, sampler="nested", rng_key=jax.random.PRNGKey(7),
                    sampler_kwargs={"num_live": 40, "num_delete": 8, "num_inner_steps": 10,
                                    "logZ_tol": -1.0},
                    output_dir="/tmp/ceridwen_bits_ns", verbose=False)
    _flat(f"{tag}/ns/samples", result.samples, out)
    out[f"{tag}/ns/log_weights"] = np.asarray(result.log_weights)
    out[f"{tag}/ns/log_likelihoods"] = np.asarray(result.log_likelihoods)
    out[f"{tag}/ns/log_likelihoods_birth"] = np.asarray(result.log_likelihoods_birth)
    out[f"{tag}/ns/logZ"] = np.asarray([result.log_evidence, result.log_evidence_err])


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
    else:
        print("SPS_HOME not set: nebular configurations skipped")
    for tag, cfg in configs:
        t0 = time.perf_counter()
        model = _build(**cfg)
        theta = _draws(model, N_THETA, rng)
        _forward(model, theta, tag, out)
        _postprocess(model, theta, tag, out)
        if args.ns and tag == "plain":
            _nested(model, tag, out)
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
    if args.save:
        np.savez(args.save, **out)
        print(f"saved {len(out)} arrays -> {args.save}")
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
        sys.exit(1 if (bad or missing) else 0)


if __name__ == "__main__":
    main()
