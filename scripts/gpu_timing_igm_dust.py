#!/usr/bin/env python3
"""GPU timing of the nested-sampling log-posterior for the 2026-09-22 features.

Times the jitted, value-only log-posterior that nested sampling calls (built as ``fitSED``
builds it: ``MultiObservationLikelihood`` of ``fit._likelihood_for``), ``vmap``-ed at the
nested-sampling batch widths, which are read from ``BlackJAXNestedSamplerAdapter``'s defaults
(``num_live`` and ``num_delete = max(1, num_live // 5)``, ``sampler/nested.py:158-160``).

Cases, all on one model (Photometry + Spectrum at z = 6.5):

    off      Madau (1995) IGM, powerlaw bin + kriek_conroy diffuse dust, DL07 dust emission
    igm      as off with MadauDampingDLA, x_HI and logN_HI sampled
    dust     as off with noll / gordon03_smcbar bins and reddy15 diffuse, Ebump sampled
    themis   as off with THEMIS dust emission

A case the package does not have (the pre-change commit has only ``off``) is skipped.  Calls are
interleaved across cases, warm-up excluded, ``block_until_ready`` on every call; median and IQR
per case and width, compile time separately, all written as JSON.

    python scripts/gpu_timing_igm_dust.py --out timing_new.json            # new code
    (cd /path/to/base && python /path/to/this/script --out timing_base.json)   # pre-change

Pass: ``off`` (new) not slower than ``off`` (base); each feature case within
max(IQR of off, 1 %) of ``off``.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

import jax                                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                           # noqa: E402

from _gridfixture import find_test_grid                           # noqa: E402
from ceridwen import SSPData, CSPBasis, SedModel, Cosmology       # noqa: E402
from ceridwen.broadening import Instrument                        # noqa: E402
from ceridwen.fit import _likelihood_for                          # noqa: E402
from ceridwen.likelihood import MultiObservationLikelihood        # noqa: E402
from ceridwen.observation import Photometry, Spectrum             # noqa: E402
from ceridwen.priors import Uniform                               # noqa: E402
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter  # noqa: E402

ZRED = 6.5
FILTERS = ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w",
           "jwst_f356w", "jwst_f444w", "wise_w3"]
N_TIME = 7


def ns_widths():
    sig = inspect.signature(BlackJAXNestedSamplerAdapter.__init__).parameters
    num_live = int(sig["num_live"].default)
    return num_live, max(1, num_live // 5)          # sampler/nested.py:158-160


def case_kwargs(case):
    """CSPBasis kwargs and extra sampled parameters {name: (lo, hi)}; None = not available."""
    kw = dict(add_dust=True, add_diffuse_dust=True, add_dust_emission=True, add_igm=True)
    extra = {}
    try:
        if case == "igm":
            from ceridwen.igm import MadauDampingDLA
            kw["igm_model"] = MadauDampingDLA(Ob0=0.04897)
            extra = {"x_HI": (0.0, 1.0), "logN_HI": (19.0, 22.5)}
        elif case == "dust":
            from ceridwen.dust.attenuation_laws import ATTENUATION_LAWS
            if "gordon03_smcbar" not in ATTENUATION_LAWS:
                return None
            kw.update(init_dust_params={"bin_edges": [(-np.inf, -1.97)], "laws": ["noll"]},
                      diffuse_law="reddy15")
            extra = {"tau_noll": (0.0, 2.0), "Ebump": (0.0, 3.0), "diffuse_tau_reddy": (0.0, 1.5)}
        elif case == "themis":
            if "duste_model" not in inspect.signature(CSPBasis.__init__).parameters:
                return None
            kw["duste_model"] = "THEMIS"
    except ImportError:
        return None
    return kw, extra


def build(case, ssp, cosmo):
    got = case_kwargs(case)
    if got is None:
        return None
    kw, extra = got
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
                   cosmo=cosmo, zh_const=True, sfh_interp="step", add_neb=False,
                   sps_home=os.environ["SPS_HOME"], verbose=False, **kw)
    sampled = {"logmass": (8.0, 11.0), "logzsol": (-2.0, 0.2)}
    if case != "dust":
        sampled.update({"diffuse_tau_kc": (0.0, 1.5)})
    sampled.update(extra)
    fixed = {k: v for k, v in csp.theta_init.items() if k not in sampled}
    tr = {k: (lambda th, v=jnp.atleast_1d(v): v) for k, v in fixed.items()}
    wave = np.linspace(1150.0, 1500.0, 300) * (1 + ZRED)
    rng = np.random.default_rng(0)
    obs = [Photometry(filters=FILTERS, flux=1e-9 * (1 + 0.1 * rng.standard_normal(len(FILTERS))),
                      uncertainty=np.full(len(FILTERS), 1e-10), name="phot"),
           Spectrum(wavelength=wave, flux=1e-9 * np.ones(wave.size),
                    uncertainty=np.full(wave.size, 1e-10), instrument=Instrument.R_fwhm(1000.0),
                    name="spec")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SedModel(csp, obs, zred=ZRED, transforms=tr,
                     free_param_init={k: (a + b) / 2 for k, (a, b) in sampled.items()},
                     priors={k: Uniform(low=a, high=b) for k, (a, b) in sampled.items()})
    lh = MultiObservationLikelihood(keys=tuple(m.obs_dict), likelihoods=tuple(
        _likelihood_for(o, m.param_names) for o in m.observations))
    lnp = lh.make_lnprobfn(m.obs_dict, m, m)
    names = sorted(sampled)
    lo = np.array([sampled[n][0] for n in names])
    hi = np.array([sampled[n][1] for n in names])
    return lnp, names, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cases", nargs="+", default=["off", "igm", "dust", "themis"])
    ap.add_argument("--widths", nargs="+", type=int, default=None,
                    help="override the nested-sampling widths (CPU smoke test only)")
    args = ap.parse_args()
    import ceridwen
    print("package:", ceridwen.__file__, "device:", jax.devices())
    widths = args.widths or list(ns_widths())
    ssp = SSPData.load(str(find_test_grid()))
    cosmo = Cosmology.planck18()
    fns, compile_s = {}, {}
    rng = np.random.default_rng(1)
    for case in args.cases:
        b = build(case, ssp, cosmo)
        if b is None:
            print(f"  {case}: not available in this package, skipped")
            continue
        lnp, names, lo, hi = b
        for w in widths:
            x = {n: jnp.asarray(rng.uniform(lo[i], hi[i], (w, 1))) for i, n in enumerate(names)}
            f = jax.jit(jax.vmap(lnp))
            t0 = time.perf_counter()
            f(x).block_until_ready()
            compile_s[f"{case}/{w}"] = time.perf_counter() - t0
            fns[f"{case}/{w}"] = (f, x)
    times = {k: [] for k in fns}
    for it in range(args.warmup + args.calls):       # interleaved across cases
        for k, (f, x) in fns.items():
            t0 = time.perf_counter()
            f(x).block_until_ready()
            if it >= args.warmup:
                times[k].append(time.perf_counter() - t0)
    res = {}
    for k, t in times.items():
        t = np.asarray(t) * 1e3
        q1, med, q3 = np.percentile(t, [25, 50, 75])
        res[k] = {"median_ms": med, "iqr_ms": q3 - q1, "n": int(t.size),
                  "compile_s": compile_s[k]}
        print(f"{k:<14} median {med:9.3f} ms  IQR {q3 - q1:7.3f} ms  compile {compile_s[k]:6.1f} s")
    out = {"package": ceridwen.__file__, "devices": [str(d) for d in jax.devices()],
           "widths": widths, "results": res}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
