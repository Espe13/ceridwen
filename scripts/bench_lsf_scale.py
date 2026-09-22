#!/usr/bin/env python3
"""Timing of the instrumental LSF scale (``Instrument(..., scale="lsf_scale")``).

Times the jitted, VALUE-ONLY log posterior that the nested sampler calls (fitSED's path:
``fit._likelihood_for`` + ``MultiObservationLikelihood`` + ``make_lnprobfn``), vmapped at
the nested-sampling widths read from the code: ``num_live`` (the default of
``BlackJAXNestedSamplerAdapter``) and ``num_delete = max(1, num_live // 5)``
(``ceridwen/sampler/nested.py``).  Arms:

    off  the default Instrument (scale = 1: the unscaled code path)
    on   the scale sampled, Uniform(0.8, 1.3)

Arms run in one process are interleaved; warm-up and compile are excluded (compile time is
printed separately); every call is ``block_until_ready``.  For the pre-change commit, run
``--arms off`` from that checkout (``git archive <base> | tar -x``), same GPU.

    python scripts/bench_lsf_scale.py --arms off on --reps 200
"""
from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

import jax                                                     # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

from _gridfixture import find_test_grid                        # noqa: E402
from ceridwen import SSPData, CSPBasis, SedModel, Cosmology    # noqa: E402
from ceridwen.broadening import Kinematics, Instrument         # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines   # noqa: E402
from ceridwen.priors import Uniform                            # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh     # noqa: E402
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter  # noqa: E402

ZRED = 0.8
N_TIME = 7
SPEC_WAVE = np.exp(np.arange(np.log(4400.0), np.log(7000.0), 45.0 / 2.99792458e5)) * (1 + ZRED)
LINES = {"Hbeta": 4862.71, "OIII5007": 5008.24, "Halpha": 6564.61}
BOUNDS = {"logzsol": (-1.0, 0.2), "gas_logz": (-1.0, 0.2), "gas_logu": (-3.5, -1.5),
          "diffuse_tau_kc": (0.0, 2.0), "diffuse_dust_index": (-1.0, 0.4),
          "logsfr_ratios": (-3.0, 3.0), "logmass": (9.0, 11.0), "lsf_scale": (0.8, 1.3)}


def build(csp, arm):
    ins = (Instrument.R_fwhm(2700.0) if arm == "off"
           else Instrument.R_fwhm(2700.0, scale="lsf_scale"))
    n = SPEC_WAVE.size
    obs = [Spectrum(wavelength=SPEC_WAVE, flux=np.ones(n), uncertainty=np.full(n, 0.1),
                    instrument=ins, name="spec"),
           Photometry(filters=["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"], flux=[1.0] * 4,
                      uncertainty=[0.1] * 4, name="phot"),
           Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                 wavelength=np.array(list(LINES.values())), flux=np.ones(len(LINES)),
                 uncertainty=np.full(len(LINES), 0.1), name="lines")]
    t = np.array(csp.sfh_times)

    def _sfh(th, _t=t):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    if arm == "on":
        init["lsf_scale"] = jnp.array([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SedModel(csp, obs, priors={"lsf_scale": Uniform(low=0.8, high=1.3)} if arm == "on" else {},
                     transforms={"sfh": _sfh}, free_param_init=init, zred=ZRED,
                     kinematics=Kinematics(sigma_gal=150.0, sigma_gas=80.0))
    m.priors = {k: Uniform(low=lo, high=hi) for k, (lo, hi) in BOUNDS.items()
                if k in m.param_names}
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood.likelihood import MultiObservationLikelihood, make_lnprobfn
    od = m.obs_dict
    keys = tuple(od)
    lh = tuple(_likelihood_for(od[k], m.param_names) for k in keys)
    return m, make_lnprobfn(od, m, m, MultiObservationLikelihood(keys=keys, likelihoods=lh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    import ceridwen
    print("package:", ceridwen.__file__, "| device:", jax.devices()[0])
    num_live = inspect.signature(BlackJAXNestedSamplerAdapter.__init__).parameters["num_live"].default
    widths = [max(1, num_live // 5), num_live]
    print(f"widths (num_delete, num_live): {widths}")
    ssp = SSPData.load(str(find_test_grid()))
    cosmo = Cosmology.planck18()
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=True, verbose=False, cosmo=cosmo,
                   sps_home=os.environ["SPS_HOME"])
    rng = np.random.default_rng(0)
    for W in widths:
        fns, batches = {}, {}
        for arm in args.arms:
            m, lnp = build(csp, arm)
            f = jax.jit(jax.vmap(lnp))
            b = {}
            for k, v in m.theta_init.items():
                v = np.asarray(v, float)
                b[k] = jnp.asarray(v[None] + 0.01 * rng.standard_normal((W,) + v.shape))
            if "lsf_scale" in b:
                b["lsf_scale"] = jnp.asarray(rng.uniform(0.8, 1.3, (W, 1)))
            t0 = time.perf_counter()
            f(b).block_until_ready()
            print(f"W={W:<4d} {arm:<3s}: compile + first call {time.perf_counter() - t0:.1f} s")
            for _ in range(3):
                f(b).block_until_ready()
            fns[arm], batches[arm] = f, b
        ts = {a: [] for a in fns}
        for _ in range(args.reps):
            for arm in fns:
                t0 = time.perf_counter()
                fns[arm](batches[arm]).block_until_ready()
                ts[arm].append(time.perf_counter() - t0)
        for arm, v in ts.items():
            q = np.percentile(v, [25, 50, 75])
            print(f"W={W:<4d} {arm:<3s}: median {1e3 * q[1]:8.2f} ms   IQR [{1e3 * q[0]:.2f}, "
                  f"{1e3 * q[2]:.2f}] ms   (n={len(v)})")
        if {"on", "off"} <= set(ts):
            print(f"W={W:<4d} on/off median ratio {np.median(ts['on']) / np.median(ts['off']):.3f}")


if __name__ == "__main__":
    main()
