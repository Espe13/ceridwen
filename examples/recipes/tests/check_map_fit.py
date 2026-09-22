#!/usr/bin/env python3
"""
check_map_fit.py -- acceptance check of examples/recipes/map_fit.py (CPU, a few minutes).

Mock: the configuration of ``examples/make_mock_data.py`` (its TRUTH, FILTERS, SPEC_WAVE,
SPEC_RES, SNR_*, ZRED, N_TIME, T_OLDEST, and its CSPBasis / SedModel setup, l.60-96),
regenerated in memory with the same ``np.random.default_rng(42)`` draw order (l.97-106) so no
file is written (``make_mock_data.main()`` writes ``examples/mock_galaxy.npz``).  The fit model
adds the priors of ``examples/demo_1_mock_test.py:84-91``.

Checks
  1. ln p(MAP) >= max ln p over 1000 prior draws (same jitted ln posterior).
  2. |MAP - truth| <= 1 sigma for every parameter element, sigma = Laplace width
     (inverse Hessian of -ln p at the MAP, constrained parameters).  Note: for 9 independent
     Gaussian elements the chance that all 9 lie within 1 sigma is 0.683^9 ~ 3 %; the pulls are
     printed so a FAIL can be judged.  A 3-sigma line is printed for information.
  Wall time of map_fit (compile included) is reported.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # examples/recipes
sys.path.insert(0, str(HERE.parent.parent))   # examples

import make_mock_data as mm                   # noqa: E402  (module constants only)
from map_fit import build_lnprob, laplace_sigma, map_fit  # noqa: E402

from ceridwen import CSPBasis, Instrument, SSPData, SedModel  # noqa: E402
from ceridwen.cosmology import Cosmology  # noqa: E402
from ceridwen.model import logsfr_ratios_to_sfh  # noqa: E402
from ceridwen.observation import Photometry, Spectrum  # noqa: E402
from ceridwen.priors import ClippedNormal, StudentT, Uniform  # noqa: E402

N_STARTS = 16
N_PRIOR = 1000


def main() -> int:
    ssp = SSPData.load(str(mm.SSP_FILE))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, mm.T_OLDEST, mm.N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=False,
                   add_diffuse_dust=True, add_neb=False, verbose=False,
                   cosmo=Cosmology.planck18())
    t_yr = np.array(csp.sfh_times)
    inst = Instrument.sigma_kms(mm.SPEC_RES)

    def build(obs, priors=None):
        return SedModel(
            csp, observations=obs, priors=priors,
            transforms={"sfh": lambda th, _t=t_yr:
                        logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
            free_param_init={"logsfr_ratios": jnp.zeros(mm.N_TIME - 1),
                             "logmass": jnp.array([10.0])},
            zred=mm.ZRED)

    gen = build([Photometry(filters=mm.FILTERS, name="phot"),
                 Spectrum(wavelength=mm.SPEC_WAVE, instrument=inst, name="spec")])
    noiseless = gen.predict(mm.TRUTH)
    rng = np.random.default_rng(42)
    mag = np.asarray(noiseless["phot"])
    mag_unc = mag / mm.SNR_PHOT
    mag_obs = mag + mag_unc * rng.standard_normal(mag.shape)
    flux = np.asarray(noiseless["spec"])
    flux_unc = np.abs(flux) / mm.SNR_SPEC
    flux_obs = flux + flux_unc * rng.standard_normal(flux.shape)

    priors = {
        "logzsol": Uniform(low=-2.0, high=0.2),
        "logmass": Uniform(low=9.0, high=12.0),
        "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
        "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
        "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=1.0),
    }
    model = build([Photometry(filters=mm.FILTERS, flux=mag_obs, uncertainty=mag_unc,
                              name="phot"),
                   Spectrum(wavelength=mm.SPEC_WAVE, flux=flux_obs, uncertainty=flux_unc,
                            instrument=inst, name="spec")], priors)
    print("free parameters:", {k: tuple(np.shape(v)) for k, v in model.theta_init.items()})

    lnprob = build_lnprob(model)
    res = map_fit(model, n_starts=N_STARTS, rng_key=jax.random.PRNGKey(1), lnprob=lnprob)
    print(f"map_fit: {N_STARTS} starts, wall time {res.wall_time:.1f} s (compile included); "
          f"steps per start {res.n_steps.tolist()}")
    print("  final ln p per start:", np.array2string(res.lnp_starts, precision=3))

    # 1. versus the best of 1000 prior draws
    key = jax.random.PRNGKey(123)
    draws = {}
    for name, init in model.theta_init.items():
        key, sub = jax.random.split(key)
        draws[name] = priors[name].sample(sub, shape=(N_PRIOR, *np.shape(init)))
    t0 = time.perf_counter()
    lnp_draws = np.asarray(jax.jit(jax.vmap(lnprob))(draws))
    t_draws = time.perf_counter() - t0
    best_draw = np.nanmax(np.where(np.isfinite(lnp_draws), lnp_draws, -np.inf))
    ok1 = res.lnp >= best_draw
    lnp_truth = float(lnprob({k: jnp.asarray(v) for k, v in mm.TRUTH.items()}))
    print(f"{'PASS' if ok1 else 'FAIL'} lnp_MAP >= best of {N_PRIOR} prior draws: "
          f"lnp_MAP = {res.lnp:.4f}, best draw = {best_draw:.4f} "
          f"(ln p at truth = {lnp_truth:.4f}; draws took {t_draws:.1f} s)")

    # 2. MAP within 1 Laplace sigma of the truth
    sig, _ = laplace_sigma(lnprob, res.theta, model.theta_init)
    pulls = []
    print(f"  {'parameter':>22} {'truth':>8} {'MAP':>9} {'sigma':>8} {'pull':>6}")
    for k in model.theta_init:
        tr, mp, s = (np.ravel(np.asarray(a)) for a in (mm.TRUTH[k], res.theta[k], sig[k]))
        for i in range(tr.size):
            p = (mp[i] - tr[i]) / s[i]
            pulls.append(p)
            lab = k if tr.size == 1 else f"{k}[{i}]"
            print(f"  {lab:>22} {tr[i]:+8.3f} {mp[i]:+9.4f} {s[i]:8.5f} {p:+7.4f}")
    pulls = np.array(pulls)
    ok2 = bool(np.all(np.isfinite(pulls)) and np.all(np.abs(pulls) <= 1.0))
    print(f"{'PASS' if ok2 else 'FAIL'} |MAP - truth| <= 1 Laplace sigma for all "
          f"{pulls.size} elements: max |pull| = {np.max(np.abs(pulls)):.4f}, "
          f"{int(np.sum(np.abs(pulls) <= 1.0))}/{pulls.size} within 1 sigma")
    print(f"(info) within 3 sigma: {int(np.sum(np.abs(pulls) <= 3.0))}/{pulls.size}")
    print(f"wall time map_fit: {res.wall_time:.1f} s")
    ok = ok1 and ok2
    print("PASS" if ok else "FAIL", "check_map_fit")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
