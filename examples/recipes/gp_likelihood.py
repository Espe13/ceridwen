"""Fit a spectrum whose residuals are correlated between neighbouring pixels, with the
Gaussian-process (GP) likelihood and its two hyperparameters sampled.

    JAX_PLATFORMS=cpu python examples/recipes/gp_likelihood.py [--nuts] [--out DIR]

What it does
------------
1. Builds a small model on the test SSP grid (``tests/_gridfixture.py``; set
   ``$CERIDWEN_TEST_SSP`` or pass ``--grid``): a constant SFH, fixed metallicity, no dust
   or nebular emission, redshift 0.1, a 600-pixel spectrum from 5000 to 6200 A (observed).
2. Draws mock data: the model at log M = 10.5, plus noise sigma * L z with
   L L^T = I + a^2 exp(-dlambda^2 / 2 l^2), a = 1 and l = 30 A, sigma = model / 30.
   Half the variance per pixel is correlated over ~15 pixels.
3. Fits log M together with the GP hyperparameters, sampled in natural log:

       log_gp_amp_spec    = ln a   (a in units of the per-pixel sigma_eff)
       log_gp_length_spec = ln l   (l in observed-frame Angstrom)

   Both names switch the GP on for the (only) spectrum; with several spectra use
   ``log_gp_amp_spec_<obs.name>``.  To FIX the hyperparameters instead, give
   ``Spectrum(noise=GaussianProcess(a, l))`` and do not sample the names.
4. Prints the posterior of a, l and log M, and the diagonal-likelihood fit for comparison.
   One run (nested sampling, defaults below): GP fit log M = 10.498 [10.496, 10.500],
   a = 0.73 [0.64, 0.84], l = 23.6 [20.1, 27.5] A; diagonal fit log M = 10.498
   [10.497, 10.499], half as wide.  This noise draw (seed 42) sits low in (a, l): its
   maximum-likelihood values from the independent numpy GaussianProcess are 0.72 and 23.6 A,
   while over 20 draws the estimates average a = 1.01 +/- 0.17, l = 30.4 +/- 5.2 A.

Nested sampling by default (``--nuts`` for NUTS).  Both fits together took 11 min on a
laptop CPU (Apple M3 Pro, shared with other jobs).
See ``docs/gp_likelihood.md``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ceridwen import CSPBasis, SSPData, SedModel, Cosmology, fitSED   # noqa: E402
from ceridwen.observation import Spectrum                              # noqa: E402
from ceridwen.priors import Uniform                                     # noqa: E402
from ceridwen.likelihood.gp_likelihood import gp_sqdist, GP_JITTER      # noqa: E402

A_TRUE, L_TRUE, LOGM_TRUE, ZRED = 1.0, 30.0, 10.5, 0.1


def build(grid_path, flux=None, sigma=None, gp=True):
    ssp = SSPData.load(str(grid_path))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 10.0, 4), zh_const=True,
                   sfh_interp="step", add_dust=False, add_diffuse_dust=False, add_neb=False,
                   add_igm=False, verbose=False, cosmo=Cosmology.planck18())
    w = np.linspace(5000.0, 6200.0, 600)
    spec = Spectrum(wavelength=w, flux=np.ones(w.size) if flux is None else flux,
                    uncertainty=np.ones(w.size) if sigma is None else sigma, name="spec")
    priors = {"logmass": Uniform(low=10.0, high=11.0)}
    init = {"logmass": jnp.array([LOGM_TRUE])}
    if gp:
        priors.update({"log_gp_amp_spec": Uniform(low=-3.0, high=1.5),        # a in [0.05, 4.5]
                       "log_gp_length_spec": Uniform(low=1.0, high=5.5)})     # l in [2.7, 245] A
        init.update({"log_gp_amp_spec": jnp.array([0.0]),
                     "log_gp_length_spec": jnp.array([3.0])})
    fixed = {"sfh": lambda th: jnp.ones(4), "logzsol": lambda th: jnp.array([-0.3])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, [spec], priors=priors, transforms=fixed, free_param_init=init,
                        zred=ZRED)


def mock_data(model, seed=42):
    w = np.asarray(model.observations[0].wavelength)
    mu = np.asarray(model.predict({k: jnp.asarray(v) for k, v in model.theta_init.items()})["spec"])
    sig = mu / 30.0
    C = (A_TRUE ** 2 * np.exp(-0.5 * np.asarray(gp_sqdist(w)) / L_TRUE ** 2)
         + (1.0 + GP_JITTER) * np.eye(w.size))
    y = mu + sig * (np.linalg.cholesky(C) @ np.random.default_rng(seed).normal(size=w.size))
    return y, sig


def summarise(result, names):
    s = result.samples
    lw = np.asarray(result.log_weights, dtype=float)
    wts = np.exp(lw - lw.max()); wts /= wts.sum()
    for n in names:
        x = np.asarray(s[n], dtype=float).reshape(-1)
        if n.startswith("log_gp_"):
            x = np.exp(x)
            n = {"log_gp_amp_spec": "a", "log_gp_length_spec": "l [A]"}[n]
        order = np.argsort(x)
        cdf = np.cumsum(wts[order])
        q = [x[order][np.searchsorted(cdf, p)] for p in (0.16, 0.5, 0.84)]
        print(f"    {n:8s} = {q[1]:.3f}  [{q[0]:.3f}, {q[2]:.3f}]  (median [16, 84 %])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=None, help="SSP grid (default: the test grid)")
    ap.add_argument("--nuts", action="store_true")
    ap.add_argument("--out", default="gp_likelihood_out")
    args = ap.parse_args()
    grid = args.grid
    if grid is None:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
        from _gridfixture import find_test_grid
        grid = find_test_grid()
        if grid is None:
            sys.exit("no SSP grid: pass --grid or set $CERIDWEN_TEST_SSP")
    y, sig = mock_data(build(grid))
    sampler = "nuts" if args.nuts else "nested"
    skw = ({"num_samples": 500, "num_chains": 2} if args.nuts
           else {"num_live": 150, "num_inner_steps": 15})
    for gp in (True, False):
        model = build(grid, y, sig, gp=gp)
        res = fitSED(model, output_dir=pathlib.Path(args.out) / ("gp" if gp else "diag"),
                     sampler=sampler, sampler_kwargs=dict(skw), verbose=False, mfrac=False)
        print(f"\n{'GP' if gp else 'diagonal'} likelihood (truth: a = {A_TRUE}, "
              f"l = {L_TRUE} A, logmass = {LOGM_TRUE}):")
        summarise(res, model.param_names)


if __name__ == "__main__":
    main()
