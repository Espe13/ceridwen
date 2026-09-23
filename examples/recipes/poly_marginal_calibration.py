"""Usage: the three routes for a spectrum's calibration polynomial on one mock.

    CERIDWEN_TEST_SSP=/path/to/ssp_grid.h5 JAX_PLATFORMS=cpu \
        python examples/recipes/poly_marginal_calibration.py

A mock spectrum (S/N 30) of a known galaxy is multiplied by a known order-3 Chebyshev response
and fitted together with 4-band photometry.  At the true physical parameters the script prints
the log-likelihood of

* the profiled polynomial   Spectrum(polynomial_order=3)
* the marginalised one      Spectrum(polynomial_order=3, polynomial_mode="marginalize",
                                     polynomial_prior_sigma=0.1)
* no polynomial

and the conditional posterior of the coefficients (mean +- sd) against the injected ones.
The marginal lies below the profile by the Occam factor of the four coefficients.
"""
import os
import sys
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology
from ceridwen.broadening import Instrument
from ceridwen.fit import _likelihood_for
from ceridwen.likelihood.likelihood import observation_data
from ceridwen.likelihood.poly_calibration import chebyshev_design_matrix
from ceridwen.observation import Photometry, Spectrum

grid = os.environ.get("CERIDWEN_TEST_SSP") or (sys.argv[1] if len(sys.argv) > 1 else None)
if grid is None:
    raise SystemExit("set CERIDWEN_TEST_SSP to an SSP grid (.h5), or pass its path")

csp = CSPBasis(SSPData.load(grid), lookback_time=jnp.linspace(0.0, 9.0, 4), zh_const=True,
               add_dust=True, add_diffuse_dust=True, add_neb=False, add_igm=False,
               verbose=False, cosmo=Cosmology.planck18())
wave = np.linspace(4500.0, 8500.0, 400)
c_true = np.array([0.03, 0.06, -0.04, 0.025])
filters = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]


def model(**spec_kw):
    sp = Spectrum(wavelength=wave, flux=np.ones(wave.size), uncertainty=np.ones(wave.size),
                  instrument=Instrument.sigma_kms(150.0), name="s", **spec_kw)
    ph = Photometry(filters=filters, flux=[1.0] * 4, uncertainty=[0.1] * 4, name="p")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, [sp, ph], priors={}, zred=0.3,
                        free_param_init={"logmass": jnp.array([10.0]),
                                         "diffuse_tau_kc": jnp.array([0.6])})


m0 = model()
theta = {k: jnp.asarray(v) for k, v in m0.theta_init.items()}
pred = m0.predict(theta)
A = chebyshev_design_matrix(wave, np.ones(wave.size, bool), 3)
rng = np.random.default_rng(1)
mu_cal = np.asarray(pred["s"]) * (1.0 + A @ c_true)
data = {"s": (mu_cal + mu_cal / 30.0 * rng.standard_normal(wave.size), mu_cal / 30.0),
        "p": (np.asarray(pred["p"]), 0.03 * np.asarray(pred["p"]))}

for label, kw in (("profiled", dict(polynomial_order=3)),
                  ("marginalised", dict(polynomial_order=3, polynomial_mode="marginalize",
                                        polynomial_prior_sigma=0.1)),
                  ("no polynomial", {})):
    m = model(**kw)
    for o in m.observations:
        o.flux, o.uncertainty = (jnp.asarray(a) for a in data[o.name])
    lh = _likelihood_for(m.obs_dict["s"], m.param_names, model=m)
    y, sig, mask, _c, _u = observation_data(m.obs_dict["s"])
    mu = m.predict(theta)["s"]
    print(f"{label:>14}: ln L(spectrum) = {float(lh(y, mu, sig, mask, theta)[0]):.3f}"
          f"   [{type(lh).__name__}]")
    if hasattr(lh, "conditional"):
        mean, cov, _resp = lh.conditional(y, mu, sig, mask, theta)
        for i, (ci, si) in enumerate(zip(np.asarray(mean), np.sqrt(np.diag(np.asarray(cov))))):
            print(f"{'':>16}c_{i} = {ci:+.4f} +- {si:.4f}   (injected {c_true[i]:+.4f})")
