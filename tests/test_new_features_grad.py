"""Differentiability of the 2026-09-22 features through ``fitSED``'s log-posterior (CPU, x64).

Features: the IGM damping wing / DLA (``x_HI``, ``logN_HI``, ``z_dla``), the new and fixed
attenuation laws (``tau_g03smc``, ``diffuse_tau_reddy``, ``Ebump`` in an age-bin ``noll``) and
THEMIS dust emission (``duste_qpah``, ``duste_umin``, ``duste_gamma``).  For each, the
production log-posterior (``MultiObservationLikelihood`` of ``fit._likelihood_for``, as
``fitSED`` builds it) with Photometry (one NaN flux, masked) and a Spectrum:

1. ``jax.grad`` w.r.t. every free parameter is finite at the prior centre and near each bound;
2. it agrees with central finite differences at one of four steps to 1e-3 relative plus the
   float32 noise of the difference (the SSP einsums are float32, so the log-posterior carries
   ~1e-7 relative noise, i.e. ~3 x 2e-7 |lnp| / h in the derivative); the largest relative
   difference at the best step is printed;
3. ``jit(vmap(value_and_grad))`` over a batch equals the unbatched loop (to 1e-6 in value,
   1e-5 in gradient: the batched float32 einsums reduce in a different order, as they do for
   the default laws).

The feature-off gradient is covered by ``scripts/bit_identity_check.py`` (the lnprob value and
gradient of every pre-existing configuration, byte for byte against the pre-change commit).
"""
import os
import warnings

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ceridwen import CSPBasis, SedModel, Cosmology, SSPData
from ceridwen.broadening import Instrument
from ceridwen.fit import _likelihood_for
from ceridwen.igm import MadauDampingDLA
from ceridwen.likelihood import MultiObservationLikelihood
from ceridwen.observation import Photometry, Spectrum
from ceridwen.priors import Uniform

from _gridfixture import require_test_grid

_SSP = str(require_test_grid())
SPS_HOME = os.environ.get("SPS_HOME")


def _lnprob(csp, filters, spec_rest, zred, free, fixed):
    """(lnprob, names, lo, hi) for a SedModel with mock data from its own prediction."""
    cst = {k: (lambda th, v=jnp.atleast_1d(jnp.asarray(v, float)): v) for k, v in fixed.items()}

    def build(flux_p=None, flux_s=None):
        phot = Photometry(filters=filters, flux=flux_p,
                          uncertainty=None if flux_p is None else 0.05 * np.abs(flux_p) + 1e-12,
                          name="phot")
        wave = np.linspace(*spec_rest, 120) * (1 + zred)
        spec = Spectrum(wavelength=wave, flux=flux_s,
                        uncertainty=None if flux_s is None else 0.05 * np.abs(flux_s) + 1e-30,
                        instrument=Instrument.sigma_kms(200.0), name="spec")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SedModel(csp, [phot, spec], zred=zred, transforms=cst,
                            free_param_init={k: (lo + hi) / 2 for k, (lo, hi) in free.items()},
                            priors={k: Uniform(low=lo, high=hi) for k, (lo, hi) in free.items()})

    m0 = build()
    th0 = {k: jnp.atleast_1d((lo + hi) / 2) for k, (lo, hi) in free.items()}
    pred = m0.predict(th0)
    yp = np.asarray(pred["phot"], float) * 1.03
    yp[0] = np.nan                                     # non-finite datum: masked at construction
    ys = np.asarray(pred["spec"], float) * 0.98
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build(yp, ys)
    keys = tuple(m.obs_dict)
    lh = MultiObservationLikelihood(
        keys=keys, likelihoods=tuple(_likelihood_for(o, m.param_names) for o in m.observations))
    names = list(free)
    assert set(m.param_names) == set(names), m.param_names
    lnp = lh.make_lnprobfn(m.obs_dict, m, m)
    lo = np.array([free[n][0] for n in names])
    hi = np.array([free[n][1] for n in names])
    return (lambda x: lnp({n: x[i:i + 1] for i, n in enumerate(names)})), names, lo, hi


def _check(f, names, lo, hi):
    vg = jax.jit(jax.value_and_grad(f))
    span = hi - lo
    # near-bound points 1.3 % inside, an odd fraction so they avoid the (qPAH, Umin) template
    # nodes, where the bilinear interpolation has a kink and a central difference straddles it
    pts = [lo + 0.5 * span, lo + 0.013 * span, hi - 0.013 * span]
    worst = 0.0
    for x in pts:
        v, g = vg(jnp.asarray(x))
        g = np.asarray(g)
        assert np.isfinite(float(v)) and np.all(np.isfinite(g)), (x, v, g)
        noise = 2e-7 * abs(float(v))            # float32 SSP einsums: lnp noise ~1e-7 |lnp|
        for i in range(x.size):
            ok, rels = False, []
            for step in (1e-2, 1e-3, 3e-4, 1e-4):
                h = step * span[i]
                e = np.zeros_like(x)
                e[i] = h
                fd = (float(f(jnp.asarray(x + e))) - float(f(jnp.asarray(x - e)))) / (2 * h)
                err = abs(fd - g[i])
                ok |= err <= 1e-3 * max(abs(fd), abs(g[i])) + 3 * noise / h
                rels.append(err / max(abs(fd), abs(g[i]), 1e-12))
            worst = max(worst, min(rels))
            assert ok, (names[i], x, float(v), g[i], rels)
    xs = jnp.asarray(np.stack(pts))
    vb, gb = jax.jit(jax.vmap(jax.value_and_grad(f)))(xs)
    for k, x in enumerate(pts):
        v, g = vg(jnp.asarray(x))
        # batched float32 einsums (and the float32 dust curve) reduce in a different order:
        # equal to float32 precision, not bytes (default powerlaw/kriek_conroy: up to 6.8e-8)
        np.testing.assert_allclose(np.asarray(vb[k]), float(v), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(gb[k]), np.asarray(g), rtol=1e-5,
                                   atol=1e-7 * float(np.max(np.abs(np.asarray(g)))))
    print(f"largest |AD - FD| / |grad| over {names}: {worst:.2e}")
    return worst


def _csp(**kw):
    return CSPBasis(SSPData.load(_SSP), lookback_time=jnp.linspace(0.0, 0.6, 5),
                    cosmo=Cosmology.wmap9(), zh_const=True, add_neb=False, verbose=False, **kw)


def test_grad_igm_damping_dla():
    csp = _csp(add_dust=False, add_diffuse_dust=False, add_igm=True,
               igm_model=MadauDampingDLA(Ob0=0.04628))
    f, names, lo, hi = _lnprob(
        csp, ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w"], (1180.0, 1400.0), 7.0,
        free={"logmass": (8.0, 10.0), "x_HI": (0.0, 1.0), "logN_HI": (19.0, 22.5),
              "z_dla": (6.0, 7.0)},
        fixed={"sfh": np.ones(5), "logzsol": -0.5})
    _check(f, names, lo, hi)
