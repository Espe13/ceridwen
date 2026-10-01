"""E2 (opt-in, ``SedModel._hoist_flux_factor``): the flux factor of a FIXED redshift is
computed once at SedModel construction (by the
CSP's own ``_flux_factor`` under jit) and passed to ``csp.predict`` as a private argument.

It must be byte-identical to the per-call quadrature (the reference: the same model with
``_flux_factor_fixed = None``), add no theta key, and stay per call whenever theta brings its
own ``zred`` / ``lumdist_mpc`` or the redshift is sampled.
"""
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import require_test_grid  # noqa: E402

jax.config.update("jax_enable_x64", True)

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology   # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics        # noqa: E402

GRID = require_test_grid()
HAVE_SPS = bool(os.environ.get("SPS_HOME"))


@pytest.fixture(autouse=True)
def _hoist_on(monkeypatch):
    monkeypatch.setattr(SedModel, "_hoist_flux_factor", True)     # E2 is opt-in


def _model(neb, zred=0.8, lumdist=None, free_z=False):
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(GRID)),
                   lookback_time=jnp.linspace(0.0, float(cosmo.age(zred)), 5), zh_const=True,
                   add_neb=neb, add_igm=True, verbose=False, cosmo=cosmo,
                   **({"sps_home": os.environ["SPS_HOME"]} if neb else {}))
    lam = np.linspace(3800.0, 8800.0, 300) * (1.0 + zred)
    obs = [Photometry(filters=["sdss_g0", "sdss_r0", "sdss_i0"], flux=[1e-9] * 3,
                      uncertainty=[1e-10] * 3, name="phot"),
           Spectrum(wavelength=lam, flux=np.full(lam.size, 1e-29),
                    uncertainty=np.full(lam.size, 1e-30),
                    instrument=Instrument.sigma_kms(150.0), name="spec")]
    if neb:
        obs.append(Lines(line_ind=np.arange(2), line_names=["Hbeta", "Halpha"],
                         wavelength=np.array([4862.71, 6564.61]), flux=np.full(2, 1e-17),
                         uncertainty=np.full(2, 1e-18), name="lines"))
    kw = {}
    if free_z:
        from ceridwen.priors import Uniform
        kw = dict(priors={"zred": Uniform(low=0.7, high=0.9)},
                  free_param_init={"zred": jnp.array([zred])})
        obs = obs[1:2]
    else:
        kw = dict(priors={}, zred=zred, lumdist_mpc=lumdist)
    return SedModel(csp, obs, kinematics=Kinematics(sigma_gal=250.0), **kw)


def _theta(m, k):
    rng = np.random.default_rng(k)
    return {n: jnp.asarray(np.asarray(m.theta_init[n]) + 0.05 * rng.standard_normal(
        np.shape(m.theta_init[n]))) for n in m.param_names}


def _bytes(tree):
    return [np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(tree)]


CASES = [dict(neb=False), dict(neb=False, lumdist=50.0),
         pytest.param(dict(neb=True), marks=pytest.mark.skipif(not HAVE_SPS, reason="$SPS_HOME"))]


@pytest.mark.parametrize("kw", CASES)
def test_hoisted_equals_per_call(kw):
    m = _model(**kw)
    hoisted = m._flux_factor_fixed
    th0 = {"zred": m._zred_fixed}
    if m._lumdist_fixed is not None:
        th0["lumdist_mpc"] = m._lumdist_fixed
    assert np.asarray(hoisted).tobytes() == np.asarray(m.csp._flux_factor(th0)).tobytes()
    lnl = lambda t: sum(jnp.sum(jnp.log(jnp.abs(v) + 1e-300)) for v in m.predict(t).values())
    res = {}
    for on in (False, True):
        m._flux_factor_fixed = hoisted if on else None
        ths = [_theta(m, k) for k in range(3)]
        tb = {n: jnp.stack([t[n] for t in ths]) for n in ths[0]}
        res[on] = (_bytes([jax.jit(m.predict)(t) for t in ths])
                   + _bytes(jax.jit(jax.vmap(m.predict))(tb))
                   + _bytes([jax.jit(jax.value_and_grad(lnl))(t) for t in ths]))
    m._flux_factor_fixed = hoisted
    assert res[False] == res[True]


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(SedModel, "_hoist_flux_factor", False)
    assert _model(neb=False)._flux_factor_fixed is None


def test_no_theta_key_and_per_call_fallbacks():
    m = _model(neb=False)
    assert m._flux_factor_fixed is not None
    assert not any("flux" in k for k in m.param_names)
    assert not any("flux_factor" in k for k in m.csp._known_theta_keys)
    t = m.apply_transforms(_theta(m, 0))
    assert m._fixed_flux_factor(t) is not None
    assert m._fixed_flux_factor({**t, "zred": jnp.array([0.5])}) is None         # own zred
    assert m._fixed_flux_factor({**t, "lumdist_mpc": jnp.array([3.0])}) is None   # own D_L
    # a theta that brings its own zred is predicted at that zred, not the construction one
    a = jax.jit(m.predict)({**_theta(m, 0), "zred": jnp.array([0.5])})
    m2 = _model(neb=False)
    m2._flux_factor_fixed = None
    b = jax.jit(m2.predict)({**_theta(m2, 0), "zred": jnp.array([0.5])})
    assert _bytes(a) == _bytes(b)


def test_sampled_zred_stays_per_call():
    m = _model(neb=False, free_z=True)
    assert m._flux_factor_fixed is None and m._zred_fixed is None
    g = jax.grad(lambda t: jnp.sum(m.predict(t)["spec"]))(_theta(m, 0))
    assert np.isfinite(float(g["zred"][0])) and float(g["zred"][0]) != 0.0


def test_zred_zero_has_no_flux_factor():
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(GRID)), lookback_time=jnp.linspace(0.0, 13.0, 4),
                   add_neb=False, verbose=False, cosmo=cosmo)
    with pytest.warns(UserWarning):
        m = SedModel(csp, [Photometry(filters=["sdss_r0"], flux=[1.0], uncertainty=[0.1],
                                      name="p")], priors={})
    assert m._flux_factor_fixed is None
