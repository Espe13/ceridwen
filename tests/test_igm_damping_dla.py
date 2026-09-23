"""``ceridwen.igm.MadauDampingDLA``: IGM damping wing and DLA with theta-level parameters.

The physics (vs Prospector @ a78d153 to 1e-10) is checked by
``examples/recipes/tests/check_igm_damping_dla.py`` and pinned by the ``igm_damping_dla``
regression category; this file checks the package wiring:

1. ``x_HI``, ``logN_HI``, ``z_dla`` in theta reproduce the constructor values bit for bit, and
   ``x_HI = 0`` / ``logN_HI = -inf`` in theta give the Madau-only CSP prediction bit for bit.
2. The keys are known to the CSP (no unknown-key warning) and can be sampled through
   ``SedModel`` like ``igm_factor``.
3. Gradients w.r.t. the three keys through ``csp.predict`` are finite and non-zero.
4. Construction guards: missing ``Ob0`` and a mismatched cosmology raise.
"""
import warnings

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ceridwen import CSPBasis, SedModel, Cosmology, SSPData
from ceridwen.igm import MadauDampingDLA, make_igm_model
from ceridwen.observation import Photometry
from ceridwen.priors import Uniform

from _gridfixture import require_test_grid

_SSP = str(require_test_grid())
COSMO = Cosmology.wmap9()
Z = 7.0
FILTERS = ["jwst_f090w", "jwst_f115w", "jwst_f150w"]


def _csp(igm):
    return CSPBasis(SSPData.load(_SSP), lookback_time=jnp.linspace(0.0, 0.7, 6), cosmo=COSMO,
                    zh_const=True, add_neb=False, add_dust=False, add_diffuse_dust=False,
                    add_igm=True, igm_model=igm, verbose=False)


@pytest.fixture(scope="module")
def setup():
    csp_m = _csp("madau1995")
    csp_d = _csp(MadauDampingDLA(Ob0=0.04628))
    phot = Photometry(filters=FILTERS, name="phot")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phot.setup_for_model(csp_m.wave, zred=Z)
    th = dict(csp_m.theta_init, zred=jnp.array([Z]), logmass=jnp.array([9.0]))
    return csp_m, csp_d, phot, th


def _pred(csp, th, phot):
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # also: no unknown-theta-key warning
        return np.asarray(csp.predict(th, [phot])["phot"])


@pytest.mark.parametrize("extra", [
    {},
    {"x_HI": 0.0},
    {"logN_HI": -np.inf},
    {"x_HI": 0.0, "logN_HI": -np.inf, "z_dla": 6.5},
])
def test_madau_limit_bytes(setup, extra):
    csp_m, csp_d, phot, th = setup
    ref = _pred(csp_m, th, phot)
    out = _pred(csp_d, dict(th, **{k: jnp.array([v]) for k, v in extra.items()}), phot)
    assert out.tobytes() == ref.tobytes()


def test_theta_overrides_constructor(setup):
    _, _, phot, th = setup
    fixed = _csp(MadauDampingDLA(Ob0=0.04628, x_HI=0.7, logN_HI=21.3, z_dla=6.9))
    ref = _pred(fixed, th, phot)
    via_theta = _pred(_csp(MadauDampingDLA(Ob0=0.04628)), dict(
        th, x_HI=jnp.array([0.7]), logN_HI=jnp.array([21.3]), z_dla=jnp.array([6.9])), phot)
    assert via_theta.tobytes() == ref.tobytes()
    madau = _pred(setup[0], th, phot)
    assert ref[0] < 0.9 * madau[0]              # F090W straddles Ly-alpha at z=7: absorbed


def test_gradients(setup):
    _, csp_d, phot, th = setup

    def f(x, n, zd):
        t = dict(th, x_HI=jnp.atleast_1d(x), logN_HI=jnp.atleast_1d(n), z_dla=jnp.atleast_1d(zd))
        return jnp.sum(jnp.log(csp_d.predict(t, [phot])["phot"]))

    g = np.asarray(jax.jit(jax.grad(f, argnums=(0, 1, 2)))(0.5, 21.0, 6.95))
    assert np.all(np.isfinite(g)) and np.all(g != 0.0), g


def test_sampled_through_sedmodel(setup):
    csp_m, csp_d, phot, th = setup
    y = _pred(csp_m, th, phot)

    def model(csp, extra_init=None, extra_priors=None):
        obs = Photometry(filters=FILTERS, flux=y, uncertainty=0.05 * y, name="phot")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SedModel(csp, [obs], zred=Z,
                            free_param_init={"logmass": 9.0, **(extra_init or {})},
                            priors={"logmass": Uniform(low=8.0, high=10.0),
                                    **(extra_priors or {})})

    m_ref = model(csp_m)
    m = model(csp_d, {"x_HI": 0.5, "logN_HI": 21.0},
              {"x_HI": Uniform(low=0.0, high=1.0), "logN_HI": Uniform(low=19.0, high=23.0)})
    assert {"x_HI", "logN_HI"} <= set(m.param_names)
    ref = np.asarray(m_ref.predict(dict(m_ref.theta_init))["phot"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p0 = m.predict({**m.theta_init, "x_HI": jnp.array([0.0]),
                        "logN_HI": jnp.array([-np.inf])})["phot"]
        p1 = m.predict({**m.theta_init, "x_HI": jnp.array([1.0])})["phot"]
    assert np.asarray(p0).tobytes() == ref.tobytes()
    assert float(p1[0]) < 0.9 * float(p0[0])


def test_guards():
    with pytest.raises(ValueError, match="Ob0"):
        MadauDampingDLA(x_HI=0.5)
    with pytest.raises(ValueError, match="x_HI must be >= 0"):
        MadauDampingDLA(Ob0=0.05, x_HI=-0.1)
    with pytest.raises(ValueError, match="cosmology"):
        _csp(MadauDampingDLA(Ob0=0.05, cosmo=Cosmology.planck18()))
    # the registry name gives the Madau-equivalent default; a sampled x_HI needs Ob0 and says so
    csp = _csp("madau1995_damping_dla")
    th = dict(csp.theta_init, zred=jnp.array([Z]), x_HI=jnp.array([0.5]))
    with pytest.raises(ValueError, match="Ob0"):
        csp._igm_transmission(jnp.asarray(Z), th)
    assert isinstance(make_igm_model("madau1995_damping_dla"), MadauDampingDLA)
