"""Attenuation-law registry: consistency of every entry, and the laws fixed or added 2026-09-22.

* Every ``params`` / ``defaults`` key of every registered law is a parameter of its function
  (``Dust`` passes a law only the signature names that are also ``params`` keys, so a mismatch
  is silently dropped: the ``noll`` ``E_bump`` / ``Ebump`` bug).
* ``noll``: ``Ebump`` reaches the law through an age-bin ``Dust`` (single and repeated use).
* ``drude``: the registered law takes Angstrom and peaks at 1 at x0 (it took inverse microns).
* ``smc`` / ``lmc``: tau(5500 A) = amplitude (the registry said 1500 A).
* ``gordon03_smcbar`` / ``reddy15``: registered, read through ``Dust`` and ``DiffuseDust``, and
  normalised as documented.  Their physics against FSPS and Prospector is checked by
  ``examples/recipes/tests/check_extra_dust_laws.py`` and pinned by the ``dust_laws`` regression
  category.
"""
import contextlib
import inspect
import io

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ceridwen.dust.attenuation_laws import (ATTENUATION_LAWS, noll, drude, gordon03_smcbar,
                                            reddy15, smc, lmc)
from ceridwen.dust.DustModel import Dust, DiffuseDust

BUILTIN = ["smc", "lmc", "kriek_conroy", "powerlaw", "calzetti", "drude", "noll", "chevallard",
           "cardelli", "conroy", "gordon03_smcbar", "reddy15"]
W = jnp.array([1216.0, 1500.0, 2175.0, 2178.6, 3000.0, 5500.0, 9000.0, 20000.0])


@pytest.mark.parametrize("name", BUILTIN)
def test_registry_names_match_signature(name):
    entry = ATTENUATION_LAWS[name]
    sig = [p for p in inspect.signature(entry["func"]).parameters
           if p != "wave" and inspect.signature(entry["func"]).parameters[p].kind
           not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)]
    assert set(entry["params"]) == set(entry["defaults"]), name
    assert set(entry["params"]) <= set(sig), (name, sig)
    assert entry["doc"]
    # every documented parameter is passed through Dust and changes the curve
    d = Dust(bin_edges=[(-jnp.inf, jnp.inf)], laws=[name])
    base = {k: jnp.asarray(v) for k, v in entry["defaults"].items()}
    ref = d.compute_attenuation(W, base)[0]
    for k, v in entry["defaults"].items():
        bumped = dict(base, **{k: jnp.asarray(v + 0.37)})
        assert not np.array_equal(np.asarray(d.compute_attenuation(W, bumped)[0]),
                                  np.asarray(ref)), f"{name}: {k} does not reach the law"


def test_noll_bump_reaches_age_bin_dust():
    d = Dust(bin_edges=[(-jnp.inf, jnp.inf)], laws=["noll"])
    p = {"tau_noll": 1.0, "delta": 0.0, "c_r": 0.0, "Ebump": 3.0}
    got = np.asarray(d.compute_attenuation(W, p)[0])
    np.testing.assert_array_equal(got, np.asarray(noll(W, 1.0, 0.0, 0.0, 3.0)))
    assert got[2] > 1.3 * np.asarray(noll(W, 1.0, 0.0, 0.0, 0.0))[2]     # the bump is there
    with contextlib.redirect_stdout(io.StringIO()):
        d2 = Dust(bin_edges=[(-jnp.inf, -1.0), (-1.0, jnp.inf)], laws=["noll", "noll"])
    assert "Ebump2" in d2.get_param_names()
    p2 = {f"{k}{i}": v for i in (1, 2) for k, v in p.items()}
    p2["Ebump1"] = 0.0
    att = np.asarray(d2.compute_attenuation(W, p2))
    np.testing.assert_array_equal(att[0], np.asarray(noll(W, 1.0, 0.0, 0.0, 0.0)))
    np.testing.assert_array_equal(att[1], np.asarray(noll(W, 1.0, 0.0, 0.0, 3.0)))
    # the diffuse path always worked; it must agree
    dd = DiffuseDust("noll")
    np.testing.assert_array_equal(
        np.asarray(dd.compute_attenuation(W, {f"diffuse_{k}": v for k, v in p.items()})), got)


def test_drude_law_in_angstrom():
    d = Dust(bin_edges=[(-jnp.inf, jnp.inf)], laws=["drude"])
    got = np.asarray(d.compute_attenuation(W, {"x0": 4.59, "gamma": 0.9})[0])
    np.testing.assert_array_equal(got, np.asarray(drude(1e4 / W)))
    peak = float(d.compute_attenuation(jnp.array([1e4 / 4.59]), {"x0": 4.59, "gamma": 0.9})[0, 0])
    assert peak == pytest.approx(1.0, abs=1e-15)
    assert got.max() > 0.99                           # was ~1.7e-7 with Angstrom fed as x


@pytest.mark.parametrize("func,key", [(smc, "tau_smc"), (lmc, "tau_lmc"),
                                      (gordon03_smcbar, "tau_g03smc")])
def test_normalised_at_5500(func, key):
    assert float(func(jnp.array([5500.0]), 0.7)[0]) == pytest.approx(0.7, rel=1e-15)
    assert "5500" in ATTENUATION_LAWS[func.__name__]["params"][key]
    assert "Pei" in ATTENUATION_LAWS["smc"]["doc"] and "Pei" in ATTENUATION_LAWS["lmc"]["doc"]


def test_reddy_normalisation_and_dust_wrappers():
    r = float(reddy15(jnp.array([5500.0]), 1.0)[0])
    k55 = (-5.726 + 4.004 / 0.55 - 0.525 / 0.55**2 + 0.029 / 0.55**3 + 2.505) / 2.505
    assert r == pytest.approx(k55, rel=1e-15)
    for law, key in (("gordon03_smcbar", "tau_g03smc"), ("reddy15", "tau_reddy")):
        direct = np.asarray(ATTENUATION_LAWS[law]["func"](W, 0.7))
        d = Dust(bin_edges=[(-jnp.inf, jnp.inf)], laws=[law])
        dd = DiffuseDust(law)
        np.testing.assert_array_equal(np.asarray(d.compute_attenuation(W, {key: 0.7})[0]), direct)
        np.testing.assert_array_equal(
            np.asarray(dd.compute_attenuation(W, {f"diffuse_{key}": 0.7})), direct)


@pytest.mark.parametrize("law", ["gordon03_smcbar", "reddy15", "drude", "noll"])
def test_grad_finite(law):
    f = ATTENUATION_LAWS[law]["func"]
    g = jax.grad(lambda wv: jnp.sum(f(wv)))(jnp.geomspace(912.0, 5e4, 300))
    assert bool(jnp.all(jnp.isfinite(g)))
