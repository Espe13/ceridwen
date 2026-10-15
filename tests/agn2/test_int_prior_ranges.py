"""Priors on the tabulated AGN2 axes are checked at setup (session INT; VERDICT_NLR_r1 note:
``FeltreNLR.check_prior_ranges`` had no caller, so a prior beyond the grid was clamped silently).

V-INT-PR-1  a bounded prior on an NLR axis beyond the Feltre+16 grid raises; inside it passes.
V-INT-PR-2  a bounded prior on a SKIRTOR axis beyond the grid raises; inside it passes.
V-INT-PR-3  agn_mode="cigale": a prior on fagn reaching 1 raises (CIGALE's fracAGN < 1).
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import require_test_grid  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData, SedModel  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.agn.nlr import default_nlr_file  # noqa: E402
from ceridwen.agn.skirtor import default_skirtor_path  # noqa: E402
from ceridwen.observation import Photometry  # noqa: E402
from ceridwen.priors import Uniform  # noqa: E402


def _model(agn, priors, init):
    csp = CSPBasis(SSPData.load(str(require_test_grid())),
                   lookback_time=jnp.array([0.0, 0.1, 1.0]), cosmo=Cosmology.planck18(),
                   zh_const=True, add_dust=False, add_diffuse_dust=True, add_neb=False,
                   add_igm=False, verbose=False, agn=agn)
    obs = [Photometry(filters=["jwst_f150w", "jwst_f444w"], flux=np.ones(2) * 1e-9,
                      uncertainty=np.ones(2) * 1e-10, name="phot")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, obs, priors={"logmass": Uniform(low=8.0, high=11.0), **priors},
                        free_param_init={"logmass": jnp.array([9.0]), **init}, zred=2.0)


def _need(path):
    if not pathlib.Path(path).exists():
        pytest.skip(f"MISSING {path}: did NOT run")


def test_nlr_prior_beyond_grid_raises():
    _need(default_nlr_file())
    agn = AGNConfig(torus=False, disk=False, nlr_model="feltre16")
    init = {"nlr_logu": jnp.array([-2.5])}
    _model(agn, {"nlr_logu": Uniform(low=-4.0, high=-1.5)}, init)
    with pytest.raises(ValueError, match="outside the Feltre"):
        _model(agn, {"nlr_logu": Uniform(low=-6.0, high=-1.5)}, init)


def test_skirtor_prior_beyond_grid_raises():
    _need(default_skirtor_path())
    agn = AGNConfig(torus_model="skirtor16", lines=False)
    init = {"skirtor_i": jnp.array([40.0])}
    _model(agn, {"skirtor_i": Uniform(low=0.0, high=90.0)}, init)
    with pytest.raises(ValueError, match="outside the SKIRTOR grid"):
        _model(agn, {"skirtor_i": Uniform(low=0.0, high=100.0)}, init)


def test_cigale_fagn_prior_reaching_one_raises():
    _need(default_skirtor_path())
    agn = AGNConfig(agn_mode="cigale", torus_model="skirtor16", lines=False)
    init = {"fagn": jnp.array([0.3])}
    _model(agn, {"fagn": Uniform(low=0.0, high=0.99)}, init)
    with pytest.raises(ValueError, match="fracAGN"):
        _model(agn, {"fagn": Uniform(low=0.0, high=1.0)}, init)
