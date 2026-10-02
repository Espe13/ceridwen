"""Shared fixtures of the AGN tests (AGN_PLAN §4).

* ``torus_path``: ``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat``; skips without it.
* ``model_wave``: the test SSP grid's wavelength axis (``ssp_grid_path``).
* ``agn_ref(name)``: a reference npz of session REF (``tests/reference/agn/<name>``, or the
  directory in ``$CERIDWEN_AGN_REF``); skips, saying so, when it is not there.
* ``build_csp(agn, **kw)``: a small CSPBasis on the test grid (5 nodes, constant Z).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

REF_DIR = pathlib.Path(__file__).resolve().parents[1] / "reference" / "agn"


@pytest.fixture(scope="session")
def torus_path():
    root = os.environ.get("SPS_HOME")
    if not root:
        pytest.skip("$SPS_HOME is unset: the Nenkova+08 torus templates live in $SPS_HOME/dust")
    p = pathlib.Path(root) / "dust" / "Nenkova08_y010_torusg_n10_q2.0.dat"
    if not p.exists():
        pytest.skip(f"{p} not found")
    return p


@pytest.fixture(scope="session")
def ssp_data(ssp_grid_path):
    from ceridwen import SSPData
    return SSPData.load(ssp_grid_path)


@pytest.fixture(scope="session")
def model_wave(ssp_data):
    return np.asarray(ssp_data.ssp_wave, dtype=np.float64)


@pytest.fixture(scope="session")
def agn_ref():
    def load(name):
        d = pathlib.Path(os.environ.get("CERIDWEN_AGN_REF", REF_DIR))
        p = d / name
        if not p.exists():
            pytest.skip(f"reference {p} not present (session REF writes tests/reference/agn/; "
                        "set $CERIDWEN_AGN_REF to another directory)")
        return np.load(p, allow_pickle=False)
    return load


@pytest.fixture(scope="session")
def build_csp(ssp_data, torus_path):
    from ceridwen import CSPBasis, Cosmology
    cosmo = Cosmology.planck18()

    def build(agn=None, **kw):
        args = dict(lookback_time=jnp.linspace(0.0, 5.0, 5), zh_const=True, add_neb=False,
                    add_igm=False, verbose=False, cosmo=cosmo)
        args.update(kw)
        return CSPBasis(ssp_data, agn=agn, **args)
    return build
