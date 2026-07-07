"""Construction-time API of ``CSPBasis``: the ``lookback_time=`` shortcut and
the informative errors for malformed / missing theta.

Covers:

1. Shortcut equivalence — ``CSPBasis(ssp, lookback_time=lb)`` produces the
   same static structure (``sfh_times``, ``n_time``, ``sfh_per_bin``,
   metallicity mode) as the equivalent full ``theta=`` construction, and its
   neutral initial metallicity is inside the SSP grid (no clamp warning).
2. ``sfh_per_bin=True`` gives a per-bin sfh of shape ``(n_time - 1,)``.
3. Mutual exclusion — passing both ``theta=`` and ``lookback_time=`` raises.
4. No structure at all (``theta=None``, no shortcut) raises with a message
   naming both construction routes.
5. Missing ``sfh`` / ``lookback_time`` keys raise informative ValueErrors
   (not raw KeyErrors).
6. Fewer than 2 lookback nodes raises (a 1-node grid must not slip through
   the monotonicity check, whose diff is empty for a single node).

Pure construction tests: no FSPS, no fitting. CPU only.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import require_test_grid  # noqa: E402

from ceridwen import CSPBasis, SSPData  # noqa: E402

KW = dict(zh_const=True, add_neb=False, add_igm=False,
          add_dust=False, add_diffuse_dust=True, verbose=False)


@pytest.fixture(scope="module")
def ssp():
    return SSPData.load(str(require_test_grid()))


def test_shortcut_matches_full_theta_structure(ssp):
    lb = jnp.linspace(0.0, 12.0, 6)
    via_theta = CSPBasis(
        ssp, theta={"lookback_time": lb, "sfh": jnp.ones(6),
                    "Z": jnp.array([-1.85])}, **KW)
    via_shortcut = CSPBasis(ssp, lookback_time=lb, **KW)

    np.testing.assert_allclose(np.asarray(via_shortcut.sfh_times),
                               np.asarray(via_theta.sfh_times))
    assert via_shortcut.n_time == via_theta.n_time == 6
    assert via_shortcut.sfh_per_bin is False
    assert via_shortcut.zh_is_scalar is True          # zh_const -> 'Z' mode
    # Neutral initial Z must lie inside the grid: no clamp messages.
    assert via_shortcut.check_param_ranges(warn=False) == []


def test_shortcut_per_bin_sfh(ssp):
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
                   sfh_per_bin=True, **KW)
    assert csp.sfh_per_bin is True
    assert csp.theta_init["sfh"].shape == (5,)


def test_shortcut_time_varying_zh(ssp):
    kw = dict(KW, zh_const=False)
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6), **kw)
    assert csp.zh_is_scalar is False
    assert csp.theta_init["zh"].shape == (6,)


def test_theta_and_shortcut_are_mutually_exclusive(ssp):
    lb = jnp.linspace(0.0, 12.0, 6)
    with pytest.raises(ValueError, match="not both"):
        CSPBasis(ssp, theta={"lookback_time": lb, "sfh": jnp.ones(6),
                             "Z": jnp.array([-1.85])},
                 lookback_time=lb, **KW)


def test_no_structure_raises_with_both_routes_named(ssp):
    with pytest.raises(ValueError, match="lookback_time"):
        CSPBasis(ssp, **KW)


def test_missing_sfh_is_a_valueerror_not_keyerror(ssp):
    with pytest.raises(ValueError, match="'sfh'"):
        CSPBasis(ssp, theta={"lookback_time": jnp.linspace(0.0, 12.0, 6),
                             "Z": jnp.array([-1.85])}, **KW)


def test_missing_lookback_time_is_a_valueerror_not_keyerror(ssp):
    with pytest.raises(ValueError, match="'lookback_time'"):
        CSPBasis(ssp, theta={"sfh": jnp.ones(6), "Z": jnp.array([-1.85])},
                 **KW)


def test_single_node_grid_rejected(ssp):
    with pytest.raises(ValueError, match="at least"):
        CSPBasis(ssp, lookback_time=jnp.array([0.0]), **KW)
