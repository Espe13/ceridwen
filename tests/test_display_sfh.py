"""
Tests for ``CSPBasis.display_sfh``.

Cover the four (sfh_interp, sfh-input-shape) combinations:
    (a) step,   per-node
    (b) step,   per-bin
    (c) linear, per-node
    (d) linear, per-bin

Each case:
    * builds a CSPBasis,
    * calls display_sfh with no overlays/edges,
    * asserts the mass-conservation contract did not raise, and
    * confirms ``ax`` contains the expected number of line segments:
        n_bin = n_time - 1   for step
        n_node - 1           for linear  (equal to n_bin)
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis


from _gridfixture import require_test_grid

SSP_FILE = str(require_test_grid())


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ssp():
    return SSPData.load(SSP_FILE)


def _make_lookback(n_time=10, tuniv=13.8):
    """Lookback grid in Gyr, NEW convention: T_0 = 0 (today), T_{-1} ≈ tuniv."""
    return jnp.linspace(0.0, tuniv, n_time)


def _make_sfh(lookback, per_bin: bool):
    """A rising SFH: SFR(lookback) = exp(-lookback / tau)."""
    sfr_node = jnp.exp(-lookback / 1.0)
    if per_bin:
        return 0.5 * (sfr_node[:-1] + sfr_node[1:])
    return sfr_node


def _build_csp(ssp, *, sfh_interp, per_bin, n_time=10, tuniv=13.8):
    lookback = _make_lookback(n_time=n_time, tuniv=tuniv)
    sfh      = _make_sfh(lookback, per_bin=per_bin)
    theta    = {
        "lookback_time": lookback,
        "sfh":           sfh,
        "Z":             jnp.array([-0.5]),
    }
    return CSPBasis(
        ssp,
        theta             = theta,
        tuniv             = tuniv,
        zh_const          = True,
        add_dust          = False,
        add_diffuse_dust  = False,
        add_dust_emission = False,
        add_neb           = False,
        verbose           = False,
        sfh_interp        = sfh_interp,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("sfh_interp,per_bin", [
    ("step",   False),
    ("step",   True),
    ("linear", False),
    ("linear", True),
])
def test_display_sfh_segment_count(ssp, sfh_interp, per_bin):
    """display_sfh draws exactly n_bin line segments (no extra)."""
    import matplotlib.pyplot as plt

    n_time = 10
    csp    = _build_csp(ssp, sfh_interp=sfh_interp, per_bin=per_bin,
                        n_time=n_time)

    fig, ax = plt.subplots()
    out_ax = csp.display_sfh(ax=ax, overlay_nodes=False,
                             show_bin_edges=False)
    assert out_ax is ax

    n_bin = n_time - 1
    assert len(ax.lines) == n_bin, (
        f"({sfh_interp}, per_bin={per_bin}): expected {n_bin} segments, "
        f"got {len(ax.lines)}"
    )
    plt.close(fig)


def test_display_sfh_step_segments_are_horizontal(ssp):
    """Step mode segments must be exactly piecewise-constant — y0 == y1."""
    import matplotlib.pyplot as plt

    csp = _build_csp(ssp, sfh_interp="step", per_bin=False, n_time=8)
    fig, ax = plt.subplots()
    csp.display_sfh(ax=ax, overlay_nodes=False)

    for line in ax.lines:
        ys = line.get_ydata()
        assert ys[0] == ys[1], (
            f"step-mode segment is not horizontal: y={ys}"
        )
    plt.close(fig)


def test_display_sfh_step_height_matches_weight_branch(ssp):
    """Per-bin SFR drawn in step mode must equal _ssp_weights' bar_psi."""
    import matplotlib.pyplot as plt

    n_time = 8
    csp    = _build_csp(ssp, sfh_interp="step", per_bin=False, n_time=n_time)

    # Expected per-bin SFR -- the same branch _ssp_weights uses for per-node:
    psi      = np.asarray(csp.theta_init["sfh"])
    expected = 0.5 * (psi[:-1] + psi[1:])

    fig, ax = plt.subplots()
    csp.display_sfh(ax=ax, overlay_nodes=False)
    drawn = np.array([line.get_ydata()[0] for line in ax.lines])
    # Segments are emitted in oldest-to-youngest order — same as the
    # per-bin SFR returned by _ssp_weights (t_hi[i] = sfh_times[i]).
    np.testing.assert_allclose(drawn, expected, rtol=1e-12)
    plt.close(fig)


def test_display_sfh_linear_segments_connect_to_nodes(ssp):
    """Linear mode segments must chord between consecutive SFH nodes."""
    import matplotlib.pyplot as plt

    n_time = 8
    csp    = _build_csp(ssp, sfh_interp="linear", per_bin=False, n_time=n_time)
    psi    = np.asarray(csp.theta_init["sfh"])

    fig, ax = plt.subplots()
    csp.display_sfh(ax=ax, overlay_nodes=False)
    # Each segment i has y = [psi_nodes[i+1], psi_nodes[i]] (older T on the
    # right of the segment in data order).
    for i, line in enumerate(ax.lines):
        ys = line.get_ydata()
        np.testing.assert_allclose(ys, [psi[i + 1], psi[i]], rtol=1e-12)
    plt.close(fig)


def test_display_sfh_mass_conservation_does_not_raise(ssp):
    """The mass-conservation assertion must not fire on a normal SFH."""
    import matplotlib.pyplot as plt

    for sfh_interp in ("step", "linear"):
        for per_bin in (False, True):
            csp = _build_csp(ssp, sfh_interp=sfh_interp, per_bin=per_bin,
                             n_time=10)
            fig, ax = plt.subplots()
            csp.display_sfh(ax=ax)
            plt.close(fig)


def test_display_sfh_units_rescales_x(ssp):
    """The x-axis scales with the chosen unit."""
    import matplotlib.pyplot as plt

    csp = _build_csp(ssp, sfh_interp="step", per_bin=False, n_time=6)
    psi = np.asarray(csp.theta_init["sfh"])
    bar_psi_expected = 0.5 * (psi[:-1] + psi[1:])

    for units, factor in [("Gyr", 1.0), ("Myr", 1e3), ("yr", 1e9)]:
        fig, ax = plt.subplots()
        csp.display_sfh(ax=ax, overlay_nodes=False, units=units)
        x_max = max(line.get_xdata().max() for line in ax.lines)
        # T_max in Gyr = sfh_times.max() / 1e9
        expected = float(np.asarray(csp.sfh_times).max()) / 1e9 * factor
        assert np.isclose(x_max, expected, rtol=1e-6), (
            f"units={units}: x_max={x_max} vs expected {expected}"
        )
        plt.close(fig)


def test_display_sfh_default_uses_theta_init(ssp):
    """Calling with no theta uses self.theta_init + self.sfh_times grid."""
    import matplotlib.pyplot as plt

    csp = _build_csp(ssp, sfh_interp="step", per_bin=False, n_time=10)

    fig, ax1 = plt.subplots()
    csp.display_sfh(ax=ax1, overlay_nodes=False)
    drawn_default = [line.get_ydata()[0] for line in ax1.lines]
    plt.close(fig)

    fig, ax2 = plt.subplots()
    csp.display_sfh(ax=ax2, overlay_nodes=False,
                    theta={"sfh": csp.theta_init["sfh"],
                           "lookback_time": np.asarray(csp.sfh_times) / 1e9})
    drawn_explicit = [line.get_ydata()[0] for line in ax2.lines]
    plt.close(fig)

    np.testing.assert_allclose(drawn_default, drawn_explicit, rtol=1e-12)


def test_display_sfh_xaxis_t0_on_left(ssp):
    """T = 0 (today) must be at the origin on the LEFT of the plot."""
    import matplotlib.pyplot as plt

    csp = _build_csp(ssp, sfh_interp="step", per_bin=False, n_time=6)
    fig, ax = plt.subplots()
    csp.display_sfh(ax=ax, overlay_nodes=False)
    xlo, xhi = ax.get_xlim()
    assert xlo < xhi, (
        f"x-axis is not left-to-right increasing: xlim = ({xlo}, {xhi})"
    )
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
