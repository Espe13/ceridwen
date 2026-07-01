"""
Regression: the LOSVD smoother must not contaminate pixels just above the
Lyman limit (≥ 912 Å) with circular-FFT wrap-around from the bright red end
of the smoothing window.

Before the fix in ``sedpy_jax.smoothing.smooth_fft_padded`` (2026-06-04), the
factory-built velocity smoother in ``CSPBasis._setup_losvd_kernel`` ran a
periodic FFT convolution on a ~16k-pixel log-uniform grid spanning
``(912, 25000) Å``.  The blue-edge pixels (913 / 915 / 917 / 919 Å) had
raw amplitudes ≈ 1e-21 W/Hz (heavy dust attenuation under the Lyman limit),
while the red-edge pixels were ≈ 1e-7 W/Hz; the cyclic FFT treated them
as adjacent and the red-end flux leaked back into the blue-end pixels,
inflating pixel 913 by ~14 orders of magnitude.

This test pins the corrected behaviour: pixel 913 must not exceed its
raw counterpart (``get_spectrum_dattn_nodem_neb``) by more than ~10×.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis

from _gridfixture import require_test_grid

# Uses NebularModelFSPSMatch, which reads CLOUDY grids from $SPS_HOME/nebular/;
# those ship with FSPS, so this module requires a working FSPS install.
pytestmark = pytest.mark.fsps

# This regression also needs the BPASS+AGB grid variant; commit it to
# tests/fixtures/ (see tests/fixtures/README.md) or the test skips.
SSP_FILE = str(require_test_grid("ssp_data_bpass_agb_dust.h5"))


def _build_dust_neb_csp():
    T_UNIV = 13.8
    N = 10
    lookback = jnp.linspace(0.0, T_UNIV - 0.01, N)

    def gb(t, c, w, a=1.0):
        return a * jnp.exp(-0.5 * ((t - c) / w) ** 2)

    sfh = gb(lookback, 0.05, 0.03, 1.0) + gb(lookback, 11.0, 0.8, 0.7)

    ssp = SSPData.load(SSP_FILE)
    csp = CSPBasis(
        ssp,
        theta={"lookback_time": lookback, "sfh": sfh, "Z": jnp.array([-4.0])},
        tuniv=T_UNIV, zh_const=True,
        add_dust=True, add_diffuse_dust=True, add_dust_emission=False,
        add_neb=True, nebemlineinspec=True,
        # isoc_type is auto-resolved from the SSP grid's recorded provenance.
        init_neb_params={"cloudy_dust": False, "match_fsps": True},
        init_dust_params={"bin_edges": [(-jnp.inf, -1.97)], "laws": ["powerlaw"]},
        diffuse_law="kriek_conroy",
        verbose=False, sps_home=os.environ.get("SPS_HOME"),
        sfh_interp="linear",
    )
    return csp


def _theta(csp):
    t = dict(csp.theta_init)
    t["diffuse_tau_kc"] = 2.0
    t["diffuse_dust_index"] = -0.7
    t["tau_pow"] = 2.0
    t["alpha"] = -1.0
    t["gas_logz"] = jnp.array([0.0])
    t["gas_logu"] = jnp.array([-2.0])
    return t


def test_losvd_no_lyman_wrap_around():
    """Pixel 913 Å must not be inflated by FFT wrap-around past 10× its raw value."""
    csp = _build_dust_neb_csp()
    theta = _theta(csp)

    spec_raw = np.asarray(csp.get_spectrum_dattn_nodem_neb(theta))
    spec_smoothed = np.asarray(csp.get_spectrum(theta))

    w = np.asarray(csp.wave)
    i913 = int(np.where(w == 913.0)[0][0])

    raw_913 = float(spec_raw[i913])
    smoothed_913 = float(spec_smoothed[i913])

    # Pre-fix smoothed_913 was ≈ 3.6e-7 with raw_913 ≈ 7.4e-21 (14 orders of
    # magnitude inflation).  Post-fix smoothed_913 sits within a small
    # multiple of raw_913 — the only legitimate sources of inflation here
    # are the genuine velocity-broadening of nearby pixels and the
    # boundary one-sidedness of the smoother window.
    assert smoothed_913 / raw_913 < 10.0, (
        f"LOSVD smoother at 913 Å is contaminated by FFT wrap-around: "
        f"raw={raw_913:.3e}, smoothed={smoothed_913:.3e}, "
        f"ratio={smoothed_913/raw_913:.3e} (must be < 10)."
    )


def test_lyman_edge_no_isolated_spike():
    """No single pixel between 905 and 925 Å should exceed its immediate
    neighbours by more than 30× in the smoothed spectrum.  A genuine Lyman
    edge is a *step* across pixels 911→913, not a delta at a single grid
    point."""
    csp = _build_dust_neb_csp()
    theta = _theta(csp)
    spec = np.asarray(csp.get_spectrum(theta))
    w = np.asarray(csp.wave)
    mask = (w >= 905) & (w <= 925)
    idx = np.where(mask)[0]
    # Skip the genuine 911→913 Lyman jump (allowed); flag any other pixel
    # that is >30× either neighbour.
    for k in range(1, len(idx) - 1):
        i = idx[k]
        if w[i] in (911.0, 913.0):  # legitimate edge
            continue
        ratio_left  = spec[i] / max(spec[idx[k - 1]], 1e-30)
        ratio_right = spec[i] / max(spec[idx[k + 1]], 1e-30)
        assert max(ratio_left, ratio_right) < 30.0, (
            f"Spike at λ = {w[i]:.0f} Å: spec[i]={spec[i]:.3e}, "
            f"neighbours = {spec[idx[k-1]]:.3e} / {spec[idx[k+1]]:.3e}"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
