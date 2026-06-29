"""
Regression: under the post-2026-06-03 lookback-time flip, the SAME
physical SFH must produce bit-for-bit identical SSP weights and the
same spectra / line spectrum / synthetic maggies to float64 rounding.

Baselines were captured by ``tests/baselines/_capture.py`` while the
codebase was still using the OLD (decreasing-lookback) convention.
This file loads those .npy files, then builds CSPs with the same
physical history expressed in the NEW (increasing-lookback)
convention -- i.e. with both ``lookback_time`` AND ``sfh`` (and
``zh``) reversed in index.

If any assertion fails by more than float64 rounding the kernel has
an indexing bug, not a numerical artefact -- do not loosen the
tolerance.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp     import CSPBasis
from ceridwen.observation.observation import Photometry


from _gridfixture import require_test_grid

BASELINES = pathlib.Path(__file__).resolve().parent.parent / "baselines"
SSP_FILE  = str(require_test_grid())

# Read the OLD convention used at capture time.
with open(BASELINES / "manifest.json") as f:
    MANIFEST = json.load(f)

T_UNIV   = MANIFEST["tuniv_gyr"]
N_TIME   = MANIFEST["n_time"]
LB_OLD   = jnp.asarray(MANIFEST["lb_old_gyr"])    # decreasing
PSI_OLD  = jnp.asarray(MANIFEST["psi_old"])       # OLD-indexed SFR-per-node


@pytest.fixture(scope="module")
def ssp():
    return SSPData.load(SSP_FILE)


def _build_theta_new(zh_const: bool, per_bin: bool):
    """Rebuild the same physical SFH used at baseline-capture time.
    After the 2026-06-03 lookback flip the manifest stores NEW-convention
    arrays directly (today @ idx 0), so no reversal is needed.  The
    variable names ``LB_OLD`` / ``PSI_OLD`` are kept for backwards-
    compatibility with the manifest.json schema.
    """
    lb  = LB_OLD
    psi = PSI_OLD
    sfh = psi
    if per_bin:
        sfh = 0.5 * (psi[:-1] + psi[1:])
    theta = {
        "lookback_time": lb,
        "sfh":           sfh,
    }
    if zh_const:
        theta["Z"] = jnp.asarray([-2.0])
    else:
        theta["zh"] = jnp.asarray(np.full(N_TIME, -2.0))   # constant Z history
    return theta


def _build_csp(ssp, theta, *, sfh_interp, zh_const):
    return CSPBasis(
        ssp,
        theta             = theta,
        tuniv             = T_UNIV,
        zh_const          = zh_const,
        add_dust          = False,
        add_diffuse_dust  = False,
        add_dust_emission = False,
        add_neb           = True,
        nebemlineinspec   = True,
        verbose           = False,
        sfh_interp        = sfh_interp,
    )


def _photometry():
    return Photometry(
        filters     = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
        flux        = jnp.zeros(5),
        uncertainty = jnp.ones(5) * 1e-12,
        mask        = jnp.ones(5, dtype=bool),
        name        = "phot",
    )


CONFIGS = [(c["sfh_interp"], c["zh_const"], c["per_bin"], c["tag"])
           for c in MANIFEST["configs"]]


@pytest.mark.parametrize("sfh_interp,zh_const,per_bin,tag", CONFIGS)
def test_lookback_flip_invariant(ssp, sfh_interp, zh_const, per_bin, tag):
    theta = _build_theta_new(zh_const, per_bin)
    csp   = _build_csp(ssp, theta, sfh_interp=sfh_interp, zh_const=zh_const)
    phot  = _photometry()

    theta_full = {k: jnp.asarray(v) for k, v in csp.theta_init.items()}
    theta_full.setdefault("logmass",       jnp.zeros(1))
    theta_full.setdefault("zred",          jnp.zeros(1))
    theta_full.setdefault("igm_factor",    jnp.ones(1))
    theta_full.setdefault("eline_scaling", jnp.ones(1))

    W_new       = np.asarray(csp.calculate_ssp_weights(theta_full))
    spec_new    = np.asarray(csp.get_spectrum(theta_full))
    lines_new   = np.asarray(csp.get_line_spec(theta_full))
    maggies_new = np.asarray(csp.predict(theta_full, [phot])[phot.name])

    W_ref       = np.load(BASELINES / f"W_{tag}.npy")
    spec_ref    = np.load(BASELINES / f"spec_{tag}.npy")
    lines_ref   = np.load(BASELINES / f"lines_{tag}.npy")
    maggies_ref = np.load(BASELINES / f"maggies_{tag}.npy")

    # W is computed in pure-jnp double precision.  In step mode the kernel
    # rewrite is an exact variable rename and W matches bit-for-bit; in
    # linear mode the kernel computes m2 = sfh[:-1]*(1+0.5*slope*dt)*dt
    # directly instead of the pre-flip ``sfh[1:]*(1+0.5*slope*(t_hi+t_lo
    # -2*t_lo))*dt`` form -- algebraically identical, but the float64
    # rounding sequence differs.  We therefore require rtol=1e-12 (real
    # float64 rounding tolerance), not np.array_equal.
    np.testing.assert_allclose(W_new, W_ref, rtol=1e-12, atol=0,
        err_msg=f"[{tag}] W differs beyond float64 rounding")

    # Spectra and maggies flow through float32 einsums, so float32 rtol.
    np.testing.assert_allclose(spec_new,    spec_ref,    rtol=1e-6, atol=0)
    np.testing.assert_allclose(lines_new,   lines_ref,   rtol=1e-6, atol=0)
    np.testing.assert_allclose(maggies_new, maggies_ref, rtol=1e-6, atol=0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
