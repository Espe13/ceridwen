"""The picket-fence spectrum from the factored nebular path equals the dense
(n_z, n_age, n_wave) formulation it replaced, with and without dust emission."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid                                   # noqa: E402

from ceridwen import SSPData, CSPBasis, Cosmology                          # noqa: E402


def _csp(dust_emission):
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    return CSPBasis(SSPData.load(str(path)), lookback_time=jnp.linspace(0.0, 5.0, 5),
                    zh_const=True, add_dust=True, add_diffuse_dust=True, add_neb=True,
                    add_dust_emission=dust_emission, fesc_geometry="picket",
                    sps_home=os.environ["SPS_HOME"], verbose=False, cosmo=Cosmology.planck18())


def _dense_picket(csp, theta, include_lines):
    """The previous dense formulation (materialised nebular cube, masked flux copy)."""
    W = csp.calculate_ssp_weights(theta=theta).astype(jnp.float32)
    fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
    attn, attn_diffuse = csp.attenuate_dust(csp.wave, theta)
    attn_age = jnp.exp(-jnp.einsum("ab,bw->aw", csp._age_bin_mix, attn.astype(jnp.float32)))
    diffuse = jnp.exp(-attn_diffuse.astype(jnp.float32))
    young = csp.young_mask.astype(jnp.float32)
    neb_all = csp._build_neb_array(theta, include_lines=include_lines).astype(jnp.float32)
    F_cov = jnp.where(csp.kill_ion[None, :, :], jnp.float32(0.0), csp.flux)
    cov_w = (1.0 - fo) * young + (1.0 - young)
    stellar_cov = jnp.einsum("za,zaw,aw,a->w", W, F_cov, attn_age, cov_w)
    neb_cov = (1.0 - fo) * jnp.einsum("za,zaw,aw->w", W, neb_all, attn_age)
    covered = (stellar_cov + neb_cov) * diffuse
    clear = fo * jnp.einsum("za,zaw,a->w", W, csp.flux, young)
    attenuated = covered + clear
    dust_free = (clear + jnp.einsum("za,zaw,a->w", W, F_cov, cov_w)
                 + (1.0 - fo) * jnp.einsum("za,zaw->w", W, neb_all))
    return attenuated, dust_free, diffuse


@pytest.mark.parametrize("include_lines", [False, True])
def test_picket_nodem_matches_dense(include_lines):
    csp = _csp(False)
    theta = dict(csp.theta_init, logzsol=jnp.array([-2.0 - csp.log10_zsun]), frac_obrun=jnp.array([0.3]))
    new = np.asarray(csp.get_spectrum(theta, include_lines=include_lines))
    old = np.asarray(_dense_picket(csp, theta, include_lines)[0])
    np.testing.assert_allclose(new, old, rtol=2e-5, atol=1e-12 * np.max(old))


def test_picket_dem_matches_dense():
    csp = _csp(True)
    theta = dict(csp.theta_init, logzsol=jnp.array([-2.0 - csp.log10_zsun]), frac_obrun=jnp.array([0.3]))
    new = np.asarray(csp.get_spectrum(theta, include_lines=True))
    attenuated, dust_free, diffuse = _dense_picket(csp, theta, True)
    old, _, _ = csp.dust_emi.compute_dust_emission(
        spec_attn=attenuated, spec_dustfree=dust_free, spec_lambda=csp.wave,
        diffuse_curve=diffuse, duste_qpah=theta["duste_qpah"],
        duste_umin=theta["duste_umin"], duste_gamma=theta["duste_gamma"])
    old = np.asarray(old)
    np.testing.assert_allclose(new, old, rtol=2e-5, atol=1e-12 * np.max(old))


def test_picket_frac_obrun_zero_equals_full_covering():
    """fo = 0: every young photon is covered; the picket spectrum equals the runaway
    geometry with frac_obrun = 0 (no bypass, LyC absorbed)."""
    csp = _csp(False)
    theta = dict(csp.theta_init, logzsol=jnp.array([-2.0 - csp.log10_zsun]), frac_obrun=jnp.array([0.0]))
    picket = np.asarray(csp.get_spectrum(theta, include_lines=True))
    csp.fesc_geometry = "runaway_bc"
    runaway = np.asarray(csp.get_spectrum(theta, include_lines=True))
    np.testing.assert_allclose(picket, runaway, rtol=1e-4, atol=1e-12 * np.max(runaway))
