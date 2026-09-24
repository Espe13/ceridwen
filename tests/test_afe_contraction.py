"""B1-022: a sampled [alpha/Fe] is folded into the SSP weights, never built as a plane.

Reference: the explicit two-plane interpolation (1 - w) flux[k-1] + w flux[k]
(``CSPBasis_afe._flux_at_afe``) contracted with the same weights and attenuation.  The two
agree to float32 rounding (the fold rounds the weights, the plane rounds the flux; each
spectrum pixel is a float32 sum over n_z * n_age terms): tolerance 2e-6 of the spectrum
maximum.  Memory: XLA's own analysis of jit(vmap(get_spectrum)) must stay below one
[alpha/Fe] plane at W=64 (the plane-per-sample form needs about W planes).
"""
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen import Cosmology

GRID = (pathlib.Path(__file__).resolve().parents[1] / "ceridwen" / "data" / "test_data"
        / "amist_c3k_lr_chab_afe.h5")
pytestmark = pytest.mark.skipif(not GRID.is_file(), reason=f"{GRID.name} not found")


def _csp(**kw):
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe
    return CSPBasis_afe(SSPDataAfe.load(str(GRID)), lookback_time=np.linspace(0.0, 12.0, 6),
                        verbose=False, cosmo=Cosmology.planck18(), **kw)


def _reference(csp, theta, dusty):
    """The spectrum with the interpolated plane built explicitly (the pre-B1-022 algorithm)."""
    flux = csp._flux_at_afe(theta)
    W = csp.calculate_ssp_weights(theta).astype(jnp.float32)
    if not dusty:
        return jnp.einsum("za,zaw->w", W, flux)
    attn, attn_diffuse = csp.attenuate_dust(csp.wave, theta)
    A = jnp.exp(-jnp.einsum("ab,bw->aw", csp._age_bin_mix, attn.astype(jnp.float32)))
    if "frac_obrun" in theta:
        fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
        A = (jnp.float32(1.0) - fo) * A + fo
    return jnp.einsum("za,zaw,aw->w", W, flux, A) * jnp.exp(-attn_diffuse.astype(jnp.float32))


CASES = [
    dict(add_dust=False, add_diffuse_dust=False, zh_const=True, sfh_interp="step"),
    dict(zh_const=True, sfh_interp="step"),
    dict(zh_const=False, sfh_interp="linear"),
]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("obrun", [False, True])
def test_folded_afe_equals_explicit_plane(kw, obrun):
    dusty = kw.get("add_dust", True)
    if obrun and not dusty:
        pytest.skip("frac_obrun needs dust")
    csp = _csp(**kw)
    th0 = dict(csp.theta_init)
    if obrun:
        th0["frac_obrun"] = jnp.array([0.3])
    for afe in (-0.2, 0.0, 0.13, 0.2, 0.3, 0.6, -0.5, 0.9):   # nodes, interior, clamped edges
        th = {**th0, "afe": jnp.array([afe])}
        new = np.asarray(csp.get_spectrum(th))
        ref = np.asarray(_reference(csp, th, dusty))
        assert np.max(np.abs(new - ref)) <= 2e-6 * np.max(np.abs(ref)), afe


def test_afe_absent_unchanged():
    csp = _csp(zh_const=True)
    th = dict(csp.theta_init)
    np.testing.assert_array_equal(np.asarray(csp.get_spectrum(th)),
                                  np.asarray(_reference(csp, th, True)))


@pytest.mark.skipif(not os.environ.get("SPS_HOME"), reason="dust emission needs $SPS_HOME")
def test_folded_afe_dust_emission():
    """With dust emission both the dust-free and the attenuated contractions are folded."""
    csp = _csp(zh_const=True, add_dust_emission=True)
    th = {**csp.theta_init, "afe": jnp.array([0.3])}
    new = np.asarray(csp.get_spectrum(th))
    flux = csp._flux_at_afe(th)
    W = csp.calculate_ssp_weights(th).astype(jnp.float32)
    attn, attn_diffuse = csp.attenuate_dust(csp.wave, th)
    A = jnp.exp(-jnp.einsum("ab,bw->aw", csp._age_bin_mix, attn.astype(jnp.float32)))
    d = jnp.exp(-attn_diffuse.astype(jnp.float32))
    ref, _, _ = csp.dust_emi.compute_dust_emission(
        spec_attn=jnp.einsum("za,zaw,aw->w", W, flux, A) * d,
        spec_dustfree=jnp.einsum("za,zaw->w", W, flux), spec_lambda=csp.wave,
        diffuse_curve=d, duste_qpah=th["duste_qpah"], duste_umin=th["duste_umin"],
        duste_gamma=th["duste_gamma"])
    ref = np.asarray(ref)
    assert np.max(np.abs(new - ref)) <= 2e-6 * np.max(np.abs(ref))


def test_vmap_memory_stays_below_one_plane():
    csp = _csp(zh_const=True)
    th = {**csp.theta_init, "afe": jnp.array([0.1])}
    W = 64
    thb = {k: jnp.broadcast_to(v, (W,) + v.shape) for k, v in th.items()}
    ma = jax.jit(jax.vmap(lambda t: csp.get_spectrum(t))).lower(thb).compile().memory_analysis()
    if ma is None:
        pytest.skip("this backend gives no memory analysis")
    plane = csp.flux[0].nbytes
    assert ma.temp_size_in_bytes < plane, (ma.temp_size_in_bytes, plane)


@pytest.mark.parametrize("afe", [0.0, 0.2, 0.3, -0.2, 0.6])
def test_grad_afe_finite(afe):
    """d spectrum / d afe is finite in a cell, exactly at a node and at the grid edges."""
    csp = _csp(zh_const=True)
    th0 = dict(csp.theta_init)
    g = jax.grad(lambda a: jnp.sum(jnp.log1p(csp.get_spectrum({**th0, "afe": jnp.array([a])}))))
    assert np.isfinite(float(g(afe)))
