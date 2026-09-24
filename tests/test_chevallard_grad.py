"""B1-004: the chevallard law has a finite gradient at tau_chev = 0 (its natural prior
bound) and is unchanged, bit for bit, for tau_chev > 0."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ceridwen.dust.attenuation_laws import chevallard  # noqa: E402

WAVE = jnp.linspace(1000.0, 20000.0, 50)


def test_gradient_finite_at_zero_and_matches_the_limit():
    g0 = jax.grad(lambda t: jnp.sum(chevallard(WAVE, tau_chev=t)))(0.0)
    g_small = jax.grad(lambda t: jnp.sum(chevallard(WAVE, tau_chev=t)))(1e-14)
    assert np.isfinite(g0)
    assert float(g0) == float(jnp.sum((WAVE / 5500.0) ** (-(2.8 + 0.3 * (WAVE * 1e-4 - 0.55)))))
    assert abs(float(g_small) - float(g0)) < 1e-4 * abs(float(g0))


def test_value_unchanged_for_positive_tau():
    for t in (1e-8, 0.3, 1.0, 2.5):
        ref = t * (WAVE / 5500.0) ** (-(2.8 / (1.0 + jnp.sqrt(t))
                                         + (0.3 - 0.05 * t) * (WAVE * 1e-4 - 0.55)))
        assert np.asarray(chevallard(WAVE, tau_chev=t)).tobytes() == np.asarray(ref).tobytes()


def test_missing_law_parameter_warns_or_names_the_key():
    """B1-007: a law parameter missing from theta used the registry default silently (Dust)
    or raised a bare KeyError (DiffuseDust)."""
    import pytest
    from ceridwen.dust.DustModel import make_law_wrapper
    from ceridwen.dust.attenuation_laws import powerlaw
    w = make_law_wrapper(powerlaw, ["tau_pow", "alpha_pow"], {"tau_pow": 1.0, "alpha_pow": -1.0})
    with pytest.warns(UserWarning, match=r"theta has no \['tau_pow'\]"):
        a = w(WAVE, {"alpha_pow": -0.7})
    assert np.array_equal(np.asarray(a), np.asarray(powerlaw(WAVE, 1.0, -0.7)))
    w2 = make_law_wrapper(powerlaw, ["diffuse_tau_pow", "diffuse_alpha_pow"])
    with pytest.raises(KeyError, match="powerlaw.*diffuse_tau_pow"):
        w2(WAVE, {"diffuse_alpha_pow": -0.7})
