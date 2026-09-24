"""B2-002: prior draws for a vector parameter have shape (n, *param_shape) whether the
prior has scalar parameters (broadcast), per-element parameters, or is multivariate."""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.sampler.priors import (MultivariateNormalPrior, Uniform,  # noqa: E402
                                     sample_for_parameter)

PRIORS = {
    "scalar": Uniform(low=-1.0, high=1.0),
    "per_element": Uniform(low=jnp.array([-1.0, -2.0, -3.0]), high=jnp.array([1.0, 2.0, 3.0])),
    "mvn": MultivariateNormalPrior(mean=jnp.zeros(3), Sigma=jnp.eye(3)),
}


@pytest.mark.parametrize("kind", sorted(PRIORS))
def test_sample_for_parameter_shape(kind):
    d = sample_for_parameter(PRIORS[kind], jax.random.PRNGKey(0), 7, (3,))
    assert d.shape == (7, 3)
    if kind == "per_element":
        assert np.all(np.abs(np.asarray(d)) <= np.array([1.0, 2.0, 3.0]))


def test_scalar_prior_draw_is_unchanged():
    """The scalar-prior path is the old call, bit for bit."""
    p, k = PRIORS["scalar"], jax.random.PRNGKey(3)
    assert np.array_equal(np.asarray(sample_for_parameter(p, k, 5, (4,))),
                          np.asarray(p.sample(k, shape=(5, 4))))


def test_mismatched_prior_shape_is_refused():
    with pytest.raises(ValueError, match="parameter of shape"):
        sample_for_parameter(PRIORS["per_element"], jax.random.PRNGKey(0), 5, (4,))


@pytest.mark.parametrize("kind", sorted(PRIORS))
def test_nested_live_points_and_map_starts(kind):
    pytest.importorskip("blackjax.ns")
    from ceridwen.optimize import _prior_starts
    from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
    ad = BlackJAXNestedSamplerAdapter({"x": PRIORS[kind]}, num_live=8, verbose=False)
    live = ad._sample_prior({"x": jnp.zeros(3)}, jax.random.PRNGKey(1))
    assert live["x"].shape == (8, 3)
    model = types.SimpleNamespace(theta_init={"x": jnp.zeros(3)}, priors={"x": PRIORS[kind]})
    starts = _prior_starts(model, 4, jax.random.PRNGKey(2), include_init=True)
    assert starts["x"].shape == (5, 3)


def test_detect_bounds_per_element_and_nuts_transform():
    """B2-005: per-element bounds are kept per element (was TypeError); scalar bounds stay
    floats; the NUTS logit map takes them element by element."""
    from ceridwen.fit import _detect_bounds
    from ceridwen.sampler.nuts import _build_transforms
    model = types.SimpleNamespace(priors={"x": PRIORS["per_element"], "s": PRIORS["scalar"]})
    b = _detect_bounds(model)
    assert b["s"] == (-1.0, 1.0) and type(b["s"][0]) is float
    np.testing.assert_array_equal(b["x"][0], [-1.0, -2.0, -3.0])
    lo, hi, isb = _build_transforms(b, {"x": jnp.zeros(3), "s": jnp.zeros(1)})
    np.testing.assert_array_equal(np.asarray(lo), [-1.0, -2.0, -3.0, -1.0])
    np.testing.assert_array_equal(np.asarray(hi), [1.0, 2.0, 3.0, 1.0])


def test_build_adapter_does_not_mutate_sampler_kwargs():
    """B2-004: fitSED popped 'bounds' out of the caller's dict, so a second fit with the same
    dict silently used auto-detected bounds."""
    from ceridwen.fit import _build_adapter
    model = types.SimpleNamespace(priors={"a": Uniform(low=0.0, high=1.0)},
                                  theta_init={"a": jnp.array([0.5])})
    skw = {"bounds": {"a": (0.2, 0.3)}, "num_samples": 10}
    ad1 = _build_adapter("nuts", model, skw, verbose=False)
    assert skw == {"bounds": {"a": (0.2, 0.3)}, "num_samples": 10}
    ad2 = _build_adapter("nuts", model, skw, verbose=False)
    assert ad1.bounds == ad2.bounds == {"a": (0.2, 0.3)}
