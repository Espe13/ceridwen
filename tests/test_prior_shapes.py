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
