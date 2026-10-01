"""Inner slice kernels of BlackJAXNestedSamplerAdapter: "carry" and "lanes" must be bitwise
equal to the stock blackjax.nss kernel on CPU (same random numbers, same evaluations).

Adapted from the tests of Liu Hao's fork (potatoist314/ceridwen, tests/test_nss_diagnostics.py
at 2d5edc9) to this adapter, which has no iteration callback: the comparisons run the kernels'
``step`` functions directly, and one test compares whole adapter runs.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from blackjax.mcmc.slice import build_kernel, stepping_out          # noqa: E402
from blackjax.ns.base import init_state_strategy                     # noqa: E402
from blackjax.ns.nss import covariance_proposal, slice_constrained_step   # noqa: E402

from ceridwen.sampler.nested import (BlackJAXNestedSamplerAdapter,  # noqa: E402
                                     SLICE_KERNELS, stepping_out_carry)


def _assert_trees_equal(want, got):
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_carry_stepping_out_counts_the_same_calls_as_stock():
    counts = {}
    for interval in (stepping_out, stepping_out_carry):
        calls = []

        def loglike(position):
            calls.append(1)
            return -jnp.sum(position['x'] ** 2)

        def initialize(position, loglikelihood_birth=jnp.nan):
            return init_state_strategy(position, lambda p: jnp.zeros(()),
                                       loglike, loglikelihood_birth)

        move = slice_constrained_step(
            initialize, build_kernel(interval=interval, max_expansions=10, max_shrinkage=100),
            covariance_proposal)
        with jax.disable_jit():
            state = initialize({'x': jnp.array([0.1, 0.2])})
            calls.clear()
            move(jax.random.PRNGKey(37), state, -2., cov=jnp.eye(2))
        counts[interval] = len(calls)
    assert counts[stepping_out_carry] == counts[stepping_out] > 1


def test_unknown_slice_kernel_raises():
    with pytest.raises(ValueError, match="slice_kernel"):
        BlackJAXNestedSamplerAdapter({}, slice_kernel="fast")


def test_resume_config_is_the_same_for_every_kernel():
    """Checkpoints written before the option existed, or with another kernel, resume with any
    kernel: the kernels make the same moves (tests/test_ns_checkpoint.py resumes across them)."""
    theta = {'x': jnp.zeros(2)}
    configs = [BlackJAXNestedSamplerAdapter({}, verbose=False, slice_kernel=k)._run_config(
        theta, 6, 8) for k in SLICE_KERNELS]
    assert all(c == configs[0] for c in configs) and "slice_kernel" not in configs[0]


@pytest.mark.parametrize("kernel", ["carry", "lanes"])
def test_kernel_steps_are_bitwise_equal_to_stock(kernel):
    """A narrow, correlated peak: the stepping-out loops reach their caps early and the
    shrink loop runs many times later."""
    def loglike(p):
        return -50. * jnp.sum((p['x'] - .7) ** 2) - 40. * jnp.prod(p['x'])

    def logprior(p):
        return jnp.where(jnp.all(jnp.abs(p['x']) < 5.), 0., -jnp.inf)

    num_inner_steps, num_delete = 6, 8
    stock = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                         num_inner_steps=num_inner_steps, num_delete=num_delete)
    other = BlackJAXNestedSamplerAdapter(
        {}, verbose=False, slice_kernel=kernel)._build_nested_sampler(
            loglike, logprior, num_inner_steps, num_delete)
    keys = jax.random.split(jax.random.key(20260930), 2)
    x0 = jax.random.uniform(keys[0], (40, 3), minval=-5., maxval=5.)
    state_s = jax.jit(stock.init)({'x': x0})
    state_o = jax.jit(other.init)({'x': x0})
    _assert_trees_equal(state_s, state_o)
    stock_step, other_step = jax.jit(stock.step), jax.jit(other.step)
    expansions = []
    for key in jax.random.split(keys[1], 25):
        want, got = stock_step(key, state_s), other_step(key, state_s)
        _assert_trees_equal(want, got)
        expansions.append(np.asarray(want[1].update_info.num_expansions))
        state_s = want[0]
    expansions = np.concatenate([e.ravel() for e in expansions])
    assert expansions.max() > 0          # the stepping-out loops did run


def test_lane_kernel_matches_stock_when_shrinkage_runs_out():
    """Two likelihood levels: once every live point is on the upper level, no candidate is
    above the contour and each slice uses all of max_shrinkage."""
    inside = lambda p: jnp.all(jnp.abs(p['x']) < 1.2) & (jnp.abs(p['y']) < 1.2)  # noqa: E731
    loglike = lambda p: jnp.where(inside(p), 1., 0.)  # noqa: E731
    logprior = lambda p: jnp.where(  # noqa: E731
        jnp.all(jnp.abs(p['x']) < 2.) & (jnp.abs(p['y']) < 2.), 0., -jnp.inf)
    stock = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                         num_inner_steps=4, num_delete=6)
    lanes = BlackJAXNestedSamplerAdapter(
        {}, verbose=False, slice_kernel="lanes")._build_nested_sampler(loglike, logprior, 4, 6)
    keys = jax.random.split(jax.random.key(5), 3)
    state = jax.jit(stock.init)({
        'x': jax.random.uniform(keys[0], (30, 2), minval=-2., maxval=2.),
        'y': jax.random.uniform(keys[1], (30,), minval=-2., maxval=2.)})
    stock_step, lane_step = jax.jit(stock.step), jax.jit(lanes.step)
    accepted, shrinks = [], []
    for key in jax.random.split(keys[2], 12):
        want, got = stock_step(key, state), lane_step(key, state)
        _assert_trees_equal(want, got)
        state = want[0]
        accepted.append(np.asarray(want[1].update_info.is_accepted))
        shrinks.append(np.asarray(want[1].update_info.num_shrink))
    accepted, shrinks = np.concatenate(accepted), np.concatenate(shrinks)
    assert accepted.any() and not accepted.all()
    assert shrinks[~accepted].min() == shrinks.max() == 100


@pytest.mark.parametrize("prior", ["flat", "normal"])
def test_lane_kernel_skips_and_stores_evaluations_bitwise(prior):
    """A flat prior: every batch slot saved comes from a stored look-ahead result.  A normal
    prior: candidates below the prior slice level also skip the batch."""
    batches = []

    def loglike(p):
        jax.debug.callback(lambda: batches.append(1))
        return -8. * jnp.sum((p['x'] - .5) ** 2) - 6. * p['x'][0] * p['x'][1]

    def logprior(p):
        return jnp.zeros(()) if prior == "flat" else -.5 * jnp.sum(p['x'] ** 2)

    num_delete, num_inner_steps = 10, 8
    stock = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                         num_inner_steps=num_inner_steps, num_delete=num_delete)
    lanes = BlackJAXNestedSamplerAdapter(
        {}, verbose=False, slice_kernel="lanes")._build_nested_sampler(
            loglike, logprior, num_inner_steps, num_delete)
    keys = jax.random.split(jax.random.key(11), 2)
    state = jax.jit(stock.init)({'x': jax.random.normal(keys[0], (60, 3))})
    stock_step, lane_step = jax.jit(stock.step), jax.jit(lanes.step)
    unbatched = 0
    for key in jax.random.split(keys[1], 10):
        want = stock_step(key, state)
        batches.clear()
        got = jax.block_until_ready(lane_step(key, state))
        jax.effects_barrier()     # the callback runs once per batch
        _assert_trees_equal(want, got)
        info = want[1].update_info
        per_lane = np.sum(np.asarray(info.num_expansions) + np.asarray(info.num_shrink) + 2,
                          axis=1)
        # one batch per candidate of the slowest lane, less the batches saved
        unbatched += per_lane.max() - len(batches)
        state = want[0]
    assert unbatched > 0


def test_adapter_runs_are_bitwise_equal_across_kernels():
    from ceridwen.sampler import Uniform

    def run(kernel):
        adapter = BlackJAXNestedSamplerAdapter(
            {'x': Uniform(low=-5., high=5.)}, num_live=40, num_delete=8, num_inner_steps=6,
            verbose=False, checkpoint_interval_s=0, slice_kernel=kernel)
        return adapter.run(lambda p: -50. * jnp.sum((p['x'] - .7) ** 2),
                           lambda p: jnp.zeros(()), {'x': jnp.zeros(3)},
                           jax.random.PRNGKey(7))

    results = {k: run(k) for k in SLICE_KERNELS}
    ref = results["stock"]
    for kernel in ("carry", "lanes"):
        res = results[kernel]
        for name in ("log_likelihoods", "log_likelihoods_birth", "log_weights"):
            np.testing.assert_array_equal(np.asarray(getattr(res, name)),
                                          np.asarray(getattr(ref, name)))
        np.testing.assert_array_equal(np.asarray(res.samples['x']),
                                      np.asarray(ref.samples['x']))
        assert res.log_evidence == ref.log_evidence
        assert res.n_likelihood_calls == ref.n_likelihood_calls


def test_log_evidence_err_is_reproducible_and_leaves_global_rng_alone():
    """anesthetic draws the ln Z error from numpy's global RNG; the adapter seeds it from
    rng_key (and restores it), so the same key gives the same error."""
    pytest.importorskip("anesthetic")
    from ceridwen.sampler import Uniform

    def run():
        adapter = BlackJAXNestedSamplerAdapter(
            {'x': Uniform(low=-5., high=5.)}, num_live=30, num_delete=6, num_inner_steps=4,
            verbose=False, checkpoint_interval_s=0)
        return adapter.run(lambda p: -.5 * jnp.sum(p['x'] ** 2), lambda p: jnp.zeros(()),
                           {'x': jnp.zeros(2)}, jax.random.PRNGKey(3))

    np.random.seed(1)
    a = run()
    after_a = np.random.rand()
    np.random.seed(2)
    b = run()
    np.random.seed(1)
    expected = np.random.rand()
    assert np.isfinite(a.log_evidence_err) and a.log_evidence_err > 0
    assert a.log_evidence_err == b.log_evidence_err
    assert after_a == expected          # the global RNG continued as if the run never drew


def test_result_provenance_records_an_opt_in_kernel_only():
    """The result file's sampler settings and the fit log name the kernel when it is not
    "stock"; for "stock" they are unchanged, so earlier files read the same."""
    from ceridwen.fit import adapter_settings
    stock = adapter_settings(BlackJAXNestedSamplerAdapter({}, verbose=False))
    lanes = adapter_settings(BlackJAXNestedSamplerAdapter({}, verbose=False,
                                                          slice_kernel="lanes"))
    assert "slice_kernel" not in stock
    assert lanes.pop("slice_kernel") == "lanes" and lanes == stock
