"""Inner slice kernels of BlackJAXNestedSamplerAdapter. "lanes" (the default) is tested on its
own: its steps keep every replacement above the contour and born at it, use all of
max_shrinkage when no candidate is above it, save batch evaluations, and reproduce bitwise at
the same key. Its agreement with "stock" is not a requirement: on Apple-silicon CPUs the two are
bitwise equal, on x86 they differ in the last bits (ln Z, the analytic value and the contour
invariant of whole runs are in tests/test_ns_version_independent.py). "carry" is still bitwise
the stock kernel on CPU.

Adapted from tests by Liu Hao (@potatoist314) to this adapter, which has no iteration callback: the comparisons run the kernels'
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


def _replaced(state, num_delete):
    """Indices of the particles a step replaces (the num_delete lowest ln L), as blackjax's
    base.delete_fn picks them."""
    return jax.lax.top_k(-state.particles.loglikelihood, num_delete)[1]


def _check_lane_step(state, out, num_delete, strict=True):
    """A lane step's output: the replaced particles are born at the contour (the highest
    deleted ln L) and lie above it; the others are untouched."""
    new, info = out
    idx = np.asarray(_replaced(state, num_delete))
    contour = float(np.max(np.asarray(info.particles.loglikelihood)))
    logl = np.asarray(new.particles.loglikelihood)
    birth = np.asarray(new.particles.loglikelihood_birth)
    np.testing.assert_array_equal(birth[idx], contour)
    assert np.all(logl[idx] > contour) if strict else np.all(logl[idx] >= contour)
    keep = np.setdiff1d(np.arange(logl.size), idx)
    np.testing.assert_array_equal(logl[keep], np.asarray(state.particles.loglikelihood)[keep])


def test_carry_steps_are_bitwise_equal_to_stock():
    """"carry": a narrow, correlated peak: the stepping-out loops reach their caps early and
    the shrink loop runs many times later."""
    kernel = "carry"
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


def test_lane_steps_on_a_narrow_peak_keep_the_contour_and_reproduce():
    """The same narrow, correlated peak with "lanes": every step keeps the contour invariant,
    the stepping-out loops run, and the same key gives the same step bit for bit."""
    def loglike(p):
        return -50. * jnp.sum((p['x'] - .7) ** 2) - 40. * jnp.prod(p['x'])

    def logprior(p):
        return jnp.where(jnp.all(jnp.abs(p['x']) < 5.), 0., -jnp.inf)

    num_delete = 8
    lanes = BlackJAXNestedSamplerAdapter(
        {}, verbose=False, slice_kernel="lanes")._build_nested_sampler(
            loglike, logprior, 6, num_delete)
    keys = jax.random.split(jax.random.key(20260930), 2)
    state = jax.jit(lanes.init)({'x': jax.random.uniform(keys[0], (40, 3), minval=-5.,
                                                            maxval=5.)})
    step = jax.jit(lanes.step)
    expansions = []
    for key in jax.random.split(keys[1], 25):
        out = step(key, state)
        _assert_trees_equal(out, step(key, state))
        _check_lane_step(state, out, num_delete)
        expansions.append(np.asarray(out[1].update_info.num_expansions))
        state = out[0]
    assert np.concatenate([e.ravel() for e in expansions]).max() > 0


def test_lane_kernel_when_shrinkage_runs_out():
    """Two likelihood levels: once every live point is on the upper level, no candidate is
    above the contour and each slice uses all of max_shrinkage. Chains whose moves were all
    rejected are born at the contour (blackjax 1.7's rule) at their start particle's level."""
    inside = lambda p: jnp.all(jnp.abs(p['x']) < 1.2) & (jnp.abs(p['y']) < 1.2)  # noqa: E731
    loglike = lambda p: jnp.where(inside(p), 1., 0.)  # noqa: E731
    logprior = lambda p: jnp.where(  # noqa: E731
        jnp.all(jnp.abs(p['x']) < 2.) & (jnp.abs(p['y']) < 2.), 0., -jnp.inf)
    lanes = BlackJAXNestedSamplerAdapter(
        {}, verbose=False, slice_kernel="lanes")._build_nested_sampler(loglike, logprior, 4, 6)
    keys = jax.random.split(jax.random.key(5), 3)
    state = jax.jit(lanes.init)({
        'x': jax.random.uniform(keys[0], (30, 2), minval=-2., maxval=2.),
        'y': jax.random.uniform(keys[1], (30,), minval=-2., maxval=2.)})
    lane_step = jax.jit(lanes.step)
    accepted, shrinks = [], []
    for key in jax.random.split(keys[2], 12):
        got = lane_step(key, state)
        _check_lane_step(state, got, 6, strict=False)
        state = got[0]
        accepted.append(np.asarray(got[1].update_info.is_accepted))
        shrinks.append(np.asarray(got[1].update_info.num_shrink))
    accepted, shrinks = np.concatenate(accepted), np.concatenate(shrinks)
    assert accepted.any() and not accepted.all()
    assert shrinks[~accepted].min() == shrinks.max() == 100


@pytest.mark.parametrize("prior", ["flat", "normal"])
def test_lane_kernel_skips_and_stores_evaluations(prior):
    """A flat prior: every batch slot saved comes from a stored look-ahead result.  A normal
    prior: candidates below the prior slice level also skip the batch."""
    batches = []

    def loglike(p):
        jax.debug.callback(lambda: batches.append(1))
        return -8. * jnp.sum((p['x'] - .5) ** 2) - 6. * p['x'][0] * p['x'][1]

    def logprior(p):
        return jnp.zeros(()) if prior == "flat" else -.5 * jnp.sum(p['x'] ** 2)

    num_delete, num_inner_steps = 10, 8
    lanes = BlackJAXNestedSamplerAdapter(
        {}, verbose=False, slice_kernel="lanes")._build_nested_sampler(
            loglike, logprior, num_inner_steps, num_delete)
    keys = jax.random.split(jax.random.key(11), 2)
    state = jax.jit(lanes.init)({'x': jax.random.normal(keys[0], (60, 3))})
    lane_step = jax.jit(lanes.step)
    unbatched = 0
    for key in jax.random.split(keys[1], 10):
        batches.clear()
        got = jax.block_until_ready(lane_step(key, state))
        jax.effects_barrier()     # the callback runs once per batch
        _check_lane_step(state, got, num_delete)
        info = got[1].update_info
        per_lane = np.sum(np.asarray(info.num_expansions) + np.asarray(info.num_shrink) + 2,
                          axis=1)
        # one batch per candidate of the slowest lane, less the batches saved
        unbatched += per_lane.max() - len(batches)
        state = got[0]
    assert unbatched > 0


def test_adapter_runs_reproduce_and_carry_equals_stock():
    """A whole adapter run: "lanes" gives the same run twice at the same key, bit for bit;
    "carry" is bitwise the stock run on CPU."""
    from ceridwen.sampler import Uniform

    def run(kernel):
        adapter = BlackJAXNestedSamplerAdapter(
            {'x': Uniform(low=-5., high=5.)}, num_live=40, num_delete=8, num_inner_steps=6,
            verbose=False, checkpoint_interval_s=0, slice_kernel=kernel)
        return adapter.run(lambda p: -50. * jnp.sum((p['x'] - .7) ** 2),
                           lambda p: jnp.zeros(()), {'x': jnp.zeros(3)},
                           jax.random.PRNGKey(7))

    for res, ref in ((run("lanes"), run("lanes")), (run("carry"), run("stock"))):
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


def test_result_provenance_records_a_non_stock_kernel_only():
    """The result file's sampler settings name the kernel when it is not "stock"; for "stock"
    they have no slice_kernel entry, so files written before the option existed (all stock)
    read the same.  The default, "lanes", is therefore recorded."""
    from ceridwen.fit import adapter_settings
    stock = adapter_settings(BlackJAXNestedSamplerAdapter({}, verbose=False,
                                                          slice_kernel="stock"))
    default = adapter_settings(BlackJAXNestedSamplerAdapter({}, verbose=False))
    assert "slice_kernel" not in stock
    assert default.pop("slice_kernel") == "lanes" and default == stock


def test_default_slice_kernel_is_lanes_everywhere():
    """The adapter and fitSED(sampler="nested") (the Quick start's path) both run "lanes"
    unless told otherwise; an explicit "stock" is still honoured."""
    import types
    from ceridwen.fit import _build_adapter
    assert BlackJAXNestedSamplerAdapter({}, verbose=False).slice_kernel == "lanes"
    model = types.SimpleNamespace(priors={})
    for name in ("nested", "ns", "nss"):
        assert _build_adapter(name, model, {}, verbose=False).slice_kernel == "lanes"
    assert _build_adapter("nested", model, {"slice_kernel": "stock"},
                          verbose=False).slice_kernel == "stock"
