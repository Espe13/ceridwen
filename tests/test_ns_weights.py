"""nested_log_weights: exact n_live bookkeeping on a toy nested-sampling run
with an analytic evidence, batch deletion, the unsorted live tail that
BlackJAX's finalise appends, permutation invariance and invalid points."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ceridwen.sampler.ns_weights import nested_log_weights, log_evidence_from_weights

S = 0.05                                   # likelihood width; prior U(0, 1)


def _logL(x):
    return -0.5 * (x / S) ** 2


def _analytic_logZ():
    return math.log(S * math.sqrt(math.pi / 2) * math.erf(1 / (S * math.sqrt(2))))


def _toy_run(nlive, num_delete, seed):
    """Exact constrained-prior nested sampling; returns (logL, logL_birth) in
    BlackJAX order: deletion batches (ascending logL inside a batch, as
    top_k(-logL) yields), then the live points in slot order."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, nlive)
    b = np.full(nlive, -np.inf)
    dead_x, dead_b = [], []
    n_iter = int(nlive * (12 + 3 * math.log(1 / S)) / num_delete)
    for _ in range(n_iter):
        kill = np.argsort(_logL(x))[:num_delete]        # lowest logL first
        dead_x.extend(x[kill]); dead_b.extend(b[kill])
        contour = _logL(x[kill]).max()
        x[kill] = rng.uniform(0.0, np.min(x[kill]), num_delete)
        b[kill] = contour
    xs = np.array(dead_x + list(x)); bs = np.array(dead_b + list(b))
    return _logL(xs), bs


@pytest.mark.parametrize("num_delete", [1, 200])
def test_evidence_unbiased_with_batch_deletion_and_live_tail(num_delete):
    nlive = 1000
    diffs = []
    for seed in range(8):
        lL, lB = _toy_run(nlive, num_delete, seed)
        lw = nested_log_weights(lL, lB)
        assert lw.shape == lL.shape and np.isfinite(lw).all()
        diffs.append(log_evidence_from_weights(lw) - _analytic_logZ())
    diffs = np.array(diffs)
    sigma = math.sqrt(2.1 / nlive)                     # sqrt(H / n_live), H ~ 2.1 nats
    assert abs(diffs.mean()) < 3 * sigma / math.sqrt(len(diffs))
    assert diffs.std() < 2.5 * sigma


def test_posterior_mean_and_permutation_invariance():
    lL, lB = _toy_run(500, 40, 3)
    lw = nested_log_weights(lL, lB)
    x = np.sqrt(-2.0 * lL) * S
    w = np.exp(lw - lw.max()); w /= w.sum()
    Ex = float(np.sum(w * x))
    Ex_an = S * S * (1 - math.exp(-1 / (2 * S * S))) / math.exp(_analytic_logZ())
    assert abs(Ex - Ex_an) < 0.1 * Ex_an
    p = np.random.default_rng(1).permutation(lL.size)
    np.testing.assert_allclose(nested_log_weights(lL[p], lB[p]), lw[p])


def test_invalid_points_get_minus_inf():
    lL, lB = _toy_run(200, 10, 0)
    lw = nested_log_weights(lL, lB)
    lL2 = np.concatenate([lL, [-np.inf, np.nan, 0.0]])
    lB2 = np.concatenate([lB, [-np.inf, -np.inf, 0.0]])          # zero-L draw, NaN, logL == birth
    lw2 = nested_log_weights(lL2, lB2)
    assert np.isneginf(lw2[-3:]).all()
    np.testing.assert_allclose(lw2[:-3], lw)
    with pytest.raises(ValueError, match="differ in length"):
        nested_log_weights(lL, lB[:-1])
    with pytest.raises(ValueError, match="no point"):
        nested_log_weights([-np.inf, 1.0], [-np.inf, 1.0])
