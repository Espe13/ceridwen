"""Nested-sampling checks that hold on every supported blackjax (1.6, 1.7, ...).

The bitwise tests in test_nss_slice_kernels.py compare the lane kernel with the stock kernel of
the installed blackjax; they follow its version's details (1.7 draws its directions from a
Cholesky factor and births rejected chains at the contour). These do not:

  * ln Z of a Gaussian in a uniform box against its analytic value;
  * the contour invariant of the lane kernel: a point is born at a contour that an earlier
    batch died at, and lies above it;
  * a lanes checkpoint resumed with lanes reproduces the uninterrupted lanes run bit for bit,
    births included;
  * the lane and stock kernels agree on ln Z statistically, at independent keys.
"""
import glob
import math
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

pytest.importorskip("blackjax.ns")

from ceridwen.sampler import Uniform                                   # noqa: E402
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter       # noqa: E402
from test_ns_checkpoint import _KW, _equal, _run_killed_after, _toy    # noqa: E402

HALF, NDIM, NLIVE = 5.0, 3, 200
# ln Z of exp(-|x|^2 / 2) under a uniform prior on [-HALF, HALF]^NDIM
LNZ_TRUE = NDIM * (0.5 * math.log(2.0 * math.pi) + math.log(math.erf(HALF / math.sqrt(2.0)))
                   - math.log(2.0 * HALF))
# the nested-sampling error of ln Z, sqrt(H / n_live), H the prior-to-posterior KL divergence
# (for this Gaussian H = -LNZ_TRUE - NDIM / 2, to the erf factor)
SIGMA_LNZ = math.sqrt((-LNZ_TRUE - NDIM / 2.0) / NLIVE)


def _run(kernel, seed):
    priors = {"x": Uniform(low=-HALF, high=HALF)}
    ad = BlackJAXNestedSamplerAdapter(priors, num_live=NLIVE, num_delete=40,
                                      num_inner_steps=5 * NDIM, verbose=False,
                                      checkpoint_interval_s=0, slice_kernel=kernel)
    return ad.run(lambda p: -0.5 * jnp.sum(p["x"] ** 2), lambda p: jnp.zeros(()),
                  {"x": jnp.zeros(NDIM)}, jax.random.PRNGKey(seed))


@pytest.fixture(scope="module")
def lanes_runs():
    return [_run("lanes", s) for s in (1, 2, 3)]


def test_log_evidence_matches_the_analytic_value(lanes_runs):
    for res in lanes_runs:
        assert abs(float(res.log_evidence) - LNZ_TRUE) < 4.0 * SIGMA_LNZ


def test_lane_points_are_born_at_an_earlier_contour_and_lie_above_it(lanes_runs):
    for res in lanes_runs:
        logl = np.asarray(res.log_likelihoods)
        birth = np.asarray(res.log_likelihoods_birth)
        reborn = np.isfinite(birth)          # the first live points are born at the prior
        assert reborn.any() and (~reborn).sum() == NLIVE
        assert np.all(logl[reborn] > birth[reborn])
        assert np.all(np.isin(birth[reborn], logl))


def test_lanes_checkpoint_resumes_bitwise(tmp_path, monkeypatch):
    priors, loglike, logprior, theta_init = _toy()
    kw = dict(_KW, slice_kernel="lanes")
    ref = BlackJAXNestedSamplerAdapter(priors, checkpoint_dir=str(tmp_path / "full"),
                                       **kw).run(loglike, logprior, theta_init,
                                                 jax.random.PRNGKey(0))
    _run_killed_after(monkeypatch, 4, checkpoint_dir=str(tmp_path / "killed"), **kw)
    ck_path = glob.glob(os.path.join(tmp_path / "killed", "ns_checkpoint_*.pkl"))[0]
    res = BlackJAXNestedSamplerAdapter(priors, checkpoint_dir=str(tmp_path / "resumed"),
                                       resume_from=ck_path, **kw).run(
        loglike, logprior, theta_init, jax.random.PRNGKey(0))
    for k in theta_init:
        assert _equal(res.samples[k], ref.samples[k]), k
    for name in ("log_likelihoods", "log_likelihoods_birth", "log_weights"):
        assert _equal(getattr(res, name), getattr(ref, name)), name
    assert res.log_evidence == ref.log_evidence
    assert res.n_likelihood_calls == ref.n_likelihood_calls


def test_lane_and_stock_log_evidence_agree_statistically(lanes_runs):
    stock = [float(_run("stock", s).log_evidence) for s in (11, 12, 13)]
    lanes = [float(r.log_evidence) for r in lanes_runs]
    # the difference of two means of three independent runs: sd SIGMA_LNZ sqrt(2 / 3)
    assert abs(np.mean(lanes) - np.mean(stock)) < 4.0 * SIGMA_LNZ * math.sqrt(2.0 / 3.0)
