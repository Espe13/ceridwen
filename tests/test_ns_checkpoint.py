"""Periodic checkpointing + rescue pickle for the nested-sampling adapter.

BlackJAX-ns has no native checkpoint/resume, so ``BlackJAXNestedSamplerAdapter``
adds:
  * a periodic checkpoint (default every 20 min) that finalises the dead points
    against the live ensemble and dumps a snapshot, so a run killed by the
    scheduler wall-time / a node failure still yields a recoverable posterior;
  * an end-of-run rescue pickle in the same format, so a crash anywhere on the
    post-convergence save path can't discard a multi-hour run (the regression
    that lost job 229509 and the 2026-06-19 FMR / N80-extended runs);
  * ``load_checkpoint`` to read either back.

These tests use a fast 3-D Gaussian toy so they run in well under a second.
"""
import glob
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
from ceridwen.sampler import Uniform


def _toy():
    priors = {k: Uniform(low=-5.0, high=5.0) for k in ("x", "y", "z")}

    def loglike(t):
        v = jnp.array([t["x"][0], t["y"][0], t["z"][0]])
        return -0.5 * jnp.sum(v ** 2)

    def logprior(t):
        return jnp.array(0.0)

    theta_init = {k: jnp.array([0.0]) for k in ("x", "y", "z")}
    return priors, loglike, logprior, theta_init


def test_periodic_checkpoint_and_rescue(tmp_path):
    priors, loglike, logprior, theta_init = _toy()
    # tiny interval -> a partial checkpoint fires on (effectively) every iter.
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False,
        checkpoint_interval_s=1e-6, checkpoint_dir=str(tmp_path))
    res = ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    assert np.isfinite(float(res.log_evidence))

    ckpts = glob.glob(os.path.join(tmp_path, "ns_checkpoint_*.pkl"))
    rescues = glob.glob(os.path.join(tmp_path, "ns_raw_dead_*.pkl"))
    assert ckpts, "no periodic checkpoint written"
    assert rescues, "no end-of-run rescue pickle written"

    # Partial checkpoint is a complete, loadable snapshot.
    d = BlackJAXNestedSamplerAdapter.load_checkpoint(ckpts[0])
    assert d["partial"] is True
    assert d["n_dead"] > 0
    assert set(d["positions"]) == {"x", "y", "z"}
    assert np.isfinite(d["logZ"])
    # Rescue is the same format, flagged converged.
    assert BlackJAXNestedSamplerAdapter.load_checkpoint(rescues[0])["partial"] is False


def test_checkpoint_recovers_a_posterior(tmp_path):
    """A killed run's checkpoint must reconstruct an anesthetic posterior."""
    anesthetic = __import__("anesthetic")
    priors, loglike, logprior, theta_init = _toy()
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False,
        checkpoint_interval_s=1e-6, checkpoint_dir=str(tmp_path))
    ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))

    ck = BlackJAXNestedSamplerAdapter.load_checkpoint(
        glob.glob(os.path.join(tmp_path, "ns_checkpoint_*.pkl"))[0])
    data = np.column_stack([ck["positions"][k].reshape(ck["n_dead"], -1)
                            for k in ("x", "y", "z")])
    ns = anesthetic.NestedSamples(
        data=data, logL=ck["loglikelihood"],
        logL_birth=ck["loglikelihood_birth"], columns=["x", "y", "z"])
    assert np.isfinite(float(ns.logZ()))


def test_checkpoint_disabled_when_no_dir(tmp_path, monkeypatch):
    """No checkpoint dir resolvable -> silently no files (no surprise writes)."""
    monkeypatch.delenv("CERIDWEN_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("CERIDWEN_RESCUE_DIR", raising=False)
    priors, loglike, logprior, theta_init = _toy()
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False,
        checkpoint_interval_s=1e-6, checkpoint_dir=None)
    ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    assert ad._resolve_ckpt_dir() is None
    assert not glob.glob(os.path.join(tmp_path, "*.pkl"))


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        test_periodic_checkpoint_and_rescue(Path(d))
    print("checkpoint tests passed")
