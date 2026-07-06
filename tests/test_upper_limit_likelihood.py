"""
Tests for the upper-limit-aware diagonal Gaussian likelihood added in
``ceridwen.likelihood.likelihood``.

The kernel under test is
:func:`ceridwen.likelihood.lnlike_diag_gaussian_with_upper_limits`.  We
check four properties:

1. **Backward compatibility.**  With all ``is_upper_limit = False`` the
   kernel reduces bit-for-bit to :func:`lnlike_diag_gaussian` (and to
   the standard ``numpyro.distributions.Normal.log_prob`` sum).

2. **Below-the-limit no-penalty.**  When a flagged datum has
   :math:`\\mu \\le y`, its contribution to the log-likelihood is just
   the log-normalisation ``-log_det`` — no chi-squared term.

3. **Above-the-limit penalty.**  When a flagged datum has
   :math:`\\mu > y`, the contribution matches
   :math:`-\\tfrac{1}{2}\\chi^2 - \\log\\!\\det`, i.e. the standard
   Gaussian penalty.

4. **JIT + grad smoke test.**  The kernel composes with ``jax.jit`` and
   ``jax.grad`` (gradient w.r.t. ``mu``) without raising.

Pure unit test: no FSPS, no CSP, no fitting machinery.  Runs in
roughly a few hundred ms on a laptop CPU.
"""
from __future__ import annotations

import os
# Stay on CPU for this unit test so it runs on any laptop without GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen.likelihood import (
    lnlike_diag_gaussian,
    lnlike_diag_gaussian_with_upper_limits,
    DiagonalGaussianLikelihood,
    DiagonalGaussianLikelihoodWithUpperLimits,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _noise_to_invvar_logdet(sigma):
    """Translate sigma -> (inv_var, log_det) the same way the
    DiagonalNoiseModel does.  Keeps the test independent of that class
    so a noise-model regression cannot mask a kernel bug."""
    var      = sigma ** 2
    inv_var  = 1.0 / var
    log_det  = 0.5 * jnp.log(2.0 * jnp.pi * var)
    return inv_var, log_det


# --------------------------------------------------------------------------- #
# 1. Backward compatibility                                                    #
# --------------------------------------------------------------------------- #

def test_no_upper_limits_matches_standard_gaussian():
    """All-detection input must give identical lnL to the legacy kernel."""
    rng = np.random.default_rng(2026)
    n   = 12
    y      = jnp.asarray(rng.standard_normal(n))
    mu     = jnp.asarray(rng.standard_normal(n))
    sigma  = jnp.asarray(0.1 + rng.random(n))
    mask   = jnp.ones(n, dtype=bool)
    is_ul  = jnp.zeros(n, dtype=bool)

    inv_var, log_det = _noise_to_invvar_logdet(sigma)

    lnl_old, _ = lnlike_diag_gaussian(y, mu, inv_var, log_det, mask)
    lnl_new, _ = lnlike_diag_gaussian_with_upper_limits(
        y, mu, inv_var, log_det, mask, is_ul,
    )

    assert jnp.allclose(lnl_old, lnl_new, atol=1e-12, rtol=0.0), (
        f"Backward compat broken: lnl_old={lnl_old}, lnl_new={lnl_new}"
    )


def test_class_no_upper_limits_matches_legacy_class():
    """The new class with is_upper_limit=None must equal the legacy class."""
    rng    = np.random.default_rng(7)
    n      = 8
    y      = jnp.asarray(rng.standard_normal(n))
    mu     = jnp.asarray(rng.standard_normal(n))
    sigma  = jnp.asarray(0.2 + rng.random(n))
    mask   = jnp.ones(n, dtype=bool)

    lh_old = DiagonalGaussianLikelihood()
    lh_new = DiagonalGaussianLikelihoodWithUpperLimits()

    lnl_old, _ = lh_old(y, mu, sigma, mask)
    lnl_new, _ = lh_new(y, mu, sigma, mask, is_upper_limit=None)

    assert jnp.allclose(lnl_old, lnl_new, atol=1e-12, rtol=0.0), (
        f"Class drop-in compat broken: {float(lnl_old)} vs {float(lnl_new)}"
    )


# --------------------------------------------------------------------------- #
# 2. Below-the-limit: no chi-squared penalty                                   #
# --------------------------------------------------------------------------- #

def test_upper_limit_no_penalty_when_model_below_limit():
    """For an UL band with mu < y, contribution must be just -log_det."""
    n      = 4
    y      = jnp.array([10.0, 10.0, 10.0, 10.0])
    mu     = jnp.array([ 5.0,  3.0,  1.0,  9.9])   # all below the UL
    sigma  = jnp.array([ 2.0,  2.0,  2.0,  2.0])
    mask   = jnp.ones(n, dtype=bool)
    is_ul  = jnp.ones(n, dtype=bool)

    inv_var, log_det = _noise_to_invvar_logdet(sigma)

    lnl, aux = lnlike_diag_gaussian_with_upper_limits(
        y, mu, inv_var, log_det, mask, is_ul,
    )

    # Expected: sum over n of -log_det_i.
    expected = -jnp.sum(log_det)
    assert jnp.allclose(lnl, expected, atol=1e-12, rtol=0.0), (
        f"UL below-limit penalty leaked: lnl={float(lnl)}, "
        f"expected={float(expected)}"
    )

    # All chi^2 contributions to lnl_pointwise should be zero on this branch.
    chi2_contrib = -2.0 * (aux.lnl_pointwise + log_det)   # = chi^2_used per datum
    assert jnp.allclose(chi2_contrib, 0.0, atol=1e-12, rtol=0.0), (
        f"Per-datum chi^2 should be 0 for all UL below limit; got "
        f"{np.asarray(chi2_contrib)}"
    )


# --------------------------------------------------------------------------- #
# 3. Above-the-limit: full Gaussian penalty                                    #
# --------------------------------------------------------------------------- #

def test_upper_limit_full_penalty_when_model_above_limit():
    """For an UL band with mu > y, contribution must match -0.5 chi^2 - log_det."""
    n      = 3
    y      = jnp.array([1.0, 1.0, 1.0])
    mu     = jnp.array([2.0, 3.0, 5.0])              # all above the UL
    sigma  = jnp.array([1.0, 1.0, 1.0])
    mask   = jnp.ones(n, dtype=bool)
    is_ul  = jnp.ones(n, dtype=bool)

    inv_var, log_det = _noise_to_invvar_logdet(sigma)

    lnl, _ = lnlike_diag_gaussian_with_upper_limits(
        y, mu, inv_var, log_det, mask, is_ul,
    )

    chi      = (y - mu) * jnp.sqrt(inv_var)
    expected = jnp.sum(-0.5 * chi ** 2 - log_det)
    assert jnp.allclose(lnl, expected, atol=1e-12, rtol=0.0), (
        f"UL above-limit penalty wrong: lnl={float(lnl)}, "
        f"expected={float(expected)}"
    )


def test_mixed_detections_and_upper_limits():
    """Two detections + two upper-limit bands.  Hand-computed expectation."""
    y      = jnp.array([ 5.0,  8.0, 10.0, 10.0])
    mu     = jnp.array([ 5.5,  7.5,  3.0, 15.0])
    sigma  = jnp.array([ 0.5,  0.5,  4.0,  5.0])
    mask   = jnp.array([True, True, True, True])
    is_ul  = jnp.array([False, False, True, True])

    inv_var, log_det = _noise_to_invvar_logdet(sigma)

    lnl, _ = lnlike_diag_gaussian_with_upper_limits(
        y, mu, inv_var, log_det, mask, is_ul,
    )

    # By hand:
    #  i=0: detection.  chi = (5 - 5.5)/0.5 = -1.0  -> -0.5*1 - log_det[0]
    #  i=1: detection.  chi = (8 - 7.5)/0.5 = +1.0  -> -0.5*1 - log_det[1]
    #  i=2: UL, mu=3 < y=10.  No penalty, just -log_det[2].
    #  i=3: UL, mu=15 > y=10.  chi = (10-15)/5 = -1.  -> -0.5*1 - log_det[3].
    expected_pointwise = jnp.array([
        -0.5 * 1.0 - log_det[0],
        -0.5 * 1.0 - log_det[1],
                 - log_det[2],
        -0.5 * 1.0 - log_det[3],
    ])
    expected = jnp.sum(expected_pointwise)

    assert jnp.allclose(lnl, expected, atol=1e-12, rtol=0.0), (
        f"Mixed branch wrong: lnl={float(lnl):.8f}, "
        f"expected={float(expected):.8f}"
    )


# --------------------------------------------------------------------------- #
# 4. JIT + grad smoke test                                                     #
# --------------------------------------------------------------------------- #

def test_kernel_jits_and_differentiates():
    """End-to-end: jax.jit + jax.grad through the kernel must succeed."""
    y      = jnp.array([1.0, 1.0, 1.0])
    sigma  = jnp.array([0.5, 0.5, 0.5])
    mask   = jnp.array([True, True, True])
    is_ul  = jnp.array([True, False, True])
    inv_var, log_det = _noise_to_invvar_logdet(sigma)

    def loss(mu):
        lnl, _ = lnlike_diag_gaussian_with_upper_limits(
            y, mu, inv_var, log_det, mask, is_ul,
        )
        return -lnl

    g_jit = jax.jit(jax.grad(loss))
    grad_below = g_jit(jnp.array([0.0, 0.0, 0.0]))     # UL bands safely below
    grad_above = g_jit(jnp.array([5.0, 0.0, 5.0]))     # UL bands above limit

    # Gradient w.r.t. mu of the *negative* log-likelihood:
    #   detection (i=1): d(-lnl)/dmu = -(y - mu)/sigma^2
    #   UL below (mu<y): contribution to -lnl is +log_det only -> grad = 0
    #   UL above (mu>y): same as detection.
    expected_below = jnp.array([0.0,                  # UL, mu=0 < y=1: zero
                                -(1.0 - 0.0) / 0.25,  # detection
                                0.0])                 # UL, mu=0 < y=1
    assert jnp.allclose(grad_below, expected_below, atol=1e-10), (
        f"Below-limit grad wrong: got {grad_below}, expected {expected_below}"
    )

    expected_above = jnp.array([-(1.0 - 5.0) / 0.25,  # UL above: same as Gaussian
                                -(1.0 - 0.0) / 0.25,
                                -(1.0 - 5.0) / 0.25])
    assert jnp.allclose(grad_above, expected_above, atol=1e-10), (
        f"Above-limit grad wrong: got {grad_above}, expected {expected_above}"
    )


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        test_no_upper_limits_matches_standard_gaussian,
        test_class_no_upper_limits_matches_legacy_class,
        test_upper_limit_no_penalty_when_model_below_limit,
        test_upper_limit_full_penalty_when_model_above_limit,
        test_mixed_detections_and_upper_limits,
        test_kernel_jits_and_differentiates,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
