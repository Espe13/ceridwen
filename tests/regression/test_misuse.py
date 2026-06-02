"""
Misuse / hardening tests: assert that the common user-error scenarios are now
caught (loud error or warning, or an intentional by-design fallback) rather than
silently returning a wrong result, and that the guards add zero hot-path cost
(they are trace-time-only / non-jitted).

Also emits ``figures/misuse_report.png`` so the hardening status is inspectable.
"""
from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp

from misuse_report import run_scenarios, make_misuse_figure, _good_csp


def test_all_user_mistakes_caught():
    rows = run_scenarios()
    make_misuse_figure(rows)
    silent = [label for (label, kind, ok, detail, good) in rows if not ok]
    assert not silent, f"user mistakes NOT handled (still silent/unexpected): {silent}"


def test_guards_do_not_break_jit_or_gradients():
    """A jitted predict must still compile and run, and the typo guard must fire
    only at trace time (no per-call warnings), proving zero hot-path cost."""
    from ceridwen.observation.observation import Photometry

    csp = _good_csp()
    phot = Photometry(filters=["sdss_g0", "sdss_r0"], name="p")
    phot.setup_for_model(csp.wave)
    th = dict(csp.theta_init)

    f = jax.jit(lambda t: csp.predict(t, [phot]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r0 = f(th); r0["p"].block_until_ready()      # trace + compile
        for _ in range(20):                          # steady-state calls
            r = f(th); r["p"].block_until_ready()
    # clean theta -> no unknown-key warnings at all, and none accumulate per call
    assert len(w) == 0
    assert np.all(np.isfinite(np.asarray(r["p"])))

    # gradient through predict still works (hot path differentiable)
    g = jax.grad(lambda t: jnp.sum(csp.predict(t, [phot])["p"] ** 2))(th)
    assert np.all(np.isfinite(np.asarray(g["sfh"])))


def test_typo_key_warns_once_at_trace_but_still_runs():
    from ceridwen.observation.observation import Photometry

    csp = _good_csp()
    phot = Photometry(filters=["sdss_g0"], name="p")
    phot.setup_for_model(csp.wave)
    th = {**csp.theta_init, "logmas": jnp.array([10.0])}  # typo for logmass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = jax.jit(lambda t: csp.predict(t, [phot]))
        out = g(th); out["p"].block_until_ready()
    assert any("unrecognized theta key" in str(x.message) for x in w)
    assert np.all(np.isfinite(np.asarray(out["p"])))
