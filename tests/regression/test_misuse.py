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


def _weight_csp(zh_const, sfh, sfh_interp):
    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.csp.csp import CSPBasis
    import pathlib
    ssp = SSPData.load(str(pathlib.Path(__file__).resolve().parent.parent.parent
                           / "ceridwen" / "data" / "test_data" / "ssp_data.h5"))
    T = 13.8
    lb = T - jnp.linspace(1e-2, T, 10)
    Zc = -1.85
    theta = {"lookback_time": lb, "sfh": sfh}
    theta["Z" if zh_const else "zh"] = (jnp.array([Zc]) if zh_const
                                        else jnp.full(10, Zc))
    csp = CSPBasis(ssp, theta=theta, tuniv=T, zh_const=zh_const,
                   add_dust=False, add_diffuse_dust=False, add_dust_emission=False,
                   add_neb=False, add_igm=False, verbose=False, sfh_interp=sfh_interp)
    return csp


def test_weight_calcs_share_metallicity_and_sfr_units():
    """All four calculate_ssp_weights_* must consume metallicity and SFR in the
    SAME units. With a constant metallicity (zh = full(Z)) and the same SFH,
    const-zh and var-zh must therefore yield identical SSP-weight matrices, in
    both the linear and step SFH schemes."""
    T = 13.8
    lb = T - jnp.linspace(1e-2, T, 10)
    sfh = jnp.exp(-0.5 * ((lb - 0.05) / 0.03) ** 2) + 0.7 * jnp.exp(-0.5 * ((lb - 11.) / 0.8) ** 2)

    for sfh_interp in ("linear", "step"):
        cc = _weight_csp(True, sfh, sfh_interp)
        cv = _weight_csp(False, sfh, sfh_interp)
        if sfh_interp == "linear":
            wc = cc.calculate_ssp_weights_const_zh(dict(cc.theta_init))
            wv = cv.calculate_ssp_weights_var_zh(dict(cv.theta_init))
        else:
            wc = cc.calculate_ssp_weights_const_zh_step(dict(cc.theta_init))
            wv = cv.calculate_ssp_weights_var_zh_step(dict(cv.theta_init))
        wc = np.asarray(wc); wv = np.asarray(wv)
        assert np.all(np.isfinite(wc)) and np.all(np.isfinite(wv))
        np.testing.assert_allclose(
            wv, wc, atol=1e-10, rtol=1e-7,
            err_msg=f"const-zh and var-zh disagree for constant Z ({sfh_interp}) "
                    f"-> metallicity/SFR units are not consistent",
        )


def test_var_zh_zero_sfr_is_finite():
    """var-zh with an exactly-zero SFR node must produce finite weights (the
    unified 1e-30 SFR floor; previously NaN via the tiny_logt mis-floor)."""
    T = 13.8
    lb = T - jnp.linspace(1e-2, T, 10)
    sfh = (jnp.exp(-0.5 * ((lb - 0.05) / 0.03) ** 2)
           + 0.7 * jnp.exp(-0.5 * ((lb - 11.) / 0.8) ** 2)).at[3].set(0.0)
    for sfh_interp in ("linear", "step"):
        cv = _weight_csp(False, sfh, sfh_interp)
        w = (cv.calculate_ssp_weights_var_zh(dict(cv.theta_init)) if sfh_interp == "linear"
             else cv.calculate_ssp_weights_var_zh_step(dict(cv.theta_init)))
        assert np.all(np.isfinite(np.asarray(w))), f"var-zh NaN with zero SFR ({sfh_interp})"


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
