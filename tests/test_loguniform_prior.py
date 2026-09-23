"""``LogUniform(mini, maxi)``: the log-uniform (reciprocal) prior, as Prospector's.

a. logpdf / cdf / ppf equal ``scipy.stats.reciprocal`` and Prospector's ``LogUniform``
   (live, by path import when a clone is found; always against a golden table);
b. KS, 1e5 draws, vs the recipe's ``Uniform`` in log10 + transform and vs the analytic CDF;
c. unit_transform / inverse_unit_transform round trip; construction-time validation;
   serialize round trip;
d. jit / vmap / grad of logpdf; -inf and a finite (zero) gradient outside the support;
e. ``fit._detect_bounds`` gives (mini, maxi) on a SedModel (test SSP grid; skips without);
f. nested sampling on a toy with an analytic evidence and posterior;
g. NUTS (logit map from the auto-detected bounds) on a toy: prior-only and with a likelihood;
h. gradient of the fitSED log-posterior wrt every free parameter, with a LogUniform on
   ``diffuse_tau_kc``, vs central finite differences at the centre and near both bounds.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pathlib
import sys
import warnings

import numpy as np
import scipy.stats
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from ceridwen.priors import LogUniform, Uniform            # noqa: E402

# ---------------------------------------------------------------------------
# Golden table.  Produced once from Prospector bd-j/prospector commit a78d153,
# prospect/models/priors.py loaded by path (numpy + scipy only):
#     p = LogUniform(mini=a, maxi=b)                 # scipy.stats.reciprocal(a, b)
#     p(np.array(x)).tolist()                        # ln-prior
#     p.inverse_unit_transform(np.array(x)).tolist() # CDF
#     p.unit_transform(np.array(u)).tolist()         # ppf
# with x = [1.5 a, sqrt(a b), 0.9 b], u = [0.1, 0.5, 0.9] (scipy 1.x, float64).
# ---------------------------------------------------------------------------
GOLDEN = {
    (1e-3, 10.0): dict(
        x=[0.0015, 0.1, 9.0],
        lnp=[4.281963364506126, 0.08225828662619909, -4.417551383704065],
        cdf=[0.04402281476392033, 0.5, 0.9885606273598312],
        ppf=[0.00251188643150958, 0.10000000000000006, 3.981071705534978]),
    (0.5, 2.0): dict(
        x=[0.75, 1.0, 1.8],
        lnp=[-0.03895218752649999, -0.326634259978281, -0.9144209248804],
        cdf=[0.2924812503605781, 0.5, 0.9239984532774751],
        ppf=[0.5743491774985175, 1.0, 1.7411011265922482]),
}
U_GOLDEN = [0.1, 0.5, 0.9]


def _prospector_priors():
    """Prospector's priors.py (numpy + scipy only) from a clone at ``$PROSPECTOR_DIR``."""
    d = os.environ.get("PROSPECTOR_DIR", "")
    path = pathlib.Path(d) / "prospect" / "models" / "priors.py"
    if not (d and path.is_file()):
        return None
    spec = importlib.util.spec_from_file_location("prospector_priors_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- a. values -------------------------------------------------------------
@pytest.mark.parametrize("ab", list(GOLDEN))
def test_values_match_golden_and_scipy(ab):
    a, b = ab
    g = GOLDEN[ab]
    lu = LogUniform(mini=a, maxi=b)
    x, u = np.array(g["x"]), np.array(U_GOLDEN)
    np.testing.assert_allclose(np.asarray(lu.logpdf(x)), g["lnp"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(lu.inverse_unit_transform(x)), g["cdf"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(lu.unit_transform(u)), g["ppf"], rtol=1e-14, atol=0)
    ref = scipy.stats.reciprocal(a, b)
    xs = np.geomspace(a, b, 501)
    us = np.linspace(0.0, 1.0, 501)
    np.testing.assert_allclose(np.asarray(lu.logpdf(xs)), ref.logpdf(xs), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(lu.inverse_unit_transform(xs)), ref.cdf(xs), rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(lu.unit_transform(us)), ref.ppf(us), rtol=1e-13, atol=0)
    # the TFP form (Exp of Uniform in ln x) agrees strictly inside the support
    np.testing.assert_allclose(np.asarray(lu.tfp_dist().log_prob(xs[1:-1])), ref.logpdf(xs[1:-1]),
                               rtol=0, atol=1e-12)
    # outside the support (incl. x <= 0 and NaN): -inf, CDF clamped to [0, 1]
    out = jnp.array([0.5 * a, 2.0 * b, 0.0, -1.0, jnp.nan])
    assert np.all(np.isneginf(np.asarray(lu.logpdf(out))))
    np.testing.assert_array_equal(np.asarray(lu.inverse_unit_transform(out[:2])), [0.0, 1.0])


def test_values_match_live_prospector():
    pp = _prospector_priors()
    if pp is None:
        pytest.skip("no Prospector clone ($PROSPECTOR_DIR); the golden table covers it")
    for (a, b) in GOLDEN:
        p, lu = pp.LogUniform(mini=a, maxi=b), LogUniform(mini=a, maxi=b)
        xs = np.geomspace(a, b, 301)
        us = np.linspace(0.0, 1.0, 301)
        np.testing.assert_allclose(np.asarray(lu.logpdf(xs)), p(xs), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(lu.inverse_unit_transform(xs)),
                                   p.inverse_unit_transform(xs), rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(lu.unit_transform(us)), p.unit_transform(us),
                                   rtol=1e-13, atol=0)
        np.testing.assert_allclose(np.asarray(lu.range), p.range, rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(lu.bounds), p.bounds(), rtol=0, atol=0)


# ---- b. KS -----------------------------------------------------------------
def test_ks_vs_recipe_and_analytic():
    sys.path.insert(0, str(REPO / "examples" / "recipes"))
    from loguniform_prior import loguniform, sampled_name
    a, b, n = 1e-3, 10.0, 100_000
    x_new = np.asarray(LogUniform(mini=a, maxi=b).sample(jax.random.PRNGKey(0), (n,)))
    assert x_new.dtype == np.float64 and x_new.shape == (n,)
    assert x_new.min() >= a and x_new.max() <= b
    prior, transform = loguniform("x", a, b)
    x_rec = np.asarray(transform({sampled_name("x"): prior.sample(jax.random.PRNGKey(1), (n,))}))
    ks2 = scipy.stats.ks_2samp(x_new, x_rec)
    ksc = scipy.stats.kstest(x_new, scipy.stats.reciprocal(a, b).cdf)
    print(f"KS vs recipe D={ks2.statistic:.5f} p={ks2.pvalue:.3f}; "
          f"vs reciprocal CDF D={ksc.statistic:.5f} p={ksc.pvalue:.3f}")
    assert ks2.pvalue > 0.01 and ksc.pvalue > 0.01


# ---- c. round trip, validation, serialisation ------------------------------
def test_unit_transform_round_trip_and_shapes():
    lu = LogUniform(mini=1e-2, maxi=3.0)
    u = jnp.linspace(0.0, 1.0, 1001)
    np.testing.assert_allclose(np.asarray(lu.inverse_unit_transform(lu.unit_transform(u))),
                               np.asarray(u), rtol=0, atol=1e-14)
    x = jnp.geomspace(1e-2, 3.0, 1001)
    np.testing.assert_allclose(np.asarray(lu.unit_transform(lu.inverse_unit_transform(x))),
                               np.asarray(x), rtol=1e-13, atol=0)
    assert lu.sample(jax.random.PRNGKey(0)).shape == ()
    assert lu.sample(jax.random.PRNGKey(0), shape=(7, 1)).shape == (7, 1)
    vec = LogUniform(mini=jnp.array([0.1, 1.0]), maxi=jnp.array([1.0, 100.0]))
    s = np.asarray(vec.sample(jax.random.PRNGKey(2), (1000,)))
    assert s.shape == (1000, 2) and len(vec) == 2
    assert np.all((s[:, 0] >= 0.1) & (s[:, 0] <= 1.0) & (s[:, 1] >= 1.0) & (s[:, 1] <= 100.0))


@pytest.mark.parametrize("kw", [dict(mini=0.0, maxi=1.0), dict(mini=-1.0, maxi=1.0),
                                dict(mini=2.0, maxi=1.0), dict(mini=1.0, maxi=1.0),
                                dict(mini=1.0, maxi=np.inf), dict(mini=np.nan, maxi=1.0)])
def test_rejects_bad_limits(kw):
    with pytest.raises(ValueError, match="0 < mini < maxi"):
        LogUniform(**kw)


def test_rejects_wrong_kwargs():
    with pytest.raises(TypeError):
        LogUniform(low=0.1, high=1.0)
    with pytest.raises(TypeError):
        LogUniform(mini=0.1)


def test_serialize_round_trip():
    lu = LogUniform(mini=1e-2, maxi=3.0, name="tau")
    spec = json.loads(json.dumps(lu.serialize()))
    assert spec == {"type": "LogUniform", "name": "tau", "mini": 0.01, "maxi": 3.0}
    kind = spec.pop("type")
    back = getattr(sys.modules["ceridwen.priors"], kind)(**spec)
    assert back.serialize() == lu.serialize()
    alias = LogUniform(parnames=("lo", "hi"), lo=0.1, hi=2.0)
    assert float(alias.params["mini"]) == 0.1 and float(alias.params["maxi"]) == 2.0


# ---- d. jit / vmap / grad --------------------------------------------------
def test_jit_vmap_grad():
    a, b = 1e-2, 3.0
    lu = LogUniform(mini=a, maxi=b)
    xs = jnp.geomspace(a * 1.001, b * 0.999, 64)
    np.testing.assert_array_equal(np.asarray(jax.jit(lu.logpdf)(xs)), np.asarray(lu.logpdf(xs)))
    np.testing.assert_array_equal(np.asarray(jax.vmap(lu.logpdf)(xs)), np.asarray(lu.logpdf(xs)))
    g = np.asarray(jax.vmap(jax.grad(lambda x: lu.logpdf(x)))(xs))
    np.testing.assert_allclose(g, -1.0 / np.asarray(xs), rtol=1e-13)
    gout = np.asarray(jax.vmap(jax.grad(lambda x: lu.logpdf(x)))(jnp.array([-1.0, 0.0, 10.0])))
    np.testing.assert_array_equal(gout, [0.0, 0.0, 0.0])
    gu = np.asarray(jax.vmap(jax.grad(lu.unit_transform))(jnp.linspace(0.0, 1.0, 11)))
    assert np.all(np.isfinite(gu)) and np.all(gu > 0)


# ---- e. _detect_bounds on a SedModel ---------------------------------------
@pytest.fixture(scope="module")
def grid():
    from _gridfixture import find_test_grid
    path = find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (tests/_gridfixture.py)")
    from ceridwen import SSPData
    return SSPData.load(str(path))


TAU_LO, TAU_HI = 1e-2, 3.0


def _sed_model(ssp):
    from ceridwen import CSPBasis, SedModel
    from ceridwen.cosmology import Cosmology
    from ceridwen.observation import Photometry
    n_time = 5
    csp = CSPBasis(ssp, theta={"lookback_time": jnp.linspace(0.0, 12.0, n_time),
                               "sfh": jnp.ones(n_time), "logzsol": jnp.array([-0.2])},
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=False, verbose=False, cosmo=Cosmology.planck18())
    filters = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
    priors = {"sfh": Uniform(low=0.01, high=10.0), "logzsol": Uniform(low=-2.0, high=0.2),
              "diffuse_tau_kc": LogUniform(mini=TAU_LO, maxi=TAU_HI),
              "diffuse_dust_index": Uniform(low=-1.0, high=0.4)}
    init = {"sfh": jnp.array([1.0, 2.0, 1.5, 0.7, 0.3]), "logzsol": jnp.array([-0.3]),
            "diffuse_tau_kc": jnp.array([0.3]), "diffuse_dust_index": jnp.array([-0.2])}
    kw = dict(priors=priors, free_param_init=init, zred=0.1, broaden_photometry=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = SedModel(csp, [Photometry(filters=filters, name="phot", flux=np.ones(5),
                                          uncertainty=0.1 * np.ones(5))], **kw)
        f0 = np.asarray(probe.predict(probe.theta_init)["phot"])
        phot = Photometry(filters=filters, name="phot", flux=f0 * 1.03,
                          uncertainty=0.05 * np.abs(f0))
        return SedModel(csp, [phot], **kw)


def test_detect_bounds_sees_loguniform(grid):
    from ceridwen.fit import _detect_bounds
    from ceridwen.csp.csp import prior_support
    model = _sed_model(grid)
    b = _detect_bounds(model)
    assert b["diffuse_tau_kc"] == (TAU_LO, TAU_HI)
    assert prior_support(model.priors["diffuse_tau_kc"]) == (TAU_LO, TAU_HI)
    assert np.isfinite(float(model.ln_prior(model.theta_init)))


# ---- f. nested sampling toy --------------------------------------------------
# L(x) = N(ln x; MU, S) and p(x) = 1 / (x ln(B/A)) on [A, B], so
#   Z = [Phi((ln B - MU)/S) - Phi((ln A - MU)/S)] / ln(B/A)
# and the posterior of ln x is N(MU, S) truncated to [ln A, ln B].
A, B, MU, S = 1e-2, 1e2, 0.5, 0.4


def _toy_loglike(t):
    y = jnp.log(t["x"][0])
    return -0.5 * ((y - MU) / S) ** 2 - jnp.log(S * jnp.sqrt(2.0 * jnp.pi))


def _analytic_logz():
    Phi = scipy.stats.norm.cdf
    return math.log((Phi((math.log(B) - MU) / S) - Phi((math.log(A) - MU) / S)) / math.log(B / A))


def test_nested_sampling_evidence_and_posterior():
    pytest.importorskip("blackjax.ns")
    from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
    prior = LogUniform(mini=A, maxi=B)
    ad = BlackJAXNestedSamplerAdapter({"x": prior}, num_live=300, num_delete=30,
                                      num_inner_steps=15, logZ_tol=-4.0, verbose=False,
                                      checkpoint_interval_s=0.0)
    res = ad.run(_toy_loglike, lambda t: jnp.sum(prior.logpdf(t["x"])),
                 {"x": jnp.array([1.0])}, jax.random.PRNGKey(3))
    lz, lz_err, lz_true = float(res.log_evidence), float(res.log_evidence_err), _analytic_logz()
    x = np.asarray(res.samples["x"]).ravel()
    w = np.exp(np.asarray(res.log_weights) - np.max(res.log_weights))
    w /= w.sum()
    m = float(np.sum(w * np.log(x)))
    sd = float(np.sqrt(np.sum(w * (np.log(x) - m) ** 2)))
    print(f"NS lnZ = {lz:.4f} +- {lz_err:.4f}, analytic {lz_true:.4f}; "
          f"posterior ln x mean {m:.4f} (analytic {MU}), sd {sd:.4f} (analytic {S})")
    assert np.all((x >= A) & (x <= B))
    assert abs(lz - lz_true) < max(4.0 * lz_err, 0.15)
    assert abs(m - MU) < 0.05 and abs(sd - S) < 0.05


# ---- g. NUTS toy ---------------------------------------------------------------
def _nuts(loglike, key):
    from ceridwen.sampler.nuts import BlackJAXNUTSAdapter
    from ceridwen.fit import _detect_bounds

    class _M:  # the one attribute _detect_bounds reads
        priors = {"x": LogUniform(mini=A, maxi=B)}
    bounds = _detect_bounds(_M)
    assert bounds == {"x": (A, B)}
    prior = _M.priors["x"]
    ad = BlackJAXNUTSAdapter(num_warmup=300, num_samples=1000, num_chains=2, bounds=bounds,
                             verbose=False)
    res = ad.run(loglike, lambda t: jnp.sum(prior.logpdf(t["x"])), {"x": jnp.array([1.0])}, key)
    return np.asarray(res.samples["x"]).ravel()


def test_nuts_prior_only_and_toy_posterior():
    pytest.importorskip("blackjax")
    x0 = _nuts(lambda t: jnp.array(0.0), jax.random.PRNGKey(4))     # target = the prior
    x1 = _nuts(_toy_loglike, jax.random.PRNGKey(5))
    for x in (x0, x1):
        assert np.all(np.isfinite(x)) and np.all((x >= A) & (x <= B))
    m0 = float(np.mean(np.log(x0)))
    m1, s1 = float(np.mean(np.log(x1))), float(np.std(np.log(x1)))
    print(f"NUTS prior-only mean ln x {m0:.3f} (analytic {0.5 * math.log(A * B):.3f}, sd "
          f"{math.log(B / A) / math.sqrt(12):.3f}); toy mean {m1:.3f} (analytic {MU}), sd {s1:.3f}"
          f" (analytic {S})")
    assert abs(m0 - 0.5 * math.log(A * B)) < 0.5
    assert abs(m1 - MU) < 0.1 and abs(s1 - S) < 0.08


# ---- h. gradient of the fitSED log-posterior -----------------------------------
def _lnprob(model):
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood.likelihood import MultiObservationLikelihood, make_lnprobfn
    obs = model.obs_dict
    keys = tuple(obs)
    lhs = tuple(_likelihood_for(obs[k], model.param_names, model=model) for k in keys)
    multi = MultiObservationLikelihood(keys=keys, likelihoods=lhs)
    return make_lnprobfn(obs, model, model, multi)


@pytest.mark.parametrize("tau", [0.3, TAU_LO * 1.01, TAU_HI * 0.99])
def test_fitsed_logpost_gradient_matches_fd(grid, tau):
    model = _sed_model(grid)
    lnprob = _lnprob(model)
    th = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in model.theta_init.items()}
    th["diffuse_tau_kc"] = jnp.array([tau])
    v, g = jax.value_and_grad(lnprob)(th)
    assert np.isfinite(float(v))
    worst, rows = 0.0, []
    for name in model.param_names:
        base = np.asarray(th[name], dtype=np.float64)
        for i in range(base.size):
            # the photometric prediction is float32 (see CLAUDE.md T3: float32 einsums), so
            # ln p carries ~1e-5 of rounding noise; a 1e-3 relative step keeps the FD error
            # from it at the per-cent level (1e-6 steps give FD errors of order unity).
            h = 1e-3 * max(abs(base.flat[i]), 1e-2)
            up, dn = base.copy(), base.copy()
            up.flat[i] += h
            dn.flat[i] -= h
            fd = (float(lnprob({**th, name: jnp.asarray(up)}))
                  - float(lnprob({**th, name: jnp.asarray(dn)}))) / (2 * h)
            ad = float(np.asarray(g[name]).flat[i])
            assert np.isfinite(ad)
            rel = abs(ad - fd) / max(abs(fd), 1.0)
            rows.append((name, i, ad, fd, rel))
            worst = max(worst, rel)
    print(f"tau={tau}: max rel diff (|ad-fd| / max(|fd|, 1)) = {worst:.3e} over {len(rows)} "
          "elements")
    assert worst < 2e-2, rows
