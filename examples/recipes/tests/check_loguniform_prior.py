"""Check ``examples/recipes/loguniform_prior.py`` against Prospector (CPU, < 1 min).

    python examples/recipes/tests/check_loguniform_prior.py

Prospector's ``prospect/models/priors.py`` is loaded standalone (numpy + scipy
only) from ``$PROSPECTOR_DIR`` (a clone at commit a78d153). Without it, the
checks that need Prospector FAIL rather than skip.

Checks
  1. KS, 1e5 draws: helper draws vs Prospector ``LogUniform`` draws (two-sample),
     and each vs the analytic CDF ln(x/a)/ln(b/a).        PASS if every p > 0.01
  2. Unit transform (nested sampling path): 10**Uniform.unit_transform(u) ==
     Prospector LogUniform.unit_transform(u) on 1e5 u's.  PASS if max rel diff < 1e-12
  3. Density: log p_x(x) = Uniform.logpdf(log10 x) - ln(x ln 10) equals the
     Prospector ln-prior on [a, b].                       PASS if max abs diff < 1e-10
  4. SedModel wiring on the test SSP grid: log10_<name> sampled, <name> derived,
     fit._detect_bounds gives (log10 a, log10 b), the eager SedModel.predict is
     bit-identical to csp.predict with <name> = 10**log10_<name> set directly, the
     jitted predict agrees with the eager one to 1e-6 (the jit/eager gap is 1.2e-7
     here; its cause, e.g. float32 einsum reordering, is not verified), and the
     gradient is finite and non-zero.
  5. LogNormal: CERIDWEN LogNormal(**prospector_lognormal_to_ceridwen(m, s))
     logpdf == Prospector LogNormal(m, s) ln-prior.       PASS if max abs diff < 1e-10
     (and the unconverted pair differs, to show the mismatch is real)
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import warnings

import numpy as np
import scipy.stats
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

from loguniform_prior import (loguniform, loguniform_setup, sampled_name,   # noqa: E402
                              prospector_lognormal_to_ceridwen)
from ceridwen.priors import LogNormal                                        # noqa: E402

PROSPECTOR_DIR = os.environ.get("PROSPECTOR_DIR", "")   # clone at a78d153; FAIL if unset

A, B = 1e-3, 10.0
N = 100_000
results = []


def report(name, ok, msg):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")


def load_prospector_priors():
    path = pathlib.Path(PROSPECTOR_DIR) / "prospect" / "models" / "priors.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("prospector_priors_a78d153", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = load_prospector_priors()
if pp is None:
    report("prospector", False, f"priors.py not found under {PROSPECTOR_DIR} (set $PROSPECTOR_DIR)")
    print("\nOVERALL: FAIL")
    sys.exit(1)

prior, transform = loguniform("x", A, B)
key = sampled_name("x")

# ---- 1. KS ---------------------------------------------------------------
draws = prior.sample(jax.random.PRNGKey(0), (N,))
x_cer = np.asarray(transform({key: draws}))
p_lu = pp.LogUniform(mini=A, maxi=B)
# Prospector's own sampling call, Prior.sample (priors.py:137-138), with size=N
# instead of len(self)=1:
x_pro = p_lu.distribution.rvs(*p_lu.args, size=N, loc=p_lu.loc, scale=p_lu.scale,
                              random_state=np.random.default_rng(0))
cdf = lambda x: np.log(np.asarray(x) / A) / np.log(B / A)   # noqa: E731
ks2 = scipy.stats.ks_2samp(x_cer, x_pro)
ksc = scipy.stats.kstest(x_cer, cdf)
ksp = scipy.stats.kstest(x_pro, cdf)
report("KS 1e5 draws", min(ks2.pvalue, ksc.pvalue, ksp.pvalue) > 0.01,
       f"2-sample D={ks2.statistic:.5f} p={ks2.pvalue:.3f}; "
       f"helper vs CDF D={ksc.statistic:.5f} p={ksc.pvalue:.3f}; "
       f"Prospector vs CDF D={ksp.statistic:.5f} p={ksp.pvalue:.3f}; "
       f"dtype={x_cer.dtype}, range=[{x_cer.min():.4g}, {x_cer.max():.4g}]")

# ---- 2. unit transform ---------------------------------------------------
u = np.random.default_rng(1).uniform(size=N)
x_ut_cer = np.asarray(transform({key: prior.unit_transform(jnp.asarray(u))}))
x_ut_pro = p_lu.unit_transform(u)
rel = np.max(np.abs(x_ut_cer / x_ut_pro - 1.0))
report("unit transform", rel < 1e-12, f"max |rel diff| = {rel:.3e} over {N} u's")

# ---- 3. density with Jacobian -------------------------------------------
xg = np.geomspace(A, B, 2001)[1:-1]
lp_cer = np.asarray(prior.logpdf(jnp.log10(jnp.asarray(xg)))) - np.log(xg * np.log(10.0))
lp_pro = p_lu(xg)
d = np.max(np.abs(lp_cer - lp_pro))
report("log density", d < 1e-10, f"max |diff| = {d:.3e} on 1999 points in ({A}, {B})")

# ---- 4. SedModel wiring --------------------------------------------------
try:
    from ceridwen import SSPData, CSPBasis, SedModel
    from ceridwen.observation import Photometry
    from ceridwen.priors import Uniform
    from ceridwen.cosmology import Cosmology
    from ceridwen.fit import _detect_bounds

    grid = os.environ.get("CERIDWEN_TEST_SSP",
                          str(REPO / "ceridwen" / "data" / "test_data" / "ssp_data_bpass.h5"))
    ssp = SSPData.load(grid)
    n_time = 5
    csp = CSPBasis(ssp, theta={"lookback_time": jnp.linspace(0.0, 12.0, n_time),
                               "sfh": jnp.ones(n_time), "logzsol": jnp.array([-0.2])},
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=False, verbose=False, cosmo=Cosmology.planck18())
    phot = Photometry(filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0"], name="phot",
                      flux=np.ones(4), uncertainty=0.1 * np.ones(4))
    lu = loguniform_setup("diffuse_tau_kc", 1e-2, 3.0, init=0.3)
    others = [p for p in csp.param_names if p != "diffuse_tau_kc"]
    priors = {p: Uniform(low=-5.0, high=5.0) for p in others}
    priors["logzsol"] = Uniform(low=-2.0, high=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, [phot], priors={**priors, **lu["priors"]},
                         transforms=lu["transforms"], free_param_init=lu["free_param_init"],
                         zred=0.1, broaden_photometry=False)
    k = sampled_name("diffuse_tau_kc")
    names_ok = (k in model.param_names) and ("diffuse_tau_kc" not in model.param_names)
    b = _detect_bounds(model)[k]
    b_ok = np.allclose(b, (np.log10(1e-2), np.log10(3.0)), rtol=0, atol=1e-15)

    th = dict(model.theta_init)
    tau = lu["transforms"]["diffuse_tau_kc"](th)      # 10**log10(0.3)
    direct = {kk: v for kk, v in th.items() if kk != k}
    direct["diffuse_tau_kc"] = tau
    direct["zred"] = jnp.array([0.1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_e = np.asarray(model.predict(th)["phot"])                 # eager, via transform
        pred_d = np.asarray(csp.predict(direct, model.observations, kinematics=model.kinematics,
                                        broaden_photometry=False)["phot"])
        pred_t = np.asarray(jax.jit(lambda t: model.predict(t)["phot"])(th))
    exact = bool(np.array_equal(pred_e, pred_d))
    # jit vs eager differ at ~1e-7 (cause not verified); tolerance as CLAUDE.md T3 (1e-6)
    pj = float(np.max(np.abs(pred_t / pred_e - 1.0)))
    tau_err = abs(float(tau[0]) / 0.3 - 1.0)
    g = jax.grad(lambda t: jnp.sum(model.predict({**th, k: t})["phot"]))(th[k])
    g_ok = bool(np.all(np.isfinite(np.asarray(g))) and np.all(np.asarray(g) != 0))
    report("SedModel wiring", names_ok and b_ok and exact and pj < 1e-6 and g_ok,
           f"{k} sampled={k in model.param_names}, diffuse_tau_kc derived="
           f"{'diffuse_tau_kc' not in model.param_names}; _detect_bounds[{k}]={b}; "
           f"eager predict == direct CSP predict bitwise: {exact}; jit vs eager max rel "
           f"diff={pj:.2e}; |10**log10(0.3)/0.3-1|={tau_err:.1e}; "
           f"d(sum flux)/d{k}={np.asarray(g)}")
except Exception as exc:  # report, do not hide
    report("SedModel wiring", False, f"{type(exc).__name__}: {exc}")

# ---- 5. LogNormal conversion --------------------------------------------
worst_conv, worst_raw = 0.0, 0.0
for m, s in [(0.0, 0.5), (np.log(2.0), 0.3), (-1.0, 1.0)]:
    xs = np.geomspace(np.exp(m - 4 * s), np.exp(m + 4 * s + 2 * s * s), 400)
    lp_p = pp.LogNormal(mode=m, sigma=s)(xs)
    lp_c = np.asarray(LogNormal(**prospector_lognormal_to_ceridwen(m, s)).logpdf(jnp.asarray(xs)))
    lp_raw = np.asarray(LogNormal(mode=m, sigma=s).logpdf(jnp.asarray(xs)))
    worst_conv = max(worst_conv, float(np.max(np.abs(lp_c - lp_p))))
    worst_raw = max(worst_raw, float(np.max(np.abs(lp_raw - lp_p))))
    peak_p = xs[np.argmax(lp_p)]
    if m == 0.0:
        print(f"      LogNormal(mode=0, sigma=0.5): Prospector pdf peak at x={peak_p:.3f} "
              f"(exp(mode)=1); CERIDWEN median exp(mode)=1 but its pdf peak is "
              f"exp(mode - sigma^2)={np.exp(-0.25):.3f}")
report("LogNormal conversion", worst_conv < 1e-10 and worst_raw > 1e-3,
       f"converted max |dlnp| = {worst_conv:.3e}; unconverted max |dlnp| = {worst_raw:.3f}")

print(f"\nOVERALL: {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)})")
sys.exit(0 if all(results) else 1)
