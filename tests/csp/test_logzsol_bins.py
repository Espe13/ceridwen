"""One stellar metallicity per SFH bin: theta["logzsol_bins"] (shape (n_time - 1,), index 0 =
youngest), in CSPBasis and CSPBasis_afe (2026-11-07).  It replaces the per-node history
"logzsol_hist", which the kernel only ever used through its bin means 0.5*(zh[k] + zh[k+1]).
Ported from the prototype tests of ``METAL/zhmap/tests/test_proto_logzsol_bins.py``:

  1  the base commit's node history and the bin means of it as "logzsol_bins" give the same
     BYTES (W, spectrum; step and linear); a constant metallicity is unchanged too
  2  constant bins == the zh_const path (step, linear; nodes, between nodes, both axis ends)
  4  independent: W == sum over bins of the zh_const weights of that bin alone (step)
  5  out-of-grid bins clamp to the edge node, and check_param_ranges warns
  6  differentiability of the full fitSED log-posterior (finite, central FD, bounds, masked data;
     jit + vmap == loop)
  7  validation (missing key, shape, the removed "logzsol_hist" in theta / fixed / priors),
     gas_tied reads bin 0, CSPBasis_afe
  8  end to end: SedModel + fitSED (nested) + PostProcess + result-file round trip
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision)
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import require_test_grid, REPO_ROOT                       # noqa: E402

from ceridwen import CSPBasis, SSPData, SedModel, Cosmology, fitSED, read_result_h5  # noqa: E402
from ceridwen.observation import Photometry                                # noqa: E402
from ceridwen.priors import Uniform                                        # noqa: E402

BASE = "c3e743706981b21c14b1f0ac8c2ea7a029615e66"     # the last commit with logzsol_hist
LB = jnp.linspace(0.0, 2.0, 6)                       # 6 nodes, 5 bins
N = LB.size - 1
KW = dict(add_neb=False, add_dust=False, add_diffuse_dust=False, add_igm=False,
          verbose=False, cosmo=Cosmology.planck18())
FILTERS = ["galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0", "twomass_Ks"]
X64 = bool(jax.config.jax_enable_x64)
f64_only = pytest.mark.skipif(not X64, reason="float64 identities (float32 mode is on)")


@pytest.fixture(scope="module")
def ssp():
    return SSPData.load(str(require_test_grid()))


def _sfr(seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.1, 2.0, N))


def _mu(seed=1, lo=-1.8, hi=0.25):
    return np.random.default_rng(seed).uniform(lo, hi, N)


def _csp(ssp, key, interp, **over):
    zh = {"logzsol": jnp.array([0.0]), "logzsol_bins": jnp.zeros(N)}[key]
    return CSPBasis(ssp, theta={"lookback_time": LB, "sfh": jnp.ones(N), key: zh},
                    zh_const=(key == "logzsol"), sfh_interp=interp, **{**KW, **over})


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


# --- 1 ---------------------------------------------------------------------------------------

_ZERO_SCRIPT = textwrap.dedent("""
    import sys, numpy as np, jax.numpy as jnp, ceridwen
    from ceridwen import CSPBasis, SSPData, Cosmology
    assert ceridwen.__file__.startswith(sys.argv[1]), ceridwen.__file__
    ssp = SSPData.load(sys.argv[2])
    hist_key = sys.argv[4]                    # base: "logzsol_hist" (nodes); head: "logzsol_bins"
    out = {}
    for interp in ("step", "linear"):
        for key in ("logzsol", hist_key):
            if key == "logzsol":
                zh = jnp.array([-0.3])
            else:
                zh = jnp.linspace(-0.1, -1.3, 6)              # a non-constant node history
                if key == "logzsol_bins":
                    zh = 0.5 * (zh[:-1] + zh[1:])             # its bin means
            sfh = jnp.linspace(2.0, 0.3, 6)                     # per node at the base commit
            if hist_key == "logzsol_bins" and interp == "step":
                sfh = 0.5 * (sfh[:-1] + sfh[1:])                  # step: one SFR per bin
            th = {"lookback_time": jnp.linspace(0.0, 2.0, 6), "sfh": sfh, key: zh}
            c = CSPBasis(ssp, theta=th, zh_const=(key == "logzsol"), sfh_interp=interp,
                         add_neb=False, add_igm=False, verbose=False, cosmo=Cosmology.planck18())
            t = {k: jnp.asarray(v) for k, v in c.theta_init.items()}
            t.update(diffuse_tau_kc=jnp.array([0.4]), tau_pow=jnp.array([0.6]))
            name = "const" if key == "logzsol" else "history"
            out[f"{interp}/{name}/W"] = np.asarray(c.calculate_ssp_weights(t))
            out[f"{interp}/{name}/spec"] = np.asarray(c.get_spectrum(t))
    np.savez(sys.argv[3], **out)
""")


@f64_only
def test_1_bins_equal_the_base_commits_node_history(tmp_path):
    grid = str(require_test_grid())
    base = tmp_path / "base"
    base.mkdir()
    arch = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", BASE], capture_output=True)
    if arch.returncode != 0:
        pytest.skip(f"base commit {BASE[:12]} not in this clone: zero-change test did NOT run")
    subprocess.run(["tar", "-x", "-C", str(base)], input=arch.stdout, check=True)
    script = tmp_path / "zero.py"
    script.write_text(_ZERO_SCRIPT)
    res = {}
    for name, tree, key in (("base", base, "logzsol_hist"), ("head", REPO_ROOT, "logzsol_bins")):
        env = dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu",
                   CERIDWEN_NO_UPDATE_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("CERIDWEN_X64", None)
        out = tmp_path / f"{name}.npz"
        r = subprocess.run([sys.executable, str(script),
                            str(tree / "ceridwen"), grid, str(out), key],
                           env=env, cwd=str(tmp_path), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        res[name] = np.load(out)
    assert sorted(res["base"].files) == sorted(res["head"].files)
    for k in res["base"].files:
        a, b = res["base"][k], res["head"][k]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k


# --- 2 - 4 -----------------------------------------------------------------------------------

@f64_only
@pytest.mark.parametrize("interp", ["step", "linear"])
def test_2_constant_bins_equal_zh_const(ssp, interp):
    var = _csp(ssp, "logzsol_bins", interp)
    const = _csp(ssp, "logzsol", interp)
    z = np.asarray(var.zmet)
    values = [float(z[0]), float(z[-1]), float(z[3]), 0.5 * float(z[3] + z[4]), -0.37]
    for l in values:
        a = var.calculate_ssp_weights({"sfh": _sfr(), "logzsol_bins": jnp.full(N, l)})
        b = const.calculate_ssp_weights({"sfh": _sfr(), "logzsol": jnp.array([l])})
        assert _rel(a, b) <= 1e-13, l


@f64_only
@pytest.mark.parametrize("seed", [1, 2])
def test_4_superposition_of_single_bins(ssp, seed):
    """Independent of the new code path: each bin alone through the constant-metallicity
    kernel, summed (step SFH: a bin's SFR touches only that bin's ages)."""
    var = _csp(ssp, "logzsol_bins", "step")
    const = _csp(ssp, "logzsol", "step")
    mu, sfr = _mu(seed), np.asarray(_sfr(seed))
    a = np.asarray(var.calculate_ssp_weights({"sfh": jnp.asarray(sfr), "logzsol_bins": jnp.asarray(mu)}))
    b = np.zeros_like(a)
    for i in range(N):
        s = np.zeros(N)
        s[i] = sfr[i]
        b += np.asarray(const.calculate_ssp_weights({"sfh": jnp.asarray(s),
                                                     "logzsol": jnp.array([mu[i]])}))
    assert _rel(a, b) <= 1e-12


# --- 5 ---------------------------------------------------------------------------------------

def test_5_out_of_grid_bins_clamp_and_warn(ssp):
    var = _csp(ssp, "logzsol_bins", "step")
    lo, hi = float(var.zmet.min()), float(var.zmet.max())

    def W(v):
        return np.asarray(var.calculate_ssp_weights({"sfh": _sfr(), "logzsol_bins": jnp.full(N, v)}))
    assert np.array_equal(W(lo - 0.7), W(lo))
    assert np.array_equal(W(hi + 0.7), W(hi))
    bad = np.array([0.0, 0.1, hi + 0.5, -0.2, -0.4])
    with pytest.warns(UserWarning, match=r"theta\['logzsol_bins'\].*outside the SSP metallicity grid"):
        var.check_param_ranges({**var.theta_init, "logzsol_bins": jnp.asarray(bad)})
    inside = np.linspace(lo + 0.1, hi - 0.1, N)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert var.check_param_ranges({**var.theta_init, "logzsol_bins": jnp.asarray(inside)},
                                      warn=False) == []


# --- 6 ---------------------------------------------------------------------------------------

def _mock_model(ssp, *, masked=False):
    """Photometry of a per-bin metallicity mock at z = 0.5, its data the model's own prediction
    at ``truth`` (in maggies); ``masked`` masks one band."""
    csp = CSPBasis(ssp, theta={"lookback_time": LB, "sfh": jnp.ones(N),
                               "logzsol_bins": jnp.zeros(N)},
                   zh_const=False, add_neb=False, add_dust=False, add_diffuse_dust=True,
                   verbose=False, cosmo=Cosmology.planck18())
    truth = {"logzsol_bins": jnp.linspace(-0.1, -1.1, N), "logmass": jnp.array([10.0]),
             "diffuse_tau_kc": jnp.array([0.4]), "diffuse_dust_index": jnp.array([-0.5])}
    sfh = jnp.linspace(2.0, 0.4, N)
    priors = {"logzsol_bins": Uniform(low=-1.5, high=0.2), "logmass": Uniform(low=9.5, high=10.5),
              "diffuse_tau_kc": Uniform(low=0.0, high=1.5),
              "diffuse_dust_index": Uniform(low=-1.0, high=0.4)}

    def build(flux, mask):
        phot = Photometry(filters=FILTERS, flux=flux, uncertainty=np.abs(flux) / 20.0,
                          mask=mask, name="phot")
        return SedModel(csp, [phot], transforms={"sfh": lambda th, _s=sfh: _s}, priors=priors,
                        free_param_init={"logzsol_bins": jnp.full(N, -0.5),
                                         "logmass": jnp.array([10.0])},
                        zred=0.5, broaden_photometry=False)
    mask = np.ones(len(FILTERS), bool)
    flux = np.asarray(build(np.ones(len(FILTERS)), mask).predict(truth)["phot"])
    if masked:
        mask[1] = False
    return build(flux, mask), truth


def _log_posterior(model):
    """The ln posterior fitSED samples: the model's observation likelihoods + model.ln_prior
    (tests/baselines/_capture_f32.py:log_posterior)."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood import MultiObservationLikelihood
    keys = tuple(model.obs_dict)
    lhs = tuple(_likelihood_for(model.obs_dict[k], model.param_names, model=model) for k in keys)

    class _Prior:
        @staticmethod
        def log_prob(theta):
            return model.ln_prior(theta)
    return MultiObservationLikelihood(keys=keys, likelihoods=lhs).make_lnprobfn(
        model.obs_dict, model, _Prior())


@f64_only
@pytest.mark.parametrize("masked", [False, True])
def test_6_gradient_of_the_log_posterior(ssp, masked):
    model, _ = _mock_model(ssp, masked=masked)
    lnp = _log_posterior(model)
    base = {k: jnp.asarray(v) for k, v in model.theta_init.items() if k in model.param_names}
    lo, hi = -1.5, 0.2
    vg = jax.jit(jax.value_and_grad(lambda mu, b=base: lnp({**b, "logzsol_bins": mu})))
    f = jax.jit(lambda mu, b=base: lnp({**b, "logzsol_bins": mu}))
    centre = np.full(N, 0.5 * (lo + hi)) + np.linspace(-0.05, 0.05, N)
    for mu0 in (centre, np.full(N, lo + 0.013 * (hi - lo)), np.full(N, hi - 0.013 * (hi - lo))):
        mu0 = mu0 + np.array([0.0, 0.0071, -0.0043, 0.0029, -0.0011])   # off the grid nodes
        v, g = vg(jnp.asarray(mu0))
        g = np.asarray(g)
        assert np.isfinite(float(v)) and np.all(np.isfinite(g)), (mu0, v, g)
        noise = 2e-7 * abs(float(v))            # float32 SSP einsums: lnp noise ~1e-7 |lnp|
        for i in range(N):
            ok, rels = False, []
            for h in (1e-3, 3e-4, 1e-4):
                e = np.zeros(N)
                e[i] = h
                fd = (float(f(jnp.asarray(mu0 + e))) - float(f(jnp.asarray(mu0 - e)))) / (2 * h)
                err = abs(fd - g[i])
                ok |= err <= 1e-3 * max(abs(fd), abs(g[i])) + 3 * noise / h
                rels.append(err / max(abs(fd), abs(g[i]), 1e-12))
            assert ok, (i, mu0, float(v), g[i], rels)


@f64_only
def test_6b_jit_and_vmap_equal_the_loop(ssp):
    model, _ = _mock_model(ssp)
    lnp = _log_posterior(model)
    base = {k: jnp.asarray(v) for k, v in model.theta_init.items() if k in model.param_names}
    vg = jax.value_and_grad(lambda mu, b=base: lnp({**b, "logzsol_bins": mu}))
    mus = jnp.asarray(np.stack([_mu(s, -1.4, 0.15) for s in range(6)]))
    loop = [vg(m) for m in mus]
    vb, gb = jax.jit(jax.vmap(vg))(mus)
    for k, (v, g) in enumerate(loop):
        np.testing.assert_allclose(np.asarray(vb[k]), float(v), rtol=1e-6)
        # the gradient's float32 noise under vmap (the SSP einsums reorder), measured
        # 2026-11-07 on this mock: 5.1e-5 of max|g| with logzsol_bins, 3.3e-5 with the
        # equivalent logzsol_hist (the node path, since removed); eager vs jit 9e-7
        np.testing.assert_allclose(np.asarray(gb[k]), np.asarray(g), rtol=0,
                                   atol=2e-4 * float(np.max(np.abs(np.asarray(g)))))
    # the CSP weights themselves: jit(vmap) == loop to float64 rounding
    var = _csp(ssp, "logzsol_bins", "step")
    w = lambda mu: var.calculate_ssp_weights({"sfh": _sfr(), "logzsol_bins": mu})  # noqa: E731
    lw = np.stack([np.asarray(w(m)) for m in mus])
    assert np.max(np.abs(np.asarray(jax.jit(jax.vmap(w))(mus)) - lw)) <= 1e-14 * np.max(np.abs(lw))


# --- 7 ---------------------------------------------------------------------------------------

def test_7_validation(ssp):
    th = {"lookback_time": LB, "sfh": jnp.ones(N)}
    with pytest.raises(ValueError, match=r"requires a metallicity history theta\['logzsol_bins'\]"):
        CSPBasis(ssp, theta=th, zh_const=False, **KW)
    # the removed per-node history: a teaching error with the bin means that give the same model
    hist = jnp.asarray([0.0, -0.2, -0.4, -0.6, -0.8, -1.0])
    with pytest.raises(ValueError, match=r"'logzsol_hist'\] was removed.*-0\.1\s+-0\.3\s+-0\.5\s+-0\.7\s+-0\.9"):
        CSPBasis(ssp, theta={**th, "logzsol_hist": hist}, zh_const=False, **KW)
    with pytest.raises(ValueError, match="'logzsol_hist'.*was removed"):
        CSPBasis(ssp, theta={**th, "logzsol_hist": hist, "logzsol_bins": jnp.zeros(N)},
                 zh_const=False, **KW)
    with pytest.raises(ValueError, match=r"one metallicity per\s+SFH bin, shape \(5,\)"):
        CSPBasis(ssp, theta={**th, "logzsol_bins": jnp.zeros(N + 1)}, zh_const=False, **KW)
    with pytest.raises(ValueError, match="zh_const=True but theta also contains 'logzsol_bins'"):
        CSPBasis(ssp, theta={**th, "logzsol": jnp.zeros(1), "logzsol_bins": jnp.zeros(N)},
                 zh_const=True, **KW)
    with pytest.raises(ValueError, match="not finite"):
        CSPBasis(ssp, theta={**th, "logzsol_bins": jnp.array([0.0, np.nan, 0, 0, 0])},
                 zh_const=False, **KW)
    # the lookback_time= shortcut and the attributes every reader uses
    c = CSPBasis(ssp, lookback_time=LB, zh_const=False, **KW)
    assert c.zh_key == "logzsol_bins"
    assert c.theta_init["logzsol_bins"].shape == (N,) and "logzsol_hist" not in c.theta_init
    k = CSPBasis(ssp, lookback_time=LB, zh_const=True, **KW)
    assert k.zh_key == "logzsol"
    # fixed= names the right switch, and fixes the bins
    with pytest.raises(ValueError, match="zh_const=False \\(theta\\['logzsol_bins'\\]"):
        CSPBasis(ssp, lookback_time=LB, zh_const=True, fixed={"logzsol_bins": [0.0] * N}, **KW)
    with pytest.raises(ValueError, match="fixed\\['logzsol_hist'\\] was removed"):
        CSPBasis(ssp, lookback_time=LB, zh_const=False, fixed={"logzsol_hist": [0.0] * (N + 1)},
                 **KW)
    f = CSPBasis(ssp, lookback_time=LB, zh_const=False,
                 fixed={"logzsol_bins": np.linspace(-0.2, -1.0, N)}, **KW)
    assert "logzsol_bins" not in f.param_names
    # a prior on the removed key is refused, and one outside the grid's axis like one on logzsol
    with pytest.raises(ValueError, match="priors\\['logzsol_hist'\\] was removed"):
        SedModel(c, [Photometry(filters=FILTERS[:3], flux=[1e-9] * 3, uncertainty=[1e-10] * 3,
                                name="p")],
                 priors={"logzsol_hist": Uniform(low=-1.0, high=0.0),
                         "logmass": Uniform(low=9.0, high=11.0)}, zred=0.5)
    with pytest.raises(ValueError, match="prior on 'logzsol_bins'.*outside this grid"):
        SedModel(c, [Photometry(filters=FILTERS[:3], flux=[1e-9] * 3, uncertainty=[1e-10] * 3,
                                name="p")],
                 priors={"logzsol_bins": Uniform(low=-5.0, high=0.0),
                         "logmass": Uniform(low=9.0, high=11.0)}, zred=0.5)


def test_7_copy_rebind_recipe(ssp):
    """The shared-template recipe (copy.copy, re-bind bound methods, initialize_model_structure
    with new nodes; tests/agn/test_copy_rebind.py) keeps the per-bin metallicity."""
    import copy
    import types
    tpl = CSPBasis(ssp, lookback_time=LB, zh_const=False, sfh_per_bin=True, **KW)
    nodes = jnp.asarray([0.0, 0.03, 0.2, 0.5, 1.0, 2.5])
    cp = copy.copy(tpl)
    for k, v in list(vars(cp).items()):
        if isinstance(v, types.MethodType) and v.__self__ is tpl:
            setattr(cp, k, types.MethodType(v.__func__, cp))
    cp.initialize_model_structure({"lookback_time": nodes, **dict(tpl.theta_init)})
    fresh = CSPBasis(ssp, theta={"lookback_time": nodes, **dict(tpl.theta_init)},
                     zh_const=False, **KW)
    th = {"sfh": _sfr(), "logzsol_bins": jnp.asarray(_mu(3))}
    assert cp.zh_key == "logzsol_bins"
    a, b = np.asarray(cp.calculate_ssp_weights(th)), np.asarray(fresh.calculate_ssp_weights(th))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != np.asarray(tpl.calculate_ssp_weights(th)).tobytes()


@pytest.mark.skipif(not os.environ.get("SPS_HOME"), reason="SPS_HOME not set (nebular grids)")
@pytest.mark.fsps
def test_7_gas_tied_reads_the_youngest_bin(ssp):
    mu = jnp.asarray([-0.35, -0.6, -0.8, -1.0, -1.2])
    th = {"lookback_time": LB, "sfh": jnp.ones(N), "logzsol_bins": mu}
    kw = dict(KW, add_neb=True, sps_home=os.environ["SPS_HOME"])
    tied = CSPBasis(ssp, theta=th, zh_const=False, gas_tied=True, **kw)
    free = CSPBasis(ssp, theta={**th, "gas_logz": mu[:1]}, zh_const=False, **kw)
    t = {k: jnp.asarray(v) for k, v in tied.theta_init.items()}
    assert np.asarray(tied._gas_logz(t)).tolist() == [-0.35]
    a = np.asarray(tied.get_spectrum(t))
    b = np.asarray(free.get_spectrum({**t, "gas_logz": mu[:1]}))
    assert a.tobytes() == b.tobytes()
    msgs = tied.check_param_ranges({**t, "logzsol_bins": jnp.full(N, 5.0)}, warn=False)
    assert any("logzsol_bins'][0] (the youngest bin's value)" in m for m in msgs), msgs


AFE_GRID = REPO_ROOT / "ceridwen" / "data" / "test_data" / "amist_c3k_lr_chab_afe.h5"


@pytest.mark.skipif(not AFE_GRID.is_file(), reason=f"{AFE_GRID.name} not found: afe bins test did NOT run")
def test_7_csp_afe_with_bins():
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe
    sa = SSPDataAfe.load(str(AFE_GRID))
    kw = dict(lookback_time=LB, verbose=False, cosmo=Cosmology.planck18(),
              add_dust=False, add_diffuse_dust=False)
    c = CSPBasis_afe(sa, zh_const=False, **kw)
    assert c.zh_key == "logzsol_bins" and c.theta_init["logzsol_bins"].shape == (N,)
    mu = jnp.linspace(-0.2, -1.0, N)
    th = {**{k: jnp.asarray(v) for k, v in c.theta_init.items()}, "logzsol_bins": mu,
          "afe": jnp.array([0.2])}
    tot = np.asarray(c.logzsol_total(th))
    assert tot.shape == (N,)
    for i in range(N):                          # elementwise: each bin on its own
        one = np.asarray(c.logzsol_total({"logzsol_bins": mu[i:i + 1], "afe": jnp.array([0.2])}))
        assert np.isclose(tot[i], one[0])
    const = CSPBasis_afe(sa, zh_const=True, **kw)
    W_bins = c.calculate_ssp_weights({**th, "logzsol_bins": jnp.full(N, -0.4)})
    W_const = const.calculate_ssp_weights({"sfh": th["sfh"], "logzsol": jnp.array([-0.4]),
                                           "afe": jnp.array([0.2])})
    assert _rel(W_bins, W_const) <= 1e-13
    cells = c.refused_cells_logzsol()
    if cells:
        z_lo, _, a_lo, _, _ = cells[0]
        bad = {**th, "logzsol_bins": jnp.full(N, z_lo + 0.05), "afe": jnp.array([a_lo + 0.05])}
        with pytest.raises(ValueError, match="theta\\['logzsol_bins'\\]"):
            c._check_refused_cells_theta(bad)


# --- 8 ---------------------------------------------------------------------------------------

@f64_only
def test_8_fit_postprocess_and_result_file(ssp, tmp_path):
    from ceridwen.postprocess import PostProcess
    from ceridwen.resultfile import check_model_against_result
    model, truth = _mock_model(ssp)
    result = fitSED(model, output_dir=tmp_path, rng_key=jax.random.PRNGKey(3), verbose=False,
                    sampler_kwargs={"num_live": 40, "num_delete": 8, "num_inner_steps": 8,
                                    "logZ_tol": -0.5, "slice_kernel": "lanes"})
    s = np.asarray(result.samples["logzsol_bins"])
    assert s.ndim == 2 and s.shape[1] == N and np.all(np.isfinite(s))
    assert np.all((s >= -1.5) & (s <= 0.2))
    path = tmp_path / "ceridwen_result.h5"
    r = read_result_h5(path)
    assert np.asarray(r["samples"]["logzsol_bins"]).shape == s.shape
    cmp = check_model_against_result(model, path)
    assert cmp.ok, str(cmp)
    import h5py
    import json
    with h5py.File(path, "r") as f:
        cfg = json.loads(f["model"].attrs["csp_config"])
    assert cfg["zh_const"] is False and "zh_per_bin" not in cfg
    out = PostProcess(model, result, n_samples=60, uv=False, ionizing=False, mfrac=False).run()
    assert np.asarray(out["theta"]["logzsol_bins"]).shape == (60, N)
    assert np.all(np.isfinite(np.asarray(out["extras"]["sfh"]["sfr100"])))
