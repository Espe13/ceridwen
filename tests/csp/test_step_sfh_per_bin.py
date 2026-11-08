"""A step SFH takes one SFR per bin, as Prospector's FastStepBasis (2026-11-07).

``sfh_interp="step"`` holds the SFR constant within each bin.  It used to accept one SFR per
node and use only the bin means 0.5*(sfh[k] + sfh[k+1]), so the continuity prior acted on node
ratios and one alternating combination of the nodes changed no output.  It now takes
theta["sfh"] of shape (n_time-1,), and ``logsfr_ratios_to_sfh`` gives n_time-2 ratios of
adjacent BIN SFRs, normalised so sum_i SFR_i dt_i = 1 Msun.

  1  the base commit's step model fed per-node SFRs and today's fed their bin means give the
     same BYTES (W, spectrum with dust + IGM at z > 0, photometry); "linear" is unchanged
  2  logsfr_ratios_to_sfh per bin == Prospector's logsfr_ratios_to_masses (transcribed from
     prospect/models/transforms.py:184-198, prospector 40e2614) and sum SFR dt = 1
  3  no invisible direction: d W / d logsfr_ratios has full rank (the old node parametrisation
     had a null vector, shown here at the base commit's construction)
  4  refusals: a per-node sfh in step mode, sfh_per_bin=False with step, an sfh transform of the
     wrong length at SedModel setup; the lookback_time= shortcut shapes
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision)
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import require_test_grid, REPO_ROOT                       # noqa: E402

from ceridwen import CSPBasis, SSPData, SedModel, Cosmology                 # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh                 # noqa: E402
from ceridwen.observation import Photometry                                # noqa: E402
from ceridwen.priors import Uniform, StudentT                              # noqa: E402

BASE = "c3e743706981b21c14b1f0ac8c2ea7a029615e66"     # the last commit with per-node step SFRs
LB = jnp.asarray([0.0, 0.01, 0.03, 0.1, 0.5, 1.5, 3.0])  # Gyr, 7 nodes, 6 bins
X64 = bool(jax.config.jax_enable_x64)
f64_only = pytest.mark.skipif(not X64, reason="float64 identities (float32 mode is on)")
KW = dict(add_neb=False, add_dust=False, add_diffuse_dust=False, add_igm=False,
          verbose=False, cosmo=Cosmology.planck18())


@pytest.fixture(scope="module")
def ssp():
    return SSPData.load(str(require_test_grid()))


# --- 1 ---------------------------------------------------------------------------------------

_SCRIPT = textwrap.dedent("""
    import sys, numpy as np, jax.numpy as jnp, ceridwen
    from ceridwen import CSPBasis, SSPData, Cosmology
    from ceridwen.observation import Photometry
    assert ceridwen.__file__.startswith(sys.argv[1]), ceridwen.__file__
    ssp = SSPData.load(sys.argv[2])
    per_bin = sys.argv[4] == "head"
    lb = jnp.asarray([0.0, 0.01, 0.03, 0.1, 0.5, 1.5, 3.0])
    nodes = jnp.asarray([2.0, 1.4, 3.0, 0.7, 0.9, 0.2, 0.05])   # a non-monotonic SFR per node
    out = {}
    for interp in ("step", "linear"):
        sfh = 0.5 * (nodes[:-1] + nodes[1:]) if (per_bin and interp == "step") else nodes
        th = {"lookback_time": lb, "sfh": sfh, "logzsol": jnp.array([-0.3])}
        c = CSPBasis(ssp, theta=th, zh_const=True, sfh_interp=interp, add_neb=False,
                     add_dust=True, add_diffuse_dust=True, add_igm=True, verbose=False,
                     cosmo=Cosmology.planck18())
        t = {k: jnp.asarray(v) for k, v in c.theta_init.items()}
        t.update(zred=jnp.array([1.2]), logmass=jnp.array([10.0]),
                 diffuse_tau_kc=jnp.array([0.4]), tau_pow=jnp.array([0.6]))
        ph = Photometry(filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
                        flux=np.ones(5), uncertainty=np.ones(5), name="p")
        ph.setup_for_model(c.wave, zred=1.2)
        out[f"{interp}/W"] = np.asarray(c.calculate_ssp_weights(t))
        out[f"{interp}/spec"] = np.asarray(c.get_spectrum(t))
        out[f"{interp}/maggies"] = np.asarray(c.predict(t, [ph])["p"])
    np.savez(sys.argv[3], **out)
""")


@f64_only
def test_1_bin_means_reproduce_the_base_commits_step_model(tmp_path):
    grid = str(require_test_grid())
    base = tmp_path / "base"
    base.mkdir()
    arch = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", BASE], capture_output=True)
    if arch.returncode != 0:
        pytest.skip(f"base commit {BASE[:12]} not in this clone: base comparison did NOT run")
    subprocess.run(["tar", "-x", "-C", str(base)], input=arch.stdout, check=True)
    script = tmp_path / "step.py"
    script.write_text(_SCRIPT)
    res = {}
    for name, tree in (("base", base), ("head", REPO_ROOT)):
        env = dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu",
                   CERIDWEN_NO_UPDATE_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("CERIDWEN_X64", None)
        out = tmp_path / f"{name}.npz"
        r = subprocess.run([sys.executable, str(script), str(tree / "ceridwen"), grid, str(out),
                            name], env=env, cwd=str(tmp_path), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        res[name] = np.load(out)
    assert sorted(res["base"].files) == sorted(res["head"].files)
    for k in res["base"].files:
        a, b = res["base"][k], res["head"][k]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k


# --- 2 ---------------------------------------------------------------------------------------

def _prospector_masses(logmass, logsfr_ratios, agebins):
    """prospect.models.transforms.logsfr_ratios_to_masses (prospector 40e2614, lines 184-198),
    transcribed: masses formed per bin from log10(SFR_j / SFR_{j+1}), j = 0 the youngest bin."""
    nbins = agebins.shape[0]
    sratios = 10 ** np.clip(logsfr_ratios, -10, 10)
    dt = (10 ** agebins[:, 1] - 10 ** agebins[:, 0])
    coeffs = np.array([(1. / np.prod(sratios[:i])) * (np.prod(dt[1: i + 1]) / np.prod(dt[: i]))
                       for i in range(nbins)])
    m1 = (10 ** logmass) / coeffs.sum()
    return m1 * coeffs


@f64_only
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_2_per_bin_ratios_are_prospectors_continuity_sfh(seed):
    edges = np.asarray(LB) * 1e9
    edges[0] = 1.0                                   # Prospector's youngest edge, log10(1 yr) = 0
    r = np.random.default_rng(seed).normal(0.0, 0.6, edges.size - 2)
    sfr = np.asarray(logsfr_ratios_to_sfh(jnp.asarray(r), sfh_times_yr=edges))
    assert sfr.shape == (edges.size - 1,)
    assert abs(np.sum(sfr * np.diff(edges)) - 1.0) < 1e-14
    agebins = np.log10(np.stack([edges[:-1], edges[1:]], axis=1))
    ref = _prospector_masses(10.0, r, agebins)
    assert np.max(np.abs(1e10 * sfr * np.diff(edges) - ref) / ref) < 1e-13
    # the ratios are those of adjacent BIN SFRs
    assert np.allclose(np.log10(sfr[:-1] / sfr[1:]), r, rtol=0, atol=1e-13)


def test_2b_per_node_ratios_keep_the_trapezoidal_normalisation():
    t = np.asarray(LB) * 1e9
    s = np.asarray(logsfr_ratios_to_sfh(jnp.asarray([0.2, -0.1, 0.3, 0.0, -0.4, 0.1]),
                                        sfh_times_yr=t))
    assert s.shape == (t.size,)
    assert abs(np.sum(0.5 * (s[:-1] + s[1:]) * np.diff(t)) - 1.0) < 1e-12
    with pytest.raises(ValueError, match="give 7 nodes for one SFR per bin"):
        logsfr_ratios_to_sfh(jnp.zeros(5), sfh_times_yr=t[:4])


# --- 3 ---------------------------------------------------------------------------------------

@f64_only
def test_3_no_direction_of_the_ratios_is_invisible(ssp):
    csp = CSPBasis(ssp, lookback_time=LB, zh_const=True, **KW)
    t = np.asarray(csp.sfh_times)
    lz = jnp.array([-0.3])

    def W(r):
        return jnp.ravel(csp.calculate_ssp_weights(
            {"sfh": logsfr_ratios_to_sfh(r, sfh_times_yr=t), "logzsol": lz}))
    r0 = jnp.asarray([0.3, -0.2, 0.4, -0.1, 0.2])
    sv = np.linalg.svd(np.asarray(jax.jacfwd(W)(r0)), compute_uv=False)
    assert sv.size == 5 and sv.min() > 1e-6 * sv.max(), sv
    # the old per-node construction (n_time-1 node ratios, bin means in the kernel) had a null
    # vector: +a, -a, +a, ... on the nodes leaves every bin mean, and so W, unchanged
    nodes = np.asarray(logsfr_ratios_to_sfh(jnp.asarray([0.3, -0.2, 0.4, -0.1, 0.2, 0.1]),
                                            sfh_times_yr=t))
    alt = 0.4 * nodes.min() * (-1.0) ** np.arange(nodes.size)
    for n in (nodes, nodes + alt):
        assert np.allclose(0.5 * (n[:-1] + n[1:]), 0.5 * (nodes[:-1] + nodes[1:]), rtol=1e-15)


# --- 4 ---------------------------------------------------------------------------------------

def test_4_refusals_and_shortcut_shapes(ssp):
    n = LB.size
    with pytest.raises(ValueError, match=r"one value per NODE.*FastStepBasis.*0\.5\*\(sfh"):
        CSPBasis(ssp, theta={"lookback_time": LB, "sfh": jnp.ones(n), "logzsol": jnp.array([0.0])},
                 zh_const=True, **KW)
    with pytest.raises(ValueError, match="sfh_per_bin=False with sfh_interp='step'"):
        CSPBasis(ssp, lookback_time=LB, zh_const=True, sfh_per_bin=False, **KW)
    step = CSPBasis(ssp, lookback_time=LB, zh_const=True, **KW)
    assert step.theta_init["sfh"].shape == (n - 1,) and step.sfh_per_bin
    lin = CSPBasis(ssp, lookback_time=LB, zh_const=True, sfh_interp="linear", **KW)
    assert lin.theta_init["sfh"].shape == (n,) and not lin.sfh_per_bin
    linb = CSPBasis(ssp, lookback_time=LB, zh_const=True, sfh_interp="linear", sfh_per_bin=True,
                    **KW)
    assert linb.theta_init["sfh"].shape == (n - 1,)
    # an sfh transform with one ratio too many (the old per-node count) is refused at setup
    t = np.asarray(step.sfh_times)
    ph = Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1e-9, 1e-9], uncertainty=[1e-10, 1e-10],
                    name="p")
    priors = {"logsfr_ratios": StudentT(mean=0.0, scale=0.3, df=2.0),
              "logmass": Uniform(low=9.0, high=11.0), "logzsol": Uniform(low=-1.0, high=0.0)}
    sfh = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)}
    with pytest.raises(ValueError, match=rf"give logsfr_ratios_to_sfh {n - 2} ratios"):
        SedModel(step, [ph], zred=0.5, priors=priors, transforms=sfh,
                 free_param_init={"logsfr_ratios": jnp.zeros(n - 1), "logmass": jnp.array([10.0])})
    m = SedModel(step, [ph], zred=0.5, priors=priors, transforms=sfh,
                 free_param_init={"logsfr_ratios": jnp.zeros(n - 2), "logmass": jnp.array([10.0])})
    assert np.all(np.isfinite(np.asarray(m.predict(dict(m.theta_init))["p"])))
