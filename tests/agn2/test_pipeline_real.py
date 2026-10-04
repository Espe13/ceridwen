"""The REAL AGN2 components through CSPBasis + SedModel (session VAL, Phase B/C). Each config
skips loudly while one of its components (or its data) is missing, so every writer's branch runs
its own rows and the merged branch runs them all.

V-PIPE-1  zero amplitudes (fagn = agn_elum = blr_lum = 0; every other AGN2 key off its default):
          every prediction (Spectrum, Photometry, Lines) byte-identical to the CSP without agn=,
          with and without IGM.
V-PIPE-2  jit(value_and_grad) of a log-likelihood: finite, non-zero in every AGN2 theta key of
          the config, also at the grid / prior edges; vmap (W = 64) == loop (rel 1e-6 value,
          1e-4 gradient: float32 einsums reorder under vmap).
V-PIPE-3  linear in the amplitude of each component at the prediction level.
V-PIPE-4  "corrected" with DL07: L_absorbed (CSPBasis._agn_parts) = galaxy-absorbed nuclear
          continuum (from the component's own nuclear_parts) + absorbed line light; the AGN
          luminosity added = its nuclear luminosity minus nothing (DL07 re-emits), rel 2e-4.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ref  # noqa: E402
import test_emission_lines as T  # noqa: E402
import test_scaffold as S  # noqa: E402

import ceridwen.agn as agnmod  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402

CFG = {
    "skirtor-corrected": (("skirtor16", "polar"),
                          dict(agn_mode="corrected", torus_model="skirtor16", polar_dust=True,
                               lines=False)),
    "skirtor-cigale": (("skirtor16", "polar"),
                       dict(agn_mode="cigale", torus_model="skirtor16", polar_dust=True)),
    "nlr-corrected": (("feltre16",), dict(agn_mode="corrected", torus=False, disk=False,
                                          nlr_model="feltre16")),
    "nlr-cigale": (("feltre16",), dict(agn_mode="cigale", torus=False, nlr_model="feltre16")),
    "blr-corrected": (("blr",), dict(agn_mode="corrected", torus=False, disk=False, lines=False,
                                     blr=True)),
    "blr-cigale": (("blr",), dict(agn_mode="cigale", torus=False, blr=True)),
    "all-corrected": (("skirtor16", "polar", "feltre16", "blr"), S.ALL_ON),
    "all-cigale": (("skirtor16", "polar", "feltre16", "blr"), dict(S.ALL_ON, agn_mode="cigale")),
}
ON = {"fagn": 0.3, "skirtor_t": 5.0, "skirtor_pl": 0.5, "skirtor_q": 1.2, "skirtor_oa": 50.0,
      "skirtor_i": 20.0, "polar_ebv": 0.2, "polar_temp": 150.0, "polar_emis": 1.5,
      "agn_elum": 3e-3, "agn_eline_sigma": 150.0, "nlr_logu": -2.2, "nlr_logz": -0.3,
      "nlr_alpha": -1.4, "nlr_xid": 0.4, "nlr_lognh": 2.5, "blr_lum": 1e-2, "blr_sigma": 1800.0}
AMPL = ("fagn", "agn_elum", "blr_lum")
EDGES = {"skirtor_t": (3.0, 11.0), "skirtor_pl": (0.0, 1.5), "skirtor_q": (0.0, 1.5),
         "skirtor_oa": (10.0, 80.0), "skirtor_i": (0.0, 90.0), "polar_ebv": (0.0, 1.0),
         "polar_temp": (50.0, 300.0), "nlr_logu": (-4.0, -1.5), "nlr_logz": (-2.0, 0.2),
         "nlr_alpha": (-2.0, -1.2), "nlr_xid": (0.1, 0.5), "nlr_lognh": (2.0, 4.0),
         "blr_sigma": (400.0, 5000.0)}

_CACHE = {}


def _csp(name, *, igm=False, dem=False):
    comps, kw = CFG[name]
    test = f"V-PIPE[{name}]"
    for c in comps:
        try:
            agnmod._agn2_class(c)
        except NotImplementedError as e:
            _ref.missing(f"component {c} ({e})", test)
    key = (name, igm, dem)
    if key in _CACHE:
        return _CACHE[key]
    path = S._need()
    cosmo = Cosmology.planck18()
    try:
        csp = CSPBasis(SSPData.load(str(path)),
                       lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
                       zh_const=True, sfh_interp="step", add_dust=dem, add_diffuse_dust=True,
                       add_neb=True, add_igm=igm, add_dust_emission=dem, verbose=False,
                       cosmo=cosmo, sps_home=os.environ["SPS_HOME"], agn=AGNConfig(**kw))
    except (FileNotFoundError, OSError) as e:
        _ref.missing(f"AGN2 data ({e})", test)
    _CACHE[key] = csp
    return csp


def _vals(csp, d):
    return {k: jnp.array([float(v)]) for k, v in d.items() if k in csp._agn.param_names}


@pytest.mark.parametrize("igm", [False, True])
@pytest.mark.parametrize("name", list(CFG))
def test_V_PIPE_1_zero_amplitude_byte_identical(name, igm):
    on, _, _, _ = S._model(_csp(name, igm=igm))
    off, _, _, _ = S._model(S._csp(None, igm=igm))
    zero = {k: (0.0 if k in AMPL else v) for k, v in ON.items()}
    base = off.predict_jit(T._theta(off))
    got = on.predict_jit({**T._theta(on), **_vals(on.csp, zero)})
    assert set(got) == set(base)
    assert S._bytes(got) == S._bytes(base)
    lit = on.predict_jit({**T._theta(on), **_vals(on.csp, ON)})
    for k in base:
        if k == "lines" and not on.csp._agn.model.line_sets:
            continue                          # continuum only: line fluxes do not move
        assert np.any(np.asarray(lit[k]) != np.asarray(base[k])), (name, k)


@pytest.mark.parametrize("name", list(CFG))
def test_V_PIPE_2_jit_vmap_grad(name):
    model, _, _, _ = S._model(_csp(name, igm=True))
    vals = _vals(model.csp, ON)
    f, x0, pos = S._lnlike_factory(model, {k: float(v[0]) for k, v in vals.items()})
    vg = jax.jit(jax.value_and_grad(f))
    v, g = vg(jnp.asarray(x0))
    g = np.asarray(g)
    assert np.isfinite(float(v)) and np.all(np.isfinite(g))
    for k in vals:
        assert g[pos[k]] != 0.0, (name, k)
    for k, (lo, hi) in EDGES.items():
        if k not in pos:
            continue
        for x in (lo, hi):
            xe = x0.copy()
            xe[pos[k]] = x
            ve, ge = vg(jnp.asarray(xe))
            assert np.isfinite(float(ve)) and np.all(np.isfinite(np.asarray(ge))), (name, k, x)
    rng = np.random.default_rng(2)
    W = 64
    X = np.repeat(x0[None], W, axis=0)
    for k, (lo, hi) in EDGES.items():
        if k in pos:
            X[:, pos[k]] = rng.uniform(lo, hi, W)
    for k in AMPL:
        if k in pos:
            X[:, pos[k]] = x0[pos[k]] * rng.uniform(0.1, 2.0, W)
    bv, bg = jax.jit(jax.vmap(jax.value_and_grad(f)))(jnp.asarray(X))
    assert np.all(np.isfinite(np.asarray(bv))) and np.all(np.isfinite(np.asarray(bg)))
    for i in (0, W // 2, W - 1):
        lv, lg = vg(jnp.asarray(X[i]))
        assert float(bv[i]) == pytest.approx(float(lv), rel=1e-6)
        np.testing.assert_allclose(np.asarray(bg[i]), np.asarray(lg), rtol=1e-4,
                                   atol=1e-5 * np.abs(np.asarray(lg)).max())


@pytest.mark.parametrize("name", list(CFG))
def test_V_PIPE_3_linear_in_amplitudes(name):
    model, _, _, _ = S._model(_csp(name))
    present = [a for a in AMPL if a in model.csp._agn.param_names]

    def pred(scale):
        d = dict(ON, **{a: ON[a] * scale for a in present})
        return {k: np.asarray(v, np.float64) for k, v in
                model.predict({**T._theta(model), **_vals(model.csp, d)}).items()}

    p0, p1, p2 = pred(0.0), pred(1.0), pred(2.0)
    for k in p0:
        d1, d2 = p1[k] - p0[k], p2[k] - p0[k]
        if k == "lines" and not model.csp._agn.model.line_sets:
            assert np.all(d1 == 0) and np.all(d2 == 0), (name, k)
            continue
        assert np.any(d1 != 0), (name, k)
        np.testing.assert_allclose(d2, 2 * d1, rtol=1e-4,
                                   atol=1e-4 * np.abs(d2).max() + 1e-6 * np.abs(p0[k]).max())


@pytest.mark.parametrize("name", ["skirtor-corrected", "all-corrected", "blr-corrected",
                                  "nlr-corrected"])
def test_V_PIPE_4_energy_balance(name):
    csp = _csp(name, dem=True)
    assert csp._agn.energy_balance
    model, _, _, _ = S._model(csp)
    _, mt = S._mt(model, {k: float(v[0]) for k, v in _vals(csp, ON).items()},
                  diffuse_tau_kc=jnp.array([0.5]))
    hook = csp._agn
    th = hook.filled(mt)
    W = csp.calculate_ssp_weights(theta=mt)
    msum = float(jnp.sum(W))
    ctx = csp._agn_context(mt, W)
    wave = np.asarray(csp.wave)
    w = np.asarray(hook._w)
    dc = S._diffuse(csp, mt)
    nuc = (np.asarray(hook.nuclear_parts(th, ctx)["nuclear"])
           if "skirtor" in hook.model.components else np.zeros_like(wave))
    exp_abs = (nuc * (1 - dc)) @ w
    if hook.model.lines is not None:
        L = np.asarray(hook.model.lines.line_luminosity(th)) * msum
        exp_abs += np.sum(L * (1 - np.interp(hook.model.lines.line_wave, wave, dc)))
    if hook.model.blr is not None:
        L = np.asarray(hook.model.blr.line_luminosity(th)) * msum
        tp = (np.interp(hook.model.blr.line_wave, wave,
                        np.asarray(hook.model.polar.transmission(th)))
              if hook.model.polar is not None else 1.0)
        exp_abs += np.sum(L * tp * (1 - np.interp(hook.model.blr.line_wave, wave, dc)))
    parts = csp._agn_parts(mt)
    assert float(parts["L_absorbed"]) == pytest.approx(exp_abs, rel=1e-10)
    agn = np.asarray(csp._agn_rest(mt))
    assert agn @ w == pytest.approx((nuc * dc) @ w + exp_abs, rel=2e-4)
