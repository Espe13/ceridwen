"""The AGN2 contract (``ceridwen/agn/__init__.py``) checked on the REAL component classes
(session VAL). Each test skips loudly while the component's module or data is missing.

V-CON-1  param_names == AGN2_THETA_KEYS[name]; default_params() has exactly those keys, finite.
V-CON-2  outputs: float64, the documented shapes, finite at the defaults.
V-CON-3  zero amplitude (fagn, agn_elum, blr_lum = 0) gives exactly 0; linear in the amplitude.
V-CON-4  jit == eager to max|diff| <= 1e-14 max|value| (not byte-equal: XLA fuses a*b + c into an
         FMA under jit on CPU, DISPUTE_TORUS_r1, upheld) and vmap (W = 64) == loop (rel 1e-12).
V-CON-5  jax.grad of a scalar of every traced output is finite in every theta key of the
         component, at the defaults.
V-CON-6  line sets: region / width_key; line_wave equals FSPS's emlines_info.dat value of each
         line_names entry; rows_for maps every own line_wave back to itself.
V-CON-7  modes: "fsps" and "prospector" are refused by the constructor (ValueError).
V-CON-8  float32 theta in, float64 out (common prompt rule 7).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ref  # noqa: E402

import ceridwen.agn as agnmod  # noqa: E402
from ceridwen.agn import AGN2_THETA_KEYS, AGNContext  # noqa: E402

WAVE = np.geomspace(500.0, 1.0e7, 1200)
AMPL = {"skirtor16": "fagn", "feltre16": "agn_elum", "blr": "blr_lum"}
ON_AMPL = {"fagn": 0.2, "agn_elum": 1e-3, "blr_lum": 1e-2}


def _cls(name, test):
    try:
        return agnmod._agn2_class(name)
    except NotImplementedError as e:
        _ref.missing(f"component {name} ({e})", test)


def _build(name, test, mode="corrected"):
    cls = _cls(name, test)
    try:
        if name == "skirtor16":
            return cls(WAVE, mode=mode, R=None, file=None)
        if name == "polar":
            return cls(WAVE, mode=mode, law=agnmod.POLAR_LAW_DEFAULT)
        return cls(mode=mode, file=None, sps_home=os.environ.get("SPS_HOME"))
    except (FileNotFoundError, OSError) as e:
        _ref.missing(f"data of {name} ({e})", test)
    except ValueError as e:
        if "not found" in str(e) or "MISSING" in str(e) or "No such file" in str(e) \
                or "SPS_HOME" in str(e):
            _ref.missing(f"data of {name} ({e})", test)
        raise


def _ctx(L=3.0):
    return AGNContext(jnp.asarray(WAVE), jnp.asarray(L), None, jnp.ones(WAVE.size),
                      "corrected", jnp.asarray(2.0))


def _theta(comp, **over):
    th = {k: jnp.asarray(float(v)) for k, v in comp.default_params().items()}
    th.update({k: jnp.asarray(float(v)) for k, v in over.items()})
    return th


def _outputs(name, comp, th):
    """Every traced output of the component as a dict of arrays."""
    if name == "skirtor16":
        p = comp.rest_parts(th, _ctx())
        return {"disk": p["disk"], "dust": p["dust"]}
    if name == "polar":
        t = comp.transmission(th)
        disk = jnp.asarray(np.exp(-0.5 * (np.log(WAVE / 2000.0) / 0.7) ** 2))
        return {"T": t, "ir": comp.reemission(th, disk, t, _ctx())}
    return {"L": comp.line_luminosity(th), "sigma": jnp.asarray(comp.sigma_kms(th))}


NAMES = ["skirtor16", "polar", "feltre16", "blr"]


@pytest.mark.parametrize("name", NAMES)
def test_V_CON_1_keys(name):
    comp = _build(name, f"V-CON-1[{name}]")
    assert tuple(comp.param_names) == AGN2_THETA_KEYS[name] or (
        set(comp.param_names) == set(AGN2_THETA_KEYS[name]))
    d = comp.default_params()
    assert set(d) == set(AGN2_THETA_KEYS[name])
    assert all(np.isfinite(float(v)) for v in d.values())


@pytest.mark.parametrize("name", NAMES)
def test_V_CON_2_outputs(name):
    comp = _build(name, f"V-CON-2[{name}]")
    th = _theta(comp, **({AMPL[name]: ON_AMPL[AMPL[name]]} if name in AMPL else {}))
    out = _outputs(name, comp, th)
    for k, v in out.items():
        v = jnp.asarray(v)
        assert v.dtype == jnp.float64, (k, v.dtype)
        assert np.all(np.isfinite(np.asarray(v))), k
    if name in ("skirtor16",):
        assert out["disk"].shape == (WAVE.size,) and out["dust"].shape == (WAVE.size,)
    if name == "polar":
        t = np.asarray(out["T"])
        assert t.shape == (WAVE.size,) and np.all((t >= 0) & (t <= 1))
    if name in ("feltre16", "blr"):
        assert out["L"].shape == (len(comp.line_names),) and out["sigma"].shape == ()


@pytest.mark.parametrize("name", ["skirtor16", "feltre16", "blr"])
def test_V_CON_3_zero_and_linear(name):
    comp = _build(name, f"V-CON-3[{name}]")
    a = AMPL[name]
    x = ON_AMPL[a]
    o0 = _outputs(name, comp, _theta(comp, **{a: 0.0}))
    o1 = _outputs(name, comp, _theta(comp, **{a: x}))
    o2 = _outputs(name, comp, _theta(comp, **{a: 2 * x}))
    for k in o0:
        if k == "sigma":
            continue
        assert np.all(np.asarray(o0[k]) == 0.0), k
        assert np.any(np.asarray(o1[k]) != 0.0), k
        np.testing.assert_allclose(np.asarray(o2[k]), 2 * np.asarray(o1[k]), rtol=1e-13,
                                   atol=0)


@pytest.mark.parametrize("name", NAMES)
def test_V_CON_4_jit_vmap(name):
    comp = _build(name, f"V-CON-4[{name}]")
    keys = list(comp.default_params())
    th = _theta(comp, **({AMPL[name]: ON_AMPL[AMPL[name]]} if name in AMPL else {}))

    def f(t):
        return _outputs(name, comp, t)

    eager = f(th)
    jitted = jax.jit(f)(th)
    for k in eager:
        e, j = np.asarray(eager[k]), np.asarray(jitted[k])
        assert e.shape == j.shape and e.dtype == j.dtype, k
        assert np.max(np.abs(e - j)) <= 1e-14 * max(np.max(np.abs(e)), 1e-300), k
    rng = np.random.default_rng(3)
    W = 64
    batch = {k: jnp.asarray(float(th[k]) * (1.0 + 0.02 * rng.standard_normal(W))
                            + (0.01 * rng.standard_normal(W) if float(th[k]) == 0 else 0.0))
             for k in keys}
    # keep amplitudes non-negative and the defaults' sign
    vm = jax.jit(jax.vmap(f))(batch)
    for i in (0, W // 2, W - 1):
        one = f({k: batch[k][i] for k in keys})
        for k in one:
            np.testing.assert_allclose(np.asarray(vm[k][i]), np.asarray(one[k]), rtol=1e-12,
                                       atol=1e-300)


@pytest.mark.parametrize("name", NAMES)
def test_V_CON_5_grad_finite(name):
    comp = _build(name, f"V-CON-5[{name}]")
    th = _theta(comp, **({AMPL[name]: ON_AMPL[AMPL[name]]} if name in AMPL else {}))

    def s(t):
        o = _outputs(name, comp, t)
        return sum(jnp.sum(jnp.log1p(jnp.abs(v))) for v in o.values())

    g = jax.grad(s)(th)
    for k in comp.param_names:
        assert np.isfinite(float(g[k])), (name, k, g[k])


@pytest.mark.parametrize("name", ["feltre16", "blr"])
def test_V_CON_6_line_sets(name):
    comp = _build(name, f"V-CON-6[{name}]")
    assert comp.region == ("nlr" if name == "feltre16" else "blr")
    assert comp.width_key == ("agn_eline_sigma" if name == "feltre16" else "blr_sigma")
    from ceridwen.agn.lines import read_emlines_info
    wave, names, _ = read_emlines_info(os.environ.get("SPS_HOME"))
    for n, w in zip(comp.line_names, comp.line_wave):
        assert n in names, n
        assert w == wave[names.index(n)], (n, w)
    rows = comp.rows_for(comp.line_wave)
    assert list(rows) == list(range(len(comp.line_names)))
    assert len(set(comp.line_names)) == len(comp.line_names)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mode", ["fsps", "prospector"])
def test_V_CON_7_reproduction_modes_refused(name, mode):
    _cls(name, f"V-CON-7[{name}]")
    with pytest.raises(ValueError):
        _build(name, f"V-CON-7[{name}]", mode=mode)


@pytest.mark.parametrize("name", NAMES)
def test_V_CON_8_float64_for_float32_theta(name):
    """Common prompt rule 7 (float64 for everything new): a theta given in float32 (e.g. prior
    draws cast by a caller) still gives float64 outputs; review finding VERDICT_BLR_r1."""
    comp = _build(name, f"V-CON-8[{name}]")
    th = _theta(comp, **({AMPL[name]: ON_AMPL[AMPL[name]]} if name in AMPL else {}))
    th32 = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in th.items()}
    for k, v in _outputs(name, comp, th32).items():
        assert jnp.asarray(v).dtype == jnp.float64, (name, k, jnp.asarray(v).dtype)
