"""AGN screen (ceridwen/agn/screen.py): dust4 power law combined with the diffuse curve,
the Prospector ``fake_fsps.attenuate`` dust2 port, setup-time refusals, and AGN_PLAN §4 B8
(jit / vmap / grad in ``dust4``, ``dust4_index``, ``fagn_bbb``) for disc + screen.

Component-level: stub ``AGNContext``, no CSP.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ceridwen.agn import AGNContext
from ceridwen.agn.disk import TempleDisk
from ceridwen.agn.screen import AGNScreen, ProspectorDiffuseCurve

_LOG = np.geomspace(1.0, 1e8, 3000)
WAVE = np.concatenate([_LOG[_LOG < 5000.0], np.arange(5001.0, 6000.0, 2.0), _LOG[_LOG > 6000.0]])
WAVE = np.sort(np.append(WAVE, 5500.0))
I5500 = int(np.flatnonzero(WAVE == 5500.0)[0])
GAL = 1e-8 * np.exp(-0.5 * (np.log10(WAVE / 5500.0) / 0.8) ** 2) * (1.0 + 0.01 * np.sin(WAVE / 3.0))


def _ctx(diffuse, gal=GAL):
    return AGNContext(wave=jnp.asarray(WAVE), L_bol_star=jnp.asarray(1.0),
                      gal_nodust_young_old=jnp.asarray(gal),
                      diffuse_curve=jnp.asarray(diffuse), mode="prospector")


def _np_fake_fsps_dust2(lam, dust_type, dust_index, dust2):
    """Independent numpy transcription of prospect/sources/fake_fsps.py::attenuate
    (91f8e13) with dust1 = dust4 = 0, used to check the JAX port."""
    lamv = 5500.0
    if dust_type == 0:
        a = (lam / lamv) ** dust_index * dust2
    else:
        cal00 = np.where(lam > 6300.0, 1.17 * (-1.857 + 1.04 * (1e4 / lam)) + 1.78,
                         1.17 * (-2.156 + 1.509 * (1e4 / lam) - 0.198 * (1e4 / lam) ** 2
                                 + 0.011 * (1e4 / lam) ** 3) + 1.78)
        cal00 = np.clip(cal00 / 0.44 / 4.05, 0.0, np.inf)
        if dust_type == 2:
            a = cal00 * dust2
        elif dust_type == 4:
            eb = 0.85 - 1.9 * dust_index
            drude = eb * (lam * 350.0) ** 2 / ((lam ** 2 - 2175.0 ** 2) ** 2 + (lam * 350.0) ** 2)
            a = dust2 * (cal00 + drude / 4.05) * (lam / lamv) ** dust_index
        else:
            k = np.zeros_like(lam)
            m = lam / 1e4
            w1, w2 = np.abs(lam - 1500).argmin(), np.abs(lam - 6000).argmin()
            k[w1:w2] = -5.726 + 4.004 / m[w1:w2] - 0.525 / m[w1:w2] ** 2 + 0.029 / m[w1:w2] ** 3 + 2.505
            k[:w1] = k[w1]
            w1, w2 = np.abs(lam - 6000).argmin(), np.abs(lam - 28500).argmin()
            k[w1:w2] = (-2.672 - 0.010 / m[w1:w2] + 1.532 / m[w1:w2] ** 2 - 0.412 / m[w1:w2] ** 3
                        + 2.505 - 0.036221981)
            a = dust2 * k / 2.505
    return np.exp(-a)


@pytest.mark.parametrize("dust_type", [0, 2, 4, 6])
@pytest.mark.parametrize("dust2", [0.0, 0.3, 1.5])
def test_prospector_diffuse_curve_matches_transcription(dust_type, dust2):
    curve = ProspectorDiffuseCurve(WAVE, dust_type=dust_type)
    got = np.asarray(curve.transmission({"dust2": dust2, "dust_index": -0.7}))
    ref = _np_fake_fsps_dust2(WAVE, dust_type, -0.7, dust2)
    normal = ref >= np.finfo(float).tiny
    np.testing.assert_allclose(got[normal], ref[normal], rtol=1e-12, atol=0)
    assert np.all(got[~normal] < np.finfo(float).tiny)
    if dust2 == 0.0:
        assert np.all(got == 1.0)


def test_dust4_analytic_values():
    """transmission = diffuse * exp(-(lam/5500)**dust4_index * dust4): at 5500 A it is
    diffuse * exp(-dust4) for any index."""
    rng = np.random.default_rng(1)
    diffuse = rng.uniform(0.2, 1.0, WAVE.size)
    s = AGNScreen(WAVE, dust4_type=1)
    for d4, idx in ((0.5, -1.8), (2.0, -0.8), (4.0, -1.0)):
        t = np.asarray(s.transmission({"dust4": d4, "dust4_index": idx}, _ctx(diffuse)))
        assert t[I5500] == pytest.approx(diffuse[I5500] * np.exp(-d4), rel=1e-15)
        ref = diffuse * np.exp(-(WAVE / 5500.0) ** idx * d4)
        np.testing.assert_allclose(t, ref, rtol=1e-12, atol=1e-300)


def test_zero_dust4_and_type0_are_exactly_the_diffuse_curve():
    rng = np.random.default_rng(2)
    diffuse = rng.uniform(0.2, 1.0, WAVE.size)
    s1 = AGNScreen(WAVE, dust4_type=1)
    s0 = AGNScreen(WAVE, dust4_type=0)
    t1 = np.asarray(s1.transmission({"dust4": 0.0, "dust4_index": -1.3}, _ctx(diffuse)))
    t0 = np.asarray(s0.transmission({"dust4": 3.0, "dust4_index": -1.3}, _ctx(diffuse)))
    assert t1.tobytes() == diffuse.tobytes()
    assert t0.tobytes() == diffuse.tobytes()
    assert s0.param_names == () and s1.param_names == ("dust4", "dust4_index")
    assert s1.default_params() == {"dust4": 0.0, "dust4_index": -1.0}


def test_attenuate_is_spec_times_transmission():
    diffuse = _np_fake_fsps_dust2(WAVE, 0, -0.7, 1.0)
    s = AGNScreen(WAVE, dust4_type=1)
    th = {"dust4": 0.5, "dust4_index": -1.8}
    spec = jnp.asarray(GAL)
    a = np.asarray(s.attenuate(spec, th, _ctx(diffuse)))
    assert np.array_equal(a, np.asarray(spec * s.transmission(th, _ctx(diffuse))))


def test_setup_refusals():
    with pytest.raises(NotImplementedError, match="dust4_type=2"):
        AGNScreen(WAVE, dust4_type=2)
    for bad in (3, -1, True, "1"):
        with pytest.raises(ValueError):
            AGNScreen(WAVE, dust4_type=bad)
    for dt in (1, 3, 5):
        with pytest.raises(NotImplementedError, match="fake_fsps"):
            ProspectorDiffuseCurve(WAVE, dust_type=dt)
    with pytest.raises(ValueError):
        ProspectorDiffuseCurve(WAVE, dust_type=7)
    with pytest.raises(ValueError, match="positive"):
        AGNScreen(np.array([0.0, 1.0, 2.0]), dust4_type=1)


# --------------------------------------------------------------------------- B8
DISK = TempleDisk(WAVE, mode="prospector")
SCREEN = AGNScreen(WAVE, dust4_type=1)
DIFF = ProspectorDiffuseCurve(WAVE, dust_type=0)


def _agn(fb, d4, d4i, dust2):
    th = {"fagn_bbb": fb, "dust4": d4, "dust4_index": d4i, "dust2": dust2, "dust_index": -0.7}
    ctx = _ctx(DIFF.transmission(th))
    return SCREEN.attenuate(DISK.rest_spectrum(th, ctx), th, ctx)


def _loss(p):
    # a smooth scalar of the attenuated disc, restricted to pixels that stay normal numbers
    spec = _agn(p[0], p[1], p[2], p[3])
    return jnp.sum(spec[(WAVE > 2000.0) & (WAVE < 1e6)]) / 1e-8


def test_B8_jit_matches_eager():
    """Not bitwise: XLA fuses pow/exp under jit (measured max rel 5.7e-14, at 15 A where
    the dust4 optical depth is ~400 and exp amplifies the last-ulp difference of tau)."""
    p = (0.8, 0.5, -1.2, 0.4)
    eager, jitted = np.asarray(_agn(*p)), np.asarray(jax.jit(_agn)(*p))
    assert np.array_equal(eager == 0, jitted == 0)
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=0)


@pytest.mark.parametrize("W", [64, 100])
def test_B8_vmap_matches_loop(W):
    rng = np.random.default_rng(3)
    P = np.stack([10 ** rng.uniform(-5, 2, W), rng.uniform(0, 4, W),
                  rng.uniform(-1.8, -0.8, W), rng.uniform(0, 2, W)], axis=1)
    batched = np.asarray(jax.jit(jax.vmap(lambda p: _agn(p[0], p[1], p[2], p[3])))(jnp.asarray(P)))
    single = jax.jit(_agn)
    loop = np.stack([np.asarray(single(*P[i])) for i in range(W)])
    np.testing.assert_allclose(batched, loop, rtol=1e-14, atol=1e-300)


@pytest.mark.parametrize("p", [
    (1e-5, 0.0, -1.8, 0.0),      # lower prior bounds (dust4 TopHat(0,4), index TopHat(-1.8,-0.8))
    (100.0, 4.0, -0.8, 2.0),     # upper prior bounds
    (1.0, 2.0, -1.3, 0.5),       # prior centre
    (1e-3, 0.5, -1.0, 1.0),
])
def test_B8_grad_finite_and_matches_fd(p):
    """jax.grad of the attenuated disc in fagn_bbb, dust4, dust4_index (and dust2) is
    finite at the prior bounds and centre and matches central differences."""
    p = np.asarray(p, dtype=float)
    g = np.asarray(jax.grad(_loss)(jnp.asarray(p)))
    assert np.all(np.isfinite(g))
    for k in range(4):
        h = 1e-6 * max(abs(p[k]), 1e-3)
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        fd = (float(_loss(jnp.asarray(pp))) - float(_loss(jnp.asarray(pm)))) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-5, abs=1e-9 * abs(float(_loss(jnp.asarray(p))))), k
