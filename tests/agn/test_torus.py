"""NenkovaTorus (ceridwen/agn/torus.py): AGN_PLAN §4 A1, A2, B1, B2, B4, B5, B8 at the
component level (the CSP wiring is tests/agn/test_hook*.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ceridwen.agn import AGNContext, AGNConfig
from ceridwen.agn.torus import (NenkovaTorus, TORUS_TAU_NODES, fsps_locate, load_nenkova,
                                resample_corrected, trapezoid_nu_weights)

TAUS = np.asarray(TORUS_TAU_NODES)
MODES = ("fsps", "prospector", "corrected")


def _ctx(wave, lbol=1.0, mode="corrected"):
    n = len(wave)
    return AGNContext(jnp.asarray(wave), jnp.asarray(lbol), None,
                      jnp.ones(n), mode)


@pytest.fixture(scope="module")
def tori(model_wave, torus_path):
    return {m: NenkovaTorus(model_wave, mode=m, torus_file=torus_path) for m in MODES}


# ------------------------------------------------------------------ the data file
def test_file_shapes_and_native_normalisation(torus_path):
    """The FSPS data file as the 23 Sep design read it: 9 taus, 125 rows, row 0 (100 A) all
    zero, 'unity' only to ~7e-4 on its own 125 points (0.99927-0.99933)."""
    tau, lam, spec = load_nenkova(torus_path)
    assert tau.tolist() == list(TORUS_TAU_NODES)
    assert lam[0] == 100.0 and lam[-1] == 3.6e8
    assert np.all(spec[0] == 0) and np.all(spec[1:] > 0)
    t = NenkovaTorus(np.geomspace(50.0, 4e8, 4000), mode="corrected", torus_file=torus_path)
    assert np.all((t.file_integrals > 0.99926) & (t.file_integrals < 0.99934)), t.file_integrals
    # the smooth (log-log) SED integrates to 0.989-0.996: the file's unity is a trapezoid artefact
    assert np.all((t.dense_integrals > 0.988) & (t.dense_integrals < 0.997)), t.dense_integrals


def test_trapezoid_weights_bitwise_dl07(model_wave):
    """The torus integrates with DustEmission's own weights (its __init__ expression)."""
    import os
    from ceridwen.dust.DustEmission import DustEmission
    sps = os.environ.get("SPS_HOME")
    if not sps:
        pytest.skip("$SPS_HOME unset (DustEmission reads the DL07 templates)")
    de = DustEmission(spec_lambda=jnp.asarray(model_wave), dust_file=sps)
    w = trapezoid_nu_weights(model_wave)
    assert np.asarray(w).tobytes() == np.asarray(de._trap_w).tobytes()


# ------------------------------------------------------------------ B1
def test_B1_unit_normalisation_corrected(tori, model_wave):
    """B1: sum(T_j w) == 1 for every tau node on the CERIDWEN grid; native integrals kept."""
    t = tori["corrected"]
    w = -np.asarray(trapezoid_nu_weights(model_wave))
    integ = np.asarray(t.templates).T @ w
    np.testing.assert_allclose(integ, 1.0, rtol=0, atol=2e-15)
    # recorded: the un-normalised integral on this grid, = the dense one (no truncation)
    np.testing.assert_allclose(t.native_integrals, t.dense_integrals, rtol=1e-5)
    print("native integrals on the test grid:", np.array2string(t.native_integrals, precision=6))


def test_B1_any_grid(torus_path):
    """B1 on other grids, coarse and truncated: still exactly 1; a truncating grid warns."""
    for wave in (np.geomspace(91.0, 1e8, 300), np.geomspace(912.0, 3e7, 5000)):
        t = NenkovaTorus(wave, mode="corrected", torus_file=torus_path)
        w = -np.asarray(trapezoid_nu_weights(wave))
        np.testing.assert_allclose(np.asarray(t.templates).T @ w, 1.0, rtol=0, atol=2e-15)
    with pytest.warns(UserWarning, match="truncated"):
        NenkovaTorus(np.geomspace(912.0, 3e5, 3000), mode="corrected", torus_file=torus_path)
    with pytest.raises(ValueError, match="does not overlap"):
        NenkovaTorus(np.linspace(10.0, 90.0, 50), mode="corrected", torus_file=torus_path)


# ------------------------------------------------------------------ B2
@pytest.mark.parametrize("mode", ("fsps", "corrected"))
def test_B2_node_exactness(tori, mode):
    """B2: at every tau node the template is the stored column, bit for bit."""
    t = tori[mode]
    for k, tau in enumerate(TAUS):
        assert np.asarray(t.template(tau)).tobytes() == np.asarray(t.templates[:, k]).tobytes(), tau


def test_B2_nodes_prospector(tori):
    """Prospector's scipy expression y_lo + slope (x - x_lo) is exact at tau = 5 only; at the
    other nodes it reproduces the column to rounding (as scipy does)."""
    t = tori["prospector"]
    assert np.asarray(t.template(5.0)).tobytes() == np.asarray(t.templates[:, 0]).tobytes()
    for k, tau in enumerate(TAUS):
        np.testing.assert_allclose(np.asarray(t.template(tau)), np.asarray(t.templates[:, k]),
                                   rtol=1e-14, atol=1e-16 * float(np.abs(t.templates).max()))


@pytest.mark.parametrize("mode", ("fsps", "corrected"))
def test_B2_clamping(tori, mode):
    """B2: no extrapolation in fsps/corrected (agn_dust.f90:21): below 5 -> the tau=5 column,
    above 150 -> the tau=150 column, exactly."""
    t = tori[mode]
    for tau in (0.1, 1.0, 4.999):
        assert np.asarray(t.template(tau)).tobytes() == np.asarray(t.templates[:, 0]).tobytes()
    for tau in (150.001, 1e3, 1e4):
        assert np.asarray(t.template(tau)).tobytes() == np.asarray(t.templates[:, -1]).tobytes()


def test_B2_no_negative_no_nan_corrected(tori):
    """B2: tau in [0.1, 1e4]: finite, never negative, in corrected mode (linear and log)."""
    t = tori["corrected"]
    taus = jnp.asarray(np.geomspace(0.1, 1e4, 257))
    T = jax.vmap(t.template)(taus)
    assert bool(jnp.all(jnp.isfinite(T))) and float(T.min()) >= 0.0


def test_B2_log_interp(model_wave, torus_path):
    """tau_interp='log' (corrected only): nodes reproduced to rounding, clamped, non-negative."""
    t = NenkovaTorus(model_wave, mode="corrected", tau_interp="log", torus_file=torus_path)
    for k, tau in enumerate(TAUS):
        np.testing.assert_allclose(np.asarray(t.template(tau)), np.asarray(t.templates[:, k]),
                                   rtol=1e-12, atol=0)
    T = jax.vmap(t.template)(jnp.asarray(np.geomspace(0.1, 1e4, 101)))
    assert bool(jnp.all(jnp.isfinite(T))) and float(T.min()) >= 0.0
    # midway in log between 10 and 20 is sqrt(200): the plain average of the two columns
    np.testing.assert_allclose(np.asarray(t.template(np.sqrt(200.0))),
                               0.5 * np.asarray(t.templates[:, 1] + t.templates[:, 2]),
                               rtol=1e-12, atol=1e-300)
    with pytest.raises(ValueError):
        AGNConfig(agn_mode="fsps", tau_interp="log")


def test_prospector_extrapolates(tori):
    """D6 reproduced in 'prospector' mode: linear EXTRApolation with the end segments
    (scipy interp1d fill_value='extrapolate'), independently re-derived here."""
    t = tori["prospector"]
    T = np.asarray(t.templates)
    for tau in (1.0, 2.5):
        expect = T[:, 0] + (T[:, 1] - T[:, 0]) / (10.0 - 5.0) * (tau - 5.0)
        np.testing.assert_allclose(np.asarray(t.template(tau)), expect, rtol=1e-12,
                                   atol=1e-15 * np.abs(T).max())
    expect = T[:, 7] + (T[:, 8] - T[:, 7]) / 50.0 * (1e4 - 100.0)
    got = np.asarray(t.template(1e4))
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-15 * np.abs(expect).max())
    assert got.min() < 0          # the documented failure mode of D6: negative flux


def test_fsps_locate():
    """FSPS locate.f90 semantics the fsps resampling relies on (1-based)."""
    xx = np.array([1.0, 2.0, 3.0, 4.0])
    assert fsps_locate(xx, 0.5) == 0
    assert fsps_locate(xx, 1.0) == 1
    assert fsps_locate(xx, 2.5) == 2
    assert fsps_locate(xx, 3.0) == 3
    assert fsps_locate(xx, 4.0) == 3          # x == xx(n) -> n-1
    assert fsps_locate(xx, 9.0) == 4


def test_corrected_resampling_zero_outside(torus_path):
    """D4 fixed: exactly 0 below the first positive template row (150 A) and above 3.6e8 A."""
    tau, lam, spec = load_nenkova(torus_path)
    wave = np.array([50.0, 100.0, 120.0, 149.9, 150.0, 1e4, 3.6e8, 4e8])
    T = resample_corrected(wave, lam, spec)
    assert np.all(T[:4] == 0) and np.all(T[-1] == 0)
    np.testing.assert_allclose(T[4], spec[1], rtol=1e-14)
    np.testing.assert_allclose(T[6], spec[-1], rtol=1e-14)


# ------------------------------------------------------------------ B4, B5
@pytest.mark.parametrize("mode", MODES)
def test_B4_linear_in_fagn(tori, model_wave, mode):
    """B4: rest_spectrum(2 f) == 2 rest_spectrum(f) bit for bit; fagn = 0 gives exact zeros."""
    t = tori[mode]
    ctx = _ctx(model_wave, lbol=3.7, mode=mode)
    a = np.asarray(t.rest_spectrum({"fagn": jnp.array([0.3]), "agn_tau": jnp.array([17.0])}, ctx))
    b = np.asarray(t.rest_spectrum({"fagn": jnp.array([0.6]), "agn_tau": jnp.array([17.0])}, ctx))
    assert (2.0 * a).tobytes() == b.tobytes()
    z = np.asarray(t.rest_spectrum({"fagn": jnp.array([0.0]), "agn_tau": jnp.array([17.0])}, ctx))
    assert not np.any(z)


@pytest.mark.parametrize("tau", (5.0, 7.5, 17.0, 150.0, 1.0, 1e4))
def test_B5_luminosity_corrected(tori, model_wave, tau):
    """B5: L_torus = fagn * L_bol to rtol 1e-10 (no screen), any agn_tau (clamped)."""
    t = tori["corrected"]
    L = 2.5e10
    ctx = _ctx(model_wave, lbol=L)
    s = t.rest_spectrum({"fagn": jnp.array([0.37]), "agn_tau": jnp.array([tau])}, ctx)
    w = -trapezoid_nu_weights(model_wave)
    np.testing.assert_allclose(float(jnp.dot(s, w)), 0.37 * L, rtol=1e-10)


# ------------------------------------------------------------------ B8 (component level)
@pytest.mark.parametrize("mode", MODES)
def test_B8_jit_vmap_grad(tori, model_wave, mode):
    """B8: jit == eager (to 1e-14: XLA CPU contracts (1-dj) a + dj b into FMAs under jit), vmap
    at W = 64 and 100 == the loop, finite non-zero gradients in fagn and agn_tau inside
    (5, 150) (off the nodes)."""
    t = tori[mode]
    ctx = _ctx(model_wave, lbol=1.0, mode=mode)
    w = -trapezoid_nu_weights(model_wave)

    def f(fagn, tau):
        return t.rest_spectrum({"fagn": fagn, "agn_tau": tau}, ctx)

    eager = f(jnp.array([0.2]), jnp.array([23.0]))
    np.testing.assert_allclose(np.asarray(jax.jit(f)(jnp.array([0.2]), jnp.array([23.0]))),
                               np.asarray(eager), rtol=1e-14, atol=0)
    rng = np.random.default_rng(0)
    for W in (64, 100):
        fa = jnp.asarray(rng.uniform(1e-3, 3.0, (W, 1)))
        ta = jnp.asarray(rng.uniform(5.0, 150.0, (W, 1)))
        batched = jax.jit(jax.vmap(f))(fa, ta)
        loop = jnp.stack([f(fa[i], ta[i]) for i in range(W)])
        np.testing.assert_allclose(np.asarray(batched), np.asarray(loop), rtol=1e-14, atol=0)

    def lum_at(fagn, tau, lam_idx):
        return f(fagn, tau)[lam_idx]

    i_ir = int(np.searchsorted(model_wave, 1e5))      # 10 um, where the shape moves with tau
    for tau in (7.5, 23.0, 55.0, 120.0):
        gf, gt = jax.grad(lum_at, argnums=(0, 1))(jnp.array([0.2]), jnp.array([tau]), i_ir)
        assert np.all(np.isfinite(gf)) and np.all(np.isfinite(gt))
        assert float(gf[0]) != 0.0 and float(gt[0]) != 0.0
    if mode != "prospector":      # flat outside the grid (clamped): gradient exactly 0
        g = jax.grad(lum_at, argnums=1)(jnp.array([0.2]), jnp.array([300.0]), i_ir)
        assert float(g[0]) == 0.0
    gL = jax.grad(lambda a: jnp.dot(f(a, jnp.array([40.0])), w))(jnp.array([0.2]))
    assert np.isfinite(float(gL[0])) and float(gL[0]) > 0


# ------------------------------------------------------------------ A1, A2 (REF oracles)
def test_A1_fsps_mode_vs_python_fsps(agn_ref, torus_path):
    """A1: fsps-mode torus x diffuse power law (CERIDWEN's own 'powerlaw' DiffuseDust law =
    FSPS dust_type 0) x 10**log_lbol x fagn, vs python-fsps spec(fagn) - spec(0) on the FSPS
    grid, over fagn x agn_tau x dust2 x zred (IGM on in FSPS: its torus is added after the
    IGM, so the difference carries none).  Tolerance: REF's recommended rule."""
    from ceridwen.dust.DustModel import DiffuseDust
    d = agn_ref("A1_torus_fsps.npz")
    wave = d["wave"]
    t = NenkovaTorus(wave, mode="fsps", torus_file=torus_path)
    dd = DiffuseDust("powerlaw")
    rtol = float(d["recommended_rtol"])
    eps = np.finfo(np.float64).eps
    worst = (0.0, None)
    for iz in range(len(d["zred"])):
        for i2, dust2 in enumerate(d["dust2"]):
            tau_d = dd.compute_attenuation(jnp.asarray(wave), {
                "diffuse_tau_pow": jnp.array([dust2]),
                "diffuse_alpha_pow": jnp.array([float(d["dust_index"])])})
            dc = jnp.exp(-tau_d)
            L = 10.0 ** d["log_lbol"][iz, i2]
            ctx = AGNContext(jnp.asarray(wave), jnp.asarray(L), None, dc, "fsps")
            for it, tau in enumerate(d["agn_tau"]):
                for jf, fagn in enumerate(d["fagn"]):
                    ref = d["dspec"][iz, i2, it, jf]
                    s0 = d["spec0"][iz, i2]
                    got = np.asarray(t.rest_spectrum(
                        {"fagn": jnp.array([fagn]), "agn_tau": jnp.array([tau])}, ctx) * dc)
                    m = np.abs(ref) > 1e-6 * np.abs(ref).max()
                    rel = np.abs(got - ref)[m] / np.abs(ref)[m]
                    allow = rtol + 8 * eps * (np.abs(s0) + np.abs(ref))[m] / np.abs(ref)[m]
                    bad = rel > allow
                    assert not np.any(bad), (
                        f"z={d['zred'][iz]} dust2={dust2} tau={tau} fagn={fagn}: "
                        f"{int(bad.sum())} pixels, worst rel {rel.max():.3e} at "
                        f"{wave[m][np.argmax(rel)]:.6g} A")
                    if rel.max() > worst[0]:
                        worst = (rel.max(), (float(d['zred'][iz]), dust2, tau, fagn,
                                             float(wave[m][np.argmax(rel)])))
    print("A1 worst rel", worst, "provenance:", json.loads(str(d["provenance"]))["python_fsps"])


def test_A2_prospector_mode_vs_fake_fsps(agn_ref, torus_path):
    """A2: prospector-mode template vs prospect.sources.fake_fsps.agn_torus on the FSPS and the
    CERIDWEN grids, agn_tau inside and outside [5, 150] (extrapolation): rel <= 1e-12 where
    |ref| > 1e-12 max|ref|, abs <= 1e-12 max|ref| elsewhere."""
    d = agn_ref("A2_torus_prospector.npz")
    for wkey, skey in (("wave_fsps", "torus_fsps"), ("wave_ceridwen", "torus_ceridwen")):
        wave = d[wkey]
        t = NenkovaTorus(wave, mode="prospector", torus_file=torus_path)
        for i, tau in enumerate(d["agn_tau"]):
            ref = d[skey][i]
            got = np.asarray(t.template(tau))
            big = np.abs(ref) > 1e-12 * np.abs(ref).max()
            rel = np.abs(got - ref)[big] / np.abs(ref)[big]
            assert rel.max() <= 1e-12, (wkey, tau, rel.max(), wave[big][np.argmax(rel)])
            assert np.abs(got - ref)[~big].max(initial=0) <= 1e-12 * np.abs(ref).max()
    assert float(d["default_agn_tau"]) == 50.0
    assert float(NenkovaTorus(d["wave_fsps"], mode="prospector",
                              torus_file=torus_path).default_params()["agn_tau"]) == 50.0
