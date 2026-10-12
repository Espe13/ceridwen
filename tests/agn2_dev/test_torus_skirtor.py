"""SKIRTOR torus + CIGALE polar dust (session TORUS development tests).

The reference is a line-by-line NumPy transcription of CIGALE's ``skirtor2016._init_code`` and
``process`` (cigale 29cb909f, pcigale/sed_modules/skirtor2016.py l.293-547) applied to ONE node,
written without any traced code of ``ceridwen.agn.skirtor``. It shares only the converted file
(CIGALE's database arrays) and ``polar.k_ext`` (itself a transcription; tested separately against
the closed forms).

T-SK-1  cigale mode at nodes == the transcription (disc, torus dust, polar dust), polar on/off,
        3 laws, type-1 and type-2 views (rel 1e-12 where > 1e-6 x peak).
T-SK-2  off-node == multilinear blend of the per-node transcription (rel 1e-12).
T-SK-3  corrected mode: torus dust + polar dust carry P = fagn L_bol to quadrature (5e-6 on
        6000 points), each part its template luminosity; polar transmission <= 1.
T-SK-4  polar re-emission == the angle-averaged absorption of the face-on disc (CIGALE l_ext).
T-SK-5  fagn = 0 -> exactly 0; linear in fagn (corrected); fagn/(1-fagn) (cigale).
T-SK-6  clamped outside the grid; continuous across nodes; finite grads at nodes and edges.
T-SK-7  jit (rel 1e-14), vmap W = 64 == loop (rel 1e-12).
T-SK-8  setup errors: missing file names the converter; R not on the grid; bad mode.
T-SK-10 corrected pixels do not depend on the model grid's extent (byte-equal).
T-SK-9  derived (CIGALE's sed.info AGN quantities) == the transcription at nodes (rel 1e-12).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as cst

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ceridwen.agn import AGNContext  # noqa: E402

h5py = pytest.importorskip("h5py")

WAVE = np.geomspace(1.0, 1.0e8, 6000)
C_NM = 2.99792458e17


def _file():
    root = os.environ.get("CERIDWEN_AGN_DATA") or str(Path.home() / ".ceridwen" / "agn")
    p = Path(root) / "skirtor2016.h5"
    if not p.exists():
        pytest.skip(f"MISSING {p}: SKIRTOR test did NOT run (scripts/agn2/build_skirtor_h5.py)")
    return p


def _axes(path):
    with h5py.File(path) as f:
        return {k: f["axes"][k][()] for k in f["axes"]}


def _ctx(mode, lbol=3.0, ldust=2.0):
    return AGNContext(jnp.asarray(WAVE), jnp.asarray(lbol), None, jnp.ones(WAVE.size), mode,
                      jnp.asarray(ldust))


# ------------------------------------------------------------------ the transcription
def _schartmann(wl):
    limits = np.array([8.0, 50.0, 125.0, 10000.0, 1e6])
    coefs = np.array([1.0, -0.2, -1.5, -4.0])
    ss = np.searchsorted(wl, limits)
    norms = np.ones_like(coefs)
    for idx in range(1, coefs.size):
        norms[idx] = norms[idx - 1] * limits[idx] ** (coefs[idx - 1] - coefs[idx])
    spec = np.zeros_like(wl)
    for lo, hi, coef, norm in zip(ss[:-1], ss[1:], coefs, norms):
        spec[lo:hi] = wl[lo:hi] ** coef * norm
    return spec / np.trapezoid(spec, x=wl)


def cigale_node(path, R, t, pl, q, oa, i, *, polar, EBV=0.0, T=100.0, emis=1.6, law=0):
    """(wl [nm], torus dust, polar dust, disc) per unit agn_power, F_lambda per nm."""
    from ceridwen.agn.polar import k_ext
    with h5py.File(path) as f:
        ax = {k: f["axes"][k][()] for k in f["axes"]}

        def ix(k, v):
            return int(np.flatnonzero(ax[k] == v)[0])
        sl = (ix("R", R), ix("t", t), ix("pl", pl), ix("q", q), ix("oa", oa))
        wl = f["wl_nm"][()]
        disk_i = f["disk"][sl + (ix("i", i),)].astype(np.float64)
        dust_i = f["dust"][sl + (ix("i", i),)].astype(np.float64)
        norm_i = float(f["norm"][sl + (ix("i", i),)])
        disk_0 = f["disk"][sl + (0,)].astype(np.float64)
        norm_0 = float(f["norm"][sl + (0,)])
    agn1 = disk_0 * norm_0 / norm_i
    disk = _schartmann(wl) * np.trapezoid(agn1, x=wl)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk_disk = np.nan_to_num(disk * disk_i / agn1)
    agn1 = disk
    if not polar:
        return wl, dust_i, np.zeros_like(wl), sk_disk
    ext = 10 ** (-0.4 * k_ext(wl, law) * EBV)
    if i <= 90.0 - oa:
        sk_disk = sk_disk * ext
    s = np.sin(np.deg2rad(oa))
    l_ext = (7.0 / 18.0 - s ** 2 / 6.0 - s ** 3 * 2.0 / 9.0) * np.trapezoid(agn1 * (1 - ext), x=wl)
    c = cst.c * 1e9
    with np.errstate(over="ignore"):
        bb = (c / wl ** 2 * (1.0 - np.exp(-((200e3 / wl) ** emis))) * (c / wl) ** 3
              / (np.exp(cst.h * c / (wl * cst.k * T)) - 1.0))
    bb = bb * l_ext / np.trapezoid(bb, x=wl)
    dust = dust_i + bb
    norm = 1.0 / np.trapezoid(dust, x=wl)
    return wl, (dust - bb) * norm, bb * norm, sk_disk * norm


def cigale_info(path, R, t, pl, q, oa, i, *, polar, fracAGN, Ldust, **pol):
    """CIGALE's sed.info AGN quantities (skirtor2016.py l.397-531) at one node."""
    wl, dust, pdust, disk = cigale_node(path, R, t, pl, q, oa, i, polar=polar, **pol)
    # AGN1.disk after the replacement, in this node's normalisation, times norm
    _, _, _, disk0 = cigale_node(path, R, t, pl, q, oa, 0.0, polar=False)
    with h5py.File(path) as f:
        a = {k: f["axes"][k][()] for k in f["axes"]}
        ix = [int(np.flatnonzero(a[k] == v)[0]) for k, v in
              zip(("R", "t", "pl", "q", "oa"), (R, t, pl, q, oa))]
        n_i = float(f["norm"][(*ix, int(np.flatnonzero(a["i"] == i)[0]))])
        n_0 = float(f["norm"][(*ix, 0)])
    # dust, pdust, disk above already carry CIGALE's norm (l.392); AGN1.disk does not.
    l_pol = np.trapezoid(pdust, x=wl)
    lumin_disk = np.trapezoid(disk, x=wl)
    agn1 = disk0 * n_0 / n_i
    # CIGALE's norm = 1 / (1 + l_ext) and l_pol = l_ext / (1 + l_ext), so norm = 1 - l_pol
    nrm = 1.0 - l_pol
    c = cst.c * 1e9
    cos30 = np.cos(np.radians(30.0))
    norm_fac = cos30 * (2.0 * cos30 + 1.0) / 3.0 * nrm
    intrin = np.trapezoid(agn1, x=wl) * norm_fac
    l2500 = np.interp(250, wl, agn1) * norm_fac * 250.0 ** 2 / c
    P = Ldust * (1.0 / (1.0 - fracAGN) - 1.0)
    L6, L12 = np.interp([6000, 12000], wl, dust + pdust + disk) * P * np.array([6000, 12000])
    return {"total_dust_luminosity": P, "polar_dust_luminosity": P * l_pol,
            "torus_dust_luminosity": P - P * l_pol, "disk_luminosity": P * lumin_disk,
            "luminosity": P + P * lumin_disk, "accretion_power": P * intrin * 0.493,
            "intrin_Lnu_2500A_30deg": P * l2500, "L_6um": L6, "L_12um": L12,
            "frac_agn_cigale": P / (P + Ldust)}


def _to_model(wl, y):
    return np.interp(WAVE / 10.0, wl, y, left=0.0, right=0.0) * (WAVE / 10.0) ** 2 / C_NM


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    big = np.abs(b) > 1e-6 * np.abs(b).max()
    if not big.any():
        return float(np.max(np.abs(a)))
    return float(np.max(np.abs(a[big] - b[big]) / np.abs(b[big])))


def _theta(t, pl, q, oa, i, fagn=0.3, **pol):
    th = {"fagn": fagn, "skirtor_t": t, "skirtor_pl": pl, "skirtor_q": q, "skirtor_oa": oa,
          "skirtor_i": i}
    th.update(pol)
    return {k: jnp.asarray(float(v)) for k, v in th.items()}


def _nodes(ax, n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(tuple(float(rng.choice(ax[k])) for k in ("t", "pl", "q", "oa", "i")))
    # a type-1 and a type-2 view at the same oa
    oa = float(ax["oa"][len(ax["oa"]) // 2])
    out.append((float(ax["t"][0]), float(ax["pl"][0]), float(ax["q"][0]), oa, 0.0))
    out.append((float(ax["t"][-1]), float(ax["pl"][-1]), float(ax["q"][-1]), oa,
                float(ax["i"][-1])))
    return out


LAWS = {"smc": 0, "calzetti": 1, "gaskell": 2}


@pytest.mark.usefixtures("polar_data")
@pytest.mark.parametrize("law", list(LAWS))
@pytest.mark.parametrize("polar_on", [False, True])
def test_T_SK_1_cigale_nodes(law, polar_on):
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    ax = _axes(path)
    R = float(ax["R"][len(ax["R"]) // 2])
    sk = SkirtorTorus(WAVE, mode="cigale", R=R, file=str(path))
    pd = PolarDust(WAVE, mode="cigale", law=law) if polar_on else None
    fagn, ldust = 0.3, 2.0
    P = ldust * (1.0 / (1.0 - fagn) - 1.0)
    pol = {"polar_ebv": 0.25, "polar_temp": 180.0, "polar_emis": 1.4}
    worst = 0.0
    for node in _nodes(ax):
        p = sk.nuclear_parts(_theta(*node, fagn=fagn, **pol), _ctx("cigale", ldust=ldust), pd)
        wl, dust, pdust, disk = cigale_node(path, R, *node, polar=polar_on, EBV=0.25, T=180.0,
                                            emis=1.4, law=LAWS[law])
        for mine, ref in ((p["dust"], dust), (p["polar_ir"], pdust),
                          (p["disk"] * p["polar_t"], disk)):
            if not np.any(ref):
                assert not np.any(np.asarray(mine))
                continue
            worst = max(worst, _rel(mine, P * _to_model(wl, ref)))
        np.testing.assert_allclose(np.asarray(p["nuclear"]),
                                   np.asarray(p["disk"] * p["polar_t"] + p["dust"]
                                              + p["polar_ir"]), rtol=1e-13, atol=0)
    print(f"T-SK-1 {law} polar={polar_on}: worst rel {worst:.2e}")
    assert worst < 1e-12


@pytest.mark.usefixtures("polar_data")
def test_T_SK_2_off_node_multilinear():
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    ax = _axes(path)
    R = float(ax["R"][0])
    sk = SkirtorTorus(WAVE, mode="cigale", R=R, file=str(path))
    pd = PolarDust(WAVE, mode="cigale", law="smc")
    keys = ("t", "pl", "q", "oa", "i")
    # a point strictly inside the first cell of every axis (the i cell crosses type 1 / 2
    # for oa near 90 - i)
    pt = [float(ax[k][0] + 0.37 * (ax[k][1] - ax[k][0])) if ax[k].size > 1 else float(ax[k][0])
          for k in keys]
    pol = dict(EBV=0.2, T=150.0, emis=1.6, law=0)
    p = sk.nuclear_parts(_theta(*pt, polar_ebv=0.2, polar_temp=150.0, polar_emis=1.6),
                         _ctx("cigale"), pd)
    ref = {"dust": 0.0, "pol": 0.0, "disk": 0.0}
    for corner in range(32):
        w, node = 1.0, []
        for d, k in enumerate(keys):
            a = ax[k]
            if a.size == 1:
                node.append(float(a[0]))
                continue
            fr = (pt[d] - a[0]) / (a[1] - a[0])
            b = (corner >> d) & 1
            w *= fr if b else 1.0 - fr
            node.append(float(a[b]))
        if w == 0.0:
            continue
        wl, dust, pdust, disk = cigale_node(path, R, *node, polar=True, **pol)
        ref["dust"] = ref["dust"] + w * dust
        ref["pol"] = ref["pol"] + w * pdust
        ref["disk"] = ref["disk"] + w * disk
    P = 2.0 * 0.3 / 0.7
    assert _rel(p["dust"], P * _to_model(wl, ref["dust"])) < 1e-12
    assert _rel(p["polar_ir"], P * _to_model(wl, ref["pol"])) < 1e-12
    assert _rel(p["disk"] * p["polar_t"], P * _to_model(wl, ref["disk"])) < 1e-12


def _nu_int(y):
    from ceridwen.agn.torus import trapezoid_nu_weights
    return float(np.asarray(y) @ -np.asarray(trapezoid_nu_weights(WAVE)))


@pytest.mark.usefixtures("polar_data")
@pytest.mark.parametrize("polar_on", [False, True])
def test_T_SK_3_corrected_energy(polar_on):
    """Torus + polar dust carry P = fagn L_bol to the model grid's quadrature (|L/P - 1| measured
    <= 1.9e-6 on this 6000-point grid, 3e-7 on 15000 points); each part keeps its template
    luminosity to the same order; the transmitted disc is never brighter than the emitted one."""
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    ax = _axes(path)
    sk = SkirtorTorus(WAVE, mode="corrected", R=float(ax["R"][0]), file=str(path))
    pd = PolarDust(WAVE, mode="corrected", law="calzetti") if polar_on else None
    lbol = 3.0
    for node in _nodes(ax, n=3):
        th = _theta(*node, fagn=0.2, polar_ebv=0.4, polar_temp=120.0, polar_emis=1.6)
        p = sk.nuclear_parts(th, _ctx("corrected", lbol=lbol), pd)
        L = _nu_int(p["dust"]) + _nu_int(p["polar_ir"])
        assert abs(L / (0.2 * lbol) - 1.0) < 5e-6, (node, L)
        nat = sk.native_parts(th, pd)
        tw = np.asarray(sk.tw)
        for k_model, k_nat in (("dust", "dust"), ("polar_ir", "polar")):
            want = 0.2 * lbol * float(np.asarray(nat[k_nat]) @ tw)
            got = _nu_int(p[k_model])
            assert (want == 0 and got == 0) or abs(got / want - 1) < 5e-6
        assert np.all(np.asarray(p["polar_t"]) <= 1.0)


@pytest.mark.usefixtures("polar_data")
def test_T_SK_10_corrected_pixels_independent_of_grid_coverage():
    """VERDICT_TORUS_r1 R1: the corrected SED at a pixel does not depend on the grid's extent."""
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    cut = WAVE[(WAVE >= 2000.0) & (WAVE <= 3e6)]
    a_sk = SkirtorTorus(WAVE, mode="corrected", file=str(path))
    b_sk = SkirtorTorus(cut, mode="corrected", file=str(path))
    pd_a = PolarDust(WAVE, mode="corrected", law="smc")
    pd_b = PolarDust(cut, mode="corrected", law="smc")
    th = _theta(7.0, 1.0, 1.0, 40.0, 30.0, polar_ebv=0.3, polar_temp=100.0, polar_emis=1.6)
    ctx_b = AGNContext(jnp.asarray(cut), jnp.asarray(3.0), None, jnp.ones(cut.size), "corrected")
    a = a_sk.nuclear_parts(th, _ctx("corrected"), pd_a)
    b = b_sk.nuclear_parts(th, ctx_b, pd_b)
    sel = (WAVE >= 2000.0) & (WAVE <= 3e6)
    for k in ("disk", "dust", "polar_ir", "polar_t", "nuclear"):
        np.testing.assert_array_equal(np.asarray(a[k])[sel], np.asarray(b[k]), err_msg=k)


@pytest.mark.usefixtures("polar_data")
def test_T_SK_4_polar_reemission_is_angle_averaged_absorption():
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust, k_ext
    path = _file()
    ax = _axes(path)
    R = float(ax["R"][0])
    sk = SkirtorTorus(WAVE, mode="cigale", R=R, file=str(path))
    pd = PolarDust(WAVE, mode="cigale", law="gaskell")
    node = tuple(float(ax[k][0]) for k in ("t", "pl", "q", "oa")) + (float(ax["i"][-1]),)
    th = _theta(*node, polar_ebv=0.5, polar_temp=200.0, polar_emis=1.8)
    nat = sk.native_parts(th, pd)
    wl = sk.wl_nm
    # face-on disc of this node after CIGALE's replacement, in its normalisation
    _, dust, _, _ = cigale_node(path, R, *node, polar=False)
    _, _, _, disk0 = cigale_node(path, R, *node[:4], 0.0, polar=False)
    with h5py.File(path) as f:
        a = {k: f["axes"][k][()] for k in f["axes"]}
        ix = [int(np.flatnonzero(a[k] == v)[0]) for k, v in zip(("t", "pl", "q", "oa"), node)]
        n_i = float(f["norm"][(0, *ix, a["i"].size - 1)])
        n_0 = float(f["norm"][(0, *ix, 0)])
    face_on = disk0 * n_0 / n_i         # AGN1 in the inclined node's normalisation
    ext = 10 ** (-0.4 * k_ext(wl, 2) * 0.5)
    s = np.sin(np.deg2rad(node[3]))
    l_ext = (7 / 18 - s ** 2 / 6 - 2 * s ** 3 / 9) * np.trapezoid(face_on * (1 - ext), x=wl)
    got = np.trapezoid(np.asarray(nat["polar"]), x=wl)
    assert abs(got / (l_ext / (1 + l_ext)) - 1) < 1e-12
    # a type-2 view: the disc is not extinguished, but the cone still re-emits
    assert node[4] > 90 - node[3]
    np.testing.assert_allclose(np.asarray(nat["disk"]), np.asarray(nat["disk_u"]), rtol=0, atol=0)
    assert got > 0


@pytest.mark.usefixtures("polar_data")
@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_T_SK_5_zero_and_linear(mode):
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    sk = SkirtorTorus(WAVE, mode=mode, file=str(path), R=float(_axes(path)["R"][0]))
    pd = PolarDust(WAVE, mode=mode, law="calzetti")
    th = {k: jnp.asarray(v) for k, v in sk.default_params().items()}
    th.update(polar_ebv=jnp.asarray(0.3), polar_temp=jnp.asarray(100.0),
              polar_emis=jnp.asarray(1.6))
    p0 = sk.nuclear_parts(dict(th, fagn=jnp.asarray(0.0)), _ctx(mode), pd)
    for k in ("disk", "dust", "polar_ir", "nuclear"):
        assert not np.any(np.asarray(p0[k])), k
    a = sk.nuclear_parts(dict(th, fagn=jnp.asarray(0.1)), _ctx(mode), pd)["nuclear"]
    b = sk.nuclear_parts(dict(th, fagn=jnp.asarray(0.2)), _ctx(mode), pd)["nuclear"]
    ratio = 2.0 if mode == "corrected" else (0.2 / 0.8) / (0.1 / 0.9)
    np.testing.assert_allclose(np.asarray(b), ratio * np.asarray(a), rtol=1e-13, atol=0)


@pytest.mark.usefixtures("polar_data")
def test_T_SK_6_clamp_continuity_grads():
    from ceridwen.agn.skirtor import SkirtorTorus, SKIRTOR_AXES, THETA_OF_AXIS
    from ceridwen.agn.polar import PolarDust
    path = _file()
    ax = _axes(path)
    sk = SkirtorTorus(WAVE, mode="corrected", file=str(path), R=float(ax["R"][0]))
    pd = PolarDust(WAVE, mode="corrected", law="smc")
    base = {k: jnp.asarray(v) for k, v in sk.default_params().items()}
    base.update(fagn=jnp.asarray(0.2), polar_ebv=jnp.asarray(0.2),
                polar_temp=jnp.asarray(150.0), polar_emis=jnp.asarray(1.6))
    for a in SKIRTOR_AXES:
        base[THETA_OF_AXIS[a]] = jnp.asarray(float(ax[a][0]))

    def f(th):
        return sk.nuclear_parts(th, _ctx("corrected"), pd)["nuclear"]
    for a in SKIRTOR_AXES:
        key = THETA_OF_AXIS[a]
        lo, hi = float(ax[a][0]), float(ax[a][-1])
        # clamped below and above
        assert np.array_equal(np.asarray(f(dict(base, **{key: jnp.asarray(lo - 5.0)}))),
                              np.asarray(f(dict(base, **{key: jnp.asarray(lo)}))))
        assert np.array_equal(np.asarray(f(dict(base, **{key: jnp.asarray(hi + 5.0)}))),
                              np.asarray(f(dict(base, **{key: jnp.asarray(hi)}))))
        if ax[a].size > 1:
            node = float(ax[a][1]) if ax[a].size > 2 else hi
            eps = 1e-9 * max(1.0, abs(node))
            l = np.asarray(f(dict(base, **{key: jnp.asarray(node - eps)})))
            r = np.asarray(f(dict(base, **{key: jnp.asarray(node + eps)})))
            assert np.max(np.abs(l - r)) <= 1e-6 * np.max(np.abs(l)), a
        # finite gradients at a node, at both edges and between nodes
        for x in (lo, hi, 0.5 * (lo + hi)):
            g = jax.grad(lambda th: jnp.sum(jnp.log1p(f(th))))(dict(base, **{key: jnp.asarray(x)}))
            assert all(np.isfinite(np.asarray(v)) for v in g.values()), (a, x)


@pytest.mark.usefixtures("polar_data")
@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_T_SK_7_jit_vmap(mode):
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    ax = _axes(path)
    sk = SkirtorTorus(WAVE, mode=mode, file=str(path), R=float(ax["R"][0]))
    pd = PolarDust(WAVE, mode=mode, law="smc")
    rng = np.random.default_rng(1)
    W = 64
    ths = {"fagn": rng.uniform(0.01, 0.5, W), "polar_ebv": rng.uniform(0, 1, W),
           "polar_temp": rng.uniform(50, 300, W), "polar_emis": rng.uniform(1, 2, W)}
    for a, key in (("t", "skirtor_t"), ("pl", "skirtor_pl"), ("q", "skirtor_q"),
                   ("oa", "skirtor_oa"), ("i", "skirtor_i")):
        ths[key] = rng.uniform(ax[a][0], ax[a][-1], W)
    ths = {k: jnp.asarray(v) for k, v in ths.items()}

    def f(th):
        return sk.nuclear_parts(th, _ctx(mode), pd)["nuclear"]
    batch = np.asarray(jax.jit(jax.vmap(f))(ths))
    loop = np.stack([np.asarray(f({k: v[j] for k, v in ths.items()})) for j in range(W)])
    scale = np.max(np.abs(loop), axis=1, keepdims=True)
    assert np.max(np.abs(batch - loop) / scale) < 1e-12
    one = {k: v[0] for k, v in ths.items()}
    assert np.max(np.abs(np.asarray(jax.jit(f)(one)) - loop[0])) <= 1e-14 * scale[0, 0]


def test_T_SK_8_setup_errors(tmp_path):
    from ceridwen.agn.skirtor import SkirtorTorus
    path = _file()
    with pytest.raises(ValueError, match="build_skirtor_h5.py"):
        SkirtorTorus(WAVE, mode="corrected", file=str(tmp_path / "nope.h5"))
    with pytest.raises(ValueError, match="skirtor_R must be one of"):
        SkirtorTorus(WAVE, mode="corrected", file=str(path), R=17.0)
    for mode in ("fsps", "prospector"):
        with pytest.raises(ValueError, match="only in"):
            SkirtorTorus(WAVE, mode=mode, file=str(path))


@pytest.mark.usefixtures("polar_data")
@pytest.mark.parametrize("polar_on", [False, True])
def test_T_SK_9_derived(polar_on):
    from ceridwen.agn.skirtor import SkirtorTorus
    from ceridwen.agn.polar import PolarDust
    path = _file()
    ax = _axes(path)
    R = float(ax["R"][0])
    sk = SkirtorTorus(WAVE, mode="cigale", R=R, file=str(path))
    pd = PolarDust(WAVE, mode="cigale", law="smc") if polar_on else None
    pol = dict(EBV=0.3, T=140.0, emis=1.6, law=0)
    for node in _nodes(ax, n=3, seed=3):
        got = sk.derived(_theta(*node, fagn=0.25, polar_ebv=0.3, polar_temp=140.0,
                                polar_emis=1.6), _ctx("cigale", ldust=2.0), pd)
        ref = cigale_info(path, R, *node, polar=polar_on, fracAGN=0.25, Ldust=2.0, **pol)
        for k, v in ref.items():
            g = float(got[k])
            assert (v == 0 and abs(g) < 1e-300) or abs(g / v - 1) < 1e-12, (node, k, g, v)
