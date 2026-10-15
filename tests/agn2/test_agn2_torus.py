"""SKIRTOR torus + polar dust (``skirtor.SkirtorTorus``, ``polar.PolarDust``), session VAL,
Phase B. Skips loudly while the modules, the SKIRTOR data or the pcigale reference are missing.

Requirements (PLAN_AGN2.md sections 2, 3, 5; CIGALE skirtor2016.py at 29cb909f):

V-TOR-1  L1 parity, "cigale" mode, polar dust off: on CIGALE's own wavelength grid the disc and
         the torus dust equal pcigale's ``agn.SKIRTOR2016_disk`` / ``agn.SKIRTOR2016_torus``
         (tests/reference/agn2/skirtor_cigale.npz, kind "node") to rel 1e-6 where the flux
         is > 1e-6 x its peak, for every node set (both disk_type 1 and 0, all t, pl, q, oa, R, i).
V-TOR-2  L1 off-node: inside one cell, CERIDWEN equals the multilinear interpolation (NumPy,
         tests/agn2/_ref.py) of pcigale's 32 corner spectra (kind "cell") in (t, pl, q, oa, i),
         rel 1e-6 (TORUS's convention, endorsed by PLAN_AGN2 2: blend the node spectra).
V-TOR-3  L3 parity with polar dust (kind "polar"): disc, torus and polar grey body each equal
         pcigale's to rel 1e-6 (> 1e-6 peak), laws 0/1/2, type-1 and type-2 views.
V-TOR-4  amplitude "cigale": torus + polar dust luminosity (frequency integral on the model
         grid) = L_dust_gal x fagn / (1 - fagn) (skirtor2016.py:498), rel 1e-3 (resampling).
V-TOR-5  amplitude "corrected": torus + polar dust luminosity = fagn x L_bol,* (PLAN_AGN2 2),
         rel 1e-6.
V-TOR-6  polar dust energy (skirtor2016.py:351-368, both modes): at i = 0 the grey-body luminosity
         = (7/18 - sin^2 oa / 6 - 2 sin^3 oa / 9) x the face-on disc's absorbed luminosity,
         int disk (1 - T_los) dnu, rel 1e-2 (model grid).
V-TOR-7  type-1/2 switch (skirtor2016.py:348): oa = 40: at i = 30 the disc is extinguished
         (polar_t < 1 in the UV), at i = 70 (> 90 - oa) polar_t == 1 exactly; the grey body is
         emitted in both.
V-TOR-10 local SED independent of the model grid coverage (review finding r1).
V-TOR-11 "corrected": every polar transmission <= 1 (review finding r1).
V-TOR-8  inclination limit: the disc seen edge-on (i = 90) carries < 10 % of the face-on (i = 0)
         disc luminosity; the dust changes by < a factor 3 (the torus re-emits nearly
         isotropically in the IR, Stalevski+16).
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
from ceridwen.agn import AGNContext  # noqa: E402

C_NM_S = 2.99792458e17
REF = _ref.REF_DIR / "skirtor_cigale.npz"
LAWS = {0: "smc", 1: "calzetti", 2: "gaskell"}
MODEL_WAVE = np.geomspace(100.0, 1.0e7, 3000)            # A


def _classes(test):
    try:
        return agnmod._agn2_class("skirtor16"), agnmod._agn2_class("polar")
    except NotImplementedError as e:
        _ref.missing(f"component ({e})", test)


def _build(cls, wave, mode, test, **kw):
    try:
        return cls(wave, mode=mode, **kw)
    except (FileNotFoundError, OSError) as e:
        _ref.missing(f"SKIRTOR data ({e})", test)
    except ValueError as e:
        if "MISSING the SKIRTOR template file" not in str(e):
            raise
        _ref.missing(f"SKIRTOR data ({e})", test)


def _ctx(wave, mode, L_bol=1.0, L_dust=1.0):
    return AGNContext(jnp.asarray(wave), jnp.asarray(L_bol), None, jnp.ones(len(wave)), mode,
                      jnp.asarray(L_dust))


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        _ref.missing(f"{REF} (scripts/agn2_reference/cigale_skirtor_reference.py in the "
                     "pcigale env)", "V-TOR parity")
    d = np.load(REF, allow_pickle=False)
    return {k: d[k] for k in d.files}


def _theta(p, fagn=0.5):
    return {"fagn": jnp.asarray(fagn), "skirtor_t": jnp.asarray(p["t"]),
            "skirtor_pl": jnp.asarray(p["pl"]), "skirtor_q": jnp.asarray(p["q"]),
            "skirtor_oa": jnp.asarray(p["oa"]), "skirtor_i": jnp.asarray(p["i"]),
            "polar_ebv": jnp.asarray(p["EBV"]), "polar_temp": jnp.asarray(p["temperature"]),
            "polar_emis": jnp.asarray(p["emissivity"])}


def _params(ref, k):
    return dict(zip([str(s) for s in ref["param_keys"]], ref["params"][k]))


def _to_lam(fnu, wl_nm):
    """L_nu [per Hz] -> L_lambda [per nm] (per unit luminosity in both codes)."""
    return np.asarray(fnu) * C_NM_S / wl_nm ** 2


def _rel(got, want, frac=1e-6):
    m = want > frac * want.max()
    return float(np.max(np.abs(got[m] / want[m] - 1))) if m.any() else 0.0


_INST = {}


def _instances(ref, p, test, with_polar):
    Sk, Po = _classes(test)
    wl = ref["wl_nm"]
    key = (int(p["R"]), int(p["disk_type"]), int(p["law"]) if with_polar else None)
    if key not in _INST:
        sk = _build(Sk, wl * 10.0, "cigale", test, R=int(p["R"]), file=None,
                    disk_type=int(p["disk_type"]), delta=float(p["delta"]))
        po = _build(Po, wl * 10.0, "cigale", test, law=LAWS[int(p["law"])]) if with_polar \
            else None
        _INST[key] = (sk, po)
    return _INST[key]


def _parts(ref, k, test, with_polar):
    p = _params(ref, k)
    sk, po = _instances(ref, p, test, with_polar)
    wl = ref["wl_nm"]
    out = sk.nuclear_parts(_theta(p, fagn=p["fracAGN"]), _ctx(wl * 10, "cigale",
                                                               L_dust=float(ref["L_dust"])), po)
    return p, {"disk": _to_lam(out["disk"] * out["polar_t"], wl),
               "torus": _to_lam(out["dust"], wl), "polar": _to_lam(out["polar_ir"], wl)}


def test_V_TOR_1_node_parity(ref):
    worst = {"disk": 0.0, "torus": 0.0}
    for k in np.nonzero(ref["kind"] == "node")[0]:
        p, got = _parts(ref, k, "V-TOR-1", with_polar=False)
        for c in worst:
            worst[c] = max(worst[c], _rel(got[c], ref[c][k]))
        assert np.all(got["polar"] == 0)
    assert worst["disk"] <= 1e-6 and worst["torus"] <= 1e-6, worst


def test_V_TOR_2_off_node_cell(ref):
    idx = np.nonzero(ref["kind"] == "cell")[0]
    P = np.array([[_params(ref, k)[a] for a in ("t", "pl", "q", "oa", "i")] for k in idx])
    axes = [np.unique(P[:, j]) for j in range(5)]
    cube = {c: np.zeros((ref["wl_nm"].size,) + tuple(a.size for a in axes))
            for c in ("disk", "torus")}
    for n, k in enumerate(idx):
        pos = tuple(int(np.searchsorted(axes[j], P[n, j])) for j in range(5))
        for c in cube:
            cube[c][(slice(None),) + pos] = ref[c][k]
    rng = np.random.default_rng(11)
    base = _params(ref, idx[0])
    worst = 0.0
    for _ in range(12):
        pt = [rng.uniform(a[0], a[-1]) for a in axes]
        p = dict(base, t=pt[0], pl=pt[1], q=pt[2], oa=pt[3], i=pt[4])
        sk, _ = _instances(ref, p, "V-TOR-2", with_polar=False)
        wl = ref["wl_nm"]
        out = sk.nuclear_parts(_theta(p), _ctx(wl * 10, "cigale"), None)
        for c, got in (("disk", _to_lam(out["disk"], wl)), ("torus", _to_lam(out["dust"], wl))):
            want = _ref.multilinear(axes, cube[c], pt)
            worst = max(worst, _rel(got, want))
    assert worst <= 1e-6, worst


def test_V_TOR_3_polar_parity(ref):
    worst = {"disk": 0.0, "torus": 0.0, "polar": 0.0}
    for k in np.nonzero(ref["kind"] == "polar")[0]:
        p, got = _parts(ref, k, "V-TOR-3", with_polar=True)
        for c in worst:
            worst[c] = max(worst[c], _rel(got[c], ref[c][k]))
    assert max(worst.values()) <= 1e-6, worst


DEF = {"t": 7.0, "pl": 1.0, "q": 1.0, "oa": 40.0, "i": 30.0, "EBV": 0.2, "temperature": 100.0,
       "emissivity": 1.6}


def _model_parts(mode, p, test, polar=True, fagn=0.3, L_bol=2.0, L_dust=5.0):
    Sk, Po = _classes(test)
    key = ("model", mode, polar)
    if key not in _INST:
        _INST[key] = (_build(Sk, MODEL_WAVE, mode, test, R=None, file=None),
                      _build(Po, MODEL_WAVE, mode, test, law="smc") if polar else None)
    sk, po = _INST[key]
    return sk.nuclear_parts(_theta(p, fagn=fagn), _ctx(MODEL_WAVE, mode, L_bol, L_dust), po)


def _L(x):
    return _ref.nu_integral(MODEL_WAVE, np.asarray(x))


def test_V_TOR_4_amplitude_cigale():
    out = _model_parts("cigale", DEF, "V-TOR-4", fagn=0.3, L_dust=5.0)
    assert _L(out["dust"] + out["polar_ir"]) == pytest.approx(5.0 * 0.3 / 0.7, rel=1e-3)


def test_V_TOR_5_amplitude_corrected():
    out = _model_parts("corrected", DEF, "V-TOR-5", fagn=0.3, L_bol=2.0)
    assert _L(out["dust"] + out["polar_ir"]) == pytest.approx(0.3 * 2.0, rel=1e-6)


@pytest.mark.parametrize("mode", ["corrected", "cigale"])
@pytest.mark.parametrize("oa", [20.0, 40.0, 70.0])
def test_V_TOR_6_polar_energy(mode, oa):
    out = _model_parts(mode, dict(DEF, i=0.0, oa=oa), f"V-TOR-6[{mode}]")
    s = np.sin(np.deg2rad(oa))
    f = 7.0 / 18.0 - s ** 2 / 6.0 - 2.0 * s ** 3 / 9.0
    absorbed = _L(np.asarray(out["disk"]) * (1.0 - np.asarray(out["polar_t"])))
    assert absorbed > 0
    assert _L(out["polar_ir"]) == pytest.approx(f * absorbed, rel=1e-2), (mode, oa)


@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_TOR_7_type_switch(mode):
    t1 = _model_parts(mode, dict(DEF, i=30.0), f"V-TOR-7[{mode}]")
    t2 = _model_parts(mode, dict(DEF, i=70.0), f"V-TOR-7[{mode}]")
    uv = (MODEL_WAVE > 1000) & (MODEL_WAVE < 3000) & (np.asarray(t1["disk"]) > 0)
    assert np.all(np.asarray(t1["polar_t"])[uv] < 1.0)
    np.testing.assert_array_equal(np.asarray(t2["polar_t"]), 1.0)
    assert _L(t1["polar_ir"]) > 0 and _L(t2["polar_ir"]) > 0


@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_TOR_8_inclination_limits(mode):
    face = _model_parts(mode, dict(DEF, i=0.0, EBV=0.0), f"V-TOR-8[{mode}]", polar=False)
    edge = _model_parts(mode, dict(DEF, i=90.0, EBV=0.0), f"V-TOR-8[{mode}]", polar=False)
    assert _L(edge["disk"]) < 0.1 * _L(face["disk"])
    r = _L(edge["dust"]) / _L(face["dust"])
    assert 1 / 3 < r < 3, r


def test_V_TOR_9_cigale_luminosities(ref):
    """PLAN_AGN2 5 L1: CIGALE's module quantities (skirtor2016.py:397-531) per unit agn_power:
    disc and polar-dust luminosity, intrinsic disc at 30 deg x 0.493 (accretion power),
    L_nu(2500 A) at 30 deg; via SkirtorTorus.derived (TORUS's API), rel 1e-6."""
    keys = [str(k) for k in ref["info_keys"]]
    for k in np.nonzero(ref["kind"] != "cell")[0][::7]:
        p = _params(ref, k)
        sk, po = _instances(ref, p, "V-TOR-9", with_polar=p["EBV"] > 0)
        if not hasattr(sk, "derived"):
            _ref.missing("SkirtorTorus.derived", "V-TOR-9")
        wl = ref["wl_nm"]
        d = sk.derived(_theta(p, fagn=p["fracAGN"]), _ctx(wl * 10, "cigale",
                                                          L_dust=float(ref["L_dust"])), po)
        info = dict(zip(keys, ref["info"][k]))
        for ours, theirs in (("disk_luminosity", "agn.disk_luminosity"),
                             ("polar_dust_luminosity", "agn.polar_dust_luminosity"),
                             ("accretion_power", "agn.accretion_power"),
                             ("total_dust_luminosity", "agn.total_dust_luminosity")):
            want = info[theirs]
            got = float(d[ours])
            assert got == pytest.approx(want, rel=1e-6, abs=1e-12 * abs(info[
                "agn.total_dust_luminosity"])), (k, ours, got, want)


def test_V_TOR_10_local_sed_independent_of_grid_coverage():
    """Review finding VERDICT_TORUS_r1: a model grid that does not cover the whole SKIRTOR
    template range (1 nm - 10 mm) must not change the SED at the wavelengths it does cover. On
    a grid starting at 1000 A the disc must equal, pixel for pixel, the disc on a grid starting
    at 100 A (same pixels), rel 1e-6, in both modes. (19.7 % of the Schartmann disc luminosity
    lies below 912 A: renormalising each part to its full-template luminosity puts it into the
    covered pixels.)"""
    Sk, Po = _classes("V-TOR-10")
    cut = MODEL_WAVE[MODEL_WAVE >= 1000.0]
    for mode in ("corrected", "cigale"):
        full = _build(Sk, MODEL_WAVE, mode, "V-TOR-10", R=None, file=None)
        part = _build(Sk, cut, mode, "V-TOR-10", R=None, file=None)
        th = _theta(dict(DEF, EBV=0.0), fagn=0.3)
        a = full.nuclear_parts(th, _ctx(MODEL_WAVE, mode, 2.0, 5.0), None)
        b = part.nuclear_parts(th, _ctx(cut, mode, 2.0, 5.0), None)
        sel = MODEL_WAVE >= 1000.0
        for k in ("disk", "dust"):
            x, y = np.asarray(a[k])[sel], np.asarray(b[k])
            m = x > 1e-6 * x.max()
            assert np.max(np.abs(y[m] / x[m] - 1)) <= 1e-6, (mode, k)


def test_V_TOR_11_corrected_transmission_physical():
    """Review finding VERDICT_TORUS_r1: CIGALE's law 1 polynomial (skirtor2016.py:76-77) is
    negative above ~3.1 um, so 10^(-0.4 k EBV) > 1 (light created; 1.99 at 20 um for EBV = 1).
    "cigale" reproduces CIGALE; "corrected" is CERIDWEN's consistent framework: every polar
    transmission <= 1 and the grey body >= 0, for all three laws."""
    Sk, Po = _classes("V-TOR-11")
    sk = _build(Sk, MODEL_WAVE, "corrected", "V-TOR-11", R=None, file=None)
    for law in ("smc", "calzetti", "gaskell"):
        po = _build(Po, MODEL_WAVE, "corrected", "V-TOR-11", law=law)
        for ebv in (0.3, 1.0, 3.0):
            th = _theta(dict(DEF, EBV=ebv, i=30.0), fagn=0.3)
            t = np.asarray(po.transmission(th))
            assert np.all(t <= 1.0), (law, ebv, t.max())
            out = sk.nuclear_parts(th, _ctx(MODEL_WAVE, "corrected", 2.0, 5.0), po)
            assert np.all(np.asarray(out["polar_t"]) <= 1.0 + 1e-12), (law, ebv)
            assert np.all(np.asarray(out["polar_ir"]) >= 0.0), (law, ebv)
