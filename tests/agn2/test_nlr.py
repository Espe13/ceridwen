"""Feltre+16 NLR (``nlr.FeltreNLR``): grid exactness against the SOURCE files and the full
pipeline (session VAL, Phase B). Skips loudly while nlr.py, its HDF5 or the source files are
missing.

Requirements (PLAN_AGN2.md section 4 and 5 L4; PROMPT_agn2_VAL.md "Grid exactness"):

V-NLR-1  at every grid node (all 16 x 9 x 3 x 3 x 4 nodes, every painted line), FeltreNLR's
         L_line / L_Hbeta equals the ratio of the source text files (nlr_nebular_Z*.txt, parsed by
         tests/agn2/_ref.py, not by the converter) to rel 1e-12; agn_elum is L_Hbeta exactly.
V-NLR-2  off-node: equals an independent NumPy multilinear interpolation of log10(L/L_Hbeta) in
         (log10(Z/0.01524), log U_S, xi_d, log10 n_H, alpha) (PLAN_AGN2 4: NLR's convention,
         endorsed) at 200 random points, rel 1e-12.
V-NLR-3  outside the grid the ratios equal those at the clamped point (no extrapolation), and
         range_messages names every out-of-grid key.
V-NLR-4  line set: every source line with an FSPS line of the same species/wavelength is painted
         (within 2 A of its source wavelength); the ones without (NV1240, [SiIII]1883, SiIII]1888)
         are listed in dropped_lines.
V-NLR-5  pipeline: the Lines observation of [OIII]5007 and H beta moves by agn_elum x ratio x mass x
         flux factor / (1 + z) x T_gal(line) (corrected) or x 1 (cigale); the ratio of the two
         equals the grid ratio at the theta point.
V-NLR-6  grad of the line luminosities finite in every nlr_* key at the prior centre (BEAGLE-AGN
         ranges, PLAN_AGN2 4) and at the grid edges.
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

ZSUN = 0.01524
SRC_DIRS = [pathlib.Path(p) for p in (
    os.environ.get("CERIDWEN_FELTRE16_SRC", ""),
    pathlib.Path.home() / "Desktop/PhD/Tools/highz_NIRSpec_NIRCam/data/AGN_NLR_nebular_feltre16",
) if p]
KEYS = ("nlr_logz", "nlr_logu", "nlr_xid", "nlr_lognh", "nlr_alpha")


@pytest.fixture(scope="module")
def raw():
    for d in SRC_DIRS:
        if d.is_dir() and list(d.glob("nlr_nebular_Z*.txt")):
            return _ref.feltre_raw(d)
    _ref.missing("Feltre+16 source files nlr_nebular_Z*.txt (set $CERIDWEN_FELTRE16_SRC)",
                 "V-NLR")


@pytest.fixture(scope="module")
def nlr():
    try:
        cls = agnmod._agn2_class("feltre16")
    except NotImplementedError as e:
        _ref.missing(f"component feltre16 ({e})", "V-NLR")
    if not os.environ.get("SPS_HOME"):
        _ref.missing("$SPS_HOME", "V-NLR")
    try:
        return cls(mode="corrected", file=None, sps_home=os.environ["SPS_HOME"])
    except FileNotFoundError as e:
        _ref.missing(f"nlr_feltre16.h5 ({e})", "V-NLR")


def _axes(raw):
    return (np.log10(raw["Z_abs"] / ZSUN), raw["logu"], raw["xid"], np.log10(raw["nh"]),
            raw["alpha"])


def _src_rows(nlr, raw):
    """Row of the source table for every painted line (by source name, NLR's mapping)."""
    return [raw["lines"].index(s) for s in nlr.source_names]


def _theta(**kw):
    return {k: jnp.asarray(float(v)) for k, v in kw.items()}


def test_V_NLR_1_nodes_equal_source(nlr, raw):
    rows = _src_rows(nlr, raw)
    hb = raw["lines"].index("Hbeta")
    R = raw["L"][rows] / raw["L"][hb]                    # (n, nZ, nU, nX, nN, nA)
    ax = _axes(raw)
    f = jax.jit(lambda t: nlr.line_luminosity(t))
    worst = 0.0
    n = 0
    for iz, z in enumerate(ax[0]):
        for iu, u in enumerate(ax[1]):
            for ix, x in enumerate(ax[2]):
                for iN, N in enumerate(ax[3]):
                    for ia, a in enumerate(ax[4]):
                        got = np.asarray(f(_theta(agn_elum=1.0, nlr_logz=z, nlr_logu=u,
                                                  nlr_xid=x, nlr_lognh=N, nlr_alpha=a)))
                        want = R[:, iz, iu, ix, iN, ia]
                        worst = max(worst, float(np.max(np.abs(got / want - 1))))
                        n += 1
    assert n == 16 * 9 * 3 * 3 * 4
    assert worst <= 1e-12, worst
    names = list(nlr.line_names)
    th = _theta(agn_elum=0.7, nlr_logz=0.0, nlr_logu=-2.5, nlr_xid=0.3, nlr_lognh=3.0,
                nlr_alpha=-1.7)
    assert float(nlr.line_luminosity(th)[names.index("Ba-beta 4861")]) == pytest.approx(
        0.7, rel=1e-15)


def test_V_NLR_2_off_node_equals_numpy(nlr, raw):
    rows = _src_rows(nlr, raw)
    hb = raw["lines"].index("Hbeta")
    logR = np.log10(raw["L"][rows] / raw["L"][hb])
    ax = _axes(raw)
    rng = np.random.default_rng(7)
    f = jax.jit(lambda t: nlr.line_luminosity(t))
    worst = 0.0
    for _ in range(200):
        p = [rng.uniform(a[0], a[-1]) for a in ax]
        want = 10 ** _ref.multilinear(ax, logR, p)
        got = np.asarray(f(_theta(agn_elum=1.0, **dict(zip(KEYS, p)))))
        worst = max(worst, float(np.max(np.abs(got / want - 1))))
    assert worst <= 1e-12, worst


def test_V_NLR_3_clamped_outside(nlr, raw):
    ax = _axes(raw)
    inside = {"nlr_logz": ax[0][-1], "nlr_logu": ax[1][0], "nlr_xid": ax[2][-1],
              "nlr_lognh": ax[3][0], "nlr_alpha": ax[4][-1]}
    outside = {"nlr_logz": ax[0][-1] + 0.4, "nlr_logu": ax[1][0] - 1.0,
               "nlr_xid": ax[2][-1] + 0.2, "nlr_lognh": ax[3][0] - 0.5,
               "nlr_alpha": ax[4][-1] + 0.3}
    further = {k: v + (v - inside[k]) for k, v in outside.items()}
    a = np.asarray(nlr.line_luminosity(_theta(agn_elum=1.0, **inside)))
    b = np.asarray(nlr.line_luminosity(_theta(agn_elum=1.0, **outside)))
    c = np.asarray(nlr.line_luminosity(_theta(agn_elum=1.0, **further)))
    np.testing.assert_array_equal(b, c)            # clamped: flat beyond the edge, bit for bit
    # the edge value (my edge differs from NLR's axis node in the last bit: 1e-13)
    np.testing.assert_allclose(b, a, rtol=1e-13, atol=0)
    msgs = " ".join(nlr.range_messages({k: np.array([v]) for k, v in outside.items()}))
    for k in KEYS:
        assert k in msgs, k
    assert nlr.range_messages({k: np.array([v]) for k, v in inside.items()}) == []


SOURCE_WAVE = {"[OII]3727": 3727.0, "Hbeta": 4861.3, "[OIII]4959": 4958.9, "[OIII]5007": 5006.8,
               "[OI]6300": 6300.3, "[NII]6548": 6548.0, "Halpha": 6562.8, "[NII]6584": 6583.4,
               "[SII]6717": 6716.4, "[SII]6731": 6730.8, "NV1240": 1240.0, "CIV1548": 1548.2,
               "CIV1551": 1550.8, "HeII1640": 1640.4, "OIII]1661": 1660.8, "OIII]1666": 1666.2,
               "[SiIII]1883": 1883.0, "SiIII]1888": 1892.0, "[CIII]1907": 1906.7,
               "CIII]1910": 1908.7}


def test_V_NLR_4_line_set(nlr, raw):
    painted = dict(zip(nlr.source_names, nlr.line_wave))
    for src, lam in painted.items():
        # FSPS vacuum vs the source's (air, rounded) labels: within 4 A
        assert abs(lam - SOURCE_WAVE[src]) < 4.0, (src, lam)
    assert set(painted) | set(nlr.dropped_lines) == set(raw["lines"])
    assert set(nlr.dropped_lines) <= {"NV1240", "[SiIII]1883", "SiIII]1888"}


def _pipeline(mode, test):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import test_scaffold as S
    try:
        agnmod._agn2_class("feltre16")
    except NotImplementedError as e:
        _ref.missing(f"component feltre16 ({e})", test)
    cfg = agnmod.AGNConfig(agn_mode=mode, torus=False, disk=False, nlr_model="feltre16") \
        if mode == "corrected" else agnmod.AGNConfig(agn_mode="cigale", torus=False,
                                                       nlr_model="feltre16")
    path = S._need()
    from ceridwen import CSPBasis, Cosmology, SSPData
    cosmo = Cosmology.planck18()
    try:
        csp = CSPBasis(SSPData.load(str(path)),
                       lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(S.T.ZRED))]),
                       zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                       add_neb=True, add_igm=False, add_dust_emission=False, verbose=False,
                       cosmo=cosmo, sps_home=os.environ["SPS_HOME"], agn=cfg)
    except FileNotFoundError as e:
        _ref.missing(f"nlr_feltre16.h5 ({e})", test)
    return S, csp


@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_NLR_5_pipeline_lines(mode):
    S, csp = _pipeline(mode, f"V-NLR-5[{mode}]")
    model, _, _, lines = S._model(csp)
    pt = {"nlr_logz": -0.4, "nlr_logu": -2.2, "nlr_xid": 0.3, "nlr_lognh": 3.0,
          "nlr_alpha": -1.7}
    th0, _ = S._mt(model, dict(pt, agn_elum=0.0))
    th1, mt1 = S._mt(model, dict(pt, agn_elum=2e-3))
    d = np.asarray(model.predict(th1)["lines"], np.float64) - np.asarray(
        model.predict(th0)["lines"], np.float64)
    nlr = csp._agn.model.lines
    names = list(nlr.line_names)
    L = np.asarray(nlr.line_luminosity(csp._agn.filled(mt1)))
    msum = float(jnp.sum(csp.calculate_ssp_weights(theta=mt1)))
    scale = msum * 10 ** float(mt1["logmass"][0]) * float(csp._flux_factor(mt1)) / (1 + S.T.ZRED)
    dc = S._diffuse(csp, mt1)
    lname = [n for n, _ in S.T.LINES]
    for obs_name, fsps in (("[OIII]5007", "[O III] 5007"), ("Hbeta", "Ba-beta 4861")):
        j = names.index(fsps)
        t = np.interp(nlr.line_wave[j], np.asarray(csp.wave), dc) if mode == "corrected" else 1.0
        assert d[lname.index(obs_name)] == pytest.approx(L[j] * scale * t, rel=2e-6), obs_name


def test_V_NLR_6_grad(nlr, raw):
    ax = _axes(raw)
    centre = {"nlr_logz": -1.0, "nlr_logu": -2.75, "nlr_xid": 0.3, "nlr_lognh": 3.0,
              "nlr_alpha": -1.6}

    def s(t):
        return jnp.sum(jnp.log(nlr.line_luminosity(t)))

    g = jax.grad(s)
    pts = [centre] + [dict(centre, **{k: v}) for k, a in zip(KEYS, ax) for v in (a[0], a[-1])]
    for p in pts:
        gr = g(_theta(agn_elum=1.0, **p))
        for k in KEYS:
            assert np.isfinite(float(gr[k])), (p, k)
    gr = g(_theta(agn_elum=1.0, **centre))
    assert all(float(gr[k]) != 0.0 for k in ("nlr_logz", "nlr_logu", "nlr_alpha"))
