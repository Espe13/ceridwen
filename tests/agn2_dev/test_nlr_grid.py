"""Development tests of the Feltre+16 NLR grid (``scripts/agn2/build_nlr_h5.py``,
``ceridwen/agn/nlr.py``). Session NLR's own tests; VAL's acceptance tests are ``tests/agn2/``.

D-nlr-1  the converted cube equals the raw ASCII files (an independent parse) at every node.
D-nlr-2  FeltreNLR.ratios equals the table at random grid nodes (rel <= 1e-12).
D-nlr-3  off-node values equal scipy's RegularGridInterpolator on log10 ratios (rel <= 1e-12),
         and lie between the corner values.
D-nlr-4  outside the grid the value is the edge value (clamped, never extrapolated).
D-nlr-5  line_luminosity is linear in agn_elum; agn_elum = 0 gives exactly 0.
D-nlr-6  jit, vmap (W = 64, 100) equal the loop; jax.grad finite in every NLR key at the centre
         and at the edges, and equal to central finite differences inside a cell.
D-nlr-7  refusals at setup: reproduction modes, missing file, a different FSPS line list,
         priors/values outside the grid.
"""
from __future__ import annotations

import os
import pathlib
import shutil

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

SPS_HOME = os.environ.get("SPS_HOME")
RAW_DIR = pathlib.Path(os.environ.get(
    "CERIDWEN_NLR_RAW",
    "/Users/amanda/Desktop/PhD/Tools/highz_NIRSpec_NIRCam/data/AGN_NLR_nebular_feltre16"))


def _nlr_file():
    from ceridwen.agn.nlr import default_nlr_file
    return default_nlr_file()


pytestmark = [
    pytest.mark.skipif(not SPS_HOME, reason="MISSING $SPS_HOME: NLR tests did NOT run"),
    pytest.mark.skipif(not _nlr_file().is_file(),
                       reason=f"MISSING {_nlr_file()}: NLR tests did NOT run "
                              "(build it with scripts/agn2/build_nlr_h5.py)"),
]

KEYS = ("nlr_logz", "nlr_logu", "nlr_xid", "nlr_lognh", "nlr_alpha")


@pytest.fixture(scope="module")
def nlr():
    from ceridwen.agn.nlr import FeltreNLR
    return FeltreNLR(mode="corrected")


@pytest.fixture(scope="module")
def table():
    import h5py
    with h5py.File(_nlr_file(), "r") as h:
        return {"L": h["L"][()], "ratio": h["ratio"][()],
                "src": [s.decode() for s in h["lines/source_name"][()]],
                "z_abs": h["axes/Z"][()], "logu": h["axes/logu"][()], "xid": h["axes/xid"][()],
                "nh": h["axes/nh"][()], "alpha": h["axes/alpha"][()]}


def _theta(nlr, idx, elum=1.0):
    th = {k: float(nlr.axes[k][i]) for k, i in zip(KEYS, idx)}
    th["agn_elum"] = elum
    return th


def test_D_nlr_1_cube_equals_raw_files(table):
    if not RAW_DIR.is_dir():
        pytest.skip(f"MISSING {RAW_DIR}: D-nlr-1 did NOT run (set $CERIDWEN_NLR_RAW)")
    names = sorted(RAW_DIR.glob("nlr_nebular_Z*.txt"))
    assert len(names) == table["z_abs"].size
    n = 0
    for p in names:
        z = float("0." + p.stem.split("_Z")[1])
        iz = int(np.flatnonzero(np.isclose(table["z_abs"], z, rtol=0, atol=1e-12))[0])
        cols = open(p).readline()[1:].split()
        assert cols[4:] == table["src"]
        for row in np.loadtxt(p):
            iu = int(np.flatnonzero(table["logu"] == row[0])[0])
            ix = int(np.flatnonzero(table["xid"] == row[1])[0])
            ih = int(np.flatnonzero(table["nh"] == row[2])[0])
            ia = int(np.flatnonzero(table["alpha"] == row[3])[0])
            np.testing.assert_array_equal(table["L"][:, iz, iu, ix, ih, ia], row[4:])
            n += 1
    assert n == table["L"][0].size          # every node seen once (5184)
    hb = table["src"].index("Hbeta")
    np.testing.assert_array_equal(table["ratio"], table["L"] / table["L"][hb][None])


def test_D_nlr_2_nodes_exact(nlr, table):
    src = table["src"]
    keep = [src.index(s) for s in nlr.source_names]
    hb = src.index("Hbeta")
    f = jax.jit(nlr.ratios)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(400):
        idx = [int(rng.integers(nlr.axes[k].size)) for k in KEYS]
        ref = table["L"][(keep,) + tuple(idx)] / table["L"][(hb,) + tuple(idx)]
        got = np.asarray(f(_theta(nlr, idx)))
        worst = max(worst, float(np.max(np.abs(got / ref - 1.0))))
    assert worst <= 1e-12, worst
    # H beta itself is exactly 1 at any point
    assert nlr.ratios(_theta(nlr, [0, 0, 0, 0, 0]))[nlr.line_names.index("Ba-beta 4861")] == 1.0


def test_D_nlr_3_off_node_vs_scipy(nlr):
    from scipy.interpolate import RegularGridInterpolator
    axes = [nlr.axes[k] for k in KEYS]
    logr = np.log10(nlr.ratio_nodes)
    interp = RegularGridInterpolator(axes, np.moveaxis(logr, 0, -1), method="linear")
    rng = np.random.default_rng(2)
    f = jax.jit(nlr.ratios)
    worst = 0.0
    for _ in range(300):
        p = [rng.uniform(a[0], a[-1]) for a in axes]
        th = dict(zip(KEYS, p))
        got = np.asarray(f(th))
        ref = 10.0 ** interp(p)[0]
        worst = max(worst, float(np.max(np.abs(got / ref - 1.0))))
        # bracketing: inside the min/max of the 32 corners
        idx = [np.clip(np.searchsorted(a, x) - 1, 0, a.size - 2) for a, x in zip(axes, p)]
        sl = (slice(None),) + tuple(slice(i, i + 2) for i in idx)
        corners = nlr.ratio_nodes[sl].reshape(nlr.n_lines, -1)
        assert np.all(got >= corners.min(1) * (1 - 1e-12))
        assert np.all(got <= corners.max(1) * (1 + 1e-12))
    assert worst <= 1e-12, worst


def test_D_nlr_4_clamped_outside(nlr):
    lo = {k: float(nlr.axes[k][0]) for k in KEYS}
    hi = {k: float(nlr.axes[k][-1]) for k in KEYS}
    for k in KEYS:
        span = hi[k] - lo[k]
        a = dict(lo); a[k] = lo[k] - span
        np.testing.assert_array_equal(nlr.ratios(a), nlr.ratios(lo))
        b = dict(hi); b[k] = hi[k] + span
        np.testing.assert_array_equal(nlr.ratios(b), nlr.ratios(hi))


def test_D_nlr_5_linear_in_elum(nlr):
    th = {"nlr_logz": -0.4, "nlr_logu": -2.3, "nlr_xid": 0.25, "nlr_lognh": 2.7,
          "nlr_alpha": -1.6}
    r = nlr.ratios(th)
    for e in (0.0, 1e-4, 3.7e-2):
        np.testing.assert_array_equal(nlr.line_luminosity({**th, "agn_elum": e}), e * r)
    assert np.all(np.asarray(nlr.line_luminosity({**th, "agn_elum": 0.0})) == 0.0)


def test_D_nlr_6_jit_vmap_grad(nlr):
    rng = np.random.default_rng(3)
    for W in (64, 100):
        th = {k: jnp.asarray(rng.uniform(nlr.axes[k][0], nlr.axes[k][-1], W)) for k in KEYS}
        th["agn_elum"] = jnp.asarray(rng.uniform(0, 1e-2, W))
        batched = np.asarray(jax.jit(jax.vmap(nlr.line_luminosity))(th))
        loop = np.stack([np.asarray(nlr.line_luminosity({k: v[i] for k, v in th.items()}))
                         for i in range(W)])
        np.testing.assert_allclose(batched, loop, rtol=1e-14, atol=0)

    def f(t):
        return jnp.sum(jnp.log(nlr.line_luminosity(t)) * jnp.arange(1, nlr.n_lines + 1))

    g = jax.jit(jax.grad(f))
    centre = {k: 0.5 * (nlr.axes[k][0] + nlr.axes[k][-1]) for k in KEYS}
    centre["agn_elum"] = 1e-3
    pts = [centre]
    for k in KEYS:
        pts.append({**centre, k: float(nlr.axes[k][0])})
        pts.append({**centre, k: float(nlr.axes[k][-1])})
    for p in pts:
        gr = g({k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()})
        assert all(np.isfinite(float(v)) for v in gr.values()), (p, gr)
    # central finite differences inside a cell (away from nodes: piecewise linear)
    p = {"nlr_logz": -0.71, "nlr_logu": -2.27, "nlr_xid": 0.23, "nlr_lognh": 2.61,
         "nlr_alpha": -1.53, "agn_elum": 2e-3}
    gr = g({k: jnp.asarray(v) for k, v in p.items()})
    for k in KEYS + ("agn_elum",):
        h = 1e-6 * max(1.0, abs(p[k]))
        fd = (f({**p, k: p[k] + h}) - f({**p, k: p[k] - h})) / (2 * h)
        assert abs(float(gr[k]) - float(fd)) <= 1e-6 * max(1.0, abs(float(fd))), (k, gr[k], fd)


def test_D_nlr_7_refusals(nlr, tmp_path):
    from ceridwen.agn.nlr import FeltreNLR
    from ceridwen.priors import Uniform
    from ceridwen.csp.csp import prior_support
    for mode in ("fsps", "prospector"):
        with pytest.raises(ValueError, match="a42"):
            FeltreNLR(mode=mode)
    with pytest.raises(FileNotFoundError, match="build_nlr_h5.py"):
        FeltreNLR(mode="corrected", file=tmp_path / "nope.h5")
    # an FSPS list without '[O III] 5007'
    fake = tmp_path / "sps"
    (fake / "data").mkdir(parents=True)
    src = pathlib.Path(SPS_HOME) / "data" / "emlines_info.dat"
    txt = src.read_text().replace("[O III] 5007", "[O III] 5008")
    (fake / "data" / "emlines_info.dat").write_text(txt)
    with pytest.raises(ValueError, match=r"\[O III\] 5007"):
        FeltreNLR(mode="corrected", sps_home=str(fake))
    # values / priors outside the grid
    lo, hi = nlr.ranges["nlr_logu"]
    assert nlr.range_messages({"nlr_logu": lo}) == []
    assert "outside" in nlr.range_messages({"nlr_logu": hi + 0.1})[0]
    with pytest.raises(ValueError, match="nlr_logu"):
        nlr.check_theta({"nlr_logu": lo - 0.1})
    nlr.check_prior_ranges({"nlr_logu": Uniform(low=lo, high=hi)}, prior_support)
    with pytest.raises(ValueError, match="outside the Feltre"):
        nlr.check_prior_ranges({"nlr_logu": Uniform(low=lo - 0.5, high=hi)}, prior_support)
