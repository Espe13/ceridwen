"""Development tests of the Feltre+16 NLR through the CSP (``AGNConfig(nlr_model="feltre16")``).

D-nlr-8   Lines observation: the AGN part of every observed line (incl. the [O II] doublet as
          one component) equals agn_elum x ratio x formed mass x flux factor x T_gal at the line,
          recomputed from the grid file.
D-nlr-9   agn_elum = 0: every prediction byte-identical to the same CSP's agn_elum = 0 with the
          a42 lines, i.e. the NLR adds exactly zero (Spectrum, Photometry, Lines).
D-nlr-10  jit(value_and_grad) of a chi^2 through predict: finite in every NLR key and agn_elum.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import test_emission_lines as T  # noqa: E402

from ceridwen import CSPBasis, Cosmology, SSPData  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402


def _nlr_file():
    from ceridwen.agn.nlr import default_nlr_file
    return default_nlr_file()


pytestmark = [
    pytest.mark.skipif(not os.environ.get("SPS_HOME"),
                       reason="MISSING $SPS_HOME: NLR CSP tests did NOT run"),
    pytest.mark.skipif(not _nlr_file().is_file(),
                       reason=f"MISSING {_nlr_file()}: NLR CSP tests did NOT run"),
]

NLR = {"agn_elum": 3e-3, "agn_eline_sigma": 150.0, "nlr_logu": -2.2, "nlr_logz": -0.35,
       "nlr_alpha": -1.55, "nlr_xid": 0.4, "nlr_lognh": 2.6}
_CACHE = {}


def _csp(nlr_model="feltre16", igm=False):
    key = (nlr_model, igm)
    if key in _CACHE:
        return _CACHE[key]
    path = T.find_test_grid()
    if path is None:
        pytest.skip("MISSING test SSP grid: test did NOT run")
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(path)),
                   lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=igm, add_dust_emission=False, verbose=False,
                   cosmo=cosmo, sps_home=os.environ["SPS_HOME"],
                   agn=AGNConfig(nlr_model=nlr_model, torus=False, disk=False))
    _CACHE[key] = csp
    return csp


def _setup(csp):
    spec, phot, lines = T._obs()
    model = T._model(csp, (spec, phot, lines))
    return model, lines


def _th(model, vals, dust2=0.4):
    th = T._theta(model, diffuse_tau_kc=jnp.array([dust2]))
    th.update({k: jnp.array([float(v)]) for k, v in vals.items()})
    return th


def test_D_nlr_8_lines_by_name():
    csp = _csp()
    model, lines = _setup(csp)
    nlr = csp._agn.model.lines
    assert type(nlr).__name__ == "FeltreNLR"
    th1 = _th(model, NLR)
    th0 = _th(model, dict(NLR, agn_elum=0.0))
    d = (np.asarray(model.predict(th1)["lines"], np.float64)
         - np.asarray(model.predict(th0)["lines"], np.float64))
    mt = T._csp_theta(model, th1)
    msum = float(jnp.sum(csp.calculate_ssp_weights(theta=mt)))
    scale = msum * 10 ** float(mt["logmass"][0]) * float(csp._flux_factor(mt)) / (1 + T.ZRED)
    wave = np.asarray(csp.wave)
    dc = np.exp(-np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64))
    # ratios from the file, by an independent scipy interpolation of log10 ratio
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    with h5py.File(_nlr_file(), "r") as h:
        ratio = h["ratio"][()]
        fs = [s.decode() for s in h["lines/fsps_name"][()]]
        ax = [np.log10(h["axes/Z"][()] / 0.01524), h["axes/logu"][()], h["axes/xid"][()],
              np.log10(h["axes/nh"][()]), h["axes/alpha"][()]]
    pt = [NLR["nlr_logz"], NLR["nlr_logu"], NLR["nlr_xid"], NLR["nlr_lognh"], NLR["nlr_alpha"]]
    r = 10 ** RegularGridInterpolator(ax, np.moveaxis(np.log10(ratio), 0, -1))(pt)[0]
    fw, fn, _ = __import__("ceridwen.agn.lines", fromlist=["x"]).read_emlines_info()
    exp = np.zeros(len(lines.line_components))
    for k, comp in enumerate(lines.line_components):
        for lam in comp:
            name = fn[int(np.argmin(np.abs(fw - lam)))]
            if name in fs:
                lw = fw[fn.index(name)]
                exp[k] += NLR["agn_elum"] * r[fs.index(name)] * scale * np.interp(lw, wave, dc)
    assert np.ptp(dc) > 0.1     # the galaxy screen is on
    assert np.all(exp > 0)          # Hb, [OIII], Ha, [NII], [OII] doublet all carry NLR flux
    np.testing.assert_allclose(d, exp, rtol=2e-6, atol=0)


def test_D_nlr_9_zero_adds_zero():
    csp = _csp()
    model, _ = _setup(csp)
    csp_a = _csp("a42")
    model_a, _ = _setup(csp_a)
    p = model.predict(_th(model, dict(NLR, agn_elum=0.0)))
    q = model_a.predict(_th(model_a, {"agn_elum": 0.0, "agn_eline_sigma": 150.0}))
    for k in p:
        a, b = np.asarray(p[k]), np.asarray(q[k])
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), k


def test_D_nlr_10_grad_through_predict():
    csp = _csp()
    model, _ = _setup(csp)
    base = _th(model, NLR)
    keys = list(NLR)

    def chi2(v):
        th = dict(base)
        th.update({k: v[i][None] for i, k in enumerate(keys)})
        out = model.predict(th)
        return sum(jnp.sum(jnp.asarray(out[o], jnp.float64) ** 2) for o in out)

    v0 = jnp.asarray([NLR[k] for k in keys])
    val, g = jax.jit(jax.value_and_grad(chi2))(v0)
    assert np.isfinite(float(val)) and np.all(np.isfinite(np.asarray(g))), g
    gi = dict(zip(keys, np.asarray(g)))
    assert gi["agn_elum"] != 0 and gi["nlr_logu"] != 0 and gi["nlr_logz"] != 0
