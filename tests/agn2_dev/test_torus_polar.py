"""CIGALE polar dust pieces (session TORUS development tests).

T-PO-1  k_ext above 100 nm == the closed forms of skirtor2016.py l.57-99, written out again here.
T-PO-2  below 100 nm: the vendored curve / (ext(440) - ext(550)), joined at the last point < 100 nm.
T-PO-3  vendored curves are CIGALE's files (sha256).
T-PO-4  grey body == CIGALE's expression (l.371-384) where exp does not overflow; finite and
        with finite gradients in T and beta where it does.
T-PO-5  transmission: 10^(-0.4 k E(B-V)); E(B-V) = 0 -> exactly 1.
T-PO-6  generic reemission: frequency integral == that of disk (1 - T) on the model grid.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import scipy.constants as cst

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ceridwen.agn import AGNContext  # noqa: E402
from ceridwen.agn.polar import (EXTFUN_FILES, EXTFUN_SHA256, PolarDust, greybody_shape,  # noqa
                                k_ext)

WL = np.geomspace(1.0, 1e7, 3000)          # nm


def _closed(wl, law):
    um = wl * 1e-3
    if law == 0:
        return 1.39 * um ** -1.2
    if law == 1:
        return np.where(wl < 630,
                        2.659 * (-2.156 + 1.509 / um - 0.198 / um ** 2 + 0.011 / um ** 3) + 4.05,
                        2.659 * (-1.857 + 1.040 / um) + 4.05)
    x = 1 / um
    a = np.where(x < 3.69, -0.8175 + 1.5848 * x - 0.3774 * x ** 2 + 0.0296 * x ** 3,
                 1.3468 + 0.0087 * x)
    return np.clip(a, 0.0, None) / 0.182


@pytest.mark.parametrize("law", [0, 1, 2])
def test_T_PO_1_closed_forms(law):
    k = k_ext(WL, law)
    m = WL >= 100.0
    np.testing.assert_allclose(k[m], _closed(WL, law)[m], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("law", [0, 1, 2])
def test_T_PO_2_short_wavelengths(law):
    k = k_ext(WL, law)
    d = np.loadtxt(EXTFUN_FILES[law])
    wl_c = d[:, 0] * 1e3
    ab, av = np.interp([440.0, 550.0], wl_c, d[:, 1])
    short = np.interp(WL, wl_c, d[:, 1] / (ab - av))
    m = WL < 100.0
    j = np.flatnonzero(m)[-1]
    scale = _closed(WL, law)[j] / short[j]
    np.testing.assert_allclose(k[m], short[m] * scale, rtol=1e-13)
    assert np.isclose(k[j], _closed(WL, law)[j], rtol=1e-13)


def test_T_PO_3_vendored_curves():
    for law, path in EXTFUN_FILES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == EXTFUN_SHA256[law]


def test_T_PO_4_greybody():
    c = cst.c * 1e9
    for T, beta in ((100.0, 1.6), (40.0, 2.0), (500.0, 1.0)):
        x = cst.h * c / (WL * cst.k * T)
        ok = x < 600
        ref = (c / WL ** 2 * (1 - np.exp(-((200e3 / WL) ** beta))) * (c / WL) ** 3
               / (np.exp(np.where(ok, x, 0.0)) - 1.0))
        got = np.asarray(greybody_shape(jnp.asarray(WL), T, beta))
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-11)
        assert np.all(np.isfinite(got)) and np.all(got >= 0)
        assert np.all(got[x > 745] == 0.0)
    g = jax.grad(lambda tb: jnp.sum(greybody_shape(jnp.asarray(WL), tb[0], tb[1])))(
        jnp.array([30.0, 1.6]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_T_PO_5_transmission():
    wave = WL * 10.0
    pd = PolarDust(wave, mode="cigale", law="smc")
    t0 = pd.transmission({"polar_ebv": jnp.asarray(0.0)})
    assert np.all(np.asarray(t0) == 1.0)
    t = np.asarray(pd.transmission({"polar_ebv": jnp.asarray(0.3)}))
    np.testing.assert_allclose(t, 10 ** (-0.4 * k_ext(WL, 0) * 0.3), rtol=1e-14)


def test_T_PO_6_generic_reemission_energy():
    from ceridwen.agn.torus import trapezoid_nu_weights
    wave = WL * 10.0
    pd = PolarDust(wave, mode="corrected", law="gaskell")
    th = {"polar_ebv": jnp.asarray(0.4), "polar_temp": jnp.asarray(120.0),
          "polar_emis": jnp.asarray(1.6)}
    disk = jnp.asarray(np.exp(-0.5 * (np.log(wave / 2000.0) / 0.7) ** 2))
    t = pd.transmission(th)
    ctx = AGNContext(jnp.asarray(wave), jnp.asarray(1.0), None, jnp.ones(wave.size), "corrected")
    ir = pd.reemission(th, disk, t, ctx)
    w = -np.asarray(trapezoid_nu_weights(wave))
    assert abs(float(np.asarray(ir) @ w) / float(np.asarray(disk * (1 - t)) @ w) - 1) < 1e-12
