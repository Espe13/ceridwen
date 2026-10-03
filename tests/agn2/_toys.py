"""Toy AGN2 components for the scaffold tests (session VAL, Phase A).

They implement the contracts of ``ceridwen/agn/__init__.py`` (``SkirtorTorus``, ``PolarDust``,
``AGN2LineSet``) with simple closed forms, so every quantity the hook and the CSP compute from
them can be recomputed here independently. Setup in float64 NumPy; the traced methods are pure
JAX. They are NOT the physics (the TORUS / NLR / BLR sessions write that); they test the wiring.

* ``ToySkirtor``: disc = fagn L_bol * a_disk(theta) * g(ln lam; 1500 A, 0.6), dust = fagn L_bol *
  a_dust(theta) * g(ln lam; 20 um, 0.5), each shape normalised to unit integral over frequency
  with the DL07 trapezoid weights; a_disk, a_dust smooth in every SKIRTOR key.
* ``ToyPolar``: T = exp(-polar_ebv * 3 (lam / 5500)^-1.2); re-emission = grey body
  nu^polar_emis B_nu(polar_temp) normalised to the frequency integral of disk * (1 - T).
* ``ToyNLR`` / ``ToyBLR``: FSPS lines by name; L = amplitude * ratio(theta).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ceridwen.agn import (AGN2_THETA_KEYS, AGN2LineSet, PolarDust, SkirtorTorus)
from ceridwen.agn.lines import read_emlines_info
from ceridwen.agn.torus import trapezoid_nu_weights

H_OVER_K = 6.62607015e-27 / 1.380649e-16          # s K
C_AA_S = 2.99792458e18

TOY_DEFAULTS = {
    "fagn": 0.0, "skirtor_t": 7.0, "skirtor_pl": 1.0, "skirtor_q": 1.0, "skirtor_oa": 40.0,
    "skirtor_i": 30.0,
    "polar_ebv": 0.0, "polar_temp": 100.0, "polar_emis": 1.6,
    "agn_elum": 0.0, "agn_eline_sigma": 100.0, "nlr_logu": -2.5, "nlr_logz": 0.0,
    "nlr_alpha": -1.7, "nlr_xid": 0.3, "nlr_lognh": 3.0,
    "blr_lum": 0.0, "blr_sigma": 2000.0,
}


def _w(wave):
    return -np.asarray(trapezoid_nu_weights(np.asarray(wave, dtype=np.float64)), dtype=np.float64)


def _unit_shape(wave, lam0, width):
    lam = np.asarray(wave, dtype=np.float64)
    g = np.exp(-0.5 * (np.log(lam / lam0) / width) ** 2)
    return g / float(g @ _w(lam))


def _get(theta, k):
    v = theta[k] if k in theta else TOY_DEFAULTS[k]
    return jnp.ravel(jnp.asarray(v, dtype=jnp.float64))[0]


def disk_amp(th):
    """Smooth in every SKIRTOR key; 1 at the defaults."""
    i = jnp.deg2rad(_get(th, "skirtor_i"))
    oa = jnp.deg2rad(_get(th, "skirtor_oa"))
    return (jnp.cos(i) / jnp.cos(jnp.deg2rad(30.0))) * (1.0 + 0.1 * (oa - jnp.deg2rad(40.0)))


def dust_amp(th):
    t = _get(th, "skirtor_t")
    return (t / 7.0) ** 0.3 * jnp.exp(0.05 * (_get(th, "skirtor_pl") - 1.0)
                                      + 0.03 * (_get(th, "skirtor_q") - 1.0))


class ToySkirtor(SkirtorTorus):
    param_names = AGN2_THETA_KEYS["skirtor16"]

    def __init__(self, wave, *, mode, R, file):
        self.mode, self.R, self.file = mode, R, file
        self.wave = np.asarray(wave, dtype=np.float64)
        self.disk_shape = jnp.asarray(_unit_shape(self.wave, 1500.0, 0.6))
        self.dust_shape = jnp.asarray(_unit_shape(self.wave, 2.0e5, 0.5))

    def default_params(self):
        return {k: TOY_DEFAULTS[k] for k in self.param_names}

    def rest_parts(self, theta, ctx):
        L = _get(theta, "fagn") * ctx.L_bol_star
        return {"disk": L * disk_amp(theta) * self.disk_shape,
                "dust": L * dust_amp(theta) * self.dust_shape}

    def provenance(self):
        return {"file": "toy", "R": self.R}


class ToyPolar(PolarDust):
    param_names = AGN2_THETA_KEYS["polar"]

    def __init__(self, wave, *, mode, law):
        self.mode, self.law = mode, law
        self.wave = np.asarray(wave, dtype=np.float64)
        self.k = jnp.asarray(3.0 * (self.wave / 5500.0) ** -1.2)
        self.w = jnp.asarray(_w(self.wave))
        self.nu = jnp.asarray(C_AA_S / self.wave)

    def default_params(self):
        return {k: TOY_DEFAULTS[k] for k in self.param_names}

    def transmission(self, theta):
        return jnp.exp(-_get(theta, "polar_ebv") * self.k)

    def reemission(self, theta, disk, transmission, ctx):
        L_abs = jnp.dot(disk * (1.0 - transmission), self.w)
        T = _get(theta, "polar_temp")
        x = H_OVER_K * self.nu / T
        nu_rel = self.nu / 1e13
        shape = nu_rel ** (3.0 + _get(theta, "polar_emis")) / jnp.expm1(jnp.minimum(x, 700.0))
        shape = shape / jnp.dot(shape, self.w)
        return L_abs * shape


class _ToyLines(AGN2LineSet):
    NAMES = ()

    def __init__(self, *, mode, file, sps_home):
        self.mode, self.file = mode, file
        wave, names, path = read_emlines_info(sps_home)
        idx = [names.index(n) for n in self.NAMES]
        self.line_names = tuple(self.NAMES)
        self.line_wave = wave[idx].copy()
        self.fsps_index = np.asarray(idx, dtype=np.int64)
        self.emlines_wave, self.emlines_names, self.emlines_file = wave, names, str(path)

    def default_params(self):
        return {k: TOY_DEFAULTS[k] for k in self.param_names}

    def sigma_kms(self, theta):
        return _get(theta, self.width_key)


class ToyNLR(_ToyLines):
    param_names = AGN2_THETA_KEYS["feltre16"]
    region = "nlr"
    width_key = "agn_eline_sigma"
    NAMES = ("Ba-beta 4861", "[O III] 5007", "Ba-alpha 6563", "[N II] 6584")

    def ratios(self, theta):
        u = _get(theta, "nlr_logu")
        z = _get(theta, "nlr_logz")
        a = _get(theta, "nlr_alpha")
        x = _get(theta, "nlr_xid")
        n = _get(theta, "nlr_lognh")
        o3 = 10.0 ** (1.0 + 0.2 * (u + 2.5) + 0.1 * z + 0.05 * (a + 1.7) - 0.1 * (x - 0.3)
                      + 0.02 * (n - 3.0))
        n2 = 10.0 ** (0.0 - 0.1 * (u + 2.5) + 0.3 * z)
        return jnp.stack([jnp.ones_like(o3), o3, 2.86 * jnp.ones_like(o3), n2])

    def line_luminosity(self, theta):
        return _get(theta, "agn_elum") * self.ratios(theta)


class ToyBLR(_ToyLines):
    param_names = AGN2_THETA_KEYS["blr"]
    region = "blr"
    width_key = "blr_sigma"
    NAMES = ("Ba-alpha 6563", "Ba-beta 4861")
    RATIOS = np.array([1.0, 1.0 / 3.1])

    def line_luminosity(self, theta):
        return _get(theta, "blr_lum") * jnp.asarray(self.RATIOS)


TOYS = {"skirtor16": ToySkirtor, "polar": ToyPolar, "feltre16": ToyNLR, "blr": ToyBLR}
