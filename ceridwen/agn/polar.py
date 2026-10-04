"""CIGALE's polar dust (X-CIGALE, Yang et al. 2020): extinction of the SKIRTOR disc in the
polar cone and its grey-body re-emission.

Reference: ``pcigale/sed_modules/skirtor2016.py`` at cigale 29cb909f (2025.1), functions
``k_ext_short``, ``k_ext`` and ``SKIRTOR2016._init_code`` (lines 341-395). This module holds the
pieces of that code that do not depend on the SKIRTOR templates; ``skirtor.SkirtorTorus``
composes them with the templates on the templates' own wavelength grid (``nuclear_parts``),
because CIGALE's polar dust is not separable from the torus (see below).

Extinction ``k(lambda) = A(lambda) / E(B-V)``, ``polar_law``:

``"smc"`` (law 0)       ``1.39 (lambda/um)^-1.2`` (Bongiorno+2012)
``"calzetti"`` (law 1)  Calzetti+2000 (``dustatt_calzleit.py``): 120-630 nm and >= 630 nm branches
``"gaskell"`` (law 2)   Gaskell+2004 appendix, ``A/A_V / 0.182``, negative values set to 0

Below 100 nm the law is replaced by CIGALE's dust-model curve (``extFun_SMC`` for law 0,
``extFun_MRN`` for law 1, ``extFun_Gaskel04`` for law 2; ``k = ext / (ext(440 nm) -
ext(550 nm))``), scaled to match the analytic law at the LAST grid point below 100 nm. That point
depends on the grid, so ``k_ext`` is a function of the grid it is evaluated on; the curves are
vendored as ``data/cigale_extFun_*.txt`` (unchanged; sha256 in ``EXTFUN_SHA256``). CIGALE's
``interp1d`` raises below 1 nm (the curves start there); here the curve is held constant below
1 nm (only the model-grid ``transmission`` reaches there; the SKIRTOR grid starts at 1 nm).

Transmission ``10^(-0.4 k E(B-V))``. CIGALE's law 1 polynomial is negative above ~3.1 um (k < 0,
transmission > 1, light created); ``"cigale"`` keeps it (parity), ``"corrected"`` clips k at 0
(no polar extinction there), so every transmission is <= 1 and the absorbed energy >= 0.

Re-emission (CIGALE, l. 351-386): the luminosity the polar dust absorbs is the ANGLE-AVERAGED
absorption of the face-on disc, ``l_ext = (7/18 - sin^2(oa)/6 - 2 sin^3(oa)/9) x
trapz(L_disc(i=0) (1 - T))`` (an anisotropic disc ``cos(theta)(1 + 2 cos(theta))`` integrated
over the cone ``theta < 90 - oa``), re-emitted as a Casey (2012) modified black body,
``(c/lambda^2) (1 - exp(-(lambda0/lambda)^beta)) (c/lambda)^3 / (exp(hc/lambda k T) - 1)``,
``lambda0 = 200 um``, normalised to ``l_ext`` with ``trapz`` over the grid. Only a type-1 view
(``i <= 90 - oa``) sees the disc through the polar dust; a type-2 view sees the disc unextinguished
by it but still receives the re-emission. Then torus dust, grey body and disc are renormalised
together (``1 / trapz(dust + grey body)``), which is why ``SkirtorTorus`` does the composition.

The grey body here uses ``exp(-x) / (-expm1(-x))`` for ``1 / (exp(x) - 1)`` (equal in exact
arithmetic, finite with finite gradients where ``exp(x)`` overflows, which CIGALE silences with
``np.seterr``).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import AGN2_THETA_KEYS, AGN_MODE_CIGALE, AGN_MODE_CORRECTED, POLAR_LAWS
from . import PolarDust as _PolarDustContract

__all__ = ["PolarDust", "POLAR_DEFAULTS", "LAW_INDEX", "k_ext", "k_ext_short",
           "greybody_shape", "polar_geometry", "EXTFUN_FILES", "EXTFUN_SHA256"]

# CIGALE skirtor2016 ``parameters`` defaults: EBV 0.03, temperature 100 K, emissivity 1.6
POLAR_DEFAULTS = {"polar_ebv": 0.03, "polar_temp": 100.0, "polar_emis": 1.6}
LAW_INDEX = {"smc": 0, "calzetti": 1, "gaskell": 2}
_DATA = Path(os.path.dirname(__file__)) / "data"
EXTFUN_FILES = {0: _DATA / "cigale_extFun_SMC.txt", 1: _DATA / "cigale_extFun_MRN.txt",
                2: _DATA / "cigale_extFun_Gaskel04.txt"}
EXTFUN_SHA256 = {
    0: "6200117d5f1a5f5fccc9be1e5b02e408541742e836aec458d72b608878e3b189",
    1: "19b3dc8e7f9752500070a63e2684e548464bcc40afb77588108db5de7e6334a0",
    2: "927a8903d55bf2159f2c864446e41622e1eee3f0bd0dcc20049378e904a8cc95",
}
LAMBDA0_NM = 200e3                       # Casey (2012) lambda_0, skirtor2016.py l.372
# scipy.constants (CODATA 2018 exact values), as CIGALE's ``cst``
C_NM_S = 299792458.0 * 1e9
H_SI = 6.62607015e-34
K_SI = 1.380649e-23

_SHORT_CACHE = {}


def k_ext_short(law):
    """``(wl_nm, k_short)`` of CIGALE's ``k_ext_short``: the dust-model curve divided by
    ``ext(440 nm) - ext(550 nm)`` (``np.interp`` on the curve, as CIGALE)."""
    if law not in EXTFUN_FILES:
        raise ValueError(f"Extinction law {law} unknown (0 SMC, 1 Calzetti, 2 Gaskell).")
    hit = _SHORT_CACHE.get(law)
    if hit is None:
        d = np.loadtxt(EXTFUN_FILES[law], comments="#", dtype=np.float64)
        wl = d[:, 0] * 1e3
        ab, av = np.interp([440.0, 550.0], wl, d[:, 1])
        hit = (wl, d[:, 1] / (ab - av))
        _SHORT_CACHE[law] = hit
    return hit


def k_ext(wavelength, law):
    """``k(lambda) = A(lambda)/E(B-V)`` on ``wavelength`` [nm], CIGALE's ``k_ext`` (numpy,
    float64). ``law``: 0 SMC, 1 Calzetti+2000, 2 Gaskell+2004."""
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if law == 0:
        k = 1.39 * (wavelength * 1e-3) ** -1.2
    elif law == 1:
        result = np.zeros(len(wavelength))
        mask = wavelength < 630
        result[mask] = (2.659 * (-2.156 + 1.509e3 / wavelength[mask]
                                 - 0.198e6 / wavelength[mask] ** 2
                                 + 0.011e9 / wavelength[mask] ** 3) + 4.05)
        mask = wavelength >= 630
        result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05
        k = result
    elif law == 2:
        x = 1e3 / wavelength
        alam_av = np.zeros(len(wavelength))
        mask = x < 3.69
        alam_av[mask] = (-0.8175 + 1.5848 * x[mask] - 0.3774 * x[mask] ** 2
                         + 0.0296 * x[mask] ** 3)
        mask = x >= 3.69
        alam_av[mask] = 1.3468 + 0.0087 * x[mask]
        alam_av[alam_av < 0.0] = 0.0
        k = alam_av / 0.182
    else:
        raise KeyError("Extinction law is different from the expected ones")
    mask = np.where(wavelength < 100.0)
    if mask[0].size > 0:
        wl_s, k_s = k_ext_short(law)
        # CIGALE: interp1d (linear, raises outside the curve); here np.interp (linear, held
        # constant below the curve's 1 nm)
        k_short = np.interp(wavelength[mask], wl_s, k_s)
        k[mask] = k_short * (k[mask][-1] / k_short[-1])
    return k


def polar_geometry(oa_deg):
    """``7/18 - sin^2(oa)/6 - 2 sin^3(oa)/9`` (skirtor2016.py l.365-368), traced."""
    s = jnp.sin(jnp.deg2rad(oa_deg))
    return 7.0 / 18.0 - s ** 2 / 6.0 - s ** 3 * 2.0 / 9.0


def greybody_shape(wl_nm, temp, emis):
    """Casey (2012) modified black body of skirtor2016.py l.371-384 on ``wl_nm`` (F_lambda,
    arbitrary units), traced in ``temp`` [K] and ``emis``."""
    wl = jnp.asarray(wl_nm)
    x = H_SI * C_NM_S / (wl * K_SI * temp)
    planck = jnp.exp(-x) / (-jnp.expm1(-x))                    # 1 / (exp(x) - 1)
    return (C_NM_S / wl ** 2) * (1.0 - jnp.exp(-((LAMBDA0_NM / wl) ** emis))) \
        * (C_NM_S / wl) ** 3 * planck


def _get(theta, key):
    v = theta[key] if key in theta else POLAR_DEFAULTS[key]
    return jnp.ravel(jnp.asarray(v, dtype=jnp.float64))[0]


class PolarDust(_PolarDustContract):
    """CIGALE polar dust: ``polar_ebv`` (E(B-V) [mag]), ``polar_temp`` [K], ``polar_emis``.

    Parameters
    ----------
    wave : (n_wave,) rest model grid [A]
    mode : "corrected" or "cigale" (the same physics; see ``skirtor.SkirtorTorus``)
    law : "smc", "calzetti" or "gaskell" (CIGALE ``law`` 0, 1, 2)
    """

    param_names = AGN2_THETA_KEYS["polar"]

    def __init__(self, wave, *, mode, law):
        if mode not in (AGN_MODE_CORRECTED, AGN_MODE_CIGALE):
            raise ValueError(f"PolarDust(mode={mode!r}): the polar dust exists only in "
                             "agn_mode='corrected' or 'cigale'.")
        if law not in POLAR_LAWS:
            raise ValueError(f"PolarDust(law={law!r}): choose one of {POLAR_LAWS}.")
        wave = np.asarray(wave, dtype=np.float64)
        if wave.ndim != 1 or not np.all(np.isfinite(wave)) or not np.all(wave > 0) \
                or not np.all(np.diff(wave) > 0):
            raise ValueError("PolarDust: wave must be a 1-D ascending grid of positive "
                             "wavelengths [A].")
        for law_i, path in EXTFUN_FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"PolarDust: missing {path} (CIGALE extinction curve, "
                                        "shipped with ceridwen).")
        self.mode = mode
        self.law = law
        self.law_index = LAW_INDEX[law]
        self.wave = wave
        self.k_model = jnp.asarray(self.k_on(wave / 10.0))               # on the model grid

    def default_params(self) -> dict:
        return dict(POLAR_DEFAULTS)

    def k_on(self, wl_nm):
        """CIGALE's ``k_ext`` on a grid in nm (numpy; setup time). ``"corrected"``: clipped at 0
        (CIGALE's Calzetti polynomial is negative above ~3.1 um, which would make the
        transmission exceed 1; VERDICT_TORUS_r1 R2)."""
        k = k_ext(wl_nm, self.law_index)
        if self.mode == AGN_MODE_CORRECTED:
            k = np.maximum(k, 0.0)
        return k

    def transmission(self, theta):
        """(n_wave,) ``10^(-0.4 k E(B-V))`` on the model grid: the polar-cone extinction along a
        line of sight through it (a type-1 view). The SKIRTOR disc gets it on the templates'
        grid inside ``SkirtorTorus.nuclear_parts`` (for type-1 views only, as CIGALE)."""
        return 10.0 ** (-0.4 * self.k_model * _get(theta, "polar_ebv"))

    def reemission(self, theta, disk, transmission, ctx):
        """Grey body carrying the frequency integral of ``disk * (1 - transmission)`` on the
        model grid (the line-of-sight absorption): the scaffold's generic form. The SKIRTOR
        path does NOT use it: CIGALE re-emits the angle-averaged absorption of the face-on
        disc, which ``SkirtorTorus.nuclear_parts`` computes on the templates' grid."""
        from .torus import trapezoid_nu_weights
        w = -trapezoid_nu_weights(jnp.asarray(self.wave))
        l_abs = jnp.dot(disk * (1.0 - transmission), w)
        wl_nm = jnp.asarray(self.wave) / 10.0
        g_lam = greybody_shape(wl_nm, _get(theta, "polar_temp"), _get(theta, "polar_emis"))
        g_nu = g_lam * wl_nm ** 2 / C_NM_S
        norm = jnp.dot(g_nu, w)
        return l_abs * g_nu / jnp.where(norm > 0, norm, 1.0)

    def range_messages(self, theta) -> list:
        msgs = []
        for key, lo in (("polar_ebv", 0.0), ("polar_temp", 0.0), ("polar_emis", 0.0)):
            if key in theta:
                v = np.asarray(theta[key], float)
                if v.size and np.nanmin(v) < lo:
                    msgs.append(f"theta['{key}'] = {np.array2string(v, precision=3)} is "
                                f"negative; CIGALE requires {key} >= 0.")
        return msgs

    def provenance(self) -> dict:
        return {"law": self.law, "law_cigale": self.law_index,
                "extfun_file": str(EXTFUN_FILES[self.law_index]),
                "extfun_sha256": EXTFUN_SHA256[self.law_index]}
