"""
Intergalactic-medium absorption models: transmission curves ``exp(-tau * factor)``
on the rest-frame wavelength grid at a given source redshift.
"""
from __future__ import annotations

import abc
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from .constants import C_CMS

Array = jax.Array


class IGMModel(abc.ABC):
    """Abstract IGM attenuation model; subclasses implement ``tau(lam_rest, zred)``.

    A model that reads theta keys of its own lists them in ``param_names``; the CSP then passes
    ``attenuation(..., params={key: scalar})`` with those of the keys present in theta (and only
    then, so a subclass with the three-argument ``attenuation`` keeps working)."""

    name: str = "igm"
    param_names: tuple = ()

    @abc.abstractmethod
    def tau(self, lam_rest: Array, zred: Array) -> Array:
        """Optical depth on the rest-frame wavelength grid [Å] at source redshift ``zred``."""

    def attenuation(self, lam_rest: Array, zred: Array,
                    factor: Union[Array, float] = 1.0) -> Array:
        """Transmission curve ``exp(-tau * factor)``."""
        return jnp.exp(-self.tau(lam_rest, zred) * factor)


class NoIGM(IGMModel):
    """Identity IGM: zero optical depth at every wavelength."""

    name = "none"

    def tau(self, lam_rest, zred):
        return jnp.zeros_like(lam_rest)


class Madau1995(IGMModel):
    """Madau (1995) IGM attenuation: 17 Lyman-series lines, Ly-α metal blanketing and
    Lyman-continuum absorption; tau is capped at its short-wavelength peak and is zero at zred = 0."""

    name = "madau1995"

    # Lyman-series rest wavelengths [Å], Ly-α first
    _LYW = jnp.asarray([
        1215.67, 1025.72,  972.537, 949.743, 937.803,
         930.748, 926.226, 923.150, 920.963, 919.352,
         918.129, 917.181, 916.429, 915.824, 915.329,
         914.919, 914.576,
    ])

    _LYCOEFF = jnp.asarray([
        0.0036,    0.0017,    0.0011846, 0.0009410, 0.0007960,
        0.0006967, 0.0006236, 0.0005665, 0.0005200, 0.0004817,
        0.0004487, 0.0004200, 0.0003947, 0.000372,  0.000352,
        0.0003334, 0.00031644,
    ])

    _LYLIM   = 911.75    # Lyman limit [Å]
    _A_METAL = 0.0017

    def tau(self, lam_rest, zred):
        zred = jnp.maximum(zred, 0.0)
        lam  = lam_rest
        z1   = 1.0 + zred
        ratio = lam[None, :] / self._LYW[:, None]
        series = jnp.sum(jnp.where(lam[None, :] < self._LYW[:, None],
                                   self._LYCOEFF[:, None] * ratio ** 3.46, 0.0), axis=0)
        metal = self._A_METAL * (lam / self._LYW[0]) ** 1.68 * (lam < self._LYW[0])
        tau_series = series * z1 ** 3.46
        tau_metal  = metal * z1 ** 1.68

        mask_lyc = lam < self._LYLIM
        xc_safe = jnp.where(mask_lyc, lam / self._LYLIM * z1, 1.0)
        tau_lyc = (
             0.25  * xc_safe**3.0  * (z1**0.46 - xc_safe**0.46)
            + 9.4  * xc_safe**1.5  * (z1**0.18 - xc_safe**0.18)
            - 0.7  * xc_safe**3.0  * (xc_safe**(-1.32) - z1**(-1.32))
            - 0.023 * (z1**1.68 - xc_safe**1.68)
        )
        tau_lyc = jnp.where(mask_lyc, tau_lyc, 0.0)
        tau = tau_series + tau_metal + tau_lyc

        idx_peak = jnp.argmax(tau)
        tau = jnp.where(jnp.arange(tau.shape[0]) <= idx_peak, tau[idx_peak], tau)
        return jnp.where(zred > 0.0, tau, jnp.zeros_like(tau))


# ---- IGM damping wing and damped Ly-alpha absorber ------------------------------------------
# Ported from Prospector (prospect/models/sedmodel.py @ a78d153): DLA ``add_dla`` :808-819,
# ``voigt_profile`` :1201-1245, ``H`` :1192-1198; damping wing ``add_damping_wing`` :821-827,
# ``tau_damping`` :1261-1309, ``tau_gp`` :1312-1327, ``Ix`` :1330-1333.  Constants as there.
_C_CMS = C_CMS                 # :1233 has 2.99792e10 (1.5e-6 lower)
_VOIGT_CONST = 0.0149736082    # :1234, sqrt(pi) e^2 / (m_e c) [cgs]
_LYA = 1215.6696               # :1201, :1261
_F_LYA = 4.16e-1               # :1201
_GAMMA_LYA = 6.265e8           # :1201
_R_ALPHA = 2.02e-8             # :1297
_X_SAFE = 1e-3


def _voigt_H(a, x):
    """Tepper-Garcia (2006) Voigt-Hjerting approximation, with the ``x -> 0`` singularity moved to
    ``|x| = 1e-3`` (|lambda - l0| < 1.6e-4 A; tau > 3e6 there for any DLA column), so the value
    and the gradient are finite where Prospector returns NaN."""
    small = jnp.abs(x) < _X_SAFE
    x = jnp.where(small, jnp.where(x < 0, -_X_SAFE, _X_SAFE), x)
    P = x ** 2
    H0 = jnp.exp(-x ** 2)
    Q = 1.5 / x ** 2
    return H0 - a / jnp.sqrt(jnp.pi) / P * (H0 * H0 * (4. * P * P + 7. * P + 4. + Q) - Q - 1)


def voigt_tau(wave_abs, N, bkms=40.0, l0=_LYA, f=_F_LYA, gamma=_GAMMA_LYA):
    """Optical depth of a Voigt line (Krogager 2018 normalisation) on ``wave_abs`` [A, absorber
    rest frame] for a column density ``N`` [cm^-2]; Ly-alpha by default."""
    l0_cm = l0 * 1.e-8
    b = bkms * 1e5
    C_a = _VOIGT_CONST * f * l0_cm / b
    a = l0_cm * gamma / (4. * jnp.pi * b)
    x = (_C_CMS / b) * (1. - l0 / wave_abs)
    return C_a * N * _voigt_H(a, x)


def _Ix(x):
    v = x ** (9 / 2) / (1 - x) + 9 / 7 * x ** 3.5 + 9 / 5 * x ** 2.5 + 3 * x ** 1.5 + 9 * x ** 0.5
    return v - 9 / 2 * jnp.log((1 + x ** 0.5) / (1 - x ** 0.5))


def tau_gp(zred, h, Om0, Ob0, Y=0.25):
    """Gunn-Peterson optical depth, Miralda-Escude (1998) normalisation."""
    scale = (1 - Y) * h * Om0 ** (-0.5) * (Ob0 / 0.03)
    return 2.6e5 * scale * ((1 + zred) / 7) ** (3 / 2)


def tau_damping(wave_rest, zred, x_HI, h, Om0, Ob0, zmin=5.0, Y=0.25, l0=_LYA):
    """Damping-wing optical depth of a uniformly neutral IGM (neutral fraction ``x_HI``) between
    ``zmin`` and ``zred`` (Miralda-Escude 1998; Totani et al. 2006), on the rest-frame grid.
    Zero blueward of Ly-alpha and for ``zred <= zmin``; masked with safe arguments so ``jit``
    and gradients stay finite."""
    wave_obs = wave_rest * (1 + zred)
    zobs = wave_obs / l0 - 1
    red = ((zobs - zred) / (1 + zobs) > (100 * _R_ALPHA)) & (zred > zmin)
    x1 = jnp.where(red, (1 + zred) / (1 + zobs), 0.5)
    x2 = jnp.where(red, (1 + zmin) / (1 + zobs), 0.25)
    xx = _Ix(x1) - _Ix(x2)
    zobs_s = jnp.where(red, zobs, zred)
    tau = _R_ALPHA / jnp.pi * x_HI * tau_gp(zred, h, Om0, Ob0, Y=Y)
    tau = tau * ((1 + zobs_s) / (1 + zred)) ** (3 / 2) * xx
    return jnp.where(red, tau, 0.0)


class MadauDampingDLA(Madau1995):
    """Madau (1995) x IGM damping wing x damped Ly-alpha absorber (DLA).

        attenuation = exp(-(igm_factor * tau_Madau + tau_damp(x_HI) + tau_DLA(logN_HI, z_dla)))

    ``igm_factor`` scales the Madau forest only; the neutral fraction is its own parameter
    (Prospector instead reads ``x_HI`` from ``igm_factor``, ``sedmodel.py:824``).

    ``x_HI``, ``logN_HI`` and ``z_dla`` are theta keys like ``igm_factor``: the constructor
    values are the defaults, and a theta entry (fixed or sampled) overrides them.  A component
    is evaluated when its constructor value switches it on or its key is in theta; otherwise it
    is skipped, so ``MadauDampingDLA()`` is ``Madau1995`` bit for bit, and so are ``x_HI = 0``
    and ``logN_HI = -inf``.

    Parameters
    ----------
    Ob0 : float -- baryon density today; required for the damping wing (CERIDWEN's Cosmology
        carries no Ob0, so it is not guessed).  Planck18: 0.04897; WMAP9: 0.04628.
    cosmo : Cosmology or None -- supplies ``h`` and ``Om0``; ``None`` = the CSP's cosmology,
        bound at ``CSPBasis`` construction (a different one given here raises there).
    x_HI : float -- IGM neutral fraction (> 1 mimics a local overdensity); 0 = no damping wing.
    zmin : float -- lower redshift of the neutral IGM; no wing for ``zred <= zmin``.
    Y : float -- helium mass fraction.
    logN_HI : float or None -- DLA column [log10 cm^-2]; ``None`` = no DLA.
    z_dla : float or None -- absorber redshift; ``None`` = the source redshift.  No DLA for
        ``z_dla > zred``.  A foreground absorber is placed at rest wavelength
        ``1215.67 (1 + z_dla) / (1 + zred)`` (Prospector's ``add_dla``, ``sedmodel.py:816``,
        divides the other way and puts it redward of Ly-alpha; they agree at ``z_dla = zred``).
    b_kms : float -- Doppler parameter of the DLA [km/s].
    """

    name = "madau1995_damping_dla"
    param_names = ("x_HI", "logN_HI", "z_dla")

    def __init__(self, Ob0=None, cosmo=None, x_HI=0.0, zmin=5.0, Y=0.25,
                 logN_HI=None, z_dla=None, b_kms=40.0):
        x_HI = float(x_HI)
        if not x_HI >= 0.0:
            raise ValueError(f"x_HI must be >= 0, got {x_HI}")
        if Ob0 is not None and not float(Ob0) > 0.0:
            raise ValueError(f"Ob0 must be > 0, got {Ob0}")
        if logN_HI is not None and np.isnan(float(logN_HI)):
            raise ValueError("logN_HI is NaN; pass None for no DLA")
        self.x_HI = x_HI
        self.zmin = float(zmin)
        self.Y = float(Y)
        self.Ob0 = None if Ob0 is None else float(Ob0)
        self.cosmo = None
        self.h = self.Om0 = None
        if cosmo is not None:
            self.bind_cosmology(cosmo)
        self.logN_HI = None if logN_HI is None else float(logN_HI)
        self.z_dla = None if z_dla is None else float(z_dla)
        self.b_kms = float(b_kms)
        if self.x_HI > 0.0 and self.Ob0 is None:
            raise ValueError(
                f"x_HI = {self.x_HI} > 0 switches on the damping wing, which needs Ob0= "
                "(baryon density; Planck18 0.04897, WMAP9 0.04628): CERIDWEN's Cosmology "
                "has no Ob0, so it is not guessed")

    def bind_cosmology(self, cosmo):
        """Take ``h`` and ``Om0`` from ``cosmo``; a model already bound to a different
        cosmology raises (the damping wing and the flux factor would disagree)."""
        h, Om0 = float(cosmo.h), float(cosmo.Om0)
        if self.cosmo is not None and (h, Om0) != (self.h, self.Om0):
            raise ValueError(
                f"{type(self).__name__} was built with cosmology h={self.h}, Om0={self.Om0} "
                f"but the CSP uses h={h}, Om0={Om0}; pass the same Cosmology, or cosmo=None "
                "to take the CSP's")
        self.cosmo, self.h, self.Om0 = cosmo, h, Om0

    @staticmethod
    def _get(params, key, default):
        if params is not None and key in params:
            return jnp.ravel(jnp.asarray(params[key]))[0]
        return default

    # --- components -----------------------------------------------------------------------
    def tau_madau(self, lam_rest, zred):
        return super().tau(lam_rest, zred)

    def tau_damp(self, lam_rest, zred, params=None):
        """Damping-wing optical depth; ``params['x_HI']`` overrides the constructor value."""
        sampled = params is not None and "x_HI" in params
        if not sampled and self.x_HI == 0.0:
            return jnp.zeros_like(lam_rest)
        if self.Ob0 is None:
            raise ValueError(
                "theta['x_HI'] switches on the damping wing, which needs the baryon density: "
                "build the model as MadauDampingDLA(Ob0=...) and pass it as CSPBasis(igm_model=...)")
        if self.h is None:
            raise ValueError(
                "MadauDampingDLA has no cosmology: pass cosmo= or use it through "
                "CSPBasis(igm_model=...), which binds the CSP's")
        x_HI = self._get(params, "x_HI", self.x_HI)
        return tau_damping(lam_rest, zred, x_HI, self.h, self.Om0, self.Ob0,
                           zmin=self.zmin, Y=self.Y)

    def tau_dla(self, lam_rest, zred, params=None):
        """DLA Voigt optical depth; ``params['logN_HI']`` / ``params['z_dla']`` override."""
        sampled = params is not None and "logN_HI" in params
        if not sampled and self.logN_HI is None:
            return jnp.zeros_like(lam_rest)
        N = 10.0 ** self._get(params, "logN_HI", self.logN_HI)
        z_abs = self._get(params, "z_dla", zred if self.z_dla is None else self.z_dla)
        wave_abs = lam_rest * (1 + zred) / (1 + z_abs)
        tau = voigt_tau(wave_abs, N, bkms=self.b_kms)
        return jnp.where(z_abs > zred, 0.0, tau)

    # --- IGMModel interface ----------------------------------------------------------------
    def tau(self, lam_rest, zred, params=None):
        return (self.tau_madau(lam_rest, zred) + self.tau_damp(lam_rest, zred, params)
                + self.tau_dla(lam_rest, zred, params))

    def attenuation(self, lam_rest, zred, factor=1.0, params=None):
        """``exp(-(factor * tau_Madau + tau_damp + tau_DLA))``: ``igm_factor`` scales the forest only."""
        return jnp.exp(-(self.tau_madau(lam_rest, zred) * factor
                         + self.tau_damp(lam_rest, zred, params)
                         + self.tau_dla(lam_rest, zred, params)))


_MODEL_REGISTRY = {
    "madau1995": Madau1995,
    "madau1995_damping_dla": MadauDampingDLA,
    "none":      NoIGM,
}


def make_igm_model(name_or_model):
    """Return an :class:`IGMModel` from a registry name, an instance (as-is), or ``None`` (NoIGM)."""
    if name_or_model is None:
        return NoIGM()
    if isinstance(name_or_model, IGMModel):
        return name_or_model
    name = str(name_or_model).lower().strip()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown IGM model {name!r}.  "
            f"Available: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name]()


__all__ = ["IGMModel", "NoIGM", "Madau1995", "MadauDampingDLA", "make_igm_model",
           "tau_damping", "tau_gp", "voigt_tau"]
