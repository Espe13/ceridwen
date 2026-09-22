"""IGM damping wing and damped Ly-alpha absorber (DLA) as a CERIDWEN ``IGMModel`` subclass.

What it does
------------
``MadauDampingDLA`` is Madau (1995) IGM absorption (CERIDWEN's own ``Madau1995``, unchanged)
times two optional, FIXED-parameter extra absorbers:

* an IGM **damping wing** redward of rest-frame Ly-alpha from a uniformly neutral IGM between
  ``zmin`` and the source redshift, with a fixed neutral fraction ``x_HI``;
* a **DLA**: a Voigt Ly-alpha profile of fixed column density ``logN_HI`` [log10 cm^-2] at a
  fixed absorber redshift ``z_dla`` (``None`` = the source redshift, Prospector's default).

    tau_total(lam_rest, z) = tau_Madau(lam_rest, z) + tau_damp(lam_rest, z; x_HI) + tau_DLA

    attenuation(lam_rest, z, factor) = exp(-(factor * tau_Madau + tau_damp + tau_DLA))

Use it through the existing extension point, no package change::

    from ceridwen import CSPBasis
    from examples.recipes.igm_damping_dla import MadauDampingDLA
    igm = MadauDampingDLA(cosmo=cosmo, Ob0=0.04897, x_HI=0.5, logN_HI=21.0)
    csp = CSPBasis(ssp, ..., cosmo=cosmo, add_igm=True, igm_model=igm)

Public API relied on
--------------------
* ``ceridwen.igm.IGMModel`` (``ceridwen/igm.py:16``): abstract ``tau(lam_rest, zred)``
  (``:21-23``) and ``attenuation(lam_rest, zred, factor)`` (``:25-28``); overridden here.
* ``ceridwen.igm.Madau1995.tau`` (``ceridwen/igm.py:64-88``), reused unchanged via ``super()``.
* ``ceridwen.igm.make_igm_model`` returns an ``IGMModel`` instance as-is (``ceridwen/igm.py:101-102``);
  ``CSPBasis(add_igm=True, igm_model=instance)`` stores it (``ceridwen/csp/csp.py:302-307``).
* The CSP calls ``self.igm.attenuation(self.wave, z_scalar, factor=ig_factor)`` on the REST-frame
  grid with ``z_scalar = theta["zred"]`` (a traced value) at ``ceridwen/csp/csp.py:857``, ``:976``,
  ``:1105`` and ``:1132``; ``ig_factor`` is ``theta["igm_factor"]`` or the constructor
  ``igm_factor``.  IGM is only applied when ``"zred"`` is in theta (``csp.py:847``).
* ``ceridwen.cosmology.Cosmology`` ``.h`` (``ceridwen/cosmology.py:141``) and ``.Om0`` (field,
  ``:46``).  CERIDWEN's ``Cosmology`` has no baryon density, so ``Ob0`` is a required argument.

Prospector equivalent (prospector @ a78d153, ``prospect/models/sedmodel.py``)
-----------------------------------------------------------------------------
* ``SpecModel.add_dla`` ``:808-819`` -> ``voigt_profile`` ``:1201-1245`` with ``H`` ``:1192-1198``.
* ``SpecModel.add_damping_wing`` ``:821-827`` -> ``tau_damping`` ``:1261-1309``,
  ``tau_gp`` ``:1312-1327``, ``Ix`` ``:1330-1333``.
* Both are applied to the rest-frame smoothed spectrum after LOSVD smoothing (``:213-215``).

Literature (author/year as cited in those Prospector docstrings/comments; the journal details
were added from memory and are not verified against ADS here)
----------------------------------------------------------------------------------------------
* Voigt approximation: Tepper-Garcia (2006, MNRAS 369, 2025; 2007, MNRAS 382, 1375) (``sedmodel.py:1193``).
* Voigt optical-depth normalisation: Krogager (2018, VoigtFit, arXiv:1803.01187) (``:1205``).
* Damping wing of a uniform neutral IGM: Miralda-Escude (1998, ApJ 501, 15) and Totani et al.
  (2006, PASJ 58, 485) (``:1262-1264``), Gunn-Peterson depth normalisation from the same
  (``:1313-1318``).
* Madau (1995, ApJ 441, 18) for the forest, as in ``ceridwen/igm.py:40``.

Deliberate differences from Prospector
--------------------------------------
1. **``igm_factor`` is NOT ``x_HI``.**  Prospector reads the neutral fraction from
   ``params["igm_factor"]`` (``sedmodel.py:824``), the same parameter that scales the FSPS
   Madau forest optical depth, so one number sets both.  Here ``x_HI`` is its own constructor
   argument and CERIDWEN's ``igm_factor`` scales ``tau_Madau`` only.
2. **DLA wavelength for ``z_dla != zred``.**  Prospector maps the source rest frame to the
   absorber frame with ``wave_rest * (1 + dla_z) / (1 + zred)`` (``sedmodel.py:816``); the
   observed wavelength is ``wave_rest (1+zred)``, so the absorber-frame wavelength is
   ``wave_rest (1 + zred) / (1 + z_dla)``, which is what is used here.  The two agree exactly for
   ``z_dla = zred`` (Prospector's default, ``:814``) and the check compares only that case; for a
   foreground absorber Prospector puts the trough redward of Ly-alpha (at
   ``1215.67 (1+zred)/(1+z_dla)`` rest) instead of blueward.
3. JAX: the boolean indexing of ``tau_damping`` and the ``x = 0`` singularity of ``H`` (``Q =
   1.5 / x**2``) are replaced by masked ``jnp.where`` with safe arguments, so values, ``jit`` and
   gradients are finite.  ``|x| < 1e-3`` (|lambda - l0| < 1.6e-4 Å) is evaluated at ``|x| = 1e-3``;
   there ``tau`` = 3.8e6 already at ``logN_HI = 20.3`` (the DLA threshold; measured by the check
   script), so the transmission is 0 either way.  Prospector returns
   NaN at an exact ``x = 0`` pixel.

Out of scope: sampling ``x_HI`` / ``logN_HI`` / ``z_dla``
---------------------------------------------------------
Needs a package change (after the rebuild).  ``IGMModel.attenuation`` receives only
``(lam_rest, zred, factor)``, so a model cannot see other theta keys.  The change:
(a) give ``IGMModel.attenuation`` / ``tau`` a ``params`` mapping (default ``{}``) and pass the
theta sub-dict of the model's declared keys at the four call sites ``csp.py:857, 976, 1105,
1132`` (and the ``CSPBasis_afe`` equivalents it inherits); (b) let an ``IGMModel`` declare
``param_names`` (e.g. ``igm_x_HI``, ``dla_logNh``, ``dla_zred``) and add them to
``_known_theta_keys`` (``csp.py:505-509``) so ``SedModel`` samples them and the unknown-key
warning stays quiet; (c) serialise the IGM model name and fixed args in ``fit.py`` result
metadata; (d) a new golden configuration (CLAUDE.md section 4).  The kernels below already take
traced ``x_HI`` / ``N_HI`` / ``z_dla`` and are differentiable in them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ceridwen.igm import Madau1995  # noqa: E402

# ---- constants copied from prospect/models/sedmodel.py @ a78d153 --------------------------
_C_CMS = 2.99792e10            # :1233
_VOIGT_CONST = 0.0149736082    # :1234, sqrt(pi) e^2 / (m_e c) [cgs]
_LYA = 1215.6696               # :1201, :1261
_F_LYA = 4.16e-1               # :1201
_GAMMA_LYA = 6.265e8           # :1201
_R_ALPHA = 2.02e-8             # :1297
_X_SAFE = 1e-3


def _H(a, x):
    """Tepper-Garcia (2006) Voigt-Hjerting approximation (``sedmodel.py:1192-1198``), with the
    ``x -> 0`` singularity moved to ``|x| = 1e-3`` (value and gradient finite)."""
    small = jnp.abs(x) < _X_SAFE
    x = jnp.where(small, jnp.where(x < 0, -_X_SAFE, _X_SAFE), x)
    P = x ** 2
    H0 = jnp.exp(-x ** 2)
    Q = 1.5 / x ** 2
    return H0 - a / jnp.sqrt(jnp.pi) / P * (H0 * H0 * (4. * P * P + 7. * P + 4. + Q) - Q - 1)


def voigt_tau(wave_abs, N, bkms=40.0, l0=_LYA, f=_F_LYA, gamma=_GAMMA_LYA):
    """Optical depth of a Voigt line on ``wave_abs`` [Å, absorber rest frame] for column ``N``
    [cm^-2] (``sedmodel.py:1201-1245``)."""
    l0_cm = l0 * 1.e-8
    b = bkms * 1e5
    C_a = _VOIGT_CONST * f * l0_cm / b
    a = l0_cm * gamma / (4. * jnp.pi * b)
    x = (_C_CMS / b) * (1. - l0 / wave_abs)
    return C_a * N * _H(a, x)


def _Ix(x):
    """``sedmodel.py:1330-1333``."""
    v = x ** (9 / 2) / (1 - x) + 9 / 7 * x ** 3.5 + 9 / 5 * x ** 2.5 + 3 * x ** 1.5 + 9 * x ** 0.5
    return v - 9 / 2 * jnp.log((1 + x ** 0.5) / (1 - x ** 0.5))


def tau_gp(zred, h, Om0, Ob0, Y=0.25):
    """Gunn-Peterson depth, Miralda-Escude (1998) normalisation (``sedmodel.py:1312-1327``)."""
    scale = (1 - Y) * h * Om0 ** (-0.5) * (Ob0 / 0.03)
    return 2.6e5 * scale * ((1 + zred) / 7) ** (3 / 2)


def tau_damping(wave_rest, zred, x_HI, h, Om0, Ob0, zmin=5.0, Y=0.25, l0=_LYA):
    """Damping-wing optical depth of a uniform IGM between ``zmin`` and ``zred``
    (``sedmodel.py:1261-1309``), zero blueward of Ly-alpha and for ``zred <= zmin``
    (the ``zred > zmin`` switch of ``add_damping_wing``, ``:823``)."""
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
    """Madau (1995) x IGM damping wing (fixed ``x_HI``) x optional DLA (fixed ``logN_HI``, ``z_dla``).

    Parameters
    ----------
    cosmo : ceridwen Cosmology -- supplies ``h`` and ``Om0`` for the Gunn-Peterson depth.
    Ob0 : float -- baryon density today (not carried by CERIDWEN's Cosmology; required when
        ``x_HI != 0``).  Planck18: 0.04897; WMAP9 (astropy, Prospector's default): 0.04628.
    x_HI : float -- IGM neutral fraction (may exceed 1 to mimic a local overdensity).  0 = off.
    zmin : float -- lower redshift of the neutral IGM; the wing is off for ``zred <= zmin``.
    Y : float -- helium mass fraction.
    logN_HI : float or None -- DLA column [log10 cm^-2]; ``None`` = no DLA.
    z_dla : float or None -- absorber redshift; ``None`` = the source redshift.  No DLA when
        ``z_dla > zred``.
    b_kms : float -- Doppler parameter of the DLA [km/s].
    """

    name = "madau1995_damping_dla"

    def __init__(self, cosmo=None, Ob0=None, x_HI=0.0, zmin=5.0, Y=0.25,
                 logN_HI=None, z_dla=None, b_kms=40.0):
        x_HI = float(x_HI)
        if not x_HI >= 0.0:
            raise ValueError(f"x_HI must be >= 0, got {x_HI}")
        if x_HI > 0.0 and (cosmo is None or Ob0 is None):
            raise ValueError("x_HI > 0 needs cosmo= (h, Om0) and Ob0= (baryon density); "
                             "CERIDWEN's Cosmology has no Ob0, so it is not guessed")
        self.x_HI = x_HI
        self.zmin = float(zmin)
        self.Y = float(Y)
        self.h = float(cosmo.h) if cosmo is not None else 0.0
        self.Om0 = float(cosmo.Om0) if cosmo is not None else 1.0
        self.Ob0 = float(Ob0) if Ob0 is not None else 0.0
        self.N_HI = None if logN_HI is None else 10.0 ** float(logN_HI)
        self.z_dla = None if z_dla is None else float(z_dla)
        self.b_kms = float(b_kms)

    # --- components -----------------------------------------------------------------------
    def tau_madau(self, lam_rest, zred):
        return super().tau(lam_rest, zred)

    def tau_damp(self, lam_rest, zred):
        if self.x_HI == 0.0:
            return jnp.zeros_like(lam_rest)
        return tau_damping(lam_rest, zred, self.x_HI, self.h, self.Om0, self.Ob0,
                           zmin=self.zmin, Y=self.Y)

    def tau_dla(self, lam_rest, zred):
        if self.N_HI is None:
            return jnp.zeros_like(lam_rest)
        z_abs = zred if self.z_dla is None else self.z_dla
        wave_abs = lam_rest * (1 + zred) / (1 + z_abs)
        tau = voigt_tau(wave_abs, self.N_HI, bkms=self.b_kms)
        return jnp.where(z_abs > zred, 0.0, tau)

    # --- IGMModel interface ----------------------------------------------------------------
    def tau(self, lam_rest, zred):
        return (self.tau_madau(lam_rest, zred) + self.tau_damp(lam_rest, zred)
                + self.tau_dla(lam_rest, zred))

    def attenuation(self, lam_rest, zred, factor=1.0):
        """``exp(-(factor * tau_Madau + tau_damp + tau_DLA))``: ``igm_factor`` scales the forest only."""
        return jnp.exp(-(self.tau_madau(lam_rest, zred) * factor
                         + self.tau_damp(lam_rest, zred) + self.tau_dla(lam_rest, zred)))


__all__ = ["MadauDampingDLA", "tau_damping", "tau_gp", "voigt_tau"]
