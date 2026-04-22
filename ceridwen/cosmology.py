r"""
ceridwen/cosmology.py
=====================
Redshift helpers for ceridwen, with two back-ends:

1. **JAX native (default, always available)** — flat LambdaCDM, matter
   only, Planck-2018-like parameters, Simpson integrator on a fixed
   128-node grid.  JIT-compilable, differentiable in z — **required
   whenever zred is a sampled free parameter** (NUTS needs gradients).

2. **astropy (optional, auto-detected)** — uses
   ``astropy.cosmology.Planck18.luminosity_distance`` and therefore
   includes radiation + massive neutrinos, closing the ~0.5-1% gap
   between the native integrator and published Planck18 tables.
   astropy's code is NumPy-based, not JAX — so it can only be used when
   ``z`` is a concrete Python scalar (not a traced value), typically
   for the one-off precompute step at ``SedModel(csp, obs, priors,
   zred=0.5)``.  Set ``backend='astropy'`` in the top-level helpers to
   opt in explicitly.

References
----------
Hogg, "Distance measures in cosmology" (astro-ph/9905116).
Planck Collaboration 2020, A&A 641, A6 (Planck 2018 VI).
astropy Cosmology docs: https://docs.astropy.org/en/stable/cosmology/

Conventions
-----------
- distances in megaparsec (Mpc); luminosity distance ``D_L(z)`` in Mpc.
- Native back-end integrates :math:`\int_0^z dz'/E(z')` with Simpson's
  rule on a fixed ``n_nodes`` grid — fully differentiable, constant XLA
  graph cost independent of ``z``.

Use
---
>>> from ceridwen.cosmology import luminosity_distance_mpc
>>> luminosity_distance_mpc(0.5)                   # JAX, any z
>>> luminosity_distance_mpc(0.5, backend='astropy') # Planck18 exact, scalar z only
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array

# Astropy-like constants
c_km_s = 299792.458              # speed of light [km/s]
MPC_TO_CM = 3.0856775814913673e24  # 1 Mpc in cm
PC_TO_CM = 3.0856775814913673e18   # 1 pc in cm

# Constants used in the radiation / neutrino density:
#   Omega_gamma0 * h^2 = a_B T0^4 c / (3 H0^2_ref * 100^2) = 2.473e-5 (T0/2.7255)^4
#   Omega_nu,mat * h^2 = sum(m_nu)/93.14 eV  (Mangano et al. 2005)
#   Omega_nu,rel,0 = (7/8)(4/11)^(4/3) * N_eff,relativistic * Omega_gamma0
# These match astropy.cosmology.FlatLambdaCDM exactly in the limit that
# the massive neutrino is treated as cold matter (a good approximation
# for m_nu = 0.06 eV at z <= 10, as needed for SED fitting).
_OGAMMA_H2_REF = 2.473e-5        # Omega_gamma h^2 at Tcmb = 2.7255 K
_NU_MASS_CONST_H2 = 93.14        # eV; sum(m_nu)/93.14/h^2 -> Omega_nu (matter)
_NU_REL_FACTOR = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)  # 0.22710...


# Planck 2018 VI TT,TE,EE+lowE+lensing (astropy.cosmology.Planck18 defaults).
@dataclass(frozen=True)
class Cosmology:
    """Flat LambdaCDM cosmology with photons + effective neutrinos.

    This is a deliberately-minimal JAX-friendly port of astropy's
    ``FlatLambdaCDM``.  The massive-neutrino contribution is folded into
    ``Om0`` as cold matter (the late-time approximation; valid for
    m_nu <= 1 eV at z <= 10 to <0.1% on D_L), so we do NOT need the
    full relativistic transition that astropy evaluates point-by-point.

    All properties are Python scalars and constant-fold into the XLA
    graph when this object is passed as a closure to a jitted function.
    """
    # Directly match astropy.cosmology.Planck18 defaults.
    H0: float = 67.66            # km/s/Mpc
    Om0: float = 0.30966         # cold matter today (NO neutrinos here)
    Tcmb0: float = 2.7255        # K; Fixsen 2009
    Neff: float = 3.046          # effective number of neutrino species
    m_nu_ev_sum: float = 0.06    # sum of neutrino masses [eV]

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def hubble_distance_mpc(self) -> float:
        return c_km_s / self.H0

    @property
    def Ogamma0(self) -> float:
        """Photon density today :math:`\\Omega_\\gamma`."""
        return (_OGAMMA_H2_REF
                * (self.Tcmb0 / 2.7255) ** 4
                / (self.h ** 2))

    @property
    def Onu0_massive_as_matter(self) -> float:
        """Massive-neutrino density today treated as cold matter
        (Mangano et al. 2005 relation :math:`\\Omega_\\nu h^2 = \\sum m_\\nu/93.14\\,\\mathrm{eV}`)."""
        return self.m_nu_ev_sum / (_NU_MASS_CONST_H2 * self.h ** 2)

    @property
    def Onu0_relativistic(self) -> float:
        """Relativistic (massless-equivalent) neutrino density today.

        For Planck18 with one 0.06 eV neutrino, Neff = 3.046 counts the
        total at CMB epoch but one species becomes non-relativistic at
        late times.  Subtract one for the late-time relativistic count.
        """
        n_rel = max(0.0, self.Neff - 1.0)
        return _NU_REL_FACTOR * n_rel * self.Ogamma0

    @property
    def Om0_eff(self) -> float:
        """Effective matter density today (CDM + baryons + massive nu)."""
        return self.Om0 + self.Onu0_massive_as_matter

    @property
    def Or0(self) -> float:
        """Radiation + massless-neutrino density today."""
        return self.Ogamma0 + self.Onu0_relativistic

    @property
    def Ode0(self) -> float:
        """Dark-energy density today (flat: 1 - Om - Or)."""
        return 1.0 - self.Om0_eff - self.Or0


DEFAULT_COSMO = Cosmology()


def E_of_z(z: Array, cosmo: Cosmology = DEFAULT_COSMO) -> Array:
    r"""Dimensionless expansion rate

    .. math::
        E(z) = \sqrt{\Omega_m^\mathrm{eff}(1+z)^3
                   + \Omega_r (1+z)^4
                   + \Omega_\Lambda}.

    Includes radiation (photons + massless neutrinos) exactly; the
    single massive neutrino is folded into :math:`\Omega_m^\mathrm{eff}`
    as cold matter.  This matches
    :class:`astropy.cosmology.FlatLambdaCDM` with the same cosmological
    parameters to < 0.1% on :math:`D_L(z)` for :math:`z \le 10`.

    The formula is identical to astropy's ``efunc`` implementation
    modulo the massive-neutrino approximation; structurally

    .. math::
        E(z)^2 = (1+z)^3\bigl[\Omega_r(1+z) + \Omega_m^\mathrm{eff}\bigr]
                 + \Omega_\Lambda,

    which is the form astropy uses to factor out one ``(1+z)^3`` for
    numerical stability at high z.
    """
    opz = 1.0 + z
    return jnp.sqrt(
        opz * opz * opz * (cosmo.Or0 * opz + cosmo.Om0_eff)
        + cosmo.Ode0
    )


def _integrate_dz_over_E(z: Array, cosmo: Cosmology,
                         n_nodes: int = 128) -> Array:
    r"""Simpson's-rule integral :math:`\int_0^z dz'/E(z')` on a fixed grid.

    ``n_nodes`` must be odd for Simpson; we pad to the next odd.
    """
    if n_nodes % 2 == 0:
        n_nodes += 1

    # Uniform grid from 0 to z.  z may be a scalar, so we build the grid
    # relative to z; this keeps the traced shape static.
    t = jnp.linspace(0.0, 1.0, n_nodes)       # shape (n_nodes,)
    zp = z * t                                 # shape (..., n_nodes) via broadcast
    y = 1.0 / E_of_z(zp, cosmo)

    # Simpson weights: 1, 4, 2, 4, 2, ..., 4, 1 scaled by h/3 where h = z / (n_nodes - 1).
    # Build once at trace time.
    w = jnp.ones(n_nodes)
    w = w.at[1::2].set(4.0)
    w = w.at[2:-1:2].set(2.0)
    h = z / (n_nodes - 1)
    return (h / 3.0) * jnp.sum(w * y, axis=-1)


def comoving_distance_mpc(z: Array, cosmo: Cosmology = DEFAULT_COSMO,
                          n_nodes: int = 128) -> Array:
    r"""Comoving (line-of-sight) distance
    :math:`D_C(z) = D_H \int_0^z dz'/E(z')`.
    """
    return cosmo.hubble_distance_mpc * _integrate_dz_over_E(z, cosmo, n_nodes)


def _astropy_luminosity_distance_mpc(z) -> float:
    """Use astropy.cosmology.Planck18 for the luminosity distance.

    This is a *scalar-only* fallback (astropy returns a Quantity, not a
    JAX array) used when the user opts in with ``backend='astropy'``.
    Not importable into a traced function.
    """
    from astropy.cosmology import Planck18
    # astropy accepts numpy floats; coerce JAX scalars to Python floats so
    # there is no traced path through this function.
    z_val = float(z)
    return float(Planck18.luminosity_distance(z_val).to("Mpc").value)


def luminosity_distance_mpc(z, cosmo: Cosmology = DEFAULT_COSMO,
                            n_nodes: int = 128,
                            backend: str = "native") -> Array:
    r"""Luminosity distance :math:`D_L(z) = (1+z) D_C(z)` in Mpc.

    Parameters
    ----------
    z : Array or float
        Redshift.  JAX arrays are supported by the native backend;
        scalars only for the astropy backend.
    cosmo : Cosmology
        Only used by the native backend (astropy uses Planck18 internally).
    n_nodes : int
        Simpson-integrator node count (native only).  128 is accurate to
        ~1e-6 for the integral itself; the residual error vs astropy is
        dominated by missing radiation + neutrinos, not the integrator.
    backend : {'native', 'astropy'}
        ``'native'`` (default) uses the JAX Simpson integrator and is
        differentiable; ``'astropy'`` calls
        :func:`astropy.cosmology.Planck18.luminosity_distance` and is
        limited to concrete scalar z but is "ground-truth" accurate.
    """
    if backend == "astropy":
        return _astropy_luminosity_distance_mpc(z)
    if backend != "native":
        raise ValueError(f"Unknown cosmology backend {backend!r}")
    return (1.0 + z) * comoving_distance_mpc(z, cosmo, n_nodes)


def flux_factor(z: Array, cosmo: Cosmology = DEFAULT_COSMO,
                n_nodes: int = 128) -> Array:
    r"""Cosmological flux factor to turn rest-frame per-Hz luminosity into
    observed-frame flux density, including the :math:`(1+z)` term for
    :math:`F_\nu`.

    Specifically, for a rest-frame luminosity per unit frequency
    :math:`L_\nu^\mathrm{rest}` in erg s^{-1} Hz^{-1} one has

    .. math::
        F_\nu^\mathrm{obs}(\nu_\mathrm{obs}) =
        \frac{(1+z)\, L_\nu^\mathrm{rest}(\nu_\mathrm{rest})}
             {4 \pi\, D_L(z)^2}.

    This helper returns :math:`(1+z) / (4\pi D_L^2)` in CGS-compatible units
    such that multiplying a spectrum in erg s^{-1} Hz^{-1} yields the
    observed-frame :math:`F_\nu` in erg s^{-1} cm^{-2} Hz^{-1}.

    For the ceridwen "maggies"-style convention where the rest-frame
    spectrum is in L_sun Hz^{-1} M_sun^{-1} (as produced by FSPS / SSP
    tables), this factor should be composed with any unit conversion
    the forward model already does.  See
    :func:`flux_factor_maggies` for the maggies-ready version.
    """
    dL_mpc = luminosity_distance_mpc(z, cosmo, n_nodes)
    dL_cm = dL_mpc * MPC_TO_CM
    return (1.0 + z) / (4.0 * jnp.pi * dL_cm * dL_cm)


# Convenience wrappers that match the units used in ceridwen's CSP (F_nu
# in maggies): the CSP spectrum is in L_sun Hz^{-1} M_sun^{-1}, and
# distance-less maggies are computed at a fiducial ``d = 10 pc``.  The
# correction from d = 10 pc to d = D_L(z) is ``(10 pc / D_L)^2 * (1+z)``.

_MAGGIES_D_FID_PC = 10.0  # absolute-magnitude convention, 10 pc


def flux_factor_maggies(z, cosmo: Cosmology = DEFAULT_COSMO,
                        n_nodes: int = 128,
                        backend: str = "native") -> Array:
    r"""Redshift scaling relative to the CSP's fiducial ``d = 10 pc``.

    Multiplies a CSPBasis-computed (rest-frame, per-source) spectrum to
    produce the observed-frame spectrum at the luminosity distance for
    redshift ``z``.  At ``z = 0`` the CSP normalisation already corresponds
    to 10 pc, so :math:`\mathrm{ff}(0) = 1`.  For ``z > 0``:

    .. math::
        \mathrm{ff}(z) = \frac{(1 + z)\, D_L(z=0)^2}{D_L(z)^2}
                      = \frac{(1 + z)\,(10\,\mathrm{pc})^2}{D_L(z)^2}.

    This assumes the observation flux unit is maggies relative to the
    same absolute magnitude convention.

    Parameters
    ----------
    z : Array or float
        Redshift.
    backend : {'native', 'astropy'}
        See :func:`luminosity_distance_mpc`.  Use ``'astropy'`` for a
        one-off scalar at model construction time (e.g. inside
        ``SedModel.__init__`` when ``zred`` is a fixed float); use
        ``'native'`` inside the forward-evaluation loop so the call is
        JIT-compilable and differentiable.  ``SedModel`` automatically
        applies the astropy backend when a ``float`` ``zred`` is
        provided and the package is installed.
    """
    dL_pc = 1e6 * luminosity_distance_mpc(z, cosmo, n_nodes, backend=backend)
    # Avoid 0/0 at z = 0: when z is exactly zero, dL is zero by integration;
    # fall through to the algebraic limit ff(0) = 1.
    safe_dL = jnp.where(dL_pc > 0, dL_pc, 1.0)
    ff_nonzero = (1.0 + z) * (_MAGGIES_D_FID_PC / safe_dL) ** 2
    return jnp.where(dL_pc > 0, ff_nonzero, 1.0)


def have_astropy() -> bool:
    """True if astropy is importable.  Used by SedModel to decide whether
    to use the higher-accuracy Planck18 backend for fixed-z precompute."""
    try:
        import astropy.cosmology  # noqa: F401
        return True
    except Exception:
        return False


__all__ = [
    "Cosmology", "DEFAULT_COSMO", "c_km_s", "MPC_TO_CM", "PC_TO_CM",
    "E_of_z", "comoving_distance_mpc", "luminosity_distance_mpc",
    "flux_factor", "flux_factor_maggies", "have_astropy",
]
