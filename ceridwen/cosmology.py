r"""Flat LambdaCDM redshift helpers: :class:`Cosmology` presets, D_L(z), age(z)
and the maggies flux factor, JAX-native (differentiable in z) with an optional
scalar-only astropy backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

Array = jax.Array

c_km_s = 299792.458
MPC_TO_CM = 3.0856775814913673e24
PC_TO_CM = 3.0856775814913673e18

_OGAMMA_H2_REF = 2.473e-5        # Omega_gamma h^2 at Tcmb = 2.7255 K
_NU_MASS_CONST_H2 = 93.14        # eV; Omega_nu h^2 = sum(m_nu)/93.14
_NU_REL_FACTOR = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)


_PRESETS = {
    "Planck18": dict(H0=67.66, Om0=0.30966, Tcmb0=2.7255, Neff=3.046, m_nu_ev_sum=0.06),
    "Planck15": dict(H0=67.74, Om0=0.3075,  Tcmb0=2.7255, Neff=3.046, m_nu_ev_sum=0.06),
    "WMAP9":    dict(H0=69.32, Om0=0.2865,  Tcmb0=2.725,  Neff=3.04,  m_nu_ev_sum=0.0),
}
_FIELDS = ("H0", "Om0", "Tcmb0", "Neff", "m_nu_ev_sum")


@dataclass(frozen=True)
class Cosmology:
    """Flat LambdaCDM cosmology with photons and effective neutrinos; the
    massive neutrino is folded into Om0 as cold matter.  Defaults to Planck 2018.

    Parameters
    ----------
    H0 : float, km/s/Mpc
    Om0 : float -- cold matter today, EXCLUDING massive neutrinos.
    Tcmb0 : float, K
    Neff : float -- effective number of neutrino species.
    m_nu_ev_sum : float, eV -- sum of neutrino masses.
    name : str -- label only; ignored in equality.
    """
    H0: float = 67.66
    Om0: float = 0.30966
    Tcmb0: float = 2.7255
    Neff: float = 3.046
    m_nu_ev_sum: float = 0.06
    name: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        for k in _FIELDS:
            v = float(getattr(self, k))
            if v != v or v < 0.0 or (k in ("H0", "Om0") and v == 0.0):
                raise ValueError(
                    f"Cosmology.{k} must be a finite non-negative number"
                    f"{' > 0' if k in ('H0', 'Om0') else ''}, got {v}")
            object.__setattr__(self, k, v)
        if self.Om0 > 1.0:
            raise ValueError(
                f"Cosmology.Om0 = {self.Om0} > 1 leaves no room for Omega_Lambda "
                "in a flat model")
        if not (10.0 <= self.H0 <= 1000.0):
            raise ValueError(
                f"Cosmology.H0 = {self.H0} km/s/Mpc is outside [10, 1000]; the "
                "argument order is (H0, Om0), e.g. Cosmology.flat(70.0, 0.3)")
        if self.name is not None:
            object.__setattr__(self, "name", str(self.name))

    @classmethod
    def planck18(cls) -> "Cosmology":
        return cls(**_PRESETS["Planck18"], name="Planck18")

    @classmethod
    def planck15(cls) -> "Cosmology":
        return cls(**_PRESETS["Planck15"], name="Planck15")

    @classmethod
    def wmap9(cls) -> "Cosmology":
        """WMAP9 preset."""
        return cls(**_PRESETS["WMAP9"], name="WMAP9")

    @classmethod
    def flat(cls, H0: float, Om0: float, *, Tcmb0: float = 2.7255,
             Neff: float = 3.046, m_nu_ev_sum: float = 0.06,
             name: Optional[str] = None) -> "Cosmology":
        """Flat LCDM from H0 and Om0; radiation and neutrinos default to Planck 2018."""
        return cls(H0=H0, Om0=Om0, Tcmb0=Tcmb0, Neff=Neff,
                   m_nu_ev_sum=m_nu_ev_sum, name=name)

    @classmethod
    def from_name(cls, name: str) -> "Cosmology":
        """Preset by name, case-insensitive (see ``available_cosmologies()``)."""
        key = {k.lower(): k for k in _PRESETS}.get(str(name).strip().lower())
        if key is None:
            raise KeyError(
                f"unknown cosmology preset {name!r}; known: {', '.join(_PRESETS)}")
        return cls(**_PRESETS[key], name=key)

    @classmethod
    def from_dict(cls, d) -> "Cosmology":
        """Inverse of :meth:`to_dict`; also accepts ``cosmo_*``-prefixed HDF5 attrs."""
        d = dict(d)
        if "H0" not in d and "cosmo_H0" in d:
            d = {k[len("cosmo_"):]: v for k, v in d.items() if k.startswith("cosmo_")}
        missing = [k for k in _FIELDS if k not in d]
        if missing:
            raise KeyError(f"Cosmology.from_dict: missing {missing} (have {sorted(d)})")
        name = d.get("name", None)
        if isinstance(name, bytes):
            name = name.decode()
        if name is not None:
            name = str(name).strip() or None
        return cls(**{k: float(d[k]) for k in _FIELDS}, name=name)

    def to_dict(self) -> dict:
        """Plain floats plus ``name`` ("" when unnamed)."""
        d = {k: float(getattr(self, k)) for k in _FIELDS}
        d["name"] = self.name or ""
        return d

    def describe(self) -> str:
        tag = f"{self.name}: " if self.name else ""
        return (f"{tag}flat LCDM, H0 = {self.H0:g} km/s/Mpc, Om0 = {self.Om0:g}, "
                f"Tcmb0 = {self.Tcmb0:g} K, Neff = {self.Neff:g}, "
                f"sum m_nu = {self.m_nu_ev_sum:g} eV")

    def __str__(self) -> str:
        return self.describe()

    def age(self, z) -> Array:
        """Age of the Universe at ``z`` [Gyr]."""
        return age_gyr(z, self)

    def luminosity_distance(self, z) -> Array:
        """D_L(z) [Mpc]."""
        return luminosity_distance_mpc(z, self)

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def hubble_distance_mpc(self) -> float:
        return c_km_s / self.H0

    @property
    def Ogamma0(self) -> float:
        """Photon density today."""
        return (_OGAMMA_H2_REF
                * (self.Tcmb0 / 2.7255) ** 4
                / (self.h ** 2))

    @property
    def Onu0_massive_as_matter(self) -> float:
        """Massive-neutrino density today, treated as cold matter."""
        return self.m_nu_ev_sum / (_NU_MASS_CONST_H2 * self.h ** 2)

    @property
    def Onu0_relativistic(self) -> float:
        """Relativistic neutrino density today (Neff - 1 species)."""
        n_rel = max(0.0, self.Neff - 1.0)
        return _NU_REL_FACTOR * n_rel * self.Ogamma0

    @property
    def Om0_eff(self) -> float:
        """Matter density today including massive neutrinos."""
        return self.Om0 + self.Onu0_massive_as_matter

    @property
    def Or0(self) -> float:
        """Radiation + relativistic-neutrino density today."""
        return self.Ogamma0 + self.Onu0_relativistic

    @property
    def Ode0(self) -> float:
        """Dark-energy density today (flat: 1 - Om0_eff - Or0)."""
        return 1.0 - self.Om0_eff - self.Or0

    @classmethod
    def from_astropy(cls, cosmo) -> "Cosmology":
        """Build from a flat astropy cosmology; a non-flat input raises."""
        Ok0 = float(getattr(cosmo, "Ok0", 0.0))
        if abs(Ok0) > 1e-8:
            raise ValueError(
                f"ceridwen's cosmology integrator assumes flatness, but the "
                f"supplied cosmology has Ok0 = {Ok0:.3e}."
            )
        m_nu = getattr(cosmo, "m_nu", None)
        if m_nu is None:
            m_nu_sum = 0.0
        else:
            try:
                m_nu_sum = float(m_nu.to("eV").value.sum())
            except AttributeError:
                m_nu_sum = float(m_nu.to("eV").value)
        return cls(
            H0=float(cosmo.H0.to("km/(s Mpc)").value),
            Om0=float(cosmo.Om0),
            Tcmb0=float(cosmo.Tcmb0.to("K").value),
            Neff=float(cosmo.Neff),
            m_nu_ev_sum=m_nu_sum,
            name=(str(cosmo.name) if getattr(cosmo, "name", None) else None),
        )

    def to_astropy(self):
        """Return the matching ``astropy.cosmology.FlatLambdaCDM`` (requires astropy)."""
        import math

        import astropy.units as u
        from astropy.cosmology import FlatLambdaCDM

        kwargs = dict(
            H0=self.H0 * u.km / u.s / u.Mpc,
            Om0=self.Om0,
            Tcmb0=self.Tcmb0 * u.K,
            Neff=self.Neff,
        )
        # astropy needs len(m_nu) == floor(Neff); total mass on the last species
        n_species = math.floor(self.Neff)
        if n_species > 0 and self.Tcmb0 > 0.0:
            masses = [0.0] * n_species
            masses[-1] = self.m_nu_ev_sum
            kwargs["m_nu"] = u.Quantity(masses, u.eV)
        return FlatLambdaCDM(**kwargs)

    @property
    def is_planck18(self) -> bool:
        """True when the parameters equal the Planck-2018 preset (name ignored)."""
        return self == Cosmology.planck18()


def available_cosmologies() -> tuple:
    """Names accepted by :meth:`Cosmology.from_name`."""
    return tuple(_PRESETS)


DEFAULT_COSMO = Cosmology.planck18()


def resolve_cosmology(cosmo: "Cosmology | None" = None) -> Cosmology:
    """Return ``cosmo``, or Planck 2018 when ``None``."""
    return DEFAULT_COSMO if cosmo is None else cosmo


def E_of_z(z: Array, cosmo: Cosmology | None = None) -> Array:
    r"""Dimensionless expansion rate E(z) = H(z)/H0; ``z`` is clamped to >= 0
    so transient negative proposals cannot produce NaN."""
    cosmo = resolve_cosmology(cosmo)
    z = jnp.maximum(z, 0.0)
    opz = 1.0 + z
    return jnp.sqrt(
        opz * opz * opz * (cosmo.Or0 * opz + cosmo.Om0_eff)
        + cosmo.Ode0
    )


def _integrate_dz_over_E(z: Array, cosmo: Cosmology,
                         n_nodes: int = 128) -> Array:
    r"""Simpson integral of 1/E from 0 to z on a fixed grid (n_nodes padded to odd)."""
    cosmo = resolve_cosmology(cosmo)
    if n_nodes % 2 == 0:
        n_nodes += 1

    t = jnp.linspace(0.0, 1.0, n_nodes)
    zp = z * t                                 # (..., n_nodes)
    y = 1.0 / E_of_z(zp, cosmo)

    w = jnp.ones(n_nodes)
    w = w.at[1::2].set(4.0)
    w = w.at[2:-1:2].set(2.0)
    h = z / (n_nodes - 1)
    return (h / 3.0) * jnp.sum(w * y, axis=-1)


def comoving_distance_mpc(z: Array, cosmo: Cosmology | None = None,
                          n_nodes: int = 128) -> Array:
    r"""Line-of-sight comoving distance D_C(z) [Mpc]."""
    cosmo = resolve_cosmology(cosmo)
    return cosmo.hubble_distance_mpc * _integrate_dz_over_E(z, cosmo, n_nodes)


_INV_H0_GYR_NUM = 977.7922216807891       # 1/H0 in Gyr = this / H0[km/s/Mpc]


def age_gyr(z: Array, cosmo: Cosmology | None = None,
            n_nodes: int = 257) -> Array:
    r"""Age of the Universe at ``z`` [Gyr], JAX-native and differentiable in z."""
    cosmo = resolve_cosmology(cosmo)
    if n_nodes % 2 == 0:
        n_nodes += 1
    z = jnp.asarray(z, dtype=float)
    a = 1.0 / (1.0 + jnp.maximum(z, 0.0))
    u = jnp.linspace(0.0, 1.0, n_nodes)
    x = a[..., None] * u                           # (..., n_nodes): a' from 0..a
    pos = x > 0.0
    x_safe = jnp.where(pos, x, 1.0)
    zp = 1.0 / x_safe - 1.0
    g = jnp.where(pos, 1.0 / (x_safe * E_of_z(zp, cosmo)), 0.0)
    w = jnp.ones(n_nodes).at[1::2].set(4.0).at[2:-1:2].set(2.0)
    h = a / (n_nodes - 1)
    integral = (h / 3.0) * jnp.sum(w * g, axis=-1)
    return (_INV_H0_GYR_NUM / cosmo.H0) * integral


def _astropy_luminosity_distance_mpc(z, cosmo: Cosmology | None = None) -> float:
    """Scalar-only D_L(z) [Mpc] via astropy; not usable under a JAX trace."""
    cosmo = resolve_cosmology(cosmo)
    z_val = float(z)
    if cosmo.is_planck18:
        from astropy.cosmology import Planck18
        ap = Planck18
    else:
        ap = cosmo.to_astropy()
    return float(ap.luminosity_distance(z_val).to("Mpc").value)


def luminosity_distance_mpc(z, cosmo: Cosmology | None = None,
                            n_nodes: int = 128,
                            backend: str = "native") -> Array:
    r"""Luminosity distance D_L(z) [Mpc].

    Parameters
    ----------
    backend : {'native', 'astropy'} -- 'native' is JAX and differentiable;
        'astropy' is scalar-z only.
    """
    cosmo = resolve_cosmology(cosmo)
    if backend == "astropy":
        return _astropy_luminosity_distance_mpc(z, cosmo)
    if backend != "native":
        raise ValueError(f"Unknown cosmology backend {backend!r}")
    return (1.0 + z) * comoving_distance_mpc(z, cosmo, n_nodes)


def flux_factor(z: Array, cosmo: Cosmology | None = None,
                n_nodes: int = 128) -> Array:
    r"""(1+z) / (4 pi D_L^2) with D_L in cm: converts rest-frame L_nu
    [erg/s/Hz] to observed F_nu [erg/s/cm^2/Hz]."""
    cosmo = resolve_cosmology(cosmo)
    dL_mpc = luminosity_distance_mpc(z, cosmo, n_nodes)
    dL_cm = dL_mpc * MPC_TO_CM
    return (1.0 + z) / (4.0 * jnp.pi * dL_cm * dL_cm)


_D_FID_10PC = 10.0

_LSUN_HZ_TO_FNU_CGS_AT_10PC = 3.1967965e-7   # erg s^-1 cm^-2 Hz^-1 per L_sun Hz^-1 at 10 pc


def flux_factor_cgs(z, cosmo: Cosmology | None = None,
                    n_nodes: int = 128,
                    backend: str = "native",
                    lumdist_mpc=None) -> Array:
    r"""Factor turning a CSP spectrum [L_sun/Hz/M_sun] into observed-frame
    F_nu [erg/s/cm^2/Hz] (cgs): (1+z) (10 pc / D_L)^2 times the 10 pc unit
    constant. z <= 0 is pinned to D_L = 10 pc.  The AB-maggies photometry is
    obtained downstream by projecting this cgs spectrum through the filters
    (dividing by the 3631 Jy AB zero point); this factor itself is NOT maggies.

    Parameters
    ----------
    backend : {'native', 'astropy'} -- see :func:`luminosity_distance_mpc`.
    lumdist_mpc : float or Array, Mpc -- explicit D_L replacing D_L(z); the
        (1+z) term still comes from z and the z <= 0 pin does not apply.
    """
    cosmo = resolve_cosmology(cosmo)
    if lumdist_mpc is not None:
        dL_pc = 1e6 * jnp.asarray(lumdist_mpc, dtype=float)
        return ((1.0 + z) * (_D_FID_10PC / dL_pc) ** 2
                * _LSUN_HZ_TO_FNU_CGS_AT_10PC)
    dL_pc = 1e6 * luminosity_distance_mpc(z, cosmo, n_nodes, backend=backend)
    positive_z = z > 0
    safe_dL = jnp.where(positive_z, dL_pc, float(_D_FID_10PC))
    ff_distance = jnp.where(
        positive_z,
        (1.0 + z) * (_D_FID_10PC / safe_dL) ** 2,
        1.0,
    )
    return ff_distance * _LSUN_HZ_TO_FNU_CGS_AT_10PC


flux_factor_maggies = flux_factor_cgs   # backwards-compatible alias (old jades_full predictor)


def have_astropy() -> bool:
    """True if astropy is importable."""
    try:
        import astropy.cosmology  # noqa: F401
        return True
    except Exception:
        return False


__all__ = [
    "Cosmology", "DEFAULT_COSMO", "resolve_cosmology", "available_cosmologies",
    "c_km_s", "MPC_TO_CM", "PC_TO_CM",
    "E_of_z", "comoving_distance_mpc", "luminosity_distance_mpc",
    "age_gyr", "flux_factor", "flux_factor_cgs", "have_astropy",
]
