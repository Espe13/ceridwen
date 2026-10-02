"""AGN components for the CERIDWEN forward model.

This file is the interface every AGN component is written against (AGN_PLAN §3).
It imports no component module at import time, so a checkout that has only some of
``torus.py`` / ``disk.py`` / ``screen.py`` / ``lines.py`` still imports cleanly.
Changes after the first commit: ``agn_mode`` defaults to ``"corrected"`` (session INT, after
the parity suite AGN_PLAN §4 A passed; see CHANGELOG.md and docs/agn.md).

Public API
----------
``CSPBasis(..., agn=AGNConfig())`` switches the AGN on in the default ``"corrected"`` mode
(``agn=True`` is short for it, ``agn="fsps"`` for ``AGNConfig(agn_mode="fsps")``). Without
``agn=`` (the default) every existing path is unchanged, byte for byte.

Modes (static, validated when ``AGNConfig`` is built):

``"fsps"``
    Reproduces FSPS (``add_agn_dust=1``; Prospector ``TemplateLibrary["agn"]``):
    Nenkova+08 torus resampled exactly as ``sps_setup.f90`` does (tiny30 log-log,
    ``locate`` edges, no renormalisation), attenuated by the diffuse ``dust2`` screen
    only, added AFTER the IGM, no energy balance. Only the torus exists in FSPS, so
    ``disk`` and ``dust4_type`` are refused; ``lines`` (Prospector ``AGNSpecModel``,
    which runs on top of FSPS) may be switched on explicitly.
``"prospector"``
    Reproduces Prospector's ``AGNSSPBasis`` path: the ``fake_fsps.agn_torus`` torus
    (argmin edges, log-log, linear EXTRApolation in ``agn_tau``, not attenuated),
    the Temple+21 disc, the ``dust4`` screen, AGN lines, IGM applied to the AGN as
    ``agnssp_basis.get_spectrum`` does. ``agn_tau`` defaults to 50 there.
``"corrected"``
    Prospector's components with the defects fixed (23 Sep design D1-D6): torus
    renormalised to exactly ``fagn * L_bol`` on the model grid, positive-rows-only
    resampling, ``agn_tau`` clamped to [5, 150], AGN added inside the CSP before the
    IGM, energy balance into DL07 when dust emission is on.

theta keys (registered in ``CSPBasis._known_theta_keys``; defaults from
``AGNModel.default_params()``):

    fagn             0      L_AGN,torus / L_bol,*  (L_bol,* = dust-free stellar integral)
    agn_tau          10     torus optical depth (50 in "prospector" mode when not given)
    fagn_bbb         0      F_nu,disc / F_nu,* at rest 5500 A (no-dust young+old galaxy)
    dust4            0      AGN-only screen optical depth at 5500 A (dust4_type=1)
    dust4_index      -1.0   AGN-only screen power-law index
    agn_elum         0      L_Hbeta,AGN [L_sun] per M_formed
    agn_eline_sigma  100    AGN line width [km/s]

Units: every ``Component.rest_spectrum`` returns L_sun/Hz per M_formed in the rest
frame, on ``ctx.wave``, BEFORE any attenuation. ``hook.py`` applies the screens, sums
the components, handles energy balance and the IGM order per mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax

__all__ = [
    "AGN_MODE_FSPS",
    "AGN_MODE_PROSPECTOR",
    "AGN_MODE_CORRECTED",
    "AGN_MODES",
    "TAU_INTERP_MODES",
    "DUST4_TYPES",
    "AGN_THETA_DEFAULTS",
    "AGN_TAU_DEFAULT_PROSPECTOR",
    "AGNConfig",
    "AGNContext",
    "Component",
    "Screen",
    "LineComponent",
    "AGNModel",
]

AGN_MODE_FSPS = "fsps"
AGN_MODE_PROSPECTOR = "prospector"
AGN_MODE_CORRECTED = "corrected"
AGN_MODES = (AGN_MODE_FSPS, AGN_MODE_PROSPECTOR, AGN_MODE_CORRECTED)

TAU_INTERP_MODES = ("linear", "log")
# 0: no AGN screen; 1: dust2 (diffuse) + dust4 power law; 2: Prospector NotImplemented.
DUST4_TYPES = (0, 1)

# Defaults of every AGN theta key (AGN_PLAN §2). "prospector" mode overrides agn_tau.
AGN_THETA_DEFAULTS = {
    "fagn": 0.0,
    "agn_tau": 10.0,
    "fagn_bbb": 0.0,
    "dust4": 0.0,
    "dust4_index": -1.0,
    "agn_elum": 0.0,
    "agn_eline_sigma": 100.0,
}
AGN_TAU_DEFAULT_PROSPECTOR = 50.0  # prospect/sources/agnssp_basis.py: params.get("agn_tau", 50)


@dataclass(frozen=True)
class AGNConfig:
    """Static AGN configuration, validated on construction.

    Parameters
    ----------
    agn_mode : {"corrected", "fsps", "prospector"}, default "corrected"
        "corrected" is the physically consistent model; "fsps" and "prospector" reproduce
        those codes (defects included) for comparison with existing fits.
    torus, disk, lines : bool or None
        Which components exist. ``None`` = the mode's default: ``"fsps"`` -> torus
        only; ``"prospector"`` / ``"corrected"`` -> all three. Their shapes are static,
        so a component switched off costs nothing; one switched on with its amplitude
        at 0 (``fagn``, ``fagn_bbb``, ``agn_elum``) adds exactly zero.
    dust4_type : {0, 1}
        AGN screen: 0 none, 1 diffuse dust2 curve times a ``dust4`` power law.
        2 is refused (Prospector raises NotImplementedError for it).
    tau_interp : {"linear", "log"}
        Interpolation of the torus templates in ``agn_tau`` ("linear" = FSPS).
        Only ``"corrected"`` mode may use ``"log"``.
    agn_energy_balance : bool or None
        Re-emit the AGN light absorbed by the galaxy screen through DL07.
        ``None`` = ON in ``"corrected"`` mode when ``add_dust_emission=True``,
        OFF otherwise. ``True`` is refused in the reproduction modes (FSPS and
        Prospector have no AGN energy balance) and, at CSP setup, without dust
        emission.
    torus_file : str or None
        Nenkova+08 template file; default
        ``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat``.
    """

    agn_mode: str = AGN_MODE_CORRECTED
    torus: Optional[bool] = None
    disk: Optional[bool] = None
    lines: Optional[bool] = None
    dust4_type: int = 0
    tau_interp: str = "linear"
    agn_energy_balance: Optional[bool] = None
    torus_file: Optional[str] = None

    def __post_init__(self):
        if self.agn_mode not in AGN_MODES:
            raise ValueError(
                f"AGNConfig(agn_mode={self.agn_mode!r}): choose one of {AGN_MODES}. "
                "'fsps' reproduces FSPS add_agn_dust, 'prospector' reproduces "
                "Prospector's AGNSSPBasis, 'corrected' is the physically consistent model.")
        if self.dust4_type == 2:
            raise NotImplementedError(
                "AGNConfig(dust4_type=2): Prospector raises NotImplementedError for it "
                "(prospect/sources/fake_fsps.py) and so does CERIDWEN. Use 0 (no AGN "
                "screen) or 1 (dust2 curve times a dust4 power law).")
        if self.dust4_type not in DUST4_TYPES:
            raise ValueError(f"AGNConfig(dust4_type={self.dust4_type!r}): choose 0 or 1.")
        if self.tau_interp not in TAU_INTERP_MODES:
            raise ValueError(
                f"AGNConfig(tau_interp={self.tau_interp!r}): choose one of {TAU_INTERP_MODES}.")
        if self.tau_interp != "linear" and self.agn_mode != AGN_MODE_CORRECTED:
            raise ValueError(
                f"AGNConfig(tau_interp={self.tau_interp!r}) with agn_mode={self.agn_mode!r}: "
                "the reproduction modes interpolate linearly in agn_tau as FSPS/Prospector "
                "do; 'log' is only available in agn_mode='corrected'.")
        if self.agn_energy_balance and self.agn_mode != AGN_MODE_CORRECTED:
            raise ValueError(
                f"AGNConfig(agn_energy_balance=True) with agn_mode={self.agn_mode!r}: "
                "FSPS and Prospector do not re-emit absorbed AGN light; energy balance "
                "exists only in agn_mode='corrected'.")
        if self.agn_mode == AGN_MODE_FSPS:
            if self.disk:
                raise ValueError(
                    "AGNConfig(agn_mode='fsps', disk=True): FSPS has no accretion disc; "
                    "the Temple+21 disc is Prospector's AGNSSPBasis. Use "
                    "agn_mode='prospector' or 'corrected'.")
            if self.dust4_type != 0:
                raise ValueError(
                    "AGNConfig(agn_mode='fsps', dust4_type=1): FSPS has no dust4 screen; "
                    "use agn_mode='prospector' or 'corrected'.")
        for name in ("torus", "disk", "lines", "agn_energy_balance"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, bool):
                raise TypeError(f"AGNConfig({name}={v!r}): expected True, False or None.")

    @property
    def use_torus(self) -> bool:
        return True if self.torus is None else self.torus

    @property
    def use_disk(self) -> bool:
        if self.disk is None:
            return self.agn_mode != AGN_MODE_FSPS
        return self.disk

    @property
    def use_lines(self) -> bool:
        if self.lines is None:
            return self.agn_mode != AGN_MODE_FSPS
        return self.lines

    def energy_balance(self, add_dust_emission: bool) -> bool:
        """Resolved energy-balance switch for a CSP with/without dust emission."""
        if self.agn_energy_balance is None:
            return self.agn_mode == AGN_MODE_CORRECTED and bool(add_dust_emission)
        if self.agn_energy_balance and not add_dust_emission:
            raise ValueError(
                "AGNConfig(agn_energy_balance=True) needs CSPBasis(add_dust_emission=True): "
                "the absorbed AGN light is re-emitted through DL07. Without dust emission "
                "it is lost, exactly as absorbed starlight is.")
        return self.agn_energy_balance

    def default_agn_tau(self) -> float:
        return AGN_TAU_DEFAULT_PROSPECTOR if self.agn_mode == AGN_MODE_PROSPECTOR \
            else AGN_THETA_DEFAULTS["agn_tau"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AGNContext:
    """Per-call quantities ``hook.py`` builds once and hands to every component.

    wave : (n_wave,) rest-frame wavelength [A], static.
    L_bol_star : scalar, dust-free stellar bolometric luminosity [L_sun per M_formed]
        (grid integral with the DL07 trapezoid-in-frequency weights).
    gal_nodust_young_old : (n_wave,) dust-free young+old stellar spectrum
        [L_sun/Hz per M_formed], for the disc's 5500 A normalisation.
    diffuse_curve : (n_wave,) exp(-tau_diffuse(lambda)) of the galaxy screen.
    mode : one of ``AGN_MODES`` (static, pytree aux data).
    """

    wave: jax.Array
    L_bol_star: jax.Array
    gal_nodust_young_old: jax.Array
    diffuse_curve: jax.Array
    mode: str

    def tree_flatten(self):
        return ((self.wave, self.L_bol_star, self.gal_nodust_young_old,
                 self.diffuse_curve), (self.mode,))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


class Component:
    """One AGN continuum component (torus, disc). Pure JAX in ``rest_spectrum``;
    everything static (templates, resampling, masks) is built in ``__init__`` in
    float64 numpy, and validation raises there, never inside a kernel."""

    param_names: tuple = ()

    def default_params(self) -> dict:
        """Defaults of the theta keys in ``param_names``."""
        raise NotImplementedError

    def rest_spectrum(self, theta, ctx: AGNContext):
        """(n_wave,) L_sun/Hz per M_formed, rest frame, on ``ctx.wave``, BEFORE any
        attenuation."""
        raise NotImplementedError


class Screen:
    """The AGN-only screen (``dust4``). ``hook.py`` multiplies every AGN continuum
    component by ``transmission(theta, ctx)``."""

    param_names: tuple = ()

    def default_params(self) -> dict:
        raise NotImplementedError

    def transmission(self, theta, ctx: AGNContext):
        """(n_wave,) total transmission applied to the AGN continuum, INCLUDING the
        galaxy's diffuse curve ``ctx.diffuse_curve`` (prospect ``attenuate`` combines
        dust2 and dust4 for the AGN)."""
        raise NotImplementedError


class LineComponent:
    """AGN narrow emission lines: a static line set, painted by the CSP."""

    param_names: tuple = ()
    line_names: tuple = ()

    def default_params(self) -> dict:
        raise NotImplementedError

    def line_luminosity(self, theta):
        """(n_lines,) L_sun per M_formed, in the order of ``line_names``."""
        raise NotImplementedError

    def sigma_kms(self, theta):
        """Gaussian line width [km/s] (scalar)."""
        raise NotImplementedError


class AGNModel:
    """The composite the CSP holds: the components ``config`` switches on, built on
    the model grid ``wave`` (rest A). Component modules are imported here, lazily.

    Constructor signatures the component modules implement:

        torus.NenkovaTorus(wave, *, mode, tau_interp, torus_file, sps_home)  -> Component
        disk.TempleDisk(wave, *, mode)                                     -> Component
        screen.AGNScreen(wave, *, dust4_type)                              -> Screen
        lines.AGNLines(*, mode, sps_home)                                  -> LineComponent
    """

    def __init__(self, config: AGNConfig, wave, *, sps_home=None):
        if not isinstance(config, AGNConfig):
            raise TypeError(f"agn= expects an AGNConfig, got {type(config).__name__}.")
        self.config = config
        self.mode = config.agn_mode
        self.wave = wave
        self.components = {}
        self.lines = None
        if config.use_torus:
            from .torus import NenkovaTorus
            self.components["torus"] = NenkovaTorus(
                wave, mode=config.agn_mode, tau_interp=config.tau_interp,
                torus_file=config.torus_file, sps_home=sps_home)
        if config.use_disk:
            from .disk import TempleDisk
            self.components["disk"] = TempleDisk(wave, mode=config.agn_mode)
        if config.dust4_type != 0:
            from .screen import AGNScreen
            self.screen = AGNScreen(wave, dust4_type=config.dust4_type)
        else:
            self.screen = None
        if config.use_lines:
            from .lines import AGNLines
            self.lines = AGNLines(mode=config.agn_mode, sps_home=sps_home)

    @property
    def param_names(self) -> tuple:
        names = []
        parts = list(self.components.values())
        parts += [p for p in (self.screen, self.lines) if p is not None]
        for p in parts:
            names += [n for n in p.param_names if n not in names]
        return tuple(names)

    def default_params(self) -> dict:
        out = {}
        parts = list(self.components.values())
        parts += [p for p in (self.screen, self.lines) if p is not None]
        for p in parts:
            out.update(p.default_params())
        return out
