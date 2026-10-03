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
``"cigale"`` (AGN2)
    Reproduces CIGALE's conventions for the AGN2 components (SKIRTOR torus + its disc,
    polar dust, Feltre+16 NLR, BLR): the AGN is added after CIGALE's galaxy dust modules, so
    it sees no galaxy dust and has no energy balance; it is added before the IGM
    (CIGALE's ``redshifting`` module acts on every contribution). The Nenkova+08 torus, the
    Temple+21 disc, the ``dust4`` screen and the a42 line ratios are refused (CIGALE has none
    of them). The exact CIGALE normalisations live in the component modules (PLAN_AGN2.md).

AGN2 components (``AGNConfig`` fields; all off by default, so ``AGNConfig()`` is unchanged):

    torus_model="skirtor16"   skirtor.SkirtorTorus     (replaces Nenkova+08; carries its own
                                                        disc, so the Temple disc is refused)
    polar_dust=True           polar.PolarDust          (extinction of the SKIRTOR disc +
                                                        grey-body re-emission)
    nlr_model="feltre16"      nlr.FeltreNLR            (replaces the a42 ratios)
    blr=True                  blr.BroadLines           (own amplitude blr_lum, own width
                                                        blr_sigma)

They are refused in ``"fsps"`` / ``"prospector"`` modes. The modules are imported lazily
through ``AGN2_REGISTRY``; a module that does not exist yet raises NotImplementedError when
the CSP is built (never at import). The constructor signatures and the methods ``hook.py``
calls are the contract below (``SkirtorTorus``, ``PolarDust``, ``AGN2LineSet``); tests swap in
toy classes through ``AGN2_REGISTRY``.

Attenuation of the AGN2 parts (``hook.py``; T_pol = ``PolarDust.transmission``, T_gal = the
galaxy's diffuse curve, T_d4 = the dust4 power law):

    ==================  =====================================  ======================
    part                "corrected"                            "cigale"
    ==================  =====================================  ======================
    SKIRTOR disc        T_pol x T_gal                          T_pol
    SKIRTOR dust        T_gal                                  1
    polar re-emission   T_gal                                  1
    NLR lines           T_gal at the line                      1
    BLR lines           T_pol (or T_d4) x T_gal at the line    T_pol at the line
    ==================  =====================================  ======================

Energy balance ("corrected" with dust emission): what T_gal removes goes to DL07; what T_pol
removes is re-emitted by ``PolarDust.reemission`` and never reaches DL07 (it is not galaxy dust).
The "cigale" column is the scaffold's reading of CIGALE's module order and is confirmed or
corrected against PLAN_AGN2.md by the validation suite.

theta keys (registered in ``CSPBasis._known_theta_keys``; defaults from
``AGNModel.default_params()``):

    fagn             0      L_AGN,torus / L_bol,*  (L_bol,* = dust-free stellar integral)
    agn_tau          10     torus optical depth (50 in "prospector" mode when not given)
    fagn_bbb         0      F_nu,disc / F_nu,* at rest 5500 A (no-dust young+old galaxy)
    dust4            0      AGN-only screen optical depth at 5500 A (dust4_type=1)
    dust4_index      -1.0   AGN-only screen power-law index
    agn_elum         0      L_Hbeta,AGN [L_sun] per M_formed
    agn_eline_sigma  100    AGN line width [km/s]

AGN2 theta keys (``AGN2_THETA_KEYS``; defaults come from each component's ``default_params``,
written by its session from PLAN_AGN2.md):

    skirtor_t, skirtor_pl, skirtor_q, skirtor_oa, skirtor_i      SKIRTOR (fagn: amplitude)
    polar_ebv, polar_temp, polar_emis                            polar dust
    nlr_logu, nlr_logz, nlr_alpha, nlr_xid, nlr_lognh            Feltre+16 NLR (agn_elum:
                                                                 amplitude, agn_eline_sigma)
    blr_lum, blr_sigma                                           BLR

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
    "AGN_MODE_CIGALE",
    "AGN_MODES",
    "IN_CSP_MODES",
    "TORUS_MODELS",
    "NLR_MODELS",
    "POLAR_LAWS",
    "POLAR_LAW_DEFAULT",
    "AGN2_THETA_KEYS",
    "AGN2_REGISTRY",
    "TAU_INTERP_MODES",
    "DUST4_TYPES",
    "AGN_THETA_DEFAULTS",
    "AGN_TAU_DEFAULT_PROSPECTOR",
    "AGNConfig",
    "AGNContext",
    "Component",
    "Screen",
    "LineComponent",
    "AGN2LineSet",
    "SkirtorTorus",
    "PolarDust",
    "AGNModel",
]

AGN_MODE_FSPS = "fsps"
AGN_MODE_PROSPECTOR = "prospector"
AGN_MODE_CORRECTED = "corrected"
AGN_MODE_CIGALE = "cigale"
AGN_MODES = (AGN_MODE_FSPS, AGN_MODE_PROSPECTOR, AGN_MODE_CORRECTED, AGN_MODE_CIGALE)
# the modes that add the AGN inside the CSP, before the IGM (the reproduction modes do not)
IN_CSP_MODES = (AGN_MODE_CORRECTED, AGN_MODE_CIGALE)

TORUS_MODELS = ("nenkova08", "skirtor16")
NLR_MODELS = ("a42", "feltre16")
POLAR_LAWS = ("smc", "calzetti", "gaskell")
# CIGALE skirtor2016 ``law`` default 0 = SMC (UNVERIFIED in the scaffold: confirmed against the
# pcigale source by PLAN_AGN2.md; changing it here changes only polar_law=None)
POLAR_LAW_DEFAULT = "smc"

# The theta keys of each AGN2 component (AGN2_BRIEF "Config and theta names"). A component
# class must declare exactly these in ``param_names`` (tests/agn2 checks it); the defaults are
# the component's own ``default_params()``.
AGN2_THETA_KEYS = {
    "skirtor16": ("fagn", "skirtor_t", "skirtor_pl", "skirtor_q", "skirtor_oa", "skirtor_i"),
    "polar": ("polar_ebv", "polar_temp", "polar_emis"),
    "feltre16": ("agn_elum", "agn_eline_sigma", "nlr_logu", "nlr_logz", "nlr_alpha",
                 "nlr_xid", "nlr_lognh"),
    "blr": ("blr_lum", "blr_sigma"),
}

# name -> (module, class) of the AGN2 components, imported lazily by AGNModel. Tests replace an
# entry with a class object (``monkeypatch.setitem(AGN2_REGISTRY, "skirtor16", Toy)``).
AGN2_REGISTRY = {
    "skirtor16": ("ceridwen.agn.skirtor", "SkirtorTorus"),
    "polar": ("ceridwen.agn.polar", "PolarDust"),
    "feltre16": ("ceridwen.agn.nlr", "FeltreNLR"),
    "blr": ("ceridwen.agn.blr", "BroadLines"),
}

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
    torus_model : {"nenkova08", "skirtor16"}, default "nenkova08"
        Which torus ``torus`` switches on. "skirtor16" (Stalevski+2012, 2016, CIGALE's
        ``skirtor2016``) carries its own anisotropic disc, so the Temple disc is off by
        default and refused when asked for (``disk=True``).
    polar_dust : bool, default False
        CIGALE's polar dust (extinction of the SKIRTOR disc and its grey-body re-emission).
        Needs ``torus_model="skirtor16"``.
    polar_law : {"smc", "calzetti", "gaskell"} or None
        Extinction law of the polar dust; None = CIGALE's default (``POLAR_LAW_DEFAULT``).
        Only with ``polar_dust=True``.
    nlr_model : {"a42", "feltre16"}, default "a42"
        The narrow-line set ``lines`` switches on: Richardson+14 fixed ratios or the
        Feltre+16 photoionisation grid.
    blr : bool, default False
        Broad lines with their own amplitude (``blr_lum``) and width (``blr_sigma``).
    skirtor_R : float or None
        Static ratio of outer to inner torus radius (a node of the SKIRTOR grid; None = the
        component's default). Only with ``torus_model="skirtor16"``.
    skirtor_file, nlr_file, blr_file : str or None
        Data files of the AGN2 components (default: under ``$CERIDWEN_AGN_DATA``, resolved by
        each component). Each only with its component switched on.

    Every combination that would silently ignore a setting is refused here, with the reason.
    """

    agn_mode: str = AGN_MODE_CORRECTED
    torus: Optional[bool] = None
    disk: Optional[bool] = None
    lines: Optional[bool] = None
    dust4_type: int = 0
    tau_interp: str = "linear"
    agn_energy_balance: Optional[bool] = None
    torus_file: Optional[str] = None
    torus_model: str = "nenkova08"
    polar_dust: bool = False
    polar_law: Optional[str] = None
    nlr_model: str = "a42"
    blr: bool = False
    skirtor_R: Optional[float] = None
    skirtor_file: Optional[str] = None
    nlr_file: Optional[str] = None
    blr_file: Optional[str] = None

    def __post_init__(self):
        if self.agn_mode not in AGN_MODES:
            raise ValueError(
                f"AGNConfig(agn_mode={self.agn_mode!r}): choose one of {AGN_MODES}. "
                "'fsps' reproduces FSPS add_agn_dust, 'prospector' reproduces "
                "Prospector's AGNSSPBasis, 'cigale' reproduces CIGALE's AGN (SKIRTOR, polar "
                "dust, AGN lines), 'corrected' is the physically consistent model.")
        self._validate_agn2()
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
                "FSPS and Prospector do not re-emit absorbed AGN light, and CIGALE adds its "
                "AGN after the galaxy's dust (nothing to re-emit); energy balance exists only "
                "in agn_mode='corrected'.")
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

    def _validate_agn2(self):
        """The AGN2 fields (AGN2_BRIEF "Modes"): refused combinations, with the reason."""
        for name in ("polar_dust", "blr"):
            v = getattr(self, name)
            if not isinstance(v, bool):
                raise TypeError(f"AGNConfig({name}={v!r}): expected True or False.")
        if self.torus_model not in TORUS_MODELS:
            raise ValueError(f"AGNConfig(torus_model={self.torus_model!r}): choose one of "
                             f"{TORUS_MODELS}.")
        if self.nlr_model not in NLR_MODELS:
            raise ValueError(f"AGNConfig(nlr_model={self.nlr_model!r}): choose one of "
                             f"{NLR_MODELS}.")
        if self.polar_law is not None and self.polar_law not in POLAR_LAWS:
            raise ValueError(f"AGNConfig(polar_law={self.polar_law!r}): choose one of "
                             f"{POLAR_LAWS} (None = CIGALE's default, {POLAR_LAW_DEFAULT!r}).")
        mode = self.agn_mode
        skirtor = self.torus_model == "skirtor16"
        feltre = self.nlr_model == "feltre16"
        new = [n for n, on in (("torus_model='skirtor16'", skirtor),
                               ("polar_dust=True", self.polar_dust),
                               ("nlr_model='feltre16'", feltre), ("blr=True", self.blr)) if on]
        if new and mode in (AGN_MODE_FSPS, AGN_MODE_PROSPECTOR):
            raise ValueError(
                f"AGNConfig(agn_mode={mode!r}, {', '.join(new)}): the AGN2 components (SKIRTOR, "
                f"polar dust, Feltre+16 NLR, BLR) do not exist in {mode.upper()} and its mode "
                "reproduces that code exactly; use agn_mode='corrected' (CERIDWEN's framework) "
                "or agn_mode='cigale' (CIGALE's conventions).")
        if self.torus is False and skirtor:
            raise ValueError(
                "AGNConfig(torus=False, torus_model='skirtor16'): the SKIRTOR torus is switched "
                "off, so torus_model would be ignored; drop one of the two.")
        if not skirtor:
            for name in ("skirtor_R", "skirtor_file"):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"AGNConfig({name}=...) needs torus_model='skirtor16' (got "
                        f"{self.torus_model!r}); it would be ignored.")
        if skirtor:
            if self.skirtor_R is not None:
                r = self.skirtor_R
                if isinstance(r, bool) or not isinstance(r, (int, float)) or not r > 1:
                    raise ValueError(
                        f"AGNConfig(skirtor_R={r!r}): the ratio of outer to inner torus radius "
                        "is a number > 1 (one of the SKIRTOR grid values).")
            if self.disk:
                raise ValueError(
                    "AGNConfig(torus_model='skirtor16', disk=True): SKIRTOR carries its own "
                    "anisotropic accretion disc; adding the Temple+21 disc would count the disc "
                    "twice. Leave disk unset (it is off with SKIRTOR).")
            if self.torus_file is not None:
                raise ValueError(
                    "AGNConfig(torus_model='skirtor16', torus_file=...): torus_file is the "
                    "Nenkova+08 template file; the SKIRTOR file is skirtor_file.")
            if self.tau_interp != "linear":
                raise ValueError(
                    f"AGNConfig(torus_model='skirtor16', tau_interp={self.tau_interp!r}): "
                    "tau_interp is the Nenkova+08 interpolation in agn_tau; SKIRTOR is "
                    "interpolated on its own grid axes.")
            if self.dust4_type != 0:
                raise ValueError(
                    "AGNConfig(torus_model='skirtor16', dust4_type=1): the dust4 screen acts on "
                    "the Temple disc, which SKIRTOR replaces; the nuclear screen of SKIRTOR is "
                    "the polar dust (polar_dust=True).")
        if self.polar_dust and not skirtor:
            raise ValueError(
                "AGNConfig(polar_dust=True) needs torus_model='skirtor16': CIGALE's polar dust "
                "extinguishes the SKIRTOR disc and re-emits what it absorbs.")
        if self.polar_law is not None and not self.polar_dust:
            raise ValueError(f"AGNConfig(polar_law={self.polar_law!r}) needs polar_dust=True; "
                             "it would be ignored.")
        if self.lines is False and feltre:
            raise ValueError(
                "AGNConfig(lines=False, nlr_model='feltre16'): the narrow lines are switched "
                "off, so nlr_model would be ignored; drop one of the two.")
        if self.nlr_file is not None and not feltre:
            raise ValueError("AGNConfig(nlr_file=...) needs nlr_model='feltre16'; it would be "
                             "ignored.")
        if self.blr_file is not None and not self.blr:
            raise ValueError("AGNConfig(blr_file=...) needs blr=True; it would be ignored.")
        if mode == AGN_MODE_CIGALE:
            if self.torus is not False and not skirtor:
                raise ValueError(
                    "AGNConfig(agn_mode='cigale') with the Nenkova+08 torus: CIGALE has no "
                    "Nenkova torus; use torus_model='skirtor16' (or torus=False).")
            if self.disk:
                raise ValueError(
                    "AGNConfig(agn_mode='cigale', disk=True): CIGALE has no Temple+21 disc "
                    "(SKIRTOR carries its own).")
            if self.dust4_type != 0:
                raise ValueError(
                    "AGNConfig(agn_mode='cigale', dust4_type=1): CIGALE has no dust4 screen; "
                    "its nuclear screen is the polar dust (polar_dust=True).")
            if self.lines and not feltre:
                raise ValueError(
                    "AGNConfig(agn_mode='cigale', lines=True) with nlr_model='a42': CIGALE has "
                    "no Richardson+14 ratios; use nlr_model='feltre16'.")

    @property
    def use_torus(self) -> bool:
        return True if self.torus is None else self.torus

    @property
    def use_skirtor(self) -> bool:
        return self.use_torus and self.torus_model == "skirtor16"

    @property
    def use_nenkova(self) -> bool:
        return self.use_torus and self.torus_model == "nenkova08"

    @property
    def use_disk(self) -> bool:
        if self.disk is None:
            return (self.agn_mode not in (AGN_MODE_FSPS, AGN_MODE_CIGALE)
                    and self.torus_model != "skirtor16")
        return self.disk

    @property
    def use_lines(self) -> bool:
        """The narrow lines (``nlr_model``) are on."""
        if self.lines is None:
            if self.agn_mode == AGN_MODE_CIGALE:
                return self.nlr_model == "feltre16"
            return self.agn_mode != AGN_MODE_FSPS
        return self.lines

    @property
    def use_any_lines(self) -> bool:
        """Any AGN line set (narrow or broad) is on."""
        return self.use_lines or self.blr

    @property
    def polar_law_resolved(self) -> str:
        return POLAR_LAW_DEFAULT if self.polar_law is None else self.polar_law

    @property
    def uses_agn2(self) -> bool:
        """Any AGN2 component is on (the hook then takes its AGN2 branches)."""
        return (self.use_skirtor or self.polar_dust or self.blr
                or (self.use_lines and self.nlr_model == "feltre16"))

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
    L_dust_gal : scalar or None, the luminosity the galaxy's own dust absorbs from the stars
        and the nebular emission [L_sun per M_formed] (what DL07 re-emits; the energy-balance
        integral of the ``*_dem_*`` kernels). Built only when a component sets
        ``needs_ldust_gal = True`` (CIGALE's ``fracAGN`` is relative to the dust luminosity),
        else None.
    """

    wave: jax.Array
    L_bol_star: jax.Array
    gal_nodust_young_old: jax.Array
    diffuse_curve: jax.Array
    mode: str
    L_dust_gal: Optional[jax.Array] = None

    def tree_flatten(self):
        return ((self.wave, self.L_bol_star, self.gal_nodust_young_old,
                 self.diffuse_curve, self.L_dust_gal), (self.mode,))

    @classmethod
    def tree_unflatten(cls, aux, children):
        wave, lbol, gal, dc, ldust = children
        return cls(wave, lbol, gal, dc, aux[0], ldust)


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


class AGN2LineSet(LineComponent):
    """Contract of an AGN2 line set (``nlr.FeltreNLR``, ``blr.BroadLines``), on top of
    ``LineComponent``. The CSP paints every line set separately, with its own width and its
    own transmission, into Spectrum, Photometry and Lines observations (a Lines observation
    sums every set and the nebular lines BY FSPS NAME).

    Class attributes
        region : "nlr" or "blr" -- selects the transmission (see the module docstring).
        width_key : the theta key of the Gaussian width (``"agn_eline_sigma"`` for the NLR,
            ``"blr_sigma"`` for the BLR); ``sigma_kms(theta)`` returns it.

    Attributes (set in ``__init__``, static NumPy)
        line_names : tuple of FSPS line names (``$SPS_HOME/data/emlines_info.dat``).
        line_wave : (n_lines,) float64 vacuum rest wavelengths [A], from the FSPS list.
        emlines_wave, emlines_names, emlines_file : the whole FSPS list (``rows_for``).

    Methods
        line_luminosity(theta) -> (n_lines,) L_sun per M_formed, BEFORE any attenuation.
        sigma_kms(theta) -> scalar km/s.
        default_params() -> dict of every key in ``param_names``.
        rows_for(wave_rest, tol=1.0) -> (n,) int64: provided here (match BY NAME).
        provenance() -> dict (optional): files and hashes for the result file.
        range_messages(theta) -> list of str (optional): setup-time range warnings.
    """

    region: str = "nlr"
    width_key: str = "agn_eline_sigma"

    @property
    def n_lines(self) -> int:
        return len(self.line_names)

    def rows_for(self, wave_rest, tol=1.0):
        """For each rest wavelength: the index of the line of this set it is, or -1. Each
        wavelength is resolved to the nearest line of the FSPS list (within ``tol`` A, else
        ValueError) and that NAME is looked up in ``line_names`` (``AGNLines.rows_for``)."""
        import numpy as np
        lam = np.atleast_1d(np.asarray(wave_rest, dtype=np.float64))
        out = np.full(lam.size, -1, dtype=np.int64)
        pos = {n: k for k, n in enumerate(self.line_names)}
        ew = np.asarray(self.emlines_wave, dtype=np.float64)
        for i, l in enumerate(lam):
            j = int(np.argmin(np.abs(ew - l)))
            if abs(ew[j] - l) > tol:
                raise ValueError(
                    f"line at {l:.2f} A has no FSPS line within {tol:g} A (nearest "
                    f"{self.emlines_names[j]!r} at {ew[j]:.2f} A in "
                    f"{getattr(self, 'emlines_file', 'the FSPS list')})")
            out[i] = pos.get(self.emlines_names[j], -1)
        return out


class SkirtorTorus(Component):
    """Contract of ``skirtor.SkirtorTorus(wave, *, mode, R, file)`` (a ``Component``).

    ``wave``: rest model grid [A]; ``mode``: "corrected" or "cigale"; ``R``: ``skirtor_R``
    (None = the component's default); ``file``: ``skirtor_file`` (None = the default under
    ``$CERIDWEN_AGN_DATA``; missing file -> ValueError at setup naming the converter script).

    rest_parts(theta, ctx) -> {"disk": (n_wave,), "dust": (n_wave,)} L_sun/Hz per M_formed,
        rest frame, BEFORE polar dust and galaxy dust (the hook applies them; the SKIRTOR
        templates' own internal extinction is part of the templates).
    rest_spectrum(theta, ctx) = disk + dust (provided here).
    nuclear_parts(theta, ctx, polar) (optional) -> {"disk", "dust", "polar_t", "polar_ir",
        "nuclear"}: when defined, ``hook.py`` uses it instead of composing
        ``disk * polar.transmission + dust + polar.reemission`` itself (CIGALE renormalises
        the three together and re-emits the angle-averaged absorption of the face-on disc).
        ``polar`` is the PolarDust or None; "polar_t" is the effective line-of-sight
        transmission of the disc, "nuclear" what leaves the nucleus (before galaxy dust).
    needs_galaxy / needs_ldust_gal : class flags; True makes the CSP fill
        ``ctx.gal_nodust_young_old`` / ``ctx.L_dust_gal``.
    param_names must equal ``AGN2_THETA_KEYS["skirtor16"]``.
    """

    needs_galaxy = False
    needs_ldust_gal = False

    def rest_parts(self, theta, ctx):
        raise NotImplementedError

    def rest_spectrum(self, theta, ctx):
        p = self.rest_parts(theta, ctx)
        return p["disk"] + p["dust"]


class PolarDust:
    """Contract of ``polar.PolarDust(wave, *, mode, law)``: one class, both the screen and the
    re-emission. ``law``: one of ``POLAR_LAWS`` (resolved, never None).

    transmission(theta) -> (n_wave,) exp(-tau_polar) along the line of sight (theta holds the
        SKIRTOR keys too, e.g. the inclination, if the extinction depends on them). Does NOT
        include the galaxy's diffuse curve (unlike ``Screen.transmission``).
    reemission(theta, disk, transmission, ctx) -> (n_wave,) L_sun/Hz per M_formed: the grey
        body re-emitting what the polar dust removes from ``disk`` (the unattenuated SKIRTOR
        disc). Without ``SkirtorTorus.nuclear_parts`` its frequency integral equals that of
        ``disk * (1 - transmission)``; a SKIRTOR class with ``nuclear_parts`` may feed it the
        angle-averaged absorption instead (4 pi energy conservation, CIGALE's convention).
    param_names must equal ``AGN2_THETA_KEYS["polar"]``.
    """

    param_names: tuple = ()

    def default_params(self) -> dict:
        raise NotImplementedError

    def transmission(self, theta):
        raise NotImplementedError

    def reemission(self, theta, disk, transmission, ctx):
        raise NotImplementedError


def _agn2_class(name):
    """The class registered under ``name`` in ``AGN2_REGISTRY`` (imported lazily). A module
    that does not exist yet raises NotImplementedError (the component is not implemented on
    this branch), at setup, never at import."""
    entry = AGN2_REGISTRY[name]
    if isinstance(entry, type):
        return entry
    mod_name, cls_name = entry
    import importlib
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name != mod_name:
            raise
        raise NotImplementedError(
            f"AGN component {name!r} ({mod_name}.{cls_name}) is not implemented on this branch "
            f"yet: {mod_name} does not exist. Switch it off in AGNConfig or use a branch where "
            "its session (AGN2 TORUS / NLR / BLR) has landed.") from None
    try:
        return getattr(mod, cls_name)
    except AttributeError:
        raise NotImplementedError(
            f"AGN component {name!r}: {mod_name} has no class {cls_name}; it is not "
            "implemented on this branch yet.") from None


class AGNModel:
    """The composite the CSP holds: the components ``config`` switches on, built on
    the model grid ``wave`` (rest A). Component modules are imported here, lazily.

    Constructor signatures the component modules implement:

        torus.NenkovaTorus(wave, *, mode, tau_interp, torus_file, sps_home)  -> Component
        disk.TempleDisk(wave, *, mode)                                     -> Component
        screen.AGNScreen(wave, *, dust4_type)                              -> Screen
        lines.AGNLines(*, mode, sps_home)                                  -> LineComponent

    AGN2 (through ``AGN2_REGISTRY``; contracts: ``SkirtorTorus``, ``PolarDust``,
    ``AGN2LineSet`` in this module):

        skirtor.SkirtorTorus(wave, *, mode, R, file)        -> components["skirtor"]
        polar.PolarDust(wave, *, mode, law)                 -> polar
        nlr.FeltreNLR(*, mode, file, sps_home)              -> lines   (region "nlr")
        blr.BroadLines(*, mode, file, sps_home)             -> blr     (region "blr")

    ``line_sets``: ``{"nlr": lines, "blr": blr}`` (the ones that are on, in that order).
    """

    def __init__(self, config: AGNConfig, wave, *, sps_home=None):
        if not isinstance(config, AGNConfig):
            raise TypeError(f"agn= expects an AGNConfig, got {type(config).__name__}.")
        self.config = config
        self.mode = config.agn_mode
        self.wave = wave
        self.components = {}
        self.lines = None
        self.polar = None
        self.blr = None
        if config.use_skirtor:
            self.components["skirtor"] = _agn2_class("skirtor16")(
                wave, mode=config.agn_mode, R=config.skirtor_R, file=config.skirtor_file)
        if config.use_nenkova:
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
        if config.polar_dust:
            self.polar = _agn2_class("polar")(wave, mode=config.agn_mode,
                                              law=config.polar_law_resolved)
        if config.use_lines:
            if config.nlr_model == "feltre16":
                self.lines = _agn2_class("feltre16")(mode=config.agn_mode,
                                                     file=config.nlr_file, sps_home=sps_home)
            else:
                from .lines import AGNLines
                self.lines = AGNLines(mode=config.agn_mode, sps_home=sps_home)
        if config.blr:
            self.blr = _agn2_class("blr")(mode=config.agn_mode, file=config.blr_file,
                                          sps_home=sps_home)
        self._check_agn2()

    def _check_agn2(self):
        """Setup-time checks of the AGN2 classes against the contract (theta keys, line-set
        regions and width keys), so a component that drifts from AGN2_THETA_KEYS fails here."""
        checks = [("skirtor16", self.components.get("skirtor")), ("polar", self.polar),
                  ("blr", self.blr)]
        if self.config.use_lines and self.config.nlr_model == "feltre16":
            checks.append(("feltre16", self.lines))
        for name, obj in checks:
            if obj is None:
                continue
            got = tuple(obj.param_names)
            want = AGN2_THETA_KEYS[name]
            if set(got) != set(want) or len(got) != len(want):
                raise ValueError(
                    f"AGN component {type(obj).__name__} ({name}) declares theta keys {got}; "
                    f"the AGN2 contract (ceridwen.agn.AGN2_THETA_KEYS) is {want}.")
        for key, obj, region, width in (("nlr", self.lines, "nlr", "agn_eline_sigma"),
                                        ("blr", self.blr, "blr", "blr_sigma")):
            if obj is None or (key == "nlr" and self.config.nlr_model == "a42"):
                continue
            if getattr(obj, "region", None) != region or getattr(obj, "width_key", None) != width:
                raise ValueError(
                    f"AGN line set {type(obj).__name__}: region={getattr(obj, 'region', None)!r}, "
                    f"width_key={getattr(obj, 'width_key', None)!r}; the {key} set must have "
                    f"region={region!r}, width_key={width!r}.")

    @property
    def line_sets(self) -> dict:
        """``{"nlr": ..., "blr": ...}``: every AGN line set that is on, in painting order."""
        return {k: v for k, v in (("nlr", self.lines), ("blr", self.blr)) if v is not None}

    def _parts(self):
        parts = list(self.components.values())
        parts += [p for p in (self.screen, self.polar, self.lines, self.blr) if p is not None]
        return parts

    @property
    def param_names(self) -> tuple:
        names = []
        for p in self._parts():
            names += [n for n in p.param_names if n not in names]
        return tuple(names)

    def default_params(self) -> dict:
        out = {}
        for p in self._parts():
            out.update(p.default_params())
        return out
