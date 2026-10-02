"""The single point where the AGN enters the CSP forward model.

``CSPBasis(..., agn=AGNConfig(...))`` builds an ``AGNHook``; ``CSPBasis`` wraps the chosen
``get_spectrum_*`` kernel ONCE (``CSPBasis._with_agn``), so the eight kernels stay untouched.

Per call the CSP builds an ``AGNContext`` (``CSPBasis._agn_context``) and calls
``AGNHook.__call__(theta, ctx)`` -> ``(agn_continuum_rest, agn_absorbed_lum)``:

* each continuum component's ``rest_spectrum`` (intrinsic, L_sun/Hz per M_formed);
* its screen, per mode:

  ===========  ==========================  ==========================================
  component    "fsps"                      "prospector" / "corrected"
  ===========  ==========================  ==========================================
  torus        diffuse curve (dust2 only)  none (prospector) / diffuse (corrected)
  disc         (refused)                   ``Screen.transmission`` with dust4_type=1,
                                           else the diffuse curve (prospect add_dust)
  ===========  ==========================  ==========================================

* energy balance (``"corrected"`` with dust emission): the absorbed AGN light, every screen
  included, is re-emitted through the CSP's own DL07 ``compute_dust_emission``: DL07 is
  linear in the absorbed luminosity (both its passes), so the AGN's re-emission is
  ``L_abs,AGN x`` (DL07's response to a unit absorbed luminosity), which equals feeding the
  AGN into the stellar call up to DL07's 1e-70 floor and is exactly 0 at fagn = 0;
* the IGM order: ``own_igm`` is True in the reproduction modes, where the AGN does not see
  the galaxy's IGM transmission T_gal but its own T_agn (``agn_transmission``): ``"fsps"``
  T_agn = 1 (FSPS adds the torus after ``igm_absorb``, ``compsp.f90:138-151``);
  ``"prospector"`` T_agn = ``fake_fsps.add_igm`` (Madau 1995 as Prospector codes it for the
  AGN, ``agnssp_basis.get_spectrum``; it differs from CERIDWEN's madau1995 blueward of
  Ly-alpha by 1 % at z = 2, 4 % at z = 6). The CSP adds ``agn * (T_agn - T_gal)`` after the
  IGM (``CSPBasis._agn_after_igm``). ``"corrected"`` adds the AGN before the IGM, as every
  other component.

Without ``agn=`` none of this exists and the kernels are called as before.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import (AGN_MODE_CORRECTED, AGN_MODE_FSPS, AGN_MODE_PROSPECTOR, AGN_THETA_DEFAULTS,
               AGNConfig, AGNContext, AGNModel)
from .torus import trapezoid_nu_weights

__all__ = ["AGNHook", "AGN_TAU_RANGE", "prospector_igm_transmission"]

AGN_TAU_RANGE = (5.0, 150.0)

# prospect/sources/fake_fsps.py add_igm (Madau 1995)
_LYLIM = 911.75
_LYW = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150,
                 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919,
                 914.576])
_LYCOEFF = np.array([0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960, 0.0006967, 0.0006236,
                     0.0005665, 0.0005200, 0.0004817, 0.0004487, 0.0004200, 0.0003947,
                     0.000372, 0.000352, 0.0003334, 0.00031644])


def prospector_igm_transmission(wave_rest, zred, igm_factor=1.0):
    """exp(-tau * igm_factor) of ``fake_fsps.add_igm`` on the rest-frame grid (pure JAX,
    float64). Its final ``clip(spec * T, 1e-70)`` acts on the spectrum, not on T, and is not
    reproduced (it is below float32's range)."""
    zred = jnp.asarray(zred, dtype=jnp.float64)
    rw = jnp.asarray(wave_rest, dtype=jnp.float64) * (1.0 + zred)
    tau_line = jnp.zeros_like(rw)
    for lyw, c in zip(_LYW, _LYCOEFF):
        inside = (rw >= lyw) & (rw <= lyw * (1.0 + zred))
        tau_line = tau_line + jnp.where(inside, c * jnp.exp(3.46 * jnp.log(rw / lyw)), 0.0)
    xem = 1.0 + zred
    xc = jnp.clip(rw / _LYLIM, 1.0, None)
    xc = jnp.where(xc > xem, xem, xc)
    tau_cont = 0.25 * xc**3 * (jnp.exp(0.46 * jnp.log(xem)) - jnp.exp(0.46 * jnp.log(xc)))
    tau_cont = tau_cont + 9.4 * jnp.exp(1.5 * jnp.log(xc)) * (
        jnp.exp(0.18 * jnp.log(xem)) - jnp.exp(0.18 * jnp.log(xc)))
    tau_cont = tau_cont - 0.7 * xc**3 * (jnp.exp(-1.32 * jnp.log(xc))
                                          - jnp.exp(-1.32 * jnp.log(xem)))
    tau_cont = tau_cont - 0.023 * (jnp.exp(1.68 * jnp.log(xem)) - jnp.exp(1.68 * jnp.log(xc)))
    return jnp.exp(-(tau_line + tau_cont) * igm_factor)


def _sha256(path) -> str:
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AGNHook:
    """Static AGN wiring for one CSP.

    Parameters
    ----------
    config : AGNConfig
    wave : (n_wave,) rest-frame model grid [A]
    add_dust_emission : bool -- whether the CSP has DL07 (decides the energy balance)
    sps_home : str or None
    """

    def __init__(self, config: AGNConfig, wave, *, add_dust_emission, sps_home=None):
        if not isinstance(config, AGNConfig):
            raise TypeError(
                f"agn= expects ceridwen.agn.AGNConfig(agn_mode=...), got {type(config).__name__}")
        self.config = config
        self.mode = config.agn_mode
        self.energy_balance = config.energy_balance(add_dust_emission)
        self.own_igm = self.mode in (AGN_MODE_FSPS, AGN_MODE_PROSPECTOR)
        self.model = AGNModel(config, wave, sps_home=sps_home)
        self.wave = wave
        # |DL07 trapezoid weights| (positive): integrates F_nu over frequency
        self._w = -trapezoid_nu_weights(wave)
        self.need_galaxy = "disk" in self.model.components
        self.param_names = self.model.param_names
        self._defaults = self.model.default_params()
        # spec_dustfree of unit absorbed luminosity for DL07 (one pixel, 1/|w|; spec_attn = 0)
        w_np = np.asarray(self._w, dtype=np.float64)
        j = int(np.argmax(w_np))
        unit = np.zeros_like(w_np)
        unit[j] = 1.0 / w_np[j]
        self._unit_absorbed = jnp.asarray(unit)

    # ---------------------------------------------------------------- static helpers
    def default_params(self) -> dict:
        return dict(self._defaults)

    def filled(self, theta) -> dict:
        """The AGN theta keys, each from ``theta`` when present else its default (static)."""
        return {k: (theta[k] if k in theta else self._defaults[k]) for k in self.param_names}

    def provenance(self) -> dict:
        """Static facts a result file should record (INT writes them into fit.py)."""
        out = {"agn_mode": self.mode, "tau_interp": self.config.tau_interp,
               "agn_energy_balance": bool(self.energy_balance),
               "dust4_type": int(self.config.dust4_type),
               "agn_components": ",".join(sorted(self.model.components))
               + (",lines" if self.model.lines is not None else "")}
        torus = self.model.components.get("torus")
        if torus is not None:
            import hashlib
            out["agn_torus_file"] = str(torus.path)
            out["agn_torus_sha256"] = hashlib.sha256(torus.path.read_bytes()).hexdigest()
            out["agn_torus_native_integrals"] = np.asarray(torus.native_integrals).tolist()
        disk = self.model.components.get("disk")
        if disk is not None:
            out["agn_disk_file"] = str(disk.template_file)
            out["agn_disk_sha256"] = _sha256(disk.template_file)
        if self.model.screen is not None:
            out["agn_dust4"] = "dust2 diffuse curve x exp(-dust4 (lam/5500)^dust4_index)"
        lines = self.model.lines
        if lines is not None:
            from .lines import A42_FILE
            out["agn_lines_table"] = str(A42_FILE)
            out["agn_lines_table_sha256"] = _sha256(A42_FILE)
            out["agn_lines_emlines_file"] = str(lines.emlines_file)
            out["agn_lines_emlines_sha256"] = _sha256(lines.emlines_file)
        return out

    def range_messages(self, theta) -> list:
        """``check_param_ranges`` messages for the AGN keys (setup-time, NumPy)."""
        msgs = []
        if "agn_tau" in theta and "torus" in self.model.components:
            v = np.asarray(theta["agn_tau"], float)
            lo, hi = AGN_TAU_RANGE
            if v.size and (np.nanmin(v) < lo or np.nanmax(v) > hi):
                what = ("EXTRApolated linearly with the end segments, as Prospector's "
                        "fake_fsps.agn_torus does (the SED can go negative)"
                        if self.mode == AGN_MODE_PROSPECTOR else
                        "clamped to the edge template (FSPS agn_dust.f90), so the likelihood "
                        "is flat in agn_tau there")
                msgs.append(f"theta['agn_tau'] = {np.array2string(v, precision=3)} is outside "
                            f"the Nenkova+08 grid [{lo:g}, {hi:g}]; it is {what}.")
        for key in ("fagn", "fagn_bbb", "agn_elum", "dust4"):
            if key in theta and key in self.param_names:
                v = np.asarray(theta[key], float)
                if v.size and np.nanmin(v) < 0:
                    msgs.append(f"theta['{key}'] = {np.array2string(v, precision=3)} is "
                                "negative; the AGN term is linear in it and goes negative.")
        return msgs

    # ---------------------------------------------------------------- traced
    def agn_transmission(self, zred, igm_factor):
        """(n_wave,) T_agn of the reproduction modes (see the module docstring)."""
        if self.mode == AGN_MODE_FSPS:
            return jnp.ones(self.wave.shape, dtype=jnp.float64)
        return prospector_igm_transmission(self.wave, zred, igm_factor)

    def components(self, theta, ctx: AGNContext):
        """``(intrinsic, attenuated)`` (n_wave,) float64 sums over the continuum components."""
        th = self.filled(theta)
        dc = ctx.diffuse_curve
        intrinsic = jnp.zeros_like(self.wave, dtype=jnp.float64)
        attenuated = jnp.zeros_like(self.wave, dtype=jnp.float64)
        for name, comp in self.model.components.items():
            s = comp.rest_spectrum(th, ctx)
            intrinsic = intrinsic + s
            if name == "torus":
                att = s if self.mode == AGN_MODE_PROSPECTOR else s * dc
            elif self.model.screen is not None:
                att = s * self.model.screen.transmission(th, ctx)
            else:
                att = s * dc
            attenuated = attenuated + att
        return intrinsic, attenuated

    def __call__(self, theta, ctx: AGNContext, dust_emission=None, dust_theta=None,
                 formed_mass=1.0):
        """``(agn_continuum_rest (n_wave,) float64, agn_absorbed_lum scalar)``.

        ``dust_emission``: the CSP's ``DustEmission`` (needed when ``energy_balance``);
        ``dust_theta``: its (qpah, umin, gamma) values; ``formed_mass``: the formed mass of the
        SFH in the units of ``ctx`` (sum of the SSP weights), which the AGN line luminosities
        (per M_formed) are multiplied by before their absorbed part joins the balance."""
        intrinsic, attenuated = self.components(theta, ctx)
        absorbed = jnp.dot(intrinsic - attenuated, self._w)
        if self.energy_balance and self.model.lines is not None:
            # the AGN line light the galaxy screen absorbs joins the re-emitted luminosity
            th = self.filled(theta)
            L = self.model.lines.line_luminosity(th) * formed_mass
            absorbed = absorbed + jnp.sum(
                L * (1.0 - self.line_dust_transmission(ctx.diffuse_curve)))
        if not self.energy_balance:
            return attenuated, absorbed
        _unit, tduste = self.dust_response(ctx, dust_emission, dust_theta)
        return attenuated + absorbed * tduste, absorbed

    def line_dust_transmission(self, diffuse_curve):
        """(n_lines,) dust transmission of the AGN lines, or None. ``"corrected"``: the galaxy's
        diffuse curve at each line (the narrow-line region lies outside the nuclear ``dust4``
        screen, so only the galaxy's diffuse dust acts on it). ``"fsps"`` / ``"prospector"``:
        None, Prospector's ``AGNSpecModel`` attenuates its AGN lines by nothing."""
        if self.mode != AGN_MODE_CORRECTED or self.model.lines is None:
            return None
        return jnp.interp(jnp.asarray(self.model.lines.line_wave), self.wave, diffuse_curve)

    def dust_response(self, ctx, dust_emission, dust_theta):
        """``(mdust, tduste)`` of DL07 for an absorbed luminosity of exactly 1 L_sun (per
        M_formed): the CSP's own ``compute_dust_emission`` on a one-pixel unit input."""
        qpah, umin, gamma = dust_theta
        _spec, mdust, tduste = dust_emission.compute_dust_emission(
            spec_attn=jnp.zeros_like(self._unit_absorbed), spec_dustfree=self._unit_absorbed,
            spec_lambda=self.wave, diffuse_curve=ctx.diffuse_curve,
            duste_qpah=qpah, duste_umin=umin, duste_gamma=gamma)
        return mdust, tduste
