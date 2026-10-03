"""SKIRTOR 2016 clumpy two-phase torus (Stalevski et al. 2012, 2016) with its anisotropic
accretion disc, as CIGALE's ``skirtor2016`` module uses it.

Reference: ``pcigale/sed_modules/skirtor2016.py`` and ``database_builder/skirtor2016`` at cigale
29cb909f (2025.1). The templates are the SKIRTOR grid converted by
``scripts/agn2/build_skirtor_h5.py`` into ``skirtor2016.h5`` (CIGALE's database arrays: F_lambda
per nm on CIGALE's grid, torus dust normalised to a unit integral).

Grid (theta keys, all interpolated multilinearly in the tabulated variable, clamped at the edges):

    skirtor_t   tau_9.7um edge-on        3, 5, 7, 9, 11
    skirtor_pl  radial density index p   0, 0.5, 1, 1.5
    skirtor_q   polar density index q    0, 0.5, 1, 1.5
    skirtor_oa  opening angle [deg]      10 ... 80 (step 10)  (equator to torus edge)
    skirtor_i   inclination [deg]        0 ... 90 (step 10)   (0 = face-on)
    skirtor_R   R_out / R_in (static)    10, 20, 30           (AGNConfig(skirtor_R=...))
    Mcl = 0.97 (the only value)

Setup (NumPy float64, once per (file, R)), CIGALE ``_init_code`` for every node: the SKIRTOR disc
is replaced by CIGALE's parametric disc (``disk_type=1``, Schartmann 2005, ``delta=0``: CIGALE's
defaults), keeping SKIRTOR's angular / absorption pattern:

    AGN1      = the i = 0 node of the same (t, pl, q, oa)
    AGN1.disk *= AGN1.norm / norm
    disk_new  = S(lambda) x trapz(AGN1.disk)                 (S: unit-integral broken power law)
    disk      = nan_to_num(disk_new x disk / AGN1.disk)

Per call (traced, pure JAX), on the templates' grid (``nuclear_parts``):

* the 2^5 corner nodes of (t, pl, q, oa, i) and their multilinear weights w_c;
* per corner, CIGALE's polar dust: ``l_c = geom(oa_c) trapz(AGN1.disk_c) trapz(S (1 - T))``,
  the disc extinguished by ``T`` only if ``i_c <= 90 - oa_c`` (type 1), and every part divided
  by ``1 + l_c`` (CIGALE's ``norm = 1 / trapz(dust + grey body)`` with ``trapz(dust) = 1``);
* the parts are blended with w_c AFTER this, so the result is CIGALE's at every node and
  continuous between nodes (the type-1 switch moves with the corner weights);
* amplitude ``P`` (the luminosity of torus dust + polar dust, CIGALE's ``agn_power``):
  ``"cigale"``: ``P = L_dust,gal x fagn / (1 - fagn)`` (CIGALE: ``scale (1/(1 - fracAGN) - 1)``
  with ``scale = dust.luminosity``, ``lambda_fracAGN = "0/0"``), so ``fagn`` IS CIGALE's
  ``fracAGN`` = P / (P + L_dust,gal); ``"corrected"``: ``P = fagn x L_bol,*`` (as the Nenkova
  torus: L_bol,* the dust-free stellar luminosity), and ``frac_agn_cigale = P / (P + L_dust,gal)``
  is derived in post-processing (``frac_agn_cigale``).

Resampling onto the model grid (static indices and weights, applied in the trace):

``"cigale"``     CIGALE's own: ``np.interp`` of L_lambda, 0 outside the template grid
                 (``pcigale/sed/utils.py:interpolate_lumin``), then L_nu = L_lambda lambda^2 / c.
``"corrected"``  log-log between neighbouring template points that are both > 0 (0 across a zero),
                 then each part rescaled so that its frequency integral on the model grid (the
                 DL07 trapezoid weights) equals its luminosity on the template grid: the model
                 carries exactly ``P`` in dust + polar dust on any grid.

Units: ``rest_parts`` / ``nuclear_parts`` return L_sun/Hz per M_formed (``ctx`` units).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import (AGN2_THETA_KEYS, AGN_MODE_CIGALE, AGN_MODE_CORRECTED)
from . import SkirtorTorus as _SkirtorContract
from .polar import C_NM_S, greybody_shape, polar_geometry
from .polar import POLAR_DEFAULTS
from .torus import trapezoid_nu_weights

__all__ = ["SkirtorTorus", "SKIRTOR_FILE_NAME", "SKIRTOR_DEFAULTS", "SKIRTOR_AXES",
           "default_skirtor_path", "schartmann2005_disk", "skirtor_disk", "parametric_disk",
           "cigale_disk_replacement"]

SKIRTOR_FILE_NAME = "skirtor2016.h5"
SKIRTOR_AXES = ("t", "pl", "q", "oa", "i")
THETA_OF_AXIS = {"t": "skirtor_t", "pl": "skirtor_pl", "q": "skirtor_q", "oa": "skirtor_oa",
                 "i": "skirtor_i"}
# CIGALE skirtor2016 ``parameters`` defaults (t 7, pl 1.0, q 1.0, oa 40, R 20, i 30); fagn 0 so
# that a switched-on torus with the default amplitude adds exactly nothing (fracAGN 0.1 in CIGALE)
SKIRTOR_DEFAULTS = {"fagn": 0.0, "skirtor_t": 7.0, "skirtor_pl": 1.0, "skirtor_q": 1.0,
                    "skirtor_oa": 40.0, "skirtor_i": 30.0}
SKIRTOR_R_DEFAULT = 20.0
DISK_TYPES = (0, 1)                 # 0 SKIRTOR power law, 1 Schartmann 2005 (CIGALE default)
CONVERTER = "scripts/agn2/build_skirtor_h5.py"

_CACHE = {}


def default_skirtor_path():
    root = os.environ.get("CERIDWEN_AGN_DATA") or str(Path.home() / ".ceridwen" / "agn")
    return Path(root) / SKIRTOR_FILE_NAME


def parametric_disk(wl, limits, coefs):
    """CIGALE ``disk``: broken power law in F_lambda, unit ``trapz`` over ``wl`` [nm]."""
    ss = np.searchsorted(wl, limits)
    wpl = [slice(lo, hi) for lo, hi in zip(ss[:-1], ss[1:])]
    norms = np.ones_like(coefs)
    for idx in range(1, coefs.size):
        norms[idx] = norms[idx - 1] * limits[idx] ** (coefs[idx - 1] - coefs[idx])
    spectrum = np.zeros_like(wl)
    for w, coef, norm in zip(wpl, coefs, norms):
        spectrum[w] = wl[w] ** coef * norm
    return spectrum * (1.0 / np.trapezoid(spectrum, x=wl))


def schartmann2005_disk(wl, delta=0.0):
    return parametric_disk(wl, np.array([8.0, 50.0, 125.0, 10000.0, 1e6]),
                           np.array([1.0, -0.2, -1.5 + delta, -4.0]))


def skirtor_disk(wl, delta=0.0):
    return parametric_disk(wl, np.array([8.0, 10.0, 100.0, 5000.0, 1e6]),
                           np.array([0.2, -1.0, -1.5 + delta, -4.0]))


def cigale_disk_replacement(wl, disk, norm, disk_type=1, delta=0.0):
    """CIGALE ``_init_code`` l.312-339 for every node at once.

    ``disk`` (..., n_i, n_wl), ``norm`` (..., n_i), the i axis second to last with i = 0 first.
    Returns ``(disk_new (..., n_i, n_wl), agn1_int (..., n_i), S (n_wl,))``: the replaced disc,
    ``trapz(AGN1.disk)`` after AGN1's renormalisation to each node's norm (the factor of the
    face-on disc in the polar-dust absorption), and the unit-integral parametric shape."""
    if disk_type == 0:
        S = skirtor_disk(wl, delta=delta)
    elif disk_type == 1:
        S = schartmann2005_disk(wl, delta=delta)
    else:
        raise ValueError(f"disk_type={disk_type!r}: 0 (SKIRTOR) or 1 (Schartmann 2005); CIGALE's "
                         "2 (ADAF) is not implemented in CERIDWEN.")
    agn1 = disk[..., :1, :] * (norm[..., :1] / norm)[..., None]           # AGN1.disk rescaled
    agn1_int = np.trapezoid(agn1, x=wl, axis=-1)                         # (..., n_i)
    new_agn1 = S * agn1_int[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.nan_to_num(new_agn1 * disk / agn1)
    return out, agn1_int, S


def _trapz_weights(x):
    """``w`` with ``y @ w == trapz(y, x)`` (up to summation order)."""
    d = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def _load(path, R):
    import h5py
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != "ceridwen skirtor2016 v1":
            raise ValueError(f"{path}: not a ceridwen skirtor2016 v1 file (write it with "
                             f"{CONVERTER}).")
        axes = {k: np.asarray(f["axes"][k][()], dtype=np.float64) for k in SKIRTOR_AXES + ("R",)}
        r = np.flatnonzero(axes["R"] == float(R))
        if r.size != 1:
            raise ValueError(f"SkirtorTorus(R={R!r}): skirtor_R must be one of the grid values "
                             f"{axes['R'].tolist()} of {path}.")
        r = int(r[0])
        wl = np.asarray(f["wl_nm"][()], dtype=np.float64)
        disk = np.asarray(f["disk"][r], dtype=np.float64)
        dust = np.asarray(f["dust"][r], dtype=np.float64)
        norm = np.asarray(f["norm"][r], dtype=np.float64)
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    if axes["i"][0] != 0.0:
        raise ValueError(f"{path}: the inclination axis must start at 0 (CIGALE's AGN1 is i=0).")
    return axes, wl, disk, dust, norm, attrs


def _resample_linear(wave_nm, wl):
    """Static (lo, frac, inside) of ``np.interp(wave_nm, wl, y, left=0, right=0)``."""
    hi = np.clip(np.searchsorted(wl, wave_nm, side="right"), 1, wl.size - 1)
    lo = hi - 1
    frac = (wave_nm - wl[lo]) / (wl[hi] - wl[lo])
    inside = (wave_nm >= wl[0]) & (wave_nm <= wl[-1])
    frac = np.where(inside, np.clip(frac, 0.0, 1.0), 0.0)
    return lo, frac, inside


class SkirtorTorus(_SkirtorContract):
    """SKIRTOR torus + disc. ``SkirtorTorus(wave, *, mode, R=None, file=None)``.

    ``rest_parts(theta, ctx)`` -> {"disk", "dust"} without polar dust; ``nuclear_parts(theta,
    ctx, polar)`` -> {"disk", "dust", "polar_t", "polar_ir", "nuclear"} with CIGALE's polar dust
    composed per corner node (``polar``: a ``polar.PolarDust`` or None). See the module docstring.
    Extra keyword arguments (static): ``disk_type`` (1 = CIGALE's default, 0) and ``delta``.
    """

    param_names = AGN2_THETA_KEYS["skirtor16"]
    needs_galaxy = False

    def __init__(self, wave, *, mode, R=None, file=None, disk_type=1, delta=0.0):
        if mode not in (AGN_MODE_CORRECTED, AGN_MODE_CIGALE):
            raise ValueError(f"SkirtorTorus(mode={mode!r}): SKIRTOR exists only in "
                             "agn_mode='corrected' or 'cigale'.")
        if disk_type not in DISK_TYPES or isinstance(disk_type, bool):
            raise ValueError(f"SkirtorTorus(disk_type={disk_type!r}): 1 (Schartmann 2005, "
                             "CIGALE's default) or 0 (SKIRTOR power law).")
        wave = np.asarray(wave, dtype=np.float64)
        if wave.ndim != 1 or wave.size < 2 or not np.all(np.isfinite(wave)) \
                or not np.all(wave > 0) or not np.all(np.diff(wave) > 0):
            raise ValueError("SkirtorTorus: wave must be a 1-D ascending grid of positive "
                             "wavelengths [A].")
        self.mode = mode
        self.R = SKIRTOR_R_DEFAULT if R is None else float(R)
        self.path = Path(file) if file is not None else default_skirtor_path()
        if not self.path.exists():
            raise ValueError(
                f"SkirtorTorus: MISSING the SKIRTOR template file {self.path}. Build it with "
                f"`python {CONVERTER} --raw-dir <SKIRTOR *_sed.dat> --cigale-db <pcigale/data/"
                "skirtor2016>` (writes $CERIDWEN_AGN_DATA/skirtor2016.h5) or pass "
                "AGNConfig(skirtor_file=...).")
        self.disk_type = int(disk_type)
        self.delta = float(delta)
        # CIGALE's fracAGN is relative to the galaxy's dust luminosity
        self.needs_ldust_gal = mode == AGN_MODE_CIGALE

        key = (str(self.path.resolve()), self.path.stat().st_mtime_ns, self.R, self.disk_type,
               self.delta)
        hit = _CACHE.get(key)
        if hit is None:
            hit = self._build_templates()
            _CACHE[key] = hit
        (self.axes_np, self.wl_nm, disk_np, dust_np, agn1_np, S_np, self.file_attrs) = hit
        self.grid_shape = disk_np.shape[:-1]
        self.axes = {k: jnp.asarray(v) for k, v in self.axes_np.items() if k in SKIRTOR_AXES}
        self.disk_t = jnp.asarray(disk_np)
        self.dust_t = jnp.asarray(dust_np)
        self.agn1_int = jnp.asarray(agn1_np)
        self.S = jnp.asarray(S_np)
        self.tw = jnp.asarray(_trapz_weights(self.wl_nm))
        self.wl_j = jnp.asarray(self.wl_nm)
        self._polar_k = {}

        # resampling onto the model grid
        self.wave = wave
        wave_nm = wave / 10.0
        lo, frac, inside = _resample_linear(wave_nm, self.wl_nm)
        self._lo = jnp.asarray(lo)
        self._frac = jnp.asarray(frac)
        self._inside = jnp.asarray(inside)
        self._lam2_c = jnp.asarray(wave_nm ** 2 / C_NM_S)               # L_lambda/nm -> L_nu
        self._w_model = -trapezoid_nu_weights(jnp.asarray(wave))         # |dnu| weights
        if mode == AGN_MODE_CORRECTED:
            wl = self.wl_nm
            hi = np.minimum(lo + 1, wl.size - 1)
            llo, lhi = np.log(wl[lo]), np.log(wl[hi])
            t = np.where(lhi > llo, (np.log(wave_nm) - llo) / np.where(lhi > llo, lhi - llo, 1.0),
                         0.0)
            self._hi = jnp.asarray(hi)
            self._t = jnp.asarray(np.where(inside, np.clip(t, 0.0, 1.0), 0.0))
            self._llam2c = jnp.asarray(np.log(wave_nm ** 2 / C_NM_S))
            if not inside.any():
                raise ValueError(
                    f"SkirtorTorus: the model grid [{wave[0]:.4g}, {wave[-1]:.4g}] A does not "
                    f"overlap the SKIRTOR templates [{10 * wl[0]:.4g}, {10 * wl[-1]:.4g}] A.")

    # ------------------------------------------------------------------ setup
    def _build_templates(self):
        axes, wl, disk, dust, norm, attrs = _load(self.path, self.R)
        disk_new, agn1_int, S = cigale_disk_replacement(wl, disk, norm, self.disk_type,
                                                        self.delta)
        if not (np.all(np.isfinite(disk_new)) and np.all(disk_new >= 0)):
            raise ValueError(f"{self.path}: the CIGALE disc replacement is not finite/positive.")
        if np.any(disk_new > 1e300):
            raise ValueError(f"{self.path}: the CIGALE disc replacement hit nan_to_num's inf "
                             "(a zero face-on disc under a non-zero inclined one).")
        return axes, wl, disk_new, dust, agn1_int, S, attrs

    def default_params(self) -> dict:
        return dict(SKIRTOR_DEFAULTS)

    def provenance(self) -> dict:
        import hashlib
        out = {"file": str(self.path), "R": self.R, "Mcl": float(self.file_attrs.get("Mcl", 0.97)),
               "disk_type": self.disk_type, "delta": self.delta,
               "source_manifest_sha256": str(self.file_attrs.get("source_manifest_sha256", "")),
               "cigale_exact": bool(self.file_attrs.get("cigale_exact", False))}
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
        return out

    def range_messages(self, theta) -> list:
        msgs = []
        for ax in SKIRTOR_AXES:
            key = THETA_OF_AXIS[ax]
            if key in theta:
                v = np.asarray(theta[key], float)
                a = self.axes_np[ax]
                if v.size and (np.nanmin(v) < a[0] or np.nanmax(v) > a[-1]):
                    msgs.append(f"theta['{key}'] = {np.array2string(v, precision=3)} is outside "
                                f"the SKIRTOR grid [{a[0]:g}, {a[-1]:g}]; it is clamped to the "
                                "edge, so the likelihood is flat in it there.")
        if "fagn" in theta and self.mode == AGN_MODE_CIGALE:
            v = np.asarray(theta["fagn"], float)
            if v.size and np.nanmax(v) >= 1.0:
                msgs.append(f"theta['fagn'] = {np.array2string(v, precision=3)}: in "
                            "agn_mode='cigale' fagn is CIGALE's fracAGN = L_AGN,dust / "
                            "(L_AGN,dust + L_dust,gal) and must be < 1 (CIGALE raises at 1).")
        return msgs

    # ------------------------------------------------------------------ traced
    def corners(self, theta):
        """``(idx, w)``: 32 corner indices per axis (dict of (32,) int) and weights (32,)."""
        lo, fr = {}, {}
        for ax in SKIRTOR_AXES:
            a = self.axes[ax]
            x = theta[THETA_OF_AXIS[ax]] if THETA_OF_AXIS[ax] in theta \
                else SKIRTOR_DEFAULTS[THETA_OF_AXIS[ax]]
            x = jnp.ravel(jnp.asarray(x, dtype=jnp.float64))[0]
            n = a.shape[0]
            if n == 1:
                lo[ax] = jnp.zeros((), dtype=jnp.int32)
                fr[ax] = jnp.zeros((), dtype=jnp.float64)
                continue
            j = jnp.clip(jnp.searchsorted(a, x, side="right") - 1, 0, n - 2)
            lo[ax] = j
            fr[ax] = jnp.clip((x - a[j]) / (a[j + 1] - a[j]), 0.0, 1.0)
        bits = np.array([[(c >> d) & 1 for d in range(len(SKIRTOR_AXES))]
                         for c in range(2 ** len(SKIRTOR_AXES))])           # (32, 5) static
        idx, w = {}, jnp.ones(bits.shape[0], dtype=jnp.float64)
        for d, ax in enumerate(SKIRTOR_AXES):
            b = jnp.asarray(bits[:, d])
            n = self.axes[ax].shape[0]
            idx[ax] = jnp.minimum(lo[ax] + b, n - 1)
            w = w * jnp.where(b == 1, fr[ax], 1.0 - fr[ax])
        return idx, w

    def native_parts(self, theta, polar=None):
        """Per unit amplitude P, on the templates' grid (F_lambda per nm): ``disk`` (after the
        polar dust for type-1 corners), ``dust`` (torus), ``polar`` (grey body), ``disk_u`` (the
        disc with T = 1, same per-corner 1 / (1 + l_c)) -- CIGALE's normalisation (dust + polar
        integrates to 1)."""
        idx, w = self.corners(theta)
        sel = (idx["t"], idx["pl"], idx["q"], idx["oa"], idx["i"])
        D = self.disk_t[sel]                                    # (32, n_wl)
        U = self.dust_t[sel]
        if polar is None:
            disk0 = w @ D
            return {"disk": disk0, "dust": w @ U, "polar": jnp.zeros_like(disk0),
                    "disk_u": disk0}
        k = self._polar_k.get(polar.law_index)
        if k is None:                       # NumPy (static): never cache a traced value
            k = np.asarray(polar.k_on(self.wl_nm), dtype=np.float64)
            self._polar_k[polar.law_index] = k
        k = jnp.asarray(k)
        ebv = jnp.ravel(jnp.asarray(theta.get("polar_ebv", POLAR_DEFAULTS["polar_ebv"]),
                                    dtype=jnp.float64))[0]
        temp = jnp.ravel(jnp.asarray(theta.get("polar_temp", POLAR_DEFAULTS["polar_temp"]),
                                     dtype=jnp.float64))[0]
        emis = jnp.ravel(jnp.asarray(theta.get("polar_emis", POLAR_DEFAULTS["polar_emis"]),
                                     dtype=jnp.float64))[0]
        ext = 10.0 ** (-0.4 * k * ebv)                          # (n_wl,)
        J = jnp.dot(self.S * (1.0 - ext), self.tw)               # trapz(S (1 - ext))
        oa_c = self.axes["oa"][idx["oa"]]
        i_c = self.axes["i"][idx["i"]]
        l_c = polar_geometry(oa_c) * self.agn1_int[sel] * J      # (32,)
        a_c = w / (1.0 + l_c)
        type1 = (i_c <= 90.0 - oa_c).astype(jnp.float64)
        disk = ext * ((a_c * type1) @ D) + (a_c * (1.0 - type1)) @ D
        dust = a_c @ U
        bb = greybody_shape(self.wl_j, temp, emis)
        bb_int = jnp.dot(bb, self.tw)
        frac_pol = jnp.sum(w * l_c / (1.0 + l_c))
        pol = frac_pol * bb / jnp.where(bb_int > 0, bb_int, 1.0)
        return {"disk": disk, "dust": dust, "polar": pol, "disk_u": a_c @ D}

    def amplitude(self, theta, ctx):
        """P [L_sun per M_formed]: the luminosity of torus dust + polar dust."""
        f = jnp.ravel(jnp.asarray(theta.get("fagn", SKIRTOR_DEFAULTS["fagn"]),
                                  dtype=jnp.float64))[0]
        if self.mode == AGN_MODE_CIGALE:
            if ctx.L_dust_gal is None:
                raise ValueError("SkirtorTorus(mode='cigale') needs ctx.L_dust_gal (CIGALE's "
                                 "fracAGN is relative to the galaxy's dust luminosity).")
            return jnp.asarray(ctx.L_dust_gal, dtype=jnp.float64) * (f / (1.0 - f))
        return f * jnp.asarray(ctx.L_bol_star, dtype=jnp.float64)

    def to_model(self, y):
        """(n_wl,) F_lambda per nm on the templates' grid -> (n_wave,) F_nu per Hz on the model
        grid (the mode's resampling; see the module docstring)."""
        lo, frac = self._lo, self._frac
        if self.mode == AGN_MODE_CIGALE:
            y_lo, y_hi = y[lo], y[jnp.minimum(lo + 1, y.shape[0] - 1)]
            lam = jnp.where(self._inside, y_lo + (y_hi - y_lo) * frac, 0.0)
            return lam * self._lam2_c
        # log-log: logs on the template grid (n_wl), one exp per model pixel
        pos_nat = y > 0
        ly = jnp.log(jnp.where(pos_nat, y, 1.0))
        seg = self._inside & pos_nat[lo] & pos_nat[self._hi]
        ly_lo, ly_hi = ly[lo], ly[self._hi]
        fnu = jnp.where(seg, jnp.exp(ly_lo + self._t * (ly_hi - ly_lo) + self._llam2c), 0.0)
        target = jnp.dot(y, self.tw)                    # luminosity on the template grid
        have = jnp.dot(fnu, self._w_model)
        ok = have > 0
        return jnp.where(ok, fnu * (target / jnp.where(ok, have, 1.0)), 0.0)

    def nuclear_parts(self, theta, ctx, polar=None):
        """The AGN2 nuclear continuum, (n_wave,) float64 L_sun/Hz per M_formed each:
        ``disk`` (unattenuated SKIRTOR disc), ``dust`` (torus), ``polar_t`` (the disc's effective
        polar transmission, transmitted / unattenuated, 1 where the disc is 0), ``polar_ir``
        (grey body), ``nuclear`` = transmitted disc + dust + polar_ir."""
        P = self.amplitude(theta, ctx)
        nat = self.native_parts(theta, polar)
        if polar is None:
            disk0 = P * self.to_model(nat["disk_u"])
            dust = P * self.to_model(nat["dust"])
            zero = jnp.zeros_like(disk0)
            return {"disk": disk0, "dust": dust, "polar_t": jnp.ones_like(disk0),
                    "polar_ir": zero, "nuclear": disk0 + dust}
        disk_t = P * self.to_model(nat["disk"])
        # the unattenuated disc in the same normalisation (CIGALE divides by 1 + l_c per corner)
        disk_u = P * self.to_model(nat["disk_u"])
        dust = P * self.to_model(nat["dust"])
        ir = P * self.to_model(nat["polar"])
        ok = disk_u > 0
        t = jnp.where(ok, disk_t / jnp.where(ok, disk_u, 1.0), 1.0)
        return {"disk": disk_u, "dust": dust, "polar_t": t, "polar_ir": ir,
                "nuclear": disk_t + dust + ir}

    def rest_parts(self, theta, ctx):
        """{"disk", "dust"}: SKIRTOR without polar dust (the scaffold's contract)."""
        p = self.nuclear_parts(theta, ctx, None)
        return {"disk": p["disk"], "dust": p["dust"]}
