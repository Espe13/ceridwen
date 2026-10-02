"""Nenkova et al. (2008) clumpy AGN torus, in the three ``agn_mode`` conventions.

The templates are FSPS's ``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat``: three
comment lines, one row of the nine optical depths 5..150, then 125 rows of
``lambda [A], F_nu * 9`` with F_nu "normalised to unity" (integral over frequency; on the
native 125 points it is 0.9993, see ``native_integrals``).

The resampling onto the model grid is done once, in float64 NumPy, at construction:

``"fsps"``        ``sps_setup.f90:919-923``: ``i1 = locate(lambda, lam_1)``,
                  ``i2 = locate(lambda, lam_125)``, log10-log10 linear interpolation of
                  ``F + tiny30`` (``linterparr``: linear EXTRApolation at the ends) on
                  ``[i1, i2]``, minus ``tiny30``; zero elsewhere. No renormalisation.
``"prospector"``  ``prospect/sources/fake_fsps.py:agn_torus``: ``i1, i2`` = argmin of
                  ``|lambda - lam_1|``, ``|lambda - lam_125|`` (inclusive), ``np.interp``
                  (constant ends) of ``log10(F + 1e-30)``, minus ``1e-30``.
``"corrected"``   log10-log10 over the strictly positive rows only, exactly 0 outside
                  ``[lam_first_positive, lam_last]``, then each column renormalised to
                  ``sum(T_j * w) == 1`` with the DL07 trapezoid-in-frequency weights ``w``,
                  so ``L_torus = fagn * L_bol`` on any grid (23 Sep design D4, D5).

The interpolation in ``agn_tau`` is traced (pure JAX, branch-free):

``"fsps"`` / ``"corrected"``  ``agn_dust.f90:17-23``: clamped linear (no extrapolation),
                  ``tau_interp="log"`` (corrected only) does the same in log10(tau).
``"prospector"``  ``scipy.interpolate.interp1d(..., fill_value="extrapolate")``: linear,
                  EXTRApolated outside [5, 150] with the end segments (may go negative), with
                  scipy's own formula ``y_lo + slope * (x - x_lo)``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from . import (AGN_MODE_CORRECTED, AGN_MODE_FSPS, AGN_MODE_PROSPECTOR, AGN_MODES,
               AGN_THETA_DEFAULTS, AGN_TAU_DEFAULT_PROSPECTOR, Component)

__all__ = ["NenkovaTorus", "TORUS_FILE_NAME", "TORUS_TAU_NODES", "load_nenkova",
           "trapezoid_nu_weights", "resample_fsps", "resample_prospector",
           "resample_corrected", "fsps_locate"]

TORUS_FILE_NAME = "Nenkova08_y010_torusg_n10_q2.0.dat"
TORUS_TAU_NODES = (5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0)
CLIGHT_AA_S = 2.99792458e18          # the constant of dust/DustEmission.py
# FSPS sps_vars.f90:438 ``tiny30 = 10**(-30.0)``: -30.0 is a default (single-precision) REAL
# literal, so the constant is float32(1e-30) promoted to double, 1.0000000031710769e-30; on
# the D4 ramp (100-150 A) that moves the FSPS template by 2e-10 relative against 1e-30
TINY30_FSPS = float(np.float32(1e-30))
TINY30_PROSPECT = 1e-30              # prospect/sources/fake_fsps.py agn_torus
NORM_WARN = 0.01                     # warn when the model grid misses > 1 % of the torus energy
N_DENSE = 20000                      # log grid on which the template's own integral is taken

_TORUS_CACHE = {}


def default_torus_path(sps_home=None):
    """``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat`` (``sps_home`` overrides)."""
    root = sps_home if sps_home is not None else os.environ.get("SPS_HOME")
    if not root:
        raise FileNotFoundError(
            f"The AGN torus templates ({TORUS_FILE_NAME}) are read from $SPS_HOME/dust/, but "
            "$SPS_HOME is unset and no sps_home was given. Set `export SPS_HOME=/path/to/fsps` "
            "or pass AGNConfig(torus_file=...).")
    return Path(root) / "dust" / TORUS_FILE_NAME


def load_nenkova(path):
    """``(tau (9,), lam (125,) [A], spec (125, 9) F_nu)``, float64, validated."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Error opening AGN torus file: {path}. File does not exist (FSPS ships it as "
            f"$SPS_HOME/dust/{TORUS_FILE_NAME}).")
    tau = np.loadtxt(path, skiprows=3, max_rows=1, dtype=np.float64)
    table = np.loadtxt(path, skiprows=4, dtype=np.float64)
    if tau.shape != (9,) or table.shape != (125, 10):
        raise RuntimeError(f"{path}: expected 9 optical depths and 125 x 10 rows, got "
                           f"{tau.shape} and {table.shape}")
    if not np.array_equal(tau, np.asarray(TORUS_TAU_NODES)):
        raise RuntimeError(f"{path}: optical depths {tau.tolist()} != {list(TORUS_TAU_NODES)}")
    lam = table[:, 0]
    if not np.all(np.diff(lam) > 0):
        raise RuntimeError(f"{path}: wavelengths not strictly increasing")
    return tau, lam, table[:, 1:]


def trapezoid_nu_weights(wave):
    """Trapezoid-in-frequency weights, the expression of ``DustEmission.__init__`` verbatim
    (``dnu < 0`` on an ascending grid, so these are negative; ``-w`` integrates F_nu dnu)."""
    nu = CLIGHT_AA_S / jnp.asarray(wave)
    dnu = jnp.diff(nu)
    return jnp.concatenate([
        jnp.array([0.5 * dnu[0]]),
        0.5 * (dnu[:-1] + dnu[1:]),
        jnp.array([0.5 * dnu[-1]]),
    ])


def fsps_locate(xx, x):
    """FSPS ``locate.f90`` (Numerical Recipes), 1-based: ``j`` with ``xx(j) <= x < xx(j+1)``,
    0 below the table, ``n`` above, 1 at ``xx(1)`` and ``n-1`` at ``xx(n)``."""
    n = len(xx)
    ascnd = xx[n - 1] >= xx[0]
    jl, ju = 0, n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascnd == (x >= xx[jm - 1]):
            jl = jm
        else:
            ju = jm
    if x == xx[0]:
        return 1
    if x == xx[n - 1]:
        return n - 1
    return jl


def _linterparr(xin, yin, xout):
    """FSPS ``linterparr.f90``: klo = max(min(locate(xin, x), n-1), 1), linear (extrapolating)."""
    n = len(xin)
    out = np.empty(len(xout), dtype=np.float64)
    for i, x in enumerate(xout):
        klo = max(min(fsps_locate(xin, x), n - 1), 1) - 1          # to 0-based
        out[i] = yin[klo] + (yin[klo + 1] - yin[klo]) * (x - xin[klo]) / (xin[klo + 1] - xin[klo])
    return out


def resample_fsps(wave, lam, spec):
    """(n_wave, 9): ``sps_setup.f90:919-923`` on ``wave`` (zero outside ``[i1, i2]``)."""
    wave = np.asarray(wave, dtype=np.float64)
    out = np.zeros((wave.size, spec.shape[1]), dtype=np.float64)
    i1 = fsps_locate(wave, lam[0])
    i2 = fsps_locate(wave, lam[-1])
    if i1 < 1:          # Fortran would index spec_lambda(0); never on a grid reaching 100 A
        i1 = 1
    if i2 < i1:
        return out
    seg = slice(i1 - 1, i2)                                     # Fortran i1:i2 inclusive
    lx = np.log10(wave[seg])
    for j in range(spec.shape[1]):
        out[seg, j] = 10.0 ** _linterparr(np.log10(lam), np.log10(spec[:, j] + TINY30_FSPS), lx) \
            - TINY30_FSPS
    return out


def resample_prospector(wave, lam, spec):
    """(n_wave, 9): the resampling of ``fake_fsps.agn_torus`` on ``wave``."""
    wave = np.asarray(wave, dtype=np.float64)
    out = np.zeros((wave.size, spec.shape[1]), dtype=np.float64)
    i1 = int(np.argmin(np.abs(wave - lam[0])))
    i2 = int(np.argmin(np.abs(wave - lam[-1])))
    for j in range(spec.shape[1]):
        out[i1:i2 + 1, j] = 10 ** np.interp(np.log10(wave[i1:i2 + 1]), np.log10(lam),
                                            np.log10(spec[:, j] + TINY30_PROSPECT)) - TINY30_PROSPECT
    return out


def resample_corrected(wave, lam, spec):
    """(n_wave, 9) un-normalised: log10-log10 over each column's strictly positive rows,
    exactly 0 outside [first positive lambda, last lambda]. Never negative."""
    wave = np.asarray(wave, dtype=np.float64)
    out = np.zeros((wave.size, spec.shape[1]), dtype=np.float64)
    for j in range(spec.shape[1]):
        pos = spec[:, j] > 0
        lp, sp = lam[pos], spec[pos, j]
        inside = (wave >= lp[0]) & (wave <= lp[-1])
        out[inside, j] = 10.0 ** np.interp(np.log10(wave[inside]), np.log10(lp), np.log10(sp))
    return out


class NenkovaTorus(Component):
    """Nenkova+08 torus: ``rest_spectrum = fagn * L_bol_star * T(agn_tau)`` [L_sun/Hz per
    M_formed], before any attenuation. ``T`` integrates to 1 over frequency exactly in
    ``"corrected"`` mode and to the FSPS/Prospector resampled value otherwise.

    Attributes: ``templates`` (n_wave, 9) jnp float64, ``tau_grid`` (9,), ``native_integrals``
    (9,) = sum(T_j |w|) of the resampled columns before any renormalisation (on this grid),
    ``file_integrals`` (9,) the trapezoid on the file's own 125 points (the "unity" of the
    file header, 0.9993), ``dense_integrals`` (9,) the log-log SED on a dense log grid over
    the whole template (0.989-0.996: the file's unity is a coarse-trapezoid artefact),
    ``path``.
    """

    param_names = ("fagn", "agn_tau")

    def __init__(self, wave, *, mode, tau_interp="linear", torus_file=None, sps_home=None):
        if mode not in AGN_MODES:
            raise ValueError(f"NenkovaTorus(mode={mode!r}): choose one of {AGN_MODES}")
        if tau_interp not in ("linear", "log"):
            raise ValueError(f"NenkovaTorus(tau_interp={tau_interp!r}): 'linear' or 'log'")
        if tau_interp == "log" and mode != AGN_MODE_CORRECTED:
            raise ValueError("tau_interp='log' exists only in agn_mode='corrected'")
        self.mode = mode
        self.tau_interp = tau_interp
        self.path = Path(torus_file) if torus_file is not None else default_torus_path(sps_home)
        wave_np = np.asarray(wave, dtype=np.float64)
        if wave_np.ndim != 1 or not np.all(np.diff(wave_np) > 0):
            raise ValueError("NenkovaTorus: wave must be 1-D and strictly increasing")

        key = (mode, str(self.path), wave_np.shape, wave_np.tobytes())
        hit = _TORUS_CACHE.get(key)
        if hit is None:
            hit = self._build(wave_np)
            _TORUS_CACHE[key] = hit
        tau, T, native, file_int, dense = hit
        self.tau_grid_np = tau
        self.tau_grid = jnp.asarray(tau)
        self.log_tau_grid = jnp.asarray(np.log10(tau))
        self.n_tau = tau.size
        self.templates = jnp.asarray(T)
        self.native_integrals = native
        self.file_integrals = file_int
        self.dense_integrals = dense
        if mode == AGN_MODE_CORRECTED:
            frac = native / dense
            if np.any(frac < 1.0 - NORM_WARN):
                warnings.warn(
                    "NenkovaTorus: the model wavelength grid holds only "
                    f"{np.array2string(frac, precision=4)} of the torus energy "
                    f"(tau = {tau.tolist()}; 1 = the templates' own integral on a dense log "
                    "grid); 'corrected' mode renormalises to L_torus = fagn * L_bol, but the "
                    "torus SED is truncated on this grid.", stacklevel=3)

    def _build(self, wave):
        tau, lam, spec = load_nenkova(self.path)
        w = -np.asarray(trapezoid_nu_weights(wave), dtype=np.float64)
        nu = CLIGHT_AA_S / lam
        wf = -np.concatenate([[0.5 * (nu[1] - nu[0])], 0.5 * (nu[2:] - nu[:-2]),
                              [0.5 * (nu[-1] - nu[-2])]])
        file_int = spec.T @ wf
        lam_pos = lam[np.all(spec > 0, axis=1)]
        dense_wave = np.geomspace(lam_pos[0], lam[-1], N_DENSE)
        dense = resample_corrected(dense_wave, lam, spec).T @ \
            (-np.asarray(trapezoid_nu_weights(dense_wave), dtype=np.float64))
        if self.mode == AGN_MODE_FSPS:
            T = resample_fsps(wave, lam, spec)
            native = T.T @ w
        elif self.mode == AGN_MODE_PROSPECTOR:
            T = resample_prospector(wave, lam, spec)
            native = T.T @ w
        else:
            T = resample_corrected(wave, lam, spec)
            native = T.T @ w
            if np.any(native <= 0):
                raise ValueError(
                    "NenkovaTorus: the model wavelength grid "
                    f"[{wave[0]:.4g}, {wave[-1]:.4g}] A does not overlap the torus templates "
                    f"[{lam[1]:.4g}, {lam[-1]:.4g}] A; no torus can be added on it.")
            T = T / native[None, :]
        return tau, T, native, file_int, dense

    def default_params(self) -> dict:
        tau0 = AGN_TAU_DEFAULT_PROSPECTOR if self.mode == AGN_MODE_PROSPECTOR \
            else AGN_THETA_DEFAULTS["agn_tau"]
        return {"fagn": jnp.asarray(AGN_THETA_DEFAULTS["fagn"]),
                "agn_tau": jnp.asarray(tau0)}

    def template(self, agn_tau):
        """(n_wave,) T(agn_tau): unit-normalised SED shape (pure JAX)."""
        t = jnp.ravel(jnp.asarray(agn_tau, dtype=self.templates.dtype))[0]
        n = self.n_tau
        if self.mode == AGN_MODE_PROSPECTOR:
            # scipy interp1d(kind="linear", fill_value="extrapolate"), its own expression
            hi = jnp.clip(jnp.searchsorted(self.tau_grid, t, side="left"), 1, n - 1)
            lo = hi - 1
            x_lo, x_hi = self.tau_grid[lo], self.tau_grid[hi]
            y_lo, y_hi = self.templates[:, lo], self.templates[:, hi]
            slope = (y_hi - y_lo) / (x_hi - x_lo)
            return slope * (t - x_lo) + y_lo
        grid = self.tau_grid
        if self.tau_interp == "log":
            grid = self.log_tau_grid
            t = jnp.log10(jnp.maximum(t, 1e-300))
        # agn_dust.f90:17-23 (locate: grid[jlo] <= t < grid[jlo+1]), dj clamped to [0, 1]
        jlo = jnp.clip(jnp.searchsorted(grid, t, side="right") - 1, 0, n - 2)
        dj = jnp.clip((t - grid[jlo]) / (grid[jlo + 1] - grid[jlo]), 0.0, 1.0)
        return (1.0 - dj) * self.templates[:, jlo] + dj * self.templates[:, jlo + 1]

    def rest_spectrum(self, theta, ctx):
        fagn = jnp.ravel(theta["fagn"])[0]
        return (fagn * ctx.L_bol_star) * self.template(theta["agn_tau"])
