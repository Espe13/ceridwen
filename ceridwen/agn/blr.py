"""AGN broad-line region: a fixed set of broad H I lines with their own amplitude and width.

``AGNConfig(blr=True)`` (modes "corrected" and "cigale") switches on ``BroadLines``, an
``AGN2LineSet`` (``ceridwen/agn/__init__.py``) that the CSP paints separately from the narrow
lines, with its own width and its own transmission.

Line set and ratios (PLAN_AGN2.md section 1.2): the table ``data/blr_caseB_T1e4_ne1e4.csv``, H I
Case B recombination (Storey & Hummer 1995 via PyNeb 1.1.32, T = 1e4 K, n_e = 1e4 cm^-3), eight
lines stored by FSPS NAME with their luminosity relative to H alpha (H alpha / H beta = 2.847).
``AGNConfig(blr_file=...)`` takes another table of the same layout (columns
``name,fsps_index,wave_vac,ratio_halpha``, a row ``Ba-alpha 6563`` with ratio exactly 1).

theta keys

    blr_lum    0       L_Halpha,BLR [L_sun] per M_formed, BEFORE any attenuation; every line is
                       ``blr_lum * ratio_halpha``. Linear in it.
    blr_sigma  1500    Gaussian dispersion SIGMA [km/s] of every broad line (NOT the FWHM:
                       FWHM = 2 sqrt(2 ln 2) sigma = 2.35482 sigma; CIGALE's ``lines_width_BLR``
                       is a FWHM, so its default 800 km/s is blr_sigma = 339.7). Gaussian in
                       velocity; the painter adds the instrument in quadrature,
                       ``sqrt(blr_sigma^2 + sigma_inst^2)``. Prior support [400, 5000] km/s.

``blr_sigma`` does not depend on ``Kinematics.sigma_max``: the AGN line painters
(``broadening.make_line_painter_free_z`` for a Spectrum, ``broadening.agn_line_profiles_on_grid``
for Photometry) evaluate the Gaussian on every pixel, with no static window and no clip.

Transmission (``AGNHook.line_set_transmission``): "corrected" T_pol (or the dust4 power law) x
T_gal at each line; "cigale" T_pol at the line (CIGALE nebular_AGN.py:339-344). In "cigale" mode
the amplitude is still ``blr_lum`` with the Case-B ratios: CIGALE's own BLR scaling (ionising
photons x f_BLR x Cloudy tables) lives on an unreleased branch and is not reproduced.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from . import AGN_MODE_CIGALE, AGN_MODE_CORRECTED, AGN2LineSet
from .lines import read_emlines_info

__all__ = ["BLR_FILE", "BLR_THETA_DEFAULTS", "BLR_SIGMA_SUPPORT", "HALPHA_NAME",
           "load_blr_table", "BroadLines"]

BLR_FILE = Path(__file__).resolve().parent / "data" / "blr_caseB_T1e4_ne1e4.csv"
HALPHA_NAME = "Ba-alpha 6563"
BLR_THETA_DEFAULTS = {"blr_lum": 0.0, "blr_sigma": 1500.0}
# prior support of blr_sigma [km/s] (AGN2_BRIEF); outside it range_messages warns at setup
BLR_SIGMA_SUPPORT = (400.0, 5000.0)
_HEADER = ["name", "fsps_index", "wave_vac", "ratio_halpha"]


def load_blr_table(path=None) -> dict:
    """``{"names", "fsps_index", "wave_vac", "ratio", "path"}`` of a BLR ratio table (comment lines
    start with '#'). Refuses a table that would make ``blr_lum`` mean something other than
    L_Halpha: no ``Ba-alpha 6563`` row, its ratio not exactly 1, a duplicate name, or a ratio that
    is not finite and > 0."""
    path = BLR_FILE if path is None else Path(path)
    if not path.is_file():
        raise ValueError(f"BLR line table not found: {path} (AGNConfig(blr_file=...); the "
                         f"default is {BLR_FILE})")
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#"))
                if r]
    if not rows or rows[0] != _HEADER:
        raise ValueError(f"{path}: header must be {_HEADER}, got {rows[0] if rows else None}")
    body = rows[1:]
    if not body:
        raise ValueError(f"{path}: no lines")
    try:
        names = tuple(r[0].strip() for r in body)
        idx = np.asarray([int(r[1]) for r in body], dtype=np.int64)
        wave = np.asarray([float(r[2]) for r in body], dtype=np.float64)
        ratio = np.asarray([float(r[3]) for r in body], dtype=np.float64)
    except (IndexError, ValueError) as e:
        raise ValueError(f"{path}: every row needs {_HEADER} ({e})") from None
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate lines {dup}; each broad line is one row")
    if HALPHA_NAME not in names:
        raise ValueError(
            f"{path}: no {HALPHA_NAME!r} row. blr_lum is the broad H alpha luminosity and every "
            "ratio is relative to it, so the table must contain H alpha with ratio 1")
    if ratio[names.index(HALPHA_NAME)] != 1.0:
        raise ValueError(
            f"{path}: the {HALPHA_NAME!r} ratio is {ratio[names.index(HALPHA_NAME)]!r}, not 1. "
            "Ratios are L_line / L_Halpha (column ratio_halpha); renormalise the table")
    bad = [n for n, r in zip(names, ratio) if not (np.isfinite(r) and r > 0.0)]
    if bad:
        raise ValueError(f"{path}: ratios must be finite and > 0; bad rows {bad}")
    return {"names": names, "fsps_index": idx, "wave_vac": wave, "ratio": ratio, "path": path}


class BroadLines(AGN2LineSet):
    """The BLR line set (static: the rows of the table, fixed at setup).

    Parameters
    ----------
    mode : "corrected" or "cigale" (the hook applies the mode's transmission)
    file : the ratio table (``AGNConfig.blr_file``; None = ``BLR_FILE``)
    sps_home : FSPS directory holding ``data/emlines_info.dat`` (default ``$SPS_HOME``)

    Every table line is matched to the FSPS list BY NAME and its wavelength must agree with that
    list within 1 A (else ValueError); ``line_wave`` is taken from the list in use, as for the
    narrow lines, so a Lines observation and the nebular lines sum by the same name.
    """

    region = "blr"
    width_key = "blr_sigma"
    param_names = ("blr_lum", "blr_sigma")

    def __init__(self, *, mode, file=None, sps_home=None):
        if mode not in (AGN_MODE_CORRECTED, AGN_MODE_CIGALE):
            raise ValueError(
                f"BroadLines(mode={mode!r}): the broad-line region exists only in "
                f"agn_mode='corrected' and 'cigale' (FSPS and Prospector have none)")
        self.mode = mode
        tab = load_blr_table(file)
        self.table_file = str(tab["path"])
        wave, names, path = read_emlines_info(sps_home)
        self.emlines_file = str(path)
        pos = {n: i for i, n in enumerate(names)}
        bad, rows = [], []
        for n, w in zip(tab["names"], tab["wave_vac"]):
            i = pos.get(n)
            if i is None:
                bad.append(f"{n!r} is not in the FSPS list")
            elif abs(wave[i] - w) > 1.0:
                bad.append(f"{n!r}: table {w:.2f} A, FSPS {wave[i]:.2f} A")
            else:
                rows.append(i)
        if bad:
            raise ValueError(
                f"BLR table {self.table_file} does not match the FSPS line list {path}: "
                + "; ".join(bad) + ". The broad lines are stored by FSPS name; use names from "
                "that list (or the FSPS data directory the table was written from)")
        self.line_names = tab["names"]
        self.fsps_index = np.asarray(rows, dtype=np.int64)
        self.line_wave = wave[self.fsps_index].copy()
        self.ratios = tab["ratio"]
        self.emlines_wave = wave
        self.emlines_names = names
        self._ratios_j = jnp.asarray(self.ratios)

    def default_params(self) -> dict:
        return dict(BLR_THETA_DEFAULTS)

    @staticmethod
    def _scalar(theta, key):
        if key in theta:
            return jnp.ravel(jnp.asarray(theta[key]))[0]
        return jnp.asarray(BLR_THETA_DEFAULTS[key], dtype=jnp.float64)

    def line_luminosity(self, theta):
        """(n_lines,) ``blr_lum * ratio_halpha`` [L_sun per M_formed], in the order of
        ``line_names``, before any attenuation."""
        return self._scalar(theta, "blr_lum") * self._ratios_j

    def sigma_kms(self, theta):
        """``blr_sigma`` [km/s] (scalar Gaussian sigma, not FWHM)."""
        return self._scalar(theta, "blr_sigma")

    def provenance(self) -> dict:
        return {"table": self.table_file,
                "table_sha256": hashlib.sha256(Path(self.table_file).read_bytes()).hexdigest(),
                "emlines_file": self.emlines_file,
                "lines": ",".join(self.line_names),
                "width": "blr_sigma = Gaussian sigma [km/s] (FWHM = 2.35482 sigma)"}

    def range_messages(self, theta) -> list:
        """Setup-time warnings for ``blr_sigma`` (``blr_lum`` < 0 is warned by the hook)."""
        msgs = []
        if "blr_sigma" in theta:
            v = np.asarray(theta["blr_sigma"], float)
            lo, hi = BLR_SIGMA_SUPPORT
            if v.size and (not np.all(np.isfinite(v)) or np.nanmin(v) <= 0.0):
                msgs.append(f"theta['blr_sigma'] = {np.array2string(v, precision=3)}: the broad "
                            "line sigma must be finite and > 0 km/s.")
            elif v.size and (np.nanmin(v) < lo or np.nanmax(v) > hi):
                msgs.append(f"theta['blr_sigma'] = {np.array2string(v, precision=3)} km/s is "
                            f"outside the BLR support [{lo:g}, {hi:g}] km/s. blr_sigma is a "
                            "Gaussian SIGMA; a FWHM must be divided by 2.35482.")
        return msgs

    def __repr__(self):
        return (f"BroadLines(mode={self.mode!r}, {self.n_lines} lines, "
                f"{self.line_wave.min():.0f}-{self.line_wave.max():.0f} A, "
                f"table={self.table_file})")
