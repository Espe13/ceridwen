"""AGN narrow emission lines: Prospector's ``AGNSpecModel`` line set, stored by FSPS line name.

Prospector (``prospect/models/sedmodel.py``, ``AGNSpecModel.init_aline_info``) puts 23 line
ratios relative to H beta (Richardson et al. 2014, 'a42') on fixed indices of FSPS's
``$SPS_HOME/data/emlines_info.dat`` and asserts that index 59 is H beta. Here the same ratios are
stored by FSPS line NAME (``data/richardson14_a42.csv``); at setup the FSPS list in use is read
and a list that does not carry those names at those indices is refused, so a ratio can never land
on a different line than it does in Prospector.

Luminosities: ``L_line = agn_elum * ratio`` [L_sun per M_formed] (Prospector's unit: ``agn_elum``
is the H beta luminosity per unit formed mass, multiplied by the mass in ``flux_norm``). The CSP
scales them by ``10**logmass`` and the flux factor like every nebular line.

Widths: one Gaussian dispersion ``agn_eline_sigma`` [km/s] for every AGN line. How the lines are
painted is decided by the mode (``ceridwen.broadening``):

``"prospector"`` / ``"fsps"`` (``AGNSpecModel`` runs on top of FSPS in both)
    Spectrum: Prospector's own painter (``AGNSpecModel.predict_spec``/``predict_aline_spec``):
    Gaussians in velocity of width ``agn_eline_sigma`` alone (no instrument), added after the
    instrumental smoothing, on the pixels within ``nsigma = 5 max(agn_eline_sigma / eline_sigma)``
    widths of a nebular line only, each normalised by a trapezoid in frequency over those pixels
    (``broadening.make_agn_line_painter_prospector``). Photometry: the line flux at the observed
    line centre through the filter transmission (``line_photometry_delta``; Prospector's
    ``nebline_photometry``). No dust, no IGM on the AGN lines.
``"corrected"``
    Spectrum: the analytic painter of the nebular lines with ``agn_eline_sigma`` in place of
    ``sigma_gas`` (``sqrt(agn_eline_sigma^2 + sigma_inst^2)``). Photometry: Gaussian profiles on the
    model grid (``broadening.agn_line_profiles_on_grid``), the nebular convention.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from . import AGN_MODES, AGN_THETA_DEFAULTS, LineComponent

__all__ = ["A42_FILE", "HBETA_INDEX", "HBETA_NAME", "load_a42_table", "read_emlines_info",
           "AGNLines", "line_photometry_delta"]

A42_FILE = Path(__file__).resolve().parent / "data" / "richardson14_a42.csv"
HBETA_INDEX = 59            # AGNSpecModel.init_aline_info: "index=59 is Hbeta"
HBETA_NAME = "Ba-beta 4861"


def load_a42_table(path=None) -> dict:
    """``{"names", "fsps_index", "wave_vac", "ratio"}`` of the AGN line table (comment lines
    start with '#')."""
    path = A42_FILE if path is None else Path(path)
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#"))]
    head, body = rows[0], rows[1:]
    if head != ["name", "fsps_index", "wave_vac", "ratio_hbeta"]:
        raise ValueError(f"{path}: unexpected header {head}")
    return {
        "names": tuple(r[0] for r in body),
        "fsps_index": np.asarray([int(r[1]) for r in body], dtype=np.int64),
        "wave_vac": np.asarray([float(r[2]) for r in body], dtype=np.float64),
        "ratio": np.asarray([float(r[3]) for r in body], dtype=np.float64),
    }


def read_emlines_info(sps_home=None):
    """``(wave (n,), names tuple)`` of ``<sps_home>/data/emlines_info.dat`` (Prospector reads it
    with ``np.genfromtxt(..., delimiter=',')``: wavelength, name)."""
    sps_home = sps_home or os.environ.get("SPS_HOME")
    if not sps_home:
        raise ValueError(
            "AGN lines read FSPS's line list from $SPS_HOME/data/emlines_info.dat, as Prospector "
            "does; set $SPS_HOME to your FSPS directory or pass sps_home=... to the basis")
    path = Path(sps_home) / "data" / "emlines_info.dat"
    if not path.is_file():
        raise ValueError(f"AGN lines need FSPS's line list, which was not found at {path}")
    wave, names = [], []
    with open(path) as f:
        for row in f:
            parts = row.split(",", 1)
            if len(parts) == 2:
                wave.append(float(parts[0]))
                names.append(parts[1].strip())
    return np.asarray(wave, dtype=np.float64), tuple(names), path


class AGNLines(LineComponent):
    """The AGN narrow-line set (static: 23 lines, fixed at setup).

    Parameters
    ----------
    mode : one of ``AGN_MODES`` -- selects the painter (see the module docstring)
    sps_home : FSPS directory holding ``data/emlines_info.dat`` (default ``$SPS_HOME``)
    table_file : the ratio table (default ``data/richardson14_a42.csv``)

    Attributes
    ----------
    line_names : tuple of FSPS names; ``line_wave`` (n,) vacuum rest wavelengths [A] from the
    FSPS list in use; ``ratios`` (n,) L_line / L_Hbeta; ``fsps_index`` (n,) the rows of that list;
    ``emlines_wave`` / ``emlines_names``: the whole FSPS list (for matching observed lines by name).
    """

    param_names = ("agn_elum", "agn_eline_sigma")

    def __init__(self, *, mode, sps_home=None, table_file=None):
        if mode not in AGN_MODES:
            raise ValueError(f"AGNLines(mode={mode!r}): choose one of {AGN_MODES}")
        self.mode = mode
        tab = load_a42_table(table_file)
        wave, names, path = read_emlines_info(sps_home)
        self.emlines_file = str(path)
        if wave.size <= HBETA_INDEX or abs(wave[HBETA_INDEX] - 4863.0) >= 2.0:
            raise ValueError(
                f"{path}: line {HBETA_INDEX} is not H beta "
                f"({'missing' if wave.size <= HBETA_INDEX else f'{wave[HBETA_INDEX]:.2f} A'}). "
                "Prospector's AGN line ratios sit on fixed indices of FSPS's line list and assert "
                "that index 59 is H beta; this $SPS_HOME has a different list, so the AGN lines "
                "would land on the wrong lines. Use the FSPS version the table was built from "
                "(see the header of ceridwen/agn/data/richardson14_a42.csv)")
        bad = []
        for n, i, w in zip(tab["names"], tab["fsps_index"], tab["wave_vac"]):
            if i >= wave.size or names[i] != n or abs(wave[i] - w) > 1.0:
                got = "nothing" if i >= wave.size else f"{names[i]!r} at {wave[i]:.2f} A"
                bad.append(f"index {i}: table {n!r} at {w:.2f} A, FSPS {got}")
        if bad:
            raise ValueError(
                f"{path} differs from the FSPS line list the AGN line table was built from "
                f"({len(bad)} of {len(tab['names'])} lines): " + "; ".join(bad) + ". Prospector "
                "applies its ratios by index, so with this list it would put them on other "
                "lines; CERIDWEN refuses instead. Use the matching FSPS data directory")
        self.line_names = tab["names"]
        self.fsps_index = tab["fsps_index"]
        self.line_wave = wave[tab["fsps_index"]].copy()
        self.ratios = tab["ratio"]
        self.emlines_wave = wave
        self.emlines_names = names
        self._ratios_j = jnp.asarray(self.ratios)

    @property
    def n_lines(self) -> int:
        return len(self.line_names)

    def default_params(self) -> dict:
        return {k: AGN_THETA_DEFAULTS[k] for k in self.param_names}

    @staticmethod
    def _scalar(theta, key):
        if key in theta:
            return jnp.ravel(jnp.asarray(theta[key]))[0]
        return jnp.asarray(AGN_THETA_DEFAULTS[key], dtype=jnp.float64)

    def line_luminosity(self, theta):
        """(n_lines,) ``agn_elum * ratio`` [L_sun per M_formed], in the order of ``line_names``."""
        return self._scalar(theta, "agn_elum") * self._ratios_j

    def sigma_kms(self, theta):
        """``agn_eline_sigma`` [km/s] (scalar)."""
        return self._scalar(theta, "agn_eline_sigma")

    def rows_for(self, wave_rest, tol=1.0) -> np.ndarray:
        """For each rest wavelength in ``wave_rest``: the index of the AGN line it is, or -1.

        Matched BY NAME: each wavelength is first resolved to the nearest line of the FSPS list
        (within ``tol`` A, else ValueError, the tolerance of the nebular cube matching), and that
        line's name is looked up in ``line_names``. So two FSPS lines closer than ``tol`` (He I 3889
        and H 8 3890) are never confused."""
        lam = np.atleast_1d(np.asarray(wave_rest, dtype=np.float64))
        out = np.full(lam.size, -1, dtype=np.int64)
        pos = {n: k for k, n in enumerate(self.line_names)}
        for i, l in enumerate(lam):
            j = int(np.argmin(np.abs(self.emlines_wave - l)))
            if abs(self.emlines_wave[j] - l) > tol:
                raise ValueError(
                    f"line at {l:.2f} A has no FSPS line within {tol:g} A (nearest "
                    f"{self.emlines_names[j]!r} at {self.emlines_wave[j]:.2f} A in "
                    f"{self.emlines_file})")
            out[i] = pos.get(self.emlines_names[j], -1)
        return out

    def observed_fluxes(self, theta, flux_factor_over_opz, *, mass=None, transmission=None):
        """(n_lines,) observed integrated fluxes [erg/s/cm^2]: ``line_luminosity * mass *
        flux_factor_over_opz`` times ``transmission`` (n_lines,) when given. ``mass`` defaults to
        ``10**theta['logmass']`` (1 without it); ``flux_factor_over_opz`` is the CSP's
        ``_flux_factor(theta) / (1 + z)`` (L_sun -> erg/s/cm^2), the factor its nebular lines use."""
        L = self.line_luminosity(theta)
        if mass is None:
            mass = 10.0 ** jnp.ravel(theta["logmass"])[0] if "logmass" in theta else 1.0
        F = L * mass * flux_factor_over_opz
        if transmission is not None:
            F = F * transmission
        return F

    def __repr__(self):
        return (f"AGNLines(mode={self.mode!r}, {self.n_lines} lines, "
                f"{self.line_wave.min():.0f}-{self.line_wave.max():.0f} A, "
                f"list={self.emlines_file})")


def line_photometry_delta(filters, line_wave_obs, line_flux):
    """(n_filters,) maggies of unresolved lines: ``sum_l T(lambda_l) lambda_l F_l /
    ab_zero_counts``, ``T`` linearly interpolated on the filter's own curve (0 outside).

    Prospector's ``SpecModel.nebline_photometry`` (used by ``AGNSpecModel.predict_phot`` for the
    AGN lines). ``filters``: ``Filter`` objects (``wavelength``, ``transmission``,
    ``ab_zero_counts``); ``line_wave_obs`` (n,) observed-frame [A] (may be traced);
    ``line_flux`` (n,) [erg/s/cm^2]."""
    lam = jnp.asarray(line_wave_obs)
    out = []
    for f in filters:
        T = jnp.interp(lam, jnp.asarray(np.asarray(f.wavelength, dtype=np.float64)),
                       jnp.asarray(np.asarray(f.transmission, dtype=np.float64)),
                       left=0.0, right=0.0)
        out.append(jnp.sum(T * lam * line_flux) / float(f.ab_zero_counts))
    return jnp.stack(out) if out else jnp.zeros(0)
