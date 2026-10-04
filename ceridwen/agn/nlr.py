"""AGN narrow-line region from the Feltre, Charlot & Gutkin (2016) photoionisation grid.

The grid (``scripts/agn2/build_nlr_h5.py`` -> ``nlr_feltre16.h5``) tabulates the luminosities of
20 lines on the Cartesian product of five axes: gas metallicity Z, ionisation parameter at the
Stroemgren radius log U_S, dust-to-metal mass ratio xi_d, hydrogen density n_H and the UV slope
alpha of the accretion-disc spectrum (Feltre+16 sec. 2.1, eq. 4-5, Table 1).

Luminosities: ``L_line = agn_elum * R_line(theta)`` [L_sun per M_formed], ``R_line`` = L_line /
L_Hbeta of the grid, interpolated multilinearly at the theta point. ``agn_elum`` keeps the meaning
it has for the Richardson+14 'a42' lines (``ceridwen.agn.lines``): the H beta luminosity of the
narrow-line region per formed solar mass.

Interpolation (the convention of ``ceridwen.neb.NebularGridModel``): log10 of the ratios, linear in
each axis variable, the axes in the variable the source tabulates, logarithmic where the source is
log-spaced:

    nlr_logz   log10(Z / Z_sun), Z_sun = 0.01524 (Feltre+16 sec. 2.1, Bressan+12): solar-relative
               like every metallicity in CERIDWEN; the grid's Z is log-spaced
    nlr_logu   log10 U_S (tabulated in log)
    nlr_xid    xi_d (tabulated linearly: 0.1, 0.3, 0.5)
    nlr_lognh  log10(n_H / cm^-3) (tabulated in decades: 100, 1000, 10^4)
    nlr_alpha  alpha (tabulated linearly)

Interpolating log10 R is the same as interpolating log10 L of each line and of H beta and dividing,
so the ratios at a grid node are the table's to rounding, and the line set never goes negative.
Outside the grid the axis value is clamped to the edge (inside traced code nothing can raise); a
fixed value or prior outside it is refused at setup (``range_messages`` / ``check_prior_ranges``).

Line set (static): the grid lines that have an FSPS ``emlines_info.dat`` name (matched BY NAME by
the converter; a name the FSPS list in use does not carry is refused at setup, with the list).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from . import AGN2LineSet, AGN_THETA_DEFAULTS
from .lines import read_emlines_info

__all__ = ["NLR_FILE_NAME", "NLR_ZSUN", "NLR_AXES", "NLR_THETA_DEFAULTS", "default_nlr_file",
           "load_nlr_h5", "FeltreNLR", "multilinear"]

NLR_FILE_NAME = "nlr_feltre16.h5"
NLR_ZSUN = 0.01524          # Feltre+16 sec. 2.1: "present-age solar metallicity ... 0.01524"
NLR_MODES = ("corrected", "cigale")

# theta key -> (h5 axis dataset, transform from the stored axis to the interpolation variable)
NLR_AXES = (
    ("nlr_logz", "logz"),
    ("nlr_logu", "logu"),
    ("nlr_xid", "xid"),
    ("nlr_lognh", "lognh"),
    ("nlr_alpha", "alpha"),
)
NLR_THETA_DEFAULTS = {
    "nlr_logz": 0.0,
    "nlr_logu": -2.5,
    "nlr_xid": 0.3,
    "nlr_lognh": 3.0,
    "nlr_alpha": -1.7,
}


def default_nlr_file() -> Path:
    """``$CERIDWEN_AGN_DATA/nlr_feltre16.h5``, else ``~/.ceridwen/agn/nlr_feltre16.h5``."""
    base = os.environ.get("CERIDWEN_AGN_DATA")
    base = Path(base) if base else Path.home() / ".ceridwen" / "agn"
    return base / NLR_FILE_NAME


def load_nlr_h5(path) -> dict:
    """Arrays and attributes of an ``nlr_feltre16.h5`` (see ``scripts/agn2/build_nlr_h5.py``)."""
    import h5py
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feltre+16 NLR grid not found at {path}. Build it once with "
            "`python scripts/agn2/build_nlr_h5.py <dir with nlr_nebular_Z*.txt>` (writes "
            f"$CERIDWEN_AGN_DATA/{NLR_FILE_NAME}, default ~/.ceridwen/agn/), or pass "
            "AGNConfig(nlr_file=...)")
    with h5py.File(path, "r") as h:
        fmt = h.attrs.get("format", "")
        if fmt != "ceridwen nlr_feltre16 v1":
            raise ValueError(f"{path}: format {fmt!r}, expected 'ceridwen nlr_feltre16 v1'")
        out = {k: np.asarray(h["axes"][k][()], dtype=np.float64) for k in h["axes"]}
        out["ratio"] = np.asarray(h["ratio"][()], dtype=np.float64)
        out["source_name"] = tuple(s.decode() for s in h["lines/source_name"][()])
        out["fsps_name"] = tuple(s.decode() for s in h["lines/fsps_name"][()])
        out["attrs"] = {k: (v.decode() if isinstance(v, bytes) else v)
                        for k, v in h.attrs.items()}
    return out


def _cell(x, grid):
    """(index i, weight w) of ``x`` clamped to ``[grid[0], grid[-1]]`` in ``[grid[i], grid[i+1]]``
    (static ``grid``, ascending, >= 2 nodes). At an interior node ``x = grid[k]`` the cell is
    ``k-1`` with ``w = 1`` (``searchsorted`` side left), so the node value is reproduced."""
    g = jnp.asarray(grid)
    x = jnp.clip(x, grid[0], grid[-1])
    i = jnp.clip(jnp.searchsorted(g, x) - 1, 0, g.size - 2)
    w = (x - g[i]) / (g[i + 1] - g[i])
    return i, w


def multilinear(cube, axes, point):
    """Multilinear interpolation of ``cube`` (n_lines, n_0, ..., n_{d-1}) at ``point`` (d scalars,
    possibly traced) on the static ascending ``axes``; one axis is contracted at a time."""
    c = cube
    for grid, x in zip(axes, point):
        i, w = _cell(x, grid)
        lo = jnp.take(c, i, axis=1)
        hi = jnp.take(c, i + 1, axis=1)
        c = (1.0 - w) * lo + w * hi
    return c


class FeltreNLR(AGN2LineSet):
    """The Feltre+16 narrow-line region (static line set).

    Parameters
    ----------
    mode : "corrected" or "cigale" (the reproduction modes "fsps" / "prospector" have no such
        grid and are refused)
    file : the converted grid (default ``default_nlr_file()``)
    sps_home : FSPS directory holding ``data/emlines_info.dat`` (default ``$SPS_HOME``)

    Attributes
    ----------
    line_names : FSPS names of the painted lines; ``line_wave`` (n,) their vacuum rest wavelengths
    from the FSPS list in use; ``fsps_index`` (n,) their rows; ``dropped_lines``: grid lines with
    no FSPS line (not painted); ``axes`` : {theta key: axis nodes}; ``emlines_wave`` /
    ``emlines_names`` the whole FSPS list.
    """

    # the order of AGN2_THETA_KEYS["feltre16"] (ceridwen/agn/__init__.py)
    param_names = ("agn_elum", "agn_eline_sigma", "nlr_logu", "nlr_logz", "nlr_alpha",
                   "nlr_xid", "nlr_lognh")
    region = "nlr"
    width_key = "agn_eline_sigma"

    def __init__(self, *, mode, file=None, sps_home=None):
        if mode not in NLR_MODES:
            raise ValueError(
                f"FeltreNLR(mode={mode!r}): the Feltre+16 NLR grid exists only in agn_mode "
                f"{NLR_MODES}; FSPS and Prospector have no NLR grid (their AGN lines are the fixed "
                "Richardson+14 ratios, AGNConfig(nlr_model='a42')).")
        self.mode = mode
        self.path = Path(file) if file is not None else default_nlr_file()
        tab = load_nlr_h5(self.path)
        self.attrs = tab["attrs"]
        wave, names, epath = read_emlines_info(sps_home)
        self.emlines_file = str(epath)
        self.emlines_wave = wave
        self.emlines_names = names
        pos = {n: i for i, n in enumerate(names)}

        keep, dropped, unknown = [], [], []
        for k, (src, fname) in enumerate(zip(tab["source_name"], tab["fsps_name"])):
            if not fname:
                dropped.append(src)
            elif fname not in pos:
                unknown.append(f"{src!r} -> {fname!r}")
            else:
                keep.append(k)
        if unknown:
            raise ValueError(
                f"{self.path}: {len(unknown)} NLR line(s) have no line of that name in "
                f"{epath}: {', '.join(unknown)}. The grid's lines are matched to FSPS by name "
                f"(the list {self.path} was built against is "
                f"{self.attrs.get('fsps_emlines_file', '?')}); use that FSPS data directory or "
                "rebuild the grid against this one with scripts/agn2/build_nlr_h5.py.")
        src = list(tab["source_name"])
        if "Hbeta" not in src:
            raise ValueError(f"{self.path}: no H beta line; the ratios are normalised to it")
        self.line_names = tuple(tab["fsps_name"][k] for k in keep)
        self.source_names = tuple(src[k] for k in keep)
        self.dropped_lines = tuple(dropped)
        self.fsps_index = np.asarray([pos[n] for n in self.line_names], dtype=np.int64)
        self.line_wave = wave[self.fsps_index].copy()

        self.axes = {key: tab[ds] for key, ds in NLR_AXES}
        for key, g in self.axes.items():
            if g.ndim != 1 or g.size < 2 or np.any(np.diff(g) <= 0):
                raise ValueError(f"{self.path}: axis for {key} must be ascending with >= 2 nodes")
        if "logz" in tab and "Z" in tab:
            # stored as absolute log10 Z; the theta key is solar-relative
            self.axes["nlr_logz"] = tab["logz"] - np.log10(NLR_ZSUN)
        ratio = tab["ratio"][keep]
        expect = (len(keep),) + tuple(g.size for g in self.axes.values())
        if ratio.shape != expect:
            raise ValueError(f"{self.path}: ratio cube {ratio.shape}, axes give {expect}")
        if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
            raise ValueError(f"{self.path}: ratios must be finite and > 0")
        self.ratio_nodes = ratio
        self._log_ratio = jnp.asarray(np.log10(ratio))
        self._axes_static = tuple(np.asarray(g) for g in self.axes.values())

    # ---------------------------------------------------------------- static
    @property
    def ranges(self) -> dict:
        """{theta key: (lo, hi)} of the grid axes."""
        return {k: (float(g[0]), float(g[-1])) for k, g in self.axes.items()}

    def default_params(self) -> dict:
        out = {k: AGN_THETA_DEFAULTS[k] for k in ("agn_elum", "agn_eline_sigma")}
        out.update(NLR_THETA_DEFAULTS)
        return out

    def provenance(self) -> dict:
        """Files and hashes for the result file (the hook prefixes the keys with ``agn_nlr_``)."""
        import hashlib
        return {"file": str(self.path),
                "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "source_tarball_sha256": str(self.attrs.get("source_tarball_sha256", "")),
                "emlines_file": self.emlines_file,
                "lines": ",".join(self.line_names),
                "not_painted": ",".join(self.dropped_lines),
                "zsun": NLR_ZSUN}

    def range_messages(self, theta) -> list:
        """Messages for NLR keys of ``theta`` outside the grid (setup-time, NumPy)."""
        msgs = []
        for key, (lo, hi) in self.ranges.items():
            if key in theta:
                v = np.asarray(theta[key], float)
                if v.size and (np.nanmin(v) < lo - 1e-12 or np.nanmax(v) > hi + 1e-12):
                    msgs.append(
                        f"theta['{key}'] = {np.array2string(v, precision=3)} is outside the "
                        f"Feltre+16 NLR grid [{lo:+.4g}, {hi:+.4g}]; the interpolation clamps to "
                        "the edge (no extrapolation), so the lines freeze there. Give a value "
                        "inside the grid.")
        return msgs

    def check_theta(self, theta):
        """Raise ValueError for NLR keys of ``theta`` outside the grid (setup-time)."""
        msgs = self.range_messages(theta)
        if msgs:
            raise ValueError(" ".join(msgs))

    def check_prior_ranges(self, priors, prior_support):
        """Raise for a bounded prior on an NLR key whose support leaves the grid (the nebular
        rule of ``SedModel._check_gas_range``). ``prior_support(prior) -> (lo, hi)``."""
        for key, (lo, hi) in self.ranges.items():
            p = priors.get(key)
            if p is None:
                continue
            plo, phi = prior_support(p)
            if not (np.isfinite(plo) and np.isfinite(phi)):
                raise ValueError(
                    f"the prior on {key!r} is unbounded; the Feltre+16 NLR grid covers "
                    f"[{lo:+.4g}, {hi:+.4g}] and is not extrapolated. Use a bounded prior, e.g. "
                    f"Uniform(low={lo:+.4g}, high={hi:+.4g}).")
            if plo < lo - 1e-9 or phi > hi + 1e-9:
                raise ValueError(
                    f"the prior on {key!r} covers [{plo:+.4g}, {phi:+.4g}], outside the Feltre+16 "
                    f"NLR grid [{lo:+.4g}, {hi:+.4g}]; the interpolation clamps there, so the "
                    "lines freeze at the edge and the posterior piles up in a flat tail. Narrow "
                    f"the prior, e.g. Uniform(low={lo:+.4g}, high={hi:+.4g}).")

    # ---------------------------------------------------------------- traced
    @staticmethod
    def _scalar(theta, key):
        if key in theta:
            return jnp.ravel(jnp.asarray(theta[key], dtype=jnp.float64))[0]
        d = NLR_THETA_DEFAULTS.get(key, AGN_THETA_DEFAULTS.get(key))
        return jnp.asarray(d, dtype=jnp.float64)

    def ratios(self, theta):
        """(n_lines,) L_line / L_Hbeta at the theta point (float64)."""
        point = [self._scalar(theta, k) for k in self.axes]
        return jnp.power(10.0, multilinear(self._log_ratio, self._axes_static, point))

    def line_luminosity(self, theta):
        """(n_lines,) ``agn_elum * ratio`` [L_sun per M_formed], in the order of ``line_names``."""
        return self._scalar(theta, "agn_elum") * self.ratios(theta)

    def sigma_kms(self, theta):
        """``agn_eline_sigma`` [km/s] (scalar)."""
        return self._scalar(theta, "agn_eline_sigma")

    def __repr__(self):
        return (f"FeltreNLR(mode={self.mode!r}, {self.n_lines} lines "
                f"{self.line_wave.min():.0f}-{self.line_wave.max():.0f} A, "
                f"not painted: {list(self.dropped_lines)}, file={self.path})")
