"""Solar metallicity of every known SSP grid: the single source of truth for Z_sun.

CERIDWEN's stellar metallicity is ``logzsol = log10(Z / Z_sun)`` with Z_sun the solar
metallicity *of the grid in use*.  A grid's Z_sun is resolved at construction
(:class:`~ceridwen.ssps.ssp_data.SSPData`), in this order:

1. an explicit ``zsun=`` keyword,
2. the ``log10_zsun`` provenance attribute written by builds since v1.0.5,
3. :data:`CHASH_TABLE` below, keyed by the grid's content hash (``chash-v1``),
4. otherwise a teaching error.  Nothing is guessed from ``isoc_type``: the MIST Z_sun of
   FSPS changed 0.0142 -> 0.0191 -> 0.0185 between versions under the same name.

Every source that is present must give the same value bit for bit.  ``log10_zsun`` is the
grid's own solar-node value, so ``logzsol = 0`` lands on that node exactly.

``chash-v1``: SHA-256 over ``ssp_wave``, ``ssp_lg_age_gyr``, ``ssp_lgmet``, ``ssp_afe`` (if
the grid has one) and ``ssp_flux``, in that order.  Each array contributes the ASCII header
``"<name>|<dtype.str>|<shape>;"`` followed by its C-contiguous little-endian bytes.  HDF5
attributes are excluded, so a re-saved or converted file with the same arrays has the same
chash (the file sha256 of such a copy changes; A2 of the Phase A report).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

CHASH_VERSION = "chash-v1"

# [Z/H] - [Fe/H] for the MIST v2.5 / GS98 alpha mixture: f = log10(1 - x + x 10^[a/Fe]).
# Derived from the MESA input_XYZ of all 85 MIST v2.5 compositions (Dotter & Bauer 2025,
# zenodo 15232687, rotating_workdirs.tgz, 00095M_dir of each feh/afe directory) with
# [Z/H] = log10((Z/X)/(Z/X)_sun), (Z/X)_sun = 0.016357678062/0.71337734046: f is independent
# of [Fe/H] to 1.9e-11, f = -0.12709428, 0, 0.14678160, 0.30950399, 0.48422267 at
# [a/Fe] = -0.2 .. +0.6, and this closed form reproduces all 85 values to 1.7e-11 dex with
# x = 0.68749037712 (x = 0.687490 rounded to 6 digits would give 1.6e-7 dex).
# x is the alpha-element (O, Ne, Mg, Si, S, Ar, Ca, Ti) mass fraction of Z in that mixture.
X_ALPHA_MIST25 = 0.68749037712

AXIS_MEANINGS = ("feh", "log_z_over_zsun")


def f_alpha(afe, x: float = X_ALPHA_MIST25):
    """[Z/H] - [Fe/H] at [alpha/Fe] = ``afe`` (jnp- and numpy-compatible, differentiable)."""
    try:
        import jax.numpy as xp
    except ImportError:          # pragma: no cover
        xp = np
    return xp.log10(1.0 - x + x * 10.0 ** afe)


def logzsol_total(logzsol, afe, x: float = X_ALPHA_MIST25):
    """Total metallicity [Z/H] (Z/X definition) from ``logzsol`` = [Fe/H] on an alpha grid."""
    return logzsol + f_alpha(afe, x)


@dataclass(frozen=True)
class GridMeta:
    """Metallicity metadata of one grid (one chash).  ``refused_cells``: (iz, ia) node
    indices whose interpolation cell (z in (zmet[iz-1], zmet[iz]], afe in (afe[ia-1],
    afe[ia]]) must not be reached."""
    name: str
    log10_zsun: float
    zsun_nominal: float
    axis_meaning: str
    alpha_axis: bool
    isoc_type: str
    spec_library: str
    fsps_version: Optional[str]
    evidence: str
    refused_cells: tuple = ()
    refused_reason: str = ""
    file_sha256: tuple = field(default=())


_A3_CORNER = (
    "[Fe/H]=+0.5, [alpha/Fe]=+0.6 is built on a copy of the +0.4 isochrone: "
    "$SPS_HOME/ISOCHRONES/MIST/isoc_feh_p050_afe_p6 and _p4 share git blob "
    "4f54baed9c58 since 2ac6af2, MIST v2.5 dropped that model (Dotter+26 Sec. 2.1; its "
    "MESA input has Z = 0.110), and the grid's L_bol(+0.6)/L_bol(+0.4) at [Fe/H]=+0.5 "
    "is within 0.31% (HR) / 0.15% (LR) of 1 at every age >= 0.1 Gyr against 10-17% "
    "between real neighbouring models (Phase A, A3)")

# Values: log10_zsun is the float64 repr of the grid's solar node (Phase A, A1: FSPS-built
# MIST grids hold log10(float32 zsol); BPASS and the HR alpha grid log10 of the float64
# value).  zsun_nominal is for display only.
CHASH_TABLE: dict[str, GridMeta] = {}


def _register(chash: str, meta: GridMeta) -> None:
    if chash in CHASH_TABLE:
        raise RuntimeError(f"duplicate chash {chash}")
    CHASH_TABLE[chash] = meta


# ---------------------------------------------------------------------------------------
# Table entries.  Evidence (Phase A report, Claude outputs/metallicity_phase0_2026-09-22/):
#   A1 FSPS source: src/sps_vars.f90 `#elif (MIST)` zsol = 0.0142 (049875039497, 2016),
#      0.0191 (c8752a1d7951, 2026-07-24), 0.0185 (1c9d8763a4e9, 2026-07-28); BPASS
#      zsol = 0.020.  python-fsps 0.5.0 bundles 0.0185 (get_zsol() = 0.01850000023841858),
#      0.4.7 bundles 0.0142.  sps_setup.f90:141-145: MIST zlegend = 10**tag * zsol, tag =
#      the [Fe/H] of ISOCHRONES/MIST/zlegend.dat; other sets read linear Z.
#   Axis check: ssp_lgmet == log10(sp.zlegend) of the matching build to <= 8.9e-16; every
#      log10_zsun below is the exact solar node, and the node - log10_zsun list is the FSPS
#      tag list ([Fe/H]) or log10(Z/0.020) (BPASS).
# ---------------------------------------------------------------------------------------

_register("chash-v1:6049a6ea0487a96eba1e293f8cdf8a8de448aa3fc055be8307b12d3fcae48ae1", GridMeta(
    name="mist_miles_chab", log10_zsun=-1.73282826600002, zsun_nominal=0.0185,
    axis_meaning="feh", alpha_axis=False, isoc_type="mist", spec_library="miles",
    fsps_version="0.5.0",
    evidence="python-fsps 0.5.0 MIST+MILES: axis = log10(float32(0.0185) 10^[Fe/H]), "
             "[Fe/H] = -2.5..+0.5 in 0.25; node[10]",
    file_sha256=("d52f1940e4cfcf739a50e8afaea0389871bec9404653a7e023faa53f86382f31",
                 "56142e9cc9ec6362d927d1401e14af11980b1e9fd3d82ce43a45df1d7a8cfcd7",
                 "2f6777a848f211b7b93a50ff206466a3c2faab073e8acab09d4c0e2dcea378fb")))

_register("chash-v1:39200b1bbccb1144cfd3ce43d0ddc1fdf921c7f7a5434f674a60dbaa009ec114", GridMeta(
    name="mist_c3k_lr_chab", log10_zsun=-1.73282826600002, zsun_nominal=0.0185,
    axis_meaning="feh", alpha_axis=False, isoc_type="mist", spec_library="c3k_lr",
    fsps_version="0.5.0",
    evidence="python-fsps 0.5.0 MIST+C3K_LR (prospector env build): axis as mist_miles_chab",
    file_sha256=("3822ddda07ab9d0ee5e092d01fd70d7a3bed7d37954fd7402dc76008ec8588a6",
                 "e871b3ef2bb69a3ed70b7291e3bfa7af79ac80669e2c1337847374e8d2378aaf")))

_register("chash-v1:f1bab2d130c2b99bd7222ac72d09398d5639f614093a2d09d0fc381ed94ba2f1", GridMeta(
    name="mist_bpass_v2", log10_zsun=-1.6989700043360187, zsun_nominal=0.020,
    axis_meaning="log_z_over_zsun", alpha_axis=False, isoc_type="bpss",
    spec_library="bpass", fsps_version="0.5.0",
    evidence="FSPS BPASS zsol = 0.020; axis = log10 of ISOCHRONES/BPASS/zlegend.dat "
             "(1e-4 .. 0.04); node[9] = log10(0.020)",
    file_sha256=("64c93ea751133cf3d34f4f33222af767d029a69b9ac3549059568055a489b9e9",
                 "017d3311c2abf0f9fa8331fb4131ebb9a9b16bc7405a84585b6f8c41025a2f58",
                 "294bfbfdc5c6b04239fc508049c18ad530cc83ba7d60aac9727e9635ac508944",
                 "5034b12a1d92cd09896a99def7f60601186807429ab02b52a817302341fc808f",
                 "c119d19e6ade6f1de72ebf0a887e70e88330a1d36c6323b4d314d8baa952d151")))

_register("chash-v1:c5db3f9772f63ebe303df0553189efaa7556c48a6016693b8358d0d97492bfe0", GridMeta(
    name="bpass_agb_dust", log10_zsun=-1.6989700043360185, zsun_nominal=0.020,
    axis_meaning="log_z_over_zsun", alpha_axis=False, isoc_type="bpss",
    spec_library="bpass", fsps_version=None,
    evidence="BPASS axis (same 12 Z as mist_bpass_v2, node[9] 2.2e-16 from it); the file "
             "records no isochrone provenance, supplied here (tests/fixtures, git-tracked)",
    file_sha256=("685894502fe07cc3843a749d9c24975c3425a2e4324b31b581c8fa12fe59bf6a",
                 "d629489e14102b1acb427dea4cf114c65f9cbf6da59a723e35a64da641eafef7")))

_register("chash-v1:cffb378cc04f57c62504eea61ecef0c989f147c83efb40feef5d8b67f9825d11", GridMeta(
    name="mist_miles_fsps047", log10_zsun=-1.8477116527913624, zsun_nominal=0.0142,
    axis_meaning="feh", alpha_axis=False, isoc_type="mist", spec_library="miles",
    fsps_version="0.4.7",
    evidence="python-fsps 0.4.7 (MIST zsol 0.0142, pre-2ac6af2 12-tag zlegend without "
             "-2.25): axis - log10(float32(0.0142)) = the tags exactly; node[9]",
    file_sha256=("f315bf1dde3483df570efcead9448700a7909fa3a972a651a6806f3c3bb99b4b",
                 "57c54face30e78b3ecc25ac859fdc8578c36b886392fc47d9106398d94ed2854")))

_register("chash-v1:6f8492e9dc00118eaf7658f6647ec513c5ad07a72f2b7ea7c6d483a7a19c7640", GridMeta(
    name="amist_c3k_lr_chab_afe", log10_zsun=-1.7328282660000198, zsun_nominal=0.0185,
    axis_meaning="feh", alpha_axis=True, isoc_type="mist", spec_library="c3k_lr",
    fsps_version="0.4.9.dev16+gc18b87021",
    evidence="FSPS AFE_FLAG=1: node z of plane a loads isoc_feh_<tag>_afe_<a> and "
             "c3k_*_feh<tag>_afe<a> (sps_setup.f90:303-305,710-713), zlegend plane-"
             "independent, so the axis is [Fe/H] + log10(zsol) on every plane; node[10]",
    refused_cells=((12, 4),), refused_reason=_A3_CORNER,
    file_sha256=("0ae3ca192f1ba3a7825c83d77dd927ec069f4107a52051b2e4484f80d5a47ef7",
                 "47d192578b3e1e9c73e6b1c5807ff9c8b9247bf1a61f7d2f4765e4a8e8fbf1c2")))

_register("chash-v1:3b386b55520bbcd93dd6847a072ab71f9b1df48767d1a4277cc3b656a9985d74", GridMeta(
    name="amist_c3k_hr_krou_afe", log10_zsun=-1.7328282715969863, zsun_nominal=0.0185,
    axis_meaning="feh", alpha_axis=True, isoc_type="mist", spec_library="c3k_hr",
    fsps_version=None,
    evidence="M. J. Park FITS (ext2 column feh) + log10(0.0185) in float64 "
             "(scripts_afe/build_afe_hr_grid.py:129), bitwise; node[10]; Fe5270 moves "
             "-0.11 A and Mgb +2.3 A from [a/Fe] 0 to +0.4 at a fixed node, i.e. fixed [Fe/H]",
    refused_cells=((12, 4),), refused_reason=_A3_CORNER,
    file_sha256=("f6af03d813569f5982891d969f030d9345278a60de907b90b2a910d56af32a16",
                 "7a51fda352d8c2455ba58a8333e3e438b9e46043341d2b28f9ff963ac7b30833")))


FILE_SHA_ALIASES: dict[str, str] = {}      # file sha256 -> chash (published / known copies)


def _rebuild_aliases() -> None:
    FILE_SHA_ALIASES.clear()
    for ch, m in CHASH_TABLE.items():
        for s in m.file_sha256:
            FILE_SHA_ALIASES[s] = ch


_rebuild_aliases()


# ------------------------------- content hash --------------------------------------------

def _update(h, name: str, a) -> None:
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and not _LITTLE):
        a = a.astype(a.dtype.newbyteorder("<"))
    h.update(f"{name}|{a.dtype.str}|{tuple(int(s) for s in a.shape)};".encode("ascii"))
    h.update(a.tobytes(order="C"))


_LITTLE = np.little_endian


def chash_arrays(ssp_wave, ssp_lg_age_gyr, ssp_lgmet, ssp_flux, ssp_afe=None) -> str:
    """``chash-v1`` of a grid's arrays (see the module docstring)."""
    h = hashlib.sha256()
    _update(h, "ssp_wave", ssp_wave)
    _update(h, "ssp_lg_age_gyr", ssp_lg_age_gyr)
    _update(h, "ssp_lgmet", ssp_lgmet)
    if ssp_afe is not None:
        _update(h, "ssp_afe", ssp_afe)
    _update(h, "ssp_flux", ssp_flux)
    return f"{CHASH_VERSION}:{h.hexdigest()}"


def _cache_path() -> Optional[Path]:
    try:
        from .grid_fetch import grid_cache_dir
        return grid_cache_dir() / "chash_cache.json"
    except OSError:
        return None


def _cache_key(path: Path, extras: tuple = ()) -> str:
    """(path, size, mtime) plus the extra datasets that went INTO the hash: the same file
    hashed as a 3-D grid and as a 4-D alpha grid has two different chashes, and caching them
    under one key would hand the wrong one back (and hide the grid from the table)."""
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{'+'.join(extras)}"


def _read_cache() -> dict:
    p = _cache_path()
    if p is None or not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _write_cache(entries: dict) -> None:
    p = _cache_path()
    if p is None:
        return
    try:
        cur = _read_cache()
        cur.update(entries)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur, indent=0, sort_keys=True))
        tmp.replace(p)
    except OSError:
        pass


def cached_chash(path, extras: tuple = ()) -> Optional[str]:
    """chash of ``path`` from the (path, size, mtime, hashed datasets) cache, or None."""
    try:
        return _read_cache().get(_cache_key(Path(path), tuple(extras)))
    except OSError:
        return None


def remember_chash(path, chash: str, extras: tuple = ()) -> None:
    """Store ``chash`` for ``path`` at its current (size, mtime) and dataset set."""
    try:
        _write_cache({_cache_key(Path(path), tuple(extras)): chash})
    except OSError:
        pass


# ------------------------------- resolution ----------------------------------------------

def solar_node(ssp_lgmet, log10_zsun: float, *, tol: float = 1e-8) -> float:
    """The node of ``ssp_lgmet`` within ``tol`` of ``log10_zsun`` (its exact float value);
    ValueError if there is none."""
    ax = np.asarray(ssp_lgmet, dtype=np.float64)
    i = int(np.argmin(np.abs(ax - log10_zsun)))
    if abs(ax[i] - log10_zsun) > tol:
        raise ValueError(
            f"no metallicity node at log10(Z_sun) = {log10_zsun:.10f} (closest node "
            f"{ax[i]:.10f}; axis {np.array2string(ax, precision=6)}).  A grid's Z_sun must "
            "be one of its own nodes, so that logzsol = 0 is a grid point.")
    return float(ax[i])


def resolve_zsun(*, ssp_lgmet, zsun_kwarg=None, provenance_log10_zsun=None,
                 chash=None, describe="this grid"):
    """``(log10_zsun, sources, meta)``: the grid's solar-node value, the names of the sources
    that gave it, and its :class:`GridMeta` (None when not in the table).  Raises when no
    source exists, when sources disagree, or when the value is not a node of the axis."""
    ax = np.asarray(ssp_lgmet, dtype=np.float64)
    vals = {}
    if zsun_kwarg is not None:
        z = float(zsun_kwarg)
        if not (z > 0.0 and math.isfinite(z)):
            raise ValueError(f"zsun= must be a positive metal mass fraction, got {zsun_kwarg!r}")
        vals["zsun= keyword"] = solar_node(ax, math.log10(z))
    if provenance_log10_zsun is not None:
        v = float(provenance_log10_zsun)
        if not np.any(ax == v):
            raise ValueError(
                f"{describe}: its provenance log10_zsun = {v!r} is not a node of its "
                f"metallicity axis {np.array2string(ax, precision=10)}; the file is "
                "inconsistent (was ssp_lgmet edited?).")
        vals["log10_zsun provenance"] = v
    meta = CHASH_TABLE.get(chash) if chash else None
    if meta is not None:
        vals[f"chash table ({meta.name})"] = meta.log10_zsun
    if not vals:
        raise ValueError(
            f"{describe}: its solar metallicity Z_sun is unknown.  CERIDWEN's metallicity "
            "is logzsol = log10(Z/Z_sun) with the Z_sun of the grid itself, and this grid "
            "records none (no log10_zsun provenance) and is not in "
            "ceridwen.ssps.grid_metadata.CHASH_TABLE"
            + (f" (chash {chash})" if chash else "") + ".  Pass it explicitly, e.g. "
            "SSPData.load(path, zsun=0.020) for a BPASS grid, zsun=0.0185 for FSPS >= "
            "2026-07 MIST/C3K, zsun=0.0142 for FSPS <= 0.4.7 MIST; the value must be one "
            "of the grid's nodes: 10**ssp_lgmet = "
            f"{np.array2string(10.0 ** ax, precision=6)}.  Never guess it from isoc_type.")
    uniq = {np.float64(v).tobytes() for v in vals.values()}
    if len(uniq) > 1:
        lines = "\n".join(f"    {k:32s}: log10_zsun = {v!r}  (Z_sun = {10.0 ** v:.10g})"
                          for k, v in vals.items())
        raise ValueError(f"{describe}: the Z_sun sources disagree:\n{lines}\n"
                         "Remove the wrong one (a zsun= that contradicts the grid's recorded "
                         "solar node is refused rather than silently preferred).")
    log10_zsun = next(iter(vals.values()))
    if not np.any(ax == log10_zsun):                      # pragma: no cover (guarded above)
        raise ValueError(f"{describe}: log10_zsun {log10_zsun!r} is not a grid node")
    return float(log10_zsun), tuple(vals), meta


DEFAULT_REFUSED_REASON = (
    "this (metallicity, [alpha/Fe]) cell is built on a duplicated isochrone file "
    "(see ceridwen.ssps.grid_metadata / SSPDataAfe.from_fsps)")


def refused_cell_bounds(refused_cells, reason, zmet_logzsol, afe_grid):
    """[(z_lo, z_hi, a_lo, a_hi, reason)] for each refused (iz, ia) cell, in logzsol."""
    if not refused_cells or afe_grid is None:
        return []
    z = np.asarray(zmet_logzsol, dtype=np.float64)
    a = np.asarray(afe_grid, dtype=np.float64)
    return [(float(z[iz - 1]), float(z[iz]), float(a[ia - 1]), float(a[ia]),
             reason or DEFAULT_REFUSED_REASON)
            for iz, ia in refused_cells]
