"""
Alpha-enhanced SSP data container: :class:`SSPDataAfe`.

This is the [alpha/Fe]-aware sibling of :mod:`ceridwen.ssps.ssp_data`.
The grid gains one leading axis:

    ssp_flux : (n_afe, n_met, n_ages, n_wave)      [Lsun / Hz / Msun]
    ssp_afe  : (n_afe,)                            [alpha/Fe] values

Everything else (units, conventions, provenance philosophy, the FSPS
kwarg whitelist) is inherited unchanged from ``ssp_data.py``:
``ssp_lgmet`` remains log10 of the *absolute total* metallicity Z — the
alpha axis re-partitions that Z between the Fe-peak and alpha elements
at fixed total Z (the aMIST/C3K convention), it does not change it.

Requirements
------------
Building (``from_fsps``) requires python-fsps compiled from main
(post 2026-08-02, the alpha-MC merge targeting FSPS v4.0) with
``AFE_FLAG=1``, MIST isochrones and a C3K spectral library, and an FSPS
v4.0 ``$SPS_HOME`` data tree containing the aMIST/C3K alpha files.  The
FSPS grid is nafe = 5 at [alpha/Fe] = {-0.2, 0.0, +0.2, +0.4, +0.6}.
An ``AFE_FLAG=0`` build yields n_afe = 1 and produces a valid
(single-plane) grid — the correct null model for isolating alpha
effects from the C3K-vs-MILES library change.

Loading (``load``) has no FSPS dependency, as usual.

Nebular caveat
--------------
FSPS v4.0 ships NO alpha-enhanced CLOUDY nebular tables (the ``ZAU_*``
grids are solar-scaled, keyed only by isochrone type).  Grids built here
are therefore consumed by :class:`ceridwen.csp.csp_afe.CSPBasis_afe`,
which carries no nebular model.
"""

import json
import typing
import h5py
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional

# Re-use the parent module's kwarg policy and constants so the two grid
# builders can never drift apart.
from ceridwen.ssps.ssp_data import (
    LIBRARY_IMF_KWARGS,
    _validate_fsps_kwargs,
    _read_fsps_provenance,
    HAS_FSPS,
)

if HAS_FSPS:
    import fsps

# Bumped whenever the on-disk metadata schema changes.  "2.0" adds the
# ssp_afe dataset and the 4-D ssp_flux layout.
SSP_AFE_SCHEMA_VERSION = "2.0"

# The FSPS v4.0 aMIST/C3K [alpha/Fe] grid (sps_vars.f90 ``afe_val``;
# ``afe_sol_indx = 2``, i.e. the second, 1-based, entry is solar-scaled).
# python-fsps does not (yet) expose afe_val, so nafe == 5 is mapped onto
# these documented values; any other nafe (except 1) must be supplied
# explicitly via ``afe_values=``.
FSPS_AFE_VALUES_NAFE5 = np.array([-0.2, 0.0, +0.2, +0.4, +0.6])


@dataclass(frozen=True)
class SSPDataAfe:
    """
    Immutable container for alpha-enhanced SSP interpolation grids.

    Attributes
    ----------
    ssp_lgmet : jnp.ndarray, shape (n_met,)
        ``log10`` of the absolute TOTAL metallicity grid (mass fraction of
        all elements heavier than He) — same convention as
        :class:`ceridwen.ssps.ssp_data.SSPData`, NOT [Fe/H].
    ssp_afe : jnp.ndarray, shape (n_afe,)
        [alpha/Fe] grid, strictly increasing.  aMIST/C3K: -0.2 .. +0.6
        in steps of 0.2.
    ssp_lg_age_gyr : jnp.ndarray, shape (n_ages,)
        ``log10(age / Gyr)``.
    ssp_wave : jnp.ndarray, shape (n_wave,)
        Wavelength grid in Angstroms.  NB: for alpha grids this is the
        C3K wavelength sampling, NOT the MILES sampling of older ceridwen
        grids — downstream LSF assumptions must match.
    ssp_flux : jnp.ndarray, shape (n_afe, n_met, n_ages, n_wave)
        SSP flux density in ``Lsun / Hz`` per Msun of initial stellar
        mass, with the [alpha/Fe] axis LEADING.

    Provenance fields are identical in meaning to ``SSPData``.
    """

    ssp_lgmet: jnp.ndarray          # (n_met,) log10 absolute total Z
    ssp_afe: jnp.ndarray            # (n_afe,) [alpha/Fe]
    ssp_lg_age_gyr: jnp.ndarray     # (n_ages,) log10(age / Gyr)
    ssp_wave: jnp.ndarray           # (n_wave,) Angstrom
    ssp_flux: jnp.ndarray           # (n_afe, n_met, n_ages, n_wave)

    # --- provenance (static Python metadata; not array leaves) ------------
    isoc_type: Optional[str] = field(default=None, compare=False)
    spec_library: Optional[str] = field(default=None, compare=False)
    imf_type: Optional[int] = field(default=None, compare=False)
    fsps_version: Optional[str] = field(default=None, compare=False)
    fsps_kwargs: dict = field(default_factory=dict, compare=False)
    wave_min: Optional[float] = field(default=None, compare=False)
    wave_max: Optional[float] = field(default=None, compare=False)
    schema_version: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        """Validate grid consistency (4-D layout, monotone afe axis)."""
        expected = (self.ssp_afe.size, self.ssp_lgmet.size,
                    self.ssp_lg_age_gyr.size, self.ssp_wave.size)
        if self.ssp_flux.shape != expected:
            raise ValueError(
                f"SSPDataAfe flux grid shape mismatch: expected {expected} "
                f"(n_afe, n_met, n_ages, n_wave) but got "
                f"{self.ssp_flux.shape}."
            )
        afe = np.asarray(self.ssp_afe, dtype=float)
        if afe.size > 1 and not np.all(np.diff(afe) > 0):
            raise ValueError(
                f"ssp_afe must be strictly increasing (required by the "
                f"searchsorted interpolation in CSPBasis_afe); got "
                f"{afe.tolist()}."
            )

    @property
    def n_afe(self) -> int:
        return int(self.ssp_afe.size)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    def display(self, *, return_str: bool = False, file=None):
        """Print a summary of the grid and its provenance."""
        import sys as _sys

        lgmet = np.asarray(self.ssp_lgmet)
        afe   = np.asarray(self.ssp_afe)
        lgage = np.asarray(self.ssp_lg_age_gyr)
        wave  = np.asarray(self.ssp_wave)
        n_afe, n_met, n_age, n_wave = self.ssp_flux.shape
        age_gyr = 10.0 ** lgage

        def _fmt(v, na="—"):
            return na if v is None else str(v)

        size = float(np.asarray(self.ssp_flux).nbytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                size_str = f"{size:.1f} {unit}"
                break
            size /= 1024.0

        lines = [
            "SSPDataAfe",
            "-" * 66,
            "provenance",
            f"  isochrones (isoc_type)   : {_fmt(self.isoc_type)}",
            f"  spectral library         : {_fmt(self.spec_library)}",
            f"  IMF (imf_type)           : {_fmt(self.imf_type)}",
            f"  FSPS version             : {_fmt(self.fsps_version)}",
            f"  schema version           : {_fmt(self.schema_version)}",
            f"  recorded wave_min/max    : {_fmt(self.wave_min)} / {_fmt(self.wave_max)}",
            f"  build kwargs             : {self.fsps_kwargs or '{}'}",
            "grids",
            f"  [alpha/Fe]               : {n_afe:>4d} pts   "
            f"{np.array2string(afe, precision=2)}",
            f"  metallicity  log10 Z     : {n_met:>4d} pts   "
            f"[{lgmet.min():+.3f}, {lgmet.max():+.3f}]  "
            f"(absolute TOTAL Z, NOT Z/Zsun, NOT [Fe/H])",
            f"  age          log10(Gyr)  : {n_age:>4d} pts   "
            f"[{lgage.min():+.3f}, {lgage.max():+.3f}]  "
            f"= [{age_gyr.min():.3g}, {age_gyr.max():.3g}] Gyr",
            f"  wavelength   Angstrom    : {n_wave:>4d} pts   "
            f"[{wave.min():.1f}, {wave.max():.1f}]",
            f"  flux (n_afe,n_met,n_age,n_wave): "
            f"{tuple(int(s) for s in self.ssp_flux.shape)}  "
            f"[L_sun Hz^-1 M_sun^-1]  {np.asarray(self.ssp_flux).dtype}  "
            f"{size_str}",
        ]
        if n_afe == 1:
            lines += [
                "note",
                "  single [alpha/Fe] plane (AFE_FLAG=0 build or legacy "
                "promotion):",
                "  CSPBasis_afe compiles the alpha interpolation away "
                "(static no-op).",
            ]

        txt = "\n".join(lines)
        if return_str:
            return txt
        print(txt, file=file or _sys.stdout)
        return None

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------
    def save(self, filename):
        """Serialise the alpha-enhanced SSP grids to HDF5 (schema 2.0)."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('ssp_lgmet',      data=np.array(self.ssp_lgmet))
            f.create_dataset('ssp_afe',        data=np.array(self.ssp_afe))
            f.create_dataset('ssp_lg_age_gyr', data=np.array(self.ssp_lg_age_gyr))
            f.create_dataset('ssp_wave',       data=np.array(self.ssp_wave))
            f.create_dataset('ssp_flux',       data=np.array(self.ssp_flux))

            f.attrs['description']        = ('FSPS alpha-enhanced SSP '
                                             'interpolation grids')
            f.attrs['units_lgmet']        = 'log10(absolute_total_metallicity)'
            f.attrs['units_afe']          = '[alpha/Fe] (dex)'
            f.attrs['units_lg_age_gyr']   = 'log10(age/Gyr)'
            f.attrs['units_wave']         = 'Angstrom'
            f.attrs['units_flux']         = 'L_sun Hz^-1 M_sun^-1'
            f.attrs['flux_axis_order']    = '(afe, met, age, wave)'

            for key in ('schema_version', 'isoc_type', 'spec_library',
                        'fsps_version'):
                val = getattr(self, key)
                if val is not None:
                    f.attrs[key] = str(val)
            if self.imf_type is not None:
                f.attrs['imf_type'] = int(self.imf_type)
            if self.wave_min is not None:
                f.attrs['wave_min'] = float(self.wave_min)
            if self.wave_max is not None:
                f.attrs['wave_max'] = float(self.wave_max)
            f.attrs['fsps_kwargs_json'] = json.dumps(self.fsps_kwargs or {})

    @classmethod
    def load(cls, filename):
        """Load an :class:`SSPDataAfe` from HDF5.

        Backward compatible with schema-1.0 (3-D, no alpha axis) files:
        those are promoted to ``n_afe = 1`` at [alpha/Fe] = 0, so a single
        loader serves both old and new grids.
        """
        def _decode(v):
            if isinstance(v, (bytes, bytearray)):
                return v.decode()
            return v

        with h5py.File(filename, 'r') as f:
            ssp_lgmet      = jnp.array(f['ssp_lgmet'][:])
            ssp_lg_age_gyr = jnp.array(f['ssp_lg_age_gyr'][:])
            ssp_wave       = jnp.array(f['ssp_wave'][:])
            ssp_flux       = jnp.array(f['ssp_flux'][:])

            if 'ssp_afe' in f:
                ssp_afe = jnp.array(f['ssp_afe'][:])
            else:
                # Legacy 3-D grid: promote to a single solar-scaled plane.
                ssp_afe  = jnp.zeros(1)
                ssp_flux = ssp_flux[None, ...]

            a = f.attrs
            meta = {
                'schema_version': _decode(a['schema_version'])
                                  if 'schema_version' in a else None,
                'isoc_type':      _decode(a['isoc_type'])
                                  if 'isoc_type' in a else None,
                'spec_library':   _decode(a['spec_library'])
                                  if 'spec_library' in a else None,
                'fsps_version':   _decode(a['fsps_version'])
                                  if 'fsps_version' in a else None,
                'imf_type':       int(a['imf_type']) if 'imf_type' in a else None,
                'wave_min':       float(a['wave_min']) if 'wave_min' in a else None,
                'wave_max':       float(a['wave_max']) if 'wave_max' in a else None,
            }
            if 'fsps_kwargs_json' in a:
                meta['fsps_kwargs'] = json.loads(_decode(a['fsps_kwargs_json']))
            else:
                meta['fsps_kwargs'] = {}

        return cls(ssp_lgmet, ssp_afe, ssp_lg_age_gyr, ssp_wave, ssp_flux,
                   **meta)

    # ------------------------------------------------------------------
    # Generation from FSPS
    # ------------------------------------------------------------------
    @classmethod
    def from_fsps(cls, save_to: Optional[str] = None,
                  afe_values=None, **fsps_kwargs) -> "SSPDataAfe":
        """
        Build an :class:`SSPDataAfe` directly from FSPS.

        Loops the parent builder's ``get_spectrum(tage=0, zmet=i)`` pattern
        over the [alpha/Fe] axis via the ``afeindx`` selector (honoured
        because the grid is built with ``zcontinuous=0``; NB ``afeindx``
        is 1-based and is silently IGNORED for ``zcontinuous > 0``).

        Requires python-fsps main (>= alpha-MC merge, 2026-08-02) compiled
        with ``AFE_FLAG=1`` against FSPS v4.0 + its v4.0 ``$SPS_HOME``
        data tree.  With an ``AFE_FLAG=0`` build (``n_afe == 1``) the
        result is a valid single-plane grid at [alpha/Fe] = 0 — the
        C3K/aMIST null model.

        Parameters
        ----------
        save_to : str or Path, optional
            If given, the result is persisted via :meth:`save`.
        afe_values : array-like, optional
            Explicit [alpha/Fe] values of the compiled FSPS grid, in
            ``afeindx`` order.  Defaults: n_afe == 5 -> the documented
            aMIST/C3K grid (-0.2 .. +0.6); n_afe == 1 -> [0.0]; anything
            else must be supplied explicitly.
        **fsps_kwargs
            Stellar-library / IMF kwargs, validated against the SAME
            whitelist as ``SSPData.from_fsps`` (``afe`` / ``afeindx`` are
            grid axes here and are rejected as build kwargs).
        """
        for bad in ('afe', 'afeindx'):
            if bad in fsps_kwargs:
                raise ValueError(
                    f"from_fsps() rejects the FSPS kwarg {bad!r}: the "
                    f"[alpha/Fe] axis is spanned by the grid itself "
                    f"(ssp_afe); CSPBasis_afe samples it through "
                    f"theta['afe'], so a fixed alpha must not be set at "
                    f"build time."
                )
        kwargs = _validate_fsps_kwargs(fsps_kwargs)

        if not HAS_FSPS:
            raise ImportError(
                "FSPS is required for SSP data generation but is not "
                "available. Alpha-enhanced grids additionally need "
                "python-fsps main (>= 2026-08-02) compiled with AFE_FLAG=1. "
                "See https://dfm.io/python-fsps/current/installation/"
            )

        # Discrete grid: zcontinuous=0 (zlegend points; also the mode in
        # which afeindx is honoured), sfh=0 (SSP mode).  Fixed here, never
        # caller-supplied.
        ssp = fsps.StellarPopulation(zcontinuous=0, sfh=0, **kwargs)

        n_afe = int(getattr(ssp, "n_afe", 1))
        if afe_values is not None:
            ssp_afe = np.atleast_1d(np.asarray(afe_values, dtype=float))
            if ssp_afe.size != n_afe:
                raise ValueError(
                    f"afe_values has {ssp_afe.size} entries but the "
                    f"compiled FSPS grid has n_afe = {n_afe}."
                )
        elif n_afe == 5:
            ssp_afe = FSPS_AFE_VALUES_NAFE5.copy()
        elif n_afe == 1:
            ssp_afe = np.zeros(1)
        else:
            raise ValueError(
                f"Compiled FSPS grid has n_afe = {n_afe}, which does not "
                f"match the documented aMIST/C3K layout (5) or a "
                f"solar-scaled build (1); pass afe_values= explicitly "
                f"(the [alpha/Fe] of each afeindx plane, in order)."
            )

        ssp_lgmet      = jnp.log10(ssp.zlegend)            # absolute log Z
        nzmet          = int(ssp_lgmet.size)
        ssp_lg_age_gyr = jnp.array(ssp.log_age - 9.0)      # log(yr)->log(Gyr)

        planes = []
        _wave = None
        for afe_indx in range(1, n_afe + 1):               # FSPS is 1-based
            if n_afe > 1:
                ssp.params["afeindx"] = afe_indx
                print(f"[alpha/Fe] plane {afe_indx}/{n_afe} "
                      f"(afe = {ssp_afe[afe_indx - 1]:+.1f})")
            spectrum_collector = []
            for zmet_indx in range(1, nzmet + 1):          # FSPS is 1-based
                print(f"...retrieving metallicity {zmet_indx}/{nzmet} "
                      f"[Z = {ssp.zlegend[zmet_indx - 1]:.4f}]")
                _wave, _fluxes = ssp.get_spectrum(
                    tage=0.0, zmet=zmet_indx, peraa=False)
                spectrum_collector.append(_fluxes)
            planes.append(np.array(spectrum_collector))

            # The zlegend must be identical for every alpha plane (FSPS
            # regularises the aMIST grids to a common nz); zlegend is a
            # compile-time constant per library, but assert anyway so a
            # future ragged-Z data release fails loudly here rather than
            # silently interpolating a ragged grid.
            _z_now = np.log10(np.asarray(ssp.zlegend))
            if not np.allclose(_z_now, np.asarray(ssp_lgmet)):
                raise RuntimeError(
                    f"zlegend changed between alpha planes (afeindx="
                    f"{afe_indx}); the Z grids are ragged across "
                    f"[alpha/Fe] and cannot form a rectangular "
                    f"(afe, Z, age, wave) grid."
                )

        ssp_wave = jnp.array(_wave)
        ssp_flux = jnp.array(np.stack(planes, axis=0))     # (n_afe, n_z, n_age, n_wave)

        meta = _read_fsps_provenance(ssp, kwargs, ssp_wave)
        meta['schema_version'] = SSP_AFE_SCHEMA_VERSION

        data = cls(ssp_lgmet, jnp.array(ssp_afe), ssp_lg_age_gyr,
                   ssp_wave, ssp_flux, **meta)
        if save_to is not None:
            data.save(save_to)
        return data
