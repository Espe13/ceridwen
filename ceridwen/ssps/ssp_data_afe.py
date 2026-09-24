"""Alpha-enhanced SSP grid container :class:`SSPDataAfe`: an ``SSPData`` whose flux
cube carries a leading [alpha/Fe] axis, ``ssp_flux`` (n_afe, n_met, n_ages, n_wave).
"""

import json
import typing
import h5py
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional

from ceridwen.ssps.ssp_data import (
    SSPData,
    LIBRARY_IMF_KWARGS,
    _validate_fsps_kwargs,
    _read_fsps_provenance,
    _import_fsps,
    fsps_stellar_mass_source,
    isochrone_points_or_none,
    corrected_table,
)

SSP_AFE_SCHEMA_VERSION = "3.1"   # 3.1: optional ssp_stellar_mass (n_afe, n_met, n_ages), as SSPData 3.0

FSPS_AFE_VALUES_NAFE5 = np.array([-0.2, 0.0, +0.2, +0.4, +0.6])


_MIST_AFE_ISO = ("m2", "p0", "p2", "p4", "p6")      # FSPS sps_vars.f90 afe_str_iso (AFE_FLAG=1)


def _duplicated_isochrone_cells(n_afe: int, n_z: int) -> tuple:
    """(iz, ia) cells whose MIST isochrone file in $SPS_HOME is byte-identical to another
    [alpha/Fe] plane's file at the same [Fe/H] tag (e.g. isoc_feh_p050_afe_p6 == _p4): FSPS
    then builds that plane on the wrong isochrone, so the cell is refused at fit time."""
    import hashlib
    import os
    import warnings
    home = os.environ.get("SPS_HOME")
    if not home or n_afe != len(_MIST_AFE_ISO):
        warnings.warn("cannot check the MIST isochrone files for duplicated [alpha/Fe] planes "
                      "($SPS_HOME unset or unexpected n_afe); no cell is refused", stacklevel=3)
        return ()
    iso = os.path.join(home, "ISOCHRONES", "MIST")
    try:
        tags = [ln[:4] for ln in open(os.path.join(iso, "zlegend.dat")) if ln.strip()]
    except OSError:
        return ()
    cells = []
    for iz, tag in enumerate(tags[:n_z]):
        seen = {}
        for ia, a in enumerate(_MIST_AFE_ISO):
            fn = os.path.join(iso, f"isoc_feh_{tag}_afe_{a}_vvcrit0.4_full.dat")
            try:
                with open(fn, "rb") as fh:
                    d = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                continue
            if d in seen and ia > 0:
                cells.append((iz, ia))
            seen.setdefault(d, ia)
    return tuple(c for c in cells if c[0] >= 1)


@dataclass(frozen=True)
class SSPDataAfe(SSPData):
    """Immutable alpha-enhanced SSP grid: an ``SSPData`` whose flux cube has a leading [alpha/Fe] axis.

    Parameters
    ----------
    ssp_lgmet : jnp.ndarray (n_met,) -- native axis [Fe/H] + log10(Z_sun), the SAME on every [alpha/Fe]
        plane (FSPS AFE_FLAG=1 loads isoc_feh_<tag>_afe_<a> per node): logzsol = [Fe/H] here, and
        the total metallicity is [Z/H] = logzsol + grid_metadata.f_alpha(afe); NOT the total Z
    ssp_afe : jnp.ndarray (n_afe,), keyword-only -- [alpha/Fe] grid, strictly increasing
    ssp_lg_age_gyr : jnp.ndarray (n_ages,) -- log10(age / Gyr)
    ssp_wave : jnp.ndarray (n_wave,) -- Angstrom
    ssp_flux : jnp.ndarray (n_afe, n_met, n_ages, n_wave) -- Lsun / Hz per Msun formed
    ssp_resolution : np.ndarray (n_wave,) or None -- sigma_v(lambda) [km/s] on ``ssp_wave``; required by save/load
    ssp_stellar_mass : np.ndarray (n_afe, n_met, n_ages) or None -- surviving mass per M_sun formed (optional)
    """

    ssp_afe: jnp.ndarray = field(kw_only=True)

    _display_title = "SSPDataAfe"
    _schema_label = "SSPDataAfe schema 3.1"
    _extra_datasets = ("ssp_afe",)

    @classmethod
    def _schema_version(cls) -> str:
        return SSP_AFE_SCHEMA_VERSION

    def _grid_arrays_for_chash(self):
        return (self.ssp_wave, self.ssp_lg_age_gyr, self.ssp_lgmet, self.ssp_flux, self.ssp_afe)

    def _check_metallicity_meta(self):
        if self.n_afe > 1 and self.axis_meaning != "feh":
            raise ValueError(
                f"an alpha-enhanced grid (n_afe = {self.n_afe}) must have axis_meaning='feh' "
                f"(got {self.axis_meaning!r}): CERIDWEN's alpha interpolation and logzsol_total "
                "assume the FSPS layout, where every [alpha/Fe] plane shares one [Fe/H] axis.  "
                "Pass axis_meaning='feh' if the grid was built that way; a grid on a total-Z "
                "axis per plane is not supported.")

    def _expected_flux_shape(self) -> tuple:
        return (int(self.ssp_afe.size), int(self.ssp_lgmet.size),
                int(self.ssp_lg_age_gyr.size), int(self.ssp_wave.size))

    def _check_flux_shape(self):
        expected = self._expected_flux_shape()
        if tuple(self.ssp_flux.shape) != expected:
            raise ValueError(
                f"SSPDataAfe flux grid shape mismatch: expected {expected} "
                f"(n_afe, n_met, n_ages, n_wave) but got "
                f"{tuple(self.ssp_flux.shape)}."
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

    def _display_grid_lines(self, size_str: str) -> list:
        lgmet = np.asarray(self.ssp_lgmet)
        afe   = np.asarray(self.ssp_afe)
        lgage = np.asarray(self.ssp_lg_age_gyr)
        wave  = np.asarray(self.ssp_wave)
        n_afe, n_met, n_age, n_wave = self.ssp_flux.shape
        age_gyr = 10.0 ** lgage
        return [
            f"  [alpha/Fe]               : {n_afe:>4d} pts   "
            f"{np.array2string(afe, precision=2)}",
        ] + self._metallicity_display_lines(n_met) + [
            "  total metallicity        : [Z/H] = logzsol + log10(1 - x + x 10^[a/Fe]), "
            "x = 0.687490 (MIST v2.5 GS98; ceridwen.ssps.grid_metadata.logzsol_total)",
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

    def _display_note_lines(self) -> list:
        notes = []
        if self.refused_cells:
            notes += ["refused interpolation cells (construction raises if a prior reaches them)"]
            for iz, ia in self.refused_cells:
                notes.append(f"  logzsol in ({self.logzsol_axis[iz - 1]:+.2f}, "
                             f"{self.logzsol_axis[iz]:+.2f}] x [alpha/Fe] in "
                             f"({float(self.ssp_afe[ia - 1]):+.1f}, {float(self.ssp_afe[ia]):+.1f}]"
                             ": duplicated isoc_feh_p050_afe_p6 (= p4) in FSPS")
        if self.n_afe > 1 and "chash table" in (self.zsun_source or ""):
            notes += ["note",
                      "  the metallicity axis is [Fe/H] + log10 Z_sun on every plane; CERIDWEN "
                      "takes this from its metadata table (by chash), not from the file's "
                      "units_lgmet attribute: nothing to do."]
        if self.n_afe == 1:
            return notes + [
                "note",
                "  single [alpha/Fe] plane (AFE_FLAG=0 build or legacy "
                "promotion):",
                "  CSPBasis_afe compiles the alpha interpolation away "
                "(static no-op).",
            ]
        return notes

    def _save_extra(self, f):
        f.create_dataset('ssp_afe', data=np.array(self.ssp_afe))
        f.attrs['description']     = ('FSPS alpha-enhanced SSP '
                                      'interpolation grids')
        f.attrs['units_lgmet']     = self._units_lgmet()
        f.attrs['units_afe']       = '[alpha/Fe] (dex)'
        f.attrs['flux_axis_order'] = '(afe, met, age, wave)'

    @classmethod
    def load(cls, filename, flux_dtype=None, zsun=None):
        """Load from HDF5; a 3-D grid without ``ssp_afe`` is promoted to n_afe = 1 at [alpha/Fe] = 0.
        ``zsun`` as for :meth:`SSPData.load`."""
        arrays, extra, meta = cls._read_h5(filename, flux_dtype=flux_dtype)
        ssp_afe = extra['ssp_afe']
        if ssp_afe is None:
            ssp_afe = jnp.zeros(1)
            arrays['ssp_flux'] = arrays['ssp_flux'][None, ...]
            if meta.get('ssp_stellar_mass') is not None:
                meta['ssp_stellar_mass'] = meta['ssp_stellar_mass'][None, ...]
        return cls(**arrays, ssp_afe=ssp_afe, **meta, zsun=zsun)

    @classmethod
    def from_fsps(cls, save_to: Optional[str] = None,
                  afe_values=None,
                  resolution_segments=None,
                  resolution_source: Optional[str] = None,
                  **fsps_kwargs) -> "SSPDataAfe":
        """Build the grid by looping ``get_spectrum(tage=0, zmet=i)`` over the [alpha/Fe] planes.

        Parameters
        ----------
        afe_values : array-like, optional -- [alpha/Fe] of each plane in ``afeindx`` order; defaults exist for n_afe 5 and 1 only
        resolution_segments : list, optional -- documented library LSF segments, max-combined with the 2-pixel sampling floor
        resolution_source : str, optional -- citation for ``resolution_segments``; each raises if given without the other
        """
        if resolution_source is not None and resolution_segments is None:
            raise ValueError(
                "from_fsps(): resolution_source was given without "
                "resolution_segments.  The sampling-floor provenance is "
                "recorded automatically; a source note only accompanies "
                "explicit library-LSF segments."
            )
        if resolution_segments is not None and resolution_source is None:
            raise ValueError(
                "from_fsps(): resolution_segments were given without "
                "resolution_source.  Cite where the LSF numbers come from "
                "— uncited resolution numbers must not ship in a released "
                "grid."
            )
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

        fsps = _import_fsps(
            "SSPDataAfe.load(ceridwen.ssps.fetch_grid('amist_c3k_hr_krou_afe'))")

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

        ssp_lgmet      = jnp.log10(ssp.zlegend)
        nzmet          = int(ssp_lgmet.size)
        ssp_lg_age_gyr = jnp.array(ssp.log_age - 9.0)

        planes = []
        mass_planes = []
        iso_planes = []
        _wave = None
        for afe_indx in range(1, n_afe + 1):               # afeindx is 1-based
            if n_afe > 1:
                ssp.params["afeindx"] = afe_indx
                print(f"[alpha/Fe] plane {afe_indx}/{n_afe} "
                      f"(afe = {ssp_afe[afe_indx - 1]:+.1f})")
            spectrum_collector = []
            mass_collector = []
            iso_collector = []
            for zmet_indx in range(1, nzmet + 1):
                print(f"...retrieving metallicity {zmet_indx}/{nzmet} "
                      f"[Z = {ssp.zlegend[zmet_indx - 1]:.4f}]")
                _wave, _fluxes = ssp.get_spectrum(
                    tage=0.0, zmet=zmet_indx, peraa=False)
                spectrum_collector.append(_fluxes)
                mass_collector.append(np.array(ssp.stellar_mass, dtype=np.float64))
                iso_collector.append(isochrone_points_or_none(ssp))
            planes.append(np.array(spectrum_collector))
            mass_planes.append(np.array(mass_collector))
            iso_planes.append(iso_collector)

            _z_now = np.log10(np.asarray(ssp.zlegend))
            if not np.allclose(_z_now, np.asarray(ssp_lgmet)):
                raise RuntimeError(
                    f"zlegend changed between alpha planes (afeindx="
                    f"{afe_indx}); the Z grids are ragged across "
                    f"[alpha/Fe] and cannot form a rectangular "
                    f"(afe, Z, age, wave) grid."
                )

        ssp_wave = jnp.array(_wave)
        ssp_flux = jnp.array(np.stack(planes, axis=0))

        meta = _read_fsps_provenance(ssp, kwargs, ssp_wave, ssp_lgmet)
        meta['schema_version'] = SSP_AFE_SCHEMA_VERSION
        iso = (None if iso_planes[0][0] is None
               else np.array(iso_planes, dtype=np.float64))
        meta['ssp_stellar_mass'], note = corrected_table(ssp, np.stack(mass_planes, axis=0), iso)
        meta['stellar_mass_source'] = fsps_stellar_mass_source(meta['fsps_version']) + note
        if n_afe > 1 and meta.get('isoc_type') == 'mist':
            meta['refused_cells'] = _duplicated_isochrone_cells(n_afe, int(nzmet))

        from .library_resolution import combined_sigma_v, combined_source
        sigma_v = combined_sigma_v(np.asarray(ssp_wave),
                                   segments=resolution_segments)

        data = cls(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux,
                   ssp_afe=jnp.array(ssp_afe),
                   ssp_resolution=sigma_v,
                   resolution_source=combined_source(resolution_source),
                   **meta)
        if save_to is not None:
            data.save(save_to)
        return data
