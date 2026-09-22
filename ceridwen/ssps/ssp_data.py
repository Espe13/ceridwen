"""SSP interpolation grids (:class:`SSPData`): construction, provenance and HDF5 I/O."""

import json
import math
import typing
import h5py
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field, InitVar
from typing import Optional

def _import_fsps():
    try:
        import fsps
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "FSPS is required for SSP data generation but is not available. "
            "See https://dfm.io/python-fsps/current/installation/") from exc
    return fsps


DEFAULT_SSP_BNAME = "ssp_data_fsps_v3.2_lgmet_age.h5"

SSP_SCHEMA_VERSION = "3.0"

# Schema history: 2.0 added the library resolution curve ``ssp_resolution`` (required);
# 3.0 adds the optional surviving stellar-mass table ``ssp_stellar_mass`` (grid shape minus
# the wavelength axis), which a schema-2 file lacks and scripts/attach_stellar_mass.py adds.
STELLAR_MASS_SCRIPT = "scripts/attach_stellar_mass.py"


_IMF_KWARGS = frozenset({
    "imf_type", "imf1", "imf2", "imf3",
    "imf_lower_limit", "imf_upper_limit", "vdmc", "mdave",
})

_LIBRARY_KWARGS = frozenset({
    "tpagb_norm_type", "agb", "pagb", "redgb", "fbhb", "sbss",
    "delt", "dell", "evtype", "masscut", "use_wr_spectra",
    "logt_wmb_hot", "add_stellar_remnants", "fcstar",
})

LIBRARY_IMF_KWARGS = _IMF_KWARGS | _LIBRARY_KWARGS


def _owned(names, mechanism):
    return {n: mechanism for n in names}


_CSP_OWNED_KWARGS = {
    **_owned(
        ["sfh", "tage", "tau", "const", "sf_start", "sf_trunc",
         "tburst", "fburst", "sf_slope", "compute_light_ages"],
        "the star-formation history, which CSPBasis builds itself "
        "(set it through theta, e.g. 'lookback_time' / 'logsfr_ratios')",
    ),
    **_owned(
        ["zmet", "logzsol", "pmetals"],
        "metallicity — the SSP grid spans metallicity itself (ssp_lgmet, on "
        "FSPS's discrete zlegend); the metallicity is a FIT parameter, "
        "theta['logzsol'] = log10(Z/Z_sun) (or theta['logzsol_hist']), sampled "
        "by CSPBasis on that axis, so a fixed metallicity must not be set at "
        "build time",
    ),
    **_owned(
        ["dust_type", "dust1", "dust2", "dust3", "dust_index", "dust1_index",
         "dust_tesc", "dust_clumps", "frac_nodust", "frac_obrun", "mwr", "uvb",
         "wgp1", "wgp2", "wgp3", "agb_dust", "add_agb_dust_model"],
        "dust attenuation, which CSPBasis applies at fit time "
        "(Dust / DiffuseDust, and its own AGB circumstellar-dust treatment)",
    ),
    **_owned(
        ["add_dust_emission", "duste_gamma", "duste_qpah", "duste_umin"],
        "dust emission, which CSPBasis applies at fit time (DustEmission)",
    ),
    **_owned(
        ["fagn", "agn_tau"],
        "AGN emission, which is applied downstream by the CSP forward model, "
        "not baked into the SSP grid",
    ),
    **_owned(
        ["add_neb_emission", "add_neb_continuum", "gas_logz", "gas_logu",
         "nebemlineinspec", "cloudy_dust"],
        "nebular emission, which CSPBasis applies via NebularModel "
        "(gas_logz / gas_logu are sampled through theta)",
    ),
    **_owned(
        ["add_igm_absorption", "igm_factor"],
        "IGM attenuation, which CSPBasis applies at fit time (add_igm=...)",
    ),
    **_owned(
        ["zred", "redshift_colors"],
        "redshift / cosmological normalisation, which CSPBasis applies at fit "
        "time (theta 'zred')",
    ),
    **_owned(
        ["sigma_smooth", "smooth_velocity", "smooth_lsf",
         "min_wave_smooth", "max_wave_smooth"],
        "LOSVD / line-spread smoothing, which CSPBasis applies at fit time; "
        "the SSP grid is stored unsmoothed",
    ),
    **_owned(
        ["add_xrb_emission", "frac_xrb"],
        "X-ray binary emission, which the CSP forward model does not model; "
        "baking it into the grid would make it inconsistent with CSPBasis",
    ),
    **_owned(
        ["zcontinuous"],
        "the metallicity-grid mode, which is fixed internally (zcontinuous=0) "
        "so the grid is built on FSPS's discrete zlegend points",
    ),
}


def _validate_fsps_kwargs(kwargs: dict) -> dict:
    """Return ``kwargs`` unchanged; raise ValueError on any kwarg that is not library/IMF-defining."""
    for name in kwargs:
        if name in LIBRARY_IMF_KWARGS:
            continue
        if name in _CSP_OWNED_KWARGS:
            raise ValueError(
                f"from_fsps() rejects the FSPS kwarg {name!r}: it controls "
                f"{_CSP_OWNED_KWARGS[name]}. An SSP grid must contain only the "
                f"stellar library / IMF; remove {name!r} from the build. "
                f"Allowed build-time kwargs: {sorted(LIBRARY_IMF_KWARGS)}."
            )
        raise ValueError(
            f"from_fsps() rejects the kwarg {name!r}: it is not a "
            f"stellar-library / IMF-defining FSPS parameter, so it does not "
            f"belong at SSP-build time (it is either applied later by CSPBasis "
            f"or not a recognised FSPS StellarPopulation parameter). "
            f"Allowed build-time kwargs: {sorted(LIBRARY_IMF_KWARGS)}."
        )
    return kwargs


@dataclass(frozen=True)
class SSPData:
    """Immutable container for the SSP interpolation grids plus static provenance.

    Parameters
    ----------
    ssp_lgmet : array (n_met,) -- the grid's NATIVE metallicity axis, log10 Z: absolute Z for
        BPASS, the FSPS label log10(zsun 10^[Fe/H]) for MIST/aMIST (not MIST's physical initial Z)
    ssp_lg_age_gyr : array (n_ages,) -- log10(age / Gyr)
    ssp_wave : array (n_wave,), Angstrom
    ssp_flux : array (n_met, n_ages, n_wave), L_sun/Hz per M_sun formed
    ssp_resolution : ndarray (n_wave,), km/s -- library sigma_v(lambda) on ssp_wave, NaN where unknown; optional in memory, required by save()/load()
    ssp_stellar_mass : ndarray (n_met, n_ages) or None -- surviving mass (stars + remnants) per
        M_sun formed of each SSP, FSPS ``stellar_mass`` (schema 3.0, optional); None when the
        grid carries no table (``scripts/attach_stellar_mass.py`` adds one to an existing file)
    stellar_mass_source : str or None -- provenance of ``ssp_stellar_mass``
    zsun : float, init-only -- the grid's solar metallicity (mass fraction) when it is neither
        recorded (``log10_zsun``) nor in ``grid_metadata.CHASH_TABLE``; must match a node
    log10_zsun : float -- resolved: the grid's solar node (exact), so logzsol = ssp_lgmet - log10_zsun
    axis_meaning : {"feh", "log_z_over_zsun"} -- what logzsol is on this grid ([Fe/H] for MIST/aMIST)
    """

    ssp_lgmet: jnp.ndarray
    ssp_lg_age_gyr: jnp.ndarray
    ssp_wave: jnp.ndarray
    ssp_flux: jnp.ndarray

    ssp_resolution: Optional[np.ndarray] = field(default=None, compare=False)
    resolution_source: Optional[str] = field(default=None, compare=False)
    ssp_stellar_mass: Optional[np.ndarray] = field(default=None, compare=False)
    stellar_mass_source: Optional[str] = field(default=None, compare=False)

    isoc_type: Optional[str] = field(default=None, compare=False)
    spec_library: Optional[str] = field(default=None, compare=False)
    imf_type: Optional[int] = field(default=None, compare=False)
    fsps_version: Optional[str] = field(default=None, compare=False)
    fsps_kwargs: dict = field(default_factory=dict, compare=False)
    wave_min: Optional[float] = field(default=None, compare=False)
    wave_max: Optional[float] = field(default=None, compare=False)
    schema_version: Optional[str] = field(default=None, compare=False)

    log10_zsun: Optional[float] = field(default=None, compare=False)
    zsun_nominal: Optional[float] = field(default=None, compare=False)
    axis_meaning: Optional[str] = field(default=None, compare=False)
    zsun_source: Optional[str] = field(default=None, compare=False)
    chash: Optional[str] = field(default=None, compare=False)
    refused_cells: tuple = field(default=(), compare=False)
    refused_reason: Optional[str] = field(default=None, compare=False)
    zsun: InitVar[Optional[float]] = None
    _chash_hint: InitVar[Optional[str]] = None

    _display_title = "SSPData"
    _schema_label = "SSP schema 3.0"
    _extra_datasets: tuple = ()

    def _expected_flux_shape(self) -> tuple:
        return (int(self.ssp_lgmet.size), int(self.ssp_lg_age_gyr.size),
                int(self.ssp_wave.size))

    def _check_flux_shape(self):
        expected = self._expected_flux_shape()
        if tuple(self.ssp_flux.shape) != expected:
            hint = ""
            if self.ssp_flux.ndim == 4:
                hint = ("  A 4-D flux cube (n_afe, n_met, n_ages, n_wave) is an "
                        "alpha-enhanced grid: use ceridwen.ssps.SSPDataAfe.")
            raise ValueError(
                f"SSP flux grid shape mismatch: expected {expected} but got "
                f"{tuple(self.ssp_flux.shape)}.  Grid dimensions must be "
                f"consistent (n_met, n_ages, n_wave).{hint}"
            )

    def __post_init__(self, zsun=None, _chash_hint=None):
        self._check_flux_shape()
        self._resolve_metallicity(zsun, _chash_hint)
        if self.ssp_resolution is not None:
            res = np.asarray(self.ssp_resolution, dtype=np.float64)
            if res.shape != (int(self.ssp_wave.size),):
                raise ValueError(
                    f"ssp_resolution shape {res.shape} must match ssp_wave "
                    f"({int(self.ssp_wave.size)},): one sigma_v(lambda) "
                    f"[km/s] per wavelength pixel (NaN where unknown)."
                )
            finite = res[np.isfinite(res)]
            if finite.size and (finite <= 0.0).any():
                raise ValueError(
                    "ssp_resolution must be positive (km/s) where finite; "
                    "use NaN to mark pixels of unknown library resolution."
                )
            object.__setattr__(self, "ssp_resolution", res)
        if self.ssp_stellar_mass is not None:
            object.__setattr__(self, "ssp_stellar_mass",
                               self._checked_stellar_mass(self.ssp_stellar_mass))

    def _checked_stellar_mass(self, mass) -> np.ndarray:
        """``mass`` as float64 of shape ``ssp_flux.shape[:-1]``, finite and positive, else ValueError."""
        m = np.asarray(mass, dtype=np.float64)
        want = self._expected_flux_shape()[:-1]
        if m.shape != want:
            raise ValueError(
                f"ssp_stellar_mass shape {m.shape} must be the grid shape without the "
                f"wavelength axis, {want}: one surviving mass per SSP (M_sun per M_sun formed).")
        if not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError("ssp_stellar_mass must be finite and positive (M_sun surviving per "
                             "M_sun formed); got non-finite or non-positive entries.")
        return m

    def with_stellar_mass(self, mass, *, source):
        """Return a copy carrying the surviving stellar-mass table ``mass`` (shape
        ``ssp_flux.shape[:-1]``, M_sun per M_sun formed) with its provenance ``source``."""
        import dataclasses as _dc
        if not source:
            raise ValueError("with_stellar_mass: pass source= (where the masses come from)")
        return _dc.replace(self, ssp_stellar_mass=self._checked_stellar_mass(mass),
                           stellar_mass_source=str(source), schema_version=self._schema_version())

    @classmethod
    def _schema_version(cls) -> str:
        return SSP_SCHEMA_VERSION

    def require_stellar_mass(self) -> np.ndarray:
        """The surviving stellar-mass table; ValueError naming the attach script when absent."""
        if self.ssp_stellar_mass is None:
            raise ValueError(missing_stellar_mass_message(type(self).__name__))
        return self.ssp_stellar_mass

    def _grid_arrays_for_chash(self):
        return (self.ssp_wave, self.ssp_lg_age_gyr, self.ssp_lgmet, self.ssp_flux, None)

    def _resolve_metallicity(self, zsun, chash_hint):
        """Resolve Z_sun (``grid_metadata.resolve_zsun``) and fill the metallicity provenance."""
        from .grid_metadata import chash_arrays, resolve_zsun, AXIS_MEANINGS
        wave, age, lgmet, flux, afe = self._grid_arrays_for_chash()
        ch = chash_hint or chash_arrays(wave, age, lgmet, flux, afe)
        log10_zsun, sources, meta = resolve_zsun(
            ssp_lgmet=lgmet, zsun_kwarg=zsun, provenance_log10_zsun=self.log10_zsun,
            chash=ch, describe=f"this {type(self).__name__} grid")
        axis = self.axis_meaning
        if meta is not None:
            if axis is not None and axis != meta.axis_meaning:
                raise ValueError(
                    f"axis_meaning={axis!r} contradicts the metadata table entry "
                    f"{meta.name!r} ({meta.axis_meaning!r}) for this grid's chash {ch}")
            axis = meta.axis_meaning
        if axis is not None and axis not in AXIS_MEANINGS:
            raise ValueError(f"axis_meaning must be one of {AXIS_MEANINGS}, got {axis!r}")
        nominal = self.zsun_nominal
        if nominal is None:
            nominal = (meta.zsun_nominal if meta is not None
                       else float(zsun) if zsun is not None else 10.0 ** log10_zsun)
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731  (frozen dataclass)
        set_("log10_zsun", float(log10_zsun))
        set_("zsun_nominal", float(nominal))
        set_("axis_meaning", axis)
        set_("zsun_source", " + ".join(sources))
        set_("chash", ch)
        if meta is not None:
            for k in ("isoc_type", "spec_library"):
                if getattr(self, k) is None:
                    set_(k, getattr(meta, k))
            if self.fsps_version is None and meta.fsps_version is not None:
                set_("fsps_version", meta.fsps_version)
            set_("refused_cells", tuple(tuple(c) for c in meta.refused_cells))
            set_("refused_reason", meta.refused_reason)
        else:
            set_("refused_cells", tuple(tuple(int(i) for i in c) for c in self.refused_cells))
        self._check_metallicity_meta()

    def _check_metallicity_meta(self):
        """Subclass hook: extra consistency checks on the resolved metallicity metadata."""
        return None

    @property
    def logzsol_axis(self) -> np.ndarray:
        """The metallicity axis in logzsol = ssp_lgmet - log10_zsun (float64)."""
        return np.asarray(self.ssp_lgmet, dtype=np.float64) - self.log10_zsun

    def _metadata_entry(self):
        from .grid_metadata import CHASH_TABLE
        return CHASH_TABLE.get(self.chash)

    def with_resolution(self, *, sigma_v=None, segments=None, source=None):
        """Return a copy carrying a library resolution curve from exactly one of ``sigma_v`` (km/s on ssp_wave, NaN where unknown) or ``segments``."""
        import dataclasses as _dc
        from .library_resolution import sigma_v_from_segments
        if (sigma_v is None) == (segments is None):
            raise ValueError(
                "with_resolution: pass exactly one of sigma_v= or segments=")
        if segments is not None:
            sigma_v = sigma_v_from_segments(
                np.asarray(self.ssp_wave), segments)
        return _dc.replace(
            self,
            ssp_resolution=np.asarray(sigma_v, dtype=np.float64),
            resolution_source=(str(source) if source is not None else None),
        )

    def _display_grid_lines(self, size_str: str) -> list:
        lgmet = np.asarray(self.ssp_lgmet)
        lgage = np.asarray(self.ssp_lg_age_gyr)
        wave  = np.asarray(self.ssp_wave)
        n_met, n_age, n_wave = self.ssp_flux.shape
        age_gyr = 10.0 ** lgage
        return self._metallicity_display_lines(n_met) + [
            f"  age          log10(Gyr)  : {n_age:>4d} pts   "
            f"[{lgage.min():+.3f}, {lgage.max():+.3f}]  "
            f"= [{age_gyr.min():.3g}, {age_gyr.max():.3g}] Gyr",
            f"  wavelength   Angstrom    : {n_wave:>4d} pts   "
            f"[{wave.min():.1f}, {wave.max():.1f}]",
            f"  flux (n_met,n_age,n_wave): {tuple(int(s) for s in self.ssp_flux.shape)}  "
            f"[L_sun Hz^-1 M_sun^-1]  {np.asarray(self.ssp_flux).dtype}  {size_str}",
        ]

    def _metallicity_display_lines(self, n_met) -> list:
        lgmet = np.asarray(self.ssp_lgmet, dtype=np.float64)
        lz = self.logzsol_axis
        meaning = {"feh": "[Fe/H] (FSPS MIST node label)",
                   "log_z_over_zsun": "log10(Z/Z_sun), Z = metal mass fraction",
                   None: "not recorded"}[self.axis_meaning]
        out = [
            f"  metallicity  logzsol     : {n_met:>4d} pts   "
            f"[{lz.min():+.3f}, {lz.max():+.3f}]  = {meaning}",
            f"  metallicity  native      :            "
            f"[{lgmet.min():+.4f}, {lgmet.max():+.4f}]  log10 Z = logzsol + log10 Z_sun",
            f"  Z_sun                    : {self.zsun_nominal:.6g}  "
            f"(log10 Z_sun = {self.log10_zsun!r}, the solar node; source: {self.zsun_source})",
        ]
        if self.axis_meaning == "feh":
            out.append("  native values are FSPS labels log10(Z_sun 10^[Fe/H]), not MIST's "
                       "physical initial Z (0.0164 at [Fe/H]=0)")
        return out

    def _display_note_lines(self) -> list:
        if self.isoc_type is None:
            return [
                "note",
                "  isoc_type is None (legacy grid, built before provenance "
                "tracking):",
                "  CSPBasis will warn and fall back to 'mist' for the nebular grid.",
            ]
        return []

    def display(self, *, return_str: bool = False, file=None):
        """Print (or return as str when ``return_str``) a summary of the grid and its provenance."""
        import sys as _sys

        wave  = np.asarray(self.ssp_wave)

        def _fmt(v, na="—"):
            return na if v is None else str(v)

        size = float(np.asarray(self.ssp_flux).nbytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                size_str = f"{size:.1f} {unit}"
                break
            size /= 1024.0

        lines = [
            self._display_title,
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
        ] + self._display_grid_lines(size_str)
        if self.ssp_resolution is None:
            lines += ["  library resolution       : MISSING "
                      "(cannot be saved; attach with with_resolution)"]
        else:
            res = np.asarray(self.ssp_resolution, dtype=np.float64)
            fin = np.isfinite(res)
            if fin.any():
                cov_lo = wave[fin].min(); cov_hi = wave[fin].max()
                lines += [
                    f"  library resolution       : sigma_v "
                    f"[{res[fin].min():.1f}, {res[fin].max():.1f}] km/s over "
                    f"[{cov_lo:.0f}, {cov_hi:.0f}] AA "
                    f"({100.0 * fin.mean():.0f}% of pixels; NaN elsewhere)",
                ]
            else:
                lines += ["  library resolution       : all-NaN "
                          "(unknown everywhere; no subtraction will occur)"]
            if self.resolution_source:
                lines += [f"  resolution source        : {self.resolution_source}"]
        if self.ssp_stellar_mass is None:
            lines += [f"  surviving stellar mass   : not in this grid (add it with "
                      f"{STELLAR_MASS_SCRIPT})"]
        else:
            m = self.ssp_stellar_mass
            lines += [f"  surviving stellar mass   : [{m.min():.4g}, {m.max():.4g}] M_sun per "
                      f"M_sun formed ({self.stellar_mass_source or 'source not recorded'})"]
        lines += self._display_note_lines()

        txt = "\n".join(lines)
        if return_str:
            return txt
        print(txt, file=file or _sys.stdout)
        return None

    def save(self, filename):
        """Write grids, resolution curve and provenance attrs to HDF5 (overwrites); raises ValueError if ``ssp_resolution`` is None."""
        if self.ssp_resolution is None:
            raise ValueError(
                f"{type(self).__name__}.save(): this grid carries no library "
                f"resolution curve (ssp_resolution is None).  "
                f"{self._schema_label} files require one — attach it with "
                "with_resolution(segments=...) or with_resolution(sigma_v=...) "
                "before saving."
            )
        with h5py.File(filename, 'w') as f:
            f.create_dataset('ssp_lgmet',      data=np.array(self.ssp_lgmet))
            f.create_dataset('ssp_lg_age_gyr', data=np.array(self.ssp_lg_age_gyr))
            f.create_dataset('ssp_wave',       data=np.array(self.ssp_wave))
            f.create_dataset('ssp_flux',       data=np.array(self.ssp_flux))
            f.create_dataset('ssp_resolution',
                             data=np.asarray(self.ssp_resolution,
                                             dtype=np.float64))

            f.attrs['description']        = 'FSPS SSP interpolation grids'
            f.attrs['units_lgmet']        = self._units_lgmet()
            f.attrs['units_lg_age_gyr']   = 'log10(age/Gyr)'
            f.attrs['units_wave']         = 'Angstrom'
            f.attrs['units_flux']         = 'L_sun Hz^-1 M_sun^-1'
            f.attrs['units_resolution']   = 'sigma_v [km/s]; NaN = unknown'
            self._save_extra(f)
            if self.ssp_stellar_mass is not None:
                f.create_dataset('ssp_stellar_mass',
                                 data=np.asarray(self.ssp_stellar_mass, dtype=np.float64))
                f.attrs['units_stellar_mass'] = ('M_sun surviving (stars + remnants) per '
                                                 'M_sun formed, per SSP')
                if self.stellar_mass_source is not None:
                    f.attrs['stellar_mass_source'] = str(self.stellar_mass_source)
            if self.resolution_source is not None:
                f.attrs['resolution_source'] = str(self.resolution_source)

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
            f.attrs['log10_zsun'] = float(self.log10_zsun)
            f.attrs['zsun_nominal'] = float(self.zsun_nominal)
            if self.axis_meaning is not None:
                f.attrs['axis_meaning'] = str(self.axis_meaning)
            f.attrs['chash'] = str(self.chash)
            if self.refused_cells:
                f.attrs['refused_cells_json'] = json.dumps([list(c) for c in self.refused_cells])

    def _units_lgmet(self) -> str:
        if self.axis_meaning == "feh":
            return ("log10(Z_label) = [Fe/H] + log10_zsun: FSPS MIST node label, not the "
                    "total metal mass fraction; logzsol = [Fe/H]")
        if self.axis_meaning == "log_z_over_zsun":
            return "log10(Z), Z = metal mass fraction; logzsol = log10(Z) - log10_zsun"
        return "log10(Z); logzsol = log10(Z) - log10_zsun (axis meaning not recorded)"

    def _save_extra(self, f):
        """Subclass hook for extra datasets / attrs, called inside save()."""
        return None

    @classmethod
    def _read_h5(cls, filename, flux_dtype=None):
        """Return ``(arrays, extra, meta)`` read from an HDF5 grid file."""
        def _decode(v):
            if isinstance(v, (bytes, bytearray)):
                return v.decode()
            return v

        with h5py.File(filename, 'r') as f:
            if 'ssp_resolution' not in f:
                raise ValueError(
                    f"{filename}: no 'ssp_resolution' dataset — this grid "
                    f"predates {cls._schema_label}.  Convert it (no FSPS rebuild "
                    f"needed) with scripts/convert_grids_schema2.py, which "
                    f"copies the existing arrays and attaches the library "
                    f"resolution curve."
                )
            from .grid_metadata import cached_chash, remember_chash, chash_arrays
            raw = {k: f[k][:] for k in ('ssp_lgmet', 'ssp_lg_age_gyr', 'ssp_wave', 'ssp_flux')}
            raw_extra = {name: (f[name][:] if name in f else None)
                         for name in cls._extra_datasets}
            extras = tuple(k for k, v in raw_extra.items() if v is not None)
            chash = cached_chash(filename, extras)
            if chash is None:
                chash = chash_arrays(raw['ssp_wave'], raw['ssp_lg_age_gyr'], raw['ssp_lgmet'],
                                     raw['ssp_flux'], raw_extra.get('ssp_afe'))
                remember_chash(filename, chash, extras)
            arrays = {
                'ssp_lgmet':      jnp.array(raw['ssp_lgmet']),
                'ssp_lg_age_gyr': jnp.array(raw['ssp_lg_age_gyr']),
                'ssp_wave':       jnp.array(raw['ssp_wave']),
                'ssp_flux':       jnp.asarray(raw['ssp_flux'] if flux_dtype is None
                                              else raw['ssp_flux'].astype(flux_dtype)),
            }
            extra = {name: (jnp.array(v) if v is not None else None)
                     for name, v in raw_extra.items()}
            ssp_resolution = np.asarray(f['ssp_resolution'][:],
                                        dtype=np.float64)
            # schema 3.0, optional; deliberately NOT part of the content hash (chash), which
            # identifies the flux grid: attaching a mass table keeps the grid's identity
            ssp_stellar_mass = (np.asarray(f['ssp_stellar_mass'][:], dtype=np.float64)
                                if 'ssp_stellar_mass' in f else None)

            a = f.attrs
            meta = {
                'ssp_resolution':    ssp_resolution,
                'ssp_stellar_mass':  ssp_stellar_mass,
                'stellar_mass_source': _decode(a['stellar_mass_source'])
                                       if 'stellar_mass_source' in a else None,
                'resolution_source': _decode(a['resolution_source'])
                                     if 'resolution_source' in a else None,
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
                'log10_zsun':     float(a['log10_zsun']) if 'log10_zsun' in a else None,
                'zsun_nominal':   float(a['zsun_nominal']) if 'zsun_nominal' in a else None,
                'axis_meaning':   _decode(a['axis_meaning']) if 'axis_meaning' in a else None,
                'refused_cells':  (tuple(tuple(c) for c in json.loads(_decode(a['refused_cells_json'])))
                                   if 'refused_cells_json' in a else ()),
                '_chash_hint':    chash,
            }
            if 'fsps_kwargs_json' in a:
                meta['fsps_kwargs'] = json.loads(_decode(a['fsps_kwargs_json']))
            else:
                meta['fsps_kwargs'] = {}
            if 'chash' in a and _decode(a['chash']) != chash:
                import warnings
                warnings.warn(
                    f"{filename}: its recorded chash {_decode(a['chash'])} differs from the "
                    f"content hash of its arrays ({chash}); the arrays were modified after the "
                    "file was written.", UserWarning, stacklevel=3)
        return arrays, extra, meta

    @classmethod
    def load(cls, filename, flux_dtype=None, zsun=None):
        """Load a grid from HDF5; raises ValueError if the file lacks ``ssp_resolution``. ``flux_dtype`` casts the flux cube on read.
        ``zsun`` supplies the grid's solar metallicity when it is neither recorded in the file nor in ``grid_metadata.CHASH_TABLE``."""
        arrays, _extra, meta = cls._read_h5(filename, flux_dtype=flux_dtype)
        return cls(**arrays, **meta, zsun=zsun)

    @classmethod
    def from_fsps(cls, save_to: Optional[str] = None,
                  resolution_segments=None,
                  resolution_source: Optional[str] = None,
                  **fsps_kwargs) -> "SSPData":
        """Build a grid from the SPS backend, attach the library resolution curve and record provenance.

        Only library/IMF-defining kwargs (``LIBRARY_IMF_KWARGS``) are accepted.
        The resolution curve is the element-wise maximum of the 2-pixel sampling
        floor of ``ssp_wave`` and ``resolution_segments`` (if given, which then
        require ``resolution_source``). Raises ValueError otherwise.
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
                "(e.g. resolution_source='MILES FWHM 2.54A "
                "(Falcon-Barroso et al. 2011)') — uncited resolution "
                "numbers must not ship in a released grid."
            )
        from .library_resolution import combined_sigma_v, combined_source
        data = collect_ssp_data_wrapper(**fsps_kwargs)
        sigma_v = combined_sigma_v(np.asarray(data.ssp_wave),
                                   segments=resolution_segments)
        data = data.with_resolution(
            sigma_v=sigma_v, source=combined_source(resolution_source))
        if save_to is not None:
            data.save(save_to)
        return data


def _collect_ssp_and_meta(**kwargs):
    """Return ``(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, meta)`` built from the SPS backend."""
    kwargs = _validate_fsps_kwargs(kwargs)

    fsps = _import_fsps()
    ssp = fsps.StellarPopulation(zcontinuous=0, sfh=0, **kwargs)

    ssp_lgmet      = jnp.log10(ssp.zlegend)
    nzmet          = ssp_lgmet.size
    ssp_lg_age_gyr = ssp.log_age - 9.0

    spectrum_collector = []
    mass_collector = []
    for zmet_indx in range(1, nzmet + 1):              # 1-based metallicity index
        print(f"...retrieving metallicity {zmet_indx}/{nzmet} "
              f"[Z = {ssp.zlegend[zmet_indx-1]:.4f}]")
        _wave, _fluxes = ssp.get_spectrum(tage=0.0, zmet=zmet_indx, peraa=False)
        spectrum_collector.append(_fluxes)
        mass_collector.append(np.array(ssp.stellar_mass, dtype=np.float64))

    ssp_wave       = jnp.array(_wave)
    ssp_flux       = jnp.array(spectrum_collector)
    ssp_lg_age_gyr = jnp.array(ssp_lg_age_gyr)

    meta = _read_fsps_provenance(ssp, kwargs, ssp_wave, ssp_lgmet)
    meta["ssp_stellar_mass"] = np.array(mass_collector)
    meta["stellar_mass_source"] = fsps_stellar_mass_source(meta["fsps_version"])
    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, meta


def _read_fsps_provenance(ssp, kwargs: dict, ssp_wave, ssp_lgmet) -> dict:
    """Return the provenance dict for a built StellarPopulation."""
    def _dec(x):
        if isinstance(x, (bytes, bytearray)):
            return x.decode()
        return None if x is None else str(x)

    libs = getattr(ssp, "libraries", ()) or ()
    isoc_type    = _dec(libs[0]) if len(libs) > 0 else None
    spec_library = _dec(libs[1]) if len(libs) > 1 else None

    try:
        imf_type = int(ssp.params["imf_type"])
    except Exception:
        imf_type = kwargs.get("imf_type")
        imf_type = int(imf_type) if imf_type is not None else None

    zsol = float(getattr(ssp, "solar_metallicity"))     # python-fsps: driver.get_zsol()
    from .grid_metadata import solar_node
    log10_zsun = solar_node(np.asarray(ssp_lgmet, dtype=np.float64), math.log10(zsol))
    # sps_setup.f90:141-145: MIST zlegend = 10**[Fe/H] * zsol (a label), others read linear Z
    axis_meaning = "feh" if isoc_type == "mist" else "log_z_over_zsun"

    return {
        "isoc_type":      isoc_type,
        "spec_library":   spec_library,
        "imf_type":       imf_type,
        "log10_zsun":     log10_zsun,
        "zsun_nominal":   float(np.float32(zsol)),
        "axis_meaning":   axis_meaning,
        "fsps_version":   getattr(_import_fsps(), "__version__", None),
        "fsps_kwargs":    dict(kwargs),
        "wave_min":       float(np.min(np.array(ssp_wave))),
        "wave_max":       float(np.max(np.array(ssp_wave))),
        "schema_version": SSP_SCHEMA_VERSION,
    }


def missing_stellar_mass_message(what="this grid") -> str:
    return (f"{what} carries no surviving stellar-mass table (ssp_stellar_mass, SSP schema "
            f"3.0), so mfrac and the surviving mass cannot be computed.  Add it to a COPY of the "
            f"grid file, without rebuilding the spectra, with\n"
            f"    python {STELLAR_MASS_SCRIPT} <grid.h5>\n"
            f"(needs python-fsps compiled with the grid's isochrones and $SPS_HOME).")


def fsps_stellar_mass_source(fsps_version) -> str:
    """Provenance string of a stellar-mass table read from FSPS."""
    import os
    return (f"FSPS StellarPopulation.stellar_mass (sfh=0, tage=0; stars + remnants per M_sun "
            f"formed), python-fsps {fsps_version}, SPS_HOME={os.environ.get('SPS_HOME')}")


def collect_ssp_data(**kwargs) -> typing.Tuple[jnp.ndarray, jnp.ndarray,
                                               jnp.ndarray, jnp.ndarray]:
    """Return ``(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux)`` for all backend metallicities and ages (units as in :class:`SSPData`)."""
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, _meta = \
        _collect_ssp_and_meta(**kwargs)
    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux


def collect_ssp_data_wrapper(**kwargs) -> SSPData:
    """Return a provenance-aware :class:`SSPData` built from the SPS backend."""
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, meta = \
        _collect_ssp_and_meta(**kwargs)
    return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, **meta)
