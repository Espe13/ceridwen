"""
Module for handling Simple Stellar Population (SSP) data.

This module provides utilities to retrieve, store, and manage SSP data
from FSPS in a JAX-friendly form.  The core component is the
:class:`SSPData` frozen dataclass which holds the interpolation grids
needed to build composite stellar populations.

SSPs represent the integrated light from a single burst of star
formation with uniform metallicity and age.  They serve as building
blocks for the more complex stellar populations used in SED fitting.

Provenance
----------
Grids built with :meth:`SSPData.from_fsps` record *how* they were made
(isochrone / spectral library, IMF, FSPS version, the exact FSPS kwargs
used, wavelength range, and a schema tag).  This metadata is stored as
plain-Python static fields on :class:`SSPData` — it is never a JAX array
leaf and never enters a ``@jit`` kernel — and is persisted to / restored
from the HDF5 attrs.

Library resolution (schema 2.0)
-------------------------------
Every grid carries the intrinsic spectral resolution of its stellar
library as a per-pixel Gaussian dispersion in velocity units,
``ssp_resolution`` = sigma_v(lambda) [km/s] on ``ssp_wave`` (velocity
units are redshift-invariant, so the observation layer needs no frame
bookkeeping).  Grids written by :meth:`SSPData.from_fsps` or by
``scripts/convert_grids_schema2.py`` build this curve automatically as
the element-wise maximum of the grid's own 2-pixel sampling floor
(derived from ``ssp_wave`` itself) and any documented library LSF
segments, so their curves are finite at every pixel.  NaN marks pixels
where the resolution is unknown (possible only in hand-built curves);
the Spectrum projection subtracts nothing there.  The observation layer
subtracts this curve in quadrature from the target instrumental
resolution automatically, so models are never over-broadened by the
library width.  Schema 2.0 files REQUIRE this dataset: :meth:`load`
raises on files without it (convert old grids with
``scripts/convert_grids_schema2.py``) and :meth:`save` refuses to write
a grid whose curve is missing.

Only the kwargs that legitimately define the stellar library / IMF are
accepted by :meth:`from_fsps`; anything the composite-stellar-population
forward model (:class:`ceridwen.csp.CSPBasis`) applies itself — star
formation history, dust, nebular emission, IGM, redshift, LOSVD
smoothing, or a fixed metallicity — is rejected with a clear error, so
the SSP grid can never be silently double-processed.

Note on ``log_qq``
------------------
This module does not store a precomputed ionising photon rate
``log_qq`` per (Z, age).  :class:`ceridwen.neb.NebularGridModel.NebularModel`
derives ``log_qq`` directly from ``ssp_flux`` at construction time, so the
value is always self-consistent with the SSP grid in use and with FSPS's
own run-time formula.  Legacy HDF5 files that contain a ``log_qq`` dataset
are loaded transparently — the field is simply ignored.
"""

import json
import typing
import h5py
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional

# Import FSPS lazily; only the generator paths need it.
try:
    import fsps
    HAS_FSPS = True
except (ImportError, RuntimeError):
    HAS_FSPS = False


# Default filename for cached SSP data.
DEFAULT_SSP_BNAME = "ssp_data_fsps_v3.2_lgmet_age.h5"

# Bumped whenever the on-disk metadata schema changes.
# 2.0 (2026-08): ssp_resolution (library sigma_v(lambda) [km/s]) is a
# REQUIRED dataset; loaders reject schema-1.x files.
SSP_SCHEMA_VERSION = "2.0"


# ----------------------------------------------------------------------
# FSPS kwarg policy for SSP-grid construction
# ----------------------------------------------------------------------
# A from_fsps grid may only carry the parameters that define the *stellar
# library and IMF* — the things that legitimately belong at SSP-build time.
# Everything the CSP forward model applies itself (SFH, dust, nebular
# emission, IGM, redshift, LOSVD smoothing, a fixed metallicity) is rejected,
# because baking it into the grid would double-process it or make the grid
# inconsistent with :class:`ceridwen.csp.CSPBasis`.
#
# NB: the isochrone / spectral library itself is NOT a runtime kwarg — it is
# compiled into libfsps and read back as provenance from ``ssp.libraries``.

# IMF shape.
_IMF_KWARGS = frozenset({
    "imf_type", "imf1", "imf2", "imf3",
    "imf_lower_limit", "imf_upper_limit", "vdmc", "mdave",
})

# Stellar-evolution / isochrone-phase knobs that shape the SSP itself and are
# NOT touched by the CSP forward model.
_LIBRARY_KWARGS = frozenset({
    "tpagb_norm_type", "agb", "pagb", "redgb", "fbhb", "sbss",
    "delt", "dell", "evtype", "masscut", "use_wr_spectra",
    "logt_wmb_hot", "add_stellar_remnants", "fcstar",
})

#: The complete whitelist of kwargs :meth:`SSPData.from_fsps` accepts.
LIBRARY_IMF_KWARGS = _IMF_KWARGS | _LIBRARY_KWARGS


def _owned(names, mechanism):
    return {n: mechanism for n in names}


# Maps a rejected FSPS kwarg -> the CSP mechanism that already handles it,
# used to build an informative error.  Any FSPS param that is neither
# whitelisted nor listed here still gets rejected, with a generic message.
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
        "FSPS's discrete zlegend); CSPBasis samples Z from the grid, so a "
        "fixed metallicity must not be set at build time",
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
    """
    Reject any FSPS kwarg that does not define the stellar library / IMF.

    Parameters
    ----------
    kwargs : dict
        The kwargs the caller wants to forward to ``fsps.StellarPopulation``.

    Returns
    -------
    dict
        ``kwargs`` unchanged, once every key is confirmed to be a
        library/IMF-defining parameter.

    Raises
    ------
    ValueError
        On the first disallowed kwarg, naming it and (when known) the CSP
        mechanism that already owns it.
    """
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
    """
    Immutable container for the SSP interpolation grids (+ provenance).

    Attributes
    ----------
    ssp_lgmet : jnp.ndarray, shape (n_met,)
        ``log10`` of the absolute metallicity grid.  ``Z`` is the mass
        fraction of elements heavier than helium.  Typical range
        ~-2.3 to +0.2 dex.
    ssp_lg_age_gyr : jnp.ndarray, shape (n_ages,)
        ``log10(age / Gyr)``.
    ssp_wave : jnp.ndarray, shape (n_wave,)
        Wavelength grid in Angstroms.
    ssp_flux : jnp.ndarray, shape (n_met, n_ages, n_wave)
        SSP flux density in ``Lsun / Hz`` per Msun of initial stellar
        mass.

    isoc_type : str or None
        Isochrone library the grid was built with (e.g. ``'mist'``), read
        from FSPS's compiled-in library set.  ``None`` for legacy grids.
        :class:`ceridwen.csp.CSPBasis` uses this to pick the matching
        nebular CLOUDY grid automatically.
    spec_library : str or None
        Spectral library (e.g. ``'miles'``).  ``None`` for legacy grids.
    imf_type : int or None
        FSPS IMF selector the grid was built with.  ``None`` for legacy.
    fsps_version : str or None
        ``python-fsps`` version string used to build the grid.
    fsps_kwargs : dict
        The (whitelisted) FSPS build kwargs actually used.  ``{}`` for
        legacy grids.
    wave_min, wave_max : float or None
        Wavelength range (Å) of ``ssp_wave`` at build time.
    schema_version : str or None
        On-disk metadata schema tag.  ``None`` for legacy grids.

    Notes
    -----
    The provenance fields are ordinary Python objects (str / int / dict /
    None); they are never JAX arrays and never enter a ``@jit`` kernel.
    They are excluded from equality/hashing (``compare=False``).

    No ``log_qq`` table is stored; the nebular model computes the ionising
    photon rate internally.  HDF5 files that contain a ``log_qq`` dataset
    are loaded transparently — the field is simply ignored.
    """

    ssp_lgmet: jnp.ndarray          # log10 absolute metallicity grid
    ssp_lg_age_gyr: jnp.ndarray     # log10(age / Gyr)
    ssp_wave: jnp.ndarray           # wavelength grid (Angstrom)
    ssp_flux: jnp.ndarray           # (n_met, n_ages, n_wave) in Lsun/Hz/Msun

    # --- library resolution (schema 2.0; static numpy, not a JAX leaf) ----
    # sigma_v(lambda) [km/s] on ssp_wave; NaN = unknown at that pixel.
    # Optional at the CONSTRUCTOR level only (intermediate in-memory
    # objects); save()/load() REQUIRE it, so every on-disk grid carries it.
    ssp_resolution: Optional[np.ndarray] = field(default=None, compare=False)
    resolution_source: Optional[str] = field(default=None, compare=False)

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
        """Validate grid consistency."""
        if self.ssp_flux.shape != (self.ssp_lgmet.size,
                                   self.ssp_lg_age_gyr.size,
                                   self.ssp_wave.size):
            raise ValueError(
                f"SSP flux grid shape mismatch: expected "
                f"({self.ssp_lgmet.size}, {self.ssp_lg_age_gyr.size}, "
                f"{self.ssp_wave.size}) but got {self.ssp_flux.shape}.  "
                f"Grid dimensions must be consistent (n_met, n_ages, n_wave)."
            )
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
            # normalise storage to a plain float64 numpy array
            object.__setattr__(self, "ssp_resolution", res)

    # ------------------------------------------------------------------
    # Library resolution attachment
    # ------------------------------------------------------------------
    def with_resolution(self, *, sigma_v=None, segments=None, source=None):
        """Return a copy carrying the library resolution curve.

        Exactly one of ``sigma_v`` (a per-pixel sigma_v(lambda) [km/s]
        array on ``ssp_wave``, NaN where unknown) or ``segments`` (a
        piecewise spec understood by
        :func:`ceridwen.ssps.library_resolution.sigma_v_from_segments`)
        must be given.  ``source`` is a short free-text provenance note
        (e.g. the literature reference for the numbers) stored alongside.
        """
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

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    def display(self, *, return_str: bool = False, file=None):
        """Print a summary of the grid and its provenance.

        Intended as a sanity check: call it right after ``from_fsps`` or
        ``load`` to confirm the isochrone set, spectral library, IMF, and
        grid coverage are what you expect before you build a ``CSPBasis``.
        Purely diagnostic — none of this is touched by the forward model.

        Parameters
        ----------
        return_str : bool, optional
            Return the formatted string instead of printing it.  Default False.
        file : file-like, optional
            Destination for the print (default ``sys.stdout``).

        Returns
        -------
        str or None
            The formatted string if ``return_str=True``, else ``None``.
        """
        import sys as _sys

        lgmet = np.asarray(self.ssp_lgmet)
        lgage = np.asarray(self.ssp_lg_age_gyr)
        wave  = np.asarray(self.ssp_wave)
        n_met, n_age, n_wave = self.ssp_flux.shape
        age_gyr = 10.0 ** lgage

        def _fmt(v, na="—"):
            return na if v is None else str(v)

        # Human-readable in-memory size of the (dominant) flux array.
        size = float(np.asarray(self.ssp_flux).nbytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                size_str = f"{size:.1f} {unit}"
                break
            size /= 1024.0

        lines = [
            "SSPData",
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
            f"  metallicity  log10 Z     : {n_met:>4d} pts   "
            f"[{lgmet.min():+.3f}, {lgmet.max():+.3f}]  (absolute Z, NOT Z/Zsun)",
            f"  age          log10(Gyr)  : {n_age:>4d} pts   "
            f"[{lgage.min():+.3f}, {lgage.max():+.3f}]  "
            f"= [{age_gyr.min():.3g}, {age_gyr.max():.3g}] Gyr",
            f"  wavelength   Angstrom    : {n_wave:>4d} pts   "
            f"[{wave.min():.1f}, {wave.max():.1f}]",
            f"  flux (n_met,n_age,n_wave): {tuple(int(s) for s in self.ssp_flux.shape)}  "
            f"[L_sun Hz^-1 M_sun^-1]  {np.asarray(self.ssp_flux).dtype}  {size_str}",
        ]
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
        if self.isoc_type is None:
            lines += [
                "note",
                "  isoc_type is None (legacy grid, built before provenance "
                "tracking):",
                "  CSPBasis will warn and fall back to 'mist' for the nebular grid.",
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
        """
        Serialise the SSP grids (and provenance metadata) to HDF5.

        Schema 2.0: the library resolution curve is REQUIRED — a grid
        without ``ssp_resolution`` cannot be written (attach one with
        :meth:`with_resolution` first).  Provenance fields are written to
        the file ``attrs``; any that are ``None`` are omitted.
        ``fsps_kwargs`` is stored as a JSON string.

        Parameters
        ----------
        filename : str or Path
            Output file path.  Will be overwritten if it exists.
        """
        if self.ssp_resolution is None:
            raise ValueError(
                "SSPData.save(): this grid carries no library resolution "
                "curve (ssp_resolution is None).  Schema 2.0 files require "
                "one — attach it with with_resolution(segments=...) or "
                "with_resolution(sigma_v=...) before saving."
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
            f.attrs['units_lgmet']        = 'log10(absolute_metallicity)'
            f.attrs['units_lg_age_gyr']   = 'log10(age/Gyr)'
            f.attrs['units_wave']         = 'Angstrom'
            f.attrs['units_flux']         = 'L_sun Hz^-1 M_sun^-1'
            f.attrs['units_resolution']   = 'sigma_v [km/s]; NaN = unknown'
            if self.resolution_source is not None:
                f.attrs['resolution_source'] = str(self.resolution_source)

            # --- provenance -------------------------------------------------
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
            # Always record the build-kwargs dict (possibly empty) as JSON.
            f.attrs['fsps_kwargs_json'] = json.dumps(self.fsps_kwargs or {})

    @classmethod
    def load(cls, filename):
        """
        Load an :class:`SSPData` from an HDF5 file (schema 2.0).

        The file MUST carry the library resolution dataset
        ``ssp_resolution`` (sigma_v(lambda) [km/s], NaN where unknown);
        files written under schema 1.x raise with a pointer to the
        converter.  Any legacy ``log_qq`` dataset is silently ignored —
        the nebular model computes its own ionising-photon rate from
        ``ssp_flux``.
        """
        def _decode(v):
            if isinstance(v, (bytes, bytearray)):
                return v.decode()
            return v

        with h5py.File(filename, 'r') as f:
            if 'ssp_resolution' not in f:
                raise ValueError(
                    f"{filename}: no 'ssp_resolution' dataset — this grid "
                    f"predates SSP schema 2.0.  Convert it (no FSPS rebuild "
                    f"needed) with scripts/convert_grids_schema2.py, which "
                    f"copies the existing arrays and attaches the library "
                    f"resolution curve."
                )
            ssp_lgmet      = jnp.array(f['ssp_lgmet'][:])
            ssp_lg_age_gyr = jnp.array(f['ssp_lg_age_gyr'][:])
            ssp_wave       = jnp.array(f['ssp_wave'][:])
            ssp_flux       = jnp.array(f['ssp_flux'][:])
            ssp_resolution = np.asarray(f['ssp_resolution'][:],
                                        dtype=np.float64)

            a = f.attrs
            meta = {
                'ssp_resolution':    ssp_resolution,
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
            }
            if 'fsps_kwargs_json' in a:
                meta['fsps_kwargs'] = json.loads(_decode(a['fsps_kwargs_json']))
            else:
                meta['fsps_kwargs'] = {}

        return cls(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, **meta)

    # ------------------------------------------------------------------
    # Generation from FSPS
    # ------------------------------------------------------------------
    @classmethod
    def from_fsps(cls, save_to: Optional[str] = None,
                  resolution_segments=None,
                  resolution_source: Optional[str] = None,
                  **fsps_kwargs) -> "SSPData":
        """
        Build an :class:`SSPData` directly from FSPS, recording provenance.

        Schema 2.0: the library resolution curve is built AUTOMATICALLY as
        the element-wise maximum of the grid's own 2-pixel sampling floor
        — derived from the built ``ssp_wave`` itself, no external numbers
        needed (:func:`ceridwen.ssps.library_resolution.sampling_floor_sigma_v`)
        — and, when given, ``resolution_segments`` describing the parent
        library's documented line-spread function (the piecewise spec of
        :func:`ceridwen.ssps.library_resolution.sigma_v_from_segments`).
        Supply segments only when the library LSF is broader than the
        stored sampling (e.g. MILES: ``[(3525., 7500., 'fwhm_AA', 2.54)]``,
        Falcon-Barroso et al. 2011) and cite them via ``resolution_source``;
        for libraries whose only documented resolution IS their tabulation
        (e.g. BPASS at 1 Angstrom) pass nothing — the sampling floor is
        the honest curve.  The stored curve is finite at every pixel.

        Only kwargs that define the **stellar library / IMF** are accepted —
        the things that legitimately belong at SSP-build time.  Anything the
        CSP forward model applies itself (star-formation history, dust,
        nebular emission, IGM, redshift, LOSVD smoothing, or a fixed
        metallicity) raises :class:`ValueError`, so the grid can never be
        silently double-processed or made inconsistent with
        :class:`ceridwen.csp.CSPBasis`.  ``zcontinuous`` and ``sfh`` are
        fixed internally (the grid is built on FSPS's discrete ``zlegend``
        metallicity points).

        Parameters
        ----------
        save_to : str or Path, optional
            If given, the result is also persisted to this path via
            :meth:`save` so subsequent runs can use :meth:`load`.
        **fsps_kwargs
            Forwarded to :class:`fsps.StellarPopulation`.  Allowed keys are
            the IMF parameters (``imf_type``, ``imf1``, ``imf2``, ``imf3``,
            ``imf_lower_limit``, ``imf_upper_limit``, ``vdmc``, ``mdave``)
            and stellar-evolution / isochrone-phase knobs (``tpagb_norm_type``,
            ``agb``, ``pagb``, ``redgb``, ``fbhb``, ``sbss``, ``delt``,
            ``dell``, ``evtype``, ``masscut``, ``use_wr_spectra``,
            ``logt_wmb_hot``, ``add_stellar_remnants``, ``fcstar``).
            Common choice: ``imf_type=1`` (Chabrier).

        Raises
        ------
        ValueError
            If any kwarg is not a library/IMF-defining parameter, or if
            ``resolution_source`` is given without ``resolution_segments``
            (the floor's provenance is generated automatically; a source
            note only makes sense for supplied LSF numbers).
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


# ----------------------------------------------------------------------
# FSPS-driven generators
# ----------------------------------------------------------------------
def _collect_ssp_and_meta(**kwargs):
    """
    Build the FSPS SSP grid and capture its provenance.

    Validates ``kwargs`` against the library/IMF whitelist first (so an
    unsafe kwarg is rejected even when FSPS is not installed), then builds
    the discrete-metallicity SSP grid and reads back provenance metadata.

    Returns
    -------
    (ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, meta) where ``meta`` is
    a dict of the :class:`SSPData` provenance fields.
    """
    kwargs = _validate_fsps_kwargs(kwargs)

    if not HAS_FSPS:
        raise ImportError(
            "FSPS is required for SSP data generation but is not available. "
            "See https://dfm.io/python-fsps/current/installation/"
        )

    # Discrete-metallicity SSP grid: zcontinuous=0 builds on FSPS's zlegend
    # points; sfh=0 is SSP mode.  Both are fixed here, never caller-supplied.
    ssp = fsps.StellarPopulation(zcontinuous=0, sfh=0, **kwargs)

    ssp_lgmet      = jnp.log10(ssp.zlegend)            # absolute log Z
    nzmet          = ssp_lgmet.size
    ssp_lg_age_gyr = ssp.log_age - 9.0                  # log(age/yr) -> log(age/Gyr)

    spectrum_collector = []
    for zmet_indx in range(1, nzmet + 1):              # FSPS is 1-based
        print(f"...retrieving metallicity {zmet_indx}/{nzmet} "
              f"[Z = {ssp.zlegend[zmet_indx-1]:.4f}]")
        _wave, _fluxes = ssp.get_spectrum(tage=0.0, zmet=zmet_indx, peraa=False)
        spectrum_collector.append(_fluxes)

    ssp_wave       = jnp.array(_wave)
    ssp_flux       = jnp.array(spectrum_collector)
    ssp_lg_age_gyr = jnp.array(ssp_lg_age_gyr)

    meta = _read_fsps_provenance(ssp, kwargs, ssp_wave)
    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, meta


def _read_fsps_provenance(ssp, kwargs: dict, ssp_wave) -> dict:
    """Extract static provenance metadata from a built StellarPopulation."""
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

    return {
        "isoc_type":      isoc_type,
        "spec_library":   spec_library,
        "imf_type":       imf_type,
        "fsps_version":   getattr(fsps, "__version__", None),
        "fsps_kwargs":    dict(kwargs),
        "wave_min":       float(np.min(np.array(ssp_wave))),
        "wave_max":       float(np.max(np.array(ssp_wave))),
        "schema_version": SSP_SCHEMA_VERSION,
    }


def collect_ssp_data(**kwargs) -> typing.Tuple[jnp.ndarray, jnp.ndarray,
                                               jnp.ndarray, jnp.ndarray]:
    """
    Retrieve SSP spectra from FSPS for all available metallicities and
    ages.

    Only stellar-library / IMF-defining kwargs are accepted (see
    :meth:`SSPData.from_fsps`); anything owned by the CSP forward model
    raises :class:`ValueError`.  ``sfh=0`` / ``zcontinuous=0`` are fixed
    internally.

    Returns
    -------
    ssp_lgmet : jnp.ndarray, shape (n_met,)
        ``log10`` of the absolute metallicity grid.
    ssp_lg_age_gyr : jnp.ndarray, shape (n_ages,)
        ``log10(age / Gyr)``.
    ssp_wave : jnp.ndarray, shape (n_wave,)
        Wavelength grid (Angstrom).
    ssp_flux : jnp.ndarray, shape (n_met, n_ages, n_wave)
        SSP flux in ``Lsun / Hz`` per Msun.
    """
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, _meta = \
        _collect_ssp_and_meta(**kwargs)
    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux


def collect_ssp_data_wrapper(**kwargs) -> SSPData:
    """
    High-level helper: generate a provenance-aware :class:`SSPData` from FSPS.

    Only stellar-library / IMF-defining kwargs are accepted (see
    :meth:`SSPData.from_fsps`).
    """
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, meta = \
        _collect_ssp_and_meta(**kwargs)
    return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, **meta)
