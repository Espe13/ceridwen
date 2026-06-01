"""Villaume, Conroy & Johnson (2015) circumstellar AGB dust shell.

This module adds the FSPS ``add_agb_dust_model=True`` physics to
ceridwen.  Defaults are matched to FSPS / python-fsps:

* ``add_agb_dust_model = True``
* ``agb_dust = 1.0``

so calling ``AGBDustShellModel(ssp_data).apply(ssp_data.ssp_flux)``
gives the SSP fluxes with the same AGB-dust treatment prospector /
FSPS use by default.

Algorithm
---------
FSPS implements the model per-star, on an isochrone:

  1. For every TP-AGB star, compute ``tau1`` (1-micron optical depth)
     from physical inputs (mass, T_eff, L, log g, mass-loss rate, C/O).
  2. Look up the DUSTY template ``flux_dagb(lambda, cstar, T_eff, tau1)``
     at that point.
  3. Heavy-smooth the bare-star spectrum at 1-3 micron, then multiply
     by the DUSTY template (which is in ``flux_out / flux_in`` units).
  4. Sum over the IMF / isochrone to get the SSP-level contribution.

Ceridwen does not carry isochrones at run time -- it consumes
pre-computed SSPs.  So a faithful implementation must source the
SSP-level correction from FSPS itself.  The cleanest way is to query
python-fsps once: compute the SSP fluxes with and without
``add_agb_dust_model`` for the same (zmet, age) grid and store the
ratio.  Applied multiplicatively to ceridwen's ``ssp_flux``, this
matches FSPS to round-off.

A linear-in-``agb_dust`` interpolation between the unmodified
template (``agb_dust=0``) and the full ``agb_dust=1.0`` correction is
implemented, which is exact in the dust-thin limit and a good
approximation elsewhere.  For a value-dependent fit the caller can
recompute the templates with ``agb_dust=`` set on the FSPS side at
template-build time.

Author: written for the ceridwen <-> prospector comparison.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Cache file naming
# --------------------------------------------------------------------------- #

def _default_cache_path(isoc_type: str, agb_dust: float,
                        n_wave: int, n_age: int, n_z: int) -> Path:
    """Cache file lives under ``$XDG_CACHE_HOME/ceridwen/`` if set,
    otherwise ``~/.cache/ceridwen/``.  The filename encodes the SSP
    grid shape so a different FSPS install / different isoc_type
    cannot accidentally pick up the wrong cache.
    """
    base = (Path(os.environ.get("XDG_CACHE_HOME",
                                 Path.home() / ".cache"))
            / "ceridwen" / "agb_dust")
    base.mkdir(parents=True, exist_ok=True)
    fname = (f"agb_dust_ratio_{isoc_type}_agb{agb_dust:.3f}_"
             f"nz{n_z}_nage{n_age}_nwave{n_wave}.h5")
    return base / fname


# --------------------------------------------------------------------------- #
# Template builder (uses python-fsps)
# --------------------------------------------------------------------------- #

def build_agb_ratio_via_fsps(
        ssp_data,
        agb_dust: float = 1.0,
        isoc_type: str = "bpss",
        cache_path: Optional[Path] = None,
        force_rebuild: bool = False,
        sps_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Compute the SSP-level AGB-dust-shell flux ratio.

    Builds two python-fsps ``StellarPopulation`` instances -- one with
    ``add_agb_dust_model=True`` and ``agb_dust=agb_dust``, one with
    ``add_agb_dust_model=False`` -- on the same isoc_type / spectral
    library / wavelength grid as ``ssp_data``, asks both for the
    spectrum at every metallicity in ``ssp_data.ssp_lgmet``, and
    returns the per-pixel ratio ``flux_with / flux_without`` as a
    ``(n_z, n_age, n_wave)`` numpy array on ``ssp_data.ssp_wave``.

    Cached to disk so a subsequent run loads in milliseconds.

    Parameters
    ----------
    ssp_data : ceridwen.ssps.ssp_data.SSPData
        The SSP grid to align with.  ``ssp_data.ssp_lgmet`` is
        ``log10(Z_abs)``; ``ssp_data.ssp_lg_age_gyr`` is
        ``log10(age/Gyr)``.
    agb_dust : float, default 1.0
        Scales the AGB dust optical depth (FSPS's ``agb_dust`` knob).
        Default 1.0 matches the FSPS / python-fsps default.
    isoc_type : str, default ``'bpss'``
        FSPS isochrone tag.  Must match what ``ssp_data`` was built
        with.  ``'bpss'`` => BPASS isochrones (the production
        ceridwen default).
    cache_path : Path, optional
        Override the default cache location.
    force_rebuild : bool, default False
        If True, ignore any existing cache and recompute.
    sps_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fsps.StellarPopulation``.
        Pass the same ones that were used when ``ssp_data`` was built,
        otherwise the ratios will be evaluated on a different SPS
        configuration and the cache key will silently mismatch.
    """
    n_z, n_age, n_wave = ssp_data.ssp_flux.shape
    if cache_path is None:
        cache_path = _default_cache_path(isoc_type, agb_dust,
                                          n_wave, n_age, n_z)
    if cache_path.is_file() and not force_rebuild:
        with h5py.File(cache_path, "r") as f:
            ratio = np.asarray(f["ratio"][:], dtype=np.float32)
            cached_w = np.asarray(f["ssp_wave"][:])
        if (ratio.shape == (n_z, n_age, n_wave)
                and np.allclose(cached_w,
                                 np.asarray(ssp_data.ssp_wave),
                                 rtol=1e-6, atol=1e-3)):
            return ratio
        # Shape / grid mismatch -- fall through and rebuild.

    try:
        import fsps                                        # python-fsps
    except ImportError as exc:                              # noqa: BLE001
        raise RuntimeError(
            "AGB-dust template builder requires python-fsps; "
            "install it (and an FSPS data directory at $SPS_HOME) "
            "before running ``build_agb_ratio_via_fsps``."
        ) from exc

    # python-fsps does NOT accept ``isoc_type`` as a constructor kwarg
    # -- the isochrone library is hard-baked into FSPS at compile time
    # (you pick it when you build libfsps).  The ``isoc_type`` argument
    # to this function is kept purely as a cache-key tag and a
    # consistency check: the caller passes the same string that
    # ssp_data was built with, and we sanity-check the resulting
    # libraries match.  Strip it before forwarding to StellarPopulation.
    sps_kwargs = dict(sps_kwargs or {})
    sps_kwargs.pop("isoc_type", None)
    sps_kwargs.setdefault("zcontinuous", 0)                 # discrete grid
    sps_kwargs.setdefault("sfh", 0)                         # SSP

    sp_on  = fsps.StellarPopulation(add_agb_dust_model=True,
                                     agb_dust=float(agb_dust),
                                     **sps_kwargs)
    sp_off = fsps.StellarPopulation(add_agb_dust_model=False,
                                     agb_dust=0.0,
                                     **sps_kwargs)

    # Sanity: the FSPS install must agree with the requested isoc_type.
    libs = [b.decode() if isinstance(b, bytes) else str(b)
            for b in (sp_on.libraries or [])]
    if isoc_type not in libs and isoc_type != "":
        print(f"  [AGB warn] requested isoc_type={isoc_type!r} but FSPS "
              f"libraries report {libs}; proceeding under the assumption "
              "that the compiled-in isochrones are correct.")

    # Quick sanity check: same n_z?
    if sp_on.zlegend.size != n_z:
        raise RuntimeError(
            f"FSPS reports {sp_on.zlegend.size} metallicities for "
            f"isoc_type={isoc_type!r}; ssp_data has {n_z}.  Cannot "
            "align AGB-dust templates."
        )

    # Pull native FSPS wavelength grid and the on/off SSP fluxes per
    # metallicity.  ``peraa=False`` returns Lsun/Hz, matching ceridwen.
    fsps_wave = None
    ratio = np.ones((n_z, n_age, n_wave), dtype=np.float64)
    for zmet_indx in range(1, n_z + 1):                     # FSPS is 1-based
        w_on,  flux_on  = sp_on.get_spectrum(tage=0.0, zmet=zmet_indx,
                                             peraa=False)
        w_off, flux_off = sp_off.get_spectrum(tage=0.0, zmet=zmet_indx,
                                              peraa=False)
        if fsps_wave is None:
            fsps_wave = np.asarray(w_on, dtype=np.float64)
        flux_on  = np.asarray(flux_on,  dtype=np.float64)  # (n_age, n_wave_fsps)
        flux_off = np.asarray(flux_off, dtype=np.float64)
        # ratio on FSPS grid, then interpolate to ceridwen's
        safe = flux_off > 0
        r_fsps = np.where(safe, flux_on / np.where(safe, flux_off, 1.0), 1.0)
        for a in range(flux_on.shape[0]):
            if a >= n_age:
                break
            ratio[zmet_indx - 1, a, :] = np.interp(
                np.asarray(ssp_data.ssp_wave, dtype=np.float64),
                fsps_wave, r_fsps[a, :],
                left=1.0, right=1.0,
            )

    # Persist
    try:
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("ratio", data=ratio.astype(np.float32))
            f.create_dataset("ssp_wave",
                              data=np.asarray(ssp_data.ssp_wave))
            f.create_dataset("ssp_lgmet",
                              data=np.asarray(ssp_data.ssp_lgmet))
            f.create_dataset("ssp_lg_age_gyr",
                              data=np.asarray(ssp_data.ssp_lg_age_gyr))
            f.attrs["isoc_type"] = isoc_type
            f.attrs["agb_dust"] = float(agb_dust)
            f.attrs["source"] = "fsps.StellarPopulation"
    except OSError:
        # Read-only filesystem: skip the cache silently.
        pass

    return ratio.astype(np.float32)


# --------------------------------------------------------------------------- #
# Runtime apply
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AGBDustShellModel:
    """Pre-computed FSPS Villaume+15 AGB dust shell, applied
    multiplicatively to ceridwen SSP fluxes.

    Construct with ``AGBDustShellModel.build(ssp_data)`` for the
    on-by-default ``agb_dust = 1.0`` configuration, or pass
    ``agb_dust=0`` to disable.  The ``apply`` method returns
    ``ssp_flux`` with the AGB dust correction folded in; it does
    NOT mutate the input array.
    """
    ratio: jnp.ndarray      # (n_z, n_age, n_wave), float32
    agb_dust: float         # the value used to build ``ratio``

    @classmethod
    def build(cls,
              ssp_data,
              agb_dust: float = 1.0,
              isoc_type: str = "bpss",
              cache_path: Optional[Path] = None,
              force_rebuild: bool = False,
              sps_kwargs: Optional[dict] = None) -> "AGBDustShellModel":
        """Build (or load from cache) the (Z, age, wave) ratio cube."""
        ratio = build_agb_ratio_via_fsps(
            ssp_data, agb_dust=agb_dust, isoc_type=isoc_type,
            cache_path=cache_path, force_rebuild=force_rebuild,
            sps_kwargs=sps_kwargs,
        )
        return cls(jnp.asarray(ratio, dtype=jnp.float32),
                    float(agb_dust))

    def apply(self, ssp_flux, agb_dust_runtime: Optional[float] = None):
        """Multiply ``ssp_flux`` by the AGB correction.

        ``ssp_flux`` must have shape ``(n_z, n_age, n_wave)`` and align
        with ``ssp_data`` used at build time.

        ``agb_dust_runtime`` linearly interpolates between the
        unmodified template (``agb_dust_runtime = 0`` -> identity)
        and the cached ``agb_dust`` template (``agb_dust_runtime =
        self.agb_dust``).  This is exact in the optically-thin limit
        and a useful first-order knob otherwise; for an exact
        ``agb_dust != self.agb_dust``, rebuild the cache.
        """
        if agb_dust_runtime is None or agb_dust_runtime == self.agb_dust:
            r = self.ratio
        else:
            scale = jnp.float32(agb_dust_runtime / self.agb_dust)
            r = jnp.float32(1.0) + scale * (self.ratio - jnp.float32(1.0))
        return jnp.asarray(ssp_flux) * r.astype(ssp_flux.dtype)


# --------------------------------------------------------------------------- #
# Bake the AGB correction into an SSPData grid
# --------------------------------------------------------------------------- #

def ssp_data_with_agb_dust(
        ssp_data,
        agb_dust: float = 1.0,
        isoc_type: str = "bpss",
        cache_path: Optional[Path] = None,
        force_rebuild: bool = False,
        sps_kwargs: Optional[dict] = None):
    """Return a new ``SSPData`` whose ``ssp_flux`` has the FSPS
    Villaume+15 AGB-dust-shell correction folded in.

    This is the recommended way to turn the AGB dust model on by
    default: bake the correction into the SSP grid at load time, and
    every downstream consumer (``CSPBasis``, observation projection,
    predictions) automatically uses the FSPS-equivalent fluxes without
    any further code changes.  CSPBasis does not need to know AGB
    physics was applied.

    Defaults match FSPS: ``add_agb_dust_model = True`` and
    ``agb_dust = 1.0`` are the python-fsps defaults, so calling
    ``ssp_data_with_agb_dust(ssp_data)`` with no further arguments
    gives the same SSP fluxes a fresh ``fsps.StellarPopulation(
    isoc_type='bpss')`` would produce.

    Parameters
    ----------
    ssp_data : ceridwen.ssps.ssp_data.SSPData
        The bare-star SSP grid to modify.
    agb_dust : float, default 1.0
        FSPS's ``agb_dust`` knob (scales the optical depth).
    isoc_type : str, default ``'bpss'``
        FSPS isochrone tag.
    cache_path, force_rebuild, sps_kwargs :
        Forwarded to ``build_agb_ratio_via_fsps``.

    Returns
    -------
    SSPData
        New ``SSPData`` with ``ssp_flux`` multiplied by the AGB ratio.
        All other fields are preserved bit-for-bit from the input.
    """
    # Local import so this module remains importable without ceridwen
    # installed (e.g. in unit tests / linting).
    from ceridwen.ssps.ssp_data import SSPData

    ratio = build_agb_ratio_via_fsps(
        ssp_data, agb_dust=agb_dust, isoc_type=isoc_type,
        cache_path=cache_path, force_rebuild=force_rebuild,
        sps_kwargs=sps_kwargs,
    )
    new_flux = jnp.asarray(ssp_data.ssp_flux) * jnp.asarray(
        ratio, dtype=ssp_data.ssp_flux.dtype)
    return SSPData(
        ssp_lgmet      = ssp_data.ssp_lgmet,
        ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr,
        ssp_wave       = ssp_data.ssp_wave,
        ssp_flux       = new_flux,
    )


__all__ = [
    "AGBDustShellModel",
    "build_agb_ratio_via_fsps",
    "ssp_data_with_agb_dust",
]
