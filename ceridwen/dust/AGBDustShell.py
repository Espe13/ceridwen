"""Villaume, Conroy & Johnson (2015) circumstellar AGB dust shell as a
multiplicative (n_z, n_age, n_wave) correction to SSP fluxes."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import jax.numpy as jnp



def _default_cache_path(isoc_type: str, agb_dust: float,
                        n_wave: int, n_age: int, n_z: int) -> Path:
    """Cache path under $XDG_CACHE_HOME/ceridwen (default ~/.cache/ceridwen)."""
    base = (Path(os.environ.get("XDG_CACHE_HOME",
                                 Path.home() / ".cache"))
            / "ceridwen" / "agb_dust")
    base.mkdir(parents=True, exist_ok=True)
    fname = (f"agb_dust_ratio_{isoc_type}_agb{agb_dust:.3f}_"
             f"nz{n_z}_nage{n_age}_nwave{n_wave}.h5")
    return base / fname



def build_agb_ratio_via_fsps(
        ssp_data,
        agb_dust: float = 1.0,
        isoc_type: str = "bpss",
        cache_path: Optional[Path] = None,
        force_rebuild: bool = False,
        sps_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Return the AGB-dust flux ratio (with / without shell), shape
    (n_z, n_age, n_wave) float32 on ``ssp_data.ssp_wave``; disk-cached.

    Parameters
    ----------
    agb_dust : float -- optical-depth scale of the shell
    isoc_type : str -- isochrone tag used only as cache key / consistency check
    sps_kwargs : dict -- must match those ``ssp_data`` was built with
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

    try:
        import fsps
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "AGB-dust template builder requires python-fsps; "
            "install it (and an FSPS data directory at $SPS_HOME) "
            "before running ``build_agb_ratio_via_fsps``."
        ) from exc

    sps_kwargs = dict(sps_kwargs or {})
    sps_kwargs.pop("isoc_type", None)
    sps_kwargs.setdefault("zcontinuous", 0)
    sps_kwargs.setdefault("sfh", 0)

    sp_on  = fsps.StellarPopulation(add_agb_dust_model=True,
                                     agb_dust=float(agb_dust),
                                     **sps_kwargs)
    sp_off = fsps.StellarPopulation(add_agb_dust_model=False,
                                     agb_dust=0.0,
                                     **sps_kwargs)

    libs = [b.decode() if isinstance(b, bytes) else str(b)
            for b in (sp_on.libraries or [])]
    if isoc_type not in libs and isoc_type != "":
        print(f"  [AGB warn] requested isoc_type={isoc_type!r} but FSPS "
              f"libraries report {libs}; proceeding under the assumption "
              "that the compiled-in isochrones are correct.")

    if sp_on.zlegend.size != n_z:
        raise RuntimeError(
            f"FSPS reports {sp_on.zlegend.size} metallicities for "
            f"isoc_type={isoc_type!r}; ssp_data has {n_z}.  Cannot "
            "align AGB-dust templates."
        )

    fsps_wave = None
    ssp_wave_np = np.asarray(ssp_data.ssp_wave, dtype=np.float64)
    ratio = np.ones((n_z, n_age, n_wave), dtype=np.float64)
    for zmet_indx in range(1, n_z + 1):  # zmet is 1-based
        w_on,  flux_on  = sp_on.get_spectrum(tage=0.0, zmet=zmet_indx,
                                             peraa=False)
        w_off, flux_off = sp_off.get_spectrum(tage=0.0, zmet=zmet_indx,
                                              peraa=False)
        if fsps_wave is None:
            fsps_wave = np.asarray(w_on, dtype=np.float64)
        flux_on  = np.asarray(flux_on,  dtype=np.float64)
        flux_off = np.asarray(flux_off, dtype=np.float64)
        safe = flux_off > 0
        r_fsps = np.where(safe, flux_on / np.where(safe, flux_off, 1.0), 1.0)
        for a in range(flux_on.shape[0]):
            if a >= n_age:
                break
            ratio[zmet_indx - 1, a, :] = np.interp(
                ssp_wave_np,
                fsps_wave, r_fsps[a, :],
                left=1.0, right=1.0,
            )

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
        pass

    return ratio.astype(np.float32)



@dataclass(frozen=True)
class AGBDustShellModel:
    """Pre-computed AGB dust shell correction applied multiplicatively to SSP fluxes.

    Parameters
    ----------
    ratio : array (n_z, n_age, n_wave), float32 -- flux with / without shell
    agb_dust : float -- optical-depth scale ``ratio`` was built at
    """
    ratio: jnp.ndarray
    agb_dust: float

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
        """Return ``ssp_flux`` (n_z, n_age, n_wave) times the correction;
        ``agb_dust_runtime`` interpolates linearly between identity (0) and ``self.agb_dust``."""
        if agb_dust_runtime is None or agb_dust_runtime == self.agb_dust:
            r = self.ratio
        else:
            scale = jnp.float32(agb_dust_runtime / self.agb_dust)
            r = jnp.float32(1.0) + scale * (self.ratio - jnp.float32(1.0))
        return jnp.asarray(ssp_flux) * r.astype(ssp_flux.dtype)



def ssp_data_with_agb_dust(
        ssp_data,
        agb_dust: float = 1.0,
        isoc_type: str = "bpss",
        cache_path: Optional[Path] = None,
        force_rebuild: bool = False,
        sps_kwargs: Optional[dict] = None):
    """Return a new ``SSPData`` with the AGB-dust ratio folded into ``ssp_flux``;
    all other fields are preserved (arguments forwarded to ``build_agb_ratio_via_fsps``)."""
    from ceridwen.ssps.ssp_data import SSPData

    ratio = build_agb_ratio_via_fsps(
        ssp_data, agb_dust=agb_dust, isoc_type=isoc_type,
        cache_path=cache_path, force_rebuild=force_rebuild,
        sps_kwargs=sps_kwargs,
    )
    new_flux = jnp.asarray(ssp_data.ssp_flux) * jnp.asarray(
        ratio, dtype=ssp_data.ssp_flux.dtype)
    import dataclasses as _dc
    return _dc.replace(ssp_data, ssp_flux=new_flux)


__all__ = [
    "AGBDustShellModel",
    "build_agb_ratio_via_fsps",
    "ssp_data_with_agb_dust",
]
