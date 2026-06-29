"""
Module for handling Simple Stellar Population (SSP) data.

This module provides utilities to retrieve, store, and manage SSP data
from FSPS in a JAX-friendly form.  The core component is the
:class:`SSPData` frozen dataclass which holds the interpolation grids
needed to build composite stellar populations.

SSPs represent the integrated light from a single burst of star
formation with uniform metallicity and age.  They serve as building
blocks for the more complex stellar populations used in SED fitting.

Note on ``log_qq``
------------------
This module does not store a precomputed ionising photon rate
``log_qq`` per (Z, age).  :class:`ceridwen.neb.NebularGridModel.NebularModel`
derives ``log_qq`` directly from ``ssp_flux`` at construction time, so the
value is always self-consistent with the SSP grid in use and with FSPS's
own run-time formula.  Legacy HDF5 files that contain a ``log_qq`` dataset
are loaded transparently — the field is simply ignored.
"""

import typing
import h5py
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

# Import FSPS lazily; only the generator paths need it.
try:
    import fsps
    HAS_FSPS = True
except (ImportError, RuntimeError):
    HAS_FSPS = False


# Default filename for cached SSP data.
DEFAULT_SSP_BNAME = "ssp_data_fsps_v3.2_lgmet_age.h5"


@dataclass(frozen=True)
class SSPData:
    """
    Immutable container for the SSP interpolation grids.

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

    Notes
    -----
    No ``log_qq`` table is stored; the nebular model computes the ionising
    photon rate internally.  HDF5 files that contain a ``log_qq`` dataset
    are loaded transparently — the field is simply ignored.
    """

    ssp_lgmet: jnp.ndarray          # log10 absolute metallicity grid
    ssp_lg_age_gyr: jnp.ndarray     # log10(age / Gyr)
    ssp_wave: jnp.ndarray           # wavelength grid (Angstrom)
    ssp_flux: jnp.ndarray           # (n_met, n_ages, n_wave) in Lsun/Hz/Msun

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

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------
    def save(self, filename):
        """
        Serialise the SSP grids to HDF5.

        Parameters
        ----------
        filename : str or Path
            Output file path.  Will be overwritten if it exists.
        """
        with h5py.File(filename, 'w') as f:
            f.create_dataset('ssp_lgmet',      data=np.array(self.ssp_lgmet))
            f.create_dataset('ssp_lg_age_gyr', data=np.array(self.ssp_lg_age_gyr))
            f.create_dataset('ssp_wave',       data=np.array(self.ssp_wave))
            f.create_dataset('ssp_flux',       data=np.array(self.ssp_flux))

            f.attrs['description']        = 'FSPS SSP interpolation grids'
            f.attrs['units_lgmet']        = 'log10(absolute_metallicity)'
            f.attrs['units_lg_age_gyr']   = 'log10(age/Gyr)'
            f.attrs['units_wave']         = 'Angstrom'
            f.attrs['units_flux']         = 'L_sun Hz^-1 M_sun^-1'

    @classmethod
    def load(cls, filename):
        """
        Load an SSPData from an HDF5 file.

        Any legacy ``log_qq`` dataset is silently ignored — the nebular
        model computes its own ionising-photon rate from ``ssp_flux``.
        """
        with h5py.File(filename, 'r') as f:
            ssp_lgmet      = jnp.array(f['ssp_lgmet'][:])
            ssp_lg_age_gyr = jnp.array(f['ssp_lg_age_gyr'][:])
            ssp_wave       = jnp.array(f['ssp_wave'][:])
            ssp_flux       = jnp.array(f['ssp_flux'][:])
        return cls(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux)

    # ------------------------------------------------------------------
    # Generation from FSPS
    # ------------------------------------------------------------------
    @classmethod
    def from_fsps(cls, save_to: Optional[str] = None,
                  **fsps_kwargs) -> "SSPData":
        """
        Build an :class:`SSPData` directly from FSPS.

        Parameters
        ----------
        save_to : str or Path, optional
            If given, the result is also persisted to this path via
            :meth:`save` so subsequent runs can use :meth:`load`.
        **fsps_kwargs
            Forwarded to :class:`fsps.StellarPopulation`.  Common
            choices: ``imf_type=1`` (Chabrier).
        """
        data = collect_ssp_data_wrapper(**fsps_kwargs)
        if save_to is not None:
            data.save(save_to)
        return data


# ----------------------------------------------------------------------
# FSPS-driven generators
# ----------------------------------------------------------------------
def collect_ssp_data(**kwargs) -> typing.Tuple[jnp.ndarray, jnp.ndarray,
                                               jnp.ndarray, jnp.ndarray]:
    """
    Retrieve SSP spectra from FSPS for all available metallicities and
    ages.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`fsps.StellarPopulation`.  ``sfh=0``
        (SSP mode) is fixed internally; everything else is at the
        caller's discretion.

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
    if not HAS_FSPS:
        raise ImportError(
            "FSPS is required for SSP data generation but is not available. "
            "See https://dfm.io/python-fsps/current/installation/"
        )

    # Discrete-metallicity SSP grid (sfh=0).
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

    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux


def collect_ssp_data_wrapper(**kwargs) -> SSPData:
    """
    High-level helper: generate an :class:`SSPData` from FSPS.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`fsps.StellarPopulation`.
    """
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux = collect_ssp_data(**kwargs)
    return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux)
