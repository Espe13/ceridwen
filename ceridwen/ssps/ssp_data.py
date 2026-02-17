"""
Module for handling Simple Stellar Population (SSP) data.

This module provides utilities to retrieve, store, and manage SSP data from FSPS
(Flexible Stellar Population Synthesis) in a JAX-optimized format. The core component
is the SSPData frozen dataclass which holds interpolation grids for stellar population
spectra as a function of metallicity and age.

SSPs represent the integrated light from a single burst of star formation with uniform
metallicity and age. These serve as building blocks for more complex stellar population
models in astronomical spectral energy distribution (SED) fitting.

Key Features:
- Immutable SSPData storage with HDF5 persistence
- JAX-compatible arrays for GPU acceleration  
- Automatic addition of zero-age SSP (log age = -6) not available in FSPS
- Ionizing photon rate calculations for nebular modeling

Globals:
--------
DEFAULT_SSP_BNAME : str
    Default filename for cached SSP data (ssp_data_fsps_v3.2_lgmet_age.h5)
HAS_FSPS : bool
    Flag indicating whether FSPS is available for data generation
"""

import typing
import h5py
import numpy as np
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from typing import Optional
from astropy import constants as const

# Import FSPS (Flexible Stellar Population Synthesis) and check availability
# FSPS is required for generating SSP data but may not be installed
try:
    import fsps
    HAS_FSPS = True
except (ImportError, RuntimeError):
    HAS_FSPS = False

#from .ssp import SSPBasis



# Default filename for cached SSP data - includes FSPS version for compatibility
DEFAULT_SSP_BNAME = "ssp_data_fsps_v3.2_lgmet_age.h5"


@dataclass(frozen=True)
class SSPData:
    """
    Immutable container for Simple Stellar Population interpolation grids from FSPS.
    
    This frozen dataclass stores the complete set of SSP spectra needed for stellar
    population synthesis. SSPs represent single bursts of star formation and serve
    as the fundamental building blocks for composite stellar populations in SED fitting.
    
    The data forms a 3D interpolation grid spanning metallicity, age, and wavelength.
    All arrays are JAX-compatible for efficient GPU-accelerated operations.
    
    Attributes:
    -----------
    ssp_lgmet : jnp.ndarray, shape (n_met,)
        Log10 absolute metallicity grid [log10(Z)]. Z is the mass fraction of elements 
        heavier than Helium. Typical range: ~-2.3 to +0.2 dex.
        
    ssp_lg_age_gyr : jnp.ndarray, shape (n_ages,)
        Log10 age grid [log10(age/Gyr)]. Includes artificially added zero-age
        SSP at log age = -6 (1 Myr) not available in FSPS. Range: -6 to ~1.15 dex.
        
    ssp_wave : jnp.ndarray, shape (n_wave,)
        Wavelength grid in Angstroms. Typical FSPS range: ~91 to 160,000 Å
        covering UV through near-IR.
        
    ssp_flux : jnp.ndarray, shape (n_met, n_ages, n_wave)  
        SSP spectral flux density in units of L_sun Hz^-1 per solar mass of
        initial stellar mass. This is the core interpolation grid for SED modeling.
        
    log_qq : jnp.ndarray, shape (n_met, n_ages)
        Log10 ionizing photon production rate [s^-1] for wavelengths < 912 Å.
        Used for nebular emission modeling. Units: log10(photons/s) per solar mass.
    
    Notes:
    ------
    - All arrays are frozen after initialization to prevent accidental modification
    - Shape validation ensures consistent grid dimensions across all attributes
    - Data can be persisted to/from HDF5 files for efficient caching
    """

    ssp_lgmet: jnp.ndarray          # Log10 absolute metallicity grid [log10(Z)]
    ssp_lg_age_gyr: jnp.ndarray     # Log10 age grid [log10(age/Gyr)]
    ssp_wave: jnp.ndarray           # Wavelength grid [Angstroms]
    ssp_flux: jnp.ndarray           # SSP flux grid [L_sun Hz^-1 M_sun^-1]
    log_qq: jnp.ndarray             # Log10 ionizing photon rate [s^-1 M_sun^-1]
    

    def __post_init__(self):
        """Validate grid consistency after initialization.
        
        Ensures that the flux grid dimensions match the metallicity, age, and
        wavelength coordinate arrays. This prevents inconsistent interpolation
        grids that could cause runtime errors in downstream calculations.
        
        Raises:
        -------
        ValueError
            If flux grid shape doesn't match coordinate array dimensions.
        """
        if self.ssp_flux.shape != (self.ssp_lgmet.size, self.ssp_lg_age_gyr.size, self.ssp_wave.size):
            raise ValueError(
                f"SSP flux grid shape mismatch: expected "
                f"({self.ssp_lgmet.size}, {self.ssp_lg_age_gyr.size}, {self.ssp_wave.size}) "
                f"but got {self.ssp_flux.shape}. Grid dimensions must be consistent "
                f"for interpolation: (n_met, n_ages, n_wave)."
            )
    

    def save(self, filename):
        """
        Serialize SSPData to HDF5 format for persistent storage.
        
        Saves all SSP grid data to an HDF5 file for efficient loading in future
        sessions. This avoids expensive re-computation from FSPS which can take
        several minutes for large grids.
        
        Parameters:
        -----------
        filename : str or Path
            Output HDF5 file path. Will be overwritten if it exists.
            
        Notes:
        ------
        JAX arrays are converted to NumPy for HDF5 compatibility.
        """
        with h5py.File(filename, 'w') as f:
            # Convert JAX arrays to NumPy for HDF5 storage
            f.create_dataset('ssp_lgmet', data=np.array(self.ssp_lgmet))
            f.create_dataset('ssp_lg_age_gyr', data=np.array(self.ssp_lg_age_gyr))
            f.create_dataset('ssp_wave', data=np.array(self.ssp_wave))
            f.create_dataset('ssp_flux', data=np.array(self.ssp_flux))
            f.create_dataset('log_qq', data=np.array(self.log_qq))
            
            # Add metadata for provenance tracking
            f.attrs['description'] = 'FSPS Simple Stellar Population interpolation grids'
            f.attrs['units_lgmet'] = 'log10(absolute_metallicity)'
            f.attrs['units_lg_age_gyr'] = 'log10(age/Gyr)'
            f.attrs['units_wave'] = 'Angstroms'
            f.attrs['units_flux'] = 'L_sun Hz^-1 M_sun^-1'
            f.attrs['units_log_qq'] = 'log10(photons s^-1 M_sun^-1)'

    @classmethod
    def load(cls, filename):
        """
        Load SSPData from HDF5 file.
        
        Reconstructs the SSPData object from previously saved HDF5 data.
        Arrays are automatically converted to JAX format for GPU compatibility.
        
        Parameters:
        -----------
        filename : str or Path
            Input HDF5 file path containing SSP data.
            
        Returns:
        --------
        SSPData
            Loaded SSP interpolation grids ready for SED modeling.
            
        Raises:
        -------
        FileNotFoundError
            If the specified file doesn't exist.
        KeyError
            If required datasets are missing from the HDF5 file.
        """
        with h5py.File(filename, 'r') as f:
            # Load datasets and convert to JAX arrays for GPU compatibility
            ssp_lgmet = jnp.array(f['ssp_lgmet'][:])
            ssp_lg_age_gyr = jnp.array(f['ssp_lg_age_gyr'][:])
            ssp_wave = jnp.array(f['ssp_wave'][:])
            ssp_flux = jnp.array(f['ssp_flux'][:])
            log_qq = jnp.array(f['log_qq'][:])

        return cls(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, log_qq)


def collect_ssp_data(**kwargs) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate SSP interpolation grids from FSPS with zero-age extension.
    
    Retrieves stellar population spectra from FSPS for all available metallicities
    and ages, then adds an artificial zero-age SSP by duplicating the youngest
    available template (log age = -4) and assigning it log age = -6 (1 Myr).
    This fills a gap in FSPS coverage needed for modeling very young stellar populations.
    
    The function also calculates ionizing photon production rates (log_qq) from
    the flux shortward of the Lyman limit (912 Å) for nebular emission modeling.
    
    Parameters
    ----------
    **kwargs : dict
        FSPS StellarPopulation initialization parameters (e.g., imf_type, sfh, etc.).
        See FSPS documentation or use print(fsps.PARAMS.all_params) to view all 
        available options.
        
    Returns
    -------
    tuple of jnp.ndarray
        - ssp_lgmet : shape (n_met,) - Log10 absolute metallicity grid
        - ssp_lg_age_gyr : shape (n_ages,) - Log10 age grid with added zero-age SSP  
        - ssp_wave : shape (n_wave,) - Wavelength grid [Angstroms]
        - ssp_flux : shape (n_met, n_ages, n_wave) - SSP flux grid [L_sun Hz^-1 M_sun^-1]
        - log_qq : shape (n_met, n_ages) - Log10 ionizing photon rates [s^-1 M_sun^-1]
        
    Raises
    ------
    ImportError
        If FSPS is not available for data generation.
        
    Notes
    -----
    - Zero-age SSP at log age = -6 is created by duplicating log age = -4 template
    - Ionizing photon rates integrate flux from 91-912 Å using trapezoidal rule
    - Processing time scales with grid size (typically 1-5 minutes for full grids)
    """
    if not HAS_FSPS:
        raise ImportError(
            "FSPS is required for SSP data generation but is not available. "
            "Please install FSPS following the instructions at: "
            "https://dfm.io/python-fsps/current/installation/"
        )

    # Initialize FSPS stellar population with discrete metallicity grid
    # sfh=0 specifies single stellar population (SSP) mode
    ssp = fsps.StellarPopulation(zcontinuous=0, sfh=0, **kwargs)

    # Extract metallicity and age grids from FSPS
    ssp_lgmet = jnp.log10(ssp.zlegend)  # Convert linear Z to log10(Z)
    nzmet = ssp_lgmet.size
    ssp_lg_age_gyr = ssp.log_age - 9.0  # Convert log(age/yr) to log(age/Gyr)

    # Find youngest available age (log age = -4) to duplicate for zero-age SSP
    #age_minus_4_idx = np.where(ssp_lg_age_gyr == -4)[0][0]

    # Collect SSP spectra for each metallicity bin
    # FSPS uses 1-based indexing for metallicity (zmet)
    spectrum_collector = []
    for zmet_indx in range(1, nzmet + 1):
        print(f"...retrieving metallicity {zmet_indx}/{nzmet} [Z = {ssp.zlegend[zmet_indx-1]:.4f}]")
        # Get full age sequence for this metallicity
        # tage=0.0 returns all ages, peraa=False gives flux in L_sun/Hz
        _wave, _fluxes = ssp.get_spectrum(tage=0.0, zmet=zmet_indx, peraa=False)
        spectrum_collector.append(_fluxes)


    # Convert to JAX arrays for GPU compatibility
    ssp_wave = jnp.array(_wave)  # Wavelength grid from last metallicity (same for all)
    ssp_flux = jnp.array(spectrum_collector)  # Shape: (n_met, n_ages, n_wave)

    # Create zero-age SSP by duplicating youngest available template # DO NOT DO THIS ANYMORE
    # This addresses FSPS limitation of not providing SSPs younger than ~10 Myr
    #duplicated_flux = ssp_flux[:, age_minus_4_idx, :]  # Extract log age = -4 template
    
    # Insert artificial zero-age SSP at beginning of age grid
    #new_age = -6  # log10(1 Myr / 1 Gyr) = -6
    #ssp_lg_age_gyr = np.insert(ssp_lg_age_gyr, 0, new_age)
    
    # Insert duplicated flux at beginning of flux grid for all metallicities
    #ssp_flux = np.concatenate([duplicated_flux[:, np.newaxis, :], ssp_flux], axis=1)
    
    # Final conversion to JAX arrays
    ssp_lg_age_gyr = jnp.array(ssp_lg_age_gyr)
    ssp_flux = jnp.array(ssp_flux)

    # Calculate ionizing photon production rates for nebular modeling
    # Integrate flux shortward of Lyman limit (912 Å) to get ionizing photon rate
    LL = 912.0  # Å

    mask = ssp_wave < LL                                  # (n_wave,)
    ssp_wave_ion  = ssp_wave[mask]                        # (n_ion,)
    ssp_flux_ion  = ssp_flux[..., mask]                   # (..., n_ion)

        
    # Convert flux density to photon rate using E = hν = hc/λ
    # Q(H) = ∫ L_ν / (hν) dν = ∫ (L_ν λ / hc) dλ
    qq = jnp.trapezoid(ssp_flux_ion/ssp_wave_ion, x=ssp_wave_ion)  # [L_sun Å / Å]
    
    # Convert to photons/s units: [L_sun] / [h] = [erg s^-1] / [erg s] = [s^-1]
    log_qq = jnp.log10(qq / const.h) + jnp.log10(const.L_sun)

    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, log_qq


def collect_ssp_data_wrapper(**kwargs) -> SSPData:
    """
    High-level interface for generating SSPData from FSPS.
    
    Convenience wrapper that generates SSP interpolation grids using FSPS and
    packages the results in an immutable SSPData container. This is the recommended
    entry point for creating new SSP datasets.
    
    Parameters
    ----------
    **kwargs : dict
        FSPS StellarPopulation configuration parameters. Common options:
        - imf_type : int, Initial mass function (0=Salpeter, 1=Chabrier, 2=Kroupa)
        - dust_type : int, Dust attenuation law (0=Charlot & Fall, 1=Calzetti, etc.)
        - add_neb_emission : bool, Include nebular emission lines
        Use print(fsps.PARAMS.all_params) to see complete parameter list.
        
    Returns
    -------
    SSPData
        Immutable container with complete SSP interpolation grids ready for
        stellar population synthesis and SED modeling.
        
    Examples
    --------
    >>> # Generate SSPs with Chabrier IMF and nebular emission
    >>> ssp_data = collect_ssp_data_wrapper(imf_type=1, add_neb_emission=True)
    >>> ssp_data.save('my_ssp_data.h5')  # Cache for future use
    """
    # Generate interpolation grids from FSPS
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, log_qq = collect_ssp_data(**kwargs)
    
    # Package into immutable container for safe downstream use
    return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, log_qq)


