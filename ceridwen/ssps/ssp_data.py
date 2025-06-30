"""
Module for handling Simple Stellar Population (SSP) data.

This module provides utilities to retrieve and store SSP data using 
FSPS and JAX for efficient operations. It defines a frozen dataclass 
for storing SSP-related data, as well as functions for data collection.

Globals:
--------
DEFAULT_SSP_BNAME : str
    The default file name for the SSP data.
"""

import typing
import h5py
import numpy as np
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from typing import Optional

# Import fsps and check availability
try:
    import fsps
    HAS_FSPS = True
except (ImportError, RuntimeError):
    HAS_FSPS = False

from .ssp import SSPBasis



# Default filename for SSP data
DEFAULT_SSP_BNAME = "ssp_data_fsps_v3.2_lgmet_age.h5"

@dataclass(frozen=True)
class SSPData:
    """
    A frozen dataclass to store information about Simple Stellar Population (SSP) templates.

    Attributes:
    -----------
    ssp_lgmet : jnp.ndarray
        1D array of log10(Z) representing the metallicity values of the SSP templates.
        Z is the mass fraction of elements heavier than Helium (He).

    ssp_lg_age_gyr : jnp.ndarray
        1D array of log10(age/Gyr) representing the ages of the SSP templates.

    ssp_wave : jnp.ndarray
        1D array of wavelength values for the SSP spectra.

    ssp_flux : jnp.ndarray
        3D array of shape (n_met, n_ages, n_wave) containing the Spectral Energy 
        Distribution (SED) of the SSP in units of Lsun/Hz. This represents the 
        flux for each combination of metallicity, age, and wavelength.

    ssp_flux_aa : Optional[jnp.ndarray]
        3D array of shape (n_met, n_ages, n_wave) containing the SED of the SSP in
        units of Lsun/AA, but with wavelengths in Angstroms (Å). This is an
        optional attribute that may not be present in all SSP datasets. If not provided,
    """

    ssp_lgmet: jnp.ndarray
    ssp_lg_age_gyr: jnp.ndarray
    ssp_wave: jnp.ndarray
    ssp_flux: jnp.ndarray
    ssp_flux_aa: jnp.array

    def __post_init__(self):
        """Validate the shapes of the SSP data after initialization."""
        if self.ssp_flux.shape != (self.ssp_lgmet.size, self.ssp_lg_age_gyr.size, self.ssp_wave.size):
            raise ValueError(
                f"Shape mismatch: 'ssp_flux' should have shape "
                f"(n_met, n_ages, n_wave), but got {self.ssp_flux.shape}. "
                f"Expected: ({self.ssp_lgmet.size}, {self.ssp_lg_age_gyr.size}, {self.ssp_wave.size})"
            )
    

    def save(self, filename):
        """
        Save the SSPData object to an HDF5 file.

        Parameters:
        -----------
        filename : str
            The file to which the data will be saved.
        """
        with h5py.File(filename, 'w') as f:
            # Save each array as a dataset in the HDF5 file
            f.create_dataset('ssp_lgmet', data=np.array(self.ssp_lgmet))
            f.create_dataset('ssp_lg_age_gyr', data=np.array(self.ssp_lg_age_gyr))
            f.create_dataset('ssp_wave', data=np.array(self.ssp_wave))
            f.create_dataset('ssp_flux', data=np.array(self.ssp_flux))
            f.create_dataset('ssp_flux_aa', data=np.array(self.ssp_flux_aa))


    @classmethod
    def load(cls, filename):
        """
        Load the SSPData object from an HDF5 file.

        Parameters:
        -----------
        filename : str
            The file from which the data will be loaded.
        
        Returns:
        --------
        SSPData : SSPData
            An instance of SSPData containing the loaded data.
        """
        with h5py.File(filename, 'r') as f:
            # Load each dataset from the HDF5 file
            ssp_lgmet = jnp.array(f['ssp_lgmet'][:])
            ssp_lg_age_gyr = jnp.array(f['ssp_lg_age_gyr'][:])
            ssp_wave = jnp.array(f['ssp_wave'][:])
            ssp_flux = jnp.array(f['ssp_flux'][:])
        
        return cls(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux)   
        
    def plot_spectrum(self, lg_metallicity, lg_age, ax=None, **kwargs):
        import warnings
        import matplotlib.pyplot as plt

        """
        Plot the flux array for a given metallicity and age over the wavelength array.
        
        Parameters:
        -----------
        self : object
            The object that contains the SSP data, including ssp_lgmet, ssp_lg_age_gyr, 
            ssp_wave, and ssp_flux.
        
        lg_metallicity : float
            The desired log10(metallicity) value.
        
        lg_age : float
            The desired log10(age/Gyr) value.

        ax : matplotlib.axes.Axes, optional
            The axis on which to plot the spectrum. If None, a new plot will be created.
        
        **plot_kwargs : keyword arguments
            Additional keyword arguments to customize the plot, passed directly to plt.plot().
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axis with the plotted spectrum.
        """
        # Unpack plot-related keyword arguments from kwargs
        plot_kwargs = kwargs.get('plot_kwargs', {})
        title_kwargs = kwargs.get('title_kwargs', {})
        label_kwargs = kwargs.get('label_kwargs', {})
        show_legend = kwargs.get('show_legend', True)

        # Find the closest available metallicity
        closest_met_idx = np.argmin(np.abs(self.ssp_lgmet - lg_metallicity))
        closest_metallicity = self.ssp_lgmet[closest_met_idx]

        if closest_metallicity != lg_metallicity:
            warnings.warn(f"Requested lg_metallicity = {lg_metallicity} not found, using closest value: {closest_metallicity}")

        # Find the closest available age
        closest_age_idx = np.argmin(np.abs(self.ssp_lg_age_gyr - lg_age))
        closest_age = self.ssp_lg_age_gyr[closest_age_idx]

        if closest_age != lg_age:
            warnings.warn(f"Requested lg_age = {lg_age} not found, using closest value: {closest_age}")

        # Extract the flux array for the closest metallicity and age
        flux = self.ssp_flux[closest_met_idx, closest_age_idx, :]
        
        # If no axis is provided, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot flux over wavelength with additional plot parameters passed via **plot_kwargs
        ax.plot(self.ssp_wave, flux, label=f'lg(Z) = {closest_metallicity:.3f}, lg(age/Gyr) = {closest_age:.3f}', **plot_kwargs)
        ax.set_xlabel('Wavelength [Å]', **label_kwargs)
        ax.set_ylabel('Flux [Lsun/Hz/Msun]', **label_kwargs)

        default_title = f'Spectrum for lg(Z) = {lg_metallicity:.3f}, lg(age/Gyr) = {lg_age:.3f}'
        title = title_kwargs.pop('title', default_title)
        ax.set_title(title, **title_kwargs)

        if show_legend:
            ax.legend()
        ax.set_xlim(0, 8000)
        ax.grid(True)

        return ax

# Default keys for SSPData attributes
DEFAULT_SSP_KEYS = SSPData.__annotations__.keys()



def collect_ssp_data(**kwargs) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Collects Simple Stellar Population (SSP) data using the SSPBasis class and 
    adds an extra SSP with log age -6 because zero age SSPs are not available in FSPS. 

    The new SSP with log age -6 is inserted at the beginning of the age and flux arrays.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to configure the SSPBasis object.

    Returns
    -------
    Tuple of jnp.ndarray:
        - ssp_lgmet : 1D array of log10(Z) representing the metallicities.
        - ssp_lg_age_gyr : 1D array of log10(age/Gyr) for the SSP templates, including the new SSP with log age -6.
        - ssp_wave : 1D array of wavelength values for the SSP spectra.
        - ssp_flux : 3D array containing the spectral fluxes for each metallicity, including the new SSP.
    """
    if not HAS_FSPS:
        raise ImportError("FSPS is required for SSP data collection but is not installed.")

    # Initialize the SSPBasis object
    ssp = SSPBasis(zcontinuous=0, sfh=0, **kwargs)

    # Retrieve logarithmic metallicity and age data
    ssp_lgmet = jnp.log10(ssp.ssp.zlegend)
    nzmet = ssp_lgmet.size
    ssp_lg_age_gyr = ssp.ssp.log_age - 9.0  # Log age in Gyr

    # Identify the index where log age is -4
    age_minus_4_idx = np.where(ssp_lg_age_gyr == -4)[0][0]

    # Collect spectral data for each metallicity
    spectrum_collector = []
    spectrum_collector_aa = []
    for zmet_indx in range(1, nzmet + 1):
        print(f"...retrieving zmet = {zmet_indx} of {nzmet}")
        _wave, _fluxes, _ = ssp.get_galaxy_spectrum(zmet=zmet_indx)
        _wave, _fluxes_aa, _ = ssp.get_galaxy_spectrum(zmet=zmet_indx, peraa=True)
        spectrum_collector.append(_fluxes)
        spectrum_collector_aa.append(_fluxes_aa)

    # Convert collected data to JAX arrays
    ssp_wave = jnp.array(_wave)
    ssp_flux = jnp.array(spectrum_collector)
    ssp_flux_aa = jnp.array(spectrum_collector_aa)

    # Step 1: Duplicate the SSP with age -4 to create one with age -6
    # Select flux corresponding to log age -4 for all metallicities
    duplicated_flux = ssp_flux[:, age_minus_4_idx, :]  # Shape: (nzmet, wavelength)
    duplicated_flux_aa = ssp_flux_aa[:, age_minus_4_idx, :]  # Shape: (nzmet, wavelength)
    # Step 2: Modify the age to log age -6 and insert it at the beginning of ssp_lg_age_gyr
    new_age = -6
    ssp_lg_age_gyr = np.insert(ssp_lg_age_gyr, 0, new_age)

    # Step 3: Insert the duplicated flux at the beginning of ssp_flux for each metallicity
    ssp_flux = np.concatenate([duplicated_flux[:, np.newaxis, :], ssp_flux], axis=1)
    ssp_flux_aa = np.concatenate([duplicated_flux_aa[:, np.newaxis, :], ssp_flux_aa], axis=1)

    # Convert the updated arrays to JAX arrays
    ssp_lg_age_gyr = jnp.array(ssp_lg_age_gyr)
    ssp_flux = jnp.array(ssp_flux)
    ssp_flux_aa = jnp.array(ssp_flux_aa)

    return ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, ssp_flux_aa


def collect_ssp_data_wrapper(**kwargs) -> SSPData:
    """
    Wrapper function to collect Simple Stellar Population (SSP) data and return it
    as an SSPData object.

    This function calls the lower-level JIT-compiled function `collect_ssp_data` 
    to retrieve the SSP data in the form of JAX arrays. It then wraps the data into 
    an SSPData dataclass for easy access and further usage.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments passed to the `collect_ssp_data` function to configure
        the SSPBasis object.

    Returns
    -------
    SSPData
        A frozen dataclass containing the SSP data, including metallicity, age,
        wavelength, and flux arrays.
    """
    # Collect SSP data in the form of JAX arrays
    ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, ssp_flux_aa = collect_ssp_data(**kwargs)

    # Return the data wrapped in an SSPData dataclass
    return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux, ssp_flux_aa)