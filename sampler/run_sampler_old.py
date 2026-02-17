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

import h5py
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import tqdm
import blackjax
from functools import partial
from jax import jit, grad

import astropy.constants as const

import matplotlib.pyplot as plt



"""
Composite Stellar Population (CSP) Module

This module implements the CSP modeling framework for generating synthetic spectra
from complex star formation and metallicity histories. It provides tools for:

- Building CSPs from Simple Stellar Population (SSP) libraries
- Handling time-dependent star formation and metallicity evolution  
- Computing spectral weights through integration of stellar population models
- Supporting both tabular and parametric stellar formation histories

The core algorithm follows the approach of flexible stellar population synthesis,
where CSPs are constructed by weighting and combining SSP spectra across age
and metallicity grids based on the input star formation and chemical enrichment
histories.

Key classes:
    CSPBasis: Main class for CSP spectrum generation and weight calculation

Key functions:
    add_sfh: Add star formation history to CSP model
    add_zh: Add metallicity evolution history to CSP model  
    intsfwght: Core integration function for stellar formation weights
"""

# JAX imports for high-performance numerical computing
import jax.numpy as jnp  # JAX's numpy-compatible array operations
from jax import jit, vmap      # Just-in-time compilation decorator for performance

import astropy.constants as const

def fnu2flam(lam, fnu):
    """Convert f_nu [erg/s/cm^2/Hz] to f_lambda [erg/s/cm^2/Å]."""
    c = 2.998e18  # Å/s
    return c * (fnu / (lam**2))

def add_zh(zh, lookback_time=None, forward_time=None, tuniv=13.8):
    """
    Add a star formation history (SFH) for a CSP.
    Parameters:
        zh (array-like): The metallicity at the given ages.
            Example: [0.1, 0.2, 0.3, 0.4]
        
        lookback_time (array-like, optional): Lookback times for the SFH in Gyr.
            Example: [1.0, 2.0, 3.0, 4.0]
        
        forward_time (array-like, optional): Forward times (age of the universe) for the SFH in Gyr.
            Example: [9.8, 10.8, 11.8, 12.8]
        
        tuniv (float, default 13.8): The age of the stellar population (in Gyr) for which to obtain a spectrum.
            Default: 13.8 Gyr

    Returns:
        tuple: A tuple containing:
            - sfh (jnp.ndarray): The star formation history array.
            Example output: jnp.array([0.1, 0.2, 0.3, 0.4])
            
            - sfh_times (jnp.ndarray): The corresponding times (lookback or forward) for the SFH.
            Example output with lookback_time: jnp.array([1.0, 2.0, 3.0, 4.0])
            Example output with forward_time: jnp.array([1.0, 2.0, 3.0, 4.0]) (computed as tuniv - forward_time)

    Usage:
        >>> add_sfh(sfh=[0.1, 0.2, 0.3, 0.4], lookback_time=[1.0, 2.0, 3.0, 4.0])
        >>> add_sfh(sfh=[0.1, 0.2, 0.3, 0.4], forward_time= [9.8, 10.8, 11.8, 12.8])
    """
    # Convert metallicity history to JAX array for efficient computation
    zh = jnp.array(zh)  # Ensure zh is a JAX array

    # Process time input - either lookback time (time before present) or forward time (age of universe)
    if lookback_time is not None:
        print('lookback time')  # Debug output for time mode
        lookback_time = jnp.array(lookback_time)
        # Convert from Gyr to years for internal calculations
        zh_times = lookback_time*1e9  
    elif forward_time is not None:
        print('forward time')  # Debug output for time mode  
        forward_time = jnp.array(forward_time)
        # Convert forward time to lookback time: lookback = tuniv - forward_time
        zh_times = tuniv*1e9 - forward_time*1e9  # Compute lookback time from forward time  
        print(zh_times)  # Debug output for computed times
    else:
        # Neither time specification provided - this is an error condition
        raise ValueError("Either 'lookback_time' or 'forward_time' must be provided.")
    
    # Note: This condition seems incorrect - zh_times should never be None at this point
    if zh_times != None:
        print('No times added for metallicity history, use lookback_time or forward_time of SFH')
    
    # Validate that metallicity and time arrays have consistent dimensions
    if zh.shape != zh_times.shape:
        raise ValueError(
            f"Shape mismatch: zh has shape {zh.shape}, but zh_times has shape {zh_times.shape}."
        )
    
    return zh, zh_times


def add_sfh(sfh, lookback_time=None, forward_time=None, tuniv=13.8):
    """
    Add a star formation history (SFH) for a CSP.
    Parameters:
        sfh (array-like): The star formation rate in solar masses per year at the given ages.
            Example: [0.1, 0.2, 0.3, 0.4]
        
        lookback_time (array-like, optional): Lookback times for the SFH in Gyr.
            Example: [1.0, 2.0, 3.0, 4.0]
        
        forward_time (array-like, optional): Forward times (age of the universe) for the SFH in Gyr.
            Example: [9.8, 10.8, 11.8, 12.8]
        
        tuniv (float, default 13.8): The age of the stellar population (in Gyr) for which to obtain a spectrum.
            Default: 13.8 Gyr

    Returns:
        tuple: A tuple containing:
            - sfh (jnp.ndarray): The star formation history array.
            Example output: jnp.array([0.1, 0.2, 0.3, 0.4])
            
            - sfh_times (jnp.ndarray): The corresponding times (lookback or forward) for the SFH.
            Example output with lookback_time: jnp.array([1.0, 2.0, 3.0, 4.0])
            Example output with forward_time: jnp.array([1.0, 2.0, 3.0, 4.0]) (computed as tuniv - forward_time)

    Usage:
        >>> add_sfh(sfh=[0.1, 0.2, 0.3, 0.4], lookback_time=[1.0, 2.0, 3.0, 4.0])
        >>> add_sfh(sfh=[0.1, 0.2, 0.3, 0.4], forward_time= [9.8, 10.8, 11.8, 12.8])
    """
    # Convert star formation history to JAX array for efficient computation
    sfh = jnp.array(sfh)  # Ensure sfh is a JAX array

    # Process time input - either lookback time (time before present) or forward time (age of universe)
    if lookback_time is not None:
        print('lookback time')  # Debug output for time mode
        lookback_time = jnp.array(lookback_time)
        # Convert from Gyr to years for internal calculations  
        sfh_times = lookback_time*1e9  

    elif forward_time is not None:
        print('forward time')  # Debug output for time mode
        forward_time = jnp.array(forward_time)
        # Convert forward time to lookback time: lookback = tuniv - forward_time
        sfh_times = tuniv*1e9 - forward_time*1e9  # Compute lookback time from forward time  
        print(sfh_times)  # Debug output for computed times
    else:
        # Neither time specification provided - this is an error condition
        raise ValueError("Either 'lookback_time' or 'forward_time' must be provided.")
    
    # Validate that star formation rate and time arrays have consistent dimensions
    if sfh.shape != sfh_times.shape:
        raise ValueError(
            f"Shape mismatch: sfh has shape {sfh.shape}, but sfh_times has shape {sfh_times.shape}."
        )
    
    
    return sfh, sfh_times

@jit
def intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_broadcasted):
    """
    Compute the integrated star formation weight for CSP modeling.
    
    This function calculates the contribution of stellar populations formed within 
    specific age bins to the overall CSP spectrum. It handles the integration of
    the star formation rate over time intervals, accounting for linear interpolation
    between adjacent time points.
    
    The integration accounts for:
    - Linear interpolation of SFR between time bins (sf_slope term)
    - Proper weighting by formation time and current age
    - Log-linear time scaling typical in stellar population models
    
    Args:
        tlimhi: Upper integration limit in log(age) (younger ages)
        tlimlo: Lower integration limit in log(age) (older ages) 
        a_broadcasted: Linear term coefficient for SFH interpolation
        sf_slope_broadcasted: Slope term for linear SFH interpolation
        logage_broadcasted: Log age of SSP bin being weighted
        
    Returns:
        intsfwght: Integrated star formation weight for the age interval
    """
    # Natural log conversion factor (log10(e)) for mathematical convenience
    loge = jnp.log10(jnp.e)

    # Convert log ages back to linear scale for integration calculations
    exp_tlimlo = 10**tlimlo  # Shape: (i, j) - Linear age at lower limit (older)
    exp_tlimhi = 10**tlimhi  # Shape: (i, j) - Linear age at upper limit (younger)  
    exp_tlimlo_squared = exp_tlimlo**2  # Shape: (i, j) - Squared for quadratic terms
    exp_tlimhi_squared = exp_tlimhi**2  # Shape: (i, j) - Squared for quadratic terms

    # Compute star formation weight at lower integration limit (older ages)
    # b1: Linear term contribution at older boundary
    b1 = a_broadcasted * exp_tlimlo * (logage_broadcasted - tlimlo + loge)
    # c1: Quadratic term contribution from SFH slope at older boundary  
    c1 = sf_slope_broadcasted * exp_tlimlo_squared / 2 * (logage_broadcasted - tlimlo + loge / 2)
    sfwght_lo = b1 + c1  # Total weight contribution at older boundary

    # Compute star formation weight at upper integration limit (younger ages)
    # b2: Linear term contribution at younger boundary
    b2 = a_broadcasted * exp_tlimhi * (logage_broadcasted - tlimhi + loge)
    # c2: Quadratic term contribution from SFH slope at younger boundary
    c2 = sf_slope_broadcasted * exp_tlimhi_squared / 2 * (logage_broadcasted - tlimhi + loge / 2)
    sfwght_hi = b2 + c2  # Total weight contribution at younger boundary

    # Fundamental theorem of calculus: integrated weight = upper_limit - lower_limit
    intsfwght = sfwght_hi - sfwght_lo
    return intsfwght

class CSPBasis:
    """
    A class to wrap the CSP object, providing the spectrum of a CSP for a given SFH.
    """

    def __init__(self, SSPData, tuniv = 13.8, tiny_logt = -10, 
                 add_neb = False, init_neb_params = {"isoc_type": "mist", "cloudy_dust": True},
                 dusty = False, dust_emission = False, sps_home = '/Users/amanda/Prospector/fsps',
                 init_dust_params = {'bin_edges': [(-jnp.inf, -1.97), (-1.97, jnp.inf)], 
                                'laws': ['powerlaw', 'kriek_conroy'], 'diffuse_law': 'kriek_conroy'}, **kwargs):
        """
        Initialize CSPBasis with the given SSPData object and age bounds from gal_t_table.
        
        Parameters:
            SSPData: An object holding SSP data (ages, metallicities, wavelengths, fluxes).
            tuniv: Age of the universe in Gyr (default: 13.8 Gyr)
            tiny_logt: Minimum lookback time in log(years) to prevent numerical issues
            **kwargs: Additional keyword arguments (not used here, but available for future extensions).
        """
        
        # Extract and convert SSP data to JAX arrays for efficient computation
        self.flux = jnp.array(SSPData.ssp_flux)        # SSP flux array: shape (n_metallicity, n_age, n_wavelength)
        self.wave = jnp.array(SSPData.ssp_wave)        # Wavelength grid in Angstroms: shape (n_wavelength,)
        self.ages = jnp.array(SSPData.ssp_lg_age_gyr)  # SSP ages in log10(Gyr): shape (n_age,)
        self.zmet = jnp.array(SSPData.ssp_lgmet)       # SSP metallicities in log10(Z/Zsun): shape (n_metallicity,)
        self.logqq = jnp.array(SSPData.log_qq)         # SSP log10(Q/Qsun): shape (n_age, n_metallicity)

        # Convert log metallicities to linear scale for interpolation calculations
        self.zlegend = 10**jnp.array(SSPData.ssp_lgmet)  # Linear metallicity scale: shape (n_metallicity,)

        # Convert SSP age grid from log(Gyr) to log(yr) for internal time calculations
        # This standardizes time units throughout the CSP calculations
        self.time_full = self.ages + 9  # Convert from log(Gyr) to log(yr): log10(Gyr) + log10(1e9) = log10(yr)
        
        # Universe age and numerical precision parameters
        self.tuniv = tuniv        # Age of the Universe in Gyr (cosmological parameter)
        self.tiny_logt = tiny_logt  # Smallest lookback time we accept in log(years) to prevent numerical issues

        self.sps_home = sps_home  # Path to FSPS home directory for stellar population synthesis

        # If dust emission is on, dusty must be on too
        if dust_emission:
            dusty = True
            self.emi = DustEmission(spec_lambda=self.wave, dust_file=sps_home)

        # Decide spectrum methods based on dusty/dust_emission
        if dusty:
            if dust_emission:
                self.get_spectrum = self.get_spectrum_dust_emission
            else:
                self.get_spectrum = self.get_spectrum_dusty
            # self.get_spectrum_direct = self.get_spectrum_direct_dusty  # enable if needed
            self.dust = Dust(**init_dust_params)
            self.bin_low = jnp.array([edge[0] for edge in self.dust.bin_edges])
            self.bin_high = jnp.array([edge[1] for edge in self.dust.bin_edges])
        else:
            self.get_spectrum = self.get_spectrum_dustfree
            self.get_spectrum_direct = self.get_spectrum_direct_dustfree

        if add_neb:
            self.get_spectrum_neb = self.get_spectrum_neb
            self.neb_model = NebularModel(
                sps_home=self.sps_home,
                csp_lambda=self.wave,
                nebular_smooth_init=2,
                smooth_velocity=True,
                mypi = jnp.pi,
                **init_neb_params)


    def __repr__(self):
        """
        Provide a string representation of the CSPBasis object, including
        dust attenuation and dust emission models if present.
        """
        repr_str = (
            f"<CSPBasis Object>\n"
            f"-----------------------------------\n"
            f"Universe Age (tuniv): {self.tuniv} Gyr\n"
            f"Tiny Log Time (tiny_logt): {self.tiny_logt}\n"
            f"Number of SSP Ages: {len(self.ages)}\n"
            f"Number of SSP Metallicities: {len(self.zmet)}\n"
            f"Wavelength Range: {self.wave.min()} - {self.wave.max()} Å\n"
        )

        # SFH
        if hasattr(self, "sfh"):
            repr_str += (
                "Star Formation History:\n"
                f"  SFH Times (lookback): {self.sfh_times}\n"
                f"  SFH Values: {self.sfh}\n"
            )
        else:
            repr_str += "Star Formation History: Not added yet\n"

        # Spectrum status
        repr_str += "Spectrum: Computed\n" if hasattr(self, "spectrum") else "Spectrum: Not computed yet\n"

        # Dust attenuation (if present)
        if getattr(self, "dust", None) is not None:
            repr_str += f"Dust Model: {repr(self.dust)}\n"
            if hasattr(self, "bin_low") and hasattr(self, "bin_high"):
                repr_str += f"  Dust Bins: {len(self.bin_low)} (low={self.bin_low}, high={self.bin_high})\n"
        else:
            repr_str += "Dust Model: None\n"

        # Dust emission (if present)
        if getattr(self, "emi", None) is not None:
            repr_str += f"Dust Emission: {repr(self.emi)}\n"
        else:
            repr_str += "Dust Emission: None\n"

        return repr_str

    def add_sfh(self, sfh, lookback_time=None, forward_time=None, tuniv=13.8):
        """
        Add a star formation history to the CSP.

        Parameters:
            sfh: The star formation rate at the given ages.
            lookback_time: Array of lookback times for the SFH in Gyr.
            forward_time: Array of forward times (age of the universe) for the SFH in Gyr.
            tuniv: The age of the universe in Gyr.
        """
        sfh, sfh_times = add_sfh(sfh, lookback_time = lookback_time, forward_time = forward_time, tuniv=tuniv)  # Use the JIT-compiled pure function
        self.sfh = sfh
        self.sfh_times = sfh_times

    def add_zh(self, zh, lookback_time=None, forward_time=None, tuniv=13.8):
        """
        Add a metallicity history to the CSP.

        Parameters:
            zh: The metallicity at the given ages.
            lookback_time: Array of lookback times for the SFH in Gyr.
            forward_time: Array of forward times (age of the universe) for the SFH in Gyr.
            tuniv: The age of the universe in Gyr.
        """
        zh, zh_times = add_zh(zh=zh, lookback_time = lookback_time, forward_time = forward_time, tuniv=tuniv)  # Use the JIT-compiled pure function
        #check that the metallicity history is the same length as the SFH
        if self.sfh_times.shape != zh_times.shape:
            raise ValueError("The metallicity history must have the same length as the SFH.")
        self.sfh_times = zh_times
        self.zh = zh
    
    def change_history(self, sfh=None, zh=None):
        """
        Change the star formation history or metallicity history of the CSP.
        
        Parameters:
            sfh: New star formation history to set.
            zh: New metallicity history to set.
        """
        self.sfh = sfh
        self.zh = zh

    def get_spectrum_dustfree(self, **kwargs):
        """
        Get the spectrum of the CSP for the given SFH. SFH (and optionally zh) must be added to the CSP object using the 'add_sfh' method.
        """
    
        total_weights = self.calculate_ssp_weights()
        
        spectrum = jnp.sum(total_weights[:, :, None] * self.flux, axis=(0,1))  # Shape: (n_wave,)
        self.spectrum = spectrum / (len(self.sfh) - 1) # Normalize by the number of time bins

        return self.spectrum

    def get_spectrum_neb(self, neb_params={'logZ': -0.82, 'logU': -2.0}, **kwargs):

        # Ensure logqq has age on axis 0 and metallicity on axis 1
        logqq = self.logqq.T

        def eval_single(logage, logQ):
            return self.neb_model.evaluate(logage=logage, logQ=logQ, **neb_params)

        # Map over ages for one metallicity row
        eval_over_age = vmap(eval_single, in_axes=(0, 0))  # (n_age,) x (n_age,) -> (n_age, n_wave)

        # Map over metallicities — keep them as the first axis in the output
        self.flux_neb = vmap(lambda logQ_row: eval_over_age(self.ages, logQ_row),
                        in_axes=0, out_axes=0)(self.logqq)
        
        total_weights = self.calculate_ssp_weights()

        self.spectrum_neb = jnp.sum(total_weights[:, :, None] * self.flux_neb, axis=(0,1)) / (len(self.sfh) - 1)  # Shape: (n_wave,)
        self.spectrum_noneb = jnp.sum(total_weights[:, :, None] * self.flux, axis=(0,1)) / (len(self.sfh) - 1) 
        self.spectrum = (self.spectrum_noneb+self.spectrum_neb) # Normalize by the number of time bins

        return self.spectrum
    
    

    def get_spectrum_dusty(self, dust_params, **kwargs):

        attenuation, diffuse = self.dust._compute_with_diffuse(self.wave, dust_params)
        self.diffuse = diffuse

        is_in_bin = (self.ages[:, None] >= self.bin_low[None, :]) & (self.ages[:, None] < self.bin_high[None, :])

        bin_indices = jnp.argmax(is_in_bin, axis=1)
        has_bin = jnp.any(is_in_bin, axis=1)
        self.attenuation_matrix = jnp.where(has_bin[:, None], jnp.exp(-attenuation[bin_indices]), jnp.ones_like(attenuation[0]))

        dusty_flux = self.flux * self.attenuation_matrix[None, :, :]  # Apply attenuation to SSP fluxes

        total_weights = self.calculate_ssp_weights()
        spectrum_dust = jnp.sum(total_weights[:, :, None] * dusty_flux, axis=(0,1))

        # --- 6) Apply diffuse attenuation at the end
        self.spectrum = spectrum_dust * jnp.exp(-diffuse)/ (len(self.sfh) - 1)  # Normalize by the number of time bins

        return self.spectrum
    
    def get_spectrum_dust_emission(self, dust_params, emi_params, **kwargs):

        self.emi.update_dust_params(**emi_params)
        total_weights = self.calculate_ssp_weights()

        attenuation, diffuse = self.dust._compute_with_diffuse(self.wave, dust_params)
        self.diffuse = diffuse

        is_in_bin = (self.ages[:, None] >= self.bin_low[None, :]) & (self.ages[:, None] < self.bin_high[None, :])

        bin_indices = jnp.argmax(is_in_bin, axis=1)
        has_bin = jnp.any(is_in_bin, axis=1)
        self.attenuation_matrix = jnp.where(has_bin[:, None], jnp.exp(-attenuation[bin_indices]), jnp.ones_like(attenuation[0]))

        dusty_flux = self.flux * self.attenuation_matrix[None, :, :]  # Apply attenuation to SSP fluxes

        spectrum_dustfree = jnp.sum(total_weights[:, :, None] * self.flux, axis=(0,1))/ (len(self.sfh) - 1)

        spectrum_dust = jnp.sum(total_weights[:, :, None] * dusty_flux, axis=(0,1))
        spectrum_diffuse = spectrum_dust * jnp.exp(-diffuse)/ (len(self.sfh) - 1)

        # --- 6) Apply diffuse attenuation at the end
        self.spectrum, self.mdust = self.emi.compute_dust_emission(spec_attn = spectrum_diffuse, spec_dustfree=spectrum_dustfree, spec_lambda=self.wave, diffuse_curve=self.diffuse)


        return self.spectrum
    

    


    def calculate_ssp_weights(self):
        """
        Calculate SSP weights for CSP spectrum generation.
        
        This is the core method that computes how much each SSP (defined by age and metallicity)
        contributes to the final CSP spectrum based on the star formation and metallicity histories.
        
        The algorithm:
        1. Interpolates SFH linearly between time bins
        2. Computes mass formed in each time interval  
        3. Distributes this mass across SSP age bins via integration
        4. Handles metallicity evolution through linear interpolation
        
        Dimensions: i = number of SFH time bins, j = number of SSP age bins
        
        Returns:
            total_weights: Array of SSP weights, shape depends on metallicity history:
                         - No metallicity history: (1, n_age)
                         - With metallicity history: (n_metallicity, n_age)
        """
        if not hasattr(self, "sfh"):
            raise ValueError("Please add an SFH to the CSP object using the 'add_sfh' method.")
        
        # Ensure SFH values are positive to avoid numerical issues
        self.sfh = jnp.clip(self.sfh, 1e-30, None)  # Ensure SFH is non-negative
        # === TIME BINNING AND SFH INTERPOLATION SETUP ===
        # Define time intervals from the SFH grid
        t1 = self.sfh_times[1:] # Beginning of time intervals (older times) - shape: (i,)
        t2 = self.sfh_times[:-1] # End of time intervals (younger times) - shape: (i,)
        
        # Compute slope for linear interpolation of SFH between adjacent points
        # This allows smooth SFH evolution within each time bin rather than step functions
        sf_slope = jnp.diff(self.sfh) / ((t1 - t2) * self.sfh[1:])  # Shape: (i,) - Normalized SFH slope

        # === TIME CLIPPING AND MASS CALCULATION ===
        # Clip times to physically valid range to avoid extrapolation beyond SSP grid
        tq = jnp.clip(t1, 10**self.tiny_logt, 10**self.time_full[-1])  # Shape: (i,) - Clipped older times
        tage = jnp.clip(t2, 10**self.tiny_logt, 10**self.time_full[-1]) # Shape: (i,) - Clipped younger times
        sf_trunc = tage - tq  # Shape: (i,) - Effective time interval after clipping
        
        # Calculate total stellar mass formed in each time interval
        # Accounts for linear SFH variation within the interval via trapezoidal rule
        m2 = (
                self.sfh[1:]  # SFR at younger edge of interval
                * (1 + sf_slope / 2.0 * (tage + tq - 2 * t1))  # Correction for linear SFH slope
                * sf_trunc  # Multiply by time interval duration
            )  # Shape: (i,) - Total stellar mass formed in each interval

        # Calculate parameters for linear SFH interpolation within integration
        tprime = jnp.maximum(0.0, tage - sf_trunc)  # Shape: (i,) - Time offset for slope calculation
        a = 1 - sf_slope * tprime  # Shape: (i,) - Linear interpolation coefficient

        # SSP-related computations
        ssp_dt = jnp.diff(self.time_full)  # Time intervals in SSP (shape: (107,))
        logage_lft = self.time_full[1:]    # Left edge of log-age bins (shape: (107,))
        logage_rght = self.time_full[:-1]  # Right edge of log-age bins (shape: (107,))

        # Broadcasting integration limits
        tq_broadcasted = jnp.log10(tq)[:, None]  # Expand tq for broadcasting (shape: (9, 1))
        tage_broadcasted = jnp.log10(tage)[:, None]  # Expand tage for broadcasting (shape: (9, 1))

        # Compute integration limits with broadcasting
        tlimlo = jnp.clip(logage_lft[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (9, 107)
        tlimhi = jnp.clip(logage_rght[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (9, 107)

        # Mask computation
        j_indices = jnp.arange(len(self.time_full))  # Indices for SSP bins (shape: (108,))
        jmin = jnp.clip(jnp.searchsorted(self.time_full, jnp.log10(t1)) - 1, 0, len(self.time_full) - 1)  # Shape: (9,)
        jmax = jnp.clip(jnp.searchsorted(self.time_full, jnp.log10(t2)) + 2, 0, len(self.time_full) - 1)  # Shape: (9,)

        # Create mask for relevant SSP bins
        mask = (j_indices >= jmin[:, None]) & (j_indices < jmax[:, None])  # Shape: (9, 108)
        mask_lft = mask[:, 1:]  # Mask for left edges (shape: (9, 107))
        mask_rght = mask[:, :-1]  # Mask for right edges (shape: (9, 107))

        # Boadcast bin edges
        logage_lft_broadcasted = logage_lft[None, :]  # Shape: (1, j), broadcast to (i, j)
        logage_rght_broadcasted = logage_rght[None, :]  # Shape: (1, j), broadcast to (i, j)

        a_broadcasted = a[:, None]  # Shape: (i, 1), broadcast to (i, j)
        sf_slope_broadcasted = sf_slope[:, None]  # Shape: (i, 1), broadcast to (i, j)

        # Left weights  
        intsfwght_lft = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_lft_broadcasted)
        tmp_weights_lft = jnp.zeros_like(intsfwght_lft) - intsfwght_lft / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_lft = jnp.where(mask_lft, tmp_weights_lft, 0.0)

        # Right weights
        intsfwght_rght = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_rght_broadcasted)
        tmp_weights_rght = jnp.zeros_like(intsfwght_rght) + intsfwght_rght / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_rght = jnp.where(mask_rght, tmp_weights_rght, 0.0)

        # Combine left and right weights
        result = jnp.zeros((tmp_weights_lft.shape[0], tmp_weights_lft.shape[1]+1))  
        w1 = result.at[:, :-1].add(tmp_weights_lft).at[:, 1:].add(tmp_weights_rght) # Shape: (i, j)

        m1 = jnp.sum(w1, axis=1) #shape (9,)

        sfh_weights = w1 * (m2[:, None] / m1[:, None])  # Shape: (j,)
        
        if not hasattr(self, "zh"):
            total_weights = sfh_weights.sum(axis=0)
            total_weights = total_weights[None, :]
        else:
            zbin = (self.zh[:-1] + self.zh[1:]) / 2 # Shape: (9,)  # Compute metallicity bin (zbin) from a simple average of adjacent metallicities
            k = jnp.clip(jnp.searchsorted(self.zlegend, zbin) - 1, 0, len(self.zlegend) - 2)  # Shape: (i,)
            bin_size = jnp.log10(self.zlegend[k + 1]) - jnp.log10(self.zlegend[k])  # Shape: (i,)

            dz = (jnp.log10(zbin) - jnp.log10(self.zlegend[k])) / bin_size  # Shape: (i,)
            dz = jnp.clip(dz, -1.0, 1.0)  # Clamping dz to avoid extrapolation

            total_weights = jnp.zeros((len(self.sfh_times)-1, len(self.zlegend), len(self.ages)))

            total_weights = total_weights.at[:, k].add((1 - dz[:, None]) * sfh_weights)
            total_weights = total_weights.at[:, k + 1].add(dz[:, None] * sfh_weights)
            total_weights = total_weights.sum(axis=0)#/(len(self.sfh_times)-1)  # Shape: (n_z, n_time)

            

            # See how much is going into each z bin:
            z_weights = jnp.sum(total_weights, axis=(1))  # shape (n_z,)
            
                
        self.ssp_weights = total_weights
        self.mass_formed = m2
        self.m1 = m1    
        self.m2 = m2
        self.w1 = w1
        
        return total_weights
    
    def get_spectrum_direct_dustfree(self, sfh, zh):
        """
        Get the spectrum of the CSP for the given SFH. SFH (and optionally zh) must be added to the CSP object using the 'add_sfh' method.
        """
    
        total_weights = self.calculate_ssp_weights_direct( sfh = sfh, zh = zh)
        
        spectrum = jnp.sum(total_weights[:, :, None] * self.flux, axis=(0,1))  # Shape: (n_wave,)

        return spectrum / (len(sfh) - 1)

    def calculate_ssp_weights_direct(self, sfh, zh):
        """
        Calculate SSP weights directly from input SFH and metallicity history.
        
        This is the core computational method that implements the CSP algorithm
        without requiring the histories to be stored as object attributes.
        
        Parameters:
            sfh: Star formation history array
            zh: Metallicity history array
            
        Returns:
            total_weights: SSP weights array (n_metallicity, n_age)
        """

        sfh = jnp.clip(sfh, 1e-70, None)  # Ensure SFH is non-negative

        # === TIME BINNING SETUP (same logic as calculate_ssp_weights) ===
        # Use the time grid stored in the object but operate on input SFH arrays directly
        t1 = self.sfh_times[1:]  # Beginning of time intervals (older times) - shape: (i,)
        t2 = self.sfh_times[:-1]  # End of time intervals (younger times) - shape: (i,)
        
        # Compute slope for linear interpolation using input SFH (not self.sfh)
        sf_slope = jnp.diff(sfh) / ((t1 - t2) * sfh[1:])  # Shape: (i,) - SFH slope per interval
        
        # === TIME CLIPPING AND MASS CALCULATION ===
        # Clip integration times to valid SSP grid range (same logic as calculate_ssp_weights)
        tq = jnp.clip(t1, 10**self.tiny_logt, 10**self.time_full[-1])  # Shape: (i,) - Clipped older times
        tage = jnp.clip(t2, 10**self.tiny_logt, 10**self.time_full[-1]) # Shape: (i,) - Clipped younger times
        sf_trunc = tage - tq  # Shape: (i,) - Effective time interval after clipping
        
        # Calculate stellar mass formed using input SFH (key difference: uses sfh parameter, not self.sfh)
        m2 = (sfh[1:]  # Use input SFH array instead of self.sfh
                * (1 + sf_slope / 2.0 * (tage + tq - 2 * t1))  # Trapezoidal rule correction
                * sf_trunc  # Time interval duration
            )  # Shape: (i,) - Stellar mass formed per time interval

        # Linear interpolation parameters for SFH integration
        tprime = jnp.maximum(0.0, tage - sf_trunc)  # Shape: (i,) - Time offset for slope calculation
        a = 1 - sf_slope * tprime  # Shape: (i,) - Linear interpolation coefficient

        # === REMAINDER OF ALGORITHM IDENTICAL TO calculate_ssp_weights ===
        # The following sections implement the same SSP weight calculation algorithm:
        # 1. SSP grid preparation 
        # 2. Integration limit setup and broadcasting
        # 3. Efficiency masking for sparse operations
        # 4. Weight integration using intsfwght function
        # 5. Edge contribution combination via finite differences
        # 6. Mass normalization
        # 7. Metallicity interpolation (using input zh instead of self.zh)
        # See calculate_ssp_weights method for detailed comments on each section
        
        # SSP-related computations
        ssp_dt = jnp.diff(self.time_full)  # Time intervals in SSP (shape: (107,))
        logage_lft = self.time_full[1:]    # Left edge of log-age bins (shape: (107,))
        logage_rght = self.time_full[:-1]  # Right edge of log-age bins (shape: (107,))

        # Broadcasting integration limits
        tq_broadcasted = jnp.log10(tq)[:, None]  # Expand tq for broadcasting (shape: (9, 1))
        tage_broadcasted = jnp.log10(tage)[:, None]  # Expand tage for broadcasting (shape: (9, 1))

        # Compute integration limits with broadcasting
        tlimlo = jnp.clip(logage_lft[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (9, 107)
        tlimhi = jnp.clip(logage_rght[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (9, 107)

        # Mask computation
        j_indices = jnp.arange(len(self.time_full))  # Indices for SSP bins (shape: (108,))
        jmin = jnp.clip(jnp.searchsorted(self.time_full, jnp.log10(t1)) - 1, 0, len(self.time_full) - 1)  # Shape: (9,)
        jmax = jnp.clip(jnp.searchsorted(self.time_full, jnp.log10(t2)) + 2, 0, len(self.time_full) - 1)  # Shape: (9,)

        # Create mask for relevant SSP bins
        mask = (j_indices >= jmin[:, None]) & (j_indices < jmax[:, None])  # Shape: (9, 108)
        mask_lft = mask[:, 1:]  # Mask for left edges (shape: (9, 107))
        mask_rght = mask[:, :-1]  # Mask for right edges (shape: (9, 107))

        # Boadcast bin edges
        logage_lft_broadcasted = logage_lft[None, :]  # Shape: (1, j), broadcast to (i, j)
        logage_rght_broadcasted = logage_rght[None, :]  # Shape: (1, j), broadcast to (i, j)

        a_broadcasted = a[:, None]  # Shape: (i, 1), broadcast to (i, j)
        sf_slope_broadcasted = sf_slope[:, None]  # Shape: (i, 1), broadcast to (i, j)

        # Left weights  
        intsfwght_lft = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_lft_broadcasted)
        tmp_weights_lft = jnp.zeros_like(intsfwght_lft) - intsfwght_lft / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_lft = jnp.where(mask_lft, tmp_weights_lft, 0.0)

        # Right weights
        intsfwght_rght = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_rght_broadcasted)
        tmp_weights_rght = jnp.zeros_like(intsfwght_rght) + intsfwght_rght / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_rght = jnp.where(mask_rght, tmp_weights_rght, 0.0)

        # Combine left and right weights
        result = jnp.zeros((tmp_weights_lft.shape[0], tmp_weights_lft.shape[1]+1))  
        w1 = result.at[:, :-1].add(tmp_weights_lft).at[:, 1:].add(tmp_weights_rght) # Shape: (i, j)

        m1 = jnp.sum(w1, axis=1) #shape (9,)

        sfh_weights = w1 * (m2[:, None] / m1[:, None])  # Shape: (j,)
        
        #if not hasattr(self, "zh"):
        #    total_weights = sfh_weights.sum(axis=0)
        #    total_weights = total_weights[None, :]
        #else:
        zbin = (zh[:-1] + zh[1:]) / 2 # Shape: (9,)  # Compute metallicity bin (zbin) from a simple average of adjacent metallicities
        k = jnp.clip(jnp.searchsorted(self.zlegend, zbin) - 1, 0, len(self.zlegend) - 2)  # Shape: (i,)
        bin_size = jnp.log10(self.zlegend[k + 1]) - jnp.log10(self.zlegend[k])  # Shape: (i,)

        dz = (jnp.log10(zbin) - jnp.log10(self.zlegend[k])) / bin_size  # Shape: (i,)
        dz = jnp.clip(dz, -1.0, 1.0)  # Clamping dz to avoid extrapolation

        total_weights = jnp.zeros((len(self.sfh_times)-1, len(self.zlegend), len(self.ages)))

        total_weights = total_weights.at[:, k].add((1 - dz[:, None]) * sfh_weights)
        total_weights = total_weights.at[:, k + 1].add(dz[:, None] * sfh_weights)
        total_weights = total_weights.sum(axis=0)#/(len(self.sfh_times)-1)  # Shape: (n_z, n_time)
                
        self.ssp_weights = total_weights
        self.mass_formed = m2
        self.m1 = m1    
        self.m2 = m2
        self.w1 = w1
        
        return total_weights
    


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
    age_minus_4_idx = np.where(ssp_lg_age_gyr == -4)[0][0]

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

    # Create zero-age SSP by duplicating youngest available template
    # This addresses FSPS limitation of not providing SSPs younger than ~10 Myr
    duplicated_flux = ssp_flux[:, age_minus_4_idx, :]  # Extract log age = -4 template
    
    # Insert artificial zero-age SSP at beginning of age grid
    new_age = -6  # log10(1 Myr / 1 Gyr) = -6
    ssp_lg_age_gyr = np.insert(ssp_lg_age_gyr, 0, new_age)
    
    # Insert duplicated flux at beginning of flux grid for all metallicities
    ssp_flux = np.concatenate([duplicated_flux[:, np.newaxis, :], ssp_flux], axis=1)
    
    # Final conversion to JAX arrays
    ssp_lg_age_gyr = jnp.array(ssp_lg_age_gyr)
    ssp_flux = jnp.array(ssp_flux)

    # Calculate ionizing photon production rates for nebular modeling
    # Integrate flux shortward of Lyman limit (912 Å) to get ionizing photon rate
    idx_ion = jnp.searchsorted(ssp_wave, 912.0)  # Find Lyman limit index
    
    ssp_flux_ion = ssp_flux[:, :, :idx_ion]  # Ionizing flux [L_sun Hz^-1]
    ssp_wave_ion = ssp_wave[:idx_ion]         # Ionizing wavelengths [Å]
    
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




def main():
    import jax
    from jax.lib import xla_bridge
    from time import perf_counter as time
    import matplotlib.pyplot as plt

    print("Backend platform:", xla_bridge.get_backend().platform)
    print("JAX devices:", jax.devices())
    try:
        # Attempt to load SSP data from a file
        ssp_data = SSPData.load('ssp_data.h5')
        print("SSP Data Loaded Successfully")
    except Exception as e:
        print(f"Error loading SSP Data: {e}")
        print('Ask Amanda where the fuck the ssp_data.h5 file is. It should be in the folder she send.')


    T_UNIV = 13.8  # Gyr, age of the universe
    N_T = 20     # number of time steps for SFH

    def gaussian_burst(tau, center_tau, width_tau, amp=1.0):
        """Gaussian in lookback time, evaluated on tau grid (Gyr)."""
        return amp * jnp.exp(-0.5 * ((tau - center_tau) / width_tau)**2)

    t = jnp.linspace(1e-2, T_UNIV, N_T)   # Gyr (avoid exactly 0)
    tau = T_UNIV - t                      # lookback time (Gyr)

    sfr_bimodal = (gaussian_burst(tau, 0.05, 0.03, 1.0) +
                gaussian_burst(tau, 11.0, 0.8, 0.7))
    zh = jnp.full_like(t, 0.02)  # constant metallicity (Z=0.02)
    zh = jnp.linspace(0.0002, 0.2, len(tau))

    t = jnp.linspace(1e-2, T_UNIV, N_T)   # Gyr (avoid exactly 0)
    tau = T_UNIV - t                      # lookback time (Gyr)

    def gaussian_burst(center_tau, width_tau, amp=1.0):
        """Gaussian in lookback time, evaluated on tau grid (Gyr)."""
        return amp * jnp.exp(-0.5 * ((tau - center_tau) / width_tau)**2)



    sfr_old = gaussian_burst(center_tau=12.0, width_tau=1.0, amp=1.0)


    csp = CSPBasis(ssp_data, dusty=False, add_neb=False)
    csp.add_sfh(sfr_old, lookback_time=tau)
    csp.add_zh(zh, lookback_time=tau)
    spec = csp.get_spectrum()

    csp = CSPBasis(ssp_data, dusty=False, add_neb=False)
    csp.add_sfh(sfr_old, lookback_time=tau)
    csp.add_zh(zh, lookback_time=tau)
    spec = csp.get_spectrum()

    rng_key = jax.random.PRNGKey(42)

    key, rng_key = jax.random.split(rng_key)
    sigma = 0.2 * spec  # relative noise level
    noise = sigma * jax.random.normal(key, shape=spec.shape)
    spectrum_obs = spec + noise


    plt.figure(figsize=(20/3, 3))
    plt.plot(csp.wave, fnu2flam(csp.wave, spec*1e-23*3631), label='Spectrum', color='dodgerblue')
    plt.plot(csp.wave, fnu2flam(csp.wave, spectrum_obs*1e-23*3631), label='Observed', color='coral', zorder = 0)
    plt.xlim(0, 10000)
    plt.yscale('log')
    plt.legend()
    plt.savefig('/home/aas208/Ceridwen/forHarveyDir/figsspectrum_comparison.pdf')

    print('____________________________________________________________')


    # -------------------------- MAKE THE MODEL FOR THE LIKELIHOOD FUNCTION -------------------------

    from functools import partial
    csp = CSPBasis(ssp_data)

    csp.add_sfh(sfh=sfr_old, lookback_time=tau) #need to set sfh_times 
    model = partial(csp.get_spectrum_direct_dustfree)  # csp is now static
    print(model)
    print('____________________________________________________________')


    # -------------------------- CHECK IF JIT AND GRAD WORKS -------------------------

    model = jax.jit(model)
    spectrum = model(sfr_old, zh)

    # grad also works (with respect to sfh, for example)
    grad_spectrum = grad(lambda sfh: jnp.sum(model(sfh, zh)))(sfr_old)
    print(model)
    print('JIT and grad work!')
    print('____________________________________________________________')

    # -------------------------- SET UP THE LIKELIHOOD FUNCTION -------------------------

 
    def unpack_params(params_flat, n_bins):
        sfh = jnp.array([params_flat[f"sfh_{i}"] for i in range(n_bins)])
        zh  = jnp.array([params_flat[f"zh_{i}"]  for i in range(n_bins)])
        sigma = jnp.array(params_flat["sigma"])
        return sfh, zh, sigma

    @jit
    def loglikelihood_fn(params_flat):
        sfh, zh, sigma = unpack_params(params_flat, n_bins=N_T)
        mu = model(sfh, zh)
        cov = sigma ** 2
        lgl = jax.scipy.stats.multivariate_normal.logpdf(spectrum_obs, mu, cov)
        return lgl

    # -------------------------- SET UP THE PRIOR FUNCTION -------------------------

    from blackjax.ns.utils import uniform_prior



    # Create prior_bounds dictionary
    prior_bounds = {**{f"sfh_{i}": (0.0, 10.0) for i in range(N_T)},
                **{f"zh_{i}": (5e-5, 0.02) for i in range(N_T)},
                "sigma": (1e-7, 1e-5)}

    rng_key = jax.random.PRNGKey(0)
    num_dims = N_T * 2 + 1 #star formation history, metallicity history and sigma
    num_live = 1000
    num_inner_steps = num_dims * 5
    num_delete = num_live // 2

    rng_key, prior_key = jax.random.split(rng_key)
    particles, logprior_fn = uniform_prior(prior_key, num_live, prior_bounds)


    # -------------------------- TEST -------------------------
    sfh_test0 = jnp.linspace(2.0, 6.0, N_T)     # example trend for test 0
    sfh_test1 = jnp.linspace(3.0, 7.0, N_T)     # example trend for test 1
    sfh_batch = jnp.stack([sfh_test0, sfh_test1], axis=0)  # (B, N_T)

    # zh in [5e-5, 2e-2] — keep well inside bounds
    zh_test0 = jnp.linspace(1.0e-4, 9.0e-4, N_T)
    zh_test1 = jnp.linspace(4.0e-4, 1.5e-3, N_T)
    zh_batch = jnp.stack([zh_test0, zh_test1], axis=0)     # (B, N_T)

    # sigma (one per test)
    sigma_batch = jnp.array([5.9143e-3, 1.0e-3])           # (B,)

    # ---- assemble per-bin dict expected by your current loglikelihood_fn ----
    params_test = {**{f"sfh_{i}": sfh_batch[:, i] for i in range(N_T)},
                **{f"zh_{i}":  zh_batch[:, i]  for i in range(N_T)},
                "sigma": sigma_batch}


    lls = jax.vmap(loglikelihood_fn)(params_test)
    print(f"Log-likelihoods for test parameters: {lls}")
    print('____________________________________________________________')

    nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
    )

    init_fn = jax.jit(nested_sampler.init)
    step_fn = jax.jit(nested_sampler.step)
    live = init_fn(particles)
    
    # -------------------------- Sampling -------------------------

    dead = []
    print('Starting sampling...')

    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not live.logZ_live - live.logZ < -3:
            print(f"Current logZ: {live.logZ_live - live.logZ}")
            rng_key, subkey = jax.random.split(rng_key, 2)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)

    print('Sampling completed.')

    from anesthetic import NestedSamples
    columns = dead.particles.keys()
    data = jnp.vstack([dead.particles[key] for key in columns]).T
    samples = NestedSamples(
        data,
        logL=dead.loglikelihood,
        logL_birth=dead.loglikelihood_birth,
        columns=columns,
        logzero=jnp.nan,
    )
    samples.to_csv("/home/aas208/Ceridwen/forHarveyDir/figs/galaxy.csv")



if __name__ == "__main__":
    main()