
# --- Standard library ---
import inspect
from pprint import pprint
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence
from dataclasses import dataclass

import typing
from pathlib import Path


# --- Third-party libraries ---
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from astropy import constants as const

# --- JAX ---
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, grad
from jax.lib import xla_bridge
from time import perf_counter as time

# --- Local modules ---
from sedpy_jax.attenuation_dust import ATTENUATION_LAWS

# --- Sampler ---
import tqdm
import blackjax

# --- Constants ---
tiny_number = 1e-70

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

def fnu2flam(lam, fnu):
    """Convert f_nu [erg/s/cm^2/Hz] to f_lambda [erg/s/cm^2/Å]."""
    c = 2.998e18  # Å/s
    return c * (fnu / (lam**2))

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

    def __init__(self, SSPData, theta_dict = {'lookback_time': 13.8 - jnp.linspace(1e-2, 13.8, 100)}, tuniv = 13.8, tiny_logt = -70, zh_const = False,
                 add_neb = True, init_neb_params = {"isoc_type": "mist", "cloudy_dust": True}, 
                 add_dust = True, add_diffuse_dust = True, add_dust_emission = False, sps_home = '/home/aas208/Prospector/fsps', #'/Users/amanda/Prospector/fsps', #
                 init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 'laws': ['powerlaw']},
                 diffuse_law = 'kriek_conroy', verbose = True, **kwargs):
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
        self.ssp_ages_lgyr = self.ages + 9  # Convert from log(Gyr) to log(yr): log10(Gyr) + log10(1e9) = log10(yr)
        
        # Universe age and numerical precision parameters
        self.tuniv = tuniv        # Age of the Universe in Gyr (cosmological parameter)
        self.tiny_logt = tiny_logt  # Smallest lookback time we accept in log(years) to prevent numerical issues

        self.sps_home = sps_home  # Path to FSPS home directory for stellar population synthesis

        if add_neb:
            self.neb_model = NebularModel(sps_home=self.sps_home, csp_lambda=self.wave, nebular_smooth_init=2,
                                          smooth_velocity=True, mypi = jnp.pi, **init_neb_params)

        if add_diffuse_dust or add_dust:
            self.set_attenuation_function(add_diffuse_dust, add_dust)

        theta_dict = self.initialize_dust_components(
                                                    add_dust,
                                                    add_diffuse_dust,
                                                    add_dust_emission,
                                                    theta_dict,
                                                    init_dust_params,
                                                    diffuse_law,
                                                    sps_home,
                                                )
   
        self.configure_spectrum_model(add_dust, add_diffuse_dust, add_dust_emission, add_neb, sps_home)
    
        # SSP weight calculation method depending on metallicity history    
        if zh_const:
            self.calculate_ssp_weights = self.calculate_ssp_weights_const_zh    
        else:
            self.calculate_ssp_weights = self.calculate_ssp_weights_var_zh

        # Determine Model Structure
        self.summary = self.initialize_model_structure(theta_dict)
        if verbose:
            pprint(self.summary.keys())

        self.params = self.build_params_from_summary(self.summary)

    def build_params_from_summary(self, summary):
        """Create a self.params-style summary dictionary from a summary dict."""
        params = {}

        for key, val in summary.items():
            if not key.endswith('_idx'):
                continue

            # parameter base name, e.g. "sfh_idx" → "sfh"
            pname = key[:-4]
            idx = summary[key]

            if len(idx) == 1:
                params[pname] = {'pos': f"{idx[0]}", 'length': 1}
            elif len(idx) <= 3:
                params[pname] = {'pos': str(idx), 'length': len(idx)}
            else:
                params[pname] = {'pos': f"{idx[0]}–{idx[-1]}", 'length': len(idx)}

        return params
        
    def set_attenuation_function(self, add_diffuse_dust, add_dust):
        """
        Define and assign the appropriate dust attenuation function based on model configuration.
        """

        # --- Case 1: both binwise and diffuse dust are included ---
        if add_diffuse_dust and add_dust: 
            
            def attenuate(wave, theta):
                # binwise
                dust_param_values = tuple(theta[getattr(self, f"{name}_idx")] for name in self.dust_param_names)
                dust_params = self.DustParams(*dust_param_values)
                attn = self.dust_attn.compute_attenuation(wave, dust_params)

                # diffuse
                diffuse_param_values = tuple(theta[getattr(self, f"{name}_idx")] for name in self.diff_param_names)
                diffuse_params = self.DiffDustParams(*diffuse_param_values)
                attn_diffuse = self.diff_dust.compute_attenuation(wave, diffuse_params)

                return attn, attn_diffuse
            print("Using combined (binwise + diffuse) dust attenuation.")
            self.attenuate_dust = attenuate

        elif add_dust and not add_diffuse_dust:

            def attenuate_without_diffuse(wave, theta):
                # --- Extract bin-based dust parameters
                dust_param_values = tuple(theta[getattr(self, f"{name}_idx")] for name in self.dust_param_names)
                dust_params = self.DustParams(*dust_param_values)
                attn = self.dust_attn.compute_attenuation(wave, dust_params)  # shape (num_bins, len(wave))

                # --- Dummy diffuse attenuation: shape (1, len(wave)), all ones
                attn_diffuse = jnp.zeros((1, wave.shape[0]))

                return attn, attn_diffuse

            print("Using only binwise dust attenuation.")
            self.attenuate_dust = attenuate_without_diffuse

        elif add_diffuse_dust and not add_dust:

            self.bin_low = jnp.array([-jnp.inf])
            self.bin_high = jnp.array([jnp.inf])

            def attenuate_diffuse_only(wave, theta):
                # --- Extract diffuse dust parameters
                diffuse_param_values = tuple(theta[getattr(self, f"{name}_idx")] for name in self.diff_param_names)
                diffuse_params = self.DiffDustParams(*diffuse_param_values)
                attn_diffuse = self.diff_dust.compute_attenuation(wave, diffuse_params)

                # --- Dummy binwise attenuation: shape (1, len(wave)), all ones
                attn = jnp.zeros((1, wave.shape[0]))

                return attn, attn_diffuse

            print("Using only diffuse dust attenuation.")
            self.attenuate_dust = attenuate_diffuse_only

            
    def initialize_dust_components( self, add_dust: bool, add_diffuse_dust: bool, add_dust_emission: bool,
                                    theta_dict: dict, init_dust_params: dict, diffuse_law: str, sps_home: str) -> dict:
        """
        Initialize dust-related models and register their parameters.
        """
        if add_dust:
            print("Initializing Dust attenuation model...")
            self.dust_attn = Dust(**init_dust_params)

            self.bin_low = jnp.array([edge[0] for edge in self.dust_attn.bin_edges])
            self.bin_high = jnp.array([edge[1] for edge in self.dust_attn.bin_edges])

            dust_fit_dict = self.dust_attn.get_default_fit_params()._asdict()
            for k, v in dust_fit_dict.items():
                if k not in theta_dict:
                    theta_dict[k] = v

            self.dust_param_names = list(dust_fit_dict.keys())
            self.DustParams = NamedTuple(
                "DustParams", [(name, float) for name in self.dust_param_names]
            )

        if add_diffuse_dust:
            print("Initializing DiffuseDust model...")
            self.diff_dust = DiffuseDust(diffuse_law)

            diff_fit_dict = self.diff_dust.get_default_params()._asdict()
            for k, v in diff_fit_dict.items():
                if k not in theta_dict:
                    theta_dict[k] = v

            self.diff_param_names = list(diff_fit_dict.keys())
            self.DiffDustParams = NamedTuple(
                "DiffDustParams", [(name, float) for name in self.diff_param_names]
            )

        if add_dust_emission:
            print("Initializing DustEmission model...")
            self.dust_emi = DustEmission(spec_lambda=self.wave, dust_file=sps_home)

            emi_fit_dict = self.dust_emi.get_default_params()._asdict()
            for k, v in emi_fit_dict.items():
                if k not in theta_dict:
                    theta_dict[k] = v

            self.emi_param_names = list(emi_fit_dict.keys())
            self.DustEmiParams = NamedTuple(
                "DustEmiParams", [(name, float) for name in self.emi_param_names]
            )

        print("Dust initialization complete.")
        return theta_dict

    def configure_spectrum_model(self, add_dust, add_diffuse_dust, add_dust_emission, add_neb, sps_home):
        """
        Configure which get_spectrum method to use depending on dust and nebular settings.
        """
        if add_dust or add_diffuse_dust:
            part1 = 'dust_'
            message1 = 'Get spectrum with dust attenuation'
        else:
            part1 = 'nodust_'
            message1 = 'Get spectrum without dust attenuation'
        if add_neb:
            part2 = 'neb_'
            message2 = ', with nebular continuum and lines'
        else:
            part2 = 'noneb_'
            message2 = ', without nebular contribution'
        if add_dust_emission:
            if not add_dust or not add_diffuse_dust:
                raise Error('without dust attenuation no dust emission possible.')
            part3 = 'dustemi'
            message3 = ', and with dust emission.'
        else:
            part3 = 'nodustemi'
            message3 = ', and without dust emission.'
        message = message1 + message2 + message3
        key = part1 + part2 + part3
        mapping = key_message_map = {
                'dust_neb_dustemi':  self.get_spectrum_dattn_dem_neb,
                'dust_neb_nodustemi': self.get_spectrum_dattn_nodem_neb,
                'dust_noneb_dustemi': self.get_spectrum_dattn_dem_noneb,
                'dust_noneb_nodustemi':  self.get_spectrum_dattn_nodem_noneb,
                'nodust_neb_nodustemi':  self.get_spectrum_nodattn_nodem_neb,
                'nodust_noneb_nodustemi':  self.get_spectrum_nodattn_nodem_noneb
            }
        print(message)
        self.get_spectrum = mapping[key]


    def initialize_model_structure(self, theta_dict: dict):
        """
        Initializes model structure from a flexible theta_dict.

        Required:
            - 'sfh': jnp.ndarray of shape (n_time,)
            - 'lookback_time': jnp.ndarray of shape (n_time,)

        Optional:
            - Any number of scalar or array parameters (e.g. 'Z', 'zh', 'dust', ...)
        """

        # --- Required base parameters
        self.sfh_times = theta_dict['lookback_time'] * 1e9  # Gyr → yr
        sfh = jnp.atleast_1d(theta_dict['sfh'])
        self.n_time = self.sfh_times.size

        assert sfh.shape == (self.n_time,), "'sfh' must match 'lookback_time' length"
        self.sfh_idx = jnp.arange(self.n_time)

        theta_parts = [sfh]
        current_idx = self.n_time  # start after sfh

        # Track special handling of Z vs. zh
        self.zh_is_scalar = None

        for key, val in theta_dict.items():
            if key in ('sfh', 'lookback_time'):
                continue  # already handled

            arr = jnp.atleast_1d(val)

            if key == 'zh':
                assert arr.shape == (self.n_time,), "'zh' must match 'lookback_time' length"
                setattr(self, f'{key}_idx', jnp.arange(current_idx, current_idx + self.n_time))
                self.zh_is_scalar = False
                current_idx += self.n_time

            elif key == 'Z':
                assert arr.shape == (1,), "'Z' must be a scalar"
                setattr(self, f'{key}_idx', jnp.array([current_idx]))
                self.zh_is_scalar = True
                current_idx += 1

            else:
                # General case: scalar or vector parameters
                arr_shape = arr.shape
                flat_len = arr.size
                setattr(self, f'{key}_idx', jnp.arange(current_idx, current_idx + flat_len))
                current_idx += flat_len

            # Append to flat vector
            theta_parts.append(arr)

        # Final flat parameter vector
        self.theta_init = jnp.concatenate(theta_parts)
        
        summary = {
            'sfh_times': self.sfh_times.tolist(),
            'sfh': sfh.tolist(),
            'theta_init': self.theta_init.tolist(),
        }

        # Dynamically include all *_idx attributes
        for name in dir(self):
            if name.endswith('_idx') and not name.startswith('__'):
                summary[name] = getattr(self, name).tolist()

        return summary
   
    def predict(self, theta):
        spectrum = self.get_spectrum(theta = theta)
        return spectrum

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

        repr_str += f"Model Params: {self.params}\n"

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

 
    def get_spectrum_dattn_dem_neb(self):
        """
        Get the spectrum of the CSP with dust attenuation, dust emission, and nebular emission.
        """
        raise NotImplementedError("This method is not yet implemented.")

    
    def get_spectrum_dattn_nodem_neb(self):
        """
        Get the spectrum of the CSP with dust attenuation and nebular emission, but no dust emission.
        """
        raise NotImplementedError("This method is not yet implemented.")
    
    def get_spectrum_dattn_nodem_noneb(self, theta):

        """
        Get the spectrum of the CSP with dust attenuation, but no dust emission or nebular emission.
        """

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        is_in_bin = (self.ages[:, None] >= self.bin_low[None, :]) & (self.ages[:, None] < self.bin_high[None, :])
        bin_indices = jnp.argmax(is_in_bin, axis=1)
        has_bin = jnp.any(is_in_bin, axis=1)

        atten_matrix = jnp.where(has_bin[:, None], jnp.exp(-attn[bin_indices]), jnp.ones_like(attn[0]))
        dusty_flux = self.flux * atten_matrix[None, :, :]
        weights = self.calculate_ssp_weights(theta)
        spectrum_dust = jnp.sum(weights[:, :, None] * dusty_flux, axis=(0, 1))

        # apply diffuse dust, diffuse dust is not included, this is just multiplying with ones
        attenuated = spectrum_dust * jnp.exp(-attn_diffuse)
        self.spectrum = jnp.reshape(attenuated, (-1,)) / (self.n_time - 1)
        return self.spectrum

    def get_spectrum_dattn_dem_noneb(self, theta):
        """
        Get the spectrum of the CSP with dust attenuation and dust emission, but no nebular emission.
        """
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        is_in_bin = (self.ages[:, None] >= self.bin_low[None, :]) & (self.ages[:, None] < self.bin_high[None, :])
        bin_indices = jnp.argmax(is_in_bin, axis=1)
        has_bin = jnp.any(is_in_bin, axis=1)

        # add age bin dust attenuation
        atten_matrix = jnp.where(has_bin[:, None], jnp.exp(-attn[bin_indices]), jnp.ones_like(attn[0]))
        dusty_flux = self.flux * atten_matrix[None, :, :]
        weights = self.calculate_ssp_weights(theta)/(self.n_time - 1)

        spectrum_dust_free = jnp.sum(weights[:, :, None] * self.flux, axis=(0, 1))
        spectrum_dust = jnp.sum(weights[:, :, None] * dusty_flux, axis=(0, 1))

        # apply diffuse dust
        attenuated = spectrum_dust * jnp.exp(-attn_diffuse)
        self.spec_attn = attenuated
        
        dust_emi_spectrum, self.mdust, self.tduste = self.dust_emi.compute_dust_emission(spec_attn=self.spec_attn, spec_dustfree=spectrum_dust_free,
                                                                                spec_lambda=self.wave, diffuse_curve=jnp.exp(-attn_diffuse),
                                                                                duste_qpah=theta[self.duste_qpah_idx],
                                                                                duste_umin=theta[self.duste_umin_idx],
                                                                                duste_gamma=theta[self.duste_gamma_idx])
        self.spectrum = dust_emi_spectrum
        return self.spectrum
        
    
    def get_spectrum_nodattn_nodem_neb(self):
        """
        Get the spectrum of the CSP with nebular emission, but no dust attenuation or dust emission.
        """
        total_weights = self.calculate_ssp_weights()
        # Compute the spectrum using the total_weights and the nebular model
        # This is a placeholder implementation
        raise NotImplementedError("This method is not yet implemented.")


    def get_spectrum_nodattn_nodem_noneb(self, theta):
        
        """
        Compute CSP spectrum at fixed metallicity (no dust, no nebular).
        """
        
        total_weights = self.calculate_ssp_weights(theta=theta)
        
        spectrum = jnp.sum(total_weights[:, :, None] * self.flux, axis=(0,1))  # Shape: (n_wave,)
        self.spectrum = spectrum / (self.n_time - 1) # Normalize by the number of time bins

        return self.spectrum


    

    def calculate_ssp_weights_const_zh(self, theta):
        """
        Calculate SSP weights for CSP spectrum generation.
        
        This is the core method that computes how much each SSP (defined by age and metallicity)
        contributes to the final CSP spectrum based on the star formation and metallicity histories.
        
        The algorithm:
        1. Interpolates SFH linearly between time bins
        2. Computes mass formed in each time interval  
        3. Distributes this mass across SSP age bins via integration
        4. Handles metallicity evolution through linear interpolation
        
        Dimensions: i = number of SFH time bins (len(sfh) - 1), j = number of SSP ages (len(ssp_ages_lgyr) - 1)
        
        Returns:
            total_weights: Array of SSP weights, shape depends on metallicity history:
                         - No metallicity history: (1, n_age)
                         - With metallicity history: (n_metallicity, n_age)
        """
        # Ensure SFH values are positive to avoid numerical issues
        sfh = theta[self.sfh_idx]
        sfh = jnp.clip(sfh, 1e-30, None)  # Ensure SFH is non-negative - shape (i+1)

        # === TIME BINNING AND SFH INTERPOLATION SETUP ===
        # Define time intervals from the SFH grid
        t1 = self.sfh_times[1:] # Beginning of time intervals (older times) - shape: (i,)
        t2 = self.sfh_times[:-1] # End of time intervals (younger times) - shape: (i,)

        
        # Compute slope for linear interpolation of SFH between adjacent points
        # This allows smooth SFH evolution within each time bin rather than step functions
        sf_slope = jnp.diff(sfh) / ((t1 - t2) * sfh[1:])  # Shape: (i,) - Normalized SFH slope

        # === TIME CLIPPING AND MASS CALCULATION ===
        # Clip times to physically valid range to avoid extrapolation beyond SSP grid
        tq = jnp.clip(t1, 10**self.tiny_logt, 10**self.ssp_ages_lgyr[-1])  # Shape: (i,) - Clipped older times
        tage = jnp.clip(t2, 10**self.tiny_logt, 10**self.ssp_ages_lgyr[-1]) # Shape: (i,) - Clipped younger times
        sf_trunc = tage - tq  # Shape: (i,) - Effective time interval after clipping


        # Calculate total stellar mass formed in each time interval
        # Accounts for linear SFH variation within the interval via trapezoidal rule
        m2 = (
                sfh[1:]  # SFR at younger edge of interval
                * (1 + sf_slope / 2.0 * (tage + tq - 2 * t1))  # Correction for linear SFH slope
                * sf_trunc  # Multiply by time interval duration
            )  # Shape: (i,) - Total stellar mass formed in each interval

        # Calculate parameters for linear SFH interpolation within integration
        tprime = jnp.maximum(0.0, tage - sf_trunc)  # Shape: (i,) - Time offset for slope calculation
        a = 1 - sf_slope * tprime  # Shape: (i,) - Linear interpolation coefficient


        # SSP-related computations
        ssp_dt = jnp.diff(self.ssp_ages_lgyr)  # Time intervals in SSP (shape: (j,))
        logage_lft = self.ssp_ages_lgyr[1:]    # Left edge of log-age bins (shape: (j,))
        logage_rght = self.ssp_ages_lgyr[:-1]  # Right edge of log-age bins (shape: (j,))


        # Broadcasting integration limits
        tq_broadcasted = jnp.log10(tq)[:, None]  # Expand tq for broadcasting (shape: (i, 1))
        tage_broadcasted = jnp.log10(tage)[:, None]  # Expand tage for broadcasting (shape: (i, 1))


        # Compute integration limits with broadcasting
        tlimlo = jnp.clip(logage_lft[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (i, j)
        tlimhi = jnp.clip(logage_rght[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (i, j)


        # Mask computation
        j_indices = jnp.arange(len(self.ssp_ages_lgyr))  # Indices for SSP bins (shape: (j,))
        jmin = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t1)) - 1, 0, len(self.ssp_ages_lgyr) - 1)  # Shape: (i,)
        jmax = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t2)) + 2, 0, len(self.ssp_ages_lgyr) - 1)  # Shape: (i,)


        # Create mask for relevant SSP bins
        mask = (j_indices >= jmin[:, None]) & (j_indices < jmax[:, None])  # Shape: (i, n_time)
        mask_lft = mask[:, 1:]  # Mask for left edges (shape: (i, j))
        mask_rght = mask[:, :-1]  # Mask for right edges (shape: (i, j))

    
        # Broadcast bin edges
        logage_lft_broadcasted = logage_lft[None, :]  # broadcast to (1, j)
        logage_rght_broadcasted = logage_rght[None, :]  # broadcast to (1, j)
 

        a_broadcasted = a[:, None]  # broadcast to (i, 1)
        sf_slope_broadcasted = sf_slope[:, None]  # broadcast to (i, 1)


        # Left weights  
        intsfwght_lft = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_lft_broadcasted) # Shape: (i, j)
        tmp_weights_lft = jnp.zeros_like(intsfwght_lft) - intsfwght_lft / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_lft = jnp.where(mask_lft, tmp_weights_lft, 0.0)


        # Right weights
        intsfwght_rght = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_rght_broadcasted)  # Shape: (i, j)
        tmp_weights_rght = jnp.zeros_like(intsfwght_rght) + intsfwght_rght / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_rght = jnp.where(mask_rght, tmp_weights_rght, 0.0)


        # Combine left and right weights
        result = jnp.zeros((tmp_weights_lft.shape[0], tmp_weights_lft.shape[1]+1))  # Shape: (i, n_time)
        w1 = result.at[:, :-1].add(tmp_weights_lft).at[:, 1:].add(tmp_weights_rght) # Shape: (i, n_time)

        m1 = jnp.sum(w1, axis=1) #shape (i,)
        
        sfh_weights = w1 * (m2[:, None] / m1[:, None])  # Shape: (i, n_time)
        total_sfh_weights = sfh_weights.sum(axis=0) # Shape: (n_time,)

        # Metallicity handling: constant metallicity case
        target_Z = theta[self.Z_idx] # scalar

        # 1. Find interpolation indices along z-axis
        z_idx = jnp.searchsorted(self.zmet, target_Z, side='left')
        z_idx = jnp.clip(z_idx, 1, len(self.zmet) - 1)

        z1 = self.zmet[z_idx - 1]
        z2 = self.zmet[z_idx]
        w = (target_Z - z1) / (z2 - z1)

        # 2. Prepare zeroed weights cube: (n_z, n_age)
        total_weights = jnp.zeros((len(self.zmet), len(self.ages)))  # (n_z, n_age)

        # 3. Add the weights to bins k and k+1 with linear interpolation
        total_weights = total_weights.at[z_idx - 1].add((1 - w) * total_sfh_weights)
        total_weights = total_weights.at[z_idx].add(w * total_sfh_weights)
        
        self.ssp_weights = total_weights
        self.mass_formed = m2
        self.m1 = m1    
        self.m2 = m2
        self.w1 = w1
        
        return total_weights
    


    def calculate_ssp_weights_var_zh(self, theta):
        """
        Calculate SSP weights for CSP spectrum generation.
        
        This is the core method that computes how much each SSP (defined by age and metallicity)
        contributes to the final CSP spectrum based on the star formation and metallicity histories.
        
        The algorithm:
        1. Interpolates SFH linearly between time bins
        2. Computes mass formed in each time interval  
        3. Distributes this mass across SSP age bins via integration
        4. Handles metallicity evolution through linear interpolation
        
        Dimensions: i = number of SFH time bins (len(sfh) - 1), j = number of SSP ages (len(ssp_ages_lgyr) - 1)
        
        Returns:
            total_weights: Array of SSP weights, shape depends on metallicity history:
                         - No metallicity history: (1, n_age)
                         - With metallicity history: (n_metallicity, n_age)
        """
        
        # Ensure SFH values are positive to avoid numerical issues
        sfh = theta[self.sfh_idx]
        sfh = jnp.clip(sfh, 1e-30, None)  # Ensure SFH is non-negative - shape (i+1)

        # === TIME BINNING AND SFH INTERPOLATION SETUP ===
        # Define time intervals from the SFH grid
        t1 = self.sfh_times[1:] # Beginning of time intervals (older times) - shape: (i,)
        t2 = self.sfh_times[:-1] # End of time intervals (younger times) - shape: (i,)

        
        # Compute slope for linear interpolation of SFH between adjacent points
        # This allows smooth SFH evolution within each time bin rather than step functions
        sf_slope = jnp.diff(sfh) / ((t1 - t2) * sfh[1:])  # Shape: (i,) - Normalized SFH slope

        # === TIME CLIPPING AND MASS CALCULATION ===
        # Clip times to physically valid range to avoid extrapolation beyond SSP grid
        tq = jnp.clip(t1, 10**self.tiny_logt, 10**self.ssp_ages_lgyr[-1])  # Shape: (i,) - Clipped older times
        tage = jnp.clip(t2, 10**self.tiny_logt, 10**self.ssp_ages_lgyr[-1]) # Shape: (i,) - Clipped younger times
        sf_trunc = tage - tq  # Shape: (i,) - Effective time interval after clipping


        # Calculate total stellar mass formed in each time interval
        # Accounts for linear SFH variation within the interval via trapezoidal rule
        m2 = (
                sfh[1:]  # SFR at younger edge of interval
                * (1 + sf_slope / 2.0 * (tage + tq - 2 * t1))  # Correction for linear SFH slope
                * sf_trunc  # Multiply by time interval duration
            )  # Shape: (i,) - Total stellar mass formed in each interval

        # Calculate parameters for linear SFH interpolation within integration
        tprime = jnp.maximum(0.0, tage - sf_trunc)  # Shape: (i,) - Time offset for slope calculation
        a = 1 - sf_slope * tprime  # Shape: (i,) - Linear interpolation coefficient


        # SSP-related computations
        ssp_dt = jnp.diff(self.ssp_ages_lgyr)  # Time intervals in SSP (shape: (j,))
        logage_lft = self.ssp_ages_lgyr[1:]    # Left edge of log-age bins (shape: (j,))
        logage_rght = self.ssp_ages_lgyr[:-1]  # Right edge of log-age bins (shape: (j,))


        # Broadcasting integration limits
        tq_broadcasted = jnp.log10(tq)[:, None]  # Expand tq for broadcasting (shape: (i, 1))
        tage_broadcasted = jnp.log10(tage)[:, None]  # Expand tage for broadcasting (shape: (i, 1))


        # Compute integration limits with broadcasting
        tlimlo = jnp.clip(logage_lft[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (i, j)
        tlimhi = jnp.clip(logage_rght[None, :], tq_broadcasted, tage_broadcasted)  # Shape: (i, j)


        # Mask computation
        j_indices = jnp.arange(len(self.ssp_ages_lgyr))  # Indices for SSP bins (shape: (j,))
        jmin = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t1)) - 1, 0, len(self.ssp_ages_lgyr) - 1)  # Shape: (i,)
        jmax = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t2)) + 2, 0, len(self.ssp_ages_lgyr) - 1)  # Shape: (i,)


        # Create mask for relevant SSP bins
        mask = (j_indices >= jmin[:, None]) & (j_indices < jmax[:, None])  # Shape: (i, n_time)
        mask_lft = mask[:, 1:]  # Mask for left edges (shape: (i, j))
        mask_rght = mask[:, :-1]  # Mask for right edges (shape: (i, j))

    
        # Broadcast bin edges
        logage_lft_broadcasted = logage_lft[None, :]  # broadcast to (1, j)
        logage_rght_broadcasted = logage_rght[None, :]  # broadcast to (1, j)
 

        a_broadcasted = a[:, None]  # broadcast to (i, 1)
        sf_slope_broadcasted = sf_slope[:, None]  # broadcast to (i, 1)


        # Left weights  
        intsfwght_lft = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_lft_broadcasted) # Shape: (i, j)
        tmp_weights_lft = jnp.zeros_like(intsfwght_lft) - intsfwght_lft / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_lft = jnp.where(mask_lft, tmp_weights_lft, 0.0)


        # Right weights
        intsfwght_rght = intsfwght(tlimhi, tlimlo, a_broadcasted, sf_slope_broadcasted, logage_rght_broadcasted)  # Shape: (i, j)
        tmp_weights_rght = jnp.zeros_like(intsfwght_rght) + intsfwght_rght / ssp_dt[None, :]  # Shape: (i, j)
        tmp_weights_rght = jnp.where(mask_rght, tmp_weights_rght, 0.0)


        # Combine left and right weights
        result = jnp.zeros((tmp_weights_lft.shape[0], tmp_weights_lft.shape[1]+1))  # Shape: (i, n_time)
        w1 = result.at[:, :-1].add(tmp_weights_lft).at[:, 1:].add(tmp_weights_rght) # Shape: (i, n_time)

        m1 = jnp.sum(w1, axis=1) #shape (i,)
        
        sfh_weights = w1 * (m2[:, None] / m1[:, None])  # Shape: (i, n_time)


        zh = theta[self.zh_idx]  # Shape: (i,)
        zbin = (zh[:-1] + zh[1:]) / 2 # Shape: (i,)  # Compute metallicity bin (zbin) from a simple average of adjacent metallicities
        k = jnp.clip(jnp.searchsorted(self.zlegend, zbin) - 1, 0, len(self.zlegend) - 2)  # Shape: (i,)
                
        # Logarithmic bin size — protect against divide-by-zero
        logz_k     = self.zlegend[k]
        logz_k1    = self.zlegend[k + 1]
        bin_size   = jnp.maximum(logz_k1 - logz_k, tiny_number) # Shape: (i,)

        dz = (jnp.log10(zbin) - jnp.log10(self.zlegend[k])) / bin_size  # Shape: (i,)
        dz = jnp.clip(dz, -1.0, 1.0)  # Clamping dz to avoid extrapolation
        total_weights = jnp.zeros((len(self.sfh_times)-1, len(self.zlegend), len(self.ages))) # Shape: (i, n_z, n_time)

        total_weights = total_weights.at[:, k].add((1 - dz[:, None]) * sfh_weights) # Shape: (i, n_z, n_time)
        total_weights = total_weights.at[:, k + 1].add(dz[:, None] * sfh_weights) # Shape: (i, n_z, n_time)
        total_weights = total_weights.sum(axis=0) #/(len(self.sfh_times)-1)  # Shape: (n_z, n_time)

        # See how much is going into each z bin:
        z_weights = jnp.sum(total_weights, axis=(1))  # shape (n_z,)
            
                
        self.ssp_weights = total_weights
        self.mass_formed = m2
        self.m1 = m1    
        self.m2 = m2
        self.w1 = w1
        
        return total_weights

class Dust:
    """
    JAX-compatible modular dust model that supports multiple attenuation laws per bin.

    Use this to compute attenuation curves given a binning in stellar age
    and a choice of fixed attenuation law per bin. Parameter fitting
    is only applied to the `fit_params` dictionary passed at runtime.

    Call `Dust.describe_attenuation_laws()` to list all available models.
    """

    def __init__(self, bin_edges = [(-jnp.inf, -1.97)], laws = ['powerlaw']):
        """
        Parameters:
            bin_edges (list of tuple): Age bins in Gyr.
            laws (list of str): Attenuation law names (e.g., 'smc', 'kriek_conroy') for each bin.
        """
        assert len(bin_edges) == len(laws), "Must have one dust law per bin"
        self.bin_edges = jnp.array(bin_edges)
        self.num_bins = len(bin_edges)
        self.laws = laws

        self.law_names_resolved = []
        law_name_counter = defaultdict(int)
        law_occurrences = {name: laws.count(name) for name in set(laws)}

        self.law_funcs = []
        self.law_params = []

        for name in laws:
            count = law_name_counter[name]
            law_name_counter[name] += 1

            law_entry = ATTENUATION_LAWS[name]
            base_func = law_entry["func"]
            defaults = law_entry.get("defaults", {})
            params = law_entry.get("params", {})
            doc = law_entry.get("doc", "")

            if law_occurrences[name] == 1:
                # Only used once → keep original name and function
                resolved_name = name
                func = base_func
                param_dict = self._get_law_params(name)
            else:
                # Used multiple times → rename function and parameters
                number = count + 1
                resolved_name = f"{name}{number}"
                func = modify_function(base_func, number, defaults)

                # Register modified version
                ATTENUATION_LAWS[resolved_name] = {
                    "func": func,
                    "defaults": {f"{k}{number}": v for k, v in defaults.items()},
                    "params": {f"{k}{number}": d for k, d in params.items()},
                    "doc": doc
                }

                # Notify user of renaming
                renamed_keys = [f"{k}{number}" for k in defaults.keys()]
                ignored_keys = list(defaults.keys())
                print(f"[dust setup] Law '{name}' is used multiple times. Registered as '{resolved_name}'.")
                print(f"             Use parameters: {', '.join(renamed_keys)}")
                print(f"             Parameters like {', '.join(ignored_keys)} will be ignored.")

                # Rename parameters
                param_dict = self._get_law_params(name)
                param_dict = {f"{k}{number}": v for k, v in param_dict.items()}

            sig = inspect.signature(func)

            # Keep parameters after 'wave' and ignore *args/**kwargs
            ordered_param_names = [
                p.name for p in sig.parameters.values()
                if p.name != "wave"
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]

            ordered_param_names = [n for n in ordered_param_names if n in param_dict]

            # Fallback: if signature-based detection fails, use param_dict keys (stably sorted)
            if not ordered_param_names:
                ordered_param_names = sorted(param_dict.keys())

            missing = [n for n in ordered_param_names if n not in param_dict]
            if missing:
                raise ValueError(f"Parameters {missing} expected for '{resolved_name}' but not found in param_dict")

            # Create the wrapped function (this is done in Python, not inside jit)
            wrapped_func = make_law_wrapper(func, ordered_param_names)
        

            self.law_names_resolved.append(resolved_name)
            self.law_funcs.append(wrapped_func)
            self.law_params.append(param_dict)

    def _get_law_fn(self, name):
        try:
            return ATTENUATION_LAWS[name]["func"]
        except KeyError:
            raise ValueError(f"Unknown attenuation law: '{name}'. Please add '{name}' to the sedpy_jax.attenuation module, as a function and as an entry in the ATTENUATION_LAWS dictionary.")
    
    def _get_law_params(self, name):
        try:
            return ATTENUATION_LAWS[name]["params"]
        except KeyError:
            raise ValueError(f"Unknown attenuation law: '{name}'. Please add '{name}' to the sedpy_jax.attenuation module, as a function and as an entry in the ATTENUATION_LAWS dictionary.")
        
    def __repr__(self):
        def format_array(arr):
            return "\n    " + "\n    ".join(map(str, arr))

        info = [
            "Dust Model Configuration",
            "=" * 60,
            f"Number of bins          : {self.num_bins}",
            f"Bin edges (Gyr)         : {format_array(self.bin_edges)}",
            f"Dust laws               : {', '.join(self.laws)}",
            f"Dust parameters        : {', '.join(map(str, self.law_params))}",
            "-" * 60,
        ]

        for i, lawname in enumerate(self.laws):
            lawinfo = ATTENUATION_LAWS.get(lawname, {})
            doc = lawinfo.get("doc", "No description.")
            defaults = lawinfo.get("defaults", {})
            params = lawinfo.get("params", {})
            info.append(f"Bin {i} → {lawname}")
            info.append(f"  Description: {doc}")
            for p, d in params.items():
                info.append(f"    {p:10s}: {d}")
            info.append(f"  Defaults: {defaults}")
            info.append("-" * 60)

        return "\n".join(info)


    def compute_attenuation(self, wave, fit_params):
        """
        Compute bin-wise attenuation curves using the provided parameters.

        Parameters:
            wave (jnp.ndarray): Wavelength array in Angstroms.
            fit_params (NamedTuple): Parameters for the attenuation laws.
        Returns:
            jnp.ndarray: shape (num_bins, len(wave)) attenuation per bin
        """

        def curve_fn(i, wave):
            return lax.switch(i, self.law_funcs, wave, fit_params)

        curves = vmap(curve_fn, in_axes=(0, None))(jnp.arange(self.num_bins), wave)
        return curves

    def display(self, fit_params=None):
        """
        Display the attenuation curves for each bin using matplotlib.
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        import matplotlib.pyplot as plt
        wave = jnp.linspace(0, 10000, 1000)

        if fit_params is None:
            fit_params = self.get_default_fit_params()

        curves = self.compute_attenuation(wave, fit_params=fit_params)
        fig, ax = plt.subplots()

        for i in range(len(curves)):
            law = self.law_names_resolved[i]
            t_start, t_end = self.bin_edges[i]
            ax.plot(wave, curves[i], label=f"Bin {i+1}: {law} ({t_start:.0f}–{t_end:.0f} Myr)")

        ax.set_xlabel("Wavelength (Angstroms)")
        ax.set_ylabel("Attenuation")
        ax.set_yscale("log")
        ax.legend()
        plt.show()
        return fig, ax
    
    @staticmethod
    def describe_attenuation_laws():
        """
        Print all available dust laws with descriptions and parameter metadata.
        """
        print("=" * 70)
        print("Available Dust Attenuation Laws in sedpy_jax:\n")
        for name, info in ATTENUATION_LAWS.items():
            print(f"• {name}")
            print(f"  Description: {info.get('doc', 'No description.')}")
            print("  Parameters:")
            for param, desc in info.get("params", {}).items():
                print(f"    {param:12s}: {desc}")
            print("-" * 70)

   

    def get_default_fit_params(self):
        from typing import NamedTuple, Any
        """
        Return a NamedTuple of default fit parameters based on chosen dust laws.
        Returns:
            NamedTuple: e.g. FitParams(tau_smc1=..., tau_smc2=..., dust_index=...)
        """
        defaults = {}

        for law in self.law_names_resolved:
            for v, p in ATTENUATION_LAWS[law].get("defaults", {}).items():
                defaults[v] = p

        # Dynamically create a NamedTuple class with fields matching defaults
        FitParams = NamedTuple("FitParams", [(k, Any) for k in defaults.keys()])
        return FitParams(**defaults)
    
    def get_param_names(self):
        """
        Return a list of all parameter names used in the dust laws.
        Returns:
            list of str: Parameter names.
        """
        param_names = []
        for law in self.law_names_resolved:
            param_names.extend(ATTENUATION_LAWS[law].get("params", {}).keys())
        return param_names

def make_law_wrapper(f, param_names):
    """Return a JAX‑traceable wrapper that extracts given params from a NamedTuple."""
    def wrapped(wave, fit_params):
        args = tuple(getattr(fit_params, name) for name in param_names)
        return f(wave, *args)
    return wrapped

# In case the same dust law is applied multiple times, this function takes care of the renaming.

def modify_function(func, number, defaults_dict=None):
    import inspect

    func_name = func.__name__
    sig = inspect.signature(func)

    new_func_name = f"{func_name}{number}"

    new_params = []
    call_params = []
    original_names = []
    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            continue
        new_name = f"{name}{number}" if name != "wave" else f"wave"
        original_names.append(name)

        if param.default is not inspect.Parameter.empty:
            default_val = repr(param.default)
        elif defaults_dict and name in defaults_dict:
            default_val = repr(defaults_dict[name])
        else:
            default_val = None

        if default_val is not None:
            new_params.append(f"{new_name}={default_val}")
        else:
            new_params.append(new_name)

        call_params.append(f"{name}={new_name}")

    arg_str = ", ".join(new_params + ["**kwargs"])
    call_str = f"{func_name}({', '.join(call_params)}, **filtered_kwargs)"

    func_def = f"""
        def {new_func_name}({arg_str}):
            filtered_kwargs = {{k: v for k, v in kwargs.items() if k not in {original_names!r}}}
            return {call_str}
        """

    local_ns = {func_name: func}
    exec(func_def, local_ns)
    return local_ns[new_func_name]

class DiffuseDust(Dust):
    def __init__(self, law="kriek_conroy"):
        """
        A simplified dust model that applies one dust law to all ages (bin spans all time).
        
        Parameters:
            law (str): Name of the attenuation law (must be in ATTENUATION_LAWS).
        """
        # Initialize parent class with one bin spanning all time
        super().__init__(bin_edges=[(-jnp.inf, jnp.inf)], laws=[law])

        # Rename all parameters in the single bin to use prefix 'diffuse_'
        old_name = self.law_names_resolved[0]
        param_dict = self.law_params[0]

        # Create remapped version
        self.diffuse_param_map = {}
        renamed_param_dict = {}
        for k, v in param_dict.items():
            new_k = f"diffuse_{k}"
            renamed_param_dict[new_k] = v
            self.diffuse_param_map[new_k] = k

        # Replace the first bin’s param dict with the renamed version
        self.law_params[0] = renamed_param_dict

        # Also update the wrapped function to use the renamed keys
        func = ATTENUATION_LAWS[old_name]["func"]
        sig = inspect.signature(func)

        param_names = [
            f"diffuse_{p.name}" for p in sig.parameters.values()
            if p.name != "wave"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

        wrapped_func = make_law_wrapper(func, param_names)
        self.law_funcs[0] = wrapped_func

        # Save param name list for reference
        self.dust_param_names = param_names

    def get_default_params(self):
        """
        Return a NamedTuple of default fit parameters, with names prefixed by 'diffuse_'.
        """
        from typing import NamedTuple, Any

        defaults = {}
        law = self.law_names_resolved[0]
        for k, v in ATTENUATION_LAWS[law].get("defaults", {}).items():
            defaults[f"diffuse_{k}"] = v

        FitParams = NamedTuple("DiffuseParams", [(k, Any) for k in defaults.keys()])
        return FitParams(**defaults)
    

    def compute_attenuation(self, wave, fit_params):
        """
        Compute the diffuse attenuation curve (single bin).

        Parameters:
            wave (jnp.ndarray): Wavelength array in Angstroms.
            fit_params (NamedTuple): Parameters for the attenuation law.

        Returns:
            jnp.ndarray: shape (len(wave),) attenuation curve
        """
        # Always use the single law function (index 0)
        return self.law_funcs[0](wave, fit_params)
    
    def get_param_names(self):
        return list(self.law_params[0].keys())

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr.replace("Dust Model Configuration", "Diffuse Dust Model Configuration")


class DustEmission:

    def __init__(self, duste_model="DL07",
                 dust_file=None, spec_lambda=None, **kwargs):
        """
        Initialize the DustEmission object with parameters for dust emission modeling.

        Parameters
        ----------
        duste_model : str
            Dust emission model to use: 'DL07' or 'THEMIS'.
        dust_file : str
            Path to the dust emission file (required).
        spec_lambda : ndarray
            Wavelength grid over which dust emission will be evaluated (required).
        kwargs : dict
            Optional keyword arguments to override default dust parameters.
            Supported: duste_qpah, duste_umin, duste_gamma
        """

        # Store model choice (e.g., 'DL07' or 'THEMIS')
        self.duste_model = duste_model

        # Dust parameter values
        self.duste_qpah = None
        self.duste_umin = None
        self.duste_gamma = None

        # Model grid arrays for allowed values of qPAH and Umin
        self.qpaharr = None
        self.uminarr = None

        # Placeholder for loaded dust emission spectra
        self.dustem2_dustem = None

        # File path and wavelength grid
        self.dust_file = None
        self.spec_lambda = None

        # Store any additional keyword arguments for later use
        self.dwargs = kwargs

        # Read in key dust parameters, allowing overrides via kwargs
        self.duste_qpah = kwargs.pop("duste_qpah", 3.5)
        self.duste_umin = kwargs.pop("duste_umin", 1.0)
        self.duste_gamma = kwargs.pop("duste_gamma", 0.01)

        # Set parameter grids based on selected dust model
        if self.duste_model == "DL07":
            self.qpaharr = jnp.array([0.47,1.12,1.77,2.50,3.19,3.90,4.58])
            self.uminarr = jnp.array([0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.8,1.0,1.2,1.5,
                                      2.0,2.5,3.0,4.0,5.0,7.0,8.0,12.0,15.0,20.0,25.0])
            self.nqpah_dustem = self.qpaharr.size
            self.numin_dustem = self.uminarr.size
        elif self.duste_model == "THEMIS":
            # THEMIS model uses smaller qPAH values rescaled to percent
            self.qpaharr = jnp.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24,
                                      0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
            self.uminarr = jnp.array([
                0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
                0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
                6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
                35.0, 40.0, 50.0, 80.0
            ])
            self.nqpah_dustem = self.qpaharr.size
            self.numin_dustem = self.uminarr.size
        else:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        # Ensure that necessary data is provided
        if dust_file is None or spec_lambda is None:
            raise ValueError("If `duste=True`, both `dust_file` and `spec_lambda` must be provided.")

        self.dust_file = dust_file
        self.spec_lambda = spec_lambda

        # Load emission templates or model data from file
        self.load_dust_emission(dust_file, spec_lambda)

    def __repr__(self):
        """
        Custom string representation of the DustEmission object.
        Provides a readable summary of model settings and parameters.
        """

        def format_array(arr):
            """Helper to format short arrays inline; longer arrays multiline."""
            if arr is None:
                return "None"
            if arr.ndim == 1 and len(arr) <= 5:
                return f"[{', '.join(map(str, arr))}]"
            return f"\n    " + "\n    ".join(map(str, arr))

        attributes = {
            "Duste model": self.duste_model,
            "DUST qPAH": self.duste_qpah,
            "DUST Umin": self.duste_umin,
            "DUST Gamma": self.duste_gamma,
            f"qpaharr ({self.duste_model})": format_array(self.qpaharr),
            f"uminarr ({self.duste_model})": format_array(self.uminarr),
            "dust_file": self.dust_file,
            "spec_lambda": self.spec_lambda.shape if self.spec_lambda is not None else None,
        }

        if self.dwargs:
            attributes["Extra parameters (dwargs)"] = self.dwargs

        attr_str = "\n".join(f"  {k:<30}: {v}" for k, v in attributes.items() if v is not None)
        return f"\nDustEmission Model:\n{'='*50}\n{attr_str}\n{'='*50}"

    def get_default_params(self):
        """
        Return a NamedTuple of default fit parameters, with names prefixed by 'diffuse_'.
        """
        from typing import NamedTuple, Any

        defaults = {
            "duste_qpah": self.duste_qpah,
            "duste_umin": self.duste_umin,
            "duste_gamma": self.duste_gamma
        }
        FitParams = NamedTuple("DustEmissionParams", [(k, Any) for k in defaults.keys()])
        return FitParams(**defaults)
    
    def load_dust_emission(self, dust_file=None, spec_lambda=None):
        
        # Use default paths if not provided
        if dust_file is None:
            dust_file = self.dust_file
        if spec_lambda is None:
            spec_lambda = self.spec_lambda

        # Select dust model parameters
        dust_model_params = {
            "DL07": (7, 1001, 22),
            "THEMIS": (11, 576, 37),
        }
        
        nqpah_dustem, ndim_dustem, numin_dustem = dust_model_params[self.duste_model]

        # Initialize storage for interpolated spectra (JAX-compatible)
        dustem2_dustem = np.zeros((len(spec_lambda), nqpah_dustem, numin_dustem * 2))

        # Read and interpolate dust emission spectra
        for k in range(nqpah_dustem):
            filename = Path(dust_file) / "dust" / "dustem" / f"{self.duste_model}_MW3.1_{'100' if k == 10 else f'{k}0'}.dat"

            if not filename.exists():
                raise FileNotFoundError(f"Error opening dust emission file: {filename}. File does not exist.")

            with filename.open('r') as f:
                next(f)  # Skip first header line
                next(f)  # Skip second header line

                lambda_dustem = np.zeros(ndim_dustem)
                dustem_dustem = np.zeros((ndim_dustem, numin_dustem * 2))

                for i in range(ndim_dustem):
                    try:
                        values = list(map(float, f.readline().strip().split()))
                        lambda_dustem[i], dustem_dustem[i, :] = values[0], values[1:]
                    except Exception:
                        raise RuntimeError(f"Error reading dust emission file: {filename}")

            # Convert wavelength from microns to Angstroms
            lambda_dustem *= 1E4

            # Interpolate dust spectra onto the master wavelength array
            jj = jnp.searchsorted(spec_lambda / 1E4, 1, side='left')
            for j in range(numin_dustem * 2):
                dustem2_dustem[jj:, k, j] = jnp.interp(spec_lambda[jj:], lambda_dustem, dustem_dustem[:, j])

        self.dustem2_dustem = jnp.array(dustem2_dustem)

    def update_dust_params(self, duste_qpah = 3.5, duste_umin = 1.0, duste_gamma = 0.01):
        """
        Update dust parameters for emission calculations.

        Parameters:
            duste_qpah (float, optional): New PAH fraction.
            duste_umin (float, optional): New minimum U radiation field.
            duste_gamma (float, optional): New fraction of high U component.
        """
        self.duste_qpah = duste_qpah
        self.duste_umin = duste_umin
        self.duste_gamma = duste_gamma

    def compute_dust_emission(self, spec_attn, spec_dustfree, spec_lambda, diffuse_curve,
                                        duste_qpah, duste_umin, duste_gamma):
        """
        Compute dust emission using JAX-optimized vectorization for GPU acceleration.

        Parameters:
            spec_attn (jnp.ndarray): Attenuated spectrum after dust absorption.
            spec_dustfree (jnp.ndarray): Stellar spectrum before attenuation.
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            duste_qpah (float, optional): PAH fraction. Defaults to self.duste_qpah if None.
            duste_umin (float, optional): Minimum U radiation field. Defaults to self.duste_umin if None.
            duste_gamma (float, optional): Fraction of high U component. Defaults to self.duste_gamma if None.

        Returns:
            tuple: (Updated spectrum with dust emission added, Estimated dust mass)
        """

        clight = const.c.value  # m/s
        tiny_number = 1e-70
        
        nu = clight * 1e10 / spec_lambda  # Hz

        # --- Compute Lbol before and after attenuation ---
        lbold = jnp.trapezoid(spec_attn, nu)
        lboln = jnp.trapezoid(spec_dustfree, nu)

        # --- QPAH interpolation setup ---
        qlo = jnp.clip(jnp.searchsorted(self.qpaharr, duste_qpah) - 1, 0, self.nqpah_dustem - 2)
        dq = (duste_qpah - self.qpaharr[qlo]) / (self.qpaharr[qlo + 1] - self.qpaharr[qlo])
        dq = jnp.clip(dq, 0.0, 1.0)

        # --- Umin interpolation setup ---
        ulo = jnp.clip(jnp.searchsorted(self.uminarr, duste_umin) - 1, 0, self.numin_dustem - 2)
        du = (duste_umin - self.uminarr[ulo]) / (self.uminarr[ulo + 1] - self.uminarr[ulo])
        du = jnp.clip(du, 0.0, 1.0)
    
        # --- gamma limiting ---
        gamma = jnp.clip(duste_gamma, 0.0, 1.0)

        # --- Bilinear interpolation ---
        dumin = (
            (1 - dq) * (1 - du) * self.dustem2_dustem[:, qlo, 2 * ulo] +
            dq * (1 - du) * self.dustem2_dustem[:, qlo + 1, 2 * ulo] +
            dq * du * self.dustem2_dustem[:, qlo + 1, 2 * (ulo + 1)] +
            (1 - dq) * du * self.dustem2_dustem[:, qlo, 2 * (ulo + 1)] 
        )

        dumax = (
            (1 - dq) * (1 - du) * self.dustem2_dustem[:, qlo, 2 * ulo + 1] +
            dq * (1 - du) * self.dustem2_dustem[:, qlo + 1, 2 * ulo + 1] +
            dq * du * self.dustem2_dustem[:, qlo + 1, 2 * (ulo + 1) + 1] +
            (1 - dq) * du * self.dustem2_dustem[:, qlo, 2 * (ulo + 1) + 1]
        )
    
        # Combine both parts of P(U)dU
        mduste = (1 - gamma) * dumin + gamma * dumax
        mduste = jnp.maximum(mduste, tiny_number)
        mduste = jnp.squeeze(mduste)


        # Normalize the dust emission to match the absorbed luminosity
        labs = lboln - lbold
        norm = jnp.trapezoid(mduste, nu)
        duste = mduste / norm * labs
        self.duste = jnp.maximum(duste, tiny_number)


        # --- Two explicit self-absorption iterations ---
        # Iteration 1
        oduste_1 = duste
        duste_1 = duste * diffuse_curve
        tduste_1 = duste_1
        lbold_1 =jnp.trapezoid(duste_1, nu)
        lboln_1 =jnp.trapezoid(oduste_1, nu)
        duste_1 = jnp.maximum(mduste / norm * (lboln_1 - lbold_1), tiny_number)

        # Iteration 2
        oduste_2 = duste_1
        duste_2 = duste_1 * diffuse_curve
        tduste_2 = tduste_1 + duste_2
        lbold_2 =jnp.trapezoid(duste_2, nu)
        lboln_2 =jnp.trapezoid(oduste_2, nu)
        duste_2 = jnp.maximum(mduste / norm * (lboln_2 - lbold_2), tiny_number)

        # Final results
        specdust_final = spec_attn + tduste_2

        # Dust mass
        mdust = 3.21e-3 / (4 * jnp.pi) * labs / norm

        return specdust_final, mdust, tduste_2
    



print('getting started.')



T_UNIV = 13.8  # Gyr, age of the universe
N_T = 100     # number of time steps for SFH

def gaussian_burst(tau, center_tau, width_tau, amp=1.0):
    """Gaussian in lookback time, evaluated on tau grid (Gyr)."""
    return amp * jnp.exp(-0.5 * ((tau - center_tau) / width_tau)**2)

t = jnp.linspace(1e-2, T_UNIV, N_T)   # Gyr (avoid exactly 0)
tau = T_UNIV - t                      # lookback time (Gyr)

sfr_bimodal = (gaussian_burst(tau, 0.05, 0.03, 1.0) +
               gaussian_burst(tau, 11.0, 0.8, 0.7))
zh = jnp.full_like(t, 10**-4.3477116)  # constant metallicity (Z=0.02)

ssp_data = SSPData.load('/home/aas208/Ceridwen/data/ssp_data.h5') #SSPData.load('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/ssp_data.h5')#

ssp_data.ssp_flux.shape

diffdust = 0.2
diffdust_index = -0.7
young_dust = 10
young_dust_index = -1





def main():
    print('-------------------------')
    print('Start sampling now')
    print('-------------------------')



    csp = CSPBasis(ssp_data, theta_dict = {'lookback_time': tau, 'sfh': sfr_bimodal, 'Z': zh[0]}, tuniv = 13.8, tiny_logt = -70, zh_const = True,
                    add_neb = False, init_neb_params = {"isoc_type": "mist", "cloudy_dust": True}, 
                    add_dust = True, add_diffuse_dust = True, add_dust_emission = False, 
                    init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 
                                    'laws': ['powerlaw']}, verbose = True)

    def run_spectrum(csp, theta):
        return csp.get_spectrum(theta)

    # --- Compile the function, marking `csp` as static
    run_spectrum_jit = jit(run_spectrum, static_argnums=0)

    #-------------------------------
    # CSP SPECTRUM WITHOUT DUST
    #-------------------------------

    theta = jnp.array(csp.theta_init.at[csp.tau_pow_idx].set(0))
    theta = jnp.array(theta.at[csp.alpha_idx].set(0))
    theta = jnp.array(theta.at[csp.diffuse_dust_index_idx].set(0))
    theta = jnp.array(theta.at[csp.diffuse_tau_kc_idx].set(0))

    spec_dustfree = run_spectrum_jit(csp, theta)

    #-------------------------------
    # CSP SPECTRUM WITH ALL DUST
    #-------------------------------

    theta = jnp.array(csp.theta_init.at[csp.tau_pow_idx].set(young_dust))
    theta = jnp.array(theta.at[csp.alpha_idx].set(young_dust_index))
    theta = jnp.array(theta.at[csp.diffuse_dust_index_idx].set(diffdust_index))
    theta = jnp.array(theta.at[csp.diffuse_tau_kc_idx].set(diffdust))

    spec_dust = run_spectrum_jit(csp, theta)


    #-------------------------------
    # CSP SPECTRUM WITH DIFFUSE DUST ONLY
    #-------------------------------

    theta = jnp.array(csp.theta_init.at[csp.tau_pow_idx].set(0))
    theta = jnp.array(theta.at[csp.alpha_idx].set(0))
    theta = jnp.array(theta.at[csp.diffuse_dust_index_idx].set(diffdust_index))
    theta = jnp.array(theta.at[csp.diffuse_tau_kc_idx].set(diffdust))

    spec_diffuse = run_spectrum_jit(csp, theta)

    #-------------------------------
    # CSP SPECTRUM WITH Young DUST ONLY
    #-------------------------------

    theta = jnp.array(csp.theta_init.at[csp.tau_pow_idx].set(young_dust))
    theta = jnp.array(theta.at[csp.alpha_idx].set(young_dust_index))
    theta = jnp.array(theta.at[csp.diffuse_dust_index_idx].set(0))
    theta = jnp.array(theta.at[csp.diffuse_tau_kc_idx].set(0))

    spec_young_dust = run_spectrum_jit(csp, theta)


    pprint(csp.params)


    print('dusty spectra generated\n')


    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True)

    # --- Data ---
    waves = csp.wave
    cases = [
        ('No Dust', spec_dustfree, 'dodgerblue'),
        ('Diffuse Dust Only', spec_diffuse, 'dodgerblue'),
        ('Young Dust Only', spec_young_dust, 'dodgerblue'),
        ('Full Dust', spec_dust, 'dodgerblue'),
    ]

    # --- Spectra (single row) ---
    for i, (label, model, color) in enumerate(cases):
        ax = axes[i]
        ax.loglog(waves, model, color=color, lw=1)
        ax.set_ylim(1e-20, 1e-2)
        ax.set_title(label, fontsize=9)
        if i == 0:
            ax.set_ylabel('Flux (erg/s/cm²/Å)')
        ax.set_xlabel('Wavelength (Å)')

    plt.tight_layout()
    plt.savefig('example_spectra.png')

    # ---------------------
    # SAMPLER
    # ---------------------

    print("Backend platform:", xla_bridge.get_backend().platform)
    print("JAX devices:", jax.devices())
    

    # -------------------------- MAKE THE MODEL FOR THE LIKELIHOOD FUNCTION -------------------------
    def run_spectrum(csp, theta):
        return csp.get_spectrum(theta)

    # --- Compile the function, marking `csp` as static
    run_spectrum_jit = jit(run_spectrum, static_argnums=0)


    # --- Use the model's own initial parameter vector ---
    theta0 = csp.theta_init

    # Run once to ensure the model executes
    spec0 = run_spectrum_jit(csp, theta0)
    print("Spectrum shape:", spec0.shape)

    # --- Check that gradients flow properly ---
    grad_fn = grad(lambda th: jnp.sum(run_spectrum_jit(csp, th)))
    grads = grad_fn(theta0)

    print("Gradient shape:", grads.shape)
    print("First 10 gradient elements:", grads[:10])
    print("Gradients computed successfully.")

    # -------------------------- SET UP THE LIKELIHOOD FUNCTION -------------------------

    @partial(jit, static_argnames=['csp'])
    def loglikelihood_fn(theta, sigma, spectrum_obs, csp):
        mu = run_spectrum_jit(csp, theta)
        cov = sigma ** 2
        return jax.scipy.stats.multivariate_normal.logpdf(spectrum_obs, mu, cov)
    
    # ------------------------ Get fake observation ------------------------------------
    # Create a PRNG key (you can fix or vary this as needed)
    key = jax.random.PRNGKey(0)

    # Define the noise amplitude (e.g. 1% of the mean flux)
    noise_level = 0.01 * jnp.mean(spec_dust)

    # Generate Gaussian noise with mean 0 and std = noise_level
    noise = noise_level * jax.random.normal(key, shape=spec_dust.shape)

    # Add to your spectrum
    spectrum_obs = spec_dust + noise
  
    sigma = 0.01

    grad_loglike = jax.grad(lambda th: loglikelihood_fn(th, sigma, spectrum_obs, csp))
    g = grad_loglike(csp.theta_init)
    print("Gradient shape:", g.shape)
    print("First 10 grad elements:", g[:10])

    plt.loglog(csp.wave, spectrum_obs, color = 'crimson')
    plt.loglog(csp.wave, spec_dust, color = 'black')
    plt.ylim(1e-20, 1e-2)
    plt.savefig('spectrum_obs.png')

    def get_param_slices(csp):
        """Convert csp.params position strings into slice or int indices."""
        param_slices = {}
        for name, meta in csp.params.items():
            pos = meta["pos"]
            if "–" in pos:
                start, end = map(int, pos.split("–"))
                param_slices[name] = slice(start, end + 1)
            else:
                param_slices[name] = int(pos)
        return param_slices

    param_slices = get_param_slices(csp)
    print('got the param slices.')

    # -------------------------- SET UP THE PRIOR FUNCTION -------------------------
    from blackjax.ns.utils import uniform_prior

    N_T = 100  # number of time bins for SFH and metallicity

    # --- Define uniform prior bounds for the bounded parameters ---
    prior_bounds = {
        **{f"sfh_{i}": (0.0, 1.0) for i in range(N_T)},       # flat 0–1
        **{f"zh_{i}": (4e-5, 1.0) for i in range(N_T)},       # flat 4e-5–1
        "alpha": (-1.0, 2.0),                                 # flat -1–2
        "sigma": (1e-7, 1e-5)                                 # example noise amplitude
    }

    # Parameters with Gaussian priors (we’ll handle these separately)
    gaussian_priors = {
        "diffuse_dust_index": {"mu": -0.5, "sigma": 0.3},
        "diffuse_tau_kc": {"mu": 0.2, "sigma": 0.3}
    }

    # --- Initialize BlackJAX prior ---
    rng_key = jax.random.PRNGKey(0)
    num_dims = len(prior_bounds)
    num_live = 1000
    num_inner_steps = num_dims * 5
    num_delete = num_live // 2

    rng_key, prior_key = jax.random.split(rng_key)
    particles, logprior_uniform = uniform_prior(prior_key, num_live, prior_bounds)



    from jax.scipy.stats import norm

    @partial(jit, static_argnames=['logprior_uniform'])
    def logprior_fn(theta, logprior_uniform):
        """
        Combined log-prior using uniform priors from BlackJAX
        and Gaussian priors for the dust parameters.
        """
        # Uniform component (from BlackJAX utility)
        logp_uniform = logprior_uniform(theta)

        ps = get_param_slices(csp)
        diffuse_dust_index = theta[ps["diffuse_dust_index"]]
        diffuse_tau_kc = theta[ps["diffuse_tau_kc"]]

        logp_dust_index = norm.logpdf(diffuse_dust_index, loc=-0.5, scale=0.3)
        logp_tau_kc = norm.logpdf(diffuse_tau_kc, loc=0.2, scale=0.3)

        return logp_uniform + logp_dust_index + logp_tau_kc

    @partial(jit, static_argnames=['logprior_uniform', 'csp'])
    def logposterior_fn(theta, sigma, spectrum_obs, csp, logprior_uniform):
        lp = logprior_fn(theta, logprior_uniform)
        ll = loglikelihood_fn(theta, sigma, spectrum_obs, csp)
        return lp + ll

    # -------------------------- TEST -------------------------
    theta0 = csp.theta_init
    lp0 = logprior_fn(theta0, logprior_uniform)
    ll0 = loglikelihood_fn(theta0, sigma, spectrum_obs, csp)

    print("Initial log-prior:", lp0)
    print("Initial log-likelihood:", ll0)
    print("Initial log-posterior:", lp0 + ll0)


    @partial(jit, static_argnames=['csp'])
    def loglikelihood_fn(theta, sigma, spectrum_obs, csp):
        mu = run_spectrum_jit(csp, theta)
        cov = sigma ** 2
        return jax.scipy.stats.multivariate_normal.logpdf(spectrum_obs, mu, cov)

    # Example parameters
    N = 10
    D = csp.theta_init.shape[0]
    params_test = jnp.stack([csp.theta_init + 0.01 * jax.random.normal(jax.random.PRNGKey(i), (D,))
                            for i in range(N)])

    # Vectorized evaluation
    vmap_ll = jax.vmap(loglikelihood_fn, in_axes=(0, None, None, None))
    lls = vmap_ll(params_test, sigma, spectrum_obs, csp)

    print("Log-likelihoods shape:", lls.shape)
    print("First few values:", lls[:5])
    
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

