"""
Composite Stellar Population (CSP) 

Modeling composite stellar populations (CSPs) by combining Simple Stellar Population (SSP)
spectra according to complex star formation and metallicity evolution histories.

The CSP framework implements the flexible stellar population synthesis approach,
where galaxy spectra are constructed by weighting and integrating SSP models across
age and metallicity grids based on the input evolutionary histories.

Main Components:
    CSPBasis: Primary class for CSP modeling and spectrum generation
    add_sfh: Utility function for adding star formation histories
    add_zh: Utility function for adding metallicity evolution histories
    intsfwght: Core integration function for stellar formation weights

Usage:
    from ceridwen.csp import CSPBasis, add_sfh, add_zh
    
    # Create CSP model from SSP data
    csp = CSPBasis(ssp_data)
    
    # Add star formation and metallicity histories
    csp.add_sfh(sfh_values, lookback_time=times)
    csp.add_zh(metallicities, lookback_time=times)
    
    # Generate composite spectrum
    spectrum = csp.get_spectrum()
"""

# Import main classes and functions from the csp module
from .csp import CSPBasis

# Package metadata
__version__ = "1.0.0"
__author__ = "Amanda Stoffers"

# Define what gets imported with "from ceridwen.csp import *"
__all__ = [
    "CSPBasis",
    "intsfwght",
    "fnun2flam"
]