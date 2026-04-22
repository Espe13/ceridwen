"""
Composite Stellar Population (CSP) 

Modeling composite stellar populations (CSPs) by combining Simple Stellar Population (SSP)
spectra according to complex star formation and metallicity evolution histories.

The CSP framework implements the flexible stellar population synthesis approach,
where galaxy spectra are constructed by weighting and integrating SSP models across
age and metallicity grids based on the input evolutionary histories.

Main Components:
    CSPBasis: Primary class for CSP modeling and spectrum generation
    spectrum = csp.get_spectrum()
"""

# Import main classes and functions from the csp module
from .csp import CSPBasis

# SVD-accelerated variant is optional — not present in all installations
try:
    from .csp_svd import SVDCSPBasis
    _HAS_SVD = True
except ImportError:
    SVDCSPBasis = None
    _HAS_SVD = False

# Package metadata
__version__ = "1.0.0"
__author__ = "Amanda Stoffers"

# Define what gets imported with "from ceridwen.csp import *"
__all__ = ["CSPBasis"] + (["SVDCSPBasis"] if _HAS_SVD else []) + ["fnun2flam"]