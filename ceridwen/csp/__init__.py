"""
Composite Stellar Population (CSP)

Modeling composite stellar populations (CSPs) by combining Simple Stellar Population (SSP)
spectra according to complex star formation and metallicity evolution histories.

The CSP framework implements the flexible stellar population synthesis approach,
where galaxy spectra are constructed by weighting and integrating SSP models across
age and metallicity grids based on the input evolutionary histories.

Main Components:
    CSPBasis: Primary class for CSP modeling and spectrum generation (solar-scaled
        3-D SSP grids, full nebular machinery)
    CSPBasis_afe: [alpha/Fe]-aware, nebular-free variant (4-D SSPDataAfe grids only)
    SVDCSPBasis: SVD-accelerated variant of CSPBasis (optional)
    spectrum = csp.get_spectrum()
"""

from .csp import CSPBasis, fnu2flam
from .csp_afe import CSPBasis_afe

# SVD-accelerated variant is optional — not present in all installations
try:
    from .csp_svd import SVDCSPBasis
    _HAS_SVD = True
except ImportError:
    SVDCSPBasis = None
    _HAS_SVD = False

__author__ = "Amanda Stoffers"

__all__ = ["CSPBasis", "CSPBasis_afe", "fnu2flam"] + \
    (["SVDCSPBasis"] if _HAS_SVD else [])
