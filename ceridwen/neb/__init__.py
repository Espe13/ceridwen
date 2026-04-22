"""
This module provides the Dust class for handling dust attenuation and emission in spectral modeling.
"""

# Import the Dust class from the corresponding file
from .NebularGridModel import NebularModel

# SVD-accelerated variant is optional — not present in all installations
try:
    from .NebularGridModelSVD import NebularModelSVD
    _HAS_SVD = True
except ImportError:
    NebularModelSVD = None
    _HAS_SVD = False

# Define what gets imported with `from package import *`
__all__ = ["NebularModel"] + (["NebularModelSVD"] if _HAS_SVD else [])