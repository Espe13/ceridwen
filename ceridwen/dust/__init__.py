"""
This module provides the Dust class for handling dust attenuation and emission in spectral modeling.
"""

# Import the Dust class from the corresponding file
from .DustModel import Dust, DiffuseDust
from .DustEmission import DustEmission

# Define what gets imported with `from package import *`
__all__ = ["Dust", "DiffuseDust", "DustEmission", "attn_power_law", "attn_kriek_conroy"]