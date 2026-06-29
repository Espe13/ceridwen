"""
This module provides the Dust class for handling dust attenuation and emission in spectral modeling.
"""

from .DustModel import Dust, DiffuseDust
from .DustEmission import DustEmission

__all__ = ["Dust", "DiffuseDust", "DustEmission", "attn_power_law", "attn_kriek_conroy"]