"""Dust attenuation and emission models."""

from .DustModel import Dust, DiffuseDust
from .DustEmission import DustEmission

__all__ = ["Dust", "DiffuseDust", "DustEmission"]
