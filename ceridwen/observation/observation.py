"""Backward-compatible shim.

The Observation containers were split (review 2026-06-01) into
``base.py`` (Observation), ``gp.py`` (GaussianProcess), ``photometry.py``
(Photometry), ``spectrum.py`` (Spectrum) and ``lines.py`` (Lines).  Importing
from the historical path ``ceridwen.observation.observation`` still works via
these re-exports, so no downstream import needs to change.
"""
from .gp import GaussianProcess
from .base import Observation
from .photometry import Photometry
from .spectrum import Spectrum
from .lines import Lines

__all__ = ["Observation", "Photometry", "Spectrum", "Lines", "GaussianProcess"]
