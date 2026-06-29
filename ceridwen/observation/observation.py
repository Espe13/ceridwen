"""Backward-compatible shim.

The Observation containers live in ``base.py`` (Observation), ``gp.py``
(GaussianProcess), ``photometry.py`` (Photometry), ``spectrum.py``
(Spectrum) and ``lines.py`` (Lines).  Importing from
``ceridwen.observation.observation`` still works via these re-exports.
"""
from .gp import GaussianProcess
from .base import Observation
from .photometry import Photometry
from .spectrum import Spectrum
from .lines import Lines

__all__ = ["Observation", "Photometry", "Spectrum", "Lines", "GaussianProcess"]
