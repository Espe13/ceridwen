"""Nebular emission: ``NebularModel`` interpolates the CLOUDY continuum and line cubes,
each against its own (logZ, age, logU) axes."""

from .NebularGridModel import NebularModel

__all__ = ["NebularModel"]
