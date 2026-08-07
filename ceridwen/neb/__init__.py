"""
Nebular emission models for ceridwen.

The public nebular model is:

* :class:`NebularModel` — default, *physically strict*: each CLOUDY cube
  (continuum and lines) is interpolated against its own
  ``(logZ, age, logU)`` grid, so the returned spectrum corresponds to
  the parameters CLOUDY was actually run at.  This is the recommended
  class for new work.
"""

from .NebularGridModel import NebularModel



__all__ = ["NebularModel"]
