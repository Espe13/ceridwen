"""
Nebular emission models for ceridwen.

This package exposes two variants of the CLOUDY-grid nebular model:

* :class:`NebularModel` — default, *physically strict*: each CLOUDY cube
  (continuum and lines) is interpolated against its own
  ``(logZ, age, logU)`` grid, so the returned spectrum corresponds to
  the parameters CLOUDY was actually run at.  This is the recommended
  class for new work.

* :class:`NebularModelFSPSMatch` — backup, *FSPS-compatible*:
  reproduces FSPS's run-time behaviour bit-for-bit (both cubes
  interpolated against the ``.lines`` cube's axes) and therefore
  agrees with FSPS to better than 0.5% on SFH-integrated nebular
  spectra.  Use only when you need posterior-level reproducibility
  against an upstream FSPS install; see the file's docstring for
  details on why FSPS's convention is inconsistent.

The SVD-accelerated subclass is optional and inherits from the strict
:class:`NebularModel`.
"""

from .NebularGridModel            import NebularModel
from .NebularGridModel_fsps_match import NebularModelFSPSMatch

# SVD-accelerated variant is optional — not present in all installations
try:
    from .NebularGridModelSVD import NebularModelSVD
    _HAS_SVD = True
except ImportError:
    NebularModelSVD = None
    _HAS_SVD = False

__all__ = ["NebularModel", "NebularModelFSPSMatch"] + (
    ["NebularModelSVD"] if _HAS_SVD else []
)
