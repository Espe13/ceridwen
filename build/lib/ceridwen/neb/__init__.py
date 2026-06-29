"""
Nebular emission models for ceridwen.

The public nebular model is:

* :class:`NebularModel` — default, *physically strict*: each CLOUDY cube
  (continuum and lines) is interpolated against its own
  ``(logZ, age, logU)`` grid, so the returned spectrum corresponds to
  the parameters CLOUDY was actually run at.  This is the recommended
  class for new work.

``NebularModelFSPSMatch`` (in ``NebularGridModel_fsps_match.py``) is a
*backup, FSPS-bug-replicating* variant that reproduces FSPS's run-time
behaviour bit-for-bit (both cubes interpolated against the ``.lines`` cube's
axes).  It is intentionally **no longer part of the public namespace** — it is
an internal implementation detail consumed only by ``CSPBasis(..., match_fsps=
True)`` and the FSPS-comparison validation scripts.  Import it explicitly from
``ceridwen.neb.NebularGridModel_fsps_match`` if you need posterior-level
reproducibility against an upstream FSPS install.

The SVD-accelerated subclass is optional and inherits from the strict
:class:`NebularModel`.
"""

from .NebularGridModel import NebularModel

# SVD-accelerated variant is optional — not present in all installations
try:
    from .NebularGridModelSVD import NebularModelSVD
    _HAS_SVD = True
except ImportError:
    NebularModelSVD = None
    _HAS_SVD = False

__all__ = ["NebularModel"] + (["NebularModelSVD"] if _HAS_SVD else [])
