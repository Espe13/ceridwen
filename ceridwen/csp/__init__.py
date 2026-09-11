"""Composite stellar population bases: ``CSPBasis`` (solar-scaled grids, nebular model)
and ``CSPBasis_afe`` ([alpha/Fe] grids, no nebular model)."""

from .csp import CSPBasis, fnu2flam
from .csp_afe import CSPBasis_afe

__author__ = "Amanda Stoffers"

__all__ = ["CSPBasis", "CSPBasis_afe", "fnu2flam"]
