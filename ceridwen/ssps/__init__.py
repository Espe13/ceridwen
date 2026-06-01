"""
__init__.py for the ssps subfolder.

This subfolder contains modules and utilities for handling Simple Stellar Population (SSP) data.
"""

# Import the main components from ssp_data.py for easy access
from .ssp_data import SSPData, collect_ssp_data, collect_ssp_data_wrapper
# SSPBasis / FastStepBasis live in ssp_basis.py (not ssp.py — the previous
# commented-out import named the wrong module, which is why
# ``from ceridwen.ssps import SSPBasis`` raised AttributeError).  CSPSpecBasis
# is referenced in the old __all__ but is not defined anywhere in the package,
# so it is dropped.
from .ssp_basis import SSPBasis, FastStepBasis

__all__ = ['SSPData', 'collect_ssp_data', 'collect_ssp_data_wrapper', 'SSPBasis', 'FastStepBasis']

__author__ = 'Amanda Stoffers'