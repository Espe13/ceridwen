"""
__init__.py for the ssps subfolder.

This subfolder contains modules and utilities for handling Simple Stellar Population (SSP) data.
"""

from .ssp_data import SSPData, collect_ssp_data, collect_ssp_data_wrapper
# SSPBasis / FastStepBasis live in ssp_basis.py.
from .ssp_basis import SSPBasis, FastStepBasis

__all__ = ['SSPData', 'collect_ssp_data', 'collect_ssp_data_wrapper', 'SSPBasis', 'FastStepBasis']

__author__ = 'Amanda Stoffers'