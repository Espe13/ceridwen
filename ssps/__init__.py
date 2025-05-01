"""
__init__.py for the ssps subfolder.

This subfolder contains modules and utilities for handling Simple Stellar Population (SSP) data.
"""

# Import the main components from ssp_data.py for easy access
from .ssp_data import SSPData, collect_ssp_data, collect_ssp_data_wrapper
from .ssp import SSPBasis

__all__ = ['SSPData', 'collect_ssp_data', 'collect_ssp_data_wrapper', 'SSPBasis', 'FastStepBasis', 'CSPSpecBasis']

__author__ = 'Amanda Stoffers'