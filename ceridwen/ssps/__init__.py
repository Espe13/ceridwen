"""
__init__.py for the ssps subfolder.

This subfolder contains modules and utilities for handling Simple Stellar Population (SSP) data.
"""

from .ssp_data import SSPData, collect_ssp_data, collect_ssp_data_wrapper
# Alpha-enhanced 4-D grids (leading [alpha/Fe] axis) for csp.CSPBasis_afe.
from .ssp_data_afe import SSPDataAfe, FSPS_AFE_VALUES_NAFE5
# SSPBasis / FastStepBasis live in ssp_basis.py.
from .ssp_basis import SSPBasis, FastStepBasis
# Published-grid download helpers.
from .grid_fetch import fetch_grid, available_grids, grid_cache_dir

__all__ = ['SSPData', 'SSPDataAfe', 'FSPS_AFE_VALUES_NAFE5',
           'collect_ssp_data', 'collect_ssp_data_wrapper',
           'SSPBasis', 'FastStepBasis',
           'fetch_grid', 'available_grids', 'grid_cache_dir']

__author__ = 'Amanda Stoffers'
