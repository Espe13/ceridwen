"""The Euclid NISP Y, J, H passbands (ceridwen/data/filters/euclid_nisp_*.par) load through
ceridwen.observation.filters and match the official release.

Provenance (in each file's header): Euclid Consortium (2022), NISP-PHOTO-PASSBANDS V1.0, ESA,
doi:10.5270/esa-kx8w57c, column T_TOTAL, nm converted to Angstrom; Euclid Collaboration:
Schirmer et al. 2022, A&A 662, A92. Pivot wavelengths: the SVO Filter Profile Service
(Euclid/NISP.Y, .J, .H; read 2026-10-14), computed from the same release.
"""
import numpy as np
import pytest

from ceridwen.observation.filters import Filter, list_available_filters

PIVOT_SVO = {"y": 10785.39, "j": 13620.63, "h": 17648.80}   # Angstrom


def test_nisp_filters_are_listed():
    assert {"euclid_nisp_y", "euclid_nisp_j", "euclid_nisp_h"} <= set(
        list_available_filters("euclid"))


@pytest.mark.parametrize("band", ["y", "j", "h"])
def test_nisp_filter_loads_with_the_release_pivot(band):
    f = Filter(f"euclid_nisp_{band}")
    assert f.npts > 300
    assert np.all(np.diff(f.wavelength) > 0)
    assert 0.0 <= f.transmission.min() and f.transmission.max() < 1.0
    # Angstrom (the release is in nm): a unit slip would move the pivot by 10x
    assert f.wave_pivot == pytest.approx(PIVOT_SVO[band], rel=1e-5)
