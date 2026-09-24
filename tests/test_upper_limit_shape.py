"""B2-006: an upper_limit of the wrong length is refused at construction (a length-1 flag
broadcast silently over every band)."""
import numpy as np
import pytest

from ceridwen.observation import Lines, Photometry

FILTERS = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]


@pytest.mark.parametrize("ul", [[True], [True, False]])
def test_photometry_upper_limit_length(ul):
    with pytest.raises(ValueError, match="upper_limit shape"):
        Photometry(filters=FILTERS, flux=np.ones(5), uncertainty=np.ones(5), upper_limit=ul)


def test_lines_upper_limit_length_and_ok():
    with pytest.raises(ValueError, match="upper_limit shape"):
        Lines(line_ind=[59, 61, 62], wavelength=[4862.76, 4960.37, 5008.31], flux=np.ones(3),
              uncertainty=np.ones(3), upper_limit=[True, False])
    p = Photometry(filters=FILTERS, flux=np.ones(5), uncertainty=np.ones(5),
                   upper_limit=[True, False, False, False, False])
    assert p.upper_limit.shape == (5,)
