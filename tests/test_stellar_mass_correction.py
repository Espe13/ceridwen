"""The truncated-isochrone correction of FSPS's surviving mass (ceridwen.ssps.stellar_mass).

FSPS (``imf_weight.f90``, ``ssp_gen.f90``) gives the first isochrone point the NUMBER of
stars from ``imf_lower_bound`` (0.08 M_sun) up to it, at that point's mass.  A young MIST
isochrone starting at 2.68 M_sun so gets up to 4.6 M_sun per M_sun formed.  The reference here
is an analytic limit: an isochrone on which no star has lost mass or died (mact = mini, no
remnants) holds exactly the formed mass, 1 M_sun per M_sun formed, up to the midpoint
discretisation of FSPS's IMF sum.  The FSPS-style weights below are an independent
implementation (numpy trapezoid on a fine grid), not the module's quadrature.
"""
from __future__ import annotations

import numpy as np
import pytest

from ceridwen.ssps.stellar_mass import (STELLAR_MASS_MAX, imf_number_density,
                                        truncated_isochrone_correction)

LO, HI = 0.08, 120.0


def _chabrier(m):
    """FSPS imf.f90 imf_type 1, written out again: dN/dm."""
    m = np.asarray(m, dtype=np.float64)
    ln = np.exp(-(np.log10(m) - np.log10(0.08)) ** 2 / (2 * 0.69 ** 2))
    pl = np.exp(-np.log10(0.08) ** 2 / (2 * 0.69 ** 2)) * m ** -1.3
    return np.where(m < 1.0, ln, pl) / m


def _trap(f, a, b, n=200001):
    x = np.geomspace(a, b, n)
    y = f(x)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _fsps_style(mini, mact):
    """FSPS IMF_WEIGHT + SSP_GEN (stars only): the first bin starts at the IMF's lower limit."""
    norm = _trap(lambda m: m * _chabrier(m), LO, HI)
    edges = np.concatenate([[LO], 0.5 * (mini[1:] + mini[:-1]), [mini[-1]]])
    w = np.array([_trap(_chabrier, a, b, 20001) for a, b in zip(edges[:-1], edges[1:])]) / norm
    return float(np.sum(w * mact)), w


@pytest.mark.parametrize("m_first", [2.678, 0.709, 0.131])
def test_truncated_young_isochrone_holds_the_formed_mass(m_first):
    imf = {"imf_type": 1, "imf_lower_limit": LO, "imf_upper_limit": HI,
           "imf1": 1.3, "imf2": 2.3, "imf3": 2.3}
    trunc = np.geomspace(m_first, HI, 400)          # truncated (pre-main-sequence) isochrone
    full = np.geomspace(0.1, HI, 400)               # a complete one: sets the row's floor
    fsps_t, w_t = _fsps_style(trunc, trunc)
    fsps_f, w_f = _fsps_style(full, full)
    if m_first > 1.0:
        assert fsps_t > 2.0, "the FSPS lumping this corrects is absent from the setup"
    iso = np.array([[[6.0, trunc[0], trunc[1], trunc[0], np.log10(w_t[0])],
                     [7.0, full[0], full[1], full[0], np.log10(w_f[0])]]])
    fixed, rep = truncated_isochrone_correction(np.array([[fsps_t, fsps_f]]), iso, imf)
    assert rep["n_corrected"] == 1
    assert fixed[0, 1] == fsps_f, "a complete isochrone keeps FSPS's value bit for bit"
    assert fixed[0, 0] == pytest.approx(1.0, abs=2e-3)
    assert fixed[0, 0] <= STELLAR_MASS_MAX
    assert rep["max_abs_dlogw_vs_fsps"] < 1e-5      # the module's IMF is FSPS's


def test_mass_loss_of_the_first_point_is_kept():
    """The correction scales the bin by mact_1 / mini_1, so a first point that has lost 10 %
    of its mass removes 10 % of the bin's mass, as FSPS's own term would."""
    imf = {"imf_type": 1, "imf_lower_limit": LO, "imf_upper_limit": HI,
           "imf1": 1.3, "imf2": 2.3, "imf3": 2.3}
    trunc = np.geomspace(2.678, HI, 400)
    mact = trunc.copy()
    mact[0] *= 0.9
    fsps, w = _fsps_style(trunc, mact)
    iso = np.array([[[6.0, trunc[0], trunc[1], mact[0], np.log10(w[0])],
                     [7.0, 0.1, 0.11, 0.1, -1.0]]])
    fixed, _ = truncated_isochrone_correction(np.array([[fsps, 0.9]]), iso, imf)
    dn = np.vectorize(imf_number_density(imf))
    top = 0.5 * (trunc[0] + trunc[1])
    m_bin = _trap(lambda m: m * dn(m), LO, top) / _trap(lambda m: m * _chabrier(m), LO, HI)
    assert fixed[0, 0] == pytest.approx(fsps - w[0] * mact[0] + 0.9 * m_bin, rel=1e-6)


def test_unsupported_imf_refused():
    with pytest.raises(ValueError, match="imf_type 0, 1 and 2 only"):
        imf_number_density({"imf_type": 3})


def test_first_isochrone_points_parses_fsps_table(tmp_path, monkeypatch):
    """FSPS's write_isoc format (list-directed header with a leading space; ages in blocks;
    a one-point isochrone), read back into (age, mini_1, mini_2, mact_1, logw_1); the scratch
    file is removed."""
    from ceridwen.ssps.stellar_mass import first_isochrone_points
    (tmp_path / "OUTPUTS").mkdir()
    monkeypatch.setenv("SPS_HOME", str(tmp_path))
    rows = [(5.0, 2.677882280, 2.677880320, 0.1993), (5.0, 2.70, 2.69, -1.5),
            (5.0, 3.0, 2.9, -1.7), (6.5, 0.1, 0.1, -0.5), (6.5, 0.11, 0.11, -0.9),
            (7.0, 0.1, 0.099, -0.4)]

    class Driver:
        def write_isoc(self, name):
            with open(tmp_path / "OUTPUTS" / f"{name}.cmd", "w") as f:
                f.write(" # age log(Z) mini mact logl logt logg phase composition "
                        "log(weight) log(mdot) mags\n")
                for a, mi, ma, lw in rows:
                    f.write(f"{a:7.4f} {-1.4828:8.4f} {mi:14.9f} {ma:14.9f} 1.0 3.6 2.8 "
                            f"-1.0 0.4 {lw:8.4f} -11.1 0.8 3.3\n")

    got = first_isochrone_points(None, Driver())
    assert len(got) == 3
    assert got[0] == pytest.approx((5.0, 2.677882280, 2.70, 2.677880320, 0.1993))
    assert got[1] == pytest.approx((6.5, 0.1, 0.11, 0.1, -0.5))
    assert got[2][:2] == (7.0, 0.1) and np.isnan(got[2][2])
    assert not list((tmp_path / "OUTPUTS").iterdir())
