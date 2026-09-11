"""Does the galaxy velocity dispersion change broad-band photometry?

Pure NumPy (no ceridwen import): a synthetic SED with the sharpest features
a galaxy has (Lyman and Balmer breaks, deep Balmer absorption, emission
lines with rest EW up to 1000 A) is pushed through real filter curves with
and without a velocity broadening of 300 and 1000 km/s.  The photometry is
the AB maggie integral  int T f_nu dlambda/lambda / int T f_AB dlambda/lambda.

The worst case is deliberately constructed: an emission line placed
exactly on the half-maximum edge of a narrow-band filter (JWST F187N,
edge ~140 A wide) and inside a 63 A wide H-alpha filter.  Result: broad
bands move by < 5e-4 mag at 300 km/s (< 5e-3 at 1000 km/s); the F187N
edge case moves by 1.5e-2 mag at 300 km/s; the 63 A filter by 0.15 mag.

Run as a script for the table; run under pytest for the assertions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from reference_broadening import CKMS, C_AA_S                      # noqa: E402


def fft_broaden_loguniform(spec, dv, sigma_kms):
    """Gaussian in ln lambda on a log-uniform grid (edge-continued FFT);
    the same algorithm as ceridwen.broadening.make_gaussian_fft, validated
    against direct quadrature in test_broadening.py."""
    n = spec.size
    m = int(2 ** np.ceil(np.log2(2 * n)))
    nr = (m - n) // 2
    padded = np.concatenate([spec, np.full(nr, spec[-1]), np.full(m - n - nr, spec[0])])
    nu = np.fft.rfftfreq(m)
    taper = np.exp(-2 * np.pi ** 2 * (sigma_kms / dv) ** 2 * nu ** 2)
    return np.fft.irfft(np.fft.rfft(padded) * taper, n=m)[:n]

def _default_filter_dir():
    d = os.path.join(os.path.dirname(__file__), "filters")
    if os.path.isdir(d):
        return d
    try:
        import sedpy_jax
        return os.path.join(os.path.dirname(sedpy_jax.__file__), "data", "filters")
    except ImportError:
        return d


FILTER_DIR = os.environ.get("SEDPY_FILTERS") or _default_filter_dir()


def load_filter(name):
    w, t = np.loadtxt(os.path.join(FILTER_DIR, name + ".par"), usecols=(0, 1)).T
    return w, t


def ab_maggies(wave, fnu, fw, ft):
    """AB maggies through a photon-counting filter, f_nu in erg/s/cm2/Hz."""
    t = np.interp(wave, fw, ft, left=0.0, right=0.0)
    num = np.trapezoid(t * fnu / wave, wave)
    den = np.trapezoid(t * 3631e-23 / wave, wave)
    return num / den if den > 0 else np.nan


def synthetic_sed(wave_rest, ew_halpha=1000.0):
    """f_nu (arbitrary units) with the sharpest features a galaxy spectrum has."""
    lam = wave_rest
    h, c, k = 6.626e-27, 2.998e10, 1.381e-16
    nu = c / (lam * 1e-8)
    bb = nu ** 3 / (np.exp(h * nu / (k * 6000.0)) - 1.0)
    f = bb / bb[np.argmin(np.abs(lam - 5500.))]
    f = np.where(lam < 912.0, 0.0, f)
    f = np.where(lam < 3646.0, 0.5 * f, f)
    for l0 in (3933.7, 3968.5, 4101.7, 4340.5, 4861.3, 6562.8):
        x = np.log(lam / l0) / (100.0 / CKMS)
        f *= 1.0 - 0.6 * np.exp(-0.5 * x * x)
    lines = {6562.8: ew_halpha, 4861.3: ew_halpha / 2.86, 5006.8: ew_halpha * 0.8,
             4958.9: ew_halpha * 0.27, 3727.1: ew_halpha * 0.5, 1215.7: ew_halpha * 2.0}
    for l0, ew in lines.items():
        cont = f[np.argmin(np.abs(lam - l0))]
        s = 30.0 / CKMS
        x = np.log(lam / l0) / s
        f += cont * ew / l0 * np.exp(-0.5 * x * x) / (np.sqrt(2 * np.pi) * s)
    return f


def broadened_photometry(z, filters, sigmas=(0.0, 300.0, 1000.0), ew=1000.0):
    dv = 5.0
    wave_rest = np.exp(np.arange(np.log(800.), np.log(60000.), dv / CKMS))
    f_rest = synthetic_sed(wave_rest, ew)
    wave_obs = wave_rest * (1 + z)
    out = {}
    for s in sigmas:
        f = f_rest if s == 0.0 else fft_broaden_loguniform(f_rest, dv, s)
        out[s] = {n: ab_maggies(wave_obs, f, fw, ft) for n, (fw, ft) in filters.items()}
    return out


def edge_redshift(fw, ft, line_rest=6562.8, side="red"):
    """Redshift that puts the line on the half-maximum edge of the filter."""
    half = 0.5 * ft.max()
    above = np.flatnonzero(ft > half)
    lam_edge = fw[above[-1]] if side == "red" else fw[above[0]]
    return lam_edge / line_rest - 1.0


def run(verbose=True):
    names = ["sdss_r0", "sdss_u0", "jwst_f444w", "jwst_f187n", "Halpha",
             "D4000_blue", "galex_FUV"]
    filters = {n: load_filter(n) for n in names}
    rows = []
    cases = [("z=0.00 (Halpha in sdss_r0, breaks in u)", 0.0),
             ("z=2.90 (Lyman break in galex_FUV)", 2.90)]
    fw, ft = filters["jwst_f187n"]
    cases.append(("Halpha on F187N red half-max edge", edge_redshift(fw, ft, side="red")))
    cases.append(("Halpha on F187N blue half-max edge", edge_redshift(fw, ft, side="blue")))
    fw, ft = filters["Halpha"]
    cases.append(("Halpha on Halpha.par red edge", edge_redshift(fw, ft, side="red")))
    for label, z in cases:
        phot = broadened_photometry(z, filters)
        for n in names:
            m0 = phot[0.0][n]
            if not m0 > 0:
                continue
            d300 = -2.5 * np.log10(phot[300.0][n] / m0)
            d1000 = -2.5 * np.log10(phot[1000.0][n] / m0)
            rows.append((label, z, n, d300, d1000))
    if verbose:
        print(f"{'case':45s} {'z':>6s} {'filter':12s} {'dmag(300)':>10s} {'dmag(1000)':>11s}")
        for label, z, n, d300, d1000 in rows:
            print(f"{label:45s} {z:6.3f} {n:12s} {d300:10.2e} {d1000:11.2e}")
    return rows


def test_photometry_insensitive_to_losvd():
    if not os.path.isdir(FILTER_DIR):
        import pytest
        pytest.skip("filter directory not found; set SEDPY_FILTERS")
    rows = run(verbose=False)
    broad = [r for r in rows if r[2] not in ("jwst_f187n", "Halpha", "D4000_blue")]
    narrow = [r for r in rows if r[2] in ("jwst_f187n", "Halpha", "D4000_blue")]
    assert max(abs(r[3]) for r in broad) < 5e-4
    assert max(abs(r[4]) for r in broad) < 5e-3
    assert max(abs(r[3]) for r in narrow) > 1e-2
    assert max(abs(r[4]) for r in narrow) > 1e-1


if __name__ == "__main__":
    run()
