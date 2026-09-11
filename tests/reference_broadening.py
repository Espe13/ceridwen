"""Brute-force NumPy reference for ceridwen.broadening.

Everything here is deliberately slow and obvious (direct O(N*M) quadrature
in ln lambda, no FFT, no grids) so that the fast implementation can be
checked against it.  Used only by the tests.
"""
import numpy as np

CKMS = 2.99792458e5
C_AA_S = 2.99792458e18


def direct_broaden(wave_in, spec_in, wave_out, sigma_kms):
    """Convolve f_nu(ln lambda) with a Gaussian of dispersion sigma_kms
    (scalar or per-output-pixel array) by direct trapezoid quadrature.

    spec_in must be sampled finely enough (>= 4 samples per sigma) for
    the trapezoid rule to be accurate.  The input is continued as a
    constant beyond its ends (same edge condition as the fast code).
    """
    lnw_in = np.log(np.asarray(wave_in, dtype=np.float64))
    lnw_out = np.log(np.asarray(wave_out, dtype=np.float64))
    s = np.broadcast_to(np.asarray(sigma_kms, dtype=np.float64) / CKMS, lnw_out.shape)
    out = np.empty_like(lnw_out)
    pad = 6.0 * s.max()
    dln = np.min(np.diff(lnw_in))
    n_pad = int(np.ceil(pad / dln)) + 1
    left = lnw_in[0] - dln * np.arange(n_pad, 0, -1)
    right = lnw_in[-1] + dln * np.arange(1, n_pad + 1)
    x = np.concatenate([left, lnw_in, right])
    y = np.concatenate([np.full(n_pad, spec_in[0]), spec_in, np.full(n_pad, spec_in[-1])])
    for i, (c, si) in enumerate(zip(lnw_out, s)):
        if si <= 0.0:
            out[i] = np.interp(c, x, y)
            continue
        g = np.exp(-0.5 * ((x - c) / si) ** 2)
        norm = np.trapezoid(g, x)
        out[i] = np.trapezoid(g * y, x) / norm
    return out


def analytic_line_fnu(wave, wave0, flux, sigma_kms):
    """f_nu of one Gaussian line (in ln lambda) of integrated flux ``flux``."""
    s = sigma_kms / CKMS
    x = (np.log(wave) - np.log(wave0)) / s
    phi = np.exp(-0.5 * x * x) / (np.sqrt(2 * np.pi) * s)
    return flux * phi * wave / C_AA_S


def integrate_fnu_dnu(wave, fnu):
    """Integral of f_nu over nu (trapezoid on the wavelength grid)."""
    nu = C_AA_S / np.asarray(wave, dtype=np.float64)
    return -np.trapezoid(fnu, nu)


def gaussian_feature_lnlambda(wave, wave0, sigma_kms, amplitude):
    """Unit-peak-scaled Gaussian feature in ln lambda (a synthetic 'absorption
    line' with a library width) on a flat continuum of 1."""
    s = sigma_kms / CKMS
    x = (np.log(wave) - np.log(wave0)) / s
    return 1.0 + amplitude * np.exp(-0.5 * x * x)
