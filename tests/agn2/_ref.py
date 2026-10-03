"""Independent references of the AGN2 validation suite (session VAL). Nothing here imports a
component module (skirtor / polar / nlr / blr): the point is independence from them.

* ``missing(what, test)``: the loud skip of the suite ("MISSING <what>: <test> did NOT run").
* ``feltre_raw(dir)``: the Feltre+16 NLR grid parsed from the SOURCE text files
  (``nlr_nebular_Z*.txt``), not from the converter's HDF5.
* ``multilinear(axes, table, point)``: plain NumPy multilinear interpolation on a rectilinear grid
  (clamped at the edges), written without JAX and without any code of the package.
* ``nu_integral(wave, fnu)``: integral over frequency of F_nu on a wavelength grid (trapezoid).
* ``gauss_moments(wave, fnu, lam0)``: (flux, mean ln lam offset, rms in ln lam) of a painted line.
"""
from __future__ import annotations

import os
import pathlib
import re

import numpy as np
import pytest

C_AA_S = 2.99792458e18
CKMS = 2.99792458e5
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))      # 2.3548...

AGN_DATA = pathlib.Path(os.environ.get("CERIDWEN_AGN_DATA",
                                       pathlib.Path.home() / ".ceridwen" / "agn"))
REF_DIR = pathlib.Path(__file__).resolve().parents[1] / "reference" / "agn2"


def missing(what, test):
    pytest.skip(f"MISSING {what}: {test} did NOT run")


def need_file(path, test):
    p = pathlib.Path(path)
    if not p.exists():
        missing(p, test)
    return p


# ------------------------------------------------------------------ Feltre+16 source files
FELTRE_Z_FROM_NAME = re.compile(r"nlr_nebular_Z(\d+)\.txt$")


def _z_of(name):
    """'Z0001' -> 0.0001, 'Z014' -> 0.014, 'Z020' -> 0.020: the digits are the decimals."""
    return float("0." + FELTRE_Z_FROM_NAME.search(name).group(1))


def feltre_raw(src_dir):
    """``{"Z_abs": (nZ,) absolute Z, "logu": , "xid": , "nh": , "alpha": , "lines": tuple of source names,
    "L": (n_lines, nZ, nU, nX, nN, nA) erg/s}`` from the source text files, axes ascending."""
    src = pathlib.Path(src_dir)
    files = sorted(src.glob("nlr_nebular_Z*.txt"), key=lambda p: _z_of(p.name))
    if not files:
        raise FileNotFoundError(f"no nlr_nebular_Z*.txt in {src}")
    blocks, header = [], None
    for f in files:
        with open(f) as fh:
            head = fh.readline()
        cols = head.lstrip("#").split()
        names = cols[4:]
        if header is None:
            header = names
        elif names != header:
            raise ValueError(f"{f}: columns differ from the first file")
        blocks.append(np.loadtxt(f, comments="#"))
    first = blocks[0]
    ax = [np.unique(first[:, i]) for i in range(4)]       # logu, xid, nh, alpha
    shape = tuple(a.size for a in ax)
    L = np.full((len(header), len(files)) + shape, np.nan)
    for iz, b in enumerate(blocks):
        idx = [np.searchsorted(ax[i], b[:, i]) for i in range(4)]
        for i in range(4):
            if not np.allclose(ax[i][idx[i]], b[:, i]):
                raise ValueError("source rows are not on a rectilinear grid")
        L[(slice(None), iz) + tuple(idx)] = b[:, 4:].T
    if np.isnan(L).any():
        raise ValueError("the source grid has holes")
    return {"Z_abs": np.array([_z_of(f.name) for f in files]), "logu": ax[0], "xid": ax[1],
            "nh": ax[2], "alpha": ax[3], "lines": tuple(header), "L": L}


# ------------------------------------------------------------------ interpolation
def multilinear(axes, table, point):
    """Multilinear interpolation of ``table`` (..., n0, n1, ..., nk-1) on the ascending
    ``axes`` (k arrays) at ``point`` (k floats), clamped to the grid; leading dims carried."""
    k = len(axes)
    lead = table.ndim - k
    out = np.zeros(table.shape[:lead])
    lo, fr = [], []
    for a, x in zip(axes, point):
        a = np.asarray(a, dtype=np.float64)
        x = float(np.clip(x, a[0], a[-1]))
        i = int(np.clip(np.searchsorted(a, x, side="right") - 1, 0, a.size - 2))
        lo.append(i)
        fr.append((x - a[i]) / (a[i + 1] - a[i]))
    for corner in range(2 ** k):
        w = 1.0
        idx = []
        for d in range(k):
            bit = (corner >> d) & 1
            w *= fr[d] if bit else 1.0 - fr[d]
            idx.append(lo[d] + bit)
        if w != 0.0:
            out = out + w * table[(Ellipsis,) + tuple(idx)]
    return out


# ------------------------------------------------------------------ integrals, moments
def nu_integral(wave, fnu):
    """Integral of F_nu over frequency [same units x Hz] on an ascending wavelength grid."""
    wave = np.asarray(wave, dtype=np.float64)
    return -np.trapezoid(np.asarray(fnu, dtype=np.float64), C_AA_S / wave)


def gauss_moments(wave, fnu, lam0):
    """(integral over nu, mean of ln(lam/lam0), rms in ln lam) of a painted F_nu line."""
    wave = np.asarray(wave, dtype=np.float64)
    fnu = np.asarray(fnu, dtype=np.float64)
    flux = nu_integral(wave, fnu)
    x = np.log(wave / lam0)
    w = fnu / wave            # f_nu = F phi(ln lam) lam / c  ->  phi ~ f_nu / lam
    mean = np.sum(w * x) / np.sum(w)
    rms = np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return flux, mean, rms


def fwhm_kms(wave, fnu, lam0):
    """FWHM [km/s] of a single painted line measured directly from the profile in ln lam
    (half-maximum crossings found by linear interpolation), independent of any moment."""
    wave = np.asarray(wave, dtype=np.float64)
    phi = np.asarray(fnu, dtype=np.float64) / wave
    x = np.log(wave / lam0)
    j = int(np.argmax(phi))
    half = 0.5 * phi[j]
    left = np.nonzero(phi[:j] < half)[0]
    right = np.nonzero(phi[j:] < half)[0]
    if not left.size or not right.size:
        raise ValueError("profile does not fall below half maximum on both sides")
    i0 = left[-1]
    xl = np.interp(half, [phi[i0], phi[i0 + 1]], [x[i0], x[i0 + 1]])
    i1 = j + right[0]
    xr = np.interp(half, [phi[i1], phi[i1 - 1]], [x[i1], x[i1 - 1]])
    return (xr - xl) * CKMS
