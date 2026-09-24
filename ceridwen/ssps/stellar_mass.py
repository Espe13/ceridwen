"""Surviving stellar mass per SSP from FSPS, corrected where FSPS's isochrone is truncated.

FSPS ``stellar_mass`` (``ssp_gen.f90``) is ``sum_i wght_i * mact_i`` (+ remnants), with the
IMF weights of ``imf_weight.f90``: point ``i`` stands for the NUMBER of stars in
``[m_{i-1/2}, m_{i+1/2}]``, and the first point for all stars from ``imf_lower_bound``
(= ``imf_lower_limit``, 0.08 M_sun, except for Geneva) up to it.  At ages below ~2.5 Myr the
MIST pre-main-sequence isochrones start far above that (2.68 M_sun at 10^5 yr), so every star
from 0.08 to 2.68 M_sun is counted at 2.68 M_sun, and FSPS returns up to 4.6 M_sun surviving
per M_sun formed.  ``truncated_isochrone_correction`` replaces that first bin's contribution by
its IMF mass integral, times the point's ``mact / mini``, in the SSPs whose isochrone starts
above the lowest first mass of its metallicity row; every other SSP keeps FSPS's value bit for
bit.  BPASS reads its masses from ``bpass.mass`` (no isochrones) and is not corrected.
"""
from __future__ import annotations

import os

import numpy as np

# A surviving-mass table (M_sun per M_sun formed) above this is refused: more mass cannot
# survive than was formed.  The 1 % margin covers FSPS's midpoint IMF sum, whose formed mass
# sum_i wght_i * mini_i is itself not exactly 1 (MIST: 1.0017 at 10^6.5 yr, logzsol +0.25).
# mist_miles_chab: the corrected SSPs peak at 1.0013; FSPS's own values on complete isochrones
# reach 1.0036 (10^6.1 yr, logzsol -2.25; 42 SSPs above 1); FSPS's uncorrected values on the
# truncated ones reach 4.62 (measured).
STELLAR_MASS_MAX = 1.0 + 1e-2


def first_isochrone_points(sp, driver):
    """``[(log_age_yr, mini_1, mini_2, mact_1, log10 wght_1), ...]`` for every isochrone age of
    ``sp`` at its current zmet / afeindx, from FSPS's ``write_isoc`` table (the file
    ``StellarPopulation.isochrones()`` reads), parsed for its first two rows per age.  The
    scratch file ``$SPS_HOME/OUTPUTS/ceridwen_isoc_<pid>.cmd`` is removed afterwards.
    Self-contained (needs only ``os``) so it can run in another python-fsps environment."""
    import os as _os
    name = f"ceridwen_isoc_{_os.getpid()}"
    path = _os.path.join(_os.environ["SPS_HOME"], "OUTPUTS", name + ".cmd")
    driver.write_isoc(name)
    rows, last, prev = [], None, None
    try:
        with open(path) as fh:
            for line in fh:
                if line.lstrip().startswith("#"):
                    continue
                v = line.split(None, 10)          # age log(Z) mini mact logl logt logg phase
                age, mini, mact, lw = float(v[0]), float(v[2]), float(v[3]), float(v[9])
                if age != last:                   # composition log(weight) ...
                    if prev is not None:          # a one-point isochrone
                        rows.append((prev[0], prev[1], float("nan"), prev[2], prev[3]))
                    last, prev = age, (age, mini, mact, lw)
                elif prev is not None:
                    rows.append((prev[0], prev[1], mini, prev[2], prev[3]))
                    prev = None
        if prev is not None:
            rows.append((prev[0], prev[1], float("nan"), prev[2], prev[3]))
    finally:
        if _os.path.exists(path):
            _os.remove(path)
    return rows


def imf_number_density(imf):
    """dN/dm of FSPS ``IMF`` (``imf.f90``) for ``imf["imf_type"]`` 0 (Salpeter), 1 (Chabrier)
    or 2 (Kroupa, ``imf1..imf3``); ValueError for the others (no correction implemented)."""
    t = int(imf["imf_type"])
    if t == 0:
        return lambda m: m ** -2.35
    if t == 1:
        mc, s2, ind = 0.08, 0.69 * 0.69, 1.3                 # sps_vars.f90 chab_*
        hi = np.exp(-np.log10(mc) ** 2 / 2.0 / s2)
        return lambda m: (np.exp(-(np.log10(m) - np.log10(mc)) ** 2 / 2.0 / s2) if m < 1.0
                          else hi * m ** -ind) / m
    if t == 2:
        a1, a2, a3 = float(imf["imf1"]), float(imf["imf2"]), float(imf["imf3"])
        c = 0.5 ** (-a1 + a2)
        return lambda m: (m ** -a1 if m < 0.5 else c * m ** -a2 if m < 1.0 else c * m ** -a3)
    raise ValueError(f"imf_type {t}: the truncated-isochrone stellar-mass correction supports "
                     "imf_type 0, 1 and 2 only")


def truncated_isochrone_correction(mass, iso_first, imf, rel_floor=0.01):
    """``(corrected, report)`` for FSPS ``mass`` (..., n_age) and ``iso_first`` (..., n_age, 5)
    from :func:`first_isochrone_points`.  In each SSP whose first isochrone mass exceeds the
    lowest first mass of its row by more than ``rel_floor``, the first bin's FSPS contribution
    ``N_bin mact_1 / M_norm`` becomes ``M_bin (mact_1 / mini_1) / M_norm`` (N_bin, M_bin: IMF
    number and mass over ``[imf_lower_limit, (mini_1 + mini_2) / 2]``; M_norm: IMF mass over
    the limits).  ``report["max_abs_dlogw_vs_fsps"]`` compares the first-bin weight with the
    one FSPS wrote, a check that this IMF is FSPS's."""
    from scipy.integrate import quad
    dndm = imf_number_density(imf)
    lo, hi = float(imf["imf_lower_limit"]), float(imf["imf_upper_limit"])
    brk = [x for x in (0.5, 1.0) if lo < x < hi]

    def integ(f, a, b):
        pts = [x for x in brk if a < x < b]
        return quad(f, a, b, points=pts or None, epsabs=0.0, epsrel=1e-12, limit=400)[0]

    m_norm = integ(lambda m: m * dndm(m), lo, hi)
    out = np.array(mass, dtype=np.float64, copy=True)
    iso = np.asarray(iso_first, dtype=np.float64)
    if iso.shape[:-1] != out.shape or iso.shape[-1] != 5:
        raise ValueError(f"isochrone points {iso.shape} do not match the mass table {out.shape}")
    worst_w, cells = 0.0, []
    for idx in np.ndindex(*out.shape[:-1]):
        row = iso[idx]
        floor = np.nanmin(row[:, 1])
        for j, (_age, m1, m2, mact1, logw) in enumerate(row):
            if not (np.isfinite(m2) and lo < m1 < hi):
                continue
            top = m1 + 0.5 * (m2 - m1)
            n_bin = integ(dndm, lo, top)
            worst_w = max(worst_w, abs(np.log10(n_bin / m_norm) - logw))
            if m1 <= floor * (1.0 + rel_floor):
                continue
            m_bin = integ(lambda m: m * dndm(m), lo, top)
            new = out[idx + (j,)] + (m_bin * mact1 / m1 - n_bin * mact1) / m_norm
            cells.append((idx + (j,), float(out[idx + (j,)]), float(new), float(m1)))
            out[idx + (j,)] = new
    return out, {"n_corrected": len(cells), "cells": cells,
                 "max_abs_change": max((abs(c[2] - c[1]) for c in cells), default=0.0),
                 "max_abs_dlogw_vs_fsps": worst_w,
                 "imf": {k: float(v) for k, v in dict(imf).items()}}


def fsps_imf_params(sp) -> dict:
    """The IMF parameters of a python-fsps ``StellarPopulation`` the correction needs."""
    return {k: float(sp.params[k]) for k in ("imf_type", "imf_lower_limit", "imf_upper_limit",
                                             "imf1", "imf2", "imf3")}


def isochrone_points_or_none(sp):
    """:func:`first_isochrone_points` of ``sp`` at its current zmet / afeindx, or None for
    BPASS (no isochrones; FSPS reads its masses from ``bpass.mass``)."""
    lib = sp.libraries[0]
    if (lib.decode() if isinstance(lib, bytes) else str(lib)) == "bpss":
        return None
    if not os.environ.get("SPS_HOME"):
        raise RuntimeError("SPS_HOME is not set: FSPS's isochrone table cannot be written")
    from fsps._fsps import driver
    return first_isochrone_points(sp, driver)


def corrected_table(sp, mass, iso):
    """``(table, note)``: ``mass`` (FSPS ``stellar_mass`` stacked over zmet, and afeindx) with
    :func:`truncated_isochrone_correction` applied when ``iso`` (the matching stack of
    :func:`isochrone_points_or_none`) is not None; ``note`` is appended to the provenance."""
    mass = np.asarray(mass, dtype=np.float64)
    if iso is None:
        return mass, ""
    table, rep = truncated_isochrone_correction(mass, iso, fsps_imf_params(sp))
    if rep["max_abs_dlogw_vs_fsps"] > 2e-4:                 # FSPS writes log(weight) as F8.4
        raise RuntimeError(
            f"the first-bin IMF weight differs from FSPS's by {rep['max_abs_dlogw_vs_fsps']:.2e} "
            "dex: this IMF is not FSPS's, so the stellar-mass correction cannot be applied")
    return table, CORRECTION_NOTE.format(n=rep["n_corrected"])


CORRECTION_NOTE = ("; lowest IMF bin mass-weighted where the isochrone is truncated ({n} SSPs, "
                   "ages < ~2.5 Myr; FSPS counts that bin by number at the first isochrone mass)")
