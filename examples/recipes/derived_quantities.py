"""Prospector-equivalent derived quantities for ``PostProcess(derived=...)``.

What it does
------------
Functions of one posterior draw (a ``ceridwen.SpectrumSample``) that return

* ``mwa``            mass-weighted age  <t>_M = int t psi(t) dt / int psi(t) dt  [Gyr, lookback]
* ``t50``, ``t90``   lookback time [Gyr] at which 50 % / 90 % of the FORMED mass had formed
                     (so t90 < t50: the 90 % point is more recent)
* ``absmag``         rest-frame absolute AB magnitudes through any filter set
* ``colors``         differences of those magnitudes for given filter pairs

All SFH integrals use the package's own piecewise SFH (``csp.sfh_interp``: "step" per bin,
"linear" between nodes) through the SAME private helpers ``PostProcess`` uses for
``extras["sfh"]["mass_formed"]``, so the cumulative mass at the oldest node equals
``mass_formed`` by construction. No new integrator is written:

* ``ceridwen/postprocess.py:116`` ``_per_bin_and_nodes`` (per-bin / per-node SFH)
* ``ceridwen/postprocess.py:132`` ``_mean_sfr_window``   (mass formed in lookback [0, w] = mean SFR x w)
* ``ceridwen/postprocess.py:147`` ``_formed_mass``       (total formed mass; ``PostProcess._derive`` at :467)

The cumulative formed mass M(<t) is piecewise linear (step) or piecewise quadratic (linear) in
t, so ``mwa = (1/M) int_0^T (M - M(<t)) dt`` is integrated exactly with Simpson's rule per SFH
bin on values of ``_mean_sfr_window``, and ``t_q`` is found by root-bracketing M(<t) inside the
bin that contains it (exact up to ``xtol``). As in ``PostProcess``, these are FORMED-mass
quantities (no return fraction / surviving mass).

Absolute magnitudes: the rest-frame L_nu ``s.full`` [L_sun/Hz] x 10**logmass (dust-attenuated,
lines included, no IGM, no (1+z): ``postprocess.py:83-85``, ``:350-367``) is placed at 10 pc,
``f_nu = L_nu L_sun / (4 pi (10 pc)^2)``, with the package's own constants (``_LSUN_ERG_S``,
``_PC_CM``, ``postprocess.py:68, :71``, the ones behind ``MUV``), and projected through the
package's vendored filters with one of two package paths:

* ``method="hires"`` (default): ``Filter.ab_mag`` (``observation/filters.py:341``, trapezoid on the
  source grid, ``obj_counts_hires`` :285), a port of sedpy's ``Filter.obj_counts_hires``. It equals
  sedpy -- and so Prospector's ``absolute_rest_maggies`` -- to 0.0 mag on the BPASS grid.
* ``method="photometry"``: ``Photometry.get_maggies`` (``observation/photometry.py:77``), the
  ``FilterSet`` gridded projection (shared log-lambda grid, dlnlam <= 1e-3, ``filters.py:444-500``)
  that ``Photometry.setup_for_model`` also uses for the fit predictions (``photometry.py:106-114``).
  It differs from sedpy by up to ~1.5e-3 mag (measured: sdss_u0 +1.5e-3, twomass_Ks -1.1e-3 on a
  BPASS SSP). Use it to match CERIDWEN's own predicted photometry rather than Prospector.

Prospector equivalents (bd-j/prospector @ a78d153)
-------------------------------------------------
* ``prospect/models/sedmodel.py:851-883`` ``SpecModel.absolute_rest_maggies``
* ``prospect/plotting/sfh.py:255-264``     ``nonpar_mwa`` (step SFH mass-weighted age)
* ``prospect/plotting/sfh.py:267-278``     ``sfh_to_cmf`` (cumulative mass fraction -> t50/t90)

Known difference: Prospector's ``lsun = 3.846e33`` erg/s (``prospect/sources/constants.py:15``)
while CERIDWEN (and FSPS) use ``3.839e33`` (``postprocess.py:68``; FSPS ``src/sps_vars.f90:422``). Feeding the same L_sun/Hz
array to both therefore differs by 2.5 log10(3.846/3.839) = 1.98 mmag; the check compares the
same PHYSICAL spectrum (erg/s/Hz).

Literature: mass-weighted age and formation-time quantiles as posterior summaries of
non-parametric SFHs, Leja et al. 2019, ApJ 876, 3 (Prospector-alpha); Pacifici et al. 2016,
ApJ 832, 79 (t50 / "half-mass" time). AB magnitudes: Oke & Gunn 1983, ApJ 266, 713.

Usage
-----
    import sys; sys.path.insert(0, "examples/recipes")      # no package __init__ here
    from derived_quantities import make_derived
    pp = PostProcess(model, result,
                     derived=make_derived(model.csp, filters=["sdss_g0", "sdss_r0"],
                                          colors=[("sdss_g0", "sdss_r0")]))
    out = pp.run()
    out["derived"]["t50"]          # (N,) Gyr
    out["derived"]["absmag"]       # (N, n_filters)

Public / package API relied on
------------------------------
* ``ceridwen.SpectrumSample`` (``postprocess.py:81-113``; fields ``sfr``, ``lookback_gyr``,
  ``full``, ``wave_rest``); ``PostProcess(derived=...)`` calls ``fn(s)`` per draw (``:541-548``).
* ``csp.sfh_interp`` ("step" | "linear"), read here because ``SpectrumSample`` does not carry it.
* ``ceridwen.observation.filters.load_filters`` / ``Filter.ab_mag`` (``observation/filters.py:680``, ``:341``).
* ``ceridwen.observation.Photometry`` / ``.get_maggies`` (``observation/photometry.py:23``, ``:77``).
* Private helpers ``_per_bin_and_nodes``, ``_mean_sfr_window``, ``_formed_mass`` and constants of
  ``ceridwen/postprocess.py`` (not public: the package version of this recipe should expose them).

``PostProcess`` runs ``derived`` callables in numpy on host arrays (``postprocess.py:547``), so
these are plain float64 numpy functions, not traced; nothing here enters a jitted kernel.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from ceridwen.postprocess import (_LSUN_ERG_S, _PC_CM, _formed_mass,
                                  _mean_sfr_window, _per_bin_and_nodes)
from ceridwen.observation import Photometry
from ceridwen.observation.filters import load_filters

_YR_PER_GYR = 1e9


def _sfh(s, interp: str):
    """(T_yr (n_time,), per-bin psi, per-node psi) of a SpectrumSample, as PostProcess builds them."""
    if interp not in ("step", "linear"):
        raise ValueError(f"interp must be 'step' or 'linear', got {interp!r}")
    T = np.asarray(s.lookback_gyr, dtype=float) * _YR_PER_GYR
    bar, nodes = _per_bin_and_nodes(s.sfr, T.size)
    return T, bar, nodes


def _mass_before(T, bar, nodes, t_yr, interp):
    """Formed mass in lookback [0, t_yr] (the package's own piecewise SFH)."""
    if t_yr <= 0.0:
        return 0.0
    return _mean_sfr_window(T, bar, nodes, float(t_yr), interp) * float(t_yr)


def cumulative_mass(s, interp: str) -> np.ndarray:
    """M(<T_k): formed mass [M_sun] between today and each SFH node, shape (n_time,).
    The last element equals ``PostProcess`` ``mass_formed``."""
    T, bar, nodes = _sfh(s, interp)
    return np.array([_mass_before(T, bar, nodes, t, interp) for t in T])


def mass_formed(s, interp: str) -> float:
    """Total formed mass [M_sun], exactly as ``PostProcess._derive`` (postprocess.py:467)."""
    T, bar, nodes = _sfh(s, interp)
    return _formed_mass(T, bar, nodes, interp)


def mass_weighted_age(s, interp: str) -> float:
    """<t>_M [Gyr] = (1/M) int_0^T_old (M - M(<t)) dt; Simpson per bin (exact: M(<t) is
    piecewise linear or quadratic on the package's SFH)."""
    T, bar, nodes = _sfh(s, interp)
    M = _formed_mass(T, bar, nodes, interp)
    if not M > 0:
        return float("nan")
    edges = np.concatenate([[0.0], T]) if T[0] > 0.0 else T
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        fa, fm, fb = (M - _mass_before(T, bar, nodes, t, interp) for t in (a, 0.5 * (a + b), b))
        total += (b - a) / 6.0 * (fa + 4.0 * fm + fb)
    return total / M / _YR_PER_GYR


def formation_time(s, interp: str, frac: float, xtol_yr: float = 1e-6) -> float:
    """Lookback time [Gyr] by which ``frac`` of the formed mass had formed, i.e. the t with
    M(>t) = frac M  <=>  M(<t) = (1 - frac) M.  Prospector's t50 / t90 (sfh_to_cmf)."""
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    T, bar, nodes = _sfh(s, interp)
    M = _formed_mass(T, bar, nodes, interp)
    if not M > 0:
        return float("nan")
    target = (1.0 - frac) * M
    cum = np.array([_mass_before(T, bar, nodes, t, interp) for t in T])
    k = int(np.searchsorted(cum, target, side="left"))        # first node with M(<T_k) >= target
    if k == 0:
        return float(T[0] / _YR_PER_GYR)
    if cum[k] == target:
        return float(T[k] / _YR_PER_GYR)
    lo, hi = T[k - 1], T[k]
    t = brentq(lambda x: _mass_before(T, bar, nodes, x, interp) - target, lo, hi,
               xtol=xtol_yr, rtol=4 * np.finfo(float).eps, maxiter=200)
    return float(t / _YR_PER_GYR)


class AbsMagFilters:
    """Filters for rest-frame absolute magnitudes; ``method`` "hires" (sedpy-equivalent,
    default) or "photometry" (the FilterSet path of CERIDWEN's Photometry predictions)."""

    def __init__(self, filters: Sequence[str], method: str = "hires"):
        if method not in ("hires", "photometry"):
            raise ValueError(f"method must be 'hires' or 'photometry', got {method!r}")
        self.names, self.method = list(filters), method
        if method == "hires":
            self._filters = load_filters(self.names)
        else:
            self._phot = Photometry(filters=self.names, name="_absmag")

    def maggies(self, wave_aa, fnu_cgs):
        wave = np.asarray(wave_aa, dtype=float)
        fnu = np.asarray(fnu_cgs, dtype=float)
        if self.method == "photometry":
            return np.asarray(self._phot.get_maggies(wave, fnu), dtype=float).reshape(-1)
        flam = fnu * 2.998e18 / wave ** 2           # the c of filters.py:26 / sedpy
        return np.array([10.0 ** (-0.4 * float(f.ab_mag(wave, flam))) for f in self._filters])


def absolute_rest_maggies(s, filters: AbsMagFilters, spectrum: str = "full") -> np.ndarray:
    """Rest-frame absolute maggies 10**(-0.4 M_AB), shape (n_filters,): the rest-frame model
    L_nu placed at 10 pc and projected through ``filters``."""
    lnu = np.asarray(getattr(s, spectrum), dtype=float)                  # L_sun/Hz x 10**logmass
    fnu_10pc = lnu * _LSUN_ERG_S / (4.0 * np.pi * (10.0 * _PC_CM) ** 2)   # erg/s/cm^2/Hz
    return filters.maggies(s.wave_rest, fnu_10pc)


def absolute_magnitudes(s, filters: AbsMagFilters, spectrum: str = "full") -> np.ndarray:
    """Rest-frame absolute AB magnitudes, shape (n_filters,)."""
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(absolute_rest_maggies(s, filters, spectrum))


def make_derived(csp, filters: Optional[Sequence[str]] = None,
                 colors: Optional[Sequence[tuple]] = None,
                 quantiles: Sequence[float] = (0.5, 0.9), spectrum: str = "full",
                 method: str = "hires") -> dict:
    """``derived=`` dict for ``PostProcess``: mwa, t<q>, mass_formed_check and, with
    ``filters``, absmag (n_filters,) and colors (n_pairs,)."""
    interp = str(csp.sfh_interp)
    out = {"mwa": lambda s: mass_weighted_age(s, interp),
           "mass_formed_check": lambda s: cumulative_mass(s, interp)[-1]}
    for q in quantiles:
        out[f"t{100 * q:g}"] = (lambda qq: (lambda s: formation_time(s, interp, qq)))(float(q))
    if colors and not filters:
        raise ValueError("colors needs filters")
    if filters:
        names = list(filters)
        phot = AbsMagFilters(names, method=method)
        out["absmag"] = lambda s: absolute_magnitudes(s, phot, spectrum)
        if colors:
            pairs = [(names.index(a), names.index(b)) for a, b in colors]

            def _colors(s):
                m = absolute_magnitudes(s, phot, spectrum)
                return np.array([m[i] - m[j] for i, j in pairs])
            out["colors"] = _colors
    return out
