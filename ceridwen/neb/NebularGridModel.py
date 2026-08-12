"""
NebularGridModel.py
===================
Physically strict JAX/JIT implementation of the FSPS-style nebular
emission model.

This module's :class:`NebularModel` interpolates each CLOUDY cube
(``ZAU_ND_<isoc>.cont`` and ``ZAU_ND_<isoc>.lines``) against its **own**
``(logZ, age, logU)`` grid, rather than against the line-cube's grid
which FSPS's run-time uses for both.  As of FSPS commit
``0db2d3e`` (2023-09-21) those two files are no longer tabulated on the
same physical grid — see the project docs / supervisor note for the
detailed trace — and using the FSPS convention silently evaluates the
continuum cube at axes that do not describe its data.  Here every cube
is interpolated against the labels it was generated with.

If you want to *match FSPS exactly* (e.g. for SED-fitting comparisons
against upstream FSPS posteriors), import
:class:`ceridwen.neb.NebularGridModel_fsps_match.NebularModelFSPSMatch`
instead.  It is bit-identical to FSPS's behaviour and agrees with FSPS
to better than 0.5% on the SFH-integrated nebular spectrum; this
default class will disagree with FSPS by up to ~50% in the FUV
continuum at young ages because *FSPS itself* is using the wrong axes
there.

Physical model
--------------
For each (single-stellar) age :math:`t` below the maximum age in the
CLOUDY grid, FSPS does::

    Q                = (L_sun / h)  ×  ∫ L_nu / λ  dλ        (0 < λ < 912 Å)
    L_neb_cont(λ)    = 10**logcont(logZ, logage, logU; λ)  ×  Q
    L_neb_line(λ)    = Σ_l  10**logline_l(logZ, logage, logU)  ×  Q  ×  g_l(λ)

with line profile

    dlam   = max(λ_l · σ_smooth / c · 1e13, 2 · Δλ_pix)   (smooth_velocity=True)
    dlam   = max(σ_smooth, 2 · Δλ_pix)                     (smooth_velocity=False)
    g_l(λ) = (1 / √(2π) / dlam) · exp(-(λ-λ_l)²/(2 dlam²)) · λ_l² / c.

The two log10 cubes are interpolated linearly in (logZ, logage, logU)
and exponentiated, identical to FSPS — only the axis arrays differ
between this class and the FSPS-matching one.

Defaults (match FSPS ``sps_vars.f90``)
--------------------------------------
gas_logz             : 0.0    (log10 Z/Zsun)
gas_logu             : -2.0   (log10 U)
smooth_velocity      : True
sigma_smooth         : 0.0
cloudy_dust          : False
isoc_type            : 'mist'

Public interface
----------------
The class signature accepted by :class:`ceridwen.csp.csp.CSPBasis` is
preserved.  Pass ``ssp_flux=`` to have the model compute a
self-consistent ``log_qq`` table from the SSP fluxes (FSPS-equivalent
formula); ``ssp_ages_lgyr=`` to flag the young SSPs whose ages lie
inside both CLOUDY grids.
"""

from pathlib import Path

import warnings
import os
import numpy as np
import jax
import jax.numpy as jnp


# ── Physical constants in CGS (match FSPS sps_vars.f90) ──────────────────────
CLIGHT_AA_S    = 2.9979e18     # speed of light, Angstrom / s        (clight)
HPLANK_ERG_S   = 6.6261e-27    # Planck constant, erg * s            (hplank)
LSUN_ERG_S     = 3.839e33      # solar luminosity, erg / s           (lsun)
LYMAN_LIMIT_AA = 912.0         # Lyman limit, Angstrom
SQRT_2PI       = float(np.sqrt(2.0 * np.pi))
TINY           = 1.0e-95       # FSPS floor for log10


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (also imported by NebularGridModelSVD).
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_age_axis_to_log10yr(age, src_file):
    """Normalise a CLOUDY-cube age axis to log10(age/yr), whatever unit the
    file was tabulated in, and fail loudly on anything unrecognisable.

    Conventions seen in FSPS ``ZAU_*`` files across libraries:
      * log10(yr)  -- values ~ 5..9        (e.g. 6.0 .. 7.301)
      * linear yr  -- values ~ 1e6..1e8
      * linear Myr -- values ~ 0.5..100

    The old heuristic (``if age.max() > 30: log10``) silently misreads a
    linear-Myr axis (max <= 30) as log10(yr), which would break BOTH the
    young-SSP mask and the age interpolation for every SSP.  This version
    disambiguates on the value range and validates the result, so a nebular
    library with a different age tabulation either works or raises
    immediately.
    """
    a = np.asarray(age, dtype=np.float64)
    if a.min() >= 4.5 and a.max() <= 10.5:            # already log10(yr)
        out = a
    elif a.max() > 1.0e4:                              # linear years
        out = np.log10(a)
    elif a.max() <= 1.0e3:                             # linear Myr
        out = np.log10(a * 1.0e6)
    else:
        raise ValueError(
            f"Unrecognisable nebular age axis in {src_file}: range "
            f"[{a.min():g}, {a.max():g}]. Expected log10(yr) (~5-10), "
            "linear yr (>1e4) or linear Myr (<=1e3).")
    if not (4.5 <= out.min() and out.max() <= 10.5):
        raise ValueError(
            f"Nebular age axis in {src_file} normalised to log10(yr) is out "
            f"of range: [{out.min():.2f}, {out.max():.2f}].")
    return out


def _resolve_line_wavelengths(line_file, line_pos, payload):
    """Return the wavelength row for a ZAU ``.lines`` cube, repairing the
    stale-wavelength-row bug shipped by upstream FSPS for the dusty grids.

    As of FSPS commit ``cbcd0ee`` (2026-08-03; first observed with the
    C3K/[alpha/Fe] data release), the ``ZAU_WD_*.lines`` files were
    regenerated with the new 166-line list in their 770 flux blocks, but
    their header (``#128 cols``) and wavelength row were left at the old
    128-line vintage.  The flux blocks ARE column-aligned with the matching
    ``ZAU_ND_*.lines`` file (verified 2026-08-05: per-column WD/ND flux
    ratios are smooth attenuation factors, e.g. H-alpha 0.19-1.0, and the
    per-block ``(logZ, age, logU)`` metadata rows are identical), so the
    correct wavelengths are the dust-free file's.  Note Fortran FSPS itself
    misreads these files: its list-directed ``READ nebem_line_pos`` consumes
    the first metadata row and part of the first flux block to fill 166
    values -- so the file cannot be "matched", only repaired or refused.

    Parameters
    ----------
    line_file : path
        The ``.lines`` file being read (for messages and sibling lookup).
    line_pos : (n,) ndarray
        Wavelength row as read from line 2 of the file.
    payload : list of str
        The remaining lines of the file (alternating meta / flux rows).

    Returns
    -------
    (nflux,) ndarray of wavelengths consistent with the flux-row width.

    Raises
    ------
    ValueError if the wavelength row is inconsistent with the flux rows and
    no trustworthy repair source is available.
    """
    nflux = len(payload[1].split())
    if line_pos.size == nflux:
        return line_pos                              # healthy file, any vintage

    line_file = Path(line_file)
    candidates = []
    if '_WD_' in line_file.name:
        candidates.append(('dust-free sibling cube',
                           line_file.with_name(
                               line_file.name.replace('_WD_', '_ND_'))))
    candidates.append(('emlines_info.dat',
                       line_file.parent.parent / 'data' / 'emlines_info.dat'))

    for label, path in candidates:
        try:
            if path.suffix == '.dat':
                waves = np.asarray([float(r.split(',')[0])
                                    for r in open(path) if ',' in r])
            else:
                with open(path) as f:
                    f.readline()                                  # header
                    waves = np.asarray(f.readline().split(),
                                       dtype=np.float64)
                    f.readline()                                  # first meta row
                    sib_nflux = len(f.readline().split())
                if sib_nflux != waves.size:
                    continue                    # sibling is broken too
        except OSError:
            continue
        if waves.size == nflux:
            warnings.warn(
                f"{line_file.name}: wavelength row lists {line_pos.size} "
                f"lines but the flux blocks contain {nflux} -- this is the "
                "known upstream FSPS bug where ZAU_WD_*.lines flux blocks "
                "were regenerated for the new line list without updating "
                f"the header/wavelength row. Using the {nflux} wavelengths "
                f"from {label} ({path.name}) instead, which the flux "
                "columns are verified to align with.", stacklevel=3)
            return waves

    raise ValueError(
        f"{line_file}: wavelength row lists {line_pos.size} lines but the "
        f"flux blocks contain {nflux}, and no consistent repair source "
        "(dust-free sibling cube or emlines_info.dat with a matching line "
        "count) was found in this $SPS_HOME. The nebular data files are "
        "internally inconsistent -- re-download or regenerate them.")


def _locate(x, grid):
    """Index of the cell with ``grid[i] <= x < grid[i+1]``, clipped to ``[0, n-2]``."""
    return jnp.clip(jnp.searchsorted(grid, x) - 1, 0, grid.size - 2)


def _frac(x, grid, i):
    """Fractional position of ``x`` in ``[grid[i], grid[i+1]]``, clipped to [0,1]."""
    return jnp.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)


def _trilinear(cube, z1, dz, a1, da, u1, du):
    """
    Trilinear interpolation on a cube of shape ``(..., nz, nage, nu)``.

    Leading axes (typically the spectral axis) are broadcast.  Implemented
    as eight scalar-weighted slice gathers so the XLA graph is just
    gather + axpy ops — no scatter / dynamic shapes; JIT-, vmap-, and
    grad-friendly.
    """
    w000 = (1.0 - dz) * (1.0 - da) * (1.0 - du)
    w001 = (1.0 - dz) * (1.0 - da) *       du
    w010 = (1.0 - dz) *       da   * (1.0 - du)
    w011 = (1.0 - dz) *       da   *       du
    w100 =       dz   * (1.0 - da) * (1.0 - du)
    w101 =       dz   * (1.0 - da) *       du
    w110 =       dz   *       da   * (1.0 - du)
    w111 =       dz   *       da   *       du
    return (w000 * cube[..., z1,     a1,     u1    ]
          + w001 * cube[..., z1,     a1,     u1 + 1]
          + w010 * cube[..., z1,     a1 + 1, u1    ]
          + w011 * cube[..., z1,     a1 + 1, u1 + 1]
          + w100 * cube[..., z1 + 1, a1,     u1    ]
          + w101 * cube[..., z1 + 1, a1,     u1 + 1]
          + w110 * cube[..., z1 + 1, a1 + 1, u1    ]
          + w111 * cube[..., z1 + 1, a1 + 1, u1 + 1])


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class NebularModel:
    """
    CLOUDY-grid nebular emission model — *physically strict* variant.

    The continuum cube and the line cube are each interpolated against
    their own ``(logZ, age, logU)`` axes, ensuring that the returned
    nebular spectrum corresponds to the parameters CLOUDY was actually
    run at.  This differs from FSPS's run-time convention; see the
    module docstring and
    :class:`ceridwen.neb.NebularGridModel_fsps_match.NebularModelFSPSMatch`
    for the FSPS-matching variant.

    Parameters
    ----------
    cloudy_dust : bool
        ``False`` → load the dust-free grids ``ZAU_ND_<isoc>.{cont,lines}``.
        ``True`` → load the dust-attenuated grids ``ZAU_WD_<isoc>.…``.
    sps_home : str | Path
        FSPS root directory; expects ``<sps_home>/nebular/`` to contain
        the ZAU grids.
    csp_lambda : array, shape (nspec,)
        Wavelength grid (Å) the nebular spectrum will be projected onto.
    ssp_flux : array, shape (n_z, n_age, n_wave) or None
        Optional.  If provided, ``log_qq`` is computed internally from
        the SSP fluxes via the FSPS-equivalent formula.
    ssp_ages_lgyr : array, shape (n_age,) or None
        Optional.  ``log10(age / yr)`` of every SSP in ``ssp_flux``.
        Used to flag the SSPs whose age lies inside **both** CLOUDY
        grids (intersection of the two age ranges).
    isoc_type : {'mist', 'pdva', 'prsc', 'bpss'}
        Isochrone tag identifying the ZAU file suffix.
    nebnz, nebnage, nebnip : int
        Dimensions of each CLOUDY grid (default 11, 10, 7 for MIST).
    smooth_velocity : bool
        ``True`` → ``sigma_smooth`` is in km/s; ``False`` → Å.
    sigma_smooth : float
        Line-broadening σ.  Default 100.0 (km/s, since ``smooth_velocity``
        defaults to ``True``); matches the FSPS ``nebular_smooth_init``.
    nebular_smooth_init : float or None
        Backwards-compatible alias for ``sigma_smooth``.

    Attributes
    ----------
    nebem_cont : (nspec, nebnz, nebnage, nebnip)
        ``log10`` of nebular continuum (L_ν / Q) on ``csp_lambda``.
    nebem_line : (nemline, nebnz, nebnage, nebnip)
        ``log10`` of per-line luminosities (Lsun / Q).
    nebem_cont_logz, nebem_cont_age, nebem_cont_logu : 1-D
        The ``.cont`` cube's *own* grid axes (ascending).
    nebem_line_logz, nebem_line_age, nebem_line_logu : 1-D
        The ``.lines`` cube's *own* grid axes (ascending).
    nebem_logz, nebem_age, nebem_logu : 1-D
        Legacy aliases — point at the **line** axes for compatibility
        with code (e.g. ``NebularGridModelSVD``) that expects the FSPS
        canonical axis names.
    nebem_line_pos : (nemline,)
        Rest-frame emission-line centroids in Å.
    gaussnebarr : (nspec, nemline)
        Pre-computed Gaussian profiles including the FSPS ``λ²/c`` factor.
    log_qq : (n_z, n_age) or None
        Self-consistent ``log10(Q)`` derived from ``ssp_flux``.
    young_mask, young_idx :
        Boolean mask / index array over ``ssp_ages_lgyr`` selecting ages
        inside the intersection of both cubes' age ranges.
    """

    # ---- construction --------------------------------------------------
    def __init__(self,
                 cloudy_dust,
                 sps_home,
                 csp_lambda,
                 ssp_flux=None,
                 ssp_ages_lgyr=None,
                 isoc_type='mist',
                 nebnz=11, nebnage=10, nebnip=7,
                 smooth_velocity=True,
                 sigma_smooth=100.0,
                 res_floor_factor=2.0,
                 nebular_smooth_init=None):
        # Defaults match FSPS verbatim:
        #   nebular_smooth_init = 100 km/s  (sps_vars.f90)
        #   smooth_velocity     = .true.    (sps_vars.f90)
        #   pixel floor         = neb_res_min * 2  (sps_setup.f90)
        # Prospector inherits this through python-fsps.
        if nebular_smooth_init is not None:
            sigma_smooth = float(nebular_smooth_init)

        self.csp_lambda = jnp.asarray(csp_lambda)
        self.nspec      = int(self.csp_lambda.size)

        self.smooth_velocity = bool(smooth_velocity)
        self.sigma_smooth    = float(sigma_smooth)
        self.res_floor_factor = float(res_floor_factor)

        self.nebnz   = int(nebnz)
        self.nebnage = int(nebnage)
        self.nebnip  = int(nebnip)

        suffix = 'WD' if cloudy_dust else 'ND'
        base = Path(sps_home) / 'nebular' / f'ZAU_{suffix}_{isoc_type}'
        self.cont_file = base.with_suffix('.cont')
        self.line_file = base.with_suffix('.lines')

        self._load_continuum()
        self._load_lines()
        self._compute_resolution_elements()
        self._build_gaussians()

        if ssp_flux is not None:
            self.log_qq = self.compute_log_qq(jnp.asarray(ssp_flux))
        else:
            self.log_qq = None

        # Restrict nebular emission to SSPs whose age lies inside BOTH
        # cubes' age ranges (intersection).  This is the safest choice
        # given that the two grids may differ.
        if ssp_ages_lgyr is not None:
            ages = jnp.asarray(ssp_ages_lgyr)
            max_age = min(float(self.nebem_cont_age[-1]),
                          float(self.nebem_line_age[-1]))
            young = ages <= max_age
            self.young_mask = young
            self.young_idx  = jnp.where(young)[0]
            # Sanity guard (2026-07-21): the mask must actually restrict to
            # the CLOUDY grid's age range.  With a mis-normalised age axis
            # every SSP passes and old populations acquire nebular emission.
            n_young = int(np.asarray(young).sum())
            oldest_young_yr = (10.0 ** float(np.asarray(ages)[
                np.asarray(young)].max()) if n_young else 0.0)
            if n_young == 0 or oldest_young_yr > 3.2e8:
                raise RuntimeError(
                    "Nebular young-SSP mask is inconsistent with the CLOUDY "
                    f"grid: {n_young} SSPs flagged young, oldest "
                    f"{oldest_young_yr/1e6:.1f} Myr (grid max_age "
                    f"{max_age:.3f} log10 yr). Check the cube age-axis "
                    "units.")
        else:
            self.young_mask = None
            self.young_idx  = None

        # ── emlines_info.dat <-> cube ordering cross-check (2026-07-21) ──
        # ``line_ind`` values used across the Prospector-compatible API
        # index $SPS_HOME/data/emlines_info.dat, whose ROW ORDER is only
        # guaranteed to match this cube if both files ship from the same
        # FSPS vintage.  Hand-installed cubes (e.g. BPASS) next to a stock
        # emlines_info.dat CAN disagree -- observed on Tursa, where an
        # index-based gather silently returned neighbouring lines.  Consumers
        # inside ceridwen match by wavelength (CSPBasis._neb_cube_rows_for)
        # and are immune; this check warns ANY consumer at construction time
        # and records the verdict in ``self.emline_index_consistent``.
        self.emline_index_consistent = None
        try:
            _info_path = str(Path(sps_home) / "data" / "emlines_info.dat")
            _info_wave = []
            with open(_info_path) as _f:
                for _row in _f:
                    _parts = _row.split(",")
                    if len(_parts) >= 2:
                        _info_wave.append(float(_parts[0]))
            _info_wave = np.asarray(_info_wave)
            _pos = np.asarray(self.nebem_line_pos)
            if _info_wave.size != _pos.size:
                self.emline_index_consistent = False
                warnings.warn(
                    f"emlines_info.dat lists {_info_wave.size} lines but the "
                    f"nebular cube {getattr(self, 'line_file', '?')} has "
                    f"{_pos.size} -- the two files are from different FSPS "
                    "vintages. Raw line_ind indices into this cube are "
                    "UNRELIABLE; ceridwen's own predictions match lines by "
                    "wavelength and are unaffected.", stacklevel=2)
            else:
                _n = min(_info_wave.size, _pos.size)
                _bad = int(np.sum(np.abs(_info_wave[:_n] - _pos[:_n]) > 1.0))
                self.emline_index_consistent = (_bad == 0)
                if _bad:
                    warnings.warn(
                        f"{_bad}/{_n} rows of emlines_info.dat disagree with "
                        "the nebular cube wavelengths by >1 A -- mixed FSPS "
                        "vintages in $SPS_HOME. Raw line_ind indices into "
                        "this cube are UNRELIABLE; ceridwen's own predictions "
                        "match lines by wavelength and are unaffected.",
                        stacklevel=2)
        except OSError:
            pass                              # no emlines_info.dat: nothing to check

    # ── file loaders ------------------------------------------------------
    #
    # ``ZAU_ND_*.cont`` and ``ZAU_ND_*.lines`` were tabulated on different
    # CLOUDY grids since FSPS commit 0db2d3e (2023-09-21).  We therefore
    # parse each file's per-block metadata separately and interpolate each
    # cube against its own axes.

    def _load_continuum(self):
        """Read ``ZAU_*.cont`` and store its cube and its OWN axes."""
        with open(self.cont_file, 'r') as f:
            f.readline()                                # header
            readlamb = np.asarray(f.readline().split(), dtype=np.float64)
            payload  = f.readlines()

        n_expected = 2 * self.nebnz * self.nebnage * self.nebnip
        if len(payload) != n_expected:
            raise ValueError(
                f"{self.cont_file}: expected {n_expected} payload lines "
                f"({self.nebnz}x{self.nebnage}x{self.nebnip} meta+flux "
                f"blocks), found {len(payload)}. Grid dimensions and file "
                "disagree.")
        nflux = len(payload[1].split())
        if readlamb.size != nflux:
            raise ValueError(
                f"{self.cont_file}: wavelength row has {readlamb.size} "
                f"entries but the flux rows have {nflux}. The continuum "
                "cube is internally inconsistent -- re-download or "
                "regenerate the nebular data files.")

        cont = np.empty((self.nspec, self.nebnz, self.nebnage, self.nebnip),
                        dtype=np.float64)
        logz = np.empty(self.nebnz,   dtype=np.float64)
        age  = np.empty(self.nebnage, dtype=np.float64)
        logu = np.empty(self.nebnip,  dtype=np.float64)

        csp_np = np.asarray(self.csp_lambda, dtype=np.float64)
        idx = 0
        for i in range(self.nebnz):
            for j in range(self.nebnage):
                for k in range(self.nebnip):
                    meta = payload[idx].split()
                    logz[i] = float(meta[0])
                    age[j]  = float(meta[1])
                    logu[k] = float(meta[2])
                    idx += 1
                    raw = np.asarray(payload[idx].split(), dtype=np.float64)
                    cont[:, i, j, k] = np.interp(
                        csp_np, readlamb, np.log10(raw + TINY))
                    idx += 1

        age = _normalise_age_axis_to_log10yr(age, self.cont_file)

        z_perm = np.argsort(logz)
        a_perm = np.argsort(age)
        u_perm = np.argsort(logu)
        logz = logz[z_perm]; age = age[a_perm]; logu = logu[u_perm]
        cont = cont[:, z_perm, :, :][:, :, a_perm, :][:, :, :, u_perm]

        self.nebem_cont      = jnp.asarray(cont)
        self.nebem_cont_logz = jnp.asarray(logz)
        self.nebem_cont_age  = jnp.asarray(age)
        self.nebem_cont_logu = jnp.asarray(logu)

    def _load_lines(self):
        """Read ``ZAU_*.lines`` and store its cube and its OWN axes."""
        with open(self.line_file, 'r') as f:
            f.readline()                                # header
            line_pos = np.asarray(f.readline().split(), dtype=np.float64)
            payload  = f.readlines()

        n_expected = 2 * self.nebnz * self.nebnage * self.nebnip
        if len(payload) != n_expected:
            raise ValueError(
                f"{self.line_file}: expected {n_expected} payload lines "
                f"({self.nebnz}x{self.nebnage}x{self.nebnip} meta+flux "
                f"blocks), found {len(payload)}. Grid dimensions and file "
                "disagree.")
        # Repair the upstream stale-wavelength-row bug in ZAU_WD_*.lines
        # (128-entry header row on 166-column flux blocks); no-op for
        # healthy files of any vintage.
        line_pos = _resolve_line_wavelengths(self.line_file, line_pos, payload)

        nem = line_pos.size
        self.nemline = nem
        cube = np.empty((nem, self.nebnz, self.nebnage, self.nebnip),
                        dtype=np.float64)
        logz = np.empty(self.nebnz,   dtype=np.float64)
        age  = np.empty(self.nebnage, dtype=np.float64)
        logu = np.empty(self.nebnip,  dtype=np.float64)

        idx = 0
        for i in range(self.nebnz):
            for j in range(self.nebnage):
                for k in range(self.nebnip):
                    meta = payload[idx].split()
                    logz[i] = float(meta[0])
                    age[j]  = float(meta[1])
                    logu[k] = float(meta[2])
                    idx += 1
                    vals = np.asarray(payload[idx].split(), dtype=np.float64)
                    cube[:, i, j, k] = np.log10(vals + TINY)
                    idx += 1

        age = _normalise_age_axis_to_log10yr(age, self.line_file)

        z_perm = np.argsort(logz)
        a_perm = np.argsort(age)
        u_perm = np.argsort(logu)
        logz = logz[z_perm]; age = age[a_perm]; logu = logu[u_perm]
        cube = cube[:, z_perm, :, :][:, :, a_perm, :][:, :, :, u_perm]

        self.nebem_line       = jnp.asarray(cube)
        self.nebem_line_pos   = jnp.asarray(line_pos)
        self.nebem_line_logz  = jnp.asarray(logz)
        self.nebem_line_age   = jnp.asarray(age)
        self.nebem_line_logu  = jnp.asarray(logu)

        # Legacy aliases — for back-compat with code that uses the
        # FSPS-canonical axis names.  Pointed at the line axes (matches
        # FSPS); per-cube users should reach for the explicit
        # ``nebem_{cont,line}_{logz,age,logu}`` attributes instead.
        self.nebem_logz = self.nebem_line_logz
        self.nebem_age  = self.nebem_line_age
        self.nebem_logu = self.nebem_line_logu

    # ── ionising-photon rate ---------------------------------------------
    def compute_log_qq(self, ssp_flux):
        """
        ``log10(Q)`` for every (Z, age) SSP, matching FSPS's run-time
        formula::

            qq = ∫ L_nu / lambda  dλ          (0 < λ < 912 Å)
            Q  = (L_sun_erg / h_erg_s) × qq   (photons / s)

        Forced to float64 because the deep-UV fluxes are tiny.
        """
        mask = self.csp_lambda < LYMAN_LIMIT_AA
        wave_ion = self.csp_lambda[mask].astype(jnp.float64)
        flux_ion = ssp_flux[..., mask].astype(jnp.float64)
        qq = jnp.trapezoid(flux_ion / wave_ion, x=wave_ion)
        scale = LSUN_ERG_S / HPLANK_ERG_S
        return jnp.log10(jnp.maximum(qq * scale, TINY))

    # ── line broadening helpers -------------------------------------------
    def _compute_resolution_elements(self):
        """
        Per-line minimum wavelength resolution element  Δλ_pix.

        Matches ``sps_setup.f90`` lines 960-965 of FSPS, using
        ``searchsorted(side='right') - 1`` (the Python equivalent of
        FSPS's ``locate``) and the upper-neighbour pixel spacing.
        """
        idx = jnp.clip(
            jnp.searchsorted(self.csp_lambda, self.nebem_line_pos, side='right') - 1,
            1, self.nspec - 2,
        )
        self.neb_res_min = self.csp_lambda[idx + 1] - self.csp_lambda[idx]

    def _build_gaussians(self):
        """Pre-compute ``gaussnebarr`` of shape ``(nspec, nemline)``."""
        line_pos = self.nebem_line_pos
        if self.smooth_velocity:
            dlam = line_pos * self.sigma_smooth / CLIGHT_AA_S * 1.0e13
        else:
            dlam = jnp.full_like(line_pos, self.sigma_smooth)
        dlam = jnp.maximum(dlam, self.neb_res_min * self.res_floor_factor)

        lam   = self.csp_lambda[:, None]
        l0    = line_pos[None, :]
        dl    = dlam[None, :]
        norm  = 1.0 / (SQRT_2PI * dl)
        prof  = jnp.exp(-0.5 * ((lam - l0) / dl) ** 2)
        scale = l0 ** 2 / CLIGHT_AA_S
        self.gaussnebarr = norm * prof * scale
        self.dlam_lines  = dlam                                          # diagnostic

    # ── public evaluation -------------------------------------------------
    #
    # PHYSICALLY STRICT: each cube is interpolated against its own
    # (logZ, age, logU) axes.

    def evaluate(self, logZ, logU, logage, logQ):
        """
        Single-point evaluation — returns ``(cont, lines)`` in Lsun/Hz.

        The continuum cube is interpolated against the **cont** axes,
        the line cube against the **line** axes (each cube is therefore
        evaluated at the physical point CLOUDY was actually run at).
        """
        # ── continuum cube on its own axes
        zc  = _locate(logZ,   self.nebem_cont_logz)
        dzc = _frac(logZ,     self.nebem_cont_logz, zc)
        uc  = _locate(logU,   self.nebem_cont_logu)
        duc = _frac(logU,     self.nebem_cont_logu, uc)
        ac  = _locate(logage, self.nebem_cont_age)
        dac = _frac(logage,   self.nebem_cont_age,  ac)
        log_cont = _trilinear(self.nebem_cont, zc, dzc, ac, dac, uc, duc)

        # ── line cube on its own axes
        zl  = _locate(logZ,   self.nebem_line_logz)
        dzl = _frac(logZ,     self.nebem_line_logz, zl)
        ul  = _locate(logU,   self.nebem_line_logu)
        dul = _frac(logU,     self.nebem_line_logu, ul)
        al  = _locate(logage, self.nebem_line_age)
        dal = _frac(logage,   self.nebem_line_age,  al)
        log_line = _trilinear(self.nebem_line, zl, dzl, al, dal, ul, dul)

        cont_flux = jnp.power(10.0, log_cont + logQ)
        line_lum  = jnp.power(10.0, log_line + logQ)
        line_spec = self.gaussnebarr @ line_lum
        return cont_flux, line_spec

    def evaluate_batch(self, logZ_gas, logU, ssp_ages_young, logqq_young,
                        return_components=False):
        """
        Vectorised evaluation for all (Z_ssp, age_young) pairs at a
        single ``(logZ_gas, logU)``.  Each cube is bilinearly collapsed
        in (Z, U) on its OWN axes, then linearly interpolated in age
        across the young-SSP set.

        Parameters
        ----------
        return_components : bool
            When ``False`` (default), return ``cont_flux + line_spec`` as a
            single ``(n_z, n_wave, n_young)`` array -- the legacy behaviour
            consumed by the four ``CSPBasis.get_spectrum_*_neb`` variants.
            When ``True``, return the tuple ``(cont_flux, line_spec)`` in the
            same layout, so the caller can decide whether to include the
            broadened emission lines in the continuum spectrum (the
            prospector ``nebemlineinspec`` switch).
        """
        logZ_gas = jnp.squeeze(logZ_gas)
        logU     = jnp.squeeze(logU)

        def _interp_cube(cube, logz_grid, age_grid, logu_grid):
            z1 = _locate(logZ_gas, logz_grid)
            dz = _frac(logZ_gas,   logz_grid, z1)
            u1 = _locate(logU,     logu_grid)
            du = _frac(logU,       logu_grid, u1)

            w00 = (1.0 - dz) * (1.0 - du)
            w01 = (1.0 - dz) *       du
            w10 =       dz   * (1.0 - du)
            w11 =       dz   *       du
            zu = (w00 * cube[..., z1,     :, u1    ]
                + w01 * cube[..., z1,     :, u1 + 1]
                + w10 * cube[..., z1 + 1, :, u1    ]
                + w11 * cube[..., z1 + 1, :, u1 + 1])

            a1 = jnp.clip(jnp.searchsorted(age_grid, ssp_ages_young) - 1,
                          0, age_grid.shape[0] - 2)
            da = jnp.clip(
                (ssp_ages_young - age_grid[a1])
                / (age_grid[a1 + 1] - age_grid[a1]),
                0.0, 1.0,
            )
            return (1.0 - da)[None, :] * zu[..., a1] + da[None, :] * zu[..., a1 + 1]

        log_cont = _interp_cube(self.nebem_cont,
                                self.nebem_cont_logz,
                                self.nebem_cont_age,
                                self.nebem_cont_logu)                     # (nspec , n_young)
        log_line = _interp_cube(self.nebem_line,
                                self.nebem_line_logz,
                                self.nebem_line_age,
                                self.nebem_line_logu)                     # (nlines, n_young)

        # ── PERF (2026-08-12): factorise the metallicity axis OUT of the
        # line-painting contraction. ``log_cont`` / ``log_line`` come out of
        # ``_interp_cube`` with NO z axis -- the only z dependence in the whole
        # expression is ``logqq_young``. The old form broadcast logqq into the
        # exponent first and then contracted
        #     einsum('wl,zly->zwy', gaussnebarr, line_lum)
        # once per metallicity, costing n_z * n_wave * n_young * n_lines
        # multiply-adds for a result that is rank-1 in z. Using
        #     10**(log_X[.,y] + logqq[z,y])
        #       = 10**(log_X[.,y] + ref[y]) * 10**(logqq[z,y] - ref[y])
        # the sum over lines commutes through the second factor, so the
        # contraction is done ONCE at (n_wave, n_young) and then scaled. That
        # is an ~n_z-fold reduction in both FLOPs and bytes moved on what is
        # otherwise the single most expensive op in the forward model.
        #
        # ``ref`` is the per-young-age MAXIMUM over z, which is what makes the
        # split safe in float32: the first factor never exceeds the largest
        # term the old code already formed (so no new overflow can appear --
        # naively splitting into 10**log_X * 10**logqq WOULD overflow, since
        # logqq ~ 10**46 photons/s alone exceeds the float32 range), and the
        # second factor lies in (0, 1]. The ``isfinite`` guard covers ages
        # whose ionising-photon rate is identically zero (log_qq = -inf):
        # ref -> 0, scale -> 0, product -> 0, matching the old behaviour
        # instead of producing 0 * inf = nan.
        ref   = jnp.max(logqq_young, axis=0)                          # (n_young,)
        ref   = jnp.where(jnp.isfinite(ref), ref, 0.0)
        scale = jnp.power(10.0, logqq_young - ref[None, :])            # (n_z, n_young)

        cont_base = jnp.power(10.0, log_cont + ref[None, :])           # (nspec , n_young)
        line_base = jnp.einsum('wl,ly->wy', self.gaussnebarr,
                               jnp.power(10.0, log_line + ref[None, :]))
        cont_flux = cont_base[None, :, :] * scale[:, None, :]          # (n_z, nspec, n_young)
        line_spec = line_base[None, :, :] * scale[:, None, :]          # (n_z, nspec, n_young)
        if return_components:
            return (cont_flux.transpose(0, 2, 1),
                    line_spec.transpose(0, 2, 1))
        return (cont_flux + line_spec).transpose(0, 2, 1)

    def evaluate_batch_factored(self, logZ_gas, logU, ssp_ages_young,
                                logqq_young, include_lines=True):
        """Nebular emission in FACTORED (rank-1-in-z) form.

        Returns ``(base, scale)`` with shapes ``(n_young, n_wave)`` and
        ``(n_z, n_young)`` such that

            neb[z, y, w] == scale[z, y] * base[y, w]

        exactly (same arithmetic as :meth:`evaluate_batch`, just not
        expanded). Consumers that immediately contract the metallicity axis
        away -- which is every ``CSPBasis.get_spectrum_*_neb`` variant --
        should use this instead of :meth:`evaluate_batch`, because the
        ``(n_z, n_young, n_wave)`` product never has to be materialised:
        the z sum can be folded into the SSP weights first.

        See ``CSPBasis._neb_factored`` for the consuming side.
        """
        logZ_gas = jnp.squeeze(logZ_gas)
        logU     = jnp.squeeze(logU)

        def _interp_cube(cube, logz_grid, age_grid, logu_grid):
            z1 = _locate(logZ_gas, logz_grid)
            dz = _frac(logZ_gas,   logz_grid, z1)
            u1 = _locate(logU,     logu_grid)
            du = _frac(logU,       logu_grid, u1)
            w00 = (1.0 - dz) * (1.0 - du)
            w01 = (1.0 - dz) *       du
            w10 =       dz   * (1.0 - du)
            w11 =       dz   *       du
            zu = (w00 * cube[..., z1,     :, u1    ]
                + w01 * cube[..., z1,     :, u1 + 1]
                + w10 * cube[..., z1 + 1, :, u1    ]
                + w11 * cube[..., z1 + 1, :, u1 + 1])
            a1 = jnp.clip(jnp.searchsorted(age_grid, ssp_ages_young) - 1,
                          0, age_grid.shape[0] - 2)
            da = jnp.clip(
                (ssp_ages_young - age_grid[a1])
                / (age_grid[a1 + 1] - age_grid[a1]),
                0.0, 1.0,
            )
            return (1.0 - da)[None, :] * zu[..., a1] + da[None, :] * zu[..., a1 + 1]

        log_cont = _interp_cube(self.nebem_cont, self.nebem_cont_logz,
                                self.nebem_cont_age, self.nebem_cont_logu)
        ref   = jnp.max(logqq_young, axis=0)
        ref   = jnp.where(jnp.isfinite(ref), ref, 0.0)
        scale = jnp.power(10.0, logqq_young - ref[None, :])            # (n_z, n_young)

        base = jnp.power(10.0, log_cont + ref[None, :])                # (nspec, n_young)
        if include_lines:
            log_line = _interp_cube(self.nebem_line, self.nebem_line_logz,
                                    self.nebem_line_age, self.nebem_line_logu)
            base = base + jnp.einsum(
                'wl,ly->wy', self.gaussnebarr,
                jnp.power(10.0, log_line + ref[None, :]))
        return base.T, scale                                           # (n_young, n_wave), (n_z, n_young)

    def evaluate_batch_line_lum(self, logZ_gas, logU, ssp_ages_young,
                                logqq_young):
        """Per-line luminosities WITHOUT the spectral painting round-trip.

        Same (Z_gas, U) bilinear collapse and young-age interpolation as
        :meth:`evaluate_batch`, but returns the raw line luminosities
        ``(n_z, n_young, n_lines)`` [Lsun] instead of painting them onto the
        wavelength grid.  Used by ``CSPBasis.predict_line_fluxes`` (2026-07-21):
        extracting line fluxes back out of the painted spectrum with Gaussian
        apertures recovers only ~0.35-0.5 of the painted flux (the aperture's
        narrow-line normalisation vs the resolution-floor + LOSVD-broadened
        line width), so ``Lines`` observations are now predicted directly
        from these grid values -- exact for any library, resolution or
        smoothing configuration.
        """
        logZ_gas = jnp.squeeze(logZ_gas)
        logU     = jnp.squeeze(logU)
        cube      = self.nebem_line
        logz_grid = self.nebem_line_logz
        age_grid  = self.nebem_line_age
        logu_grid = self.nebem_line_logu

        z1 = _locate(logZ_gas, logz_grid)
        dz = _frac(logZ_gas,   logz_grid, z1)
        u1 = _locate(logU,     logu_grid)
        du = _frac(logU,       logu_grid, u1)
        w00 = (1.0 - dz) * (1.0 - du)
        w01 = (1.0 - dz) *       du
        w10 =       dz   * (1.0 - du)
        w11 =       dz   *       du
        zu = (w00 * cube[..., z1,     :, u1    ]
            + w01 * cube[..., z1,     :, u1 + 1]
            + w10 * cube[..., z1 + 1, :, u1    ]
            + w11 * cube[..., z1 + 1, :, u1 + 1])          # (nlines, nage)

        a1 = jnp.clip(jnp.searchsorted(age_grid, ssp_ages_young) - 1,
                      0, age_grid.shape[0] - 2)
        da = jnp.clip(
            (ssp_ages_young - age_grid[a1])
            / (age_grid[a1 + 1] - age_grid[a1]),
            0.0, 1.0,
        )
        log_line = ((1.0 - da)[None, :] * zu[..., a1]
                    +       da[None, :] * zu[..., a1 + 1])   # (nlines, n_young)

        # (n_z, nlines, n_young) -> (n_z, n_young, nlines)
        line_lum = jnp.power(10.0, log_line[None, :, :]
                             + logqq_young[:, None, :])
        return line_lum.transpose(0, 2, 1)

    # ── parameter bookkeeping --------------------------------------------
    def get_default_params(self):
        """Return the FSPS default nebular parameters."""
        return {'gas_logz': jnp.asarray(0.0),
                'gas_logu': jnp.asarray(-2.0)}

    def get_param_names(self):
        return ['gas_logz', 'gas_logu']

    # ── repr --------------------------------------------------------------
    def __repr__(self):
        bits = [
            "<NebularModel (physically-strict per-cube axes)>",
            f"  cloudy file       : {self.cont_file.name}",
            f"  csp_lambda        : {self.nspec} pts, "
            f"[{float(self.csp_lambda[0]):.1f} .. {float(self.csp_lambda[-1]):.1f}] Å",
            f"  CLOUDY grid       : nz={self.nebnz} nage={self.nebnage} nu={self.nebnip}",
            f"  nemline           : {self.nemline}",
            f"  cont logZ range   : [{float(self.nebem_cont_logz[0]):.2f} .. "
            f"{float(self.nebem_cont_logz[-1]):.2f}]",
            f"  line logZ range   : [{float(self.nebem_line_logz[0]):.2f} .. "
            f"{float(self.nebem_line_logz[-1]):.2f}]",
            f"  cont logU range   : [{float(self.nebem_cont_logu[0]):.2f} .. "
            f"{float(self.nebem_cont_logu[-1]):.2f}]",
            f"  line logU range   : [{float(self.nebem_line_logu[0]):.2f} .. "
            f"{float(self.nebem_line_logu[-1]):.2f}]",
            f"  cont logage range : [{float(self.nebem_cont_age[0]):.2f} .. "
            f"{float(self.nebem_cont_age[-1]):.2f}]",
            f"  line logage range : [{float(self.nebem_line_age[0]):.2f} .. "
            f"{float(self.nebem_line_age[-1]):.2f}]",
            f"  smooth_velocity   : {self.smooth_velocity}",
            f"  sigma_smooth      : {self.sigma_smooth}",
        ]
        if self.log_qq is not None:
            bits.append(
                f"  log_qq table      : shape {tuple(self.log_qq.shape)}, "
                f"range [{float(self.log_qq.min()):.2f} .. "
                f"{float(self.log_qq.max()):.2f}]"
            )
        return "\n".join(bits)
