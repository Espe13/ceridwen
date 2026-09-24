"""CLOUDY-grid nebular emission model.

Each cube (``ZAU_<ND|WD>_<isoc>.cont`` / ``.lines``) is interpolated linearly in
(logZ, log age, logU) against its own axes and scaled by the ionising-photon rate Q of
each SSP; lines are painted as Gaussians of width max(sigma_smooth, res_floor_factor pixels).
"""

from pathlib import Path

import warnings
import os
import numpy as np
import jax
import jax.numpy as jnp


CLIGHT_AA_S    = 2.9979e18     # A/s
HPLANK_ERG_S   = 6.6261e-27    # erg s
LSUN_ERG_S     = 3.839e33      # erg/s
LYMAN_LIMIT_AA = 912.0         # Lyman limit, Angstrom
SQRT_2PI       = float(np.sqrt(2.0 * np.pi))
TINY           = 1.0e-95       # floor for log10


def _normalise_age_axis_to_log10yr(age, src_file):
    """Cube age axis as log10(age/yr), whether the file lists log10(yr), yr or Myr; raises on an unrecognisable range."""
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
    """Wavelength row of a ``.lines`` cube, taken from the dust-free sibling cube or
    ``emlines_info.dat`` when the file's own row has fewer entries than its flux rows.
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
    """Trilinear interpolation on a cube of shape ``(..., nz, nage, nu)``."""
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


class NebularModel:
    """Nebular continuum and lines from the CLOUDY cubes under ``<sps_home>/nebular``.

    Parameters
    ----------
    cloudy_dust : bool -- CLOUDY grids with dust inside the H II region, ``ZAU_WD`` (True), or
        without, ``ZAU_ND`` (False, the default, as in FSPS / python-fsps / Prospector).
    csp_lambda : (nspec,) -- model wavelength grid [A].
    ssp_flux : (n_z, n_age, n_wave) -- SSP L_nu [L_sun/Hz]; gives ``log_qq`` (n_z, n_age).
    ssp_ages_lgyr : (n_age,) -- log10(age/yr) of the SSPs; ages inside both cubes are "young".
    isoc_type : str -- ZAU file suffix.
    nebnz, nebnage, nebnip : int -- cube dimensions.
    smooth_velocity : bool -- ``sigma_smooth`` in km/s (True) or A.
    sigma_smooth : float -- intrinsic width of the painted lines (0: pixel floor only).
    res_floor_factor : float -- minimum painted width in local pixels.

    Attributes
    ----------
    nebem_cont : (nspec, nz, nage, nu) -- log10(L_nu / Q)
    nebem_line : (nemline, nz, nage, nu) -- log10(L / Q)
    nebem_line_pos : (nemline,) -- rest wavelengths [A]
    gaussnebarr : (nspec, nemline) -- painted profiles including lambda^2/c
    young_mask, young_idx : the young ages over ``ssp_ages_lgyr``
    """

    def __init__(self,
                 cloudy_dust=False,
                 sps_home=None,
                 csp_lambda=None,
                 ssp_flux=None,
                 ssp_ages_lgyr=None,
                 isoc_type='mist',
                 nebnz=11, nebnage=10, nebnip=7,
                 smooth_velocity=True,
                 sigma_smooth=0.0,
                 res_floor_factor=2.0,
                 nebular_smooth_init=None):
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

        if sps_home is None or csp_lambda is None:
            raise TypeError("NebularModel needs sps_home= and csp_lambda=")
        self.cloudy_dust = bool(cloudy_dust)
        suffix = 'WD' if self.cloudy_dust else 'ND'
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

        if ssp_ages_lgyr is not None:
            ages = jnp.asarray(ssp_ages_lgyr)
            max_age = min(float(self.nebem_cont_age[-1]),
                          float(self.nebem_line_age[-1]))
            young = ages <= max_age
            self.young_mask = young
            self.young_idx  = jnp.where(young)[0]
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

        self.emline_index_consistent = None
        self.nebem_line_names = None
        try:
            _info_path = str(Path(sps_home) / "data" / "emlines_info.dat")
            _info_wave = []
            _info_name = []
            with open(_info_path) as _f:
                for _row in _f:
                    _parts = _row.split(",")
                    if len(_parts) >= 2:
                        _info_wave.append(float(_parts[0]))
                        _info_name.append(_parts[1].strip())
            _info_wave = np.asarray(_info_wave)
            _pos = np.asarray(self.nebem_line_pos)
            # names per cube row, matched by wavelength (1 A) like CSPBasis matches Lines
            if _info_wave.size:
                _j = np.argmin(np.abs(_pos[:, None] - _info_wave[None, :]), axis=1)
                self.nebem_line_names = [
                    _info_name[j] if abs(_pos[i] - _info_wave[j]) <= 1.0 else None
                    for i, j in enumerate(_j)]
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


    def _load_continuum(self):
        """Read ``.cont``: cube interpolated onto ``csp_lambda`` as log10, axes sorted ascending."""
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
        """Read ``.lines``: log10 line cube, wavelengths, axes sorted ascending."""
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

        self.nebem_logz = self.nebem_line_logz
        self.nebem_age  = self.nebem_line_age
        self.nebem_logu = self.nebem_line_logu

    def compute_log_qq(self, ssp_flux):
        """log10 Q [photons/s] for every SSP: (L_sun/h) * int_{lambda<912} L_nu / lambda dlambda (float64)."""
        mask = self.csp_lambda < LYMAN_LIMIT_AA
        wave_ion = self.csp_lambda[mask].astype(jnp.float64)
        flux_ion = ssp_flux[..., mask].astype(jnp.float64)
        qq = jnp.trapezoid(flux_ion / wave_ion, x=wave_ion)
        scale = LSUN_ERG_S / HPLANK_ERG_S
        return jnp.log10(jnp.maximum(qq * scale, TINY))

    def _compute_resolution_elements(self):
        """Per-line local pixel width of ``csp_lambda`` [A]."""
        idx = jnp.clip(
            jnp.searchsorted(self.csp_lambda, self.nebem_line_pos, side='right') - 1,
            1, self.nspec - 2,
        )
        self.neb_res_min = self.csp_lambda[idx + 1] - self.csp_lambda[idx]

    def line_profiles(self, sigma_kms=0.0):
        """(nspec, nemline) NumPy profiles like ``gaussnebarr`` with width
        sqrt(floor^2 + (lambda sigma_kms / c)^2): a line painted at the floor and then broadened.
        """
        pos = np.asarray(self.nebem_line_pos, dtype=np.float64)
        lam = np.asarray(self.csp_lambda, dtype=np.float64)
        floor = np.asarray(self.neb_res_min, dtype=np.float64) * self.res_floor_factor
        if self.smooth_velocity:
            base = pos * self.sigma_smooth / CLIGHT_AA_S * 1.0e13
        else:
            base = np.full_like(pos, self.sigma_smooth)
        dl0 = np.maximum(base, floor)
        dl = np.sqrt(dl0 ** 2 + (pos * float(sigma_kms) / CLIGHT_AA_S * 1.0e13) ** 2)
        prof = np.exp(-0.5 * ((lam[:, None] - pos[None, :]) / dl[None, :]) ** 2)
        return prof / (SQRT_2PI * dl[None, :]) * (pos[None, :] ** 2 / CLIGHT_AA_S)

    def _build_gaussians(self):
        """``gaussnebarr`` (nspec, nemline): normalised Gaussians times lambda^2/c."""
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


    def evaluate(self, logZ, logU, logage, logQ):
        """``(cont (nspec,), lines (nspec,))`` [L_sun/Hz] at one (logZ, logU, logage, logQ)."""
        zc  = _locate(logZ,   self.nebem_cont_logz)
        dzc = _frac(logZ,     self.nebem_cont_logz, zc)
        uc  = _locate(logU,   self.nebem_cont_logu)
        duc = _frac(logU,     self.nebem_cont_logu, uc)
        ac  = _locate(logage, self.nebem_cont_age)
        dac = _frac(logage,   self.nebem_cont_age,  ac)
        log_cont = _trilinear(self.nebem_cont, zc, dzc, ac, dac, uc, duc)

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
        """(n_z, n_young, nspec) nebular spectra at one (logZ_gas, logU) for the young SSP ages,
        or ``(cont, lines)`` in that layout with ``return_components``.  The metallicity dependence
        enters only through ``logqq_young``; the per-age reference ``ref`` keeps 10**(...) in float32 range.
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
        """``(base (n_young, n_wave), scale (n_z, n_young))`` with neb[z, y, w] == scale[z, y] * base[y, w]
        (same arithmetic as ``evaluate_batch``, not expanded).  ``include_lines="both"`` returns
        ``(base_continuum, base_with_lines, scale)`` from one evaluation.
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
            base_full = base + jnp.einsum(
                'wl,ly->wy', self.gaussnebarr,
                jnp.power(10.0, log_line + ref[None, :]))
            if include_lines == "both":
                return base.T, base_full.T, scale
            base = base_full
        return base.T, scale                                           # (n_young, n_wave), (n_z, n_young)

    def evaluate_batch_line_lum(self, logZ_gas, logU, ssp_ages_young,
                                logqq_young):
        """Line luminosities (n_z, n_young, nlines) [L_sun] at one (logZ_gas, logU), no painting."""
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

        line_lum = jnp.power(10.0, log_line[None, :, :]
                             + logqq_young[:, None, :])
        return line_lum.transpose(0, 2, 1)

    def get_default_params(self):
        """Default nebular parameters (gas_logz = 0, gas_logu = -2)."""
        return {'gas_logz': jnp.asarray(0.0),
                'gas_logu': jnp.asarray(-2.0)}

    def get_param_names(self):
        return ['gas_logz', 'gas_logu']

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
