"""
NebularGridModel_fsps_match.py
===============================
**Backup of the FSPS-bug-replicating variant of NebularModel.**

This file is preserved as a faithful reproduction of FSPS's `add_nebular.f90`
run-time behaviour, *including* a subtle inconsistency introduced upstream
on 2023-09-21 in commit ``0db2d3e`` of ``cconroy20/fsps`` (X-ray binary
nebular contribution from ``@kgarofali``).  At that point
``ZAU_ND_*.lines`` was re-tabulated onto a new ``(logZ, age)`` CLOUDY
grid but ``ZAU_ND_*.cont`` was not regenerated to match.
``sps_setup.f90`` reads both files into the same global axis arrays and
the second read overwrites the first, so FSPS thereafter interpolates
the continuum cube against the lines-file axes.

This class replicates that exact behaviour and therefore agrees with FSPS
to better than 0.5% on the SFH-integrated nebular spectrum.  Keep this
file around if you need to benchmark against, or reproduce the output of,
an upstream FSPS install.

For day-to-day physics you almost certainly want the default
:class:`ceridwen.neb.NebularGridModel.NebularModel` instead — that
version interpolates each cube against its OWN axes (the physically
strict behaviour), and so its continuum corresponds to the actual
CLOUDY runs that produced the ``.cont`` file rather than to the
mis-labelled axes FSPS happens to use.
"""

from pathlib import Path

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


def _locate(x, grid):
    """Index of the cell with ``grid[i] <= x < grid[i+1]``, clipped to ``[0, n-2]``."""
    return jnp.clip(jnp.searchsorted(grid, x) - 1, 0, grid.size - 2)


def _frac(x, grid, i):
    """Fractional position of ``x`` inside ``[grid[i], grid[i+1]]``, clipped to [0,1]."""
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


class NebularModelFSPSMatch:
    """
    FSPS-matching nebular model — reproduces FSPS's behaviour bit-for-bit
    even where that behaviour is internally inconsistent (cont cube
    interpolated on the lines grid).  Use only for benchmarking.

    See module-level docstring and the default
    :class:`ceridwen.neb.NebularGridModel.NebularModel` for the
    physically-strict variant.
    """

    def __init__(self,
                 cloudy_dust,
                 sps_home,
                 csp_lambda,
                 ssp_flux=None,
                 ssp_ages_lgyr=None,
                 isoc_type='mist',
                 nebnz=11, nebnage=10, nebnip=7,
                 smooth_velocity=True,
                 sigma_smooth=0.0,
                 nebular_smooth_init=None):
        if nebular_smooth_init is not None:
            sigma_smooth = float(nebular_smooth_init)

        self.csp_lambda = jnp.asarray(csp_lambda)
        self.nspec      = int(self.csp_lambda.size)

        self.smooth_velocity = bool(smooth_velocity)
        self.sigma_smooth    = float(sigma_smooth)

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

        if ssp_ages_lgyr is not None:
            ages = jnp.asarray(ssp_ages_lgyr)
            max_age = min(float(self.nebem_cont_age[-1]),
                          float(self.nebem_line_age[-1]))
            young = ages <= max_age
            self.young_mask = young
            self.young_idx  = jnp.where(young)[0]
        else:
            self.young_mask = None
            self.young_idx  = None

    def _load_continuum(self):
        with open(self.cont_file, 'r') as f:
            f.readline()
            readlamb = np.asarray(f.readline().split(), dtype=np.float64)
            payload  = f.readlines()

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

        if age.max() > 30.0:
            age = np.log10(age)

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
        with open(self.line_file, 'r') as f:
            f.readline()
            line_pos = np.asarray(f.readline().split(), dtype=np.float64)
            payload  = f.readlines()

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

        if age.max() > 30.0:
            age = np.log10(age)

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

        # In this FSPS-matching variant, the canonical axes (the ones
        # used by both interpolations) are the LINE-cube axes — matching
        # the FSPS run-time behaviour (post-read-overwrite).
        self.nebem_logz = self.nebem_line_logz
        self.nebem_age  = self.nebem_line_age
        self.nebem_logu = self.nebem_line_logu

    def compute_log_qq(self, ssp_flux):
        mask = self.csp_lambda < LYMAN_LIMIT_AA
        wave_ion = self.csp_lambda[mask].astype(jnp.float64)
        flux_ion = ssp_flux[..., mask].astype(jnp.float64)
        qq = jnp.trapezoid(flux_ion / wave_ion, x=wave_ion)
        scale = LSUN_ERG_S / HPLANK_ERG_S
        return jnp.log10(jnp.maximum(qq * scale, TINY))

    def _compute_resolution_elements(self):
        idx = jnp.clip(
            jnp.searchsorted(self.csp_lambda, self.nebem_line_pos, side='right') - 1,
            1, self.nspec - 2,
        )
        self.neb_res_min = self.csp_lambda[idx + 1] - self.csp_lambda[idx]

    def _build_gaussians(self):
        line_pos = self.nebem_line_pos
        if self.smooth_velocity:
            dlam = line_pos * self.sigma_smooth / CLIGHT_AA_S * 1.0e13
        else:
            dlam = jnp.full_like(line_pos, self.sigma_smooth)
        dlam = jnp.maximum(dlam, self.neb_res_min * 2.0)

        lam   = self.csp_lambda[:, None]
        l0    = line_pos[None, :]
        dl    = dlam[None, :]
        norm  = 1.0 / (SQRT_2PI * dl)
        prof  = jnp.exp(-0.5 * ((lam - l0) / dl) ** 2)
        scale = l0 ** 2 / CLIGHT_AA_S
        self.gaussnebarr = norm * prof * scale
        self.dlam_lines  = dlam

    # ── evaluation: both cubes on the line axes (FSPS-matching) ──────────
    def evaluate(self, logZ, logU, logage, logQ):
        z1 = _locate(logZ,   self.nebem_line_logz)
        dz = _frac(logZ,     self.nebem_line_logz, z1)
        u1 = _locate(logU,   self.nebem_line_logu)
        du = _frac(logU,     self.nebem_line_logu, u1)
        a1 = _locate(logage, self.nebem_line_age)
        da = _frac(logage,   self.nebem_line_age,  a1)

        log_cont = _trilinear(self.nebem_cont, z1, dz, a1, da, u1, du)
        log_line = _trilinear(self.nebem_line, z1, dz, a1, da, u1, du)

        cont_flux = jnp.power(10.0, log_cont + logQ)
        line_lum  = jnp.power(10.0, log_line + logQ)
        line_spec = self.gaussnebarr @ line_lum
        return cont_flux, line_spec

    def evaluate_batch(self, logZ_gas, logU, ssp_ages_young, logqq_young):
        logZ_gas = jnp.squeeze(logZ_gas)
        logU     = jnp.squeeze(logU)

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

        cont_zu = (w00 * self.nebem_cont[:, z1,     :, u1    ]
                 + w01 * self.nebem_cont[:, z1,     :, u1 + 1]
                 + w10 * self.nebem_cont[:, z1 + 1, :, u1    ]
                 + w11 * self.nebem_cont[:, z1 + 1, :, u1 + 1])
        line_zu = (w00 * self.nebem_line[:, z1,     :, u1    ]
                 + w01 * self.nebem_line[:, z1,     :, u1 + 1]
                 + w10 * self.nebem_line[:, z1 + 1, :, u1    ]
                 + w11 * self.nebem_line[:, z1 + 1, :, u1 + 1])

        a1 = jnp.clip(jnp.searchsorted(age_grid, ssp_ages_young) - 1,
                      0, age_grid.shape[0] - 2)
        da = jnp.clip(
            (ssp_ages_young - age_grid[a1])
            / (age_grid[a1 + 1] - age_grid[a1]),
            0.0, 1.0,
        )

        log_cont = (1.0 - da)[None, :] * cont_zu[:, a1] + da[None, :] * cont_zu[:, a1 + 1]
        log_line = (1.0 - da)[None, :] * line_zu[:, a1] + da[None, :] * line_zu[:, a1 + 1]

        cont_flux = jnp.power(10.0, log_cont[None, :, :] + logqq_young[:, None, :])
        line_lum  = jnp.power(10.0, log_line[None, :, :] + logqq_young[:, None, :])
        line_spec = jnp.einsum('wl,zly->zwy', self.gaussnebarr, line_lum)
        neb_total = cont_flux + line_spec
        return neb_total.transpose(0, 2, 1)

    def get_default_params(self):
        return {'gas_logz': jnp.asarray(0.0),
                'gas_logu': jnp.asarray(-2.0)}

    def get_param_names(self):
        return ['gas_logz', 'gas_logu']
