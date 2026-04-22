import jax.numpy as jnp
from pathlib import Path
import numpy as np
import astropy.constants as const

CLIGHT_ANG_S   = 2.9979e18   # Å/s  (clight in FSPS)
HPLANK_ERG_S   = 6.6261e-27  # erg·s
LSUN_ERG_S     = 3.839e33    # erg/s per Lsun  (lsun in FSPS)
LYMAN_LIMIT_AA = 912.0       # Å


def _locate(x, grid):
    """Index of the cell just below x; clipped to valid range."""
    return jnp.clip(jnp.searchsorted(grid, x) - 1, 0, grid.size - 2)


def _trilinear(cube, z1, dz, a1, da, u1, du):
    """
    Trilinear interpolation on cube with shape (..., nz, nage, nu).
    Leading dimensions are broadcast independently; z1/a1/u1 are scalars.
    """
    w = jnp.array([
        (1-dz)*(1-da)*(1-du),
        (1-dz)*(1-da)*(  du),
        (1-dz)*(  da)*(1-du),
        (1-dz)*(  da)*(  du),
        (  dz)*(1-da)*(1-du),
        (  dz)*(1-da)*(  du),
        (  dz)*(  da)*(1-du),
        (  dz)*(  da)*(  du),
    ])  # (8,)
    slices = jnp.stack([
        cube[..., z1,   a1,   u1  ],
        cube[..., z1,   a1,   u1+1],
        cube[..., z1,   a1+1, u1  ],
        cube[..., z1,   a1+1, u1+1],
        cube[..., z1+1, a1,   u1  ],
        cube[..., z1+1, a1,   u1+1],
        cube[..., z1+1, a1+1, u1  ],
        cube[..., z1+1, a1+1, u1+1],
    ], axis=0)  # (8, ..., nspec or nlines)
    return jnp.sum(w.reshape((8,) + (1,)*(slices.ndim-1)) * slices, axis=0)



class NebularModel:
    def __init__(self, cloudy_dust, sps_home, csp_lambda,
                 smooth_velocity=True, isoc_type = 'mist', nebnz=11, nebnage=10, nebnip=7, nebular_smooth_init=0.0):
        """
        Initializes a nebular emission model for use with stellar population synthesis output.

        Parameters
        ----------
        isoc_type : str
            Isochrone type (e.g., 'mist', 'pdva', 'bpss') used to identify the correct nebular files.
        cloudy_dust : bool
            If True, uses nebular models with dust attenuation from CLOUDY.
        sps_home : str or Path
            Path to the FSPS root directory containing the nebular subdirectory.
        csp_lambda : array-like
            Wavelength grid of the composite stellar population (CSP) spectrum [in Angstrom].

        smooth_velocity : bool, optional
            If True, smoothing is velocity-based (km/s). If False, use fixed smoothing in Å.
        nebnz : int, optional
            Number of metallicity grid points.
        nebnage : int, optional
            Number of age grid points.
        nebnip : int, optional
            Number of ionization parameter grid points.
        nebular_smooth_init : float, optional
            Initial smoothing width in km/s or Å (depending on `smooth_velocity`).
        """

        # Store CSP wavelength grid and spectrum length
        self.csp_lambda = jnp.asarray(csp_lambda)
        self.nspec = self.csp_lambda.size

        # Store constants and hyperparameters
        self.smooth_velocity = smooth_velocity
        self.nebular_smooth_init = nebular_smooth_init
        self.nebnz = nebnz
        self.nebnage = nebnage
        self.nebnip = nebnip

        # Set filenames for nebular continuum and line emission based on isochrone type and dusty flag
        suffix = 'WD' if cloudy_dust else 'ND'
        base = Path(sps_home) / 'nebular' / f'ZAU_{suffix}_{isoc_type}'
        self.cont_file = base.with_suffix('.cont')   # Nebular continuum file
        self.line_file = base.with_suffix('.lines')  # Nebular emission line file

        # Load the nebular continuum cube and grids
        self._load_continuum()

        # Load the nebular emission line cube and line positions
        self._load_lines()

        # Compute the minimum spectral resolution for each line (in Å)
        self._compute_resolution_elements_fsps()

        # Build Gaussian line profiles for all emission lines on the CSP wavelength grid
        self._build_gaussians()
    
    def _load_continuum(self): 
        # Open the nebular continuum file and skip the header
        with open(self.cont_file, 'r') as f:
            f.readline()  # skip header line

            # Read the wavelength grid of the nebular continuum model
            readlambneb = np.array([float(x) for x in f.readline().split()])
            nlam = len(readlambneb)
            self.readlambneb = jnp.asarray(readlambneb)

            # Read the rest of the file content into memory
            lines = f.readlines()

            # Allocate arrays for interpolated continuum and parameter grids
            cont_cube = np.zeros((self.nspec, self.nebnz, self.nebnage, self.nebnip))
            logz_grid = np.zeros(self.nebnz)   # gas metallicity
            logu_grid = np.zeros(self.nebnip)  # ionization parameter
            age_grid = np.zeros(self.nebnage)  # age in yr

            idx = 0  # line index in the file
            for i in range(self.nebnz):
                for j in range(self.nebnage):
                    for k in range(self.nebnip):
                        # First line gives (logZ, age, logU) for this cube slice
                        parts = [float(x) for x in lines[idx].split()]
                        logz_grid[i], age_grid[j], logu_grid[k] = parts[:3]
                        idx += 1

                        # Second line gives the continuum spectrum (Lsun/Hz/Q)
                        cont = np.array([float(x) for x in lines[idx].split()])
                        logcont = np.log10(cont + 10**-95)  # avoid log(0)
                        # Interpolate onto CSP wavelength grid
                        cont_interp = jnp.interp(self.csp_lambda, readlambneb, logcont)
                        cont_cube[:, i, j, k] = cont_interp
                        idx += 1

        # Store as JAX arrays for GPU compatibility
        self.nebem_cont = jnp.asarray(cont_cube)
        self.nebem_logz = jnp.asarray(logz_grid)
        self.nebem_logu = jnp.asarray(logu_grid)
        self.nebem_age = jnp.log10(jnp.asarray(age_grid))  # log10(age/yr)
    
    def _load_lines(self):
        # Open the nebular emission line file
        with open(self.line_file, 'r') as f:
            f.readline()  # Skip header line

            # Read the rest-frame line center wavelengths (in Å)
            self.nebem_line_pos = jnp.array([float(x) for x in f.readline().split()])
            self.nemline = self.nebem_line_pos.shape[0]

            # Allocate array to store log10(line luminosities) in Lsun/Q
            line_cube = np.zeros((self.nemline, self.nebnz, self.nebnage, self.nebnip))

            # Loop over metallicity, age, and ionization parameter grid
            for i in range(self.nebnz):
                for j in range(self.nebnage):
                    for k in range(self.nebnip):
                        f.readline()  # metadata line (logZ, age, logU), already known
                        linevals = np.array([float(x) for x in f.readline().split()])
                        # Store log10(L/Q), avoiding log(0)
                        line_cube[:, i, j, k] = np.log10(linevals + 10**-95)

            # Store line emission cube as a JAX array for GPU use
            self.nebem_line = jnp.asarray(line_cube)

    def _compute_resolution_elements(self):
        '''Avoid delta function spikes in nebular lines when the broadening of the gaussian is smaller than the resolution element.'''
        # For each emission line, find the nearest wavelength index just below the line center
        idx = jnp.clip(
            jnp.searchsorted(self.csp_lambda, self.nebem_line_pos) - 1,
            1, self.nspec - 2 
        )
        self.neb_res_min = self.csp_lambda[idx + 1] - self.csp_lambda[idx]
        # Estimate the local spectral resolution element Δλ around each line
        dlam_pre = self.csp_lambda[idx + 1] - self.csp_lambda[idx]
        dlam = dlam_pre*2/2.355
        neb_res_min = jnp.maximum(dlam, self.neb_res_min*self.csp_lambda[idx]/const.c * 1e13)#sigma of the line angs
        # Store the minimum resolution per line as a JAX array
        self.neb_res_min = jnp.asarray(neb_res_min)

    def _compute_resolution_elements_fsps(self):
        '''Avoid delta function spikes in nebular lines when the broadening of the gaussian is smaller than the resolution element.'''
        # For each emission line, find the nearest wavelength index just below the line center
        idx = jnp.clip(jnp.searchsorted(self.csp_lambda, self.nebem_line_pos) - 1, 1, self.nspec - 2)
        self.neb_res_min = self.csp_lambda[idx + 1] - self.csp_lambda[idx]

    def _build_gaussians(self):
        """
        Precompute normalized Gaussian line profiles for each emission line,
        on the given wavelength grid. These profiles model the line broadening
        (either in velocity space or wavelength space), and are stored for
        later use when adding line emission to the SED.
        """

        def compute_line(i):
            lam0 = self.nebem_line_pos[i]
            dlam = jnp.where(
                self.smooth_velocity,
                lam0 * self.nebular_smooth_init / (const.c *10**10)* 1e13,  # in Angstrom
                self.nebular_smooth_init
            )
            dlam = jnp.maximum(dlam, self.neb_res_min[i] * 2.0)

            norm = 1.0 / (jnp.sqrt(2 * jnp.pi) * dlam)
            profile = jnp.exp(-0.5 * ((self.csp_lambda - lam0) / dlam) ** 2)
            scaling = lam0 ** 2 / (const.c * 10**10)

            return norm * profile * scaling  # shape: (n_lambda,)

        # Vectorize over all emission lines
        self.gaussnebarr = jnp.stack([compute_line(i) for i in range(self.nebem_line_pos.shape[0])], axis=1)
        return self.gaussnebarr  # shape: (n_lambda, n_lines)

    def evaluate_batch(self, logZ_gas, logU, ssp_ages_young, logqq_young):
        """
        Vectorized nebular evaluation for all (Z_ssp, age_young) pairs at a
        single (logZ_gas, logU) point.

        Replaces the nested ``vmap(vmap(evaluate))`` pattern with factored
        array operations.  The trilinear interpolation on the CLOUDY cube is
        decomposed as:

        1. **Bilinear in (logZ, logU)** — done once, collapses two grid dims.
        2. **Linear in age** — vectorized over all young ages simultaneously.
        3. **Exponentiate + line projection** — batched over (Z_ssp, age).

        The result is *numerically identical* to calling ``evaluate()`` in a
        nested vmap, but avoids vmap dispatch overhead and is more
        cache-friendly on GPU.

        Parameters
        ----------
        logZ_gas : scalar or shape-(1,) array
            Gas-phase log10(Z/Zsun).
        logU : scalar or shape-(1,) array
            Ionisation parameter log10(U).
        ssp_ages_young : array, shape (n_young,)
            log10(age/yr) for the young SSP ages.
        logqq_young : array, shape (n_z_ssp, n_young)
            log10(Q(H0)) for each SSP metallicity and young age.

        Returns
        -------
        neb_young : array, shape (n_z_ssp, n_young, nspec)
            Combined (continuum + line) nebular spectrum.
        """
        logZ_gas = jnp.squeeze(logZ_gas)
        logU     = jnp.squeeze(logU)

        # ── Step 1: locate (logZ, logU) in the CLOUDY grid (scalar) ──────
        z1 = _locate(logZ_gas, self.nebem_logz)
        dz = jnp.clip(
            (logZ_gas - self.nebem_logz[z1])
            / (self.nebem_logz[z1 + 1] - self.nebem_logz[z1]),
            0.0, 1.0,
        )
        u1 = _locate(logU, self.nebem_logu)
        du = jnp.clip(
            (logU - self.nebem_logu[u1])
            / (self.nebem_logu[u1 + 1] - self.nebem_logu[u1]),
            0.0, 1.0,
        )

        # ── Step 2: bilinear in (Z, U) — collapse two grid dims ─────────
        # nebem_cont shape: (nspec, nz, nage_cloudy, nu)
        # After bilinear:   (nspec, nage_cloudy)
        wzu = jnp.array([
            (1 - dz) * (1 - du),
            (1 - dz) *       du,
                  dz  * (1 - du),
                  dz  *       du,
        ])  # (4,)

        cont_zu = (wzu[0] * self.nebem_cont[:, z1,     :, u1    ] +
                   wzu[1] * self.nebem_cont[:, z1,     :, u1 + 1] +
                   wzu[2] * self.nebem_cont[:, z1 + 1, :, u1    ] +
                   wzu[3] * self.nebem_cont[:, z1 + 1, :, u1 + 1])
        # shape: (nspec, nage_cloudy)

        line_zu = (wzu[0] * self.nebem_line[:, z1,     :, u1    ] +
                   wzu[1] * self.nebem_line[:, z1,     :, u1 + 1] +
                   wzu[2] * self.nebem_line[:, z1 + 1, :, u1    ] +
                   wzu[3] * self.nebem_line[:, z1 + 1, :, u1 + 1])
        # shape: (nlines, nage_cloudy)

        # ── Step 3: linear in age — vectorized over young ages ───────────
        a1 = jnp.clip(
            jnp.searchsorted(self.nebem_age, ssp_ages_young) - 1,
            0, self.nebem_age.shape[0] - 2,
        )  # (n_young,)
        da = jnp.clip(
            (ssp_ages_young - self.nebem_age[a1])
            / (self.nebem_age[a1 + 1] - self.nebem_age[a1]),
            0.0, 1.0,
        )  # (n_young,)

        # cont_zu[:, a1] gathers age columns → (nspec, n_young)
        logcont = (1 - da)[None, :] * cont_zu[:, a1] + da[None, :] * cont_zu[:, a1 + 1]
        logline = (1 - da)[None, :] * line_zu[:, a1] + da[None, :] * line_zu[:, a1 + 1]
        # shapes: (nspec, n_young), (nlines, n_young)

        # ── Step 4: add logQ and exponentiate ────────────────────────────
        # logqq_young shape: (n_z_ssp, n_young)
        # Broadcasting: (nspec, n_young) + (n_z_ssp, 1, n_young) → (n_z_ssp, nspec, n_young)
        cont_flux = 10.0 ** (logcont[None, :, :] + logqq_young[:, None, :])
        line_lum  = 10.0 ** (logline[None, :, :] + logqq_young[:, None, :])
        # shapes: (n_z_ssp, nspec, n_young), (n_z_ssp, nlines, n_young)

        # ── Step 5: project lines onto wavelength grid ───────────────────
        # gaussnebarr: (nspec, nlines)
        # line_lum:    (n_z_ssp, nlines, n_young)
        line_spec = jnp.einsum('wl,zly->zwy', self.gaussnebarr, line_lum)
        # shape: (n_z_ssp, nspec, n_young)

        # ── Step 6: combine and transpose to (n_z_ssp, n_young, nspec) ───
        neb_total = cont_flux + line_spec
        return neb_total.transpose(0, 2, 1)

    def get_default_params(self):
        """
        Return a plain dict of default nebular fit parameters.

        Defaults match FSPS: gas_logz=0.0 (solar), gas_logu=-2.0 (moderate
        ionization).  Previously returned a NamedTuple; now returns a dict so
        that it merges directly into the global theta dict without ``._asdict()``.
        """
        return {
            'gas_logz': jnp.asarray(0.0),
            'gas_logu': jnp.asarray(-2.0),
        }

    def get_param_names(self):
        """Return a list of fittable nebular parameter names."""
        return ['gas_logz', 'gas_logu']

    def xxevaluate(self, logZ, logU, logage, logQ):

        # Locate and compute fractional offsets in grid
        z1 = locate(logZ, self.nebem_logz)
        dz = jnp.clip((logZ - self.nebem_logz[z1]) / (self.nebem_logz[z1 + 1] - self.nebem_logz[z1]), 0.0, 1.0)

        u1 = locate(logU, self.nebem_logu)
        du = jnp.clip((logU - self.nebem_logu[u1]) / (self.nebem_logu[u1 + 1] - self.nebem_logu[u1]), 0.0, 1.0)

        a1 = locate(logage, self.nebem_age)
        da = jnp.clip((logage - self.nebem_age[a1]) / (self.nebem_age[a1 + 1] - self.nebem_age[a1]), 0.0, 1.0)
        
        # Interpolate log-continuum and line luminosity
        logcont_interp = trilinear_interp(self.nebem_cont, z1, dz, a1, da, u1, du)
        loglines_interp = trilinear_interp(self.nebem_line, z1, dz, a1, da, u1, du)

        # Combine with logQ
        log_cont_flux = logcont_interp + logQ
        log_line_flux = loglines_interp + logQ
        
        # Convert to linear only where absolutely necessary
        cont_flux = 10 ** log_cont_flux  # shape (nspec,)
        line_flux = 10 ** log_line_flux  # shape (nlines,)
        
        # Combine lines
        line_spec = jnp.dot(self.gaussnebarr, line_flux)   # shape (nspec,)

        return (cont_flux, line_spec)
    
    
    def evaluate(self, logZ, logU, logage, logQ):
        """
        Interpolate the nebular grid and return continuum + line spectra.

        Parameters
        ----------
        logZ : scalar
            log10(Z/Z_sun) of the gas.
        logU : scalar
            Ionisation parameter log10(U).
        logage : scalar
            log10(age / yr).
        logQ : scalar
            log10(Q(H0)) in photons/s.

        Returns
        -------
        cont_flux : array (nspec,)   — nebular continuum in Lsun/Hz
        line_spec : array (nspec,)   — emission lines in Lsun/Hz
        line_lum  : array (nlines,)  — total luminosity per line in Lsun
        """
        # Locate grid cells and fractional offsets (no extrapolation)
        z1 = _locate(logZ, self.nebem_logz)
        dz = jnp.clip(
            (logZ - self.nebem_logz[z1])
            / (self.nebem_logz[z1 + 1] - self.nebem_logz[z1]),
            0.0, 1.0,
        )
        u1 = _locate(logU, self.nebem_logu)
        du = jnp.clip(
            (logU - self.nebem_logu[u1])
            / (self.nebem_logu[u1 + 1] - self.nebem_logu[u1]),
            0.0, 1.0,
        )
        a1 = _locate(logage, self.nebem_age)
        da = jnp.clip(
            (logage - self.nebem_age[a1])
            / (self.nebem_age[a1 + 1] - self.nebem_age[a1]),
            0.0, 1.0,
        )

        # Trilinear interpolation in log-space
        logcont_interp = _trilinear(self.nebem_cont, z1, dz, a1, da, u1, du)  # (nspec,)
        logline_interp = _trilinear(self.nebem_line, z1, dz, a1, da, u1, du)  # (nlines,)

        # Scale by Q — matches FSPS: 10**tmpnebcont * qq, 10**tmpnebline * qq
        cont_flux = 10.0 ** (logcont_interp + logQ)    # Lsun/Hz, shape (nspec,)
        line_lum  = 10.0 ** (logline_interp + logQ)    # Lsun,    shape (nlines,)

        # Distribute line luminosity onto wavelength grid as L_ν Gaussians
        line_spec = self.gaussnebarr @ line_lum         # (nspec,)

        return cont_flux, line_spec#, line_lum

def locate(x, grid):
    return jnp.clip(jnp.searchsorted(grid, x) - 1, 0, grid.size - 2)   

def trilinear_interp(cube, z1, dz, a1, da, u1, du):
    """
    Trilinear interpolation on a 4D cube: shape (nspec, nz, nage, nu)
    """
    # Interpolation weights for the 8 surrounding points
    w = jnp.array([
        (1 - dz) * (1 - da) * (1 - du),
        (1 - dz) * (1 - da) * du,
        (1 - dz) * da * (1 - du),
        (1 - dz) * da * du,
        dz * (1 - da) * (1 - du),
        dz * (1 - da) * du,
        dz * da * (1 - du),
        dz * da * du
    ])

    # Stack the 8 corners along a new axis (0), each of shape (nspec,)
    slices = jnp.stack([
        cube[:, z1,   a1,   u1],
        cube[:, z1,   a1,   u1+1],
        cube[:, z1,   a1+1, u1],
        cube[:, z1,   a1+1, u1+1],
        cube[:, z1+1, a1,   u1],
        cube[:, z1+1, a1,   u1+1],
        cube[:, z1+1, a1+1, u1],
        cube[:, z1+1, a1+1, u1+1]
    ], axis=0)  # shape (8, nspec)

    # Weighted sum over the 8 corners
    return jnp.sum(w[:, None] * slices, axis=0)  # shape (nspec,)


