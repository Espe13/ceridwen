import fsps
import matplotlib.pyplot as plt
from ceridwen.ssps.ssp_data import SSPData
import jax.numpy as jnp

ssp_data = SSPData.load('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/ssp_data.h5')

sp = fsps.StellarPopulation(zcontinuous=0, zmet=1, sfh =0, add_neb_emission=False, add_neb_continuum=False)


sp.params['add_neb_emission'] = False
sp.params['add_neb_continuum'] = False

wave, fsps_spec_no_neb = sp.get_spectrum(tage=0.001)

sp.params['add_neb_emission'] = True
sp.params['add_neb_continuum'] = True
sp.params['gas_logz'] = 0
sp.params['gas_logu'] = -4
wave, fsps_spec_with_neb = sp.get_spectrum(tage=0.001)


sp.params['add_neb_emission'] = True
sp.params['add_neb_continuum'] = False
wave, fsps_spec_only_lines = sp.get_spectrum(tage=0.001)


fsps_spec_only_cont = fsps_spec_with_neb - fsps_spec_only_lines



age_idx = 20  # Corresponds to 1 Myr
met_idx = 0

ssp_flux = ssp_data.ssp_flux[met_idx,  age_idx,:]
residuals = (fsps_spec_no_neb - ssp_flux) / ssp_flux
# Create two vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10/3, 3.6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

# Top plot: Spectra
ax1.loglog(wave, fsps_spec_no_neb, color='black', label='FSPS no nebular emission')
ax1.loglog(wave, ssp_flux, color='red', ls='dashed', label='SSP')
#ax1.set_ylim(1e-17, 1e-12)
#ax1.set_xlim(1e2, 1e5)
ax1.legend()
ax1.set_ylabel("Flux")

# Bottom plot: Normalized residuals
ax2.plot(wave, residuals, color='red')
ax2.axhline(0, color='gray', lw=1, ls='--')
ax2.set_ylabel("Residuals")
ax2.set_xlabel("Wavelength")
ax2.set_yscale('symlog')

plt.show()

print(jnp.sum(residuals))

print("Residual Sum:", jnp.sum(residuals), 'the ssp and the fsps are the same!')


# My neb model:
import jax.numpy as jnp
from pathlib import Path
from jax import vmap, jit
#from jax.scipy.interpolate import interp1d
import numpy as np
import astropy.constants as const
import math



from pathlib import Path
import jax.numpy as jnp

class NebularModel:
    def __init__(self,sps_home, csp_lambda, cloudy_dust = False,  isoc_type='mist',
                smooth_velocity=True, nebnz=11, nebnage=10, nebnip=7, nebular_smooth_init=100.0):
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
.
        jnp.pi : float
            Value of π (can be jnp.pi or float).
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
        self._compute_resolution_elements()

        # Build Gaussian line profiles for all emission lines on the CSP wavelength grid
        self._build_gaussians()

    def __repr__(self):
        def _shape(x):
            try:
                return tuple(jnp.asarray(x).shape)
            except Exception:
                try:
                    return tuple(np.asarray(x).shape)
                except Exception:
                    return "n/a"

        def _minmax(x):
            try:
                xa = jnp.asarray(x)
                return f"{float(jnp.min(xa)):.4g} – {float(jnp.max(xa)):.4g}"
            except Exception:
                return "n/a"

        lines = []
        lines.append("<NebularModel>")
        lines.append("-----------------------------------")
        # Config
        lines.append(f"Files:")
        lines.append(f"  Continuum: {str(self.cont_file)}")
        lines.append(f"  Lines    : {str(self.line_file)}")
        lines.append("")
        lines.append("Grids & settings:")
        lines.append(f"  CSP λ grid shape        : {_shape(self.csp_lambda)}  (nspec={getattr(self, 'nspec', 'n/a')})")
        lines.append(f"  smooth_velocity         : {self.smooth_velocity}")
        lines.append(f"  nebular_smooth_init     : {self.nebular_smooth_init}")
        lines.append(f"  nebnz / nebnage / nebnip: {getattr(self, 'nebnz', 'n/a')} / {getattr(self, 'nebnage', 'n/a')} / {getattr(self, 'nebnip', 'n/a')}")
        lines.append("")
        # Continuum cube + parameter grids
        lines.append("Continuum cube (log10 L/Q):")
        lines.append(f"  nebem_cont shape        : {_shape(getattr(self, 'nebem_cont', None))}  (nspec, nz, nage, nu)")
        lines.append(f"  nebem_logz shape/range  : {_shape(getattr(self, 'nebem_logz', None))} ; {_minmax(getattr(self, 'nebem_logz', None))}")
        lines.append(f"  nebem_age  shape/range  : {_shape(getattr(self, 'nebem_age', None))}  ; {_minmax(getattr(self, 'nebem_age', None))}  (log10 yr)")
        lines.append(f"  nebem_logu shape/range  : {_shape(getattr(self, 'nebem_logu', None))} ; {_minmax(getattr(self, 'nebem_logu', None))}")
        lines.append("")
        # Lines cube
        lines.append("Emission lines (log10 L/Q):")
        lines.append(f"  line positions shape    : {_shape(getattr(self, 'nebem_line_pos', None))}  (Å)")
        lines.append(f"  nebem_line shape        : {_shape(getattr(self, 'nebem_line', None))}  (nlines, nz, nage, nu)")
        lines.append(f"  neb_res_min shape/range : {_shape(getattr(self, 'neb_res_min', None))} ; {_minmax(getattr(self, 'neb_res_min', None))}  (Å)")
        lines.append("")
        # Precomputed Gaussians
        lines.append("Precomputed line profiles:")
        lines.append(f"  gaussnebarr shape       : {_shape(getattr(self, 'gaussnebarr', None))}  (nspec, nlines)")
        lines.append("")
        # Last evaluation (if present)
        lines.append("Last evaluation buffers (optional):")
        lines.append(f"  conflux shape           : {_shape(getattr(self, 'conflux', None))}")
        lines.append(f"  lineflux shape          : {_shape(getattr(self, 'lineflux', None))}")
        lines.append(f"  linespec shape          : {_shape(getattr(self, 'linespec', None))}")

        return "\n".join(lines)

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

    def evaluate(self, logZ, logU, logage, logQ):

        # Locate and compute fractional offsets in grid
        z1 = locate(logZ, self.nebem_logz)
        dz = jnp.clip((logZ - self.nebem_logz[z1]) / (self.nebem_logz[z1 + 1] - self.nebem_logz[z1]), 0.0, 1.0)

        u1 = locate(logU, self.nebem_logu)
        du = jnp.clip((logU - self.nebem_logu[u1]) / (self.nebem_logu[u1 + 1] - self.nebem_logu[u1]), 0.0, 1.0)

        a1 = locate(logage, self.nebem_age)
        da = jnp.clip((logage - self.nebem_age[a1]) / (self.nebem_age[a1 + 1] - self.nebem_age[a1]), 0.0, 1.0)

        # Interpolate log-continuum and line luminosity
        logcont_interp = trilinear(self.nebem_cont, z1, dz, a1, da, u1, du)
        loglines_interp = trilinear(self.nebem_line, z1, dz, a1, da, u1, du)

        # Combine with logQ
        log_cont_flux = logcont_interp + logQ
        log_line_flux = loglines_interp + logQ
        
        # Convert to linear only where absolutely necessary
        cont_flux = 10 ** log_cont_flux  # shape (nspec,)
        line_flux = 10 ** log_line_flux  # shape (nlines,)
        
        # Combine lines
        line_spec = jnp.dot(self.gaussnebarr, line_flux)   # shape (nspec,)
        self.linespec = line_spec  # Store for later use if needed
        self.conflux = cont_flux  # Store for later use if needed
        self.lineflux = line_flux  # Store for later use if needed


        return cont_flux, line_spec

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


def trilinear(cube, z1, dz, a1, da, u1, du):
    return (
        (1 - dz) * (1 - da) * (1 - du) * cube[:, z1,   a1,   u1  ] +
        (1 - dz) * (1 - da) * (    du) * cube[:, z1,   a1,   u1+1] +
        (1 - dz) * (    da) * (1 - du) * cube[:, z1,   a1+1, u1  ] +
        (1 - dz) * (    da) * (    du) * cube[:, z1,   a1+1, u1+1] +
        (    dz) * (1 - da) * (1 - du) * cube[:, z1+1, a1,   u1  ] +
        (    dz) * (1 - da) * (    du) * cube[:, z1+1, a1,   u1+1] +
        (    dz) * (    da) * (1 - du) * cube[:, z1+1, a1+1, u1  ] +
        (    dz) * (    da) * (    du) * cube[:, z1+1, a1+1, u1+1]
    )



model = NebularModel(
    sps_home="/Users/amanda/Prospector/fsps",
    csp_lambda=wave,
    nebular_smooth_init=0,
    smooth_velocity=True,
    cloudy_dust=False
)

cont, line = model.evaluate(
    logZ=sp.params['gas_logz'],
    logU=sp.params['gas_logu'],
    logage=ssp_data.ssp_lg_age_gyr[age_idx] + 9,  # Example age in log years
    logQ=ssp_data.log_qq[met_idx, age_idx]  # Example ionizing photon rate
)

spec = cont + line


mod_age_idx = jnp.searchsorted(model.nebem_age, ssp_data.ssp_lg_age_gyr[age_idx] + 9)
mod_met_idx = jnp.searchsorted(model.nebem_logz, sp.params['gas_logz'])
mod_logu_idx = jnp.searchsorted(model.nebem_logu, sp.params['gas_logu'])



ssp_flux = np.array(ssp_data.ssp_flux[met_idx, age_idx, :].copy())

# Zero out ionizing photon flux (below Lyman limit)
ssp_flux[wave < 912] = 0.0

csp_cont = cont + ssp_flux
csp_lines = line + ssp_flux


direct = 10**(model.nebem_cont[:, mod_met_idx, mod_age_idx, mod_logu_idx]+ssp_data.log_qq[met_idx, age_idx])



# COmpare to FSPS:


csp_flux = spec + ssp_flux

comparisons = [
    ("continuum", direct, fsps_spec_only_cont, "darkorange"),
    ("lines", csp_lines, fsps_spec_only_lines, "mediumseagreen"),
    ("ssp_flux", np.array(ssp_data.ssp_flux[met_idx, age_idx, :].copy()), fsps_spec_no_neb, "royalblue"),
    ("composite", csp_flux, fsps_spec_with_neb, "crimson")
]


fig, axes = plt.subplots(
    2, len(comparisons), figsize=(18, 6),
    sharex='col', gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15, 'wspace': 0.25}
)

for i, (label, model_spec, fsps_spec, color) in enumerate(comparisons):
    # --- top: spectra ---
    ax1 = axes[0, i]
    ax1.loglog(wave, model_spec, color='dodgerblue', label=label)
    ax1.loglog(wave, fsps_spec, color='crimson', ls='--', label='FSPS')
    ax1.set_ylim(1e-14, 1e-11)
    #ax1.set_xlim(5e3, 1e4)
    ax1.set_title(label.replace('_', ' '))
    if i == 0:
        ax1.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
    if i == 3:
        ax1.legend(frameon=False, fontsize=8, loc='lower right')

    # --- bottom: normalized residuals ---
    ax2 = axes[1, i]
    residual = (model_spec - fsps_spec) / fsps_spec
    ax2.plot(wave, residual, color='black', lw=0.8)
    ax2.axhline(0, color='gray', lw=1, ls='--')
    ax2.set_yscale('symlog')
    if i == 0:
        ax2.set_ylabel('Norm. residuals')
    ax2.set_xlabel(r'Wavelength $\lambda$ [Å]')



plt.tight_layout()
plt.savefig('/Users/amanda/Desktop/PhD/Tools/ceridwen/notebooks/neb/test_neb_comparison.png', dpi=150)
plt.close()

# Quantitative diagnostics
print("\n=== Residual Diagnostics ===")
for label, model_spec, fsps_spec, color in comparisons:
    model_arr = np.array(model_spec)
    fsps_arr = np.array(fsps_spec)
    mask = fsps_arr != 0
    rel_residual = (model_arr[mask] - fsps_arr[mask]) / fsps_arr[mask]

    # Also check specific wavelength ranges
    optical = (wave > 3000) & (wave < 10000) & mask
    rel_opt = (model_arr[optical] - fsps_arr[optical]) / fsps_arr[optical]

    print(f"\n{label}:")
    print(f"  Full range:  max|residual| = {np.max(np.abs(rel_residual)):.6e},  median = {np.median(np.abs(rel_residual)):.6e}")
    if np.sum(optical) > 0:
        print(f"  Optical:     max|residual| = {np.max(np.abs(rel_opt)):.6e},  median = {np.median(np.abs(rel_opt)):.6e}")

    # Find wavelengths with worst residuals in optical
    if np.sum(optical) > 0:
        worst_opt_idx = np.argsort(np.abs(rel_opt))[-10:]
        wave_opt = wave[optical]
        print(f"  Worst 10 optical wavelengths: {wave_opt[worst_opt_idx]}")
        print(f"  Worst 10 optical residuals:   {rel_opt[worst_opt_idx]}")

# Compare neb_res_min: check if model matches FSPS approach
# FSPS uses spec_lambda(j+1) - spec_lambda(j) where j = locate(spec_lambda, line_pos)
print("\n=== neb_res_min check (first 20 lines) ===")
for i in range(min(20, len(model.nebem_line_pos))):
    lpos = float(model.nebem_line_pos[i])
    res = float(model.neb_res_min[i])
    dlam = res * 2  # actual width used
    print(f"  line {i}: lam={lpos:.2f} A, neb_res_min={res:.4f} A, dlam_used={dlam:.4f} A")

# Check what Q values are being used
print(f"\nlog_qq from ssp_data: {float(ssp_data.log_qq[met_idx, age_idx]):.6f}")

# Compare Q: compute FSPS's Q from the spectrum directly
hplank = 6.6261e-27  # erg s
lsun = 3.839e33      # erg/s
clight_cgs = 2.9979e18  # Angstrom/s
spec_nu = clight_cgs / wave  # Hz
whlylim = np.searchsorted(wave, 912.0)
# Trapezoidal integration of photon number flux below Lyman limit
integrand = fsps_spec_no_neb[:whlylim] / spec_nu[:whlylim]  # Lsun/Hz / Hz = Lsun/Hz^2...
# Actually FSPS does: tsum(spec_nu, F_nu/nu) which integrates over frequency
# qq = integral(F_nu/nu dnu) / h * Lsun  where F_nu is in Lsun/Hz
fsps_qq = np.trapz(fsps_spec_no_neb[:whlylim] / spec_nu[:whlylim], spec_nu[:whlylim]) / hplank * lsun
# Note: spec_nu is decreasing, so integral is negative; take abs
fsps_qq = abs(fsps_qq)
print(f"FSPS Q (recomputed from spectrum): log10(Q) = {np.log10(fsps_qq):.6f}")
print(f"Stored log_qq:                     log10(Q) = {float(ssp_data.log_qq[met_idx, age_idx]):.6f}")
print(f"Q ratio (stored/recomputed):       {10**float(ssp_data.log_qq[met_idx, age_idx]) / fsps_qq:.6f}")

# Direct comparison of emission line luminosities with FSPS
print("\n=== FSPS emline comparison ===")
fsps_emline_wave = sp.emline_wavelengths  # line wavelengths
fsps_emline_lum = sp.emline_luminosity    # line luminosities at current params/tage
print(f"FSPS emline count: {len(fsps_emline_wave)}")
print(f"Model emline count: {len(model.nebem_line_pos)}")

# Match lines by wavelength and compare luminosities
model_line_lum = np.array(model.lineflux)  # line luminosities from evaluate()
print(f"\nModel line_flux (from evaluate): shape={model_line_lum.shape}")
print(f"FSPS emline_lum: shape={fsps_emline_lum.shape}")

# Sort FSPS lines by luminosity descending, show top 20 optical
print(f"\nLine-by-line comparison (top 20 strongest optical lines):")
optical_mask = (fsps_emline_wave > 3000) & (fsps_emline_wave < 10000)
opt_idx = np.where(optical_mask)[0]
opt_sorted = opt_idx[np.argsort(fsps_emline_lum[opt_idx])[::-1]]
for j in opt_sorted[:20]:
    fw = fsps_emline_wave[j]
    fl = fsps_emline_lum[j]
    if fl <= 0:
        continue
    diffs = np.abs(np.array(model.nebem_line_pos) - fw)
    mi = np.argmin(diffs)
    if diffs[mi] < 2.0:
        ml = float(model_line_lum[mi])
        ratio = ml / fl if fl > 0 else float('nan')
        print(f"  {fw:.2f} A: FSPS={fl:.4e}, model={ml:.4e}, ratio={ratio:.6f}, lum_per_Q_ratio={ratio/0.996851:.6f}")

# Check age interpolation
logage_used = float(ssp_data.ssp_lg_age_gyr[age_idx] + 9)
print(f"\n=== Age grid check ===")
print(f"Age used: logage = {logage_used:.6f} (log10 yr)")
print(f"nebem_age grid: {np.array(model.nebem_age)}")
a1 = int(locate(logage_used, model.nebem_age))
da = (logage_used - float(model.nebem_age[a1])) / (float(model.nebem_age[a1+1]) - float(model.nebem_age[a1]))
print(f"Age index a1 = {a1}, da = {da:.6f}")
print(f"  nebem_age[a1] = {float(model.nebem_age[a1]):.6f}")
print(f"  nebem_age[a1+1] = {float(model.nebem_age[a1+1]):.6f}")
print(f"  → Age is {'ON' if abs(da) < 1e-6 or abs(da - 1) < 1e-6 else 'BETWEEN'} grid points")

logZ_used = float(sp.params['gas_logz'])
z1 = int(locate(logZ_used, model.nebem_logz))
dz = (logZ_used - float(model.nebem_logz[z1])) / (float(model.nebem_logz[z1+1]) - float(model.nebem_logz[z1]))
print(f"\nZ index z1 = {z1}, dz = {dz:.6f}")
print(f"  → Z is {'ON' if abs(dz) < 1e-6 or abs(dz - 1) < 1e-6 else 'BETWEEN'} grid points")

logU_used = float(sp.params['gas_logu'])
u1 = int(locate(logU_used, model.nebem_logu))
du = (logU_used - float(model.nebem_logu[u1])) / (float(model.nebem_logu[u1+1]) - float(model.nebem_logu[u1]))
print(f"\nU index u1 = {u1}, du = {du:.6f}")
print(f"  → U is {'ON' if abs(du) < 1e-6 or abs(du - 1) < 1e-6 else 'BETWEEN'} grid points")


# =============================================================
# Energy budget analysis: does the difference matter for SED fitting?
# =============================================================
print("\n" + "=" * 70)
print("ENERGY BUDGET ANALYSIS: Does the difference matter for SED fitting?")
print("=" * 70)

clight_aa = 2.9979e18  # Angstrom/s

# Convert F_nu (Lsun/Hz) to F_lambda (Lsun/Angstrom) for integration
# F_lambda = F_nu * c / lambda^2
wave_arr = np.array(wave)
model_composite = np.array(csp_flux)
fsps_composite = np.array(fsps_spec_with_neb)
model_lines_only = np.array(line)
fsps_lines_only = np.array(fsps_spec_with_neb - fsps_spec_no_neb)  # nebular contribution only
model_cont_neb = np.array(cont)
fsps_cont_neb = np.array(fsps_spec_only_cont - fsps_spec_no_neb)  # nebular continuum only

# Define photometric band ranges (approximate rectangular filters)
bands = {
    'FUV (1300-1800)': (1300, 1800),
    'NUV (1800-2800)': (1800, 2800),
    'u (3000-4000)':   (3000, 4000),
    'g (4000-5500)':   (4000, 5500),
    'r (5500-7000)':   (5500, 7000),
    'i (7000-8500)':   (7000, 8500),
    'z (8500-10000)':  (8500, 10000),
    'Full (91-1e7)':   (91, 1e7),
}

# 1. Broadband flux differences (composite spectrum)
print("\n--- 1. Broadband flux difference (model vs FSPS, composite) ---")
print(f"{'Band':<22} {'FSPS flux':>12} {'Model flux':>12} {'Diff %':>8} {'Abs diff %':>10}")
for band_name, (lam_lo, lam_hi) in bands.items():
    mask_band = (wave_arr >= lam_lo) & (wave_arr <= lam_hi)
    if np.sum(mask_band) < 2:
        continue
    fsps_band = np.trapz(fsps_composite[mask_band], wave_arr[mask_band])
    model_band = np.trapz(model_composite[mask_band], wave_arr[mask_band])
    diff_pct = (model_band - fsps_band) / fsps_band * 100
    print(f"  {band_name:<20} {fsps_band:>12.4e} {model_band:>12.4e} {diff_pct:>+7.3f}% {abs(diff_pct):>9.3f}%")

# 2. What fraction of band flux comes from nebular emission?
print("\n--- 2. Nebular emission as fraction of total band flux ---")
print(f"{'Band':<22} {'Neb lines %':>12} {'Neb cont %':>12} {'Total neb %':>12}")
ssp_arr = np.array(ssp_data.ssp_flux[met_idx, age_idx, :].copy())
ssp_arr_zeroed = ssp_arr.copy()
ssp_arr_zeroed[wave_arr < 912] = 0.0

for band_name, (lam_lo, lam_hi) in bands.items():
    mask_band = (wave_arr >= lam_lo) & (wave_arr <= lam_hi)
    if np.sum(mask_band) < 2:
        continue
    total_flux = np.trapz(fsps_composite[mask_band], wave_arr[mask_band])
    neb_line_flux = np.trapz(np.array(fsps_spec_only_lines - fsps_spec_no_neb)[mask_band], wave_arr[mask_band])
    neb_cont_flux = np.trapz(fsps_cont_neb[mask_band], wave_arr[mask_band])
    if abs(total_flux) > 0:
        print(f"  {band_name:<20} {neb_line_flux/total_flux*100:>+11.3f}% {neb_cont_flux/total_flux*100:>+11.3f}% {(neb_line_flux+neb_cont_flux)/total_flux*100:>+11.3f}%")

# 3. Energy difference from nebular emission ONLY
print("\n--- 3. Nebular-only energy difference by band ---")
print(f"  (How much energy does the nebular model error contribute per band?)")
print(f"{'Band':<22} {'Line E diff %':>14} {'Cont E diff %':>14} {'Total neb diff %':>16} {'of total flux':>14}")
for band_name, (lam_lo, lam_hi) in bands.items():
    mask_band = (wave_arr >= lam_lo) & (wave_arr <= lam_hi)
    if np.sum(mask_band) < 2:
        continue
    # Total band flux from FSPS (for normalization)
    total_fsps = np.trapz(fsps_composite[mask_band], wave_arr[mask_band])

    # Line emission difference
    model_line_band = np.trapz(model_lines_only[mask_band], wave_arr[mask_band])
    fsps_line_band = np.trapz(np.array(fsps_spec_only_lines - fsps_spec_no_neb)[mask_band], wave_arr[mask_band])
    line_diff = model_line_band - fsps_line_band

    # Continuum emission difference
    model_cont_band = np.trapz(model_cont_neb[mask_band], wave_arr[mask_band])
    fsps_cont_band = np.trapz(fsps_cont_neb[mask_band], wave_arr[mask_band])
    cont_diff = model_cont_band - fsps_cont_band

    if abs(total_fsps) > 0:
        print(f"  {band_name:<20} {line_diff/total_fsps*100:>+13.4f}% {cont_diff/total_fsps*100:>+13.4f}% {(line_diff+cont_diff)/total_fsps*100:>+15.4f}% {'← of total band flux':>14}")

# 4. Context: typical SED fitting uncertainties
print("\n--- 4. Context for SED fitting ---")
print("  Typical photometric calibration uncertainty:  ~2-5%")
print("  Typical model uncertainty (SPS, IMF, etc.):   ~5-10%")
print("  Typical spectroscopic S/N per pixel:          ~5-50")

# Compute the RMS flux difference across bands
band_diffs = []
for band_name, (lam_lo, lam_hi) in bands.items():
    if band_name.startswith('Full'):
        continue
    mask_band = (wave_arr >= lam_lo) & (wave_arr <= lam_hi)
    if np.sum(mask_band) < 2:
        continue
    fsps_band = np.trapz(fsps_composite[mask_band], wave_arr[mask_band])
    model_band = np.trapz(model_composite[mask_band], wave_arr[mask_band])
    band_diffs.append((model_band - fsps_band) / fsps_band * 100)

band_diffs = np.array(band_diffs)
print(f"\n  Your model's RMS broadband difference:       {np.sqrt(np.mean(band_diffs**2)):.3f}%")
print(f"  Your model's max broadband difference:       {np.max(np.abs(band_diffs)):.3f}%")
print(f"\n  → If RMS << photometric uncertainty (~2-5%), the difference is negligible for SED fitting.")