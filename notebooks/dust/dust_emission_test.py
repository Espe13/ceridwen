import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import fsps
from tqdm import tqdm
from astropy import constants as const
tiny_number = 1e-70

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis, fnu2flam

T_UNIV = 13.8  # Gyr, age of the universe
N_T = 100     # number of time steps for SFH

def gaussian_burst(tau, center_tau, width_tau, amp=1.0):
    """Gaussian in lookback time, evaluated on tau grid (Gyr)."""
    return amp * jnp.exp(-0.5 * ((tau - center_tau) / width_tau)**2)

t = jnp.linspace(1e-2, T_UNIV, N_T)   # Gyr (avoid exactly 0)
tau = T_UNIV - t                      # lookback time (Gyr)

sfr_bimodal = (gaussian_burst(tau, 0.05, 0.03, 1.0) +
               gaussian_burst(tau, 11.0, 0.8, 0.7))
zh = jnp.full_like(t, 0.02)  # constant metallicity (Z=0.02)

ssp_data = SSPData.load('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/ssp_data.h5')



diffdust = 0.2
diffdust_index = -0.7
young_dust = 0
young_dust_index = -1




dust_params = {'tau_pow': young_dust,
                'alpha': -young_dust_index, 'tau_kc': 0.0, 'dust_index': 0.0,
                'diffuse_params': {'tau_kc': diffdust, 'dust_index': diffdust_index}}

print('dust parameters:', dust_params)

dust_params = {'tau_pow': young_dust,
                'alpha': -young_dust_index, 'tau_kc': 0.0, 'dust_index': 0.0,
                'diffuse_params': {'tau_kc': diffdust, 'dust_index': diffdust_index}}
emi_params = {'duste_qpah': 0.5, 
              'duste_umin': 1.0, 
              'duste_gamma': 0.01}
#-------------------------------
# CSP SPECTRUM WITH DUST
#-------------------------------
csp = CSPBasis(ssp_data, dusty=True, dust_params={'bin_edges': [(-jnp.inf, -1.97), (-1.97, 10)], 'laws': ['powerlaw', 'kriek_conroy']})
print(csp.dust.get_default_fit_params())
csp.add_sfh(sfr_bimodal, lookback_time=tau)
csp.add_zh(zh, lookback_time=tau)
emi_params = {'duste_qpah': 3.5,
              'duste_umin': 1.0,
              'duste_gamma': 0.01}
spec = csp.get_spectrum(dust_params=dust_params)
spec_dustfree = csp.get_spectrum_dustfree()


#-------------------------------
# DUST Attenuated spectra
#------------------------------

csp_attenuated = fnu2flam(csp.wave, spec*1e-23*3631)


import jax.numpy as jnp
import jax

import numpy as np
from pathlib import Path

class DustEmission:

    def __init__(self, duste_model="DL07",
                 dust_file=None, spec_lambda=None, **kwargs):
        """
        Initialize the DustEmission object with parameters for dust emission modeling.

        Parameters
        ----------
        duste_model : str
            Dust emission model to use: 'DL07' or 'THEMIS'.
        dust_file : str
            Path to the dust emission file (required).
        spec_lambda : ndarray
            Wavelength grid over which dust emission will be evaluated (required).
        kwargs : dict
            Optional keyword arguments to override default dust parameters.
            Supported: duste_qpah, duste_umin, duste_gamma
        """

        # Store model choice (e.g., 'DL07' or 'THEMIS')
        self.duste_model = duste_model

        # Dust parameter values
        self.duste_qpah = None
        self.duste_umin = None
        self.duste_gamma = None

        # Model grid arrays for allowed values of qPAH and Umin
        self.qpaharr = None
        self.uminarr = None


        # Placeholder for loaded dust emission spectra
        self.dustem2_dustem = None

        # File path and wavelength grid
        self.dust_file = None
        self.spec_lambda = None

        # Store any additional keyword arguments for later use
        self.dwargs = kwargs

        # Read in key dust parameters, allowing overrides via kwargs
        self.duste_qpah = kwargs.pop("duste_qpah", 3.5)
        self.duste_umin = kwargs.pop("duste_umin", 1.0)
        self.duste_gamma = kwargs.pop("duste_gamma", 0.01)

        # Set parameter grids based on selected dust model
        if self.duste_model == "DL07":
            self.qpaharr = jnp.array([0.47,1.12,1.77,2.50,3.19,3.90,4.58])
            self.uminarr = jnp.array([
                0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.8,1.0,1.2,1.5,2.0,
                2.5,3.0,4.0,5.0,7.0,8.0,12.0,15.0,20.0,25.0
            ])
        elif self.duste_model == "THEMIS":
            # THEMIS model uses smaller qPAH values rescaled to percent
            self.qpaharr = jnp.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24,
                                      0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
            self.uminarr = jnp.array([
                0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
                0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
                6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
                35.0, 40.0, 50.0, 80.0
            ])
        else:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        # Ensure that necessary data is provided
        if dust_file is None or spec_lambda is None:
            raise ValueError("If `duste=True`, both `dust_file` and `spec_lambda` must be provided.")

        self.dust_file = dust_file
        self.spec_lambda = spec_lambda

        # Load emission templates or model data from file
        self.load_dust_emission(dust_file, spec_lambda)

    def __repr__(self):
        """
        Custom string representation of the DustEmission object.
        Provides a readable summary of model settings and parameters.
        """

        def format_array(arr):
            """Helper to format short arrays inline; longer arrays multiline."""
            if arr is None:
                return "None"
            if arr.ndim == 1 and len(arr) <= 5:
                return f"[{', '.join(map(str, arr))}]"
            return f"\n    " + "\n    ".join(map(str, arr))

        attributes = {
            "Duste model": self.duste_model,
            "DUST qPAH": self.duste_qpah,
            "DUST Umin": self.duste_umin,
            "DUST Gamma": self.duste_gamma,
            f"qpaharr ({self.duste_model})": format_array(self.qpaharr),
            f"uminarr ({self.duste_model})": format_array(self.uminarr),
            "dust_file": self.dust_file,
            "spec_lambda": self.spec_lambda.shape if self.spec_lambda is not None else None,
        }

        if self.dwargs:
            attributes["Extra parameters (dwargs)"] = self.dwargs

        attr_str = "\n".join(f"  {k:<30}: {v}" for k, v in attributes.items() if v is not None)
        return f"\nDustEmission Model:\n{'='*50}\n{attr_str}\n{'='*50}"


    def load_dust_emission(self, dust_file=None, spec_lambda=None):
        
        # Use default paths if not provided
        if dust_file is None:
            dust_file = self.dust_file
        if spec_lambda is None:
            spec_lambda = self.spec_lambda

        # Select dust model parameters
        dust_model_params = {
            "DL07": (7, 1001, 22),
            "THEMIS": (11, 576, 37),
        }
        
        if self.duste_model not in dust_model_params:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        nqpah_dustem, ndim_dustem, numin_dustem = dust_model_params[self.duste_model]

        # Initialize storage for interpolated spectra (JAX-compatible)
        dustem2_dustem = np.zeros((len(spec_lambda), nqpah_dustem, numin_dustem * 2))

        # Read and interpolate dust emission spectra
        for k in range(nqpah_dustem):
            filename = Path(dust_file) / "dust" / "dustem" / f"{self.duste_model}_MW3.1_{'100' if k == 10 else f'{k}0'}.dat"

            if not filename.exists():
                raise FileNotFoundError(f"Error opening dust emission file: {filename}. File does not exist.")

            with filename.open('r') as f:
                next(f)  # Skip first header line
                next(f)  # Skip second header line

                lambda_dustem = np.zeros(ndim_dustem)
                dustem_dustem = np.zeros((ndim_dustem, numin_dustem * 2))

                for i in range(ndim_dustem):
                    try:
                        values = list(map(float, f.readline().strip().split()))
                        lambda_dustem[i], dustem_dustem[i, :] = values[0], values[1:]
                    except Exception:
                        raise RuntimeError(f"Error reading dust emission file: {filename}")

            # Convert wavelength from microns to Angstroms
            lambda_dustem *= 1E4

            # Interpolate dust spectra onto the master wavelength array
            jj = jnp.searchsorted(spec_lambda / 1E4, 1, side='left')
            for j in range(numin_dustem * 2):
                dustem2_dustem[jj:, k, j] = jnp.interp(spec_lambda[jj:], lambda_dustem, dustem_dustem[:, j])

        self.dustem2_dustem = jnp.array(dustem2_dustem)

    def update_dust_params(self, duste_qpah = 3.5, duste_umin = 1.0, duste_gamma = 0.01):
        """
        Update dust parameters for emission calculations.

        Parameters:
            duste_qpah (float, optional): New PAH fraction.
            duste_umin (float, optional): New minimum U radiation field.
            duste_gamma (float, optional): New fraction of high U component.
        """
        self.duste_qpah = duste_qpah
        self.duste_umin = duste_umin
        self.duste_gamma = duste_gamma

    def compute_dust_emission(self, spec_attn, spec_dustfree, spec_lambda, diffuse_curve,
                                        duste_qpah=None, duste_umin=None, duste_gamma=None):
        """
        Compute dust emission using JAX-optimized vectorization for GPU acceleration.

        Parameters:
            spec_attn (jnp.ndarray): Attenuated spectrum after dust absorption.
            spec_dustfree (jnp.ndarray): Stellar spectrum before attenuation.
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            duste_qpah (float, optional): PAH fraction. Defaults to self.duste_qpah if None.
            duste_umin (float, optional): Minimum U radiation field. Defaults to self.duste_umin if None.
            duste_gamma (float, optional): Fraction of high U component. Defaults to self.duste_gamma if None.

        Returns:
            tuple: (Updated spectrum with dust emission added, Estimated dust mass)
        """

        # Use provided parameters if given, otherwise fallback to self attributes
        duste_qpah = self.duste_qpah
        duste_umin = self.duste_umin
        duste_gamma = self.duste_gamma

        def tsum(x, y):
            nn = len(x)
            tsum = jnp.sum(jnp.abs((x[1:nn] - x[0:nn-1])) * (y[1:nn] + y[0:nn-1]) / 2.0)
            return tsum

        # Compute total luminosity before and after attenuation
        # FSPS uses clight = 2.9979E18 Angstrom/s, spec_lambda is in Angstroms
        clight = 2.9979E18  # Angstrom/s, matches FSPS
        nu = clight / spec_lambda  # Frequency in Hz
        print(f"DEBUG: nu[1000]={nu[1000]:.2e}, spec_lambda[1000]={spec_lambda[1000]:.2f}")
        lbold = tsum(nu, spec_attn)#jnp.trapezoid(spec_attn, x=nu)
        lboln = tsum(nu, spec_dustfree)#jnp.trapezoid(spec_dustfree, x=nu)  # L_bol before attenuation
        print(f"DEBUG: Initial lbold={lbold:.2e}, lboln={lboln:.2e}, labs={lboln-lbold:.2e}")
        # Interpolation indices for PAH fraction and Umin
        qlo = jnp.clip(jnp.searchsorted(self.qpaharr, duste_qpah) - 1, 0, len(self.qpaharr) - 2)
        dq = jnp.clip((duste_qpah - self.qpaharr[qlo]) / (self.qpaharr[qlo + 1] - self.qpaharr[qlo]), 0.0, 1.0)
        ulo = jnp.clip(jnp.searchsorted(self.uminarr, duste_umin) - 1, 0, len(self.uminarr) - 2)
        du = jnp.clip((duste_umin - self.uminarr[ulo]) / (self.uminarr[ulo + 1] - self.uminarr[ulo]), 0.0, 1.0)
    

        # Ensure gamma fraction is within [0,1]
        gamma = jnp.clip(duste_gamma, 0.0, 1.0)

        # Perform bilinear interpolation over qpah and Umin using `vmap`
        def interpolate_dustem(i):
            return (
                (1 - dq) * (1 - du) * self.dustem2_dustem[i, qlo, 2 * ulo - 1] +
                dq * (1 - du) * self.dustem2_dustem[i, qlo + 1, 2 * ulo - 1] +
                dq * du * self.dustem2_dustem[i, qlo + 1, 2 * (ulo + 1) - 1] +
                (1 - dq) * du * self.dustem2_dustem[i, qlo, 2 * (ulo + 1) - 1]
            ), (
                (1 - dq) * (1 - du) * self.dustem2_dustem[i, qlo, 2 * ulo] +
                dq * (1 - du) * self.dustem2_dustem[i, qlo + 1, 2 * ulo] +
                dq * du * self.dustem2_dustem[i, qlo + 1, 2 * (ulo + 1)] +
                (1 - dq) * du * self.dustem2_dustem[i, qlo, 2 * (ulo + 1)]
            )

        dumin, dumax = jax.vmap(interpolate_dustem)(jnp.arange(len(spec_lambda)))
        self.dumin = dumin
        self.dumax = dumax

        # Compute dust emission spectrum
        mduste = (1 - gamma) * dumin + gamma * dumax
        mduste = jnp.maximum(mduste, tiny_number)
        self.mduste = mduste
        # Normalize to absorbed luminosity
        labs = lboln - lbold  # Energy absorbed by dust (FSPS: lboln-lbold)
        self.labs = labs
        norm = tsum(nu, mduste)  # Normalization factor (use tsum like FSPS)
        self.norm = norm
        duste = jnp.maximum(mduste / norm * labs, tiny_number)  # Normalize dust emission

        #duste = jnp.maximum(duste, tiny_number)
        self.duste = duste
        tduste = jnp.zeros_like(duste)

        diff_dust = jnp.exp(-diffuse_curve)  # Apply diffuse dust attenuation


       # --- Simple Python loop for debugging ---
        iself = 0
        iter_count = 0
        max_iters = 100
        print(f"Starting self-absorption loop: lboln={lboln:.2e}, lbold={lbold:.2e}")
        while ((lboln - lbold) > 1e-2) or (iself == 0):
            if iter_count >= max_iters:
                print(f"Warning: Maximum iterations ({max_iters}) reached")
                break

            if iter_count < 5:
                print(f"  Iteration {iter_count}: lboln={float(lboln):.2e}, lbold={float(lbold):.2e}, diff={float(lboln-lbold):.2e}")
                print(f"    Before: duste[5000]={float(duste[5000]):.2e}, tduste[5000]={float(tduste[5000]):.2e}")

            oduste = duste
            duste = duste * diff_dust
            tduste = tduste + duste

            if iter_count < 5:
                print(f"    After: duste[5000]={float(duste[5000]):.2e}, tduste[5000]={float(tduste[5000]):.2e}")

            lbold = tsum(nu, duste)
            lboln = tsum(nu, oduste)

            duste = jnp.maximum(mduste / norm * (lboln - lbold), tiny_number)

            if iter_count < 5:
                print(f"    New duste[5000]={float(duste[5000]):.2e}")

            iself = 1
            iter_count += 1

        final_tduste = tduste
        print(f"Loop completed after {iter_count} iterations")
        print(f"final_tduste[5000]={float(final_tduste[5000]):.2e}, sum={float(jnp.sum(final_tduste)):.2e}")

        # --- Final calculations after the loop ---
        # Compute estimated dust mass
        mdust = 3.21E-3 / (4 * jnp.pi) * labs / norm
        
        # Add dust emission to the stellar spectrum
        spec_duste = spec_attn + final_tduste 

        return spec_duste, mdust
    

from pathlib import Path

file_path = Path("/Users/amanda/Prospector/fsps/dust/dustem/DL07_MW3.1_20.dat")

with open(file_path, "r") as f:
    header_lines = [next(f).rstrip("\n") for _ in range(2)]

print("Header lines:")
for line in header_lines:
    print(line)


emi = DustEmission(spec_lambda=csp.wave, dust_file='/Users/amanda/Prospector/fsps')

emi.update_dust_params(duste_qpah=3.5, duste_umin=1, duste_gamma=0.01)

# WARNING: The fsps_emi.pkl was generated using FSPS spectra as inputs,
# so we should compare apples-to-apples by using the SAME input spectra.
# Let me verify this by running compute_dust_emission with CSP spectra
print(f"\n=== Testing with CSP spectra ===")
print(f"spec (CSP attenuated) min: {np.min(spec):.2e}, max: {np.max(spec):.2e}")
print(f"spec_dustfree (CSP) min: {np.min(spec_dustfree):.2e}, max: {np.max(spec_dustfree):.2e}")
specduste, mdust_me = emi.compute_dust_emission(spec_attn = spec, spec_dustfree=spec_dustfree, spec_lambda=csp.wave, diffuse_curve=csp.diffuse)
print(f"After compute_dust_emission:")
print(f"labs: {emi.labs:.2e}")
print(f"norm: {emi.norm:.2e}")
print(f"specduste min: {np.min(specduste):.2e}, max: {np.max(specduste):.2e}")
print(f"spec (input) min: {np.min(spec):.2e}, max: {np.max(spec):.2e}")
print(f"Dust emission component: specduste[5000]-spec[5000] = {specduste[5000]-spec[5000]:.2e}")
nonzero_idx = np.where(emi.mduste > 0)[0]
if len(nonzero_idx) > 0:
    print(f"First non-zero mduste at index {nonzero_idx[0]}, wavelength {csp.wave[nonzero_idx[0]]:.1f} Angstrom")
    print(f"mduste[{nonzero_idx[0]}]: {emi.mduste[nonzero_idx[0]]:.2e}")
else:
    print("WARNING: mduste is all zeros!")

csp_emi = fnu2flam(csp.wave, specduste*1e-23*3631)

# Load FSPS reference
import pickle as pkl
with open('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/fsps_emi.pkl', 'rb') as f:
    fsps_emi = pkl.load(f)

print(f"\nDiagnostics:")
print(f"csp_emi shape: {csp_emi.shape}, min: {np.min(csp_emi):.2e}, max: {np.max(csp_emi):.2e}")
print(f"fsps_emi shape: {fsps_emi.shape}, min: {np.min(fsps_emi):.2e}, max: {np.max(fsps_emi):.2e}")
print(f"NaN in csp_emi: {np.any(np.isnan(csp_emi))}, NaN in fsps_emi: {np.any(np.isnan(fsps_emi))}")
print(f"Zeros in fsps_emi: {np.sum(fsps_emi == 0)}")

# Check specific wavelengths
test_indices = [4800, 5000, 5200, 5500]
print(f"\nComparison at specific indices:")
for idx in test_indices:
    if idx < len(csp.wave):
        wave = csp.wave[idx]
        csp_val = csp_emi[idx]
        fsps_val = fsps_emi[idx]
        diffuse_val = csp.diffuse[idx]
        ratio = csp_val / fsps_val if fsps_val != 0 else float('inf')
        print(f"  λ={wave:.1f}Å: CSP={csp_val:.2e}, FSPS={fsps_val:.2e}, ratio={ratio:.3f}, diffuse_curve={diffuse_val:.2e}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20/3,4), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
ax1.plot(csp.wave, csp_emi, color ='dodgerblue', label='CSP Dust Emission Spectrum')
ax1.plot(csp.wave, fsps_emi, '--', color = 'red', label='FSPS Dust Emission Spectrum')
ax1.set_yscale('log'); ax1.set_xlim(500, 970000); ax1.legend(); ax1.grid(True, alpha=0.3)

resid = (csp_emi - fsps_emi)
# Avoid division by zero
mask = fsps_emi != 0
resid_normalized = np.zeros_like(resid)
resid_normalized[mask] = resid[mask] / fsps_emi[mask]
resid_normalized[~mask] = 0

ax2.plot(csp.wave, resid_normalized, color='k'); ax2.axhline(0, color='gray', lw=1)
ax2.set_xlabel('Wavelength'); ax2.set_ylabel('Norm. Residuals'); ax2.grid(True, alpha=0.3)
#ax1.set_ylim(10**-14.6, 10**-14.3)
ax1.set_ylim(10**-17, 10**-13)
plt.tight_layout()
plt.savefig('dust_emission_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to dust_emission_comparison.png")
print(f"\nResidual statistics (where fsps_emi != 0):")
max_resid_idx = np.argmax(np.abs(resid_normalized[mask]))
actual_idx = np.where(mask)[0][max_resid_idx]
print(f"Max absolute residual: {np.max(np.abs(resid_normalized[mask])):.6f} at λ={csp.wave[actual_idx]:.1f}Å")
print(f"  CSP={csp_emi[actual_idx]:.2e}, FSPS={fsps_emi[actual_idx]:.2e}")
print(f"Mean absolute residual: {np.mean(np.abs(resid_normalized[mask])):.6f}")
print(f"Residual < 0.001 everywhere: {np.all(np.abs(resid_normalized[mask]) < 0.001)}")
plt.close()