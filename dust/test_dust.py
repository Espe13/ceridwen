import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import fsps
from tqdm import tqdm

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
sp = fsps.StellarPopulation(zcontinuous=3, sfh=3)  

#-------------------------------
# DUST PARAMETERS
#-------------------------------

diffdust = 0.5
diffdust_index = -0.7
young_dust = 1.5
young_dust_index = -1

sp.params['dust_type'] = 4
sp.params['dust2'] = diffdust
sp.params['dust1'] = young_dust
sp.params['dust_index'] = diffdust_index
sp.params['dust1_index'] = young_dust_index
sp.params['add_dust_emission'] = False

dust_params = {'tau_pow': young_dust,
                'alpha': -young_dust_index,
                'diffuse_params': {'tau_kc': diffdust, 'dust_index': diffdust_index}}


#-------------------------------
# DUSTY FSPS SPECTRUM
#-------------------------------
print('fsps spectrum start')
sp.set_tabular_sfh(age = np.asarray(t), sfr = np.asarray(sfr_bimodal), Z=zh)
wave, spec_fsps = sp.get_spectrum(tage=13.8) 
print('fsps spectrum done')

#-------------------------------
# CSP SPECTRUM WITH DUST
#-------------------------------
csp = CSPBasis(ssp_data, dusty=True, dust_params={'bin_edges': [(-jnp.inf, sp.params['dust_tesc'] - 9)], 'laws': ['powerlaw']})

csp.add_sfh(sfr_bimodal, lookback_time=tau)
csp.add_zh(zh, lookback_time=tau)
spec = csp.get_spectrum(dust_params=dust_params)
spec_dustfree = csp.get_spectrum_dustfree()

#-------------------------------
# Plotting the results
#-------------------------------

import matplotlib.pyplot as plt
import numpy as np

# Convert all spectra to same units first
spec_flam       = fnu2flam(csp.wave, spec * 1e-23 * 3631)
spec_dustfree_flam = fnu2flam(csp.wave, spec_dustfree * 1e-23 * 3631)
spec_fsps_flam  = fnu2flam(csp.wave, spec_fsps * 1e-23 * 3631)

norm_resid = (spec_flam - spec_fsps_flam) / spec_fsps_flam

# Create figure with two stacked subplots
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(20/3, 4.5), sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

# --- Top: Spectra ---
ax1.plot(csp.wave, spec_flam, lw=2, label='Composite Spectrum with Dust',
         color='dodgerblue', linestyle='--')
ax1.plot(csp.wave, spec_dustfree_flam, lw=2, label='Composite Spectrum without Dust',
         color='orange', linestyle='--')
ax1.plot(csp.wave, spec_fsps_flam, lw=0.5, label='FSPS Spectrum',
         color='red', linestyle='--')
ax1.set_yscale('log')
ax1.set_xlim(0, 300000)
ax1.set_ylabel('Flux (erg/s/cm²/Å)')
ax1.set_title('Composite Spectrum from CSP with Bimodal SFH')
ax1.legend()

# --- Bottom: Residuals ---
ax2.axhline(0, color='k', lw=0.8, alpha=0.6)
ax2.plot(csp.wave, norm_resid, color='black', lw=0.8)
ax2.set_xlabel('Wavelength (Å)')
ax2.set_ylabel('Norm.\nResidual')
ax2.set_xlim(0, 300000)

# Save
plt.savefig('csp_spectrum_with_dust_and_residual.png', dpi=300)
plt.show()
