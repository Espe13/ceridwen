import jax.numpy as jnp
from jax import jit
from ceridwen.ssps.ssp_data import SSPData
from CSPBasis import CSPBasis
import matplotlib.pyplot as plt
import fsps

def fnu2flam(lam,fnu): # fnu in erg/s/cm2/Hz
    c = 2.998e18 #A/s
    flam = c* fnu / lam**2
    return flam

# make random sfh
# 
t = jnp.linspace(0, 13.8, 100)  # in Gyr
sfr = jnp.exp(-t / 2.0)  # simple exponential decay

ssp_data = SSPData.load('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/ssp_data.h5')

csp = CSPBasis(ssp_data)
csp.add_sfh(sfh=sfr, forward_time=t)
csp_spec = fnu2flam(csp.wave, csp.get_spectrum()*1e-23 * 3631)


fsps_csp = fsps.StellarPopulation(zcontinuous=1, sfh=3)

fsps_csp.set_tabular_sfh(t, sfr[::-1])
fsps_wave, fsps_spec = fsps_csp.get_spectrum(tage=13.8, peraa=True)

fsps_spec = fsps_spec * 1e-23 * 3631

same = jnp.isclose(csp.wave, fsps_wave, atol=1e-19)
print(f"Spectra are the same: {jnp.all(same)}")
plt.figure(figsize=(10/3,3))
plt.plot(csp.wave, csp_spec, label='CSPBasis', color = 'dodgerblue')
#plt.plot(fsps_wave, fsps_spec, label='FSPS', color = 'orange')
#plt.xlim(3000, 10000)
#plt.ylim(1e-20, 1e-18)
plt.yscale('log')
plt.xlabel('Wavelength (A)')
plt.ylabel('Flux (erg/s/cm2/A)')
plt.legend()
plt.show()


