import jax
import jax.numpy as jnp
import tqdm
import blackjax

from jax import jit

import jax.numpy as jnp
from jax import jit
from ceridwen.ssps.ssp_data import SSPData, collect_ssp_data_wrapper
from ceridwen.csp.csp import CSPBasis
import astropy.constants as const

import matplotlib.pyplot as plt
import fsps

jax.config.update("jax_enable_x64", True)



ssp_data = SSPData.load('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/ssp_data.h5')

csp = CSPBasis(ssp_data)
tuniv = 13.8
gal_t_table = jnp.linspace(0.1, 13.78, 3)
gal_sfr_table = jnp.linspace(0.1, 10, len(gal_t_table))
gal_z_table = jnp.ones_like(gal_sfr_table)*csp.zlegend[0] #jnp.array([4.49043431e-05, 1.42000001e-04, 2.52515678e-04, 4.49043431e-04,
       #7.98524687e-04, 1.42000001e-03, 2.52515678e-03, 4.49043431e-03,
       #7.98524687e-03, 1.42000001e-02, 1.42000001e-02, 1.42000001e-02])

csp.add_sfh(sfh=gal_sfr_table, lookback_time=13.8 - gal_t_table)
csp.add_zh(zh=gal_z_table, lookback_time=13.8 - gal_t_table)
spectrum1 = csp.get_spectrum()

rng_key = jax.random.PRNGKey(0)
num_dims = len(gal_sfr_table) + len(gal_z_table) 
num_live = 1000
num_inner_steps = num_dims * 5
num_delete = num_live // 2


num_time_bins = 3
gal_t_table = jnp.linspace(0.1, 13.78, num_time_bins)

# True parameters
true_sfh = jnp.linspace(0.1, 10, num_time_bins)
true_zh = jnp.ones_like(true_sfh) * csp.zlegend[0]

# Create "true" spectrum
csp_true = CSPBasis(ssp_data)
csp_true.add_sfh(sfh=true_sfh, lookback_time=13.8 - gal_t_table)
csp_true.add_zh(zh=true_zh, lookback_time=13.8 - gal_t_table)
spectrum_clean = csp_true.get_spectrum()

# Add noise
key, rng_key = jax.random.split(rng_key)
sigma = 0.03 * jnp.max(spectrum_clean)  # relative noise level
noise = sigma * jax.random.normal(key, shape=spectrum_clean.shape)
spectrum_obs = spectrum_clean + noise

plt.figure(figsize=(10/3, 3))
plt.plot(csp_true.wave, spectrum_obs, label='Observed Spectrum', color='red')
plt.plot(csp_true.wave, csp_true.spectrum, label='CSP Spectrum', color='black')
#plt.errorbar(csp_true.wave, spectrum_obs, yerr=jnp.abs(noise), label='Clean Spectrum', color='blue')
plt.xlim(0, 8000)
plt.yscale('log')
plt.ylim(1e-5, 1e-4)

plt.close()



csp = CSPBasis(ssp_data)
from functools import partial
from jax import jit, grad
sfh = jnp.ones_like(gal_t_table)  # Initial guess for SFH
zh = jnp.ones_like(gal_t_table) * csp.zlegend[2]
csp.add_sfh(sfh=sfh, lookback_time=13.8 - gal_t_table) #need to set sfh_times 
model = partial(csp.get_spectrum_direct)  # csp is now static

# JIT works
model_jit = jit(model)
spectrum = model_jit(sfh, zh)

# grad also works (with respect to sfh, for example)
grad_spectrum = grad(lambda sfh: jnp.sum(model(sfh, zh)))(sfh)

plt.plot(csp.wave, model(true_sfh, true_zh))
plt.plot(csp.wave, model(sfh, zh))
plt.xlim(0, 8000)
plt.yscale('log')
plt.ylim(1e-7, 1e-4)

plt.close()

n_bins = len(gal_t_table)
prior_bounds = {}

for i in range(n_bins):
    prior_bounds[f"sfh_{i}"] = (0.001, 12.0)     # Msun/yr
    prior_bounds[f"zh_{i}"]  = (5e-5, 4e-5)      # Z
prior_bounds["sigma"] = (1e-7, 1e-5)              # flux error

print(prior_bounds)



def unpack_params(params_flat, n_bins):
    sfh = jnp.array([params_flat[f"sfh_{i}"] for i in range(n_bins)])
    zh  = jnp.array([params_flat[f"zh_{i}"]  for i in range(n_bins)])
    sigma = params_flat["sigma"]
    return sfh, zh, sigma

def loglikelihood_fn(params_flat):
    #sfh, zh, sigma = unpack_params(params_flat, n_bins)
    sfh =jnp.array( [params_flat[f"sfh_{i}"] for i in range(n_bins)])
    zh = jnp.array([params_flat[f"zh_{i}"] for i in range(n_bins)])
    sigma = params_flat["sigma"]
    mu = model(sfh, zh)
    cov = sigma ** 2
    lgl = jax.scipy.stats.multivariate_normal.logpdf(spectrum_obs, mu, cov)
    print(f"Log-likelihood: {lgl}")
    return lgl



from blackjax.ns.utils import uniform_prior

rng_key, prior_key = jax.random.split(rng_key)
particles, logprior_fn = uniform_prior(prior_key, num_live, prior_bounds)


i = 0  # index of the live particle you want to test
params_test = {k: v[i] for k, v in particles.items()}

# Evaluate the likelihood
ll = loglikelihood_fn(params_test)

print("Log-likelihood:", ll)



nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)
init_fn = jax.jit(nested_sampler.init)
step_fn = jax.jit(nested_sampler.step)

live = init_fn(particles)



target_dead_points = 50

dead = []
iteration = 0
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while len(dead) * num_delete < target_dead_points:
        if iteration % 1 == 0:  # every step in short run
            print(f"[{iteration}] logZ = {live.logZ:.2f}, ΔlogZ = {live.logZ_live - live.logZ:.2f}")
        rng_key, subkey = jax.random.split(rng_key, 2)
        live, dead_info = step_fn(subkey, live)
        dead.append(dead_info)
        pbar.update(num_delete)

dead = blackjax.ns.utils.finalise(live, dead)

print('Final loglikelihood:', dead.loglikelihood)
print('Final loglikelihood_birth:', dead.loglikelihood_birth)
print('Number of dead points:', len(dead.particles['sfh_0']))
print('post process')

from anesthetic import NestedSamples
columns = ['sfh_0', 'sfh_1', 'sfh_2', 'sigma', 'zh_0', 'zh_1', 'zh_2']
labels = [r"$sfh_0$", r"$sfh_1$", r"$sfh_2$", r"$\sigma$", r"$zh_0$", r"$zh_1$", r"$zh_2$"]


data = jnp.vstack([dead.particles[key] for key in columns]).T




samples = NestedSamples(
    data,
    logL=dead.loglikelihood,
    logL_birth=dead.loglikelihood_birth,
    columns=columns,
    labels=labels,
    logzero=jnp.nan,
)
samples.to_csv("test1_ceridwen.csv")


from anesthetic import read_chains
samples = read_chains("test1_ceridwen.csv")


kinds ={'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
axes = samples.prior().plot_2d(['sfh_0', 'sfh_1', 'sfh_2', 'sigma', 'zh_0', 'zh_1', 'zh_2'], kinds=kinds, label='prior')
samples.plot_2d(axes, kinds=kinds, label='posterior')
plt.show()
plt.close()