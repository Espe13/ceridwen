import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import fsps
from tqdm import tqdm

from ceridwen.ssps.ssp_data import SSPData
from csp import CSPBasis

# ------------------------
# Config & helpers
# ------------------------
T_UNIV = 13.8  # Gyr
N_T = 200
SEED = 13
REF_WIN = (4000.0, 7000.0)  # Å, normalization window for spectral shape comparison

rng = np.random.default_rng(SEED)

def fnu2flam(lam, fnu):
    """Convert f_nu [erg/s/cm^2/Hz] to f_lambda [erg/s/cm^2/Å]."""
    c = 2.998e18  # Å/s
    return c * (fnu / (lam**2))

def normalize_in_window(wave, spec, wmin, wmax, eps=1e-300):
    """Normalize spectrum by its median within [wmin, wmax]."""
    m = (wave >= wmin) & (wave <= wmax) & np.isfinite(spec)
    if not np.any(m):
        return spec  # no change if window empty
    norm = np.median(spec[m])
    return spec / (norm + eps)

def nrmse(a, b, eps=1e-300):
    """Normalized RMSE = RMSE / RMS(b). Returns scalar."""
    resid = a - b
    rmse = np.sqrt(np.mean(resid**2))
    denom = np.sqrt(np.mean(b**2)) + eps
    return rmse / denom

# ------------------------
# SFH constructors on a uniform time grid t (cosmic time)
# ------------------------
t = jnp.linspace(1e-2, T_UNIV, N_T)   # Gyr (avoid exactly 0)
tau = T_UNIV - t                      # lookback time (Gyr)

def gaussian_burst(center_tau, width_tau, amp=1.0):
    """Gaussian in lookback time, evaluated on tau grid (Gyr)."""
    return amp * jnp.exp(-0.5 * ((tau - center_tau) / width_tau)**2)

def exp_decay(tau_scale):
    """Exponential decay with lookback time scale tau_scale (Gyr)."""
    return jnp.exp(-tau / tau_scale)

def smooth_random_sfr(k=5):
    """Random positive SFH: draw nonnegative coefficients and combine smooth bases."""
    # Smooth bases = wide Gaussians across 0..T_UNIV
    centers = np.linspace(0.1, T_UNIV*0.95, k)
    widths  = np.linspace(T_UNIV/12, T_UNIV/5, k)
    coeffs  = rng.lognormal(mean=-0.2, sigma=0.7, size=k)  # positive, varied
    s = jnp.zeros_like(tau)
    for c, w, a in zip(centers, widths, coeffs):
        s = s + a * jnp.exp(-0.5 * ((tau - c)/w)**2)
    return s / (jnp.mean(s) + 1e-12)

def bursty_sfr(n_bursts=(12, 20), width_range=(0.01, 0.15), floor_frac=1e-3):
    """
    Very bursty SFH as a sum of many narrow Gaussians in lookback time.
    - n_bursts: number of bursts (uniform integer in the given range)
    - width_range: Gaussian sigma range in Gyr (narrow -> bursty)
    - floor_frac: small constant floor as a fraction of mean burst level to avoid exact zeros
    """
    n = rng.integers(n_bursts[0], n_bursts[1]+1)
    centers = rng.uniform(low=0.02, high=T_UNIV*0.98, size=n)          # Gyr (avoid exact edges)
    widths  = rng.uniform(low=width_range[0], high=width_range[1], size=n)  # Gyr
    amps    = rng.lognormal(mean=0.0, sigma=1.0, size=n)               # heavy-tail burst strengths

    s = jnp.zeros_like(tau)
    for c, w, a in zip(centers, widths, amps):
        s = s + a * jnp.exp(-0.5 * ((tau - c) / w) ** 2)

    # tiny floor to keep CSP/FSPS stable where interpolation or masking happens
    floor = floor_frac * jnp.mean(s + 1e-12)
    s = s + floor

    # normalize to unit mass over forward time t (or equivalently tau)
    s = s / (jnp.trapezoid(s, t) + 1e-30)
    return s

# 1) Very young (recent) — burst within last 50 Myr
sfr_young = gaussian_burst(center_tau=0.03, width_tau=0.02, amp=1.0)

# 2) Very old — early universe burst ~ 12 Gyr lookback
sfr_old = gaussian_burst(center_tau=12.0, width_tau=1.0, amp=1.0)

# 3) Young + old components (bimodal)
sfr_bimodal = (gaussian_burst(0.05, 0.03, 1.0) +
               gaussian_burst(11.0, 0.8, 0.7))

# 4) Very bursty — many narrow bursts
sfr_bursty = bursty_sfr()

# 5–6) Three random smooth SFHs
sfr_rand1 = smooth_random_sfr(k=6)
sfr_rand2 = smooth_random_sfr(k=6)

sfh_list = [
    ("Very young",  sfr_young),
    ("Very old",    sfr_old),
    ("Young + old", sfr_bimodal),
    ("Bursty",      sfr_bursty),
    ("Random A",    sfr_rand1),
    ("Random B",    sfr_rand2)
]

# Optional: scale each SFH to comparable total formed mass (arbitrary normalization)
# This is only for plotting and for FSPS tabular SFH stability.
for i in range(len(sfh_list)):
    name, s = sfh_list[i]
    s = s / (jnp.trapezoid(s, t) + 1e-30)
    sfh_list[i] = (name, s)

# ------------------------
# Load SSPs and set metallicity history (here: constant)
# ------------------------
ssp_data = SSPData.load('/Users/amanda/Desktop/PhD/Tools/ceridwen/ceridwen/data/test_data/ssp_data.h5')
sp = fsps.StellarPopulation(zcontinuous=3, sfh=3)   # tabular SFH, initialise only once!
print("FSPS SPS object initialised. ")
# ------------------------
# Compute spectra: CSPBasis and FSPS
# ------------------------
def get_csp_spectrum(sfr, zh):
    csp = CSPBasis(ssp_data)
    csp.add_sfh(sfh=sfr, lookback_time=tau)           # expects lookback time grid matching sfr
    csp.add_zh(zh=zh, lookback_time=tau)
    fnu = csp.get_spectrum() * 1e-23 * 3631.0         # if your CSP returns AB flux density (Jy)
    flam = fnu2flam(np.asarray(csp.wave), np.asarray(fnu))
    return np.asarray(csp.wave), np.asarray(flam)

def get_fsps_spectrum(sfr, zh):
    # FSPS takes t (Gyr, increasing), SFR(t), and Z (scalar or array). We pass scalar for simplicity.
    sp.set_tabular_sfh(age = np.asarray(t), sfr = np.asarray(sfr), Z=zh)
    wave, spec = sp.get_spectrum(tage=13.8)  # wave[Å], spec in FSPS per-Å units
    # To compare *shapes* robustly despite unit conventions, convert to f_nu then to f_lambda
    # in the same way as CSP. If your FSPS spec is already f_lambda (per Å), this conversion
    # is unnecessary; the subsequent normalization will largely neutralize unit issues.
    fnu = spec * 1e-23 * 3631.0
    flam = fnu2flam(wave, fnu)
    return wave, flam


# ------------------------
# Build plots: 6 rows × 2 cols (left: spectra; right: SFH)
# ------------------------


# --- Figure layout: per row, 2 columns; left column splits into spectra+residuals ---
fig = plt.figure(figsize=(10, 14))
outer = fig.add_gridspec(nrows=len(sfh_list), ncols=2, hspace=0.6, wspace=0.35)

metrics = []
tol = 1e-25  # flux threshold to avoid division by ~0; adjust if needed

for row, (name, sfr) in enumerate(tqdm(sfh_list, desc="Processing SFHs")):

    zh = jnp.linspace(0.0002, 0.2, sfr.shape[0])  # constant metallicity history
    # --- Spectra (already on same grid) ---
    w_csp, f_csp = get_csp_spectrum(sfr, zh)
    w_fsp, f_fsp = get_fsps_spectrum(sfr, zh)

    

    # Similarity metric (one number)
    m_valid_spec = np.isfinite(f_csp) & np.isfinite(f_fsp)
    score = nrmse(f_csp[m_valid_spec], f_fsp[m_valid_spec]) if np.any(m_valid_spec) else np.nan
    metrics.append(score)

    # --- Axes for this row (left column split into spectra + residuals) ---
    left = outer[row, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_spec = fig.add_subplot(left[0])
    ax_res  = fig.add_subplot(left[1], sharex=ax_spec)
    ax_sfh  = fig.add_subplot(outer[row, 1])

    # --- Top-left: spectra ---
    ax_spec.plot(w_csp, f_csp, color='dodgerblue', lw=2.0, label='CSPBasis')
    ax_spec.plot(w_fsp, f_fsp, color='red',    lw=0.8, label='FSPS')
    ax_spec.set_xlim(0, 2000)
    if name == "Very young":
        ax_spec.set_ylim(1e-15, 1e-11)
    elif name == "Very old":
        ax_spec.set_ylim(1e-19, 1e-16)
    elif name == "Young + old":
        ax_spec.set_ylim(1e-18, 1e-13)
    else:
        ax_spec.set_ylim(1e-18, 1e-13)
    ax_spec.set_yscale('log')
    ax_spec.set_ylabel(r'$f_\lambda$')
    ax_spec.text(0.02, 0.92, f'NRMSE={score:.3g}', transform=ax_spec.transAxes, fontsize=9)
    if row == 0:
        ax_spec.set_title('Spectra')
        ax_spec.legend(loc='best', fontsize=8)
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    # --- Bottom-left: residuals (CSP - FSPS)/CSP, masked where CSP≈0 or invalid) ---
    m_res = m_valid_spec & (np.abs(f_csp) > tol)
    resnorm = np.full_like(f_csp, np.nan, dtype=float)
    resnorm[m_res] = (f_csp[m_res] - f_fsp[m_res]) / f_csp[m_res]

    ax_res.plot(w_csp[m_res], resnorm[m_res], color='purple', lw=1.2)
    ax_res.axhline(0.0, ls='--', lw=0.8, color='gray')
    ax_res.set_xlabel('Wavelength (Å)')
    ax_res.set_ylabel('Residual (norm.)')

    # Robust limits only if we have finite residuals
    if np.any(np.isfinite(resnorm[m_res])):
        finite_res = resnorm[m_res][np.isfinite(resnorm[m_res])]
        lo, hi = np.nanpercentile(finite_res, [1, 99])
        pad = 0.1 * max(hi - lo, 1e-6)
        ax_res.set_ylim(lo - pad, hi + pad)
    else:
        ax_res.set_ylim(-0.1, 0.1)  # safe default
    

    # --- Right: SFH vs lookback ---
    ax_sfh.plot(np.asarray(tau), np.asarray(sfr), color = 'dodgerblue', lw=2.0)
    ax_sfh.set_ylabel('SFR (arb.)')
    ax_sfh.invert_xaxis()
    ax_sfh.grid(alpha=0.25)
    ax_sfh.text(0.02, 0.92, name, transform=ax_sfh.transAxes, fontsize=9)
    if row == len(sfh_list) - 1:
        ax_sfh.set_xlabel('Lookback Time (Gyr)')
    if row == 0:
        ax_sfh.set_title('SFH')


plt.tight_layout()
plt.savefig('residuals_rising_metal.pdf', dpi=300)

# Optional: print numeric summary
for (name, _), s in zip(sfh_list, metrics):
    print(f"{name:12s}  NRMSE(CSP, FSPS) = {s:.4f}")
