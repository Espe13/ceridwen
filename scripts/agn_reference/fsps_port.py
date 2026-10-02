"""Numpy port of the FSPS AGN torus path, used as an independent cross-check.

python-fsps builds and runs on this machine, so the A1 reference is produced by
python-fsps itself (``make_refs.py a1``). This port only checks that the
reference is understood: spec(fagn) - spec(0) must equal
``10**log_lbol * fagn * T(tau) * exp(-dust2 * attn_curve)``.

Ported from FSPS ``cconroy20/fsps`` (b6600ff; identical AGN files at 05b5e55,
the libfsps pinned by python-fsps v0.5.0):

- ``src/sps_setup.f90:895-925``  table read + log-log resampling with ``locate``
- ``src/agn_dust.f90:16-29``     linear tau interpolation, clamped dj, diffuse screen
- ``src/attn_curve.f90``         dust_type 0, 2, 4 (the ones used here)
- ``src/locate.f90``, ``src/linterparr.f90``

Fortran single-precision literals (``tiny30 = 10**(-30.0)``, the Calzetti
coefficients) are reproduced by rounding them through float32, because
python-fsps compiles without ``-fdefault-real-8``.
"""
import os

import numpy as np

# 10**(-30.0): default-real literal exponent, evaluated in single precision
TINY30 = float(np.float32(10.0) ** np.float32(-30.0))


def _f32(x):
    return float(np.float32(x))


def locate(xx, x):
    """FSPS locate.f90, 1-based result (0 if below xx(1), n if above xx(n))."""
    n = len(xx)
    ascnd = xx[-1] >= xx[0]
    jl, ju = 0, n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascnd == (x >= xx[jm - 1]):
            jl = jm
        else:
            ju = jm
    if x == xx[0]:
        return 1
    if x == xx[n - 1]:
        return n - 1
    return jl


def linterparr(xin, yin, xout):
    """FSPS linterparr.f90: linear, extrapolating from the end segments."""
    n = len(xin)
    out = np.empty(len(xout))
    for i, x in enumerate(xout):
        klo = max(min(locate(xin, x), n - 1), 1) - 1  # to 0-based
        out[i] = yin[klo] + (yin[klo + 1] - yin[klo]) * (x - xin[klo]) / (xin[klo + 1] - xin[klo])
    return out


def read_nenkova(sps_home=None):
    """Return (tau[9], lam[125], spec[125, 9]) as FSPS reads them."""
    sps_home = sps_home or os.environ["SPS_HOME"]
    path = os.path.join(sps_home, "dust", "Nenkova08_y010_torusg_n10_q2.0.dat")
    with open(path) as f:
        lines = f.read().splitlines()
    tau = np.array(lines[3].split(), dtype=np.float64)
    rows = np.array([ln.split() for ln in lines[4:4 + 125]], dtype=np.float64)
    return tau, rows[:, 0], rows[:, 1:]


def resample_torus(spec_lambda, sps_home=None):
    """agndust_spec(nspec, 9) exactly as sps_setup.f90:919-924 builds it."""
    tau, lam, spec = read_nenkova(sps_home)
    out = np.zeros((len(spec_lambda), len(tau)))
    i1 = locate(spec_lambda, lam[0])
    i2 = locate(spec_lambda, lam[-1])
    sl = slice(i1 - 1, i2)  # Fortran i1:i2 inclusive
    for i in range(len(tau)):
        out[sl, i] = 10 ** linterparr(np.log10(lam), np.log10(spec[:, i] + TINY30),
                                     np.log10(spec_lambda[sl])) - TINY30
    return tau, out


def interp_tau(agndust_tau, agndust_spec, agn_tau):
    """agn_dust.f90:17-23."""
    n = len(agndust_tau)
    jlo = min(max(locate(agndust_tau, agn_tau), 1), n - 1)
    dj = (agn_tau - agndust_tau[jlo - 1]) / (agndust_tau[jlo] - agndust_tau[jlo - 1])
    dj = max(min(dj, 1.0), 0.0)
    return (1 - dj) * agndust_spec[:, jlo - 1] + dj * agndust_spec[:, jlo]


def attn_curve(lam, dust_type=0, dust_index=-0.7):
    """attn_curve.f90 for dust_type 0, 2, 4 with tauv = 1."""
    lamv, dd63, dlam, lamuvb = 5500.0, 6300.0, 350.0, 2175.0
    if dust_type == 0:
        return (lam / lamv) ** dust_index
    if dust_type not in (2, 4):
        raise NotImplementedError(dust_type)
    c = [_f32(v) for v in (1.17, 1.857, 1.04, 1.78, 2.156, 1.509, 0.198, 0.011, 0.44, 4.05)]
    w63 = locate(lam, dd63)
    cal = np.zeros_like(lam)
    x = 1e4 / lam
    cal[w63:] = c[0] * (-c[1] + c[2] * x[w63:]) + c[3]
    cal[:w63] = c[0] * (-c[4] + c[5] * x[:w63] - c[6] * x[:w63] ** 2 + c[7] * x[:w63] ** 3) + c[3]
    cal = cal / c[8] / c[9]
    w = locate(cal, 0.0)
    if w != len(lam):
        cal[w:] = 0.0
    if dust_type == 2:
        return cal
    eb = _f32(0.85) - _f32(1.9) * dust_index
    drude = eb * (lam * dlam) ** 2 / ((lam ** 2 - lamuvb ** 2) ** 2 + (lam * dlam) ** 2)
    return (cal + drude / c[9]) * (lam / lamv) ** dust_index


def agn_torus_fsps(spec_lambda, log_lbol, fagn, agn_tau, dust2=0.0, dust_type=0,
                   dust_index=-0.7, sps_home=None, _cache={}):
    """The term agn_dust.f90:29 adds: 10**lbol * fagn * T(tau) * exp(-dust2*curve)."""
    key = (spec_lambda.tobytes(), sps_home)
    if key not in _cache:
        _cache[key] = resample_torus(spec_lambda, sps_home)
    tau_nodes, agndust_spec = _cache[key]
    t = interp_tau(tau_nodes, agndust_spec, agn_tau)
    t = t * np.exp(-dust2 * attn_curve(spec_lambda, dust_type, dust_index))
    return 10 ** log_lbol * fagn * t
