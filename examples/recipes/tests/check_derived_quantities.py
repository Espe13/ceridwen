#!/usr/bin/env python
"""Check examples/recipes/derived_quantities.py on CPU; prints PASS / FAIL with numbers.

1. mass    : cumulative mass at the oldest node == PostProcess mass_formed (rel 1e-10), through
             a real PostProcess run on the BPASS test grid, for sfh_interp "step" and "linear".
2. analytic: constant SFH -> mwa = T/2, t50 = T/2 (and t90 = T/10) to 1e-10 relative, both interps,
             on a non-uniform node grid; plus a step-SFH cross-check against Prospector's
             nonpar_mwa / sfh_to_cmf (prospect/plotting/sfh.py:255-278).
3. mags    : recipe absolute magnitudes (method='hires') vs Prospector's SpecModel.absolute_rest_maggies
             (prospect/models/sedmodel.py:851-883, called unbound, real code, sedpy filters)
             on the same physical spectrum, |dM| < 1e-6 mag.
"""
from __future__ import annotations

import os
import sys
import types
import pathlib
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
PROSPECTOR = os.environ.get("PROSPECTOR_DIR", "")
if not (pathlib.Path(PROSPECTOR) / "prospect").is_dir():
    print("FAIL  prospector: PROSPECTOR_DIR unset or not a checkout; set PROSPECTOR_DIR to a clone of https://github.com/bd-j/prospector at a78d153")
    print("FAIL check_derived_quantities")
    sys.exit(1)

from ceridwen import SSPData, CSPBasis, SedModel, PostProcess, SpectrumSample   # noqa: E402
from ceridwen.cosmology import Cosmology                                        # noqa: E402
from ceridwen.model import logsfr_ratios_to_sfh                                 # noqa: E402
from ceridwen.priors import Uniform                                             # noqa: E402
from ceridwen.sampler.runner import SamplingResult                              # noqa: E402
import derived_quantities as dq                                                 # noqa: E402

GRID = os.environ.get("CERIDWEN_TEST_SSP", str(REPO / "ceridwen/data/test_data/ssp_data_bpass.h5"))
FILTERS = ["galex_FUV", "galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
           "twomass_J", "twomass_Ks", "wise_w1"]
ok_all = True


def report(name, ok, msg):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")


def build(ssp, interp, n_time=7):
    lookback = jnp.array([0.0, 0.03, 0.1, 0.5, 1.5, 4.0, 13.0])[:n_time]
    csp = CSPBasis(ssp, theta={"lookback_time": lookback, "sfh": jnp.ones(n_time),
                               "logzsol": jnp.array([-0.3])},
                   zh_const=True, sfh_interp=interp, add_dust=False, add_diffuse_dust=True,
                   add_neb=False, verbose=False, cosmo=Cosmology.planck18())
    T = np.array(csp.sfh_times)
    model = SedModel(
        csp, observations=[],
        priors={"logsfr_ratios": Uniform(low=-1.0, high=1.0), "logmass": Uniform(low=9.0, high=11.0),
                "logzsol": Uniform(low=-1.0, high=0.2), "diffuse_tau_kc": Uniform(low=0.0, high=1.0),
                "diffuse_dust_index": Uniform(low=-1.0, high=0.3)},
        transforms={"sfh": lambda th, _t=T: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)},
        free_param_init={"logsfr_ratios": jnp.zeros(n_time - 1), "logmass": jnp.array([10.0])},
        broaden_photometry=False)
    return csp, model


def fake_result(model, n=6, seed=1):
    rng = np.random.default_rng(seed)
    lim = {"logsfr_ratios": (-1, 1), "logmass": (9, 11), "logzsol": (-1, 0.2),
           "diffuse_tau_kc": (0, 1), "diffuse_dust_index": (-1, 0.3)}
    samples = {}
    for k in model.param_names:
        shape = tuple(np.shape(model.theta_init[k]))
        lo, hi = lim[k]
        samples[k] = rng.uniform(lo, hi, size=(n,) if shape in ((), (1,)) else (n,) + shape)
    return SamplingResult(samples=samples, log_evidence=np.nan, log_evidence_err=np.nan,
                          log_weights=np.zeros(n), log_likelihoods=np.zeros(n),
                          param_names=list(model.param_names), n_likelihood_calls=0,
                          wall_time_s=0.0, sampler_name="fake-uniform")


# ---------------------------------------------------------------- 1. mass (real PostProcess)
ssp = SSPData.load(GRID)
captured = []
for interp in ("step", "linear"):
    csp, model = build(ssp, interp)
    derived = dq.make_derived(csp, filters=FILTERS, colors=[("sdss_g0", "sdss_r0")])
    if interp == "step":
        derived["_capture"] = lambda s: (captured.append(s), 0.0)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = PostProcess(model, fake_result(model), derived=derived, predictions=False).run()
    mf = out["extras"]["sfh"]["mass_formed"]
    mc = out["derived"]["mass_formed_check"]
    rel = np.max(np.abs(mc / mf - 1.0))
    report(f"mass[{interp}]", rel <= 1e-10,
           f"max |M(<T_old)/mass_formed - 1| = {rel:.3e} over {mf.size} draws "
           f"(mass_formed {mf.min():.4e}..{mf.max():.4e} Msun)")
    t50, t90, mwa = out["derived"]["t50"], out["derived"]["t90"], out["derived"]["mwa"]
    T_old = float(np.asarray(csp.sfh_times)[-1]) / 1e9
    sane = bool(np.all((t90 <= t50) & (t50 <= T_old) & (mwa > 0) & (mwa < T_old)))
    report(f"order[{interp}]", sane,
           f"t90 <= t50 <= T_old and 0 < mwa < T_old; draw0: mwa={mwa[0]:.6f} t50={t50[0]:.6f} "
           f"t90={t90[0]:.6f} Gyr; g-r={out['derived']['colors'][0, 0]:.6f}")

# ---------------------------------------------------------------- 2. analytic constant SFH
T_gyr = np.array([0.0, 0.01, 0.1, 0.35, 1.0, 2.2, 5.0, 9.7])
Tend = T_gyr[-1]
for interp, sfr in (("step", np.full(T_gyr.size - 1, 3.0)), ("linear", np.full(T_gyr.size, 3.0))):
    s = SpectrumSample(wave_rest=np.array([1.0, 2.0]), full=np.zeros(2), intrinsic=np.zeros(2),
                       dustfree=np.zeros(2), theta={}, zred=0.0, logmass=None, sfr=sfr,
                       lookback_gyr=T_gyr, cosmo=None)
    mwa = dq.mass_weighted_age(s, interp)
    t50 = dq.formation_time(s, interp, 0.5)
    t90 = dq.formation_time(s, interp, 0.9)
    errs = (abs(mwa / (Tend / 2) - 1), abs(t50 / (Tend / 2) - 1), abs(t90 / (Tend / 10) - 1))
    report(f"analytic[{interp}]", max(errs) <= 1e-10,
           f"T={Tend} Gyr: mwa={mwa!r} t50={t50!r} (T/2={Tend / 2}), t90={t90!r} (T/10={Tend / 10}); "
           f"rel err {errs[0]:.1e} {errs[1]:.1e} {errs[2]:.1e}")

# step SFH, non-constant: against Prospector's own formulas (ported verbatim, sfh.py:255-278)
sfr = np.array([5.0, 2.0, 7.0, 1.0, 0.5, 3.0, 0.2])
T_yr = T_gyr * 1e9
dtsq = (T_yr[1:] ** 2 - T_yr[:-1] ** 2) / 2                     # nonpar_mwa, sfh.py:261
mwa_p = (dtsq * sfr).sum() / (sfr * np.diff(T_yr)).sum() / 1e9    # sfh.py:262-264
masses = (sfr * np.diff(T_yr))[::-1]                            # sfh_to_cmf, sfh.py:270-277
cmf = np.append(0.0, masses.cumsum() / masses.sum())[::-1]      # fraction formed older than t
t50_p = np.interp(0.5, cmf[::-1], T_gyr[::-1])                  # cmf linear within a step bin
t90_p = np.interp(0.9, cmf[::-1], T_gyr[::-1])
s = SpectrumSample(wave_rest=np.array([1.0, 2.0]), full=np.zeros(2), intrinsic=np.zeros(2),
                   dustfree=np.zeros(2), theta={}, zred=0.0, logmass=None, sfr=sfr,
                   lookback_gyr=T_gyr, cosmo=None)
got = (dq.mass_weighted_age(s, "step"), dq.formation_time(s, "step", 0.5), dq.formation_time(s, "step", 0.9))
errs = [abs(g / r - 1) for g, r in zip(got, (mwa_p, t50_p, t90_p))]
report("prospector-sfh[step]", max(errs) <= 1e-10,
       f"mwa {got[0]:.12f} vs {mwa_p:.12f}; t50 {got[1]:.12f} vs {t50_p:.12f}; "
       f"t90 {got[2]:.12f} vs {t90_p:.12f}; max rel {max(errs):.1e}")

# ---------------------------------------------------------------- 3. magnitudes vs Prospector
np.trapz = getattr(np, "trapz", np.trapezoid)       # prospect @ a78d153 calls np.trapz (numpy 2 removed it)
sys.path.insert(0, PROSPECTOR)
from prospect.models.sedmodel import SpecModel                                   # noqa: E402
from prospect.sources.constants import cosmo as p_cosmo, lsun as p_lsun, \
    to_cgs_at_10pc as p_to_cgs, jansky_cgs as p_jy                              # noqa: E402
from sedpy.observate import load_filters                                         # noqa: E402

s = captured[0]
m_recipe = dq.absolute_magnitudes(s, dq.AbsMagFilters(FILTERS))
m_phot = dq.absolute_magnitudes(s, dq.AbsMagFilters(FILTERS, method="photometry"))
zred = 0.7
ld_mpc = p_cosmo.luminosity_distance(zred).to("Mpc").value
lnu_cgs = np.asarray(s.full, float) * dq._LSUN_ERG_S                 # the same PHYSICAL spectrum, erg/s/Hz
norm = p_to_cgs / (3631 * p_jy) * (1 + zred) / (ld_mpc * 1e5) ** 2    # SpecModel.flux_norm, sedmodel.py:426-445
stub = types.SimpleNamespace(_zred=zred, _wave=np.asarray(s.wave_rest, float),
                             _norm_spec=lnu_cgs / p_lsun * norm, _want_lines=False, _need_lines=False)
sedpy_filters = load_filters(FILTERS)
m_prosp = -2.5 * np.log10(SpecModel.absolute_rest_maggies(stub, sedpy_filters))
d = m_recipe - m_prosp
report("absmag vs prospector", np.max(np.abs(d)) <= 1e-6,
       f"max |dM| = {np.max(np.abs(d)):.3e} mag over {len(FILTERS)} bands; per band: "
       + " ".join(f"{f}:{x:+.1e}" for f, x in zip(FILTERS, d)))
print(f"INFO  method='photometry' (FilterSet, as Photometry predictions) - prospector: max |dM| = "
      f"{np.max(np.abs(m_phot - m_prosp)):.3e} mag (not a gate)")
naive = 2.5 * np.log10(p_lsun / dq._LSUN_ERG_S)
print(f"INFO  L_sun convention: prospector lsun={p_lsun:.4e}, ceridwen {dq._LSUN_ERG_S:.4e} -> "
      f"feeding the same L_sun/Hz array to both would differ by {naive * 1e3:.3f} mmag")

print("PASS" if ok_all else "FAIL", "check_derived_quantities")
sys.exit(0 if ok_all else 1)
