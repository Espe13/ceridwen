"""Check for ``examples/recipes/igm_damping_dla.py`` (CPU, < 1 min).

1. Damping wing and DLA transmission vs Prospector's own ``tau_damping`` / ``voigt_profile``
   (``prospect/models/sedmodel.py`` @ a78d153, imported from a clone), same wavelength grid and
   redshift, WMAP9 (Prospector's ``cosmo``, ``prospect/sources/constants.py:4``): |dT| <= 1e-10.
2. ``x_HI = 0`` and ``N_HI -> 0`` (logN_HI = -inf) give CERIDWEN's ``Madau1995`` bit for bit.
3. ``jax.grad`` of a CSP prediction (photometry and observed-frame spectrum) with the recipe's
   IGM, w.r.t. ``zred`` and ``logmass``, is finite.

Prospector location: ``$PROSPECTOR_DIR`` (a clone at a78d153); FAIL if unset.

    python examples/recipes/tests/check_igm_damping_dla.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
_src = os.environ.get("PROSPECTOR_DIR", "")
if not os.path.isdir(os.path.join(_src, "prospect")):
    print("FAIL  prospector: PROSPECTOR_DIR unset or not a checkout; set PROSPECTOR_DIR to a clone of https://github.com/bd-j/prospector at a78d153")
    print("FAIL  check_igm_damping_dla")
    sys.exit(1)
sys.path.insert(0, _src)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from ceridwen.cosmology import Cosmology  # noqa: E402
from ceridwen.igm import Madau1995  # noqa: E402
from examples.recipes.igm_damping_dla import MadauDampingDLA  # noqa: E402

TOL = 1e-10
results = []


def report(name, ok, msg):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")


# ---- grids ---------------------------------------------------------------------------------
from ceridwen.ssps.ssp_data import SSPData  # noqa: E402
from ceridwen.csp.csp import CSPBasis  # noqa: E402

SSP_FILE = os.environ.get("CERIDWEN_TEST_SSP",
                          os.path.join(REPO, "ceridwen/data/test_data/ssp_data_bpass.h5"))
ssp = SSPData.load(SSP_FILE)
cosmo_w9 = Cosmology.wmap9()
lt = jnp.linspace(0.0, 0.7, 12)   # Gyr, < age(z=7) under WMAP9
csp = CSPBasis(
    ssp, theta={"lookback_time": lt, "sfh": jnp.ones(12), "logzsol": jnp.array([0.0])},
    cosmo=cosmo_w9, zh_const=True, add_neb=False, add_dust=False, add_diffuse_dust=False,
    add_igm=True, igm_model=MadauDampingDLA(cosmo=cosmo_w9, Ob0=0.04628, x_HI=0.8,
                                            logN_HI=21.5),
    verbose=False,
)
wave_grid = np.asarray(csp.wave, dtype=np.float64)
# fine grid through the Ly-alpha core, never exactly 1215.6696 (Prospector's H is NaN there)
wave_fine = 1100.0 + 0.0137 * np.arange(22000)
waves = {"csp.wave": wave_grid, "fine 1100-1401A": wave_fine}

# ---- 1. vs Prospector ---------------------------------------------------------------------
try:
    import prospect
    from prospect.models import sedmodel as psm
    from prospect.sources.constants import cosmo as p_cosmo
    have_prosp = True
    print(f"Prospector from {os.path.dirname(prospect.__file__)}; cosmo = {p_cosmo.name} "
          f"(h={p_cosmo.h}, Om0={p_cosmo.Om0}, Ob0={p_cosmo.Ob0})")
except Exception as exc:  # pragma: no cover
    have_prosp = False
    report("prospector import", False, f"not importable ({exc!r}); set PROSPECTOR_DIR")

if have_prosp:
    assert abs(p_cosmo.h - cosmo_w9.h) == 0 and abs(p_cosmo.Om0 - cosmo_w9.Om0) == 0
    Ob0 = float(p_cosmo.Ob0)
    for zred in (5.5, 7.0, 9.3):
        for x_HI in (0.3, 1.0, 1.7):
            m = MadauDampingDLA(cosmo=cosmo_w9, Ob0=Ob0, x_HI=x_HI)
            for wname, w in waves.items():
                T_c = np.asarray(jnp.exp(-m.tau_damp(jnp.asarray(w), jnp.asarray(zred))))
                T_p = np.exp(-psm.tau_damping(w, zred, x_HI, zmin=5.0, cosmo=p_cosmo))
                d = np.max(np.abs(T_c - T_p))
                report(f"damping z={zred} x_HI={x_HI} [{wname}]", d <= TOL,
                       f"max|dT| = {d:.3e}, min T = {T_p.min():.4f}")
    # zred <= zmin: Prospector's add_damping_wing skips (sedmodel.py:823); recipe must give T = 1
    m = MadauDampingDLA(cosmo=cosmo_w9, Ob0=Ob0, x_HI=1.0)
    t = np.asarray(m.tau_damp(jnp.asarray(wave_grid), jnp.asarray(4.9)))
    report("damping off at zred=4.9 <= zmin", np.all(t == 0.0), f"max tau = {t.max():.1e}")

    for zred in (3.0, 7.0):
        for logN in (19.0, 20.3, 21.5, 22.5):
            m = MadauDampingDLA(logN_HI=logN)          # z_dla = zred (Prospector default)
            for wname, w in waves.items():
                T_c = np.asarray(jnp.exp(-m.tau_dla(jnp.asarray(w), jnp.asarray(zred))))
                # add_dla (sedmodel.py:808-819) at dla_z = zred: wave unchanged
                T_p = np.exp(-psm.voigt_profile(w, 10 ** logN))
                d = np.max(np.abs(T_c - T_p))
                report(f"DLA z={zred} logN={logN} [{wname}]", d <= TOL, f"max|dT| = {d:.3e}")
    # tau at the |x| = 1e-3 floor for the smallest DLA column (docstring claim)
    tau_floor = float(MadauDampingDLA(logN_HI=20.3).tau_dla(
        jnp.asarray([1215.6696]), jnp.asarray(3.0))[0])
    report("DLA core floor (logN=20.3, x=0)", np.isfinite(tau_floor) and tau_floor > 700,
           f"tau = {tau_floor:.3e}, T = {np.exp(-tau_floor):.1e}")

    # documented difference: foreground absorber
    zs, zd, logN = 7.0, 6.0, 21.0
    w = 1000.0 + 0.0137 * np.arange(36500)   # 1000-1500 A: both troughs inside
    Tc = np.asarray(jnp.exp(-MadauDampingDLA(logN_HI=logN, z_dla=zd).tau_dla(
        jnp.asarray(w), jnp.asarray(zs))))
    Tp = np.exp(-psm.voigt_profile(w * (1 + zd) / (1 + zs), 10 ** logN))
    print(f"INFO  foreground DLA z_dla={zd}, zred={zs}: trough (T min) at rest "
          f"{w[np.argmin(Tc)]:.2f} A (recipe; expected {1215.6696 * (1 + zd) / (1 + zs):.2f}) vs "
          f"{w[np.argmin(Tp)]:.2f} A (Prospector; {1215.6696 * (1 + zs) / (1 + zd):.2f})")

# ---- 2. limits: Madau alone, bit for bit --------------------------------------------------
mad = Madau1995()
for zred in (0.0, 2.0, 6.5, 8.0):
    z = jnp.asarray(zred)
    ref_tau = np.asarray(mad.tau(jnp.asarray(wave_grid), z))
    for fac in (1.0, 0.7):
        ref_T = np.asarray(mad.attenuation(jnp.asarray(wave_grid), z, factor=fac))
        for label, m in (("x_HI=0", MadauDampingDLA(cosmo=cosmo_w9, Ob0=0.04628, x_HI=0.0)),
                         ("logN_HI=-inf", MadauDampingDLA(logN_HI=-np.inf)),
                         ("both", MadauDampingDLA(cosmo=cosmo_w9, Ob0=0.04628, x_HI=0.0,
                                                  logN_HI=-np.inf))):
            T = np.asarray(m.attenuation(jnp.asarray(wave_grid), z, factor=fac))
            tt = np.asarray(m.tau(jnp.asarray(wave_grid), z))
            ok = T.tobytes() == ref_T.tobytes() and tt.tobytes() == ref_tau.tobytes()
            report(f"Madau limit {label} z={zred} factor={fac}", ok,
                   f"max|dT| = {np.max(np.abs(T - ref_T)):.1e} (bytes equal: {ok})")
# the damping kernel itself at x_HI = 0 (not the Python short cut)
from examples.recipes.igm_damping_dla import tau_damping  # noqa: E402
t0 = np.asarray(tau_damping(jnp.asarray(wave_grid), jnp.asarray(7.0), 0.0,
                            cosmo_w9.h, cosmo_w9.Om0, 0.04628))
report("tau_damping kernel at x_HI=0", np.all(t0 == 0.0), f"max|tau| = {np.abs(t0).max():.1e}")

# ---- 3. gradients through the CSP ---------------------------------------------------------
from ceridwen.observation import Photometry  # noqa: E402

phot = Photometry(filters=["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w"],
                  name="phot")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    phot.setup_for_model(csp.wave, zred=7.0)
theta0 = dict(csp.theta_init)


def stat_phot(zred, logmass):
    th = dict(theta0, zred=jnp.atleast_1d(zred), logmass=jnp.atleast_1d(logmass))
    return jnp.sum(jnp.log(csp.predict(th, [phot])["phot"]))


def stat_spec(zred, logmass):
    th = dict(theta0, zred=jnp.atleast_1d(zred), logmass=jnp.atleast_1d(logmass))
    cont = csp.get_spectrum(theta=th, include_lines=False)
    spec, _ = csp._apply_mass_redshift_igm(cont, cont, th)   # csp.py:840-863
    sel = (csp.wave > 1150.0) & (csp.wave < 1400.0)
    return jnp.sum(jnp.where(sel, spec, 0.0)) / jnp.sum(jnp.where(sel, cont, 0.0))


for name, f in (("photometry", stat_phot), ("observed spectrum 1150-1400A", stat_spec)):
    val = jax.jit(f)(7.0, 9.0)
    g = jax.jit(jax.grad(f, argnums=(0, 1)))(7.0, 9.0)
    g = np.array([float(g[0]), float(g[1])])
    report(f"grad {name}", bool(np.all(np.isfinite(g))) and np.isfinite(float(val)),
           f"value = {float(val):.6g}, d/dzred = {g[0]:.6g}, d/dlogmass = {g[1]:.6g}")

# damping actually reaches the prediction: vs Madau-only CSP at z=7
csp_m = CSPBasis(
    ssp, theta={"lookback_time": lt, "sfh": jnp.ones(12), "logzsol": jnp.array([0.0])},
    cosmo=cosmo_w9, zh_const=True, add_neb=False, add_dust=False, add_diffuse_dust=False,
    add_igm=True, igm_model="madau1995", verbose=False)
th = dict(theta0, zred=jnp.array([7.0]), logmass=jnp.array([9.0]))
r = np.asarray(csp.predict(th, [phot])["phot"]) / np.asarray(csp_m.predict(th, [phot])["phot"])
print(f"INFO  F090W/F115W/F150W/F200W flux ratio (x_HI=0.8 + logN=21.5) / Madau-only at z=7: "
      + ", ".join(f"{v:.4f}" for v in r))

n_fail = results.count(False)
print(f"\n{'PASS' if n_fail == 0 else 'FAIL'}  check_igm_damping_dla: "
      f"{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(0 if n_fail == 0 else 1)
