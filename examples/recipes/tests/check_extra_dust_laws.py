"""Check for examples/recipes/extra_dust_laws.py (CPU, ~1 min with FSPS, seconds without).

    python examples/recipes/tests/check_extra_dust_laws.py

References (each skipped, and reported as SKIPPED, when not available):
- REQUIRED, FAIL if missing: ``$PROSPECTOR_DIR`` = a clone of https://github.com/bd-j/prospector
  at a78d153 (``prospect/sources/fake_fsps.py``, Reddy+15).
- REQUIRED, FAIL if missing: ``$FSPS_SRC`` = a clone of https://github.com/cconroy20/fsps
  (``dust/Gordon03_table4.dat``, the file FSPS reads for dust_type=5; re-implemented here in
  numpy with FSPS's ``locate`` + ``linterp``, ``src/sps_setup.f90:1374-1394``).
- Optional, SKIPPED if missing: python-fsps with ``$SPS_HOME`` (run in a temporary cwd, since the
  Fortran writes ``fort.NN`` files there): dust_type=5 and 6 taken from the Fortran itself, as
  -ln(F(dust2) / F(0)) / dust2 of an SSP with dust1 = 0 (``src/add_dust.f90:87-104``).
- An SSP grid (``$CERIDWEN_TEST_SSP`` or ``ceridwen/data/test_data/ssp_data_bpass.h5``) for the
  gradient through the CSP spectrum.
"""

import importlib.util
import os
import pathlib
import sys
import warnings

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))

import extra_dust_laws as R  # noqa: E402
from ceridwen.dust.attenuation_laws import ATTENUATION_LAWS  # noqa: E402
from ceridwen.dust.DustModel import Dust, DiffuseDust  # noqa: E402

TOL = 1e-10
RESULTS = []   # (name, status, detail)


def record(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}: {detail}")


def skip(name, why):
    RESULTS.append((name, "SKIPPED", why))
    print(f"SKIPPED  {name}: {why}")


# ------------------------------------------------------------------ registration
names = R.register()
R.register()   # idempotent
record("register", all(n in ATTENUATION_LAWS for n in names)
       and all(set(ATTENUATION_LAWS[n]["params"]) == set(ATTENUATION_LAWS[n]["defaults"])
               for n in names),
       f"registered {names}; params keys == defaults keys")

PAR = {"gordon03_smcbar": "tau_g03smc", "reddy15": "tau_reddy"}
wave = jnp.geomspace(900.0, 5e4, 400)
for law, p in PAR.items():
    tau = 0.7
    direct = ATTENUATION_LAWS[law]["func"](wave, **{p: tau})
    d = Dust(bin_edges=[(-jnp.inf, jnp.inf)], laws=[law])
    via_dust = d.compute_attenuation(wave, {p: jnp.asarray(tau)})[0]
    dd = DiffuseDust(law)
    via_diff = dd.compute_attenuation(wave, {f"diffuse_{p}": jnp.asarray(tau)})
    names_ok = d.get_param_names() == [p] and dd.get_param_names() == [f"diffuse_{p}"]
    diff = max(float(jnp.max(jnp.abs(via_dust - direct))), float(jnp.max(jnp.abs(via_diff - direct))))
    record(f"Dust/DiffuseDust accept '{law}'", names_ok and diff == 0.0,
           f"param names {d.get_param_names()} / {dd.get_param_names()}, "
           f"max |wrapper - direct| = {diff:.1e} (tau={tau} is read)")

# ------------------------------------------------------------------ 5500 A normalisation
w55 = jnp.array([5500.0])
g = float(R.gordon03_smcbar(w55, tau_g03smc=0.7)[0])
record("gordon03_smcbar tau(5500)=tau", abs(g - 0.7) < 1e-15, f"tau(5500)/tau = {g / 0.7:.15f} (intended 1)")
r = float(R.reddy15(w55, tau_reddy=0.7)[0]) / 0.7
r_int = (-5.726 + 4.004 / 0.55 - 0.525 / 0.55**2 + 0.029 / 0.55**3 + 2.505) / 2.505
record("reddy15 tau(5500)=k(0.55um)/2.505", abs(r - r_int) < 1e-15,
       f"tau(5500)/tau = {r:.6f} (intended k(0.55)/R_V = {r_int:.6f}, as FSPS/Prospector dust2)")

# ------------------------------------------------------------------ G03 vs FSPS algorithm in numpy
FIVE_G03 = np.array([1200.0, 1830.0, 2175.0, 5500.0, 15000.0])


def fsps_locate(xx, x):
    """Numerical Recipes locate, 1-based as in FSPS: j with xx(j) <= x < xx(j+1), 0 below."""
    return int(np.searchsorted(xx, x, side="right"))


def fsps_g03(table_path, lam):
    t = np.loadtxt(table_path)
    g03lam = t[::-1, 0] * 1e4           # READ(99,*) g03lam(30-i+1), d1, g03smc(30-i+1)
    g03smc = t[::-1, 2]
    out = np.zeros_like(lam)
    n = g03lam.size
    for i, l in enumerate(lam):
        if l > g03lam[-1]:
            out[i] = 0.0
        elif l < g03lam[0]:
            out[i] = g03smc[0]
        else:
            klo = max(min(fsps_locate(g03lam, l), n - 1), 1) - 1   # to 0-based
            out[i] = g03smc[klo] + (g03smc[klo + 1] - g03smc[klo]) * (l - g03lam[klo]) / (g03lam[klo + 1] - g03lam[klo])
    return out, g03lam, g03smc


sps_home = os.environ.get("SPS_HOME")
fsps_src = os.environ.get("FSPS_SRC")
tab = pathlib.Path(fsps_src) / "dust" / "Gordon03_table4.dat" if fsps_src else None
if tab is not None and tab.is_file():
    ref, g03lam, g03smc = fsps_g03(tab, FIVE_G03)
    mine = np.asarray(R.gordon03_smcbar(jnp.asarray(FIVE_G03)))
    table_same = np.array_equal(g03lam, np.asarray(R._G03_LAM_AA)) and np.array_equal(g03smc, np.asarray(R._G03_Y))
    dmax = float(np.max(np.abs(mine - ref)))
    record("gordon03_smcbar vs FSPS table+linterp (5 lambda)", table_same and dmax < TOL,
           f"embedded table == {tab.name}: {table_same}; lambda={FIVE_G03.tolist()} "
           f"tau={np.round(mine, 6).tolist()}; max|diff| = {dmax:.1e}")
    edge = np.array([1000.0, 1160.0, 21980.0, 22000.0, 30000.0])
    ref_e = fsps_g03(tab, edge)[0]
    mine_e = np.asarray(R.gordon03_smcbar(jnp.asarray(edge)))
    record("gordon03_smcbar edges (blue const, red 0)", float(np.max(np.abs(mine_e - ref_e))) < TOL,
           f"lambda={edge.tolist()} tau={mine_e.tolist()} ref={ref_e.tolist()}")
else:
    record("gordon03_smcbar vs FSPS table+linterp", False,
           f"FSPS_SRC={fsps_src!r}: dust/Gordon03_table4.dat not found; set FSPS_SRC to a clone "
           f"of https://github.com/cconroy20/fsps")

# ------------------------------------------------------------------ Reddy vs Prospector fake_fsps
FIVE_R = np.array([1200.0, 2000.0, 5500.0, 10000.0, 20000.0])
pdir = os.environ.get("PROSPECTOR_DIR")
ff = pathlib.Path(pdir) / "prospect" / "sources" / "fake_fsps.py" if pdir else None
if ff is not None and ff.is_file():
    spec = importlib.util.spec_from_file_location("_prospector_fake_fsps", ff)
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    # a grid with nodes at the branch boundaries 1500 / 6000 / 28500 A and at the 5 test points
    lam = np.unique(np.concatenate([np.arange(1000.0, 30001.0, 10.0), FIVE_R]))
    tau = 0.7
    ext = fake.attenuate(np.ones_like(lam), lam, dust_type=6, dust2=tau)   # exp(-tau*reddy)
    ref_all = -np.log(ext)
    mine_all = np.asarray(R.reddy15(jnp.asarray(lam), tau_reddy=tau))
    idx = np.searchsorted(lam, FIVE_R)
    d5 = float(np.max(np.abs(mine_all[idx] - ref_all[idx])))
    dall = float(np.max(np.abs(mine_all - ref_all)))
    record("reddy15 vs Prospector attenuate(dust_type=6) (5 lambda)", d5 < TOL,
           f"lambda={FIVE_R.tolist()} tau={np.round(mine_all[idx], 6).tolist()}; max|diff| = {d5:.1e}")
    record("reddy15 vs Prospector, whole node-aligned grid", dall < TOL,
           f"{lam.size} pixels 1000-30000 A; max|diff| = {dall:.1e}")
else:
    record("reddy15 vs Prospector", False,
           f"PROSPECTOR_DIR={pdir!r}: prospect/sources/fake_fsps.py not found; set PROSPECTOR_DIR "
           f"to a clone of https://github.com/bd-j/prospector at a78d153")

# ------------------------------------------------------------------ both vs the FSPS Fortran
try:
    import fsps
except Exception as exc:  # noqa: BLE001
    fsps = None
    skip("vs python-fsps dust_type 5/6", f"python-fsps not importable ({exc.__class__.__name__})")
if fsps is not None and sps_home:
    import tempfile
    cwd0 = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)   # the FSPS Fortran writes fort.NN files into the cwd
        try:
            sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, dust_type=5, dust1=0.0, dust2=0.0,
                                        add_neb_emission=False, add_dust_emission=False, imf_type=1)
            tau = 0.7
            w, f0 = sp.get_spectrum(tage=1.0)
            out = {}
            for dt in (5, 6):
                sp.params["dust_type"] = dt
                sp.params["dust2"] = tau
                _, f1 = sp.get_spectrum(tage=1.0)
                out[dt] = -np.log(f1 / f0)
                sp.params["dust2"] = 0.0
        finally:
            os.chdir(cwd0)
    lib = tuple(x.decode() if isinstance(x, bytes) else x for x in sp.libraries)
    # 5 pixels each: G03 inside its table; Reddy inside [1500, 28500) A away from the
    # grid-pixel branch boundaries (the blueward constant is compared separately below)
    def pix(targets):
        return np.array([int(np.argmin(np.abs(w - t))) for t in targets])
    ig = pix([1250.0, 2000.0, 3500.0, 5500.0, 15000.0])
    ir = pix([2000.0, 3500.0, 5500.0, 10000.0, 20000.0])
    dg = float(np.max(np.abs(np.asarray(R.gordon03_smcbar(jnp.asarray(w[ig]), tau_g03smc=tau)) - out[5][ig])))
    dr = float(np.max(np.abs(np.asarray(R.reddy15(jnp.asarray(w[ir]), tau_reddy=tau)) - out[6][ir])))
    record("gordon03_smcbar vs FSPS Fortran dust_type=5 (5 pixels)", dg < TOL,
           f"fsps {fsps.__version__} {lib}, lambda={np.round(w[ig], 3).tolist()}; max|diff| = {dg:.1e}")
    # FSPS writes the Reddy coefficients as default-kind REAL literals (-5.726, 4.004, ...,
    # src/attn_curve.f90:173-182), i.e. single precision, so the Fortran curve is the formula
    # with float32-rounded coefficients. Reproduce that in numpy to 1e-10, and report the
    # (expected, ~1e-7) difference of the exact-decimal port separately.
    c = lambda v: float(np.float32(v))  # noqa: E731
    mic = w[ir] / 1e4
    k32 = np.where(w[ir] < 6000.0,
                   c(-5.726) + c(4.004) / mic - c(0.525) / mic**2 + c(0.029) / mic**3 + c(2.505),
                   c(-2.672) - c(0.010) / mic + c(1.532) / mic**2 + c(-0.412) / mic**3 + c(2.505) - c(0.036221981))
    d32 = float(np.max(np.abs(tau * k32 / c(2.505) - out[6][ir])))
    record("reddy15 formula with FSPS float32 literals vs FSPS Fortran dust_type=6 (5 pixels)", d32 < TOL,
           f"lambda={np.round(w[ir], 3).tolist()}; max|diff| = {d32:.1e}")
    print(f"INFO  reddy15 (exact-decimal coefficients, = Prospector) vs FSPS Fortran on the same pixels: "
          f"max|diff| = {dr:.2e} at tau = {tau} (FSPS single-precision literals; documented difference)")
    # blueward constant: FSPS uses k at the pixel locate(lambda, 1500), not k(1500 A) exactly
    blue = w < 1400.0
    rb = np.asarray(R.reddy15(jnp.asarray(w[blue]), tau_reddy=tau))
    print(f"INFO  reddy15 blueward of 1500 A vs FSPS (grid-pixel boundary, documented difference): "
          f"max|diff| = {np.max(np.abs(rb - out[6][blue])):.2e} (tau = {rb[0]:.6f})")
elif fsps is not None:
    skip("vs python-fsps dust_type 5/6", "SPS_HOME not set")

# ------------------------------------------------------------------ JIT and gradient
for law, p in PAR.items():
    f = jax.jit(lambda t, law=law, p=p: jnp.sum(ATTENUATION_LAWS[law]["func"](wave, **{p: t})))
    gr = jax.grad(f)(0.7)
    gw = jax.grad(lambda wv, law=law: jnp.sum(ATTENUATION_LAWS[law]["func"](wv, 0.7)))(wave)
    record(f"{law} jit + grad finite", bool(np.isfinite(gr)) and bool(jnp.all(jnp.isfinite(gw))),
           f"d sum(tau)/d{p} = {float(gr):.6f}; d/dwave finite on {wave.size} pixels")

grid = os.environ.get("CERIDWEN_TEST_SSP") or str(REPO / "ceridwen" / "data" / "test_data" / "ssp_data_bpass.h5")
if pathlib.Path(grid).is_file():
    from ceridwen import CSPBasis, SSPData
    from ceridwen.cosmology import Cosmology
    ssp = SSPData.load(grid)
    n = 6
    lb = jnp.linspace(0.0, 13.0, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csp = CSPBasis(ssp, theta={"lookback_time": lb, "sfh": jnp.ones(n), "logzsol": jnp.array([0.0])},
                       zh_const=True, add_neb=False, add_dust=True, add_diffuse_dust=True,
                       init_dust_params={"bin_edges": [(-jnp.inf, -1.97)], "laws": ["gordon03_smcbar"]},
                       diffuse_law="reddy15", verbose=False, cosmo=Cosmology.planck18())
    th0 = {"lookback_time": lb, "sfh": jnp.ones(n), "logzsol": jnp.array([0.0]),
           "tau_g03smc": jnp.asarray(0.5), "diffuse_tau_reddy": jnp.asarray(0.3)}

    def total(taus):
        th = dict(th0, tau_g03smc=taus[0], diffuse_tau_reddy=taus[1])
        return jnp.sum(csp.get_spectrum(theta=th, include_lines=False))

    gg = jax.jit(jax.grad(total))(jnp.array([0.5, 0.3]))
    record("grad through CSP spectrum", bool(jnp.all(jnp.isfinite(gg))) and bool(jnp.all(gg < 0)),
           f"{pathlib.Path(grid).name}: d sum(F)/d(tau_g03smc, diffuse_tau_reddy) = "
           f"({float(gg[0]):.4e}, {float(gg[1]):.4e}) (finite, negative)")
else:
    skip("grad through CSP spectrum", f"no SSP grid at {grid}")

# ------------------------------------------------------------------ summary
n_fail = sum(s == "FAIL" for _, s, _ in RESULTS)
n_skip = sum(s == "SKIPPED" for _, s, _ in RESULTS)
n_pass = sum(s == "PASS" for _, s, _ in RESULTS)
print(f"\ncheck_extra_dust_laws: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
print("FAIL" if n_fail else ("PASS" if not n_skip else f"PASS (with {n_skip} SKIPPED: not a full pass)"))
sys.exit(1 if n_fail else 0)
