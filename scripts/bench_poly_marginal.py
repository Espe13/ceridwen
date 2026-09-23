"""Timing of the calibration-polynomial modes (G2, 2026-09-23): likelihood-call time vs order.
(a) kernel: the spectrum likelihood given the model mu (n_pix = 2000), per call of a batch
    of W thetas (vmap), jitted, float64, CPU;
(b) full: the fitSED log-posterior (forward model + likelihood) on the test grid.
Median of repeats after a warm-up; the machine is shared with other jobs, so +-20 %."""
import sys, time, warnings
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, jax.numpy as jnp
from ceridwen.likelihood.likelihood import DiagonalGaussianLikelihood
from ceridwen.likelihood.poly_calibration import PolynomialCalibration, chebyshev_design_matrix
from ceridwen.likelihood.poly_marginal import PolyMarginalGaussianLikelihood, PolynomialMarginal
from ceridwen.csp.spectrum_calibration import legendre_design_matrix

ORDERS = [1, 3, 6, 10, 20]
WIDTHS = [1, 32]
for a in sys.argv:
    if a.startswith("--widths="):
        WIDTHS = [int(x) for x in a.split("=", 1)[1].split(",")]
N = 2000
rng = np.random.default_rng(0)
wave = np.linspace(4000., 9000., N)
mask = jnp.asarray(rng.random(N) > 0.05)
mu0 = jnp.asarray(1.0 + 0.3 * np.sin(wave / 300.))
sig = jnp.full(N, 0.02)
y = mu0 * 1.02 + sig * jnp.asarray(rng.standard_normal(N))


def bench(f, args, reps=30):
    out = f(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t)
    return np.median(ts) * 1e6


def kernel_fns(M):
    A = chebyshev_design_matrix(wave, np.asarray(mask), M)
    prof = DiagonalGaussianLikelihood(poly_calibration=PolynomialCalibration(A))
    marg = PolyMarginalGaussianLikelihood(poly_marginal=PolynomialMarginal(A, 0.1))
    P = jnp.asarray(legendre_design_matrix(wave, M))     # sampled: scaling * (1 + P c)
    plain = DiagonalGaussianLikelihood()
    f_prof = lambda s, c: prof(y, mu0 * s, sig, mask)[0]
    f_marg = lambda s, c: marg(y, mu0 * s, sig, mask)[0]
    f_samp = lambda s, c: plain(y, mu0 * s * (1.0 + P @ c), sig, mask)[0]
    return {"profile": f_prof, "marginalize": f_marg, "sampled": f_samp}


rows = []
for W in WIDTHS:
    s = jnp.linspace(0.98, 1.02, W)
    for M in ORDERS:
        c = jnp.zeros((W, M))
        for mode, f in kernel_fns(M).items():
            g = jax.jit(jax.vmap(f))
            rows.append(("kernel", W, M, mode, bench(g, (s, c))))
            gg = jax.jit(jax.vmap(jax.value_and_grad(f, argnums=(0, 1))))
            rows.append(("kernel+grad", W, M, mode, bench(gg, (s, c))))

def table(rows):
    print(f"# backend {jax.default_backend()} {jax.devices()}")
    print("| what | W | M | profile [us] | marginalize [us] | sampled [us] |")
    print("|---|---|---|---|---|---|")
    keys = sorted({(r[0], r[1], r[2]) for r in rows}, key=lambda k: (k[0], k[1], k[2]))
    for k in keys:
        v = {r[3]: r[4] for r in rows if (r[0], r[1], r[2]) == k}
        print(f"| {k[0]} | {k[1]} | {k[2]} | {v['profile']:.0f} | {v['marginalize']:.0f} | "
              f"{v['sampled']:.0f} |", flush=True)


table(rows)
rows = []
if "--full" in sys.argv:
    import os, pathlib
    sys.path.insert(0, os.environ.get("CERIDWEN_TESTS_DIR",
                                      str(pathlib.Path(__file__).resolve().parents[1] / "tests")))
    from _gridfixture import require_test_grid
    from ceridwen import SSPData, CSPBasis, SedModel, Cosmology
    from ceridwen.broadening import Instrument
    from ceridwen.observation import Photometry, Spectrum
    from ceridwen.sampler.priors import Uniform
    from ceridwen.optimize import build_lnprob
    ssp = SSPData.load(str(require_test_grid()))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 9.0, 4), zh_const=True, add_dust=True,
                   add_diffuse_dust=True, add_neb=False, add_igm=False, verbose=False,
                   cosmo=Cosmology.planck18())
    wv = np.linspace(4500., 8500., 1000)
    for M in ORDERS:
        for mode in ("profile", "marginalize", "sampled"):
            kw = {} if mode == "sampled" else dict(polynomial_order=M)
            if mode == "marginalize":
                kw.update(polynomial_mode="marginalize", polynomial_prior_sigma=0.1)
            sp = Spectrum(wavelength=wv, flux=np.full(wv.size, 1e-18),
                          uncertainty=np.full(wv.size, 1e-19),
                          instrument=Instrument.sigma_kms(150.), name="s", **kw)
            ph = Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1e-9] * 2,
                            uncertainty=[1e-10] * 2, name="p")
            init = {"logmass": jnp.array([10.0])}
            pr = {"logmass": Uniform(low=9., high=11.)}
            if mode == "sampled":
                init.update(spectrum_scaling=jnp.array([1.0]), spectrum_calib=jnp.zeros(M))
                pr.update(spectrum_scaling=Uniform(low=.5, high=1.5),
                          spectrum_calib=Uniform(low=-jnp.ones(M), high=jnp.ones(M)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SedModel(csp, [sp, ph], priors=pr, free_param_init=init, zred=0.3)
            f = build_lnprob(m)
            th0 = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
            for W in WIDTHS:
                tb = {k: jnp.stack([v] * W) for k, v in th0.items()}
                tb["logmass"] = tb["logmass"] + jnp.linspace(0, .1, W)[:, None]
                g = jax.jit(jax.vmap(f))
                rows.append(("full", W, M, mode, bench(lambda t: g(t), (tb,), reps=10)))
                gg = jax.jit(jax.vmap(jax.value_and_grad(f)))
                rows.append(("full+grad", W, M, mode, bench(lambda t: gg(t), (tb,), reps=10)))

if rows:
    table(rows)
