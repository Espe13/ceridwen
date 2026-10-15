"""Windowed emission-line basis (``SpectralProjector.enable_line_windows``, opt-in).

Each line profile is evaluated on a static window of +-40 of its largest sigma and placed with
one gather + select.  The contract is byte identity with the dense ``line_basis``:

1. the underflow the windows rely on: exp(-x^2/2) is exactly 0 in float64 beyond |x| = 38.61
   (NumPy) and 37.65 (XLA CPU, subnormals flushed), so 40 sigma leaves a margin;
2. windowed == dense (``assert_array_equal``) under jit + vmap over random sigma_gas (including
   0, sigma_max and a value the clip brings back), LSF scale and redshift draws, for a
   grating-like and a prism-like spectrum, with and without an instrument, and with windows
   clipped at both spectrum edges;
3. what stays dense: the default (off), ``eline_delta_zred`` in theta, a window as wide as the
   spectrum;
4. the full marginalised likelihood: lnL bitwise equal at the production batch width, and its
   gradient finite and equal to 1e-10 (measured: 1-2 ulp; the reverse pass of the windowed basis
   sums over the window, the dense one over every pixel) (needs the test grid and SPS_HOME).
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
import warnings

import numpy as np
import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ceridwen.broadening import (SpectralProjector, Kinematics, Instrument,  # noqa: E402
                                 _LINE_WINDOW_NSIGMA, _line_windows)
from ceridwen import SSPData, CSPBasis, SedModel, Cosmology            # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh             # noqa: E402
from ceridwen.observation import Spectrum                              # noqa: E402
from ceridwen.priors import Uniform                                    # noqa: E402
from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter       # noqa: E402
from _gridfixture import find_test_grid                                # noqa: E402

WAVE_MODEL = np.geomspace(300.0, 40000.0, 30000)
_rng = np.random.default_rng(3)
# random lines, a blended doublet, and lines at both ends of each spectrum (windows clipped)
LINES = np.sort(np.concatenate([_rng.uniform(900.0, 12000.0, 60),
                                [3727.1, 3729.9, 4862.7, 5008.2, 6564.6, 6585.3,
                                 9700.0 / 3.0 + 0.5, 18900.0 / 3.0 - 0.5,
                                 6000.0 / 3.0 + 1.0, 53000.0 / 3.0 - 1.0]]))
GRATING = (np.geomspace(9700.0, 18900.0, 4000), 1000.0)
PRISM = (np.geomspace(6000.0, 53000.0, 400), 100.0)
ZRED = 2.0
W = 64


def _projector(wave, R, *, sigma_max=500.0, gas="sigma_gas", mode="fixed", inst=True):
    ins = None
    if inst:
        ins = (Instrument.R_fwhm(R, scale="lsf_scale", scale_range=(0.8, 1.3))
               if mode == "scale" else Instrument.R_fwhm(R))
    zr = (1.9, 2.1) if mode == "free_z" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectralProjector.build(
            Kinematics(sigma_gal=100.0, sigma_gas=gas, sigma_max=sigma_max), ins, WAVE_MODEL,
            wave, ZRED, line_wave_rest=LINES, zred_range=zr)


def _draws(p, sigma_max, seed=0):
    rng = np.random.default_rng(seed)
    sg = rng.uniform(0.0, sigma_max, W)
    sg[:3] = [0.0, sigma_max, 1.7 * sigma_max]              # the last one is clipped
    opz = np.full(W, p.opz_ref)
    if p.free_z:
        opz = 1.0 + rng.uniform(*p.zred_range, W)
        opz[:2] = [1.0 + p.zred_range[0], 1.0 + p.zred_range[1]]
    sc = None
    if p.free_inst_scale:
        sc = rng.uniform(*p.inst_scale_range, W)
        sc[:2] = p.inst_scale_range
    return sg, opz, sc


def _both(p, sigma_max, sg, opz, sc):
    """(dense, windowed) bases at every draw, under jit + vmap, sigma_gas clipped as
    Kinematics.resolve clips it."""
    def dense(a, o, c=None):
        return p.line_basis(jnp.clip(a, 0.0, sigma_max), o, c)

    def win(a, o, c=None):
        return p._line_basis_windowed(jnp.clip(a, 0.0, sigma_max), o, c)
    args = (sg, opz) if sc is None else (sg, opz, sc)
    return (np.asarray(jax.jit(jax.vmap(dense))(*args)),
            np.asarray(jax.jit(jax.vmap(win))(*args)))


def test_exp_underflows_to_exactly_zero_before_the_window_edge():
    x = np.linspace(37.0, 41.0, 400001)
    for z in (np.exp(-0.5 * x * x), np.asarray(jax.jit(lambda t: jnp.exp(-0.5 * t * t))(x))):
        first = int(np.flatnonzero(z == 0.0)[0])
        assert np.all(z[first:] == 0.0)
        assert x[first] < 38.61 < _LINE_WINDOW_NSIGMA
    # the half-width as computed from a sigma: x = (40 s) / s is not below 40 by more than an ulp
    s = 10.0 ** np.random.default_rng(0).uniform(-6.0, -1.0, 100000)
    x = (_LINE_WINDOW_NSIGMA * s) / s
    assert x.min() > 39.99 and np.all(np.exp(-0.5 * x * x) == 0.0)


@pytest.mark.parametrize("sigma_max", [2000.0, 500.0])
@pytest.mark.parametrize("mode", ["fixed", "free_z", "scale"])
@pytest.mark.parametrize("spec", ["grating", "prism", "grating_noinst"])
def test_windowed_basis_equals_the_dense_basis_bitwise(spec, mode, sigma_max):
    wave, R = PRISM if spec == "prism" else GRATING
    p = _projector(wave, R, sigma_max=sigma_max, mode=mode, inst=spec != "grating_noinst")
    assert p.line_window is None                         # off by default
    win = p.enable_line_windows()
    assert win is not None and win[1] < wave.size
    dense, windowed = _both(p, sigma_max, *_draws(p, sigma_max))
    np.testing.assert_array_equal(windowed, dense)
    assert dense.tobytes() == windowed.tobytes()         # +0.0 outside, as exp gives
    # the windows really do cut: most of the basis is outside them, and they hold every nonzero
    start, width = win
    rows = np.arange(wave.size)[:, None]
    outside = (rows < start[None, :]) | (rows >= start[None, :] + width)
    assert np.all(dense[:, outside] == 0.0)
    if spec == "prism" or sigma_max == 500.0:
        assert width < wave.size // 2


def test_window_clipped_at_both_edges():
    p = _projector(*GRATING, mode="free_z")
    start, width = p.enable_line_windows()
    n = GRATING[0].size
    assert start.min() == 0 and start.max() == n - width
    dense, windowed = _both(p, 500.0, *_draws(p, 500.0, seed=1))
    np.testing.assert_array_equal(windowed, dense)


def test_fixed_sigma_gas_gives_narrower_windows():
    free = _projector(*GRATING, gas="sigma_gas").enable_line_windows()[1]
    p = _projector(*GRATING, gas=80.0)
    fixed = p.enable_line_windows()[1]
    assert fixed < free
    # both jitted (eager op-by-op evaluation of the dense basis itself differs from its jitted
    # fusion in the last bit), with the fixed width a constant, as Kinematics.resolve gives it
    a = np.asarray(jax.jit(lambda: p._line_basis_windowed(jnp.asarray(80.0), p.opz_ref))())
    b = np.asarray(jax.jit(lambda: p.line_basis(jnp.asarray(80.0), p.opz_ref))())
    np.testing.assert_array_equal(a, b)


def test_no_window_when_it_would_cover_the_spectrum():
    wave = np.geomspace(9700.0, 10100.0, 200)              # 40 sigma_max >> the spectrum
    p = _projector(wave, 1000.0, sigma_max=2000.0)
    assert p.enable_line_windows() is None and p.line_window is None
    no_lines = SpectralProjector.build(Kinematics(sigma_gal=100.0), Instrument.R_fwhm(1000.0),
                                       WAVE_MODEL, GRATING[0], ZRED)
    assert no_lines.enable_line_windows() is None


def _spec_rest(p):
    return jnp.asarray(np.interp(WAVE_MODEL, WAVE_MODEL, 1.0 + 0.1 * np.sin(WAVE_MODEL / 300.0)))


def test_predict_with_line_basis_bitwise_and_eline_delta_zred_stays_dense(monkeypatch):
    p = _projector(*GRATING, mode="scale")
    flux = jnp.asarray(np.random.default_rng(2).uniform(0.0, 1e-17, LINES.size))
    fit_pos = np.arange(0, p.line_idx.size, 3)
    sg, _, sc = _draws(p, 500.0, seed=4)
    th = {"sigma_gas": jnp.asarray(sg[:, None]), "lsf_scale": jnp.asarray(sc[:, None])}
    sr = _spec_rest(p)

    def run():
        return jax.jit(jax.vmap(lambda t: p.predict_with_line_basis(sr, flux, t, fit_pos)))(th)
    pred_d, A_d = run()
    p.enable_line_windows()
    pred_w, A_w = run()
    np.testing.assert_array_equal(np.asarray(pred_w), np.asarray(pred_d))
    np.testing.assert_array_equal(np.asarray(A_w), np.asarray(A_d))

    def boom(*a, **k):
        raise AssertionError("windowed basis used with eline_delta_zred")
    monkeypatch.setattr(p, "_line_basis_windowed", boom)
    th1 = {"sigma_gas": jnp.array([100.0]), "lsf_scale": jnp.array([1.0]),
           "eline_delta_zred": jnp.array([3e-3])}
    p.predict_with_line_basis(sr, flux, th1, fit_pos)     # dense: no AssertionError
    with pytest.raises(AssertionError):
        p.predict_with_line_basis(sr, flux, {"sigma_gas": jnp.array([100.0]),
                                             "lsf_scale": jnp.array([1.0])}, fit_pos)


def test_gradients_finite_and_equal_to_the_dense_ones():
    p = _projector(*GRATING, mode="free_z")
    flux = jnp.asarray(np.random.default_rng(5).uniform(0.0, 1.0, p.line_idx.size))
    wts = jnp.asarray(np.random.default_rng(6).standard_normal(GRATING[0].size))

    def f(fn):
        return lambda a, o: jnp.dot(wts, fn(a, o) @ flux)
    for sg, opz in [(0.0, 3.0), (250.0, 2.95), (499.9, 3.1), (120.0, 2.9)]:
        gd = jax.grad(f(p.line_basis), argnums=(0, 1))(sg, opz)
        p.enable_line_windows()
        gw = jax.jit(jax.grad(f(p._line_basis_windowed), argnums=(0, 1)))(sg, opz)
        assert all(np.isfinite(float(g)) for g in gw)
        np.testing.assert_allclose([float(g) for g in gw], [float(g) for g in gd],
                                   rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# the full model (test grid + CLOUDY)
# ---------------------------------------------------------------------------

_N_TIME = 7
_WORKLOADS = {"grating": GRATING, "prism": PRISM}


def _num_delete_default() -> int:
    num_live = inspect.signature(BlackJAXNestedSamplerAdapter.__init__).parameters["num_live"].default
    return max(1, int(num_live) // 5)


def _build_csp():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (see tests/_gridfixture.py)")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY nebular grids)")
    cosmo = Cosmology.planck18()
    return CSPBasis(SSPData.load(str(path)),
                    lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), _N_TIME),
                    zh_const=True, sfh_interp="step", add_dust=True, add_diffuse_dust=True,
                    add_neb=True, add_igm=False, verbose=False, cosmo=cosmo,
                    sps_home=os.environ["SPS_HOME"])


def _build_model(csp, wave, R, sigma_max):
    t = np.array(csp.sfh_times)
    init = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([10.0]),
            "sigma_gas": jnp.array([0.3 * sigma_max])}
    tr = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t),
          # marginalised line fluxes: the nebular parameters are fixed (eline_marginal refuses
          # them sampled)
          "gas_logu": lambda th: jnp.array([-2.5]), "gas_logz": lambda th: jnp.array([-0.3])}
    n = wave.size
    spec = Spectrum(wavelength=wave, flux=np.ones(n), uncertainty=np.full(n, 0.1),
                    instrument=Instrument.R_fwhm(R), name="spec", marginalize_elines=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, [spec], transforms=tr, free_param_init=init, zred=ZRED,
                         priors={"sigma_gas": Uniform(low=0.0, high=sigma_max)},
                         kinematics=Kinematics(sigma_gal=150.0, sigma_gas="sigma_gas",
                                               sigma_max=sigma_max))
    y = np.asarray(model.predict({k: jnp.asarray(v) for k, v in model.theta_init.items()})["spec"])
    rng = np.random.default_rng(0)
    e = 0.05 * np.abs(y) + 1e-3 * np.median(np.abs(y))
    spec.flux = jnp.asarray(y + e * rng.standard_normal(n))
    spec.uncertainty = jnp.asarray(e)
    return model


def _bench_draws(model, W, sigma_max, seed):
    rng = np.random.default_rng(seed)
    th = {}
    for k, v in model.theta_init.items():
        base = np.asarray(v, dtype=float)
        th[k] = base[None, :] + 0.05 * rng.standard_normal((W,) + base.shape)
    th["sigma_gas"] = rng.uniform(0.0, sigma_max, (W, 1))
    return {k: jnp.asarray(v) for k, v in th.items()}


@pytest.fixture(scope="module")
def bench():
    if find_test_grid() is None:
        pytest.skip("no test SSP grid (see tests/_gridfixture.py)")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY nebular grids)")
    return _build_csp()


@pytest.mark.fsps
@pytest.mark.parametrize("workload", ["grating", "prism"])
def test_marginalised_likelihood_bitwise_at_production_width(bench, workload):
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood import MultiObservationLikelihood
    csp = bench
    wave, R = _WORKLOADS[workload]
    smax = 500.0
    model = _build_model(csp, wave, R, smax)
    proj = model.obs_dict["spec"]._proj
    lh = MultiObservationLikelihood(
        keys=tuple(model.obs_dict),
        likelihoods=tuple(_likelihood_for(o, model.param_names) for o in model.observations))
    th = _bench_draws(model, _num_delete_default(), smax, seed=11)
    out = {}
    for arm in ("dense", "windowed"):
        if arm == "windowed":
            assert proj.enable_line_windows() is not None
        f = lh.make_lnprobfn(model.obs_dict, model, model)
        val = np.asarray(jax.jit(jax.vmap(f))(th))
        one = {k: v[0] for k, v in th.items()}
        g = jax.jit(jax.grad(f))(one)
        out[arm] = (val, g)
    assert np.all(np.isfinite(out["dense"][0]))
    assert out["windowed"][0].tobytes() == out["dense"][0].tobytes()
    for k in out["dense"][1]:
        gw, gd = np.asarray(out["windowed"][1][k]), np.asarray(out["dense"][1][k])
        assert np.all(np.isfinite(gw)), k
        np.testing.assert_allclose(gw, gd, rtol=1e-10, atol=0.0, err_msg=k)
