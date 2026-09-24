"""Free instrumental LSF scale: ``Instrument(..., scale=1.0 | float | "<theta key>")``.

sigma_inst -> s * sigma_inst in the continuum kernel AND the line widths
(``ceridwen/broadening.py``).  What is checked:

1. the default (and an explicit ``scale=1.0``) is byte-identical to the no-scale model
   (Spectrum, Photometry and Lines predictions; the static response arrays);
2. a fixed ``scale=s`` equals the Instrument built with the width multiplied by s
   (an independent construction), byte-identically for ``sigma_kms``;
3. a sampled scale at ``theta[key] = s`` equals the fixed-``s`` model built on the same
   log grid.  By design the only difference is the band: the sampled one is sized for the
   top of the range, so at s < s_max it keeps Gaussian tail the fixed band drops (relative
   mass < erfc((5 sigma + 1) / (sqrt 2 sigma)) < erfc(5 / sqrt 2) = 5.7e-7 per row), so the
   spectrum agrees to rtol SPEC_RTOL = 2 x 5.7e-7 (measured: < 1e-9) and to 1e-12 at
   s = s_max; photometry and line fluxes never see the instrument and agree to 1e-12.
   Checked with a fixed and a sampled redshift;
4. differentiability of the fitSED log-posterior (fit._likelihood_for +
   MultiObservationLikelihood + make_lnprobfn, as ``examples/recipes/map_fit.py``):
   finite at the prior centre, near the bounds and with masked / non-finite data;
   equal to central finite differences; jit(value_and_grad) and its vmap equal the loop;
5. the construction-time guards.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid                                   # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                # noqa: E402
from ceridwen.broadening import (Kinematics, Instrument, ScaledResponse,   # noqa: E402
                                 build_response, check_scale_range)
from ceridwen.observation import Photometry, Spectrum, Lines              # noqa: E402
from ceridwen.priors import Uniform, Normal                               # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh                # noqa: E402

ZRED = 0.8
N_TIME = 5
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.exp(np.arange(np.log(4400.0), np.log(7000.0), 90.0 / 2.99792458e5)) * (1 + ZRED)
LINES = {"Hbeta": 4862.71, "OIII5007": 5008.24, "Halpha": 6564.61}
RANGE = (0.7, 1.4)
SPEC_RTOL = 2 * 5.8e-7        # 2 x erfc(5 / sqrt 2): the fixed band's dropped tail, see 3.


@pytest.fixture(scope="module")
def csp():
    return _make_csp()


def _make_csp():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    ssp = SSPData.load(str(path))
    cosmo = Cosmology.planck18()
    return CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), N_TIME),
                    zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                    add_neb=True, add_igm=False, verbose=False, cosmo=cosmo,
                    sps_home=os.environ["SPS_HOME"])


def _obs(ins, flux=None):
    n = SPEC_WAVE.size
    return [Spectrum(wavelength=SPEC_WAVE, flux=np.ones(n) if flux is None else flux,
                     uncertainty=np.full(n, 0.1), instrument=ins, name="spec"),
            Photometry(filters=FILTERS, flux=[1.0] * len(FILTERS),
                       uncertainty=[0.1] * len(FILTERS), name="phot"),
            Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                  wavelength=np.array(list(LINES.values())), flux=np.ones(len(LINES)),
                  uncertainty=np.full(len(LINES), 0.1), name="lines")]


def _model(csp, ins, kin=None, priors=None, init=None, obs=None, **kw):
    t = np.array(csp.sfh_times)

    def _sfh(th, _t=t):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)
    fi = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    if isinstance(getattr(ins, "scale", 1.0), str):
        fi[ins.scale] = jnp.array([1.0])
    fi.update(init or {})
    if priors is None and isinstance(getattr(ins, "scale", 1.0), str):
        priors = {ins.scale: Uniform(low=RANGE[0], high=RANGE[1])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, obs if obs is not None else _obs(ins), priors=priors or {},
                        transforms={"sfh": _sfh}, free_param_init=fi, zred=ZRED,
                        kinematics=kin or Kinematics(sigma_gal=150.0, sigma_gas=80.0), **kw)


def _theta(model, **kw):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th.update({k: jnp.array([float(v)]) for k, v in kw.items()})
    return th


def _pred(model, **kw):
    return {k: np.asarray(v) for k, v in model.predict(_theta(model, **kw)).items()}


# --------------------------------------------------------------------------------------
# 1. default and scale=1.0: byte-identical
# --------------------------------------------------------------------------------------
def test_default_scale_is_byte_identical(csp):
    ref = _model(csp, Instrument.R_fwhm(1500.0))
    p_ref = _pred(ref)
    for ins in (Instrument.R_fwhm(1500.0, scale=1.0), Instrument.R_fwhm(1500.0, scale=1),
                Instrument("R_fwhm", 1500.0, None, 1.0)):
        m = _model(csp, ins)
        p = _pred(m)
        assert set(p) == {"spec", "phot", "lines"}
        for k in p_ref:
            assert p[k].dtype == p_ref[k].dtype and np.array_equal(p[k], p_ref[k]), k
        a, b = ref.observations[0]._proj, m.observations[0]._proj
        assert np.array_equal(np.asarray(a.J), np.asarray(b.J))
        assert np.array_equal(np.asarray(a.W), np.asarray(b.W))
        assert not b.free_inst_scale and b.scaled is None
    assert ref.observations[0].instrument.scale == 1.0


# --------------------------------------------------------------------------------------
# 2. fixed s != 1 equals the Instrument with the width multiplied by s
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("s", [0.85, 1.23])
def test_fixed_scale_equals_scaled_instrument(csp, s):
    a = _pred(_model(csp, Instrument.sigma_kms(140.0, scale=s)))
    b = _pred(_model(csp, Instrument.sigma_kms(140.0 * s)))
    for k in a:                                    # sigma_kms: s * 140 either way, bitwise
        assert np.array_equal(a[k], b[k]), k
    a = _pred(_model(csp, Instrument.R_fwhm(1500.0, scale=s)))
    b = _pred(_model(csp, Instrument.R_fwhm(1500.0 / s)))
    for k in a:                                    # (c/R) * s vs c/(R/s): 1 ulp in the width
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=0, err_msg=k)
    wv = np.linspace(7500.0, 13000.0, 12)          # wavelength-dependent width in km/s
    fw = np.linspace(4.0, 6.0, 12)
    a = _pred(_model(csp, Instrument.fwhm_aa(fw, wave=wv, scale=s)))
    b = _pred(_model(csp, Instrument.fwhm_aa(fw * s, wave=wv)))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=0, err_msg=k)
    assert not np.array_equal(a["spec"], _pred(_model(csp, Instrument.fwhm_aa(fw, wave=wv)))["spec"])


# --------------------------------------------------------------------------------------
# 3. sampled scale at s equals the fixed-s model on the same log grid
# --------------------------------------------------------------------------------------
def _kin_matched(sampled_proj, s, sigma_max=2000.0):
    """Kinematics whose sigma_max gives the fixed-s projector the sampled one's window
    margin (sigma_max + max sigma_fix(s_hi)), hence the same log grid."""
    p = sampled_proj
    fix = lambda sc: np.sqrt(np.clip((sc * p.sigma_inst_kms) ** 2 - p.sigma_lib_kms ** 2,
                                     0.0, None)).max()
    return Kinematics(sigma_gal=150.0, sigma_gas=80.0,
                      sigma_max=sigma_max + fix(p.inst_scale_range[1]) - fix(s))


@pytest.mark.parametrize("make", [
    lambda sc: Instrument.sigma_kms(300.0, scale=sc),
    lambda sc: Instrument.fwhm_aa(np.linspace(4.0, 6.0, 12), wave=np.linspace(7500.0, 13000.0, 12),
                                  scale=sc)], ids=["sigma_kms", "fwhm_aa_array"])
def test_sampled_scale_matches_fixed(csp, make):
    ms = _model(csp, make("lsf_scale"))
    P = ms.observations[0]._proj
    assert P.free_inst_scale and P.inst_scale_range == RANGE
    for s in (RANGE[0] + 1e-3, 1.0, 1.17, RANGE[1]):
        # the spectrum against the fixed-s model on the same log grid (its Kinematics'
        # sigma_max is shifted for that, which also moves the photometric broadener's grid)
        mf = _model(csp, make(s), kin=_kin_matched(P, s))
        Q = mf.observations[0]._proj
        assert Q.grid.n == P.grid.n and Q.grid.n_pad == P.grid.n_pad
        np.testing.assert_allclose(Q.grid.lnw, P.grid.lnw, rtol=1e-15, atol=0)
        a, b = _pred(ms, lsf_scale=s), _pred(mf)
        tol = SPEC_RTOL if s < RANGE[1] else 1e-12
        np.testing.assert_allclose(a["spec"], b["spec"], rtol=tol, atol=0, err_msg=f"s={s}")
        print(f"s={s}: max rel spec diff {np.max(np.abs(a['spec'] / b['spec'] - 1)):.2e}")
        # photometry and line fluxes never see the instrument: bitwise equal to the fixed-s
        # model with the same Kinematics
        c = _pred(_model(csp, make(s)))
        for k in ("phot", "lines"):
            assert np.array_equal(a[k], c[k]), (k, s)
    # the line path through predict_with_line_basis (eline_delta_zred in theta)
    th = _theta(ms, lsf_scale=1.17)
    th2 = dict(th, eline_delta_zred=jnp.array([0.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the two line painters order their products differently (pre-existing): 1e-10
        np.testing.assert_allclose(np.asarray(ms.predict(th2)["spec"]),
                                   np.asarray(ms.predict(th)["spec"]), rtol=1e-10, atol=0)


def test_sampled_scale_matches_fixed_free_z(csp):
    pri = {"zred": Uniform(low=0.75, high=0.85),
           "lsf_scale": Uniform(low=RANGE[0], high=RANGE[1])}
    ms = _model(csp, Instrument.R_fwhm(1200.0, scale="lsf_scale"), priors=pri,
                init={"zred": jnp.array([ZRED])})
    P = ms.observations[0]._proj
    assert P.free_z and P.free_inst_scale
    for s, z in ((0.9, 0.78), (1.3, 0.83)):
        mf = _model(csp, Instrument.R_fwhm(1200.0, scale=s), kin=_kin_matched(P, s),
                    priors={"zred": pri["zred"]}, init={"zred": jnp.array([ZRED])})
        a, b = _pred(ms, lsf_scale=s, zred=z)["spec"], _pred(mf, zred=z)["spec"]
        np.testing.assert_allclose(a, b, rtol=SPEC_RTOL, atol=0, err_msg=f"s={s}, z={z}")
        print(f"free z, s={s}, z={z}: max rel spec diff {np.max(np.abs(a / b - 1)):.2e}")


def test_scaled_response_weights_equal_build_response():
    """ScaledResponse at s_max reproduces build_response bitwise-equivalently on the same
    grid (same band), and at s < s_max differs only by the Gaussian tail beyond the fixed
    band: < erfc(5 / sqrt 2) = 5.7e-7 per weight."""
    from ceridwen.broadening import LogGrid
    wm = np.exp(np.linspace(np.log(3000.0), np.log(9000.0), 20000))
    grid = LogGrid.build(wm, 4000.0, 8000.0, 500.0)
    lnw = np.log(np.exp(np.arange(np.log(4100.0), np.log(7900.0), 70.0 / 2.99792458e5)))
    s_inst = np.linspace(60.0, 160.0, lnw.size)          # km/s, varies along the spectrum
    s_lib = np.full(lnw.size, 55.0)
    sr = ScaledResponse(grid, lnw, s_inst, s_lib, 1.5)
    for s in (0.8, 1.0, 1.5):
        s_fix = np.sqrt(np.clip((s * s_inst) ** 2 - s_lib ** 2, 0.0, None))
        J, W = build_response(grid, lnw, s_fix)
        dense_ref = np.zeros((lnw.size, grid.n))
        np.add.at(dense_ref, (np.arange(lnw.size)[:, None], J), W)
        for Ws in (sr.weights_np(s), np.asarray(sr.weights(jnp.asarray(s)))):
            dense = np.zeros_like(dense_ref)
            np.add.at(dense, (np.arange(lnw.size)[:, None], sr.J_np), Ws)
            tol = 1e-15 if s == 1.5 else 5.8e-7
            assert np.max(np.abs(dense - dense_ref)) < tol, s


# --------------------------------------------------------------------------------------
# 4. differentiability of the fitSED log-posterior
# --------------------------------------------------------------------------------------
PRIORS_BOUNDED = {"logzsol": (-1.0, 0.2), "gas_logz": (-1.0, 0.2), "gas_logu": (-3.5, -1.5),
                  "diffuse_tau_kc": (0.0, 2.0), "diffuse_dust_index": (-1.0, 0.4),
                  "logsfr_ratios": (-3.0, 3.0), "logmass": (9.0, 11.0),
                  "lsf_scale": RANGE, "sigma_gal": (50.0, 400.0)}


def _build_lnprob(model):
    """As examples/recipes/map_fit.build_lnprob (fitSED's likelihood path)."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood.likelihood import MultiObservationLikelihood, make_lnprobfn
    obs = model.obs_dict
    keys = tuple(obs)
    lh = tuple(_likelihood_for(obs[k], model.param_names, model=model) for k in keys)
    return make_lnprobfn(obs, model, model, MultiObservationLikelihood(keys=keys, likelihoods=lh))


@pytest.fixture(scope="module")
def grad_model(csp):
    return _make_grad_model(csp)


def _make_grad_model(csp):
    """Every CSP parameter, sigma_gal and the LSF scale free, bounded priors; data = the
    model at a reference theta + 3 % noise, with masked and non-finite pixels."""
    kin = Kinematics(sigma_gal="sigma_gal", sigma_gas=80.0)
    ins = Instrument.sigma_kms(300.0, scale="lsf_scale")
    m0 = _model(csp, ins, kin=kin, init={"sigma_gal": jnp.array([180.0])})
    ref = _pred(m0, lsf_scale=1.1)
    rng = np.random.default_rng(3)
    flux = ref["spec"] * (1 + 0.03 * rng.standard_normal(ref["spec"].size))
    flux[5] = np.nan                                    # non-finite datum
    unc = 0.03 * np.abs(ref["spec"])
    unc[9] = np.inf                                     # non-finite uncertainty
    mask = np.ones(flux.size, bool)
    mask[20:30] = False                                 # masked pixels
    obs = _obs(ins)
    obs[0] = Spectrum(wavelength=SPEC_WAVE, flux=flux, uncertainty=unc, mask=mask,
                      instrument=ins, name="spec")
    obs[1] = Photometry(filters=FILTERS, flux=ref["phot"] * 1.02,
                        uncertainty=0.05 * np.abs(ref["phot"]), name="phot")
    obs[2] = Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                   wavelength=np.array(list(LINES.values())), flux=ref["lines"] * 0.97,
                   uncertainty=0.05 * np.abs(ref["lines"]), name="lines")
    m = _model(csp, ins, kin=kin, init={"sigma_gal": jnp.array([180.0])}, obs=obs)
    m.priors = {k: Uniform(low=lo, high=hi) for k, (lo, hi) in PRIORS_BOUNDED.items()
                if k in m.param_names}
    assert set(m.priors) == set(m.param_names), (sorted(m.param_names), sorted(m.priors))
    return m


def _point(model, frac):
    """theta at fraction ``frac`` of every prior interval (0.5 = centre)."""
    th = {}
    for k, v in model.theta_init.items():
        lo, hi = PRIORS_BOUNDED[k]
        th[k] = jnp.full(np.shape(v), lo + frac * (hi - lo))
    return th


def test_gradient_finite_everywhere(grad_model):
    vg = jax.jit(jax.value_and_grad(_build_lnprob(grad_model)))
    for frac in (0.5, 1e-3, 1.0 - 1e-3):
        val, g = vg(_point(grad_model, frac))
        assert np.isfinite(float(val)), frac
        for k, v in g.items():
            assert np.all(np.isfinite(np.asarray(v))), (k, frac)
        assert float(jnp.abs(g["lsf_scale"][0])) > 0.0, frac


def _off_node(model):
    """A theta between grid nodes (piecewise-linear interpolation in logzsol, gas_logz and
    gas_logu has kinks at the nodes, where a central difference averages two slopes) and
    near the data-generating point."""
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th.update(logzsol=jnp.array([-0.23]), gas_logz=jnp.array([-0.17]),
              gas_logu=jnp.array([-2.27]), diffuse_tau_kc=jnp.array([0.43]),
              diffuse_dust_index=jnp.array([-0.21]), logmass=jnp.array([10.03]),
              logsfr_ratios=jnp.array([0.1, -0.2, 0.15, 0.05]),
              lsf_scale=jnp.array([1.13]), sigma_gal=jnp.array([171.0]))
    return th


def test_gradient_matches_finite_differences(grad_model):
    """Central differences, h = 1e-4 max(1, |x|).  The LSF scale acts after the CSP, in
    float64 only: agreement < 1e-6.  The CSP parameters go through float32 stages of the
    forward model (the SSP einsums), whose rounding limits a central difference to ~1e-4
    relative here (measured on the pre-change code too): < 2e-3."""
    lnp = jax.jit(_build_lnprob(grad_model))
    th = _off_node(grad_model)
    g = jax.grad(lnp)(th)
    worst = {}
    for k, v in th.items():
        v = np.asarray(v, dtype=float)
        for i in range(v.size):
            # sigma_gal (~170 km/s): a 0.017 km/s step is float32 noise with the ZAU_ND line
            # strengths (AD 0.50133; FD 0.5132 at h 0.01, 0.50120 at h 1: rel 1.3e-4, 2026-09-24)
            h = 1.0 if k == "sigma_gal" else 1e-4 * max(1.0, abs(v.flat[i]))
            up, dn = v.copy(), v.copy()
            up.flat[i] += h
            dn.flat[i] -= h
            fd = (float(lnp({**th, k: jnp.asarray(up)}))
                  - float(lnp({**th, k: jnp.asarray(dn)}))) / (2 * h)
            ad = float(np.asarray(g[k]).flat[i])
            rel = abs(ad - fd) / max(abs(fd), 1.0)   # absolute below |d lnP / dx| = 1
            worst[k] = max(worst.get(k, 0.0), rel)
    print("max relative |grad - central FD| per parameter:",
          {k: f"{v:.1e}" for k, v in worst.items()})
    assert worst["lsf_scale"] < 1e-6, worst
    assert max(worst.values()) < 2e-3, worst


def test_jit_vmap_equal_loop(grad_model):
    """vmap of jit(value_and_grad) over 4 thetas equals the loop.  Not bitwise: the batched
    and unbatched XLA programs reduce the float32 SSP einsums in a different order.  Measured
    here 1.6e-8 (ln P, |ln P| ~ 3e5) and 4.8e-6 (gradient, relative above |g| = 1); the same
    model with the scale FIXED (static response) gives 1.7e-8 and 7.0e-6, so the level is
    the forward model's, not the feature's.  Asserted: 1e-7 and 1e-4."""
    lnp = _build_lnprob(grad_model)
    vg = jax.jit(jax.value_and_grad(lnp))
    pts = []
    for s, f in zip((0.75, 0.95, 1.2, 1.38), (0.9, 1.0, 1.05, 1.1)):
        p = _off_node(grad_model)
        p["lsf_scale"] = jnp.array([s])
        p["sigma_gal"] = p["sigma_gal"] * f
        pts.append(p)
    batch = {k: jnp.stack([p[k] for p in pts]) for k in pts[0]}
    vals, grads = jax.jit(jax.vmap(jax.value_and_grad(lnp)))(batch)
    dv = dg = 0.0
    for i, p in enumerate(pts):
        v, g = vg(p)
        dv = max(dv, abs(float(vals[i]) / float(v) - 1))
        np.testing.assert_allclose(float(vals[i]), float(v), rtol=1e-7, atol=0)
        for k in g:
            a, b = np.asarray(grads[k][i]), np.asarray(g[k])
            dg = max(dg, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))))
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    print(f"vmap vs loop: max rel value diff {dv:.1e}, max rel grad diff {dg:.1e}")


def test_marginalised_lines_with_sampled_scale(csp):
    """Spectrum(marginalize_elines=True) with a sampled scale: the static precomputation is
    off (the line basis depends on s), the marginal ln L at s equals the fixed-s model's on
    the same log grid, and its gradient in s is finite and matches a central difference."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood.likelihood import MultiObservationLikelihood, make_lnprobfn
    t = np.array(csp.sfh_times)
    tr = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t),
          "gas_logu": lambda th: jnp.array([-2.23]), "gas_logz": lambda th: jnp.array([-0.37])}
    truth = _pred(_model(csp, Instrument.sigma_kms(300.0 * 1.1)))["spec"]
    unc = 0.03 * np.abs(truth)

    def build(ins, kin=None, priors=None, init=None):
        fi = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
        fi.update(init or {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SedModel(csp, [Spectrum(wavelength=SPEC_WAVE, flux=truth * 1.01, uncertainty=unc,
                                        instrument=ins, name="spec", marginalize_elines=True)],
                         priors=priors or {}, transforms=tr, free_param_init=fi, zred=ZRED,
                         kinematics=kin or Kinematics(sigma_gal=150.0, sigma_gas=80.0))
        od = m.obs_dict
        lh = tuple(_likelihood_for(od[k], m.param_names, model=m) for k in od)
        lnl = make_lnprobfn(od, m, _NoPrior(), MultiObservationLikelihood(keys=tuple(od),
                                                                          likelihoods=lh))
        return m, jax.jit(lnl)
    ms, fs = build(Instrument.sigma_kms(300.0, scale="lsf_scale"),
                   priors={"lsf_scale": Uniform(low=RANGE[0], high=RANGE[1])},
                   init={"lsf_scale": jnp.array([1.0])})
    assert ms._eline_system is not None and ms._eline_system.static is None
    P = ms.observations[0]._proj
    th = {k: jnp.asarray(v) for k, v in ms.theta_init.items()}
    for sc in (0.9, 1.1):
        mf, ff = build(Instrument.sigma_kms(300.0, scale=sc), kin=_kin_matched(P, sc))
        a = float(fs({**th, "lsf_scale": jnp.array([sc])}))
        b = float(ff({k: v for k, v in th.items() if k != "lsf_scale"}))
        assert abs(a - b) <= 1e-6 * max(1.0, abs(b)), (sc, a, b)
    g = jax.grad(fs)({**th, "lsf_scale": jnp.array([1.1])})["lsf_scale"][0]
    h = 1e-5
    fd = (float(fs({**th, "lsf_scale": jnp.array([1.1 + h])}))
          - float(fs({**th, "lsf_scale": jnp.array([1.1 - h])}))) / (2 * h)
    assert np.isfinite(float(g)) and abs(float(g) - fd) <= 1e-5 * max(abs(fd), 1.0), (g, fd)


class _NoPrior:
    def log_prob(self, theta):
        return 0.0


# --------------------------------------------------------------------------------------
# 5. guards
# --------------------------------------------------------------------------------------
def test_instrument_scale_guards():
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            Instrument.R_fwhm(1500.0, scale=bad)
    with pytest.raises(TypeError):
        Instrument.R_fwhm(1500.0, scale=True)
    with pytest.raises(TypeError):
        Instrument.R_fwhm(1500.0, scale=[1.0])
    with pytest.raises(ValueError):
        Instrument.R_fwhm(1500.0, scale="  ")
    with pytest.raises(ValueError):                      # range on a fixed scale
        Instrument.R_fwhm(1500.0, scale=1.1, scale_range=(0.9, 1.2))
    for rng in ((0.0, 1.2), (-1.0, 1.0), (1.2, 0.9), (1.0, 1.0), (0.9, np.inf), (1.0,)):
        with pytest.raises(ValueError):
            Instrument.R_fwhm(1500.0, scale="lsf_scale", scale_range=rng)
    ins = Instrument.R_fwhm(1500.0, scale="lsf_scale", scale_range=(0.9, 1.2))
    assert ins.free_keys == ("lsf_scale",) and ins.scale_range == (0.9, 1.2)
    assert Instrument.R_fwhm(1500.0).free_keys == ()
    assert check_scale_range((np.array([0.5]), 2), "x") == (0.5, 2.0)


def test_sedmodel_scale_guards(csp):
    ins = Instrument.R_fwhm(1500.0, scale="lsf_scale")
    t = np.array(csp.sfh_times)
    sfh = {"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)}
    base = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}

    def build(ins, priors, init=None, transforms=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SedModel(csp, _obs(ins), priors=priors, transforms={**sfh, **(transforms or {})},
                            free_param_init={**base, **(init or {})}, zred=ZRED)
    with pytest.raises(KeyError, match="lsf_scale"):                       # not a parameter
        build(ins, {})
    one = {"lsf_scale": jnp.array([1.0])}
    with pytest.raises(ValueError, match="unbounded"):
        build(ins, {"lsf_scale": Normal(mean=1.0, sigma=0.1)}, one)
    with pytest.raises(ValueError, match="0 < lo < hi"):                   # non-positive bound
        build(ins, {"lsf_scale": Uniform(low=0.0, high=1.5)}, one)
    with pytest.raises(ValueError, match="no prior"):
        build(ins, {}, one)
    with pytest.raises(ValueError, match="transform"):
        build(ins, {}, transforms={"lsf_scale": lambda th: jnp.array([1.1])})
    own = Instrument.R_fwhm(1500.0, scale="lsf_scale", scale_range=(0.8, 1.2))
    with pytest.raises(ValueError, match="beyond the Instrument's scale_range"):
        build(own, {"lsf_scale": Uniform(low=0.7, high=1.2)}, one)
    # accepted: a transform with the Instrument's own range; an unbounded prior likewise
    m = build(own, {}, transforms={"lsf_scale": lambda th: jnp.array([1.1])})
    assert m.observations[0]._proj.inst_scale_range == (0.8, 1.2)
    m = build(own, {"lsf_scale": Normal(mean=1.0, sigma=0.05)}, one)
    assert m.observations[0]._proj.inst_scale_range == (0.8, 1.2)
    # the sampled key is registered: predict warns about no unrecognised key
    m = build(ins, {"lsf_scale": Uniform(low=0.8, high=1.2)}, one)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.predict(_theta(m, lsf_scale=1.05))
    assert not [x for x in w if "unrecognized theta key" in str(x.message)]
    with pytest.raises(KeyError, match="lsf_scale"):                       # missing at predict
        th = _theta(m)
        th.pop("lsf_scale")
        m.predict(th)


def test_standalone_spectrum_needs_range(csp):
    spec = Spectrum(wavelength=SPEC_WAVE, instrument=Instrument.R_fwhm(1500.0, scale="s_lsf"))
    with pytest.raises(ValueError, match="no range is known"):
        spec.setup_for_model(csp.wave, zred=ZRED)
    spec.setup_for_model(csp.wave, zred=ZRED, inst_scale_range=(0.9, 1.1))
    assert spec._proj.inst_scale_range == (0.9, 1.1)
    assert "x theta['s_lsf']" in str(spec)


def test_switching_rows_warn(csp):
    """An instrument close to the library resolution: some rows cross half a log pixel
    within the range, where the response changes form (warned at setup)."""
    with pytest.warns(UserWarning, match="crosses half a log-grid pixel"):
        Spectrum(wavelength=SPEC_WAVE,
                 instrument=Instrument.R_fwhm(1500.0, scale="lsf_scale")).setup_for_model(
            csp.wave, zred=ZRED, lib_resolution=csp.lib_resolution,
            inst_scale_range=(0.7, 1.4))
