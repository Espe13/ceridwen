"""Tests for ceridwen.broadening.

Run with a real JAX:      pytest tests/test_broadening.py
Numerics-only smoke run without JAX: see tests/numpy_stub/README.md
(the stub sets jax.__fake__ = True; jit/grad/vmap tests are skipped).
The four sedpy reproduction tests need `sedpy` (a prospector dependency).
"""
import os
import sys
import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from reference_broadening import (                                  # noqa: E402
    direct_broaden, analytic_line_fnu, integrate_fnu_dnu, gaussian_feature_lnlambda,
    CKMS, C_AA_S)
from ceridwen.broadening import (                                   # noqa: E402
    Instrument, Kinematics, TIED, DEFAULT_KINEMATICS, LogGrid, make_gaussian_fft, build_response,
    apply_response, make_line_painter, make_line_painter_free_z, SpectralProjector,
    PhotometricBroadener, FWHM_TO_SIGMA)

FAKE = getattr(jax, "__fake__", False)
needs_jax = pytest.mark.skipif(FAKE, reason="needs a real JAX")

if not FAKE:
    jax.config.update("jax_enable_x64", True)


def model_grid(dv_fine=8.0, dv_coarse=40.0):
    """Rest-frame model grid: 8 km/s pixels in 3500-7500 A, 40 km/s outside
    (mimics a library that is finer in the optical), 900-30000 A."""
    def seg(a, b, dv):
        n = int(np.ceil(np.log(b / a) / (dv / CKMS)))
        return np.exp(np.linspace(np.log(a), np.log(b), n, endpoint=False))
    w = np.concatenate([seg(900., 3500., dv_coarse), seg(3500., 7500., dv_fine),
                        seg(7500., 30000., dv_coarse), [30000.]])
    return w


@pytest.fixture(scope="module")
def wm():
    return model_grid()


@pytest.fixture(scope="module")
def wo():
    return np.exp(np.arange(np.log(4800.), np.log(9000.), 60.0 / CKMS))


Z = 0.5


def test_instrument_unit_conversions_agree():
    lam = np.array([5000.0])
    R = 2000.0
    fwhm = lam[0] / R
    a = Instrument.R_fwhm(R).sigma_kms_at(lam)[0]
    b = Instrument.fwhm_aa(fwhm).sigma_kms_at(lam)[0]
    c = Instrument.sigma_aa(fwhm * FWHM_TO_SIGMA).sigma_kms_at(lam)[0]
    d = Instrument.sigma_kms(CKMS / R * FWHM_TO_SIGMA).sigma_kms_at(lam)[0]
    e = Instrument.fwhm_kms(CKMS / R).sigma_kms_at(lam)[0]
    f = Instrument.R_sigma(R / FWHM_TO_SIGMA).sigma_kms_at(lam)[0]
    assert np.allclose([a, b, c, d, e, f], a, rtol=1e-12)
    assert np.isclose(a, CKMS / (R * 2.0 * np.sqrt(2 * np.log(2))))
    assert np.isclose(Instrument.R_sigma(R).sigma_kms_at(lam)[0] / a, 2 * np.sqrt(2 * np.log(2)))


def test_instrument_rejects_bad_input():
    with pytest.raises(ValueError):
        Instrument.R_fwhm(-1.0)
    with pytest.raises(ValueError):
        Instrument.R_fwhm(np.array([1000., 2000.]))
    with pytest.raises(ValueError):
        Instrument.R_fwhm(np.array([1000., 2000.]), wave=np.array([6000., 5000.]))
    inst = Instrument.R_fwhm(np.array([1000., 2000.]), wave=np.array([5000., 6000.]))
    with pytest.raises(ValueError):
        inst.sigma_kms_at(np.array([4000.]))


def test_kinematics_validation(wm, wo):
    with pytest.raises(TypeError):
        Kinematics(sigma_gal=True)
    with pytest.raises(ValueError):
        Kinematics(sigma_gal=-5.0)
    with pytest.raises(ValueError):
        Kinematics(sigma_gal=5000.0)
    with pytest.raises(ValueError):
        Kinematics(sigma_gal=TIED)
    with pytest.raises(TypeError):
        SpectralProjector.build(Kinematics(100.0), "R=1000", wm, wo, Z)
    b = Kinematics(sigma_gal="sigma_gal")
    assert b.free_keys == ("sigma_gal",) and not b.is_static
    assert b.effective_sigma_gas == "sigma_gal"
    b2 = Kinematics(sigma_gal=150.0, sigma_gas="sigma_gas")
    assert b2.free_keys == ("sigma_gas",)
    with pytest.raises(KeyError):
        b.validate_theta({"logmass": 1.0})
    with pytest.raises(ValueError):
        b2.validate_theta({"sigma_gal": 1.0, "sigma_gas": 1.0})
    b2.validate_theta({"sigma_gas": np.array([80.0])})
    assert Kinematics.none().free_keys == () and Kinematics.none().is_static
    assert DEFAULT_KINEMATICS == Kinematics(sigma_gal=300.0) and DEFAULT_KINEMATICS.is_static
    assert DEFAULT_KINEMATICS.effective_sigma_gas == 300.0


def test_loggrid_dv_is_finest_model_pixel(wm):
    g = LogGrid.build(wm, 1000., 24000., sigma_max_kms=1000.)
    dv_model = CKMS * np.min(np.diff(np.log(g.wave_model_window)))
    assert g.dv <= dv_model * (1 + 1e-9)
    assert g.wave[0] == 1000. and g.wave[-1] == 24000.
    assert np.allclose(np.diff(g.lnw), g.dv / CKMS)
    assert g.n_pad >= 2 * g.n and g.n_pad >= g.n + 10 * 1000. / g.dv
    assert g.n_pad & (g.n_pad - 1) == 0
    n_sedpy = 2 ** int(np.ceil(np.log2(g.idx.size)))
    dv_sedpy = CKMS * (np.log(24000.) - np.log(1000.)) / (n_sedpy - 1)
    assert dv_sedpy > 1.5 * g.dv


def test_loggrid_rejects_uncovered_window(wm):
    with pytest.raises(ValueError):
        LogGrid.build(wm, 100., 7000., 1000.)


def test_stage_a_matches_direct_convolution(wm):
    g = LogGrid.build(wm, 3600., 7000., sigma_max_kms=1500.)
    smooth = make_gaussian_fft(g)
    rng = np.random.default_rng(1)
    y = 1.0 + 0.3 * np.log(g.wave / 5000.)
    for w0, a, s in zip(rng.uniform(3700, 6900, 40), rng.uniform(-0.5, 0.5, 40),
                        rng.uniform(30, 300, 40)):
        y *= gaussian_feature_lnlambda(g.wave, w0, s, a)
    for sigma in (0.0, 25.0, 300.0, 1500.0):
        fast = np.asarray(smooth(jnp.asarray(y), jnp.asarray(sigma)))
        ref = direct_broaden(g.wave, y, g.wave, sigma)
        assert np.max(np.abs(fast - ref)) < 2e-4 * np.max(np.abs(ref)), sigma


def test_stage_a_edges_not_darkened(wm):
    g = LogGrid.build(wm, 3600., 7000., sigma_max_kms=2000.)
    smooth = make_gaussian_fft(g)
    flat = np.ones(g.n)
    out = np.asarray(smooth(jnp.asarray(flat), jnp.asarray(2000.)))
    assert np.allclose(out, 1.0, atol=1e-9)
    step = np.where(g.wave < 5000., 2.0, 1.0)
    out = np.asarray(smooth(jnp.asarray(step), jnp.asarray(2000.)))
    assert abs(out[0] - 2.0) < 1e-6 and abs(out[-1] - 1.0) < 1e-6


def test_stage_a_conserves_sum(wm):
    g = LogGrid.build(wm, 3600., 7000., sigma_max_kms=500.)
    smooth = make_gaussian_fft(g)
    y = gaussian_feature_lnlambda(g.wave, 5200., 40., 3.0) - 1.0
    out = np.asarray(smooth(jnp.asarray(y), jnp.asarray(400.)))
    assert np.isclose(out.sum(), y.sum(), rtol=1e-10)


@pytest.mark.parametrize("sigma_gal,inst", [
    (0.0, Instrument.R_fwhm(3000.)),
    (120.0, Instrument.R_fwhm(3000.)),
    (250.0, Instrument.fwhm_aa(6.0)),
    (400.0, Instrument.sigma_kms(np.array([60., 250.]), wave=np.array([4000., 10000.]))),
])
def test_projector_gaussian_composition(wm, wo, sigma_gal, inst):
    s_lib = 30.0
    lib = np.full(wm.shape, s_lib)
    b = Kinematics(sigma_gal=sigma_gal, sigma_max=600.)
    proj = SpectralProjector.build(b, inst, wm, wo, Z, lib_sigma_kms=lib)
    centres = np.array([3700., 4100., 5000., 5800.])
    y = np.ones_like(wm)
    for c in centres:
        y *= gaussian_feature_lnlambda(wm, c, s_lib, -0.4)
    out = np.asarray(proj.continuum(jnp.asarray(y), jnp.asarray(sigma_gal)))
    s_inst = inst.sigma_kms_at(wo)
    expected = np.ones_like(wo)
    for c in centres:
        c_obs = c * (1 + Z)
        si = np.interp(c_obs, wo, s_inst)
        s_tot = np.sqrt(sigma_gal ** 2 + si ** 2)
        amp = -0.4 * s_lib / s_tot
        expected *= gaussian_feature_lnlambda(wo, c_obs, s_tot, amp)
    assert np.max(np.abs(out - expected)) < 3e-3 * 0.4, (sigma_gal, inst.kind)


def test_projector_matches_direct_reference(wm, wo):
    """Whole continuum path vs brute-force quadrature with the total width."""
    inst = Instrument.R_fwhm(np.array([1500., 4000.]), wave=np.array([4000., 10000.]))
    lib = np.full(wm.shape, 25.0)
    b = Kinematics(sigma_gal=200.0, sigma_max=500.)
    proj = SpectralProjector.build(b, inst, wm, wo, Z, lib_sigma_kms=lib)
    rng = np.random.default_rng(3)
    y = 1.0 + 0.2 * np.sin(np.log(wm) * 400.)
    for w0, a, s in zip(rng.uniform(3300, 6100, 30), rng.uniform(-0.4, 0.4, 30),
                        rng.uniform(25, 200, 30)):
        y *= gaussian_feature_lnlambda(wm, w0, s, a)
    out = np.asarray(proj.continuum(jnp.asarray(y), jnp.asarray(200.0)))
    s_tot = np.sqrt(200.0 ** 2 + inst.sigma_kms_at(wo) ** 2 - 25.0 ** 2)
    ref = direct_broaden(wm, y, wo / (1 + Z), s_tot)
    assert np.max(np.abs(out - ref)) < 2e-3


def test_projector_flat_is_flat(wm, wo):
    proj = SpectralProjector.build(Kinematics(sigma_gal=300.0), Instrument.R_fwhm(500.), wm, wo, Z)
    out = np.asarray(proj.continuum(jnp.ones_like(wm), jnp.asarray(300.0)))
    assert np.allclose(out, 1.0, atol=1e-9)


def test_library_wider_than_instrument_warns_and_floors(wm, wo):
    """A grid coarser than the instrument is legitimate (C3K_lr, BaSeL): warn,
    floor the fixed width at zero, deliver the continuum at library resolution."""
    lib = np.full(wm.shape, 80.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        inst = Instrument.sigma_kms(np.array([60., 200.]), wave=np.array([4000., 10000.]))
        SpectralProjector.build(Kinematics(0.0), inst, wm, wo, Z, lib_sigma_kms=lib)
    assert any("narrower than the SSP library" in str(w.message) for w in rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        proj = SpectralProjector.build(Kinematics(0.0), Instrument.sigma_kms(50.),
                                       wm, wo, Z, lib_sigma_kms=lib)
    assert len(rec) == 1 and np.all(proj.sigma_fix_kms == 0.0)
    y = gaussian_feature_lnlambda(wm, 5000., 80., -0.4)
    out = np.asarray(proj.continuum(jnp.asarray(y), jnp.asarray(0.0)))
    ref = np.interp(wo / (1 + Z), wm, y)
    assert np.max(np.abs(out - ref)) < 1e-3


def test_projector_requires_model_coverage(wm):
    wo_bad = np.linspace(40000., 45000., 100)
    with pytest.raises(ValueError):
        SpectralProjector.build(Kinematics(0.0), Instrument.R_fwhm(1000.), wm, wo_bad, Z)


def test_line_painter_flux_and_profile(wo):
    lines = np.array([3727.1, 4861.3, 5006.8]) * (1 + Z)
    flux = np.array([1.0, 3.0, 2.8])
    s_inst = np.array([80., 80., 80.])
    paint = make_line_painter(wo, lines, s_inst)
    for s_gas in (0.0, 50.0, 300.0):
        out = np.asarray(paint(jnp.asarray(flux), jnp.asarray(s_gas)))
        s_tot = np.sqrt(s_gas ** 2 + s_inst ** 2)
        ref = sum(analytic_line_fnu(wo, l0, f, s) for l0, f, s in zip(lines, flux, s_tot))
        assert np.allclose(out, ref, rtol=1e-10, atol=1e-12 * ref.max())
        assert np.isclose(integrate_fnu_dnu(wo, out), flux.sum(), rtol=2e-3)
    paint2 = make_line_painter(wo, np.array([6562.8 * (1 + Z)]), np.array([80.]))
    out2 = np.asarray(paint2(jnp.asarray([1.0]), jnp.asarray(50.0)))
    assert integrate_fnu_dnu(wo, out2) < 1e-12


def test_projector_lines_in_window_selected(wm, wo):
    line_rest = np.array([1215.7, 3727.1, 4861.3, 5006.8, 6562.8, 18750.0])
    b = Kinematics(sigma_gal=100.0, sigma_gas=50.0)
    proj = SpectralProjector.build(b, Instrument.R_fwhm(1000.), wm, wo, Z, line_wave_rest=line_rest)
    obs = line_rest * (1 + Z)
    inwin = (obs > wo[0] * 0.98) & (obs < wo[-1] * 1.02)
    assert set(proj.line_idx) == set(np.flatnonzero(inwin))
    F = np.arange(1., 7.)
    out = np.asarray(proj.predict(jnp.zeros_like(wm), jnp.asarray(F),
                                  {"logmass": jnp.array([1.0])}))
    assert np.isclose(integrate_fnu_dnu(wo, out), F[inwin].sum(), rtol=2e-3)


def test_resolve_fixed_free_tied():
    b = Kinematics(sigma_gal="sigma_gal")
    g, s = b.resolve({"sigma_gal": jnp.array([123.0])})
    assert float(g) == 123.0 and float(s) == 123.0
    b = Kinematics(sigma_gal=77.0, sigma_gas="sg")
    g, s = b.resolve({"sg": jnp.array([5.0])})
    assert float(g) == 77.0 and float(s) == 5.0
    b = Kinematics(sigma_gal="sigma_gal", sigma_max=500.)
    g, _ = b.resolve({"sigma_gal": jnp.array([9999.0])})
    assert float(g) == 500.0


@needs_jax
def test_jit_and_grad_wrt_sigma(wm, wo):
    b = Kinematics(sigma_gal="sigma_gal", sigma_gas="sigma_gas")
    line_rest = np.array([4861.3, 5006.8])
    proj = SpectralProjector.build(b, Instrument.R_fwhm(2000.), wm, wo, Z, line_wave_rest=line_rest)
    y = np.ones_like(wm)
    for c in (4300., 5200.):
        y *= gaussian_feature_lnlambda(wm, c, 40., -0.3)
    y = jnp.asarray(y)
    F = jnp.asarray([2.0, 5.0])

    @jax.jit
    def f(theta):
        return proj.predict(y, F, theta)

    def loss(sg, ss):
        return jnp.sum(f({"sigma_gal": sg, "sigma_gas": ss}) ** 2)

    g = jax.grad(loss, argnums=(0, 1))(jnp.array([150.0]), jnp.array([60.0]))
    eps = 1e-2
    fd0 = (loss(jnp.array([150.0 + eps]), jnp.array([60.0]))
           - loss(jnp.array([150.0 - eps]), jnp.array([60.0]))) / (2 * eps)
    fd1 = (loss(jnp.array([150.0]), jnp.array([60.0 + eps]))
           - loss(jnp.array([150.0]), jnp.array([60.0 - eps]))) / (2 * eps)
    assert np.isclose(float(g[0][0]), float(fd0), rtol=1e-4)
    assert np.isclose(float(g[1][0]), float(fd1), rtol=1e-4)
    assert np.all(np.isfinite(np.asarray(f({"sigma_gal": jnp.array([0.0]),
                                            "sigma_gas": jnp.array([0.0])}))))


# ---------------------------------------------------------------- sampled redshift

LINES_REST = np.array([3727.0, 4861.3, 5006.8, 6562.8, 9069.0])
LINE_FLUX = np.array([1.0, 2.0, 5.0, 8.0, 0.5])


def _free_z_setup(wm, wo):
    inst = Instrument.R_fwhm(np.array([1500., 4000.]), wave=np.array([4000., 10000.]))
    lib = np.full(wm.shape, 25.0)
    kin = Kinematics(sigma_gal="sigma_gal", sigma_gas="sigma_gas", sigma_max=500.)
    rng = np.random.default_rng(3)
    y = 1.0 + 0.2 * np.sin(np.log(wm) * 400.)
    for w0, a, s in zip(rng.uniform(3000, 6300, 40), rng.uniform(-0.4, 0.4, 40),
                        rng.uniform(25, 200, 40)):
        y *= gaussian_feature_lnlambda(wm, w0, s, a)
    return inst, lib, kin, y


def test_free_z_projector_matches_fixed_z(wm, wo):
    """A projector built with zred_range, evaluated at z, reproduces the projector
    built at that fixed z (continuum and lines), at the reference z and away from it."""
    inst, lib, kin, y = _free_z_setup(wm, wo)
    free = SpectralProjector.build(kin, inst, wm, wo, 0.5, lib_sigma_kms=lib,
                                   line_wave_rest=LINES_REST, zred_range=(0.35, 0.65))
    assert free.free_z and free.opz_ref == 1.5
    theta = {"sigma_gal": jnp.array([200.0]), "sigma_gas": jnp.array([80.0])}
    F = jnp.asarray(LINE_FLUX)
    for z in (0.5, 0.42, 0.6, 0.65):
        fixed = SpectralProjector.build(kin, inst, wm, wo, z, lib_sigma_kms=lib,
                                        line_wave_rest=LINES_REST)
        a = np.asarray(free.predict(jnp.asarray(y), F, dict(theta, zred=jnp.array([z]))))
        b = np.asarray(fixed.predict(jnp.asarray(y), F, theta))
        assert np.max(np.abs(a - b)) < 1e-4, z
        la = np.asarray(free.lines(F, jnp.asarray(80.0), 1.0 + z))
        lb = np.asarray(fixed.lines(F, jnp.asarray(80.0)))
        assert np.max(np.abs(la - lb)) < 1e-8 * lb.max(), z
    # the free-z window keeps every line that enters the observed range for some z
    assert set(free.line_idx.tolist()) >= set(
        SpectralProjector.build(kin, inst, wm, wo, 0.35, line_wave_rest=LINES_REST).line_idx.tolist())
    assert set(free.line_idx.tolist()) >= set(
        SpectralProjector.build(kin, inst, wm, wo, 0.65, line_wave_rest=LINES_REST).line_idx.tolist())


def test_free_z_fixed_path_unchanged(wm, wo):
    """zred_range=None is the fixed-z path: no 'zred' is read from theta."""
    inst, lib, kin, y = _free_z_setup(wm, wo)
    fixed = SpectralProjector.build(kin, inst, wm, wo, 0.5, lib_sigma_kms=lib,
                                    line_wave_rest=LINES_REST)
    assert not fixed.free_z
    theta = {"sigma_gal": jnp.array([200.0]), "sigma_gas": jnp.array([80.0])}
    a = fixed.predict(jnp.asarray(y), jnp.asarray(LINE_FLUX), theta)
    b = fixed.predict(jnp.asarray(y), jnp.asarray(LINE_FLUX), dict(theta, zred=jnp.array([0.6])))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_free_z_projector_guards(wm, wo):
    inst, lib, kin, y = _free_z_setup(wm, wo)
    with pytest.raises(ValueError):
        SpectralProjector.build(kin, inst, wm, wo, 0.5, zred_range=(0.6, 0.7))   # ref outside
    with pytest.raises(ValueError):
        SpectralProjector.build(kin, inst, wm, wo, 0.5, zred_range=(0.7, 0.4))   # inverted
    free = SpectralProjector.build(kin, inst, wm, wo, 0.5, zred_range=(0.4, 0.6))
    with pytest.raises(KeyError):
        free.predict(jnp.asarray(y), None, {"sigma_gal": jnp.array([200.0]),
                                            "sigma_gas": jnp.array([80.0])})


def test_free_z_line_painter_flux(wo):
    """The traced-z painter conserves the line flux and places the line at lambda_rest (1+z)."""
    s_tab = np.full(wo.shape, 60.0)
    paint = make_line_painter_free_z(wo, np.array([4861.3]), s_tab)
    for z in (0.3, 0.5, 0.8):
        fnu = np.asarray(paint(jnp.asarray([3.0]), jnp.asarray(80.0), jnp.asarray(1.0 + z)))
        assert np.isclose(integrate_fnu_dnu(wo, fnu), 3.0, rtol=2e-3), z
        assert abs(wo[np.argmax(fnu)] / (4861.3 * (1 + z)) - 1.0) < 60.0 / CKMS


@needs_jax
def test_free_z_jit_and_grad_wrt_zred(wm, wo):
    inst, lib, kin, y = _free_z_setup(wm, wo)
    free = SpectralProjector.build(kin, inst, wm, wo, 0.5, lib_sigma_kms=lib,
                                   line_wave_rest=LINES_REST, zred_range=(0.35, 0.65))
    yj, F = jnp.asarray(y), jnp.asarray(LINE_FLUX)

    @jax.jit
    def f(theta):
        return free.predict(yj, F, theta)

    def loss(z):
        return jnp.sum(f({"sigma_gal": jnp.array([200.0]), "sigma_gas": jnp.array([80.0]),
                          "zred": z}) ** 2)

    z0 = jnp.array([0.52])
    g = float(jax.grad(loss)(z0)[0])
    # the model is read by linear interpolation onto the log grid, so the loss is
    # piecewise linear in ln(1+z): every log-grid node crossing a model pixel is a kink,
    # and with ~1e4 nodes the pieces are ~1e-8 wide in z.  The analytic gradient is the
    # slope of the piece at z0 (at a kink: one of the two one-sided slopes), so compare
    # with one-sided differences small enough to stay inside a piece.
    eps = 1e-10
    l0, lp, lm = (float(loss(z)) for z in (z0, z0 + eps, z0 - eps))
    fr, fl = (lp - l0) / eps, (l0 - lm) / eps
    tol = 1e-2 * abs(g)
    assert np.isfinite(g)
    assert min(fl, fr) - tol <= g <= max(fl, fr) + tol, (g, fl, fr)


@needs_jax
def test_vmap_over_theta(wm, wo):
    b = Kinematics(sigma_gal="sigma_gal")
    proj = SpectralProjector.build(b, Instrument.R_fwhm(2000.), wm, wo, Z)
    y = jnp.asarray(gaussian_feature_lnlambda(wm, 5000., 40., -0.3))
    sig = jnp.array([[50.0], [150.0], [400.0]])
    out = jax.vmap(lambda s: proj.continuum(y, b.resolve({"sigma_gal": s})[0]))(sig)
    assert out.shape == (3, wo.size)
    mins = np.asarray(out.min(axis=1))
    assert mins[0] < mins[1] < mins[2]


def test_photometric_broadener_window_and_kernel(wm):
    pb = PhotometricBroadener.build(Kinematics(300.0, sigma_max=500.), wm, 3000., 9000.)
    marg = 5 * 500. / CKMS
    assert pb.grid.wave[0] <= 3000. * np.exp(-marg) * (1 + 1e-12)
    assert pb.grid.wave[-1] >= 9000. * np.exp(marg) * (1 - 1e-12)
    y = gaussian_feature_lnlambda(wm, 5000., 40., -0.5)
    spec_log = np.interp(pb.grid.wave, pb.grid.wave_model_window, y[pb.grid.idx])
    out_log = np.asarray(pb.window.smooth(jnp.asarray(spec_log), jnp.asarray(300.)))
    ref = gaussian_feature_lnlambda(pb.grid.wave, 5000., np.hypot(40., 300.),
                                    -0.5 * 40. / np.hypot(40., 300.))
    assert np.max(np.abs(out_log - ref)) < 1e-3


@needs_jax
def test_photometric_broadener_scatter(wm):
    pb = PhotometricBroadener.build(Kinematics(300.0, sigma_max=500.), wm, 3000., 9000.)
    y = jnp.asarray(gaussian_feature_lnlambda(wm, 5000., 40., -0.5))
    out = np.asarray(pb(y, jnp.asarray(300.)))
    outside = (wm < pb.grid.wave[0]) | (wm > pb.grid.wave[-1])
    assert np.array_equal(out[outside], np.asarray(y)[outside])
    assert out.min() > np.asarray(y).min()


def _sedpy():
    return pytest.importorskip("sedpy.smoothing")


def _feature_spectrum(wm, rng_seed=7, n=25):
    rng = np.random.default_rng(rng_seed)
    y = 1.0 + 0.2 * np.log(wm / 5000.)
    for w0, a, s in zip(rng.uniform(3400, 6100, n), rng.uniform(-0.4, 0.4, n),
                        rng.uniform(40, 150, n)):
        y *= gaussian_feature_lnlambda(wm, w0, s, a)
    return y


def test_reproduces_sedpy_smooth_vel(wm, wo):
    """sigma_gal + constant-velocity instrument - library  ==  sedpy.smooth_vel
    with sigma = sqrt(gal^2 + inst^2), inres = lib (direct quadrature)."""
    sm = _sedpy()
    y = _feature_spectrum(wm)
    gal, inst, lib = 180.0, 70.0, 30.0
    proj = SpectralProjector.build(Kinematics(gal, sigma_max=400.), Instrument.sigma_kms(inst),
                                   wm, wo, Z, lib_sigma_kms=np.full(wm.shape, lib))
    ours = np.asarray(proj.continuum(jnp.asarray(y), jnp.asarray(gal)))
    ref = sm.smooth_vel(wm, y, wo / (1 + Z), np.hypot(gal, inst), inres=lib)
    assert np.max(np.abs(ours - ref)) < 2e-3


def test_reproduces_sedpy_smooth_wave(wm, wo):
    """Instrument quoted as constant sigma in Angstrom == sedpy.smooth_wave
    (a Gaussian in lambda; ours is a Gaussian in ln lambda of the same width
    at each output pixel, identical to O(sigma_lambda/lambda))."""
    sm = _sedpy()
    y = _feature_spectrum(wm)
    sig_aa_obs = 4.5
    proj = SpectralProjector.build(Kinematics.none(), Instrument.sigma_aa(sig_aa_obs),
                                   wm, wo, Z)
    ours = np.asarray(proj.continuum(jnp.asarray(y), jnp.asarray(0.0)))
    ref = sm.smooth_wave(wm, y, wo / (1 + Z), sig_aa_obs / (1 + Z))
    assert np.max(np.abs(ours - ref)) < 2e-3


def test_reproduces_sedpy_smooth_lsf(wm, wo):
    """Wavelength-dependent LSF in Angstrom == sedpy.smooth_lsf (the row-
    normalised Gaussian matrix, output-referenced: the same construction)."""
    sm = _sedpy()
    y = _feature_spectrum(wm)
    sig_aa_obs = 2.0 + 3.0 * (wo - wo[0]) / (wo[-1] - wo[0])
    proj = SpectralProjector.build(Kinematics.none(),
                                   Instrument.sigma_aa(sig_aa_obs, wave=wo), wm, wo, Z)
    ours = np.asarray(proj.continuum(jnp.asarray(y), jnp.asarray(0.0)))
    ref = sm.smooth_lsf(wm, y, wo / (1 + Z), sigma=sig_aa_obs / (1 + Z))
    assert np.max(np.abs(ours - ref)) < 2e-3


@pytest.mark.parametrize("module", ["sedpy.smoothing", "sedpy_jax.smoothing"])
def test_reproduces_smooth_vel_fft_interior(wm, module):
    """Stage A alone vs the zero-padded circular FFT of sedpy / sedpy_jax
    (what CERIDWEN calls today): identical away from the window edges,
    where the zero pad darkens and ours does not."""
    sm = pytest.importorskip(module)
    g = LogGrid.build(wm, 3600., 7000., sigma_max_kms=500.)
    smooth = make_gaussian_fft(g)
    y = _feature_spectrum(g.wave)
    ours = np.asarray(smooth(jnp.asarray(y), jnp.asarray(300.)))
    if module.startswith("sedpy_jax"):
        theirs = np.asarray(sm.smooth_vel_fft(g.wave, jnp.asarray(y), 300., g.wave))
    else:
        theirs = sm.smooth_vel_fft(g.wave, y, g.wave, 300.)
    interior = (g.wave > 3800.) & (g.wave < 6700.)
    assert np.max(np.abs(ours - theirs)[interior]) < 1e-3
    assert np.abs(ours - theirs)[~interior].max() > 1e-2
