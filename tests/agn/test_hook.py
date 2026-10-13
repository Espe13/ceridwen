"""The AGN in the CSP (ceridwen/agn/hook.py, CSPBasis._with_agn): AGN_PLAN §4 B3-B8.

Every CSP here is the torus only (``disk=False, lines=False``: the disc and lines are wired by
the INT session); B3 runs every one of the eight spectrum kernels in all three modes."""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ceridwen import SedModel
from ceridwen.agn import AGNConfig
from ceridwen.observation import Photometry, Spectrum
from ceridwen.broadening import Instrument
from _f32bounds import B8, lnl_bound, fft_atol   # noqa: E402,F401  (float32 bounds)

MODES = ("fsps", "prospector", "corrected")
ZRED = 2.0
FILTERS = ["jwst_f090w", "jwst_f115w", "jwst_f444w", "jwst_f770w", "jwst_f1800w"]
SPEC_WAVE = np.linspace(3800.0, 8800.0, 400)

# the eight kernels: (CSP kwargs, extra theta)
CONFIGS = {
    "dattn_nodem_noneb": (dict(add_neb=False), {}),
    "dattn_dem_noneb": (dict(add_neb=False, add_dust_emission=True), {}),
    "nodattn_nodem_noneb": (dict(add_neb=False, add_dust=False, add_diffuse_dust=False), {}),
    "dattn_nodem_neb": (dict(add_neb=True), {}),
    "dattn_dem_neb": (dict(add_neb=True, add_dust_emission=True), {}),
    "nodattn_nodem_neb": (dict(add_neb=True, add_dust=False, add_diffuse_dust=False), {}),
    "picket_nodem": (dict(add_neb=True, fesc_geometry="picket"),
                     {"frac_obrun": jnp.array([0.3])}),
    "picket_dem": (dict(add_neb=True, fesc_geometry="picket", add_dust_emission=True),
                   {"frac_obrun": jnp.array([0.3])}),
}


def torus_only(mode, **kw):
    return AGNConfig(agn_mode=mode, disk=False, lines=False, **kw)


@pytest.fixture(scope="module")
def csps(build_csp):
    cache = {}

    def get(cfg, mode, **extra):
        key = (cfg, mode, tuple(sorted(extra.items())))
        if key not in cache:
            kw, _ = CONFIGS[cfg]
            agn = None if mode is None else torus_only(mode)
            cache[key] = build_csp(agn, add_igm=True, **kw, **extra)
        return cache[key]
    return get


def _theta(csp, cfg, **over):
    th = dict(csp.theta_init)
    th.update(CONFIGS[cfg][1])
    th.update(over)
    return th


def _obs(phot_only=False):
    obs = [Photometry(filters=FILTERS, flux=[2e-9] * len(FILTERS),
                      uncertainty=[2e-10] * len(FILTERS), name="phot")]
    if not phot_only:
        obs.append(Spectrum(wavelength=SPEC_WAVE * (1.0 + ZRED), flux=np.full(SPEC_WAVE.size, 1e-9),
                            uncertainty=np.full(SPEC_WAVE.size, 1e-10),
                            instrument=Instrument.sigma_kms(150.0), name="spec"))
    return obs


def _model(csp, cfg, phot_only=False, zred=ZRED):
    init = {"logmass": jnp.array([10.0])}
    init.update(CONFIGS[cfg][1])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, _obs(phot_only), priors={}, free_param_init=init, zred=zred)


def _bytes(tree):
    return {k: np.asarray(v).tobytes() for k, v in tree.items()}


# ------------------------------------------------------------------ B3
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_B3_zero_agn_byte_identical(csps, cfg, mode):
    """B3: with fagn = 0 the spectrum (eager and jit, with and without lines), the components
    and the jitted predictions (photometry + spectrum, z = 2, IGM on) are byte-identical to the
    same CSP built without agn=."""
    off = csps(cfg, None)
    on = csps(cfg, mode)
    th_off = _theta(off, cfg)
    th_on = _theta(on, cfg, fagn=jnp.array([0.0]))
    assert set(th_on) - set(th_off) == {"fagn", "agn_tau"}
    for lines in (False, True):
        a = off.get_spectrum(theta=th_off, include_lines=lines)
        b = on.get_spectrum(theta=th_on, include_lines=lines)
        assert a.dtype == b.dtype and np.asarray(a).tobytes() == np.asarray(b).tobytes(), lines
    ja = jax.jit(lambda t: off.get_spectrum(theta=t))(th_off)
    jb = jax.jit(lambda t: on.get_spectrum(theta=t))(th_on)
    assert np.asarray(ja).tobytes() == np.asarray(jb).tobytes()
    ca, la = off.get_spectrum_components(th_off)
    cb, lb = on.get_spectrum_components(th_on)
    assert np.asarray(ca).tobytes() == np.asarray(cb).tobytes()
    assert np.asarray(la).tobytes() == np.asarray(lb).tobytes()
    m_off, m_on = _model(off, cfg), _model(on, cfg)
    p_off = m_off.predict_jit({k: v for k, v in m_off.theta_init.items()})
    p_on = m_on.predict_jit({**m_on.theta_init, "fagn": jnp.array([0.0])})
    assert _bytes(p_off) == _bytes(p_on)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", ["dattn_nodem_noneb", "dattn_nodem_neb", "nodattn_nodem_neb"])
def test_B3_wavelength_support(csps, cfg, mode):
    """B3 with C1 ``wavelength_support=True`` (per-pixel kernels, photometry only)."""
    off = csps(cfg, None, wavelength_support=True)
    on = csps(cfg, mode, wavelength_support=True)
    assert getattr(on.get_spectrum, "_per_pixel", False)
    m_off, m_on = _model(off, cfg, phot_only=True), _model(on, cfg, phot_only=True)
    sup = on._support_for(m_on.observations, {**m_on.theta_init, "zred": jnp.array([ZRED])})
    assert sup is not None             # the support path is really taken
    p_off = m_off.predict_jit(dict(m_off.theta_init))
    p_on = m_on.predict_jit({**m_on.theta_init, "fagn": jnp.array([0.0])})
    assert _bytes(p_off) == _bytes(p_on)


# ------------------------------------------------------------------ a/Fe CSP
@pytest.fixture(scope="module")
def afe_ssp(ssp_data):
    """A two-plane [alpha/Fe] grid made from the BPASS test grid (planes x1 and x0.9), so the
    a/Fe path is exercised wherever the solar test grid is."""
    from types import SimpleNamespace
    s = SimpleNamespace(**{k: getattr(ssp_data, k) for k in vars(ssp_data)})
    f = np.asarray(ssp_data.ssp_flux, dtype=np.float64)
    s.ssp_flux = np.stack([f, 0.9 * f])
    s.ssp_afe = np.array([0.0, 0.4])
    if s.ssp_stellar_mass is not None:
        m = np.asarray(s.ssp_stellar_mass)
        s.ssp_stellar_mass = np.stack([m, m])
    return s


def _afe(afe_ssp, agn, **kw):
    from ceridwen import Cosmology
    from ceridwen.csp import CSPBasis_afe
    return CSPBasis_afe(afe_ssp, lookback_time=jnp.linspace(0.0, 0.5, 5), zh_const=True,
                        verbose=False, cosmo=Cosmology.planck18(), agn=agn, add_igm=True,
                        **kw)


AFE_CFGS = {"dattn_nodem_noneb": {}, "dattn_dem_noneb": dict(add_dust_emission=True),
            "nodattn_nodem_noneb": dict(add_dust=False, add_diffuse_dust=False)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", list(AFE_CFGS))
def test_B3_afe(afe_ssp, torus_path, cfg, mode):
    """B3 on CSPBasis_afe at a sampled afe (the folded contraction) and without afe."""
    off = _afe(afe_ssp, None, **AFE_CFGS[cfg])
    on = _afe(afe_ssp, torus_only(mode), **AFE_CFGS[cfg])
    for extra in ({}, {"afe": jnp.array([0.13])}):
        a = off.get_spectrum(theta={**off.theta_init, **extra})
        b = on.get_spectrum(theta={**on.theta_init, **extra, "fagn": jnp.array([0.0])})
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), extra


def test_afe_lbol_interpolates(afe_ssp, torus_path):
    """L_bol on the a/Fe CSP follows theta['afe'] exactly as the flux cube does (planes x1,
    x0.9: L_bol(afe) = (1 - 0.25 afe) L_bol(0) on this synthetic grid)."""
    on = _afe(afe_ssp, torus_only("corrected"), add_dust=False, add_diffuse_dust=False)
    th = dict(on.theta_init)
    W = on.calculate_ssp_weights(th)
    L0 = float(jnp.sum(W * on._agn_lbol_at(th)))
    for afe in (0.0, 0.2, 0.4, 0.8):
        La = float(jnp.sum(W * on._agn_lbol_at({**th, "afe": jnp.array([afe])})))
        np.testing.assert_allclose(La, (1.0 - 0.25 * min(afe, 0.4)) * L0, rtol=1e-12)


# ------------------------------------------------------------------ B4, B5
@pytest.mark.parametrize("mode", MODES)
def test_B4_linear_in_fagn_csp(csps, mode):
    """B4: the AGN term the CSP adds is linear in fagn (float64 term: 2x exactly; in the float32
    spectrum: to float32 rounding of the stellar part)."""
    cfg = "dattn_nodem_noneb"
    on = csps(cfg, mode)
    th = _theta(on, cfg, agn_tau=jnp.array([23.0]))
    a1 = on._agn_rest({**th, "fagn": jnp.array([0.25])})
    a2 = on._agn_rest({**th, "fagn": jnp.array([0.5])})
    assert np.asarray(2.0 * a1).tobytes() == np.asarray(a2).tobytes()
    s0 = np.asarray(on.get_spectrum(theta={**th, "fagn": jnp.array([0.0])}), float)
    s1 = np.asarray(on.get_spectrum(theta={**th, "fagn": jnp.array([0.25])}), float)
    s2 = np.asarray(on.get_spectrum(theta={**th, "fagn": jnp.array([0.5])}), float)
    scale = np.maximum(np.abs(s2), np.abs(s0))
    assert np.all(np.abs((s2 - s0) - 2.0 * (s1 - s0)) <= 4 * 6e-8 * scale + 1e-300)


def test_B4_energy_balance_linear(csps):
    """B4 with the energy balance (corrected + DL07): linear to 1e-12 (DL07's 1e-70 floor)."""
    cfg = "dattn_dem_noneb"
    on = csps(cfg, "corrected")
    assert on._agn.energy_balance
    th = _theta(on, cfg)
    a1 = np.asarray(on._agn_rest({**th, "fagn": jnp.array([0.25])}))
    a2 = np.asarray(on._agn_rest({**th, "fagn": jnp.array([0.5])}))
    np.testing.assert_allclose(a2, 2.0 * a1, rtol=1e-12, atol=1e-60)


def test_B5_lbol_normalisation(csps):
    """B5: without a screen L_torus = fagn * L_bol to rtol 1e-10, L_bol = the dust-free stellar
    integral (the per-SSP table, float64) = the float32 spectrum's integral to 1e-6."""
    cfg = "nodattn_nodem_noneb"
    on = csps(cfg, "corrected")
    w = on._agn._w
    for tau in (5.0, 12.0, 150.0, 300.0):
        th = _theta(on, cfg, fagn=jnp.array([0.4]), agn_tau=jnp.array([tau]))
        W = on.calculate_ssp_weights(th)
        L_bol = float(jnp.sum(W * on._agn_lbol_at(th)))
        np.testing.assert_allclose(float(jnp.dot(on._agn_rest(th), w)), 0.4 * L_bol, rtol=1e-10)
    stellar = on.get_spectrum_nodattn_nodem_noneb(th)
    np.testing.assert_allclose(float(jnp.dot(stellar.astype(jnp.float64), w)), L_bol, rtol=1e-6)


def test_screen_per_mode(csps):
    """The torus screen per mode: fsps and corrected x exp(-tau_diffuse) (dust2 only, no
    birth-cloud dust); prospector unattenuated (agnssp_basis.get_spectrum)."""
    cfg = "dattn_nodem_noneb"
    for mode in MODES:
        on = csps(cfg, mode)
        th = _theta(on, cfg, fagn=jnp.array([0.3]), diffuse_tau_kc=jnp.array([0.8]))
        ctx = on._agn_context(th, on.calculate_ssp_weights(th))
        intrinsic, att = on._agn.components(th, ctx)
        expect = intrinsic if mode == "prospector" else intrinsic * ctx.diffuse_curve
        assert np.array_equal(np.asarray(att), np.asarray(expect)), mode   # (+0 vs -0 aside)
        assert float(ctx.diffuse_curve.min()) < 0.5        # the screen is really there


# ------------------------------------------------------------------ B6
@pytest.mark.parametrize("cfg", ["dattn_dem_noneb", "dattn_dem_neb"])
def test_B6_energy_balance(csps, cfg):
    """B6: corrected + dust emission: the absorbed AGN light is re-emitted by DL07.
    (i) superposition: DL07(stellar + AGN) = DL07(stellar) + DL07(AGN) (so routing the AGN
    through its own call equals feeding it into the stellar call), spectrum and mdust;
    (ii) luminosity: the AGN's output deficit L(intrinsic) - L(out) is the DL07 self-absorption
    loss, the same fraction of the absorbed light as on the stellar path;
    (iii) without energy balance the deficit is exactly the absorbed luminosity."""
    on = csps(cfg, "corrected")
    th = _theta(on, cfg, fagn=jnp.array([0.5]), agn_tau=jnp.array([30.0]),
                diffuse_tau_kc=jnp.array([1.2]))
    hook, de, w = on._agn, on.dust_emi, on._agn._w
    W = on.calculate_ssp_weights(th)
    ctx = on._agn_context(th, W)
    I_a, A_a = hook.components(th, ctx)
    dpars = dict(duste_qpah=th["duste_qpah"], duste_umin=th["duste_umin"],
                 duste_gamma=th["duste_gamma"])
    dc = ctx.diffuse_curve
    I_s = jnp.einsum("za,zaw->w", W, jnp.asarray(on.flux, jnp.float64))
    A_s = I_s * dc * 0.7                                   # any attenuated stellar spectrum
    S_s, m_s, _ = de.compute_dust_emission(A_s, I_s, on.wave, dc, **dpars)
    S_a, m_a, _ = de.compute_dust_emission(A_a, I_a, on.wave, dc, **dpars)
    S_t, m_t, _ = de.compute_dust_emission(A_s + A_a, I_s + I_a, on.wave, dc, **dpars)
    np.testing.assert_allclose(np.asarray(S_t), np.asarray(S_s + S_a), rtol=1e-12,
                               atol=1e-12 * float(jnp.max(S_t)))
    np.testing.assert_allclose(float(m_t), float(m_s + m_a), rtol=1e-12)
    assert float(m_a) > 0
    dth = (th["duste_qpah"], th["duste_umin"], th["duste_gamma"])
    agn, absorbed = hook(th, ctx, de, dth)
    np.testing.assert_allclose(np.asarray(agn), np.asarray(S_a), rtol=1e-12,
                               atol=1e-12 * float(jnp.max(S_a)))
    m_unit, _ = hook.dust_response(ctx, de, dth)
    np.testing.assert_allclose(float(absorbed * m_unit), float(m_a), rtol=1e-12)
    L_I, L_out = float(jnp.dot(I_a, w)), float(jnp.dot(agn, w))
    loss_agn = (L_I - L_out) / float(absorbed)
    loss_star = (float(jnp.dot(I_s - S_s, w))) / float(jnp.dot(I_s - A_s, w))
    # the torus emits in the IR, where the screen is thin: ~1.6 % absorbed at tau_V = 1.2
    assert float(absorbed) > 0.005 * L_I
    assert abs(loss_agn - loss_star) < 0.02, (loss_agn, loss_star)
    assert 0 <= loss_agn < 0.2, loss_agn
    print(f"B6 {cfg}: absorbed {float(absorbed)/L_I:.4f} of L_AGN; DL07 self-absorption loss "
          f"{loss_agn:.4e} of it (stellar path {loss_star:.4e})")
    # (iii) the same CSP without energy balance: the deficit is the absorbed light, exactly
    off = csps(cfg, None)
    from ceridwen.agn.hook import AGNHook
    nob = AGNHook(torus_only("corrected", agn_energy_balance=False), off.wave,
                  add_dust_emission=True)
    a2, abs2 = nob(th, ctx)
    np.testing.assert_allclose(L_I - float(jnp.dot(a2, w)), float(abs2), rtol=1e-12)


def test_energy_balance_validation(build_csp):
    """agn_energy_balance=True needs dust emission; reproduction modes refuse it."""
    with pytest.raises(ValueError, match="add_dust_emission"):
        build_csp(torus_only("corrected", agn_energy_balance=True), add_dust_emission=False)
    with pytest.raises(ValueError, match="energy balance"):
        torus_only("fsps", agn_energy_balance=True)
    on = build_csp(torus_only("corrected"), add_dust_emission=False)
    assert on._agn.energy_balance is False


# ------------------------------------------------------------------ B7
def test_B7_igm_order(build_csp):
    """B7: at z = 8 the fsps-mode torus escapes the IGM (added after it, compsp.f90), the
    corrected one does not: (fsps - corrected-ordering) = agn x (1 - T_IGM), whose
    luminosity is the torus light blueward of Ly-alpha times the IGM absorption, 0.4-1.5 % of
    L_AGN for tau 5-150 (23 Sep design D2)."""
    z = 8.0
    fr = {}
    for mode in ("fsps", "corrected"):
        on = build_csp(torus_only(mode), add_igm=True, add_dust=False, add_diffuse_dust=False)
        for tau in (5.0, 10.0, 40.0, 150.0):
            th = {**on.theta_init, "agn_tau": jnp.array([tau]), "zred": jnp.array([z]),
                  "logmass": jnp.array([9.0])}
            th0 = {**th, "fagn": jnp.array([0.0])}
            th1 = {**th, "fagn": jnp.array([1.0])}
            s0, s1 = on.get_spectrum(theta=th0), on.get_spectrum(theta=th1)
            o0 = on._apply_mass_redshift_igm(s0, s0, th0)[0]      # same theta as the spectrum
            o1 = on._apply_mass_redshift_igm(s1, s1, th1)[0]
            trans = np.asarray(on._igm_transmission(jnp.array(z), th), np.float64)
            agn = np.asarray(on._agn_rest({**th, "fagn": jnp.array([1.0])}), np.float64)
            scale = float(10.0 ** 9.0 * on._flux_factor(th))
            expect = agn * scale * (1.0 if mode == "fsps" else trans)
            got = np.asarray(o1, np.float64) - np.asarray(o0, np.float64)
            # float32 spectra: the AGN term is exact to the rounding of the stellar part, and
            # XLA flushes float32 subnormals (products below ~1e-37 erg/s/cm^2/Hz = 1e-4 nJy)
            slack = 2e-6 * np.abs(expect) + 3e-7 * (np.abs(np.asarray(o0, np.float64))
                                                    + np.abs(np.asarray(o1, np.float64)))
            assert np.all(np.abs(got - expect) <= slack + 1e-36), (mode, tau)
            w = np.asarray(on._agn._w)
            blue = np.asarray(on.wave) < 1215.67
            fr[(mode, tau)] = (float(np.dot(agn * (1.0 - trans), w) / np.dot(agn, w)),
                               float(np.dot(agn * blue, w) / np.dot(agn, w)))
    for tau in (5.0, 10.0, 40.0, 150.0):
        absorbed, blueward = fr[("corrected", tau)]
        # the IGM at z = 8 removes nearly all of the light blueward of Ly-alpha ...
        assert 0.9 * blueward < absorbed <= blueward * (1 + 1e-12), (tau, absorbed, blueward)
        # ... which is 0.4-1.5 % of L_AGN over tau = 5-150
        assert 0.0035 < blueward < 0.0155, (tau, blueward)
    assert fr[("corrected", 5.0)][1] < fr[("corrected", 40.0)][1] < fr[("corrected", 150.0)][1]
    print("B7 (IGM-absorbed, blueward-of-Lya) fractions of L_AGN at z=8:",
          {k: tuple(round(x, 5) for x in v) for k, v in fr.items()})


@pytest.mark.parametrize("sigma_gal", (0.0, 300.0))
def test_B7_predict_after_igm(csps, sigma_gal):
    """The predict path applies the same rule, also through the kinematic before-IGM branch
    (sigma_gal = 300 km/s, the SedModel default): at z = 8 the photometry gains the AGN times
    T_agn (fsps: 1; prospector: fake_fsps.add_igm; corrected: the galaxy's IGM)."""
    from ceridwen.broadening import Kinematics
    from ceridwen.agn.hook import prospector_igm_transmission
    import warnings
    cfg = "nodattn_nodem_noneb"
    gain = {}
    for mode in MODES:
        on = csps(cfg, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SedModel(on, _obs(phot_only=True), priors={}, zred=8.0,
                         free_param_init={"logmass": jnp.array([10.0])},
                         kinematics=Kinematics(sigma_gal=sigma_gal))
        p0 = m.predict_jit({**m.theta_init, "fagn": jnp.array([0.0])})["phot"]
        p1 = m.predict_jit({**m.theta_init, "fagn": jnp.array([0.7])})["phot"]
        th = {**m.theta_init, "fagn": jnp.array([0.7]), "zred": jnp.array([8.0])}
        agn = on._agn_rest(th) * 10.0 ** 10.0 * on._flux_factor(th)
        if mode == "corrected":
            agn = agn * on._igm_transmission(jnp.array(8.0), th)
        elif mode == "prospector":
            agn = agn * prospector_igm_transmission(on.wave, 8.0, 1.0)
        expect = m.observations[0].predict(agn.astype(jnp.float32), on.wave)
        # float32 maggies: the difference carries the rounding of the stellar part; with
        # sigma_gal = 300 the galaxy kernel also smooths the AGN's Ly-alpha break
        d, e = np.asarray(p1 - p0, np.float64), np.asarray(expect, np.float64)
        slack = (1e-5 if sigma_gal == 0 else 2e-3) * np.abs(e) \
            + 3e-7 * (np.abs(np.asarray(p0)) + np.abs(np.asarray(p1)))
        assert np.all(np.abs(d - e) <= slack), (mode, d, e)
        gain[mode] = float((p1 - p0)[0])
    # F090W at z = 8 is rest 890-1110 A: the fsps torus there escapes the IGM entirely
    assert gain["fsps"] > 10 * gain["corrected"] and gain["fsps"] > 10 * gain["prospector"]


def _np_add_igm(wave, spec, zred, igm_factor=1.0):
    """prospect/sources/fake_fsps.py add_igm, transcribed line by line in NumPy."""
    redshifted_wave = np.asarray(wave) * (1 + zred)
    lylim = 911.75
    lyw = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150,
                    920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919,
                    914.576])
    lycoeff = np.array([0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960, 0.0006967, 0.0006236,
                        0.0005665, 0.0005200, 0.0004817, 0.0004487, 0.0004200, 0.0003947,
                        0.000372, 0.000352, 0.0003334, 0.00031644])
    tau_line = np.zeros_like(redshifted_wave)
    for i in range(len(lyw)):
        lmin = lyw[i]
        lmax = lyw[i] * (1.0 + zred)
        idx0 = np.where((redshifted_wave >= lmin) & (redshifted_wave <= lmax))[0]
        tau_line[idx0] += lycoeff[i] * np.exp(3.46 * np.log(redshifted_wave[idx0] / lyw[i]))
    xc = redshifted_wave / lylim
    xem = 1.0 + zred
    xc[xc < 1.0] = 1.0
    xc[xc > xem] = xem
    tau_cont = 0.25 * xc**3 * (np.exp(0.46 * np.log(xem)) - np.exp(0.46 * np.log(xc)))
    tau_cont = tau_cont + 9.4 * np.exp(1.5 * np.log(xc)) * (np.exp(0.18 * np.log(xem))
                                                            - np.exp(0.18 * np.log(xc)))
    tau_cont = tau_cont - 0.7 * xc**3 * (np.exp(-1.32 * np.log(xc)) - np.exp(-1.32 * np.log(xem)))
    tau_cont = tau_cont - 0.023 * (np.exp(1.68 * np.log(xem)) - np.exp(1.68 * np.log(xc)))
    tau = tau_line + tau_cont
    return np.clip(spec * np.exp(-tau * igm_factor), 1e-70, None)


@pytest.mark.parametrize("z", (0.5, 2.0, 6.0, 8.0))
def test_prospector_igm_port(model_wave, z):
    """prospector_igm_transmission == fake_fsps.add_igm (NumPy transcription) to 1e-12 where
    the transmission exceeds 1e-60 (below, add_igm's 1e-70 clip on the spectrum takes over)."""
    from ceridwen.agn.hook import prospector_igm_transmission
    ref = _np_add_igm(model_wave, np.ones_like(model_wave), z, 1.3)
    got = np.asarray(prospector_igm_transmission(model_wave, z, 1.3))
    m = ref > 1e-60
    np.testing.assert_allclose(got[m], ref[m], rtol=1e-12)
    assert np.all(got[~m] <= 1e-60)


# ------------------------------------------------------------------ B8
@pytest.mark.parametrize("mode", MODES)
def test_B8_vmap_grad_predict(csps, mode):
    """B8: jit(vmap(predict)) at W = 64 and 100 equals the per-draw jit, and the gradient of the
    log-likelihood-like sum is finite and non-zero in fagn and agn_tau (inside (5, 150))."""
    cfg = "dattn_dem_noneb" if mode == "corrected" else "dattn_nodem_noneb"
    on = csps(cfg, mode)
    m = _model(on, cfg, phot_only=True)
    base = dict(m.theta_init)
    rng = np.random.default_rng(1)
    for W in (64, 100):
        batch = {k: jnp.broadcast_to(v, (W,) + v.shape) for k, v in base.items()}
        batch["fagn"] = jnp.asarray(rng.uniform(1e-3, 2.0, (W, 1)))
        batch["agn_tau"] = jnp.asarray(rng.uniform(5.0, 150.0, (W, 1)))
        pv = m.predict_vmap(batch)["phot"]
        for i in (0, W // 2, W - 1):
            pi = m.predict_jit({k: v[i] for k, v in batch.items()})["phot"]
            # float32 FFT smoothing in the photometric broadening, batched vs single: rtol B8
            # plus the FFT bound (was rtol=1e-5)
            np.testing.assert_allclose(np.asarray(pv[i]), np.asarray(pi), rtol=B8,
                                       atol=fft_atol(pi, 2 ** 16))

    def loss(fagn, tau):
        p = m.predict({**base, "fagn": fagn, "agn_tau": tau})["phot"]
        return jnp.sum(jnp.log(p))
    for tau in (7.5, 33.0, 120.0):
        gf, gt = jax.grad(loss, argnums=(0, 1))(jnp.array([0.3]), jnp.array([tau]))
        assert np.isfinite(float(gf[0])) and np.isfinite(float(gt[0]))
        assert float(gf[0]) != 0.0 and float(gt[0]) != 0.0
    gz = jax.grad(loss)(jnp.array([0.0]), jnp.array([10.0]))     # smooth at fagn = 0 (D3)
    assert np.isfinite(float(gz[0])) and float(gz[0]) > 0


def test_B8_afe_and_support_grad(afe_ssp, csps, torus_path):
    """B8 on the a/Fe CSP (sampled afe) and with wavelength_support=True: finite gradients."""
    on = _afe(afe_ssp, torus_only("corrected"))
    th = dict(on.theta_init)

    def f(fagn, tau, afe):
        s = on.get_spectrum(theta={**th, "fagn": fagn, "agn_tau": tau, "afe": afe})
        return jnp.sum(jnp.log1p(s))
    g = jax.grad(f, argnums=(0, 1, 2))(jnp.array([0.3]), jnp.array([20.0]), jnp.array([0.1]))
    assert all(np.isfinite(float(x[0])) for x in g) and float(g[0][0]) != 0

    cfg = "dattn_nodem_noneb"
    sup = csps(cfg, "corrected", wavelength_support=True)
    m = _model(sup, cfg, phot_only=True)
    base = dict(m.theta_init)
    gs = jax.grad(lambda a: jnp.sum(m.predict({**base, "fagn": a})["phot"]))(jnp.array([0.2]))
    full = csps(cfg, "corrected")
    mf = _model(full, cfg, phot_only=True)
    gf = jax.grad(lambda a: jnp.sum(mf.predict({**dict(mf.theta_init), "fagn": a})["phot"]))(
        jnp.array([0.2]))
    np.testing.assert_allclose(float(gs[0]), float(gf[0]), rtol=1e-5)


# ------------------------------------------------------------------ setup-time checks
def test_theta_keys_known_and_ranges(build_csp):
    """The AGN keys are known (no 'silently ignored' warning), defaults per mode, and
    check_param_ranges warns for agn_tau outside [5, 150] and negative fagn."""
    import warnings
    on = build_csp(torus_only("corrected"))
    assert {"fagn", "agn_tau"} <= on._known_theta_keys
    assert float(on.theta_init["agn_tau"][0]) == 10.0 and float(on.theta_init["fagn"][0]) == 0.0
    pr = build_csp(torus_only("prospector"))
    assert float(pr.theta_init["agn_tau"][0]) == 50.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on._warn_unknown_theta_keys({**on.theta_init})
    with pytest.warns(UserWarning, match="SILENTLY IGNORED"):
        on._warn_unknown_theta_keys({**on.theta_init, "f_agn": jnp.array([0.1])})
    msgs = on.check_param_ranges({**on.theta_init, "agn_tau": jnp.array([300.0]),
                                  "fagn": jnp.array([-0.1])}, warn=False)
    assert any("agn_tau" in m and "clamped" in m for m in msgs)
    assert any("fagn" in m and "negative" in m for m in msgs)
    msgs = pr.check_param_ranges({**pr.theta_init, "agn_tau": jnp.array([2.0])}, warn=False)
    assert any("EXTRApolated" in m for m in msgs)
    with pytest.raises(NotImplementedError, match="dust4_type=2"):
        AGNConfig(agn_mode="prospector", dust4_type=2)
    with pytest.raises(TypeError, match="AGNConfig"):
        build_csp({"agn_mode": "fsps"})


def test_agn_off_untouched(build_csp):
    """Without agn= the CSP has no hook and its get_spectrum is the plain kernel."""
    off = build_csp(None)
    assert off._agn is None
    assert getattr(off.get_spectrum, "__func__", None) is type(off).get_spectrum_dattn_nodem_noneb
    assert "fagn" not in off.theta_init and "fagn" not in off._known_theta_keys


def test_check_reports_torus_file(tmp_path, monkeypatch, capsys):
    """python -m ceridwen.check reports the optional Nenkova+08 file, ok or missing."""
    from ceridwen.check import check_environment
    monkeypatch.setenv("SPS_HOME", str(tmp_path))
    check_environment(verbose=True)
    assert "[warn] AGN torus templates" in capsys.readouterr().out
    (tmp_path / "dust").mkdir()
    (tmp_path / "dust" / "Nenkova08_y010_torusg_n10_q2.0.dat").write_text("x")
    check_environment(verbose=True)
    assert "[ok  ] AGN torus templates" in capsys.readouterr().out
