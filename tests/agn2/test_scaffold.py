"""AGN2 scaffold (session VAL, Phase A): config, dispatch, the AGN2 wiring of the hook and the
CSP, several AGN line sets at once. The components are the toys of ``_toys.py`` (swapped in
through ``ceridwen.agn.AGN2_REGISTRY``); the physics tests of the real components are the
``V-<component>-<n>`` tests of the suite (Phase B).

Requirements (AGN2_BRIEF.md "Modes", "Config and theta names", "Physics conventions";
PROMPT_agn2_VAL.md Phase A):

V-SCAF-1   AGNConfig(): every pre-AGN2 field and resolved switch unchanged; the AGN2 fields
           default to off.
V-SCAF-2   every refused combination raises at construction, with a message naming the field.
V-SCAF-3   a component whose module does not exist raises NotImplementedError when the CSP is
           built, not at import; a component that declares other theta keys than
           AGN2_THETA_KEYS is refused at setup.
V-SCAF-4   corrected: the AGN continuum = (disk T_pol + dust + polar_ir) x T_gal (float64,
           independent recomputation from the toys), added to the base kernel in its dtype.
V-SCAF-5   cigale: the AGN2 continuum sees no galaxy dust (independent of diffuse_tau_kc) and there is
           no energy balance.
V-SCAF-6   polar dust: re-emission integrates to the absorbed disc light (toy contract), and
           the hook passes it through unchanged.
V-SCAF-7   energy balance (corrected, DL07): L_absorbed = galaxy-absorbed nuclear continuum +
           NLR x (1 - T_gal) + BLR x T_pol x (1 - T_gal); the AGN adds its nuclear luminosity
           minus nothing (DL07 re-emits the absorbed part); polar absorption never reaches DL07.
V-SCAF-8   per line set transmissions: NLR = T_gal, BLR = T_pol x T_gal (corrected);
           NLR = 1, BLR = T_pol (cigale); independent of each other's parameters.
V-SCAF-9   Lines observation: NLR and BLR summed BY FSPS NAME with the nebular line, each with
           its own transmission, equal to an independent computation.
V-SCAF-10  Spectrum: each set painted with its own width: the broad H alpha has
           rms = sqrt(blr_sigma^2 + sigma_inst^2), its flux ratio to broad H beta is the
           input ratio, its centre lam (1 + z); the NLR width is untouched by blr_sigma.
V-SCAF-11  zero amplitudes (fagn = agn_elum = blr_lum = 0, polar_ebv > 0): every prediction
           byte-identical to the CSP without agn=, with and without IGM.
V-SCAF-12  linear in blr_lum and agn_elum at the prediction level.
V-SCAF-13  jit, vmap (W = 64, 100) == loop, finite gradients of a log-likelihood in every AGN2
           theta key at the defaults and at range edges; central differences for amplitudes.
V-SCAF-14  ctx.L_dust_gal equals the luminosity DL07 re-emits (CSP with dust emission minus
           the same CSP without), when a component asks for it, with or without DL07; such a
           component is refused without any galaxy dust screen (PLAN_AGN2 2.2, v2).
V-SCAF-16  a SKIRTOR class with its own nuclear_parts(theta, ctx, polar) is used as is by the
           hook (continuum, energy balance), NOTE_TORUS_contract.md.
V-SCAF-15  _agn_line_terms keeps its contract (the "nlr" set, None with BLR only); the line
           marginalisation is refused with BLR only; provenance names the AGN2 parts.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_emission_lines as T  # noqa: E402
import _toys  # noqa: E402

import ceridwen.agn as agnmod  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData  # noqa: E402
from ceridwen.agn import AGNConfig, AGN2_THETA_KEYS  # noqa: E402

CKMS = 2.99792458e5

ALL_ON = dict(agn_mode="corrected", torus_model="skirtor16", polar_dust=True,
              nlr_model="feltre16", blr=True)
ON = {"fagn": 0.3, "skirtor_t": 5.0, "skirtor_pl": 0.5, "skirtor_q": 1.2, "skirtor_oa": 50.0,
      "skirtor_i": 20.0, "polar_ebv": 0.2, "polar_temp": 150.0, "polar_emis": 1.5,
      "agn_elum": 3e-3, "agn_eline_sigma": 150.0, "nlr_logu": -2.0, "nlr_logz": -0.3,
      "nlr_alpha": -1.4, "nlr_xid": 0.5, "nlr_lognh": 2.5, "blr_lum": 1e-2, "blr_sigma": 1800.0}
ZERO = dict(ON, fagn=0.0, agn_elum=0.0, blr_lum=0.0)
NEW_KEYS = tuple(k for k in ON if k not in ("fagn", "agn_elum", "agn_eline_sigma"))


def _vals(d):
    return {k: jnp.array([float(v)]) for k, v in d.items()}


def _need():
    path = T.find_test_grid()
    if path is None:
        pytest.skip("MISSING test SSP grid (tests/_gridfixture.py): test did NOT run")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("MISSING $SPS_HOME (FSPS line list, CLOUDY grids): test did NOT run")
    return path


class _Toys:
    """Context manager: the toy classes in AGN2_REGISTRY while a CSP is built."""

    def __enter__(self):
        self.saved = dict(agnmod.AGN2_REGISTRY)
        agnmod.AGN2_REGISTRY.update(_toys.TOYS)

    def __exit__(self, *a):
        agnmod.AGN2_REGISTRY.clear()
        agnmod.AGN2_REGISTRY.update(self.saved)


_CACHE = {}


def _csp(cfg=None, *, igm=False, dem=False, neb=True):
    key = (repr(cfg), igm, dem, neb)
    if key in _CACHE:
        return _CACHE[key]
    path = _need()
    cosmo = Cosmology.planck18()
    with _Toys():
        csp = CSPBasis(SSPData.load(str(path)),
                       lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
                       zh_const=True, sfh_interp="step", add_dust=dem, add_diffuse_dust=True,
                       add_neb=neb, add_igm=igm, add_dust_emission=dem, verbose=False,
                       cosmo=cosmo, sps_home=os.environ["SPS_HOME"], agn=cfg)
    _CACHE[key] = csp
    return csp


def _model(csp, obs=None):
    spec, phot, lines = T._obs()
    obs = (spec, phot, lines) if obs is None else obs
    return T._model(csp, obs), spec, phot, lines


def _bytes(d):
    return {k: (np.asarray(v).dtype, np.asarray(v).tobytes()) for k, v in d.items()}


def _mt(model, vals=None, **over):
    th = {**T._theta(model, **over), **_vals(vals or {})}
    return th, T._csp_theta(model, th)


# ------------------------------------------------------------------ V-SCAF-1
def test_V_SCAF_1_defaults_unchanged():
    c = AGNConfig()
    assert (c.agn_mode, c.torus, c.disk, c.lines, c.dust4_type, c.tau_interp,
            c.agn_energy_balance, c.torus_file) == ("corrected", None, None, None, 0, "linear",
                                                    None, None)
    assert (c.use_torus, c.use_disk, c.use_lines) == (True, True, True)
    assert (c.torus_model, c.polar_dust, c.polar_law, c.nlr_model, c.blr, c.skirtor_R,
            c.skirtor_file, c.nlr_file, c.blr_file) == ("nenkova08", False, None, "a42", False,
                                                        None, None, None, None)
    assert not c.uses_agn2 and c.use_nenkova and not c.use_skirtor
    for mode, disk, lines in (("fsps", False, False), ("prospector", True, True)):
        cm = AGNConfig(agn_mode=mode)
        assert (cm.use_torus, cm.use_disk, cm.use_lines, cm.uses_agn2) == (True, disk, lines,
                                                                           False)
    assert agnmod.AGN_MODES[:3] == ("fsps", "prospector", "corrected")
    assert set(AGN2_THETA_KEYS) == {"skirtor16", "polar", "feltre16", "blr"}


# ------------------------------------------------------------------ V-SCAF-2
REFUSED = [
    (dict(agn_mode="fsps", blr=True), "do not exist in FSPS"),
    (dict(agn_mode="prospector", torus_model="skirtor16"), "do not exist in PROSPECTOR"),
    (dict(agn_mode="prospector", nlr_model="feltre16"), "do not exist"),
    (dict(agn_mode="fsps", polar_dust=True), "do not exist"),
    (dict(agn_mode="cigale"), "no Nenkova torus"),
    (dict(agn_mode="cigale", torus_model="skirtor16", disk=True), "own anisotropic"),
    (dict(agn_mode="cigale", torus=False, disk=True), "no Temple"),
    (dict(agn_mode="cigale", torus=False, dust4_type=1), "no dust4"),
    (dict(agn_mode="cigale", torus=False, lines=True), "no Richardson"),
    (dict(agn_mode="cigale", torus=False, agn_energy_balance=True), "CIGALE adds its"),
    (dict(agn_mode="cigale", torus_model="skirtor16", tau_interp="log"), "tau_interp"),
    (dict(torus_model="skirtor16", disk=True), "count the disc"),
    (dict(torus_model="skirtor16", torus_file="x.dat"), "torus_file"),
    (dict(torus_model="skirtor16", dust4_type=1), "dust4 screen"),
    (dict(torus_model="skirtor16", tau_interp="log"), "tau_interp"),
    (dict(torus_model="skirtor16", torus=False), "torus=False"),
    (dict(torus_model="skirtor16", skirtor_R=0.5), "skirtor_R"),
    (dict(torus_model="skirtor16", skirtor_R=True), "skirtor_R"),
    (dict(skirtor_R=20.0), "needs torus_model='skirtor16'"),
    (dict(skirtor_file="s.h5"), "needs torus_model='skirtor16'"),
    (dict(polar_dust=True), "needs torus_model='skirtor16'"),
    (dict(torus_model="skirtor16", polar_law="smc"), "needs polar_dust=True"),
    (dict(torus_model="skirtor16", polar_dust=True, polar_law="mw"), "polar_law"),
    (dict(nlr_model="feltre16", lines=False), "lines=False"),
    (dict(nlr_file="n.h5"), "needs nlr_model='feltre16'"),
    (dict(blr_file="b.h5"), "needs blr=True"),
    (dict(torus_model="x"), "torus_model"),
    (dict(nlr_model="x"), "nlr_model"),
    (dict(blr=1), "expected True or False"),
    (dict(polar_dust=None), "expected True or False"),
    (dict(agn_mode="CIGALE"), "choose one of"),
]


@pytest.mark.parametrize("kw,msg", REFUSED, ids=[str(k) for k, _ in REFUSED])
def test_V_SCAF_2_refused(kw, msg):
    with pytest.raises((ValueError, TypeError), match=msg.replace("(", r"\(")):
        AGNConfig(**kw)


def test_V_SCAF_2_accepted():
    c = AGNConfig(agn_mode="cigale", torus_model="skirtor16", polar_dust=True,
                  nlr_model="feltre16", blr=True, skirtor_R=20.0)
    assert (c.use_skirtor, c.use_disk, c.use_lines, c.polar_law_resolved, c.uses_agn2) == (
        True, False, True, agnmod.POLAR_LAW_DEFAULT, True)
    assert AGNConfig(agn_mode="cigale", torus_model="skirtor16").use_lines is False
    assert AGNConfig(agn_mode="cigale", torus=False, blr=True).use_any_lines
    c = AGNConfig(torus_model="skirtor16")              # corrected: no Temple disc with SKIRTOR
    assert (c.use_disk, c.use_lines, c.nlr_model) == (False, True, "a42")
    assert AGNConfig(blr=True).use_disk                  # Nenkova + Temple + BLR stays possible


# ------------------------------------------------------------------ V-SCAF-3
def test_V_SCAF_3_missing_module_raises_at_setup(monkeypatch):
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "skirtor16",
                        ("ceridwen.agn._agn2_does_not_exist", "SkirtorTorus"))
    with pytest.raises(NotImplementedError, match="not implemented on this branch"):
        agnmod.AGNModel(AGNConfig(torus_model="skirtor16", lines=False),
                        np.linspace(1e3, 1e5, 50))
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "blr", ("ceridwen.agn.hook", "NoSuchClass"))
    with pytest.raises(NotImplementedError, match="no class NoSuchClass"):
        agnmod.AGNModel(AGNConfig(torus=False, disk=False, lines=False, blr=True),
                        np.linspace(1e3, 1e5, 50))


def test_V_SCAF_3_contract_checked(monkeypatch):
    class BadKeys(_toys.ToyBLR):
        param_names = ("blr_lum", "blr_width")

    class BadRegion(_toys.ToyBLR):
        region = "nlr"

    if not os.environ.get("SPS_HOME"):
        pytest.skip("MISSING $SPS_HOME (FSPS line list): test did NOT run")
    cfg = AGNConfig(torus=False, disk=False, lines=False, blr=True)
    for bad, msg in ((BadKeys, "AGN2 contract"), (BadRegion, "region")):
        monkeypatch.setitem(agnmod.AGN2_REGISTRY, "blr", bad)
        with pytest.raises(ValueError, match=msg):
            agnmod.AGNModel(cfg, np.linspace(1e3, 1e5, 50))


# ------------------------------------------------------------------ helpers
def _expected_nuclear(csp, mt):
    """(disk, dust, T_pol, polar_ir, nuclear) from the toy closed forms, float64 NumPy."""
    hook = csp._agn
    th = hook.filled(mt)
    W = csp.calculate_ssp_weights(theta=mt)
    lbol = float(jnp.sum(W * csp._agn_lbol_at(mt)))
    wave = np.asarray(csp.wave, dtype=np.float64)
    L = float(np.ravel(th["fagn"])[0]) * lbol
    disk = L * float(_toys.disk_amp(th)) * _toys._unit_shape(wave, 1500.0, 0.6)
    dust = L * float(_toys.dust_amp(th)) * _toys._unit_shape(wave, 2.0e5, 0.5)
    if hook.model.polar is None:
        return disk, dust, np.ones_like(wave), np.zeros_like(wave), disk + dust
    t = np.exp(-float(np.ravel(th["polar_ebv"])[0]) * 3.0 * (wave / 5500.0) ** -1.2)
    w = _toys._w(wave)
    L_abs = (disk * (1 - t)) @ w
    nu = _toys.C_AA_S / wave
    x = _toys.H_OVER_K * nu / float(np.ravel(th["polar_temp"])[0])
    shape = (nu / 1e13) ** (3.0 + float(np.ravel(th["polar_emis"])[0])) / np.expm1(
        np.minimum(x, 700.0))
    ir = L_abs * shape / (shape @ w)
    return disk, dust, t, ir, disk * t + dust + ir


def _diffuse(csp, mt):
    return np.exp(-np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64))


# ------------------------------------------------------------------ V-SCAF-4 / 5 / 6
@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_SCAF_4_5_continuum_wiring(mode):
    cfg = AGNConfig(agn_mode=mode, torus_model="skirtor16", polar_dust=True, lines=False)
    csp = _csp(cfg)
    model, _, _, _ = _model(csp)
    for tau in (0.0, 0.6):
        _, mt = _mt(model, ON, diffuse_tau_kc=jnp.array([tau]))
        got = np.asarray(csp._agn_rest(mt))
        disk, dust, t, ir, nuc = _expected_nuclear(csp, mt)
        gal = _diffuse(csp, mt) if mode == "corrected" else 1.0
        exp = nuc * gal
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14 * np.abs(exp).max())
        # added to the base kernel in its dtype, nothing else
        base = csp._agn_base(mt)        # the wrapped kernel, resolved on the instance
        full = csp.get_spectrum(mt)
        assert np.array_equal(np.asarray(full),
                              np.asarray(base + jnp.asarray(got).astype(base.dtype)))
    assert csp._agn.energy_balance is False
    assert csp._agn.own_igm is False


def test_V_SCAF_6_polar_energy_passthrough():
    csp = _csp(AGNConfig(torus_model="skirtor16", polar_dust=True, lines=False))
    model, _, _, _ = _model(csp)
    _, mt = _mt(model, ON)
    hook = csp._agn
    th = hook.filled(mt)
    W = csp.calculate_ssp_weights(theta=mt)
    ctx = csp._agn_context(mt, W)
    p = hook.nuclear_parts(th, ctx)
    w = np.asarray(hook._w)
    absorbed = float(np.asarray(p["disk"] * (1 - p["polar_t"])) @ w)
    assert absorbed > 0
    assert float(np.asarray(p["polar_ir"]) @ w) == pytest.approx(absorbed, rel=1e-12)
    _, _, t, ir, nuc = _expected_nuclear(csp, mt)
    np.testing.assert_allclose(np.asarray(p["nuclear"]), nuc, rtol=1e-12,
                               atol=1e-14 * nuc.max())


# ------------------------------------------------------------------ V-SCAF-7
def test_V_SCAF_7_energy_balance():
    cfg = AGNConfig(**ALL_ON)
    csp = _csp(cfg, dem=True)
    assert csp._agn.energy_balance
    model, _, _, _ = _model(csp)
    _, mt = _mt(model, ON, diffuse_tau_kc=jnp.array([0.5]))
    hook = csp._agn
    th = hook.filled(mt)
    W = csp.calculate_ssp_weights(theta=mt)
    msum = float(jnp.sum(W))
    dc = _diffuse(csp, mt)
    wave = np.asarray(csp.wave)
    w = _toys._w(wave)
    _, _, t, _, nuc = _expected_nuclear(csp, mt)
    nlr, blr = hook.model.lines, hook.model.blr
    t_pol_b = np.interp(blr.line_wave, wave, t)
    L_n = np.asarray(nlr.line_luminosity(th)) * msum
    L_b = np.asarray(blr.line_luminosity(th)) * msum
    exp_abs = ((nuc * (1 - dc)) @ w
               + np.sum(L_n * (1 - np.interp(nlr.line_wave, wave, dc)))
               + np.sum(L_b * t_pol_b * (1 - np.interp(blr.line_wave, wave, dc))))
    parts = csp._agn_parts(mt)
    assert float(parts["L_absorbed"]) == pytest.approx(exp_abs, rel=1e-10)
    # the polar absorption is not in it: the same with polar_ebv = 0 differs by exactly the
    # change of the nuclear spectrum, never by the absorbed disc light
    assert float(parts["L_polar_ir"]) > 0
    # luminosity: the AGN continuum the CSP adds = attenuated nuclear + DL07 response to all
    # galaxy-absorbed AGN light (lines included), which DL07 re-emits in full
    agn = np.asarray(csp._agn_rest(mt))
    lhs = agn @ w
    rhs = (nuc * dc) @ w + exp_abs
    assert lhs == pytest.approx(rhs, rel=2e-4)      # DL07's own float32 / 1e-70 floor loss
    assert float(parts["mdust"]) > 0


# ------------------------------------------------------------------ V-SCAF-8
@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_SCAF_8_line_set_transmissions(mode):
    cfg = AGNConfig(**dict(ALL_ON, agn_mode=mode))
    csp = _csp(cfg)
    model, _, _, _ = _model(csp)
    hook = csp._agn
    wave = np.asarray(csp.wave)
    for over in ({}, {"polar_ebv": 0.8}, {"dust2": 0.7}, {"dust2": 0.7, "polar_ebv": 0.0}):
        vals = dict(ON, **{k: v for k, v in over.items() if k != "dust2"})
        _, mt = _mt(model, vals, **({"diffuse_tau_kc": jnp.array([over["dust2"]])}
                                    if "dust2" in over else {}))
        th = hook.filled(mt)
        dc = _diffuse(csp, mt)
        t = np.exp(-vals["polar_ebv"] * 3.0 * (wave / 5500.0) ** -1.2)
        tn = hook.line_set_transmission("nlr", th, jnp.asarray(dc))
        tb = np.asarray(hook.line_set_transmission("blr", th, jnp.asarray(dc)))
        lb = hook.model.blr.line_wave
        ln = hook.model.lines.line_wave
        if mode == "corrected":
            np.testing.assert_allclose(np.asarray(tn), np.interp(ln, wave, dc), rtol=1e-13)
            np.testing.assert_allclose(tb, np.interp(lb, wave, t) * np.interp(lb, wave, dc),
                                       rtol=1e-12)
        else:
            assert tn is None
            np.testing.assert_allclose(tb, np.interp(lb, wave, t), rtol=1e-12)


# ------------------------------------------------------------------ V-SCAF-9
@pytest.mark.parametrize("igm", [False, True])
def test_V_SCAF_9_lines_observation_sums_by_name(igm):
    csp = _csp(AGNConfig(**ALL_ON), igm=igm)
    model, _, _, lines = _model(csp)
    th0, mt0 = _mt(model, ZERO, diffuse_tau_kc=jnp.array([0.4]))
    th1, mt1 = _mt(model, ON, diffuse_tau_kc=jnp.array([0.4]))
    d = np.asarray(model.predict(th1)["lines"], np.float64) - np.asarray(
        model.predict(th0)["lines"], np.float64)
    hook = csp._agn
    th = hook.filled(mt1)
    wave = np.asarray(csp.wave)
    msum = float(jnp.sum(csp.calculate_ssp_weights(theta=mt1)))
    scale = msum * 10 ** float(mt1["logmass"][0]) * float(csp._flux_factor(mt1)) / (1 + T.ZRED)
    dc = _diffuse(csp, mt1)
    tig = (np.asarray(csp._igm_transmission(jnp.asarray(T.ZRED), mt1), np.float64) if igm
           else np.ones_like(wave))
    t = np.exp(-ON["polar_ebv"] * 3.0 * (wave / 5500.0) ** -1.2)
    exp = np.zeros(len(lines.line_components))
    for k, comp in enumerate(lines.line_components):
        for lam in comp:
            for ls, extra in ((hook.model.lines, 1.0), (hook.model.blr, t)):
                j = int(ls.rows_for([lam])[0])
                if j < 0:
                    continue
                lw = ls.line_wave[j]
                tr = np.interp(lw, wave, dc) * np.interp(lw, wave, tig) * (
                    1.0 if np.isscalar(extra) else np.interp(lw, wave, extra))
                exp[k] += float(ls.line_luminosity(th)[j]) * scale * tr
    names = [n for n, _ in T.LINES]
    ha = names.index("Halpha")
    assert exp[ha] > 0 and d[ha] > 0
    np.testing.assert_allclose(d, exp, rtol=2e-6, atol=1e-6 * np.abs(exp).max())


# ------------------------------------------------------------------ V-SCAF-10
def test_V_SCAF_10_spectrum_widths_per_set():
    csp = _csp(AGNConfig(**ALL_ON))
    model, spec, _, _ = _model(csp)
    th0, mt0 = _mt(model, dict(ON, blr_lum=0.0))
    # the test spectrum covers rest 4500-7000 A: both broad lines keep +-6 sigma on it up to
    # blr_sigma ~ 3000 km/s (larger widths are tested on the model grid, V-BLR suite)
    for sig in (1200.0, 3000.0):
        th1, mt1 = _mt(model, dict(ON, blr_sigma=sig, blr_lum=1.0))
        d = np.asarray(model.predict(th1)["spec"], np.float64) - np.asarray(
            model.predict(th0)["spec"], np.float64)
        wo = np.asarray(spec.wavelength, np.float64)
        s_inst = np.asarray(spec._proj.sigma_inst_kms, np.float64)
        res = {}
        for name, lam in (("ha", 6564.7229), ("hb", 4862.7629)):
            c = lam * (1 + T.ZRED)
            s_tot = np.hypot(sig, np.interp(c, wo, s_inst)) / CKMS
            m = np.abs(np.log(wo / c)) < 6 * s_tot
            assert m.sum() > 20 and np.log(wo[m].max() / c) > 5 * s_tot and \
                np.log(c / wo[m].min()) > 5 * s_tot, (name, sig, "line wings off the spectrum")
            flux, mean, rms = T._moments(wo[m], d[m], c)
            res[name] = flux
            assert abs(mean) < 0.02 * s_tot, (name, sig, mean)
            assert rms == pytest.approx(s_tot, rel=0.02), (name, sig, rms, s_tot)
        if True:
            wave = np.asarray(csp.wave)
            t = np.exp(-ON["polar_ebv"] * 3.0 * (wave / 5500.0) ** -1.2)
            dc = _diffuse(csp, mt1)
            tr = lambda lam: np.interp(lam, wave, t) * np.interp(lam, wave, dc)  # noqa: E731
            want = (1 / 3.1) * tr(4862.7629) / tr(6564.7229)
            assert res["hb"] / res["ha"] == pytest.approx(want, rel=5e-3)


# ------------------------------------------------------------------ V-SCAF-11
@pytest.mark.parametrize("igm", [False, True])
@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_SCAF_11_zero_amplitude_byte_identical(mode, igm):
    off, _, _, _ = _model(_csp(None, igm=igm))
    on, _, _, _ = _model(_csp(AGNConfig(**dict(ALL_ON, agn_mode=mode)), igm=igm))
    base = off.predict_jit(T._theta(off))
    got = on.predict_jit({**T._theta(on), **_vals(ZERO)})
    assert set(got) == set(base)
    assert _bytes(got) == _bytes(base)
    lit = on.predict_jit({**T._theta(on), **_vals(ON)})
    for k in base:
        assert np.any(np.asarray(lit[k]) != np.asarray(base[k])), k


def test_V_SCAF_11_zero_amplitude_energy_balance():
    off, _, _, _ = _model(_csp(None, dem=True))
    on, _, _, _ = _model(_csp(AGNConfig(**ALL_ON), dem=True))
    base = off.predict_jit(T._theta(off))
    got = on.predict_jit({**T._theta(on), **_vals(ZERO)})
    assert _bytes(got) == _bytes(base)


# ------------------------------------------------------------------ V-SCAF-12
def test_V_SCAF_12_linear_in_line_amplitudes():
    model, _, _, _ = _model(_csp(AGNConfig(**ALL_ON)))
    p = {a: {k: np.asarray(v, np.float64) for k, v in model.predict(
        {**T._theta(model), **_vals(dict(ON, fagn=0.0, agn_elum=a * 3e-3, blr_lum=a * 1e-2))}
    ).items()} for a in (0.0, 1.0, 2.0)}
    for k in p[0.0]:
        d1, d2 = p[1.0][k] - p[0.0][k], p[2.0][k] - p[0.0][k]
        assert np.any(d1 != 0), k
        np.testing.assert_allclose(d2, 2 * d1, rtol=1e-4,
                                   atol=1e-4 * np.abs(d2).max() + 1e-6 * np.abs(p[0.0][k]).max())


# ------------------------------------------------------------------ V-SCAF-13
def _lnlike_factory(model, vals):
    th0 = {**T._theta(model), **_vals(vals)}
    keys = sorted(th0)
    sizes = [int(np.size(th0[k])) for k in keys]
    base = model.predict(th0)
    data = {k: np.asarray(v, np.float64) * 1.05 for k, v in base.items()}
    err = {k: 0.05 * np.abs(v).max() + 0.01 * np.abs(v) for k, v in data.items()}

    def unpack(x):
        th, i = {}, 0
        for k, n in zip(keys, sizes):
            th[k] = x[i:i + n].reshape(np.shape(th0[k]))
            i += n
        return th

    def lnlike(x):
        out = model.predict(unpack(x))
        return -0.5 * sum(jnp.sum(((out[k] - data[k]) / err[k]) ** 2) for k in data)

    x0 = np.concatenate([np.ravel(np.asarray(th0[k], np.float64)) for k in keys])
    pos, i = {}, 0
    for k, n in zip(keys, sizes):
        pos[k] = i
        i += n
    return lnlike, x0, pos


EDGES = {"skirtor_t": (3.0, 11.0), "skirtor_pl": (0.0, 1.5), "skirtor_q": (0.0, 1.5),
         "skirtor_oa": (10.0, 80.0), "skirtor_i": (0.0, 90.0), "polar_ebv": (0.0, 1.0),
         "polar_temp": (50.0, 500.0), "polar_emis": (1.0, 2.5), "nlr_logu": (-4.0, -1.0),
         "blr_sigma": (400.0, 5000.0)}


@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_SCAF_13_jit_vmap_grad(mode):
    model, _, _, _ = _model(_csp(AGNConfig(**dict(ALL_ON, agn_mode=mode)), igm=True))
    f, x0, pos = _lnlike_factory(model, ON)
    vg = jax.jit(jax.value_and_grad(f))
    v, g = vg(jnp.asarray(x0))
    g = np.asarray(g)
    assert np.isfinite(float(v)) and np.all(np.isfinite(g))
    for k in ON:
        assert g[pos[k]] != 0.0, k
    for k, (lo, hi) in EDGES.items():
        for x in (lo, hi):
            xe = x0.copy()
            xe[pos[k]] = x
            ve, ge = vg(jnp.asarray(xe))
            assert np.isfinite(float(ve)) and np.all(np.isfinite(np.asarray(ge))), (k, x)
    for k in ("agn_elum", "blr_lum"):         # lnL exactly quadratic in a line amplitude
        h = 0.25 * x0[pos[k]]
        e = np.zeros_like(x0)
        e[pos[k]] = h
        fd = (float(f(jnp.asarray(x0 + e))) - float(f(jnp.asarray(x0 - e)))) / (2 * h)
        assert g[pos[k]] == pytest.approx(fd, rel=2e-3), (k, g[pos[k]], fd)
    rng = np.random.default_rng(1)
    for W in (64, 100):
        X = np.repeat(x0[None], W, axis=0)
        X[:, pos["fagn"]] = 10 ** rng.uniform(-4, np.log10(2), W)
        for k, (lo, hi) in EDGES.items():
            X[:, pos[k]] = rng.uniform(lo, hi, W)
        X[:, pos["agn_elum"]] = rng.uniform(1e-6, 1e-2, W)
        X[:, pos["blr_lum"]] = rng.uniform(1e-6, 3e-2, W)
        bv, bg = jax.jit(jax.vmap(jax.value_and_grad(f)))(jnp.asarray(X))
        assert np.all(np.isfinite(np.asarray(bv))) and np.all(np.isfinite(np.asarray(bg)))
        for i in (0, W // 2, W - 1):
            lv, lg = vg(jnp.asarray(X[i]))
            assert float(bv[i]) == pytest.approx(float(lv), rel=1e-6)
            np.testing.assert_allclose(np.asarray(bg[i]), np.asarray(lg), rtol=1e-4,
                                       atol=1e-5 * np.abs(np.asarray(lg)).max())


# ------------------------------------------------------------------ V-SCAF-14
class _ToySkirtorLdust(_toys.ToySkirtor):
    needs_ldust_gal = True


def test_V_SCAF_14_ldust_gal(monkeypatch):
    path = _need()
    cosmo = Cosmology.planck18()
    kw = dict(lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
              zh_const=True, sfh_interp="step", add_dust=True, add_diffuse_dust=True,
              add_neb=True, add_igm=False, verbose=False, cosmo=cosmo,
              sps_home=os.environ["SPS_HOME"])
    ssp = SSPData.load(str(path))
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "skirtor16", _ToySkirtorLdust)
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "polar", _toys.ToyPolar)
    cfg = AGNConfig(torus_model="skirtor16", lines=False, agn_energy_balance=False)
    dem = CSPBasis(ssp, add_dust_emission=True, agn=cfg, **kw)
    nodem = CSPBasis(ssp, add_dust_emission=False, **kw)
    assert dem._agn.need_ldust_gal
    model, _, _, _ = _model(dem)
    for d2 in (0.3, 1.0):
        _, mt = _mt(model, ZERO, diffuse_tau_kc=jnp.array([d2]), tau_pow=jnp.array([0.5]))
        W = dem.calculate_ssp_weights(theta=mt)
        ctx = dem._agn_context(mt, W)
        w = np.asarray(dem._agn._w)
        re_emitted = (np.asarray(dem.get_spectrum(mt), np.float64)
                      - np.asarray(nodem.get_spectrum(mt), np.float64)) @ w
        assert float(ctx.L_dust_gal) > 0
        assert float(ctx.L_dust_gal) == pytest.approx(re_emitted, rel=2e-3)
    # PLAN_AGN2 2.2 (v2): allowed without dust emission (L_dust_gal is the attenuation-absorbed
    # luminosity), refused without any galaxy dust screen
    nodem_agn = CSPBasis(ssp, add_dust_emission=False, agn=cfg, **kw)
    _, mt2 = _mt(_model(nodem_agn)[0], ZERO, diffuse_tau_kc=jnp.array([0.3]),
                 tau_pow=jnp.array([0.5]))
    W2 = nodem_agn.calculate_ssp_weights(theta=mt2)
    assert float(nodem_agn._agn_context(mt2, W2).L_dust_gal) == pytest.approx(
        float(dem._agn_context(mt2, W2).L_dust_gal), rel=1e-12)
    with pytest.raises(ValueError, match="needs a galaxy dust screen"):
        CSPBasis(ssp, add_dust_emission=False, agn=cfg,
                 **dict(kw, add_dust=False, add_diffuse_dust=False))
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "skirtor16", _toys.ToySkirtor)
    off = CSPBasis(ssp, add_dust_emission=True, agn=AGNConfig(torus_model="skirtor16",
                                                              lines=False), **kw)
    assert off._agn_context(mt, W).L_dust_gal is None


# ------------------------------------------------------------------ V-SCAF-15
def test_V_SCAF_15_compat_refusal_provenance():
    csp = _csp(AGNConfig(torus=False, disk=False, lines=False, blr=True))
    model, _, _, _ = _model(csp)
    _, mt = _mt(model, {"blr_lum": 1e-2, "blr_sigma": 2000.0})
    assert csp._agn_line_terms(mt) is None
    terms = csp._agn_line_set_terms(mt)
    assert [t["key"] for t in terms] == ["blr"]
    from ceridwen.likelihood.eline_marginal import refuse_with_agn_lines
    with pytest.raises(ValueError, match="AGN narrow lines"):
        refuse_with_agn_lines(csp, T._obs()[0])
    full = _csp(AGNConfig(**ALL_ON))
    assert [t["key"] for t in full._agn_line_set_terms(_mt(_model(full)[0], ON)[1])] == [
        "nlr", "blr"]
    prov = full._agn.provenance()
    assert prov["agn_torus_model"] == "skirtor16" and prov["agn_blr"] is True
    assert prov["agn_polar_law"] == agnmod.POLAR_LAW_DEFAULT
    assert prov["agn_nlr_model"] == "feltre16"
    assert prov["agn_skirtor_file"] == "toy"
    assert prov["agn_components"] == "skirtor,lines,polar,blr"
    # the AGN2 keys are registered, so an unknown-key warning never fires for them
    for k in ON:
        assert k in csp._known_theta_keys or k not in csp._agn.param_names


# ------------------------------------------------------------------ V-SCAF-16
class _ToySkirtorOwnNuclear(_toys.ToySkirtor):
    """Composes the polar dust itself: (disk + dust) / 2 + a flat offset (a composition the
    hook's own one can never produce)."""

    def nuclear_parts(self, theta, ctx, polar):
        p = self.rest_parts(theta, ctx)
        t = polar.transmission(theta)
        nuc = 0.5 * (p["disk"] + p["dust"]) + 1e-3 * ctx.L_bol_star * self.disk_shape.max()
        return {"disk": p["disk"], "dust": p["dust"], "polar_t": t,
                "polar_ir": jnp.zeros_like(t), "nuclear": nuc}


def test_V_SCAF_16_own_nuclear_parts(monkeypatch):
    path = _need()
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "skirtor16", _ToySkirtorOwnNuclear)
    monkeypatch.setitem(agnmod.AGN2_REGISTRY, "polar", _toys.ToyPolar)
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(path)),
                   lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=False, add_dust_emission=False, verbose=False,
                   cosmo=cosmo, sps_home=os.environ["SPS_HOME"],
                   agn=AGNConfig(torus_model="skirtor16", polar_dust=True, lines=False))
    model, _, _, _ = _model(csp)
    _, mt = _mt(model, ON, diffuse_tau_kc=jnp.array([0.4]))
    W = csp.calculate_ssp_weights(theta=mt)
    ctx = csp._agn_context(mt, W)
    th = csp._agn.filled(mt)
    want = csp._agn.model.components["skirtor"].nuclear_parts(th, ctx, csp._agn.model.polar)
    dc = _diffuse(csp, mt)
    got = np.asarray(csp._agn_rest(mt))
    exp = np.asarray(want["nuclear"]) * dc
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14 * np.abs(exp).max())
    absorbed = float(csp._agn_parts(mt)["L_absorbed"])
    assert absorbed == pytest.approx(
        float((np.asarray(want["nuclear"]) * (1 - dc)) @ _toys._w(csp.wave)), rel=1e-12)
