"""Every AGN component in the CSP at once (session INT): the Temple+21 disc (D), the dust4
screen (S) and the narrow lines (L) next to the torus, through ``CSPBasis.predict`` /
``SedModel`` (``csp.py``: ``_agn_line_terms``, ``_agn_spectrum_lines``,
``_agn_photometry_lines``; ``hook.py``: ``line_dust_transmission``, line energy balance).

AGN_PLAN §4 at CSP level with D, S, L on:

B3  fagn = fagn_bbb = agn_elum = 0 (dust4 > 0, every other AGN key off its default):
    every prediction (Photometry, Spectrum, Lines) byte-identical to the CSP without agn=,
    in all three modes, with and without C1 ``wavelength_support=True``, with and without IGM.
L   the AGN lines the CSP adds equal an independent computation from the component API
    (``AGNLines.observed_fluxes``, the projector's painter, ``line_photometry_delta`` /
    ``agn_line_profiles_on_grid``), which session LINES checked against Prospector (A4).
    "corrected": the galaxy's diffuse dust and the IGM at each line act on the AGN lines;
    "prospector"/"fsps": nothing does (Prospector's AGNSpecModel).
B4  linear in fagn_bbb and agn_elum at the prediction level.
B6  energy balance with the lines: absorbed continuum + absorbed line light re-emitted by
    DL07, luminosity conserved to DL07's own loss.
B8  jit, vmap (W = 64, 100) == loop, finite gradients of a log-likelihood in every AGN key
    (fagn, agn_tau, fagn_bbb, dust4, dust4_index, agn_elum, agn_eline_sigma), central
    differences for the linear amplitudes.
    Setup refusal: AGN lines in the reproduction modes need add_neb=True.
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
import test_emission_lines as T  # noqa: E402

from ceridwen import CSPBasis, Cosmology, SSPData  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.agn.lines import line_photometry_delta  # noqa: E402
from ceridwen.broadening import agn_line_profiles_on_grid  # noqa: E402

AGN_KEYS = ("fagn", "agn_tau", "fagn_bbb", "dust4", "dust4_index", "agn_elum",
            "agn_eline_sigma")
ON = {"fagn": 0.2, "agn_tau": 23.0, "fagn_bbb": 0.7, "dust4": 0.4, "dust4_index": -1.3,
      "agn_elum": 2e-3, "agn_eline_sigma": 180.0}
ZERO = dict(ON, fagn=0.0, fagn_bbb=0.0, agn_elum=0.0)


def _cfg(mode):
    if mode == "fsps":                         # FSPS has no disc / dust4; lines on explicitly
        return AGNConfig(agn_mode="fsps", lines=True)
    return AGNConfig(agn_mode=mode, dust4_type=1)


def _agn_vals(vals, mode):
    v = dict(vals)
    if mode == "fsps":
        for k in ("fagn_bbb", "dust4", "dust4_index"):
            v.pop(k)
    return {k: jnp.array([x]) for k, x in v.items()}


_CACHE = {}


def _csp(agn=None, *, igm=False, support=False, dem=False):
    key = (None if agn is None else repr(agn), igm, support, dem)
    if key in _CACHE:
        return _CACHE[key]
    path = T.find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (see tests/_gridfixture.py)")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY nebular grids, FSPS line list, Nenkova templates)")
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(path)),
                   lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
                   zh_const=True, sfh_interp="step", add_dust=dem, add_diffuse_dust=True,
                   add_neb=True, add_igm=igm, add_dust_emission=dem, verbose=False,
                   cosmo=cosmo, sps_home=os.environ["SPS_HOME"], wavelength_support=support,
                   agn=agn)
    _CACHE[key] = csp
    return csp


def _model(csp, phot_only=False):
    spec, phot, lines = T._obs()
    obs = (phot,) if phot_only else (spec, phot, lines)
    return T._model(csp, obs), spec, phot, lines


def _bytes(d):
    return {k: (np.asarray(v).dtype, np.asarray(v).tobytes()) for k, v in d.items()}


# ------------------------------------------------------------------ B3
@pytest.mark.parametrize("mode", ["fsps", "prospector", "corrected"])
@pytest.mark.parametrize("igm", [False, True])
def test_B3_all_components_zero_is_byte_identical(mode, igm):
    off, _, _, _ = _model(_csp(None, igm=igm))
    on, _, _, _ = _model(_csp(_cfg(mode), igm=igm))
    th = T._theta(off)
    base = off.predict_jit(th)
    got = on.predict_jit({**T._theta(on), **_agn_vals(ZERO, mode)})
    assert set(got) == set(base)
    assert _bytes(got) == _bytes(base)
    # and it is on: every observation moves
    lit = on.predict_jit({**T._theta(on), **_agn_vals(ON, mode)})
    for k in base:
        assert np.any(np.asarray(lit[k]) != np.asarray(base[k])), k


@pytest.mark.parametrize("mode", ["prospector", "corrected"])
def test_B3_wavelength_support(mode):
    off, _, _, _ = _model(_csp(None, support=True), phot_only=True)
    on, _, _, _ = _model(_csp(_cfg(mode), support=True), phot_only=True)
    csp = on.csp
    assert csp._support_for(on.observations, T._csp_theta(on, T._theta(on))) is not None
    base = off.predict_jit(T._theta(off))
    got = on.predict_jit({**T._theta(on), **_agn_vals(ZERO, mode)})
    assert _bytes(got) == _bytes(base)


def test_lines_refused_without_neb_in_reproduction_modes(ssp_data, torus_path):
    with pytest.raises(ValueError, match="needs add_neb=True"):
        CSPBasis(ssp_data, lookback_time=jnp.linspace(0.0, 0.5, 5), zh_const=True,
                 add_neb=False, verbose=False, cosmo=Cosmology.planck18(),
                 agn=AGNConfig(agn_mode="prospector"))
    # corrected mode paints its lines without the nebular model; lines=False is accepted
    CSPBasis(ssp_data, lookback_time=jnp.linspace(0.0, 0.5, 5), zh_const=True, add_neb=False,
             verbose=False, cosmo=Cosmology.planck18(), agn=AGNConfig(agn_mode="corrected"))
    CSPBasis(ssp_data, lookback_time=jnp.linspace(0.0, 0.5, 5), zh_const=True, add_neb=False,
             verbose=False, cosmo=Cosmology.planck18(),
             agn=AGNConfig(agn_mode="prospector", lines=False))


def test_B9_refused_on_the_real_csp():
    """B9 with the AGN built by CSPBasis(agn=...) (the hook, not a bare AGNModel attribute)."""
    from ceridwen import SedModel
    from ceridwen.observation import Spectrum
    from ceridwen.broadening import Instrument, Kinematics
    for mode, refused in (("corrected", True), ("prospector", True)):
        csp = _csp(_cfg(mode))
        spec = Spectrum(wavelength=T.WAVE_OBS, flux=np.ones_like(T.WAVE_OBS),
                        uncertainty=np.ones_like(T.WAVE_OBS),
                        instrument=Instrument.R_fwhm(T.R_FWHM), name="spec",
                        marginalize_elines=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="marginalize_elines=True with AGN narrow lines"):
                SedModel(csp, [spec], zred=T.ZRED,
                         free_param_init={"logmass": jnp.array([9.0])},
                         kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS))
    # lines off: accepted
    csp = _csp(AGNConfig(agn_mode="corrected", lines=False))
    spec = Spectrum(wavelength=T.WAVE_OBS, flux=np.ones_like(T.WAVE_OBS),
                    uncertainty=np.ones_like(T.WAVE_OBS),
                    instrument=Instrument.R_fwhm(T.R_FWHM), name="spec", marginalize_elines=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SedModel(csp, [spec], zred=T.ZRED, free_param_init={"logmass": jnp.array([9.0])},
                 transforms={"gas_logu": lambda th: jnp.array([-2.5]),
                             "gas_logz": lambda th: jnp.array([0.0])},
                 kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS))


# ------------------------------------------------------------------ L: the wiring
@pytest.mark.parametrize("mode", ["fsps", "prospector", "corrected"])
@pytest.mark.parametrize("igm", [False, True])
def test_lines_match_independent_computation(mode, igm):
    """predict(agn_elum = x) - predict(agn_elum = 0), every other AGN key equal, against the
    AGN lines built directly from the component API."""
    on, spec, phot, lines = _model(_csp(_cfg(mode), igm=igm))
    csp = on.csp
    agn = csp._agn.model.lines
    th = {**T._theta(on), **_agn_vals(ON, mode)}
    th0 = {**th, "agn_elum": jnp.array([0.0])}
    p1, p0 = on.predict(th), on.predict(th0)
    mt = T._csp_theta(on, th)
    opz = 1.0 + T.ZRED
    ff = float(csp._flux_factor(mt))
    mf = float(jnp.sum(csp.calculate_ssp_weights(theta=mt)))              # formed mass / 10**logmass
    assert mf == pytest.approx(1.0, rel=1e-12)                            # unit-mass SFH here
    F = np.asarray(agn.observed_fluxes(mt, ff / opz)) * mf                # erg/s/cm^2
    sig = float(ON["agn_eline_sigma"])
    if mode == "corrected":
        tau = np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64)
        t_dust = np.interp(agn.line_wave, np.asarray(csp.wave), np.exp(-tau))
        t_igm = (np.asarray(csp._igm_transmission(T.ZRED, mt), np.float64) if igm
                 else np.ones(csp.wave.shape))
        t_l = t_dust * np.interp(agn.line_wave, np.asarray(csp.wave), t_igm)
        F_l = F * t_l
        prof = np.asarray(agn_line_profiles_on_grid(csp.wave, agn.line_wave, sig,
                                                    csp.neb.res_floor_factor))
        mass = 10.0 ** float(np.ravel(mt["logmass"])[0])
        grid = prof @ (np.asarray(agn.line_luminosity(mt)) * mf * mass * t_dust) * ff * t_igm
        d_phot = np.asarray(phot._T, np.float64) @ grid
    else:
        F_l = F
        d_phot = np.asarray(line_photometry_delta(phot.filters, agn.line_wave * opz,
                                                  jnp.asarray(F)))
    d_spec = np.asarray(spec._proj.agn_lines(jnp.asarray(F_l), sig, mt))
    scale = float(np.ravel(mt["eline_scaling"])[0]) if "eline_scaling" in mt else 1.0
    d_lines = lines.agn_line_matrix(agn) @ (F_l * scale)
    for name, want in (("spec", d_spec), ("phot", d_phot), ("lines", d_lines)):
        got = np.asarray(p1[name], np.float64) - np.asarray(p0[name], np.float64)
        tol = 2e-6 * np.abs(np.asarray(p1[name], np.float64)).max()      # float32 predictions
        np.testing.assert_allclose(got, want, rtol=0, atol=tol, err_msg=f"{mode} {name}")
        assert np.abs(want).max() > 50 * tol, (mode, name)               # the test has teeth


def test_lines_follow_the_sfh_formed_mass():
    """A directly given SFH carries its own normalisation (AGENTS.md item 3): doubling it
    doubles the stellar light, the torus (via L_bol) and the AGN lines alike."""
    csp = _csp(_cfg("corrected"))
    th = dict(csp.theta_init)
    th.update(_agn_vals(ON, "corrected"))
    th["zred"] = jnp.array([T.ZRED])
    t1 = csp._agn_line_terms(th)["F"]
    t2 = csp._agn_line_terms({**th, "sfh": 2.0 * jnp.asarray(th["sfh"])})["F"]
    np.testing.assert_allclose(np.asarray(t2), 2.0 * np.asarray(t1), rtol=1e-12)
    a1 = csp._agn_rest(th)
    a2 = csp._agn_rest({**th, "sfh": 2.0 * jnp.asarray(th["sfh"])})
    np.testing.assert_allclose(np.asarray(a2), 2.0 * np.asarray(a1), rtol=1e-6)


def test_corrected_lines_see_dust_reproduction_lines_do_not():
    """Raising the galaxy's diffuse dust dims the corrected-mode AGN lines by exactly the
    curve at each line and leaves the prospector-mode AGN lines untouched."""
    out = {}
    for mode in ("prospector", "corrected"):
        on, _, _, lines = _model(_csp(_cfg(mode)))
        th = {**T._theta(on), **_agn_vals(ON, mode)}
        for tau in (0.1, 0.8):
            t = {**th, "diffuse_tau_kc": jnp.array([tau])} if "diffuse_tau_kc" in th else None
            if t is None:
                pytest.skip("diffuse law without diffuse_tau_kc")
            terms = on.csp._agn_line_terms(T._csp_theta(on, t))
            out[mode, tau] = np.asarray(terms["F"])
    assert np.array_equal(out["prospector", 0.1], out["prospector", 0.8])
    assert np.all(out["corrected", 0.8] < out["corrected", 0.1])


# ------------------------------------------------------------------ B4
@pytest.mark.parametrize("mode", ["prospector", "corrected"])
@pytest.mark.parametrize("key", ["fagn_bbb", "agn_elum"])
def test_B4_linear_at_prediction_level(mode, key):
    on, _, _, _ = _model(_csp(_cfg(mode), igm=True))
    th = {**T._theta(on), **_agn_vals(ZERO, mode)}
    p = [on.predict({**th, key: jnp.array([x])}) for x in (0.0, ON[key], 2 * ON[key])]
    for name in p[0]:
        a0, a1, a2 = (np.asarray(q[name], np.float64) for q in p)
        tol = 2e-6 * np.abs(a2).max()          # float32 predictions (T3's rtol 1e-6)
        np.testing.assert_allclose(a2 - a0, 2 * (a1 - a0), rtol=0, atol=tol, err_msg=name)


# ------------------------------------------------------------------ B6 with lines
def test_B6_energy_balance_includes_lines():
    csp = _csp(_cfg("corrected"), dem=True)
    hook = csp._agn
    assert hook.energy_balance
    th = dict(csp.theta_init)
    th.update(_agn_vals(ON, "corrected"))
    th["diffuse_tau_kc"] = jnp.array([0.9])
    W = csp.calculate_ssp_weights(theta=th)
    ctx = csp._agn_context(th, W)
    intrinsic, attenuated = hook.components(th, ctx)
    # per M_formed times the SFH's formed mass (theta_init's sfh is not unit-mass)
    L = np.asarray(hook.model.lines.line_luminosity(hook.filled(th))) * float(jnp.sum(W))
    t_l = np.asarray(hook.line_dust_transmission(ctx.diffuse_curve))
    assert np.all(t_l < 1.0)
    w = np.asarray(hook._w)
    agn = np.asarray(csp._agn_rest(th))
    lhs = agn @ w + np.sum(L * t_l)                  # emitted: attenuated + DL07 + lines out
    rhs = np.asarray(intrinsic) @ w + np.sum(L)      # intrinsic
    absorbed_lines = np.sum(L * (1 - t_l))
    assert absorbed_lines > 1e-3 * rhs              # the line term matters at this tolerance
    # DL07's own self-absorption loss on the stellar path is 4.6e-7 of the absorbed light
    assert abs(lhs / rhs - 1.0) < 1e-6
    _, absorbed = hook(th, ctx, csp.dust_emi,
                       (th["duste_qpah"], th["duste_umin"], th["duste_gamma"]),
                       formed_mass=jnp.sum(W))
    np.testing.assert_allclose(float(absorbed),
                               float(np.asarray(intrinsic - attenuated) @ w) + absorbed_lines,
                               rtol=1e-12)


# ------------------------------------------------------------------ B8
def _lnlike_factory(model, mode):
    th0 = {**T._theta(model), **_agn_vals(ON, mode)}
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
    pos = {}
    i = 0
    for k, n in zip(keys, sizes):
        pos[k] = i
        i += n
    return lnlike, x0, pos


@pytest.mark.parametrize("mode", ["prospector", "corrected"])
def test_B8_full_model_jit_vmap_grad(mode):
    model, _, _, _ = _model(_csp(_cfg(mode), igm=True))
    f, x0, pos = _lnlike_factory(model, mode)
    vg = jax.jit(jax.value_and_grad(f))
    v, g = vg(jnp.asarray(x0))
    g = np.asarray(g)
    assert np.isfinite(float(v)) and np.all(np.isfinite(g))
    for k in AGN_KEYS:
        assert g[pos[k]] != 0.0, k                        # every AGN key reaches the likelihood
    # lnL is exactly quadratic in a linear amplitude, so a large central step is exact and
    # stays clear of the float32 noise of the predictions
    for k in ("fagn", "fagn_bbb", "agn_elum"):
        h = 0.25 * x0[pos[k]]
        e = np.zeros_like(x0)
        e[pos[k]] = h
        fd = (float(f(jnp.asarray(x0 + e))) - float(f(jnp.asarray(x0 - e)))) / (2 * h)
        assert g[pos[k]] == pytest.approx(fd, rel=2e-3), (k, g[pos[k]], fd)
    rng = np.random.default_rng(0)
    for W in (64, 100):
        X = np.repeat(x0[None], W, axis=0)
        X[:, pos["fagn"]] = 10 ** rng.uniform(-5, np.log10(3), W)
        X[:, pos["agn_tau"]] = 10 ** rng.uniform(np.log10(5), np.log10(150), W)
        X[:, pos["fagn_bbb"]] = 10 ** rng.uniform(-5, 2, W)
        X[:, pos["dust4"]] = rng.uniform(0.0, 2.0, W)
        X[:, pos["dust4_index"]] = rng.uniform(-1.8, -0.8, W)
        X[:, pos["agn_elum"]] = rng.uniform(1e-6, 1e-2, W)
        X[:, pos["agn_eline_sigma"]] = rng.uniform(50.0, 500.0, W)
        bv, bg = jax.jit(jax.vmap(jax.value_and_grad(f)))(jnp.asarray(X))
        assert np.all(np.isfinite(np.asarray(bv))) and np.all(np.isfinite(np.asarray(bg)))
        for i in (0, W // 2, W - 1):
            lv, lg = vg(jnp.asarray(X[i]))
            # float32 einsums reorder under vmap (T3 holds them at rtol 1e-6)
            assert float(bv[i]) == pytest.approx(float(lv), rel=1e-6)
            np.testing.assert_allclose(np.asarray(bg[i]), np.asarray(lg), rtol=1e-4,
                                       atol=1e-5 * np.abs(np.asarray(lg)).max())


def test_B8_wavelength_support_grad():
    model, _, _, _ = _model(_csp(_cfg("corrected"), support=True), phot_only=True)
    f, x0, pos = _lnlike_factory(model, "corrected")
    g = np.asarray(jax.jit(jax.grad(f))(jnp.asarray(x0)))
    assert np.all(np.isfinite(g))
    assert all(g[pos[k]] != 0.0 for k in AGN_KEYS)


# ------------------------------------------------------------------ a/Fe CSP, all components
@pytest.fixture(scope="module")
def afe_ssp(ssp_data):
    """The two-plane [alpha/Fe] grid of test_hook.py (BPASS test grid x1 and x0.9)."""
    from types import SimpleNamespace
    s = SimpleNamespace(**{k: getattr(ssp_data, k) for k in vars(ssp_data)})
    f = np.asarray(ssp_data.ssp_flux, dtype=np.float64)
    s.ssp_flux = np.stack([f, 0.9 * f])
    s.ssp_afe = np.array([0.0, 0.4])
    if s.ssp_stellar_mass is not None:
        m = np.asarray(s.ssp_stellar_mass)
        s.ssp_stellar_mass = np.stack([m, m])
    return s


def test_afe_all_components(afe_ssp, torus_path):
    """CSPBasis_afe (no nebular model) in corrected mode with disc, dust4 and lines: B3 byte
    identity at zero amplitudes and finite gradients in every AGN key at a sampled afe."""
    from ceridwen.csp import CSPBasis_afe
    from ceridwen import SedModel
    from ceridwen.observation import Photometry

    def build(agn):
        return CSPBasis_afe(afe_ssp, lookback_time=jnp.linspace(0.0, 0.5, 5), zh_const=True,
                            verbose=False, cosmo=Cosmology.planck18(), agn=agn, add_igm=True,
                            sps_home=os.environ.get("SPS_HOME"))

    filters = ["jwst_f090w", "jwst_f150w", "jwst_f444w", "jwst_f770w"]

    def model(csp):
        obs = [Photometry(filters=filters, flux=[2e-9] * 4, uncertainty=[2e-10] * 4,
                          name="phot")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SedModel(csp, obs, priors={}, free_param_init={"logmass": jnp.array([10.0])},
                            zred=2.0)

    off, on = model(build(None)), model(build(_cfg("corrected")))
    afe = {"afe": jnp.array([0.13])}
    base = off.predict_jit({**off.theta_init, **afe})
    got = on.predict_jit({**on.theta_init, **afe, **_agn_vals(ZERO, "corrected")})
    assert _bytes(got) == _bytes(base)

    th0 = {**on.theta_init, **afe, **_agn_vals(ON, "corrected")}

    def lnl(t):
        return -0.5 * jnp.sum(((on.predict(t)["phot"] - 2e-9) / 2e-10) ** 2)

    g = jax.jit(jax.grad(lnl))(th0)
    for k in AGN_KEYS + ("afe",):
        v = float(np.ravel(np.asarray(g[k]))[0])
        assert np.isfinite(v) and v != 0.0, k


# ------------------------------------------------------------------ provenance (fit.py)
def test_result_file_records_the_agn():
    """``/model@csp_config`` carries ``agn`` (mode, components, energy balance, template and
    table hashes) only when the CSP has an AGN; a model with another AGN mode differs."""
    import hashlib
    import json
    from ceridwen.resultfile import _model_record, _cfg as _json_cfg
    from ceridwen.agn.disk import TEMPLE21_FILE
    from ceridwen.agn.lines import A42_FILE
    off, _, _, _ = _model(_csp(None), phot_only=True)
    assert "agn" not in _json_cfg(_model_record(off)["/model@csp_config"])
    on, _, _, _ = _model(_csp(_cfg("corrected")), phot_only=True)
    cfg = _json_cfg(_model_record(on)["/model@csp_config"])
    a = cfg["agn"]
    assert cfg["spectrum_model"].endswith("_agn")
    assert a["agn_mode"] == "corrected" and a["dust4_type"] == 1
    assert a["agn_components"] == "disk,torus,lines"
    assert a["agn_energy_balance"] is False                    # no dust emission here
    sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert a["agn_disk_sha256"] == sha(TEMPLE21_FILE)
    assert a["agn_lines_table_sha256"] == sha(A42_FILE)
    assert a["agn_torus_sha256"] == sha(a["agn_torus_file"])
    json.dumps(a)                                               # serialisable as written
    other, _, _, _ = _model(_csp(_cfg("prospector")), phot_only=True)
    assert _json_cfg(_model_record(other)["/model@csp_config"])["agn"]["agn_mode"] == "prospector"


# ------------------------------------------------------------------ PostProcess / plotting
def _fake_result(model, n=6, seed=1):
    from ceridwen.sampler.runner import SamplingResult
    rng = np.random.default_rng(seed)
    samples = {}
    for k in model.param_names:
        v = np.asarray(model.theta_init[k], np.float64)
        d = v[None] * (1.0 + 0.02 * rng.standard_normal((n,) + v.shape))
        samples[k] = jnp.asarray(d[:, 0] if v.shape == (1,) else d)   # as tests/test_postprocess.py
    return SamplingResult(samples=samples, log_evidence=float("nan"),
                          log_evidence_err=float("nan"), log_weights=jnp.zeros(n),
                          log_likelihoods=jnp.zeros(n), param_names=list(model.param_names),
                          n_likelihood_calls=n, wall_time_s=0.0, sampler_name="fake")


def test_postprocess_agn_curve_and_mdust(tmp_path):
    """PostProcess: ``spectra_agn`` is the AGN the model adds (spectra_model - the same model
    at zero AGN), ``extras['agn']`` holds L_bol_star, L_absorbed and (energy balance) mdust =
    L_absorbed x DL07's dust mass per absorbed L_sun; the summary figure draws the AGN."""
    from ceridwen import SedModel
    from ceridwen.postprocess import PostProcess
    from ceridwen.observation import Photometry
    csp = _csp(_cfg("corrected"), dem=True)
    obs = [Photometry(filters=T.FILTERS, flux=np.ones(4) * 1e-9, uncertainty=np.ones(4) * 1e-10,
                      name="phot")]
    t = np.array(csp.sfh_times)
    from ceridwen.model.transforms import logsfr_ratios_to_sfh
    init = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.0]),
            **_agn_vals(ON, "corrected")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, obs, free_param_init=init, zred=T.ZRED,
                         transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(
                             th["logsfr_ratios"], sfh_times_yr=_t)})
    res = _fake_result(model)
    pp = PostProcess(model, res, n_samples=None)
    out = pp.run()                      # uniform weights: draw i is sample i
    pred = out["prediction"]
    agn = np.asarray(pred["spectra_agn"], np.float64)
    assert agn.shape == np.asarray(pred["spectra_model"]).shape and np.all(np.isfinite(agn))
    ex = out["extras"]["agn"]
    assert set(ex) == {"L_bol_star", "L_absorbed", "mdust"}
    assert np.all(ex["L_absorbed"] > 0) and np.all(ex["mdust"] > 0)
    # one draw by hand
    th0 = {k: jnp.asarray(np.asarray(res.samples[k])[0]).reshape(np.shape(model.theta_init[k]))
           for k in model.param_names}
    mt = model.apply_transforms(th0)
    mt = {**mt, "zred": jnp.array([T.ZRED])}
    full = np.asarray(csp.get_spectrum(mt, include_lines=True), np.float64)
    zero = np.asarray(csp.get_spectrum({**mt, **_agn_vals(ZERO, "corrected")},
                                       include_lines=True), np.float64)
    scale = 10.0 ** float(np.ravel(mt["logmass"])[0])
    np.testing.assert_allclose(agn[0], (full - zero) * scale, rtol=2e-6,
                               atol=2e-6 * np.abs(full * scale).max())
    W = csp.calculate_ssp_weights(theta=mt)
    ctx = csp._agn_context(mt, W)
    m1 = float(csp._agn.dust_response(ctx, csp.dust_emi, (mt["duste_qpah"], mt["duste_umin"],
                                                           mt["duste_gamma"]))[0])
    np.testing.assert_allclose(ex["mdust"][0], ex["L_absorbed"][0] * m1, rtol=1e-10)
    pp.figures(str(tmp_path))
    assert any(f.name.startswith("summary") for f in tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["prospector", "corrected"])
def test_first_call_under_jit_and_vmap(mode):
    """The AGN-line painter is built at setup (SedModel.setup_observations), so a model whose
    FIRST prediction is traced (jit, vmap: PostProcess, the samplers) caches no tracer, and a
    direct CSPBasis.predict on an unprepared projector builds it outside the trace."""
    model, spec, _, _ = _model(_csp(_cfg(mode), igm=True))
    assert spec._proj._agn is not None                         # built at setup
    th = {**T._theta(model), **_agn_vals(ON, mode)}
    batch = {k: jnp.stack([v, v]) for k, v in th.items()}
    out_v = jax.jit(jax.vmap(model.predict))(batch)
    out_e = model.predict(th)
    for k in out_e:
        np.testing.assert_allclose(np.asarray(out_v[k][0]), np.asarray(out_e[k]), rtol=1e-6,
                                   atol=1e-6 * np.abs(np.asarray(out_e[k])).max())
    spec._proj._agn = None                                     # a projector never prepared
    out_j = jax.jit(model.predict)(th)                         # built inside this first trace
    out_j2 = jax.jit(lambda t: model.predict(t))(th)           # a second trace reuses it
    for k in out_e:
        assert np.all(np.isfinite(np.asarray(out_j[k]))) and np.all(np.isfinite(out_j2[k]))
