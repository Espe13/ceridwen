"""BLR (``blr.BroadLines``) through the whole CERIDWEN pipeline (session VAL, Phase B).
Skips loudly while ``ceridwen/agn/blr.py`` (or its data) is missing.

Requirements (AGN2_BRIEF.md "Config and theta names", "Physics conventions"; PROMPT_agn2_VAL.md
"BLR line flux integrates to blr_lum x ratio, BLR width as measured from the painted profile"):

V-BLR-1  blr_lum is L_Halpha,BLR per M_formed: line_luminosity at "Ba-alpha 6563" == blr_lum
         exactly; every other line = blr_lum x its ratio (ratio of the BLR table, > 0).
V-BLR-2  painted broad H alpha in a Spectrum: FWHM measured from the profile (half-maximum
         crossings, no moments) = 2.3548 sqrt(blr_sigma^2 + sigma_inst^2) within 1 %, over the
         prior support [400, 5000] km/s (spectrum wide enough for 5000 km/s).
V-BLR-3  its frequency integral = blr_lum x mass x flux factor / (1 + z) x T(H alpha) within
         0.5 % (pixel sampling); the Lines observation of H alpha moves by exactly that flux.
V-BLR-4  corrected: T_BLR = T_gal (no polar dust); cigale: T_BLR = 1; independent of the NLR's
         width and amplitude (agn_eline_sigma, agn_elum do not move the broad line).
V-BLR-5  grad finite in blr_lum, blr_sigma at 400, 2000, 5000 km/s through the full prediction.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ref  # noqa: E402
import test_emission_lines as T  # noqa: E402

import ceridwen.agn as agnmod  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData, SedModel  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.broadening import Instrument, Kinematics  # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh  # noqa: E402
from ceridwen.observation import Lines, Spectrum  # noqa: E402
from _f32bounds import B8   # noqa: E402,F401  (float32 bounds)

ZRED = 0.3
HA = 6564.7229
R_FWHM = 2000.0
# rest 5600-7600 A: +-7 sigma of a 5000 km/s broad H alpha on the spectrum
WAVE_OBS = np.exp(np.arange(np.log(5600 * (1 + ZRED)), np.log(7600 * (1 + ZRED)),
                            1.0 / (2.3548 * R_FWHM) / 2.5))


def _blr_or_skip(test):
    try:
        agnmod._agn2_class("blr")
    except NotImplementedError as e:
        _ref.missing(f"component blr ({e})", test)
    if T.find_test_grid() is None:
        _ref.missing("test SSP grid", test)
    if not os.environ.get("SPS_HOME"):
        _ref.missing("$SPS_HOME", test)


_CACHE = {}


def _model(mode, test):
    _blr_or_skip(test)
    if mode in _CACHE:
        return _CACHE[mode]
    cosmo = Cosmology.planck18()
    cfg = AGNConfig(agn_mode=mode, torus=False, disk=False, lines=False, blr=True) \
        if mode == "corrected" else AGNConfig(agn_mode="cigale", torus=False, blr=True)
    try:
        csp = CSPBasis(SSPData.load(str(T.find_test_grid())),
                       lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(ZRED))]),
                       zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                       add_neb=True, add_igm=False, add_dust_emission=False, verbose=False,
                       cosmo=cosmo, sps_home=os.environ["SPS_HOME"], agn=cfg)
    except (FileNotFoundError, OSError) as e:
        _ref.missing(f"BLR data ({e})", test)
    spec = Spectrum(wavelength=WAVE_OBS, flux=np.ones_like(WAVE_OBS),
                    uncertainty=np.ones_like(WAVE_OBS), instrument=Instrument.R_fwhm(R_FWHM),
                    name="spec")
    lines = Lines(line_ind=np.arange(1), wavelength=[HA], line_names=["Halpha"],
                  flux=np.ones(1), uncertainty=np.ones(1), name="lines")
    t = np.array(csp.sfh_times)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SedModel(csp, [spec, lines],
                         transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(
                             th["logsfr_ratios"], sfh_times_yr=_t)},
                         free_param_init={"logsfr_ratios": jnp.zeros(len(t) - 1),
                                          "logmass": jnp.array([9.0])},
                         zred=ZRED, kinematics=Kinematics(sigma_gal=100.0, sigma_gas=60.0))
    _CACHE[mode] = (model, csp, spec)
    return _CACHE[mode]


def _theta(model, **kw):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th["logzsol"] = jnp.array([-0.3])
    th.update({k: jnp.array([float(v)]) for k, v in kw.items()})
    return th


def _mt(model, th):
    mt = dict(model.apply_transforms(th))
    mt["zred"] = jnp.array([ZRED])
    return mt


def test_V_BLR_1_amplitude_is_halpha():
    model, csp, _ = _model("corrected", "V-BLR-1")
    blr = csp._agn.model.blr
    names = list(blr.line_names)
    assert "Ba-alpha 6563" in names
    th = csp._agn.filled({"blr_lum": jnp.asarray(0.37)})
    L = np.asarray(blr.line_luminosity(th))
    assert L[names.index("Ba-alpha 6563")] == 0.37
    assert np.all(L > 0)
    th2 = csp._agn.filled({"blr_lum": jnp.asarray(0.74)})
    np.testing.assert_array_equal(np.asarray(blr.line_luminosity(th2)), 2 * L)


def _broad(model, sig, lum=1.0, **kw):
    on = model.predict(_theta(model, blr_lum=lum, blr_sigma=sig, **kw))
    off = model.predict(_theta(model, blr_lum=0.0, blr_sigma=sig, **kw))
    return {k: np.asarray(on[k], np.float64) - np.asarray(off[k], np.float64) for k in on}


@pytest.mark.parametrize("sig", [400.0, 1500.0, 5000.0])
def test_V_BLR_2_3_width_and_flux(sig):
    model, csp, spec = _model("corrected", "V-BLR-2")
    d = _broad(model, sig, diffuse_tau_kc=0.3)
    wo = np.asarray(spec.wavelength, np.float64)
    c = HA * (1 + ZRED)
    s_inst = float(np.interp(c, wo, np.asarray(spec._proj.sigma_inst_kms, np.float64)))
    want = _ref.FWHM_PER_SIGMA * np.hypot(sig, s_inst)
    # the broad H beta and other lines are off this spectrum; other broad lines within +-7 sigma?
    blr = csp._agn.model.blr
    others = [w for n, w in zip(blr.line_names, blr.line_wave) if n != "Ba-alpha 6563"
              and abs(np.log(w / HA)) < 8 * np.hypot(sig, s_inst) / _ref.CKMS]
    if others:
        pytest.skip(f"another BLR line within 8 sigma of H alpha ({others}): FWHM not isolated")
    got = _ref.fwhm_kms(wo, d["spec"], c)
    assert got == pytest.approx(want, rel=0.01), (sig, got, want)
    # flux: integral over frequency vs the expected integrated flux
    mt = _mt(model, _theta(model, blr_lum=1.0, blr_sigma=sig, diffuse_tau_kc=0.3))
    msum = float(jnp.sum(csp.calculate_ssp_weights(theta=mt)))
    dc = np.exp(-np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64))
    t_ha = np.interp(blr.line_wave[list(blr.line_names).index("Ba-alpha 6563")],
                     np.asarray(csp.wave), dc)
    F = 1.0 * msum * 10 ** 9.0 * float(csp._flux_factor(mt)) / (1 + ZRED) * t_ha
    assert d["lines"][0] == pytest.approx(F, rel=2e-6)
    # Spectrum units: the projector returns the same units as its painter's f_nu input
    # (line flux x phi(ln lam) x lam / c), so the integral over nu returns F
    assert _ref.nu_integral(wo, d["spec"]) == pytest.approx(F, rel=5e-3)


@pytest.mark.parametrize("mode", ["corrected", "cigale"])
def test_V_BLR_4_transmission_and_independence(mode):
    model, csp, _ = _model(mode, f"V-BLR-4[{mode}]")
    a = _broad(model, 2000.0, diffuse_tau_kc=0.0)["lines"][0]
    b = _broad(model, 2000.0, diffuse_tau_kc=0.8)["lines"][0]
    mt = _mt(model, _theta(model, diffuse_tau_kc=0.8))
    dc = np.exp(-np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64))
    blr = csp._agn.model.blr
    t_ha = np.interp(blr.line_wave[list(blr.line_names).index("Ba-alpha 6563")],
                     np.asarray(csp.wave), dc)
    want = t_ha if mode == "corrected" else 1.0
    assert b / a == pytest.approx(want, rel=2e-6)
    if "agn_elum" in csp._agn.param_names:
        c = _broad(model, 2000.0, agn_elum=0.0, agn_eline_sigma=100.0)
        e = _broad(model, 2000.0, agn_elum=0.0, agn_eline_sigma=900.0)
        np.testing.assert_array_equal(c["spec"], e["spec"])


def test_V_BLR_5_grad():
    model, csp, _ = _model("corrected", "V-BLR-5")
    th0 = _theta(model, blr_lum=0.5, blr_sigma=2000.0)

    def f(lum, sig):
        out = model.predict({**th0, "blr_lum": lum, "blr_sigma": sig})
        return jnp.sum(out["spec"] ** 2) * 1e30 + jnp.sum(out["lines"]) * 1e15

    g = jax.jit(jax.grad(f, argnums=(0, 1)))
    for sig in (400.0, 2000.0, 5000.0):
        gl, gs = g(jnp.array([0.5]), jnp.array([sig]))
        assert np.all(np.isfinite(np.asarray(gl))) and np.all(np.isfinite(np.asarray(gs))), sig
        assert float(gl[0]) != 0.0 and float(gs[0]) != 0.0, sig


# ------------------------------------------------------------------ PLAN_AGN2 1.2 / 1.3
BLR_TABLE = (pathlib.Path.home() / "Desktop/PhD/Tools/ceridwen_workspace/agn2_ref/data/blr"
             / "blr_caseB_T1e4_ne1e4.csv")
BLR_TABLE_SHA256 = "1b92ed81caf837012099e255766fec5d73ecdf7878308903fbb75dec4fa106d1"
PLAN_LINES = [("Ba-alpha 6563", 74), ("Ba-beta 4861", 59), ("Ba-gamma 4341", 51),
              ("Ba-delta 4101.76A", 50), ("Pa-alpha 1.8751um", 107),
              ("Pa-beta 1.28181um", 104), ("Pa-gamma 1.09381um", 102),
              ("Pa-delta 1.00494um", 97)]


def test_V_BLR_6_line_set_and_table():
    """PLAN_AGN2 1.2: H I only, the 8 FSPS lines by name; ratios = RESEARCH's Case-B table
    (agn2_ref/data/blr, sha256 above) exactly; default blr_sigma inside [400, 5000]."""
    model, csp, _ = _model("corrected", "V-BLR-6")
    import hashlib
    blr = csp._agn.model.blr
    assert list(zip(blr.line_names, [int(i) for i in blr.fsps_index])) == PLAN_LINES
    if not BLR_TABLE.exists():
        _ref.missing(BLR_TABLE, "V-BLR-6 (table parity)")
    assert hashlib.sha256(BLR_TABLE.read_bytes()).hexdigest() == BLR_TABLE_SHA256
    rows = [r.split(",") for r in BLR_TABLE.read_text().splitlines()
            if r and not r.startswith("#")][1:]
    want = np.array([float(r[3]) for r in rows])
    got = np.asarray(blr.line_luminosity(csp._agn.filled({"blr_lum": jnp.asarray(1.0)})))
    np.testing.assert_array_equal(got, want)
    prov = blr.provenance() if hasattr(blr, "provenance") else {}
    assert BLR_TABLE_SHA256 in " ".join(str(v) for v in prov.values()), prov
    d = blr.default_params()["blr_sigma"]
    assert 400.0 <= d <= 5000.0


def _cigale_profile(wl, lam0, fwhm_kms):
    """CIGALE nebular_AGN.py:143-159 (branch nebular_agn_new 74b43a93), in NumPy: L_lambda of
    a unit line, Gaussian in lambda with FWHM width w = lam0 FWHM / c."""
    import scipy.constants as cst
    k1 = -4.0 * np.log(2)
    k2 = 0.5 * np.sqrt(np.pi / np.log(2))
    w = lam0 * fwhm_kms * 1e3 / cst.c
    return np.exp(k1 / (w * w) * (wl - lam0) ** 2) / (w * k2)


@pytest.mark.parametrize("fwhm", [800.0, 3000.0, 7000.0])
def test_V_BLR_7_cigale_fwhm_convention(fwhm):
    """PLAN_AGN2 1.3: a CIGALE line of FWHM F equals a CERIDWEN line of sigma = F / 2.35482:
    same area and same FWHM. The CERIDWEN painter is Gaussian in ln(lambda) (nebular
    convention), CIGALE's in lambda, so the shapes differ at O(sigma/c); a relative 1e-10 of
    the peak is not reachable by construction, so the test bounds it by 2 sigma/c."""
    from ceridwen.broadening import make_line_painter_free_z
    lam0 = 6564.7229
    sig = fwhm / _ref.FWHM_PER_SIGMA
    wl = lam0 * np.exp(np.linspace(-12, 12, 40001) * sig / _ref.CKMS)
    paint = make_line_painter_free_z(wl, [lam0], np.zeros_like(wl))
    fnu = np.asarray(paint(jnp.asarray([1.0]), sig, 1.0))           # unit flux, F_nu
    lam_l = fnu * _ref.C_AA_S / wl ** 2                               # -> F_lambda per A
    cig = _cigale_profile(wl / 10.0, lam0 / 10.0, fwhm) / 10.0        # per nm -> per A
    area_c = np.trapezoid(lam_l, wl)
    area_g = np.trapezoid(cig, wl)
    assert area_c == pytest.approx(1.0, rel=B8)          # a float32 profile (was rel=1e-9)
    assert area_g == pytest.approx(1.0, rel=1e-9)
    f_c = _ref.fwhm_kms(wl, fnu, lam0)
    # CIGALE's FWHM is defined on L_lambda in lambda: measure it there (half-max crossings)
    half = 0.5 * cig.max()
    j = int(np.argmax(cig))
    il = np.nonzero(cig[:j] < half)[0][-1]
    ir = j + np.nonzero(cig[j:] < half)[0][0]
    xl = np.interp(half, [cig[il], cig[il + 1]], [wl[il], wl[il + 1]])
    xr = np.interp(half, [cig[ir], cig[ir - 1]], [wl[ir], wl[ir - 1]])
    f_g = (xr - xl) / lam0 * _ref.CKMS
    assert f_c == pytest.approx(fwhm, rel=1e-4)
    assert f_g == pytest.approx(fwhm, rel=1e-4)
    diff = np.max(np.abs(lam_l - cig)) / cig.max()
    assert diff <= 2 * sig / _ref.CKMS, diff
