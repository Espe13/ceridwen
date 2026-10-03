"""BLR development tests (session BLR): ``ceridwen/agn/blr.py`` beyond VAL's V-BLR-* suite.

D-BLR-1  the committed Case-B table is RESEARCH's file byte for byte (sha256), H alpha / H beta
         = 2.847 (Storey & Hummer 1995 Case B, T = 1e4 K, n_e = 1e4, PLAN_AGN2 1.1).
D-BLR-2  table and config refusals (teaching errors at setup, never in a kernel).
D-BLR-3  CIGALE width convention: a CERIDWEN broad line with sigma = FWHM / 2.35482 is CIGALE's
         Gaussian of that FWHM (nebular_AGN.py:143-159 on branch nebular_agn_new), to rounding.
D-BLR-4  the painted Spectrum equals an independent NumPy recomputation of every broad line
         (Gaussian in ln lambda, sigma^2 = blr_sigma^2 + sigma_inst^2, flux from the
         definitions) to 1e-10 of the peak; the Lines observation of H alpha and H beta equals
         the recomputed fluxes to 1e-12.
D-BLR-5  photometry: the BLR photometry equals the delta-function estimate
         sum T(lam) lam F / ab_zero_counts within 1 % (the line is narrow next to the filter).
D-BLR-6  transmission: corrected with the dust4 screen, T_BLR = T_d4 x T_gal at each line;
         cigale, T_BLR = 1 with galaxy dust on.
D-BLR-7  energy balance (corrected, dust emission): L_absorbed moves by
         sum L x M_formed x (1 - T_gal) when the broad lines are switched on.
D-BLR-8  blr_sigma is not limited by Kinematics.sigma_max (dense painter): a 5000 km/s line with
         sigma_max = 600 has the measured FWHM.
D-BLR-9  gradients: finite and equal to central differences in blr_lum and blr_sigma at the lower
         edge (400, 410 km/s), the middle and 5000 km/s; jit(vmap(value_and_grad)) = the loop.
D-BLR-10 range_messages: negative / non-finite / outside-support blr_sigma warns (FWHM hint).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _blr_model as B  # noqa: E402
from _blr_model import _ref, jax, jnp  # noqa: E402

from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.agn import blr as blrmod  # noqa: E402
from ceridwen.broadening import agn_line_profiles_on_grid  # noqa: E402

STAGED = pathlib.Path(os.path.expanduser(
    "~/Desktop/PhD/Tools/ceridwen_workspace/agn2_ref/data/blr/blr_caseB_T1e4_ne1e4.csv"))
SHA = "1b92ed81caf837012099e255766fec5d73ecdf7878308903fbb75dec4fa106d1"   # PLAN_AGN2 1.2
C_AA_S = 2.99792458e18


def _sps(test):
    if not os.environ.get("SPS_HOME"):
        _ref.missing("$SPS_HOME", test)
    return os.environ["SPS_HOME"]


# ---------------------------------------------------------------- D-BLR-1 table
def test_D_BLR_1_table():
    data = blrmod.BLR_FILE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA
    if STAGED.is_file():
        assert STAGED.read_bytes() == data
    tab = blrmod.load_blr_table()
    r = dict(zip(tab["names"], tab["ratio"]))
    assert r["Ba-alpha 6563"] == 1.0
    assert 1.0 / r["Ba-beta 4861"] == pytest.approx(2.847, abs=5e-4)
    bl = blrmod.BroadLines(mode="corrected", sps_home=_sps("D-BLR-1"))
    assert bl.line_names == tab["names"]
    np.testing.assert_allclose(bl.line_wave, tab["wave_vac"], atol=1e-3)
    np.testing.assert_array_equal(bl.fsps_index, tab["fsps_index"])
    assert bl.region == "blr" and bl.width_key == "blr_sigma"
    assert bl.default_params() == {"blr_lum": 0.0, "blr_sigma": 1500.0}
    prov = bl.provenance()
    assert prov["table_sha256"] == SHA and "sigma" in prov["width"]


# ---------------------------------------------------------------- D-BLR-2 refusals
_HDR = "name,fsps_index,wave_vac,ratio_halpha\n"


@pytest.mark.parametrize("body,match", [
    ("Ba-beta 4861,59,4862.7629,0.35\n", "no 'Ba-alpha 6563' row"),
    ("Ba-alpha 6563,74,6564.7229,2.8\n", "not 1"),
    ("Ba-alpha 6563,74,6564.7229,1.0\nBa-alpha 6563,74,6564.7229,1.0\n", "duplicate"),
    ("Ba-alpha 6563,74,6564.7229,1.0\nBa-beta 4861,59,4862.7629,-0.3\n", "finite and > 0"),
    ("Ba-alpha 6563,74,6564.7229,1.0\nBa-beta 4861,59,4862.7629,nan\n", "finite and > 0"),
    ("Ba-alpha 6563,74,6564.7229\n", "every row needs"),
])
def test_D_BLR_2_table_refusals(tmp_path, body, match):
    p = tmp_path / "t.csv"
    p.write_text(_HDR + body)
    with pytest.raises(ValueError, match=match):
        blrmod.load_blr_table(p)


def test_D_BLR_2_fsps_and_mode_refusals(tmp_path):
    sps = _sps("D-BLR-2")
    p = tmp_path / "t.csv"
    p.write_text(_HDR + "Ba-alpha 6563,74,6564.7229,1.0\nH-foo 1234,3,1234.0,0.1\n")
    with pytest.raises(ValueError, match="not in the FSPS list"):
        blrmod.BroadLines(mode="corrected", file=p, sps_home=sps)
    p.write_text(_HDR + "Ba-alpha 6563,74,6570.0,1.0\n")
    with pytest.raises(ValueError, match="table 6570.00 A"):
        blrmod.BroadLines(mode="corrected", file=p, sps_home=sps)
    with pytest.raises(ValueError, match="not found"):
        blrmod.BroadLines(mode="corrected", file=tmp_path / "nope.csv", sps_home=sps)
    for mode in ("fsps", "prospector"):
        with pytest.raises(ValueError, match="only in"):
            blrmod.BroadLines(mode=mode, sps_home=sps)
        with pytest.raises(ValueError, match="AGN2 components"):
            AGNConfig(agn_mode=mode, blr=True)
    with pytest.raises(ValueError, match="needs blr=True"):
        AGNConfig(blr_file=str(p))
    # a table of the same layout with another ratio is used as given (by name)
    p.write_text(_HDR + "Ba-alpha 6563,74,6564.7229,1.0\nBa-beta 4861,59,4862.7629,0.25\n")
    bl = blrmod.BroadLines(mode="cigale", file=p, sps_home=sps)
    np.testing.assert_array_equal(np.asarray(bl.line_luminosity({"blr_lum": 2.0})), [2.0, 0.5])


# ---------------------------------------------------------------- D-BLR-3 CIGALE convention
@pytest.mark.parametrize("fwhm", [800.0, 3000.0, 11774.1])
def test_D_BLR_3_cigale_fwhm_convention(fwhm):
    """CIGALE (nebular_AGN.py:143-159): L_lambda = L exp(-4 ln2 (lam - lam0)^2 / w^2) /
    (w 0.5 sqrt(pi / ln2)), w = lam0 FWHM 1e3 / c [c in m/s]. CERIDWEN's grid profile with the
    resolution floor off is a Gaussian in lambda of sigma lam0 blr_sigma / c, as L_nu
    (x lam0^2 / c); converted back to L_lambda it equals CIGALE's with blr_sigma = FWHM / 2.35482
    and CIGALE's c replaced by the nebular constant ``CLIGHT_AA_S`` = 2.9979e18 A/s, to 1e-12 of
    the peak. With CIGALE's exact c (299792458 m/s) the widths differ by 8.2e-6 relative, which
    leaves at most 2e-5 of the peak (pre-existing: the nebular painter's truncated c, while the
    Spectrum painter uses the exact one, broadening.CKMS)."""
    from ceridwen.neb.NebularGridModel import CLIGHT_AA_S
    lam0 = 6564.7229
    lam = np.linspace(lam0 * 0.8, lam0 * 1.2, 40001)
    k1 = -4.0 * np.log(2.0)
    k2 = 0.5 * np.sqrt(np.pi / np.log(2.0))
    sig = fwhm / _ref.FWHM_PER_SIGMA
    prof = np.asarray(agn_line_profiles_on_grid(lam, np.array([lam0]), jnp.asarray(sig),
                                                res_floor_factor=0.0))[:, 0]
    ours = prof * CLIGHT_AA_S / lam0 ** 2     # L_nu of a unit line -> L_lambda
    for c_ms, tol in ((CLIGHT_AA_S * 1e-10, 1e-12), (299792458.0, 2e-5)):
        w = lam0 * fwhm * 1e3 / c_ms
        cig = np.exp(k1 / w ** 2 * (lam - lam0) ** 2) / (w * k2)
        assert np.trapezoid(cig, lam) == pytest.approx(1.0, rel=1e-9)
        np.testing.assert_allclose(ours, cig, rtol=0, atol=tol * cig.max())


# ---------------------------------------------------------------- D-BLR-4 spectrum + Lines
def _analytic_spectrum(model, csp, spec, F, sig, **kw):
    """Every broad line on the Spectrum pixels, from scratch: f_nu = F phi(ln lam) lam / c with
    phi a unit Gaussian in ln lam of width sqrt(sig^2 + sigma_inst(lam_l)^2) / c."""
    wo = np.asarray(spec.wavelength, np.float64)
    si = np.asarray(spec._proj.sigma_inst_kms, np.float64)
    blr = csp._agn.model.blr
    out = np.zeros_like(wo)
    for Fl, lam in zip(F, blr.line_wave):
        c = lam * (1 + B.ZRED)
        s = np.hypot(sig, np.interp(c, wo, si)) / _ref.CKMS
        x = (np.log(wo) - np.log(c)) / s
        out += Fl * np.exp(-0.5 * x * x) / (np.sqrt(2 * np.pi) * s) * wo / C_AA_S
    return out


@pytest.mark.parametrize("sig", [400.0, 1500.0, 5000.0])
def test_D_BLR_4_spectrum_and_lines_vs_numpy(sig):
    model, csp, spec = B.build("D-BLR-4")
    kw = dict(blr_sigma=sig, diffuse_tau_kc=0.4)
    d = B.broad(model, blr_lum=0.8, **kw)
    F = B.expected_fluxes(model, csp, blr_lum=0.8, **kw)
    want = _analytic_spectrum(model, csp, spec, F, sig)
    np.testing.assert_allclose(d["spec"], want, rtol=0, atol=1e-10 * want.max())
    names = list(csp._agn.model.blr.line_names)
    np.testing.assert_allclose(d["lines"], F[[names.index("Ba-alpha 6563"),
                                              names.index("Ba-beta 4861")]], rtol=1e-12)


# ---------------------------------------------------------------- D-BLR-5 photometry
def test_D_BLR_5_photometry_delta_estimate():
    from ceridwen.agn.lines import line_photometry_delta
    model, csp, _ = B.build("D-BLR-5")
    kw = dict(blr_sigma=1000.0, diffuse_tau_kc=0.2)
    d = B.broad(model, **kw)
    F = B.expected_fluxes(model, csp, **kw)
    obs = [o for o in model.observations if o.name == "phot"][0]
    est = np.asarray(line_photometry_delta(obs.filters,
                                           csp._agn.model.blr.line_wave * (1 + B.ZRED), F))
    j = int(np.argmax(est))                    # the filter holding broad H alpha (sdss_z0)
    assert est[j] > 0
    assert d["phot"][j] == pytest.approx(est[j], rel=1e-2)


# ---------------------------------------------------------------- D-BLR-6 transmission
def test_D_BLR_6_dust4_screen_corrected():
    model, csp, _ = B.build("D-BLR-6", dust4=True, phot=False)
    kw = dict(blr_sigma=2000.0, diffuse_tau_kc=0.5)
    a = B.broad(model, dust4=0.0, **kw)["lines"]
    b = B.broad(model, dust4=0.7, dust4_index=-1.3, **kw)["lines"]
    t_d4 = np.exp(-0.7 * (np.array([B.HA, B.HB]) / 5500.0) ** -1.3)
    np.testing.assert_allclose(b / a, t_d4, rtol=1e-6)


def test_D_BLR_6_cigale_no_galaxy_dust():
    model, csp, _ = B.build("D-BLR-6c", mode="cigale", phot=False)
    a = B.broad(model, blr_sigma=2000.0, diffuse_tau_kc=0.0)
    b = B.broad(model, blr_sigma=2000.0, diffuse_tau_kc=1.0)
    np.testing.assert_allclose(b["lines"], a["lines"], rtol=1e-12)
    np.testing.assert_allclose(b["spec"], a["spec"], rtol=0, atol=1e-12 * np.abs(a["spec"]).max())


# ---------------------------------------------------------------- D-BLR-7 energy balance
def test_D_BLR_7_energy_balance():
    model, csp, _ = B.build("D-BLR-7", duste=True, phot=False)
    kw = dict(blr_sigma=1500.0, diffuse_tau_kc=0.6)
    on = csp._agn_parts(B.mtheta(model, B.theta(model, blr_lum=0.9, **kw)))
    off = csp._agn_parts(B.mtheta(model, B.theta(model, blr_lum=0.0, **kw)))
    mt = B.mtheta(model, B.theta(model, **kw))
    msum = float(jnp.sum(csp.calculate_ssp_weights(theta=mt)))
    dc = np.exp(-np.asarray(csp.diff_dust.compute_attenuation(csp.wave, mt), np.float64))
    blr = csp._agn.model.blr
    t = np.interp(blr.line_wave, np.asarray(csp.wave, np.float64), dc)
    want = np.sum(0.9 * blr.ratios * msum * (1 - t))
    got = float(on["L_absorbed"]) - float(off["L_absorbed"])
    assert want > 0 and got == pytest.approx(want, rel=1e-10)


# ---------------------------------------------------------------- D-BLR-8 sigma_max
def test_D_BLR_8_not_limited_by_sigma_max():
    model, csp, spec = B.build("D-BLR-8", sigma_max=600.0, phot=False)
    d = B.broad(model, blr_sigma=5000.0)
    wo = np.asarray(spec.wavelength, np.float64)
    c = B.HA * (1 + B.ZRED)
    s_inst = float(np.interp(c, wo, np.asarray(spec._proj.sigma_inst_kms, np.float64)))
    assert _ref.fwhm_kms(wo, d["spec"], c) == pytest.approx(
        _ref.FWHM_PER_SIGMA * np.hypot(5000.0, s_inst), rel=1e-2)


# ---------------------------------------------------------------- D-BLR-9 gradients
def test_D_BLR_9_grad_fd_and_vmap():
    model, csp, _ = B.build("D-BLR-9")
    th0 = B.theta(model, blr_lum=0.5, blr_sigma=1500.0, diffuse_tau_kc=0.3)

    def f(lum, sig):
        out = model.predict({**th0, "blr_lum": jnp.reshape(lum, (1,)),
                             "blr_sigma": jnp.reshape(sig, (1,))})
        return (jnp.sum(out["spec"] ** 2) * 1e54 + jnp.sum(out["lines"]) * 1e15
                + jnp.sum(out["phot"].astype(jnp.float64)) * 1e9)

    vg = jax.jit(jax.value_and_grad(f, argnums=(0, 1)))
    worst = 0.0
    for sig in (400.0, 410.0, 2000.0, 5000.0):
        _, (gl, gs) = vg(jnp.asarray(0.5), jnp.asarray(sig))
        assert np.isfinite(float(gl)) and np.isfinite(float(gs)) and float(gs) != 0.0, sig
        h = 1e-3 * sig
        fd = (float(f(0.5, sig + h)) - float(f(0.5, sig - h))) / (2 * h)
        rel = abs(float(gs) - fd) / abs(fd)
        worst = max(worst, rel)
        assert rel < 1e-5, (sig, float(gs), fd)
        fdl = (float(f(0.5 + 1e-4, sig)) - float(f(0.5 - 1e-4, sig))) / 2e-4
        assert float(gl) == pytest.approx(fdl, rel=1e-5), sig
    print(f"D-BLR-9 largest relative grad-vs-FD difference in blr_sigma: {worst:.2e}")
    lums = jnp.array([0.1, 0.5, 1.0])
    sigs = jnp.array([400.0, 1500.0, 5000.0])
    bv, (bgl, bgs) = jax.jit(jax.vmap(jax.value_and_grad(f, argnums=(0, 1))))(lums, sigs)
    for k in range(3):
        v, (gl, gs) = vg(lums[k], sigs[k])
        np.testing.assert_allclose([float(bv[k]), float(bgl[k]), float(bgs[k])],
                                   [float(v), float(gl), float(gs)], rtol=1e-6)


# ---------------------------------------------------------------- D-BLR-10 range messages
def test_D_BLR_10_range_messages():
    bl = blrmod.BroadLines(mode="corrected", sps_home=_sps("D-BLR-10"))
    assert bl.range_messages({"blr_sigma": np.array([1500.0])}) == []
    assert bl.range_messages({}) == []
    m = bl.range_messages({"blr_sigma": np.array([7000.0])})
    assert len(m) == 1 and "2.35482" in m[0]
    assert "finite and > 0" in bl.range_messages({"blr_sigma": np.array([-10.0])})[0]
    assert "finite and > 0" in bl.range_messages({"blr_sigma": np.array([np.nan])})[0]
