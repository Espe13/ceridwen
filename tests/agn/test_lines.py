"""AGN narrow emission lines (``ceridwen.agn.lines``; AGN_PLAN §4: A4, B3, B4, B8, B9).

A4  line luminosities equal Prospector's ``AGNSpecModel`` exactly; the "prospector" painter,
    photometry and Lines terms equal Prospector's (``tests/reference/agn/A4_lines_prospector.npz``,
    session REF) and a literal NumPy transcription of ``predict_aline_spec`` (which, run against
    the real Prospector in the AGN-LINES session, agreed to 5e-13; REPORT_feat-agn-lines.md).
B3  agn_elum = 0 adds exactly zero everywhere (so a spectrum is byte-identical to AGN off).
B4  every AGN-line output is exactly linear in agn_elum.
B8  jit, vmap at the production widths (64, 100), finite gradients in agn_elum and
    agn_eline_sigma that match central finite differences.
B9  marginalize_elines=True with AGN lines is refused at setup with a teaching error.
Also: the line table is refused at setup on a different FSPS line list; Lines observations are
matched BY NAME; the "corrected" painters are the nebular ones with agn_eline_sigma.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ceridwen.agn import AGNConfig, AGNModel, AGN_THETA_DEFAULTS
from ceridwen.agn.lines import AGNLines, line_photometry_delta, HBETA_INDEX
from ceridwen.broadening import (Instrument, Kinematics, SpectralProjector,
                                 agn_line_profiles_on_grid, make_agn_line_painter_prospector,
                                 make_line_painter_free_z, C_AA_S)
from ceridwen.observation import Lines, Photometry, Spectrum

SPS_HOME = os.environ.get("SPS_HOME")
pytestmark = pytest.mark.skipif(not SPS_HOME, reason="SPS_HOME not set: the AGN line table is "
                                "read against $SPS_HOME/data/emlines_info.dat")
# session REF's dump; $CERIDWEN_AGN_REF_DIR points at another copy of tests/reference/agn
REF_A4 = (pathlib.Path(os.environ.get("CERIDWEN_AGN_REF_DIR") or
                       pathlib.Path(__file__).resolve().parents[1] / "reference" / "agn")
          / "A4_lines_prospector.npz")
W_PROD = (64, 100)
A4_PHOT_RTOL_VENDORED = 1e-4    # see test_A4_against_prospector_reference

# Prospector AGNSpecModel.init_aline_info (prospect/models/sedmodel.py, bd-j/prospector
# 91f8e13; identical in 40e2614), verbatim
P_AINDS = np.array([38, 40, 41, 43, 45, 50, 51, 52, 59,
                    61, 62, 64, 68, 69, 70, 72, 73, 74,
                    75, 76, 77, 78, 80])
P_AFLUXES = np.array([2.96, 0.06, 0.1, 1., 0.2, 0.25, 0.48, 0.13, 1.,
                      2.87, 8.53, 0.07, 0.02, 0.1, 0.33, 0.09, 0.79, 2.86,
                      2.13, 0.03, 0.77, 0.65, 0.19])


@pytest.fixture(scope="module")
def agn():
    return AGNLines(mode="prospector")


def _trapezoid(y, x, axis=0):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x, axis=axis)


def prospector_numpy(wave, agn_rest, sf_rest, rows, mask, F, sig_agn, sig_sf, eline_sig, opz):
    """Literal NumPy transcription of AGNSpecModel.predict_spec's AGN term
    (cache_eline_parameters(nsigma) + predict_aline_spec + get_eline_gaussians), on the
    compressed pixel array as Prospector does it."""
    ckms, lightspeed = 2.998e5, 2.998e18
    ew = sf_rest * opz
    nsigma = 5 * np.max(sig_agn / eline_sig)
    linewidth = nsigma * ew / ckms * sig_sf
    pm = np.abs(wave - ew[:, None]) < linewidth[:, None]
    if mask is not None:
        pm = pm & mask
    valid = pm.any(axis=1)
    emask = pm[valid].any(axis=0)
    out = np.zeros_like(wave)
    if not emask.any():
        return out
    on = valid[rows]
    warr = wave[emask]
    mu = np.atleast_2d(agn_rest[on] * opz)
    sigma = np.atleast_2d(np.full(on.sum(), sig_agn))
    dv = ckms * (warr[:, None] / mu - 1)
    dv_dnu = ckms * warr[:, None] ** 2 / (lightspeed * mu)
    g = 1. / (sigma * np.sqrt(np.pi * 2)) * np.exp(-dv ** 2 / (2 * sigma ** 2))
    g *= dv_dnu
    g /= -_trapezoid(g, 3e18 / warr[:, None], axis=0)
    out[emask] = (F[on] * g).sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# the table, by name, refused on a different FSPS list
# ---------------------------------------------------------------------------

def test_A4_table_is_prospectors_by_index_and_luminosities_exact(agn):
    """A4 (luminosities, exact): the stored names sit at Prospector's indices of the FSPS list,
    the ratios are Prospector's, index 59 is H beta, and L = agn_elum * ratio bitwise."""
    assert agn.n_lines == 23
    np.testing.assert_array_equal(agn.fsps_index, P_AINDS)
    np.testing.assert_array_equal(agn.ratios, P_AFLUXES)
    assert tuple(agn.emlines_names[i] for i in P_AINDS) == agn.line_names
    np.testing.assert_array_equal(agn.line_wave, agn.emlines_wave[P_AINDS])
    assert abs(agn.emlines_wave[HBETA_INDEX] - 4863) < 2
    assert agn.line_names[list(agn.fsps_index).index(HBETA_INDEX)] == "Ba-beta 4861"
    for elum in (1e-6, 3.7e-3, 1e-2):
        aline = np.zeros(agn.emlines_wave.size)
        aline[P_AINDS] = P_AFLUXES                      # Prospector's _aline_lum
        L = np.asarray(agn.line_luminosity({"agn_elum": jnp.array([elum])}))
        assert np.array_equal(L, aline[P_AINDS] * elum)
    assert agn.default_params() == {"agn_elum": 0.0, "agn_eline_sigma": 100.0}
    assert float(agn.line_luminosity({}).sum()) == 0.0          # default agn_elum = 0
    assert float(agn.sigma_kms({})) == AGN_THETA_DEFAULTS["agn_eline_sigma"]


def _fake_sps_home(tmp_path, edit):
    rows = open(pathlib.Path(SPS_HOME) / "data" / "emlines_info.dat").read().splitlines()
    rows = edit(rows)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "emlines_info.dat").write_text("\n".join(rows) + "\n")
    return str(tmp_path)


def test_different_fsps_line_list_is_refused_at_setup(tmp_path):
    # a line inserted before H beta: index 59 is no longer H beta (Prospector's assert)
    home = _fake_sps_home(tmp_path / "a", lambda r: r[:10] + ["1000.0,Fake 1000"] + r[10:])
    with pytest.raises(ValueError, match="is not H beta"):
        AGNLines(mode="corrected", sps_home=home)
    # a renamed line at an AGN index: the ratio would land on another line
    home = _fake_sps_home(tmp_path / "b", lambda r: r[:62] + ["5008.3137,[X Y] 5007"] + r[63:])
    with pytest.raises(ValueError, match=r"differs from the FSPS line list.*\[O III\] 5007"):
        AGNLines(mode="prospector", sps_home=home)
    with pytest.raises(ValueError, match="choose one of"):
        AGNLines(mode="bogus")


def test_lines_observation_matched_by_name(agn):
    """He I 3889 and H 8 (Ba-6 3889) lie 0.42 A apart: matched by name, never confused; a blend
    sums its AGN components; a non-AGN line gets a zero row; > 1 A from every line raises."""
    i_he, i_h8 = agn.emlines_names.index("He I 3888.63A"), agn.emlines_names.index("Ba-6 3889")
    w = agn.emlines_wave
    obs = Lines(line_ind=[i_he, i_h8, 0, 0, 0],
                wavelength=[w[i_he], w[i_h8], 4862.7, 3727.1, w[20]],
                components=[(w[i_he],), (w[i_h8],), (4862.7,), (3727.1, 3730.1), (w[20],)],
                flux=np.ones(5), uncertainty=np.ones(5), name="l")
    M = obs.agn_line_matrix(agn)
    col = {n: k for k, n in enumerate(agn.line_names)}
    assert M.shape == (5, 23)
    assert M[0].sum() == 0.0                                       # He I 3889: not an AGN line
    assert M[1, col["Ba-6 3889"]] == 1.0 and M[1].sum() == 1.0
    assert M[2, col["Ba-beta 4861"]] == 1.0 and M[2].sum() == 1.0
    # [O II] blend: Prospector puts the whole doublet on 3726, nothing on 3729
    assert M[3, col["[O II] 3726"]] == 1.0 and M[3].sum() == 1.0
    assert M[4].sum() == 0.0
    assert obs.agn_line_matrix(agn) is M                            # cached
    bad = Lines(line_ind=[0], wavelength=[4870.0], flux=np.ones(1), uncertainty=np.ones(1))
    with pytest.raises(ValueError, match="no FSPS line within 1 A"):
        bad.agn_line_matrix(agn)


# ---------------------------------------------------------------------------
# painters
# ---------------------------------------------------------------------------

Z = 2.0
WAVE_MODEL = np.exp(np.linspace(np.log(900.0), np.log(3.0e4), 6000))
WAVE_OBS = np.exp(np.arange(np.log(3600 * (1 + Z)), np.log(7300 * (1 + Z)), 1.0 / 2000 / 2.3548 / 2))


def _projector(mode, agn, *, gas="sigma_gas", mask=None, free_z=False, scale=1.0):
    kin = Kinematics(sigma_gal=150.0, sigma_gas=gas)
    inst = Instrument.R_fwhm(2000.0, scale=scale)
    p = SpectralProjector.build(kin, inst, WAVE_MODEL, WAVE_OBS, Z,
                                lib_sigma_kms=np.full(WAVE_MODEL.size, 40.0),
                                line_wave_rest=agn.emlines_wave,
                                zred_range=(1.9, 2.1) if free_z else None,
                                inst_scale_range=(0.8, 1.2) if isinstance(scale, str) else None)
    p.setup_agn_lines(agn.line_wave, mode=mode, sf_line_wave_rest=agn.emlines_wave,
                      agn_sf_rows=agn.fsps_index, pixel_mask=mask)
    return p


def _mask():
    m = np.ones(WAVE_OBS.size, bool)
    m[(WAVE_OBS > 4950 * (1 + Z)) & (WAVE_OBS < 5020 * (1 + Z))] = False     # through [O III]
    m[::17] = False
    return m


@pytest.mark.parametrize("sig_agn", [50.0, 300.0])
@pytest.mark.parametrize("eline_sig", [80.0, 150.0])
@pytest.mark.parametrize("masked", [False, True])
def test_A4_prospector_painter_equals_numpy_transcription(agn, sig_agn, eline_sig, masked):
    """A4 (painted spectrum): static-shape painter == Prospector's compressed-array algorithm,
    pixel support included, mask gaps included in the trapezoid."""
    mask = _mask() if masked else None
    rng = np.random.default_rng(1)
    sig_sf = rng.uniform(80.0, 160.0, agn.emlines_wave.size)
    F = np.asarray(agn.line_luminosity({"agn_elum": 3.7e-3})) * 1.3e-5
    paint = make_agn_line_painter_prospector(WAVE_OBS, agn.line_wave, agn.emlines_wave,
                                             agn.fsps_index, mask)
    got = np.asarray(jax.jit(paint)(jnp.asarray(F), sig_agn, jnp.asarray(sig_sf), eline_sig, 1 + Z))
    ref = prospector_numpy(WAVE_OBS, agn.line_wave, agn.emlines_wave, agn.fsps_index, mask, F,
                           sig_agn, sig_sf, eline_sig, 1 + Z)
    peak = np.abs(ref).max()
    assert peak > 0
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * peak)
    # same pixels, up to values XLA CPU flushes to zero (subnormals)
    diff = (got != 0) ^ (ref != 0)
    assert np.all(np.abs(ref[diff]) < 1e-290)


def test_projector_prospector_widths(agn):
    """The nebular widths the Prospector painter selects pixels with are Prospector's master
    formula hypot(hypot(eline_sigma, lib), sqrt(inst^2 - lib^2) or 0), lib and inst at the line."""
    p = _projector("prospector", agn)
    th = {"sigma_gas": jnp.array([90.0]), "agn_eline_sigma": jnp.array([200.0])}
    F = jnp.asarray(agn.line_luminosity({"agn_elum": 1e-3}))
    got = np.asarray(p.agn_lines(F, 200.0, th))
    inst = np.interp(agn.emlines_wave * (1 + Z), WAVE_OBS, p.sigma_inst_kms)
    lib = np.full(agn.emlines_wave.size, 40.0)
    with np.errstate(invalid="ignore"):
        d = np.sqrt(inst ** 2 - lib ** 2)
    d[~np.isfinite(d)] = 0.0
    e = np.hypot(np.hypot(90.0, lib), d)
    ref = prospector_numpy(WAVE_OBS, agn.line_wave, agn.emlines_wave, agn.fsps_index, None,
                           np.asarray(F), 200.0, e, 90.0, 1 + Z)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_corrected_painter_is_the_nebular_profile(agn):
    """corrected: the nebular painter with agn_eline_sigma in place of sigma_gas (bitwise), unit
    line flux on a well-sampled grid, centred at rest x (1 + z + eline_delta_zred)."""
    p = _projector("corrected", agn)
    F = jnp.asarray(agn.line_luminosity({"agn_elum": 1.0}))
    th = {"sigma_gas": jnp.array([90.0])}
    got = np.asarray(p.agn_lines(F, 250.0, th))
    ref = make_line_painter_free_z(WAVE_OBS, agn.line_wave, p.sigma_inst_kms)(F, 250.0, 1 + Z)
    assert np.array_equal(got, np.asarray(ref))
    # flux conservation of an isolated line: [O III] 5007 alone
    k = agn.line_names.index("[O III] 5007")
    one = jnp.zeros(agn.n_lines).at[k].set(1.0)
    f = np.asarray(p.agn_lines(one, 250.0, th))
    nu = C_AA_S / WAVE_OBS
    assert abs(-_trapezoid(f, nu) - 1.0) < 1e-6
    # eline_delta_zred moves the AGN lines with the nebular ones
    th2 = dict(th, eline_delta_zred=jnp.array([1e-3]))
    f2 = np.asarray(p.agn_lines(one, 250.0, th2))
    c2 = np.sum(f2 / WAVE_OBS * np.log(WAVE_OBS)) / np.sum(f2 / WAVE_OBS)
    assert c2 == pytest.approx(np.log(agn.line_wave[k] * (1 + Z + 1e-3)), abs=2e-7)


def test_grid_profiles_are_the_nebular_line_profiles(agn):
    """corrected photometry: model-grid profiles == NebularModel.line_profiles' construction
    (eline_marginal.line_profiles_on_grid) at the same sigma, traced in sigma."""
    from ceridwen.likelihood.eline_marginal import line_profiles_on_grid
    for s in (0.0, 50.0, 300.0):
        got = np.asarray(jax.jit(lambda s: agn_line_profiles_on_grid(WAVE_MODEL, agn.line_wave, s))(s))
        ref = line_profiles_on_grid(WAVE_MODEL, agn.line_wave, s)
        # atol: XLA CPU flushes subnormals and underflows the far Gaussian wings differently
        # from NumPy, at <= 1e-120 of the peak; rtol: exp(-x^2/2) of a far wing carries x^2 ulp
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-100 * np.abs(ref).max())


def test_line_photometry_delta_is_prospectors_nebline_photometry(agn):
    """Prospector mode photometry: sum_l T(lambda_l) lambda_l F_l / ab_zero_counts, T linear on
    the filter's own curve, 0 outside (SpecModel.nebline_photometry)."""
    phot = Photometry(filters=["sdss_g0", "sdss_r0", "jwst_f200w"], flux=np.ones(3),
                      uncertainty=np.ones(3))
    F = np.asarray(agn.line_luminosity({"agn_elum": 1e-3})) * 2e-17
    lam = agn.line_wave * (1 + 0.3)
    got = np.asarray(line_photometry_delta(phot.filters, lam, F))
    ref = []
    for f in phot.filters:
        trans = np.interp(lam, np.asarray(f.wavelength), np.asarray(f.transmission), left=0., right=0.)
        idx = trans > 0
        ref.append((trans[idx] * lam[idx] * F[idx]).sum() / float(f.ab_zero_counts))
    np.testing.assert_allclose(got, ref, rtol=1e-14)
    assert got[2] == 0.0 and got[0] > 0 and got[1] > 0


# ---------------------------------------------------------------------------
# A4 against session REF's Prospector dump
# ---------------------------------------------------------------------------

@pytest.mark.prospector
def test_A4_against_prospector_reference(agn):
    if not REF_A4.is_file():
        pytest.skip(f"MISSING {REF_A4}: written by session REF (scripts/agn_reference/"
                    "make_refs.py); the Prospector A4 comparison did NOT run")
    d = dict(np.load(REF_A4, allow_pickle=True))
    z, elum = float(d["zred"]), float(d["agn_elum"])
    # luminosities: exact
    L = np.asarray(agn.line_luminosity({"agn_elum": elum}))
    ref_L = np.asarray(d["line_lum_lsun_per_mformed"]).reshape(-1)
    ref_L = ref_L[np.asarray(d["aline_inds"])] if ref_L.size != L.size else ref_L
    assert np.array_equal(L, ref_L)
    np.testing.assert_array_equal(np.asarray(d["aline_inds"]), agn.fsps_index)
    owave = np.asarray(d["obs_wave"], dtype=np.float64)
    fn_opz = float(np.ravel(d["flux_norm"])[0]) / (1 + z)
    paint = make_agn_line_painter_prospector(owave, agn.line_wave, agn.emlines_wave,
                                             agn.fsps_index, None)
    for k, sig in enumerate(np.asarray(d["agn_eline_sigma"])):
        sig_sf = np.asarray(d["eline_sigma_kms_restored"][k], dtype=np.float64) \
            * np.ones(agn.emlines_wave.size)
        got = np.asarray(paint(jnp.asarray(L * fn_opz), sig, jnp.asarray(sig_sf),
                               float(d["eline_sigma"]), 1 + z))
        ref = np.asarray(d["aline_spec"][k])
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-8 * np.abs(ref).max())
        # zero continuum and zero nebular lines in REF's stub: the whole spectrum is the AGN term
        np.testing.assert_allclose(np.asarray(d["spec_pred"][k]), ref, rtol=0,
                                   atol=1e-8 * np.abs(ref).max())
    # Lines: AGNSpecModel.predict_lines, F = agn_elum * ratio * line_norm (mass included)
    line_norm = float(np.ravel(d["line_norm"])[0])
    F = L * line_norm
    obs = Lines(line_ind=agn.fsps_index, wavelength=agn.line_wave, flux=np.ones(23),
                uncertainty=np.ones(23))
    for k in range(len(d["agn_eline_sigma"])):
        np.testing.assert_allclose(obs.agn_line_matrix(agn) @ F, np.asarray(d["line_pred"][k]),
                                   rtol=1e-14)
    # photometry: nebline_photometry. With the reference's own (sedpy) gridded curves the formula
    # must agree to rounding; with CERIDWEN's vendored curves the JWST NIRCam throughputs differ
    # from sedpy 1.0.0's by 2-4 % of peak (filter-file versions), which moves F115W-F200W by
    # <= 7.6e-5 of the brightest band: the filter data, not the line photometry.
    names = [str(f) for f in d["filters"]]
    curves = (SimpleNamespace(wavelength=d[f"filter_curves/{n}/wave"],
                              transmission=d[f"filter_curves/{n}/trans"],
                              ab_zero_counts=float(d[f"filter_curves/{n}/abzc"])) for n in names)
    if all(f"filter_curves/{n}/wave" in d for n in names):
        got = np.asarray(line_photometry_delta(list(curves), agn.line_wave * (1 + z), F))
        rtol = 1e-12
    else:
        phot = Photometry(filters=names, flux=np.ones(len(names)), uncertainty=np.ones(len(names)))
        got = np.asarray(line_photometry_delta(phot.filters, agn.line_wave * (1 + z), F))
        rtol = A4_PHOT_RTOL_VENDORED
    for k in range(len(d["agn_eline_sigma"])):
        ref = np.asarray(d["phot_pred"][k])
        np.testing.assert_allclose(got, ref, rtol=0, atol=rtol * np.abs(ref).max())


# ---------------------------------------------------------------------------
# B3 / B4: zero and linearity in agn_elum
# ---------------------------------------------------------------------------

def _all_outputs(agn, p_cor, p_pro, phot, lines, elum, sig=150.0):
    th = {"agn_elum": jnp.asarray(elum), "agn_eline_sigma": jnp.array([sig]),
          "sigma_gas": jnp.array([90.0]), "logmass": jnp.array([9.5])}
    F = agn.observed_fluxes(th, 3.1e-18)
    return {
        "L": agn.line_luminosity(th),
        "spec_corrected": p_cor.agn_lines(F, agn.sigma_kms(th), th),
        "spec_prospector": p_pro.agn_lines(F, agn.sigma_kms(th), th),
        "phot_delta": line_photometry_delta(phot.filters, agn.line_wave * (1 + Z), F),
        "phot_grid": agn_line_profiles_on_grid(WAVE_MODEL, agn.line_wave, agn.sigma_kms(th))
                     @ agn.line_luminosity(th),
        "lines": jnp.asarray(lines.agn_line_matrix(agn)) @ F,
    }


@pytest.fixture(scope="module")
def setup(agn):
    phot = Photometry(filters=["jwst_f115w", "jwst_f150w", "jwst_f200w"], flux=np.ones(3),
                      uncertainty=np.ones(3))
    w = agn.emlines_wave
    lines = Lines(line_ind=[59, 62, 74, 20], wavelength=w[[59, 62, 74, 20]], flux=np.ones(4),
                  uncertainty=np.ones(4), name="lines")
    return (_projector("corrected", agn), _projector("prospector", agn, mask=_mask()),
            phot, lines)


def test_B3_zero_agn_elum_adds_exact_zero(agn, setup):
    """B3: agn_elum = 0 -> every AGN-line output is exactly +0.0, so adding it leaves any
    spectrum byte-identical (x + 0.0 == x for every x except -0.0)."""
    out = _all_outputs(agn, *setup, elum=jnp.array([0.0]))
    rng = np.random.default_rng(0)
    for k, v in out.items():
        v = np.asarray(v)
        assert np.all(v == 0.0) and not np.any(np.signbit(v)), k
        x = rng.standard_normal(v.shape) * 1e-29
        assert (x + v).tobytes() == x.tobytes(), k


def test_B4_linear_in_agn_elum(agn, setup):
    """B4: exactly linear in agn_elum: bitwise for power-of-two factors (except values below
    1e-100 of the peak, where a product underflows to a subnormal that XLA CPU flushes),
    1e-14 otherwise."""
    a = _all_outputs(agn, *setup, elum=jnp.array([1.7e-3]))
    for fac, exact in ((2.0, True), (0.25, True), (3.3, False)):
        b = _all_outputs(agn, *setup, elum=jnp.array([1.7e-3 * fac]))
        for k in a:
            x, y = np.asarray(a[k]) * fac, np.asarray(b[k])
            if exact:
                big = np.abs(x) > 1e-100 * np.abs(x).max()
                assert np.array_equal(x[big], y[big]), (k, fac)
                assert np.all(np.abs((x - y)[~big]) <= 1e-100 * np.abs(x).max()), (k, fac)
            else:
                np.testing.assert_allclose(y, x, rtol=1e-14, atol=1e-15 * np.abs(x).max())


# ---------------------------------------------------------------------------
# B8: jit, vmap, gradients
# ---------------------------------------------------------------------------

def _loglike_factory(agn, p_cor, p_pro, phot, lines):
    """A stub likelihood that runs every AGN-line path (both spectrum painters, both photometry
    paths, Lines), as INT will wire them into CSPBasis.predict."""
    rng = np.random.default_rng(3)
    y_spec = rng.standard_normal(WAVE_OBS.size) * 1e-30
    prof_T = jnp.asarray(rng.uniform(0, 1e-3, (3, WAVE_MODEL.size)))
    M = jnp.asarray(lines.agn_line_matrix(agn))

    def loglike(x):
        th = {"agn_elum": x[0:1], "agn_eline_sigma": x[1:2], "sigma_gas": jnp.array([90.0]),
              "logmass": jnp.array([9.5])}
        F = agn.observed_fluxes(th, 3.1e-18)
        s = agn.sigma_kms(th)
        r1 = p_cor.agn_lines(F, s, th) - y_spec
        r2 = p_pro.agn_lines(F, s, th) - y_spec
        ph = line_photometry_delta(phot.filters, agn.line_wave * (1 + Z), F) \
            + prof_T @ (agn_line_profiles_on_grid(WAVE_MODEL, agn.line_wave, s)
                        @ (agn.line_luminosity(th) * 1e-12))
        ln = M @ F
        return -0.5 * (jnp.sum((r1 / 1e-31) ** 2) + jnp.sum((r2 / 1e-31) ** 2)
                       + jnp.sum((ph / 1e-12) ** 2) + jnp.sum((ln / 1e-20) ** 2))
    return loglike


@pytest.mark.parametrize("W", W_PROD)
def test_B8_jit_vmap_equal_loop(agn, setup, W):
    f = _loglike_factory(agn, *setup)
    rng = np.random.default_rng(W)
    X = jnp.asarray(np.column_stack([rng.uniform(1e-6, 1e-2, W), rng.uniform(50, 500, W)]))
    vg = jax.jit(jax.vmap(jax.value_and_grad(f)))(X)
    one = jax.jit(jax.value_and_grad(f))
    loop = [one(X[i]) for i in range(W)]
    np.testing.assert_allclose(np.asarray(vg[0]), [float(v) for v, _ in loop], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(vg[1]), np.stack([np.asarray(g) for _, g in loop]),
                               rtol=1e-10, atol=0)


@pytest.mark.parametrize("x0", [(1e-6, 50.0), (5e-3, 275.0), (1e-2, 500.0)])
def test_B8_grad_finite_and_matches_finite_differences(agn, setup, x0):
    """Finite gradients in agn_elum and agn_eline_sigma at the prior centre and both bounds
    (Prospector's priors: Uniform(1e-6, 1e-2), Uniform(50, 500)), equal to central differences."""
    f = jax.jit(_loglike_factory(agn, *setup))
    x = jnp.asarray(x0)
    g = np.asarray(jax.grad(f)(x))
    assert np.all(np.isfinite(g))
    for i, h in ((0, 1e-9 * max(x0[0], 1e-6)), (1, 1e-4)):
        e = np.zeros(2); e[i] = h
        fd = (float(f(x + e)) - float(f(x - e))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=2e-5), (i, g[i], fd)


def test_B8_free_z_and_sampled_instrument_scale(agn):
    """The painters also trace a sampled redshift and a sampled instrument scale."""
    for mode in ("corrected", "prospector"):
        p = _projector(mode, agn, free_z=True, scale="inst_scale")
        F = jnp.asarray(agn.line_luminosity({"agn_elum": 1e-3})) * 1e-17

        def f(x):
            th = {"zred": x[0:1], "inst_scale": x[1:2], "sigma_gas": jnp.array([90.0])}
            return jnp.sum(p.agn_lines(F, x[2], th) ** 2) * 1e60
        g = jax.grad(f)(jnp.array([2.03, 1.1, 120.0]))
        assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).sum()) > 0, mode


# ---------------------------------------------------------------------------
# B9: marginalize_elines with AGN lines
# ---------------------------------------------------------------------------

def _marg_spec():
    return Spectrum(wavelength=WAVE_OBS, flux=np.ones_like(WAVE_OBS),
                    uncertainty=np.ones_like(WAVE_OBS), instrument=Instrument.R_fwhm(2000.0),
                    marginalize_elines=True, name="spec")


@pytest.mark.parametrize("holder", ["model", "config"])
def test_B9_marginalize_elines_with_agn_lines_refused(holder):
    """B9: refused at setup, before anything else, with a message that names the cause and the
    two ways out; AGN without lines is not refused by this check."""
    from ceridwen.likelihood.eline_marginal import build_eline_system, refuse_with_agn_lines
    cfg = AGNConfig(agn_mode="corrected", torus=False, disk=False, lines=True)
    val = AGNModel(cfg, WAVE_MODEL) if holder == "model" else cfg
    csp = SimpleNamespace(agn=val)
    model = SimpleNamespace(observations=[_marg_spec()], csp=csp)
    with pytest.raises(ValueError) as e:
        build_eline_system(model)
    msg = str(e.value)
    for s in ("marginalize_elines=True with AGN narrow lines", "Prospector refuses this too",
              "lines=False", "marginalize_elines=False"):
        assert s in msg, s
    off = SimpleNamespace(agn=AGNConfig(agn_mode="corrected", torus=False, disk=False, lines=False))
    refuse_with_agn_lines(off, _marg_spec())          # no lines: not this refusal


def test_B9_refused_by_sedmodel_construction():
    """B9 through SedModel: the refusal fires while the model is built (setup), on a real
    nebular CSPBasis carrying AGN lines."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import test_emission_lines as T
    from ceridwen import SedModel
    from ceridwen.model.transforms import logsfr_ratios_to_sfh
    csp = T._csp(True)
    csp.agn = AGNModel(AGNConfig(agn_mode="corrected", torus=False, disk=False, lines=True),
                       np.asarray(csp.wave), sps_home=SPS_HOME)
    t = np.array(csp.sfh_times)
    spec = Spectrum(wavelength=T.WAVE_OBS, flux=np.ones_like(T.WAVE_OBS),
                    uncertainty=np.ones_like(T.WAVE_OBS), instrument=Instrument.R_fwhm(T.R_FWHM),
                    marginalize_elines=True, name="spec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="marginalize_elines=True with AGN narrow lines"):
            SedModel(csp, [spec], zred=T.ZRED,
                     kinematics=Kinematics(sigma_gal=150.0, sigma_gas=T.SIGMA_GAS),
                     transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(
                         th["logsfr_ratios"], sfh_times_yr=_t)},
                     free_param_init={"logsfr_ratios": jnp.zeros(len(t) - 1),
                                      "logmass": jnp.array([9.0])})
