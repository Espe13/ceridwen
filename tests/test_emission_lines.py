"""End-to-end checks of the emission-line treatment: every function a nebular line
passes through, from the CLOUDY grid to each observation type.

    NebularModel line list  ->  CSPBasis.predict_line_fluxes (default / for_spectrum /
    for_photometry)  ->  Lines (grid rows, blends, eline_scaling)
                     ->  Spectrum (SpectralProjector painter, fixed and free z; mask_lines)
                     ->  Photometry (static line basis vs painted lines)
    and the rest-frame line component (get_spectrum_components, get_line_spec).

The central claim is that ONE line flux reaches every observation consistently: the
flux a Lines observation reports equals the frequency integral of the line painted
into a Spectrum, and the photometric path carries the same flux (per Hz, x(1+z)).

The first block needs no SSP grid or FSPS and always runs; the rest skips without
the test grid or $SPS_HOME (a green run with skips is not a pass).
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
from ceridwen.broadening import (                                          # noqa: E402
    CKMS, C_AA_S, Kinematics, Instrument, make_line_painter, make_line_painter_free_z,
)
from ceridwen.observation import Photometry, Spectrum, Lines              # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh               # noqa: E402

ZRED = 0.3
SIGMA_GAS = 80.0
R_FWHM = 1000.0
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
# observed pixels, log-uniform at ~2.5 px per instrumental sigma: rest 4500-7000 A at z=0.3
WAVE_OBS = np.exp(np.arange(np.log(4500 * (1 + ZRED)), np.log(7000 * (1 + ZRED)),
                            1.0 / (2.3548 * R_FWHM) / 2.5))
# (name, vacuum rest wavelength [A]) of the lines checked individually; matched to
# grid rows by wavelength, as CSPBasis does
LINES = [("Hbeta", 4862.763), ("[OIII]5007", 5008.314), ("Halpha", 6564.723),
         ("[NII]6584", 6585.369)]
OII = (3727.118, 3730.119)


# ============================================================================
# Grid-free: the painter, mask_lines and Lines validation
# ============================================================================

def _moments(wave, fnu, lam0):
    """(integral over nu, mean of ln(lam/lam0), rms in ln lam) of a painted f_nu line."""
    nu = C_AA_S / wave
    flux = -np.trapezoid(fnu, nu)
    x = np.log(wave / lam0)
    w = fnu / wave            # f_nu = F phi(ln lam) lam / c  ->  phi ~ f_nu / lam
    mean = np.sum(w * x) / np.sum(w)
    rms = np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return flux, mean, rms


def test_painter_conserves_flux_centres_and_widths():
    """make_line_painter: a line of integrated flux F integrates to F over frequency, sits at
    its observed centre and has ln-lambda dispersion sqrt(sigma_gas^2 + sigma_inst^2)/c."""
    wave = np.exp(np.linspace(np.log(6000.0), np.log(7000.0), 6000))
    lam = np.array([6300.0, 6700.0])
    s_inst = np.array([50.0, 120.0])
    paint = make_line_painter(wave, lam, s_inst)
    for sig_gas in (0.0, 80.0, 400.0):
        for j in range(2):
            F = np.zeros(2)
            F[j] = 3.7e-17
            out = np.asarray(paint(jnp.asarray(F), sig_gas))
            flux, mean, rms = _moments(wave, out, lam[j])
            assert flux == pytest.approx(F[j], rel=1e-6, abs=0.0)
            assert abs(mean) < 1e-7
            assert rms * CKMS == pytest.approx(np.hypot(sig_gas, s_inst[j]), rel=1e-4, abs=0.0)


def test_painter_is_linear_in_flux():
    wave = np.exp(np.linspace(np.log(6000.0), np.log(7000.0), 3000))
    paint = make_line_painter(wave, np.array([6300.0, 6320.0]), np.array([60.0, 60.0]))
    a, b = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    np.testing.assert_allclose(np.asarray(paint(2.0 * a + 3.0 * b, 80.0)),
                               2.0 * np.asarray(paint(a, 80.0)) + 3.0 * np.asarray(paint(b, 80.0)),
                               rtol=1e-12, atol=0.0)


def test_free_z_painter_equals_fixed_painter_at_every_redshift():
    """The traced-redshift painter reproduces a painter built at that fixed redshift."""
    wave = np.exp(np.linspace(np.log(8000.0), np.log(9500.0), 4000))
    s_table = 30.0 + 40.0 * (wave - wave[0]) / (wave[-1] - wave[0])   # a sloped LSF
    lam_rest = np.array([6549.86, 6564.72, 6585.37])
    free = make_line_painter_free_z(wave, lam_rest, s_table)
    F = jnp.array([1.0, 3.0, 0.5])
    for z in (0.25, 0.3, 0.37):
        lw = lam_rest * (1 + z)
        fixed = make_line_painter(wave, lw, np.interp(lw, wave, s_table))
        np.testing.assert_allclose(np.asarray(free(F, 90.0, 1.0 + z)),
                                   np.asarray(fixed(F, 90.0)), rtol=1e-10, atol=1e-14)


def test_free_z_painter_is_differentiable_in_redshift():
    wave = np.exp(np.linspace(np.log(8000.0), np.log(9500.0), 4000))
    free = make_line_painter_free_z(wave, np.array([6564.72]), np.full(wave.size, 50.0))

    def centre(opz):
        f = free(jnp.array([1.0]), 90.0, opz)
        return jnp.sum(f * jnp.log(wave) / wave) / jnp.sum(f / wave)

    g = float(jax.grad(centre)(1.3))
    assert np.isfinite(g)
    assert g == pytest.approx(1.0 / 1.3, rel=1e-6, abs=0.0)      # d ln(lam_rest * opz) / d opz


def test_mask_lines_masks_exactly_the_velocity_window():
    wave = np.linspace(8000.0, 9000.0, 2001)
    spec = Spectrum(wavelength=wave, flux=np.ones_like(wave), uncertainty=np.ones_like(wave))
    spec.mask_lines([6564.72], dv=500.0, zred=0.3)
    lo = 6564.72 * 1.3
    inside = np.abs(wave - lo) <= lo * 500.0 / 2.998e5
    np.testing.assert_array_equal(np.asarray(spec.mask), ~inside)
    spec.mask_lines([6585.37], dv=300.0, zred=0.3)          # masks accumulate
    lo2 = 6585.37 * 1.3
    inside2 = np.abs(wave - lo2) <= lo2 * 300.0 / 2.998e5
    np.testing.assert_array_equal(np.asarray(spec.mask), ~(inside | inside2))


def test_lines_components_are_validated():
    kw = dict(flux=[1.0, 1.0], uncertainty=[1.0, 1.0])
    with pytest.raises(ValueError, match="components has 1 entries"):
        Lines(line_ind=[0, 1], wavelength=[4862.763, 6564.723], components=[(4862.763,)], **kw)
    with pytest.raises(ValueError, match=">= 1 wavelength"):
        Lines(line_ind=[0, 1], wavelength=[4862.763, 6564.723], components=[(4862.763,), ()], **kw)
    ok = Lines(line_ind=[0, 1], wavelength=[3727.118, 6564.723],
               components=[OII, (6564.723,)], **kw)
    assert ok.has_blends


# ============================================================================
# Full model: nebular CSP + Spectrum + Photometry + Lines at a fixed redshift
# ============================================================================

def _csp(add_neb=True):
    path = find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (see tests/_gridfixture.py)")
    if add_neb and not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY nebular grids)")
    cosmo = Cosmology.planck18()
    kw = dict(lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(ZRED))]),
              zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
              add_neb=add_neb, add_igm=False, verbose=False, cosmo=cosmo)
    if add_neb:
        kw["sps_home"] = os.environ["SPS_HOME"]
    return CSPBasis(SSPData.load(str(path)), **kw)


def _obs(lines_wave=None):
    spec = Spectrum(wavelength=WAVE_OBS, flux=np.ones_like(WAVE_OBS),
                    uncertainty=np.ones_like(WAVE_OBS),
                    instrument=Instrument.R_fwhm(R_FWHM), name="spec")
    phot = Photometry(filters=FILTERS, flux=np.ones(4), uncertainty=np.ones(4), name="phot")
    waves = [w for _, w in LINES] + [OII[0]] if lines_wave is None else list(lines_wave)
    comps = [(w,) for _, w in LINES] + [OII] if lines_wave is None else None
    lines = Lines(line_ind=np.arange(len(waves)), wavelength=waves, components=comps,
                  line_names=None if lines_wave is not None else [n for n, _ in LINES] + ["[OII]"],
                  flux=np.ones(len(waves)), uncertainty=np.ones(len(waves)), name="lines")
    return spec, phot, lines


def _model(csp, obs, extra_init=None):
    t = np.array(csp.sfh_times)
    init = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.0])}
    init.update(extra_init or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # unpriored sampled parameters
        return SedModel(csp, list(obs),
                        transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(
                            th["logsfr_ratios"], sfh_times_yr=_t)},
                        free_param_init=init, zred=ZRED,
                        kinematics=Kinematics(sigma_gal=150.0, sigma_gas=SIGMA_GAS))


def _theta(model, **over):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th["Z"] = jnp.array([-2.0])
    th["gas_logu"] = jnp.array([-2.23])          # off the CLOUDY logU nodes (kinks)
    th["gas_logz"] = jnp.array([-0.37])          # off the logZ nodes
    th.update({k: jnp.asarray(v) for k, v in over.items()})
    return th


def _csp_theta(model, th):
    """The theta the CSP sees inside SedModel.predict (transforms + injected zred)."""
    mt = dict(model.apply_transforms(th))
    mt["zred"] = jnp.array([ZRED])
    return mt


def _rows(csp, waves):
    pos = np.asarray(csp.neb.nebem_line_pos)
    idx = np.array([int(np.argmin(np.abs(pos - w))) for w in waves])
    assert np.max(np.abs(pos[idx] - np.asarray(waves))) < 1.0
    return idx


@pytest.fixture(scope="module")
def csp():
    return _csp(True)


@pytest.fixture(scope="module")
def built(csp):
    spec, phot, lines = _obs()
    model = _model(csp, (spec, phot, lines))
    th = _theta(model)
    return model, spec, phot, lines, th, model.predict(th)


def test_line_list_is_vacuum_and_matches_emlines_info(csp):
    pos = np.asarray(csp.neb.nebem_line_pos)
    assert csp.neb.emline_index_consistent is True
    assert np.all(np.isfinite(pos)) and np.all(pos > 0)
    # vacuum, not air: [O III]5007 is 5008.24 A in vacuum, 5006.84 A in air
    assert abs(pos[_rows(csp, [5008.3])[0]] - 5008.24) < 0.2


def test_line_flux_variants_are_one_flux(built, csp):
    """default == for_spectrum without eline_scaling; for_photometry == (1+z) x for_spectrum
    (per-Hz amplitude with the full f_nu flux factor, no IGM here)."""
    model, *_, th, _ = built
    mt = _csp_theta(model, th)
    F = np.asarray(csp.predict_line_fluxes(mt))
    Fs = np.asarray(csp.predict_line_fluxes(mt, for_spectrum=True))
    Fp = np.asarray(csp.predict_line_fluxes(mt, for_photometry=True))
    assert F.shape == (csp.neb.nemline,) and np.all(F >= 0) and np.all(np.isfinite(F))
    np.testing.assert_array_equal(F, Fs)
    np.testing.assert_allclose(Fp, (1.0 + ZRED) * Fs, rtol=1e-12)
    mt["eline_scaling"] = jnp.array([0.4])
    np.testing.assert_allclose(np.asarray(csp.predict_line_fluxes(mt)), 0.4 * Fs, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(csp.predict_line_fluxes(mt, for_spectrum=True)), Fs)


def test_line_flux_scales_with_mass_escape_and_dust(built, csp):
    model, *_, th, _ = built
    mt = _csp_theta(model, th)
    F = np.asarray(csp.predict_line_fluxes(mt))
    hb, ha = _rows(csp, [4862.763, 6564.723])

    m2 = dict(mt, logmass=mt["logmass"] + 1.0)
    np.testing.assert_allclose(np.asarray(csp.predict_line_fluxes(m2)), 10.0 * F, rtol=1e-10)

    m3 = dict(mt, frac_obrun=jnp.array([0.3]))
    np.testing.assert_allclose(np.asarray(csp.predict_line_fluxes(m3)), 0.7 * F, rtol=1e-10)

    dec = []
    for tau in (0.0, 0.5, 1.0):
        Ft = np.asarray(csp.predict_line_fluxes(dict(mt, diffuse_tau_kc=jnp.array([tau]))))
        dec.append(Ft[ha] / Ft[hb])
    assert 2.7 < dec[0] < 3.0, dec            # dust-free: case B, Ha/Hb ~ 2.86
    assert dec[0] < dec[1] < dec[2], dec      # dust reddens the Balmer decrement


def test_rest_frame_line_component_carries_the_line_luminosity(built, csp):
    """get_spectrum_components: the line component is non-negative and integrates over
    frequency to the summed grid line luminosity; get_line_spec is that component with
    mass and flux factor applied."""
    model, *_, th, _ = built
    mt = _csp_theta(model, th)
    rest = {k: v for k, v in mt.items() if k not in ("zred", "logmass")}
    F0 = np.asarray(csp.predict_line_fluxes(rest))
    _, lc = csp.get_spectrum_components(rest)
    lc = np.asarray(lc)
    wave = np.asarray(csp.wave)
    assert lc.min() >= 0.0
    assert -np.trapezoid(lc, C_AA_S / wave) == pytest.approx(F0.sum(), rel=1e-3, abs=0.0)

    _, lc_full = csp.get_spectrum_components(mt)
    _, scaled = csp._apply_mass_redshift_igm(lc_full, lc_full, mt)
    np.testing.assert_allclose(np.asarray(csp.get_line_spec(mt)), np.asarray(scaled),
                               rtol=1e-12, atol=0.0)


def test_spectrum_is_continuum_plus_painted_grid_lines(built, csp):
    """The Spectrum prediction is the projected line-free continuum plus every grid line
    painted from predict_line_fluxes(for_spectrum=True): nothing else, nothing twice."""
    model, spec, *_, th, pred = built
    mt = _csp_theta(model, th)
    cont = csp.get_spectrum(theta=mt, include_lines=False)
    _, cont_s = csp._apply_mass_redshift_igm(cont, cont, mt)
    Fs = csp.predict_line_fluxes(mt, for_spectrum=True)
    manual = spec._proj.predict(cont_s, Fs, mt)
    np.testing.assert_array_equal(np.asarray(pred["spec"]), np.asarray(manual))
    no_lines = spec._proj.predict(cont_s, jnp.zeros_like(Fs), mt)
    assert np.all(np.asarray(pred["spec"]) >= np.asarray(no_lines))
    assert np.max(np.asarray(pred["spec"]) - np.asarray(no_lines)) > 0.1 * np.max(np.asarray(no_lines))


def test_painted_spectral_line_equals_lines_observation(built, csp):
    """The core consistency check: for each line, the frequency integral of the line painted
    into the Spectrum equals the Lines prediction, and it sits at lambda_rest (1+z) with
    width sqrt(sigma_gas^2 + sigma_inst^2)."""
    model, spec, phot, lines, th, pred = built
    mt = _csp_theta(model, th)
    Fs = np.asarray(csp.predict_line_fluxes(mt, for_spectrum=True))
    rows = _rows(csp, [w for _, w in LINES])
    s_inst = CKMS / (2.0 * np.sqrt(2.0 * np.log(2.0)) * R_FWHM)
    for k, (name, lam) in enumerate(LINES):
        one = np.zeros_like(Fs)
        one[rows[k]] = Fs[rows[k]]
        painted = np.asarray(spec._proj.lines(jnp.asarray(one), SIGMA_GAS))
        flux, mean, rms = _moments(WAVE_OBS, painted, lam * (1 + ZRED))
        assert flux == pytest.approx(float(pred["lines"][k]), rel=1e-6, abs=0.0), name
        assert abs(mean) < 1e-6, name
        assert rms * CKMS == pytest.approx(np.hypot(SIGMA_GAS, s_inst), rel=1e-3, abs=0.0), name


def test_lines_observation_reads_grid_rows_and_sums_blends(built, csp):
    model, *_, th, pred = built
    F = np.asarray(csp.predict_line_fluxes(_csp_theta(model, th)))
    rows = _rows(csp, [w for _, w in LINES])
    oii = _rows(csp, OII)
    got = np.asarray(pred["lines"])
    np.testing.assert_allclose(got[:len(LINES)], F[rows], rtol=1e-6)
    assert got[-1] == pytest.approx(F[oii].sum(), rel=1e-6, abs=0.0)
    assert np.all(got > 0)


def test_photometry_static_line_basis_matches_painted_lines(csp):
    """Fixed sigma_gas: the static line-to-band basis gives the same photometry as painting
    the lines onto the model grid (the A/B switch _force_paint_lines)."""
    spec, phot, lines = _obs()
    model = _model(csp, (spec, phot, lines))
    th = _theta(model)
    static = np.asarray(model.predict(th)["phot"], dtype=np.float64)
    csp._force_paint_lines = True
    try:
        painted = np.asarray(model.predict(th)["phot"], dtype=np.float64)
    finally:
        csp._force_paint_lines = False
    np.testing.assert_allclose(static, painted, rtol=2e-4)
    # and the lines matter: dropping them changes the band containing H-alpha (sdss_i0)
    th0 = dict(th, frac_obrun=jnp.array([1.0]))
    assert abs(static[2] / float(model.predict(th0)["phot"][2]) - 1.0) > 1e-3


def test_eline_scaling_acts_on_lines_observation_only(csp):
    spec, phot, lines = _obs()
    model = _model(csp, (spec, phot, lines), extra_init={"eline_scaling": jnp.array([1.0])})
    a = model.predict(_theta(model, eline_scaling=[1.0]))
    b = model.predict(_theta(model, eline_scaling=[0.5]))
    np.testing.assert_allclose(np.asarray(b["lines"]), 0.5 * np.asarray(a["lines"]), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(b["spec"]), np.asarray(a["spec"]))
    np.testing.assert_array_equal(np.asarray(b["phot"]), np.asarray(a["phot"]))


def test_old_population_has_no_lines(built, csp):
    """Lines come from young stars only: zeroing the SFR at the three youngest nodes
    (0, 10, 100 Myr) removes the line flux (the nebular grid ends at ~20 Myr)."""
    model, spec, *_, th, _ = built
    mt = _csp_theta(model, th)
    F_young = np.asarray(csp.predict_line_fluxes(mt))
    mt["sfh"] = mt["sfh"] * jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    F_old = np.asarray(csp.predict_line_fluxes(mt))
    assert F_young.max() > 0.0
    assert F_old.max() < 1e-12 * F_young.max()


def test_line_wavelength_without_grid_line_raises(csp):
    spec, phot, lines = _obs(lines_wave=[5000.0])          # no grid line within 1 A
    lines.line_names = ["not-a-line"]
    model = _model(csp, (spec, phot, lines))
    with pytest.raises(ValueError, match="wavelength matching failed.*'not-a-line'"):
        model.predict(_theta(model))


def test_line_wavelength_without_grid_line_raises_unnamed(csp):
    """Regression: a Lines without line_names (line_names=None) used to crash with a
    TypeError while formatting this message (csp.py _neb_cube_rows_for)."""
    spec, phot, lines = _obs(lines_wave=[5000.0])
    assert lines.line_names is None
    model = _model(csp, (spec, phot, lines))
    with pytest.raises(ValueError, match=r"wavelength matching failed: observed line '\?' at 5000.00 A"):
        model.predict(_theta(model))


def test_lines_observation_without_nebular_model_raises():
    csp0 = _csp(add_neb=False)
    spec, phot, lines = _obs()
    model = _model(csp0, (spec, phot, lines))
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th["Z"] = jnp.array([-2.0])
    with pytest.raises(ValueError, match="add_neb=True"):
        model.predict(th)


def test_jit_vmap_and_gradients_through_the_line_path(built, csp):
    model, spec, phot, lines, th, pred = built
    jitted = model.predict_jit(th)
    for k in pred:
        np.testing.assert_allclose(np.asarray(jitted[k]), np.asarray(pred[k]), rtol=1e-6)

    us = jnp.array([-2.9, -2.23, -1.7])
    batch = {k: jnp.stack([v] * 3) for k, v in th.items()}
    batch["gas_logu"] = us[:, None]
    vm = model.predict_vmap(batch)
    for i, u in enumerate(us):
        one = model.predict(dict(th, gas_logu=jnp.array([u])))
        np.testing.assert_allclose(np.asarray(vm["lines"][i]), np.asarray(one["lines"]), rtol=1e-10)

    # Lines are float64 end to end: AD matches central differences away from grid nodes
    for key, x0 in (("gas_logu", -2.23), ("gas_logz", -0.37)):
        f = lambda x, key=key: model.predict(dict(th, **{key: jnp.array([x])}))["lines"]
        ad = np.asarray(jax.jacfwd(f)(jnp.asarray(x0)))
        h = 1e-5
        fd = (np.asarray(f(x0 + h)) - np.asarray(f(x0 - h))) / (2 * h)
        np.testing.assert_allclose(ad, fd, rtol=1e-6)

    # every observation is differentiable through the lines; d/dlogmass = ln10 x value
    def total(x):
        out = model.predict(dict(th, logmass=jnp.array([x])))
        return jnp.stack([jnp.sum(out["lines"]), jnp.sum(out["spec"]),
                          jnp.sum(out["phot"]).astype(jnp.float64)])
    x0 = float(th["logmass"][0])
    g = np.asarray(jax.jacfwd(total)(jnp.asarray(x0)))
    np.testing.assert_allclose(g, np.log(10.0) * np.asarray(total(x0)), rtol=1e-5)
    gu = np.asarray(jax.jacfwd(lambda u: jnp.stack([
        jnp.sum(model.predict(dict(th, gas_logu=jnp.array([u])))["spec"]),
        jnp.sum(model.predict(dict(th, gas_logu=jnp.array([u])))["phot"]).astype(jnp.float64)
    ]))(jnp.asarray(-2.23)))
    assert np.all(np.isfinite(gu)) and np.all(gu != 0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-ra"]))
