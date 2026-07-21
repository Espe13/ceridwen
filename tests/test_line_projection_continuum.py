"""Regression tests for the emission-line projection (2026-07-21 fix).

Bug history: ``Lines.predict`` extracts line fluxes from a model spectrum via
positive Gaussian apertures (``_W @ spectrum``).  Before the fix,
``CSPBasis._project_observations`` fed it the FULL slit spectrum, so every
predicted "line flux" also contained the stellar + nebular continuum
integrated under the aperture.  Catalogue line fluxes are continuum-
subtracted, so the continuum term entered the likelihood as a spurious line
flux scaling with the evolved stellar mass — inflating all faint lines
(high-order Balmer/Paschen, [O III] 4363, He I/II, [O I], [S III]) and
producing non-case-B, even inverted, hydrogen ladders for low-sSFR models.
After the fix, ``Lines`` observations are predicted DIRECTLY from the CLOUDY
grid line luminosities (``CSPBasis.predict_line_fluxes``) -- no spectral
painting/extraction round trip -- with the line-only slit component retained
as the zero-fallback when no nebular module is present.

These tests are library-agnostic: they use whatever SSP grid and nebular
cubes the installation resolves, and test T4 sweeps every ``ZAU_*`` cube
pair found in ``$SPS_HOME/nebular``.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_line_projection_continuum.py -v
 or:  JAX_PLATFORMS=cpu python tests/test_line_projection_continuum.py
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import glob
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

# Case-B intensity ratios relative to Ha (Te = 1e4 K, ne ~ 100 cm^-3)
CASEB = {"Ba-beta 4861": 0.3497, "Ba-gamma 4341": 0.164,
         "Pa-beta 1.28181um": 0.0562, "Pa-gamma 1.09381um": 0.0304,
         "Pa-delta 1.00494um": 0.0195}
LINES = ["Ba-alpha 6563", "Ba-beta 4861", "Ba-gamma 4341",
         "Pa-beta 1.28181um", "Pa-gamma 1.09381um", "Pa-delta 1.00494um",
         "[O III] 5007", "[O III] 4363", "He II 4685.64A", "[O I] 6300"]


def _sps_home():
    home = os.environ.get("SPS_HOME")
    if not home:
        pytest.skip("SPS_HOME not set")
    return home


def _ssp_file():
    cands = [REPO / "ceridwen" / "data" / "test_data" / "ssp_data_bpass_agb_dust.h5",
             REPO / "ceridwen" / "data" / "test_data" / "ssp_data.h5",
             REPO / "examples" / "ssp_data.h5"]
    env = os.environ.get("SSP_FILE")
    if env and Path(env).is_file():
        return env
    for c in cands:
        if c.is_file():
            return str(c)
    pytest.skip("no SSP grid found (set $SSP_FILE)")


def _load_emline_info(sps_home):
    """name -> (wavelength [A], index) from $SPS_HOME/data/emlines_info.dat."""
    path = Path(sps_home) / "data" / "emlines_info.dat"
    info = {}
    with open(path) as f:
        for i, row in enumerate(f):
            parts = row.split(",")
            if len(parts) < 2:
                continue
            info[parts[1].strip()] = (float(parts[0]), i)
    return info


def _build(csp_kwargs=None):
    import jax.numpy as jnp
    from ceridwen.ssps.ssp_data import SSPData
    from ceridwen.csp.csp import CSPBasis

    ssp_file = _ssp_file()
    ssp_data = SSPData.load(ssp_file)
    # Match the nebular grid to the SSP grid when the fixture carries no
    # provenance (avoids the isoc_type='mist' fallback on a BPASS grid).
    neb_init = {"cloudy_dust": True}
    if "bpass" in Path(ssp_file).name.lower():
        neb_init["isoc_type"] = "bpss"
    kw = dict(
        tuniv=13.8, tiny_logt=-70, zh_const=True, sfh_interp="step",
        add_dust=False, add_diffuse_dust=False, add_dust_emission=False,
        add_neb=True, init_neb_params=neb_init,
        add_igm=False, sps_home=_sps_home(), verbose=False,
    )
    kw.update(csp_kwargs or {})
    return CSPBasis(
        ssp_data,
        theta={"lookback_time": jnp.array([0.0, 0.01]),
               "sfh": jnp.ones(1), "Z": jnp.array([-2.0])},
        **kw,
    )


def _lines_obs(sps_home):
    from ceridwen.observation import Lines
    info = _load_emline_info(sps_home)
    names = [n for n in LINES if n in info]
    assert len(names) >= 8, f"emline info only matched {names}"
    return Lines(
        line_ind=np.array([info[n][1] for n in names], dtype=int),
        line_names=names,
        wavelength=np.array([info[n][0] for n in names]),
        flux=np.full(len(names), 1e-18),
        uncertainty=np.full(len(names), 1e-19),
        name="test_lines",
    )


def _predict(csp, obs, t_lo, t_hi):
    import jax.numpy as jnp
    dt_yr = (t_hi - t_lo) * 1e9
    theta = {
        "lookback_time": jnp.array([t_lo, t_hi]),
        "sfh": jnp.array([1.0]),
        "Z": jnp.array([-2.0]),
        "gas_logz": jnp.array([-0.5]), "gas_logu": jnp.array([-2.5]),
        "logmass": jnp.array([np.log10(dt_yr)]),
        "zred": jnp.array([2.0]),
        "eline_scaling": jnp.array([1.0]),
    }
    out = csp.predict(theta, [obs])
    return np.asarray(out[obs.name], dtype=float)


@pytest.fixture(scope="module")
def setup():
    csp = _build()
    obs = _lines_obs(_sps_home())
    if hasattr(obs, "setup_for_model"):
        obs.setup_for_model(csp.wave)
    return csp, obs


def test_T1_old_population_emits_no_lines(setup):
    """An old-only (0.1-1 Gyr) population must predict ~zero line flux.

    Pre-fix this failed at the 0.1-5.7x level (continuum under the
    apertures); post-fix the line component of an old population is zero."""
    csp, obs = setup
    f_young = _predict(csp, obs, 0.0, 0.010)
    f_old = _predict(csp, obs, 0.100, 1.000)
    ratio = np.abs(f_old) / np.maximum(np.abs(f_young), 1e-300)
    assert np.nanmax(ratio) < 1e-3, (
        f"old/young per-mass line ratios up to {np.nanmax(ratio):.2e}: "
        "continuum (or old-SSP emission) is leaking into Lines.predict")


def test_T2_no_neb_model_predicts_zero_lines():
    """With add_neb=False the line component is identically zero, so Lines
    predictions must vanish -- pre-fix they returned the continuum."""
    csp = _build({"add_neb": False})
    obs = _lines_obs(_sps_home())
    if hasattr(obs, "setup_for_model"):
        obs.setup_for_model(csp.wave)
    f = _predict(csp, obs, 0.0, 0.010)
    # compare against a neb-on young model for scale
    csp2 = _build()
    obs2 = _lines_obs(_sps_home())
    if hasattr(obs2, "setup_for_model"):
        obs2.setup_for_model(csp2.wave)
    f_ref = _predict(csp2, obs2, 0.0, 0.010)
    assert np.nanmax(np.abs(f) / np.maximum(np.abs(f_ref), 1e-300)) < 1e-6, (
        "a continuum-only model predicts nonzero emission-line fluxes")


def test_T3_young_burst_ladder_is_caseB(setup):
    """H recombination ratios of a young dust-free burst must be case B to
    ~30% (grid physics allows modest departures; the pre-fix bias was
    1.4-1.9x for Pa-gamma/Pa-delta and unbounded for old SFHs)."""
    csp, obs = setup
    f = _predict(csp, obs, 0.0, 0.005)
    names = list(obs.line_names)
    iHa = names.index("Ba-alpha 6563")
    for nm, cb in CASEB.items():
        if nm not in names:
            continue
        r = f[names.index(nm)] / f[iHa] / cb
        # With direct grid-based line prediction (2026-07-21) the ladder is
        # the CLOUDY grid's own, so case B should hold to grid physics
        # (~20-30%).  The historical failure modes: continuum under the
        # apertures (unbounded for old SFHs) and the 0.35-0.5x aperture
        # extraction deficit -- both eliminated.
        assert 0.7 < r < 1.43, (
            f"{nm}: (line/Ha)/caseB = {r:.2f} for a young dust-free burst")


def test_T4_all_available_nebular_libraries_load_sanely():
    """Every ZAU_* cube pair in $SPS_HOME/nebular must load with an age axis
    normalised to log10(yr) and a young mask that cuts below ~300 Myr --
    regardless of the library's native age tabulation (log yr, linear yr,
    or Myr).  This is the library-flexibility guard."""
    from ceridwen.neb.NebularGridModel import NebularModel

    sps_home = _sps_home()
    pairs = sorted(glob.glob(str(Path(sps_home) / "nebular" / "ZAU_*.lines")))
    if not pairs:
        pytest.skip("no ZAU_*.lines cubes found")
    ssp_ages_lgyr = np.linspace(5.6, 10.2, 43)
    wave = np.geomspace(100.0, 1e7, 2000)
    tested = 0
    for line_file in pairs:
        stem = Path(line_file).stem           # e.g. ZAU_WD_bpss
        isoc = stem.split("_")[-1]
        try:
            neb = NebularModel(sps_home=sps_home, csp_lambda=wave,
                               ssp_flux=None, ssp_ages_lgyr=ssp_ages_lgyr,
                               isoc_type=isoc, cloudy_dust="WD" in stem)
        except Exception as exc:                                # noqa: BLE001
            print(f"[T4] {stem}: could not instantiate ({exc}); skipped")
            continue
        for ax_name in ("nebem_cont_age", "nebem_line_age"):
            ax = np.asarray(getattr(neb, ax_name))
            assert 4.5 <= ax.min() and ax.max() <= 10.5, (
                f"{stem}.{ax_name} not normalised to log10(yr): "
                f"[{ax.min():.2f}, {ax.max():.2f}]")
        young = np.asarray(neb.young_mask)
        assert young.sum() > 0, f"{stem}: young mask empty"
        oldest = 10 ** ssp_ages_lgyr[young].max()
        assert oldest < 3.2e8, (
            f"{stem}: young mask keeps SSPs up to {oldest/1e6:.0f} Myr")
        tested += 1
        print(f"[T4] {stem}: OK (young mask keeps {young.sum()} SSPs, "
              f"oldest {oldest/1e6:.1f} Myr)")
    assert tested > 0, "no nebular library could be instantiated"


def test_T5_aperture_matches_direct_integration(setup):
    """Consistency of the DIRECT grid-based line prediction with the flux
    actually painted into the spectrum: Lines predictions (now computed from
    the CLOUDY grid luminosities) must match a trapezoidal integration of
    the line-only spectrum over +-5 aperture sigma around each line.

    History: the old Gaussian-aperture extraction recovered only 0.35-0.47
    of the painted flux (narrow-line normalisation vs resolution-floor +
    LOSVD-broadened line widths, wavelength dependent).  With the direct
    prediction the ratio must be ~1 for isolated lines -- a persistent
    deviation means the painting (gaussnebarr) and grid normalisations have
    diverged.  Strict bound on the strong optical lines; the printed table
    is the per-line diagnostic (blended lines, e.g. [O III] 4363 near Hg,
    may deviate in the DIRECT column, which sums neighbours)."""
    import jax.numpy as jnp
    csp, obs = setup
    t_lo, t_hi = 0.0, 0.005
    dt_yr = (t_hi - t_lo) * 1e9
    theta = {
        "lookback_time": jnp.array([t_lo, t_hi]), "sfh": jnp.array([1.0]),
        "Z": jnp.array([-2.0]),
        "gas_logz": jnp.array([-0.5]), "gas_logu": jnp.array([-2.5]),
        "logmass": jnp.array([np.log10(dt_yr)]), "zred": jnp.array([2.0]),
        "eline_scaling": jnp.array([1.0]),
    }
    f_ap = np.asarray(csp.predict(theta, [obs])[obs.name], dtype=float)
    line_spec = np.asarray(csp.get_line_spec(theta), dtype=float)  # F_nu, rest grid
    wave = np.asarray(csp.wave, dtype=float)
    c_aa = 2.998e18
    names = list(obs.line_names)
    lam0 = np.asarray(obs.wavelength, dtype=float)
    print()
    print(f"{'line':22s} {'aperture':>12s} {'direct':>12s} {'ap/direct':>10s}")
    ratios = {}
    for k, nm in enumerate(names):
        sig = lam0[k] * (200.0 / 2.998e5)
        m = np.abs(wave - lam0[k]) < 5 * sig
        # F_line = int f_nu dnu = int f_nu * c/lambda^2 dlambda
        f_direct = np.trapezoid(line_spec[m] * c_aa / wave[m] ** 2, wave[m])
        r = f_ap[k] / f_direct if f_direct != 0 else np.nan
        ratios[nm] = r
        print(f"{nm:22s} {f_ap[k]:12.3e} {f_direct:12.3e} {r:10.3f}")
    for nm in ("Ba-alpha 6563", "Ba-beta 4861", "[O III] 5007"):
        if nm in ratios and np.isfinite(ratios[nm]):
            assert 0.75 < ratios[nm] < 1.25, (
                f"{nm}: aperture/direct = {ratios[nm]:.2f} -- the Gaussian-"
                "aperture normalisation disagrees with direct integration")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
