"""Emission lines in photometry (ceridwen.observation.line_photometry): each line's flux through
each filter, a delta line or the exact integral of a Gaussian against the piecewise-linear
filter, and the tables for a sampled redshift or width."""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import require_test_grid                                # noqa: E402

from ceridwen.observation import Photometry                                # noqa: E402
from ceridwen.observation.line_photometry import (                         # noqa: E402
    CKMS, line_weights, LineWeightTable)

BANDS = ["sdss_g0", "sdss_r0", "jwst_f115w", "jwst_f410m"]


@pytest.fixture(scope="module")
def fs():
    return Photometry(filters=BANDS, flux=np.zeros(4), uncertainty=np.ones(4), name="p").filterset


@pytest.fixture(scope="module")
def lams(fs):
    rng = np.random.default_rng(1)
    return np.concatenate([rng.uniform(t[0][0] - 40.0, t[0][-1] + 40.0, 25)
                           for t in fs._flux_tables()])


def _brute(fs, lam0, sig):
    """int g N on a sigma/2000 grid, g = R lambda / ab linear between the filter's knots."""
    out = []
    for lam, g, *_ in fs._flux_tables():
        x = np.arange(lam0 - 10 * sig, lam0 + 10 * sig, sig / 2000.0)
        gx = np.interp(x, lam, g, left=0.0, right=0.0)
        out.append(np.trapezoid(gx * np.exp(-0.5 * ((x - lam0) / sig) ** 2), x)
                   / (np.sqrt(2.0 * np.pi) * sig))
    return np.array(out)


def test_delta_line_is_the_filter_at_the_line(fs, lams):
    E = line_weights(fs, lams, 0.0)
    for j, (lam, g, *_) in enumerate(fs._flux_tables()):
        np.testing.assert_array_equal(E[j], np.interp(lams, lam, g, left=0.0, right=0.0))


@pytest.mark.parametrize("skms", [50.0, 300.0, 1500.0])
def test_gaussian_line_is_the_exact_integral(fs, lams, skms):
    sig = lams * skms / CKMS
    E = line_weights(fs, lams, sig)
    B = np.stack([_brute(fs, l, s) for l, s in zip(lams, sig)], axis=1)
    m = B > 1e-4 * B.max()
    np.testing.assert_allclose(E[m], B[m], rtol=1e-7)


def test_line_flux_is_conserved_inside_a_band(fs):
    """Away from the edges a broad line gives the same maggies as a delta line (T flat)."""
    lam, g, *_ = fs._flux_tables()[1]
    lam0 = np.array([0.5 * (lam[0] + lam[-1])])
    d, w = line_weights(fs, lam0, 0.0)[1], line_weights(fs, lam0, lam0 * 100.0 / CKMS)[1]
    assert abs(w[0] / d[0] - 1.0) < 1e-3


def test_tables(fs, lams):
    T0 = LineWeightTable(fs, sigma_kms=0.0)
    np.testing.assert_allclose(np.asarray(T0(jnp.asarray(lams))), line_weights(fs, lams, 0.0),
                               rtol=1e-11, atol=0.0)          # knots recomputed: exp rounding
    Tf = LineWeightTable(fs, sigma_kms=300.0)
    E = line_weights(fs, lams, lams * 300.0 / CKMS)
    np.testing.assert_allclose(np.asarray(Tf(jnp.asarray(lams))), E, rtol=0.0,
                               atol=1e-5 * float(E.max()))
    Ts = LineWeightTable(fs, sigma_max_kms=2000.0)
    Tl = LineWeightTable(fs, lam0=lams, sigma_max_kms=2000.0)
    for skms in (40.0, 300.0, 1200.0):
        E = line_weights(fs, lams, lams * skms / CKMS)
        tol = 3e-5 * E.max(axis=1, keepdims=True)        # of the band's response (edges)
        Et = np.asarray(jax.jit(lambda l, v: Ts(l, v))(jnp.asarray(lams), jnp.asarray(skms)))
        assert np.all(np.abs(Et - E) <= tol), skms
        assert np.all(np.abs(np.asarray(Tl(sigma_kms=jnp.asarray(skms))) - E) <= tol), skms


# ---- through the model -----------------------------------------------------------------

@pytest.fixture(scope="module")
def csp():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    from ceridwen import SSPData, CSPBasis, Cosmology
    return CSPBasis(SSPData.load(str(require_test_grid())), lookback_time=jnp.linspace(0, 0.5, 4),
                    zh_const=True, add_neb=True, add_igm=True, verbose=False,
                    cosmo=Cosmology.planck18())


def _model(csp, free_z, gas, broaden, zred=0.3):
    from ceridwen import SedModel, Kinematics
    phot = Photometry(filters=["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"], flux=np.ones(4),
                      uncertainty=np.ones(4), name="phot")
    kw = {"free_param_init": {"zred": jnp.array([zred])}} if free_z else {"zred": zred}
    if isinstance(gas, str):
        kw.setdefault("free_param_init", {})[gas] = jnp.array([120.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ceridwen.priors import Uniform
        pri = {"zred": Uniform(low=0.1, high=0.5)} if free_z else {}
        if isinstance(gas, str):
            pri[gas] = Uniform(low=10.0, high=500.0)
        return SedModel(csp, [phot], priors=pri, broaden_photometry=broaden,
                        kinematics=Kinematics(sigma_gal=150.0, sigma_gas=gas, sigma_max=600.0),
                        **kw)


@pytest.mark.parametrize("broaden", [False, True])
def test_free_redshift_equals_fixed_redshift_at_that_redshift(csp, broaden):
    fixed, free = _model(csp, False, 120.0, broaden), _model(csp, True, 120.0, broaden)
    th = {k: jnp.asarray(v) for k, v in fixed.theta_init.items()}
    a = np.asarray(fixed.predict(th)["phot"], float)
    b = np.asarray(free.predict({**th, "zred": jnp.array([0.3])})["phot"], float)
    # the line term alone: the same at a sampled redshift as at the fixed one
    mt = dict(fixed.apply_transforms(th), zred=jnp.array([0.3]))
    F = csp.predict_line_fluxes(mt, for_spectrum=True)
    la = fixed.obs_dict["phot"]._line_photometry
    lb = free.obs_dict["phot"]._line_photometry
    np.testing.assert_allclose(np.asarray(lb(F, opz=1.3), float), np.asarray(la(F), float),
                               rtol=1e-6)
    # the whole photometry: without broadening to float32 rounding; with it, to the free-z
    # sigma_gal kernel, built on the whole 912-25000 A window (a different grid phase from the
    # fixed-z one: 1.5e-4 in sdss_z0 for the stellar continuum alone, as before 1.0.14)
    np.testing.assert_allclose(b, a, rtol=3e-4 if broaden else 2e-6)


def test_sampled_gas_width_equals_the_fixed_one(csp):
    fixed, sampled = _model(csp, False, 120.0, True), _model(csp, False, "sigma_gas", True)
    th = {k: jnp.asarray(v) for k, v in fixed.theta_init.items()}
    a = np.asarray(fixed.predict(th)["phot"], float)
    b = np.asarray(sampled.predict({**th, "sigma_gas": jnp.array([120.0])})["phot"], float)
    np.testing.assert_allclose(b, a, rtol=1e-5)


def test_marginalised_lines_reach_the_photometry_at_the_sampled_width():
    """marginalize_elines with Photometry, broaden_photometry and a sampled sigma_gas: the fitted
    lines' photometric design columns are the photometry's own line weights at the sampled
    width, and at a value of sigma_gas they equal the fixed-width model's columns."""
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    from ceridwen import SSPData, CSPBasis, Cosmology, SedModel, Kinematics, Instrument
    from ceridwen.observation import Spectrum
    from ceridwen.priors import Uniform
    csp = CSPBasis(SSPData.load(str(require_test_grid())), lookback_time=jnp.linspace(0, 0.5, 4),
                   zh_const=True, add_neb=True, add_igm=True, verbose=False,
                   cosmo=Cosmology.planck18(), fixed={"gas_logu": -2.5, "gas_logz": -0.3})
    w = np.linspace(4700.0, 6800.0, 900) * 1.3
    cols = {}
    for gas in (120.0, "sigma_gas"):
        spec = Spectrum(wavelength=w, flux=np.full(w.size, 1e-18), uncertainty=np.full(w.size, 1e-19),
                        instrument=Instrument.R_fwhm(1000.0), marginalize_elines=True, name="spec")
        phot = Photometry(filters=["sdss_g0", "sdss_r0", "sdss_i0"], flux=np.ones(3),
                          uncertainty=np.ones(3), name="phot")
        kw = {}
        if isinstance(gas, str):
            kw = dict(free_param_init={gas: jnp.array([120.0])},
                      priors={gas: Uniform(low=10.0, high=500.0)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SedModel(csp, [spec, phot], zred=0.3, broaden_photometry=True,
                         kinematics=Kinematics(sigma_gal=150.0, sigma_gas=gas, sigma_max=600.0),
                         **kw)
        th = {k: jnp.asarray(v) for k, v in m.theta_init.items()}
        _, aux = m.predict_with_elines(th)
        cols[gas] = np.asarray(aux["cols"]["phot"], dtype=np.float64)
        if isinstance(gas, str):
            lp = m.obs_dict["phot"]._line_photometry
            want = np.asarray(lp.weights(sigma_kms=jnp.asarray(120.0)))[:, m._eline_system.fit_rows]
            np.testing.assert_array_equal(cols[gas], want)
    a, b = cols[120.0], cols["sigma_gas"]
    assert np.all(np.abs(b - a) <= 3e-5 * np.abs(a).max(axis=1, keepdims=True))
