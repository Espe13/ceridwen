"""A sampled redshift with a Spectrum observation: SedModel builds the spectral
projector for the zred prior range, predict() re-projects at theta['zred'], and
the result equals a model built at that fixed redshift."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridfixture import find_test_grid                                   # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology                # noqa: E402
from ceridwen.broadening import Kinematics, Instrument                     # noqa: E402
from ceridwen.observation import Photometry, Spectrum                      # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh                # noqa: E402
from ceridwen.sampler.priors import Uniform, Normal                        # noqa: E402

Z_LO, Z_HI, Z_INIT = 0.7, 0.9, 0.8
N_TIME = 5
FILTERS = ["sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE = np.linspace(7000.0, 12000.0, 800)      # observed frame, A


def _csp(add_neb):
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    if add_neb and not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY grids)")
    ssp = SSPData.load(str(path))
    cosmo = Cosmology.planck18()
    # grid valid at every z in the range: oldest node = age(Z_HI)
    kw = dict(lookback_time=jnp.linspace(0.0, float(cosmo.age(Z_HI)), N_TIME),
              zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
              add_neb=add_neb, add_igm=False, verbose=False, cosmo=cosmo)
    if add_neb:
        kw["sps_home"] = os.environ["SPS_HOME"]
    return CSPBasis(ssp, **kw)


def _obs():
    n = SPEC_WAVE.size
    spec = Spectrum(wavelength=SPEC_WAVE, flux=np.full(n, 1e-18), uncertainty=np.full(n, 1e-19),
                    instrument=Instrument.R_fwhm(1000.0), name="spec")
    phot = Photometry(filters=FILTERS, flux=np.full(4, 1e-9), uncertainty=np.full(4, 1e-10),
                      name="phot")
    return spec, phot


def _model(csp, obs, *, zred_prior=None, zred_fixed=None, extra_priors=None):
    t = np.array(csp.sfh_times)

    def _sfh(th, _t=t):
        return logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)
    init = {"logsfr_ratios": jnp.zeros(N_TIME - 1), "logmass": jnp.array([10.0])}
    priors = dict(extra_priors or {})
    kw = {}
    if zred_fixed is not None:
        kw["zred"] = zred_fixed
    else:
        init["zred"] = jnp.array([Z_INIT])
        if zred_prior is not None:
            priors["zred"] = zred_prior
    return SedModel(csp, obs, priors=priors, transforms={"sfh": _sfh},
                    free_param_init=init, kinematics=Kinematics(sigma_gal=200.0, sigma_gas=80.0),
                    **kw)


def _theta(model, z=None):
    th = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    if "Z" in th:
        th["Z"] = jnp.array([-2.0])
    if z is not None:
        th["zred"] = jnp.array([z])
    return th


@pytest.fixture(scope="module")
def csp():
    return _csp(False)


@pytest.fixture(scope="module")
def csp_neb():
    return _csp(True)


def test_free_z_spectrum_model_builds_and_matches_fixed_z(csp):
    spec, phot = _obs()
    with pytest.warns(UserWarning):          # unpriored sampled parameters (logsfr_ratios, ...)
        free = _model(csp, [spec, phot], zred_prior=Uniform(low=Z_LO, high=Z_HI))
    assert free.zred_is_free
    assert spec._proj.free_z and spec._proj.zred_range == (Z_LO, Z_HI)
    assert spec._proj.opz_ref == 1.0 + Z_INIT
    assert phot.free_z
    for z in (Z_INIT, 0.74, 0.88):
        spec_f, phot_f = _obs()
        with pytest.warns(UserWarning):
            fixed = _model(csp, [spec_f, phot_f], zred_fixed=z)
        a = free.predict(_theta(free, z))
        b = fixed.predict(_theta(fixed))
        # at the reference redshift the free-z projector shares its log-grid nodes with the
        # fixed-z one (exact); elsewhere the two grids differ in node phase and the linear
        # interpolation of the model onto them differs at the per-mille level
        for k in ("spec", "phot"):
            tol = 1e-9 if (z == Z_INIT and k == "spec") else 1e-2
            ra = np.asarray(a[k]); rb = np.asarray(b[k])
            assert np.all(np.isfinite(ra))
            assert np.max(np.abs(ra - rb)) < tol * np.max(np.abs(rb)), (k, z)


def test_free_z_spectrum_with_lines(csp_neb):
    spec, phot = _obs()
    with pytest.warns(UserWarning):
        free = _model(csp_neb, [spec, phot], zred_prior=Uniform(low=Z_LO, high=Z_HI))
    assert spec._proj.line_idx.size > 0
    for z in (Z_INIT, 0.86):
        spec_f, phot_f = _obs()
        with pytest.warns(UserWarning):
            fixed = _model(csp_neb, [spec_f, phot_f], zred_fixed=z)
        ra = np.asarray(free.predict(_theta(free, z))["spec"])
        rb = np.asarray(fixed.predict(_theta(fixed))["spec"])
        tol = 1e-9 if z == Z_INIT else 1e-2
        assert np.max(np.abs(ra - rb)) < tol * np.max(np.abs(rb)), z


def test_free_z_spectrum_needs_a_range(csp):
    spec, phot = _obs()
    with pytest.raises(ValueError, match="zred_range"):
        with pytest.warns(UserWarning):
            _model(csp, [spec, phot], zred_prior=Normal(mean=Z_INIT, sigma=0.02))
    spec2 = Spectrum(wavelength=SPEC_WAVE, flux=np.full(SPEC_WAVE.size, 1e-18),
                     uncertainty=np.full(SPEC_WAVE.size, 1e-19),
                     instrument=Instrument.R_fwhm(1000.0), name="spec",
                     zred_range=(Z_LO, Z_HI))
    with pytest.warns(UserWarning):
        m = _model(csp, [spec2, phot], zred_prior=Normal(mean=Z_INIT, sigma=0.02))
    assert spec2._proj.free_z and spec2._proj.zred_range == (Z_LO, Z_HI)
    assert np.all(np.isfinite(np.asarray(m.predict(_theta(m, 0.81))["spec"])))


def test_free_z_spectrum_jit_and_grad(csp):
    spec, phot = _obs()
    with pytest.warns(UserWarning):
        free = _model(csp, [spec, phot], zred_prior=Uniform(low=Z_LO, high=Z_HI))
    th = _theta(free, Z_INIT)
    pred = free.predict_jit(th)
    assert np.all(np.isfinite(np.asarray(pred["spec"])))

    def loss(z):
        t = dict(th, zred=z)
        return jnp.sum((free.predict(t)["spec"] / 1e-18) ** 2)

    z0 = jnp.array([0.81])
    g = float(jax.grad(loss)(z0)[0])
    eps = 1e-5
    fd = float((loss(z0 + eps) - loss(z0 - eps)) / (2 * eps))
    assert np.isfinite(g) and np.isclose(g, fd, rtol=1e-3)
