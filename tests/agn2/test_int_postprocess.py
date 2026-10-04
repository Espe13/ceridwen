"""PostProcess with the SKIRTOR torus (session INT): ``extras["agn"]`` carries CIGALE's ``agn.*``
infos (``cigale_<name>``, per formed mass x 10**logmass like every other extras entry) and
``frac_agn_cigale`` = P / (P + L_dust,gal), CIGALE's fracAGN, which is a ratio (not mass-scaled).

V-INT-PP-1  "cigale": frac_agn_cigale == fagn (CIGALE's definition of fagn), rel 1e-10.
V-INT-PP-2  "corrected": frac_agn_cigale == P / (P + L_dust_gal) from the (mass-scaled) extras,
            rel 1e-12; P = cigale_total_dust_luminosity = fagn x L_bol_star to the model grid's
            quadrature (rel 1e-5, tests/regression/capture_baseline.py _agn2_baselines).
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import require_test_grid  # noqa: E402
from ceridwen import CSPBasis, Cosmology, SSPData, SedModel  # noqa: E402
from ceridwen.agn import AGNConfig  # noqa: E402
from ceridwen.agn.skirtor import default_skirtor_path  # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh  # noqa: E402
from ceridwen.observation import Photometry  # noqa: E402

FAGN = 0.35


def _model(mode):
    if not pathlib.Path(default_skirtor_path()).exists():
        pytest.skip(f"MISSING {default_skirtor_path()}: SKIRTOR post-processing did NOT run")
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(require_test_grid())),
                   lookback_time=jnp.array([0.0, 0.01, 0.1, 0.5, 1.5]), cosmo=cosmo,
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=False, add_igm=True, verbose=False,
                   agn=AGNConfig(agn_mode=mode, torus_model="skirtor16", polar_dust=True,
                                 lines=False))
    t = np.array(csp.sfh_times)
    init = {"logsfr_ratios": jnp.zeros(len(t) - 1), "logmass": jnp.array([9.5]),
            "diffuse_tau_kc": jnp.array([0.5]), "skirtor_i": jnp.array([40.0])}
    obs = [Photometry(filters=["jwst_f150w", "jwst_f444w", "jwst_f770w"], flux=np.ones(3) * 1e-9,
                      uncertainty=np.ones(3) * 1e-10, name="phot")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SedModel(csp, obs, free_param_init=init, zred=2.0,
                        transforms={"sfh": lambda th, _t=t: logsfr_ratios_to_sfh(
                            th["logsfr_ratios"], sfh_times_yr=_t),
                            "fagn": lambda th: jnp.array([FAGN]),
                            "polar_ebv": lambda th: jnp.array([0.2])})


def _extras(mode):
    from ceridwen.postprocess import PostProcess
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agn"))
    from test_integration import _fake_result
    model = _model(mode)
    out = PostProcess(model, _fake_result(model), n_samples=None).run()
    return model, out["extras"]["agn"]


def test_cigale_frac_is_fagn():
    _m, ex = _extras("cigale")
    assert "frac_agn_cigale" in ex and "cigale_accretion_power" in ex
    np.testing.assert_allclose(ex["frac_agn_cigale"], FAGN, rtol=1e-10)


def test_corrected_frac_is_cigale_definition():
    _m, ex = _extras("corrected")
    P = ex["cigale_total_dust_luminosity"]
    np.testing.assert_allclose(ex["frac_agn_cigale"], P / (P + ex["L_dust_gal"]), rtol=1e-12)
    np.testing.assert_allclose(P, FAGN * ex["L_bol_star"], rtol=1e-5)
    assert np.all((ex["frac_agn_cigale"] > 0) & (ex["frac_agn_cigale"] < 1))
