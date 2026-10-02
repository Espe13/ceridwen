"""AGN lines on a real nebular CSPBasis through a local stub of the hook (AGN_PLAN §3: "LINES
writes against the interface with a local stub of hook.py"; session INT does the real wiring).

``_predict_with_agn`` adds the AGN-line terms to ``SedModel.predict`` exactly as the handoff in
REPORT_LINES.md tells INT to: Spectrum += ``_proj.agn_lines``, Photometry += the
mode's photometry, Lines += ``agn_line_matrix @ F``. On it:

B3  agn_elum = 0 leaves every prediction byte-identical to the model without AGN (both modes,
    also with C1 ``wavelength_support=True``).
    One flux everywhere (corrected mode): the AGN H beta a Lines observation reports equals the
    frequency integral of the AGN H beta painted into the Spectrum, and the photometric excess.
B8  jit, vmap (64, 100) == loop, finite gradients of the full log-likelihood in every free
    parameter including agn_elum / agn_eline_sigma, matching central finite differences.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import test_emission_lines as T                                           # noqa: E402

from ceridwen import SSPData, CSPBasis, Cosmology                         # noqa: E402
from ceridwen.agn.lines import AGNLines, line_photometry_delta            # noqa: E402
from ceridwen.broadening import C_AA_S, agn_line_profiles_on_grid         # noqa: E402

W_PROD = (64, 100)
AGN_KEYS = ("agn_elum", "agn_eline_sigma")


def _csp(support=False):
    path = T.find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid (see tests/_gridfixture.py)")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME not set (CLOUDY nebular grids, FSPS line list)")
    cosmo = Cosmology.planck18()
    csp = CSPBasis(SSPData.load(str(path)),
                   lookback_time=jnp.array([0.0, 0.01, 0.1, 1.0, float(cosmo.age(T.ZRED))]),
                   zh_const=True, sfh_interp="step", add_dust=False, add_diffuse_dust=True,
                   add_neb=True, add_igm=False, verbose=False, cosmo=cosmo,
                   sps_home=os.environ["SPS_HOME"], wavelength_support=support)
    csp.register_known_theta_keys(AGN_KEYS)
    return csp


def _build(csp, mode):
    spec, phot, lines = T._obs()
    model = T._model(csp, (spec, phot, lines))
    agn = AGNLines(mode=mode)
    spec._proj.setup_agn_lines(agn.line_wave, mode=mode, sf_line_wave_rest=agn.emlines_wave,
                               agn_sf_rows=agn.fsps_index, pixel_mask=np.asarray(spec.mask))
    return SimpleModel(csp, model, agn, mode, spec, phot, lines)


class SimpleModel:
    def __init__(self, csp, model, agn, mode, spec, phot, lines):
        self.csp, self.model, self.agn, self.mode = csp, model, agn, mode
        self.spec, self.phot, self.lines = spec, phot, lines
        self.M = jnp.asarray(lines.agn_line_matrix(agn))

    def predict(self, th, a):
        """``SedModel.predict(th)`` plus the AGN lines of ``a`` (agn_elum, agn_eline_sigma)."""
        out = dict(self.model.predict(th))
        mt = T._csp_theta(self.model, th)
        thl = dict(mt, **a)
        opz = 1.0 + T.ZRED
        ff = self.csp._flux_factor(mt)
        agn = self.agn
        F = agn.observed_fluxes(thl, ff / opz)                     # erg/s/cm^2, mass included
        s = agn.sigma_kms(thl)
        sp = self.spec._proj.agn_lines(F, s, thl)
        out[self.spec.name] = out[self.spec.name] + sp.astype(out[self.spec.name].dtype)
        if self.mode == "corrected":
            prof = agn_line_profiles_on_grid(self.csp.wave, agn.line_wave, s,
                                             self.csp.neb.res_floor_factor)
            mass = 10.0 ** jnp.ravel(mt["logmass"])[0]
            ph = (self.phot._T.astype(prof.dtype) @ prof) @ (agn.line_luminosity(thl) * mass * ff)
        else:
            ph = line_photometry_delta(self.phot.filters, agn.line_wave * opz, F)
        out[self.phot.name] = out[self.phot.name] + ph.astype(out[self.phot.name].dtype)
        scale = jnp.ravel(thl["eline_scaling"])[0] if "eline_scaling" in thl else 1.0
        ln = self.M @ (F * scale)
        out[self.lines.name] = out[self.lines.name] + ln.astype(out[self.lines.name].dtype)
        return out


@pytest.fixture(scope="module")
def csp():
    return _csp(False)


@pytest.mark.parametrize("mode", ["corrected", "prospector"])
@pytest.mark.parametrize("support", [False, True])
def test_B3_zero_agn_elum_is_byte_identical_to_no_agn(csp, mode, support):
    c = csp if not support else _csp(True)
    sm = _build(c, mode)
    th = T._theta(sm.model)
    base = sm.model.predict(th)
    got = sm.predict(th, {"agn_elum": jnp.array([0.0]), "agn_eline_sigma": jnp.array([220.0])})
    for k in base:
        assert got[k].dtype == base[k].dtype and got[k].tobytes() == base[k].tobytes(), k
    # and it is on: a positive agn_elum moves every observation
    on = sm.predict(th, {"agn_elum": jnp.array([1e-3]), "agn_eline_sigma": jnp.array([220.0])})
    for k in base:
        assert np.any(np.asarray(on[k]) != np.asarray(base[k])), k


def test_one_agn_flux_in_lines_spectrum_and_photometry(csp):
    """corrected mode: AGN H beta in Lines == integral over nu of the AGN H beta painted in the
    Spectrum (1e-4: the painted line sits on observed pixels at R = 1000), and the photometric
    excess of all AGN lines matches the line fluxes through the filters to 1e-2 of the brightest
    band (filter-grid sampling and line width; see the comment below)."""
    sm = _build(csp, "corrected")
    th = T._theta(sm.model)
    a = {"agn_elum": jnp.array([2e-3]), "agn_eline_sigma": jnp.array([150.0])}
    base = sm.model.predict(th)
    on = sm.predict(th, a)
    dl = np.asarray(on["lines"], np.float64) - np.asarray(base["lines"], np.float64)
    k_hb = [n for n, _ in T.LINES].index("Hbeta")
    mt = T._csp_theta(sm.model, th)
    F = np.asarray(sm.agn.observed_fluxes(dict(mt, **a), sm.csp._flux_factor(mt) / (1 + T.ZRED)))
    j_hb = sm.agn.line_names.index("Ba-beta 4861")
    assert dl[k_hb] == pytest.approx(F[j_hb], rel=1e-6)
    one = jnp.zeros(sm.agn.n_lines).at[j_hb].set(F[j_hb])
    f = np.asarray(sm.spec._proj.agn_lines(one, 150.0, dict(mt, **a)))
    nu = C_AA_S / np.asarray(T.WAVE_OBS)
    assert -np.trapezoid(f, nu) == pytest.approx(F[j_hb], rel=1e-4)
    dp = np.asarray(on["phot"], np.float64) - np.asarray(base["phot"], np.float64)
    delta = np.asarray(line_photometry_delta(sm.phot.filters, sm.agn.line_wave * (1 + T.ZRED),
                                             jnp.asarray(F)))
    # Not exact, for two reasons that hold for the nebular lines identically (same profiles,
    # same _T): Photometry._T reads the spectrum at the FilterSet grid points (dlnlam = 1e-3,
    # ~6.5 A here), which aliases a line of a few A by up to ~1 % (it does NOT shrink with a finer
    # model grid: REPORT_LINES.md), and a resolved line on a steep filter edge is not a
    # delta function ([O III] 5007 and H alpha on the i-band edges at z = 0.3: 3 % of that faint
    # band). Prospector's nebline_photometry (the "prospector" mode) is the delta function.
    np.testing.assert_allclose(dp, delta, rtol=0, atol=1e-2 * np.abs(delta).max())


def _lnlike_factory(sm):
    th0 = T._theta(sm.model)
    keys = sorted(th0)
    sizes = [int(np.size(th0[k])) for k in keys]
    base = sm.model.predict(th0)
    data = {k: np.asarray(v, np.float64) * 1.05 for k, v in base.items()}
    err = {k: 0.05 * np.abs(v).max() + 0.01 * np.abs(v) for k, v in data.items()}

    def unpack(x):
        th, i = {}, 0
        for k, n in zip(keys, sizes):
            th[k] = x[i:i + n].reshape(np.shape(th0[k]))
            i += n
        a = {"agn_elum": x[i:i + 1], "agn_eline_sigma": x[i + 1:i + 2]}
        return th, a

    def lnlike(x):
        th, a = unpack(x)
        out = sm.predict(th, a)
        return -0.5 * sum(jnp.sum(((out[k] - data[k]) / err[k]) ** 2) for k in data)

    x0 = np.concatenate([np.ravel(np.asarray(th0[k], np.float64)) for k in keys]
                        + [np.array([2e-3, 180.0])])
    return lnlike, x0, len(x0) - 2


@pytest.mark.parametrize("mode", ["corrected", "prospector"])
def test_B8_full_model_jit_vmap_grad(csp, mode):
    sm = _build(csp, mode)
    f, x0, i_agn = _lnlike_factory(sm)
    vg = jax.jit(jax.value_and_grad(f))
    v, g = vg(jnp.asarray(x0))
    g = np.asarray(g)
    assert np.isfinite(float(v)) and np.all(np.isfinite(g))
    for i, h in ((i_agn, 1e-8), (i_agn + 1, 1e-3)):
        e = np.zeros_like(x0); e[i] = h
        fd = (float(f(jnp.asarray(x0 + e))) - float(f(jnp.asarray(x0 - e)))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-4), (i, g[i], fd)
    rng = np.random.default_rng(0)
    for W in W_PROD:
        X = np.repeat(x0[None], W, axis=0)
        X[:, i_agn] = rng.uniform(1e-6, 1e-2, W)
        X[:, i_agn + 1] = rng.uniform(50.0, 500.0, W)
        bv, bg = jax.jit(jax.vmap(jax.value_and_grad(f)))(jnp.asarray(X))
        for i in (0, W // 2, W - 1):
            lv, lg = vg(jnp.asarray(X[i]))
            # the CSP runs float32 einsums whose order vmap changes (T3 uses rtol 1e-6 for them)
            assert float(bv[i]) == pytest.approx(float(lv), rel=1e-7)
            np.testing.assert_allclose(np.asarray(bg[i]), np.asarray(lg), rtol=1e-5,
                                       atol=1e-6 * np.abs(np.asarray(lg)).max())
        assert np.all(np.isfinite(np.asarray(bg)))

