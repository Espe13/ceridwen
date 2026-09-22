"""Reference dump of Prospector's emission-line marginalisation kernel for
``tests/test_eline_marginalisation.py`` (test 2a, isolated kernel).

Run in the Prospector environment, NOT with ceridwen (neither package imports the other):

    conda activate prospector
    SPS_HOME=/Users/amanda/Prospector/fsps python tests/reference/run_prospector_elines.py

Writes ``tests/reference/prospector_elines.npz``.  For every case a ``SpecModel`` gets its
cached attributes set directly (``_outwave``, ``_zred``, ``_ewave_obs``, ``_eline_lum``,
``_eline_lum_covar``, ``_speccal``) and ``cache_eline_parameters`` + ``fit_mle_elines`` are
called, exactly as ``SpecModel.predict_spec`` would.  Two attributes that the installed
Prospector (2.0a2.dev42+gff8d5e4) reads but never creates are initialised here, at runtime
only (the install is not modified): ``_speccal`` (read at sedmodel.py:658 before it is first
set at :324) and ``_eline_lum_var`` (written at :709, created nowhere).

Saved per case ``<tag>/<key>``: the kernel inputs Prospector used (``A`` = its unit Gaussians
on the fit pixels, ``delta``, ``unc``, ``alpha_breve``, ``sigma_breve``, ``linecal``) and its
outputs (``alpha_hat``, ``sigma_hat``, ``alpha_bar``, ``sigma_bar``, ``K``, ``K_flat``,
``line_spec``), plus the line names.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import prospect
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.observation import Spectrum
from prospect.sources.constants import ckms

OUT = pathlib.Path(__file__).resolve().parent / "prospector_elines.npz"
Z = 3.0


def _model(prior_width=0.0, to_fit=None, to_fix=None, to_ignore=None):
    mp = TemplateLibrary["parametric_sfh"]
    mp.update(TemplateLibrary["nebular"])
    mp["nebemlineinspec"] = dict(N=1, isfree=False, init=False)
    mp["marginalize_elines"] = dict(N=1, isfree=False, init=True)
    mp["eline_prior_width"] = dict(N=1, isfree=False, init=prior_width)
    if to_fit is not None:
        mp["elines_to_fit"] = dict(N=len(to_fit), isfree=False, init=np.array(to_fit))
    if to_fix is not None:
        mp["elines_to_fix"] = dict(N=len(to_fix), isfree=False, init=np.array(to_fix))
    if to_ignore is not None:
        mp["elines_to_ignore"] = dict(N=len(to_ignore), isfree=False, init=np.array(to_ignore))
    m = SpecModel(mp)
    m.set_parameters(m.theta)
    return m


def _grid(lo_rest, hi_rest, R, z=Z):
    """Log-uniform observed pixels at 2 px per instrumental sigma."""
    dln = 1.0 / (2.3548 * R) / 2.0
    return np.exp(np.arange(np.log(lo_rest * (1 + z)), np.log(hi_rest * (1 + z)), dln))


def _run(tag, *, lo, hi, R, prior_width, to_fit, to_fix=None, to_ignore=None, dz=0.0,
         mask_rest=None, seed=0):
    rng = np.random.default_rng(seed)
    wave = _grid(lo, hi, R)
    sig_inst = ckms / (2.3548 * R)
    out = {}
    for pw, key in ((0.0, "flat"), (prior_width, "prior")):
        m = _model(pw, to_fit, to_fix, to_ignore)
        names = m.emline_info["name"]
        ewave = m.emline_info["wave"]
        # "CLOUDY" luminosities (arbitrary, positive) and a truth that deviates from them
        lum = np.full(len(ewave), 2e-4) * (1 + 0.5 * np.sin(np.arange(len(ewave))))
        m._zred = Z
        m._eline_wave = ewave
        m._eline_lum = lum.copy()
        m._ewave_obs = (1 + Z + dz) * ewave
        m._outwave = wave
        m._ln_eline_penalty = 0.0
        m._eline_lum_mle = lum.copy()
        m._eline_lum_covar = np.diag((pw * lum) ** 2)
        m._eline_lum_var = np.zeros((len(ewave), len(ewave)))
        m._speccal = np.ones_like(wave)
        m.params["eline_delta_zred"] = np.array([dz])
        mask = np.ones(wave.size, dtype=bool)
        if mask_rest is not None:
            mask &= ~((wave > mask_rest[0] * (1 + Z)) & (wave < mask_rest[1] * (1 + Z)))
        obs = Spectrum(wavelength=wave, flux=np.ones_like(wave),
                       uncertainty=np.full(wave.size, 0.05), mask=mask,
                       resolution=np.full(wave.size, sig_inst), response=np.ones_like(wave))
        obs.rectify()
        m.cache_eline_parameters(obs)
        # data: unit continuum + fitted lines at 1.7x their luminosity + fixed lines + noise
        norm = m.flux_norm() / (1 + Z)
        fit_idx = m._elines_to_fit
        fix_idx = m._fix_eline & m._valid_eline
        g_fit = m.get_eline_gaussians(lineidx=fit_idx, wave=wave)
        cont = np.ones_like(wave)
        calibrated = cont.copy()
        if m._fix_eline_pixelmask.any():
            fixspec = m.predict_eline_spec(line_indices=fix_idx,
                                           wave=wave[m._fix_eline_pixelmask])
            calibrated[m._fix_eline_pixelmask] += fixspec.sum(axis=1)
        truth = calibrated + (g_fit * (1.7 * lum[fit_idx] * norm)).sum(axis=1)
        obs.flux = truth + 0.05 * rng.standard_normal(wave.size)
        emask = m._fit_eline_pixelmask
        A = m.get_eline_gaussians(lineidx=fit_idx, wave=wave[emask])
        linecal = norm * np.interp(m._ewave_obs[fit_idx], wave[emask], m._speccal[emask])
        line_spec = m.fit_mle_elines(obs, calibrated)
        out[key] = dict(
            A=A, delta=(obs.flux - calibrated)[emask], unc=obs.uncertainty[emask],
            alpha_breve=lum[fit_idx] * linecal,
            sigma_breve=np.diag((pw * lum[fit_idx]) ** 2) * np.outer(linecal, linecal),
            linecal=linecal,
            alpha_hat=m._eline_lum_mle[fit_idx] * linecal,
            alpha_bar=m._eline_lum[fit_idx] * linecal,
            sigma_bar=m._eline_lum_var[np.ix_(fit_idx, fit_idx)] * np.outer(linecal, linecal),
            K=np.array(m._ln_eline_penalty), line_spec=line_spec.sum(axis=1),
            names=np.array(names[fit_idx]), emask=emask, wave=wave)
    flat, prior = out["flat"], out["prior"]
    assert np.array_equal(flat["names"], prior["names"])
    res = dict(prior)
    res["sigma_hat"] = flat["sigma_bar"]        # no-prior branch stores Sigma_hat as Sigma_bar
    res["K_flat"] = flat["K"]
    res["alpha_hat_flat"] = flat["alpha_hat"]
    res["prior_width"] = np.array(prior_width)
    print(f"{tag:14s} lines={list(res['names'])} npix={int(res['emask'].sum())} "
          f"K_prior={float(res['K']):.6f} K_flat={float(res['K_flat']):.6f}")
    return {f"{tag}/{k}": v for k, v in res.items()}


CASES = dict(
    hb_oiii=dict(lo=4800, hi=5060, R=1000, prior_width=0.2,
                 to_fit=["Ba-beta 4861", "[O III] 4959", "[O III] 5007"]),
    fixed=dict(lo=4800, hi=5060, R=1000, prior_width=0.2,
               to_fit=["Ba-beta 4861", "[O III] 5007"], to_fix=["[O III] 4959"]),
    ignored=dict(lo=4800, hi=5060, R=1000, prior_width=0.2,
                 to_fit=None, to_ignore=["[O III] 4931"]),
    masked=dict(lo=4800, hi=5060, R=1000, prior_width=0.2,
                to_fit=["Ba-beta 4861", "[O III] 4959", "[O III] 5007"],
                mask_rest=(4858.0, 4861.5)),
    edge=dict(lo=4861.5, hi=5060, R=1000, prior_width=0.2,
              to_fit=["Ba-beta 4861", "[O III] 4959", "[O III] 5007"]),
    delta_z=dict(lo=4800, hi=5060, R=1000, prior_width=0.2, dz=3e-4,
                 to_fit=["Ba-beta 4861", "[O III] 4959", "[O III] 5007"]),
    oiii_R100=dict(lo=4700, hi=5300, R=100, prior_width=0.2,
                   to_fit=["[O III] 4959", "[O III] 5007"]),
    ha_nii_R100=dict(lo=6300, hi=6900, R=100, prior_width=0.2,
                     to_fit=["[N II] 6548", "Ba-alpha 6563", "[N II] 6584"]),
)


def main():
    print(f"prospect {prospect.__version__}, SPS_HOME={os.environ.get('SPS_HOME')}")
    data = {"prospect_version": np.array(prospect.__version__)}
    for i, (tag, kw) in enumerate(CASES.items()):
        data.update(_run(tag, seed=i, **kw))
    np.savez(OUT, **data)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
