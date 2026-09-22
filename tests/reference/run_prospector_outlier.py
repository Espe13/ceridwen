"""Reference dump of Prospector's outlier (mixture) likelihood for
``tests/test_outlier_model.py`` (test 2).

Run in the Prospector environment, NOT with ceridwen (neither package imports the other):

    conda activate prospector
    SPS_HOME=/Users/amanda/Prospector/fsps python tests/reference/run_prospector_outlier.py

Writes ``tests/reference/prospector_outlier.npz``.  For every case a real
``prospect.observation.Spectrum`` / ``Photometry`` gets a real
``prospect.likelihood.NoiseModel(frac_out_name=..., nsigma_out_name=...)`` (the wiring of
Prospector's doc/noise.rst), ``noise.update(**params)`` is called exactly as
``prospect.fitting.fitting.lnprobfn`` does (fitting.py:88) and ``compute_lnlike`` returns the
value (likelihood.py:15-33).  The installed prospect (2.0a2.dev42+gff8d5e4) has
``NoiseModel.lnlike`` / ``compute`` / ``lnlikelihood`` identical to bd-j/prospector commit
a78d15362fc177c43b7cde919d26409806e50fd7 (noise_model.py:13-92).

Saved per case ``<tag>/<key>``: ``flux``, ``unc``, ``mask``, ``pred``, ``f``, ``nsigma``,
``lnl``.  The ``f = 0`` cases record Prospector's plain branch, whose value is NOT the
Gaussian ln L (noise_model.py:90 multiplies chi^2 by ln 2 pi and drops n ln 2 pi).
"""
from __future__ import annotations

import pathlib

import numpy as np
import prospect
from prospect.likelihood import NoiseModel
from prospect.likelihood.likelihood import compute_lnlike
from prospect.observation import Spectrum, Photometry

OUT = pathlib.Path(__file__).resolve().parent / "prospector_outlier.npz"


def _data(n, seed, n_out):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(1.0, 5.0, n)
    unc = rng.uniform(0.05, 0.3, n)
    flux = pred + unc * rng.standard_normal(n)
    idx = rng.choice(n, n_out, replace=False)
    flux[idx] += rng.choice([-1.0, 1.0], n_out) * 20.0 * unc[idx]
    mask = rng.uniform(size=n) > 0.1
    return flux, unc, mask, pred


def main():
    out = {}
    cases = [
        ("spec_f01",    "spec", 300, 1, 9,  0.01, 50.0),
        ("spec_f20_n5", "spec", 300, 2, 30, 0.20, 5.0),
        ("spec_f1e5",   "spec", 300, 3, 0,  1e-5, 50.0),
        ("spec_f0",     "spec", 300, 4, 9,  0.0,  50.0),
        ("phot_f05",    "phot", 12,  5, 1,  0.05, 50.0),
        ("phot_f0",     "phot", 12,  6, 1,  0.0,  50.0),
    ]
    for tag, kind, n, seed, n_out, f, nsig in cases:
        flux, unc, mask, pred = _data(n, seed, n_out)
        names = (f"f_outlier_{kind}", f"nsigma_outlier_{kind}")
        noise = NoiseModel(frac_out_name=names[0], nsigma_out_name=names[1])
        if kind == "spec":
            obs = Spectrum(wavelength=np.linspace(4000.0, 7000.0, n), flux=flux,
                           uncertainty=unc, mask=mask, noise=noise)
        else:
            obs = Photometry(filters=[], flux=flux, uncertainty=unc, mask=mask, noise=noise)
        # model.params holds N=1 arrays, fixed and free alike (models/parameters.py)
        params = {names[0]: np.array([f]), names[1]: np.array([nsig])}
        obs.noise.update(**params)
        lnl = compute_lnlike(pred, obs, vectors={})
        for k, v in dict(flux=flux, unc=unc, mask=mask, pred=pred, f=f, nsigma=nsig,
                         lnl=float(np.squeeze(lnl))).items():
            out[f"{tag}/{k}"] = np.asarray(v)
        print(f"{tag:<12} f={f:<6g} nsigma={nsig:<4g} lnL = {float(np.squeeze(lnl)):.15g}")
    out["prospect_version"] = np.asarray(str(getattr(prospect, "__version__", "?")))
    np.savez(OUT, **out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
