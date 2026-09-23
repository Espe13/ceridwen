"""Reference dump of Prospector's profiled polynomial calibration (``PolyOptCal``) for
``tests/test_noise_calibration.py``.

Run in the Prospector environment, NOT with ceridwen (neither package imports the other):

    conda activate prospector
    python tests/reference/run_prospector_polyopt.py

Writes ``tests/reference/prospector_polyopt.npz``.  Each case builds a real
``prospect.observation.Spectrum`` with the ``PolyOptCal`` mixin (the way Prospector's
``PolySpectrum`` is composed) and calls ``compute_response(spec=model)``.  The installed
prospect (2.0a2.dev42+gff8d5e4) has ``PolyOptCal.compute_response`` and ``wave_to_x``
identical in code to bd-j/prospector commit a78d153 (observation.py:579-658, 824-830; only
a docstring and an assert message differ).

Saved per case ``<tag>/<key>``: ``wave``, ``flux``, ``unc``, ``mask``, ``spec`` (the model
on the pixels), ``order``, ``reg``, ``response``, ``coeffs``.
"""
from __future__ import annotations

import pathlib

import numpy as np
import prospect
from prospect.observation.observation import PolyOptCal, Spectrum


class PolySpectrum(PolyOptCal, Spectrum):
    pass


def case(rng, n, order, reg, holes):
    wave = np.linspace(4000.0, 9000.0, n) + rng.uniform(-0.5, 0.5, n)
    wave.sort()
    spec = 1e-18 * (1.0 + 0.5 * np.sin(wave / 700.0)) * (wave / 6000.0) ** -1.5
    true = 1.0 + 0.1 * np.polynomial.chebyshev.chebval(np.linspace(-1, 1, n),
                                                       rng.normal(0, 1, order + 1))
    unc = 0.02 * spec * rng.uniform(0.5, 2.0, n)
    flux = spec * true + unc * rng.standard_normal(n)
    mask = np.ones(n, bool)
    for lo, hi in holes:
        mask[lo:hi] = False
    obs = PolySpectrum(wavelength=wave, flux=flux, uncertainty=unc, mask=mask,
                       polynomial_order=order, polynomial_regularization=np.asarray(reg))
    resp = obs.compute_response(spec=spec)
    return dict(wave=wave, flux=flux, unc=unc, mask=obs.mask.astype(bool), spec=spec,
                order=np.array(order), reg=np.broadcast_to(np.asarray(reg, float), (order + 1,)).copy(),
                response=np.asarray(resp, float), coeffs=np.asarray(obs._chebyshev_coefficients))


def main():
    rng = np.random.default_rng(20260922)
    cases = {"o1": case(rng, 300, 1, 0.0, []),
             "o3_holes": case(rng, 500, 3, 0.0, [(0, 20), (200, 240), (480, 500)]),
             "o6_reg": case(rng, 800, 6, [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], [(100, 130)])}
    out = {f"{t}/{k}": v for t, d in cases.items() for k, v in d.items()}
    out["prospect_version"] = np.array(str(getattr(prospect, "__version__", "unknown")))
    path = pathlib.Path(__file__).with_name("prospector_polyopt.npz")
    np.savez(path, **out)
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
