"""BLR golden configuration (session BLR; CLAUDE.md section 4: a new theta key ships a baseline).

``baselines/blr_golden.npz`` holds the BLR-only part (prediction with blr_lum minus blr_lum = 0)
of a Spectrum (R = 2000 around H alpha), a Lines observation (H alpha, H beta) and Photometry
(sdss_i0, sdss_z0) at z = 0.3, corrected mode, BLR only, for three (blr_lum, blr_sigma,
diffuse_tau_kc) settings, and the cigale-mode spectrum of the second.

Why the numbers are right (checked before capture, and again here on every run): the painter's
profile in float64 and the Lines equal the independent NumPy recomputation of test_blr_dev.py
D-BLR-4 (Gaussian in ln lambda, fluxes from the definitions) to 1e-10 of the peak / 1e-12, the
spectrum (float32 profiles) equals that profile within the float32 profile bound, and the
photometry equals
the delta-function estimate to 1 % (D-BLR-5). Capture: ``BLR_CAPTURE=1 pytest <this file>``
(writes the npz only when those checks pass); never re-capture to make the test pass.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _blr_model as B  # noqa: E402
import test_blr_dev as D  # noqa: E402

GOLDEN = pathlib.Path(__file__).resolve().parent / "baselines" / "blr_golden.npz"
SETTINGS = [(0.3, 400.0, 0.0), (1.0, 1500.0, 0.5), (2.5, 4500.0, 1.2)]


def _compute():
    model, csp, spec = B.build("BLR golden")
    out = {}
    for k, (lum, sig, tau) in enumerate(SETTINGS):
        kw = dict(blr_sigma=sig, diffuse_tau_kc=tau)
        d = B.broad(model, blr_lum=lum, **kw)
        F = B.expected_fluxes(model, csp, blr_lum=lum, **kw)
        D.assert_painted_spectrum(d["spec"], model, csp, spec, F, sig)
        names = list(csp._agn.model.blr.line_names)
        np.testing.assert_allclose(
            d["lines"], F[[names.index("Ba-alpha 6563"), names.index("Ba-beta 4861")]],
            rtol=1e-12)
        for name in ("spec", "lines", "phot"):
            out[f"{name}_{k}"] = d[name]
    mc, _, _ = B.build("BLR golden (cigale)", mode="cigale")
    lum, sig, tau = SETTINGS[1]
    out["spec_cigale_1"] = B.broad(mc, blr_lum=lum, blr_sigma=sig, diffuse_tau_kc=tau)["spec"]
    return out


def test_blr_golden():
    got = _compute()
    if os.environ.get("BLR_CAPTURE") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        if GOLDEN.exists():
            pytest.fail(f"{GOLDEN} exists; never re-capture a baseline (CLAUDE.md section 4)")
        np.savez(GOLDEN, **got)
    if not GOLDEN.is_file():
        pytest.fail(f"MISSING baseline {GOLDEN}")
    ref = np.load(GOLDEN)
    assert sorted(ref.files) == sorted(got)
    for k in ref.files:
        # spec / lines: deterministic on CPU (the spectrum's float32 profiles included);
        # photometry passes float32 filters
        rtol = 2e-6 if k.startswith("phot") else 1e-10
        np.testing.assert_allclose(got[k], ref[k], rtol=rtol,
                                   atol=1e-12 * float(np.abs(ref[k]).max()), err_msg=k)
