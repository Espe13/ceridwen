"""float32 error bounds for tests that compare two equivalent paths (docs/conventions.md
"Precision": spectra, curves and projections are float32). Every tolerance taken from here is
one of four bounds, so a change to a test's tolerance names its reason:

* ``B8``: 8 float32 roundings (8 * 2**-24 = 4.8e-7), for quantities that differ by a handful
  of float32 operations done in another order (the bound ``csp/test_wavelength_support.py``
  already uses as ``RTOL``).
* ``lnl_bound``: the first-order change of a Gaussian log-likelihood when every prediction
  moves by ``B8`` relative, ``sum |y - mu| * B8 * |mu| / sigma^2``
  (``test_wavelength_support.py::test_lnprob_and_gradient`` already uses this form).
* ``fft_atol``: the float32 FFT smoothing's error, absolute and set by the largest value it
  smooths, ``16 * eps * log2(n_pad) * max|x|``; a faint pixel or band therefore sees a larger
  relative error than ``B8``.
* ``profile_bound``: a float32 Gaussian line profile against the same profile in float64. Its
  argument x carries ~``B8`` relative error, so exp(-x^2 / 2) carries ~x^2 ``B8``: per element
  ``(1 + 2 ln(1 / floor)) B8`` relative where the profile exceeds ``floor`` of its peak
  (x^2 <= 2 ln(1 / floor)), and that bound times ``floor`` of the peak, absolute, below (x^2
  exp(-x^2 / 2) falls with x there).
"""
from __future__ import annotations

import numpy as np

EPS32 = 2.0 ** -24
B8 = 8 * EPS32


def lnl_bound(model, theta, B=B8):
    """First-order bound on |d lnL| for a B-relative change of every prediction of ``model``
    at ``theta`` (the observations' own data, errors and masks)."""
    pred = model.predict(theta)
    tot = 0.0
    for name, o in model.obs_dict.items():
        mu = np.asarray(pred[name], dtype=np.float64)
        y = np.asarray(o.flux, dtype=np.float64)
        s = np.asarray(o.uncertainty, dtype=np.float64)
        m = np.asarray(o.mask, dtype=bool) if np.size(o.mask) else np.ones(mu.shape, bool)
        tot += float(np.sum((np.abs(y - mu) * B * np.abs(mu) / s ** 2)[m]))
    return tot


def assert_profile_bound(got, want64, floor=1e-4):
    """``got`` (a float32 painted profile, as delivered) within ``profile_bound`` of ``want64``
    (the same profile in float64)."""
    got = np.asarray(got, dtype=np.float64)
    want64 = np.asarray(want64, dtype=np.float64)
    pk = float(np.max(np.abs(want64)))
    b = (1.0 + 2.0 * np.log(1.0 / floor)) * B8
    big = np.abs(want64) > floor * pk
    d = np.abs(got - want64)
    assert np.all(d[big] <= b * np.abs(want64[big])), float(np.max(d[big] / np.abs(want64[big])))
    assert np.all(d[~big] <= b * floor * pk), float(np.max(d[~big]) / pk)


def fft_atol(x, n_pad):
    """Absolute float32 FFT-smoothing error for arrays of magnitude max|x| (n_pad the FFT
    length)."""
    return 16 * EPS32 * np.log2(n_pad) * float(np.max(np.abs(np.asarray(x, dtype=np.float64))))
