"""Observation data are stored float64 however they are set: at construction (as always) and
by assigning ``obs.flux`` / ``obs.uncertainty`` afterwards (``observation/base.py``). The
predictions are float32 (docs/conventions.md "Precision"), and a float32 sigma at cgs scale
(~1e-30) underflows when the likelihood squares it."""
from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from ceridwen.observation import Spectrum       # noqa: E402

F64 = np.dtype(np.float64)


def test_observation_data_are_float64_however_set():
    """At construction and by assignment afterwards (as harnesses and users do)."""
    f32 = np.full(4, 1e-29, dtype=np.float32)
    s = Spectrum(wavelength=np.linspace(4000.0, 5000.0, 4), flux=f32, uncertainty=0.1 * f32,
                 name="s")
    assert s.flux.dtype == F64 and s.uncertainty.dtype == F64
    s.flux, s.uncertainty = f32, 0.1 * f32
    assert s.flux.dtype == F64 and s.uncertainty.dtype == F64
    s.uncertainty = None
    assert s.uncertainty is None
