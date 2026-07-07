"""Flux-less ("predictive") observation containers.

A container constructed with a wavelength grid but no flux is the mock-
generation / forward-modelling configuration: ``setup_for_model`` +
``predict`` need only the grid; data can be attached afterwards.

Regression: ``Observation.rectify()`` used to reset ``wavelength`` to None
whenever ``flux is None``, which silently broke predictive ``Spectrum``
containers (``setup_for_model`` then failed with ``len() of unsized
object``). It must keep the grid, and a genuinely grid-less Spectrum must
fail with an informative ValueError instead.

Pure construction/projection tests: no FSPS, no grid fixture, CPU only.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from ceridwen.observation import Spectrum


def test_fluxless_spectrum_keeps_wavelength_grid():
    wave_obs = np.linspace(4000.0, 8000.0, 50)
    spec = Spectrum(wavelength=wave_obs, name="predictive")
    assert spec.wavelength is not None
    np.testing.assert_allclose(np.asarray(spec.wavelength), wave_obs)


def test_fluxless_spectrum_setup_and_predict():
    wave_obs = np.linspace(4000.0, 8000.0, 50)
    spec = Spectrum(wavelength=wave_obs, name="predictive")

    wave_model = np.linspace(1000.0, 20000.0, 500)   # rest frame
    zred = 0.1
    spec.setup_for_model(wave_model, zred=zred)

    # A flat unit model spectrum must project to ~1 on every pixel
    # (pure linear interpolation of a constant).
    pred = spec.predict(jnp.ones(wave_model.size), wave_model)
    assert pred.shape == (wave_obs.size,)
    np.testing.assert_allclose(np.asarray(pred), 1.0, rtol=1e-6)


def test_gridless_spectrum_raises_informative_error():
    spec = Spectrum(name="empty")
    with pytest.raises(ValueError, match="wavelength"):
        spec.setup_for_model(np.linspace(1000.0, 20000.0, 500), zred=0.1)
