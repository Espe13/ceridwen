"""The likelihood's arithmetic is float64 whatever the prediction's dtype: the noise model and the
profiled / marginalised calibration promote ``mu`` to the data's dtype before squaring it
(docs/conventions.md "Precision"). At cgs scale a float32 ``(noise_floor * |mu|)**2`` is below
float32's smallest number, so without the promotion the floor silently vanishes."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ceridwen.likelihood import DiagonalNoiseModel      # noqa: E402

F64 = np.dtype(np.float64)


def test_likelihood_is_float64_for_a_float32_prediction():
    """The noise floor of a float32 cgs-scale prediction: (0.05 * 1e-30)**2 = 2.5e-63 is far
    below float32's smallest number, so the variance must be formed in the data's float64."""
    mu64 = np.geomspace(1e-31, 1e-28, 50)
    sigma = jnp.asarray(0.1 * mu64)                   # float64, as Observation stores data
    mask = jnp.ones(mu64.size, dtype=bool)
    nm = DiagonalNoiseModel(noise_floor=0.05)
    ref = nm.compute(sigma, jnp.asarray(mu64), mask)
    got = nm.compute(sigma, jnp.asarray(mu64, dtype=jnp.float32), mask)
    assert got.inv_var.dtype == F64 and got.log_det.dtype == F64
    # the floor is present: 1/inv_var exceeds sigma**2 by (0.05 mu)**2, i.e. 25 %
    np.testing.assert_allclose(1.0 / np.asarray(got.inv_var), 1.25 * np.asarray(sigma) ** 2,
                               rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got.inv_var), np.asarray(ref.inv_var), rtol=1e-6)
