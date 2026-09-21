"""The common error-scale term of DiagonalNoiseModel (log_err_scale)."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ceridwen.likelihood.noise_model import DiagonalNoiseModel
from ceridwen.likelihood.likelihood import lnlike_diag_gaussian


def test_error_scale_rescales_variance_and_penalises():
    sig = jnp.array([1.0, 2.0, 3.0]); mu = jnp.array([10.0, 10.0, 10.0])
    mask = jnp.array([True, True, False])
    base = DiagonalNoiseModel().compute(sig, mu, mask)
    sc = DiagonalNoiseModel(use_error_scale=True)
    assert sc.nuisance_param_names == ("log_err_scale",)
    out = sc.compute(sig, mu, mask, {"log_err_scale": jnp.log(2.0)})
    assert np.allclose(np.asarray(base.inv_var[:2] / out.inv_var[:2]), 4.0)
    # the log-normalisation term grows with the scale, so inflating the errors is not free
    y = jnp.array([10.5, 9.0, 0.0])
    lnl0, _ = lnlike_diag_gaussian(y, mu, base.inv_var, base.log_det, mask)
    lnl2, _ = lnlike_diag_gaussian(y, mu, out.inv_var, out.log_det, mask)
    big = sc.compute(sig, mu, mask, {"log_err_scale": jnp.log(50.0)})
    lnl50, _ = lnlike_diag_gaussian(y, mu, big.inv_var, big.log_det, mask)
    assert float(lnl50) < float(lnl2)
    # unit scale reproduces the default model exactly
    one = sc.compute(sig, mu, mask, {"log_err_scale": jnp.array(0.0)})
    assert np.array_equal(np.asarray(one.inv_var), np.asarray(base.inv_var))
    g = jax.grad(lambda s: jnp.sum(
        sc.compute(sig, mu, mask, {"log_err_scale": s}).inv_var))(jnp.array(0.3))
    assert np.isfinite(float(g)) and float(g) < 0.0
