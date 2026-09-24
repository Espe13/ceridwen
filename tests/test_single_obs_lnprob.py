"""B2-003: the single-observation make_lnprobfn factories honour the observation's sky and
calibration as MultiObservationLikelihood does, and refuse flagged upper limits where the
kernel would treat them as detections."""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ceridwen.likelihood import (DiagonalGaussianLikelihood,  # noqa: E402
                                 DiagonalGaussianLikelihoodWithUpperLimits,
                                 MultiObservationLikelihood)

N = 20
MU = jnp.linspace(1.0, 2.0, N)


class _Model:
    def predict(self, theta):
        return MU * theta["a"][0]


class _Prior:
    def log_prob(self, theta):
        return jnp.zeros(())


def _obs(**kw):
    calib, sky = jnp.full(N, 1.5), jnp.full(N, 1.0)
    base = dict(flux=calib * MU + sky, uncertainty=jnp.full(N, 0.1), mask=jnp.ones(N, bool),
                sky=sky, calibration=calib, upper_limit=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


class _MultiModel:
    def predict(self, theta):
        return {"o": _Model().predict(theta)}


@pytest.mark.parametrize("cls", [DiagonalGaussianLikelihood,
                                 DiagonalGaussianLikelihoodWithUpperLimits])
def test_single_equals_multi_with_sky_and_calibration(cls):
    th = {"a": jnp.array([1.0])}
    obs = _obs()
    single = cls().make_lnprobfn(obs, _Model(), _Prior())(th)
    multi = MultiObservationLikelihood(keys=("o",), likelihoods=(cls(),)).make_lnprobfn(
        {"o": obs}, _MultiModel(), _Prior())(th)
    assert float(single) == pytest.approx(float(multi), rel=1e-14)
    # perfect model: chi^2 = 0, ln L = -sum ln sqrt(2 pi sigma^2)
    assert float(single) == pytest.approx(-N * np.log(np.sqrt(2 * np.pi) * 0.1), rel=1e-12)


def test_plain_kernel_refuses_upper_limits():
    obs = _obs(upper_limit=jnp.zeros(N, bool).at[3].set(True))
    with pytest.raises(ValueError, match="upper limits"):
        DiagonalGaussianLikelihood().make_lnprobfn(obs, _Model(), _Prior())
    DiagonalGaussianLikelihoodWithUpperLimits().make_lnprobfn(obs, _Model(), _Prior())
