"""``logsfr_ratios_to_sfh`` in float32 (the Apple GPU's precision): a StudentT(df=2) prior draw
can push log10 SFR above 38.5, where ``10**x`` overflows float32 and inf/inf gave a NaN SFH.
The float32 path shifts the peak to 0 first; float64 is unchanged."""

import jax
import jax.numpy as jnp
import numpy as np

import ceridwen  # noqa: F401  (sets the precision)
from ceridwen.model import logsfr_ratios_to_sfh

T_YR = np.array([0.0, 1e8, 3e8, 1e9, 3e9, 1e10])
EXTREME = [[0.86, -2.29, -38.93, 2.00, -6.07], [-8.0] * 5, [-7.0] * 5]


def _float32():
    if hasattr(jax, "enable_x64"):                       # jax >= 0.6
        return jax.enable_x64(False)
    from jax.experimental import disable_x64             # older jax
    return disable_x64()


def test_float32_sfh_stays_finite_and_matches_float64():
    ref = [np.asarray(logsfr_ratios_to_sfh(jnp.array(r), sfh_times_yr=T_YR)) for r in EXTREME]
    with _float32():
        got = [np.asarray(logsfr_ratios_to_sfh(jnp.array(r, dtype=jnp.float32),
                                               sfh_times_yr=T_YR.astype(np.float32)))
               for r in EXTREME]
    for r, g in zip(ref, got):
        assert g.dtype == np.float32 and np.all(np.isfinite(g))
        big = r > 1e-30                                  # float32 flushes the tiny bins to 0
        np.testing.assert_allclose(g[big], r[big], rtol=1e-5)
        # unit mass: the trapezoid integral of the SFH is 1 Msun
        w = 0.5 * (np.r_[0, np.diff(T_YR)] + np.r_[np.diff(T_YR), 0])
        assert abs(float(np.sum(g * w)) - 1.0) < 1e-5


def test_nested_sampler_treats_nan_loglike_as_outside_the_support():
    """A NaN ln L at a prior draw used to make logZ_live NaN and end the run at iteration 0."""
    from ceridwen.priors import Uniform
    from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter

    prior = Uniform(low=-1.0, high=1.0)

    def loglike(th):                                     # NaN on a quarter of the prior
        x = th["x"][0]
        return jnp.where(x > 0.5, jnp.nan, -0.5 * (x / 0.1) ** 2)

    def logprior(th):
        return jnp.sum(prior.logpdf(th["x"]))

    ad = BlackJAXNestedSamplerAdapter(priors={"x": prior}, num_live=50, num_delete=10,
                                      verbose=False)
    res = ad.run(loglike, logprior, {"x": jnp.zeros(1)}, jax.random.PRNGKey(0))
    # the Gaussian's evidence over the uniform prior: ln(0.1 sqrt(2 pi) / 2)
    assert np.isfinite(res.log_evidence)
    assert abs(res.log_evidence - np.log(0.1 * np.sqrt(2 * np.pi) / 2)) < 0.5
