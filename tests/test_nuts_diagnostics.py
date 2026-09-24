"""B2-012: the NUTS convergence printout.

ESS is each chain's own autocorrelation summed over chains, checked against the analytic
ESS of AR(1) chains; the old estimator concatenated the chains, so an offset between chain
means read as autocorrelation.  The printed R-hat says what it can detect: without VI every
chain starts from the one warmup end state.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ceridwen.sampler.nuts import BlackJAXNUTSAdapter


def _ar1(rng, n, phi):
    x = np.empty(n)
    x[0] = rng.standard_normal() / np.sqrt(1 - phi ** 2)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_ess_is_per_chain_and_matches_ar1():
    rng = np.random.default_rng(0)
    n, phi = 20000, 0.5
    # analytic ESS of a stationary AR(1): n (1 - phi) / (1 + phi) = n / 3 per chain
    exact = 4 * n * (1 - phi) / (1 + phi)
    chains = [_ar1(rng, n, phi) + off for off in (0.0, 3.0, -3.0, 6.0)]
    ess = BlackJAXNUTSAdapter._ess(chains)
    assert abs(ess / exact - 1) < 0.1, (ess, exact)
    # the offsets between chains do not enter: same chains, common mean
    same = BlackJAXNUTSAdapter._ess([c - c.mean() for c in chains])
    assert abs(ess / same - 1) < 1e-12


def test_printed_rhat_is_labelled(capsys):
    ad = BlackJAXNUTSAdapter(num_warmup=30, num_samples=30, num_chains=2,
                             max_num_doublings=4, verbose=True)
    th0 = {"a": jnp.zeros(1), "b": jnp.zeros(1)}
    res = ad.run(lambda t: -0.5 * (t["a"][0] ** 2 + t["b"][0] ** 2),
                 lambda t: 0.0, th0, jax.random.PRNGKey(0))
    out = capsys.readouterr().out
    assert "R-hat: within-run mixing only" in out
    assert res.samples["a"].shape == (60,)
