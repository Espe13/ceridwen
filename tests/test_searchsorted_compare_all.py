"""``jnp.searchsorted(..., method='compare_all')`` on the small grid axes (metallicity,
[alpha/Fe], dust-emission q_PAH / U_min, nebular logU / logZ / age) must return the same indices
as the default method: the forward model relies on it being a pure lowering change (fork
d89553f).  Checked on the real axes of the test grid and on values at, between and outside
the nodes."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


def _probe(grid):
    grid = np.asarray(grid, dtype=np.float64)
    mid = 0.5 * (grid[1:] + grid[:-1])
    span = grid[-1] - grid[0]
    rng = np.random.default_rng(0)
    return np.concatenate([grid, mid, grid[:1] - 0.1 * span, grid[-1:] + 0.1 * span,
                           np.nextafter(grid, np.inf), np.nextafter(grid, -np.inf),
                           rng.uniform(grid[0] - span, grid[-1] + span, 200)])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("n", [2, 3, 5, 7, 12, 22, 40])
def test_compare_all_equals_default_on_small_sorted_grids(n, side):
    grid = jnp.asarray(np.sort(np.random.default_rng(n).uniform(-3.0, 1.0, n)))
    x = jnp.asarray(_probe(grid))
    for xs in (x, x[0]):                       # vector and scalar queries, as in the model
        a = jax.jit(lambda g, v: jnp.searchsorted(g, v, side=side))(grid, xs)
        b = jax.jit(lambda g, v: jnp.searchsorted(g, v, side=side, method="compare_all"))(grid, xs)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_compare_all_equals_default_on_the_test_grid_axes():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _gridfixture import find_test_grid
    path = find_test_grid()
    if path is None:
        pytest.skip("no test SSP grid")
    from ceridwen import SSPData, CSPBasis
    csp = CSPBasis(SSPData.load(str(path)), lookback_time=jnp.linspace(0.0, 5.0, 5),
                   verbose=False)
    grid = jnp.asarray(csp.zmet)
    x = jnp.asarray(_probe(grid))
    np.testing.assert_array_equal(
        np.asarray(jnp.searchsorted(grid, x, side="left")),
        np.asarray(jnp.searchsorted(grid, x, side="left", method="compare_all")))
