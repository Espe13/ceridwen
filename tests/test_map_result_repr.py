"""Q2-015: ``print(map_fit(...))`` shows the best point, not every start's arrays."""
from __future__ import annotations

import numpy as np

from ceridwen.optimize import MAPResult


def test_repr_is_the_summary_not_every_start():
    n = 17
    rng = np.random.default_rng(0)
    r = MAPResult(theta={"logmass": np.array([10.44]), "logzsol": np.array([-0.154])},
                  lnp=38176.18, lnp_starts=rng.normal(size=n),
                  theta_starts={"logmass": rng.normal(size=(n, 1)),
                                "logzsol": rng.normal(size=(n, 1))},
                  n_steps=np.arange(n), grad_norm=rng.random(n), best_start=3, wall_time=108.5)
    text = repr(r)
    assert text.startswith("MAP: ln p = 38176.1800 (start 3 of 17; 17 finite; 108.5 s)")
    assert "logmass" in text and "10.44" in text
    assert "theta_starts=" not in text and len(text) < 400
    assert str(r) == text
