"""Parameter transforms for SedModel: callables ``fn(free_theta) -> derived value``
substituted into the CSP theta before ``csp.predict``."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "logsfr_ratios_to_sfh",
    "sfh_to_logsfr_ratios",
]


def logsfr_ratios_to_sfh(
    logsfr_ratios,
    sfh_times_yr=None,
):
    """Unit-mass SFH weight vector (n,) from ``logsfr_ratios[i] = log10(SFR[i]/SFR[i+1])``
    (n-1,), index 0 = today. With ``sfh_times_yr`` (lookback yr, (n,)) the trapezoidal
    integral is normalised to 1 Msun; otherwise sum(sfh) = 1."""
    ratios  = jnp.asarray(logsfr_ratios, dtype=float)
    log_sfr = jnp.concatenate([jnp.zeros(1),
                                -jnp.cumsum(ratios)])
    sfr     = 10.0 ** log_sfr

    if sfh_times_yr is not None:
        times = jnp.asarray(sfh_times_yr, dtype=float)
        dt    = jnp.abs(jnp.diff(times))
        w_lo  = jnp.concatenate([jnp.zeros(1), dt])
        w_hi  = jnp.concatenate([dt, jnp.zeros(1)])
        w     = 0.5 * (w_lo + w_hi)
        # normalise to total mass (1 Msun), NOT to mean SFR
        total_mass = jnp.sum(sfr * w)
        sfh   = sfr / total_mass
    else:
        sfh = sfr / jnp.sum(sfr)

    return sfh


def sfh_to_logsfr_ratios(sfh):
    """Invert :func:`logsfr_ratios_to_sfh`: (n-1,) log10 ratios of consecutive bins."""
    sfh     = jnp.asarray(sfh, dtype=float)
    sfh     = jnp.clip(sfh, 1e-30)
    log_sfr = jnp.log10(sfh)
    return log_sfr[:-1] - log_sfr[1:]
