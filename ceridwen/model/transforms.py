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
    """Unit-mass SFH from ``logsfr_ratios[i] = log10(SFR[i]/SFR[i+1])``, index 0 = today.

    The length of ``sfh_times_yr`` (the lookback NODES in yr, e.g. ``csp.sfh_times``) decides
    what the SFR values are:

    * ``len(sfh_times_yr) == len(logsfr_ratios) + 2``: one SFR per BIN, shape (n_time-1,), for
      ``sfh_interp="step"``.  Normalised so sum_i SFR_i * dt_i = 1 Msun: with ``logmass`` the
      mass formed in bin i is 10**logmass * SFR_i * dt_i, which is Prospector's continuity SFH
      (``prospect.models.transforms.logsfr_ratios_to_masses``: ratios of adjacent BIN SFRs).
    * ``len(sfh_times_yr) == len(logsfr_ratios) + 1``: one SFR per NODE, shape (n_time,), for
      ``sfh_interp="linear"`` (FSPS's tabular SFH, linear between nodes); the trapezoidal
      integral is normalised to 1 Msun.

    Without ``sfh_times_yr``, sum(sfh) = 1."""
    ratios  = jnp.asarray(logsfr_ratios, dtype=float)
    log_sfr = jnp.concatenate([jnp.zeros(1),
                                -jnp.cumsum(ratios)])
    if log_sfr.dtype == jnp.float32:
        # float32 mode (docs/mac_gpu.md): 10**x overflows above x ~ 38.5, which the StudentT(df=2)
        # prior reaches in ~0.2% of draws, and inf/inf = NaN.  The normalisation below divides
        # out any constant, so shift the peak to 0; float64 keeps the unshifted (bit-identical) path
        log_sfr = log_sfr - jnp.max(log_sfr)
    sfr     = 10.0 ** log_sfr

    if sfh_times_yr is not None:
        times = jnp.asarray(sfh_times_yr, dtype=float)
        dt    = jnp.abs(jnp.diff(times))
        n_sfr = ratios.shape[-1] + 1
        if times.shape[-1] == n_sfr + 1:            # one SFR per bin: sum_i SFR_i dt_i = 1
            return sfr / jnp.sum(sfr * dt)
        if times.shape[-1] != n_sfr:
            raise ValueError(
                f"logsfr_ratios_to_sfh: {ratios.shape[-1]} ratios give {n_sfr} SFR values, but "
                f"sfh_times_yr has {times.shape[-1]} nodes; give {n_sfr + 1} nodes for one SFR "
                f"per bin (sfh_interp='step') or {n_sfr} for one per node ('linear').")
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
