"""
ceridwen/model/transforms.py
============================
Parameter transform functions for SedModel.

These functions implement the ``transforms`` mechanism that allows fitting
reparameterised quantities instead of the raw CSP parameters.  The
canonical example is fitting ``logsfr_ratios`` — log-ratios of consecutive
SFR bins — instead of the raw SFH weight vector ``sfh``.

Design
------
A *transform* is a callable with signature::

    derived_value = fn(free_theta: dict[str, Array]) -> Array

It receives the *full* free-parameter dict and returns the derived
parameter value that is substituted into the CSP theta before calling
``csp.predict``.

Example usage with SedModel
----------------------------
::

    from ceridwen.model.transforms import logsfr_ratios_to_sfh

    model = SedModel(
        csp,
        observations=[spec],
        priors={"logsfr_ratios": Normal(mean=0.0, sigma=0.3)},
        transforms={
            "sfh": lambda t: logsfr_ratios_to_sfh(
                t["logsfr_ratios"],
                sfh_times_yr=csp.sfh_times,
            )
        },
        free_param_init={
            "logsfr_ratios": jnp.zeros(csp.n_time - 1),
        },
    )

The ``SedModel`` will:

- Remove ``"sfh"`` from its free-parameter list (it becomes derived).
- Add ``"logsfr_ratios"`` to the free-parameter list with the supplied
  initial values.
- At every ``predict(free_theta)`` call, evaluate the transform to obtain
  the CSP-compatible model_theta and pass it to ``csp.predict``.
- Evaluate ``ln_prior`` against ``free_theta`` (i.e., the prior on
  ``logsfr_ratios``, not on ``sfh``).

Parametrisation details
-----------------------
For ``n`` SFH time bins the free parameters are ``n − 1`` log-ratios::

    logsfr_ratios[i] = log10( SFR[i] / SFR[i+1] )

The first SFR bin is anchored (``log10 SFR[0] = 0``), and each subsequent
bin is derived via::

    log10 SFR[i+1] = log10 SFR[i] - logsfr_ratios[i]

The resulting linear SFR values are normalised so that the
mass-weighted mean equals unity using bin-width trapezoidal weights.
This matches the Prospector ``logsfr_ratios`` parametrisation described in
Leja et al. (2019, ApJ, 876, 3).
"""

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
    """
    Convert log-ratios of consecutive SFR bins to a normalised SFH weight
    vector.

    Parameters
    ----------
    logsfr_ratios : array_like, shape (n − 1,)
        Log10 ratios of consecutive SFR bins.
        ``logsfr_ratios[i] = log10( SFR[i] / SFR[i+1] )``.
        Positive values mean the SFR *decreases* with time (earlier burst);
        negative values mean the SFR *increases* (late assembly).
    sfh_times_yr : array_like, shape (n,), optional
        Lookback-time grid in years (same as ``CSPBasis.sfh_times``).
        When provided, trapezoidal integration weights are used for
        normalisation so that total stellar mass is conserved.
        If *None*, equal-weight normalisation is applied.

    Returns
    -------
    sfh : jnp.ndarray, shape (n,)
        Normalised SFH weight vector suitable for ``theta["sfh"]``.

    Notes
    -----
    The normalisation satisfies ``sum(sfh * w) / sum(w) = 1`` where ``w``
    are the standard trapezoidal quadrature weights (half-width at
    boundaries), which ensures that the trapezoidal integral of the SFH
    over the lookback-time grid equals the total lookback time.  This
    is consistent with the piecewise-constant mass integral used by
    ``CSPBasis.calculate_ssp_weights_const_zh_step`` and the
    piecewise-linear integral used by ``calculate_ssp_weights_const_zh``.
    """
    ratios  = jnp.asarray(logsfr_ratios, dtype=float)            # (n-1,)
    # Anchor log10(SFR[0]) = 0, then cumulate the negative ratios
    log_sfr = jnp.concatenate([jnp.zeros(1),
                                -jnp.cumsum(ratios)])             # (n,)
    sfr     = 10.0 ** log_sfr                                     # (n,)

    if sfh_times_yr is not None:
        times = jnp.asarray(sfh_times_yr, dtype=float)            # (n,)
        dt    = jnp.abs(jnp.diff(times))                          # (n-1,)
        # Standard trapezoidal quadrature weights:
        #   w[0]   = 0.5 * dt[0]
        #   w[i]   = 0.5 * (dt[i-1] + dt[i])   for 0 < i < n-1
        #   w[n-1] = 0.5 * dt[n-2]
        w_lo  = jnp.concatenate([jnp.zeros(1), dt])               # (n,)
        w_hi  = jnp.concatenate([dt, jnp.zeros(1)])               # (n,)
        w     = 0.5 * (w_lo + w_hi)                               # (n,)
        total = jnp.sum(sfr * w)
        sfh   = sfr * (jnp.sum(w) / total)
    else:
        n   = sfr.shape[0]
        sfh = sfr * (float(n) / jnp.sum(sfr))

    return sfh


def sfh_to_logsfr_ratios(sfh):
    """
    Invert :func:`logsfr_ratios_to_sfh` — useful for initialising free
    parameters from a known SFH.

    Parameters
    ----------
    sfh : array_like, shape (n,)
        SFH weight vector (need not be normalised).

    Returns
    -------
    logsfr_ratios : jnp.ndarray, shape (n − 1,)
        Log10 ratios of consecutive SFR bins satisfying
        ``logsfr_ratios[i] = log10( sfh[i] / sfh[i+1] )``.
    """
    sfh     = jnp.asarray(sfh, dtype=float)
    sfh     = jnp.clip(sfh, a_min=1e-30)
    log_sfr = jnp.log10(sfh)
    return log_sfr[:-1] - log_sfr[1:]                             # (n-1,)
