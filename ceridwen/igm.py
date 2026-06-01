r"""
ceridwen/igm.py
===============
Intergalactic-medium (IGM) absorption models for redshift-aware SED
fitting.  Applied as a wavelength-dependent transmission curve to the
rest-frame source spectrum as a function of the source redshift; the
returned factor can simply be multiplied into the spectrum before the
observation projection.

Interface
---------
:class:`IGMModel` is the abstract base.  Each concrete subclass
implements ``tau(lam_rest, zred)`` returning the optical depth on the
rest-frame wavelength grid at the given source redshift.  The public
:meth:`IGMModel.attenuation` method returns ``exp(-tau * factor)`` —
the transmission curve — with an optional scalar ``factor`` matching
FSPS's ``igm_factor`` fudge (so the IGM strength can be fit).

Models
------
:class:`Madau1995`
    Line-by-line port of FSPS's ``igm_absorb.f90``: 17 Lyman-series
    transitions (Ly-α through Ly-17) with blanketing index 3.46,
    metal blanketing on Ly-α, and the analytic Lyman-continuum
    approximation (Eq. 16 of Madau 1995).  τ is capped at its peak
    short-wavelength value so the fitting-function breakdown below
    ~100 Å does not produce a non-monotonic curve.

:class:`NoIGM`
    Identity (``τ ≡ 0``).  Useful as a null model so the forward
    pass does not need a Python-level branch on ``self.igm_model is
    None``.

Extensibility
-------------
Adding e.g. Inoue et al. (2014) is a matter of subclassing
:class:`IGMModel`, implementing ``tau``, and registering in
:data:`_MODEL_REGISTRY`.  The public constructor
:func:`make_igm_model` accepts either a string name or a pre-built
instance:

>>> from ceridwen.igm import make_igm_model, Madau1995
>>> igm = make_igm_model("madau1995")
>>> igm = make_igm_model(Madau1995())
>>> igm = make_igm_model(None)              # NoIGM

Tied to CSPBasis
----------------
``CSPBasis`` accepts ``add_igm`` and ``igm_model`` kwargs.  When
``add_igm=True``, ``CSPBasis.predict`` multiplies the post-mass-
post-flux-factor spectrum by the transmission curve at
``theta["zred"]`` (and, optionally, ``theta["igm_factor"]``) — so IGM
strength can be fit.

References
----------
Madau, P. (1995), ApJ, 441, 18.  "Radiative Transfer in a Clumpy
Universe: The Colors of High-Redshift Galaxies".
"""
from __future__ import annotations

import abc
from typing import Union

import jax
import jax.numpy as jnp

Array = jax.Array


# ======================================================================
# Abstract base
# ======================================================================

class IGMModel(abc.ABC):
    """Abstract IGM attenuation model.

    Concrete subclasses implement ``tau(lam_rest, zred)``.  The
    :meth:`attenuation` method returns ``exp(-tau * factor)`` and is
    what callers (e.g. ``CSPBasis.predict``) multiply into the
    spectrum.
    """

    name: str = "igm"

    @abc.abstractmethod
    def tau(self, lam_rest: Array, zred: Array) -> Array:
        """Optical depth on the rest-frame wavelength grid at source
        redshift ``zred``.  Must be broadcastable to ``lam_rest``'s
        shape."""

    def attenuation(self, lam_rest: Array, zred: Array,
                    factor: Union[Array, float] = 1.0) -> Array:
        """Transmission curve ``exp(-tau * factor)``."""
        return jnp.exp(-self.tau(lam_rest, zred) * factor)


# ======================================================================
# Null model — zero optical depth everywhere
# ======================================================================

class NoIGM(IGMModel):
    """Identity IGM: zero optical depth at every wavelength."""

    name = "none"

    def tau(self, lam_rest, zred):
        return jnp.zeros_like(lam_rest)


# ======================================================================
# Madau (1995) — line-by-line port of FSPS's igm_absorb.f90
# ======================================================================

class Madau1995(IGMModel):
    r"""Madau (1995) IGM attenuation.

    Direct port of FSPS's ``igm_absorb.f90``.  Three components:

    1. **Lyman-series line blanketing** (17 transitions from Ly-α
       down to Ly-17):

       .. math::
           \tau_\mathrm{Ly\,series}(\lambda_\mathrm{rest}) =
           \sum_{i=1}^{17} A_i
           \left(\frac{\lambda_\mathrm{rest}(1+z)}{\lambda_i}\right)^{3.46},

       applied only where :math:`\lambda_\mathrm{rest} < \lambda_i`.

    2. **Metal blanketing on Ly-α** with a softer index 1.68:

       .. math::
           \tau_\mathrm{metal} = 0.0017
           \bigl(\lambda_\mathrm{obs}/1215.67\bigr)^{1.68}.

    3. **Lyman-continuum photoelectric absorption** (Madau 1995 Eq. 16
       analytic approximation) for :math:`\lambda_\mathrm{rest} < 911.75\,\text{Å}`.

    A small post-processing step caps τ at its peak short-wavelength
    value, suppressing the fitting-function breakdown at ~100 Å.

    At ``zred = 0`` τ is forced to zero — the Madau 1995 approximation
    is not physically valid for a line-of-sight of zero length and
    would otherwise give spurious residual attenuation to local-
    universe sources.  This matches FSPS's own usage, which only
    invokes ``IGM_ABSORB`` when ``zred > 0``.
    """

    name = "madau1995"

    # Source of the tabulation below: Madau (1995), ApJ 441, 18 (eqs. 12-16),
    # as transcribed in FSPS's ``src/igm_absorb.f90`` (arrays ``lyw`` /
    # ``lycoeff`` and the Lyman limit ``lylim``).  Reproduced verbatim so
    # ceridwen's IGM attenuation matches FSPS's Madau mode bit-for-bit.  (The
    # more modern Inoue+ 2014 prescription is a different model, not used here.)
    #
    # Lyman-series rest wavelengths [Å] — 17 transitions, Ly-α first.
    _LYW = jnp.asarray([
        1215.67, 1025.72,  972.537, 949.743, 937.803,
         930.748, 926.226, 923.150, 920.963, 919.352,
         918.129, 917.181, 916.429, 915.824, 915.329,
         914.919, 914.576,
    ])

    # Blanketing coefficients (identical to FSPS).
    _LYCOEFF = jnp.asarray([
        0.0036,    0.0017,    0.0011846, 0.0009410, 0.0007960,
        0.0006967, 0.0006236, 0.0005665, 0.0005200, 0.0004817,
        0.0004487, 0.0004200, 0.0003947, 0.000372,  0.000352,
        0.0003334, 0.00031644,
    ])

    _LYLIM   = 911.75    # Lyman limit [Å]
    _A_METAL = 0.0017    # metal blanketing coefficient for Ly-α

    def tau(self, lam_rest, zred):
        # Clamp z >= 0: the Madau 1995 fitting formulae are undefined
        # for negative redshifts, and both VI ELBO training and NUTS
        # leapfrog steps can transiently explore ``z < 0``.  Without
        # this guard ``(1+z)^0.46`` at ``z < -1`` raises a power of a
        # negative base, producing NaN that then poisons the entire VI
        # gradient — causing 100 % divergences in the downstream NUTS
        # sampling.  Clamping at 0 keeps the trace finite; any proposal
        # with ``z < 0`` is still rejected by whichever prior / Jacobian
        # is in play.
        zred = jnp.maximum(zred, 0.0)
        lam  = lam_rest
        z1   = 1.0 + zred
        lobs = lam * z1
        xc   = lobs / self._LYLIM

        # ─── Lyman-series line blanketing (vectorised) ──────────────
        # FSPS loops over the 17 Ly lines and sets
        #   tau(lam <  lyw_i) += A_i * (lobs/lyw_i)^3.46
        # which is equivalent to masking + broadcasting sum.
        masks = lam[None, :] < self._LYW[:, None]               # (17, n_wave)
        contrib = (self._LYCOEFF[:, None]
                   * (lobs[None, :] / self._LYW[:, None]) ** 3.46)
        tau_series = jnp.sum(jnp.where(masks, contrib, 0.0), axis=0)

        # Metal blanketing on Ly-α only.
        mask_lya = lam < self._LYW[0]
        tau_metal = (
            self._A_METAL * (lobs / self._LYW[0]) ** 1.68
        ) * mask_lya

        # ─── Lyman-continuum photoelectric absorption (Madau Eq. 16) ─
        mask_lyc = lam < self._LYLIM
        # Compute over the full grid with a safe xc (≥ 1 outside the
        # mask) so no NaN/Inf leaks through the reverse-mode tape; then
        # mask to zero outside the valid region.
        xc_safe = jnp.where(mask_lyc, xc, 1.0)
        tau_lyc = (
             0.25  * xc_safe**3.0  * (z1**0.46 - xc_safe**0.46)
            + 9.4  * xc_safe**1.5  * (z1**0.18 - xc_safe**0.18)
            - 0.7  * xc_safe**3.0  * (xc_safe**(-1.32) - z1**(-1.32))
            - 0.023 * (z1**1.68 - xc_safe**1.68)
        )
        tau_lyc = jnp.where(mask_lyc, tau_lyc, 0.0)

        tau = tau_series + tau_metal + tau_lyc

        # ─── Cap τ at its short-wavelength maximum ──────────────────
        # The LyC fit fails below ~100 Å and can produce a spuriously
        # decreasing τ.  FSPS freezes τ at the peak for all shorter
        # wavelengths; we do the same.  Assumes ``lam_rest`` is sorted
        # ascending (the FSPS / ceridwen convention).
        idx_peak = jnp.argmax(tau)
        tau_peak = tau[idx_peak]
        idxs = jnp.arange(tau.shape[0])
        tau = jnp.where(idxs <= idx_peak, tau_peak, tau)

        # Gate at zred = 0: the approximation is not valid for a zero-
        # length line of sight.  ``jnp.where`` on a scalar predicate
        # constant-folds when ``zred`` is a literal 0, so this does
        # not cost anything in the typical fixed-z case.
        return jnp.where(zred > 0.0, tau, jnp.zeros_like(tau))


# ======================================================================
# Factory
# ======================================================================

_MODEL_REGISTRY = {
    "madau1995": Madau1995,
    "none":      NoIGM,
}


def make_igm_model(name_or_model):
    """Build an :class:`IGMModel` from a string name, an instance, or
    ``None``.

    Parameters
    ----------
    name_or_model : str, IGMModel, or None
        String name (key of :data:`_MODEL_REGISTRY`), an already-built
        :class:`IGMModel` instance (returned as-is), or ``None``
        (returns :class:`NoIGM`).

    Examples
    --------
    >>> make_igm_model("madau1995")      # -> Madau1995()
    >>> make_igm_model(None)             # -> NoIGM()
    >>> make_igm_model(Madau1995())      # -> same instance
    """
    if name_or_model is None:
        return NoIGM()
    if isinstance(name_or_model, IGMModel):
        return name_or_model
    name = str(name_or_model).lower().strip()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown IGM model {name!r}.  "
            f"Available: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name]()


__all__ = ["IGMModel", "NoIGM", "Madau1995", "make_igm_model"]
