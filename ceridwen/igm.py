"""
Intergalactic-medium absorption models: transmission curves ``exp(-tau * factor)``
on the rest-frame wavelength grid at a given source redshift.
"""
from __future__ import annotations

import abc
from typing import Union

import jax
import jax.numpy as jnp

Array = jax.Array


class IGMModel(abc.ABC):
    """Abstract IGM attenuation model; subclasses implement ``tau(lam_rest, zred)``."""

    name: str = "igm"

    @abc.abstractmethod
    def tau(self, lam_rest: Array, zred: Array) -> Array:
        """Optical depth on the rest-frame wavelength grid [Å] at source redshift ``zred``."""

    def attenuation(self, lam_rest: Array, zred: Array,
                    factor: Union[Array, float] = 1.0) -> Array:
        """Transmission curve ``exp(-tau * factor)``."""
        return jnp.exp(-self.tau(lam_rest, zred) * factor)


class NoIGM(IGMModel):
    """Identity IGM: zero optical depth at every wavelength."""

    name = "none"

    def tau(self, lam_rest, zred):
        return jnp.zeros_like(lam_rest)


class Madau1995(IGMModel):
    """Madau (1995) IGM attenuation: 17 Lyman-series lines, Ly-α metal blanketing and
    Lyman-continuum absorption; tau is capped at its short-wavelength peak and is zero at zred = 0."""

    name = "madau1995"

    # Lyman-series rest wavelengths [Å], Ly-α first
    _LYW = jnp.asarray([
        1215.67, 1025.72,  972.537, 949.743, 937.803,
         930.748, 926.226, 923.150, 920.963, 919.352,
         918.129, 917.181, 916.429, 915.824, 915.329,
         914.919, 914.576,
    ])

    _LYCOEFF = jnp.asarray([
        0.0036,    0.0017,    0.0011846, 0.0009410, 0.0007960,
        0.0006967, 0.0006236, 0.0005665, 0.0005200, 0.0004817,
        0.0004487, 0.0004200, 0.0003947, 0.000372,  0.000352,
        0.0003334, 0.00031644,
    ])

    _LYLIM   = 911.75    # Lyman limit [Å]
    _A_METAL = 0.0017

    def tau(self, lam_rest, zred):
        zred = jnp.maximum(zred, 0.0)
        lam  = lam_rest
        z1   = 1.0 + zred
        ratio = lam[None, :] / self._LYW[:, None]
        series = jnp.sum(jnp.where(lam[None, :] < self._LYW[:, None],
                                   self._LYCOEFF[:, None] * ratio ** 3.46, 0.0), axis=0)
        metal = self._A_METAL * (lam / self._LYW[0]) ** 1.68 * (lam < self._LYW[0])
        tau_series = series * z1 ** 3.46
        tau_metal  = metal * z1 ** 1.68

        mask_lyc = lam < self._LYLIM
        xc_safe = jnp.where(mask_lyc, lam / self._LYLIM * z1, 1.0)
        tau_lyc = (
             0.25  * xc_safe**3.0  * (z1**0.46 - xc_safe**0.46)
            + 9.4  * xc_safe**1.5  * (z1**0.18 - xc_safe**0.18)
            - 0.7  * xc_safe**3.0  * (xc_safe**(-1.32) - z1**(-1.32))
            - 0.023 * (z1**1.68 - xc_safe**1.68)
        )
        tau_lyc = jnp.where(mask_lyc, tau_lyc, 0.0)
        tau = tau_series + tau_metal + tau_lyc

        idx_peak = jnp.argmax(tau)
        tau = jnp.where(jnp.arange(tau.shape[0]) <= idx_peak, tau[idx_peak], tau)
        return jnp.where(zred > 0.0, tau, jnp.zeros_like(tau))


_MODEL_REGISTRY = {
    "madau1995": Madau1995,
    "none":      NoIGM,
}


def make_igm_model(name_or_model):
    """Return an :class:`IGMModel` from a registry name, an instance (as-is), or ``None`` (NoIGM)."""
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
