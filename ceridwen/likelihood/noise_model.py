"""Stateless noise models mapping (sigma_obs, mu, mask, params) to per-datum
inverse variance and log-normalisation for Gaussian likelihood kernels."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

Array = jax.Array

_HALF_LOG_2PI: Array = 0.5 * jnp.log(2.0 * jnp.pi)


class NoiseModelOutput:
    """Per-datum ``inv_var`` (1/sigma_eff^2) and ``log_det`` (0.5*log(2 pi sigma_eff^2)),
    both shape (n_data,)."""

    __slots__ = ("inv_var", "log_det")

    def __init__(self, inv_var: Array, log_det: Array) -> None:
        self.inv_var = inv_var
        self.log_det = log_det

    def __repr__(self) -> str:
        return (
            f"NoiseModelOutput("
            f"inv_var={self.inv_var}, "
            f"log_det={self.log_det})"
        )


jax.tree_util.register_pytree_node(
    NoiseModelOutput,
    flatten_func=lambda o: ([o.inv_var, o.log_det], None),
    unflatten_func=lambda _, leaves: NoiseModelOutput(
        inv_var=leaves[0], log_det=leaves[1]
    ),
)


class NoiseModelBase(abc.ABC):
    """Abstract noise model; subclasses implement ``compute`` as a pure,
    JIT-safe, differentiable JAX function."""

    @abc.abstractmethod
    def compute(
        self,
        sigma_obs: Array,
        mu: Array,
        mask: Array,
        params: Optional[dict[str, Array]] = None,
        data: Optional[Array] = None,
    ) -> NoiseModelOutput:
        """Return ``NoiseModelOutput`` for 1-sigma ``sigma_obs`` (data units), model ``mu``,
        bool ``mask``, nuisance ``params`` from theta, and optional observed ``data``;
        masked points must get a finite ``inv_var``."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class DiagonalNoiseModel(NoiseModelBase):
    """Independent Gaussian noise: sigma_obs^2 (optionally rescaled by a common
    factor) plus optional (f_calib*|mu|)^2, (f_data*|y|)^2 and jitter^2 terms,
    nuisance parameters sampled in log space.

    Parameters
    ----------
    use_error_scale : bool -- multiply sigma_obs^2 by exp(params["log_err_scale"])^2, a
        common rescaling of the quoted uncertainties (the log-normalisation term keeps it
        from running away)
    use_jitter : bool -- add exp(params["log_jitter"])^2 (data units)
    use_fractional : bool -- add (exp(params["log_f_calib"]) * |mu|)^2, model-anchored
    use_data_fractional : bool -- add (exp(params["log_f_data"]) * |y|)^2, data-anchored, needs ``data``
    noise_floor : float -- fixed fractional floor on |mu|
    """

    use_jitter          : bool = False
    use_fractional      : bool = False
    use_data_fractional : bool = False
    noise_floor         : float = 0.0
    use_error_scale     : bool = False

    def compute(
        self,
        sigma_obs : Array,
        mu        : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
        data      : Optional[Array] = None,
    ) -> NoiseModelOutput:
        """Return ``NoiseModelOutput``; ``params`` keys ``log_err_scale``/``log_jitter``/
        ``log_f_calib``/``log_f_data`` as configured, ``data`` (observed y) required when
        ``use_data_fractional``."""
        if params is None:
            params = {}

        var: Array = sigma_obs ** 2
        if self.use_error_scale:
            var = var * jnp.exp(2.0 * params["log_err_scale"])
        if self.noise_floor > 0.0:
            var = var + (self.noise_floor * jnp.abs(mu)) ** 2

        if self.use_fractional:
            f_calib = jnp.exp(params["log_f_calib"])
            var = var + (f_calib * jnp.abs(mu)) ** 2

        if self.use_data_fractional:
            if data is None:
                raise ValueError(
                    "DiagonalNoiseModel(use_data_fractional=True) requires the "
                    "observed data array; call compute(..., data=y)."
                )
            f_data = jnp.exp(params["log_f_data"])
            var = var + (f_data * jnp.abs(data)) ** 2

        if self.use_jitter:
            jitter = jnp.exp(params["log_jitter"])
            var = var + jitter ** 2

        var = jnp.where(mask, var, jnp.ones_like(var))

        var = jnp.maximum(var, jnp.finfo(var.dtype).tiny)

        inv_var = 1.0 / var
        log_det = 0.5 * jnp.log(var) + _HALF_LOG_2PI

        return NoiseModelOutput(inv_var=inv_var, log_det=log_det)

    @property
    def nuisance_param_names(self) -> tuple[str, ...]:
        """Names of nuisance parameters expected in ``params`` at compute time."""
        names: list[str] = []
        if self.use_error_scale:
            names.append("log_err_scale")
        if self.use_fractional:
            names.append("log_f_calib")
        if self.use_data_fractional:
            names.append("log_f_data")
        if self.use_jitter:
            names.append("log_jitter")
        return tuple(names)

    def __repr__(self) -> str:
        return (
            f"DiagonalNoiseModel("
            f"use_jitter={self.use_jitter}, "
            f"use_fractional={self.use_fractional}, "
            f"use_data_fractional={self.use_data_fractional}, "
            f"noise_floor={self.noise_floor}, "
            f"use_error_scale={self.use_error_scale})"
        )


jax.tree_util.register_pytree_node(
    DiagonalNoiseModel,
    flatten_func=lambda nm: (
        [],
        (nm.use_jitter, nm.use_fractional, nm.use_data_fractional, nm.noise_floor,
         nm.use_error_scale),
    ),
    unflatten_func=lambda aux, _: DiagonalNoiseModel(
        use_jitter=aux[0], use_fractional=aux[1], use_data_fractional=aux[2],
        noise_floor=aux[3], use_error_scale=aux[4],
    ),
)
