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
    use_error_scale : bool -- multiply sigma_obs^2 by exp(params[err_scale_key])^2, a
        common rescaling of the quoted uncertainties (the log-normalisation term keeps it
        from running away)
    use_jitter : bool -- add exp(params[jitter_key])^2 (data units)
    use_fractional : bool -- add (exp(params[f_calib_key]) * |mu|)^2, model-anchored
    use_data_fractional : bool -- add (exp(params[f_data_key]) * |y|)^2, data-anchored, needs ``data``
    noise_floor : float -- fixed fractional floor on |mu|
    f_outlier : None, float or str -- outlier mixture (Hogg, Bovy & Lang 2010): the fraction
        of data drawn from a Gaussian ``nsigma_outlier`` times broader than sigma_eff.  Default
        None = 0 = off: the mixture is on only when switched on explicitly, with a float in
        (0, 1) (fixed; 0.0 means off) or a str (the theta key of a sampled fraction).  Applied
        by the likelihood classes, not by ``compute``.
    nsigma_outlier : float or str -- width of the outlier Gaussian in units of sigma_eff,
        fixed (default 50, Prospector's) or a theta key
    err_scale_key, jitter_key, f_calib_key, f_data_key : str or float -- where each switched-on
        term reads its (log) value: a theta key (default the historical ``log_err_scale`` /
        ``log_jitter`` / ``log_f_calib`` / ``log_f_data``; ``fitSED`` passes the per-observation
        names ``log_jitter_<kind>[_<obs>]`` of v1.0.7) or a fixed float
    """

    use_jitter          : bool = False
    use_fractional      : bool = False
    use_data_fractional : bool = False
    noise_floor         : float = 0.0
    use_error_scale     : bool = False
    f_outlier           : Optional[float | str] = None
    nsigma_outlier      : float | str = 50.0
    err_scale_key       : float | str = "log_err_scale"
    jitter_key          : float | str = "log_jitter"
    f_calib_key         : float | str = "log_f_calib"
    f_data_key          : float | str = "log_f_data"

    def __post_init__(self) -> None:
        f, ns = self.f_outlier, self.nsigma_outlier
        if f is not None and not isinstance(f, str):
            if isinstance(f, bool) or not isinstance(f, (int, float)):
                raise TypeError(f"f_outlier must be None, a float or a theta key, got {f!r}")
            if not 0.0 <= float(f) < 1.0:
                raise ValueError(f"a fixed f_outlier must lie in [0, 1), got {f!r}")
            # f = 0 is the ordinary Gaussian: switch the mixture off so the graph is unchanged
            object.__setattr__(self, "f_outlier", None if float(f) == 0.0 else float(f))
        if isinstance(f, str) and not f:
            raise ValueError("f_outlier theta key must be a non-empty string")
        if not isinstance(ns, str):
            if isinstance(ns, bool) or not isinstance(ns, (int, float)) or not float(ns) > 0.0:
                raise ValueError(f"a fixed nsigma_outlier must be a positive number, got {ns!r}")
            object.__setattr__(self, "nsigma_outlier", float(ns))
        elif not ns:
            raise ValueError("nsigma_outlier theta key must be a non-empty string")
        for k in ("err_scale_key", "jitter_key", "f_calib_key", "f_data_key"):
            v = getattr(self, k)
            if isinstance(v, str):
                if not v:
                    raise ValueError(f"{k} must be a non-empty theta key or a float")
            elif isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{k} must be a theta key (str) or a fixed float, got {v!r}")
            else:
                object.__setattr__(self, k, float(v))

    @property
    def use_outlier(self) -> bool:
        """True when the outlier mixture is on (static: it selects the likelihood kernel)."""
        return self.f_outlier is not None

    @property
    def outlier_param_names(self) -> tuple[str, ...]:
        """theta keys of the sampled outlier parameters (empty when fixed or off)."""
        if not self.use_outlier:
            return ()
        return tuple(v for v in (self.f_outlier, self.nsigma_outlier) if isinstance(v, str))

    def outlier_params(self, params: Optional[dict[str, Array]]) -> tuple[Array, Array]:
        """``(f, nsigma)`` for the mixture kernel: the fixed values, or looked up in ``params``."""
        def get(v, what):
            if not isinstance(v, str):
                return v
            if params is None or v not in params:
                raise KeyError(f"the outlier model reads {what} from theta[{v!r}], which is "
                               "missing: sample it (with a prior) or give a fixed value")
            return params[v]
        return get(self.f_outlier, "f_outlier"), get(self.nsigma_outlier, "nsigma_outlier")

    def compute(
        self,
        sigma_obs : Array,
        mu        : Array,
        mask      : Array,
        params    : Optional[dict[str, Array]] = None,
        data      : Optional[Array] = None,
    ) -> NoiseModelOutput:
        """Return ``NoiseModelOutput``; ``params`` holds the theta keys ``err_scale_key`` /
        ``jitter_key`` / ``f_calib_key`` / ``f_data_key`` of the switched-on terms (a float key
        is a fixed value), ``data`` (observed y) required when ``use_data_fractional``."""
        if params is None:
            params = {}

        def get(key):
            return params[key] if isinstance(key, str) else key

        var: Array = sigma_obs ** 2
        if self.use_error_scale:
            var = var * jnp.exp(2.0 * get(self.err_scale_key))
        if self.noise_floor > 0.0:
            var = var + (self.noise_floor * jnp.abs(mu)) ** 2

        if self.use_fractional:
            f_calib = jnp.exp(get(self.f_calib_key))
            var = var + (f_calib * jnp.abs(mu)) ** 2

        if self.use_data_fractional:
            if data is None:
                raise ValueError(
                    "DiagonalNoiseModel(use_data_fractional=True) requires the "
                    "observed data array; call compute(..., data=y)."
                )
            f_data = jnp.exp(get(self.f_data_key))
            var = var + (f_data * jnp.abs(data)) ** 2

        if self.use_jitter:
            jitter = jnp.exp(get(self.jitter_key))
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
        for on, key in ((self.use_error_scale, self.err_scale_key),
                        (self.use_fractional, self.f_calib_key),
                        (self.use_data_fractional, self.f_data_key),
                        (self.use_jitter, self.jitter_key)):
            if on and isinstance(key, str):
                names.append(key)
        return tuple(names)

    def __repr__(self) -> str:
        return (
            f"DiagonalNoiseModel("
            f"use_jitter={self.use_jitter}, "
            f"use_fractional={self.use_fractional}, "
            f"use_data_fractional={self.use_data_fractional}, "
            f"noise_floor={self.noise_floor}, "
            f"use_error_scale={self.use_error_scale}"
            + (f", f_outlier={self.f_outlier!r}, nsigma_outlier={self.nsigma_outlier!r}"
               if self.use_outlier else "")
            + "".join(f", {k}={getattr(self, k)!r}" for k, d in _DEFAULT_KEYS.items()
                      if getattr(self, k) != d)
            + ")"
        )


_DEFAULT_KEYS = {"err_scale_key": "log_err_scale", "jitter_key": "log_jitter",
                 "f_calib_key": "log_f_calib", "f_data_key": "log_f_data"}


jax.tree_util.register_pytree_node(
    DiagonalNoiseModel,
    flatten_func=lambda nm: (
        [],
        (nm.use_jitter, nm.use_fractional, nm.use_data_fractional, nm.noise_floor,
         nm.use_error_scale, nm.f_outlier, nm.nsigma_outlier,
         nm.err_scale_key, nm.jitter_key, nm.f_calib_key, nm.f_data_key),
    ),
    unflatten_func=lambda aux, _: DiagonalNoiseModel(
        use_jitter=aux[0], use_fractional=aux[1], use_data_fractional=aux[2],
        noise_floor=aux[3], use_error_scale=aux[4], f_outlier=aux[5], nsigma_outlier=aux[6],
        err_scale_key=aux[7], jitter_key=aux[8], f_calib_key=aux[9], f_data_key=aux[10],
    ),
)
