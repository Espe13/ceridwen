from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple
import jax
import numpy as np
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

Array = jax.Array

__all__ = ["Prior", "Uniform", "TopHat", "Normal", "MultivariateNormalPrior", "ClippedNormal",
           "LogNormal", "LogUniform", "StudentT"]


@dataclass(frozen=True)
class Prior(abc.ABC):
    """Prior base class delegating to a TFP-JAX distribution; subclasses define
    ``prior_params`` and implement ``tfp_dist()``."""

    alias: Dict[str, str] = field(init=False)
    params: Dict[str, Array] = field(init=False)
    name: str = ""

    def __init__(self,
                 parnames: Sequence[str] = (),
                 name: str = "",
                 **kwargs: Any):

        prior_params: Tuple[str, ...] = getattr(type(self), "prior_params", None)
        if prior_params is None:
            raise ValueError(
                f"{type(self).__name__} must define class attribute "
                "`prior_params = (\"param1\", \"param2\", ...)`."
            )

        if not parnames:
            parnames = prior_params

        if len(parnames) != len(prior_params):
            raise ValueError(
                f"parnames has length {len(parnames)} but prior_params has "
                f"length {len(prior_params)}"
            )

        alias = dict(zip(prior_params, parnames))

        params: Dict[str, Array] = {}
        for intrinsic, external in alias.items():
            if external in kwargs:
                params[intrinsic] = jnp.asarray(kwargs.pop(external))
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unknown argument(s) {sorted(kwargs)}; "
                f"it takes {list(alias.values())}")
        missing = [external for intrinsic, external in alias.items() if intrinsic not in params]
        if missing:
            raise TypeError(f"{type(self).__name__} is missing argument(s) {missing}")

        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "name", name)

    def __repr__(self) -> str:
        argstring = [f"{k}={v}" for k, v in self.params.items()]
        return f"{type(self).__name__}({', '.join(argstring)})"

    def serialize(self) -> dict:
        """JSON-ready description: type, parameters (lists for arrays), name."""
        out = {"type": type(self).__name__, "name": self.name}
        for k, v in self.params.items():
            a = np.asarray(v)
            out[k] = a.tolist() if a.ndim else float(a)
        return out

    def __len__(self) -> int:
        prior_params = getattr(type(self), "prior_params", ())
        if not prior_params:
            return 1
        return int(max(jnp.size(self.params.get(k, jnp.array(1.0)))
                       for k in prior_params))


    @abc.abstractmethod
    def tfp_dist(self) -> tfd.Distribution:
        """Return the TFP-JAX distribution built from ``self.params``."""
        raise NotImplementedError

    def logpdf(self, x: Array) -> Array:
        return self.tfp_dist().log_prob(x)

    def __call__(self, x: Array) -> Array:
        return self.logpdf(x)


    def sample(self,
               key: Array,
               shape: Tuple[int, ...] | None = None) -> Array:
        return self._sample_impl(key, shape)

    def _sample_impl(self, key: Array, shape: Tuple[int, ...]) -> Array:
        if shape is None:
            shape = ()
        return self.tfp_dist().sample(seed=key, sample_shape=shape)

    def unit_transform(self, x: Array) -> Array:
        return self._ppf(x)

    def inverse_unit_transform(self, x: Array) -> Array:
        return self._cdf(x)

    def _ppf(self, x: Array) -> Array:
        return self.tfp_dist().quantile(x)

    def _cdf(self, x: Array) -> Array:
        return self.tfp_dist().cdf(x)

    def gradient(self, theta: Array) -> Array:
        def logp(t: Array) -> Array:
            return self(t).sum()
        return jax.grad(logp)(theta)


class Uniform(Prior):
    """Uniform distribution on [low, high]."""
    prior_params = ("low", "high")

    def tfp_dist(self) -> tfd.Distribution:
        low = self.params.get("low")
        high = self.params.get("high")
        return tfd.Uniform(low=low, high=high)

    @property
    def range(self):
        return self.params["low"], self.params["high"]

    @property
    def bounds(self):
        return self.params["low"], self.params["high"]

    def serialize(self):
        return {
            "type": "Uniform",
            "low": float(self.params["low"]),
            "high": float(self.params["high"]),
            "name": self.name,
        }


class TopHat(Uniform):
    """Alias of Uniform kept for backwards compatibility."""


class Normal(Prior):
    """Gaussian prior with parameters mean, sigma."""
    prior_params = ['mean', 'sigma']


    def tfp_dist(self) -> tfd.Distribution:
        mean = self.params.get("mean")
        sigma = self.params.get("sigma")
        return tfd.Normal(loc=mean, scale=sigma)

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['mean'] - nsig * self.params['sigma'],
                self.params['mean'] + nsig * self.params['sigma'])

    def bounds(self, **kwargs):
        return (-jnp.inf, jnp.inf)


class MultivariateNormalPrior(Prior):
    """Multivariate Gaussian prior.

    Parameters
    ----------
    mean : (d,)
    Sigma : (d, d) -- covariance matrix
    """
    prior_params = ("mean", "Sigma")

    def tfp_dist(self) -> tfd.Distribution:
        mean = self.params["mean"]
        Sigma = self.params["Sigma"]
        return tfd.MultivariateNormalFullCovariance(
            loc=mean,
            covariance_matrix=Sigma
        )

    @property
    def range(self):
        """Mean -/+ 4 sigma per dimension (plotting range)."""
        mu = self.params["mean"]
        Sigma = self.params["Sigma"]
        sigma_diag = jnp.sqrt(jnp.diag(Sigma))
        nsig = 4.0
        return mu - nsig * sigma_diag, mu + nsig * sigma_diag

    @property
    def bounds(self):
        dim = self.params["mean"].shape[0]
        return (
            -jnp.inf * jnp.ones(dim),
            +jnp.inf * jnp.ones(dim)
        )

    def sample(self, key, shape=None):
        if shape is None:
            shape = ()
        return self.tfp_dist().sample(seed=key, sample_shape=shape)

    def unit_transform(self, u):
        z = tfd.Normal(0.0, 1.0).quantile(u)

        L = jnp.linalg.cholesky(self.params["Sigma"])

        return self.params["mean"] + L @ z

    def inverse_unit_transform(self, theta):
        mu = self.params["mean"]
        Sigma = self.params["Sigma"]

        L = jnp.linalg.cholesky(Sigma)

        z = jax.scipy.linalg.solve_triangular(L, theta - mu, lower=True)

        standard_normal = tfd.Normal(0.0, 1.0)
        return standard_normal.cdf(z)

    def serialize(self):
        return {
            "type": "MultivariateNormalPrior",
            "mean": jnp.asarray(self.params["mean"]).tolist(),
            "Sigma": jnp.asarray(self.params["Sigma"]).tolist(),
            "name": self.name,
        }


class ClippedNormal(Prior):
    """Gaussian prior truncated to [low, high]; parameters mean, sigma, low, high."""
    prior_params = ['mean', 'sigma', 'low', 'high']
    def tfp_dist(self) -> tfd.Distribution:
        mean = self.params.get("mean")
        sigma = self.params.get("sigma")
        low = self.params.get("low")
        high = self.params.get("high")
        return tfd.TruncatedNormal(loc=mean, scale=sigma, low=low, high=high)

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return (self.params['low'], self.params['high'])

    @property
    def args(self):
        a = (self.params['low'] - self.params['mean']) / self.params['sigma']
        b = (self.params['high'] - self.params['mean']) / self.params['sigma']
        return [a, b]

    @property
    def bounds(self):
        return self.params["low"], self.params["high"]


class LogNormal(Prior):
    """Log-normal prior; ``mode`` and ``sigma`` are the mean and std of ln(x)."""
    prior_params = ['mode', 'sigma']
    def tfp_dist(self) -> tfd.Distribution:
        mode = self.params.get("mode")
        sigma = self.params.get("sigma")
        return tfd.LogNormal(loc=mode, scale=sigma)

    @property
    def args(self):
        return [self.params["sigma"]]

    @property
    def scale(self):
        return  jnp.exp(self.params["mode"] + self.params["sigma"]**2)

    @property
    def loc(self):
        return 0

    @property
    def range(self):
        nsig = 4
        return (jnp.exp(self.params['mode'] - (nsig * self.params['sigma'])),
                jnp.exp(self.params['mode'] + (nsig * self.params['sigma'])))

    def bounds(self, **kwargs):
        return (0, jnp.inf)


class LogUniform(Prior):
    """Log-uniform (reciprocal, Jeffreys) prior on [mini, maxi], 0 < mini < maxi < inf.

    pdf ``1 / (x ln(maxi/mini))`` on ``[mini, maxi]`` (``-inf`` log-density outside),
    CDF ``ln(x/mini) / ln(maxi/mini)``: uniform in ``log x``, whatever the base.
    Same parameter names and distribution as Prospector's ``LogUniform(mini, maxi)``
    (``scipy.stats.reciprocal(mini, maxi)``).  The parameter itself (not its log) is
    sampled; ``fitSED``'s NUTS path maps it to ``(mini, maxi)`` with a logit.

    ``logpdf``/``cdf``/``ppf``/``sample`` are analytic ``jnp`` (exact, float64,
    jit/grad/vmap-safe); ``tfp_dist()`` gives the equivalent TFP distribution
    (``Exp`` of a ``Uniform(ln mini, ln maxi)``) for interoperability.
    """
    prior_params = ("mini", "maxi")

    def __init__(self, parnames: Sequence[str] = (), name: str = "", **kwargs: Any):
        super().__init__(parnames=parnames, name=name, **kwargs)
        mini = np.asarray(self.params["mini"], dtype=np.float64)
        maxi = np.asarray(self.params["maxi"], dtype=np.float64)
        if not (np.all(np.isfinite(mini)) and np.all(np.isfinite(maxi))
                and np.all(mini > 0.0) and np.all(maxi > mini)):
            raise ValueError(
                f"LogUniform needs 0 < mini < maxi < inf (finite, elementwise), got "
                f"mini={mini.tolist()}, maxi={maxi.tolist()}.  For a parameter that can be "
                "<= 0 sample its log with a Uniform instead.")
        object.__setattr__(self, "params", {"mini": jnp.asarray(mini), "maxi": jnp.asarray(maxi)})

    def _log_limits(self):
        lna = jnp.log(self.params["mini"])
        lnb = jnp.log(self.params["maxi"])
        return lna, lnb

    def tfp_dist(self) -> tfd.Distribution:
        lna, lnb = self._log_limits()
        return tfd.TransformedDistribution(distribution=tfd.Uniform(low=lna, high=lnb),
                                           bijector=tfp.bijectors.Exp())

    def logpdf(self, x: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        a, b = self.params["mini"], self.params["maxi"]
        lna, lnb = self._log_limits()
        inside = (x >= a) & (x <= b)
        x_safe = jnp.where(inside, x, a)          # no log of <= 0 in the dead branch (grad-safe)
        return jnp.where(inside, -jnp.log(x_safe) - jnp.log(lnb - lna), -jnp.inf)

    def _cdf(self, x: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        a, b = self.params["mini"], self.params["maxi"]
        lna, lnb = self._log_limits()
        x_safe = jnp.clip(x, a, b)
        c = (jnp.log(x_safe) - lna) / (lnb - lna)
        return jnp.where(x >= b, 1.0, jnp.where(x <= a, 0.0, c))   # exact 0 / 1 at the ends

    def _ppf(self, u: Array) -> Array:
        u = jnp.asarray(u, dtype=jnp.float64)
        lna, lnb = self._log_limits()
        return jnp.exp(lna + u * (lnb - lna))

    def _sample_impl(self, key: Array, shape: Tuple[int, ...] | None) -> Array:
        if shape is None:
            shape = ()
        full = tuple(shape) + tuple(jnp.shape(self.params["mini"] * self.params["maxi"]))
        u = jax.random.uniform(key, full, dtype=jnp.float64)
        return self._ppf(u)

    @property
    def range(self):
        return self.params["mini"], self.params["maxi"]

    @property
    def bounds(self):
        return self.params["mini"], self.params["maxi"]


class StudentT(Prior):
    """Student's t prior with parameters mean, scale, df (degrees of freedom)."""
    prior_params = ['mean', 'scale', 'df']
    def tfp_dist(self) -> tfd.Distribution:
        mean = self.params.get("mean")
        scale = self.params.get("scale")
        df = self.params.get("df")
        return tfd.StudentT(df=df, loc=mean, scale=scale)

    @property
    def args(self):
        return [self.params['df']]

    @property
    def scale(self):
        return self.params['scale']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return self.tfp_dist().quantile(jnp.array([0.0025, 0.9975]))

    def bounds(self, **kwargs):
        return (-jnp.inf, jnp.inf)
