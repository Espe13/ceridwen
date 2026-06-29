from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Sequence, Tuple
import time
import jax
import jax.numpy as jnp
from jax import random

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

Array = jax.Array

__all__ = ["Prior", "Uniform", "TopHat", "Normal", "MultiVariateNormal", "ClippedNormal",
           "LogNormal", "StudentT"]


@dataclass(frozen=True)
class Prior(abc.ABC):
    """
    JAX-friendly prior base class that delegates all probability operations to
    a TFP-JAX distribution. Subclasses must implement `tfp_dist()`.
    """

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
                params[intrinsic] = jnp.asarray(kwargs[external])

        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "name", name)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        argstring = [f"{k}={v}" for k, v in self.params.items()]
        return f"{type(self).__name__}({', '.join(argstring)})"
    
    def __len__(self) -> int:
        prior_params = getattr(type(self), "prior_params", ())
        if not prior_params:
            return 1
        return int(max(jnp.size(self.params.get(k, jnp.array(1.0)))
                       for k in prior_params))
    

    # ------------------------------------------------------------------
    # Subclasses *must* provide a TFP distribution
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def tfp_dist(self) -> tfd.Distribution:
        """
        Return a TFP-JAX distribution object built from `self.params`.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    # ------------------------------------------------------------------
    # Probability interface (delegates to TFP)
    # ------------------------------------------------------------------
    def logpdf(self, x: Array) -> Array:
        return self.tfp_dist().log_prob(x)

    def __call__(self, x: Array) -> Array:
        return self.logpdf(x)



    # ------------------------------------------------------------------
    # Sampling and transforms via TFP
    # ------------------------------------------------------------------
    def sample(self,
               key: jax.random.KeyArray,
               shape: Tuple[int, ...] | None = None) -> Array:
        return self._sample_impl(key, shape)

    def _sample_impl(self, key: jax.random.KeyArray, shape: Tuple[int, ...]) -> Array:
        # TFP wants a concrete sample_shape; None triggers a deprecation warning.
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
    
    # ------------------------------------------------------------------
    # Auto-diff gradient
    # ------------------------------------------------------------------
    def gradient(self, theta: Array) -> Array:
        def logp(t: Array) -> Array:
            return self(t).sum()
        return jax.grad(logp)(theta)
    








class Uniform(Prior):
    """
    Uniform distribution on [low, high].
    """
    # Intrinsic parameter names:
    prior_params = ("low", "high")

    def tfp_dist(self) -> tfd.Distribution:
        low = self.params.get("low")
        high = self.params.get("high")
        return tfd.Uniform(low=low, high=high)
    
    @property
    def range(self):
        # Plotting range is the interval itself
        return self.params["low"], self.params["high"]
    
    @property
    def bounds(self):
        # Hard support boundaries
        return self.params["low"], self.params["high"]
    
    def serialize(self):
        return {
            "type": "Uniform",
            "low": float(self.params["low"]),
            "high": float(self.params["high"]),
            "name": self.name,
        }
    

class TopHat(Uniform):
    """Uniform distribution between two bounds, renamed for backwards compatibility
    :param low:
        Minimum of the distribution

    :param high:
        Maximum of the distribution
    """


class Normal(Prior):
    """A simple gaussian prior.


    :param mean:
        Mean of the distribution

    :param sigma:
        Standard deviation of the distribution
    """
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
    """
    Multivariate Gaussian prior using TFP-JAX.
    Intrinsic parameters:
      - mean   : vector (d,)
      - Sigma  : covariance matrix (d,d)
    """
    prior_params = ("mean", "Sigma")

    # ---------------------------------------------------------
    # Build TFP distribution
    # ---------------------------------------------------------
    def tfp_dist(self) -> tfd.Distribution:
        mean = self.params["mean"]
        Sigma = self.params["Sigma"]
        return tfd.MultivariateNormalFullCovariance(
            loc=mean,
            covariance_matrix=Sigma
        )

    # ---------------------------------------------------------
    # Plotting helpers
    # ---------------------------------------------------------
    @property
    def range(self):
        """
        Return mean ± 4σ along diagonal directions.
        Good for plotting marginal bands.
        """
        mu = self.params["mean"]
        Sigma = self.params["Sigma"]
        sigma_diag = jnp.sqrt(jnp.diag(Sigma))
        nsig = 4.0
        return mu - nsig * sigma_diag, mu + nsig * sigma_diag

    @property
    def bounds(self):
        """
        MVN has infinite support.
        """
        dim = self.params["mean"].shape[0]
        return (
            -jnp.inf * jnp.ones(dim),
            +jnp.inf * jnp.ones(dim)
        )

    # ---------------------------------------------------------
    # Sampling (TFP handles batching)
    # ---------------------------------------------------------
    def sample(self, key, shape=None):
        if shape is None:
            shape = ()  # draw 1 vector
        return self.tfp_dist().sample(seed=key, sample_shape=shape)

    # ---------------------------------------------------------
    # Unit transform: u in (0,1)^d → θ in R^d
    #
    # θ = μ + L @ z     where z = N^{-1}(u)
    # L = chol(Sigma)
    # ---------------------------------------------------------
    def unit_transform(self, u):
        # inverse CDF of standard normal
        z = tfd.Normal(0.0, 1.0).quantile(u)

        # Cholesky factor L where Sigma = L Lᵀ
        L = jnp.linalg.cholesky(self.params["Sigma"])

        # θ = μ + L @ z
        return self.params["mean"] + L @ z

    # ---------------------------------------------------------
    # Inverse unit transform: θ → u in (0,1)^d
    #
    # u_i = Phi( (L^{-1} (θ - μ))_i )
    #
    # MVN CDF itself is expensive; this gives dimension-wise transform.
    # ---------------------------------------------------------
    def inverse_unit_transform(self, theta):
        mu = self.params["mean"]
        Sigma = self.params["Sigma"]

        # L (Cholesky)
        L = jnp.linalg.cholesky(Sigma)

        # Solve L z = theta - mu
        z = jax.scipy.linalg.solve_triangular(L, theta - mu, lower=True)

        # Convert z to uniform using Φ(z)
        standard_normal = tfd.Normal(0.0, 1.0)
        return standard_normal.cdf(z)

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------
    def serialize(self):
        return {
            "type": "MultivariateNormalPrior",
            "mean": jnp.asarray(self.params["mean"]).tolist(),
            "Sigma": jnp.asarray(self.params["Sigma"]).tolist(),
            "name": self.name,
        }
    


class ClippedNormal(Prior):
    """A Gaussian prior clipped to some range.

    :param mean:
        Mean of the normal distribution

    :param sigma:
        Standard deviation of the normal distribution

    :param low:
        Minimum of the distribution

    :param high:
        Maximum of the distribution
    """
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
        # Hard support boundaries
        return self.params["low"], self.params["high"]
    



class LogNormal(Prior):
    """A log-normal prior, where the natural log of the variable is distributed
    normally.  Useful for parameters that cannot be less than zero.

    Note that ``LogNormal(np.exp(mode) / f) == LogNormal(np.exp(mode) * f)``
    and ``f = np.exp(sigma)`` corresponds to "one sigma" from the peak.

    :param mode:
        Natural log of the variable value at which the probability density is
        highest.

    :param sigma:
        Standard deviation of the distribution of the natural log of the
        variable.
    """
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
        return (jnp.exp(self.params['mode'] + (nsig * self.params['sigma'])),
                jnp.exp(self.params['mode'] - (nsig * self.params['sigma'])))

    def bounds(self, **kwargs):
        return (0, jnp.inf)
    

class StudentT(Prior):
    """A Student's T distribution

    :param mean:
        Mean of the distribution

    :param scale:
        Size of the distribution, analogous to the standard deviation

    :param df:
        Number of degrees of freedom
    """
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