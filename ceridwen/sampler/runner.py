"""Sampler-agnostic result container, adapter protocol, and ``run_sampler``
dispatch that builds separate JIT log-likelihood and log-prior callables."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass
class SamplingResult:
    """Backend-agnostic container for posterior samples, evidence and diagnostics.

    Parameters
    ----------
    samples : dict[str, Array] -- per-parameter, shape (n_samples,) for scalars, (n_samples, k) for vectors
    log_evidence : float -- ln Z; NaN for MCMC
    log_evidence_err : float -- 1-sigma error on ln Z; NaN for MCMC
    log_weights : Array, (n_samples,) -- log importance weights; all zeros for MCMC
    log_likelihoods : Array, (n_samples,) -- ln L at each sample
    log_likelihoods_birth : Array or None, (n_samples,) -- nested-sampling birth ln L; None for MCMC
    raw : Any -- backend-specific raw output
    """

    samples               : dict[str, Array]
    log_evidence          : float
    log_evidence_err      : float
    log_weights           : Array
    log_likelihoods       : Array
    param_names           : list[str]
    n_likelihood_calls    : int
    wall_time_s           : float
    sampler_name          : str
    log_likelihoods_birth : Optional[Array] = None
    raw                   : Any = None

    def to_anesthetic(self, labels: Optional[dict[str, str]] = None):
        """Convert to a ``NestedSamples`` object; ``labels`` maps parameter name to LaTeX label."""
        try:
            from anesthetic import NestedSamples
        except ImportError as exc:
            raise ImportError(
                "anesthetic is required for to_anesthetic(). "
                "Install: pip install 'anesthetic>=2.8'"
            ) from exc

        import numpy as np

        columns, col_labels, data_cols = [], [], []

        for name in self.param_names:
            arr = np.asarray(self.samples[name])
            if arr.ndim == 1:
                columns.append(name)
                col_labels.append(labels.get(name, name) if labels else name)
                data_cols.append(arr[:, None])
            else:
                for i in range(arr.shape[1]):
                    col = f"{name}[{i}]"
                    columns.append(col)
                    col_labels.append(
                        f"{labels.get(name, name)}$_{{{i}}}$"
                        if labels else col
                    )
                    data_cols.append(arr[:, i : i + 1])

        data = np.hstack(data_cols)

        return NestedSamples(
            data,
            logL       = np.asarray(self.log_likelihoods),
            logL_birth = (
                np.asarray(self.log_likelihoods_birth)
                if self.log_likelihoods_birth is not None
                else None
            ),
            columns    = columns,
            labels     = col_labels,
            logzero    = float("nan"),
        )

    def summary(self) -> str:
        """One-line human-readable summary."""
        n = next(iter(self.samples.values())).shape[0]
        lz = (f"{self.log_evidence:.3f} ± {self.log_evidence_err:.3f}"
              if jnp.isfinite(self.log_evidence) else "n/a")
        return (
            f"{self.sampler_name}  |  n_samples={n}  "
            f"ln Z = {lz}  |  "
            f"n_like = {self.n_likelihood_calls}  |  "
            f"wall = {self.wall_time_s:.1f} s"
        )

    def __repr__(self) -> str:
        n = next(iter(self.samples.values())).shape[0]
        return (
            f"SamplingResult(sampler={self.sampler_name!r}, "
            f"n_samples={n}, ln_Z={self.log_evidence:.3f})"
        )


class SamplerAdapter(abc.ABC):
    """Abstract sampling backend; ``run`` receives separate log-likelihood and
    log-prior callables (kept apart so nested sampling can use both)."""

    @abc.abstractmethod
    def run(
        self,
        loglike_fn  : Callable[[dict[str, Array]], Array],
        logprior_fn : Callable[[dict[str, Array]], Array],
        theta_init  : dict[str, Array],
        rng_key     : Array,
    ) -> SamplingResult:
        """Run the sampler on ``theta_dict -> scalar`` callables and return a ``SamplingResult``."""


def run_sampler(
    model      : Any,
    likelihood : Any,
    adapter    : SamplerAdapter,
    rng_key    : Array,
    theta_init : dict[str, Array] | None = None,
) -> SamplingResult:
    """Build JIT ``loglike_fn`` (summed over observations, no prior) and
    ``logprior_fn`` from ``model``/``likelihood`` and delegate to ``adapter.run``.
    ``theta_init`` (default ``model.theta_init``) is the start handed to the adapter, e.g. a
    MAP from ``ceridwen.optimize.map_fit``; it must have the keys and shapes of
    ``model.theta_init``."""
    if theta_init is None:
        theta_init = model.theta_init
    else:
        want = {k: tuple(jnp.shape(v)) for k, v in model.theta_init.items()}
        got = {k: tuple(jnp.shape(v)) for k, v in theta_init.items()}
        if want != got:
            raise ValueError(f"run_sampler: theta_init has keys/shapes {got}, the model's "
                             f"free parameters are {want}")
        theta_init = {k: jnp.asarray(theta_init[k]) for k in model.theta_init}
    _obs_dict    = model.obs_dict
    _keys        = tuple(likelihood.keys)
    _likelihoods = tuple(likelihood.likelihoods)
    from ..likelihood.likelihood import (observation_data, float32_rescaling, rescaled_params,
                                         loglike_offset)
    for key in _keys:
        if getattr(_obs_dict[key], "flux", None) is None:
            raise ValueError(f"run_sampler: observation {key!r} has no flux; give it flux= and "
                             f"uncertainty= to fit it (without them it serves predictions only)")
    _static_data = {key: observation_data(_obs_dict[key]) for key in _keys}
    # float32 mode (docs/mac_gpu.md): the kernels run on rescaled data; ln L comes back offset
    # by a constant of the observations, added to the reported ln L / ln Z below (None in float64)
    _rescalings = {key: float32_rescaling(_obs_dict[key], lh) for key, lh in zip(_keys, _likelihoods)}
    _offset = loglike_offset(_rescalings.values())

    if getattr(model, "_eline_system", None) is not None:
        from ..likelihood.eline_marginal import joint_loglike

        @jax.jit
        def loglike_fn(theta: dict[str, Array]) -> Array:
            return joint_loglike(model, _keys, _likelihoods, _static_data, theta)

        @jax.jit
        def logprior_fn(theta: dict[str, Array]) -> Array:
            return model.ln_prior(theta)

        return adapter.run(loglike_fn, logprior_fn, theta_init, rng_key)

    @jax.jit
    def loglike_fn(theta: dict[str, Array]) -> Array:
        predictions = model.predict(theta)
        lnl = jnp.zeros(())
        for key, lhood in zip(_keys, _likelihoods):
            y_k, sig_k, mask_k, calib_k, ul_k = _static_data[key]
            th = rescaled_params(theta, _rescalings[key])
            mu_k = predictions[key]
            if calib_k is not None:
                mu_k = mu_k * calib_k
            if ul_k is not None:
                lnl_k, _ = lhood(y_k, mu_k, sig_k, mask_k, params=th, is_upper_limit=ul_k)
            else:
                lnl_k, _ = lhood(y_k, mu_k, sig_k, mask_k, params=th)
            lnl = lnl + lnl_k
        return lnl

    @jax.jit
    def logprior_fn(theta: dict[str, Array]) -> Array:
        return model.ln_prior(theta)

    result = adapter.run(loglike_fn, logprior_fn, theta_init, rng_key)
    if _offset != 0.0:
        result = _with_loglike_offset(result, _offset)
    return result


def _with_loglike_offset(result: SamplingResult, offset: float) -> SamplingResult:
    """``result`` with ``offset`` added to ln Z and every ln L (float64, on the host): undoes
    the float32 data rescaling of the kernels (``likelihood.float32_rescaling``)."""
    import numpy as np
    from dataclasses import replace
    lnl = np.asarray(result.log_likelihoods, dtype=np.float64) + offset
    llb = result.log_likelihoods_birth
    if llb is not None:
        llb = np.asarray(llb, dtype=np.float64) + offset
    lnz = result.log_evidence
    if lnz is not None and np.isfinite(lnz):
        lnz = float(lnz) + offset
    return replace(result, log_evidence=lnz, log_likelihoods=lnl, log_likelihoods_birth=llb)
