"""LogUniform prior, the Prospector way, without a new ``Prior`` class.

What it does
------------
CERIDWEN has no ``LogUniform`` prior class. You get the same distribution by
sampling ``log10(x)`` from a ``Uniform`` and deriving ``x`` with a transform:

    log10_<name> ~ Uniform(log10 mini, log10 maxi)       (sampled)
    <name>       = 10 ** log10_<name>                    (derived, via ``transforms``)

This matches Prospector's ``LogUniform(mini, maxi)`` exactly: that class is
``scipy.stats.reciprocal(a=mini, b=maxi)`` (Prospector
``prospect/models/priors.py:373-399``, the distribution is set at l.384), with
pdf ``1 / (x ln(maxi/mini))`` on ``[mini, maxi]``. Its CDF
``ln(x/mini) / ln(maxi/mini)`` is uniform in ``log x``, and the base of the log
does not matter.

``loguniform(name, mini, maxi)`` returns ``(prior, transform)``, as asked.
``loguniform_setup(...)`` returns the three dicts ``SedModel`` needs:

    from ceridwen import SedModel
    from examples.recipes.loguniform_prior import loguniform_setup

    lu = loguniform_setup("diffuse_tau_kc", 1e-2, 3.0, init=0.3)
    model = SedModel(csp, [phot],
                     priors={**my_priors, **lu["priors"]},              # "log10_diffuse_tau_kc"
                     transforms={**my_transforms, **lu["transforms"]},  # "diffuse_tau_kc"
                     free_param_init={**my_init, **lu["free_param_init"]})

Why each dict is needed (derived from the code, not the docstrings):

* ``transforms``: ``SedModel`` removes every transform key from the free set
  (``ceridwen/model/model.py:104-110``), and ``apply_transforms`` computes it
  from the free theta before every prediction (``model.py:331-338``, called
  from ``predict`` at ``model.py:345``).
* ``free_param_init``: ``log10_<name>`` is not a CSP key. It becomes a sampled
  parameter only through ``free_param_init`` (``model.py:112-117``).
  Without it, ``SedModel`` rejects the prior as belonging to an unsampled name
  (``model.py:121-126``).
* ``priors``: a ``Uniform`` (``ceridwen/sampler/priors.py:127-150``). NUTS's
  bounded-prior detection, ``fit._detect_bounds`` (``ceridwen/fit.py:394-408``),
  matches on the class name ``"Uniform"``/``"TopHat"`` and reads
  ``params["low"]``/``["high"]``. So ``log10_<name>`` gets the bounds
  ``(log10 mini, log10 maxi)``, which ``fitSED(sampler="nuts")`` passes to the
  adapter (``fit.py:189-199``). Nested sampling uses ``Uniform.unit_transform``
  (the TFP quantile, ``priors.py:109-116``). The check script demonstrates both.

The transform is pure ``jax.numpy`` on a float64 array of shape ``(1,)``, the
same shape as every scalar theta entry (``free_param_init`` is stored with
``jnp.atleast_1d``, ``model.py:114``). It is safe under ``jit``, ``grad`` and
``vmap``.

The posterior, the evidence and the ``/samples`` in ``ceridwen_result.h5`` are
in ``log10_<name>``. ``x`` is a derived quantity: recompute it as
``10**samples["log10_<name>"]``.

LogNormal: the two packages mean different things by ``mode``
--------------------------------------------------------------
Both are called ``LogNormal(mode, sigma)``, but with the same arguments they
give different distributions:

* CERIDWEN ``ceridwen/sampler/priors.py:285-291`` builds
  ``tfd.LogNormal(loc=mode, scale=sigma)``, so ``ln x ~ N(mode, sigma)``.
  ``mode`` is the **mean (and median) of ln x**. The class docstring (l.286)
  agrees.
* Prospector ``prospect/models/priors.py:442-480`` is
  ``scipy.stats.lognorm(s=sigma, loc=0, scale=exp(mode + sigma**2))``
  (l.458, 461-471), so ``ln x ~ N(mode + sigma**2, sigma)``. Its peak (the true
  mode of x) is at ``exp(mu - sigma**2) = exp(mode)``, so Prospector's ``mode``
  is **ln of the peak** of the pdf in x, as its docstring (l.449-450) says.

Conversion, with the same ``sigma``:

    mode_ceridwen   = mode_prospector + sigma**2
    mode_prospector = mode_ceridwen   - sigma**2

``prospector_lognormal_to_ceridwen`` does this. The check script confirms it
numerically: the two log-pdfs agree after conversion and differ by up to
O(sigma**2) in ln x without it.

A related inconsistency, reported and not changed: CERIDWEN's ``LogNormal.scale``
property (``priors.py:297-299``) still returns Prospector's
``exp(mode + sigma**2)``. For its own ``tfd.LogNormal(loc=mode)`` that is not
the scipy scale (``exp(mode)``) of the distribution it samples. ``range``
(l.305-309) is centred on ``exp(mode)``, the CERIDWEN median, so the two
properties disagree with each other.

Reference
---------
The log-uniform ("reciprocal", scale-invariant) prior is the Jeffreys prior
for a scale parameter: Jeffreys H., 1946, Proc. R. Soc. A, 186, 453; see also
Leja J. et al., 2017, ApJ, 837, 170 (Prospector-alpha priors) and Johnson B. D.
et al., 2021, ApJS, 254, 22 (Prospector).

Package change this prepares for: a first-class ``LogUniform`` prior
(``tfd`` TransformedDistribution of a Uniform under ``Exp``) with bounds
detection in ``fit._detect_bounds``. It must reproduce the check numbers of
``tests/check_loguniform_prior.py``.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ceridwen.priors import Uniform  # ceridwen/sampler/priors.py:127

__all__ = ["loguniform", "loguniform_setup", "sampled_name",
           "prospector_lognormal_to_ceridwen", "ceridwen_lognormal_to_prospector"]


def sampled_name(name: str) -> str:
    """Name of the sampled parameter behind a log-uniform ``name``."""
    return f"log10_{name}"


def _check_limits(mini: float, maxi: float) -> tuple[float, float]:
    mini, maxi = float(mini), float(maxi)
    if not (np.isfinite(mini) and np.isfinite(maxi)) or mini <= 0.0 or maxi <= mini:
        raise ValueError(f"LogUniform needs 0 < mini < maxi < inf, got mini={mini}, maxi={maxi}")
    return mini, maxi


def loguniform(name: str, mini: float, maxi: float) -> tuple[Uniform, Callable]:
    """``(prior, transform)`` for a log-uniform ``name`` on ``[mini, maxi]``.

    ``prior`` is ``Uniform(log10 mini, log10 maxi)`` on ``log10_<name>``.
    ``transform(theta)`` returns ``10 ** theta["log10_<name>"]`` (float64, jit-safe),
    and is registered as ``transforms={name: transform}``.
    """
    mini, maxi = _check_limits(mini, maxi)
    key = sampled_name(name)
    prior = Uniform(low=np.log10(mini), high=np.log10(maxi))

    def transform(theta):
        return jnp.power(10.0, jnp.asarray(theta[key], dtype=jnp.float64))

    transform.__name__ = f"pow10_{key}"   # shows up in SedModel.summary() / result attrs
    return prior, transform


def loguniform_setup(name: str, mini: float, maxi: float, init: float | None = None) -> dict:
    """The ``priors`` / ``transforms`` / ``free_param_init`` dicts for ``SedModel``.

    ``init`` is in ``x`` units (default: the geometric mean ``sqrt(mini maxi)``).
    """
    mini, maxi = _check_limits(mini, maxi)
    prior, transform = loguniform(name, mini, maxi)
    x0 = np.sqrt(mini * maxi) if init is None else float(init)
    if not (mini <= x0 <= maxi):
        raise ValueError(f"init={x0} lies outside [{mini}, {maxi}]")
    key = sampled_name(name)
    return {"priors": {key: prior},
            "transforms": {name: transform},
            "free_param_init": {key: jnp.array([np.log10(x0)])}}


def prospector_lognormal_to_ceridwen(mode: float, sigma: float) -> dict:
    """Arguments of ``ceridwen.priors.LogNormal`` that give Prospector's ``LogNormal(mode, sigma)``."""
    return {"mode": mode + sigma ** 2, "sigma": sigma}


def ceridwen_lognormal_to_prospector(mode: float, sigma: float) -> dict:
    """Arguments of Prospector's ``LogNormal`` that give ``ceridwen.priors.LogNormal(mode, sigma)``."""
    return {"mode": mode - sigma ** 2, "sigma": sigma}
