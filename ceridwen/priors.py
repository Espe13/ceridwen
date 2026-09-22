"""Prior distributions for CERIDWEN models.

A clean, discoverable import path for the priors (implemented in
``ceridwen.sampler.priors``)::

    from ceridwen.priors import Uniform, Normal, ClippedNormal, LogNormal, LogUniform, StudentT

Each prior exposes ``logpdf`` (also ``__call__``), ``sample``, ``unit_transform``
(inverse CDF) and ``inverse_unit_transform`` (CDF); ``bounds`` gives the support.
"""
from .sampler.priors import (
    Prior,
    Uniform,
    TopHat,
    Normal,
    MultivariateNormalPrior,
    ClippedNormal,
    LogNormal,
    LogUniform,
    StudentT,
)

__all__ = [
    "Prior",
    "Uniform",
    "TopHat",
    "Normal",
    "MultivariateNormalPrior",
    "ClippedNormal",
    "LogNormal",
    "LogUniform",
    "StudentT",
]
