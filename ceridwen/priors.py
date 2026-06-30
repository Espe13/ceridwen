"""Prior distributions for CERIDWEN models.

A clean, discoverable import path for the priors (implemented in
``ceridwen.sampler.priors``)::

    from ceridwen.priors import Uniform, Normal, ClippedNormal, LogNormal, StudentT

Each prior exposes ``log_prob``, ``sample``, ``unit_transform`` (inverse-CDF,
used by nested sampling) and ``inverse_unit_transform`` (CDF), and is a JAX
PyTree so it can flow through ``jit``/``grad``.
"""
from .sampler.priors import (
    Prior,
    Uniform,
    TopHat,
    Normal,
    ClippedNormal,
    LogNormal,
    StudentT,
)

__all__ = [
    "Prior",
    "Uniform",
    "TopHat",
    "Normal",
    "ClippedNormal",
    "LogNormal",
    "StudentT",
]
