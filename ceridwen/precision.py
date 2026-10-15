"""The working float dtype of the device-side model: float64 unless ``CERIDWEN_X64=0``.

``import ceridwen`` sets ``jax_enable_x64`` from ``CERIDWEN_X64`` (``ceridwen/__init__.py``);
this module is the one place that turns that switch into a dtype.  Every device-side array
whose dtype the package chooses explicitly goes through :func:`float_dtype` / :func:`as_float`,
so that float32 mode (Apple Metal, consumer NVIDIA cards; ``docs/mac_gpu.md``) never requests
``jnp.float64`` and never trips JAX's "Explicitly requested dtype float64 ... will be truncated"
warning.  Host-side NumPy stays ``np.float64`` everywhere: the switch is about what runs on the
device, not about grids, axes or provenance.

Nothing here reads a theta value or branches on data: ``float_dtype()`` is a Python-level
decision made at trace time, the same as ``jax_enable_x64`` itself.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["FLOAT", "is_x64", "float_dtype", "as_float", "precision_name", "safe_tiny"]


def is_x64() -> bool:
    """True when ``jax_enable_x64`` is on (CERIDWEN's default; off with ``CERIDWEN_X64=0``)."""
    return bool(jax.config.read("jax_enable_x64"))


def float_dtype():
    """``jnp.float64`` under x64, else ``jnp.float32``: the dtype of theta, SFH weights,
    attenuation, cosmology, likelihood and sampler state.  Read at call (trace) time, so a
    test that flips ``jax_enable_x64`` after import still gets the right answer."""
    return jnp.float64 if is_x64() else jnp.float32


#: the working dtype at import time (``float_dtype()`` is the live value)
FLOAT = float_dtype()


def as_float(x):
    """``jnp.asarray(x, dtype=float_dtype())``: the device-side float array of ``x``."""
    return jnp.asarray(x, dtype=float_dtype())


def safe_tiny(value: float) -> float:
    """``value`` (a positive floor such as ``1e-70``) under float64; in float32 mode the smallest
    normal float32 (1.2e-38) when ``value`` would underflow it to 0 and turn a guarded division
    into inf.  A Python float either way, so a float64 trace is unchanged."""
    if is_x64():
        return value
    tiny32 = float(jnp.finfo(jnp.float32).tiny)
    return value if value >= tiny32 else tiny32


def precision_name() -> str:
    """``"float64"`` or ``"float32"``, as written to ``ceridwen_result.h5`` ``config/precision``."""
    return "float64" if is_x64() else "float32"
