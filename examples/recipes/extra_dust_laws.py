"""Gordon+03 SMC bar and Reddy+15 attenuation laws: now part of the package.

Both laws have been promoted to ``ceridwen/dust/attenuation_laws.py`` and are registered in
``ATTENUATION_LAWS`` under their own names, so no ``register()`` call is needed any more::

    csp = CSPBasis(ssp, ..., diffuse_law="reddy15", ...)       # theta["diffuse_tau_reddy"]
    Dust(bin_edges=[(-inf, -1.97)], laws=["gordon03_smcbar"])   # theta["tau_g03smc"]

This module re-exports the package functions (and a no-op-compatible ``register()``) so the
recipe's check script, ``examples/recipes/tests/check_extra_dust_laws.py``, runs unchanged
against the package code and remains the acceptance test: Gordon+03 vs FSPS's own table and
interpolation, and vs the FSPS Fortran (dust_type=5); Reddy+15 vs Prospector ``fake_fsps``
(@ a78d153) and vs the FSPS Fortran (dust_type=6, single-precision literals); finite gradients
through the CSP spectrum.  The physics and the deliberate differences from FSPS/Prospector are
documented in the function docstrings.

The ``smc`` and ``lmc`` laws are Pei (1992), not Gordon et al. (2003); ``gordon03_smcbar`` is
the Gordon curve.
"""
from ceridwen.dust.attenuation_laws import (  # noqa: F401
    ATTENUATION_LAWS, gordon03_smcbar, reddy15, _G03_LAM_AA, _G03_Y,
)

__all__ = ["gordon03_smcbar", "reddy15", "register", "LAWS"]

LAWS = {name: ATTENUATION_LAWS[name] for name in ("gordon03_smcbar", "reddy15")}


def register(overwrite=False):
    """Kept for old scripts: the laws are registered by the package already.  Returns their
    names; raises if another function has replaced one of them (unless ``overwrite=True``,
    which restores the package's)."""
    for name, entry in LAWS.items():
        existing = ATTENUATION_LAWS.get(name)
        if existing is not None and existing["func"] is not entry["func"] and not overwrite:
            raise ValueError(f"attenuation law '{name}' is already registered with a different "
                             f"function; pass overwrite=True to replace it")
        ATTENUATION_LAWS[name] = dict(entry)
    return list(LAWS)
