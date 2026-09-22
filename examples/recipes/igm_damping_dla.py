"""IGM damping wing and damped Ly-alpha absorber (DLA): now part of the package.

This recipe has been promoted to ``ceridwen.igm.MadauDampingDLA`` (with the kernels
``tau_damping``, ``tau_gp`` and ``voigt_tau``).  The module re-exports them so the recipe's
check script, ``examples/recipes/tests/check_igm_damping_dla.py``, runs unchanged against the
package code and remains its acceptance test (Prospector @ a78d153 to 1e-10; ``x_HI = 0`` and
``N_HI -> 0`` equal ``Madau1995`` byte for byte; finite ``jax.grad`` through the CSP).

What changed on promotion: ``x_HI``, ``logN_HI`` and ``z_dla`` are theta keys, like
``igm_factor``.  The constructor values are the defaults; a theta entry, fixed or sampled,
overrides them::

    from ceridwen import CSPBasis, SedModel
    from ceridwen.igm import MadauDampingDLA
    from ceridwen.priors import Uniform
    igm = MadauDampingDLA(Ob0=0.04897)                 # cosmology: the CSP's, bound at build
    csp = CSPBasis(ssp, ..., cosmo=cosmo, add_igm=True, igm_model=igm)
    model = SedModel(csp, obs, free_param_init={"x_HI": 0.5, "logN_HI": 21.0},
                     priors={"x_HI": Uniform(low=0.0, high=1.0),
                             "logN_HI": Uniform(low=19.0, high=23.0), ...})

``igm_factor`` scales the Madau forest only and is not the neutral fraction (Prospector reads
``x_HI`` from ``igm_factor``, ``prospect/models/sedmodel.py:824``).  See the class docstring for
the foreground-absorber convention, which differs from Prospector's for ``z_dla != zred``.
"""
from ceridwen.igm import MadauDampingDLA, tau_damping, tau_gp, voigt_tau  # noqa: F401

__all__ = ["MadauDampingDLA", "tau_damping", "tau_gp", "voigt_tau"]
