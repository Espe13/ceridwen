"""Two extra attenuation laws, registered at run time: Gordon+03 SMC bar and Reddy+15.

What it does
------------
Defines ``gordon03_smcbar`` and ``reddy15`` with the same calling convention as the laws in
``ceridwen/dust/attenuation_laws.py`` (``wave`` in Angstrom first, then the law's own
parameters, then ``**kwargs``; returns the optical depth tau(lambda)), and ``register()``, which
adds both to ``ATTENUATION_LAWS`` with the same ``func`` / ``params`` / ``defaults`` / ``doc``
entries as the built-in laws. Call ``register()`` once, before building the ``CSPBasis``:

    from examples.recipes.extra_dust_laws import register     # or put the file on sys.path
    register()
    csp = CSPBasis(ssp, ..., diffuse_law="reddy15", ...)       # theta["diffuse_tau_reddy"]
    Dust(bin_edges=[(-inf, -1.97)], laws=["gordon03_smcbar"])   # theta["tau_g03smc"]

The parameter names in ``params`` / ``defaults`` are exactly the function-signature names
(``tau_g03smc``, ``tau_reddy``). ``Dust`` passes a law only the signature names that are also
keys of ``params`` (``ceridwen/dust/DustModel.py:107-113``); a mismatch silently drops the
parameter (the built-in ``noll`` law's ``E_bump`` / ``Ebump`` is such a case).

Normalisation (the existing convention): the amplitude parameter is tau at 5500 A, exactly for
``powerlaw``/``calzetti``/``smc``/``lmc``/``noll``/``chevallard`` and to <= 0.12 % for the FSPS
ports ``kriek_conroy`` (0.99954) / ``cardelli`` / ``conroy``. The same holds here:

- ``gordon03_smcbar``: tau(5500 A) = ``tau_g03smc`` exactly (the table has a node at
  0.550 um with A/A_V = 1.000).
- ``reddy15``: tau(5500 A) = 0.997113 ``tau_reddy``, because the curve is k(lambda)/R_V with
  R_V = 2.505 and k(0.55 um) = 2.49777, exactly as FSPS and Prospector do (``dust2`` has the
  same meaning in all three). Not renormalised on purpose: that would break the equivalence.

Physics and sources
-------------------
``gordon03_smcbar``: Gordon, Clayton, Misselt, Landolt & Wolff 2003, ApJ 594, 279, Table 4
(SMC Bar average, A(lambda)/A(V)). This is FSPS ``dust_type=5``: FSPS reads
``$SPS_HOME/dust/Gordon03_table4.dat`` (column 1 = lambda [um], column 3 = SMC Bar A/A_V) in
``src/sps_setup.f90:1374-1394``, interpolates linearly in lambda (``src/linterp.f90``), holds the
bluest value (0.116 um) constant blueward, is 0 redward of 2.198 um, and returns
``tauv * g03smcextn`` (``src/attn_curve.f90:157-161``). FSPS commit ``bd187a0d`` (2026-08-06).
The 30 table values are embedded below, identical to that file. Two of them (2.198 um: 0.016 and
1.25 um: 0.131) are the published Table 4 values that dust_extinction's ``G03_SMCBar`` replaces
with 0.110 / 0.250 "to provide smooth interpolation as noted in Gordon et al. (2016, ApJ 826,
104)"; FSPS, and therefore this port, keeps the published values. Prospector has no own
implementation (``prospect/sources/fake_fsps.py:98-100`` raises ``NotImplementedError``; the real
path is FSPS).

``reddy15``: Reddy et al. 2015, ApJ 806, 259, eq. 8 (k(lambda) for 0.15-0.60 um and
0.60-2.85 um, R_V = 2.505). Ported from Prospector ``prospect/sources/fake_fsps.py:102-123``
(commit a78d153), which is itself a port of FSPS ``dust_type=6`` (``src/attn_curve.f90:163-187``):
blue branch on [1500, 6000) A, held constant at its 1500 A value blueward, red branch on
[6000, 28500) A including the non-published continuity offset -0.036221981, zero at
lambda >= 28500 A, divided by 2.505.

Deliberate difference from Prospector/FSPS (Reddy only): both place the branch boundaries at
grid *pixels* (``argmin |lam - 1500|`` in Prospector, ``locate`` in FSPS), so the blueward
constant is k at the pixel nearest 1500 A and the boundary pixels depend on the grid. Here the
boundaries are the exact wavelengths 1500 / 6000 / 28500 A (JIT-safe masks, grid-independent).
On a grid with nodes at those wavelengths the two agree to rounding; elsewhere only the pixels
next to a boundary and the blueward constant differ, by at most one grid step of the curve.
A second, smaller difference is FSPS's own: it writes the Reddy coefficients as default-kind
(single-precision) Fortran literals (``src/attn_curve.f90:173-182``), so FSPS dust_type=6 is the
formula with float32-rounded coefficients, ~5e-7 in tau/tau_reddy from the exact decimals used
by Prospector and here. The Gordon table is read from a file into double precision and is exact.

Public API used
---------------
- ``ceridwen.dust.attenuation_laws.ATTENUATION_LAWS`` (``ceridwen/dust/attenuation_laws.py:404``),
  a module-level dict read by ``Dust.__init__`` (``ceridwen/dust/DustModel.py:76``) and
  ``DiffuseDust.__init__`` (``DustModel.py:219-243``) at construction, so a law registered before
  the model is built is found by name.
- ``CSPBasis(diffuse_law=...)`` (``ceridwen/csp/csp.py:177``, passed to ``DiffuseDust`` at
  ``csp.py:702``) and ``CSPBasis(init_dust_params={"bin_edges":..., "laws":[...]})``
  (``csp.py:687``).
- Registered laws live only in this process. The result file does not record the law, so a
  script that reloads a result must call ``register()`` again before rebuilding the model.

Package change this prepares for
--------------------------------
Moving both functions into ``attenuation_laws.py`` (and ``__all__``) with a golden baseline
per law, and fixing the two registry bugs found while writing this (``noll``: ``E_bump`` in
``params`` vs ``Ebump`` in the signature, ``attenuation_laws.py:485`` vs ``:135``, so the bump
is never passed by ``Dust``; ``drude``: registered as a law although it expects inverse microns,
``:117-133`` / ``:467``, so ``Dust`` feeds it Angstrom and it returns ~1e-8).
"""

import jax
import jax.numpy as jnp

from ceridwen.dust.attenuation_laws import ATTENUATION_LAWS

jax.config.update("jax_enable_x64", True)

__all__ = ["gordon03_smcbar", "reddy15", "register", "LAWS"]

# Gordon et al. 2003 Table 4, SMC Bar: lambda [um] and A(lambda)/A(V), ascending lambda
# (identical to $SPS_HOME/dust/Gordon03_table4.dat columns 1 and 3, read in reverse by FSPS).
_G03_LAM_UM = (
    0.116, 0.119, 0.123, 0.127, 0.131, 0.136, 0.140, 0.145, 0.151, 0.157,
    0.163, 0.170, 0.178, 0.186, 0.195, 0.205, 0.216, 0.229, 0.242, 0.258,
    0.276, 0.296, 0.370, 0.440, 0.550, 0.650, 0.810, 1.250, 1.650, 2.198,
)
_G03_AXAV = (
    6.992, 6.436, 6.297, 6.074, 5.795, 5.575, 5.272, 5.000, 4.776, 4.472,
    4.243, 4.013, 3.866, 3.637, 3.489, 3.293, 3.161, 2.947, 2.661, 2.428,
    2.220, 2.000, 1.672, 1.374, 1.000, 0.801, 0.567, 0.131, 0.169, 0.016,
)
_G03_LAM_AA = jnp.array(_G03_LAM_UM, dtype=jnp.float64) * 1e4
_G03_Y = jnp.array(_G03_AXAV, dtype=jnp.float64)


def gordon03_smcbar(wave, tau_g03smc=1.0, **kwargs):
    """Gordon et al. (2003) SMC bar extinction curve (FSPS dust_type=5).

    :param wave: wavelengths in Angstrom
    :param tau_g03smc: optical depth at 5500 A (exact: the table has a node there)
    :returns: optical depth tau(lambda); linear in lambda between the 30 table nodes,
        constant blueward of 1160 A, 0 redward of 21980 A (as FSPS).
    """
    wave = jnp.asarray(wave, dtype=jnp.float64)
    curve = jnp.interp(wave, _G03_LAM_AA, _G03_Y, left=_G03_Y[0], right=0.0)
    return tau_g03smc * curve


def _reddy_blue(mic):
    return -5.726 + 4.004 / mic - 0.525 / mic**2 + 0.029 / mic**3 + 2.505


def _reddy_red(mic):
    # the -0.036221981 is not in Reddy+15; FSPS/Prospector add it for continuity at 0.6 um
    return -2.672 - 0.010 / mic + 1.532 / mic**2 - 0.412 / mic**3 + 2.505 - 0.036221981


def reddy15(wave, tau_reddy=1.0, **kwargs):
    """Reddy et al. (2015) attenuation curve (FSPS dust_type=6, Prospector fake_fsps).

    :param wave: wavelengths in Angstrom
    :param tau_reddy: amplitude, FSPS/Prospector ``dust2``; tau(5500 A) = 0.997113 tau_reddy
    :returns: optical depth tau(lambda) = tau_reddy * k(lambda) / 2.505
    """
    wave = jnp.asarray(wave, dtype=jnp.float64)
    mic = wave / 1e4
    # evaluate each branch only where it is used, so the gradient stays finite everywhere
    mic_blue = jnp.where((wave >= 1500.0) & (wave < 6000.0), mic, 0.55)
    mic_red = jnp.where((wave >= 6000.0) & (wave < 28500.0), mic, 1.0)
    k = jnp.where(wave < 1500.0, _reddy_blue(0.15),
        jnp.where(wave < 6000.0, _reddy_blue(mic_blue),
        jnp.where(wave < 28500.0, _reddy_red(mic_red), 0.0)))
    return tau_reddy * k / 2.505


LAWS = {
    "gordon03_smcbar": {
        "func": gordon03_smcbar,
        "params": {
            "tau_g03smc": "Optical depth at 5500 Å (exact)",
        },
        "defaults": {
            "tau_g03smc": 1.0,
        },
        "doc": "SMC bar extinction curve, Gordon et al. (2003) Table 4 (FSPS dust_type=5).",
    },
    "reddy15": {
        "func": reddy15,
        "params": {
            "tau_reddy": "Amplitude (FSPS dust2); tau(5500 Å) = 0.99711 tau_reddy",
        },
        "defaults": {
            "tau_reddy": 1.0,
        },
        "doc": "z~2 star-forming galaxy attenuation curve, Reddy et al. (2015) eq. 8 (FSPS dust_type=6).",
    },
}


def register(overwrite=False):
    """Add the laws in ``LAWS`` to ``ATTENUATION_LAWS``; returns the names registered.

    Idempotent. Refuses (``ValueError``) to replace a different law of the same name unless
    ``overwrite=True``.
    """
    for name, entry in LAWS.items():
        existing = ATTENUATION_LAWS.get(name)
        if existing is not None and existing["func"] is not entry["func"] and not overwrite:
            raise ValueError(f"attenuation law '{name}' is already registered with a different "
                             f"function; pass overwrite=True to replace it")
        ATTENUATION_LAWS[name] = dict(entry)
    return list(LAWS)
