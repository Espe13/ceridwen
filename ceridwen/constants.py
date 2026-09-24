"""Physical constants used across CERIDWEN (cgs, Angstrom, km/s).

One definition of each: every module imports from here.

``LSUN_ERG_S`` is FSPS's solar luminosity (``sps_vars.f90:422``, ``lsun = 3.839E33``).
Every shipped grid is built with python-fsps ``get_spectrum`` and stores spectra in L_sun/Hz
in FSPS's unit: MIST/aMIST isochrones are divided by this ``lsun`` in ``getspec.f90:200``,
and FSPS converts every grid (BPASS included) to magnitudes with the same ``lsun``
(``mag2cgs``, ``sps_vars.f90:432``).  The same value therefore serves every grid.

``LSUN_HZ_TO_FNU_CGS_AT_10PC`` turns L_sun/Hz into F_nu [erg/s/cm^2/Hz] at 10 pc.  It uses the
IAU parsec; FSPS's rounded ``pc2cm = 3.08568E18`` gives a value 1.6e-6 lower.
"""
from __future__ import annotations

import math

__all__ = [
    "C_AA_S", "C_KMS", "C_CMS", "LSUN_ERG_S", "HPLANCK_ERG_S",
    "PC_TO_CM", "MPC_TO_CM", "LSUN_HZ_TO_FNU_CGS_AT_10PC",
]

C_KMS = 2.99792458e5            # km/s
C_CMS = 2.99792458e10           # cm/s
C_AA_S = 2.99792458e18          # Angstrom/s

LSUN_ERG_S = 3.839e33           # erg/s, FSPS sps_vars.f90:422
HPLANCK_ERG_S = 6.6261e-27      # erg s, FSPS sps_vars.f90:430

PC_TO_CM = 3.0856775814913673e18
MPC_TO_CM = 3.0856775814913673e24

# erg s^-1 cm^-2 Hz^-1 per L_sun Hz^-1 at 10 pc = LSUN_ERG_S / (4 pi (10 pc)^2)
LSUN_HZ_TO_FNU_CGS_AT_10PC = LSUN_ERG_S / (4.0 * math.pi * (10.0 * PC_TO_CM) ** 2)
