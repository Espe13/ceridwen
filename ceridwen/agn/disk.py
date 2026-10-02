"""Temple+21 accretion-disc continuum (Prospector ``AGNSSPBasis``, Wang et al. 2024).

Reference: ``prospect/sources/agnssp_basis.py`` at bd-j/prospector 91f8e13
(``temple_template`` and ``get_spectrum``)::

    norm = np.interp(ssp.wavelengths, w, temp)                  # template on the SSP grid
    idx = (wave >= 5500 - 20) & (wave <= 5500 + 20)
    scale = np.median(norm[idx] / (young + old)[idx])           # no-dust, no-IGM galaxy
    agn_bbb_cont = (norm / scale) * fagn_bbb

So ``fagn_bbb`` is F_nu,disc / F_nu,galaxy at rest 5500 A, where "at 5500 A" means the
median of the pixel ratios in [5480, 5520] A. The galaxy is the dust-free young+old
stellar spectrum of the same call, so the normalisation is a traced function of theta.

Resampling of the template onto the model grid
----------------------------------------------
``"prospector"`` / ``"fsps"``: exactly ``np.interp`` (linear in wavelength and F_nu,
constant extrapolation beyond the template's ends, 91 A and 1e8 A). Computed once at
setup in float64 numpy, so the values are numpy's own.

``"corrected"``: the same ``np.interp`` inside [91, 1e8] A and ZERO outside. The template
has no information below 91 A (the X-ray/EUV side); holding its 91 A value constant to
the grid's blue end invents a flat F_nu there. Zero is the "no model" value; it changes
nothing redward of 91 A, so the 5500 A normalisation and every optical/IR pixel are
identical to the reproduction modes.

Gradient of the median
----------------------
``jnp.median`` sorts, so its derivative is piecewise: d median / d ratio_i is 1 for the
middle pixel (odd count) or 1/2 for each of the two middle pixels (even count; the
5480-5520 A window holds 20 pixels on the 2 A test grid) and 0 elsewhere. It jumps when
the order of the ratios changes. That is the derivative of the quantity Prospector
defines, so it is kept; the disc is exactly linear in ``fagn_bbb``.

Zero galaxy in the window
-------------------------
If the galaxy is zero in any window pixel (e.g. an all-zero SFH), Prospector divides by
zero. Here such pixels are replaced by 1 before the division and the disc is set to 0
when any window pixel is not positive, with the "double where" so values AND gradients
stay finite. For a positive galaxy the selected values are exactly Prospector's, so
this guard is applied in every mode.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from . import AGN_MODES, AGN_MODE_CORRECTED, AGN_THETA_DEFAULTS, Component

__all__ = ["TempleDisk", "TEMPLE21_FILE", "NORM_WINDOW", "load_temple21"]

TEMPLE21_FILE = os.path.join(os.path.dirname(__file__), "data", "temple21.txt")
# agnssp_basis.get_spectrum: (wave >= 5500 - 20) & (wave <= 5500 + 20), both ends included.
NORM_WINDOW = (5500.0 - 20.0, 5500.0 + 20.0)


def load_temple21(path=TEMPLE21_FILE):
    """(wave [A], F_nu) of the template, as Prospector's ``np.loadtxt(..., unpack=True)``."""
    w, t = np.loadtxt(path, unpack=True)
    return w, t


class TempleDisk(Component):
    """Accretion disc: ``fagn_bbb`` x Temple+21 template, normalised to the galaxy at 5500 A.

    Parameters
    ----------
    wave : (n_wave,) array
        Rest-frame model wavelength grid [A], ascending. Static.
    mode : {"fsps", "prospector", "corrected"}
        Template resampling (see module docstring). ``"fsps"`` behaves as
        ``"prospector"`` (FSPS has no disc; ``AGNConfig`` refuses one in that mode).
    template_file : str, optional
        Two-column (wave [A], F_nu) file; default the bundled ``temple21.txt``.
    """

    param_names = ("fagn_bbb",)

    def __init__(self, wave, *, mode, template_file=None):
        if mode not in AGN_MODES:
            raise ValueError(f"TempleDisk(mode={mode!r}): choose one of {AGN_MODES}.")
        wave = np.asarray(wave, dtype=np.float64)
        if wave.ndim != 1 or wave.size < 2 or not np.all(np.diff(wave) > 0):
            raise ValueError("TempleDisk: wave must be a 1-D strictly ascending grid [A].")
        if not np.all(np.isfinite(wave)) or wave[0] <= 0:
            raise ValueError("TempleDisk: wave must be finite and positive [A].")
        self.mode = mode
        self.template_file = TEMPLE21_FILE if template_file is None else template_file
        w_t, f_t = load_temple21(self.template_file)
        if not (np.all(np.diff(w_t) > 0) and np.all(f_t >= 0) and np.all(np.isfinite(f_t))):
            raise ValueError(f"TempleDisk: template {self.template_file} must have ascending "
                             "wavelengths and finite, non-negative F_nu.")
        self.template_range = (float(w_t[0]), float(w_t[-1]))

        tmpl = np.interp(wave, w_t, f_t)
        if mode == AGN_MODE_CORRECTED:
            tmpl = np.where((wave >= w_t[0]) & (wave <= w_t[-1]), tmpl, 0.0)

        lo, hi = NORM_WINDOW
        win = (wave >= lo) & (wave <= hi)
        if not win.any():
            raise ValueError(
                f"TempleDisk: the model grid has no pixel in [{lo:g}, {hi:g}] A, where "
                "fagn_bbb is defined (F_nu,disc / F_nu,galaxy at rest 5500 A, Prospector's "
                f"median over that window). Grid spacing near 5500 A must be < {hi - lo:g} A.")
        if not np.all(tmpl[win] > 0):
            raise ValueError("TempleDisk: the template is not positive in the 5500 A window.")
        self.win_idx = np.flatnonzero(win)          # static index array
        self.template = jnp.asarray(tmpl)            # (n_wave,) float64
        self.template_win = jnp.asarray(tmpl[win])   # (n_win,) float64

    def default_params(self) -> dict:
        return {"fagn_bbb": AGN_THETA_DEFAULTS["fagn_bbb"]}

    def norm_scale(self, gal_nodust_young_old):
        """Prospector's ``scale`` = median(template / galaxy) over the 5500 A window,
        and a bool that is False when the galaxy is not positive in every window pixel."""
        g = jnp.take(jnp.asarray(gal_nodust_young_old), self.win_idx, axis=-1)
        ok = jnp.all(g > 0)
        g_safe = jnp.where(g > 0, g, 1.0)
        return jnp.median(self.template_win / g_safe), ok

    def rest_spectrum(self, theta, ctx):
        """(n_wave,) disc F_nu [units of ``ctx.gal_nodust_young_old``], before attenuation."""
        fagn_bbb = jnp.asarray(theta.get("fagn_bbb", AGN_THETA_DEFAULTS["fagn_bbb"]))
        scale, ok = self.norm_scale(ctx.gal_nodust_young_old)
        ok = ok & (scale > 0)
        scale_safe = jnp.where(ok, scale, 1.0)
        disc = (self.template / scale_safe) * fagn_bbb
        return jnp.where(ok, disc, 0.0)
