"""AGN dust screen: the galaxy's diffuse curve times Prospector's ``dust4`` power law.

Reference: ``prospect/sources/fake_fsps.py`` and ``agnssp_basis.get_spectrum`` at
bd-j/prospector 91f8e13. The disc is attenuated by::

    add_dust(wave, [agn, 0], ..., dust_type, dust_index, dust2, dust1=0, dust1_index=0,
             dust4_type, dust4_index, dust4, frac_nodust=0, frac_obrun=0)

With ``frac_nodust = frac_obrun = dust1 = 0`` that reduces, exactly in floating point
(``x * 1.0``, ``x * 0`` and ``+ 0`` are exact), to ``agn * attenuate(...)`` with::

    attenuate = exp(-attn_curve(dust2)) * exp(-dust1 * ...) * exp(-attn_curve_4)
              = dust2_ext * 1.0 * dust4_ext
    attn_curve_4 = (lam / 5500) ** dust4_index * dust4    if dust4_type != 0, else 0

``dust4_type``: 0 = no AGN screen beyond the diffuse curve; 1 = diffuse curve times the
``dust4`` power law; 2 = Prospector raises ``NotImplementedError`` and so does this
module, at setup. ``agnssp_basis.get_spectrum`` silently treats any other value as 0;
here it is refused.

``AGNScreen.attenuate`` takes the diffuse transmission from ``ctx.diffuse_curve`` (built
by ``hook.py``). ``ProspectorDiffuseCurve`` is the dust2 part of ``fake_fsps.attenuate``
ported to JAX (``dust_type`` 0, 2, 4, 6; 1, 3, 5 raise ``NotImplementedError`` in
Prospector and are refused here), so the ``"prospector"`` mode can reproduce
Prospector's AGN attenuation exactly; the CERIDWEN galaxy's own ``DiffuseDust`` curve is
a different implementation of those laws.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import AGN_THETA_DEFAULTS, DUST4_TYPES, Screen

__all__ = ["AGNScreen", "ProspectorDiffuseCurve", "LAMV", "PROSPECTOR_DUST_TYPES"]

LAMV = 5500.0  # fake_fsps.attenuate: lamv = 5500.0
PROSPECTOR_DUST_TYPES = (0, 2, 4, 6)  # the dust_type values fake_fsps.attenuate implements


def _check_wave(wave, who):
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or not np.all(np.isfinite(wave)) or not np.all(wave > 0):
        raise ValueError(f"{who}: wave must be a 1-D array of finite positive wavelengths [A].")
    return wave


class AGNScreen(Screen):
    """Attenuation of the AGN disc: ``spec * diffuse_curve * exp(-(lam/5500)**dust4_index * dust4)``.

    Parameters
    ----------
    wave : (n_wave,) array
        Rest-frame model grid [A]. Static.
    dust4_type : {0, 1}
        0: diffuse curve only (``dust4`` / ``dust4_index`` are not read). 1: also the
        ``dust4`` power law. 2 is refused (``NotImplementedError``, as in Prospector).
    """

    def __init__(self, wave, *, dust4_type=1):
        if dust4_type == 2:
            raise NotImplementedError(
                "AGNScreen(dust4_type=2): Prospector raises NotImplementedError for it "
                "(prospect/sources/agnssp_basis.py) and so does CERIDWEN. Use 0 (diffuse "
                "curve only) or 1 (diffuse curve times the dust4 power law).")
        if isinstance(dust4_type, bool) or dust4_type not in DUST4_TYPES:
            raise ValueError(f"AGNScreen(dust4_type={dust4_type!r}): choose 0 or 1.")
        self.dust4_type = int(dust4_type)
        self.wave = jnp.asarray(_check_wave(wave, "AGNScreen"))
        self._x = self.wave / LAMV                    # lam / lamv, static
        self.param_names = ("dust4", "dust4_index") if self.dust4_type == 1 else ()

    def default_params(self) -> dict:
        return {k: AGN_THETA_DEFAULTS[k] for k in self.param_names}

    def dust4_transmission(self, theta):
        """``exp(-(lam/5500)**dust4_index * dust4)`` (all ones for ``dust4_type=0``)."""
        if self.dust4_type == 0:                       # static branch
            return jnp.ones_like(self.wave)
        d4 = jnp.asarray(theta.get("dust4", AGN_THETA_DEFAULTS["dust4"]))
        d4_index = jnp.asarray(theta.get("dust4_index", AGN_THETA_DEFAULTS["dust4_index"]))
        return jnp.exp(-(self._x ** d4_index * d4))

    def transmission(self, theta, ctx):
        """F(obs)/F(emitted) of the AGN screen: ``ctx.diffuse_curve * dust4_ext``."""
        return ctx.diffuse_curve * self.dust4_transmission(theta)

    def attenuate(self, spec, theta, ctx):
        """``spec`` (n_wave,) attenuated by the diffuse curve and the dust4 screen."""
        return spec * self.transmission(theta, ctx)


class ProspectorDiffuseCurve:
    """``exp(-attn_curve)`` of ``fake_fsps.attenuate`` for the galaxy's ``dust2``.

    Static ``dust_type`` (FSPS numbering): 0 power law ``(lam/5500)**dust_index * dust2``;
    2 Calzetti+00 (clipped at 0); 4 Kriek & Conroy 13 (Calzetti + Drude bump with
    ``eb = 0.85 - 1.9 dust_index``, times ``(lam/5500)**dust_index``); 6 Reddy+15
    (``argmin`` edges on the grid, constant blueward of 1500 A, zero redward of 2.85 um,
    as Prospector). theta keys: ``dust2`` (all types), ``dust_index`` (0 and 4).
    """

    def __init__(self, wave, *, dust_type=0):
        if dust_type in (1, 3, 5):
            raise NotImplementedError(
                f"ProspectorDiffuseCurve(dust_type={dust_type}): fake_fsps.attenuate raises "
                "NotImplementedError for it; choose one of (0, 2, 4, 6).")
        if isinstance(dust_type, bool) or dust_type not in PROSPECTOR_DUST_TYPES:
            raise ValueError(f"ProspectorDiffuseCurve(dust_type={dust_type!r}): choose one of "
                             f"{PROSPECTOR_DUST_TYPES}.")
        self.dust_type = int(dust_type)
        lam = _check_wave(wave, "ProspectorDiffuseCurve")
        self.wave = jnp.asarray(lam)
        self._x = self.wave / LAMV
        self.param_names = ("dust2", "dust_index") if self.dust_type in (0, 4) else ("dust2",)
        if self.dust_type in (2, 4):
            self._k = jnp.asarray(self._cal00(lam))
        if self.dust_type == 4:
            dlam, lamuvb = 350.0, 2175.0
            self._drude_num = jnp.asarray((lam * dlam) ** 2)
            self._drude_den = jnp.asarray((lam ** 2 - lamuvb ** 2) ** 2 + (lam * dlam) ** 2)
        if self.dust_type == 6:
            self._k = jnp.asarray(self._reddy(lam))

    @staticmethod
    def _cal00(lam):
        dd63 = 6300.00
        cal00 = np.zeros_like(lam)
        gt = lam > dd63
        le = ~gt
        cal00[gt] = 1.17 * (-1.857 + 1.04 * (1e4 / lam[gt])) + 1.78
        cal00[le] = 1.17 * (-2.156 + 1.509 * (1e4 / lam[le])
                            - 0.198 * (1E4 / lam[le]) ** 2
                            + 0.011 * (1E4 / lam[le]) ** 3) + 1.78
        cal00 = cal00 / 0.44 / 4.05
        return np.clip(cal00, 0.0, np.inf)

    @staticmethod
    def _reddy(lam):
        reddy = np.zeros_like(lam)
        w1 = np.abs(lam - 1500).argmin()
        w2 = np.abs(lam - 6000).argmin()
        reddy[w1:w2] = (-5.726 + 4.004 / (lam[w1:w2] / 1e4) - 0.525 / (lam[w1:w2] / 1e4) ** 2
                        + 0.029 / (lam[w1:w2] / 1e4) ** 3 + 2.505)
        reddy[:w1] = reddy[w1]
        w1 = np.abs(lam - 6000).argmin()
        w2 = np.abs(lam - 28500).argmin()
        reddy[w1:w2] = (-2.672 - 0.010 / (lam[w1:w2] / 1e4) + 1.532 / (lam[w1:w2] / 1e4) ** 2
                        + -0.412 / (lam[w1:w2] / 1e4) ** 3 + 2.505 - 0.036221981)
        return reddy / 2.505

    def attn_curve(self, theta):
        dust2 = jnp.asarray(theta.get("dust2", 0.0))
        if self.dust_type == 0:
            return self._x ** jnp.asarray(theta["dust_index"]) * dust2
        if self.dust_type == 2:
            return self._k * dust2
        if self.dust_type == 4:
            dust_index = jnp.asarray(theta["dust_index"])
            eb = 0.85 - 1.9 * dust_index
            drude = eb * self._drude_num / self._drude_den
            return dust2 * (self._k + drude / 4.05) * self._x ** dust_index
        return dust2 * self._k                         # dust_type 6

    def transmission(self, theta):
        """(n_wave,) ``exp(-attn_curve)`` (Prospector's ``dust2_ext``)."""
        return jnp.exp(-self.attn_curve(theta))
