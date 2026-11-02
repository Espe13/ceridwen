"""Emission lines in photometry: each line's flux through each filter, without painting it.

A line of integrated flux F [erg/s/cm^2] at observed wavelength lam0 contributes
``F * E_j(lam0, sigma)`` maggies to band j, with ``E_j = int g_j(lam) N(lam; lam0, sigma) dlam`` and
``g_j = R_j lam / ab_zero_counts`` linear between the knots of the filter's own grid (the
integrand of ``FilterSet.flux_weights``).  ``sigma = 0`` is a delta line, ``E_j = g_j(lam0)``
(Prospector's ``nebline_photometry``); ``sigma > 0`` is the exact integral of the Gaussian
against the piecewise-linear filter, which keeps the smoothing of a filter edge.  Both conserve
the line flux for any width.

* :func:`line_weights` -- (n_filters, n_lines) E, exact (NumPy, setup).
* :class:`LineWeightTable` -- E tabulated with the exact integral, for a traced redshift and /
  or a sampled width.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.special import erf

CKMS = 2.99792458e5
NSIGMA = 8.0             # a Gaussian is < 1.3e-15 of its peak beyond 8 sigma


def _gauss_rows(lam, g, s, lam0, sig, deriv=False):
    """E of one filter (knots ``lam``, values ``g``, slopes ``s``) for Gaussian lines, and with
    ``deriv`` also dE/dlam0; exact.  Lines are taken in chunks of neighbours, each against the
    filter segments within ``NSIGMA`` of it (the rest add exactly 0)."""
    out = np.zeros(lam0.size)
    dout = np.zeros(lam0.size)
    order = np.argsort(lam0)
    for c in np.array_split(order, max(1, lam0.size // 32)):
        if c.size == 0:
            continue
        reach = NSIGMA * float(sig[c].max())
        a = max(int(np.searchsorted(lam, lam0[c].min() - reach)) - 1, 0)
        b = min(int(np.searchsorted(lam, lam0[c].max() + reach)) + 1, lam.size - 1)
        if b <= a:
            continue
        kn, gk, sk = lam[a:b + 1], g[a:b + 1], s[a:b]
        l0, sg = lam0[c][:, None], sig[c][:, None]
        u = (kn[None, :] - l0) / sg
        Phi = 0.5 * (1.0 + erf(u / np.sqrt(2.0)))
        phi = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
        dPhi = Phi[:, 1:] - Phi[:, :-1]
        # int over [lam_k, lam_k+1] of (g_k + s_k (lam - lam_k)) N(lam; l0, sg)
        seg = ((gk[None, :-1] + sk[None, :] * (l0 - kn[None, :-1])) * dPhi
               + sk[None, :] * sg * (phi[:, :-1] - phi[:, 1:]))
        out[c] = seg.sum(axis=1)
        if deriv:
            # dE/dlam0 = int g' N + the jumps of g at the ends of the filter (g = 0 outside)
            d = (sk[None, :] * dPhi).sum(axis=1)
            if a == 0:
                d = d + g[0] * phi[:, 0] / sg[:, 0]
            if b == lam.size - 1:
                d = d - g[-1] * phi[:, -1] / sg[:, 0]
            dout[c] = d
    return (out, dout) if deriv else out


def line_weights(filterset, lam0, sigma_aa):
    """(n_filters, n_lines) maggies per unit line flux [erg/s/cm^2] for lines at observed
    wavelengths ``lam0`` [A] with dispersions ``sigma_aa`` [A] (a number or one per line; 0 is
    a delta line).  Exact (NumPy, float64)."""
    lam0 = np.atleast_1d(np.asarray(lam0, dtype=np.float64))
    sig = np.broadcast_to(np.asarray(sigma_aa, dtype=np.float64), lam0.shape)
    return np.stack([line_weights_one(lam, g, s, lam0, sig)
                     for lam, g, s, *_ in filterset._flux_tables()])


N_SIGMA_NODES = 65       # sigma nodes (uniform in sigma) of a table for a sampled width
SIGMA_RESOLVED_KMS = 30.0  # a sampled-width table resolves lines down to this width


class LineWeightTable:
    """E_j for a traced line wavelength and / or width, tabulated with the exact integral.

    * ``lam0`` given (fixed redshift, sampled width): per line, E at ``n_sigma`` nodes uniform in
      sigma over [0, ``sigma_max_kms``], linear in sigma^2 between them.
    * ``lam0`` None (sampled redshift): per filter, on its own log-uniform grid subdivided to at
      most half the narrowest width (the fixed one, or ``SIGMA_RESOLVED_KMS`` for a sampled one)
      and extended by the widest line on each side; cubic Hermite in lambda with the exact dE/dlambda (linear, and
      exact, at sigma = 0), linear in sigma^2 between the nodes; one node for a fixed width
      ``sigma_kms``.
    """

    def __init__(self, filterset, *, lam0=None, sigma_kms=None, sigma_max_kms=None,
                 n_sigma=N_SIGMA_NODES):
        from ceridwen.precision import float_dtype
        dt = float_dtype()
        if sigma_kms is None:
            sv = np.linspace(0.0, float(sigma_max_kms), int(n_sigma))
        else:
            sv = np.array([float(sigma_kms)])
        self.sv = sv
        self.ds = float(sv[1] - sv[0]) if sv.size > 1 else 0.0
        tables = filterset._flux_tables()
        if lam0 is not None:
            lam0 = np.asarray(lam0, dtype=np.float64)
            tab = np.stack([np.stack([line_weights_one(t[0], t[1], t[2], lam0,
                                                       lam0 * v / CKMS) for t in tables])
                            for v in sv], axis=-1)                 # (n_filt, n_lines, n_sigma)
            self.per_line = jnp.asarray(tab, dtype=dt)
            self.rows = None
            return
        self.per_line = None
        s_ext = float(sv.max()) / CKMS
        # the narrowest width the table must resolve: the fixed one, or the first sigma node
        # (at least SIGMA_RESOLVED_KMS) of a sampled width; 0 for delta lines
        pos = sv[sv > 0.0]
        s_res = (0.0 if pos.size == 0 else
                 (float(pos[0]) if sv.size == 1 else max(float(pos[0]), SIGMA_RESOLVED_KMS))
                 / CKMS)
        self.rows = []
        for lam, g, s, c, G0, G1, ln0, dln in tables:
            if not dln > 0.0:
                raise ValueError("LineWeightTable needs log-uniform filter grids")
            # the filter's own log grid, subdivided until its step is at most half the narrowest
            # line (the original knots stay on it: exact at sigma = 0), extended by the widest
            f = 1 if s_res <= 0.0 else int(min(64, max(1, np.ceil(dln / (0.5 * s_res)))))
            d = dln / f
            n_pad = int(np.ceil(NSIGMA * s_ext / d)) + 2
            knots = lam[0] * np.exp(np.arange(-n_pad, (lam.size - 1) * f + n_pad + 1) * d)
            dln = d
            E = np.empty((knots.size, sv.size))
            dE = np.zeros((knots.size, sv.size))
            for m, v in enumerate(sv):
                if v > 0.0:
                    E[:, m], dE[:, m] = _gauss_rows(lam, g, s, knots, knots * v / CKMS,
                                                    deriv=True)
                else:
                    E[:, m] = np.interp(knots, lam, g, left=0.0, right=0.0)
            self.rows.append((float(np.log(knots[0])), float(dln), jnp.asarray(knots, dtype=dt),
                              jnp.asarray(E, dtype=dt), jnp.asarray(dE, dtype=dt)))

    def _at_sigma(self, a, sigma_kms):
        """Linear interpolation along the last (sigma) axis of ``a``."""
        if a.shape[-1] == 1 or sigma_kms is None:
            return a[..., 0]
        m, h = self._sigma_weight(sigma_kms)
        return (1.0 - h) * a[..., m] + h * a[..., m + 1]

    def _sigma_weight(self, sigma_kms):
        """Node ``m`` below ``sigma_kms`` and the weight of node m + 1, linear in sigma^2 (E
        varies as sigma^2 about a node: E ~ g + sigma^2 lambda^2 g'' / 2c^2)."""
        sv = jnp.asarray(self.sv)
        v = jnp.clip(jnp.asarray(sigma_kms), 0.0, float(self.sv[-1]))
        m = jnp.clip(jnp.floor(v / self.ds).astype(jnp.int32), 0, self.sv.size - 2)
        lo, hi = sv[m], sv[m + 1]
        return m, (v * v - lo * lo) / (hi * hi - lo * lo)

    def __call__(self, lam0=None, sigma_kms=None):
        """(n_filters, n_lines) E at width ``sigma_kms`` (a traced scalar, ignored by a fixed-width
        table) and, for a sampled-redshift table, at the traced wavelengths ``lam0`` [A]."""
        if self.per_line is not None:
            return self._at_sigma(self.per_line, sigma_kms)
        x = jnp.asarray(lam0)
        lnx = jnp.log(x)
        if self.sv.size == 1 or sigma_kms is None:
            m0, wgt = 0, None
        else:
            m0, wgt = self._sigma_weight(sigma_kms)
        zero_node = jnp.asarray(self.sv == 0.0)
        out = []
        for lnk0, dln, knots, E, dE in self.rows:
            n = knots.shape[0]
            i = jnp.clip(jnp.floor((lnx - lnk0) / dln).astype(jnp.int32), 0, n - 2)
            i = jnp.clip(i - (x < knots[i]) + (x >= knots[i + 1]), 0, n - 2)
            h = knots[i + 1] - knots[i]
            t = jnp.clip((x - knots[i]) / h, 0.0, 1.0)

            def at(m):
                # linear in lambda at a sigma = 0 node (exact there), cubic Hermite otherwise
                E0, E1, D0, D1 = E[i, m], E[i + 1, m], dE[i, m], dE[i + 1, m]
                lin = (1.0 - t) * E0 + t * E1
                herm = ((2 * t ** 3 - 3 * t ** 2 + 1) * E0 + (t ** 3 - 2 * t ** 2 + t) * h * D0
                        + (-2 * t ** 3 + 3 * t ** 2) * E1 + (t ** 3 - t ** 2) * h * D1)
                return jnp.where(zero_node[m], lin, herm)

            e = at(m0) if wgt is None else (1.0 - wgt) * at(m0) + wgt * at(m0 + 1)
            inside = (x >= knots[0]) & (x <= knots[-1])
            out.append(jnp.where(inside, e, 0.0))
        return jnp.stack(out)


def line_weights_one(lam, g, s, lam0, sig):
    """E of one filter for lines at ``lam0`` with dispersions ``sig`` (0: delta); exact."""
    out = np.interp(lam0, lam, g, left=0.0, right=0.0)
    gs = (sig > 0.0) & (lam0 + NSIGMA * sig > lam[0]) & (lam0 - NSIGMA * sig < lam[-1])
    if gs.any():
        out = out.copy()
        out[gs] = _gauss_rows(lam, g, s, lam0[gs], sig[gs])
    out[(sig > 0.0) & ~gs] = 0.0
    return out


class LinePhotometry:
    """The nebular lines of one Photometry: ``maggies = E @ F`` with ``F`` the observed line fluxes
    [erg/s/cm^2] (``CSPBasis.predict_line_fluxes(theta, for_spectrum=True)``).

    ``sigma``: None or 0 for delta lines (``broaden_photometry=False``, or a fixed zero gas
    width), a number [km/s] for a fixed gas width, or ``"sampled"`` (a traced width up to
    ``sigma_max_kms``).  At a fixed redshift with a fixed width E is one exact matrix; otherwise
    a :class:`LineWeightTable` (exact for delta lines)."""

    def __init__(self, filterset, line_wave_rest, *, zred=0.0, free_z=False, sigma=None,
                 sigma_max_kms=None):
        from ceridwen.precision import float_dtype
        rest = np.asarray(line_wave_rest, dtype=np.float64)
        self.rest = jnp.asarray(rest)
        self.free_z = bool(free_z)
        self.sampled = isinstance(sigma, str)
        s = 0.0 if (sigma is None or self.sampled) else float(sigma)
        self.W = None
        self.table = None
        if not self.free_z and not self.sampled:
            lam0 = rest * (1.0 + float(zred))
            self.W = jnp.asarray(line_weights(filterset, lam0, lam0 * s / CKMS),
                                 dtype=float_dtype())
        elif not self.free_z:
            self.table = LineWeightTable(filterset, lam0=rest * (1.0 + float(zred)),
                                         sigma_max_kms=sigma_max_kms)
        elif self.sampled:
            self.table = LineWeightTable(filterset, sigma_max_kms=sigma_max_kms)
        else:
            self.table = LineWeightTable(filterset, sigma_kms=s)

    def weights(self, opz=None, sigma_kms=None):
        """(n_filters, n_lines) E: maggies per unit line flux; ``opz`` = 1 + z (traced) for a
        sampled redshift, ``sigma_kms`` the traced gas width for a sampled one."""
        if self.W is not None:
            return self.W
        if not self.free_z:
            return self.table(sigma_kms=sigma_kms)
        return self.table(self.rest * opz, sigma_kms if self.sampled else None)

    def __call__(self, line_flux, opz=None, sigma_kms=None):
        """Maggies (n_filters,) of the lines ``line_flux`` (n_lines,) (see :meth:`weights`)."""
        E = self.weights(opz=opz, sigma_kms=sigma_kms)
        return E @ jnp.asarray(line_flux).astype(E.dtype)
