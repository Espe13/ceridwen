"""Observed nebular emission-line flux container."""

import json
import jax.numpy as jnp
import numpy as np
from .base import Observation


class Lines(Observation):
    """Observed emission-line fluxes with line indices, rest wavelengths and uncertainties.

    Parameters
    ----------
    line_ind : array-like of int -- indices of the observed lines in the SPS emission-line array
    line_names : list of str, optional -- one name per line; needed by ``mask_by_name`` / ``select_by_name``
    wavelength : array-like of float, vacuum rest-frame Angstrom
    flux, uncertainty : array-like of float, same units as the model prediction (typically erg/s/cm^2)
    components : list of sequences of float, optional -- per observed line, the vacuum rest wavelengths
        [Angstrom] of all grid lines summed into that measurement (unresolved doublets); default one per line
    upper_limit : array-like of bool, optional -- True treats the line as a non-detection: chi^2 penalises
        only model > data
    sigma_v : float, km/s -- Gaussian aperture width of ``predict`` (integration of a painted
        spectrum); ``CSPBasis.predict`` does not use it, it reads the line fluxes from the grid
    """

    _kind = "lines"
    alias = dict(
        spectrum   = "flux",
        unc        = "uncertainty",
        wavelength = "wavelength",
        mask       = "mask",
        line_inds  = "line_ind",
    )
    _meta = ("kind", "name")
    _data = ("wavelength", "flux", "uncertainty", "mask", "line_ind")

    def __init__(
        self,
        line_ind,
        line_names  = None,
        wavelength  = None,
        name        = None,
        upper_limit = None,
        components  = None,
        sigma_v     = 200.0,
        **kwargs,
    ):
        self.sigma_v = float(sigma_v)
        if line_ind is None:
            raise ValueError(
                "line_ind is required: pass the indices of the observed lines "
                "in the FSPS emline_luminosity array."
            )
        if wavelength is None:
            raise ValueError(
                "wavelength is required: pass the wavelengths of the observed lines."
            )
        self.line_ind   = jnp.asarray(np.atleast_1d(line_ind), dtype=int)
        self.line_names = list(line_names) if line_names is not None else None
        self._wavelength = (
            None if wavelength is None
            else jnp.asarray(np.atleast_1d(wavelength), dtype=float)
        )
        self.upper_limit = (
            None if upper_limit is None
            else jnp.asarray(np.atleast_1d(upper_limit), dtype=bool)
        )
        n_lines = int(np.atleast_1d(line_ind).size)
        if components is None:
            self.line_components = [
                (float(w),) for w in np.atleast_1d(np.asarray(wavelength, dtype=float))
            ]
        else:
            comps = [tuple(float(x) for x in np.atleast_1d(c)) for c in components]
            if len(comps) != n_lines:
                raise ValueError(
                    f"components has {len(comps)} entries but there are "
                    f"{n_lines} lines")
            if any(len(c) == 0 for c in comps):
                raise ValueError("every components entry needs >= 1 wavelength")
            self.line_components = comps
        super().__init__(name=name, **kwargs)

    def to_json(self):
        """Base JSON plus ``line_names`` and ``line_components``."""
        d = json.loads(super().to_json())
        d["line_names"] = self.line_names
        d["line_components"] = [list(c) for c in self.line_components]
        return json.dumps(d)

    @property
    def has_blends(self) -> bool:
        """True when at least one observed line sums several grid lines."""
        return any(len(c) > 1 for c in getattr(self, "line_components", []))

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = (
            None if value is None
            else jnp.asarray(np.atleast_1d(value), dtype=float)
        )

    def setup_for_model(self, wave_model, zred: float = 0.0, sigma_v=None):
        """Record the model wavelength grid [Angstrom, rest, increasing], aperture ``sigma_v`` [km/s]
        and redshift used to build the ``_W`` aperture matrix; must precede ``predict``."""
        self._W_args = (np.asarray(wave_model, dtype=np.float64),
                        float(self.sigma_v if sigma_v is None else sigma_v), float(zred))
        self.__dict__.pop("_W", None)

    @property
    def _W(self):
        """(n_lines, n_wave) float32 NumPy Gaussian-aperture weights, built on first use (kept as
        NumPy so that a first use inside a jit trace never caches a tracer)."""
        if "_W" not in self.__dict__:
            if getattr(self, "_W_args", None) is None:
                raise RuntimeError("Lines.setup_for_model() has not been called")
            self.__dict__["_W"] = self._build_W(*self._W_args)
        return self.__dict__["_W"]

    def _build_W(self, wm_rest, sigma_v, zred):
        lam0_rest = np.asarray(self._wavelength, dtype=np.float64)
        c_kms  = 2.998e5
        opz = 1.0 + float(zred)

        # blends: row = pixel-wise max of the component Gaussians (no double counting of overlap)
        comps = getattr(self, "line_components", None)
        if comps is not None and any(len(c) > 1 for c in comps):
            wm = opz * wm_rest
            dlam = np.empty(len(wm), dtype=np.float64)
            dlam[1:-1] = 0.5 * (wm[2:] - wm[:-2])
            dlam[0] = 0.5 * (wm[1] - wm[0])
            dlam[-1] = 0.5 * (wm[-1] - wm[-2])
            c_aa_s = 2.998e18
            W = np.zeros((len(comps), len(wm)), dtype=np.float64)
            for k, comp in enumerate(comps):
                for lam_c in comp:
                    l0 = opz * float(lam_c)
                    sig = l0 * (sigma_v / c_kms)
                    W[k] = np.maximum(W[k], np.exp(-0.5 * ((wm - l0) / sig) ** 2))
            W = W * (dlam * c_aa_s / wm ** 2)[None, :]
            return W.astype(np.float32)
        return self._aperture_rows(wm_rest, lam0_rest, sigma_v, opz)

    @staticmethod
    def _aperture_rows(wm_rest, lam0_rest, sigma_v, opz):
        """(n_lines, n_wave) float32 aperture rows in the observed frame; rows are scaled by
        c / lambda_obs^2 so ``W @ F_nu`` is an integrated flux [erg/s/cm^2]; NaN centre -> zero row."""
        c_kms = 2.998e5
        lam0_rest = np.asarray(lam0_rest, dtype=np.float64)
        pad = ~np.isfinite(lam0_rest)
        lam0_rest = np.where(pad, 1.0, lam0_rest)

        wm   = opz * wm_rest
        lam0 = opz * lam0_rest

        dlam        = np.empty(len(wm), dtype=np.float64)
        dlam[1:-1]  = 0.5 * (wm[2:] - wm[:-2])
        dlam[0]     = 0.5 * (wm[1]  - wm[0])
        dlam[-1]    = 0.5 * (wm[-1] - wm[-2])

        c_aa_s = 2.998e18
        norm = c_aa_s / (lam0 ** 2)

        diff     = wm[None, :] - lam0[:, None]
        sigma_aa = lam0 * (sigma_v / c_kms)
        W = np.exp(-0.5 * (diff / sigma_aa[:, None]) ** 2)
        W = (W * dlam[None, :]).astype(np.float32)
        W = (W * norm[:, None].astype(np.float32))
        W[pad, :] = 0.0
        return W

    def predict(self, spectrum, wave_model):
        """Return ``_W @ spectrum``: Gaussian-aperture line fluxes, shape (n_lines,), from an F_nu
        model spectrum; ``wave_model`` is unused, ``setup_for_model`` must have been called."""
        if getattr(self, "_W_args", None) is None:
            raise RuntimeError(
                "Lines.predict() called before setup_for_model(): call "
                "lines.setup_for_model(wave_model) once before the first "
                "predict / JIT trace.")
        return jnp.asarray(self._W) @ spectrum

    def chi_sq(self, model_fluxes):
        """Return chi^2 over unmasked lines; upper-limit lines are penalised only when model > data."""
        mf    = jnp.asarray(model_fluxes, dtype=float)
        resid = (self.flux - mf) / self.uncertainty

        if self.upper_limit is not None:
            resid_sq = jnp.where(
                self.upper_limit,
                jnp.where(resid < 0.0, resid ** 2, 0.0),
                resid ** 2,
            )
        else:
            resid_sq = resid ** 2

        return float(jnp.sum(jnp.where(self.mask, resid_sq, 0.0)))

    def residuals(self, model_fluxes):
        """Return per-line ``(data - model) / sigma``, shape (n_lines,); masked lines NaN,
        upper-limit lines with model <= data set to 0."""
        mf    = jnp.asarray(model_fluxes, dtype=float)
        resid = (self.flux - mf) / self.uncertainty

        if self.upper_limit is not None:
            resid = jnp.where(
                self.upper_limit & (resid >= 0.0),
                0.0,
                resid,
            )

        return jnp.where(self.mask, resid, jnp.nan)

    def mask_by_name(self, names):
        """Set ``mask = False`` for lines whose name is in ``names``; no-op without ``line_names``."""
        if self.line_names is None:
            return
        names_set = set(names)
        exclude = jnp.array(
            [n in names_set for n in self.line_names], dtype=bool
        )
        self.mask = self.mask & ~exclude

    def select_by_name(self, names):
        """Return a new ``Lines`` containing only the named lines (KeyError on an unknown name)."""
        if self.line_names is None:
            raise ValueError(
                "line_names not set on this Lines object; "
                "cannot select by name."
            )
        idx = []
        for n in names:
            if n not in self.line_names:
                raise KeyError(
                    f"Line '{n}' not found.  Available: {self.line_names}"
                )
            idx.append(self.line_names.index(n))
        idx = np.array(idx)

        def _pick(arr):
            return None if arr is None else np.array(arr)[idx]

        return Lines(
            line_ind    = np.array(self.line_ind)[idx],
            line_names  = [self.line_names[i] for i in idx],
            wavelength  = _pick(self._wavelength),
            flux        = _pick(self.flux),
            uncertainty = _pick(self.uncertainty),
            mask        = np.array(self.mask)[idx],
            upper_limit = _pick(self.upper_limit) if self.upper_limit is not None else None,
            components  = [self.line_components[i] for i in idx],
            name        = self.name + "_sel",
        )

    def __str__(self):
        n = len(self.line_ind)
        if self.line_names is not None:
            names_str = ", ".join(self.line_names[:6])
            if n > 6:
                names_str += f" … (+{n - 6} more)"
        else:
            names_str = f"{n} lines (no names set)"

        if self._wavelength is not None:
            wmin = float(jnp.min(self._wavelength))
            wmax = float(jnp.max(self._wavelength))
            wave_str = f"{wmin:.1f} – {wmax:.1f} Å"
        else:
            wave_str = "none"

        text = [
            f"Lines ({self.name})",
            f"  n_lines       : {n}",
            f"  ndof          : {self.ndof}",
            f"  wavelength    : {wave_str}",
            f"  line_names    : {names_str}",
            f"  masked lines  : {n - self.ndof} / {n}",
        ]
        comps = getattr(self, "line_components", None)
        if comps is not None and any(len(c) > 1 for c in comps):
            nb = sum(1 for c in comps if len(c) > 1)
            names = self.line_names or [f"line{i}" for i in range(n)]
            blends = ", ".join(
                f"{names[i]} = sum of {len(c)} ({', '.join(f'{w:.1f}' for w in c)} A)"
                for i, c in enumerate(comps) if len(c) > 1)
            text.append(f"  blended lines : {nb} -> {blends}")
        return "\n".join(text)

    def _display_str(self, max_rows: int = 80) -> str:
        """Per-line table (name, index, wavelength, flux, sigma, S/N, mask)."""
        header = str(self)
        if self.flux is None or len(self.line_ind) == 0:
            return header + "\n  (no line vector)"

        inds  = np.asarray(self.line_ind)
        flux  = np.asarray(self.flux)
        unc   = (np.asarray(self.uncertainty)
                 if self.uncertainty is not None
                 else np.full_like(flux, np.nan))
        mask  = np.asarray(self.mask)
        waves = (np.asarray(self._wavelength)
                 if self._wavelength is not None
                 else np.full(len(inds), np.nan))
        names = (self.line_names if self.line_names is not None
                 else [f"line_{i}" for i in range(len(inds))])

        col = (f"  {'#':<3}  {'line_name':<28}  {'idx':>4}  "
               f"{'λ [Å]':>10}  {'flux':>14}  {'σ':>14}  "
               f"{'S/N':>7}  {'used':>5}")
        sep = "  " + "-" * (len(col) - 2)
        out = [header, "", col, sep]

        for i in range(len(inds)):
            name = names[i] if i < len(names) else "?"
            idx  = int(inds[i])
            w    = float(waves[i]) if i < len(waves) else float("nan")
            f    = float(flux[i])
            u    = float(unc[i]) if i < len(unc) else float("nan")
            snr  = (abs(f / u) if np.isfinite(u) and u > 0
                    else float("inf"))
            m    = bool(mask[i]) if i < len(mask) else False
            out.append(
                f"  {i:<3d}  {name:<28}  {idx:>4d}  "
                f"{w:>10.2f}  {f:>14.4e}  {u:>14.4e}  "
                f"{snr:>7.2f}  {str(m):>5}"
            )
        return "\n".join(out)
