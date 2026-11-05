"""
Broadband photometric observation container (AB maggies).
"""

import os

import jax.numpy as jnp
import numpy as np
from ceridwen.observation.filters import FILTER_DIR_ENV, FilterSet, resolve_filter_file
from .base import Observation

_FILTERSET_CACHE = {}


def _filter_directory(directory):
    return None if directory is None else os.path.abspath(os.path.expanduser(os.fspath(directory)))


def _filterset(filternames, directory=None):
    """One ``FilterSet`` per distinct (filter names, directory, ``$CERIDWEN_FILTER_DIR``),
    shared by every Photometry that uses it: the same names read from two directories are
    two filter sets."""
    names = tuple(filternames)
    directory = _filter_directory(directory)
    key = (names, directory, os.environ.get(FILTER_DIR_ENV) or None)
    fs = _FILTERSET_CACHE.get(key)
    if fs is None:
        fs = FilterSet(list(names)) if directory is None else FilterSet(list(names), directory=directory)
        _FILTERSET_CACHE[key] = fs
    return fs


def _filter_name(f, directory):
    """The name of a filter given as a string or a ``Filter``.  A ``Filter`` is reloaded by
    name, so it must be the file that name resolves to; anything else would be replaced."""
    if isinstance(f, (str, os.PathLike)):
        return os.fspath(f)
    name = getattr(f, "name", None)
    if not isinstance(name, str):
        raise TypeError(f"Photometry filters must be filter names or Filter objects, got {f!r}")
    src = getattr(f, "filename", None)
    if src is None:
        raise ValueError(
            f"Filter '{name}' was built from data= arrays, which Photometry cannot use: it loads "
            f"each filter by name.  Write the curve to '<dir>/{name}.par' (two columns: "
            f"wavelength [A], transmission) and pass Photometry(filter_directory=<dir>).")
    try:
        want = resolve_filter_file(name, directory, warn=False)
    except FileNotFoundError:
        want = None
    if want is None or os.path.abspath(src) != os.path.abspath(want):
        raise ValueError(
            f"Filter '{name}' was loaded from {src}, but Photometry loads filters by name and "
            + (f"would read {want}" if want else "finds no file of that name")
            + f".  Pass the name and Photometry(filter_directory="
            f"'{os.path.dirname(os.path.abspath(src))}') instead.")
    return name


class Photometry(Observation):
    """Broadband photometric observation; flux and uncertainty in AB maggies (1 maggie = 3631 Jy).

    Parameters
    ----------
    filters : list of str or Filter -- names are looked up in ``filter_directory`` when it is given,
        otherwise in ``$CERIDWEN_FILTER_DIR`` and then the bundled library; a ``Filter`` must
        come from the file its name resolves to (else ``ValueError``)
    filter_directory : str or path, optional -- directory of ``<name>.par`` files; only it is searched
    upper_limit : bool array (n_filters,) -- True = one-sided (model > data only) chi-squared penalty

    Attributes
    ----------
    flux : array (n_filters,), maggies (keyword argument)
    uncertainty : array (n_filters,), maggies -- 1-sigma (keyword argument)
    mask : bool array (n_filters,) -- True = used in the fit (keyword argument)
    """

    _kind = "photometry"
    alias = dict(
        maggies     = "flux",
        maggies_unc = "uncertainty",
        filters     = "filters",
        phot_mask   = "mask",
    )
    _meta = ("kind", "name", "filternames", "filterfiles")

    def __init__(self, filters=[], name=None, upper_limit=None, filter_directory=None, **kwargs):
        self.filter_directory = _filter_directory(filter_directory)
        self.set_filters(filters)
        self.upper_limit = (
            None if upper_limit is None
            else jnp.asarray(np.atleast_1d(upper_limit), dtype=bool)
        )
        super().__init__(name=name, **kwargs)

    def set_filters(self, filters):
        """Set the filter list from filter-name strings or ``Filter`` objects (read from
        ``self.filter_directory`` when set)."""
        directory = getattr(self, "filter_directory", None)
        if not filters:
            self.filters     = []
            self.filternames = []
            self.filterfiles = []
            self.filterset   = None
            return

        self.filternames = [_filter_name(f, directory) for f in filters]

        self.filterset = _filterset(self.filternames, directory)
        self.filters   = list(self.filterset.filters)
        self.filterfiles = [os.path.abspath(p) for p in self.filterset.filterfiles]
        self.wave_eff = [f.wave_effective for f in self.filters]
        self._wavelength = jnp.asarray([f.wave_effective for f in self.filters])

    @property
    def wavelength(self):
        """Effective wavelengths of the filters [Å], shape (n_filters,)."""
        if not self.filters:
            return jnp.array([], dtype=float)
        return self._wavelength

    def get_maggies(self, model_wave, model_fnu):
        """Synthetic maggies, shape (n_filters,), of an F_nu spectrum on ``model_wave`` [Å]
        (reference path; normalisation follows the input flux units)."""
        if self.filterset is None:
            raise ValueError("No FilterSet configured; call set_filters() first.")
        _c         = jnp.array(2.998e18)
        wave       = jnp.asarray(model_wave,  dtype=float)
        flux_flam  = jnp.asarray(model_fnu,   dtype=float) * _c / wave**2
        return self.filterset.get_sed_maggies(flux_flam, sourcewave=wave)

    def setup_for_model(self, wave_model, zred: float = 0.0):
        """Precompute the ``(n_filters, n_wave)`` float32 projection matrix ``_T`` (F_nu -> maggies)
        for the rest-frame grid ``wave_model`` [Å] observed at fixed ``zred``; required before ``predict``."""
        wm_rest = np.asarray(wave_model, dtype=np.float64)
        self._setup_zred = float(zred)
        opz = 1.0 + float(zred)
        wm = opz * wm_rest
        n_wave = len(wm)
        _c = 2.998e18

        fnu_to_flam = _c / wm**2

        # the exact integral over the model pixels (FilterSet.flux_weights): flux-conserving
        # for lines narrower than the filter grid
        T = self.filterset.flux_weights(wm, xp=np) * fnu_to_flam[None, :]
        # static window lengths for predict_at_redshift (a sampled redshift up to
        # filters.FLUX_WINDOW_ZMAX), on this rest-frame grid
        self._flux_windows = self.filterset.flux_window_widths(wm_rest)
        self._flux_windows_n = int(wm_rest.size)

        T32 = T.astype(np.float32)
        self._T = jnp.array(T32)
        self._has_precomputed_T = True
        self._broadener = None
        self._line_photometry = None
        cols = np.flatnonzero(np.any(T32 != 0.0, axis=0))
        self._T_support = (int(cols[0]), int(cols[-1]) + 1) if cols.size else (0, 0)

    @property
    def model_support(self):
        """``(lo, hi)``: the model-grid indices ``predict`` reads at the fixed redshift of
        ``setup_for_model`` (the non-zero columns of ``_T``, joined with the sigma_gal
        broadener's window when there is one); None before setup.  A Photometry projected at
        a sampled redshift reads the whole grid (``CSPBasis._support_for`` checks ``free_z``)."""
        s = getattr(self, "_T_support", None)
        if s is None:
            return None
        b = getattr(self, "_broadener", None)
        if b is not None:
            idx = np.asarray(b.window.grid.idx)
            w = (int(idx[0]), int(idx[-1]) + 1)
            s = w if s[1] <= s[0] else (min(s[0], w[0]), max(s[1], w[1]))
        return s

    def setup_broadening(self, wave_model, zred, kinematics, *, free_z=False, neb=None):
        """Build the sigma_gal broadener over the rest range the filters cover (the
        whole 912-25000 A window when ``free_z``; None for ``Kinematics.none()`` or no
        ``kinematics``) and, when ``neb`` is given, the line term ``_line_photometry``
        (``observation.line_photometry.LinePhotometry``): each nebular line's flux through
        each filter, without painting it, as a delta line without ``kinematics``
        (``broaden_photometry=False``) and as a Gaussian of the gas width with it.  Call after
        ``setup_for_model``."""
        from ..broadening import PhotometricBroadener
        from .line_photometry import LinePhotometry
        self._broadener = None
        self._line_photometry = None
        if neb is not None and self.filterset is not None:
            gas = None if kinematics is None else kinematics.effective_sigma_gas
            self._line_photometry = LinePhotometry(
                self.filterset, np.asarray(neb.nebem_line_pos, dtype=np.float64),
                zred=zred, free_z=free_z,
                sigma="sampled" if isinstance(gas, str) else gas,
                sigma_max_kms=None if kinematics is None else kinematics.sigma_max)
        if kinematics is None:
            return
        wm = np.asarray(wave_model, dtype=np.float64)
        static_zero = (kinematics.is_static
                       and float(kinematics.sigma_gal) == 0.0
                       and float(kinematics.effective_sigma_gas) == 0.0)
        if not static_zero:
            if free_z or self.filterset is None:
                wmin, wmax = max(912.0, wm[0]), min(25000.0, wm[-1])
            else:
                opz = 1.0 + float(zred)
                lam = np.asarray(self.filterset.lam, dtype=np.float64)
                wmin = max(lam.min() / opz, wm[0])
                wmax = min(lam.max() / opz, wm[-1])
            if wmin < wmax:
                self._broadener = PhotometricBroadener.build(kinematics, wm, wmin, wmax)

    def predict(self, spectrum, wave_model):
        """Synthetic AB maggies, shape (n_filters,), as ``_T @ spectrum`` (F_nu on ``wave_model``)."""
        if not getattr(self, "_has_precomputed_T", False):
            raise RuntimeError(
                "Photometry.predict() called before setup_for_model(): call "
                "phot.setup_for_model(wave_model, zred=...) once before the first "
                "predict / JIT trace (get_maggies(wave, spectrum) is the "
                "rest-frame reference path).")
        return self._T @ spectrum

    def predict_at_redshift(self, spectrum_fnu_observed, wave_rest, zred):
        """Observer-frame AB maggies, shape (n_filters,), for a traced ``zred``: ``spectrum_fnu_observed``
        must already be observer-frame F_nu (flux factor and IGM applied) on the rest-frame grid ``wave_rest`` [Å]."""
        if self.filterset is None:
            raise ValueError("No FilterSet configured; call set_filters() first.")
        wave_obs = (1.0 + jnp.asarray(zred)) * jnp.asarray(wave_rest)
        flux_flam = spectrum_fnu_observed * (2.998e18 / (wave_obs * wave_obs))
        widths = getattr(self, "_flux_windows", None)
        if widths is not None and jnp.shape(wave_rest)[0] == self._flux_windows_n:
            # each filter's window of model pixels only (static lengths from setup_for_model)
            return self.filterset.windowed_maggies(flux_flam, wave_obs, widths)
        return self.filterset.get_sed_maggies(flux_flam, sourcewave=wave_obs)

    def chi_sq(self, model_maggies):
        """Chi-squared (float) over unmasked bands; upper-limit bands are penalised only when model > data."""
        mf    = jnp.asarray(model_maggies, dtype=float)
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

    def residuals(self, model_maggies):
        """Per-filter (data - model) / sigma, shape (n_filters,); masked bands NaN, upper-limit bands 0 when model < data."""
        res = (self.flux - jnp.asarray(model_maggies, dtype=float)) / self.uncertainty

        if self.upper_limit is not None:
            res = jnp.where(
                self.upper_limit & (res >= 0.0),
                0.0,
                res,
            )
        return jnp.where(self.mask, res, jnp.nan)

    def __str__(self):
        weff = [float(f.wave_effective) for f in self.filters] if self.filters else []
        w_range = (
            "no filters"
            if not weff
            else f"{min(weff):.0f} – {max(weff):.0f} Å"
        )
        fnames = self.filternames
        if len(fnames) > 6:
            fname_str = ", ".join(fnames[:5]) + f" … (+{len(fnames)-5} more)"
        else:
            fname_str = ", ".join(fnames)
        lines = [
            f"Photometry ({self.name})",
            f"  n_filters     : {len(self.filters)}",
            f"  ndof          : {self.ndof}",
            f"  wave_eff range: {w_range}",
            f"  filters       : {fname_str}",
            f"  masked filters: {len(self.filters) - self.ndof} / {len(self.filters)}",
        ]
        return "\n".join(lines)

    def _display_str(self, max_rows: int = 80) -> str:
        """Per-filter table of name, λ_eff, flux, σ, S/N, used (the mask: True = fitted), UL."""
        header = str(self)
        flux = np.asarray(self.flux) if self.flux is not None else None
        unc  = np.asarray(self.uncertainty) if self.uncertainty is not None else None
        mask = np.asarray(self.mask)
        n = len(self.filters)

        if flux is None or n == 0:
            return header + "\n  (no flux vector)"

        ul = (np.asarray(self.upper_limit) if self.upper_limit is not None
              else np.zeros(n, dtype=bool))

        col = f"  {'#':<3}  {'filter':<28}  {'λ_eff [Å]':>12}  " \
              f"{'flux [maggies]':>14}  {'σ':>12}  {'S/N':>7}  {'used':>5}  {'UL':>3}"
        sep = "  " + "-" * (len(col) - 2)
        out = [header, "", col, sep]

        for i in range(n):
            fname = self.filternames[i] if i < len(self.filternames) else "?"
            try:
                weff = float(self.filters[i].wave_effective)
            except Exception:
                weff = float("nan")
            f = float(flux[i]) if i < len(flux) else float("nan")
            u = float(unc[i])  if (unc is not None and i < len(unc)) else float("nan")
            snr = (abs(f / u) if (u is not None and np.isfinite(u) and u > 0)
                   else float("inf"))
            m = bool(mask[i]) if i < len(mask) else False
            ulim = bool(ul[i]) if i < len(ul) else False
            ul_str = "UL" if ulim else "-"
            out.append(
                f"  {i:<3d}  {fname:<28}  {weff:>12.1f}  "
                f"{f:>14.4e}  {u:>12.4e}  {snr:>7.2f}  {str(m):>5}  {ul_str:>3}"
            )
        return "\n".join(out)
