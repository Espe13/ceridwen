"""SedModel: parameter manager, transforms, prior and prediction layer between a
CSPBasis and a set of observations."""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ceridwen.observation.observation import Observation, Photometry, Spectrum, Lines
from ceridwen.broadening import Kinematics, DEFAULT_KINEMATICS, TIED
from ceridwen.csp.csp import spectrum_model_name as _spectrum_model_name

Array = jax.Array

# agn_eline_sigma (the AGN narrow-line dispersion, km/s): its prior / fixed value must lie in
# (0, AGN_ELINE_SIGMA_MAX], the Kinematics.sigma_max default; a sampled value is warned about
# when no Spectrum has an instrumental sigma <= AGN_ELINE_SIGMA_RESOLVE x the prior's upper
# edge at any AGN line it covers
AGN_ELINE_SIGMA_MAX = float(Kinematics.__dataclass_fields__["sigma_max"].default)
AGN_ELINE_SIGMA_RESOLVE = 3.0


class SedModel:
    """Parameter manager and prediction layer: ``predict(theta)`` returns a dict keyed
    by observation name, ``log_prob(theta)`` the summed log-prior on free parameters.

    Parameters
    ----------
    priors : dict[str, Prior] -- free-parameter name -> prior; absent names get a flat improper prior
    transforms : dict[str, callable] -- derived CSP parameter name -> ``fn(free_theta)``; derived names leave the free set
    free_param_init : dict[str, Array] -- initial values of the free parameters replacing derived ones
    zred : float -- fixed redshift; zred = 0 without ``lumdist_mpc`` applies NO flux factor, so predictions
        are L_sun/Hz x 10^logmass, not maggies
    cosmo -- must equal ``csp.cosmo`` if given
    lumdist_mpc : float, Mpc -- explicit luminosity distance replacing D_L(zred) in the flux factor
    kinematics : Kinematics -- the galaxy's stellar / gas velocity dispersions [km/s], fixed
        (float) or sampled (theta key); default ``DEFAULT_KINEMATICS`` = 300 km/s, stars and gas
    broaden_photometry : bool -- apply the kinematic broadening to the spectrum entering the
        filters (default True; below 5e-4 mag for broad bands, per cent for a narrow band on a line)
    """

    def __init__(
        self,
        csp,
        observations: Sequence[Observation],
        priors: dict[str, Any] | None = None,
        transforms: dict[str, Callable] | None = None,
        free_param_init: dict[str, Any] | None = None,
        zred: float = 0.0,
        cosmo=None,
        lumdist_mpc: float | None = None,
        kinematics: Kinematics | None = None,
        broaden_photometry: bool = True,
    ):
        self.csp          = csp
        self.observations = list(observations)
        self.priors       = dict(priors) if priors is not None else {}
        self.transforms   = dict(transforms) if transforms is not None else {}
        self.zred         = float(zred)
        if kinematics is None:
            kinematics = DEFAULT_KINEMATICS
        if not isinstance(kinematics, Kinematics):
            raise TypeError("kinematics must be a ceridwen.broadening.Kinematics "
                            f"(e.g. Kinematics(sigma_gal=250.0)), got {type(kinematics).__name__}")
        self.kinematics   = kinematics
        self.broaden_photometry = bool(broaden_photometry)

        if not hasattr(csp, "cosmo"):
            raise TypeError(
                f"{type(csp).__name__} carries no .cosmo; build the CSP with "
                "cosmo=Cosmology.planck18() (or another Cosmology)")
        if cosmo is not None and cosmo != csp.cosmo:
            raise ValueError(
                "SedModel(cosmo=...) differs from the CSP's cosmology, and the "
                "CSP is what evaluates distances and ages:\n"
                f"    CSP     : {csp.cosmo.describe()}\n"
                f"    SedModel: {cosmo.describe()}\n"
                "Set it once, at CSP construction: CSPBasis(ssp, ..., cosmo=...)"
            )

        self.lumdist_mpc = None
        if lumdist_mpc is not None:
            self.lumdist_mpc = float(lumdist_mpc)
            if not (self.lumdist_mpc > 0.0) or self.lumdist_mpc != self.lumdist_mpc:
                raise ValueError(f"lumdist_mpc must be a finite positive distance "
                                 f"in Mpc, got {lumdist_mpc}")
            if "zred" in self.transforms:
                raise ValueError("lumdist_mpc cannot be combined with a 'zred' transform")
            if self.zred > 0.0:
                warnings.warn(
                    f"lumdist_mpc = {self.lumdist_mpc:g} Mpc replaces D_L(zred = "
                    f"{self.zred:g}) = {float(csp.cosmo.luminosity_distance(self.zred)):.1f} "
                    "Mpc in the flux factor; (1+zred) still comes from zred",
                    stacklevel=2)

        names = [obs.name for obs in self.observations]
        if len(names) != len(set(names)):
            dups = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"Observation names must be unique.  Duplicates found: {dups}"
            )

        self.theta_init  = dict(csp.theta_init)
        self.param_names = list(csp.param_names)
        self.wave        = csp.wave

        if self.transforms:
            _derived = set(self.transforms.keys())
            for p in _derived:
                if p in self.theta_init:
                    del self.theta_init[p]
                if p in self.param_names:
                    self.param_names.remove(p)

        if free_param_init is not None:
            for p, v in free_param_init.items():
                arr = jnp.atleast_1d(jnp.asarray(v, dtype=float))
                self.theta_init[p] = arr
                if p not in self.param_names:
                    self.param_names.append(p)

        self._check_metallicity_setup(free_param_init)
        self._check_agn_eline_sigma_range()
        self._check_agn2_prior_ranges()

        unknown = sorted(set(self.priors) - set(self.param_names))
        if unknown:
            raise ValueError(
                f"priors given for {unknown}, which are not sampled parameters "
                f"(a prior on a derived or misspelled name would be silently ignored); "
                f"the sampled parameters are {self.param_names}.  A new name that a "
                f"transform reads (e.g. log_fagn for fagn) must also be in free_param_init")
        unpriored = [p for p in self.param_names if p not in self.priors]
        if unpriored:
            warnings.warn(
                f"no prior for sampled parameter(s) {unpriored}: NUTS treats them as "
                "improper flat, nested sampling refuses them.  Give each a prior, or fix it "
                "with a constant transform", stacklevel=2)

        self.kinematics.validate_theta(set(self.theta_init) | set(self.transforms), self.priors)
        self._check_instrument_scales()
        if hasattr(self.csp, "register_known_theta_keys"):
            self.csp.register_known_theta_keys(
                set(self.param_names) | set(self.priors) | set(self.transforms)
                | set(self.kinematics.free_keys) | set(self._instrument_scale_keys())
            )

        if float(self.zred) < 0.0 and self.lumdist_mpc is None and "zred" not in self.transforms:
            raise ValueError(
                f"SedModel(zred={float(self.zred):g}): a negative redshift has no luminosity "
                "distance (the flux factor would put the source at 10 pc).  For a nearby, "
                "blueshifted galaxy give its distance, SedModel(zred=..., lumdist_mpc=D), or "
                "use its cosmological redshift")
        self._zred_fixed = None
        self._lumdist_fixed = None
        if (self.zred != 0.0 or self.lumdist_mpc is not None) and "zred" not in self.transforms:
            self._zred_fixed = jnp.array([self.zred])
            if self.lumdist_mpc is not None:
                self._lumdist_fixed = jnp.array([self.lumdist_mpc])

        self.zred_is_free = ("zred" in self.param_names) or ("zred" in self.transforms)
        grid_rescaled = (bool(getattr(csp, "track_zred_age", False))
                         and self._zred_fixed is not None) or ("lookback_time" in self.transforms)
        if (not self.zred_is_free and not grid_rescaled
                and hasattr(csp, "sfh_times") and hasattr(csp, "age_at")):
            oldest = float(csp.sfh_times[-1]) / 1e9
            age = float(csp.age_at(self.zred))
            if oldest > age * (1.0 + 5e-3):
                raise ValueError(
                    f"the oldest SFH node, {oldest:.3f} Gyr of lookback time, "
                    f"predates the Universe at zred = {self.zred:g}: age = "
                    f"{age:.3f} Gyr under {csp.cosmo.describe()}.  Build the "
                    f"grid from csp.age_at(zred) (or cosmo.age(zred)), e.g. "
                    f"lookback_time=jnp.linspace(0.0, {age:.3f}, n)"
                )

        if ("zred" in self.param_names and not bool(getattr(csp, "track_zred_age", False))
                and "lookback_time" not in self.transforms and "zred" in self.priors
                and hasattr(csp, "sfh_times") and hasattr(csp, "age_at")):
            from ..csp.csp import prior_support
            zlo, zhi = prior_support(self.priors["zred"])
            if np.isfinite(zhi) and zhi > 0:
                oldest = float(csp.sfh_times[-1]) / 1e9
                age_hi = float(csp.age_at(float(zhi)))
                if oldest > age_hi * (1.0 + 5e-3):
                    age_lo = float(csp.age_at(max(float(zlo), 0.0)))
                    where = ("at every zred the prior allows" if oldest > age_lo * (1.0 + 5e-3)
                             else "at the high-zred end of the prior")
                    warnings.warn(
                        f"the oldest SFH node, {oldest:.3f} Gyr of lookback time, predates the "
                        f"Universe {where} (age {age_hi:.3f} Gyr at zred = {zhi:g}); with "
                        "track_zred_age=False the grid is not rescaled with the sampled "
                        "redshift.  Use CSPBasis(track_zred_age=True) or a grid within "
                        "csp.age_at(zred_max)", stacklevel=2)

        self.setup_observations()

        if (self.zred == 0.0 and self.lumdist_mpc is None and not self.zred_is_free
                and "zred" not in self.transforms and self.observations):
            warnings.warn(
                "SedModel(zred=0) applies NO flux factor: predictions are in "
                "L_sun/Hz x 10^logmass, not maggies.  Give zred= (Hubble flow) or "
                "lumdist_mpc= (nearby object) for physical units, or ignore this if "
                "the data are in the same unitless convention",
                stacklevel=2)


    def _check_metallicity_setup(self, free_param_init):
        """Refuse the removed absolute-metallicity keys, and check every metallicity prior /
        fixed value against the grid (and the nebular axis when the gas is tied)."""
        from ..csp.csp import (LOGZSOL_KEYS, REMOVED_METALLICITY_KEYS, prior_support,
                               removed_metallicity_key_error, ABSOLUTE_LOOKING_LOGZSOL)
        csp = self.csp
        l0 = getattr(csp, "log10_zsun", None)
        if l0 is None:
            return
        for where, d in (("priors", self.priors), ("transforms", self.transforms),
                         ("free_param_init", free_param_init or {}),
                         ("theta_init", self.theta_init)):
            for old in REMOVED_METALLICITY_KEYS:
                if old in d:
                    v = d[old] if where in ("free_param_init", "theta_init") else None
                    raise removed_metallicity_key_error(old, v, l0,
                                                        getattr(csp, "axis_meaning", None),
                                                        where=where)
        if getattr(csp, "gas_tied", False):
            for where, d in (("priors", self.priors), ("transforms", self.transforms),
                             ("free_param_init", free_param_init or {})):
                if "gas_logz" in d:
                    raise ValueError(
                        f"the CSP was built with gas_tied=True (gas_logz := logzsol), so "
                        f"'gas_logz' cannot also appear in {where}; drop it, or rebuild the "
                        "CSP with gas_tied=False to sample the gas metallicity separately.")
        zlo, zhi = float(np.min(csp.zmet)), float(np.max(csp.zmet))
        for name in LOGZSOL_KEYS:
            prior = self.priors.get(name)
            if prior is None:
                fixed = self._transform_value(name)
                if fixed is None:
                    continue
                lo = float(np.min(fixed)); hi = float(np.max(fixed))
                if lo < zlo - 1e-12 or hi > zhi + 1e-12:
                    raise ValueError(
                        f"the transform for {name!r} gives {np.array2string(fixed, precision=3)}, "
                        f"outside this grid's logzsol range [{zlo:+.3f}, {zhi:+.3f}] (Z_sun = "
                        f"{getattr(csp, 'zsun_nominal', float('nan')):.6g}); the metallicity "
                        "interpolation would clamp there.")
                if hi < ABSOLUTE_LOOKING_LOGZSOL:
                    warnings.warn(
                        f"the transform for {name!r} gives {np.array2string(fixed, precision=3)}, "
                        f"below logzsol = {ABSOLUTE_LOOKING_LOGZSOL} everywhere: is it an OLD "
                        f"absolute log10 Z?  It would be logzsol = "
                        f"{np.array2string(fixed - l0, precision=3)}.", stacklevel=3)
                continue
            lo, hi = prior_support(prior)
            if np.isfinite(lo) and np.isfinite(hi) and (lo < zlo - 1e-12 or hi > zhi + 1e-12):
                raise ValueError(
                    f"the prior on {name!r} covers [{lo:+.3f}, {hi:+.3f}], outside this grid's "
                    f"logzsol range [{zlo:+.3f}, {zhi:+.3f}] (Z_sun = "
                    f"{getattr(csp, 'zsun_nominal', float('nan')):.6g}); the metallicity "
                    "interpolation clamps there, which makes a degenerate posterior tail that "
                    "looks like a constraint.  Narrow the prior to the grid, e.g. "
                    f"Uniform(low={zlo:+.3f}, high={zhi:+.3f}).")
            if not (np.isfinite(lo) and np.isfinite(hi)):
                warnings.warn(
                    f"the prior on {name!r} is unbounded; logzsol is clamped to the grid "
                    f"[{zlo:+.3f}, {zhi:+.3f}] outside it.  Prefer a bounded prior "
                    "(Uniform / TopHat / ClippedNormal).", stacklevel=3)
            elif hi < ABSOLUTE_LOOKING_LOGZSOL:
                warnings.warn(
                    f"the prior on {name!r} lies entirely below logzsol = "
                    f"{ABSOLUTE_LOOKING_LOGZSOL} ([{lo:+.3f}, {hi:+.3f}]): is it an OLD "
                    f"absolute-log10-Z prior?  logzsol = log10(Z/Z_sun) is 0 at solar; the "
                    f"absolute bounds [{lo:+.3f}, {hi:+.3f}] convert to "
                    f"[{lo - l0:+.3f}, {hi - l0:+.3f}].", stacklevel=3)
        self._check_tied_gas_range()
        self._check_gas_range()
        self._check_afe_range()
        self._check_refused_alpha_cell()

    def _agn_line_hook(self):
        """The CSP's AGN hook when its AGN lines (and so ``agn_eline_sigma``) are on, else None."""
        hook = getattr(self.csp, "_agn", None)
        if hook is None or getattr(hook.model, "lines", None) is None \
                or "agn_eline_sigma" not in hook.param_names:
            return None
        return hook

    def _check_agn_eline_sigma_range(self):
        """``agn_eline_sigma`` (the AGN narrow-line dispersion, km/s): its prior, constant
        transform or fixed value must lie in (0, AGN_ELINE_SIGMA_MAX].  Raises at setup."""
        key = "agn_eline_sigma"
        if self._agn_line_hook() is None:
            return
        from ..csp.csp import prior_support
        if key in self.priors:
            lo, hi = prior_support(self.priors[key])
            what = f"the prior on {key!r} covers [{lo:g}, {hi:g}]"
        else:
            if key in self.transforms:
                try:                     # its value at theta_init (constant or not)
                    v = np.atleast_1d(np.asarray(self.transforms[key](dict(self.theta_init)),
                                                 dtype=float))
                except Exception:
                    v = None
                where = f"the transform for {key!r} gives"
            elif key in self.theta_init:
                v = np.atleast_1d(np.asarray(self.theta_init[key], dtype=float))
                where = f"theta[{key!r}] is"
            else:
                return
            if v is None or v.size == 0:
                return
            lo, hi = float(np.min(v)), float(np.max(v))
            what = f"{where} {np.array2string(v, precision=4)}"
        fix = (f"Give it a bounded prior inside (0, {AGN_ELINE_SIGMA_MAX:g}], e.g. "
               "Uniform(low=50.0, high=500.0) (Prospector's), or fix it with a constant "
               "transform, e.g. transforms={'agn_eline_sigma': lambda th: jnp.array([100.0])}.")
        if not lo > 0.0:
            raise ValueError(
                f"{key} is the Gaussian dispersion [km/s] of the AGN narrow lines and must be "
                f"> 0, but {what}.  A width <= 0 has no meaning, and the painters divide by it "
                "(Prospector's AGNSpecModel painter by agn_eline_sigma alone), so it gives NaN "
                f"predictions and gradients.  {fix}")
        if not hi <= AGN_ELINE_SIGMA_MAX:
            raise ValueError(
                f"{what}, above {AGN_ELINE_SIGMA_MAX:g} km/s (the Kinematics.sigma_max default, "
                "the widest line the broadening is built for).  The AGN lines are narrow-line-"
                "region lines with Richardson+14 'a42' NARROW-line ratios; broad-line-region "
                "widths (thousands of km/s) need a separate BLR component with its own lines, "
                "ratios and width, which CERIDWEN does not have yet, and fitting broad lines "
                f"with {key} would give them narrow-line ratios.  {fix}")

    def _check_agn_eline_sigma_resolved(self):
        """UserWarning when ``agn_eline_sigma`` has a prior (is sampled) but no Spectrum can
        resolve the AGN lines: no Spectrum, none covers an AGN line with unmasked pixels, or the
        instrumental sigma at every covered AGN line exceeds AGN_ELINE_SIGMA_RESOLVE x the
        prior's upper edge (then every width the prior allows is unresolved and the posterior is
        the prior).  Setup time (``setup_observations``), NumPy only."""
        key = "agn_eline_sigma"
        hook = self._agn_line_hook()
        if hook is None or key not in self.priors:
            return
        from ..csp.csp import prior_support
        from ..broadening import CKMS
        hi = prior_support(self.priors[key])[1]
        lines = hook.model.lines
        lam_rest = np.asarray(lines.line_wave, dtype=np.float64)
        n_spec, best = 0, None                    # best: (sigma_inst, spectrum, line, z)
        for obs in self.observations:
            if getattr(obs, "_kind", None) != "spectrum":
                continue
            n_spec += 1
            proj = getattr(obs, "_proj", None)
            if proj is None or proj.wave_obs is None:
                continue
            w = np.asarray(proj.wave_obs, dtype=np.float64)
            # the instrument the AGN painter uses (SpectralProjector.setup_agn_lines)
            s_tab = (0.5 * CKMS * np.gradient(np.log(w)) if proj.instrument is None
                     else np.asarray(proj.sigma_inst_kms, dtype=np.float64))
            if proj.free_inst_scale:
                s_tab = s_tab * float(proj.inst_scale_range[0])
            m = np.asarray(obs.mask, dtype=bool)
            used = np.sort(w[m] if m.shape == w.shape else w)
            if used.size == 0:
                continue
            opzs = [float(proj.opz_ref)]
            if proj.free_z:
                opzs += [1.0 + float(z) for z in proj.zred_range]
            for opz in opzs:
                lam = lam_rest * opz
                for j in np.flatnonzero((lam >= w.min()) & (lam <= w.max())):
                    s = float(np.interp(lam[j], w, s_tab))
                    # covered: an unmasked pixel within 2 instrumental sigma of the centre
                    half = 2.0 * lam[j] * s / CKMS
                    k = np.searchsorted(used, lam[j] - half)
                    if k >= used.size or used[k] > lam[j] + half:
                        continue
                    if best is None or s < best[0]:
                        best = (s, obs.name, lines.line_names[j], opz - 1.0)
        if n_spec == 0:
            why = ("the model has no Spectrum: Lines fluxes do not depend on the AGN line width "
                   "and Photometry only through line light near filter edges")
        elif best is None:
            why = ("no Spectrum has unmasked pixels at any AGN line (the 23 Richardson+14 lines "
                   f"at {lam_rest.min():.0f}-{lam_rest.max():.0f} A rest, at the model redshift)")
        elif best[0] > AGN_ELINE_SIGMA_RESOLVE * hi:
            why = (f"the narrowest instrumental dispersion at an AGN line is {best[0]:.0f} km/s "
                   f"({best[2]} in Spectrum {best[1]!r} at z = {best[3]:.4g}), more than "
                   f"{AGN_ELINE_SIGMA_RESOLVE:g} x the prior's upper edge {hi:g} km/s, so every "
                   "width the prior allows is unresolved")
        else:
            return
        warnings.warn(
            f"{key} is sampled but the data cannot constrain it: {why}.  Its posterior will be "
            "its prior.  Fix it instead, "
            "transforms={'agn_eline_sigma': lambda th: jnp.array([100.0])} with no prior, or "
            "add a Spectrum that resolves the AGN lines (instrumental sigma <= "
            f"{AGN_ELINE_SIGMA_RESOLVE:g} x {hi:g} km/s at a line, R_FWHM >= "
            f"{CKMS / (2.3548 * AGN_ELINE_SIGMA_RESOLVE * hi):.0f}).  See docs/agn.md, "
            "'Fitting the AGN line width'.", UserWarning, stacklevel=4)

    def _check_tied_gas_range(self):
        """gas_tied=True (B1-019): the gas metallicity is the stellar logzsol, and a logzsol
        prior / fixed value reaching outside the CLOUDY gas axis freezes the lines there (the
        stellar part is still valid, so this warns)."""
        from ..csp.csp import LOGZSOL_KEYS, prior_support
        neb = getattr(self.csp, "neb", None)
        if not getattr(self.csp, "gas_tied", False) or neb is None \
                or getattr(neb, "nebem_logz", None) is None:
            return
        g = np.asarray(neb.nebem_logz, dtype=float)
        glo, ghi = float(g.min()), float(g.max())
        for name in LOGZSOL_KEYS:
            prior = self.priors.get(name)
            if prior is not None:
                lo, hi = prior_support(prior)
            else:
                fixed = self._transform_value(name)
                if fixed is None:
                    continue
                lo, hi = float(np.min(fixed)), float(np.max(fixed))
            if lo < glo - 1e-12 or hi > ghi + 1e-12:
                warnings.warn(
                    f"gas_tied=True: the gas metallicity is {name!r}, whose range "
                    f"[{lo:+.3f}, {hi:+.3f}] leaves the CLOUDY gas axis [{glo:+.3f}, {ghi:+.3f}]; "
                    "outside it the emission lines are frozen at the edge value (the stellar "
                    "continuum still follows logzsol)", stacklevel=4)

    def _check_gas_range(self):
        """The logzsol rule for the nebular parameters (T2-002, T4-002): a bounded prior on
        ``gas_logz`` / ``gas_logu`` whose support leaves the CLOUDY axis of the grid in use, or a
        fixed value outside it, raises (the interpolation clamps there: frozen lines, a flat
        likelihood tail that looks like a constraint); an unbounded prior warns."""
        from ..csp.csp import prior_support
        neb = getattr(self.csp, "neb", None)
        if neb is None:
            return
        grid = getattr(neb, "line_file", None)
        grid = "" if grid is None else f" ({getattr(grid, 'stem', grid)})"
        for name, attrs in (("gas_logz", ("nebem_cont_logz", "nebem_line_logz")),
                            ("gas_logu", ("nebem_cont_logu", "nebem_line_logu"))):
            if name == "gas_logz" and getattr(self.csp, "gas_tied", False):
                continue                     # follows logzsol (_check_tied_gas_range)
            axes = [np.asarray(getattr(neb, a), dtype=float) for a in attrs
                    if getattr(neb, a, None) is not None]
            if not axes:
                continue
            glo = max(float(a.min()) for a in axes)
            ghi = min(float(a.max()) for a in axes)
            prior = self.priors.get(name)
            if prior is not None:
                lo, hi = prior_support(prior)
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    warnings.warn(
                        f"the prior on {name!r} is unbounded; the nebular emission is clamped "
                        f"to the CLOUDY axis [{glo:+.3f}, {ghi:+.3f}]{grid} outside it.  Prefer "
                        "a bounded prior (Uniform / TopHat / ClippedNormal).", stacklevel=4)
                    continue
                if lo < glo - 1e-9 or hi > ghi + 1e-9:
                    raise ValueError(
                        f"the prior on {name!r} covers [{lo:+.3f}, {hi:+.3f}], outside the "
                        f"nebular grid's {name} axis [{glo:+.3f}, {ghi:+.3f}]{grid}; the CLOUDY "
                        "interpolation clamps there, so the lines freeze at the edge and the "
                        "posterior piles up in a flat tail that looks like a constraint.  "
                        f"Narrow the prior to the grid, e.g. Uniform(low={glo:+.3f}, "
                        f"high={ghi:+.3f}).")
                continue
            fixed = self._transform_value(name)          # a fixed value is a constant transform
            if fixed is None:
                continue
            if float(np.min(fixed)) < glo - 1e-9 or float(np.max(fixed)) > ghi + 1e-9:
                raise ValueError(
                    f"the transform for {name!r} gives {np.array2string(fixed, precision=3)}, outside the "
                    f"nebular grid's {name} axis [{glo:+.3f}, {ghi:+.3f}]{grid}; the CLOUDY "
                    "interpolation would clamp to the edge.  Give a value inside the axis.")

    def _check_agn2_prior_ranges(self):
        """The same rule for the tabulated AGN2 components (SKIRTOR axes, the Feltre+16 NLR axes,
        and fagn < 1 in "cigale" mode): each part that defines ``check_prior_ranges(priors,
        prior_support)`` raises for a prior that leaves its grid (a fixed value outside it warns
        through the hook's ``range_messages``)."""
        hook = getattr(self.csp, "_agn", None)
        if hook is None:
            return
        from ..csp.csp import prior_support
        m = hook.model
        parts = list(m.components.values()) + [getattr(m, a, None)
                                                for a in ("polar", "lines", "blr", "screen")]
        for part in parts:
            check = getattr(part, "check_prior_ranges", None)
            if check is not None:
                check(self.priors, prior_support)

    def _check_afe_range(self):
        """The logzsol rule for [alpha/Fe] on an alpha grid (B1-011): a bounded prior or a
        constant transform outside the afe axis raises (the interpolation clamps there, a flat
        likelihood tail); an unbounded prior warns."""
        from ..csp.csp import prior_support
        grid = getattr(self.csp, "afe_grid", None)
        if grid is None:
            return
        alo, ahi = float(np.min(grid)), float(np.max(grid))
        prior = self.priors.get("afe")
        if prior is None:
            fixed = self._transform_value("afe")
            if fixed is not None and (float(np.min(fixed)) < alo - 1e-12
                                      or float(np.max(fixed)) > ahi + 1e-12):
                raise ValueError(
                    f"the transform for 'afe' gives {np.array2string(fixed, precision=3)}, "
                    f"outside this grid's [alpha/Fe] range [{alo:+.3f}, {ahi:+.3f}]; the "
                    "interpolation would clamp there.")
            return
        lo, hi = prior_support(prior)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            warnings.warn(
                f"the prior on 'afe' is unbounded; [alpha/Fe] is clamped to the grid "
                f"[{alo:+.3f}, {ahi:+.3f}] outside it.  Prefer a bounded prior.", stacklevel=4)
        elif lo < alo - 1e-12 or hi > ahi + 1e-12:
            raise ValueError(
                f"the prior on 'afe' covers [{lo:+.3f}, {hi:+.3f}], outside this grid's "
                f"[alpha/Fe] range [{alo:+.3f}, {ahi:+.3f}]; the interpolation clamps there, "
                "which makes a flat posterior tail that looks like a constraint.  Narrow the "
                f"prior to the grid, e.g. Uniform(low={alo:+.3f}, high={ahi:+.3f}).")

    def _transform_value(self, name):
        """The value a CONSTANT transform gives for ``name`` (None when there is no transform
        for it, or when it depends on sampled parameters and cannot be evaluated here)."""
        fn = self.transforms.get(name)
        if fn is None:
            return None
        try:
            v = np.atleast_1d(np.asarray(fn(dict(self.theta_init)), dtype=float))
        except Exception:
            return None
        return v if v.size and np.all(np.isfinite(v)) else None

    def _check_refused_alpha_cell(self):
        """Raise when the logzsol and afe priors both reach into a refused interpolation cell."""
        from ..csp.csp import prior_support
        csp = self.csp
        cells = getattr(csp, "refused_cells_logzsol", None)
        if cells is None:
            return
        cells = cells()
        if not cells:
            return
        zname = "logzsol" if getattr(csp, "zh_const", True) else "logzsol_hist"
        def _hi(name):
            if name in self.priors:
                return prior_support(self.priors[name])[1]
            fixed = self._transform_value(name)          # a transform-derived value counts too
            if fixed is not None:
                return float(np.max(fixed))
            v = self.theta_init.get(name)
            return float(np.max(np.asarray(v))) if v is not None else None
        z_hi, a_hi = _hi(zname), _hi("afe")
        if zname in self.transforms and self._transform_value(zname) is None:
            warnings.warn(
                f"the transform for {zname!r} depends on sampled parameters, so the refused "
                "[Fe/H] x [alpha/Fe] cell cannot be checked at construction; check it yourself "
                "with csp.refused_cells_logzsol().", stacklevel=3)
        if "afe" in self.transforms and self._transform_value("afe") is None:
            warnings.warn(
                "the transform for 'afe' depends on sampled parameters, so the refused "
                "[Fe/H] x [alpha/Fe] cell cannot be checked at construction; check it yourself "
                "with csp.refused_cells_logzsol().", stacklevel=3)
        if z_hi is None or a_hi is None:
            return
        for z_lo, z_up, a_lo, a_up, reason in cells:
            if z_hi > z_lo and a_hi > a_lo:
                raise ValueError(
                    f"the {zname} and afe ranges (up to {z_hi:+.3f} and {a_hi:+.2f}) both reach "
                    f"into the refused interpolation cell logzsol in ({z_lo:+.3f}, {z_up:+.3f}] "
                    f"x afe in ({a_lo:+.2f}, {a_up:+.2f}]: {reason}.  Cap one of them "
                    f"(logzsol <= {z_lo!r} or afe <= {a_lo!r}).")

    @property
    def cosmo(self):
        """The CSP's cosmology (read-only)."""
        return self.csp.cosmo

    def _check_calibration_names(self):
        """spectrum_scaling / spectrum_calib: the plain name only with a single
        Spectrum, ``<name>_<obs.name>`` for each of several; ambiguous, doubled or unmatched
        names raise (they would be applied to every spectrum, or sampled and never used)."""
        from .obs_params import CALIB_FAMILIES, check_names
        given = set(self.param_names) | set(self.transforms)
        if any(f.claims(n) for f in CALIB_FAMILIES for n in given):
            check_names(self.observations, CALIB_FAMILIES, given,
                        what=" (spectrum calibration)")
        # keys the CSP configuration makes inert (B1-021): sampled, they would do nothing
        csp = self.csp
        if "igm_factor" in self.param_names and hasattr(csp, "igm") and csp.igm is None:
            raise ValueError(
                "'igm_factor' is sampled but the CSP has no IGM model (add_igm=False), so it "
                "would be sampled without being used.  Drop it, or build the CSP with add_igm=True")
        if ("frac_obrun" in self.param_names and getattr(csp, "neb", None) is None
                and hasattr(csp, "flux") and not hasattr(csp, "attenuate_dust")):
            raise ValueError(
                "'frac_obrun' is sampled but the CSP has neither a nebular model nor dust "
                "attenuation, so it would be sampled without being used.  Drop it, or build the "
                "CSP with add_neb=True or dust")
        # eline_scaling multiplies only the Lines prediction (csp.py predict_line_fluxes); with
        # observations but none of them Lines it would be sampled and never used (B1-014)
        if ("eline_scaling" in self.param_names and self.observations
                and not any(getattr(o, "kind", None) == "lines" for o in self.observations)):
            raise ValueError(
                "'eline_scaling' is sampled but no observation is a Lines: it scales only the "
                "Lines prediction (not a Spectrum or Photometry), so it would be sampled "
                "without being used.  Drop it, or use spectrum_scaling for a spectrum's level")

    def setup_observations(self):
        """Build every observation's projection for this model's grid, redshift and
        kinematics (called by ``__init__`` and by ``fitSED`` when it replaces the
        observations); drops the cached jitted predictors."""
        names = [o.name for o in self.observations]
        if len(set(names)) != len(names):
            raise ValueError(f"observation names must be unique, got {names}")
        self._check_calibration_names()
        neb = getattr(self.csp, "neb", None)
        lib = getattr(self.csp, "lib_resolution", None)
        for obs in self.observations:
            kind = getattr(obs, "_kind", None)
            if kind == "spectrum":
                zr = self._spectrum_zred_range(obs) if self.zred_is_free else None
                lines_rest = None if neb is None else neb.nebem_line_pos
                if neb is None and getattr(obs, "marginalize_elines", False):
                    # no nebular grid: the marginalised lines come from FSPS's line list
                    from ..likelihood.eline_marginal import line_table_for, refuse_without_grid
                    refuse_without_grid(self.csp, obs, self.observations)
                    lines_rest = line_table_for(self.csp)["wave"]
                kw = {}
                if self._prior_scale_range(obs) is not None:
                    kw["inst_scale_range"] = self._prior_scale_range(obs)
                obs.setup_for_model(
                    self.wave, zred=(self._spectrum_zred_ref(zr) if zr else self.zred),
                    kinematics=self.kinematics, lib_resolution=lib,
                    line_wave_rest=lines_rest,
                    zred_range=zr, **kw)
                if getattr(self.csp, "_agn", None) is not None:   # CSPBasis(agn=...) lines
                    self.csp._setup_agn_spectrum(obs)
            elif kind == "photometry":
                obs.setup_for_model(self.wave, zred=self.zred)
                obs.free_z = bool(self.zred_is_free)
                obs.setup_broadening(
                    self.wave, self.zred,
                    self.kinematics if self.broaden_photometry else None,
                    free_z=bool(getattr(obs, "free_z", False)), neb=neb)
            else:
                obs.setup_for_model(self.wave, zred=self.zred)
        for cached in ("_predict_jit_fn", "_predict_vmap_fn"):
            self.__dict__.pop(cached, None)
        from ..likelihood.eline_marginal import build_eline_system
        self._eline_system = build_eline_system(self)
        self._check_agn_eline_sigma_resolved()
        if self._eline_system is not None:
            from ..precision import is_x64
            if not is_x64():
                raise ValueError(
                    "Spectrum(marginalize_elines=True) needs float64: the line-flux marginal "
                    "forms A^T W A in flux units, which float32 cannot hold (docs/mac_gpu.md).  "
                    "Run this model in float64 (JAX_PLATFORMS=cpu on a Mac), or fit the lines "
                    "with the nebular grid instead")


    def _instrument_scale_keys(self) -> tuple:
        """Theta keys of the sampled LSF scales of this model's Spectrum instruments."""
        keys = []
        for o in self.observations:
            ins = getattr(o, "instrument", None)
            if getattr(o, "_kind", None) == "spectrum" and ins is not None:
                keys += list(getattr(ins, "free_keys", ()))
        return tuple(dict.fromkeys(keys))

    def _prior_scale_range(self, obs):
        """(lo, hi) finite bounds of the prior on a Spectrum instrument's sampled scale key,
        or None (fixed scale, no prior, or an unbounded one)."""
        ins = getattr(obs, "instrument", None)
        if ins is None or not isinstance(getattr(ins, "scale", 1.0), str):
            return None
        pr = self.priors.get(ins.scale)
        b = getattr(pr, "bounds", None)
        b = b() if callable(b) else b
        if b is None:
            return None
        lo = float(np.min(np.asarray(b[0], dtype=float)))
        hi = float(np.max(np.asarray(b[1], dtype=float)))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return (lo, hi)

    def _check_instrument_scales(self):
        """A sampled Instrument scale (``Instrument(..., scale="<key>")``) is in theta, and the
        range its compiled kernel must support is known and positive: the finite bounds of its
        prior, inside the Instrument's own ``scale_range`` when it has one."""
        from ..broadening import check_scale_range
        known = set(self.theta_init) | set(self.transforms)
        for o in self.observations:
            ins = getattr(o, "instrument", None)
            if getattr(o, "_kind", None) != "spectrum" or ins is None:
                continue
            k = getattr(ins, "scale", 1.0)
            if not isinstance(k, str):
                continue
            if k not in known:
                raise KeyError(
                    f"Spectrum {o.name!r}: its Instrument samples the LSF scale as theta['{k}'], "
                    f"which is not a parameter; add it (free_param_init={{'{k}': 1.0}} and a "
                    "bounded prior) or give a float to fix it")
            rng = self._prior_scale_range(o)
            if rng is not None:
                rng = check_scale_range(rng, f"prior on '{k}'")
                own = ins.scale_range
                if own is not None and (rng[0] < own[0] or rng[1] > own[1]):
                    raise ValueError(
                        f"prior on '{k}' spans [{rng[0]:g}, {rng[1]:g}], beyond the Instrument's "
                        f"scale_range [{own[0]:g}, {own[1]:g}] that the kernel is built for; "
                        "widen scale_range or narrow the prior")
            elif ins.scale_range is None:
                why = ("its prior is unbounded" if k in self.priors
                       else "it is derived by a transform" if k in self.transforms
                       else "it has no prior")
                raise ValueError(
                    f"Spectrum {o.name!r}: the LSF scale theta['{k}'] needs a finite range for "
                    f"the compiled kernel, but {why}; give it a bounded prior (Uniform / "
                    "ClippedNormal, lower bound > 0) or pass Instrument(..., "
                    "scale_range=(lo, hi))")

    def _spectrum_zred_range(self, obs) -> tuple:
        """(z_min, z_max) a Spectrum's projector must cover when zred is sampled: the
        observation's ``zred_range``, else the finite bounds of the zred prior."""
        zr = getattr(obs, "zred_range", None)
        if zr is None and "zred" in self.priors:
            b = getattr(self.priors["zred"], "bounds", None)
            b = b() if callable(b) else b
            if b is not None:
                lo, hi = float(np.asarray(b[0]).ravel()[0]), float(np.asarray(b[1]).ravel()[0])
                if np.isfinite(lo) and np.isfinite(hi):
                    zr = (lo, hi)
        if zr is None:
            raise ValueError(
                f"Spectrum {obs.name!r} with a sampled zred needs the redshift interval its "
                "projector must cover: give priors['zred'] a bounded prior (Uniform / "
                "ClippedNormal) or pass Spectrum(zred_range=(z_min, z_max))")
        lo, hi = float(zr[0]), float(zr[1])
        if not (-1.0 < lo < hi):
            raise ValueError(f"zred_range must satisfy -1 < z_min < z_max, got {zr}")
        return (lo, hi)

    def _spectrum_zred_ref(self, zr) -> float:
        """Reference redshift of a free-z projector: the zred start value when it lies
        inside the range, else the midpoint in ln(1 + z)."""
        z0 = self.theta_init.get("zred")
        if z0 is not None:
            z0 = float(np.ravel(np.asarray(z0))[0])
            if zr[0] <= z0 <= zr[1]:
                return z0
        return float(np.exp(0.5 * (np.log1p(zr[0]) + np.log1p(zr[1]))) - 1.0)

    def apply_transforms(self, free_theta: dict[str, Array]) -> dict[str, Array]:
        """Return ``free_theta`` plus every derived parameter ``fn(free_theta)``."""
        if not self.transforms:
            return free_theta
        model_theta = dict(free_theta)
        for derived_param, fn in self.transforms.items():
            model_theta[derived_param] = fn(free_theta)
        return model_theta


    def predict(self, theta: dict[str, Array]) -> dict[str, Array]:
        """Predictions keyed by observation name from the free-parameter dict.
        A fixed zred/lumdist_mpc is injected into the CSP theta here; without a
        'zred' the CSP applies no flux factor and the outputs are not maggies."""
        model_theta = self.apply_transforms(theta)
        if self._zred_fixed is not None and "zred" not in model_theta:
            if model_theta is theta:
                model_theta = dict(model_theta)
            model_theta["zred"] = self._zred_fixed
        if self._lumdist_fixed is not None and "lumdist_mpc" not in model_theta:
            if model_theta is theta:
                model_theta = dict(model_theta)
            model_theta["lumdist_mpc"] = self._lumdist_fixed
        return self.csp.predict(model_theta, self.observations,
                                kinematics=self.kinematics,
                                broaden_photometry=self.broaden_photometry)

    def predict_with_elines(self, theta: dict[str, Array]):
        """``(predictions, aux)`` for the emission-line marginalisation: the predictions with
        the fitted lines removed, and ``aux = {"prior_mean", "cols"}`` (their CLOUDY fluxes and
        the per-observation design columns).  Needs a Spectrum with marginalize_elines=True."""
        if getattr(self, "_eline_system", None) is None:
            raise ValueError("no Spectrum of this model has marginalize_elines=True")
        model_theta = self.apply_transforms(theta)
        if self._zred_fixed is not None and "zred" not in model_theta:
            model_theta = dict(model_theta)
            model_theta["zred"] = self._zred_fixed
        if self._lumdist_fixed is not None and "lumdist_mpc" not in model_theta:
            model_theta = dict(model_theta)
            model_theta["lumdist_mpc"] = self._lumdist_fixed
        return self.csp.predict(model_theta, self.observations,
                                kinematics=self.kinematics,
                                broaden_photometry=self.broaden_photometry,
                                eline_system=self._eline_system)


    def predict_jit(self, theta: dict[str, Array]) -> dict[str, Array]:
        """JIT-compiled :meth:`predict` (compiled on first call)."""
        return self._predict_jit_fn(theta)

    @cached_property
    def _predict_jit_fn(self) -> Callable[[dict[str, Array]], dict[str, Array]]:
        return jax.jit(self.predict)

    def predict_vmap(
        self,
        theta_batch: dict[str, Array],
    ) -> dict[str, Array]:
        """Vectorised :meth:`predict` over a leading batch axis of every theta entry."""
        return self._predict_vmap_fn(theta_batch)

    @cached_property
    def _predict_vmap_fn(self) -> Callable[[dict[str, Array]], dict[str, Array]]:
        return jax.jit(jax.vmap(self.predict))


    def ln_prior(self, theta: dict[str, Array]) -> Array:
        """Scalar sum of ``prior.logpdf(theta[p])`` over registered priors."""
        lnp = jnp.zeros(())
        for param_name, prior in self.priors.items():
            if param_name in theta:
                lnp = lnp + jnp.sum(prior.logpdf(theta[param_name]))
        return lnp

    def log_prob(self, theta: dict[str, Array]) -> Array:
        """Alias for ``ln_prior``."""
        return self.ln_prior(theta)


    @property
    def obs_dict(self) -> dict[str, Observation]:
        """Observations keyed by ``obs.name``."""
        return {obs.name: obs for obs in self.observations}

    @property
    def n_obs(self) -> int:
        """Number of registered observation objects."""
        return len(self.observations)

    def summary(self) -> str:
        """Multi-line summary of parameters, transforms, observations and CSP setup."""
        lines = [
            "SedModel",
            "=" * 50,
            f"CSP spectrum model : {_spectrum_model_name(self.csp)}",
            f"Wavelength range   : {float(self.wave.min()):.0f} – "
                                   f"{float(self.wave.max()):.0f} Å",
            f"Cosmology          : {self.cosmo.describe()}",
            f"Redshift           : {self._redshift_line()}",
            f"Kinematics         : {self._kinematics_line()}",
            "",
            "Free Parameters",
            "-" * 40,
        ]
        for name in self.param_names:
            val   = self.theta_init.get(name)
            shape = getattr(val, "shape", "(scalar)")
            prior = self.priors.get(name)
            prior_str = repr(prior) if prior is not None else "flat (no prior)"
            lines.append(f"  {name:<28s}: shape {shape}  |  {prior_str}")

        if self.transforms:
            lines += ["", "Transforms  (free → derived)", "-" * 40]
            for derived, fn in self.transforms.items():
                fn_name = getattr(fn, "__name__", repr(fn))
                lines.append(f"  {derived:<20s} ← {fn_name}")

        if "logmass" in self.param_names:
            lines += [
                "",
                "Mass scaling",
                "-" * 40,
                "  logmass ∈ free params → predicted flux × 10^logmass",
                "  (SFH transform `logsfr_ratios_to_sfh` enforces "
                "∫SFR dt = 1 M⊙ on the construction grid;",
                "   the CSP keeps that mass on a track_zred_age grid, so logmass",
                "   equals log10 of the total formed stellar mass at any redshift.)",
            ]

        lines += ["", "Observations", "-" * 40]
        for obs in self.observations:
            lines.append(f"  {obs!r}")
        es = getattr(self, "_eline_system", None)
        if es is not None:
            lines += ["", "Emission lines", "-" * 40, f"  {es.describe()}"]

        return "\n".join(lines)

    def _kinematics_line(self) -> str:
        """One line for summary() and the fit log: the galaxy widths in force."""
        k = self.kinematics
        def _w(v):
            return f"theta[{v!r}]" if isinstance(v, str) else f"{float(v):g} km/s"
        gas = "= sigma_gal" if k.sigma_gas is TIED else _w(k.sigma_gas)
        return (f"sigma_gal {_w(k.sigma_gal)}, sigma_gas {gas}, sigma_max {k.sigma_max:g} km/s; "
                f"photometry {'broadened' if self.broaden_photometry else 'not broadened'}")

    def _redshift_line(self) -> str:
        """One line for summary() and the fit log: how the distance is set."""
        if self.zred_is_free:
            projs = [o._proj for o in self.observations
                     if getattr(o, "_kind", None) == "spectrum" and getattr(o, "_proj", None) is not None]
            rng = (", ".join(f"spectrum projector for z in [{p.zred_range[0]:g}, {p.zred_range[1]:g}] "
                             f"(reference {p.opz_ref - 1:g})" for p in projs)
                   if projs else "photometry projected per sample")
            return (f"zred sampled (flux factor and, with track_zred_age, the SFH grid "
                    f"follow it; {rng})"
                    + (f"; luminosity distance fixed at {self.lumdist_mpc:g} Mpc"
                       if self.lumdist_mpc is not None else ""))
        if self.lumdist_mpc is not None:
            return (f"zred = {self.zred:g} fixed, luminosity distance = "
                    f"{self.lumdist_mpc:g} Mpc (explicit)")
        if self.zred == 0.0:
            return ("zred = 0: no flux factor, predictions in L_sun/Hz x 10^logmass "
                    "(give lumdist_mpc= for a nearby object in physical units)")
        return (f"zred = {self.zred:g} fixed, D_L = "
                f"{float(self.cosmo.luminosity_distance(self.zred)):.1f} Mpc, "
                f"age = {float(self.cosmo.age(self.zred)):.3f} Gyr")


    def latex(self, *, caption: str | None = None, label: str = "tab:sedmodel",
              settings: bool = True, table: bool = True) -> str:
        """A LaTeX table (``booktabs``) of the model: every free parameter with its prior, every
        derived parameter with the transform and free parameters it is computed from, every
        fixed parameter (a constant transform) with its value, and (``settings``) the fixed
        model settings: SSP grid and IMF, SFH grid, dust laws, redshift, kinematics, the
        observations, cosmology.  ``table=False`` returns the ``tabular`` alone.  Needs
        ``\\usepackage{booktabs}`` and ``amsmath``."""
        from .describe import latex
        return latex(self, caption=caption, label=label, settings=settings, table=table)

    def display(
        self,
        ax=None,
        figsize: tuple[float, float] | None = None,
        return_fig: bool = False,
        *,
        predictive: bool = True,
        n_draws: int = 64,
        seed: int = 0,
        title: str | None = None,
    ):
        """Draw the model: free parameters with their prior densities, the derived and fixed
        parameters, the forward model's stages in the order they are applied, and the
        observations; an edge joins each parameter to the transform or stage that reads it
        (a transform's inputs are traced from its code).  Below, ``predictive`` adds the 68 and
        95 per cent range of ``n_draws`` prior draws pushed through the model, with the data
        (it compiles the model once).  With ``ax`` only the diagram is drawn, into it.
        The figure uses smplotlib's Hershey fonts when smplotlib is installed (``pip install
        smplotlib``; it is not imported, so other figures keep their style) and a serif face
        otherwise.  Returns ``(fig, ax)`` when ``return_fig``."""
        from .describe import display
        return display(self, ax=ax, figsize=figsize, return_fig=return_fig,
                       predictive=predictive, n_draws=n_draws, seed=seed, title=title)

    def __repr__(self) -> str:
        obs_repr = ", ".join(o.name for o in self.observations)
        return (
            f"<SedModel n_params={len(self.param_names)} "
            f"n_obs={self.n_obs} obs=[{obs_repr}]>"
        )
