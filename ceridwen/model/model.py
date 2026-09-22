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

Array = jax.Array


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

        unknown = sorted(set(self.priors) - set(self.param_names))
        if unknown:
            raise ValueError(
                f"priors given for {unknown}, which are not sampled parameters "
                f"(a prior on a derived or misspelled name would be silently ignored); "
                f"the sampled parameters are {self.param_names}")
        unpriored = [p for p in self.param_names if p not in self.priors]
        if unpriored:
            warnings.warn(
                f"no prior for sampled parameter(s) {unpriored}: NUTS treats them as "
                "improper flat, nested sampling refuses them", stacklevel=2)

        self.kinematics.validate_theta(set(self.theta_init) | set(self.transforms), self.priors)
        self._check_instrument_scales()
        if hasattr(self.csp, "register_known_theta_keys"):
            self.csp.register_known_theta_keys(
                set(self.param_names) | set(self.priors) | set(self.transforms)
                | set(self.kinematics.free_keys) | set(self._instrument_scale_keys())
            )

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
        self._check_refused_alpha_cell()

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

    def setup_observations(self):
        """Build every observation's projection for this model's grid, redshift and
        kinematics (called by ``__init__`` and by ``fitSED`` when it replaces the
        observations); drops the cached jitted predictors."""
        names = [o.name for o in self.observations]
        if len(set(names)) != len(names):
            raise ValueError(f"observation names must be unique, got {names}")
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
            f"CSP spectrum model : {self.csp.get_spectrum.__name__}",
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
                "∫SFR dt = 1 M⊙;",
                "   logmass therefore equals log10 of the total formed "
                "stellar mass.)",
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


    def display(
        self,
        ax=None,
        figsize: tuple[float, float] | None = None,
        return_fig: bool = False,
    ):
        """Draw the model as a PGM diagram; returns ``(fig, ax)`` when ``return_fig``."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch

        C = dict(
            bg        = "#FFFFFF",
            param_fc  = "#FFFFFF",
            param_ec  = "#222222",
            vec_fc    = "#EEF3FF",
            vec_ec    = "#556BBB",
            sed_fc    = "#E6F2FB",
            sed_ec    = "#1A6098",
            phot_fc   = "#FFF4E6",  phot_ec = "#C95800",
            spec_fc   = "#EDFAED",  spec_ec = "#276929",
            line_fc   = "#F5EEFF",  line_ec = "#6A22A8",
            data_fc   = "#37474F",
            data_ec   = "#1A252B",
            data_tc   = "#FFFFFF",
            arr_prior = "#BBBBBB",
            arr_fwd   = "#555555",
            arr_obs   = "#777777",
        )

        def _obs_colors(obs):
            try:
                if isinstance(obs, Photometry):
                    return (C["phot_fc"], C["phot_ec"])
                if isinstance(obs, Spectrum):
                    return (C["spec_fc"], C["spec_ec"])
                if isinstance(obs, Lines):
                    return (C["line_fc"], C["line_ec"])
            except Exception:
                pass
            return {
                "Photometry": (C["phot_fc"], C["phot_ec"]),
                "Spectrum":   (C["spec_fc"], C["spec_ec"]),
                "Lines":      (C["line_fc"], C["line_ec"]),
            }.get(type(obs).__name__, (C["param_fc"], C["param_ec"]))

        _LATEX = {
            "sfh":         r"$\mathbf{w}_\mathrm{SFH}$",
            "logzsol":     r"$\log(Z_\star/Z_\odot)$",
            "logzsol_hist": r"$\log(Z_\star/Z_\odot)(t)$",
            "logzsol_total": r"$[Z/\mathrm{H}]$",
            "zred":        r"$z$",
            "tau_dust":    r"$\hat{\tau}$",
            "tau_1":       r"$\hat{\tau}_1$",
            "tau_2":       r"$\hat{\tau}_2$",
            "dust_index":  r"$\delta_\mathrm{dust}$",
            "dust_ratio":  r"$f_\mathrm{dust}$",
            "gas_logz":    r"$\log(Z_\mathrm{gas}/Z_\odot)$",
            "gas_logu":    r"$\log U$",
            "sigma_v":     r"$\sigma_v$",
            "f_agn":       r"$f_\mathrm{AGN}$",
            "agn_tau":     r"$\tau_\mathrm{AGN}$",
            "duste_qpah":  r"$q_\mathrm{PAH}$",
            "duste_umin":  r"$U_\mathrm{min}$",
            "duste_gamma": r"$\gamma_e$",
            "mass":        r"$\log M_\star$",
            "logmass":     r"$\log_{10}\,M_\star$",
        }

        def _param_label(name: str) -> str:
            if name in _LATEX:
                return _LATEX[name]
            for k, v in _LATEX.items():
                if k in name:
                    return v
            safe = name.replace("_", r"\_")
            return rf"$\theta_{{\mathrm{{{safe}}}}}$"

        def _prior_label(prior) -> str:
            if prior is None:
                return "flat"
            cls = type(prior).__name__
            p   = prior.params
            try:
                if cls in ("Uniform", "TopHat"):
                    lo = float(p["low"]);  hi = float(p["high"])
                    return rf"$\mathcal{{U}}({lo:.3g},\,{hi:.3g})$"
                if cls == "Normal":
                    mu = float(p["mean"]); sg = float(p["sigma"])
                    return rf"$\mathcal{{N}}({mu:.3g},\,{sg:.3g})$"
                if cls == "ClippedNormal":
                    mu = float(p["mean"]); sg = float(p["sigma"])
                    return rf"$\mathcal{{N}}_c({mu:.3g},\,{sg:.3g})$"
                if cls == "LogNormal":
                    return r"$\mathrm{LogNorm}$"
                if cls == "LogUniform":
                    lo = float(p["mini"]);  hi = float(p["maxi"])
                    return rf"$\log\mathcal{{U}}({lo:.3g},\,{hi:.3g})$"
                if cls == "StudentT":
                    return r"$\mathrm{Student}\text{-}t$"
                if "Multivariate" in cls:
                    d = int(p["mean"].shape[0])
                    return rf"$\mathcal{{N}}_{{{d}d}}$"
            except Exception:
                pass
            return r"$p(\theta)$"

        def _is_vector(name: str) -> bool:
            val = self.theta_init.get(name)
            if val is None:
                return False
            shape = getattr(val, "shape", ())
            return bool(shape) and shape[0] > 1

        def _shape_str(name: str) -> str:
            val = self.theta_init.get(name)
            if val is None:
                return ""
            shape = getattr(val, "shape", ())
            if not shape or (len(shape) == 1 and shape[0] == 1):
                return ""
            if len(shape) == 1:
                return rf"$\times\,{shape[0]}$"
            return str(shape)

        def _obs_dim(obs) -> str:
            try:
                n = int(jnp.size(obs.flux))
                if isinstance(obs, Photometry):
                    return rf"$n_{{\mathrm{{filt}}}}={n}$"
                if isinstance(obs, Spectrum):
                    return rf"$n_{{\mathrm{{pix}}}}={n}$"
                if isinstance(obs, Lines):
                    return rf"$n_{{\mathrm{{lines}}}}={n}$"
                cls = type(obs).__name__
                if "Phot" in cls:
                    return rf"$n_{{\mathrm{{filt}}}}={n}$"
                if "Spec" in cls:
                    return rf"$n_{{\mathrm{{pix}}}}={n}$"
                if "Line" in cls:
                    return rf"$n_{{\mathrm{{lines}}}}={n}$"
                return rf"$n={n}$"
            except Exception:
                pass
            return rf"$\hat{{y}}$"

        C["tr_fc"] = "#FFF8E1"
        C["tr_ec"] = "#E65100"
        C["arr_tr"] = "#E65100"

        n_p = len(self.param_names)
        n_o = len(self.observations)
        has_transforms = bool(self.transforms)

        fw = max(9.0, n_p * 1.10 + 2.0)
        fh = 8.2 if has_transforms else 7.4
        if figsize is not None:
            fw, fh = figsize

        if ax is None:
            fig = plt.figure(figsize=(fw, fh), facecolor=C["bg"])
            ax  = fig.add_axes([0.01, 0.01, 0.98, 0.98],
                               facecolor=C["bg"])
            _created = True
        else:
            fig = ax.get_figure()
            _created = False

        ax.set_xlim(0, fw)
        ax.set_ylim(0, fh)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        _tr_shift = 0.8 if has_transforms else 0.0
        y_prior = fh - 0.80
        y_param = fh - 2.05
        y_trans = y_param - 1.15
        y_sed   = fh / 2.0 + 0.10 + (_tr_shift / 2)
        y_obs   = 1.90
        y_data  = 0.62

        r_p  = 0.34
        r_v  = 0.36
        r_d  = 0.30

        x0, x1 = fw * 0.07, fw * 0.93
        x_p = ([fw / 2] if n_p == 1
                else list(np.linspace(x0, x1, n_p)))

        ox0, ox1 = fw * 0.15, fw * 0.85
        x_o = ([fw / 2]           if n_o == 1
               else [fw*0.33, fw*0.67] if n_o == 2
               else list(np.linspace(ox0, ox1, n_o)))

        x_sed = fw / 2.0

        def circle(x, y, r, fc, ec, lw=1.3, zorder=3, ls="-", alpha=1.0):
            ax.add_patch(mpatches.Circle(
                (x, y), r, facecolor=fc, edgecolor=ec,
                linewidth=lw, zorder=zorder, linestyle=ls, alpha=alpha,
            ))

        def rect(x, y, w, h, fc, ec, lw=1.6, zorder=3, rr=0.12):
            ax.add_patch(FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle=f"round,pad=0,rounding_size={rr}",
                facecolor=fc, edgecolor=ec,
                linewidth=lw, zorder=zorder,
            ))

        def arrow(x1, y1, x2, y2, color, lw=0.9,
                  style="->", rad=0.0, ls="solid", zorder=2):
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style, color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    linestyle=ls,
                ),
                zorder=zorder,
            )

        def txt(x, y, s, ha="center", va="center",
                fs=9, color="#222222", weight="normal",
                style="normal", zorder=6, **kw):
            ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                    fontweight=weight, fontstyle=style,
                    zorder=zorder, **kw)

        for i, pname in enumerate(self.param_names):
            xp    = x_p[i]
            prior = self.priors.get(pname)
            plbl  = _prior_label(prior)
            txt(xp, y_prior, plbl, fs=7.5, color="#444444",
                style="italic" if prior is None else "normal")
            arrow(xp, y_prior - 0.16,
                  xp, y_param + (r_v if _is_vector(pname) else r_p) + 0.05,
                  C["arr_prior"], lw=0.75, ls="dashed")

        for i, pname in enumerate(self.param_names):
            xp  = x_p[i]
            vec = _is_vector(pname)

            if vec:
                circle(xp + 0.06, y_param - 0.06, r_v,
                       fc=C["vec_fc"], ec=C["vec_ec"],
                       lw=0.7, zorder=3, alpha=0.7)
                circle(xp, y_param, r_v,
                       fc=C["vec_fc"], ec=C["vec_ec"],
                       lw=1.3, zorder=4)
                ds = _shape_str(pname)
                if ds:
                    txt(xp + r_v + 0.07, y_param + r_v - 0.08,
                        ds, fs=6.5, color=C["vec_ec"], ha="left",
                        style="italic")
            else:
                circle(xp, y_param, r_p,
                       fc=C["param_fc"], ec=C["param_ec"], lw=1.3)

            lbl = _param_label(pname)
            txt(xp, y_param, lbl, fs=8.5 if not vec else 8.0,
                weight="bold", color="#111111")

        if has_transforms:
            n_tr    = len(self.transforms)
            tr_x0   = fw * 0.15
            tr_x1   = fw * 0.85
            x_tr    = ([fw / 2] if n_tr == 1
                       else list(np.linspace(tr_x0, tr_x1, n_tr)))
            tr_w, tr_h = 1.70, 0.52

            for j, (derived_name, fn) in enumerate(self.transforms.items()):
                xt  = x_tr[j]
                fn_name = getattr(fn, "__name__", "fn")

                ax.add_patch(FancyBboxPatch(
                    (xt - tr_w / 2, y_trans - tr_h / 2), tr_w, tr_h,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=C["tr_fc"], edgecolor=C["tr_ec"],
                    linewidth=1.5, zorder=3, linestyle="--",
                ))
                txt(xt, y_trans + 0.10,
                    rf"$\mathtt{{{fn_name}}}$",
                    fs=7.5, color=C["tr_ec"], weight="bold")
                txt(xt, y_trans - 0.12,
                    rf"$\rightarrow$ {derived_name}",
                    fs=6.5, color="#555555", style="italic")

                for i, pname in enumerate(self.param_names):
                    xp  = x_p[i]
                    vec = _is_vector(pname)
                    r   = r_v if vec else r_p
                    arrow(xp, y_param - r - 0.03,
                          xt,  y_trans + tr_h / 2 + 0.05,
                          C["arr_tr"], lw=0.7, ls="dashed")

                arrow(xt, y_trans - tr_h / 2 - 0.04,
                      x_sed, y_sed + (max(3.8, min(fw * 0.42, n_p * 0.88))) / 2 * 0.0 + 0.42,
                      C["tr_ec"], lw=1.0)

        sed_w = max(3.8, min(fw * 0.42, n_p * 0.88))
        sed_h = 0.84

        rect(x_sed, y_sed, sed_w, sed_h,
             C["sed_fc"], C["sed_ec"], lw=2.2)
        rect(x_sed, y_sed, sed_w - 0.13, sed_h - 0.13,
             "none", C["sed_ec"], lw=0.6, zorder=4)

        txt(x_sed, y_sed + 0.18,
            r"$f_\nu(\lambda\,;\,\boldsymbol{\theta})$",
            fs=13, weight="bold", color=C["sed_ec"])

        variant_raw = getattr(
            getattr(self.csp, "get_spectrum", None), "__name__", "get_spectrum"
        )
        _VARIANT_MAP = {
            "dattn_dem_neb":       "dust  ·  dust-em.  ·  neb.",
            "dattn_nodem_neb":     "dust  ·  neb.",
            "dattn_dem_noneb":     "dust  ·  dust-em.",
            "dattn_nodem_noneb":   "dust  (no neb.)",
            "nodattn_nodem_neb":   "neb. only",
            "nodattn_nodem_noneb": "stellar continuum only",
        }
        variant_key  = variant_raw.replace("get_spectrum_", "")
        variant_disp = _VARIANT_MAP.get(
            variant_key, variant_key.replace("_", " · "))
        txt(x_sed, y_sed - 0.20,
            rf"$\mathtt{{get\_spectrum}}(\boldsymbol{{\theta}})$"
            rf"  ·  {variant_disp}",
            fs=7.2, color="#2B5F8A", style="italic")

        for i, pname in enumerate(self.param_names):
            if has_transforms:
                continue
            xp   = x_p[i]
            vec  = _is_vector(pname)
            r    = r_v if vec else r_p
            xt   = x_sed + (xp - x_sed) * 0.30
            arrow(xp, y_param - r - 0.03,
                  xt,  y_sed + sed_h / 2 + 0.04,
                  C["arr_fwd"], lw=0.85)

        for j, obs in enumerate(self.observations):
            xo       = x_o[j]
            fc, ec  = _obs_colors(obs)
            ow, oh  = 1.60, 0.64

            if isinstance(obs, Photometry):
                obs_type_lbl = "Photometry"
                proj_lbl     = r"$\mathbf{T}_\mathrm{filt}\!\cdot\!f_\nu$"
            elif isinstance(obs, Spectrum):
                obs_type_lbl = "Spectrum"
                proj_lbl     = r"$\mathbf{H}\!\cdot\!f_\nu$"
            elif isinstance(obs, Lines):
                obs_type_lbl = "Lines"
                proj_lbl     = r"$\mathbf{W}\!\cdot\!f_\nu$"
            else:
                obs_type_lbl = type(obs).__name__
                proj_lbl     = r"$\hat{y}$"

            txt(xo, y_obs + oh / 2 + 0.26, proj_lbl,
                fs=7.5, color=ec, style="italic")

            xt = x_sed + (xo - x_sed) * 0.22
            arrow(xt, y_sed - sed_h / 2 - 0.04,
                  xo, y_obs + oh / 2 + 0.05,
                  ec, lw=1.1)

            rect(xo, y_obs, ow, oh, fc, ec, lw=1.7)
            txt(xo, y_obs + 0.13, obs_type_lbl,
                fs=8.5, weight="bold", color=ec)
            txt(xo, y_obs - 0.13, obs.name,
                fs=7.0, color="#555555",
                family="monospace")

            arrow(xo, y_obs - oh / 2 - 0.04,
                  xo, y_data + r_d + 0.04,
                  ec, lw=1.1)

            noise_x = xo + r_d + 0.44
            noise_y = y_data + 0.20
            txt(noise_x, noise_y, r"$\sigma_k$",
                fs=8.5, color="#888888")
            arrow(noise_x - 0.06, noise_y - 0.13,
                  xo + r_d + 0.04, y_data + 0.05,
                  "#BBBBBB", lw=0.75)

            circle(xo, y_data, r_d,
                   fc=C["data_fc"], ec=C["data_ec"], lw=1.5)
            txt(xo, y_data, _obs_dim(obs),
                fs=7.2, color=C["data_tc"], weight="bold")

            txt(xo, y_data - r_d - 0.26,
                rf"$\mathbf{{y}}_\mathrm{{{obs.name}}}$",
                fs=8, color="#333333", style="italic")

        lg_y  = 0.28
        lg_r  = 0.13
        items = [
            (C["param_fc"], C["param_ec"], r"latent $\theta_i$"),
            (C["vec_fc"],   C["vec_ec"],   r"vector param"),
            (C["data_fc"],  C["data_ec"],  r"observed $y_k$"),
            (C["sed_fc"],   C["sed_ec"],   r"deterministic node"),
        ]
        if has_transforms:
            items.append((C["tr_fc"], C["tr_ec"], r"transform node"))
        n_leg = len(items)
        xs_leg = np.linspace(fw * 0.08, fw * 0.70, n_leg)
        for lx, (lfc, lec, llbl) in zip(xs_leg, items):
            circle(lx, lg_y, lg_r, fc=lfc, ec=lec, lw=1.0, zorder=5)
            txt(lx + 0.22, lg_y, llbl,
                fs=7.5, ha="left", color="#444444")
        ax.annotate(
            "", xy=(fw * 0.82, lg_y), xytext=(fw * 0.80, lg_y),
            arrowprops=dict(
                arrowstyle="->", color=C["arr_prior"], lw=0.9,
                linestyle="dashed",
            ),
            zorder=5,
        )
        txt(fw * 0.83, lg_y, r"stochastic edge",
            fs=7.5, ha="left", color="#444444")

        n_tr   = len(self.transforms)
        tr_str = (rf"  ·  ${n_tr}$ transform{'s' if n_tr != 1 else ''}"
                  if has_transforms else "")
        ax.set_title(
            rf"SedModel — ${n_p}$ free parameters  ·  "
            rf"${n_o}$ observation{'s' if n_o != 1 else ''}{tr_str}",
            fontsize=9.5, color="#333333", pad=3,
        )


        if return_fig:
            return fig, ax
        plt.show()
        return None

    def __repr__(self) -> str:
        obs_repr = ", ".join(o.name for o in self.observations)
        return (
            f"<SedModel n_params={len(self.param_names)} "
            f"n_obs={self.n_obs} obs=[{obs_repr}]>"
        )
