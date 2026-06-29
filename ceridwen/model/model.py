"""
ceridwen/model/model.py
=======================
Parameter manager and model-prediction layer for Ceridwen SED fitting.

Design
------
``SedModel`` is the central object connecting the forward model (CSPBasis) to
a set of observations and a prior specification.  It satisfies two interfaces
simultaneously:

1. **Model interface** (consumed by ``MultiObservationLikelihood``):
   ``model.predict(theta)`` must return a ``dict[str, Array]`` keyed by
   observation name, where each value is the model prediction for that
   datum type.

2. **Prior interface** (consumed by ``MultiObservationLikelihood.make_lnprobfn``
   as the ``prior`` argument):
   ``prior.log_prob(theta)`` must return a scalar log-prior.

Both are implemented on this single class; there is no separate "prior manager"
object.  This mirrors Prospector's ``SpecModel`` pattern while keeping the
code as lean as possible.

Architecture::

    SedModel
    ├── csp             : CSPBasis instance (forward model)
    ├── observations    : list[Observation]
    ├── priors          : dict[str, Prior]
    ├── transforms      : dict[derived_name → callable(free_theta)]
    ├── predict(theta)       → dict[obs.name → Array]  (calls csp.predict)
    ├── apply_transforms(θ)  → dict  (free_theta → model_theta)
    ├── ln_prior(theta)      → scalar  (sum of log-priors on free params)
    ├── log_prob(theta)      → scalar  (alias for ln_prior)
    ├── theta_init           → dict[str → Array]  (free parameters only)
    ├── param_names          → list[str]  (free parameters only)
    └── obs_dict             → dict[str → Observation]

Compatibility with ``likelihood.py``
-------------------------------------
``MultiObservationLikelihood.make_lnprobfn(observations, model, prior)``
expects::

    observations : dict[str, Observation]   # keyed by obs.name
    model.predict(theta) → dict[str, Array] # same keys
    prior.log_prob(theta) → Array           # scalar

Usage::

    model = SedModel(csp, observations=[phot, spec], priors={
        "sfh":  Uniform(low=0.0, high=1.0),
        "Z":    Normal(mean=-1.5, sigma=0.5),
    })

    # Forward prediction
    preds = model.predict(theta)   # {"my_phot": Array(n_filters,), ...}

    # Prior
    lnp = model.ln_prior(theta)

    # Build JIT-compiled posterior for blackjax
    from ceridwen.likelihood.likelihood import MultiObservationLikelihood
    lnprobfn = multi_lhood.make_lnprobfn(model.obs_dict, model, model)
"""

from __future__ import annotations

import pprint
from functools import cached_property
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from ceridwen.observation.observation import Observation, Photometry, Spectrum, Lines

Array = jax.Array


class SedModel:
    """
    Parameter manager + prediction layer for Ceridwen SED fitting.

    Parameters
    ----------
    csp : CSPBasis
        Initialised composite stellar population model.  Must expose
        ``csp.wave``, ``csp.theta_init``, and
        ``csp.predict(theta, observations)``.
    observations : list of Observation
        Data containers (Photometry, Spectrum, Lines).  Each must have a
        unique ``.name`` attribute.
    priors : dict[str, Prior], optional
        Mapping from *free*-parameter name to a prior object implementing
        ``logpdf(x) -> Array``.  Parameters absent from this dict receive
        no prior contribution (flat improper prior).
    transforms : dict[str, callable], optional
        Mapping from *derived* (CSP) parameter name to a callable that
        computes its value from the free-parameter dict::

            model_theta[derived] = fn(free_theta)

        Derived parameters listed here are removed from the free-parameter
        list and replaced by the new free parameters supplied via
        ``free_param_init``.  This mirrors Prospector's ``depends_on``
        mechanism.

        Example — fitting log-ratios of SFR bins instead of raw SFH::

            from ceridwen.model.transforms import logsfr_ratios_to_sfh

            transforms = {
                "sfh": lambda t: logsfr_ratios_to_sfh(
                    t["logsfr_ratios"],
                    sfh_times_yr=csp.sfh_times,
                )
            }

    free_param_init : dict[str, Array], optional
        Initial values for free parameters that *replace* derived ones.
        Keys in this dict are added to ``param_names`` and ``theta_init``;
        the corresponding derived params (``transforms`` keys) are removed.
        Required when ``transforms`` is not empty.

    Attributes
    ----------
    theta_init : dict[str, Array]
        Initial values for the **free** parameters only (derived params
        are absent; their replacements from ``free_param_init`` are present).
    param_names : list[str]
        Ordered list of free-parameter names.
    transforms : dict[str, callable]
        Registered transforms (empty dict if none).
    obs_dict : dict[str, Observation]
        Observations keyed by ``obs.name``.
    wave : Array, shape (n_wave,)
        Model wavelength grid [Å].
    """

    def __init__(
        self,
        csp,
        observations: Sequence[Observation],
        priors: dict[str, Any] | None = None,
        transforms: dict[str, Callable] | None = None,
        free_param_init: dict[str, Any] | None = None,
        zred: float = 0.0,
    ):
        self.csp          = csp
        self.observations = list(observations)
        self.priors       = dict(priors) if priors is not None else {}
        self.transforms   = dict(transforms) if transforms is not None else {}
        self.zred         = float(zred)

        # Validate that all observation names are unique
        names = [obs.name for obs in self.observations]
        if len(names) != len(set(names)):
            dups = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"Observation names must be unique.  Duplicates found: {dups}"
            )

        # Start from the CSP's full parameter set
        self.theta_init  = dict(csp.theta_init)
        self.param_names = list(csp.param_names)
        self.wave        = csp.wave

        # Apply transforms bookkeeping:
        #   1. Remove derived parameters (they are outputs of transforms, not
        #      free parameters that the sampler proposes).
        #   2. Add the new free parameters supplied via free_param_init.
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

        # Precompute static projection matrices for Spectrum and Lines.
        # This must happen once, at Python level, BEFORE any JIT trace of
        # predict().  Each observation's setup_for_model() stores a constant
        # JAX array (_H for Spectrum, _W for Lines) that XLA constant-folds
        # at trace time — meaning the GPU kernel contains no matrix construction,
        # only a single GEMV.  Photometry.setup_for_model() is a no-op.
        #
        # ``zred`` bakes the (1+z) factor into the projection matrices so the
        # GEMV fast path stays the same shape for non-zero fixed redshift.
        # Combine with setting ``theta_init['zred'] = jnp.array([zred])`` so
        # that CSPBasis.predict also applies the cosmological flux-factor —
        # both things are needed for observed-frame calibration.
        for obs in self.observations:
            obs.setup_for_model(self.wave, zred=self.zred)

        # If the user supplied a non-trivial fixed redshift, seed theta_init
        # so the CSP forward model applies the matching cosmological flux
        # factor.  Leaving zred out of theta_init entirely (the default at
        # zred = 0) means zero branches in CSPBasis.predict.
        #
        # When astropy is installed we prefer its Planck18 luminosity
        # distance for this one-off scalar computation (it includes
        # neutrinos + radiation and matches published tables to <0.1%),
        # and bake the resulting flux factor into a static JAX scalar.
        # The sampled path (when zred is free) continues to use the
        # native differentiable backend, so NUTS gradients still work.
        #
        # GOTCHA: only seed theta_init['zred'] when there is no user-supplied
        # ``zred`` transform.  If the user registered transforms={"zred": ...}
        # they are explicitly injecting zred at predict time from a
        # fixed external value; adding zred to theta_init on top would
        # let NUTS sample it unconstrained (no prior -> no bounds -> the
        # leapfrog integrator can push it into z < -1, where E(z) =
        # sqrt(Ω_m(1+z)^3 + ...) goes imaginary and the log-posterior
        # becomes NaN on the very first step).
        if self.zred != 0.0 and "zred" not in self.transforms:
            self.theta_init.setdefault("zred", jnp.array([self.zred]))
            try:
                from ..cosmology import (
                    flux_factor_maggies, have_astropy,
                )
                if have_astropy():
                    ff = float(flux_factor_maggies(
                        self.zred, backend="astropy"))
                    # Stored for diagnostics; the free-z fit path ignores
                    # this and recomputes via the native JAX backend.
                    self.flux_factor_astropy = ff
            except Exception:
                # Non-fatal: fall through to the native backend.
                self.flux_factor_astropy = None

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def apply_transforms(self, free_theta: dict[str, Array]) -> dict[str, Array]:
        """
        Apply all registered transforms to produce a CSP-compatible model_theta.

        Starts from a shallow copy of ``free_theta`` and computes each
        derived parameter by calling the corresponding transform callable::

            model_theta[derived] = transform_fn(free_theta)

        The free-parameter keys (e.g. ``"logsfr_ratios"``) are kept in
        ``model_theta`` alongside the derived ones; the CSP simply ignores
        any keys it does not recognise.

        Parameters
        ----------
        free_theta : dict[str, Array]
            Free-parameter dict as used by the sampler.

        Returns
        -------
        model_theta : dict[str, Array]
            Extended dict suitable for ``csp.predict``.  Contains all
            entries of ``free_theta`` plus the derived parameter values.
        """
        if not self.transforms:
            return free_theta
        model_theta = dict(free_theta)
        for derived_param, fn in self.transforms.items():
            model_theta[derived_param] = fn(free_theta)
        return model_theta

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, theta: dict[str, Array]) -> dict[str, Array]:
        """
        Project the CSP model spectrum onto all observations.

        If transforms are registered, ``theta`` is treated as the
        *free*-parameter dict; ``apply_transforms`` is called first to
        obtain the CSP-compatible model_theta before forwarding to
        ``csp.predict``.

        Internally calls ``csp.predict(model_theta, observations)`` which
        computes the spectrum once and projects it onto each observation:

        - ``Photometry`` → synthetic AB maggies via filter convolution
        - ``Spectrum``   → model F_ν interpolated onto observed wavelength grid
        - ``Lines``      → Gaussian-aperture integrated line fluxes

        **Mass scaling** — if ``"logmass"`` is present in ``theta``, the
        spectrum is multiplied by ``10 ** logmass`` inside ``csp.predict()``
        *before* projection.  The ``logsfr_ratios_to_sfh`` transform
        normalises the SFH so that the trapezoidal integral of SFR over
        the lookback grid equals 1 M⊙ (Prospector / FSPS convention), so
        this factor sets the physical amplitude for a galaxy with stellar
        mass ``M = 10^logmass`` M⊙.  Scaling once before projection is
        more efficient than scaling each observation separately.

        Parameters
        ----------
        theta : dict[str, Array]
            Free-parameter dict (before any transforms).  May optionally
            include ``"logmass"`` (shape ``(1,)``).

        Returns
        -------
        dict[str, Array]
            Keyed by ``obs.name`` for each observation in ``self.observations``.
        """
        model_theta = self.apply_transforms(theta)
        # Mass scaling is handled inside csp.predict() — the spectrum is
        # scaled once before projection, rather than per-observation.
        return self.csp.predict(model_theta, self.observations)

    # ------------------------------------------------------------------
    # JIT-compiled prediction (for interactive / PPC use)
    # ------------------------------------------------------------------

    def predict_jit(self, theta: dict[str, Array]) -> dict[str, Array]:
        """
        JIT-compiled version of :meth:`predict`.

        Identical semantics, but the first call triggers XLA compilation
        and subsequent calls with the same dict structure hit the compiled
        cache.  Use this for interactive evaluation (sanity checks,
        posterior predictive checks) outside the sampler hot path, where
        ``run_sampler`` already wraps the full log-posterior in ``@jax.jit``.

        For vectorised evaluation over many parameter draws, prefer
        :meth:`predict_vmap`.
        """
        # Built once on first access via cached_property (avoids tracing at
        # __init__ time, before observations are set up) and cached on the
        # instance thereafter.
        return self._predict_jit_fn(theta)

    @cached_property
    def _predict_jit_fn(self) -> Callable[[dict[str, Array]], dict[str, Array]]:
        return jax.jit(self.predict)

    def predict_vmap(
        self,
        theta_batch: dict[str, Array],
    ) -> dict[str, Array]:
        """
        Vectorised prediction over a batch of parameter dicts.

        Parameters
        ----------
        theta_batch : dict[str, Array]
            Each value has a leading batch dimension, e.g.
            ``{"logsfr_ratios": (N, 4), "Z": (N, 1), "logmass": (N, 1)}``.

        Returns
        -------
        dict[str, Array]
            Each value has a leading batch dimension, e.g.
            ``{"optical_spec": (N, n_pix)}``.
        """
        return self._predict_vmap_fn(theta_batch)

    @cached_property
    def _predict_vmap_fn(self) -> Callable[[dict[str, Array]], dict[str, Array]]:
        return jax.jit(jax.vmap(self.predict))

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------

    def ln_prior(self, theta: dict[str, Array]) -> Array:
        """
        Evaluate the log-prior for all registered free parameters.

        For each parameter ``p`` in ``self.priors``, computes
        ``sum(prior.logpdf(theta[p]))`` (the ``sum`` handles vector-valued
        parameters such as a non-parametric SFH) and accumulates the total.
        Parameters absent from ``self.priors`` contribute 0 (flat prior).

        Parameters
        ----------
        theta : dict[str, Array]

        Returns
        -------
        lnp : Array, scalar
        """
        lnp = jnp.zeros(())
        for param_name, prior in self.priors.items():
            if param_name in theta:
                lnp = lnp + jnp.sum(prior.logpdf(theta[param_name]))
        return lnp

    def log_prob(self, theta: dict[str, Array]) -> Array:
        """
        Alias for ``ln_prior``.

        Required by ``DiagonalGaussianLikelihood.make_lnprobfn`` and
        ``MultiObservationLikelihood.make_lnprobfn``, which call
        ``prior.log_prob(theta)``.

        Parameters
        ----------
        theta : dict[str, Array]

        Returns
        -------
        Array, scalar
        """
        return self.ln_prior(theta)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def obs_dict(self) -> dict[str, Observation]:
        """
        Observations as a dict keyed by ``obs.name``.

        Pass this to ``MultiObservationLikelihood.make_lnprobfn`` as the
        ``observations`` argument::

            lnprobfn = multi_lhood.make_lnprobfn(model.obs_dict, model, model)
        """
        return {obs.name: obs for obs in self.observations}

    @property
    def n_obs(self) -> int:
        """Number of registered observation objects."""
        return len(self.observations)

    def summary(self) -> str:
        """
        Return a multi-line human-readable summary of the model configuration.

        Covers: registered free parameters (with shapes and prior types),
        active transforms (free → derived param mappings), all observation
        objects, and the CSP physics switch configuration.
        """
        lines = [
            "SedModel",
            "=" * 50,
            f"CSP spectrum model : {self.csp.get_spectrum.__name__}",
            f"Wavelength range   : {float(self.wave.min()):.0f} – "
                                   f"{float(self.wave.max()):.0f} Å",
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
                "  Convention matches Prospector / FSPS.",
            ]

        lines += ["", "Observations", "-" * 40]
        for obs in self.observations:
            lines.append(f"  {obs!r}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Graphical model visualisation
    # ------------------------------------------------------------------

    def display(
        self,
        ax=None,
        figsize: tuple[float, float] | None = None,
        return_fig: bool = False,
    ):
        """
        Draw a publication-quality probabilistic graphical model (PGM) diagram.

        The diagram follows standard PGM conventions:

        - Open circles       — stochastic latent variables (free parameters θᵢ)
        - Stacked circles    — vector-valued parameters (e.g. SFH weight vector)
        - Double-bordered rectangle — deterministic SED computation f_ν(λ)
        - Coloured rectangles — observation projection operators
        - Filled circles     — observed data (shaded = conditioned upon)
        - Dashed arrows      — prior ↦ parameter dependency (ε ≡ stochastic edge)
        - Solid arrows       — deterministic dependency (θ → f_SED → ŷ → y)

        The figure adapts dynamically to however many parameters and
        observations are registered, making it immediately suitable for
        inclusion in a paper.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.  If *None* a new figure is created.
        figsize : (float, float), optional
            Figure size in inches ``(width, height)``.  Defaults scale
            automatically with the number of free parameters.
        return_fig : bool, optional
            If *True* return ``(fig, ax)``; otherwise call ``plt.show()``
            and return *None*.

        Returns
        -------
        (fig, ax) or None
            Only returned when ``return_fig=True``.

        Examples
        --------
        >>> model.display(return_fig=True)[0].savefig("pgm.pdf", dpi=300)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        import numpy as np

        # ── colour palette ─────────────────────────────────────────────────
        C = dict(
            bg        = "#FFFFFF",
            # parameter nodes
            param_fc  = "#FFFFFF",
            param_ec  = "#222222",
            vec_fc    = "#EEF3FF",
            vec_ec    = "#556BBB",
            # SED deterministic node
            sed_fc    = "#E6F2FB",
            sed_ec    = "#1A6098",
            # observation type nodes
            phot_fc   = "#FFF4E6",  phot_ec = "#C95800",
            spec_fc   = "#EDFAED",  spec_ec = "#276929",
            line_fc   = "#F5EEFF",  line_ec = "#6A22A8",
            # observed data nodes (filled = conditioned on)
            data_fc   = "#37474F",
            data_ec   = "#1A252B",
            data_tc   = "#FFFFFF",
            # arrows
            arr_prior = "#BBBBBB",
            arr_fwd   = "#555555",
            arr_obs   = "#777777",
        )

        def _obs_colors(obs):
            # Prefer isinstance (handles subclasses); fall back to class name.
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

        # ── label helpers ──────────────────────────────────────────────────
        _LATEX = {
            "sfh":         r"$\mathbf{w}_\mathrm{SFH}$",
            "logzsol":     r"$\log Z_\star/Z_\odot$",
            "Z":           r"$Z$",
            "zred":        r"$z$",
            "tau_dust":    r"$\hat{\tau}$",
            "tau_1":       r"$\hat{\tau}_1$",
            "tau_2":       r"$\hat{\tau}_2$",
            "dust_index":  r"$\delta_\mathrm{dust}$",
            "dust_ratio":  r"$f_\mathrm{dust}$",
            "gas_logz":    r"$\log Z_\mathrm{neb}$",
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
                # fallback: match by class name substring
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

        # ── transform colour ───────────────────────────────────────────────
        C["tr_fc"] = "#FFF8E1"   # warm amber fill
        C["tr_ec"] = "#E65100"   # deep orange border
        C["arr_tr"] = "#E65100"  # transform arrows

        # ── figure geometry ────────────────────────────────────────────────
        n_p = len(self.param_names)
        n_o = len(self.observations)
        has_transforms = bool(self.transforms)

        # 1 data-unit ≡ 1 inch when aspect='equal'
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

        # y levels — shift everything up by 0.8 when transforms are present
        _tr_shift = 0.8 if has_transforms else 0.0
        y_prior = fh - 0.80
        y_param = fh - 2.05
        y_trans = y_param - 1.15   # transform node row (only used when transforms exist)
        y_sed   = fh / 2.0 + 0.10 + (_tr_shift / 2)
        y_obs   = 1.90
        y_data  = 0.62

        r_p  = 0.34   # scalar-param radius
        r_v  = 0.36   # vector-param radius
        r_d  = 0.30   # data-node radius

        # x positions of parameters (spread across 85% of figure width)
        x0, x1 = fw * 0.07, fw * 0.93
        x_p = ([fw / 2] if n_p == 1
                else list(np.linspace(x0, x1, n_p)))

        # x positions of observation nodes
        ox0, ox1 = fw * 0.15, fw * 0.85
        x_o = ([fw / 2]           if n_o == 1
               else [fw*0.33, fw*0.67] if n_o == 2
               else list(np.linspace(ox0, ox1, n_o)))

        x_sed = fw / 2.0

        # ── drawing primitives ─────────────────────────────────────────────
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

        # ── 1 ·  prior labels ──────────────────────────────────────────────
        for i, pname in enumerate(self.param_names):
            xp    = x_p[i]
            prior = self.priors.get(pname)
            plbl  = _prior_label(prior)
            txt(xp, y_prior, plbl, fs=7.5, color="#444444",
                style="italic" if prior is None else "normal")
            # dashed stochastic edge: prior distribution → parameter
            arrow(xp, y_prior - 0.16,
                  xp, y_param + (r_v if _is_vector(pname) else r_p) + 0.05,
                  C["arr_prior"], lw=0.75, ls="dashed")

        # ── 2 ·  parameter nodes ───────────────────────────────────────────
        for i, pname in enumerate(self.param_names):
            xp  = x_p[i]
            vec = _is_vector(pname)

            if vec:
                # stacked-card visual: two offset circles
                circle(xp + 0.06, y_param - 0.06, r_v,
                       fc=C["vec_fc"], ec=C["vec_ec"],
                       lw=0.7, zorder=3, alpha=0.7)
                circle(xp, y_param, r_v,
                       fc=C["vec_fc"], ec=C["vec_ec"],
                       lw=1.3, zorder=4)
                # dimension annotation
                ds = _shape_str(pname)
                if ds:
                    txt(xp + r_v + 0.07, y_param + r_v - 0.08,
                        ds, fs=6.5, color=C["vec_ec"], ha="left",
                        style="italic")
            else:
                circle(xp, y_param, r_p,
                       fc=C["param_fc"], ec=C["param_ec"], lw=1.3)

            # parameter symbol inside circle
            lbl = _param_label(pname)
            txt(xp, y_param, lbl, fs=8.5 if not vec else 8.0,
                weight="bold", color="#111111")

        # ── 2b ·  transform nodes (deterministic diamonds) ────────────────
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

                # Diamond shape approximated via rotated rectangle (FancyBboxPatch)
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

                # Arrows: all free params that feed into this transform → transform node
                # (draw from all free params since we don't know which ones each fn uses)
                for i, pname in enumerate(self.param_names):
                    xp  = x_p[i]
                    vec = _is_vector(pname)
                    r   = r_v if vec else r_p
                    arrow(xp, y_param - r - 0.03,
                          xt,  y_trans + tr_h / 2 + 0.05,
                          C["arr_tr"], lw=0.7, ls="dashed")

                # Arrow: transform node → SED node
                arrow(xt, y_trans - tr_h / 2 - 0.04,
                      x_sed, y_sed + (max(3.8, min(fw * 0.42, n_p * 0.88))) / 2 * 0.0 + 0.42,
                      C["tr_ec"], lw=1.0)

        # ── 3 ·  SED computation node ──────────────────────────────────────
        sed_w = max(3.8, min(fw * 0.42, n_p * 0.88))
        sed_h = 0.84

        # outer box
        rect(x_sed, y_sed, sed_w, sed_h,
             C["sed_fc"], C["sed_ec"], lw=2.2)
        # inner double-border (convention for deterministic node)
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

        # ── 4 ·  arrows: parameters → SED ─────────────────────────────────
        # When transforms are present, free-param arrows go to transform
        # nodes (drawn in section 2b).  Non-transformed params still connect
        # directly to the SED node.
        for i, pname in enumerate(self.param_names):
            if has_transforms:
                # Skip — arrows already drawn in section 2b
                continue
            xp   = x_p[i]
            vec  = _is_vector(pname)
            r    = r_v if vec else r_p
            # fan tip into box proportionally
            xt   = x_sed + (xp - x_sed) * 0.30
            arrow(xp, y_param - r - 0.03,
                  xt,  y_sed + sed_h / 2 + 0.04,
                  C["arr_fwd"], lw=0.85)

        # ── 5 ·  observation nodes + data nodes ───────────────────────────
        for j, obs in enumerate(self.observations):
            xo       = x_o[j]
            fc, ec  = _obs_colors(obs)
            ow, oh  = 1.60, 0.64

            # canonical type name and projection symbol
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

            # SED → observation arrow
            xt = x_sed + (xo - x_sed) * 0.22
            arrow(xt, y_sed - sed_h / 2 - 0.04,
                  xo, y_obs + oh / 2 + 0.05,
                  ec, lw=1.1)

            # observation box
            rect(xo, y_obs, ow, oh, fc, ec, lw=1.7)
            txt(xo, y_obs + 0.13, obs_type_lbl,
                fs=8.5, weight="bold", color=ec)
            txt(xo, y_obs - 0.13, obs.name,
                fs=7.0, color="#555555",
                family="monospace")

            # observation → data arrow (with noise annotation)
            arrow(xo, y_obs - oh / 2 - 0.04,
                  xo, y_data + r_d + 0.04,
                  ec, lw=1.1)

            # noise annotation feeding into data node
            noise_x = xo + r_d + 0.44
            noise_y = y_data + 0.20
            txt(noise_x, noise_y, r"$\sigma_k$",
                fs=8.5, color="#888888")
            arrow(noise_x - 0.06, noise_y - 0.13,
                  xo + r_d + 0.04, y_data + 0.05,
                  "#BBBBBB", lw=0.75)

            # data node (filled = observed / conditioned upon)
            circle(xo, y_data, r_d,
                   fc=C["data_fc"], ec=C["data_ec"], lw=1.5)
            txt(xo, y_data, _obs_dim(obs),
                fs=7.2, color=C["data_tc"], weight="bold")

            # label below data node
            txt(xo, y_data - r_d - 0.26,
                rf"$\mathbf{{y}}_\mathrm{{{obs.name}}}$",
                fs=8, color="#333333", style="italic")

        # ── 6 ·  legend ───────────────────────────────────────────────────
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
        # dashed-arrow legend entry
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

        # ── 7 ·  title ────────────────────────────────────────────────────
        n_tr   = len(self.transforms)
        tr_str = (rf"  ·  ${n_tr}$ transform{'s' if n_tr != 1 else ''}"
                  if has_transforms else "")
        ax.set_title(
            rf"SedModel — ${n_p}$ free parameters  ·  "
            rf"${n_o}$ observation{'s' if n_o != 1 else ''}{tr_str}",
            fontsize=9.5, color="#333333", pad=3,
        )

        # no tight_layout — axes already fill the figure via add_axes

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
