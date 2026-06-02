"""
===========
Dict-based CSPBasis:
The parameter vector theta is kept as a plain Python/JAX dict at all times:

    theta = {
        "sfh":               jnp.zeros(100),   # shape (n_time,)
        "Z":                 jnp.array([0.0]),  # log10 Z/Zsun, scalar
        "gas_logz":          jnp.array([0.0]),
        "gas_logu":          jnp.array([-2.0]),
        "tau_pow":           jnp.array([1.0]),
        "diffuse_tau_kc":    jnp.array([0.3]),
        "diffuse_dust_index": jnp.array([0.0]),
        # … any other dust / emission parameters
    }

Why this is JAX-idiomatic
--------------------------
Python dicts are natively registered JAX PyTrees.  String keys are static
(part of the PyTree structure, not traced values), so passing a dict to a
``@jax.jit`` function has no overhead: JAX traces through the array values
exactly as if they were elements of a flat array.  Retracing occurs only if
the set of keys, or the shapes/dtypes of the values, change — neither of which
happens between sampling steps.

Compatibility with the likelihood module
-----------------------------------------
``DiagonalNoiseModel.compute(sigma, mu, mask, theta)`` looks up nuisance
parameters by name (``theta["log_jitter"]``).  This works transparently with
the dict theta; no ``ThetaVector`` adapter is needed.
"""

import math
import warnings

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import pprint

import astropy.constants as const

from sedpy_jax.smoothing import make_vel_smoother

from ceridwen.dust.DustModel import Dust, DiffuseDust
from ceridwen.dust.DustEmission import DustEmission
from ceridwen.neb.NebularGridModel            import NebularModel
from ceridwen.neb.NebularGridModel_fsps_match import NebularModelFSPSMatch
# Note: no Observation imports here.  Observation-type dispatch is handled by
# the polymorphic obs.predict(spectrum, wave) method on each Observation
# subclass, so CSPBasis.predict has zero isinstance/if branches.

tiny_number = 1e-70
# Plain Python constant — avoid a module-level JIT call that can fail on
# backends (e.g. Apple Metal) which do not support the configured float
# precision at package import.
LOG10E = math.log10(math.e)  # ≈ 0.4342944819032518


# ---------------------------------------------------------------------------
# Module-level JIT helper (unchanged from csp.py)
# ---------------------------------------------------------------------------

def fnu2flam(lam, fnu):
    """Convert f_nu [erg/s/cm^2/Hz] to f_lambda [erg/s/cm^2/Å]."""
    c = 2.998e18  # Å/s
    return c * (fnu / (lam ** 2))



def intsfwght(t_hi, t_lo, a, slope, logage):
    """Integrated SFH weight between log-time limits.

    Pure JAX helper. Called only from ``get_spectrum_*`` which is itself
    inlined into the outer ``@jax.jit`` lnprobfn (see
    ``MultiObservationLikelihood.make_lnprobfn``), so an inner ``@jit`` here
    is redundant — JAX would just inline it. Dropping the decorator saves a
    compile-cache slot and clarifies that this is not a separate JIT
    boundary; behaviour is unchanged.
    """

    def F(t):
        x = 10.0**t
        delta = logage - t
        return (
            a * x * (delta + LOG10E)
            + 0.5 * slope * x * x * (delta + 0.5 * LOG10E)
        )

    return F(t_hi) - F(t_lo)


# ===========================================================================
# CSPBasis
# ===========================================================================

class CSPBasis:
    """
    Composite Stellar Population basis using a dict-valued theta.

    The public interface is identical to ``csp.CSPBasis`` except that
    ``predict(theta)`` now expects (and ``theta_init`` now is) a
    ``dict[str, Array]`` rather than a flat 1-D ``jnp.ndarray``.

    Parameters
    ----------
    SSPData : SSPData
        Frozen dataclass with SSP grids (wave, flux, ages, zmet).  The
        nebular model computes its own ionising-photon rate from
        ``SSPData.ssp_flux`` at construction time, so ``log_qq`` is no
        longer carried in the SSP grid container.
    theta : dict
        Initial parameter values.  Must contain ``"sfh"`` and
        ``"lookback_time"``.  All other keys are optional.
    tuniv : float
        Age of the Universe in Gyr.  Default 13.8.
    zh_const : bool
        If True, use constant metallicity (requires key ``"Z"``).
        If False, use time-varying metallicity (requires key ``"zh"``).
    add_neb, add_dust, add_diffuse_dust, add_dust_emission : bool
        Physics switches.
    sps_home : str
        Path to the FSPS root directory (needed for nebular and dust emission
        file loading).
    init_neb_params, init_dust_params : dict
        Keyword arguments forwarded to ``NebularModel`` / ``Dust``.
    diffuse_law : str
        Attenuation law name for the diffuse dust component.
    verbose : bool
        Print parameter summary after initialization.
    """

    def __init__(
        self,
        SSPData,
        theta=None,
        tuniv=13.8,
        tiny_logt=-70,
        zh_const=False,
        add_neb=True,
        init_neb_params=None,
        nebemlineinspec=False,
        add_dust=True,
        add_diffuse_dust=True,
        add_dust_emission=False,
        add_igm=False,
        igm_model="madau1995",
        igm_factor=1.0,
        sps_home='/Users/amanda/Prospector/fsps',
        init_dust_params=None,
        diffuse_law='kriek_conroy',
        verbose=True,
        sfh_interp='step',
        sigma_losvd_kms=300.0,
        **kwargs,
    ):
        """
        sfh_interp : {'step', 'linear'}
            Controls the SFH integration scheme used when computing SSP weights.

            ``'step'`` (default) — piecewise-constant (FastStepBasis-style).
                The SFR is held at the mean of the two endpoint values within
                each SFH time bin.  The weight of each SSP age bin is the
                product of that constant SFR and the linear-time overlap between
                the SFH bin and the SSP age bin.  Weights are non-negative by
                construction — no clipping is ever needed.

            ``'linear'`` — piecewise-linear (original Ceridwen scheme).
                Analytically integrates a linearly-interpolated SFH against the
                SSP age bins in log-age space (``intsfwght``).  Higher-order
                accurate, but can produce small negative weights for steep SFH
                gradients, which are then clipped.  Kept for backwards
                compatibility.

        To switch at runtime::

            csp.calculate_ssp_weights = csp.calculate_ssp_weights_const_zh_step
            # or
            csp.calculate_ssp_weights = csp.calculate_ssp_weights_const_zh
        """
        if theta is None:
            theta = {'lookback_time': 13.8 - jnp.linspace(1e-2, 13.8, 100)}
        if init_neb_params is None:
            init_neb_params = {"isoc_type": "mist", "cloudy_dust": True}
        if init_dust_params is None:
            init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 'laws': ['powerlaw']}

        # --- SSP grids (static, never part of theta) -----------------------
        self.flux      = jnp.array(SSPData.ssp_flux, dtype=jnp.float32)  # (n_z, n_age, n_wave)
        self.wave      = jnp.array(SSPData.ssp_wave)       # (n_wave,)
        self.ages      = jnp.array(SSPData.ssp_lg_age_gyr) # (n_age,)  log10(Gyr)
        self.zmet      = jnp.array(SSPData.ssp_lgmet)      # (n_z,)    log10(Z)
        # ``log_qq`` is no longer stored on SSPData.  The nebular model
        # computes the ionising-photon rate from ``self.flux`` internally
        # (see ``initialize_neb`` and ``NebularModel.compute_log_qq``).
        self.zlegend   = 10 ** self.zmet                   # linear metallicity
        self.ssp_ages_lgyr = self.ages + 9                 # log10(yr)

        # Precomputed constants for calculate_ssp_weights (all static)
        self._logage_lo  = self.ssp_ages_lgyr[1:]
        self._logage_hi  = self.ssp_ages_lgyr[:-1]
        self._dlogage    = jnp.diff(self.ssp_ages_lgyr)
        self._j_range    = jnp.arange(self.ssp_ages_lgyr.size)
        self._age_clip_lo = 10.0 ** (-70)                  # floor for log-time clipping
        self._age_clip_hi = 10.0 ** self.ssp_ages_lgyr[-1] # ceiling
        self._n_z   = len(self.zmet)
        self._n_age = len(self.ages)

        # Precomputed SSP bin edges in linear years — retained for reference.
        # _ssp_lo_yr is the younger (smaller) edge; _ssp_hi_yr is the older
        # (larger) edge of each SSP age bin.
        self._ssp_lo_yr = 10.0 ** self._logage_hi   # (n_age-1,)
        self._ssp_hi_yr = 10.0 ** self._logage_lo   # (n_age-1,)

        # Voronoi cell boundaries for the step-function weight scheme.
        # Each SSP age POINT j owns the linear-time interval
        #   [_ssp_voronoi_lo[j], _ssp_voronoi_hi[j]]
        # where the boundaries are the midpoints to the neighbouring age points.
        #
        # This is the correct attribution for a piecewise-constant SFH: the
        # weight at SSP j equals the SFR * (width of its Voronoi cell in yr).
        # It matches what FSPS FastStepBasis does internally with ±ε offsets.
        #
        # Boundary handling:
        #   - youngest SSP (j=0): lower bound set to 0.
        #   - oldest  SSP (j=-1): upper bound set to 2× the last inter-point
        #     spacing, which safely exceeds any realistic SFH extent.
        _ssp_age_yr  = 10.0 ** self.ssp_ages_lgyr           # (n_age,) linear yr
        _voro_mid    = 0.5 * (_ssp_age_yr[:-1] + _ssp_age_yr[1:])  # (n_age-1,)
        _voro_hi_ext = _ssp_age_yr[-1] + (_ssp_age_yr[-1] - _ssp_age_yr[-2])
        self._ssp_voronoi_lo = jnp.concatenate(
            [jnp.zeros(1), _voro_mid]
        )   # (n_age,)  — lower boundary of each Voronoi cell
        self._ssp_voronoi_hi = jnp.concatenate(
            [_voro_mid, jnp.array([_voro_hi_ext])]
        )   # (n_age,)  — upper boundary of each Voronoi cell

        self.tuniv      = tuniv
        self.tiny_logt  = tiny_logt
        self.sps_home   = sps_home
        # Prospector-style ``nebemlineinspec`` switch.  It governs ONE
        # thing only: the default of the single-array public
        # ``get_spectrum(theta)``.  When False (default), that call
        # returns the line-free continuum (stellar + nebular continuum);
        # when True it returns continuum + emission lines.
        #
        # It does NOT affect ``predict`` or ``get_spectrum_components``:
        # those always compute the full ``(continuum, lines)``
        # decomposition, because the observations must always see the
        # lines -- Photometry at true strength, Spectrum / Lines scaled by
        # ``eline_scaling``.  To obtain a spectrum with or without the
        # emission lines explicitly, pass ``include_lines=`` to
        # ``get_spectrum`` or use ``get_spectrum_components``.  Mirrors
        # FSPS's ``nebemlineinspec`` parameter.
        self.nebemlineinspec = bool(nebemlineinspec)

        # --- IGM attenuation model (optional) ------------------------------
        # ``add_igm=False`` leaves ``self.igm`` as None; ``CSPBasis.predict``
        # then skips the multiplicative step entirely (zero Python
        # branches in the traced hot path — the ``is None`` is a
        # compile-time decision).  When ``add_igm=True`` the model (by
        # default Madau 1995, identical to FSPS's ``igm_absorb.f90``)
        # is applied whenever ``theta['zred']`` is present, with
        # optional runtime strength override via ``theta['igm_factor']``.
        if add_igm:
            from ..igm import make_igm_model
            self.igm = make_igm_model(igm_model)
        else:
            self.igm = None
        self.igm_factor = float(igm_factor)

        # --- Dust attenuation function (set before dust init) --------------
        if add_diffuse_dust or add_dust:
            self.set_attenuation_function(add_diffuse_dust, add_dust)

        # --- Sub-model init (populates defaults into theta) -----------
        theta = self.initialize_dust_components(
            add_dust, add_diffuse_dust, add_dust_emission,
            theta, init_dust_params, diffuse_law, sps_home,
        )
        theta = self.initialize_neb(add_neb, theta, init_neb_params, sps_home)

        # Pre-compute the FFT-domain Gaussian kernel for the FSPS-default
        # LOSVD smoothing.  Matches ``prospect/models/sedmodel.py``::
        # ``losvd_smoothing`` (sigma_smooth default 300 km/s on rest-frame
        # 912 < lambda < 25000 AA, velocity-space Gaussian convolution).
        # Must be done BEFORE configure_spectrum_model so the wrap can
        # see whether to install the smoother.
        self.sigma_losvd_kms = float(sigma_losvd_kms)
        self._setup_losvd_kernel()

        self.configure_spectrum_model(
            add_dust, add_diffuse_dust, add_dust_emission, add_neb, sps_home
        )

        # --- SFH integration scheme selection ---------------------------------
        # 'step'   → piecewise-constant, guaranteed non-negative (default)
        # 'linear' → piecewise-linear log-age integration (original scheme)
        if sfh_interp not in ('step', 'linear'):
            raise ValueError(
                f"sfh_interp must be 'step' or 'linear', got {sfh_interp!r}"
            )
        self.sfh_interp = sfh_interp
        self.zh_const = bool(zh_const)
        if zh_const:
            if sfh_interp == 'step':
                self.calculate_ssp_weights = self.calculate_ssp_weights_const_zh_step
            else:
                self.calculate_ssp_weights = self.calculate_ssp_weights_const_zh
        else:
            if sfh_interp == 'step':
                self.calculate_ssp_weights = self.calculate_ssp_weights_var_zh_step
            else:
                self.calculate_ssp_weights = self.calculate_ssp_weights_var_zh
        if verbose:
            print(f"SFH integration scheme : {sfh_interp}")

        # --- Build theta_init dict -----------------------------------------
        self.initialize_model_structure(theta)

        if verbose:
            print("\nCSPBasis (dict theta) — registered parameters:")
            pprint.pprint({k: v.shape for k, v in self.theta_init.items()})

        # Early (construction-time) warning if the initial parameters already
        # sit outside the interpolation grids (silent edge-clamping).  Cheap,
        # non-jitted; users can re-run check_param_ranges() on sampled theta.
        self.check_param_ranges(self.theta_init)



    def initialize_model_structure(self, theta):
        """
        Validate the incoming theta and store ``self.theta_init``.
        The dict is validated, converted to JAX arrays, and stored directly.

        Required keys
        -------------
        ``"sfh"``          : shape ``(n_time,)``
        ``"lookback_time"`` : shape ``(n_time,)``

        Either ``"Z"`` (scalar, constant metallicity) or ``"zh"`` (shape
        ``(n_time,)``, time-varying metallicity) must be present, depending on
        the ``zh_const`` flag set during ``__init__``.
        """
        # --- sfh_times: static (not part of theta) -------------------------
        self.sfh_times = jnp.atleast_1d(
            jnp.asarray(theta['lookback_time'], dtype=float)
        ) * 1e9   # Gyr → yr
        self.n_time = self.sfh_times.size

        sfh = jnp.atleast_1d(jnp.asarray(theta['sfh'], dtype=float))
        # ``sfh`` may carry either of two conventions:
        #
        # 1. FastStepBasis (prospector-compatible) — one SFR value per
        #    bin, length ``n_time - 1``.  ``calculate_ssp_weights_*_step``
        #    uses each entry directly, with no inter-edge averaging.
        # 2. Node-based legacy — one SFR value per lookback grid point,
        #    length ``n_time``.  ``calculate_ssp_weights_*_step``
        #    averages consecutive entries to recover per-bin SFR.
        #
        # We accept both shapes here and store the convention as a flag
        # the weight calculators consult.  Same parameter numbers
        # therefore mean the same physical SFH between ceridwen and
        # prospector when the FastStepBasis convention is used.
        if sfh.shape == (self.n_time,):
            self.sfh_per_bin = False
        elif sfh.shape == (self.n_time - 1,):
            self.sfh_per_bin = True
        else:
            raise AssertionError(
                f"'sfh' shape {sfh.shape} must be either "
                f"({self.n_time},)  (node-based, legacy)  or "
                f"({self.n_time - 1},)  (per-bin, FastStepBasis)."
            )

        # --- Static SFH sanity (construction-time; non-jitted) -------------
        # A NaN/Inf SFH silently propagates to a NaN spectrum, and an all- or
        # partly-negative SFH is silently clipped to >=0 (≈zero flux), so flag
        # both here rather than letting them pass into the hot path unnoticed.
        sfh_np = np.asarray(sfh)
        if not np.all(np.isfinite(sfh_np)):
            raise ValueError(
                "theta['sfh'] contains non-finite (NaN/Inf) values; this would "
                "silently produce a NaN spectrum."
            )
        if np.any(sfh_np < 0):
            warnings.warn(
                "theta['sfh'] contains negative values. SFR is clipped to >=0 "
                "internally, so negative bins contribute ~zero flux (no error is "
                "raised at evaluation time).",
                stacklevel=3,
            )

        # --- Metallicity mode detection + validation -----------------------
        # The metallicity key MUST match the zh_const mode chosen at __init__.
        # A mismatch otherwise constructs silently and only fails later with a
        # cryptic KeyError deep inside a jitted get_spectrum/predict trace.
        if self.zh_const:
            if 'Z' not in theta:
                raise ValueError(
                    "zh_const=True requires a constant metallicity theta['Z'] "
                    "(shape-(1,) array, log10 Z/Zsun); none was provided. Either "
                    "add theta['Z'], or construct with zh_const=False and provide "
                    "a time-varying theta['zh'] of shape (n_time,)."
                )
            if 'zh' in theta:
                warnings.warn(
                    "zh_const=True but theta also contains 'zh'; 'zh' is ignored "
                    "in constant-metallicity mode (only 'Z' is used).",
                    stacklevel=3,
                )
        else:
            if 'zh' not in theta:
                raise ValueError(
                    "zh_const=False requires a time-varying metallicity history "
                    "theta['zh'] of shape (n_time,) (log10 Z/Zsun); none was "
                    "provided. Either add theta['zh'], or construct with "
                    "zh_const=True and provide a scalar theta['Z']."
                )
            if 'Z' in theta:
                warnings.warn(
                    "zh_const=False but theta also contains 'Z'; 'Z' is ignored "
                    "in time-varying-metallicity mode (only 'zh' is used).",
                    stacklevel=3,
                )

        self.zh_is_scalar = None
        if 'zh' in theta:
            zh = jnp.atleast_1d(jnp.asarray(theta['zh'], dtype=float))
            assert zh.shape == (self.n_time,), "'zh' must match 'lookback_time' length"
            self.zh_is_scalar = False
        elif 'Z' in theta:
            Z = jnp.atleast_1d(jnp.asarray(theta['Z'], dtype=float))
            assert Z.shape == (1,), "'Z' must be a scalar (wrapped in shape-(1,) array)"
            self.zh_is_scalar = True

        # P5 (safe warning): the var-zh *linear* weight kernel divides by the
        # per-node SFR, so an exactly-zero SFR node yields NaN in that mode.
        if (not self.zh_const) and self.sfh_interp == 'linear' and np.any(sfh_np == 0):
            warnings.warn(
                "zh_const=False with sfh_interp='linear' and exactly-zero SFR "
                "node(s): the time-varying-metallicity linear weight kernel "
                "divides by the SFR and will return NaN for those nodes. Use a "
                "tiny positive floor (e.g. 1e-10) instead of 0, or use "
                "sfh_interp='step' (which is non-negative by construction).",
                stacklevel=3,
            )

        # --- Build theta_init: all params except lookback_time -------------
        self.theta_init = {}
        for k, v in theta.items():
            if k == 'lookback_time':
                continue   # static grid — not a free parameter
            arr = jnp.atleast_1d(jnp.asarray(v, dtype=float))
            self.theta_init[k] = arr

        # Ensure sfh has the correct shape stored in theta_init
        self.theta_init['sfh'] = sfh

        # Ordered list of parameter names (for printing / sampling setup)
        self.param_names = list(self.theta_init.keys())

        # Recognized theta keys (for the trace-time typo guard).  Everything
        # the physics consumes is already in param_names (dust/neb defaults were
        # merged into theta before this point); the rest are optional
        # runtime-only scalars read by predict / get_line_spec.
        self._known_theta_keys = set(self.param_names) | {
            'lookback_time', 'Z', 'zh',
            'logmass', 'zred', 'igm_factor', 'eline_scaling',
            'sigma_smooth', 'sigma_v_lines',
        }

    # -----------------------------------------------------------------------
    # Defensive helpers (all NON-jitted or trace-time-only; zero hot-path cost)
    # -----------------------------------------------------------------------

    def _warn_unknown_theta_keys(self, theta):
        """Warn about theta keys the model does not consume (usually typos).

        This inspects only the dict *keys* — static Python strings that are part
        of the PyTree structure — so when called inside a ``jax.jit`` trace it
        executes exactly once at compile time and adds **nothing** to the
        compiled hot path.  Unknown keys are otherwise silently ignored, so a
        typo like ``logmas`` for ``logmass`` would quietly drop the parameter.
        """
        unknown = [k for k in theta if k not in self._known_theta_keys]
        if unknown:
            warnings.warn(
                f"CSPBasis received unrecognized theta key(s) {sorted(unknown)} "
                f"which are SILENTLY IGNORED (likely a typo). Recognized keys: "
                f"{sorted(self._known_theta_keys)}.",
                stacklevel=3,
            )

    def check_param_ranges(self, theta=None, warn=True):
        """Diagnostic (NON-jitted): list parameters that fall outside the
        interpolation grids, where the model silently clamps to the nearest
        grid edge and thus hides extrapolation.

        Intended to be called once on your theta (or theta bounds) before a
        fit; it is never invoked from the hot path.  Returns the list of
        human-readable messages (and emits them as warnings when ``warn``).
        """
        if theta is None:
            theta = self.theta_init
        msgs = []

        zlo, zhi = float(self.zmet.min()), float(self.zmet.max())
        for key in ('Z', 'zh'):
            if key in theta:
                v = np.asarray(theta[key], float)
                if v.size and (np.nanmin(v) < zlo or np.nanmax(v) > zhi):
                    msgs.append(
                        f"theta['{key}'] has values outside the SSP metallicity "
                        f"grid [{zlo:.3f}, {zhi:.3f}]; these are silently clamped "
                        f"to the nearest grid edge. NOTE: this grid is in the "
                        f"same units as SSPData.ssp_lgmet (log10 of absolute "
                        f"metallicity), NOT log10(Z/Zsun) -- so Z=0.0 is out of "
                        f"range; use a value within the printed bounds."
                    )

        # Nebular gas parameters vs the CLOUDY grid, if the neb model exposes
        # its axis arrays (defensive: skipped if the attribute names differ).
        neb = getattr(self, 'neb', None)
        if neb is not None:
            for key, attrs in (
                ('gas_logz', ('logZ_grid', 'logz_grid', '_logZ', 'nebem_logz')),
                ('gas_logu', ('logU_grid', 'logu_grid', '_logU', 'nebem_logu')),
            ):
                if key in theta:
                    grid = next((getattr(neb, a) for a in attrs if hasattr(neb, a)),
                                None)
                    if grid is not None:
                        g = np.asarray(grid, float)
                        glo, ghi = float(g.min()), float(g.max())
                        v = np.asarray(theta[key], float)
                        if v.size and (np.nanmin(v) < glo or np.nanmax(v) > ghi):
                            msgs.append(
                                f"theta['{key}'] outside the nebular grid "
                                f"[{glo:.3f}, {ghi:.3f}]; silently clamped."
                            )

        if warn:
            for m in msgs:
                warnings.warn(m, stacklevel=2)
        return msgs

    # -----------------------------------------------------------------------
    # Dust / nebular initialisation helpers
    # -----------------------------------------------------------------------

    def set_attenuation_function(self, add_diffuse_dust, add_dust):
        """
        Build and assign ``self.attenuate_dust(wave, theta) → (attn, attn_diffuse)``.

        With the dict theta, each dust model simply reads the keys it knows
        about from the shared theta dict.  No NamedTuple construction needed.
        """
        if add_diffuse_dust and add_dust:
            def attenuate(wave, theta):
                attn         = self.dust_attn.compute_attenuation(wave, theta)
                attn_diffuse = self.diff_dust.compute_attenuation(wave, theta)
                return attn, attn_diffuse
            print("Using combined (binwise + diffuse) dust attenuation.")
            self.attenuate_dust = attenuate

        elif add_dust and not add_diffuse_dust:
            def attenuate_without_diffuse(wave, theta):
                attn         = self.dust_attn.compute_attenuation(wave, theta)
                attn_diffuse = jnp.zeros((1, wave.shape[0]))
                return attn, attn_diffuse
            print("Using only binwise dust attenuation.")
            self.attenuate_dust = attenuate_without_diffuse

        elif add_diffuse_dust and not add_dust:
            self.bin_low  = jnp.array([-jnp.inf])
            self.bin_high = jnp.array([jnp.inf])
            def attenuate_diffuse_only(wave, theta):
                attn_diffuse = self.diff_dust.compute_attenuation(wave, theta)
                attn         = jnp.zeros((1, wave.shape[0]))
                return attn, attn_diffuse
            print("Using only diffuse dust attenuation.")
            self.attenuate_dust = attenuate_diffuse_only

    def initialize_neb(self, add_neb, theta, init_neb_params, sps_home):
        if add_neb:
            # Hand the SSP fluxes and ages over to the nebular model so
            # it can compute an internally self-consistent ionising-photon
            # rate that matches the FSPS run-time formula.  ``log_qq`` is
            # no longer carried on ``SSPData``; the nebular model is the
            # single source of truth for it.
            #
            # ``match_fsps`` is the only ceridwen-specific knob that
            # is consumed here rather than forwarded to NebularModel.
            # When True, ceridwen uses the FSPS-bug-replicating variant
            # (both cubes interpolated on the line-cube axes) so output
            # matches FSPS to better than 0.5%.  When False (default),
            # ceridwen uses the physically strict per-cube-axis variant.
            match_fsps = init_neb_params.pop('match_fsps', False)
            init_neb_params.update({
                'sps_home':       sps_home,
                'csp_lambda':     self.wave,
                'ssp_flux':       self.flux,
                'ssp_ages_lgyr':  self.ssp_ages_lgyr,
            })
            NebClass = NebularModelFSPSMatch if match_fsps else NebularModel
            print(f"Initializing Nebular Emission model ({NebClass.__name__})...")
            self.neb = NebClass(**init_neb_params)

            # get_default_params() now returns a plain dict
            neb_defaults = self.neb.get_default_params()
            for k, v in neb_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.neb_param_names = list(neb_defaults.keys())

            # Restrict nebular emission to SSPs whose age lies inside the
            # CLOUDY grid (FSPS does:  DO t=1,nti  where  nti is the index
            # of nebem_age(nebnage)).
            self.young_mask  = jnp.asarray(self.neb.young_mask)
            self.ion_mask    = self.wave < 912.0
            self.kill_ion    = self.young_mask[:, None] & self.ion_mask[None, :]

            young_idx = self.neb.young_idx
            self._neb_young_idx   = young_idx
            self._neb_n_young     = int(young_idx.shape[0])
            self._neb_ages_young  = self.ssp_ages_lgyr[young_idx]      # log10(yr)
            self._neb_logqq_young = self.neb.log_qq[:, young_idx]      # (n_z, n_young)

        return theta

    def initialize_dust_components(
        self, add_dust, add_diffuse_dust, add_dust_emission,
        theta, init_dust_params, diffuse_law, sps_home
    ):
        if add_dust:
            print("Initializing Dust attenuation model...")
            self.dust_attn = Dust(**init_dust_params)

            self.bin_low  = jnp.array([edge[0] for edge in self.dust_attn.bin_edges])
            self.bin_high = jnp.array([edge[1] for edge in self.dust_attn.bin_edges])

            # get_default_fit_params() now returns a plain dict
            dust_defaults = self.dust_attn.get_default_fit_params()
            for k, v in dust_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.dust_param_names = list(dust_defaults.keys())

        if add_diffuse_dust:
            print("Initializing DiffuseDust model...")
            self.diff_dust = DiffuseDust(diffuse_law)

            diff_defaults = self.diff_dust.get_default_params()
            for k, v in diff_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.diff_param_names = list(diff_defaults.keys())

        if add_diffuse_dust or add_dust:
            self._init_age_bin_operator()

        if add_dust_emission:
            print("Initializing DustEmission model...")
            self.dust_emi = DustEmission(spec_lambda=self.wave, dust_file=sps_home)

            emi_defaults = self.dust_emi.get_default_params()
            for k, v in emi_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.emi_param_names = list(emi_defaults.keys())

        print("Dust initialization complete.")
        return theta

    def _init_age_bin_operator(self):
        ages = self.ages
        lo   = self.bin_low
        hi   = self.bin_high

        in_bin  = (ages[:, None] >= lo[None, :]) & (ages[:, None] < hi[None, :])
        M       = in_bin.astype(jnp.float32)
        row_sum = jnp.sum(M, axis=1, keepdims=True)
        # Normalise rows that fall in a bin; already float32 — no later casting needed.
        self._age_bin_mix = jnp.where(row_sum > 0, M / row_sum, M)

    # ------------------------------------------------------------------ #
    # LOSVD smoothing (FSPS / prospector default)
    # ------------------------------------------------------------------ #
    def _setup_losvd_kernel(self):
        """Pre-compute the LOSVD smoothing infrastructure.

        Mirrors the FSPS / Prospector source-side ``losvd_smoothing``:
        a velocity-space Gaussian of standard deviation
        ``sigma_losvd_kms`` applied to the rest-frame
        ``912 < lambda < 25000 AA`` window of the stellar SED, BEFORE
        any observation projection (filter convolution, line aperture
        integration, spectrum LSF convolution).  Same single-call
        design Prospector uses in
        ``prospect.models.sedmodel.SedModel.predict`` (cf. the
        ``self._smooth_spec = self.losvd_smoothing(...)`` cache near
        the top of ``predict()``), so all observation arms share one
        already-smoothed source spectrum and the convolution cost is
        paid once per likelihood call.

        Implementation.  Delegates to
        ``sedpy_jax.smoothing.make_vel_smoother``, which is the same
        factory the observation layer (``ceridwen.observation``)
        already uses for the Spectrum-side LOSVD + LSF chain.  This
        guarantees numerical parity between the CSP-side and
        observation-side smoothings, future-proofs the path (sigma_v
        is a runtime argument of the returned closure, so promoting
        it to a ``theta`` tracer for free sigma_v fitting is a
        one-line change), and inherits any future optimisations in
        sedpy_jax for free.

        Stores:
            ``self._losvd_kernel_fft``  -- sentinel kept as ``None``
                (disabled) or any non-``None`` value (enabled).  The
                wrap in ``configure_spectrum_model`` keys on this
                attribute name to decide whether to install the
                ``__losvd_smoothed`` wrap on ``get_spectrum``.
            ``self._losvd_smoother`` -- the JIT-friendly closure
                returned by ``make_vel_smoother(wave_window,
                wave_window, inres=0.0)``.  Operates on the in-window
                rest-frame subset of ``self.wave`` only.
            ``self._losvd_idx`` -- static int32 JAX index array used
                to gather the in-window pixels at apply time and
                scatter the smoothed result back into the full
                ``self.wave`` grid.
        """
        if self.sigma_losvd_kms <= 0.0:
            self._losvd_kernel_fft = None
            return

        wave_np = np.asarray(self.wave, dtype=np.float64)
        in_band_np = (wave_np > 912.0) & (wave_np < 25000.0)
        if not in_band_np.any():
            self._losvd_kernel_fft = None
            return

        idx_native_np = np.flatnonzero(in_band_np)
        wave_window = wave_np[idx_native_np]
        # ``inres=0`` because the FSPS BPASS native resolution is
        # already baked into ``self.wave``; there is no separate
        # library-resolution kernel to subtract in quadrature on the
        # source side (instrument LSF / library-res deconvolution
        # belongs to the Spectrum projection, not here).
        self._losvd_smoother = make_vel_smoother(
            wave_window, wave_window, inres=0.0,
        )
        self._losvd_idx = jnp.asarray(idx_native_np)
        # Sentinel for configure_spectrum_model: any non-None value
        # enables the wrap on get_spectrum.
        self._losvd_kernel_fft = True

    def _apply_losvd(self, spectrum):
        """Apply the LOSVD smoother to ``spectrum``.

        JIT-safe: the gating is on ``self._losvd_kernel_fft is None``,
        which is a Python-static property of the CSPBasis object set
        at construction time, so the compiled XLA graph is fixed once
        per CSPBasis instance.  Inside the branch, all ops act on
        tracers.

        Pixels outside the rest-frame 912-25000 AA window pass through
        unchanged (Prospector parity: see
        ``sedmodel.losvd_smoothing``'s ``sel`` / ``outspec[sel] = sm``
        pattern).
        """
        if self._losvd_kernel_fft is None:
            return spectrum
        # Gather in-window pixels, smooth via the sedpy_jax closure,
        # scatter back into the full native grid.  ``sigma_losvd_kms``
        # is a Python float here; promoting it to a JAX tracer for
        # free-sigma fitting is a one-line change to thread it in
        # from theta.
        spec_window = spectrum[self._losvd_idx]
        smoothed = self._losvd_smoother(spec_window, self.sigma_losvd_kms)
        return spectrum.at[self._losvd_idx].set(
            smoothed.astype(spectrum.dtype))

    def configure_spectrum_model(
        self, add_dust, add_diffuse_dust, add_dust_emission, add_neb, sps_home
    ):
        part1 = 'dust_'    if (add_dust or add_diffuse_dust) else 'nodust_'
        part2 = 'neb_'     if add_neb                        else 'noneb_'
        if add_dust_emission:
            if not add_dust or not add_diffuse_dust:
                raise ValueError(
                    "Dust emission requires both dust attenuation and diffuse dust."
                )
            part3 = 'dustemi'
        else:
            part3 = 'nodustemi'

        key = part1 + part2 + part3
        mapping = {
            'dust_neb_dustemi':        self.get_spectrum_dattn_dem_neb,
            'dust_neb_nodustemi':      self.get_spectrum_dattn_nodem_neb,
            'dust_noneb_dustemi':      self.get_spectrum_dattn_dem_noneb,
            'dust_noneb_nodustemi':    self.get_spectrum_dattn_nodem_noneb,
            'nodust_neb_nodustemi':    self.get_spectrum_nodattn_nodem_neb,
            'nodust_noneb_nodustemi':  self.get_spectrum_nodattn_nodem_noneb,
        }
        label = {
            'dust_neb_dustemi':        'dust attenuation, nebular emission, dust emission',
            'dust_neb_nodustemi':      'dust attenuation, nebular emission',
            'dust_noneb_dustemi':      'dust attenuation, dust emission',
            'dust_noneb_nodustemi':    'dust attenuation only',
            'nodust_neb_nodustemi':    'nebular emission only',
            'nodust_noneb_nodustemi':  'stellar continuum only',
        }
        print(f"Spectrum model: {label[key]}")
        raw_get_spectrum = mapping[key]
        if self._losvd_kernel_fft is None:
            # Smoothing disabled (sigma=0 or non-log-uniform wave grid):
            # use the raw spectrum without wrapping.
            self.get_spectrum = raw_get_spectrum
        else:
            def get_spectrum_smoothed(theta, *, include_lines=None,
                                       _raw=raw_get_spectrum):
                spec = _raw(theta, include_lines=include_lines)
                return self._apply_losvd(spec)
            get_spectrum_smoothed.__name__ = (
                f"{raw_get_spectrum.__name__}__losvd_smoothed"
            )
            self.get_spectrum = get_spectrum_smoothed

    # -----------------------------------------------------------------------
    # initialize_model_structure: build theta_init as a dict
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def get_spectrum_components(self, theta: dict) -> tuple:
        """Return the canonical ``(continuum, lines)`` line decomposition.

        Both arrays are on the rest-frame model grid ``self.wave`` and are
        *unscaled* -- mass / redshift / IGM factors are applied downstream
        by ``predict`` and ``get_line_spec``, exactly as for ``get_spectrum``.

        - ``continuum`` -- the line-free spectrum (stellar continuum +
          nebular *continuum*), dust-attenuated.  Identical to
          ``get_spectrum(theta, include_lines=False)``.
        - ``lines`` -- the broadened nebular emission-line component alone,
          carried through the same dust attenuation, so the full SED is
          recovered as ``continuum + lines``.

        This is the single source of truth for "spectrum with vs. without
        emission lines": ``predict`` builds the photometry and slit spectra
        from it and ``get_line_spec`` returns its ``lines`` term.
        ``nebemlineinspec`` does not affect this method -- it only sets the
        default of the single-array public ``get_spectrum``.  With
        ``add_neb=False`` there is no nebular module and ``lines`` is
        identically zero.
        """
        # Trace-time-only typo guard (operates on static dict keys; costs
        # nothing in the compiled hot path).  Covers predict(), which routes
        # through here.
        self._warn_unknown_theta_keys(theta)
        continuum = self.get_spectrum(theta=theta, include_lines=False)
        full      = self.get_spectrum(theta=theta, include_lines=True)
        return continuum, full - continuum

    def predict(self, theta: dict, observations: list) -> dict:
        """
        Compute the CSP spectrum and project it onto every observation.

        This method is the primary hot-path entry point for the sampler.
        It is designed to be fully JAX JIT-compatible with zero Python
        ``if`` / ``isinstance`` branches in the traced code path:

        - ``get_spectrum(theta)`` is pure JAX.
        - The Python ``for`` loop over ``observations`` is unrolled at
          trace time because ``observations`` is a static Python list
          (part of the closure, not a traced argument).
        - ``obs.predict(spectrum, self.wave)`` dispatches through Python's
          method resolution order (static at trace time) to the appropriate
          subclass implementation — either a dense matrix–vector multiply
          (``Spectrum``, ``Lines``) or a filter-set convolution
          (``Photometry``).  The XLA kernel contains no conditional branches.

        **Pre-condition:** every ``Observation`` in ``observations`` must have
        had ``obs.setup_for_model(self.wave)`` called before the first JIT
        trace.  ``SedModel.__init__`` does this automatically.

        For a raw model spectrum without projection, use ``get_spectrum(theta)``
        directly; for the separate line-free continuum and emission-line
        component, use ``get_spectrum_components(theta)``.

        Parameters
        ----------
        theta : dict[str, Array]
            Free-parameter dict.  Must contain at minimum ``"sfh"`` and the
            metallicity key (``"Z"`` or ``"zh"``), plus any dust / nebular
            parameters required by the active physics model.
        observations : list of Observation
            Observations to project onto.  Must be the same Python objects
            (same list structure, same types) on every call — changing the
            list forces a retrace.

        Returns
        -------
        predictions : dict[str, Array]
            Keyed by ``obs.name`` for each observation.  Values:

            - ``Photometry`` → shape (n_filters,), synthetic AB maggies
            - ``Spectrum``   → shape (n_pix,), model F_nu interpolated onto
              the observed pixel grid
            - ``Lines``      → shape (n_lines,), Gaussian-aperture fluxes
        """
        spectrum_phot, spectrum_slit = self._assemble_observer_spectra(theta)
        spectrum_phot, spectrum_slit = self._apply_mass_redshift_igm(
            spectrum_phot, spectrum_slit, theta
        )
        return self._project_observations(
            spectrum_phot, spectrum_slit, observations, theta
        )

    def _assemble_observer_spectra(self, theta):
        """Build the photometry- and slit-facing spectra from the single
        ``(continuum, lines)`` decomposition.  Mechanical extraction from
        :meth:`predict`; byte-for-byte identical logic.
        """
        # Decompose the model SED once into its line-free continuum and the
        # broadened emission-line component (see get_spectrum_components),
        # then build the two observer-facing spectra from that single
        # decomposition:
        #
        #   spectrum_phot -- continuum + emission lines at TRUE strength.
        #                    Photometry captures the full field of view, so
        #                    it always sees the intrinsic line flux.
        #   spectrum_slit -- continuum + (eline_scaling / 100) * lines.
        #                    Spectrum and Lines are slit-measured and lose
        #                    flux; eline_scaling is the aperture correction
        #                    (a percentage; 100 = no loss).  Absent from
        #                    theta -> factor 1.0, so spectrum_slit equals
        #                    spectrum_phot.
        #
        # nebemlineinspec is intentionally NOT consulted here: it governs
        # only the default of the single-array public get_spectrum.  The
        # observations must always see the lines, so predict always uses
        # the full (continuum, lines) decomposition.  All key checks are
        # Python-static so the branches fold out at trace time.
        spectrum_cont, line_component = self.get_spectrum_components(theta)

        spectrum_phot = spectrum_cont + line_component       # lines unscaled
        if "eline_scaling" in theta:
            eline_factor = jnp.ravel(theta["eline_scaling"])[0] / jnp.float32(100.0)
            spectrum_slit = (spectrum_cont
                             + eline_factor.astype(spectrum_cont.dtype)
                               * line_component)
        else:
            spectrum_slit = spectrum_phot

        return spectrum_phot, spectrum_slit

    def _apply_mass_redshift_igm(self, spectrum_phot, spectrum_slit, theta):
        """Apply mass, redshift (flux factor) and IGM multiplicative scaling
        to both observer spectra.  Mechanical extraction from :meth:`predict`.
        """
        # Mass + redshift + IGM scaling: identical multiplicative factors
        # for both spectra.  Each factor is computed once and applied to
        # both, so there is no duplicated work.
        if "logmass" in theta:
            mass_scale = jnp.float32(10.0 ** theta["logmass"][0])
            spectrum_phot = spectrum_phot * mass_scale
            spectrum_slit = spectrum_slit * mass_scale

        if "zred" in theta:
            from ..cosmology import flux_factor_maggies
            z_scalar = jnp.ravel(theta["zred"])[0]
            ff = jnp.float32(flux_factor_maggies(z_scalar))
            spectrum_phot = spectrum_phot * ff
            spectrum_slit = spectrum_slit * ff
            if self.igm is not None:
                if "igm_factor" in theta:
                    ig_factor = jnp.ravel(theta["igm_factor"])[0]
                else:
                    ig_factor = jnp.float32(self.igm_factor)
                transmission = self.igm.attenuation(
                    self.wave, z_scalar, factor=ig_factor,
                ).astype(spectrum_phot.dtype)
                spectrum_phot = spectrum_phot * transmission
                spectrum_slit = spectrum_slit * transmission

        return spectrum_phot, spectrum_slit

    def _project_observations(self, spectrum_phot, spectrum_slit, observations, theta):
        """Project the scaled spectra onto each Observation, returning the
        ``{obs.name: prediction}`` dict.  Mechanical extraction from
        :meth:`predict`.
        """
        # Project: Photometry sees the full-field-of-view spectrum;
        # Spectrum and Lines (slit-measured) see the eline_scaling-corrected
        # spectrum.  ``isinstance`` is resolved statically at trace time.
        #
        # Free-redshift dispatch:
        #   When ``obs.free_z`` is True AND ``"zred"`` was sampled (so a
        #   traced JAX scalar exists in theta), we route Photometry obs
        #   through ``predict_at_redshift`` instead of the GEMV fast path.
        #   The fast path's projection matrix ``_T`` was baked at a single
        #   Python-scalar setup zred and cannot be reused per-sample; the
        #   free-z path recomputes the observed-frame wavelength grid
        #   ``wave_obs = (1+z)*wave_rest`` per sample, interpolates the
        #   spectrum onto the filter grid via ``sedpy_jax.interp_source``
        #   (JAX-native, JIT-safe, differentiable in z), and dots with the
        #   precomputed filter transmission matrix.
        #
        #   Both ``obs.free_z`` (Python attribute, static at trace time)
        #   and the ``"zred" in theta`` check are Python-level, so the
        #   branch resolves at trace time and the compiled XLA graph
        #   contains exactly one of the two paths -- no runtime cond.
        from ..observation.observation import (
            Photometry as _Photometry,
            Spectrum   as _Spectrum,
            Lines      as _Lines,
        )
        out = {}
        free_z_in_theta = "zred" in theta
        # Velocity-broadening dispatch.
        #
        # Each Observation subclass carries a static Python flag set at
        # construction:
        #   Spectrum.fit_sigma_smooth -- when True, the runtime stellar
        #       LOSVD (Prospector convention: ``sigma_smooth`` [km/s])
        #       is pulled from ``theta`` and threaded into the closure
        #       that ``Spectrum.setup_for_model`` built.
        #   Lines.fit_sigma_v -- when True, the runtime Gaussian-aperture
        #       width ``sigma_v_lines`` [km/s] is pulled from theta and
        #       used to rebuild the (n_lines, n_wave) weight matrix per
        #       call.
        # When the flag is False, the obs.predict signature is unchanged
        # and the static fast path is preserved bit-for-bit.  The
        # ``isinstance`` + ``getattr`` checks resolve at trace time, so
        # the compiled XLA graph contains only the chosen path.
        for obs in observations:
            spec_for_obs = (spectrum_phot if isinstance(obs, _Photometry)
                            else spectrum_slit)
            if (isinstance(obs, _Photometry)
                    and getattr(obs, "free_z", False)
                    and free_z_in_theta):
                out[obs.name] = obs.predict_at_redshift(
                    spec_for_obs, self.wave, jnp.ravel(theta["zred"])[0]
                )
            elif (isinstance(obs, _Spectrum)
                  and getattr(obs, "fit_sigma_smooth", False)
                  and "sigma_smooth" in theta):
                out[obs.name] = obs.predict(
                    spec_for_obs, self.wave,
                    sigma_smooth=jnp.ravel(theta["sigma_smooth"])[0],
                )
            elif (isinstance(obs, _Lines)
                  and getattr(obs, "fit_sigma_v", False)
                  and "sigma_v_lines" in theta):
                out[obs.name] = obs.predict(
                    spec_for_obs, self.wave,
                    sigma_v=jnp.ravel(theta["sigma_v_lines"])[0],
                )
            else:
                out[obs.name] = obs.predict(spec_for_obs, self.wave)
        return out

    # -----------------------------------------------------------------------
    # Line-only spectrum (prospector-style)
    # -----------------------------------------------------------------------

    def get_line_spec(self, theta):
        """Return the broadened-emission-line component of the model spectrum.

        Computed as ``get_spectrum(include_lines=True) - get_spectrum(include_lines=False)``,
        which gives the contribution of the nebular lines alone -- the
        prospector-style "line-only" spectrum.  Mass + redshift + IGM
        scaling are applied identically to ``CSPBasis.predict`` so the
        output is at the same physical scale as the observation arrays.

        ``add_neb=False`` makes this return zero (no nebular module).
        """
        if not hasattr(self, "neb") or self.neb is None:
            return jnp.zeros_like(self.wave)

        _continuum, line_only = self.get_spectrum_components(theta)

        # Mirror the mass + zred + IGM scaling block of CSPBasis.predict.
        if "logmass" in theta:
            mass_scale = jnp.float32(10.0 ** theta["logmass"][0])
            line_only = line_only * mass_scale
        if "zred" in theta:
            from ..cosmology import flux_factor_maggies
            z_scalar = jnp.ravel(theta["zred"])[0]
            line_only = line_only * jnp.float32(flux_factor_maggies(z_scalar))
            if self.igm is not None:
                if "igm_factor" in theta:
                    ig_factor = jnp.ravel(theta["igm_factor"])[0]
                else:
                    ig_factor = jnp.float32(self.igm_factor)
                transmission = self.igm.attenuation(
                    self.wave, z_scalar, factor=ig_factor,
                )
                line_only = line_only * transmission.astype(line_only.dtype)
        # Apply eline_scaling/100 multiplier so the line-only spectrum is
        # consistent with the Lines.predict line fluxes produced by predict().
        if "eline_scaling" in theta:
            line_only = line_only * (jnp.ravel(theta["eline_scaling"])[0]
                                       / jnp.float32(100.0))
        return line_only

    @property
    def all_params(self):
        """Return the initial theta dict (one entry per free parameter)."""
        return dict(self.theta_init)

    def __repr__(self):
        lines = [
            "<CSPBasis (dict theta)>",
            "-" * 38,
            f"Universe age (tuniv) : {self.tuniv} Gyr",
            f"n_time               : {self.n_time}",
            f"n_SSP_ages           : {len(self.ages)}",
            f"n_metallicities      : {len(self.zmet)}",
            f"wavelength range     : {float(self.wave.min()):.0f} – {float(self.wave.max()):.0f} Å",
            f"SFH integration      : {self.sfh_interp}",
            "Parameters:",
        ]
        for k, v in self.theta_init.items():
            lines.append(f"  {k:<28s}: shape {v.shape}")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # SSP weight calculation (core; identical maths to csp.py)
    # -----------------------------------------------------------------------

    def _ssp_weights(self, theta, *, zh_mode, sfh_mode):
        """Unified SSP-weight kernel consolidating the four
        ``calculate_ssp_weights_{const,var}_zh{,_step}`` methods.

        Parameters
        ----------
        zh_mode : {"const", "var"}
            ``"const"`` — single constant metallicity from ``theta["Z"]``
            (shape ``(1,)``); ``"var"`` — time-varying metallicity from
            ``theta["zh"]`` (shape ``(n_time,)``).
        sfh_mode : {"linear", "step"}
            ``"linear"`` — analytic log-age integration of a piecewise-linear
            SFH via :func:`intsfwght`; ``"step"`` — piecewise-constant SFH via
            SSP-Voronoi-cell overlap (FastStepBasis-style).

        Reproduces each of the four original methods bit-for-bit, including
        their distinct ``sfh`` floors (``const`` → ``1e-30``; ``var`` →
        ``self.tiny_logt``) and the ``const``-mode ``maximum(0, .)`` on the
        summed weights.  In particular the ``var`` + ``linear`` combination
        preserves the original's behaviour for exactly-zero SFR nodes (the
        ``self.tiny_logt`` floor does not clip a zero, so ``slope`` divides by
        zero — a pre-existing latent issue, intentionally NOT altered here so
        this consolidation introduces zero numerical change).
        """
        # Per-mode SFH floor (matches the originals exactly).
        floor = 1e-30 if zh_mode == "const" else self.tiny_logt
        sfh = jnp.clip(theta["sfh"], floor, None)

        t_lo = self.sfh_times[1:]
        t_hi = self.sfh_times[:-1]
        dt   = t_hi - t_lo

        if sfh_mode == "linear":
            slope = jnp.diff(sfh) / ((t_lo - t_hi) * sfh[1:])
            m2    = sfh[1:] * (1.0 + 0.5 * slope * (t_hi + t_lo - 2.0 * t_lo)) * dt

            tprime = jnp.maximum(0.0, t_hi - dt)
            a      = 1.0 - slope * tprime

            logage_lo = self._logage_lo
            logage_hi = self._logage_hi
            dlogage   = self._dlogage
            j         = self._j_range
            n_ssp     = self.ssp_ages_lgyr.size

            log_t_lo = jnp.log10(jnp.clip(t_lo, self._age_clip_lo, self._age_clip_hi))[:, None]
            log_t_hi = jnp.log10(jnp.clip(t_hi, self._age_clip_lo, self._age_clip_hi))[:, None]

            L = jnp.clip(logage_lo[None, :], log_t_lo, log_t_hi)
            R = jnp.clip(logage_hi[None, :], log_t_lo, log_t_hi)

            jmin = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t_lo)) - 1, 0, n_ssp - 1)
            jmax = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t_hi)) + 2, 0, n_ssp - 1)

            mask    = (j[None, :] >= jmin[:, None]) & (j[None, :] < jmax[:, None])
            mask_lo = mask[:, 1:]
            mask_hi = mask[:, :-1]

            A = a[:, None]
            S = slope[:, None]

            I_lo = intsfwght(R, L, A, S, logage_lo[None, :])
            I_hi = intsfwght(R, L, A, S, logage_hi[None, :])

            w_lo = jnp.where(mask_lo, -I_lo / dlogage[None, :], 0.0)
            w_hi = jnp.where(mask_hi,  I_hi / dlogage[None, :], 0.0)

            w1 = jnp.pad(w_lo, ((0, 0), (0, 1))) + jnp.pad(w_hi, ((0, 0), (1, 0)))
            w1 = jnp.maximum(0.0, w1)
        else:  # sfh_mode == "step"
            if self.sfh_per_bin:
                sfh_mid = sfh
            else:
                sfh_mid = 0.5 * (sfh[:-1] + sfh[1:])
            m2 = sfh_mid * dt

            overlap = jnp.maximum(
                0.0,
                jnp.minimum(t_hi[:, None], self._ssp_voronoi_hi[None, :])
                - jnp.maximum(t_lo[:, None], self._ssp_voronoi_lo[None, :])
            )
            w1 = sfh_mid[:, None] * overlap

        m1          = jnp.maximum(w1.sum(axis=1), 1e-30)
        sfh_weights = w1 * (m2 / m1)[:, None]

        if zh_mode == "const":
            total_sfh_weights = jnp.maximum(0.0, sfh_weights.sum(axis=0))

            target_Z = theta["Z"]
            z_idx = jnp.clip(
                jnp.searchsorted(self.zmet, target_Z, side='left'),
                1, self._n_z - 1,
            )
            z1 = self.zmet[z_idx - 1]
            z2 = self.zmet[z_idx]
            w  = jnp.clip((target_Z - z1) / (z2 - z1), 0.0, 1.0)

            total_weights = jnp.zeros((self._n_z, self._n_age))
            total_weights = total_weights.at[z_idx - 1].add((1 - w) * total_sfh_weights)
            total_weights = total_weights.at[z_idx    ].add(      w  * total_sfh_weights)
            return total_weights

        # zh_mode == "var"
        zh   = theta["zh"]
        zbin = 0.5 * (zh[:-1] + zh[1:])
        k    = jnp.clip(jnp.searchsorted(self.zmet, zbin) - 1, 0, self._n_z - 2)

        z0 = self.zmet[k]
        z1 = self.zmet[k + 1]
        dz = jnp.clip((zbin - z0) / jnp.maximum(z1 - z0, tiny_number), 0.0, 1.0)

        n_bin = self.n_time - 1
        rows  = jnp.arange(n_bin)
        M     = jnp.zeros((n_bin, self._n_z))
        M     = M.at[rows, k    ].add(1.0 - dz)
        M     = M.at[rows, k + 1].add(dz)
        return M.T @ sfh_weights

    def calculate_ssp_weights_const_zh(self, theta):
        """Constant-metallicity, piecewise-linear SFH weights.

        Thin wrapper over :meth:`_ssp_weights`; reads ``theta["sfh"]``
        (shape ``(n_time,)``) and ``theta["Z"]`` (shape ``(1,)``, log10 Z/Zsun).
        """
        return self._ssp_weights(theta, zh_mode="const", sfh_mode="linear")

    def calculate_ssp_weights_const_zh_step(self, theta):
        """Constant-metallicity, piecewise-constant (FastStepBasis-style) SFH
        weights.  Thin wrapper over :meth:`_ssp_weights`.  Reads
        ``theta["sfh"]`` and ``theta["Z"]``.
        """
        return self._ssp_weights(theta, zh_mode="const", sfh_mode="step")

    def calculate_ssp_weights_var_zh(self, theta):
        """Time-varying-metallicity, piecewise-linear SFH weights.

        Thin wrapper over :meth:`_ssp_weights`; reads ``theta["sfh"]``
        (shape ``(n_time,)``) and ``theta["zh"]`` (shape ``(n_time,)``,
        log10 Z/Zsun at each lookback time).
        """
        return self._ssp_weights(theta, zh_mode="var", sfh_mode="linear")

    def calculate_ssp_weights_var_zh_step(self, theta):
        """Time-varying-metallicity, piecewise-constant (FastStepBasis-style)
        SFH weights.  Thin wrapper over :meth:`_ssp_weights`.  Reads
        ``theta["sfh"]`` and ``theta["zh"]``.
        """
        return self._ssp_weights(theta, zh_mode="var", sfh_mode="step")

    # -----------------------------------------------------------------------
    # Spectrum methods (all read theta["key"] directly)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Nebular helper: compute the (n_z, n_age, n_wave) nebular array, with or
    # without the broadened emission lines.  Used by every `_neb` variant
    # below.  ``include_lines`` defaults to ``self.nebemlineinspec`` for
    # external callers (matching the prospector switch), but ``csp.predict``
    # forces ``include_lines=True`` so that ``Lines.predict``'s
    # Gaussian-aperture integration still sees the lines in the spectrum.
    # -----------------------------------------------------------------------
    def _build_neb_array(self, theta, *, include_lines):
        """Return ``(n_z, n_age, n_wave)`` nebular array with or without lines."""
        logZ_gas = theta["gas_logz"]
        logU     = theta["gas_logu"]
        cont_young, line_young = self.neb.evaluate_batch(
            logZ_gas, logU, self._neb_ages_young, self._neb_logqq_young,
            return_components=True,
        )
        neb_young = cont_young + line_young if include_lines else cont_young
        n_z, n_age, n_wave = self.flux.shape
        neb_all = jnp.zeros((n_z, n_age, n_wave), dtype=jnp.float32)
        neb_all = neb_all.at[:, self._neb_young_idx, :].set(
            neb_young.astype(jnp.float32))
        return neb_all

    def get_spectrum_dattn_nodem_neb(self, theta, *, include_lines=None):
        """Dust attenuation + nebular emission, no dust emission.

        ``include_lines``:
          None (default) -> use ``self.nebemlineinspec``.
          True / False  -> override (csp.predict always passes True).
        """
        if include_lines is None:
            include_lines = self.nebemlineinspec
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        neb_all = self._build_neb_array(theta, include_lines=include_lines)

        # Lyman / ionising-photon escape fraction (prospector-equivalent
        # ``frac_obrun``).  When present in theta:
        #   * fraction (1 - f_esc) of ionising photons is absorbed in the
        #     HII region and reprocessed → nebular continuum + lines,
        #     so the nebular grid amplitude scales by (1 - f_esc).
        #   * fraction f_esc of the stellar ionising flux escapes and is
        #     restored to the spectrum (the kill_ion mask zeroed it
        #     out by default, assuming f_esc = 0).
        # Absent → defaults to f_esc = 0, which collapses to the
        # legacy behaviour bit-for-bit (no kill-ion restoration, no
        # nebular scaling).  The check is a Python-static dict-key
        # lookup so this fold-out is free at trace time.
        if "frac_obrun" in theta:
            f_esc = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            stellar_fluxes  = jnp.where(
                self.kill_ion[None, :, :],
                f_esc * self.flux,                        # restore frac_obrun
                self.flux,
            )
            neb_all = neb_all * (jnp.float32(1.0) - f_esc)
        else:
            stellar_fluxes = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)
        combined_fluxes = stellar_fluxes + neb_all                      # (n_z, n_age, n_wave) float32

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        # Cast dust curves to float32 for the einsum — keeps the entire
        # forward model in single precision until the likelihood.
        M        = self._age_bin_mix
        tau_age  = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age = jnp.exp(-tau_age)

        # FSPS-style OB-runaway dust escape (``add_dust.f90`` L93-94).
        # A fraction ``frac_obrun`` of the young-star flux bypasses the
        # birth-cloud (``attn_age``) attenuation, while still passing
        # through the diffuse component below.  This is the second
        # physical effect of FSPS's ``frac_obrun`` knob -- the first
        # (LyC escape into the spectrum + ``Q`` scaling) is applied
        # above.  Old SSP ages already have ``attn_age = 1`` (no
        # birth-cloud bin assignment), so the mix is a no-op for them.
        # When ``frac_obrun`` is absent or 0, this expression is
        # identically ``attn_age`` -- bit-for-bit backward compatible.
        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        W_f32 = W.astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", W_f32, combined_fluxes, attn_age)
        spectrum = spectrum * jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_neb(self, theta, *, include_lines=None):
        """Dust attenuation + nebular emission + dust emission."""
        if include_lines is None:
            include_lines = self.nebemlineinspec
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        neb_all = self._build_neb_array(theta, include_lines=include_lines)

        # Lyman / ionising-photon escape fraction.  See the long
        # comment on this block in get_spectrum_dattn_nodem_neb above.
        if "frac_obrun" in theta:
            f_esc = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            stellar_fluxes  = jnp.where(
                self.kill_ion[None, :, :],
                f_esc * self.flux,
                self.flux,
            )
            neb_all = neb_all * (jnp.float32(1.0) - f_esc)
        else:
            stellar_fluxes = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)
        combined_fluxes = stellar_fluxes + neb_all                      # float32

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

        # FSPS-style OB-runaway dust escape (``add_dust.f90`` L93-94).
        # See the long comment in ``get_spectrum_dattn_nodem_neb``.
        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        W_f32 = W.astype(jnp.float32)
        spectrum_dust_free = jnp.einsum("za,zaw->w",       W_f32, combined_fluxes)
        attenuated         = jnp.einsum("za,zaw,aw->w",    W_f32, combined_fluxes, attn_age)
        attenuated         = attenuated * diffuse_curve

        dust_emi_spectrum, _mdust, _tduste = self.dust_emi.compute_dust_emission(
            spec_attn     = attenuated,
            spec_dustfree = spectrum_dust_free,
            spec_lambda   = self.wave,
            diffuse_curve = diffuse_curve,
            duste_qpah    = theta["duste_qpah"],
            duste_umin    = theta["duste_umin"],
            duste_gamma   = theta["duste_gamma"],
        )

        return dust_emi_spectrum

    def get_spectrum_dattn_nodem_noneb(self, theta, *, include_lines=None):
        """Dust attenuation, no nebular, no dust emission.

        ``include_lines`` is accepted but ignored: there is no nebular
        emission to include / exclude when ``add_neb=False``.
        """
        _ = include_lines
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M       = self._age_bin_mix
        tau_age = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age= jnp.exp(-tau_age)

        # FSPS-style OB-runaway dust escape (``add_dust.f90`` L93-94).
        # See the long comment in ``get_spectrum_dattn_nodem_neb``.
        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        weights  = self.calculate_ssp_weights(theta).astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", weights, self.flux, attn_age)
        spectrum *= jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_noneb(self, theta, *, include_lines=None):
        """Dust attenuation + dust emission, no nebular.  ``include_lines`` accepted but ignored."""
        _ = include_lines
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

        # FSPS-style OB-runaway dust escape (``add_dust.f90`` L93-94).
        # See the long comment in ``get_spectrum_dattn_nodem_neb``.
        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        weights           = self.calculate_ssp_weights(theta).astype(jnp.float32)
        spectrum_dust_free= jnp.einsum("za,zaw->w", weights, self.flux)
        attenuated        = jnp.einsum("za,zaw,aw->w", weights, self.flux, attn_age)
        attenuated       *= diffuse_curve

        dust_emi_spectrum, _mdust, _tduste = self.dust_emi.compute_dust_emission(
            spec_attn      = attenuated,
            spec_dustfree  = spectrum_dust_free,
            spec_lambda    = self.wave,
            diffuse_curve  = diffuse_curve,
            duste_qpah     = theta["duste_qpah"],
            duste_umin     = theta["duste_umin"],
            duste_gamma    = theta["duste_gamma"],
        )
        return dust_emi_spectrum

    def get_spectrum_nodattn_nodem_noneb(self, theta, *, include_lines=None):
        """Stellar continuum only — no dust, no nebular.  ``include_lines`` ignored."""
        _ = include_lines
        weights  = self.calculate_ssp_weights(theta=theta).astype(jnp.float32)
        return jnp.einsum("za,zaw->w", weights, self.flux)

    def get_spectrum_nodattn_nodem_neb(self, theta, *, include_lines=None):
        """Nebular emission only, no dust."""
        if include_lines is None:
            include_lines = self.nebemlineinspec
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        neb_all = self._build_neb_array(theta, include_lines=include_lines)

        # Suppress stellar ionizing part — with the optional
        # frac_obrun escape fraction (see get_spectrum_dattn_nodem_neb).
        if "frac_obrun" in theta:
            f_esc = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            stellar_fluxes = jnp.where(
                self.kill_ion[None, :, :],
                f_esc * self.flux,
                self.flux,
            )
            neb_all = neb_all * (jnp.float32(1.0) - f_esc)
        else:
            stellar_fluxes = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)

        combined_fluxes = stellar_fluxes + neb_all  # float32
        W_f32 = W.astype(jnp.float32)
        return jnp.einsum("za,zaw->w", W_f32, combined_fluxes)
