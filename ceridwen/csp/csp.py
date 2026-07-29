"""
Dict-based CSPBasis.

The parameter vector theta is a plain Python/JAX dict at all times:

    theta = {
        "sfh":               jnp.zeros(100),   # shape (n_time,), linear SFR
        "Z":                 jnp.array([-1.85]),# log10 ABSOLUTE metallicity
                                                # (ssp_lgmet grid units), scalar
                                                # — NOT log10 Z/Zsun. Use "zh"
                                                # (same units, shape (n_time,))
                                                # for a metallicity history.
        "gas_logz":          jnp.array([0.0]),
        "gas_logu":          jnp.array([-2.0]),
        "tau_pow":           jnp.array([1.0]),
        "diffuse_tau_kc":    jnp.array([0.3]),
        "diffuse_dust_index": jnp.array([0.0]),
        # … any other dust / emission parameters
    }

Python dicts are natively registered JAX PyTrees.  String keys are static
(part of the PyTree structure, not traced values), so passing a dict to a
``@jax.jit`` function has no overhead, and retracing occurs only if the set of
keys, or the shapes/dtypes of the values, change.

``DiagonalNoiseModel.compute(sigma, mu, mask, theta)`` looks up nuisance
parameters by name (``theta["log_jitter"]``), which works transparently with
the dict theta.
"""

import math
import os
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
# Observation-type dispatch is handled by the polymorphic
# obs.predict(spectrum, wave) method on each Observation subclass, so
# CSPBasis.predict has zero isinstance/if branches and needs no imports here.

tiny_number = 1e-70
# Plain Python constant — avoid a module-level JIT call that can fail on
# backends (e.g. Apple Metal) which do not support the configured float
# precision at package import.
LOG10E = math.log10(math.e)  # ≈ 0.4342944819032518


def fnu2flam(lam, fnu):
    """Convert f_nu [erg/s/cm^2/Hz] to f_lambda [erg/s/cm^2/Å]."""
    c = 2.998e18  # Å/s
    return c * (fnu / (lam ** 2))



def intsfwght(t_hi, t_lo, a, slope, logage):
    """Integrated SFH weight between log-time limits.

    Pure JAX helper. Deliberately not ``@jit``-decorated: it is only ever
    called from within the outer ``@jax.jit`` lnprobfn, which inlines it, so a
    separate JIT boundary here would be redundant.
    """

    def F(t):
        x = 10.0**t
        delta = logage - t
        return (
            a * x * (delta + LOG10E)
            + 0.5 * slope * x * x * (delta + 0.5 * LOG10E)
        )

    return F(t_hi) - F(t_lo)


# Legacy default used when an SSP grid records no isochrone library (i.e. it
# predates provenance tracking) and the caller did not specify one.  Matches
# the historical hard-coded CSPBasis default so old grids reproduce exactly.
_LEGACY_ISOC_TYPE = "mist"


def _resolve_isoc_type(recorded, user, *, default=_LEGACY_ISOC_TYPE):
    """
    Pick the nebular isochrone type, preferring the SSP grid's provenance.

    A nebular CLOUDY grid whose isochrone set does not match the SSP grid is
    wrong physics with no visible symptom, so a conflict is a hard error.

    Parameters
    ----------
    recorded : str or None
        ``isoc_type`` recorded in the SSPData (``None`` for legacy grids).
    user : str or None
        ``isoc_type`` the caller passed via ``init_neb_params`` (or ``None``).

    Returns
    -------
    str
        The isochrone type to use for the nebular grid.

    Raises
    ------
    ValueError
        If the caller specified an ``isoc_type`` that conflicts with the
        value recorded in the SSP grid.
    """
    if user is None:
        if recorded is None:
            warnings.warn(
                "SSPData carries no recorded isochrone library (it was loaded "
                "from a legacy grid built before provenance tracking). Falling "
                f"back to isoc_type={default!r} for the nebular grid, which is "
                "WRONG if this SSP grid used a different isochrone set. Rebuild "
                "the grid with SSPData.from_fsps to record provenance, or pass "
                "init_neb_params={'isoc_type': ...} explicitly.",
                UserWarning, stacklevel=3,
            )
            return default
        return recorded
    if recorded is not None and str(user) != str(recorded):
        raise ValueError(
            f"isoc_type conflict: init_neb_params requested isoc_type={user!r}, "
            f"but the SSP grid was built with isoc_type={recorded!r} (from its "
            "recorded provenance). A nebular CLOUDY grid that does not match the "
            "SSP isochrone set is wrong physics with no visible symptom. Drop "
            "the explicit isoc_type to use the grid's recorded value, or rebuild "
            "the SSP grid with the matching isochrones."
        )
    return user


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
    theta : dict, optional
        Initial parameter values.  Must contain ``"sfh"`` and
        ``"lookback_time"`` (>= 2 nodes; n_time nodes define n_time-1 SFH
        bins).  All other keys are optional.  Instead of ``theta`` you can
        pass the ``lookback_time=`` shortcut (below), which fills the initial
        values with neutral defaults — use ``theta`` only when you want
        control over the initial values themselves (e.g. a specific initial
        SFH for ``display_sfh`` or a chosen starting point).

        ``"lookback_time"`` is required even when the redshift is sampled:
        with ``track_zred_age=True`` the forward pass rescales this grid to
        track ``age_gyr(zred)``, preserving its length and relative node
        spacing — the construction-time grid is the *template* that fixes
        ``n_time`` and the bin structure.

        The construction-time grid is a *default*, not a straitjacket: an
        explicit ``theta["lookback_time"]`` passed to ``predict`` /
        ``get_spectrum`` takes precedence and is used verbatim in the weight
        kernel (see ``_ssp_weights``), e.g. for transform-derived grids
        computed from a sampled ``zred``.  Per-call grids are traced values
        and therefore CANNOT be validated (monotonicity, range) inside the
        compiled path — that fail-fast validation happens only on the
        concrete construction-time grid, which is why one is required here.

        Lookback-time convention
            ``theta["lookback_time"]`` is **monotonically increasing**
            in Gyr, with index 0 the present-day node (≈ 0 Gyr) and
            the last index the oldest sampled node (≤ ``tuniv``).
            ``theta["sfh"]`` is indexed to match: ``sfh[0]`` is the
            SFR at today (per-node input) or the SFR of the
            youngest bin (per-bin / FastStepBasis input).  Likewise
            ``theta["zh"]`` (per-node) has ``zh[0]`` = today's
            metallicity.

            Example::

                T_univ = 13.8
                lookback = jnp.linspace(0.0, T_univ, 10)        # today → oldest
                sfh      = jnp.exp(-lookback / 1.0)             # late-assembly burst
                theta    = {"lookback_time": lookback,
                            "sfh":           sfh,
                            "Z":             jnp.array([-0.5])}

            A decreasing grid (``lookback = T_univ - t_grid``) raises a
            ``ValueError`` at construction — see the assertion in
            ``initialize_model_structure``.
    tuniv : float
        Age of the Universe in Gyr.  Default 13.8.
    zh_const : bool
        If True, use constant metallicity (requires key ``"Z"``).
        If False, use time-varying metallicity (requires key ``"zh"``).
    add_neb, add_dust, add_diffuse_dust, add_dust_emission : bool
        Physics switches.
    sps_home : str, optional
        Path to the FSPS data directory (needed for nebular and dust-emission
        grid loading). Defaults to the ``$SPS_HOME`` environment variable that
        FSPS users set on install; pass explicitly to override. Required only
        when ``add_neb`` or ``add_dust_emission`` is True.
    init_neb_params, init_dust_params : dict
        Keyword arguments forwarded to ``NebularModel`` / ``Dust``.
        ``isoc_type`` no longer needs to be set here: it is taken
        automatically from the SSP grid's recorded provenance
        (``SSPData.isoc_type``), so the nebular CLOUDY grid always matches
        the SSP isochrone set. Passing an ``isoc_type`` that conflicts with
        the grid raises ``ValueError``; passing one for a legacy grid with no
        recorded library is honoured. A legacy grid with no recorded library
        and no explicit ``isoc_type`` warns and falls back to ``'mist'``.
    diffuse_law : str
        Attenuation law name for the diffuse dust component.
    verbose : bool
        Print parameter summary after initialization.
    lookback_time : array-like, optional
        Shortcut alternative to ``theta``: the static SFH node grid (Gyr,
        monotonically increasing, index 0 = today, >= 2 nodes).  Initial
        values are filled with neutral defaults (``sfh`` = 1 in every
        node/bin; metallicity = the median of the SSP grid, guaranteed
        in-grid).  Mutually exclusive with ``theta``.

        Example::

            csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 12.0, 6),
                           zh_const=True, add_neb=False)
    sfh_per_bin : bool
        Only used with the ``lookback_time=`` shortcut: if True, the SFH is
        one SFR per bin (shape ``(n_time-1,)``, FastStepBasis / prospector
        convention) instead of one per node (shape ``(n_time,)``).  Default
        False.  (With ``theta=`` the convention is inferred from the shape
        of ``theta['sfh']``.)
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
        sps_home=None,
        init_dust_params=None,
        diffuse_law='kriek_conroy',
        verbose=True,
        sfh_interp='step',
        sigma_losvd_kms=300.0,
        track_zred_age=False,
        lookback_time=None,
        sfh_per_bin=False,
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

            ``'linear'`` — piecewise-linear.
                Analytically integrates a linearly-interpolated SFH against the
                SSP age bins in log-age space (``intsfwght``).  Higher-order
                accurate, but can produce small negative weights for steep SFH
                gradients, which are then clipped.

        To switch at runtime::

            csp.calculate_ssp_weights = csp.calculate_ssp_weights_const_zh_step
            # or
            csp.calculate_ssp_weights = csp.calculate_ssp_weights_const_zh
        """
        # --- Shortcut construction: lookback_time= instead of theta= --------
        # The init theta exists to fix STATIC structure (n_time + node spacing,
        # the per-node/per-bin sfh convention, the Z-vs-zh metallicity mode) —
        # things the JIT-compiled kernels bake in at trace time.  The initial
        # VALUES only seed theta_init and the early range check, so the
        # shortcut fills them with neutral defaults: sfh = 1 everywhere and
        # the median metallicity of the SSP grid (always in-grid).  Every
        # predict/get_spectrum call still takes its own theta as usual.
        if lookback_time is not None:
            if theta is not None:
                raise ValueError(
                    "Pass either theta= (full control over the initial "
                    "parameter values) or the lookback_time= shortcut, not "
                    "both."
                )
            _lb = jnp.atleast_1d(jnp.asarray(lookback_time, dtype=float))
            _n = int(_lb.size)
            theta = {
                'lookback_time': _lb,
                'sfh': jnp.ones(max(_n - 1, 1) if sfh_per_bin else _n),
            }
            _z_mid = float(jnp.median(jnp.asarray(SSPData.ssp_lgmet)))
            if zh_const:
                theta['Z'] = jnp.array([_z_mid])
            else:
                theta['zh'] = jnp.full((_n,), _z_mid)
        if theta is None:
            raise ValueError(
                "CSPBasis needs the static SFH grid structure. Pass either\n"
                "  lookback_time=jnp.linspace(0.0, T_oldest, n_nodes)   "
                "(shortcut; neutral initial values), or\n"
                "  theta={'lookback_time': ..., 'sfh': ..., 'Z' or 'zh': ...} "
                "(full control).\n"
                "lookback_time is in Gyr, monotonically increasing, index 0 = "
                "today, >= 2 nodes."
            )
        if init_neb_params is None:
            # isoc_type is intentionally NOT set here: it is taken from the
            # SSP grid's recorded provenance (see initialize_neb). Pass
            # init_neb_params={'isoc_type': ...} only to override.
            init_neb_params = {"cloudy_dust": True}
        if init_dust_params is None:
            init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 'laws': ['powerlaw']}

        # --- SSP grids (static, never part of theta) -----------------------
        self.flux      = jnp.array(SSPData.ssp_flux, dtype=jnp.float32)  # (n_z, n_age, n_wave)
        self.wave      = jnp.array(SSPData.ssp_wave)       # (n_wave,)
        self.ages      = jnp.array(SSPData.ssp_lg_age_gyr) # (n_age,)  log10(Gyr)
        self.zmet      = jnp.array(SSPData.ssp_lgmet)      # (n_z,) log10 absolute Z
        # The nebular model computes the ionising-photon rate from ``self.flux``
        # internally (see ``initialize_neb`` / ``NebularModel.compute_log_qq``).
        self.zlegend   = 10 ** self.zmet                   # linear metallicity
        self.ssp_ages_lgyr = self.ages + 9                 # log10(yr)

        # Static provenance carried by the SSP grid (Python-level only, never a
        # JAX leaf). ``isoc_type`` is auto-propagated to the nebular model so
        # users never have to set it by hand; ``None`` for legacy grids.
        self._ssp_isoc_type    = getattr(SSPData, "isoc_type", None)
        self._ssp_spec_library = getattr(SSPData, "spec_library", None)

        # Precomputed constants for calculate_ssp_weights (all static)
        self._logage_lo  = self.ssp_ages_lgyr[1:]
        self._logage_hi  = self.ssp_ages_lgyr[:-1]
        self._dlogage    = jnp.diff(self.ssp_ages_lgyr)
        self._j_range    = jnp.arange(self.ssp_ages_lgyr.size)
        self._age_clip_lo = 10.0 ** (-70)                  # floor for log-time clipping
        self._age_clip_hi = 10.0 ** self.ssp_ages_lgyr[-1] # ceiling
        self._n_z   = len(self.zmet)
        self._n_age = len(self.ages)

        # SSP bin edges in linear years. _ssp_lo_yr is the younger (smaller)
        # edge; _ssp_hi_yr is the older (larger) edge of each SSP age bin.
        self._ssp_lo_yr = 10.0 ** self._logage_hi   # (n_age-1,)
        self._ssp_hi_yr = 10.0 ** self._logage_lo   # (n_age-1,)

        # Voronoi cell boundaries for the step-function weight scheme.
        # Each SSP age POINT j owns the linear-time interval
        #   [_ssp_voronoi_lo[j], _ssp_voronoi_hi[j]]
        # where the boundaries are the midpoints to the neighbouring age points.
        # For a piecewise-constant SFH the weight at SSP j equals the
        # SFR * (width of its Voronoi cell in yr); this matches FSPS
        # FastStepBasis's internal ±ε offset scheme.
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
        # Resolve the FSPS data directory: explicit arg wins, else $SPS_HOME
        # (the variable FSPS users already set on install).
        if sps_home is None:
            sps_home = os.environ.get("SPS_HOME")
        if (add_neb or add_dust_emission) and not sps_home:
            raise ValueError(
                "sps_home is required for nebular / dust emission but was not "
                "given and $SPS_HOME is unset. Set `export SPS_HOME=/path/to/fsps` "
                "(your FSPS data directory) or pass sps_home=... explicitly."
            )
        self.sps_home   = sps_home
        # Free-redshift age-grid tracking.  When True AND theta carries a
        # sampled ``zred`` (and NO explicit ``lookback_time``), the SFH
        # lookback grid is rescaled inside the forward pass so its oldest node
        # tracks the age of the universe at the sampled redshift, via the
        # differentiable :func:`ceridwen.cosmology.age_gyr` (see
        # :meth:`_lookback_from_zred` and :meth:`_ssp_weights`).  Default
        # False keeps the fixed-z path bit-for-bit unchanged.
        self.track_zred_age = bool(track_zred_age)
        # Prospector-style ``nebemlineinspec`` switch.  It governs ONE
        # thing only: the default of the single-array public
        # ``get_spectrum(theta)``.  When False (default), that call
        # returns the line-free continuum (stellar + nebular continuum);
        # when True it returns continuum + emission lines.  It does NOT
        # affect ``predict`` or ``get_spectrum_components``, which always
        # compute the full ``(continuum, lines)`` decomposition so the
        # observations always see the lines (Photometry at true strength,
        # Spectrum / Lines scaled by ``eline_scaling``).  To force lines
        # on/off explicitly, pass ``include_lines=`` to ``get_spectrum`` or
        # use ``get_spectrum_components``.
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

        # Pre-compute the FSPS-default LOSVD smoothing kernel (sigma_smooth
        # default 300 km/s, velocity-space Gaussian on rest-frame
        # 912 < lambda < 25000 AA; matches prospect/models/sedmodel.py
        # losvd_smoothing).  Must run BEFORE configure_spectrum_model so the
        # wrap can see whether to install the smoother.
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
        # --- Required keys: nice errors instead of raw KeyErrors -----------
        if 'lookback_time' not in theta:
            raise ValueError(
                "theta must contain 'lookback_time' — the static SFH node grid "
                "(Gyr, monotonically increasing, index 0 = today). It is "
                "required even when the redshift is a free parameter: with "
                "track_zred_age=True the grid is rescaled inside the forward "
                "pass to track age(zred), but its LENGTH and RELATIVE spacing "
                "come from this construction-time grid, so it defines n_time "
                "and the bin structure rather than a fixed absolute age range."
            )
        if 'sfh' not in theta:
            raise ValueError(
                "theta must contain 'sfh' — star-formation-rate values, either "
                "one per lookback node (shape (n_time,)) or one per bin "
                "(shape (n_time-1,), FastStepBasis convention), where n_time = "
                "len(theta['lookback_time'])."
            )

        # --- sfh_times: static (not part of theta) -------------------------
        self.sfh_times = jnp.atleast_1d(
            jnp.asarray(theta['lookback_time'], dtype=float)
        ) * 1e9   # Gyr → yr
        self.n_time = self.sfh_times.size

        # --- Minimum grid size ---------------------------------------------
        # n_time nodes define n_time-1 SFH bins; with fewer than 2 nodes there
        # is no bin to integrate over. (This must precede the monotonicity
        # check: np.diff of a single node is empty and np.all([]) is True, so
        # a 1-node grid would otherwise slip through and fail later inside a
        # jitted weight kernel with a cryptic shape error.)
        if self.n_time < 2:
            raise ValueError(
                f"theta['lookback_time'] has {self.n_time} node(s); at least "
                "2 are required (n_time nodes define n_time-1 SFH bins). "
                "Typical fits use 5-10 nodes, e.g. "
                "jnp.linspace(0.0, T_UNIV, 6)."
            )

        # Convention check: lookback_time must be monotonically *increasing*,
        # starting at 0 (today).  A decreasing grid trips here loudly rather
        # than silently producing wrong-physics weights.
        _lb = np.asarray(self.sfh_times, dtype=np.float64)
        _diffs = np.diff(_lb)
        if not (np.all(_diffs > 0.0) and _lb[0] >= 0.0 and _lb[0] < 1e8):
            raise ValueError(
                "theta['lookback_time'] must be monotonically *increasing* "
                "(NEW convention, post-2026-06-03 refactor):\n"
                f"  - index 0 = today (≈ 0 Gyr): got {_lb[0]/1e9:.3f} Gyr\n"
                f"  - index -1 = oldest (≈ T_univ): got {_lb[-1]/1e9:.3f} Gyr\n"
                f"  - first three values [Gyr]: {(_lb[:3]/1e9).tolist()}\n"
                "If you see this from a pre-refactor script, replace e.g.\n"
                "    lookback = T_UNIV - jnp.linspace(eps, T_UNIV, N)\n"
                "with\n"
                "    lookback = jnp.linspace(0.0, T_UNIV, N)\n"
                "and reverse theta['sfh'] (and theta['zh'] if present) to match."
            )

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
        # Both shapes are accepted; the convention is stored as a flag the
        # weight calculators consult.  With the FastStepBasis convention,
        # the same parameter numbers mean the same physical SFH in ceridwen
        # and prospector.
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
                    "(shape-(1,) array, log10 absolute metallicity in ssp_lgmet "
                    "grid units); none was provided. Either add theta['Z'], or "
                    "construct with zh_const=False and provide a time-varying "
                    "theta['zh'] of shape (n_time,)."
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
                    "theta['zh'] of shape (n_time,) (log10 absolute metallicity "
                    "in ssp_lgmet grid units, same as theta['Z']); none was "
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
            'sigma_smooth', 'frac_obrun',
        }

    # -----------------------------------------------------------------------
    # Defensive helpers (all NON-jitted or trace-time-only; zero hot-path cost)
    # -----------------------------------------------------------------------

    def register_known_theta_keys(self, keys):
        """Register additional recognized theta keys so they are not mis-flagged
        as typos by :meth:`_warn_unknown_theta_keys`.

        ``SedModel`` calls this with its model-level free parameters (e.g.
        ``logsfr_ratios``, which the ``sfh`` transform consumes): those keys are
        forwarded through to ``predict`` in the full theta dict but are not CSP
        parameters, so without this they would trigger a spurious typo warning.
        """
        self._known_theta_keys |= set(keys)

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
            # Hand the SSP fluxes and ages to the nebular model so it can
            # compute an internally self-consistent ionising-photon rate
            # matching the FSPS run-time formula; the nebular model is the
            # single source of truth for ``log_qq``.
            #
            # ``match_fsps`` is consumed here rather than forwarded to
            # NebularModel.  When True, ceridwen uses the FSPS-bug-replicating
            # variant (both cubes interpolated on the line-cube axes), matching
            # FSPS to better than 0.5%.  When False (default), it uses the
            # physically strict per-cube-axis variant.
            match_fsps = init_neb_params.pop('match_fsps', False)

            # Auto-propagate the isochrone type from the SSP grid's recorded
            # provenance so the CLOUDY nebular grid always matches the SSP
            # isochrone set. A caller-supplied isoc_type that conflicts with
            # the grid is a hard error; a legacy grid with no provenance warns.
            init_neb_params['isoc_type'] = _resolve_isoc_type(
                self._ssp_isoc_type, init_neb_params.get('isoc_type'),
            )

            init_neb_params.update({
                'sps_home':       sps_home,
                'csp_lambda':     self.wave,
                'ssp_flux':       self.flux,
                'ssp_ages_lgyr':  self.ssp_ages_lgyr,
            })
            NebClass = NebularModelFSPSMatch if match_fsps else NebularModel
            print(f"Initializing Nebular Emission model ({NebClass.__name__}, "
                  f"isoc_type={init_neb_params['isoc_type']!r})...")
            self.neb = NebClass(**init_neb_params)

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
        # Normalise rows that fall in a bin.
        self._age_bin_mix = jnp.where(row_sum > 0, M / row_sum, M)

    # ------------------------------------------------------------------ #
    # LOSVD smoothing (FSPS / prospector default)
    # ------------------------------------------------------------------ #
    def _setup_losvd_kernel(self):
        """Pre-compute the LOSVD smoothing infrastructure.

        Mirrors the FSPS / Prospector source-side ``losvd_smoothing``: a
        velocity-space Gaussian of standard deviation ``sigma_losvd_kms``
        applied to the rest-frame ``912 < lambda < 25000 AA`` window of the
        stellar SED, BEFORE any observation projection (filter convolution,
        line aperture integration, spectrum LSF convolution).  As in
        ``prospect.models.sedmodel.SedModel.predict``, all observation arms
        share one already-smoothed source spectrum, so the convolution cost is
        paid once per likelihood call.

        Delegates to ``sedpy_jax.smoothing.make_vel_smoother`` -- the same
        factory the observation layer uses for the Spectrum-side LOSVD + LSF
        chain -- guaranteeing numerical parity between the CSP-side and
        observation-side smoothings.  ``sigma_v`` is a runtime argument of the
        returned closure, so promoting it to a ``theta`` tracer for free
        sigma_v fitting is a one-line change.

        Stores:
            ``self._losvd_kernel_fft``  -- sentinel: ``None`` (disabled) or any
                non-``None`` value (enabled).  ``configure_spectrum_model`` keys
                on it to decide whether to install the ``__losvd_smoothed``
                wrap on ``get_spectrum``.
            ``self._losvd_smoother`` -- the JIT-friendly closure from
                ``make_vel_smoother``, operating on the in-window rest-frame
                subset of ``self.wave`` only.
            ``self._losvd_idx`` -- static int32 index array used to gather the
                in-window pixels and scatter the smoothed result back into the
                full ``self.wave`` grid.
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
        # ``inres=0`` because the library native resolution is already baked
        # into ``self.wave``; there is no separate library-resolution kernel to
        # subtract in quadrature on the source side (instrument LSF / library-res
        # deconvolution belongs to the Spectrum projection, not here).
        self._losvd_smoother = make_vel_smoother(
            wave_window, wave_window, inres=0.0,
        )
        self._losvd_idx = jnp.asarray(idx_native_np)
        # Sentinel for configure_spectrum_model: any non-None value enables it.
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
        # Gather in-window pixels, smooth via the sedpy_jax closure, scatter
        # back into the full native grid.  ``sigma_losvd_kms`` is a Python float
        # here; threading it from theta would enable free-sigma fitting.
        spec_window = spectrum[self._losvd_idx]
        smoothed = self._losvd_smoother(spec_window, self.sigma_losvd_kms)
        # Edge repair: sedpy_jax's ``jax_interp`` zero-fills OUTSIDE its
        # internal log-uniform grid (left=0, right=0), and that grid's
        # endpoints are built as exp(log(lambda)), which can land 1 ulp
        # inside the true window endpoints.  The window's first/last pixel
        # then tests as out-of-range and comes back EXACTLY 0 -- seen as a
        # spurious notch at the red edge of the smoothing window (rest
        # 24950 A on the FSPS grid; blue-edge sibling of the Lyman-spike
        # bug covered by tests/test_losvd_no_lyman_spike.py).  Keep the raw
        # endpoint pixels instead: a 1-pixel unsmoothed edge is invisible,
        # a zeroed pixel is a hole in every SED.
        smoothed = smoothed.at[0].set(spec_window[0]) \
                           .at[-1].set(spec_window[-1])
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
            # Smoothing disabled (sigma=0 or non-log-uniform wave grid).
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
        # nothing in the compiled hot path).  Also covers predict().
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

        .. warning::
            The outputs are observed-frame AB maggies **only if** ``theta``
            contains ``"zred"``: that key gates the cosmological flux factor
            ``(1+z) (10pc/D_L)^2`` (and the L_sun/Hz → cgs conversion)
            inside :func:`ceridwen.cosmology.flux_factor_maggies`.  Without
            it the values are raw 10 pc-frame numbers, ~6e21 too bright at
            z = 0.1.  ``SedModel.predict`` injects its fixed ``zred``
            automatically; only direct callers of this method (and of
            ``get_line_spec``) need to supply it themselves.
        """
        spectrum_phot, spectrum_slit, line_slit = \
            self._assemble_observer_spectra(theta)
        spectrum_phot, spectrum_slit, line_slit = self._apply_mass_redshift_igm(
            spectrum_phot, spectrum_slit, line_slit, theta
        )
        return self._project_observations(
            spectrum_phot, spectrum_slit, line_slit, observations, theta
        )

    def _assemble_observer_spectra(self, theta):
        """Build the photometry- and slit-facing spectra from the single
        ``(continuum, lines)`` decomposition.
        """
        # Decompose the model SED once (see get_spectrum_components), then
        # build the two observer-facing spectra from it:
        #
        #   spectrum_phot -- continuum + emission lines at TRUE strength.
        #                    Photometry captures the full field of view, so
        #                    it always sees the intrinsic line flux.
        #   spectrum_slit -- continuum + eline_scaling * lines.
        #                    Spectrum and Lines are slit-measured and lose
        #                    flux; eline_scaling is the aperture correction
        #                    (a FRACTION; 1.0 = no loss, 0.65 = lines at 65%,
        #                    2.0 = lines doubled).  Absent from theta -> factor
        #                    1.0, so spectrum_slit equals spectrum_phot.
        #
        # nebemlineinspec is intentionally NOT consulted here (it governs only
        # the default of the public get_spectrum); the observations must always
        # see the lines.  All key checks are Python-static so the branches fold
        # out at trace time.
        spectrum_cont, line_component = self.get_spectrum_components(theta)

        spectrum_phot = spectrum_cont + line_component       # lines unscaled
        if "eline_scaling" in theta:
            eline_factor = jnp.ravel(theta["eline_scaling"])[0]   # fraction; 1.0 = no loss
            line_slit = (eline_factor.astype(spectrum_cont.dtype)
                         * line_component)
        else:
            line_slit = line_component
        spectrum_slit = spectrum_cont + line_slit

        # ``line_slit`` is the emission-line-ONLY slit component.  Lines
        # observations must be projected from this term (2026-07-21 fix):
        # previously the full slit spectrum was used, so the Gaussian line
        # apertures integrated the stellar + nebular CONTINUUM under every
        # line as well.  Catalogue line fluxes are continuum-subtracted, so
        # that continuum term entered the likelihood as a spurious "line"
        # flux scaling with the evolved stellar mass -- the dominant
        # faint-line bias in the JADES fits (inflated high-order H ladder,
        # auroral and He lines; inverted ladders in low-sSFR posteriors).
        return spectrum_phot, spectrum_slit, line_slit

    def _apply_mass_redshift_igm(self, spectrum_phot, spectrum_slit,
                                 line_slit, theta):
        """Apply mass, redshift (flux factor) and IGM multiplicative scaling
        to the observer spectra (photometry-facing, slit-facing, and the
        emission-line-only slit component).
        """
        # Identical multiplicative factors for all three; each is computed
        # once and applied to all.
        if "logmass" in theta:
            mass_scale = jnp.float32(10.0 ** theta["logmass"][0])
            spectrum_phot = spectrum_phot * mass_scale
            spectrum_slit = spectrum_slit * mass_scale
            line_slit     = line_slit     * mass_scale

        if "zred" in theta:
            from ..cosmology import flux_factor_maggies
            z_scalar = jnp.ravel(theta["zred"])[0]
            ff = jnp.float32(flux_factor_maggies(z_scalar))
            spectrum_phot = spectrum_phot * ff
            spectrum_slit = spectrum_slit * ff
            line_slit     = line_slit     * ff
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
                line_slit     = line_slit     * transmission

        return spectrum_phot, spectrum_slit, line_slit

    def _project_observations(self, spectrum_phot, spectrum_slit, line_slit,
                              observations, theta):
        """Project the scaled spectra onto each Observation, returning the
        ``{obs.name: prediction}`` dict.
        """
        # Project: Photometry sees the full-field-of-view spectrum;
        # Spectrum (slit-measured) sees the eline_scaling-corrected FULL
        # spectrum (a real spectrograph records the continuum); Lines sees
        # the emission-line-ONLY slit component, because catalogue line
        # fluxes are continuum-subtracted -- projecting the full spectrum
        # through the positive Gaussian apertures added the continuum under
        # each line to the prediction (2026-07-21 fix).
        # ``isinstance`` is resolved statically at trace time.
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
        )
        from ..observation.lines import Lines as _Lines
        # Direct grid-based line fluxes (2026-07-21): predicting Lines by
        # Gaussian-aperture extraction from the painted spectrum recovers
        # only ~0.35-0.5 of the painted flux (narrow-line aperture
        # normalisation vs resolution-floor + LOSVD line widths, with a
        # wavelength-dependent trend).  When a nebular module exists, Lines
        # observations are therefore predicted straight from the CLOUDY
        # grid line luminosities (exact; library/resolution/smoothing
        # independent).  The static any()/getattr checks fold at trace time.
        _has_lines_obs = any(isinstance(o, _Lines) for o in observations)
        _line_fluxes = (self.predict_line_fluxes(theta)
                        if _has_lines_obs
                        and getattr(self, "neb", None) is not None
                        else None)
        out = {}
        free_z_in_theta = "zred" in theta
        # Velocity-broadening dispatch.
        #
        # ``Spectrum.fit_sigma_smooth`` is a static Python flag set at
        # construction.  When True, the runtime stellar LOSVD
        # (Prospector convention: ``sigma_smooth`` [km/s]) is pulled
        # from ``theta`` and threaded into the closure that
        # ``Spectrum.setup_for_model`` built.  When False, the
        # obs.predict signature is unchanged and the static fast path
        # is preserved bit-for-bit.  The ``isinstance`` + ``getattr``
        # checks resolve at trace time, so the compiled XLA graph
        # contains only the chosen path.
        for obs in observations:
            if isinstance(obs, _Lines):
                if _line_fluxes is not None:
                    out[obs.name] = _line_fluxes[self._neb_cube_rows_for(obs)]
                else:
                    # no nebular module: line component is identically zero
                    out[obs.name] = obs.predict(line_slit, self.wave)
                continue
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
            else:
                out[obs.name] = obs.predict(spec_for_obs, self.wave)
        return out

    def _neb_cube_rows_for(self, obs):
        """Map an observation's lines onto the nebular cube rows BY REST
        WAVELENGTH (cached on the observation; static at trace time).

        ``obs.line_ind`` indexes ``$SPS_HOME/data/emlines_info.dat``, but the
        line-luminosity cube rows follow the ZAU ``.lines`` file's own
        ordering.  On installations where the two files come from different
        FSPS vintages (e.g. hand-installed BPASS cubes next to a stock
        emlines_info.dat) the orderings DISAGREE, and an index-based gather
        silently returns neighbouring lines -- observed 2026-07-21 on Tursa,
        where the direct line predictions were wrong per-line (Hb off by
        1000x) while the painted spectrum, which places lines by the cube's
        own wavelengths, was correct.  Wavelength matching is immune to the
        vintage mismatch; a >1 A discrepancy raises immediately instead of
        corrupting the likelihood.  See also the construction-time
        cross-check in ``NebularModel.__init__`` (emline_index_consistent).
        """
        rows = getattr(obs, "_neb_cube_rows", None)
        if rows is not None:
            return rows
        pos = np.asarray(self.neb.nebem_line_pos, dtype=float)
        lam = np.asarray(obs.wavelength, dtype=float)
        idx = np.array([int(np.argmin(np.abs(pos - l))) for l in lam])
        dmax = float(np.max(np.abs(pos[idx] - lam)))
        if dmax > 1.0:
            worst = int(np.argmax(np.abs(pos[idx] - lam)))
            raise ValueError(
                "Emission-line wavelength matching failed: observed line "
                f"{getattr(obs, 'line_names', ['?'] * len(lam))[worst]!r} at "
                f"{lam[worst]:.2f} A has no nebular-cube line within 1 A "
                f"(nearest {pos[idx[worst]]:.2f} A). The ZAU .lines cube and "
                "emlines_info.dat likely come from different FSPS versions.")
        li_ext = np.asarray(obs.line_ind)
        if not np.array_equal(idx, li_ext):
            warnings.warn(
                "emlines_info.dat indices and ZAU cube rows disagree for "
                f"{int((idx != li_ext).sum())}/{idx.size} lines of obs "
                f"{obs.name!r}; using wavelength-matched cube rows. Your "
                "$SPS_HOME mixes file vintages -- consider aligning them.",
                stacklevel=2)
        obs._neb_cube_rows = jnp.asarray(idx)
        return obs._neb_cube_rows

    def predict_line_fluxes(self, theta):
        """Observed-frame integrated emission-line fluxes for EVERY line in
        the nebular grid, computed directly from the CLOUDY line
        luminosities (no spectral painting / aperture round-trip).

        Pipeline mirrors the spectrum path exactly: SFH/metallicity weights,
        (1 - frac_obrun) nebular scaling, birth-cloud (per-age) + diffuse
        dust evaluated AT the line wavelengths, OB-runaway bypass, mass,
        cosmological flux factor, IGM at the line wavelengths, and
        eline_scaling.  Returns shape ``(n_lines_grid,)`` in the same
        integrated-flux units as ``Lines.flux`` (erg s^-1 cm^-2 when theta
        carries ``zred``); index with ``Lines.line_ind`` to compare with an
        observation.  Normalisation matches the total flux painted into the
        spectrum, so photometry and line predictions stay consistent.
        """
        W = self.calculate_ssp_weights(theta=theta)          # (n_z, n_age)
        logZ_gas = theta["gas_logz"]
        logU     = theta["gas_logu"]
        line_lum = self.neb.evaluate_batch_line_lum(
            logZ_gas, logU, self._neb_ages_young, self._neb_logqq_young,
        )                                                    # (n_z, n_young, n_lines)
        if "frac_obrun" in theta:
            f_esc = jnp.ravel(theta["frac_obrun"])[0]
            line_lum = line_lum * (1.0 - f_esc)

        # Static linear-interp weights of the line wavelengths on the model
        # grid (both static arrays -> constant-folded under jit).
        lam = self.neb.nebem_line_pos                        # (n_lines,)
        li = jnp.clip(jnp.searchsorted(self.wave, lam) - 1,
                      0, self.wave.shape[0] - 2)
        lf = jnp.clip((lam - self.wave[li])
                      / (self.wave[li + 1] - self.wave[li]), 0.0, 1.0)

        # Birth-cloud (per SSP age) + diffuse attenuation at line positions.
        # ``attenuate_dust`` / ``_age_bin_mix`` only exist when the CSP was
        # built with the corresponding dust components -- mirror the
        # get_spectrum_* dispatch and fall back to no attenuation (factor 1)
        # when they are absent (e.g. add_dust=False test/intrinsic models).
        n_young = self._neb_young_idx.shape[0]
        attn_age_lines = jnp.ones((n_young, lam.shape[0]))
        diff_lines = jnp.ones(lam.shape[0])
        if hasattr(self, "attenuate_dust"):
            attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
            if hasattr(self, "_age_bin_mix"):
                M = self._age_bin_mix
                tau_age = jnp.einsum("ab,bw->aw", M, attn)   # (n_age, n_wave)
                tau_lines = ((1.0 - lf)[None, :] * tau_age[:, li]
                             +        lf[None, :] * tau_age[:, li + 1])
                aal = jnp.exp(-tau_lines)                    # (n_age, n_lines)
                if "frac_obrun" in theta:
                    fo = jnp.ravel(theta["frac_obrun"])[0]
                    aal = (1.0 - fo) * aal + fo
                attn_age_lines = aal[self._neb_young_idx, :]
            diff_lines = jnp.exp(-((1.0 - lf) * attn_diffuse[li]
                                   + lf * attn_diffuse[li + 1]))

        F = jnp.einsum("zy,zyl,yl->l",
                       W[:, self._neb_young_idx], line_lum, attn_age_lines)
        F = F * diff_lines

        if "logmass" in theta:
            F = F * 10.0 ** jnp.ravel(theta["logmass"])[0]
        if "zred" in theta:
            from ..cosmology import flux_factor_maggies
            z_scalar = jnp.ravel(theta["zred"])[0]
            # flux_factor_maggies carries the (1+z) bandwidth-compression
            # Jacobian appropriate for PER-Hz flux densities (f_nu).  An
            # INTEGRATED line flux F = L / (4 pi d_L^2) must not include it:
            # integrating f_nu,obs over d nu_obs cancels the (1+z) against
            # the compressed bandwidth.  Dividing here keeps the direct
            # predictions equal to the observed-frame integral of the
            # painted line spectrum (photometry/lines consistency) -- found
            # 2026-07-21 on the 1025955 rerun, where predictions were
            # exactly (1+z) = 2.87x above the painted-spectrum line fluxes.
            F = F * flux_factor_maggies(z_scalar) / (1.0 + z_scalar)
            if self.igm is not None:
                if "igm_factor" in theta:
                    ig_factor = jnp.ravel(theta["igm_factor"])[0]
                else:
                    ig_factor = jnp.float32(self.igm_factor)
                trans = self.igm.attenuation(self.wave, z_scalar,
                                             factor=ig_factor)
                F = F * ((1.0 - lf) * trans[li] + lf * trans[li + 1])
        if "eline_scaling" in theta:
            F = F * jnp.ravel(theta["eline_scaling"])[0]
        return F

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
        # Apply the eline_scaling fraction (1.0 = no loss) so the line-only
        # spectrum is consistent with the Lines.predict fluxes from predict().
        if "eline_scaling" in theta:
            line_only = line_only * jnp.ravel(theta["eline_scaling"])[0]
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
    # SFH visualisation
    # -----------------------------------------------------------------------

    def display_sfh(self, theta=None, ax=None, *,
                    overlay_nodes=True, show_bin_edges=False,
                    units="Gyr", **plot_kwargs):
        """Plot the SFH against lookback time, rendered identically to the
        interpretation used by :meth:`_ssp_weights`.

        For ``sfh_interp == "step"`` this draws a piecewise-constant function
        with one horizontal segment per bin ``[T_{i+1}, T_i]`` at height
        :math:`\\bar\\psi_i` (the per-bin SFR consumed by
        ``calculate_ssp_weights_*_step``).  For ``sfh_interp == "linear"`` it
        draws the piecewise-linear interpolant between per-node SFR values --
        the same function whose analytic integral against the SSP age grid
        is computed by ``intsfwght``.

        Lookback time is read from ``theta["lookback_time"]`` if supplied
        (units: Gyr) and otherwise falls back to ``self.sfh_times`` (which
        is stored in years and converted back to Gyr here).  ``theta_init``
        intentionally does NOT carry ``lookback_time`` -- it is a static
        grid, not a free parameter -- so the default fallback path is the
        common case.

        The x-axis runs left-to-right in increasing lookback time: present
        day (T = 0) sits at the origin on the left, and the oldest sampled
        node sits on the right.  This matches the natural index order of
        ``theta["lookback_time"]``.

        Parameters
        ----------
        theta : dict, optional
            Parameter dict to display.  Defaults to ``self.theta_init``.
            If it carries a ``"lookback_time"`` entry, that takes
            precedence over ``self.sfh_times`` for the x-axis grid.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.  If None, a new figure is created.
        overlay_nodes : bool
            If True, mark per-bin SFR values at bin midpoints (step mode)
            or per-node SFR values at lookback nodes (linear mode).
        show_bin_edges : bool
            If True, draw vertical dotted lines at every node ``T_i``.
        units : {"Gyr", "yr", "Myr"}
            X-axis units for the lookback-time axis.  The SFR axis is
            always [M_sun / yr].
        **plot_kwargs
            Forwarded to the per-segment ``ax.plot`` calls (e.g.
            ``color``, ``lw``, ``linestyle``, ``label``).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.

        Raises
        ------
        AssertionError
            If the per-bin integral of the displayed SFR disagrees with the
            per-bin mass ``m_target`` used by :meth:`_ssp_weights` by more
            than 1e-6 relative.  This pins the visual to the weight code so
            future refactors of either side cannot silently diverge.
        """
        import matplotlib.pyplot as plt

        theta = self.theta_init if theta is None else theta

        # Lookback-time grid (Gyr).  theta_init does not carry it (stripped
        # in initialize_model_structure), so the default path falls back to
        # self.sfh_times (yr) converted to Gyr.
        if "lookback_time" in theta:
            T_gyr = np.asarray(theta["lookback_time"], dtype=float)
        else:
            T_gyr = np.asarray(self.sfh_times, dtype=float) / 1e9
        T_gyr = np.atleast_1d(T_gyr).ravel()
        n_time = T_gyr.size

        psi = np.atleast_1d(np.asarray(theta["sfh"], dtype=float)).ravel()
        if psi.size not in (n_time, n_time - 1):
            raise AssertionError(
                f"theta['sfh'] has length {psi.size}; expected {n_time} "
                f"(per-node) or {n_time - 1} (per-bin, FastStepBasis)."
            )
        per_bin = (psi.size == n_time - 1)

        # Bin widths in years (physical units for the mass-conservation check).
        # Lookback strictly INCREASING, so dt > 0 via T_yr[1:]-T_yr[:-1].
        T_yr = T_gyr * 1e9
        dt_yr = T_yr[1:] - T_yr[:-1]
        if not np.all(dt_yr > 0):
            raise AssertionError(
                "lookback-time grid must be strictly increasing (today at "
                f"index 0, oldest last); got dt_yr = {dt_yr}"
            )

        # Per-bin SFR -- same branch as _ssp_weights in step mode.  sfh[:-1] is
        # the younger-side node, sfh[1:] the older-side node.
        if per_bin:
            bar_psi = psi
        else:
            bar_psi = 0.5 * (psi[:-1] + psi[1:])

        # Per-node SFR for the linear interpolant.  For per-bin input
        # (non-canonical in linear mode -- _ssp_weights expects per-node),
        # invert the step-mode collapse: interior nodes are the mean of
        # the two adjacent per-bin values; endpoint nodes take the
        # neighbouring bin's value.
        if per_bin:
            psi_nodes = np.empty(n_time, dtype=float)
            psi_nodes[0]    = psi[0]
            psi_nodes[-1]   = psi[-1]
            psi_nodes[1:-1] = 0.5 * (psi[:-1] + psi[1:])
        else:
            psi_nodes = psi

        if units == "Gyr":
            scale, xlabel = 1.0,   "Lookback time [Gyr]"
        elif units == "Myr":
            scale, xlabel = 1e3,   "Lookback time [Myr]"
        elif units == "yr":
            scale, xlabel = 1e9,   "Lookback time [yr]"
        else:
            raise ValueError(
                f"units must be 'Gyr', 'Myr', or 'yr'; got {units!r}"
            )
        T_plot = T_gyr * scale

        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.0))

        style = {"color": "C0", "lw": 1.5}
        style.update(plot_kwargs)
        marker_color = style.get("color", "C0")

        n_bin = n_time - 1

        if self.sfh_interp == "step":
            # One horizontal segment per bin -- exactly the piecewise-constant
            # function the step-mode weight calculator integrates against the
            # SSP Voronoi cells.
            for i in range(n_bin):
                ax.plot([T_plot[i + 1], T_plot[i]],
                        [bar_psi[i],   bar_psi[i]],
                        **style)
            if overlay_nodes:
                T_mid = 0.5 * (T_plot[:-1] + T_plot[1:])
                ax.scatter(T_mid, bar_psi, marker="o",
                           color=marker_color, s=20, zorder=3)
        else:  # "linear"
            for i in range(n_bin):
                ax.plot([T_plot[i + 1], T_plot[i]],
                        [psi_nodes[i + 1], psi_nodes[i]],
                        **style)
            if overlay_nodes:
                ax.scatter(T_plot, psi_nodes, marker="o",
                           color=marker_color, s=20, zorder=3)

        if show_bin_edges:
            for t in T_plot:
                ax.axvline(t, color="grey", lw=0.5, linestyle=":")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\dot{M}_\star\;[\mathrm{M_\odot\,yr^{-1}}]$")

        # T = 0 (today) sits at the origin on the left; lookback time
        # increases to the right.  No axis inversion.

        total_mass = float(np.sum(bar_psi * dt_yr))
        ax.set_title(
            f"sfh_interp={self.sfh_interp!r}, n_time={n_time}, "
            f"M_total = {total_mass:.3e} M_sun"
        )

        # ------------------------------------------------------------------
        # Mass-conservation contract.
        #
        # m_target  : per-bin mass m2 that _ssp_weights distributes onto
        #             the SSP grid (linear m2 reduces analytically to the
        #             trapezoid between psi nodes; step m2 is bar_psi * dt).
        # m_display : trapezoidal integral of the polyline this method just
        #             drew, segment by segment.  For step we drew a constant
        #             over each bin; for linear we drew the chord between
        #             adjacent nodes.  The integrals must agree by the same
        #             formulas -- a future refactor that changes either side
        #             alone will trip this assertion.
        # ------------------------------------------------------------------
        if self.sfh_interp == "step":
            m_target  = bar_psi * dt_yr
            m_display = bar_psi * dt_yr
        else:
            m_target  = 0.5 * (psi_nodes[:-1] + psi_nodes[1:]) * dt_yr
            m_display = 0.5 * (psi_nodes[:-1] + psi_nodes[1:]) * dt_yr

        rel = np.abs(m_display - m_target) / np.maximum(np.abs(m_target), 1e-30)
        if not np.all(rel < 1e-6):
            raise AssertionError(
                "display_sfh: integrated displayed SFR disagrees with the "
                f"per-bin mass used by _ssp_weights ({self.sfh_interp!r} "
                f"mode); max rel diff = {float(rel.max()):.3e}.  This means "
                "the plot and the weight calculation have drifted out of "
                "sync -- one of them was refactored without the other."
            )

        return ax

    # -----------------------------------------------------------------------
    # SSP weight calculation (core; identical maths to csp.py)
    # -----------------------------------------------------------------------

    def _lookback_from_zred(self, zred):
        """SFH lookback grid (years) rescaled so its oldest node tracks the
        age of the universe at the sampled redshift.

        Used in the free-redshift forward pass (``track_zred_age=True``).  The
        construction grid ``self.sfh_times`` (oldest node ``self.sfh_times[-1]``
        ~ age at the build redshift) is scaled by
        ``age_gyr(zred) / age_gyr(z_build)`` so that the oldest node equals
        ``age_gyr(zred)`` while the relative node spacing (and therefore
        ``n_time``) is preserved; the result is clipped to the SSP age ceiling.

        Fully differentiable in ``zred`` through
        :func:`ceridwen.cosmology.age_gyr` (a JAX Simpson integral), so a
        gradient of the likelihood w.r.t. ``zred`` flows through the age-grid
        construction, not only the flux factor.  No Python-scalar ``zred`` is
        baked at trace time and there is no host-side branching on the traced
        value -- the only branch (in :meth:`_ssp_weights`) is on the static
        presence of dict keys.
        """
        from ceridwen.cosmology import age_gyr
        z = jnp.ravel(jnp.asarray(zred, dtype=float))[0]
        tuniv_yr = age_gyr(z) * 1.0e9
        ref_old_yr = self.sfh_times[-1]
        scaled = self.sfh_times * (tuniv_yr / ref_old_yr)
        return jnp.clip(scaled, 0.0, self._age_clip_hi)

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

        Units (identical across all four combinations)
        -----------------------------------------------
        * Metallicity — ``theta["Z"]`` (const) and ``theta["zh"]`` (var) are
          BOTH ``log10`` of the *absolute* metallicity, on the SSP grid
          ``self.zmet`` (== ``SSPData.ssp_lgmet``).  ``Z`` is a single scalar;
          ``zh`` is one value per lookback-time node — same unit, different
          shape.
        * SFR — ``theta["sfh"]`` is a *linear* star-formation rate, floored
          identically at ``1e-30`` in every mode.

        The summed-weight ``maximum(0, .)`` clamp is applied in ``const`` mode.
        Both modes share the ``1e-30`` SFR floor, keeping the SFR units
        consistent and avoiding a divide-by-zero NaN in the var-zh linear slope
        for (near-)zero SFR nodes.
        """
        # Single SFR floor, identical for EVERY (zh_mode, sfh_mode) combination,
        # so all four weight calculations consume the star-formation-rate
        # history in the same units (linear SFR) with the same regularisation.
        # ``1e-30`` is a tiny positive floor that keeps the var-zh linear slope
        # (which divides by the per-node SFR) finite.
        floor = 1e-30
        sfh = jnp.clip(theta["sfh"], floor, None)

        # ── Lookback convention ─────────────────────────────────────────────
        # self.sfh_times is monotonically INCREASING in lookback (yr):
        #   sfh_times[0]   = 0           (today, present-day node)
        #   sfh_times[-1]  ≈ T_universe  (oldest sampled node)
        # Bin i (i = 0 .. n_time-2) lies between consecutive nodes:
        #   younger end  t_young[i] = sfh_times[i]
        #   older  end   t_old[i]   = sfh_times[i+1]
        #   width        dt[i]      = t_old[i] - t_young[i]   (> 0)
        # Bin 0 is therefore the YOUNGEST bin (touching today) and bin
        # n_time-2 is the oldest.  ``theta["sfh"]`` is indexed to match:
        # sfh[0] is the SFR at the present-day node (per-node) or the
        # SFR of the youngest bin (per-bin).
        #
        # Per-sample lookback grid (free-redshift).  Precedence (branch on the
        # STATIC presence of dict keys only -- never on a traced value):
        #   1. explicit theta["lookback_time"] (Gyr) -- a transform recomputed
        #      the SFH age-bins (e.g. the model-specific extra_young grid) from
        #      the sampled zred; use it verbatim.
        #   2. track_zred_age and theta["zred"] -- derive the grid HERE from
        #      age_gyr(zred) by rescaling the construction grid so the oldest
        #      node tracks the age of the universe (the first-class ceridwen
        #      free-z path; see _lookback_from_zred).
        #   3. otherwise the cached self.sfh_times (the fixed-z path,
        #      bit-for-bit unchanged).
        # n_time is unchanged in every case, so the traced shapes are static.
        if "lookback_time" in theta:
            _times = jnp.atleast_1d(
                jnp.asarray(theta["lookback_time"], dtype=float)) * 1e9  # Gyr->yr
        elif self.track_zred_age and "zred" in theta:
            _times = self._lookback_from_zred(theta["zred"])             # years
        else:
            _times = self.sfh_times
        t_young = _times[:-1]
        t_old   = _times[1:]
        dt      = t_old - t_young

        if sfh_mode == "linear":
            # SFR(t) is linearly interpolated between adjacent nodes:
            #   SFR(t_young) = sfh[:-1]   (younger-end node SFR)
            #   SFR(t_old)   = sfh[1:]    (older-end node SFR)
            # Parametrised as SFR(t) = sfh[:-1] * (1 + slope * (t - t_young)),
            # so that SFR(t_old) = sfh[:-1] * (1 + slope * dt) = sfh[1:].
            # Solving for slope:
            slope = jnp.diff(sfh) / (sfh[:-1] * dt)
            m2    = sfh[:-1] * (1.0 + 0.5 * slope * dt) * dt   # = 0.5*(sfh[:-1]+sfh[1:])*dt

            # intsfwght expects the affine form  SFR(t) = sfh_ref * (a + slope*t):
            #   a = 1 - slope * t_young.
            tprime = jnp.maximum(0.0, t_young)
            a      = 1.0 - slope * tprime

            logage_lo = self._logage_lo
            logage_hi = self._logage_hi
            dlogage   = self._dlogage
            j         = self._j_range
            n_ssp     = self.ssp_ages_lgyr.size

            log_t_young = jnp.log10(jnp.clip(t_young, self._age_clip_lo, self._age_clip_hi))[:, None]
            log_t_old   = jnp.log10(jnp.clip(t_old,   self._age_clip_lo, self._age_clip_hi))[:, None]

            L = jnp.clip(logage_lo[None, :], log_t_young, log_t_old)
            R = jnp.clip(logage_hi[None, :], log_t_young, log_t_old)

            jmin = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t_young)) - 1, 0, n_ssp - 1)
            jmax = jnp.clip(jnp.searchsorted(self.ssp_ages_lgyr, jnp.log10(t_old))   + 2, 0, n_ssp - 1)

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
            # Per-bin SFR.  Per-node input averages adjacent nodes:
            # sfh[:-1] (younger-side) and sfh[1:] (older-side).
            if self.sfh_per_bin:
                sfh_mid = sfh
            else:
                sfh_mid = 0.5 * (sfh[:-1] + sfh[1:])
            m2 = sfh_mid * dt

            # Overlap between bin i's [t_young, t_old] window and SSP age
            # cell j's Voronoi interval [voronoi_lo, voronoi_hi] (which is
            # in years on the SSP axis, untouched by the lookback flip).
            overlap = jnp.maximum(
                0.0,
                jnp.minimum(t_old[:, None],   self._ssp_voronoi_hi[None, :])
                - jnp.maximum(t_young[:, None], self._ssp_voronoi_lo[None, :])
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
        (shape ``(n_time,)``, linear SFR) and ``theta["Z"]`` (shape ``(1,)``,
        log10 absolute metallicity on the ``self.zmet`` / ``ssp_lgmet`` grid —
        NOT log10 Z/Zsun).  Same units as the var-zh variants' ``theta["zh"]``.
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
        (shape ``(n_time,)``, linear SFR) and ``theta["zh"]`` (shape
        ``(n_time,)``, log10 absolute metallicity at each lookback time, on the
        ``self.zmet`` / ``ssp_lgmet`` grid — NOT log10 Z/Zsun).  Identical units
        to the const-zh variants' ``theta["Z"]``.
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

    # Nebular helper: build the (n_z, n_age, n_wave) nebular array, with or
    # without broadened emission lines.  ``include_lines`` defaults to
    # ``self.nebemlineinspec`` for external callers, but ``csp.predict`` forces
    # ``include_lines=True`` so ``Lines.predict``'s Gaussian-aperture
    # integration still sees the lines in the spectrum.
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
        # Absent → defaults to f_esc = 0 (no kill-ion restoration, no nebular
        # scaling).  The check is a Python-static dict-key lookup, free at
        # trace time.
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
        # Cast dust curves to float32 — keeps the forward model in single
        # precision until the likelihood.
        M        = self._age_bin_mix
        tau_age  = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age = jnp.exp(-tau_age)

        # FSPS-style OB-runaway dust escape (``add_dust.f90`` L93-94).
        # A fraction ``frac_obrun`` of the young-star flux bypasses the
        # birth-cloud (``attn_age``) attenuation while still passing through
        # the diffuse component below (the second physical effect of FSPS's
        # ``frac_obrun`` knob; the first -- LyC escape + ``Q`` scaling -- is
        # applied above).  Old SSP ages already have ``attn_age = 1``, so the
        # mix is a no-op for them.  When ``frac_obrun`` is absent or 0, this is
        # identically ``attn_age``.
        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo
            # FIX (fesc double-count): the escaped LyC restored above
            # (kill_ion: young ages, lambda<912, amplitude f_esc*flux = fo*F)
            # already left through the density-bounded channel, so it bypasses
            # the birth-cloud dust COMPLETELY -- attn_age = 1, not the runaway
            # mix (1-fo)*a_bc + fo.  Without this line the emergent LyC is
            #     fo*F * [(1-fo)*a_bc + fo] * d     with a_bc = exp(-tau_bc(912)),
            #                                       d = diffuse transmission,
            # i.e. the intended fo*F*d times a SPURIOUS factor a_bc+fo*(1-a_bc)
            # that lies in [a_bc, 1].  At 912 A the birth cloud is ~opaque
            # (a_bc<<1), so the factor -> fo and the escaping LyC -> fo**2*F*d
            # (the frac_obrun**2 regime); only if a_bc -> 1 (no BC dust) does the
            # double-count vanish on its own.  attn_age = 1 removes the factor
            # identically for ANY a_bc, giving fo*F*d.  Diffuse dust still
            # applies below, matching FSPS: runaways bypass birth-cloud (dust1)
            # but see the diffuse ISM (dust2).  No effect when frac_obrun absent/0.
            attn_age = jnp.where(self.kill_ion, jnp.float32(1.0), attn_age)

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
            # FIX (fesc double-count): the escaped LyC restored above
            # (kill_ion: young ages, lambda<912, amplitude f_esc*flux = fo*F)
            # already left through the density-bounded channel, so it bypasses
            # the birth-cloud dust COMPLETELY -- attn_age = 1, not the runaway
            # mix (1-fo)*a_bc + fo.  Without this line the emergent LyC is
            #     fo*F * [(1-fo)*a_bc + fo] * d     with a_bc = exp(-tau_bc(912)),
            #                                       d = diffuse transmission,
            # i.e. the intended fo*F*d times a SPURIOUS factor a_bc+fo*(1-a_bc)
            # that lies in [a_bc, 1].  At 912 A the birth cloud is ~opaque
            # (a_bc<<1), so the factor -> fo and the escaping LyC -> fo**2*F*d
            # (the frac_obrun**2 regime); only if a_bc -> 1 (no BC dust) does the
            # double-count vanish on its own.  attn_age = 1 removes the factor
            # identically for ANY a_bc, giving fo*F*d.  Diffuse dust still
            # applies below, matching FSPS: runaways bypass birth-cloud (dust1)
            # but see the diffuse ISM (dust2).  No effect when frac_obrun absent/0.
            attn_age = jnp.where(self.kill_ion, jnp.float32(1.0), attn_age)

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
