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

import jax.numpy as jnp
from jax import jit, vmap
import pprint

import astropy.constants as const

from ceridwen.dust.DustModel import Dust, DiffuseDust
from ceridwen.dust.DustEmission import DustEmission
from ceridwen.neb.NebularGridModel import NebularModel
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



@jit
def intsfwght(t_hi, t_lo, a, slope, logage):
    """Integrated SFH weight between log-time limits."""

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
        Frozen dataclass with SSP grids (wave, flux, ages, zmet, logqq).
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
        self.logqq     = jnp.array(SSPData.log_qq)         # (n_z, n_age)
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
        assert sfh.shape == (self.n_time,), (
            f"'sfh' shape {sfh.shape} must match 'lookback_time' length {self.n_time}"
        )

        # --- Metallicity mode detection ------------------------------------
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
            init_neb_params.update({'sps_home': sps_home, 'csp_lambda': self.wave})
            print("Initializing Nebular Emission model...")
            self.neb = NebularModel(**init_neb_params)

            # get_default_params() now returns a plain dict
            neb_defaults = self.neb.get_default_params()
            for k, v in neb_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.neb_param_names = list(neb_defaults.keys())

            young_thresh     = jnp.log10(21.0e6 / 1.0e9)
            self.young_mask  = self.ages < young_thresh
            self.ion_mask    = self.wave < 912.0
            self.kill_ion    = self.young_mask[:, None] & self.ion_mask[None, :]

            # Precompute young-age slices for efficient nebular evaluation.
            # Instead of vmapping over all n_age=107 SSP ages and then
            # zeroing out old ages with young_mask, we only evaluate the
            # nebular model at the n_young ages where it is non-zero.
            young_idx = jnp.where(self.young_mask)[0]
            self._neb_young_idx   = young_idx
            self._neb_n_young     = int(young_idx.shape[0])
            self._neb_ages_young  = self.ssp_ages_lgyr[young_idx]   # log10(yr)
            self._neb_logqq_young = self.logqq[:, young_idx]        # (n_z, n_young)

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
        self.get_spectrum = mapping[key]

    # -----------------------------------------------------------------------
    # initialize_model_structure: build theta_init as a dict
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def predict(self, theta, observations):
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
        directly.

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
        spectrum = self.get_spectrum(theta=theta)

        # Apply mass scaling here (once) rather than per-observation in
        # model.predict().  CSP normalises SFH to 1 M_sun; logmass sets
        # the physical amplitude.  Scaling the spectrum before projection
        # avoids N_obs separate multiplications.
        if "logmass" in theta:
            # Keep in float32: the spectrum is already float32 from the
            # forward model; cast the scalar to match.
            mass_scale = jnp.float32(10.0 ** theta["logmass"][0])
            spectrum = spectrum * mass_scale

        # Redshift flux-norm factor.  ``zred`` is OPTIONAL in theta; when
        # absent, the spectrum stays at the fiducial 10 pc / z=0 calibration
        # that ``setup_for_model(..., zred=0.0)`` assumes, so every existing
        # unit test is untouched.  When present, multiply by the
        # cosmologically-correct (1+z) (10 pc / D_L(z))^2 scaling — one
        # JAX-jittable scalar-scaling op, zero branches.
        if "zred" in theta:
            from ..cosmology import flux_factor_maggies
            z_scalar = jnp.ravel(theta["zred"])[0]
            spectrum = spectrum * jnp.float32(flux_factor_maggies(z_scalar))

            # IGM attenuation — optional.  ``self.igm is None`` when
            # constructed with ``add_igm=False`` and is a compile-time
            # constant, so the whole block constant-folds out of the
            # traced graph in that case.  When active, ``theta['igm_factor']``
            # can override the default ``igm_factor`` attribute to let the
            # sampler fit IGM strength.
            if self.igm is not None:
                if "igm_factor" in theta:
                    ig_factor = jnp.ravel(theta["igm_factor"])[0]
                else:
                    ig_factor = jnp.float32(self.igm_factor)
                transmission = self.igm.attenuation(
                    self.wave, z_scalar, factor=ig_factor,
                )
                spectrum = spectrum * transmission.astype(spectrum.dtype)

        # Python loop unrolled at trace time.  No Python branches inside
        # the traced code; obs.predict() is a concrete method call resolved
        # before XLA compilation.
        return {obs.name: obs.predict(spectrum, self.wave)
                for obs in observations}

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

    def calculate_ssp_weights_const_zh(self, theta):
        """
        SSP weights for a constant metallicity history.

        Reads ``theta["sfh"]`` (shape ``(n_time,)``) and ``theta["Z"]``
        (shape ``(1,)``, log10 Z/Zsun).
        """
        sfh = jnp.clip(theta["sfh"], 1e-30, None)

        t_lo = self.sfh_times[1:]
        t_hi = self.sfh_times[:-1]
        dt   = t_hi - t_lo

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

        # Physical constraint: SSP weights are masses formed — they must be
        # non-negative.  The analytical log-age integration can produce small
        # negative values for steep SFH slopes due to sign cancellation in
        # intsfwght.  Clip to zero to enforce the physical bound, matching the
        # FSPS convention where tabular-SFH weights are always non-negative.
        w1 = jnp.maximum(0.0, w1)

        # Guard m1 against zero (can occur if all weights in a bin were clipped)
        # so that the m2/m1 mass-conservation rescaling does not produce NaN.
        m1 = jnp.maximum(w1.sum(axis=1), 1e-30)
        sfh_weights       = w1 * (m2 / m1)[:, None]
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

    def calculate_ssp_weights_const_zh_step(self, theta):
        """
        SSP weights for a constant metallicity history — piecewise-constant
        (FastStepBasis-style) SFH integration.

        Within each SFH time bin the SFR is held at the mean of the two
        endpoint values.  The contribution of SFH bin *i* to SSP age bin *j*
        is ``sfh_mid[i] * overlap_yr[i, j]``, where ``overlap_yr`` is the
        intersection length (in yr) of the two bins on a linear-time axis.
        All entries are non-negative by construction; no clipping is required.

        Reads ``theta["sfh"]`` (shape ``(n_time,)``) and ``theta["Z"]``
        (shape ``(1,)``, log10 Z/Zsun).
        """
        sfh = jnp.clip(theta["sfh"], 1e-30, None)

        t_lo = self.sfh_times[1:]    # (n_bin,)  younger edge, yr
        t_hi = self.sfh_times[:-1]   # (n_bin,)  older  edge, yr
        dt   = t_hi - t_lo           # (n_bin,)  positive

        # Constant SFR within each bin: mean of the two node values
        sfh_mid = 0.5 * (sfh[:-1] + sfh[1:])   # (n_bin,) >= 0

        # Physically correct mass per SFH bin under constant-SFR assumption
        m2 = sfh_mid * dt                        # (n_bin,)

        # Overlap in linear yr between each SFH bin and each SSP Voronoi cell.
        #
        # Each SSP age POINT j owns the Voronoi cell
        #   [_ssp_voronoi_lo[j], _ssp_voronoi_hi[j]]
        # whose boundaries are the linear-time midpoints to the neighbouring
        # SSP age points.  Overlapping this cell with the SFH bin directly
        # gives the correct mass attribution without any post-hoc splitting.
        # This matches the FSPS FastStepBasis ±ε convention.
        #
        #   overlap[i, j] = max(0,
        #       min(t_hi[i], voronoi_hi[j]) - max(t_lo[i], voronoi_lo[j]) )
        #
        overlap = jnp.maximum(
            0.0,
            jnp.minimum(t_hi[:, None], self._ssp_voronoi_hi[None, :])   # (n_bin, n_age)
            - jnp.maximum(t_lo[:, None], self._ssp_voronoi_lo[None, :])
        )

        # Weight at each SSP age POINT: SFR * Voronoi overlap.  Non-negative
        # by construction — no clipping or splitting needed.
        w1 = sfh_mid[:, None] * overlap          # (n_bin, n_age)

        # Mass-conserving rescale: force each SFH bin's weight sum to m2.
        m1 = jnp.maximum(w1.sum(axis=1), 1e-30)
        w1 = w1 * (m2 / m1)[:, None]            # (n_bin, n_age)

        total_sfh_weights = w1.sum(axis=0)        # (n_age,)

        # Metallicity interpolation (identical to the linear-scheme version)
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

    def calculate_ssp_weights_var_zh(self, theta):
        """
        SSP weights for a time-varying metallicity history.

        Reads ``theta["sfh"]`` (shape ``(n_time,)``) and ``theta["zh"]``
        (shape ``(n_time,)``, log10 Z/Zsun at each lookback time).
        """
        sfh = jnp.clip(theta["sfh"], self.tiny_logt, None)

        t_lo = self.sfh_times[1:]
        t_hi = self.sfh_times[:-1]
        dt   = t_hi - t_lo

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

        # Physical constraint — same rationale as in calculate_ssp_weights_const_zh.
        w1 = jnp.maximum(0.0, w1)
        m1 = jnp.maximum(w1.sum(axis=1), 1e-30)
        sfh_weights = w1 * (m2 / m1)[:, None]

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

    def calculate_ssp_weights_var_zh_step(self, theta):
        """
        SSP weights for a time-varying metallicity history — piecewise-constant
        (FastStepBasis-style) SFH integration.

        Identical non-negativity guarantee as
        ``calculate_ssp_weights_const_zh_step``, but distributes mass across
        metallicity bins using the per-SFH-bin mean metallicity from
        ``theta["zh"]`` (shape ``(n_time,)``, log10 Z/Zsun).

        Reads ``theta["sfh"]`` (shape ``(n_time,)``) and ``theta["zh"]``
        (shape ``(n_time,)``, log10 Z/Zsun at each lookback time).
        """
        sfh = jnp.clip(theta["sfh"], self.tiny_logt, None)

        t_lo = self.sfh_times[1:]    # (n_bin,)  younger edge, yr
        t_hi = self.sfh_times[:-1]   # (n_bin,)  older  edge, yr
        dt   = t_hi - t_lo           # (n_bin,)  positive

        # Constant SFR within each bin: mean of the two node values
        sfh_mid = 0.5 * (sfh[:-1] + sfh[1:])   # (n_bin,) >= 0
        m2      = sfh_mid * dt                   # (n_bin,)

        # Voronoi-cell overlap — same scheme as calculate_ssp_weights_const_zh_step.
        # Gives (n_bin, n_age) directly; no splitting required.
        overlap = jnp.maximum(
            0.0,
            jnp.minimum(t_hi[:, None], self._ssp_voronoi_hi[None, :])   # (n_bin, n_age)
            - jnp.maximum(t_lo[:, None], self._ssp_voronoi_lo[None, :])
        )

        w1    = sfh_mid[:, None] * overlap       # (n_bin, n_age)  >= 0

        # Mass-conserving rescale
        m1    = jnp.maximum(w1.sum(axis=1), 1e-30)
        sfh_weights = w1 * (m2 / m1)[:, None]   # (n_bin, n_age)

        # Per-SFH-bin mean metallicity (identical to the linear-scheme version)
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
        return M.T @ sfh_weights   # (n_z, n_age)

    # -----------------------------------------------------------------------
    # Spectrum methods (all read theta["key"] directly)
    # -----------------------------------------------------------------------

    def get_spectrum_dattn_nodem_neb(self, theta):
        """Dust attenuation + nebular emission, no dust emission."""
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        logZ_gas = theta["gas_logz"]
        logU     = theta["gas_logu"]

        # Vectorized nebular evaluation: factored trilinear interpolation
        # bilinear in (logZ, logU) then linear in age, batched over all
        # young ages and SSP metallicities in a single call.
        neb_young = self.neb.evaluate_batch(
            logZ_gas, logU, self._neb_ages_young, self._neb_logqq_young
        )

        # Scatter young-age results back into the full (n_z, n_age, n_wave) array
        n_z, n_age, n_wave = self.flux.shape
        neb_all = jnp.zeros((n_z, n_age, n_wave), dtype=jnp.float32)
        neb_all = neb_all.at[:, self._neb_young_idx, :].set(
            neb_young.astype(jnp.float32))

        stellar_fluxes  = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)
        combined_fluxes = stellar_fluxes + neb_all                      # (n_z, n_age, n_wave) float32

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        # Cast dust curves to float32 for the einsum — keeps the entire
        # forward model in single precision until the likelihood.
        M        = self._age_bin_mix
        tau_age  = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age = jnp.exp(-tau_age)

        W_f32 = W.astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", W_f32, combined_fluxes, attn_age)
        spectrum = spectrum * jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_neb(self, theta):
        """Dust attenuation + nebular emission + dust emission."""
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        logZ_gas = theta["gas_logz"]
        logU     = theta["gas_logu"]

        # Vectorized nebular evaluation (same factored trilinear as above)
        neb_young = self.neb.evaluate_batch(
            logZ_gas, logU, self._neb_ages_young, self._neb_logqq_young
        )

        n_z, n_age, n_wave = self.flux.shape
        neb_all = jnp.zeros((n_z, n_age, n_wave), dtype=jnp.float32)
        neb_all = neb_all.at[:, self._neb_young_idx, :].set(
            neb_young.astype(jnp.float32))

        stellar_fluxes  = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)
        combined_fluxes = stellar_fluxes + neb_all                      # float32

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

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

    def get_spectrum_dattn_nodem_noneb(self, theta):
        """Dust attenuation, no nebular, no dust emission."""
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M       = self._age_bin_mix
        tau_age = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age= jnp.exp(-tau_age)

        weights  = self.calculate_ssp_weights(theta).astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", weights, self.flux, attn_age)
        spectrum *= jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_noneb(self, theta):
        """Dust attenuation + dust emission, no nebular."""
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

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

    def get_spectrum_nodattn_nodem_noneb(self, theta):
        """Stellar continuum only — no dust, no nebular."""
        weights  = self.calculate_ssp_weights(theta=theta).astype(jnp.float32)
        return jnp.einsum("za,zaw->w", weights, self.flux)

    def get_spectrum_nodattn_nodem_neb(self, theta):
        """Nebular emission only, no dust."""
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        logZ_gas = theta["gas_logz"]
        logU = theta["gas_logu"]

        # Vectorized nebular evaluation (factored trilinear)
        neb_young = self.neb.evaluate_batch(
            logZ_gas, logU, self._neb_ages_young, self._neb_logqq_young
        )

        n_z, n_age, n_wave = self.flux.shape
        neb_all = jnp.zeros((n_z, n_age, n_wave), dtype=jnp.float32)
        neb_all = neb_all.at[:, self._neb_young_idx, :].set(
            neb_young.astype(jnp.float32))

        # Suppress stellar ionizing part
        stellar_fluxes = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)

        combined_fluxes = stellar_fluxes + neb_all  # float32
        W_f32 = W.astype(jnp.float32)
        return jnp.einsum("za,zaw->w", W_f32, combined_fluxes)
