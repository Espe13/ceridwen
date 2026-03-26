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

import jax.numpy as jnp
from jax import jit, vmap
import pprint

import astropy.constants as const

from ceridwen.dust.DustModel import Dust, DiffuseDust
from ceridwen.dust.DustEmission import DustEmission
from ceridwen.neb.NebularGridModel import NebularModel
from ceridwen.observation.observation import Photometry, Spectrum, Lines

tiny_number = 1e-70
LOG10E = jnp.log10(jnp.e)   # ≈ 0.4343, precomputed constant


print('changed import!!!!')
print()


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
        sps_home='/Users/amanda/Prospector/fsps',
        init_dust_params=None,
        diffuse_law='kriek_conroy',
        verbose=True,
        **kwargs,
    ):
        if theta is None:
            theta = {'lookback_time': 13.8 - jnp.linspace(1e-2, 13.8, 100)}
        if init_neb_params is None:
            init_neb_params = {"isoc_type": "mist", "cloudy_dust": True}
        if init_dust_params is None:
            init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 'laws': ['powerlaw']}

        # --- SSP grids (static, never part of theta) -----------------------
        self.flux      = jnp.array(SSPData.ssp_flux)       # (n_z, n_age, n_wave)
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

        self.tuniv      = tuniv
        self.tiny_logt  = tiny_logt
        self.sps_home   = sps_home

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

        if zh_const:
            self.calculate_ssp_weights = self.calculate_ssp_weights_const_zh
        else:
            self.calculate_ssp_weights = self.calculate_ssp_weights_var_zh

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

    def predict(self, theta, observations=None):
        """
        Compute a CSP spectrum and, optionally, project it onto observations.

        Parameters
        ----------
        theta : dict[str, Array]
            Free-parameter dict.  Must contain at minimum ``"sfh"`` and the
            metallicity key (``"Z"`` or ``"zh"``), plus any dust / nebular
            parameters required by the active physics model.
        observations : list of Observation, optional
            If None, returns the raw model spectrum as an Array of shape
            (n_wave,).  If provided, returns a dict keyed by ``obs.name``
            for every observation in the list.  Each value has shape matching
            the observation:

            - ``Photometry`` → shape (n_filters,), synthetic AB maggies
            - ``Spectrum``   → shape (n_pix,), model F_nu on the observed grid
            - ``Lines``      → shape (n_lines,), Gaussian-aperture line fluxes

        Returns
        -------
        spectrum : Array, shape (n_wave,)
            Returned when ``observations is None``.
        predictions : dict[str, Array]
            Returned when ``observations`` is provided.
        """
        spectrum = self.get_spectrum(theta=theta)
        if observations is None:
            return spectrum
        return {obs.name: self._project_spectrum(spectrum, obs)
                for obs in observations}

    # -----------------------------------------------------------------------
    # Observation-projection helpers
    # -----------------------------------------------------------------------

    def _project_spectrum(self, spectrum, obs):
        """
        Project the model spectrum (on ``self.wave``) onto a single observation.

        Parameters
        ----------
        spectrum : Array, shape (n_wave,)
            Model spectrum in F_nu units (L_sun Hz^{-1} M_sun^{-1} as output
            by ``get_spectrum``).
        obs : Photometry | Spectrum | Lines
            Observation object that defines the projection target.

        Returns
        -------
        Array
            Shape depends on ``obs`` type; see ``predict`` docstring.
        """
        if isinstance(obs, Photometry):
            # Filter convolution via sedpy_jax FilterSet.
            # get_maggies handles F_nu -> F_lambda conversion internally.
            return obs.get_maggies(self.wave, spectrum)

        elif isinstance(obs, Spectrum):
            # Linear interpolation of the model onto the observed wavelength
            # grid. jnp.interp clamps to boundary values outside the model
            # range, which is the correct behaviour (model coverage should
            # always exceed the spectral window).
            return jnp.interp(obs.wavelength, self.wave, spectrum)

        elif isinstance(obs, Lines):
            return self._extract_line_fluxes(spectrum, obs)

        else:
            raise TypeError(
                f"Unsupported observation type: {type(obs).__name__}.  "
                "Expected Photometry, Spectrum, or Lines."
            )

    def _extract_line_fluxes(self, spectrum, lines_obs):
        """
        Extract emission-line fluxes from the model spectrum by Gaussian-
        aperture integration centred on each line.

        The aperture width is 200 km/s (1-sigma), sufficient to capture the
        narrow-line profiles generated by ``NebularGridModel`` (~tens of km/s
        intrinsic width broadened to the model wavelength grid resolution)
        while rejecting adjacent continuum and neighbouring lines spaced by
        more than ~600 km/s.

        Parameters
        ----------
        spectrum : Array, shape (n_wave,)
            Model spectrum in F_nu units.
        lines_obs : Lines
            Observed emission-line object.  ``lines_obs.wavelength`` holds the
            vacuum rest-frame wavelengths [Å] of the lines to extract.

        Returns
        -------
        Array, shape (n_lines,)
            Integrated line flux for each line, in the same units as
            ``spectrum * d_lambda`` (L_sun M_sun^{-1}).

        Notes
        -----
        The Gaussian aperture weight is

        .. math::

            w_k(\\lambda) = \\exp\\!\\left[-\\frac{1}{2}
            \\left(\\frac{\\lambda - \\lambda_k}{\\sigma_k}\\right)^2\\right],
            \\qquad \\sigma_k = \\lambda_k \\cdot \\frac{\\sigma_v}{c}

        with :math:`\\sigma_v = 200\\,\\text{km/s}`.  The flux is the
        trapezoidal integral :math:`\\int w_k(\\lambda)\\,f_\\nu(\\lambda)\\,
        \\mathrm{d}\\lambda`.  As long as the same aperture is applied to data
        and model predictions, the choice of :math:`\\sigma_v` cancels in
        the likelihood.
        """
        c_kms = 2.998e5        # km/s
        sigma_v = 200.0        # km/s

        def _flux_one(lam0):
            sigma_aa = lam0 * (sigma_v / c_kms)
            weights  = jnp.exp(-0.5 * ((self.wave - lam0) / sigma_aa) ** 2)
            return jnp.trapz(weights * spectrum, self.wave)

        return vmap(_flux_one)(lines_obs.wavelength)

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

        w1              = jnp.pad(w_lo, ((0, 0), (0, 1))) + jnp.pad(w_hi, ((0, 0), (1, 0)))
        m1              = w1.sum(axis=1)
        sfh_weights     = w1 * (m2 / m1)[:, None]
        total_sfh_weights = sfh_weights.sum(axis=0)

        target_Z = theta["Z"]
        z_idx = jnp.clip(
            jnp.searchsorted(self.zmet, target_Z, side='left'),
            1, self._n_z - 1,
        )
        z1 = self.zmet[z_idx - 1]
        z2 = self.zmet[z_idx]
        w  = (target_Z - z1) / (z2 - z1)

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

        w1          = jnp.pad(w_lo, ((0, 0), (0, 1))) + jnp.pad(w_hi, ((0, 0), (1, 0)))
        m1          = w1.sum(axis=1)
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

    # -----------------------------------------------------------------------
    # Spectrum methods (all read theta["key"] directly)
    # -----------------------------------------------------------------------

    def get_spectrum_dattn_nodem_neb(self, theta):
        """Dust attenuation + nebular emission, no dust emission."""
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        logZ_gas = theta["gas_logz"]
        logU     = theta["gas_logu"]

        def nebular_one(logage, logQ):
            return self.neb.evaluate(
                logZ=logZ_gas, logU=logU, logage=logage, logQ=logQ,
            )

        neb_cont, neb_lines = vmap(
            lambda logQ_row: vmap(nebular_one)(self.ssp_ages_lgyr, logQ_row)
        )(self.logqq)

        neb_all = (neb_cont + neb_lines)[..., 0]                        # (n_z, n_age, n_wave)
        neb_all = jnp.where(self.young_mask[None, :, None], neb_all, 0.0)

        stellar_fluxes  = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)
        combined_fluxes = stellar_fluxes + neb_all                      # (n_z, n_age, n_wave)

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        M        = self._age_bin_mix.astype(attn.dtype)
        tau_age  = jnp.einsum("ab,bw->aw", M, attn)
        attn_age = jnp.exp(-tau_age)

        spectrum = jnp.einsum("za,zaw,aw->w", W, combined_fluxes, attn_age)
        spectrum = spectrum * jnp.exp(-attn_diffuse)

        self.total_weights      = W
        self.nebular_fluxes_all = neb_all
        self.stellar_fluxes     = stellar_fluxes
        self.spectrum           = spectrum.reshape((-1,))
        return self.spectrum

    def get_spectrum_dattn_dem_neb(self, theta):
        """Dust attenuation + nebular emission + dust emission."""
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        logZ_gas = theta["gas_logz"]
        logU     = theta["gas_logu"]

        def nebular_one(logage, logQ):
            return self.neb.evaluate(
                logZ=logZ_gas, logU=logU, logage=logage, logQ=logQ,
            )

        neb_cont, neb_lines = vmap(
            lambda logQ_row: vmap(nebular_one)(self.ssp_ages_lgyr, logQ_row)
        )(self.logqq)

        neb_all = (neb_cont + neb_lines)[..., 0]                        # (n_z, n_age, n_wave)
        neb_all = jnp.where(self.young_mask[None, :, None], neb_all, 0.0)

        stellar_fluxes  = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)
        combined_fluxes = stellar_fluxes + neb_all                      # (n_z, n_age, n_wave)

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        M             = self._age_bin_mix.astype(attn.dtype)
        tau_age       = jnp.einsum("ab,bw->aw", M, attn)
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse)

        spectrum_dust_free = jnp.einsum("za,zaw->w",       W, combined_fluxes)
        attenuated         = jnp.einsum("za,zaw,aw->w",    W, combined_fluxes, attn_age)
        attenuated         = attenuated * diffuse_curve
        self.spec_attn     = attenuated

        dust_emi_spectrum, self.mdust, self.tduste = self.dust_emi.compute_dust_emission(
            spec_attn     = self.spec_attn,
            spec_dustfree = spectrum_dust_free,
            spec_lambda   = self.wave,
            diffuse_curve = diffuse_curve,
            duste_qpah    = theta["duste_qpah"],
            duste_umin    = theta["duste_umin"],
            duste_gamma   = theta["duste_gamma"],
        )

        self.total_weights      = W
        self.nebular_fluxes_all = neb_all
        self.stellar_fluxes     = stellar_fluxes
        self.spectrum           = dust_emi_spectrum
        return self.spectrum

    def get_spectrum_dattn_nodem_noneb(self, theta):
        """Dust attenuation, no nebular, no dust emission."""
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M       = self._age_bin_mix.astype(attn.dtype)
        tau_age = jnp.einsum("ab,bw->aw", M, attn)
        attn_age= jnp.exp(-tau_age)

        weights  = self.calculate_ssp_weights(theta)
        spectrum = jnp.einsum("za,zaw,aw->w", weights, self.flux, attn_age)
        spectrum *= jnp.exp(-attn_diffuse)

        self.spectrum = spectrum.reshape((-1,))
        return self.spectrum

    def get_spectrum_dattn_dem_noneb(self, theta):
        """Dust attenuation + dust emission, no nebular."""
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M             = self._age_bin_mix.astype(attn.dtype)
        tau_age       = jnp.einsum("ab,bw->aw", M, attn)
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse)

        weights           = self.calculate_ssp_weights(theta)
        spectrum_dust_free= jnp.einsum("za,zaw->w", weights, self.flux)
        attenuated        = jnp.einsum("za,zaw,aw->w", weights, self.flux, attn_age)
        attenuated       *= diffuse_curve
        self.spec_attn    = attenuated

        dust_emi_spectrum, self.mdust, self.tduste = self.dust_emi.compute_dust_emission(
            spec_attn      = self.spec_attn,
            spec_dustfree  = spectrum_dust_free,
            spec_lambda    = self.wave,
            diffuse_curve  = diffuse_curve,
            duste_qpah     = theta["duste_qpah"],
            duste_umin     = theta["duste_umin"],
            duste_gamma    = theta["duste_gamma"],
        )
        self.spectrum = dust_emi_spectrum
        return self.spectrum

    def get_spectrum_nodattn_nodem_noneb(self, theta):
        """Stellar continuum only — no dust, no nebular."""
        weights       = self.calculate_ssp_weights(theta=theta)
        self.spectrum = jnp.einsum("za,zaw->w", weights, self.flux)
        return self.spectrum




    def get_spectrum_nodattn_nodem_neb(self, theta):
        """Nebular emission only, no dust."""
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        logZ_gas = theta["gas_logz"]
        logU = theta["gas_logu"]

        def nebular_one(logage, logQ):
            return self.neb.evaluate(
                logZ=logZ_gas,
                logU=logU,
                logage=logage,
                logQ=logQ,
            )

        # self.logqq assumed shape (n_z, n_age)
        neb_cont, neb_lines = vmap(
            lambda logQ_row: vmap(nebular_one)(self.ssp_ages_lgyr, logQ_row)
        )(self.logqq)

        neb_all = (neb_cont + neb_lines)[..., 0]   # (n_z, n_age, n_wave)

        # Optional: suppress nebular emission for old ages
        neb_all = jnp.where(self.young_mask[None, :, None], neb_all, 0.0)

        # Optional: suppress stellar ionizing part if needed
        stellar_fluxes = jnp.where(self.kill_ion[None, :, :], 0.0, self.flux)

        combined_fluxes = stellar_fluxes + neb_all
        spectrum = jnp.einsum("za,zaw->w", W, combined_fluxes)

        self.total_weights = W
        self.nebular_fluxes_all = neb_all
        self.stellar_fluxes = stellar_fluxes
        self.spectrum = spectrum
        return spectrum
