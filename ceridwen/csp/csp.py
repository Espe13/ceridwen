"""Composite stellar population basis with a dict-valued theta.

theta keys: "sfh" (linear SFR, per node (n_time,) or per bin (n_time-1,)),
"logzsol" or "logzsol_hist" (stellar metallicity log10(Z/Z_sun), Z_sun the SSP grid's own solar
node; = [Fe/H] on MIST / aMIST grids), "gas_logz" (log10(Z_gas/Z_sun,neb) of the CLOUDY grid),
dust / nebular parameters, and the runtime scalars logmass, zred, lumdist_mpc,
igm_factor, eline_scaling, frac_obrun, spectrum_scaling, spectrum_calib, plus the IGM model's
own keys (``IGMModel.param_names``; x_HI, logN_HI, z_dla for MadauDampingDLA).
"""

import math
import os
import warnings

import numpy as np
import jax.numpy as jnp
import pprint

from ceridwen.dust.DustModel import Dust, DiffuseDust
from ceridwen.dust.DustEmission import DustEmission
from ceridwen.neb.NebularGridModel            import NebularModel

tiny_number = 1e-70
LOG10E = math.log10(math.e)


def fnu2flam(lam, fnu):
    """f_nu [erg/s/cm^2/Hz] -> f_lambda [erg/s/cm^2/A]."""
    c = 2.998e18  # Å/s
    return c * (fnu / (lam ** 2))


def intsfwght(t_hi, t_lo, a, slope, logage):
    """Integral of a linearly interpolated SFH between log-time limits (piecewise-linear weights)."""

    def F(t):
        x = 10.0**t
        delta = logage - t
        return (
            a * x * (delta + LOG10E)
            + 0.5 * slope * x * x * (delta + 0.5 * LOG10E)
        )

    return F(t_hi) - F(t_lo)


_LEGACY_ISOC_TYPE = "mist"


def _resolve_isoc_type(recorded, user, *, default=_LEGACY_ISOC_TYPE):
    """Nebular isochrone type: the SSP grid's recorded one, else the caller's, else ``default`` (with a warning).
    A caller value that conflicts with the recorded one raises ValueError.
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


LOGZSOL_KEYS = ("logzsol", "logzsol_hist")
REMOVED_METALLICITY_KEYS = {"Z": "logzsol", "zh": "logzsol_hist"}
ABSOLUTE_LOOKING_LOGZSOL = -1.2   # logzsol below this looks like an old absolute log10 Z


def _log10_zsun_of(ssp):
    v = getattr(ssp, "log10_zsun", None)
    if v is None:
        raise TypeError(
            f"{type(ssp).__name__} carries no resolved solar metallicity (log10_zsun); "
            "build the grid with ceridwen.ssps.SSPData / SSPDataAfe (load, from_fsps), which "
            "resolve Z_sun from the file's provenance, the metadata table or zsun=")
    return float(v)


def _default_logzsol(ssp) -> float:
    """Neutral starting metallicity: the median native node converted to logzsol."""
    return float(np.median(np.asarray(ssp.ssp_lgmet, dtype=np.float64))) - _log10_zsun_of(ssp)


def removed_metallicity_key_error(key, value, log10_zsun, axis_meaning=None, where="theta"):
    """ValueError for the removed absolute-metallicity keys 'Z' / 'zh', with the converted value."""
    new = REMOVED_METALLICITY_KEYS[key]
    conv = ""
    if value is not None:
        try:
            v = np.atleast_1d(np.asarray(value, dtype=np.float64))
            c = v - float(log10_zsun)
            conv = (f" For this grid, {where}[{key!r}] = {np.array2string(v, precision=4)} "
                    f"(absolute log10 Z) is {where}[{new!r}] = "
                    f"{np.array2string(c, precision=4)}.")
        except (TypeError, ValueError):
            conv = ""
    feh = (" On this grid logzsol is [Fe/H]." if axis_meaning == "feh" else "")
    return ValueError(
        f"{where}[{key!r}] was removed in v1.0.5: the stellar metallicity is now "
        f"{where}[{new!r}] = log10(Z/Z_sun), with Z_sun the SSP grid's own solar node "
        f"(log10 Z_sun = {float(log10_zsun)!r}, so logzsol = log10 Z - ({float(log10_zsun):.6f})), "
        f"not the absolute log10 Z.{feh}{conv} Nothing is reinterpreted silently: rename the "
        f"key and convert the value (priors and free_param_init too).")


def prior_support(prior):
    """(lo, hi) floats of a prior's support (vector priors: min of lo, max of hi).

    ``bounds`` is a property on Uniform/TopHat/ClippedNormal but a method on
    Normal/StudentT/LogNormal (a pre-existing inconsistency in ceridwen.sampler.priors), so
    both spellings are accepted here."""
    b = prior.bounds
    lo, hi = b() if callable(b) else b
    return float(np.min(np.asarray(lo, dtype=np.float64))), float(np.max(np.asarray(hi, dtype=np.float64)))


def _check_duste_model(duste_model, add_dust_emission):
    """``duste_model`` validated at construction: 'DL07' or 'THEMIS', and only meaningful with
    ``add_dust_emission=True`` (a non-default choice without it would silently do nothing)."""
    if duste_model not in ("DL07", "THEMIS"):
        raise ValueError(f"duste_model must be 'DL07' or 'THEMIS', got {duste_model!r}")
    if duste_model != "DL07" and not add_dust_emission:
        raise ValueError(f"duste_model={duste_model!r} selects dust-emission templates, but "
                         "add_dust_emission=False; set add_dust_emission=True or drop duste_model")
    return duste_model


class CSPBasis:
    """Composite stellar population basis.  ``predict(theta, observations)`` projects the model onto observations.

    Parameters
    ----------
    SSPData : SSPData -- SSP grids (wave, flux, ages, metallicities, optional resolution curve).
    theta : dict -- initial values; must contain "lookback_time" (Gyr, increasing, index 0 = today,
        >= 2 nodes) and "sfh", plus "logzsol" (zh_const=True, shape (1,)) or "logzsol_hist"
        (zh_const=False, shape (n_time,)), both log10(Z/Z_sun) of the grid.  Mutually exclusive
        with ``lookback_time=``.
    lookback_time : array -- shortcut for ``theta``: the node grid only, neutral initial values
        (sfh = 1, metallicity = grid median).
    sfh_per_bin : bool -- with ``lookback_time=``, one SFR per bin (n_time-1,) instead of per node.
    zh_const : bool -- constant metallicity ("logzsol") or a history ("logzsol_hist", one per node).
    gas_tied : bool -- gas-phase metallicity follows the stellar one, gas_logz := logzsol (or
        logzsol_hist[0], today's value), as Prospector ties them; "gas_logz" is then not a
        parameter.  The two are the SAME NUMBER, not the same absolute Z: logzsol uses the SSP
        grid's Z_sun (e.g. 0.0185 for MIST) and gas_logz the CLOUDY grid's own reference
        (Byler+2017; its node set is 0.019- or 0.020-based), which is the FSPS/Prospector
        convention.
    add_neb, add_dust, add_diffuse_dust, add_dust_emission, add_igm : bool -- physics switches.
    duste_model : {'DL07', 'THEMIS'} -- dust-emission templates with ``add_dust_emission``:
        Draine & Li (2007) or THEMIS (Jones et al. 2013, 2017), both from $SPS_HOME/dust/dustem
        with FSPS's (qPAH, Umin) axes.  THEMIS's ``duste_qpah`` axis spans 0.91-18.2 (the FSPS
        mass-fraction nodes x 100/2.2), DL07's 0.47-4.58.
    sps_home : str -- data directory for the nebular and dust-emission grids; defaults to $SPS_HOME.
    init_neb_params, init_dust_params : dict -- forwarded to NebularModel / Dust.  ``isoc_type`` is
        taken from the SSP grid's provenance when recorded.
    sfh_interp : {'step', 'linear'} -- piecewise-constant (non-negative weights) or
        piecewise-linear (analytic log-age integral, small negative weights clipped) SFH.
    track_zred_age : bool -- with a sampled ``zred``, rescale the lookback grid so its oldest node
        is the age of the Universe at that redshift.
    fesc_geometry : {'runaway_bc', 'picket'} -- how the ``frac_obrun`` escape channel bypasses dust.
    cosmo : Cosmology -- required; used for the flux factor and for the age of the Universe.
    nebemlineinspec : bool -- default of ``include_lines`` in ``get_spectrum`` only.
    """

    def __init__(
        self,
        SSPData,
        theta=None,
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
        track_zred_age=False,
        lookback_time=None,
        sfh_per_bin=False,
        fesc_geometry="runaway_bc",
        duste_model="DL07",
        cosmo=None,
        gas_tied=False,
        **kwargs,
    ):
        self.verbose = bool(verbose)
        self.gas_tied = bool(gas_tied)
        if self.gas_tied and not add_neb:
            raise ValueError("gas_tied=True ties the nebular gas metallicity to the stars, "
                             "but add_neb=False: there is no nebular model to tie")
        if kwargs:
            hint = ""
            if "sigma_losvd_kms" in kwargs:
                hint = (" ('sigma_losvd_kms' was removed: the galaxy velocity "
                        "dispersion is set once on the model, "
                        "SedModel(kinematics=Kinematics(sigma_gal=...)))")
            elif "tuniv" in kwargs:
                hint = (" ('tuniv' was removed: the age of the Universe comes from "
                        "the cosmology, csp.age_at(z) / cosmo.age(z))")
            raise TypeError(
                f"CSPBasis got unexpected keyword argument(s) {sorted(kwargs)}{hint}")
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
            _lz_mid = _default_logzsol(SSPData)
            if zh_const:
                theta['logzsol'] = jnp.array([_lz_mid])
            else:
                theta['logzsol_hist'] = jnp.full((_n,), _lz_mid)
        if theta is None:
            raise ValueError(
                "CSPBasis needs the static SFH grid structure. Pass either\n"
                "  lookback_time=jnp.linspace(0.0, T_oldest, n_nodes)   "
                "(shortcut; neutral initial values), or\n"
                "  theta={'lookback_time': ..., 'sfh': ..., 'logzsol' or 'logzsol_hist': ...} "
                "(full control).\n"
                "lookback_time is in Gyr, monotonically increasing, index 0 = "
                "today, >= 2 nodes."
            )
        if init_neb_params is None:
            init_neb_params = {"cloudy_dust": True}
        if init_dust_params is None:
            init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 'laws': ['powerlaw']}

        self.flux      = jnp.array(SSPData.ssp_flux, dtype=jnp.float32)  # (n_z, n_age, n_wave)
        self.wave      = jnp.array(SSPData.ssp_wave)       # (n_wave,)
        self.ages      = jnp.array(SSPData.ssp_lg_age_gyr) # (n_age,)  log10(Gyr)
        self._setup_metallicity(SSPData)                   # self.zmet: logzsol axis
        self.lib_resolution = (
            (np.asarray(SSPData.ssp_wave, dtype=np.float64),
             np.asarray(SSPData.ssp_resolution, dtype=np.float64))
            if getattr(SSPData, "ssp_resolution", None) is not None else None)
        self.ssp_ages_lgyr = self.ages + 9                 # log10(yr)

        self._ssp_isoc_type    = getattr(SSPData, "isoc_type", None)
        self._ssp_spec_library = getattr(SSPData, "spec_library", None)

        if fesc_geometry not in ("runaway_bc", "picket"):
            raise ValueError(
                f"fesc_geometry must be 'runaway_bc' or 'picket', got {fesc_geometry!r}")
        self.fesc_geometry = str(fesc_geometry)

        self._logage_lo  = self.ssp_ages_lgyr[1:]
        self._logage_hi  = self.ssp_ages_lgyr[:-1]
        self._dlogage    = jnp.diff(self.ssp_ages_lgyr)
        self._j_range    = jnp.arange(self.ssp_ages_lgyr.size)
        self._age_clip_lo = 10.0 ** (-70)                  # floor for log-time clipping
        self._age_clip_hi = 10.0 ** self.ssp_ages_lgyr[-1] # ceiling
        self._n_z   = len(self.zmet)
        self._n_age = len(self.ages)

        self._ssp_lo_yr = 10.0 ** self._logage_hi   # (n_age-1,)
        self._ssp_hi_yr = 10.0 ** self._logage_lo   # (n_age-1,)

        _ssp_age_yr  = 10.0 ** self.ssp_ages_lgyr           # (n_age,) linear yr
        _voro_mid    = 0.5 * (_ssp_age_yr[:-1] + _ssp_age_yr[1:])  # (n_age-1,)
        _voro_hi_ext = _ssp_age_yr[-1] + (_ssp_age_yr[-1] - _ssp_age_yr[-2])
        self._ssp_voronoi_lo = jnp.concatenate(
            [jnp.zeros(1), _voro_mid]
        )
        self._ssp_voronoi_hi = jnp.concatenate(
            [_voro_mid, jnp.array([_voro_hi_ext])]
        )

        self.tiny_logt  = tiny_logt
        if sps_home is None:
            sps_home = os.environ.get("SPS_HOME")
        if (add_neb or add_dust_emission) and not sps_home:
            raise ValueError(
                "sps_home is required for nebular / dust emission but was not "
                "given and $SPS_HOME is unset. Set `export SPS_HOME=/path/to/fsps` "
                "(your FSPS data directory) or pass sps_home=... explicitly."
            )
        self.sps_home   = sps_home
        self.duste_model = _check_duste_model(duste_model, add_dust_emission)
        from ..cosmology import Cosmology as _Cosmology
        if cosmo is None:
            raise TypeError(
                f"{type(self).__name__} needs an explicit cosmology: pass "
                "cosmo=Cosmology.planck18(), Cosmology.wmap9(), "
                "Cosmology.flat(H0, Om0), Cosmology.from_name(...) or "
                "Cosmology.from_astropy(...)  (from ceridwen import Cosmology)")
        if not isinstance(cosmo, _Cosmology):
            raise TypeError(
                f"cosmo must be a ceridwen Cosmology, got {type(cosmo).__name__}; "
                "for an astropy cosmology use Cosmology.from_astropy(...)")
        self._cosmo = cosmo
        self.track_zred_age = bool(track_zred_age)
        self.nebemlineinspec = bool(nebemlineinspec)

        self._setup_igm(add_igm, igm_model, igm_factor)

        if add_diffuse_dust or add_dust:
            self.set_attenuation_function(add_diffuse_dust, add_dust)

        theta = self.initialize_dust_components(
            add_dust, add_diffuse_dust, add_dust_emission,
            theta, init_dust_params, diffuse_law, sps_home,
        )
        theta = self.initialize_neb(add_neb, theta, init_neb_params, sps_home)

        self.configure_spectrum_model(
            add_dust, add_diffuse_dust, add_dust_emission, add_neb, sps_home
        )

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

        self.check_param_ranges(self.theta_init)


    def _setup_igm(self, add_igm, igm_model, igm_factor):
        """``self.igm`` (an ``IGMModel`` or None) and the default ``igm_factor``; a model that takes
        a cosmology (``bind_cosmology``) is given the CSP's, so the two cannot disagree."""
        if add_igm:
            from ..igm import make_igm_model
            self.igm = make_igm_model(igm_model)
            if hasattr(self.igm, "bind_cosmology"):
                self.igm.bind_cosmology(self._cosmo)
        else:
            self.igm = None
        self.igm_factor = float(igm_factor)

    def _igm_transmission(self, z_scalar, theta):
        """IGM transmission on ``self.wave`` at ``z_scalar``: ``igm_factor`` from theta or the
        constructor, plus the model's own theta keys (``IGMModel.param_names``) when it has any."""
        if "igm_factor" in theta:
            ig_factor = jnp.ravel(theta["igm_factor"])[0]
        else:
            ig_factor = jnp.float32(self.igm_factor)
        names = getattr(self.igm, "param_names", ())
        if not names:
            return self.igm.attenuation(self.wave, z_scalar, factor=ig_factor)
        params = {k: jnp.ravel(theta[k])[0] for k in names if k in theta}
        return self.igm.attenuation(self.wave, z_scalar, factor=ig_factor, params=params)

    def _setup_metallicity(self, ssp):
        """The single metallicity conversion: ``self.zmet`` = ssp_lgmet - log10_zsun (float64,
        the logzsol axis theta is looked up on); native axis and Z_sun kept for display/provenance."""
        self.log10_zsun   = _log10_zsun_of(ssp)
        self.zsun_nominal = getattr(ssp, "zsun_nominal", None)
        self.zsun_source  = getattr(ssp, "zsun_source", None)
        self.axis_meaning = getattr(ssp, "axis_meaning", None)
        self.grid_chash   = getattr(ssp, "chash", None)
        self._refused_cells = tuple(getattr(ssp, "refused_cells", ()) or ())
        self._refused_reason = getattr(ssp, "refused_reason", None)
        self.zmet_native  = jnp.asarray(np.asarray(ssp.ssp_lgmet, dtype=np.float64))
        self.zmet         = self.zmet_native - self.log10_zsun      # (n_z,) logzsol
        self.zlegend      = 10 ** self.zmet_native                  # native linear Z (FSPS label)

    def _gas_logz(self, theta):
        """Gas-phase metallicity log10(Z_gas/Z_sun,neb): theta['gas_logz'], or the stellar
        logzsol (today's value for a history) when ``gas_tied`` (a static branch).

        The tie sets the same NUMBER on two axes with different solar references (the SSP
        grid's Z_sun and the CLOUDY grid's); that is what FSPS/Prospector mean by tying the
        gas to the stars, and it is exact only in the sense of "equally solar"."""
        if not self.gas_tied:
            return theta["gas_logz"]
        if self.zh_const:
            return theta["logzsol"]
        return jnp.ravel(theta["logzsol_hist"])[0:1]

    def initialize_model_structure(self, theta):
        """Validate ``theta`` (grid, sfh shape, metallicity key) and build ``theta_init`` / ``param_names``."""
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

        self.sfh_times = jnp.atleast_1d(
            jnp.asarray(theta['lookback_time'], dtype=float)
        ) * 1e9   # Gyr → yr
        self.n_time = self.sfh_times.size

        if self.n_time < 2:
            raise ValueError(
                f"theta['lookback_time'] has {self.n_time} node(s); at least "
                "2 are required (n_time nodes define n_time-1 SFH bins). "
                "Typical fits use 5-10 nodes, e.g. "
                "jnp.linspace(0.0, T_UNIV, 6)."
            )

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
                "and reverse theta['sfh'] (and theta['logzsol_hist'] if present) to match."
            )

        sfh = jnp.atleast_1d(jnp.asarray(theta['sfh'], dtype=float))
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

        for old in REMOVED_METALLICITY_KEYS:
            if old in theta:
                raise removed_metallicity_key_error(old, theta[old], self.log10_zsun,
                                                    self.axis_meaning)
        if self.zh_const:
            if 'logzsol' not in theta:
                raise ValueError(
                    "zh_const=True requires a constant metallicity theta['logzsol'] "
                    "(shape-(1,) array, log10(Z/Z_sun) of the SSP grid, e.g. 0.0 = solar); "
                    "none was provided. Either add theta['logzsol'], or construct with "
                    "zh_const=False and provide a history theta['logzsol_hist'] of shape "
                    "(n_time,)."
                )
            if 'logzsol_hist' in theta:
                raise ValueError(
                    "zh_const=True but theta also contains 'logzsol_hist'; a constant-"
                    "metallicity basis reads only 'logzsol'. Remove 'logzsol_hist' or "
                    "construct with zh_const=False.")
        else:
            if 'logzsol_hist' not in theta:
                raise ValueError(
                    "zh_const=False requires a metallicity history theta['logzsol_hist'] "
                    "of shape (n_time,) (log10(Z/Z_sun) of the SSP grid, one per lookback "
                    "node, index 0 = today); none was provided. Either add it, or "
                    "construct with zh_const=True and provide theta['logzsol']."
                )
            if 'logzsol' in theta:
                raise ValueError(
                    "zh_const=False but theta also contains 'logzsol'; a metallicity-"
                    "history basis reads only 'logzsol_hist'. Remove 'logzsol' or "
                    "construct with zh_const=True.")
        if self.gas_tied and 'gas_logz' in theta:
            raise ValueError(
                "gas_tied=True sets gas_logz from the stellar logzsol, but theta also "
                "gives 'gas_logz'; remove one (a tied gas metallicity is not a parameter).")

        self.zh_is_scalar = None
        if 'logzsol_hist' in theta:
            zh = jnp.atleast_1d(jnp.asarray(theta['logzsol_hist'], dtype=float))
            assert zh.shape == (self.n_time,), "'logzsol_hist' must match 'lookback_time' length"
            self.zh_is_scalar = False
        elif 'logzsol' in theta:
            Z = jnp.atleast_1d(jnp.asarray(theta['logzsol'], dtype=float))
            assert Z.shape == (1,), "'logzsol' must be a scalar (wrapped in shape-(1,) array)"
            self.zh_is_scalar = True


        self.theta_init = {}
        for k, v in theta.items():
            if k == 'lookback_time':
                continue   # static grid — not a free parameter
            arr = jnp.atleast_1d(jnp.asarray(v, dtype=float))
            self.theta_init[k] = arr

        self.theta_init['sfh'] = sfh

        self.param_names = list(self.theta_init.keys())

        self._known_theta_keys = set(self.param_names) | {
            'lookback_time', 'logzsol', 'logzsol_hist',
            'logmass', 'zred', 'lumdist_mpc', 'igm_factor', 'eline_scaling',
            'frac_obrun', 'spectrum_scaling', 'spectrum_calib',
        } | set(getattr(getattr(self, "igm", None), "param_names", ()))


    def register_known_theta_keys(self, keys):
        """Add keys that ``_warn_unknown_theta_keys`` must accept (model-level parameters consumed by transforms)."""
        self._known_theta_keys |= set(keys)

    def _warn_unknown_theta_keys(self, theta):
        """Warn on theta keys nothing consumes (static dict keys; runs once at trace time)."""
        for old in REMOVED_METALLICITY_KEYS:
            if old in theta:
                raise removed_metallicity_key_error(old, None, self.log10_zsun, self.axis_meaning)
        unknown = [k for k in theta if k not in self._known_theta_keys]
        if unknown:
            warnings.warn(
                f"CSPBasis received unrecognized theta key(s) {sorted(unknown)} "
                f"which are SILENTLY IGNORED (likely a typo). Recognized keys: "
                f"{sorted(self._known_theta_keys)}.",
                stacklevel=3,
            )

    def check_param_ranges(self, theta=None, warn=True):
        """Messages (and warnings) for metallicity / nebular parameters outside the interpolation grids, where the model clamps silently."""
        if theta is None:
            theta = self.theta_init
        msgs = []

        for old in REMOVED_METALLICITY_KEYS:
            if old in theta:
                raise removed_metallicity_key_error(old, theta[old], self.log10_zsun,
                                                    self.axis_meaning)
        zlo, zhi = float(self.zmet.min()), float(self.zmet.max())
        zs = (f"Z_sun = {self.zsun_nominal:.6g}, log10 Z_sun = {self.log10_zsun:.6f}"
              if self.zsun_nominal is not None else f"log10 Z_sun = {self.log10_zsun:.6f}")
        for key in LOGZSOL_KEYS:
            if key in theta:
                v = np.asarray(theta[key], float)
                if v.size and (np.nanmin(v) < zlo or np.nanmax(v) > zhi):
                    msgs.append(
                        f"theta['{key}'] = {np.array2string(v, precision=3)} is outside the "
                        f"SSP metallicity grid, logzsol in [{zlo:+.3f}, {zhi:+.3f}] "
                        f"({zs}); the interpolation clamps to the edge node there."
                    )
                if v.size and np.nanmax(v) < ABSOLUTE_LOOKING_LOGZSOL:
                    msgs.append(
                        f"theta['{key}'] = {np.array2string(v, precision=3)} is below "
                        f"{ABSOLUTE_LOOKING_LOGZSOL} everywhere: is it an OLD absolute log10 Z? "
                        f"logzsol = log10(Z/Z_sun) is 0 at solar on this grid ({zs}); an "
                        f"absolute log10 Z = {float(np.nanmax(v)):+.3f} would be logzsol = "
                        f"{float(np.nanmax(v)) - self.log10_zsun:+.3f}.  Ignore this if a "
                        "metal-poor population is intended."
                    )

        neb = getattr(self, 'neb', None)
        if neb is not None:
            for key, attrs in (
                ('gas_logz', ('logZ_grid', 'logz_grid', '_logZ', 'nebem_logz')),
                ('gas_logu', ('logU_grid', 'logu_grid', '_logU', 'nebem_logu')),
            ):
                if key == 'gas_logz' and self.gas_tied:
                    src = 'logzsol' if self.zh_const else 'logzsol_hist'
                    if src not in theta:
                        continue
                    g = np.asarray(neb.nebem_logz, float)
                    v = np.asarray(theta[src], float)
                    v = v if self.zh_const else v[:1]
                    if v.size and (np.nanmin(v) < g.min() or np.nanmax(v) > g.max()):
                        msgs.append(
                            f"gas_tied=True: the gas metallicity follows theta['{src}'] = "
                            f"{np.array2string(v, precision=3)}, outside the nebular grid "
                            f"[{g.min():.3f}, {g.max():.3f}] (log10 Z_gas/Z_sun of the "
                            "CLOUDY grid); the nebular emission clamps there.")
                    continue
                if key in theta:
                    grid = next((getattr(neb, a) for a in attrs if hasattr(neb, a)),
                                None)
                    if grid is not None:
                        g = np.asarray(grid, float)
                        glo, ghi = float(g.min()), float(g.max())
                        v = np.asarray(theta[key], float)
                        if v.size and (np.nanmin(v) < glo or np.nanmax(v) > ghi):
                            msgs.append(
                                f"theta['{key}'] = {np.array2string(v, precision=3)} outside "
                                f"the nebular grid [{glo:.3f}, {ghi:.3f}]"
                                + (" (log10 Z_gas/Z_sun of the CLOUDY grid)"
                                   if key == 'gas_logz' else "")
                                + "; the nebular emission clamps there."
                            )

        if warn:
            for m in msgs:
                warnings.warn(m, stacklevel=2)
        return msgs


    def set_attenuation_function(self, add_diffuse_dust, add_dust):
        """Assign ``self.attenuate_dust(wave, theta) -> (attn_birthcloud (n_bins, n_wave), attn_diffuse (n_wave,))`` as optical depths."""
        if add_diffuse_dust and add_dust:
            def attenuate(wave, theta):
                attn         = self.dust_attn.compute_attenuation(wave, theta)
                attn_diffuse = self.diff_dust.compute_attenuation(wave, theta)
                return attn, attn_diffuse
            if self.verbose:
                print("Using combined (binwise + diffuse) dust attenuation.")
            self.attenuate_dust = attenuate

        elif add_dust and not add_diffuse_dust:
            def attenuate_without_diffuse(wave, theta):
                attn         = self.dust_attn.compute_attenuation(wave, theta)
                attn_diffuse = jnp.zeros((wave.shape[0],))
                return attn, attn_diffuse
            if self.verbose:
                print("Using only binwise dust attenuation.")
            self.attenuate_dust = attenuate_without_diffuse

        elif add_diffuse_dust and not add_dust:
            self.bin_low  = jnp.array([-jnp.inf])
            self.bin_high = jnp.array([jnp.inf])
            def attenuate_diffuse_only(wave, theta):
                attn_diffuse = self.diff_dust.compute_attenuation(wave, theta)
                attn         = jnp.zeros((1, wave.shape[0]))
                return attn, attn_diffuse
            if self.verbose:
                print("Using only diffuse dust attenuation.")
            self.attenuate_dust = attenuate_diffuse_only

    def initialize_neb(self, add_neb, theta, init_neb_params, sps_home):
        if add_neb:
            match_fsps = init_neb_params.pop('match_fsps', False)
            if match_fsps:
                warnings.warn(
                    "match_fsps=True is obsolete and ignored: NebularModelFSPSMatch "
                    "has been removed (upstream FSPS was fixed and now matches the "
                    "strict NebularModel). Remove match_fsps from your call.",
                    stacklevel=2,
                )

            init_neb_params['isoc_type'] = _resolve_isoc_type(
                self._ssp_isoc_type, init_neb_params.get('isoc_type'),
            )

            init_neb_params.update({
                'sps_home':       sps_home,
                'csp_lambda':     self.wave,
                'ssp_flux':       self.flux,
                'ssp_ages_lgyr':  self.ssp_ages_lgyr,
            })
            if self.verbose:
                print(f"Initializing Nebular Emission model (NebularModel, "
                      f"isoc_type={init_neb_params['isoc_type']!r})...")
            self.neb = NebularModel(**init_neb_params)

            neb_defaults = self.neb.get_default_params()
            if self.gas_tied:
                neb_defaults.pop('gas_logz', None)   # follows logzsol; not a parameter
            for k, v in neb_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.neb_param_names = list(neb_defaults.keys())

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
            if self.verbose:
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
            if self.verbose:
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
            if self.verbose:
                print("Initializing DustEmission model...")
            self.dust_emi = DustEmission(duste_model=getattr(self, "duste_model", "DL07"),
                                         spec_lambda=self.wave, dust_file=sps_home)

            emi_defaults = self.dust_emi.get_default_params()
            for k, v in emi_defaults.items():
                if k not in theta:
                    theta[k] = v
            self.emi_param_names = list(emi_defaults.keys())

        return theta

    def _init_age_bin_operator(self):
        ages = self.ages
        lo   = self.bin_low
        hi   = self.bin_high

        in_bin  = (ages[:, None] >= lo[None, :]) & (ages[:, None] < hi[None, :])
        M       = in_bin.astype(jnp.float32)
        row_sum = jnp.sum(M, axis=1, keepdims=True)
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
        if self.verbose:
            print(f"Spectrum model: {label[key]}")
        self.get_spectrum = mapping[key]


    def get_spectrum_components(self, theta: dict) -> tuple:
        """``(continuum, lines)`` on the rest-frame grid, unscaled (no mass, distance or IGM):
        ``continuum`` is ``get_spectrum(include_lines=False)`` and ``lines`` the difference to the
        full spectrum (the painted lines; with dust emission also their re-emitted energy).
        """
        self._warn_unknown_theta_keys(theta)
        continuum = self.get_spectrum(theta=theta, include_lines=False)
        full      = self.get_spectrum(theta=theta, include_lines=True)
        return continuum, full - continuum

    def predict(self, theta: dict, observations: list, *, kinematics=None,
                broaden_photometry=False, eline_system=None) -> dict:
        """``{obs.name: prediction}``: Photometry -> maggies (n_filters,), Spectrum -> F_nu [erg/s/cm^2/Hz]
        on the observed pixels, Lines -> integrated fluxes (n_lines,).  Observed-frame only when
        ``theta`` carries ``zred``.

        ``kinematics`` (``Kinematics`` or None) supplies sigma_gal / sigma_gas; ``broaden_photometry``
        applies them to the spectrum entering each Photometry's ``_broadener``.  Lines are painted on
        the model grid only when a consumer needs them there (free-z Photometry, Photometry with a
        sampled sigma_gas and ``broaden_photometry``, or ``_force_paint_lines``); otherwise fixed-z
        Photometry adds them through the static basis ``G = obs._T @ (IGM * profiles)``, Spectrum
        paints them on the observed pixels and Lines reads them from the grid.

        ``eline_system`` (``ElineSystem`` of ``SedModel``, line marginalisation): returns
        ``(predictions, aux)`` with the fitted lines removed from every prediction and
        ``aux = {"prior_mean": CLOUDY fluxes of the fitted lines, "cols": {obs.name: design}}``.
        """
        from ..observation.observation import (
            Spectrum as _Spectrum, Photometry as _Photometry,
        )
        if kinematics is not None:
            self._known_theta_keys |= set(kinematics.free_keys)
        has_neb = getattr(self, "neb", None) is not None
        gas_free = (kinematics is not None
                    and isinstance(kinematics.effective_sigma_gas, str))
        paint_lines = has_neb and (
            getattr(self, "_force_paint_lines", False)
            or any(
                isinstance(o, _Photometry)
                and ((getattr(o, "free_z", False) and "zred" in theta)
                     or (gas_free and broaden_photometry))
                for o in observations
            )
        )
        cont, lines = self._assemble_components(theta, paint_lines)
        cont, lines_s = self._apply_mass_redshift_igm(
            cont, cont if lines is None else lines, theta)
        return self._project_observations(
            cont if lines is None else cont + lines_s, cont,
            observations, theta, paint_lines=paint_lines,
            line_component=None if lines is None else lines_s,
            kinematics=kinematics, broaden_photometry=broaden_photometry,
            eline_system=eline_system,
        )

    def _assemble_components(self, theta, paint_lines):
        """``(continuum, lines or None)`` on the rest-frame grid, unscaled; ``lines`` is the painted
        line component (``get_spectrum_components``) or None when not painted."""
        if not paint_lines:
            self._warn_unknown_theta_keys(theta)
            return self.get_spectrum(theta=theta, include_lines=False), None
        return self.get_spectrum_components(theta)

    def _assemble_observer_spectra(self, theta, *, paint_lines=True):
        """``(continuum + lines, continuum)``; the lines term is absent when ``paint_lines=False``."""
        cont, lines = self._assemble_components(theta, paint_lines)
        return (cont if lines is None else cont + lines), cont

    def _apply_mass_redshift_igm(self, spectrum_phot, spectrum_slit, theta):
        """Multiply both spectra by 10**logmass, the cgs flux factor and the IGM transmission, each only when its key is present."""
        if "logmass" in theta:
            mass_scale = jnp.float32(10.0 ** theta["logmass"][0])
            spectrum_phot = spectrum_phot * mass_scale
            spectrum_slit = spectrum_slit * mass_scale

        if "zred" in theta:
            z_scalar = jnp.ravel(theta["zred"])[0]
            ff = jnp.float32(self._flux_factor(theta))
            spectrum_phot = spectrum_phot * ff
            spectrum_slit = spectrum_slit * ff
            if self.igm is not None:
                transmission = self._igm_transmission(
                    z_scalar, theta).astype(spectrum_phot.dtype)
                spectrum_phot = spectrum_phot * transmission
                spectrum_slit = spectrum_slit * transmission

        return spectrum_phot, spectrum_slit

    def _project_observations(self, spectrum_phot, spectrum_slit,
                              observations, theta, *, paint_lines=True,
                              line_component=None, kinematics=None,
                              broaden_photometry=False, eline_system=None):
        """``{obs.name: prediction}`` from the scaled observer-frame spectra: ``spectrum_phot`` is the
        continuum (+ painted lines), ``spectrum_slit`` the continuum alone (Spectrum projector input),
        ``line_component`` the painted lines alone (or None).
        """
        from ..observation.observation import (
            Photometry as _Photometry,
            Spectrum   as _Spectrum,
        )
        from ..observation.lines import Lines as _Lines
        from ..broadening import TIED
        _has_lines_obs = any(isinstance(o, _Lines) for o in observations)
        _has_spec_obs = any(isinstance(o, _Spectrum) for o in observations)
        _has_phot_obs = any(isinstance(o, _Photometry) for o in observations)
        _has_neb = getattr(self, "neb", None) is not None
        if _has_lines_obs and not _has_neb:
            raise ValueError(
                "this CSP has no nebular model (add_neb=False), so a Lines "
                "observation would receive identically zero line fluxes; build "
                "the basis with add_neb=True or remove the Lines observation")
        _line_fluxes = (self.predict_line_fluxes(theta)
                        if _has_lines_obs and _has_neb else None)
        _line_fluxes_spec = (self.predict_line_fluxes(theta, for_spectrum=True)
                             if _has_spec_obs and _has_neb else None)
        _line_fluxes_phot = (
            self.predict_line_fluxes(theta, for_photometry=True)
            if (not paint_lines) and _has_neb and _has_phot_obs
            else None)
        es = eline_system
        aux_cols = {}
        if es is not None:
            if paint_lines:
                raise ValueError(
                    "line marginalisation needs the static photometric line basis, but this "
                    "prediction paints the lines on the model grid (free-z Photometry, a sampled "
                    "sigma_gas with broaden_photometry, or _force_paint_lines)")
            keep = jnp.asarray(es.keep_grid)
            if _line_fluxes_spec is None:          # no nebular grid (CSPBasis_afe): flat prior only
                prior_mean = jnp.zeros(es.m)
            else:
                prior_mean = _line_fluxes_spec[es.fit_rows]
                _line_fluxes_spec = _line_fluxes_spec * keep
            if _line_fluxes is not None:
                _line_fluxes = _line_fluxes * keep
            if _line_fluxes_phot is not None:
                _line_fluxes_phot = _line_fluxes_phot * keep
        s_gal = s_gas = None
        gas_untied = False
        if kinematics is not None and broaden_photometry and _has_phot_obs:
            s_gal, s_gas = kinematics.resolve(theta)
            gas_untied = kinematics.sigma_gas is not TIED
        out = {}
        z_in_theta = "zred" in theta
        from .spectrum_calibration import spectrum_calibration_factor as _spec_calib
        for obs in observations:
            if isinstance(obs, _Lines):
                _B = self._neb_blend_matrix_for(obs)
                if _B is not None:
                    out[obs.name] = _B @ _line_fluxes
                else:
                    out[obs.name] = _line_fluxes[self._neb_cube_rows_for(obs)]
                if es is not None and obs.name in es.lines_cols:
                    scale = (jnp.ravel(theta["eline_scaling"])[0]
                             if "eline_scaling" in theta else 1.0)
                    aux_cols[obs.name] = jnp.asarray(es.lines_cols[obs.name]) * scale
                continue
            if isinstance(obs, _Spectrum):
                A = None
                if es is not None and obs.name == es.spec_key:
                    pred, A = obs._proj.predict_with_line_basis(
                        spectrum_slit, _line_fluxes_spec, theta, es.fit_pos,
                        basis=None if es.static is None else es.static["basis"])
                elif "eline_delta_zred" in theta and getattr(obs, "_proj", None) is not None:
                    pred, _ = obs._proj.predict_with_line_basis(
                        spectrum_slit, _line_fluxes_spec, theta)
                else:
                    pred = obs.predict(spectrum_slit, self.wave, _line_fluxes_spec, theta)
                calib = _spec_calib(obs, theta, dtype=pred.dtype)
                if calib is not None:
                    pred = pred * calib
                    if A is not None:
                        A = A * (calib[:, None] if jnp.ndim(calib) else calib)
                if A is not None:
                    aux_cols[obs.name] = A
                out[obs.name] = pred
                continue
            spec_for_obs = spectrum_phot
            pb = getattr(obs, "_broadener", None)
            if pb is not None and s_gal is not None:
                if line_component is not None and gas_untied:
                    spec_for_obs = pb(spectrum_slit, s_gal) + pb(line_component, s_gas)
                else:
                    spec_for_obs = pb(spectrum_phot, s_gal)
            if getattr(obs, "free_z", False) and z_in_theta:
                out[obs.name] = obs.predict_at_redshift(
                    spec_for_obs, self.wave, jnp.ravel(theta["zred"])[0]
                )
                continue
            pred = obs.predict(spec_for_obs, self.wave)
            if _line_fluxes_phot is not None:
                basis = getattr(obs, "_line_basis", None)
                gnb = (jnp.asarray(basis) if basis is not None
                       else self.neb.gaussnebarr).astype(obs._T.dtype)
                if self.igm is not None and z_in_theta:
                    z_scalar = jnp.ravel(theta["zred"])[0]
                    trans = self._igm_transmission(z_scalar, theta).astype(obs._T.dtype)
                    G = (obs._T * trans[None, :]) @ gnb
                else:
                    G = obs._T @ gnb
                pred = pred + G @ _line_fluxes_phot.astype(pred.dtype)
            if es is not None and obs.name in es.phot_cols:
                aux_cols[obs.name] = jnp.asarray(es.phot_cols[obs.name])
            out[obs.name] = pred
        if es is not None:
            return out, {"prior_mean": prior_mean, "cols": aux_cols}
        return out

    def _neb_cube_rows_for(self, obs):
        """Nebular-cube row of every line of a Lines observation, matched by rest wavelength (1 A
        tolerance) and cached on the observation as a NumPy int array.
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
                f"{(getattr(obs, 'line_names', None) or ['?'] * len(lam))[worst]!r} at "
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
        obs._neb_cube_rows = np.asarray(idx, dtype=np.int32)
        return obs._neb_cube_rows

    def _neb_blend_matrix_for(self, obs):
        """(n_obs, n_grid) 0/1 summation matrix for blended lines (``Lines.components``), or None when
        every observed line is a single grid line; cached on the observation as NumPy.
        """
        if hasattr(obs, "_neb_blend_matrix"):
            return obs._neb_blend_matrix
        comps = getattr(obs, "line_components", None)
        if comps is None or not any(len(c) > 1 for c in comps):
            obs._neb_blend_matrix = None
            return None
        pos = np.asarray(self.neb.nebem_line_pos, dtype=float)
        names = getattr(obs, "line_names", None) or ["?"] * len(comps)
        B = np.zeros((len(comps), pos.size), dtype=np.float32)
        for k, comp in enumerate(comps):
            for lam in comp:
                j = int(np.argmin(np.abs(pos - float(lam))))
                if abs(pos[j] - float(lam)) > 1.0:
                    raise ValueError(
                        "Emission-line blend matching failed: component at "
                        f"{float(lam):.2f} A of observed line {names[k]!r} has "
                        f"no nebular-cube line within 1 A (nearest "
                        f"{pos[j]:.2f} A).")
                B[k, j] += 1.0
        obs._neb_blend_matrix = np.asarray(B, dtype=np.float32)
        return obs._neb_blend_matrix

    def predict_line_fluxes(self, theta, *, for_photometry=False, for_spectrum=False):
        """Observed-frame fluxes of every nebular grid line (n_lines,), through the same weights,
        dust, escape, mass and distance factors as the spectrum.

        Default: integrated fluxes [erg/s/cm^2] with the 1/(1+z) Jacobian, IGM at the line
        wavelength and ``eline_scaling`` (the Lines observation).  ``for_spectrum``: the same
        without ``eline_scaling`` (painted by the Spectrum projector).  ``for_photometry``: per-Hz
        amplitudes with the full f_nu flux factor and no IGM (applied inside the photometric line
        basis) and no ``eline_scaling``.
        """
        W = self.calculate_ssp_weights(theta=theta)          # (n_z, n_age)
        logZ_gas = self._gas_logz(theta)
        logU     = theta["gas_logu"]
        line_lum = self.neb.evaluate_batch_line_lum(
            logZ_gas, logU, self._neb_ages_young, self._neb_logqq_young,
        )                                                    # (n_z, n_young, n_lines)
        if "frac_obrun" in theta:
            f_esc = jnp.ravel(theta["frac_obrun"])[0]
            line_lum = line_lum * (1.0 - f_esc)

        lam = self.neb.nebem_line_pos                        # (n_lines,)
        li = jnp.clip(jnp.searchsorted(self.wave, lam) - 1,
                      0, self.wave.shape[0] - 2)
        lf = jnp.clip((lam - self.wave[li])
                      / (self.wave[li + 1] - self.wave[li]), 0.0, 1.0)

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
                if "frac_obrun" in theta and self.fesc_geometry != "picket":
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
            z_scalar = jnp.ravel(theta["zred"])[0]
            ff = self._flux_factor(theta)
            F = F * (ff if for_photometry else ff / (1.0 + z_scalar))
            if self.igm is not None and not for_photometry:
                trans = self._igm_transmission(z_scalar, theta)
                F = F * ((1.0 - lf) * trans[li] + lf * trans[li + 1])
        if not for_photometry and not for_spectrum and "eline_scaling" in theta:
            F = F * jnp.ravel(theta["eline_scaling"])[0]
        return F


    def get_line_spec(self, theta):
        """Painted line component alone on the rest-frame grid, scaled by mass, flux factor and IGM as in
        ``predict``, times ``eline_scaling`` (the Lines-observation aperture factor)."""
        if not hasattr(self, "neb") or self.neb is None:
            return jnp.zeros_like(self.wave)

        _continuum, line_only = self.get_spectrum_components(theta)

        if "logmass" in theta:
            mass_scale = jnp.float32(10.0 ** theta["logmass"][0])
            line_only = line_only * mass_scale
        if "zred" in theta:
            z_scalar = jnp.ravel(theta["zred"])[0]
            line_only = line_only * jnp.float32(self._flux_factor(theta))
            if self.igm is not None:
                transmission = self._igm_transmission(z_scalar, theta)
                line_only = line_only * transmission.astype(line_only.dtype)
        if "eline_scaling" in theta:
            line_only = line_only * jnp.ravel(theta["eline_scaling"])[0]
        return line_only

    @property
    def all_params(self):
        """Return the initial theta dict (one entry per free parameter)."""
        return dict(self.theta_init)

    @property
    def cosmo(self):
        """The cosmology; fixed at construction (assignment raises)."""
        return self._cosmo

    @cosmo.setter
    def cosmo(self, value):
        raise AttributeError(
            "the cosmology is fixed at construction (compiled predictions would "
            "silently keep the old one); build a new basis with cosmo=...")

    def _flux_factor(self, theta):
        """(1+z) (10 pc / D)^2 x L_sun/Hz -> cgs; D from ``theta['lumdist_mpc']`` when
        present, else D_L(``theta['zred']``)."""
        from ..cosmology import flux_factor_cgs
        z_scalar = jnp.ravel(theta["zred"])[0]
        lumdist = (jnp.ravel(theta["lumdist_mpc"])[0]
                   if "lumdist_mpc" in theta else None)
        return flux_factor_cgs(z_scalar, self.cosmo, lumdist_mpc=lumdist)

    def age_at(self, z):
        """Age of the Universe [Gyr] at redshift ``z`` under ``self.cosmo``
        (a float for a scalar, a JAX array otherwise)."""
        from ceridwen.cosmology import age_gyr
        out = age_gyr(jnp.asarray(z, dtype=float), self.cosmo)
        return float(out) if jnp.ndim(out) == 0 else out

    def __repr__(self):
        lines = [
            "<CSPBasis (dict theta)>",
            "-" * 38,
            f"Universe age at z=0  : {self.age_at(0.0):.3f} Gyr",
            f"Cosmology            : {self.cosmo.describe()}",
            f"n_time               : {self.n_time}",
            f"n_SSP_ages           : {len(self.ages)}",
            f"n_metallicities      : {len(self.zmet)}   logzsol "
            f"[{float(self.zmet.min()):+.2f} .. {float(self.zmet.max()):+.2f}]"
            + ("  = [Fe/H]" if self.axis_meaning == "feh" else ""),
            f"Z_sun (grid)         : {self.zsun_nominal:.6g}"
            f"  (log10 Z_sun = {self.log10_zsun:.6f}; {self.zsun_source})",
            f"wavelength range     : {float(self.wave.min()):.0f} – {float(self.wave.max()):.0f} Å",
            f"SFH integration      : {self.sfh_interp}",
            "Parameters:",
        ]
        for k, v in self.theta_init.items():
            lines.append(f"  {k:<28s}: shape {v.shape}")
        return "\n".join(lines)


    def display_sfh(self, theta=None, ax=None, *,
                    overlay_nodes=True, show_bin_edges=False,
                    units="Gyr", **plot_kwargs):
        """Plot the SFH exactly as ``_ssp_weights`` interprets it (step: constant per bin; linear:
        chords between nodes); ``theta`` defaults to ``theta_init``, ``units`` in {"Gyr", "Myr", "yr"}.
        Returns the axes.
        """
        import matplotlib.pyplot as plt

        theta = self.theta_init if theta is None else theta

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

        T_yr = T_gyr * 1e9
        dt_yr = T_yr[1:] - T_yr[:-1]
        if not np.all(dt_yr > 0):
            raise AssertionError(
                "lookback-time grid must be strictly increasing (today at "
                f"index 0, oldest last); got dt_yr = {dt_yr}"
            )

        if per_bin:
            bar_psi = psi
        else:
            bar_psi = 0.5 * (psi[:-1] + psi[1:])

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


        total_mass = float(np.sum(bar_psi * dt_yr))
        ax.set_title(
            f"sfh_interp={self.sfh_interp!r}, n_time={n_time}, "
            f"M_total = {total_mass:.3e} M_sun"
        )

        return ax


    def _lookback_from_zred(self, zred):
        """Construction lookback grid [yr] rescaled so its oldest node equals the age of the Universe
        at ``zred`` (differentiable), clipped to the SSP age ceiling.
        """
        from ceridwen.cosmology import age_gyr
        z = jnp.ravel(jnp.asarray(zred, dtype=float))[0]
        tuniv_yr = age_gyr(z, self.cosmo) * 1.0e9
        ref_old_yr = self.sfh_times[-1]
        scaled = self.sfh_times * (tuniv_yr / ref_old_yr)
        return jnp.clip(scaled, 0.0, self._age_clip_hi)

    def _ssp_weights(self, theta, *, zh_mode, sfh_mode):
        """(n_z, n_age) SSP weights.  ``zh_mode`` "const" reads theta["logzsol"], "var"
        theta["logzsol_hist"] (both log10(Z/Z_sun), looked up on the logzsol axis self.zmet); ``sfh_mode`` "linear" integrates a piecewise-linear SFH in
        log age (a per-bin SFH is first mapped to nodes as in ``display_sfh``), "step" a
        piecewise-constant SFH over the SSP Voronoi cells.  Grid precedence:
        theta["lookback_time"] (Gyr), then the zred-tracked grid, then ``self.sfh_times``.
        """
        floor = 1e-30
        sfh = jnp.clip(theta["sfh"], floor, None)

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
            if self.sfh_per_bin:   # per-bin values to nodes: interior means, end bins at the end nodes
                sfh = jnp.concatenate([sfh[:1], 0.5 * (sfh[:-1] + sfh[1:]), sfh[-1:]])
            slope = jnp.diff(sfh) / (sfh[:-1] * dt)
            m2    = sfh[:-1] * (1.0 + 0.5 * slope * dt) * dt   # = 0.5*(sfh[:-1]+sfh[1:])*dt

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
            if self.sfh_per_bin:
                sfh_mid = sfh
            else:
                sfh_mid = 0.5 * (sfh[:-1] + sfh[1:])
            m2 = sfh_mid * dt

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

            target_Z = theta["logzsol"]
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

        zh   = theta["logzsol_hist"]
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
        """Weights for constant metallicity, piecewise-linear SFH."""
        return self._ssp_weights(theta, zh_mode="const", sfh_mode="linear")

    def calculate_ssp_weights_const_zh_step(self, theta):
        """Weights for constant metallicity, piecewise-constant SFH."""
        return self._ssp_weights(theta, zh_mode="const", sfh_mode="step")

    def calculate_ssp_weights_var_zh(self, theta):
        """Weights for a metallicity history, piecewise-linear SFH."""
        return self._ssp_weights(theta, zh_mode="var", sfh_mode="linear")

    def calculate_ssp_weights_var_zh_step(self, theta):
        """Weights for a metallicity history, piecewise-constant SFH."""
        return self._ssp_weights(theta, zh_mode="var", sfh_mode="step")


    def _build_neb_array(self, theta, *, include_lines):
        """Dense (n_z, n_age, n_wave) nebular array (young rows only non-zero); not used by the forward model."""
        logZ_gas = self._gas_logz(theta)
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

    def _neb_weights_and_base(self, W_f32, theta, *, include_lines, amplitude):
        """``(v (n_young,), base (n_young, n_wave))`` with sum_z W[z,a] neb[z,a,w] == v[y] base[y,w] for
        the young rows: the metallicity axis is contracted before the wavelength axis.
        """
        base, scale = self.neb.evaluate_batch_factored(
            self._gas_logz(theta), theta["gas_logu"],
            self._neb_ages_young, self._neb_logqq_young,
            include_lines=include_lines,
        )
        yi = self._neb_young_idx
        v = jnp.einsum("zy,zy->y", W_f32[:, yi],
                       scale.astype(jnp.float32)) * amplitude
        return v, base.astype(jnp.float32)

    def _neb_spectrum_term(self, W_f32, theta, *, include_lines,
                           attn_age=None, amplitude=jnp.float32(1.0)):
        """Nebular contribution (n_wave,), optionally attenuated by the young rows of ``attn_age``."""
        v, base = self._neb_weights_and_base(
            W_f32, theta, include_lines=include_lines, amplitude=amplitude)
        if attn_age is None:
            return jnp.einsum("y,yw->w", v, base)
        return jnp.einsum("y,yw,yw->w", v, base,
                          attn_age[self._neb_young_idx, :])

    def _ion_multiplier(self, theta):
        """``(ion_mult (n_age, n_wave), neb_amp)``: ionising flux of young ages scaled by frac_obrun (0 without it), nebular amplitude 1 - frac_obrun."""
        if "frac_obrun" in theta:
            f_esc = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            return (jnp.where(self.kill_ion, f_esc, jnp.float32(1.0)),
                    jnp.float32(1.0) - f_esc)
        return (jnp.where(self.kill_ion, jnp.float32(0.0), jnp.float32(1.0)),
                jnp.float32(1.0))

    def _picket_terms(self, theta, include_lines):
        """Picket-fence ingredients: ``(W, fo, attn_age, diffuse_curve, A_cov, A_clear, neb_v, neb_base)``
        with the stellar cube contracted against per-(age, wave) multipliers only:
        ``A_cov`` = covered channel (LyC removed, young weight 1 - fo, birth-cloud dust),
        ``A_clear`` = fo x young rows (no dust, LyC kept); the nebular term is the factored one
        scaled by 1 - fo."""
        W  = self.calculate_ssp_weights(theta=theta).astype(jnp.float32)
        fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        tau_age = jnp.einsum("ab,bw->aw", self._age_bin_mix, attn.astype(jnp.float32))
        attn_age = jnp.exp(-tau_age)                 # full birth-cloud, no bypass
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))
        young = self.young_mask.astype(jnp.float32)  # (n_age,)
        cov_w = (jnp.float32(1.0) - fo) * young + (jnp.float32(1.0) - young)
        no_ion = jnp.where(self.kill_ion, jnp.float32(0.0), jnp.float32(1.0))   # (n_age, n_wave)
        A_cov = no_ion * cov_w[:, None]
        A_clear = fo * young[:, None]
        neb_v, neb_base = self._neb_weights_and_base(
            W, theta, include_lines=include_lines, amplitude=jnp.float32(1.0) - fo)
        return W, fo, attn_age, diffuse_curve, A_cov, A_clear, neb_v, neb_base

    def _spectrum_picket_nodem(self, theta, include_lines):
        """Picket-fence geometry, no dust emission: a fraction ``frac_obrun`` of the young light (LyC
        included) escapes free of all dust; the covered fraction powers the nebular emission and is
        attenuated by birth-cloud and diffuse dust.  One contraction of the stellar cube.
        """
        W, fo, attn_age, diffuse_curve, A_cov, A_clear, neb_v, neb_base = \
            self._picket_terms(theta, include_lines)
        stellar = jnp.einsum("za,zaw,aw->w", W, self.flux,
                             A_cov * attn_age * diffuse_curve[None, :] + A_clear)
        neb = jnp.einsum("y,yw,yw->w", neb_v, neb_base,
                         attn_age[self._neb_young_idx, :]) * diffuse_curve
        return (stellar + neb).reshape((-1,))

    def _spectrum_picket_dem(self, theta, include_lines):
        """Picket-fence geometry with energy-balance dust emission; the clear channel cancels in L_abs.
        Two contractions of the stellar cube (dust-free and attenuated), like the mainline."""
        W, fo, attn_age, diffuse_curve, A_cov, A_clear, neb_v, neb_base = \
            self._picket_terms(theta, include_lines)
        yi = self._neb_young_idx
        spectrum_dust_free = (
            jnp.einsum("za,zaw,aw->w", W, self.flux, A_cov + A_clear)
            + jnp.einsum("y,yw->w", neb_v, neb_base))
        attenuated = (
            jnp.einsum("za,zaw,aw->w", W, self.flux,
                       A_cov * attn_age * diffuse_curve[None, :] + A_clear)
            + jnp.einsum("y,yw,yw->w", neb_v, neb_base, attn_age[yi, :]) * diffuse_curve)

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

    def get_spectrum_dattn_nodem_neb(self, theta, *, include_lines=None):
        """Dust attenuation + nebular emission.  ``include_lines`` None -> ``self.nebemlineinspec``."""
        if include_lines is None:
            include_lines = self.nebemlineinspec
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        if self.fesc_geometry == "picket" and "frac_obrun" in theta:
            return self._spectrum_picket_nodem(theta, include_lines)

        ion_mult, neb_amp = self._ion_multiplier(theta)

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        M        = self._age_bin_mix
        tau_age  = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age = jnp.exp(-tau_age)

        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo   # runaway fraction skips the birth cloud
            attn_age = jnp.where(self.kill_ion, jnp.float32(1.0), attn_age)   # escaped LyC: no birth-cloud dust at all

        W_f32 = W.astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", W_f32, self.flux,
                              ion_mult * attn_age)
        spectrum = spectrum + self._neb_spectrum_term(
            W_f32, theta, include_lines=include_lines,
            attn_age=attn_age, amplitude=neb_amp)
        spectrum = spectrum * jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_neb(self, theta, *, include_lines=None):
        """Dust attenuation + nebular emission + dust emission."""
        if include_lines is None:
            include_lines = self.nebemlineinspec
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        if self.fesc_geometry == "picket" and "frac_obrun" in theta:
            return self._spectrum_picket_dem(theta, include_lines)

        ion_mult, neb_amp = self._ion_multiplier(theta)

        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)
        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo   # runaway fraction skips the birth cloud
            attn_age = jnp.where(self.kill_ion, jnp.float32(1.0), attn_age)   # escaped LyC: no birth-cloud dust at all

        W_f32 = W.astype(jnp.float32)
        neb_v, neb_base = self._neb_weights_and_base(
            W_f32, theta, include_lines=include_lines, amplitude=neb_amp)
        yi = self._neb_young_idx
        spectrum_dust_free = (
            jnp.einsum("za,zaw,aw->w", W_f32, self.flux, ion_mult)
            + jnp.einsum("y,yw->w", neb_v, neb_base))
        attenuated = (
            jnp.einsum("za,zaw,aw->w", W_f32, self.flux, ion_mult * attn_age)
            + jnp.einsum("y,yw,yw->w", neb_v, neb_base, attn_age[yi, :]))
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
        """Dust attenuation only (``include_lines`` ignored)."""
        _ = include_lines
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M       = self._age_bin_mix
        tau_age = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age= jnp.exp(-tau_age)

        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        weights  = self.calculate_ssp_weights(theta).astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", weights, self.flux, attn_age)
        spectrum *= jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_noneb(self, theta, *, include_lines=None):
        """Dust attenuation + dust emission, no nebular (``include_lines`` ignored)."""
        _ = include_lines
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

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
        """Stellar continuum only."""
        _ = include_lines
        weights  = self.calculate_ssp_weights(theta=theta).astype(jnp.float32)
        return jnp.einsum("za,zaw->w", weights, self.flux)

    def get_spectrum_nodattn_nodem_neb(self, theta, *, include_lines=None):
        """Nebular emission, no dust."""
        if include_lines is None:
            include_lines = self.nebemlineinspec
        W = self.calculate_ssp_weights(theta=theta)   # (n_z, n_age)

        ion_mult, neb_amp = self._ion_multiplier(theta)

        W_f32 = W.astype(jnp.float32)
        return (jnp.einsum("za,zaw,aw->w", W_f32, self.flux, ion_mult)
                + self._neb_spectrum_term(W_f32, theta,
                                          include_lines=include_lines,
                                          amplitude=neb_amp))
