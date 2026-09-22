"""Alpha-enhanced CSP basis without nebular emission.

``ssp_flux`` carries a leading [alpha/Fe] axis, (n_afe, n_z, n_age, n_wave); a scalar
theta["afe"] selects the plane by linear interpolation.  Lines observations are refused.

theta["logzsol"] (or "logzsol_hist") is [Fe/H] on these grids: every [alpha/Fe] plane shares
one [Fe/H] axis (FSPS AFE_FLAG=1).  The total metallicity [Z/H] = logzsol + f([alpha/Fe]) is
available as :meth:`CSPBasis_afe.logzsol_total` / ``grid_metadata.logzsol_total``.
"""

import os
import warnings

import numpy as np
import jax.numpy as jnp
import pprint

from ceridwen.csp.csp import (CSPBasis, LOGZSOL_KEYS, REMOVED_METALLICITY_KEYS,
                              _default_logzsol, removed_metallicity_key_error)
from ceridwen.ssps.grid_metadata import logzsol_total as _logzsol_total, refused_cell_bounds


class CSPBasis_afe(CSPBasis):
    """Composite stellar population basis with [alpha/Fe] interpolation and no nebular model.

    Same constructor and interface as ``CSPBasis`` minus the nebular arguments; requires an
    ``SSPDataAfe`` grid (4-D ``ssp_flux`` with an ``ssp_afe`` axis).  theta["afe"] is
    optional: absent, the plane closest to [alpha/Fe] = 0 is used.  The SFH weights,
    dust, projection, flux factor and cosmology are those of ``CSPBasis``.
    """

    def __init__(
        self,
        SSPData,
        theta=None,
        tiny_logt=-70,
        zh_const=False,
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
        cosmo=None,
        **kwargs,
    ):
        self.verbose = bool(verbose)
        self.gas_tied = False          # no nebular model on alpha grids
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
                f"CSPBasis_afe got unexpected keyword argument(s) {sorted(kwargs)}{hint}")
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
        if init_dust_params is None:
            init_dust_params = {'bin_edges': [(-jnp.inf, -1.97)], 'laws': ['powerlaw']}

        _flux_in = jnp.asarray(SSPData.ssp_flux)
        _afe_in  = getattr(SSPData, "ssp_afe", None)
        if _flux_in.ndim != 4 or _afe_in is None:
            raise TypeError(
                "CSPBasis_afe requires an alpha-enhanced SSP grid: a 4-D "
                "ssp_flux (n_afe, n_z, n_age, n_wave) WITH an ssp_afe axis "
                "(ceridwen.ssps.ssp_data_afe.SSPDataAfe). Got "
                f"ssp_flux.ndim={_flux_in.ndim} and ssp_afe="
                f"{'absent' if _afe_in is None else 'present'} "
                f"({type(SSPData).__name__}). For solar-scaled 3-D grids "
                "switch to the matching basis:\n"
                "    from ceridwen.csp import CSPBasis   # solar-scaled, "
                "with nebular model\n"
                "or rebuild the grid with SSPDataAfe.from_fsps (python-fsps "
                ">= 4.0, AFE_FLAG=1) to fit [alpha/Fe]."
            )
        _afe_in = jnp.atleast_1d(jnp.asarray(_afe_in, dtype=float))
        if _afe_in.size != _flux_in.shape[0]:
            raise ValueError(
                f"ssp_afe has {_afe_in.size} points but ssp_flux leads "
                f"with {_flux_in.shape[0]}; grid is inconsistent."
            )
        if _afe_in.size > 1 and not bool(np.all(np.diff(np.asarray(_afe_in)) > 0)):
            raise ValueError(
                "ssp_afe must be strictly increasing (required by the "
                "searchsorted-based interpolation in _flux_at_afe); got "
                f"{np.asarray(_afe_in).tolist()}."
            )
        self.flux      = jnp.array(_flux_in, dtype=jnp.float32)  # (n_afe, n_z, n_age, n_wave)
        self.afe_grid  = _afe_in                           # (n_afe,) [alpha/Fe]
        self._n_afe    = int(_afe_in.size)
        self._afe_solar_idx = int(np.argmin(np.abs(np.asarray(_afe_in))))
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
        if add_dust_emission and not sps_home:
            raise ValueError(
                "sps_home is required for dust emission but was not "
                "given and $SPS_HOME is unset. Set `export SPS_HOME=/path/to/fsps` "
                "(your FSPS data directory) or pass sps_home=... explicitly."
            )
        self.sps_home   = sps_home
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
        if add_igm:
            from ..igm import make_igm_model
            self.igm = make_igm_model(igm_model)
        else:
            self.igm = None
        self.igm_factor = float(igm_factor)

        if add_diffuse_dust or add_dust:
            self.set_attenuation_function(add_diffuse_dust, add_dust)

        theta = self.initialize_dust_components(
            add_dust, add_diffuse_dust, add_dust_emission,
            theta, init_dust_params, diffuse_law, sps_home,
        )

        self.configure_spectrum_model(
            add_dust, add_diffuse_dust, add_dust_emission, sps_home
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

    def initialize_model_structure(self, theta):
        """``CSPBasis.initialize_model_structure`` plus the optional scalar theta['afe']."""
        super().initialize_model_structure(theta)
        if 'afe' in theta:
            afe = jnp.atleast_1d(jnp.asarray(theta['afe'], dtype=float))
            assert afe.shape == (1,), \
                "'afe' must be a scalar (wrapped in shape-(1,) array)"
            if self._n_afe == 1:
                warnings.warn(
                    "theta contains 'afe' but the SSP grid has a single "
                    "[alpha/Fe] plane (n_afe=1, legacy or AFE_FLAG=0 grid); "
                    "'afe' is IGNORED. Build an alpha-enhanced grid with "
                    "SSPDataAfe.from_fsps to fit alpha enhancement.",
                    stacklevel=3,
                )
        self._known_theta_keys = (self._known_theta_keys | {'afe'}) - {'eline_scaling'}

    def check_param_ranges(self, theta=None, warn=True):
        """``CSPBasis.check_param_ranges`` plus theta['afe'] against the [alpha/Fe] grid."""
        if theta is None:
            theta = self.theta_init
        msgs = super().check_param_ranges(theta, warn=False)
        self._check_refused_cells_theta(theta)
        if 'afe' in theta and self._n_afe > 1:
            alo = float(self.afe_grid.min())
            ahi = float(self.afe_grid.max())
            v = np.asarray(theta['afe'], float)
            if v.size and (np.nanmin(v) < alo or np.nanmax(v) > ahi):
                msgs.append(
                    f"theta['afe'] has values outside the SSP [alpha/Fe] grid "
                    f"[{alo:+.2f}, {ahi:+.2f}]; these are silently clamped to "
                    f"the nearest grid edge (aMIST/C3K support is "
                    f"-0.2 .. +0.6)."
                )
        if warn:
            for m in msgs:
                warnings.warn(m, stacklevel=2)
        return msgs

    def logzsol_total(self, theta):
        """[Z/H] = logzsol + f([alpha/Fe]) for this theta (Z/X definition; grid_metadata)."""
        key = 'logzsol' if self.zh_const else 'logzsol_hist'
        afe = jnp.ravel(theta['afe'])[0] if 'afe' in theta else self.afe_grid[self._afe_solar_idx]
        return _logzsol_total(theta[key], afe)

    def refused_cells_logzsol(self):
        """[(logzsol_lo, logzsol_hi, afe_lo, afe_hi, reason)] of the refused interpolation cells.

        Taken from the GRID (``SSPData.refused_cells``), which is filled from the metadata
        table, from the file's own ``refused_cells_json`` provenance, or by the duplicate-
        isochrone detection in ``SSPDataAfe.from_fsps`` -- so a re-saved, converted or freshly
        built alpha grid keeps its refusals even when its content hash is not a table key."""
        return refused_cell_bounds(self._refused_cells, self._refused_reason,
                                   np.asarray(self.zmet), np.asarray(self.afe_grid))

    def _check_refused_cells_theta(self, theta):
        """Raise when a fixed theta lands in a refused (logzsol, afe) interpolation cell."""
        cells = self.refused_cells_logzsol()
        if not cells or self._n_afe == 1:
            return
        key = 'logzsol' if self.zh_const else 'logzsol_hist'
        if key not in theta or 'afe' not in theta:
            return
        z = np.asarray(theta[key], dtype=float)
        a = np.asarray(theta['afe'], dtype=float)
        for z_lo, z_hi, a_lo, a_hi, reason in cells:
            if z.size and a.size and np.nanmax(z) > z_lo and np.nanmax(a) > a_lo:
                raise ValueError(
                    f"theta['{key}'] = {np.array2string(z, precision=3)} with theta['afe'] = "
                    f"{np.array2string(a, precision=3)} reaches the refused interpolation cell "
                    f"logzsol in ({z_lo:+.3f}, {z_hi:+.3f}] x afe in ({a_lo:+.2f}, {a_hi:+.2f}]: "
                    f"{reason}.  Cap logzsol at {z_lo!r} or afe at {a_lo!r}.")

    def _afe_coords(self, theta):
        """``(k, w)`` with the interpolated plane ``(1 - w) * flux[k - 1] + w * flux[k]``; edges clamp like the metallicity interpolation."""
        target_afe = jnp.ravel(theta["afe"])[0]
        k = jnp.clip(
            jnp.searchsorted(self.afe_grid, target_afe, side='left'),
            1, self._n_afe - 1,
        )
        a0 = self.afe_grid[k - 1]
        a1 = self.afe_grid[k]
        w  = jnp.clip((target_afe - a0) / (a1 - a0), 0.0, 1.0)
        return k, w

    def _flux_at_afe(self, theta):
        """(n_z, n_age, n_wave) flux cube at theta['afe']: the single plane for n_afe == 1, the solar plane without 'afe', else the two-plane interpolation."""
        if self._n_afe == 1:
            return self.flux[0]
        if "afe" not in theta:
            return self.flux[self._afe_solar_idx]
        k, w = self._afe_coords(theta)
        f_lo = jnp.take(self.flux, k - 1, axis=0)
        f_hi = jnp.take(self.flux, k,     axis=0)
        w32  = w.astype(jnp.float32)
        return (jnp.float32(1.0) - w32) * f_lo + w32 * f_hi

    def configure_spectrum_model(
        self, add_dust, add_diffuse_dust, add_dust_emission, sps_home
    ):
        part1 = 'dust_'    if (add_dust or add_diffuse_dust) else 'nodust_'
        if add_dust_emission:
            if not add_dust or not add_diffuse_dust:
                raise ValueError(
                    "Dust emission requires both dust attenuation and diffuse dust."
                )
            part3 = 'dustemi'
        else:
            part3 = 'nodustemi'

        key = part1 + 'noneb_' + part3
        mapping = {
            'dust_noneb_dustemi':      self.get_spectrum_dattn_dem_noneb,
            'dust_noneb_nodustemi':    self.get_spectrum_dattn_nodem_noneb,
            'nodust_noneb_nodustemi':  self.get_spectrum_nodattn_nodem_noneb,
        }
        label = {
            'dust_noneb_dustemi':      'dust attenuation, dust emission (alpha-enhanced, no nebular)',
            'dust_noneb_nodustemi':    'dust attenuation only (alpha-enhanced, no nebular)',
            'nodust_noneb_nodustemi':  'stellar continuum only (alpha-enhanced, no nebular)',
        }
        if self.verbose:
            print(f"Spectrum model: {label[key]}")
        self.get_spectrum = mapping[key]

    def get_spectrum_components(self, theta: dict) -> tuple:
        """``(continuum, zeros)``: no nebular model, so the line component is identically zero."""
        self._warn_unknown_theta_keys(theta)
        continuum = self.get_spectrum(theta=theta, include_lines=False)
        return continuum, jnp.zeros_like(continuum)

    def _project_observations(self, spectrum_phot, spectrum_slit,
                              observations, theta, **kw):
        from ..observation.lines import Lines as _Lines
        if any(isinstance(o, _Lines) for o in observations):
            raise ValueError(
                "CSPBasis_afe carries no nebular model (there are no "
                "alpha-enhanced CLOUDY grids), so Lines observations cannot "
                "be predicted.  Remove the Lines observation(s), or use "
                "csp.CSPBasis with a solar-scaled grid.")
        return super()._project_observations(
            spectrum_phot, spectrum_slit, observations, theta, **kw)

    def get_line_spec(self, theta):
        """Zeros (no nebular model)."""
        _ = theta
        return jnp.zeros_like(self.wave)

    def __repr__(self):
        _afe = np.asarray(self.afe_grid)
        lines = [
            "<CSPBasis_afe (dict theta; alpha-enhanced, no nebular)>",
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
            f"n_afe                : {self._n_afe}   "
            f"[{_afe.min():+.2f} .. {_afe.max():+.2f}]",
            f"wavelength range     : {float(self.wave.min()):.0f} – {float(self.wave.max()):.0f} Å",
            f"SFH integration      : {self.sfh_interp}",
            "Parameters:",
        ]
        for k, v in self.theta_init.items():
            lines.append(f"  {k:<28s}: shape {v.shape}")
        return "\n".join(lines)

    def get_spectrum_dattn_nodem_noneb(self, theta, *, include_lines=None):
        """Dust attenuation only (``include_lines`` ignored)."""
        _ = include_lines
        flux = self._flux_at_afe(theta)               # (n_z, n_age, n_wave)
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M       = self._age_bin_mix
        tau_age = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age= jnp.exp(-tau_age)

        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        weights  = self.calculate_ssp_weights(theta).astype(jnp.float32)
        spectrum = jnp.einsum("za,zaw,aw->w", weights, flux, attn_age)
        spectrum *= jnp.exp(-attn_diffuse.astype(jnp.float32))

        return spectrum.reshape((-1,))

    def get_spectrum_dattn_dem_noneb(self, theta, *, include_lines=None):
        """Dust attenuation + dust emission (``include_lines`` ignored)."""
        _ = include_lines
        flux = self._flux_at_afe(theta)               # (n_z, n_age, n_wave)
        attn, attn_diffuse = self.attenuate_dust(self.wave, theta)

        M             = self._age_bin_mix
        tau_age       = jnp.einsum("ab,bw->aw", M, attn.astype(jnp.float32))
        attn_age      = jnp.exp(-tau_age)
        diffuse_curve = jnp.exp(-attn_diffuse.astype(jnp.float32))

        if "frac_obrun" in theta:
            fo = jnp.ravel(theta["frac_obrun"])[0].astype(jnp.float32)
            attn_age = (jnp.float32(1.0) - fo) * attn_age + fo

        weights           = self.calculate_ssp_weights(theta).astype(jnp.float32)
        spectrum_dust_free= jnp.einsum("za,zaw->w", weights, flux)
        attenuated        = jnp.einsum("za,zaw,aw->w", weights, flux, attn_age)
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
        flux     = self._flux_at_afe(theta)           # (n_z, n_age, n_wave)
        weights  = self.calculate_ssp_weights(theta=theta).astype(jnp.float32)
        return jnp.einsum("za,zaw->w", weights, flux)
