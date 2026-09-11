from itertools import chain
import jax.numpy as jnp
from typing import Dict, Tuple
from jax import jit, vmap
from copy import deepcopy
from .ssp_data import _import_fsps


__all__ = ["SSPBasis", "FastStepBasis"]


class SSPBasis:
    """Wraps ``fsps.StellarPopulation`` to produce SSP spectra."""

    def __init__(self, zcontinuous=1, reserved_params=['tage', 'sigma_smooth'],
                interp_type='logarithmic', flux_interp='linear',
                mint_log=-3, compute_vega_mags=False, **kwargs):
        fsps = _import_fsps()

        self.interp_type = interp_type
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.reserved_params = reserved_params
        self.params = {}

        fsps_kwargs = {
            "compute_vega_mags": compute_vega_mags,
            "zcontinuous": zcontinuous,
        }
        
        fsps_kwargs.update(kwargs)

        self.ssp = fsps.StellarPopulation(sfh = 0, **fsps_kwargs)

        self.ssp.params['sfh'] = 0
        self.update(**kwargs)

    def update(self, **params):
        """Store ``params``, passing non-reserved ones through to the wrapped stellar population."""
        for k, v in params.items():
            try:
                if (len(v) == 1) and callable(v[0]):
                    self.params[k] = v[0]
                else:
                    self.params[k] = jnp.squeeze(v)
            except:
                self.params[k] = v
            if k in self.reserved_params:
                continue
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = deepcopy(v)

    def get_galaxy_spectrum(self, **params):
        """Return (rest-frame wave [AA], spectrum [Lsun/Hz per Msun formed], surviving mass fraction)."""
        self.update(**params)
        wave, spec = self.ssp.get_spectrum(tage=float(self.ssp.params['tage']), peraa=False)
        return wave, spec, self.ssp.stellar_mass

    def get_galaxy_elines(self):
        """Return (rest-frame line wavelengths [AA], line luminosities [Lsun per Msun formed]);
        call after ``get_galaxy_spectrum``."""
        ewave = self.ssp.emline_wavelengths
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            elum = self.ssp.emline_luminosity.copy()
            if elum.ndim > 1:
                elum = elum[0]
            if self.ssp.params["sfh"] == 3:
                mass = jnp.sum(self.params.get('mass', 1.0))
                elum /= mass

        return ewave, elum

    @property
    def logage(self) -> jnp.ndarray:
        return jnp.array(self.ssp.ssp_ages.copy())

    @property
    def wavelengths(self) -> jnp.ndarray:
        return jnp.array(self.ssp.wavelengths.copy())

    @property
    def spectral_resolution(self) -> jnp.ndarray:
        return jnp.array(getattr(self.ssp, "resolutions", jnp.array(0)))


class FastStepBasis(SSPBasis):
    """SSPBasis with a tabular (non-parametric) step SFH."""

    def get_galaxy_spectrum(self, **params) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Return (wave [AA], spectrum [Lsun/Hz per Msun formed], surviving mass fraction)
        for the step SFH given by ``agebins`` (log10 yr) and ``mass``; not JIT-able."""
        self.update(**params)
        if float(jnp.min(jnp.diff(10 ** jnp.asarray(self.params['agebins'])))) < 1e6:
            raise ValueError("Minimum age bin spacing must be at least 1 million years.")

        mtot = jnp.sum(self.params['mass'])
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3
        self.ssp.set_tabular_sfh(time, sfr)
        wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)
        return jnp.array(wave), jnp.array(spec) / mtot, self.ssp.stellar_mass / mtot

    @staticmethod
    def convert_sfh(agebins, mformed, epsilon=1e-4, maxage=None) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Return (time [Gyr, increasing], sfr [Msun/yr], tmax [Gyr]) for step ``agebins`` (log10 yr); not JIT-able."""
        agebins_yrs = 10 ** jnp.array(agebins).T
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        bin_edges = jnp.unique(agebins_yrs)
        if maxage is None:
            maxage = agebins_yrs.max()

        t = jnp.concatenate((bin_edges * (1. - epsilon), bin_edges * (1 + epsilon)))
        t = t[1:-1]
        fsps_time = maxage - t

        sfr = mformed / dt
        sfrout = jnp.zeros_like(t)
        sfrout = sfrout.at[::2].set(sfr)
        sfrout = sfrout.at[1::2].set(sfr)

        return (fsps_time / 1e9)[::-1], sfrout[::-1], maxage / 1e9

