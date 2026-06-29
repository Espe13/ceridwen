from itertools import chain
import jax.numpy as jnp  # Use jax.numpy instead of numpy
from typing import Dict, Tuple
from jax import jit, vmap
from copy import deepcopy
try:
    import fsps
except (ImportError, RuntimeError):
    fsps = None  # Ensure fsps is handled correctly


__all__ = ["SSPBasis", "FastStepBasis"]


class SSPBasis:
    """
    A class that wraps the fsps.StellarPopulation object, which is used for producing SSPs.
    This class allows for custom calculation of relative SSP weights to produce spectra from arbitrary composite SFHs.
    """

    def __init__(self, zcontinuous=1, reserved_params=['tage', 'sigma_smooth'],
                interp_type='logarithmic', flux_interp='linear',
                mint_log=-3, compute_vega_mags=False, **kwargs):
        """
        Initialize the SSPBasis class with given parameters.
        """
        if fsps is None:
            raise ImportError("FSPS is required for this class but could not be imported.")

        self.interp_type = interp_type
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.reserved_params = reserved_params
        self.params = {}

        # Ensure kwargs take precedence over explicitly defined arguments
        fsps_kwargs = {
            "compute_vega_mags": compute_vega_mags,
            "zcontinuous": zcontinuous,
        }
        
        # Update with any kwargs passed in, allowing them to override defaults
        fsps_kwargs.update(kwargs)

        # Pass the combined dictionary to StellarPopulation
        self.ssp = fsps.StellarPopulation(sfh = 0, **fsps_kwargs)

        self.ssp.params['sfh'] = 0  # 0: Compute a simple stellar population (SSP)
        self.update(**kwargs)  # Ensure any additional parameters are handled

    def update(self, **params):
        """Update the parameters, passing the *unreserved* FSPS parameters
        through to the ``fsps.StellarPopulation`` object.

        :param params:
            A parameter dictionary.
        """
        for k, v in params.items():
            # try to make parameters scalar
            try:
                if (len(v) == 1) and callable(v[0]):
                    self.params[k] = v[0]
                else:
                    self.params[k] = jnp.squeeze(v)
            except:
                self.params[k] = v
            # Parameters named like FSPS params but that we reserve for use
            # here.  Do not pass them to FSPS.
            if k in self.reserved_params:
                continue
            # Otherwise if a parameter exists in the FSPS parameter set, pass a
            # copy of it in.
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = deepcopy(v)

        # We use FSPS for SSPs !!ONLY!!
        # except for FastStepBasis.  And CSPSpecBasis. and...
        # assert self.ssp.params['sfh'] == 0
    
    def get_galaxy_spectrum(self, **params):
        """Update parameters, then get the SSP spectrum

        Returns
        -------
        wave : ndarray
            Restframe avelength in angstroms.

        spectrum : ndarray
            Spectrum in units of Lsun/Hz per solar mass formed.

        mass_fraction : float
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        wave, spec = self.ssp.get_spectrum(tage=float(self.ssp.params['tage']), peraa=False)
        return wave, spec, self.ssp.stellar_mass

    def get_galaxy_elines(self):
        """Get the wavelengths and specific emission line luminosity of the nebular emission lines
        predicted by FSPS. These lines are in units of Lsun/solar mass formed.
        This assumes that `get_galaxy_spectrum` has already been called.

        :returns ewave:
            The *restframe* wavelengths of the emission lines, AA

        :returns elum:
            Specific luminosities of the nebular emission lines,
            Lsun/stellar mass formed
        """
        ewave = self.ssp.emline_wavelengths
        # This allows subclasses to set their own specific emission line
        # luminosities within other methods, e.g., get_galaxy_spectrum, by
        # populating the `_specific_line_luminosity` attribute.
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            elum = self.ssp.emline_luminosity.copy()
            if elum.ndim > 1:
                elum = elum[0]
            if self.ssp.params["sfh"] == 3:
                # tabular sfh
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
    """
    Subclass of SSPBasis that implements a "nonparameteric" SFH.
    """

    def get_galaxy_spectrum(self, **params) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Construct the tabular SFH and feed it to the SSP.

        Not JIT-compiled: the body calls into FSPS Fortran
        (``self.ssp.set_tabular_sfh`` / ``self.ssp.get_spectrum``), which is
        not JAX-traceable, so a ``@jit`` here cannot work — in fact the
        previous bound-method ``@jit`` made this method *uncallable* (``self``
        was passed as a would-be abstract array and the trace raised
        immediately, so neither the FSPS path nor the validation below ever
        executed).  The only purely-numerical part, :meth:`convert_sfh`, is
        JIT-compiled on its own below.  The age-bin validation therefore runs
        on concrete arrays in this (un-jitted) method, which is the correct
        place for a data-validation guard.
        """
        self.update(**params)
        if float(jnp.min(jnp.diff(10 ** jnp.asarray(self.params['agebins'])))) < 1e6:
            raise ValueError("Minimum age bin spacing must be at least 1 million years.")

        mtot = jnp.sum(self.params['mass'])
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3  # Set SFH to tabular
        self.ssp.set_tabular_sfh(time, sfr)
        wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)
        return jnp.array(wave), jnp.array(spec) / mtot, self.ssp.stellar_mass / mtot

    @staticmethod
    def convert_sfh(agebins, mformed, epsilon=1e-4, maxage=None) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Calculate a tabular SFH based on age bins and formed masses.

        Not JIT-compiled: ``jnp.unique`` below has a value-dependent output
        size, which raises ``ConcretizationTypeError`` under ``jax.jit`` (the
        previous ``@jit`` made this helper uncallable).  It is only ever called
        from the un-jitted :meth:`get_galaxy_spectrum` (which itself wraps FSPS
        Fortran), so there is no hot path that benefits from compiling it.
        """
        agebins_yrs = 10 ** jnp.array(agebins).T
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        bin_edges = jnp.unique(agebins_yrs)
        if maxage is None:
            maxage = agebins_yrs.max()

        t = jnp.concatenate((bin_edges * (1. - epsilon), bin_edges * (1 + epsilon)))
        t = t[1:-1]  # Remove older than oldest bin, younger than youngest bin
        fsps_time = maxage - t

        sfr = mformed / dt
        sfrout = jnp.zeros_like(t)
        sfrout = sfrout.at[::2].set(sfr)
        sfrout = sfrout.at[1::2].set(sfr)

        return (fsps_time / 1e9)[::-1], sfrout[::-1], maxage / 1e9

