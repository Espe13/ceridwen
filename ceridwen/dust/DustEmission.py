import jax.numpy as jnp
import jax

import numpy as np
from pathlib import Path

import astropy.constants as const   

class DustEmission:

    def __init__(self, duste_model="DL07",
                 dust_file=None, spec_lambda=None, **kwargs):
        """
        Initialize the DustEmission object with parameters for dust emission modeling.

        Parameters
        ----------
        duste_model : str
            Dust emission model to use: 'DL07' or 'THEMIS'.
        dust_file : str
            Path to the dust emission file (required).
        spec_lambda : ndarray
            Wavelength grid over which dust emission will be evaluated (required).
        kwargs : dict
            Optional keyword arguments to override default dust parameters.
            Supported: duste_qpah, duste_umin, duste_gamma
        """

        # Store model choice (e.g., 'DL07' or 'THEMIS')
        self.duste_model = duste_model

        # Dust parameter values
        self.duste_qpah = None
        self.duste_umin = None
        self.duste_gamma = None

        # Model grid arrays for allowed values of qPAH and Umin
        self.qpaharr = None
        self.uminarr = None

        # Placeholder for loaded dust emission spectra
        self.dustem2_dustem = None

        # File path and wavelength grid
        self.dust_file = None
        self.spec_lambda = None

        # Store any additional keyword arguments for later use
        self.dwargs = kwargs

        # Read in key dust parameters, allowing overrides via kwargs
        self.duste_qpah = kwargs.pop("duste_qpah", 3.5)
        self.duste_umin = kwargs.pop("duste_umin", 1.0)
        self.duste_gamma = kwargs.pop("duste_gamma", 0.01)

        # Set parameter grids based on selected dust model
        if self.duste_model == "DL07":
            self.qpaharr = jnp.array([0.47,1.12,1.77,2.50,3.19,3.90,4.58])
            self.uminarr = jnp.array([0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.8,1.0,1.2,1.5,
                                      2.0,2.5,3.0,4.0,5.0,7.0,8.0,12.0,15.0,20.0,25.0])
            self.nqpah_dustem = self.qpaharr.size
            self.numin_dustem = self.uminarr.size
        elif self.duste_model == "THEMIS":
            # THEMIS model uses smaller qPAH values rescaled to percent
            self.qpaharr = jnp.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24,
                                      0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
            self.uminarr = jnp.array([
                0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
                0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
                6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
                35.0, 40.0, 50.0, 80.0
            ])
            self.nqpah_dustem = self.qpaharr.size
            self.numin_dustem = self.uminarr.size
        else:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        # Ensure that necessary data is provided
        if dust_file is None or spec_lambda is None:
            raise ValueError("If `duste=True`, both `dust_file` and `spec_lambda` must be provided.")

        self.dust_file = dust_file
        self.spec_lambda = spec_lambda

        # Load emission templates or model data from file
        self.load_dust_emission(dust_file, spec_lambda)

    def __repr__(self):
        """
        Custom string representation of the DustEmission object.
        Provides a readable summary of model settings and parameters.
        """

        def format_array(arr):
            """Helper to format short arrays inline; longer arrays multiline."""
            if arr is None:
                return "None"
            if arr.ndim == 1 and len(arr) <= 5:
                return f"[{', '.join(map(str, arr))}]"
            return f"\n    " + "\n    ".join(map(str, arr))

        attributes = {
            "Duste model": self.duste_model,
            "DUST qPAH": self.duste_qpah,
            "DUST Umin": self.duste_umin,
            "DUST Gamma": self.duste_gamma,
            f"qpaharr ({self.duste_model})": format_array(self.qpaharr),
            f"uminarr ({self.duste_model})": format_array(self.uminarr),
            "dust_file": self.dust_file,
            "spec_lambda": self.spec_lambda.shape if self.spec_lambda is not None else None,
        }

        if self.dwargs:
            attributes["Extra parameters (dwargs)"] = self.dwargs

        attr_str = "\n".join(f"  {k:<30}: {v}" for k, v in attributes.items() if v is not None)
        return f"\nDustEmission Model:\n{'='*50}\n{attr_str}\n{'='*50}"

    def get_default_params(self):
        """
        Return a NamedTuple of default fit parameters, with names prefixed by 'diffuse_'.
        """
        from typing import NamedTuple, Any

        defaults = {
            "duste_qpah": self.duste_qpah,
            "duste_umin": self.duste_umin,
            "duste_gamma": self.duste_gamma
        }
        FitParams = NamedTuple("DustEmissionParams", [(k, Any) for k in defaults.keys()])
        return FitParams(**defaults)
    
    def load_dust_emission(self, dust_file=None, spec_lambda=None):
        
        # Use default paths if not provided
        if dust_file is None:
            dust_file = self.dust_file
        if spec_lambda is None:
            spec_lambda = self.spec_lambda

        # Select dust model parameters
        dust_model_params = {
            "DL07": (7, 1001, 22),
            "THEMIS": (11, 576, 37),
        }
        
        nqpah_dustem, ndim_dustem, numin_dustem = dust_model_params[self.duste_model]

        # Initialize storage for interpolated spectra (JAX-compatible)
        dustem2_dustem = np.zeros((len(spec_lambda), nqpah_dustem, numin_dustem * 2))

        # Read and interpolate dust emission spectra
        for k in range(nqpah_dustem):
            filename = Path(dust_file) / "dust" / "dustem" / f"{self.duste_model}_MW3.1_{'100' if k == 10 else f'{k}0'}.dat"

            if not filename.exists():
                raise FileNotFoundError(f"Error opening dust emission file: {filename}. File does not exist.")

            with filename.open('r') as f:
                next(f)  # Skip first header line
                next(f)  # Skip second header line

                lambda_dustem = np.zeros(ndim_dustem)
                dustem_dustem = np.zeros((ndim_dustem, numin_dustem * 2))

                for i in range(ndim_dustem):
                    try:
                        values = list(map(float, f.readline().strip().split()))
                        lambda_dustem[i], dustem_dustem[i, :] = values[0], values[1:]
                    except Exception:
                        raise RuntimeError(f"Error reading dust emission file: {filename}")

            # Convert wavelength from microns to Angstroms
            lambda_dustem *= 1E4

            # Interpolate dust spectra onto the master wavelength array
            jj = jnp.searchsorted(spec_lambda / 1E4, 1, side='left')
            for j in range(numin_dustem * 2):
                dustem2_dustem[jj:, k, j] = jnp.interp(spec_lambda[jj:], lambda_dustem, dustem_dustem[:, j])

        self.dustem2_dustem = jnp.array(dustem2_dustem)

    def update_dust_params(self, duste_qpah = 3.5, duste_umin = 1.0, duste_gamma = 0.01):
        """
        Update dust parameters for emission calculations.

        Parameters:
            duste_qpah (float, optional): New PAH fraction.
            duste_umin (float, optional): New minimum U radiation field.
            duste_gamma (float, optional): New fraction of high U component.
        """
        self.duste_qpah = duste_qpah
        self.duste_umin = duste_umin
        self.duste_gamma = duste_gamma

    def compute_dust_emission(self, spec_attn, spec_dustfree, spec_lambda, diffuse_curve,
                                        duste_qpah, duste_umin, duste_gamma):
        """
        Compute dust emission using JAX-optimized vectorization for GPU acceleration.

        Parameters:
            spec_attn (jnp.ndarray): Attenuated spectrum after dust absorption.
            spec_dustfree (jnp.ndarray): Stellar spectrum before attenuation.
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            duste_qpah (float, optional): PAH fraction. Defaults to self.duste_qpah if None.
            duste_umin (float, optional): Minimum U radiation field. Defaults to self.duste_umin if None.
            duste_gamma (float, optional): Fraction of high U component. Defaults to self.duste_gamma if None.

        Returns:
            tuple: (Updated spectrum with dust emission added, Estimated dust mass)
        """

        clight = const.c.value  # m/s
        tiny_number = 1e-70
        
        nu = clight * 1e10 / spec_lambda  # Hz

        # --- Compute Lbol before and after attenuation ---
        lbold = jnp.trapezoid(spec_attn, nu)
        lboln = jnp.trapezoid(spec_dustfree, nu)

        # --- QPAH interpolation setup ---
        qlo = jnp.clip(jnp.searchsorted(self.qpaharr, duste_qpah) - 1, 0, self.nqpah_dustem - 2)
        dq = (duste_qpah - self.qpaharr[qlo]) / (self.qpaharr[qlo + 1] - self.qpaharr[qlo])
        dq = jnp.clip(dq, 0.0, 1.0)

        # --- Umin interpolation setup ---
        ulo = jnp.clip(jnp.searchsorted(self.uminarr, duste_umin) - 1, 0, self.numin_dustem - 2)
        du = (duste_umin - self.uminarr[ulo]) / (self.uminarr[ulo + 1] - self.uminarr[ulo])
        du = jnp.clip(du, 0.0, 1.0)
    
        # --- gamma limiting ---
        gamma = jnp.clip(duste_gamma, 0.0, 1.0)

        # --- Bilinear interpolation ---
        dumin = (
            (1 - dq) * (1 - du) * self.dustem2_dustem[:, qlo, 2 * ulo] +
            dq * (1 - du) * self.dustem2_dustem[:, qlo + 1, 2 * ulo] +
            dq * du * self.dustem2_dustem[:, qlo + 1, 2 * (ulo + 1)] +
            (1 - dq) * du * self.dustem2_dustem[:, qlo, 2 * (ulo + 1)] 
        )

        dumax = (
            (1 - dq) * (1 - du) * self.dustem2_dustem[:, qlo, 2 * ulo + 1] +
            dq * (1 - du) * self.dustem2_dustem[:, qlo + 1, 2 * ulo + 1] +
            dq * du * self.dustem2_dustem[:, qlo + 1, 2 * (ulo + 1) + 1] +
            (1 - dq) * du * self.dustem2_dustem[:, qlo, 2 * (ulo + 1) + 1]
        )
    
        # Combine both parts of P(U)dU
        mduste = (1 - gamma) * dumin + gamma * dumax
        mduste = jnp.maximum(mduste, tiny_number)
        mduste = jnp.squeeze(mduste)


        # Normalize the dust emission to match the absorbed luminosity
        labs = lboln - lbold
        norm = jnp.trapezoid(mduste, nu)
        duste = mduste / norm * labs
        self.duste = jnp.maximum(duste, tiny_number)


        # --- Two explicit self-absorption iterations ---
        # Iteration 1
        oduste_1 = duste
        duste_1 = duste * diffuse_curve
        tduste_1 = duste_1
        lbold_1 =jnp.trapezoid(duste_1, nu)
        lboln_1 =jnp.trapezoid(oduste_1, nu)
        duste_1 = jnp.maximum(mduste / norm * (lboln_1 - lbold_1), tiny_number)

        # Iteration 2
        oduste_2 = duste_1
        duste_2 = duste_1 * diffuse_curve
        tduste_2 = tduste_1 + duste_2
        lbold_2 =jnp.trapezoid(duste_2, nu)
        lboln_2 =jnp.trapezoid(oduste_2, nu)
        duste_2 = jnp.maximum(mduste / norm * (lboln_2 - lbold_2), tiny_number)

        # Final results
        specdust_final = spec_attn + tduste_2

        # Dust mass
        mdust = 3.21e-3 / (4 * jnp.pi) * labs / norm

        return specdust_final, mdust, tduste_2
    