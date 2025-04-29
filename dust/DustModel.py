import jax.numpy as jnp
from jax import vmap
import numpy as np
from pathlib import Path
import jax



class Dust:
    """
    A JAX-compatible dust attenuation model optimized for fast parameter updates.
    
    Parameters:
        bin_edges: Age bin edges in Myr.
        dust_laws: Attenuation law for each bin (0 = power-law, 1 = Kriek & Conroy 2013).
        tau_values: Optical depths for each bin.
        dust_indices: Controls wavelength dependence.
        diffuse_tau: Diffuse dust optical depth.
        diffuse_index: Diffuse dust index.
        duste: Whether to enable dust emission modeling.
        duste_model: The dust emission model to use. Options: "DL07" (default), "THEMIS".
        dust_file: Path to the dust emission data files.
        spec_lambda: Wavelength array in Angstroms.
        dwargs: Additional dust model arguments.
    """

    def __init__(self, bin_edges=None, dust_laws=None, tau_values=None, dust_indices=None,
                 diffuse_tau=0.2, diffuse_index=-0.7, diffuse_law=1, duste=False, duste_model="DL07",
                 dust_file=None, spec_lambda=None, **kwargs):
        # Initialize fundamental parameters
        self.bin_edges = bin_edges if bin_edges is not None else jnp.array([(0, 10), (10, jnp.inf)])
        self.num_bins = len(self.bin_edges)

        self.dust_laws = dust_laws if dust_laws is not None else jnp.array([0, 1])
        self.tau_values = tau_values if tau_values is not None else jnp.ones(self.num_bins)
        self.dust_indices = dust_indices if dust_indices is not None else jnp.array([-0.7, -0.3])

        self.diffuse_tau = diffuse_tau
        self.diffuse_index = diffuse_index
        self.diffuse_law = {0: attn_power_law, 1: attn_kriek_conroy}.get(diffuse_law, attn_kriek_conroy) 
        self.duste = duste
        self.duste_model = duste_model
        self.dwargs = kwargs  # Store extra parameters

        # Initialize dust emission attributes (default to None)
        self.duste_qpah = None
        self.duste_umin = None
        self.duste_gamma = None
        self.qpaharr = None
        self.uminarr = None
        self.dustem2_dustem = None  # Will store dust emission spectra
        self.dust_file = None
        self.spec_lambda = None

        # If dust emission is enabled, initialize the relevant properties
        if self.duste:
            self.duste_qpah = kwargs.pop("duste_qpah", 1.1)
            self.duste_umin = kwargs.pop("duste_umin", 0.72)
            self.duste_gamma = kwargs.pop("duste_gamma", 0.5)

            # Set emission model-specific arrays
            if self.duste_model == "DL07":
                self.qpaharr = jnp.array([0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58])
                self.uminarr = jnp.array([
                    0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0,
                    2.5, 3.0, 4.0, 5.0, 7.0, 8.0, 12.0, 15.0, 20.0, 25.0
                ])
            elif self.duste_model == "THEMIS":
                self.qpaharr = jnp.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
                self.uminarr = jnp.array([
                    0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0,
                    1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
                    12.0, 15.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 80.0
                ])
            else:
                raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

            # Ensure dust_file and spec_lambda are provided if duste=True
            if dust_file is None or spec_lambda is None:
                raise ValueError("If `duste=True`, both `dust_file` and `spec_lambda` must be provided.")
            self.dust_file = dust_file
            self.spec_lambda = spec_lambda
            # Load dust emission spectra
            self.load_dust_emission(dust_file, spec_lambda)

    def __repr__(self):
        def format_dust_laws(dust_laws):
            """Convert numerical dust laws to human-readable format."""
            return [("Power-law" if law == 0 else "Kriek & Conroy 13") for law in dust_laws]

        def format_array(arr):
            """Format small arrays inline, larger ones multi-line."""
            if arr is None:
                return "None"
            if arr.ndim == 1 and len(arr) <= 5:
                return f"[{', '.join(map(str, arr))}]"
            return f"\n    " + "\n    ".join(map(str, arr))

        attributes = {
            "Duste enabled": self.duste,
            "Duste model": self.duste_model if self.duste else "N/A",
            "Number of bins": self.num_bins,
            "Bin edges (Myr)": format_array(self.bin_edges),
            "Dust laws": format_dust_laws(self.dust_laws),
            "Tau values": format_array(self.tau_values),
            "Dust indices": format_array(self.dust_indices),
            "Diffuse tau": self.diffuse_tau,
            "Diffuse index": self.diffuse_index,
        }

        # Only include dust emission properties if duste=True
        if self.duste:
            attributes.update({
                "DUST qPAH": self.duste_qpah,
                "DUST Umin": self.duste_umin,
                "DUST Gamma": self.duste_gamma,
                f"qpaharr ({self.duste_model})": self.qpaharr,
                f"uminarr ({self.duste_model})": self.uminarr,
                "dust_file": self.dust_file,
                "spec_lambda": self.spec_lambda.shape,
            })

        # Only show dwargs if it's not empty
        if self.dwargs:
            attributes["Extra parameters (dwargs)"] = self.dwargs

        attr_str = "\n".join(f"  {k:<30}: {v}" for k, v in attributes.items() if v is not None)
        return f"\nDust Model:\n{'='*50}\n{attr_str}\n{'='*50}"

    def load_dust_emission(self, dust_file=None, spec_lambda=None):
        """
        Reads and interpolates dust emission spectra from files and returns a new Dust instance with the loaded emission.
        Optimized for JAX using jnp and vmap.

        Parameters:
            dust_file: Path to the dust emission file directory.
            spec_lambda: Wavelength array in Angstroms.

        Returns:
            dustem2_dustem: JAX array of dust emission spectra interpolated onto spec_lambda.
        """
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
        
        if self.duste_model not in dust_model_params:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        nqpah_dustem, ndim_dustem, numin_dustem = dust_model_params[self.duste_model]

        # Initialize storage for interpolated spectra, in numpy not jnp bc it just reads in the data once, no need for jax and it was faster to code.
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

        # Convert to JAX array for compatibility
        self.dustem2_dustem = jnp.array(dustem2_dustem)

    def compute_attenuation(self, spec_lambda):
        """
        Compute dust attenuation curves for each bin with updated parameters.

        Parameters:
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.

        Returns:
            jnp.ndarray: Attenuation curves (shape: `(num_bins, len(spec_lambda))`).
        """
        if spec_lambda is None:
            spec_lambda = self.spec_lambda
            
        def compute_curve(i):
            """Compute attenuation for a single bin."""
            tau = self.tau_values[i]
            dust_index = self.dust_indices[i]
            dust_law = self.dust_laws[i]

            return jnp.where(
                dust_law == 0,  # 0 = power-law, 1 = Kriek & Conroy 2013
                attn_power_law(spec_lambda, tau, dust_index),
                attn_kriek_conroy(spec_lambda, tau, dust_index)
            )

        # Vectorized computation over bins using `vmap`
        self.attn_curves = vmap(compute_curve)(jnp.arange(self.num_bins))
        return self.attn_curves

    def apply_attenuation(self, spec_lambda, csp_spectra_list):
        """
        Applies dust attenuation to each input spectrum.

        Parameters:
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            csp_spectra_list (jnp.ndarray): Input CSP spectra (shape: `(num_bins, len(spec_lambda))`).

        Returns:
            jnp.ndarray: Attenuated spectra (shape: `(num_bins, len(spec_lambda))`).
        """

        # Compute attenuation curves for each bin
        attn_curves = self.compute_attenuation(spec_lambda)

        # Element-wise multiplication to apply attenuation
        return csp_spectra_list * attn_curves

    def apply_diffuse_dust(self, spec_lambda, dusty_spectra):
        """
        Apply diffuse dust attenuation to all spectra efficiently using `vmap`.

        Parameters:
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            dusty_spectra (jnp.ndarray): Input spectra (shape: `(num_bins, len(spec_lambda))`).

        Returns:
            jnp.ndarray: Attenuated spectra after applying diffuse dust attenuation.
        """

        # Compute diffuse dust attenuation curve (same for all bins)
        diffuse = self.diffuse_law(spec_lambda, self.diffuse_tau, self.diffuse_index)

        # Use vmap to efficiently apply the attenuation to all spectra
        return vmap(lambda spectrum: spectrum * diffuse)(dusty_spectra)

    def apply_diffuse_and_specific_dust(self, spec_lambda, spectra):
        """
        Apply both diffuse and specific dust attenuation to spectra efficiently.

        Parameters:
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            spectra (list or jnp.ndarray): List of dust-attenuated spectra for each bin.

        Returns:
            jnp.ndarray: Attenuated spectra after applying both diffuse and specific dust attenuation.
        """

        # Compute diffuse dust attenuation curve (same for all bins)
        diffuse = attn_kriek_conroy(spec_lambda, self.diffuse_tau, self.diffuse_index)

        # Convert spectra list to JAX array if needed
        spectra = jnp.stack(spectra) if isinstance(spectra, list) else spectra

        # Compute specific dust attenuation and sum across bins
        specific_dust_spectra = jnp.sum(self.apply_attenuation(spec_lambda, spectra), axis=0)

        # Apply diffuse dust attenuation
        return specific_dust_spectra * diffuse

    def compute_dust_emission(self, specdust, csp_spectra, spec_lambda, 
                                        duste_qpah=None, duste_umin=None, duste_gamma=None):
        """
        Compute dust emission using JAX-optimized vectorization for GPU acceleration.

        Parameters:
            specdust (jnp.ndarray): Attenuated spectrum after dust absorption.
            csp_spectra (jnp.ndarray): Stellar spectrum before attenuation.
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            duste_qpah (float, optional): PAH fraction. Defaults to self.duste_qpah if None.
            duste_umin (float, optional): Minimum U radiation field. Defaults to self.duste_umin if None.
            duste_gamma (float, optional): Fraction of high U component. Defaults to self.duste_gamma if None.

        Returns:
            tuple: (Updated spectrum with dust emission added, Estimated dust mass)
        """

        csp_spectra = jnp.sum(csp_spectra, axis=0) if csp_spectra.ndim > 1 else csp_spectra

        # Use provided parameters if given, otherwise fallback to self attributes
        duste_qpah = duste_qpah if duste_qpah is not None else self.duste_qpah
        duste_umin = duste_umin if duste_umin is not None else self.duste_umin
        duste_gamma = duste_gamma if duste_gamma is not None else self.duste_gamma


        nu = jnp.array(2.9979E18 / spec_lambda )[::-1] # Frequency in Hz (c / λ)
        lbold = jnp.trapezoid(spec_lambda * specdust/nu, nu)  # L_bol after attenuation
        lboln = jnp.trapezoid(spec_lambda * csp_spectra/nu, nu)  # L_bol before attenuation

        # Interpolation indices for PAH fraction and Umin
        qlo = jnp.clip(jnp.searchsorted(self.qpaharr, duste_qpah) - 1, 0, len(self.qpaharr) - 2)
        dq = jnp.clip((duste_qpah - self.qpaharr[qlo]) / (self.qpaharr[qlo + 1] - self.qpaharr[qlo]), 0.0, 1.0)

        ulo = jnp.clip(jnp.searchsorted(self.uminarr, duste_umin) - 1, 0, len(self.uminarr) - 2)
        du = jnp.clip((duste_umin - self.uminarr[ulo]) / (self.uminarr[ulo + 1] - self.uminarr[ulo]), 0.0, 1.0)


        # Ensure gamma fraction is within [0,1]
        gamma = jnp.clip(duste_gamma, 0.0, 1.0)

        # Perform bilinear interpolation over qpah and Umin using `vmap`
        def interpolate_dustem(i):
            return (
                (1 - dq) * (1 - du) * self.dustem2_dustem[i, qlo, 2 * ulo - 1] +
                dq * (1 - du) * self.dustem2_dustem[i, qlo + 1, 2 * ulo - 1] +
                dq * du * self.dustem2_dustem[i, qlo + 1, 2 * (ulo + 1) - 1] +
                (1 - dq) * du * self.dustem2_dustem[i, qlo, 2 * (ulo + 1) - 1]
            ), (
                (1 - dq) * (1 - du) * self.dustem2_dustem[i, qlo, 2 * ulo] +
                dq * (1 - du) * self.dustem2_dustem[i, qlo + 1, 2 * ulo] +
                dq * du * self.dustem2_dustem[i, qlo + 1, 2 * (ulo + 1)] +
                (1 - dq) * du * self.dustem2_dustem[i, qlo, 2 * (ulo + 1)]
            )

        dumin, dumax = vmap(interpolate_dustem)(jnp.arange(len(spec_lambda)))

        # Compute dust emission spectrum
        mduste = (1 - gamma) * dumin + gamma * dumax
        mduste = jnp.maximum(mduste, 1e-70)

        # Normalize to absorbed luminosity
        labs = lboln - lbold  # Energy absorbed by dust
        norm = jnp.trapezoid(spec_lambda * mduste/nu, nu)  # Normalization factor
        duste = mduste / norm * labs  # Normalize dust emission
        duste = jnp.maximum(duste, 1e-70)

        # Define the stopping condition with a max iteration count
        def cond_fn(state):
            lbold, lboln, _, i = state
            return (jnp.abs(lboln - lbold) > 1e-2) & (i < max_iterations)  # Stop after 1 iterations

        # Define the iterative update
        def body_fn(state):
            lbold, lboln, tduste, i = state  # Extract iteration count
            oduste = duste
            duste_att = duste * jnp.exp(-self.diffuse_tau)  # Apply diffuse attenuation
            tduste = tduste + duste_att
            lbold = jnp.trapezoid(spec_lambda * duste_att/nu, nu)  # Update L_bol after self-absorption
            lboln = jnp.trapezoid(spec_lambda * oduste/nu, nu)  # Before self-absorption
            dusten = jnp.maximum(mduste / norm * (lboln - lbold), 1e-70)

            return lbold, lboln, tduste, i + 1  # Increment iteration count

        max_iterations = 1
        lbold, lboln, tduste, final_iter = jax.lax.while_loop(
            cond_fn, body_fn, (lbold, lboln, jnp.zeros_like(duste), 0)
        )

        # Compute estimated dust mass
        mdust = 3.21E-3 / (4 * jnp.pi) * labs / norm
  
        # Add dust emission to the stellar spectrum
        specdust = specdust + tduste

        return specdust, mdust, tduste, duste, mduste, dumin, dumax, nu


def attn_power_law(spec_lambda, tau, dust_index):
    """Compute power-law attenuation."""
    return jnp.exp(-tau * (spec_lambda / 5500.0) ** dust_index)

def attn_kriek_conroy(spec_lambda, tau, dust_index):
    """
    Computes the attenuation factor based on Kriek & Conroy (2013) with a Drude profile for the UV bump.

    Parameters:
        spec_lambda: Array of wavelengths in Angstroms.
        tau: Normalization of attenuation (optical depth).
        dust_index: Power-law slope for the attenuation curve.

    Returns:
        attenuation: The attenuation factor (exp(-tau_lambda)), same form as power-law attenuation.
    """

    # Define constants
    lamuvb = 2175.0  # Central wavelength of the UV bump in Angstroms
    dlam = 350.0     # Width of the UV bump
    lamv = 5500.0    # Normalization wavelength

    # Locate transition wavelength (6300 Å)
    w63 = jnp.argmax(spec_lambda >= 6300.0)

    # Kriek & Conroy (2013) attenuation curve
    cal00 = jnp.zeros_like(spec_lambda)
    cal00 = jnp.where(
        spec_lambda >= 6300.0,
        1.17 * (-1.857 + 1.04 * (1e4 / spec_lambda)) + 1.78,  # λ > 6300 Å
        1.17 * (-2.156 + 1.509 * (1e4 / spec_lambda) - 
                0.198 * (1e4 / spec_lambda) ** 2 + 
                0.011 * (1e4 / spec_lambda) ** 3) + 1.78  # λ < 6300 Å
    )

    # Normalize by R_V = 4.05
    cal00 /= (0.44 * 4.05)

    # Cut off negative values
    cal00 = jnp.where(cal00 < 0, 0.0, cal00)

    # Compute UV bump strength (Kriek & Conroy 2013, Eq. 3)
    eb = 0.85 - 1.9 * dust_index

    # Drude profile for the 2175 Å bump
    drude = (eb * (spec_lambda * dlam) ** 2) / (
        (spec_lambda**2 - lamuvb**2) ** 2 + (spec_lambda * dlam) ** 2
    )

    # Compute attenuation optical depth (τ_λ)
    tau_lambda = tau * (cal00 + drude / 4.05) * (spec_lambda / lamv) ** dust_index
    
    # Return attenuation factor, exp(-τ_λ), so that it behaves like power-law attenuation
    return jnp.exp(-tau_lambda)
