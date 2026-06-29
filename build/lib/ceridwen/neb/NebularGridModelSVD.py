"""
SVD-accelerated Nebular Emission Model.

This module provides ``NebularModelSVD``, which wraps the standard
``NebularModel`` and adds an SVD-compressed evaluation path.

Motivation
----------
The standard ``NebularModel.evaluate()`` performs trilinear interpolation
on the CLOUDY continuum cube (nspec, nz, nage, nu) and line cube
(nlines, nz, nage, nu), then distributes line luminosities onto the
wavelength grid via ``gaussnebarr @ line_lum``.  When called via
``vmap`` over n_z * n_young ~ 360 (metallicity, age) pairs, the
dominant costs are:

    1. The trilinear interpolation: 8 index-gather operations on arrays
       of size nspec (~7000), repeated 360 times.
    2. The Gaussian line projection: a (nspec, nlines) @ (nlines,)
       matmul, repeated 360 times.

Both operations scale with nspec.  By pre-evaluating the total nebular
spectrum (continuum + lines) at every CLOUDY grid node and compressing
the resulting cube via SVD, we can:

    - Replace trilinear interpolation on (nspec,) vectors with
      interpolation on (k_neb,) coefficient vectors (k_neb << nspec).
    - Eliminate the per-call Gaussian matmul entirely (folded into the
      pre-evaluated spectra).
    - Reconstruct the full-wavelength nebular spectrum only once at the
      end, after the weighted sum over ages.

The pre-evaluation is done at ``__init__`` time and stored as static
arrays.

Usage
-----
    >>> from ceridwen.neb.NebularGridModelSVD import NebularModelSVD
    >>> neb_svd = NebularModelSVD(
    ...     cloudy_dust=True, sps_home=SPS_HOME,
    ...     csp_lambda=wave, isoc_type='mist',
    ...     n_svd_components=15,
    ... )
    >>> # Single-point evaluation (same API as NebularModel)
    >>> cont, lines = neb_svd.evaluate(logZ, logU, logage, logQ)
    >>>
    >>> # Batch evaluation returning SVD coefficients (fast path)
    >>> coeffs = neb_svd.evaluate_svd_coeffs(logZ, logU, logage_arr, logQ_arr)
    >>> spectrum = neb_svd.reconstruct(coeffs)
"""

import jax.numpy as jnp
import numpy as np

from ceridwen.neb.NebularGridModel import NebularModel, _locate, _trilinear


class NebularModelSVD(NebularModel):
    """
    SVD-compressed nebular emission model.

    Inherits the full NebularModel and adds a pre-evaluated, SVD-compressed
    nebular spectrum cube for fast batch evaluation.

    Parameters
    ----------
    n_svd_components : int
        Number of SVD components for the nebular spectrum compression.
        Default 15.  Nebular spectra contain narrow emission lines, so
        they typically require more components than smooth stellar
        continua.  20--30 is a safe choice; 10--15 is aggressive.
    svd_variance_threshold : float or None
        If set, overrides n_svd_components by choosing k such that
        the retained variance fraction exceeds this threshold.
    **kwargs
        All arguments forwarded to NebularModel.__init__.
    """

    def __init__(self, *args, n_svd_components=15,
                 svd_variance_threshold=None, **kwargs):
        # Initialise the parent (loads CLOUDY grids, builds Gaussians)
        super().__init__(*args, **kwargs)

        self.use_svd = True
        self._build_svd_cube(n_svd_components, svd_variance_threshold)

    def _build_svd_cube(self, n_svd_components, svd_variance_threshold):
        """
        Pre-evaluate the total nebular spectrum (continuum + line) at
        every CLOUDY grid node and compress via SVD.

        The CLOUDY grids have shape:
            nebem_cont: (nspec, nz, nage, nu)     -- log10(L_cont / Q)
            nebem_line: (nlines, nz, nage, nu)    -- log10(L_line / Q)
            gaussnebarr: (nspec, nlines)           -- Gaussian profiles

        At each grid node (iz, ia, iu), the total nebular spectrum
        (per unit Q) is:

            neb_spec[w] = 10^cont[w,iz,ia,iu]
                        + sum_l gaussnebarr[w,l] * 10^line[l,iz,ia,iu]

        We evaluate this at all nz*nage*nu nodes, reshape to
        (nz*nage*nu, nspec), and compute a truncated SVD.
        """
        nspec = self.nspec
        nz = self.nebnz
        nage = self.nebnage
        nu = self.nebnip

        # Pre-evaluate total nebular spectrum at every grid node
        # Work in numpy for the init (this is done once)
        cont_cube = np.array(self.nebem_cont)   # (nspec, nz, nage, nu)
        line_cube = np.array(self.nebem_line)    # (nlines, nz, nage, nu)
        gauss = np.array(self.gaussnebarr)       # (nspec, nlines)

        n_nodes = nz * nage * nu
        total_spec = np.zeros((n_nodes, nspec))

        idx = 0
        for iz in range(nz):
            for ia in range(nage):
                for iu in range(nu):
                    # Continuum: 10^(log_cont), normalised per Q
                    cont = 10.0 ** cont_cube[:, iz, ia, iu]
                    # Lines: 10^(log_line), normalised per Q
                    line_lum = 10.0 ** line_cube[:, iz, ia, iu]
                    # Distribute lines onto wavelength grid
                    line_spec = gauss @ line_lum
                    total_spec[idx] = cont + line_spec
                    idx += 1

        # SVD of the total nebular spectrum matrix
        # total_spec: (n_nodes, nspec)
        U, S, Vt = np.linalg.svd(total_spec, full_matrices=False)

        # Determine truncation rank
        cumvar = np.cumsum(S ** 2)
        totvar = cumvar[-1]

        if svd_variance_threshold is not None:
            k = int(np.searchsorted(cumvar / totvar, svd_variance_threshold)) + 1
            k = max(k, 1)
            print(f"NebularSVD: variance threshold {svd_variance_threshold} -> "
                  f"k = {k} (captures {cumvar[k-1]/totvar*100:.4f}%)")
        else:
            k = min(n_svd_components, min(n_nodes, nspec))
            print(f"NebularSVD: using k = {k} components "
                  f"(captures {cumvar[k-1]/totvar*100:.4f}%)")

        self._neb_svd_k = k

        # Store truncated SVD factors
        # coeffs_cube: (nz, nage, nu, k) -- SVD coordinates at each grid node
        # basis: (k, nspec) -- spectral basis vectors
        coeffs = (U[:, :k] * S[None, :k])  # (n_nodes, k)
        self._neb_svd_coeffs_cube = jnp.array(
            coeffs.reshape(nz, nage, nu, k)
        )
        self._neb_svd_basis = jnp.array(Vt[:k, :])       # (k, nspec)
        self._neb_svd_S = jnp.array(S[:k])
        self._neb_svd_S_full = jnp.array(S)

        # Memory report
        orig_mb = cont_cube.nbytes / 1024**2 + line_cube.nbytes / 1024**2
        svd_mb = (self._neb_svd_coeffs_cube.nbytes +
                  self._neb_svd_basis.nbytes) / 1024**2
        print(f"NebularSVD: original grids {orig_mb:.2f} MB -> "
              f"SVD storage {svd_mb:.4f} MB "
              f"({svd_mb/orig_mb*100:.1f}%)")

    def evaluate_svd_coeffs(self, logZ, logU, logage, logQ):
        """
        Evaluate the nebular model and return SVD coefficients instead
        of the full spectrum.

        This is the fast path: trilinear interpolation operates on
        (k,)-vectors instead of (nspec,)-vectors.

        Parameters
        ----------
        logZ : scalar
            Gas metallicity log10(Z/Z_sun).
        logU : scalar
            Ionisation parameter log10(U).
        logage : scalar
            log10(age / yr).
        logQ : scalar
            log10(Q(H0)) in photons/s.

        Returns
        -------
        neb_coeffs : array, shape (k,)
            SVD coefficients for the total nebular spectrum.
            To reconstruct: spectrum = neb_coeffs @ self._neb_svd_basis
        """
        # Grid cell location
        z1 = _locate(logZ, self.nebem_logz)
        dz = jnp.clip(
            (logZ - self.nebem_logz[z1])
            / (self.nebem_logz[z1 + 1] - self.nebem_logz[z1]),
            0.0, 1.0,
        )
        u1 = _locate(logU, self.nebem_logu)
        du = jnp.clip(
            (logU - self.nebem_logu[u1])
            / (self.nebem_logu[u1 + 1] - self.nebem_logu[u1]),
            0.0, 1.0,
        )
        a1 = _locate(logage, self.nebem_age)
        da = jnp.clip(
            (logage - self.nebem_age[a1])
            / (self.nebem_age[a1 + 1] - self.nebem_age[a1]),
            0.0, 1.0,
        )

        # Trilinear interpolation on the SVD coefficient cube
        # _neb_svd_coeffs_cube has shape (nz, nage, nu, k)
        # We need to interpolate along the first 3 axes.
        # Use the same trilinear logic but on the (k,) trailing dim.
        cube = self._neb_svd_coeffs_cube  # (nz, nage, nu, k)

        w = jnp.array([
            (1 - dz) * (1 - da) * (1 - du),
            (1 - dz) * (1 - da) * (du),
            (1 - dz) * (da) * (1 - du),
            (1 - dz) * (da) * (du),
            (dz) * (1 - da) * (1 - du),
            (dz) * (1 - da) * (du),
            (dz) * (da) * (1 - du),
            (dz) * (da) * (du),
        ])  # (8,)

        corners = jnp.stack([
            cube[z1, a1, u1],
            cube[z1, a1, u1 + 1],
            cube[z1, a1 + 1, u1],
            cube[z1, a1 + 1, u1 + 1],
            cube[z1 + 1, a1, u1],
            cube[z1 + 1, a1, u1 + 1],
            cube[z1 + 1, a1 + 1, u1],
            cube[z1 + 1, a1 + 1, u1 + 1],
        ], axis=0)  # (8, k)

        # Interpolated SVD coefficients (per unit Q)
        coeffs_per_Q = jnp.sum(w[:, None] * corners, axis=0)  # (k,)

        # Scale by Q (in linear space, since the SVD was built on
        # linear spectra per unit Q)
        Q = 10.0 ** logQ
        return coeffs_per_Q * Q  # (k,)

    def reconstruct(self, neb_coeffs):
        """
        Reconstruct the full nebular spectrum from SVD coefficients.

        Parameters
        ----------
        neb_coeffs : array, shape (k,) or (n, k)
            SVD coefficient vector(s).

        Returns
        -------
        spectrum : array, shape (nspec,) or (n, nspec)
        """
        return neb_coeffs @ self._neb_svd_basis

    def evaluate_svd_full(self, logZ, logU, logage, logQ):
        """
        Evaluate via SVD and reconstruct the full spectrum.

        Drop-in replacement for evaluate() with the SVD path.
        Returns (total_spectrum, zeros) to maintain API compatibility
        (continuum and lines are not separated in the SVD representation).
        """
        coeffs = self.evaluate_svd_coeffs(logZ, logU, logage, logQ)
        total = self.reconstruct(coeffs)
        # Return as (continuum + lines, zero) to match the parent API
        # where caller does cont + lines
        return total, jnp.zeros_like(total)

    def svd_diagnostics(self):
        """Return diagnostic information about the nebular SVD."""
        k = self._neb_svd_k
        S = self._neb_svd_S_full
        cumvar = jnp.cumsum(S ** 2)
        totvar = cumvar[-1]

        # Reconstruction error at all grid nodes
        nz, nage, nu, kk = self._neb_svd_coeffs_cube.shape
        coeffs_flat = self._neb_svd_coeffs_cube.reshape(-1, kk)
        reconstructed = coeffs_flat @ self._neb_svd_basis

        # We need the original spectra for comparison -- re-evaluate
        # (this is only for diagnostics, not called in hot path)
        cont_cube = self.nebem_cont   # (nspec, nz, nage, nu)
        line_cube = self.nebem_line   # (nlines, nz, nage, nu)
        gauss = self.gaussnebarr      # (nspec, nlines)

        originals = []
        for iz in range(nz):
            for ia in range(nage):
                for iu in range(nu):
                    cont = 10.0 ** cont_cube[:, iz, ia, iu]
                    line_lum = 10.0 ** line_cube[:, iz, ia, iu]
                    line_spec = gauss @ line_lum
                    originals.append(cont + line_spec)
        originals = jnp.stack(originals, axis=0)

        residual = originals - reconstructed
        rms_orig = jnp.sqrt(jnp.mean(originals ** 2))
        rms_err = jnp.sqrt(jnp.mean(residual ** 2))

        return {
            'k': k,
            'variance_explained': float(cumvar[k - 1] / totvar),
            'reconstruction_error_rms': float(rms_err / rms_orig),
            'max_abs_error': float(jnp.max(jnp.abs(residual))),
            'singular_values': self._neb_svd_S,
        }
