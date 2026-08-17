"""
ceridwen/observation/base.py
============================
Abstract base class for observed SEDs.

Defines :class:`Observation`, which stores flux, uncertainty, mask, and an
optional noise model, and the GPU/JIT projection interface
(``setup_for_model`` / ``predict``) that the concrete subclasses
(``Photometry``, ``Spectrum``, ``Lines``) override.
"""

import json
import jax.numpy as jnp
import numpy as np          # only for dtype and IO, not computation
from sedpy_jax.observate import FilterSet
from sedpy_jax.smoothing import (
    make_vel_smoother,
    make_wave_smoother,
    make_lsf_smoother,
)

_CKMS = 2.998e5   # speed of light [km/s]


class Observation:
    """
    Base class for a single observed dataset.

    Parameters
    ----------
    flux : array-like, optional
        Observed data vector (maggies, F_nu, F_lambda, …).
    uncertainty : array-like, optional
        1-sigma uncertainty, same units as `flux`.
    mask : array-like of bool or slice
        True where a datum is *used* (unmasked).  Defaults to all-True.
    noise : object, optional
        Noise model instance (used by a fitter; stored but not applied here).
    name : str, optional
        Human-readable label.

    Notes
    -----
    `wavelength` is *not* set by the base class – subclasses must define it
    either as a property (Photometry: derived from filter effective wavelengths;
    Spectrum: stored directly) or as an instance attribute.
    """

    _kind          = "observation"
    logify_spectrum = False
    alias          = {}
    _meta          = ("kind", "name")
    _data          = ("wavelength", "flux", "uncertainty", "mask")

    # Class-level sentinel so `self.wavelength` is always defined without
    # raising AttributeError on bare Observation instances.
    wavelength = None

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 noise=None,
                 name=None,
                 **kwargs):

        # Unknown keyword arguments are a hard error: silently swallowing
        # them hides typos with real consequences (e.g. ``res_type=`` in
        # place of ``smoothtype=`` used to disable instrumental smoothing
        # without any sign of it).
        if kwargs:
            hints = {"res_type": "smoothtype", "restype": "smoothtype",
                     "sigma_v": "sigma_losvd", "lsf": "resolution",
                     "convention": "res_convention",
                     "resolution_convention": "res_convention",
                     "res_units": "res_convention",
                     "fwhm": "res_convention"}
            hint = "; ".join(f"did you mean {hints[k]!r} instead of {k!r}?"
                             for k in kwargs if k in hints)
            raise TypeError(
                f"{type(self).__name__}: unknown keyword argument(s) "
                f"{sorted(kwargs)}.{' ' + hint if hint else ''}"
            )

        self.flux        = None if flux        is None else jnp.asarray(flux,        dtype=float)
        self.uncertainty = None if uncertainty is None else jnp.asarray(uncertainty, dtype=float)

        # Initialise mask (bool array matching flux length, or empty)
        if isinstance(mask, slice):
            if self.flux is None:
                self.mask = jnp.array([], dtype=bool)
            else:
                m = np.zeros(len(self.flux), dtype=bool)
                m[mask] = True
                self.mask = jnp.asarray(m)
        else:
            self.mask = jnp.asarray(mask, dtype=bool)

        self.noise = noise

        if name is None:
            addr      = f"{id(self):016x}"
            self.name = f"{self.kind[:4]}-{addr[-6:]}"
        else:
            self.name = name

        self.rectify()

    # ------------------------------------------------------------------
    def __str__(self):
        wmin = (None if self.wavelength is None
                else float(jnp.min(self.wavelength)))
        wmax = (None if self.wavelength is None
                else float(jnp.max(self.wavelength)))
        lines = [
            f"Observation ({self._kind}, {self.name})",
            f"  ndata         : {self.ndata}",
            f"  ndof          : {self.ndof}",
            f"  wavelength    : {wmin} – {wmax} Å",
            f"  masked points : {self.ndata - self.ndof} / {self.ndata}",
            f"  flux finite   : "
            f"{int(jnp.sum(jnp.isfinite(self.flux))) if self.flux is not None else 0}"
            f" / {self.ndata}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' ndof={self.ndof}>"

    # ------------------------------------------------------------------
    # display() — tabular, row-by-row view of the stored data
    # ------------------------------------------------------------------
    def display(self, max_rows: int = 80, return_str: bool = False,
                file=None):
        """Print a tabular summary of every datum stored in this
        observation (flux, uncertainty, wavelength, mask, and any
        subclass-specific columns like filter or line name).

        Intended as a *sanity check* in fitting pipelines — call once
        after building the observation and before launching the sampler
        to confirm that the catalogue parsing + unit conversions
        actually produced the expected values.

        Parameters
        ----------
        max_rows : int, optional
            Cap on the number of rows printed.  Long tables (e.g. a
            Spectrum with 10 000 pixels) are head + tail truncated with
            a ``...`` marker in the middle; short tables (Photometry,
            Lines) are printed in full regardless of this limit as long
            as ``n <= max_rows``.  Default 80.
        return_str : bool, optional
            If True, return the formatted string instead of printing.
            Default False.
        file : file-like, optional
            Destination for the print when ``return_str=False``.
            Default ``sys.stdout``.

        Returns
        -------
        str or None
            The formatted string if ``return_str=True``, else ``None``.
        """
        txt = self._display_str(max_rows=max_rows)
        if return_str:
            return txt
        import sys as _sys
        print(txt, file=file or _sys.stdout)
        return None

    def _display_str(self, max_rows: int = 80) -> str:
        """Default table formatter — overridden by each subclass to add
        the columns that matter for it (filter, line name, wavelength,
        etc.).  The fall-through on the base class just returns the
        :func:`__str__` summary, which is always available."""
        return str(self)

    def __getitem__(self, item):
        k = self.alias.get(item, item)
        return getattr(self, k)

    def get(self, item, default=None):
        try:
            return self[item]
        except AttributeError:
            return default

    # ------------------------------------------------------------------
    def rectify(self):
        """
        Validate arrays and build the boolean mask.
        Called automatically at the end of ``__init__``.
        """
        if self.flux is None:
            # No data yet: skip the flux/uncertainty validation but KEEP any
            # user-supplied wavelength grid.  A flux-less container with a
            # pixel grid is the legitimate *predictive* configuration (mock
            # generation / forward modelling): setup_for_model + predict only
            # need the grid, and the data can be attached afterwards.
            # (Historically the grid was reset to None here, which silently
            # broke predictive Spectrum containers.)
            return

        assert self.flux.ndim == 1,        "flux must be 1-D"
        assert self.uncertainty is not None, "uncertainty is required when flux is provided"
        assert self.uncertainty.ndim == 1, "uncertainty must be 1-D"
        assert len(self.flux) == len(self.uncertainty), \
            "flux and uncertainty lengths differ"

        if self.wavelength is not None:
            assert self.wavelength.ndim == 1, "wavelength must be 1-D"
            assert len(self.wavelength) == len(self.flux), \
                f"wavelength length ({len(self.wavelength)}) ≠ flux length ({len(self.flux)})"

        self._automask()
        assert self.ndof > 0, "no valid unmasked data points after masking"

    # ------------------------------------------------------------------
    def _automask(self):
        """
        AND the user-supplied mask with finite-flux and positive-uncertainty
        constraints.  Pure JAX — safe inside JIT-traced code.
        """
        if self.flux is None:
            return
        valid = (
            jnp.isfinite(self.flux) &
            jnp.isfinite(self.uncertainty) &
            (self.uncertainty > 0)
        )
        self.mask = self.mask & valid

    # ------------------------------------------------------------------
    @property
    def kind(self):
        return self._kind

    @property
    def ndof(self):
        return int(jnp.sum(self.mask))

    @property
    def ndata(self):
        return 0 if self.flux is None else len(self.flux)

    @property
    def wave_min(self):
        return None if self.wavelength is None else float(jnp.min(self.wavelength))

    @property
    def wave_max(self):
        return None if self.wavelength is None else float(jnp.max(self.wavelength))

    @property
    def metadata(self):
        meta = {m: getattr(self, m, None) for m in self._meta}
        if "filternames" in meta and meta["filternames"] is not None:
            meta["filters"] = ",".join(meta["filternames"])
        return meta

    @property
    def maggies_to_nJy(self):
        """Conversion factor: 1 maggie = maggies_to_nJy nJy."""
        return 1e9 * 3631.0

    # ------------------------------------------------------------------
    # GPU / JIT-compatible projection interface
    # ------------------------------------------------------------------

    def setup_for_model(self, wave_model):
        """
        Precompute any static projection matrices that depend on the CSP
        wavelength grid ``wave_model``.

        Call this **once** after the model is constructed, before JIT-compiling
        any function that calls ``predict``.  ``SedModel.__init__`` calls it
        automatically for every registered observation.

        The base-class implementation is a no-op (``Photometry`` does not need
        precomputation because its ``FilterSet`` already holds the transmission
        curves).  Override in subclasses that need a precomputed matrix
        (``Spectrum``, ``Lines``).

        Parameters
        ----------
        wave_model : array-like, shape (n_wave,)
            Model wavelength grid [Å], as stored in ``CSPBasis.wave``.
        """
        pass   # no-op by default; overridden in Spectrum and Lines

    def predict(self, spectrum, wave_model):
        """
        Project the model ``spectrum`` onto this observation type.

        This method is called inside ``CSPBasis.predict(theta, observations)``
        which is JIT-compiled.  Implementations MUST be pure JAX, with no
        Python ``if`` / ``isinstance`` branches over traced values.

        For ``Spectrum`` and ``Lines``, this reduces to a single dense
        matrix–vector multiply using a constant matrix precomputed in
        ``setup_for_model``.  For ``Photometry``, the ``FilterSet``
        convolution is used directly.

        Parameters
        ----------
        spectrum : jax.Array, shape (n_wave,)
            Model spectrum in F_nu units (e.g. L_sun Hz^{-1} M_sun^{-1}).
        wave_model : jax.Array, shape (n_wave,)
            Model wavelength grid [Å] (same grid used in ``setup_for_model``).

        Returns
        -------
        jax.Array
            Projection result; shape depends on subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict(). "
            "Subclasses must override this method."
        )

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def to_struct(self, data_dtype=np.float32):
        """Return a NumPy structured array suitable for writing to FITS/HDF5."""
        self._automask()
        cols = []
        for c in self._data:
            dat = getattr(self, c, None)
            if dat is None:
                continue
            dat_np = np.asarray(dat)
            if dat_np.ndim != 1 or len(dat_np) != self.ndata:
                continue
            cols.append((c, dat_np.dtype))

        dtype  = np.dtype(cols)
        struct = np.zeros(self.ndata, dtype=dtype)
        for c in dtype.names:
            struct[c] = np.asarray(getattr(self, c))
        return struct

    def to_fits(self, filename=""):
        from astropy.io import fits
        hdus = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(self.to_struct()),
        ])
        for hdu in hdus:
            hdu.header.update(self.metadata)
        if filename:
            hdus.writeto(filename, overwrite=True)
        return hdus

    def to_h5_dataset(self, handle):
        dset = handle.create_dataset(self.name, data=self.to_struct())
        dset.attrs.update(self.metadata)

    def to_json(self):
        obs     = {m: getattr(self, m, None) for m in self._meta + self._data}
        convert = {
            k: (np.asarray(v).tolist() if isinstance(v, jnp.ndarray) else v)
            for k, v in obs.items()
        }
        return json.dumps(convert)


# =====================================================================
# Photometry
# =====================================================================

