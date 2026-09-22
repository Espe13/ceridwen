"""Abstract base class for observed SEDs (flux, uncertainty, mask, noise model)
and the ``setup_for_model`` / ``predict`` projection interface."""

import json
import warnings
import jax.numpy as jnp
import numpy as np


class Observation:
    """Base class for a single observed dataset.

    Parameters
    ----------
    mask : array-like of bool or slice -- True where a datum is USED (unmasked).
    noise : object -- noise model instance; stored, not applied here.
    """

    _kind          = "observation"
    logify_spectrum = False
    alias          = {}
    _meta          = ("kind", "name")
    _data          = ("wavelength", "flux", "uncertainty", "mask")

    wavelength = None

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 noise=None,
                 name=None,
                 **kwargs):

        if kwargs:
            hints = {"res_type": "instrument", "restype": "instrument",
                     "resolution": "instrument", "smoothtype": "instrument",
                     "lsf": "instrument", "res_convention": "instrument",
                     "sigma_v": "SedModel(kinematics=...)",
                     "sigma_losvd": "SedModel(kinematics=...)"}
            hint = "; ".join(f"did you mean {hints[k]!r} instead of {k!r}?"
                             for k in kwargs if k in hints)
            raise TypeError(
                f"{type(self).__name__}: unknown keyword argument(s) "
                f"{sorted(kwargs)}.{' ' + hint if hint else ''}"
            )

        self.flux        = None if flux        is None else jnp.asarray(flux,        dtype=float)
        self.uncertainty = None if uncertainty is None else jnp.asarray(uncertainty, dtype=float)

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

    def display(self, max_rows: int = 80, return_str: bool = False,
                file=None):
        """Print (or return, if ``return_str``) a per-datum table; long tables
        are head/tail truncated to ``max_rows``."""
        txt = self._display_str(max_rows=max_rows)
        if return_str:
            return txt
        import sys as _sys
        print(txt, file=file or _sys.stdout)
        return None

    def _display_str(self, max_rows: int = 80) -> str:
        return str(self)

    def __getitem__(self, item):
        k = self.alias.get(item, item)
        return getattr(self, k)

    def get(self, item, default=None):
        try:
            return self[item]
        except AttributeError:
            return default

    def rectify(self):
        """Validate arrays and build the boolean mask; a flux-less container
        keeps its wavelength grid."""
        if self.flux is None:
            return

        name = f"{type(self).__name__} {self.name!r}"
        if self.flux.ndim != 1:
            raise ValueError(f"{name}: flux must be 1-D, got shape {self.flux.shape}")
        if self.uncertainty is None:
            raise ValueError(f"{name}: uncertainty is required when flux is provided")
        if self.uncertainty.ndim != 1 or len(self.uncertainty) != len(self.flux):
            raise ValueError(f"{name}: uncertainty shape {self.uncertainty.shape} does not "
                             f"match flux shape {self.flux.shape}")
        if self.wavelength is not None and (self.wavelength.ndim != 1
                                            or len(self.wavelength) != len(self.flux)):
            raise ValueError(f"{name}: wavelength length {len(self.wavelength)} != flux "
                             f"length {len(self.flux)}")
        if self.mask.shape != self.flux.shape:
            raise ValueError(f"{name}: mask shape {self.mask.shape} != flux shape "
                             f"{self.flux.shape}")

        bad = self.mask & ~(jnp.isfinite(self.flux) & jnp.isfinite(self.uncertainty)
                            & (self.uncertainty > 0))
        if bool(jnp.any(bad)):
            idx = np.flatnonzero(np.asarray(bad))
            warnings.warn(
                f"{name}: {idx.size} data point(s) with non-finite flux or non-finite / "
                f"non-positive uncertainty were not in the mask (indices "
                f"{idx[:10].tolist()}{' ...' if idx.size > 10 else ''}); they are masked now "
                "and ignored by the likelihood", stacklevel=3)
        self._automask()
        if self.ndof <= 0:
            raise ValueError(f"{name}: no valid unmasked data points after masking")

    def _automask(self):
        """AND the mask with finite-flux and positive-uncertainty constraints."""
        if self.flux is None:
            return
        valid = (
            jnp.isfinite(self.flux) &
            jnp.isfinite(self.uncertainty) &
            (self.uncertainty > 0)
        )
        self.mask = self.mask & valid

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
        """nJy per maggie."""
        return 1e9 * 3631.0

    def setup_for_model(self, wave_model, **kwargs):
        """Precompute static projection data for the model grid ``wave_model``
        [Å] (``zred=`` is passed by ``SedModel``); call once, outside JIT.  No-op here."""
        pass

    def predict(self, spectrum, wave_model):
        """Project the model F_nu ``spectrum`` (n_wave,) onto this observation;
        must be pure JAX (called inside JIT)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict(). "
            "Subclasses must override this method."
        )

    def to_struct(self, data_dtype=np.float32):
        """Return a NumPy structured array of the data columns."""
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
