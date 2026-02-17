import json
import jax.numpy as jnp
import numpy as np          # only for dtype and IO, not computation
from sedpy_jax.observate import FilterSet 

class Observation:

    _kind = "observation"
    logify_spectrum = False
    alias = {}
    _meta = ("kind", "name")
    _data = ("wavelength", "flux", "uncertainty", "mask")

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 noise=None,
                 name=None,
                 **kwargs):

        # convert to JAX arrays
        self.flux = None if flux is None else jnp.asarray(flux)
        self.uncertainty = None if uncertainty is None else jnp.asarray(uncertainty)

        # handle initial mask
        if isinstance(mask, slice):
            # convert slice → boolean mask
            if self.flux is None:
                self.mask = jnp.array([], dtype=bool)
            else:
                m = np.zeros(len(self.flux), dtype=bool)
                m[mask] = True
                self.mask = jnp.asarray(m)
        else:
            self.mask = jnp.asarray(mask, dtype=bool)

        self.noise = noise

        # name
        if name is None:
            addr = f"{id(self):04x}"
            self.name = f"{self.kind[:5]}-{addr[:6]}"
        else:
            self.name = name

        # validate + auto-mask
        self.rectify()

    # ------------------------------------------------------------
    def __str__(self):
        info = [
            f"Observation object: {self.kind} ({self.name})",
            f"  ndata         : {self.ndata}",
            f"  ndof          : {self.ndof}",
            f"  wavelength min: {None if not hasattr(self, 'wavelength') or self.wavelength is None else float(jnp.min(self.wavelength))}",
            f"  wavelength max: {None if not hasattr(self, 'wavelength') or self.wavelength is None else float(jnp.max(self.wavelength))}",
            f"  masked points : {int(jnp.sum(~self.mask))} / {self.ndata}",
            f"  flux finite   : {int(jnp.sum(jnp.isfinite(self.flux)))} / {self.ndata}",
            f"  unc finite    : {int(jnp.sum(jnp.isfinite(self.uncertainty)))} / {self.ndata}",
            f"  noise model   : {self.noise.__class__.__name__}"
        ]
        return "\n".join(info)

    def __getitem__(self, item):
        k = self.alias.get(item, item)
        return getattr(self, k)

    def get(self, item, default):
        try:
            return self[item]
        except AttributeError:
            return default


    # ------------------------------------------------------------
    def rectify(self):
        """
        Validation and automatic masking (host-side).
        JAX arrays are OK here — no JIT paths.
        """
        n = self.__repr__

        if self.flux is None:
            print(f"{n} has no data")
            self.wavelength = None
            return

        # Ensure 1D arrays
        assert self.flux.ndim == 1, "flux must be 1D"
        assert self.uncertainty.ndim == 1, "uncertainty must be 1D"
        assert len(self.flux) == len(self.uncertainty), \
            "flux and uncertainty lengths differ"

        if self.wavelength is not None:
            assert self.wavelength.ndim == 1, "wavelength must be 1D"
            assert len(self.wavelength) == len(self.flux), \
                "wavelength length mismatch"

        # update mask using JAX
        self._automask()

        assert self.ndof > 0, "no valid unmasked datapoints"
        assert hasattr(self, "noise")

    # ------------------------------------------------------------
    def _automask(self):
        """JAX-compatible auto-mask; no numpy ops."""
        if self.flux is None:
            return

        # base mask
        base = self.mask

        # new valid mask
        valid = (
            jnp.isfinite(self.flux) &
            jnp.isfinite(self.uncertainty) &
            (self.uncertainty > 0)
        )

        self.mask = base & valid


    # ------------------------------------------------------------
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
        meta = {m: getattr(self, m) for m in self._meta}
        if "filternames" in meta:
            meta["filters"] = ",".join(meta["filternames"])
        return meta

    # ------------------------------------------------------------
    def to_struct(self, data_dtype=np.float32):
        """
        Needs numpy structured array → convert jax → numpy
        (This is pure IO; safe.)
        """
        # force auto-mask to run
        self._automask()

        cols = []
        for c in self._data:
            dat = getattr(self, c)
            if dat is None:
                continue
            dat_np = np.asarray(dat)
            if len(dat_np) != self.ndata:
                continue
            cols.append((c, dat_np.dtype))

        dtype = np.dtype(cols)
        struct = np.zeros(self.ndata, dtype=dtype)

        for c in dtype.names:
            struct[c] = np.asarray(getattr(self, c))

        return struct

    def to_fits(self, filename=""):
        from astropy.io import fits
        hdus = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(self.to_struct())
        ])
        for hdu in hdus:
            hdu.header.update(self.metadata)
        if filename:
            hdus.writeto(filename, overwrite=True)

    def to_h5_dataset(self, handle):
        dset = handle.create_dataset(self.name, data=self.to_struct())
        dset.attrs.update(self.metadata)

    def to_json(self):
        obs = {m: getattr(self, m) for m in self._meta + self._data}
        convert = {k: (np.asarray(v).tolist() if isinstance(v, jnp.ndarray)
                       else v)
                   for k, v in obs.items()}
        return json.dumps(convert)

    @property
    def maggies_to_nJy(self):
        return 1e9 * 3631.0


# =====================================================================
# Subclass: PHOTOMETRY (unchanged API, JAX arrays inside)
# =====================================================================

class Photometry(Observation):

    _kind = "photometry"
    alias = dict(maggies="flux",
                 maggies_unc="uncertainty",
                 filters="filters",
                 phot_mask="mask")
    _meta = ("kind", "name", "filternames")

    def __init__(self, filters=[],
                 name=None,
                 **kwargs):

        self.set_filters(filters)
        super().__init__(name=name, **kwargs)

    def set_filters(self, filters):
        if (len(filters) == 0) or (filters is None):
            self.filters = filters
            self.filternames = []
            self.filterset = None
            return

        try:
            self.filternames = [f.name for f in filters]
        except AttributeError:
            self.filternames = filters

        self.filterset = FilterSet(self.filternames)
        self.filters = [f for f in self.filterset.filters]

    @property
    def wavelength(self):
        # convert to JAX array later if needed
        return jnp.asarray([f.wave_effective for f in self.filters])