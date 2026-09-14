"""Vendored from sedpy-jax v0.1.1 (MIT, (c) 2026 Amanda Stoffers) on
2026-09-14, so that CERIDWEN carries its own filter and attenuation
machinery and has no sedpy dependency.  Behaviour is unchanged: the
module is the upstream file with its package-internal imports and data
paths repointed at ``ceridwen``.
"""
import numpy as np
import os
import jax.numpy as jnp
from typing import  List, Optional, Tuple
from jax import vmap
# Try to use rich for fancy table output
try:
    from rich.table import Table
    from rich.console import Console
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from ceridwen.observation.reference_spectra import vega, solar

from pathlib import Path

_PKGDIR = Path(__file__).resolve().parent.parent   # the ceridwen package root

lightspeed = 2.998e18  # AA/s






class Filter:
    """
    A class representing a single photometric filter, with methods to load,
    clean, rebin, and compute properties of the transmission curve.
    """

    # Constants for AB system and speed of light in Angstrom/s
    ab_gnu = 3.631e-20
    lightspeed = 2.998e18  # in AA/s
    npts = 0  # number of non-zero transmission points (set later)

    def __init__(
        self,
        kname: str = 'sdss_r0',
        nick: Optional[str] = None,
        directory: Optional[str] = None,
        dlnlam: Optional[float] = None,
        wmin: float = 1e2,
        min_trans: float = 1e-5,
        data: Optional[Tuple] = None,
        **extras
    ):
        """
        Initialize a Filter object from a filter name or provided transmission data.

        Parameters
        ----------
        kname : str
            The name of the filter (used to resolve the filename if data is not provided).

        nick : str, optional
            A short nickname for the filter (defaults to kname).

        directory : str, optional
            Directory from which to load the filter file.

        dlnlam : float, optional
            If provided, resample the transmission to a log-lambda grid.

        wmin : float
            Minimum wavelength for resampling, if dlnlam is given.

        min_trans : float
            Minimum threshold for transmission (used for trimming and masking).

        data : tuple of (wave, trans), optional
            Directly provide wavelength and transmission arrays.
        """
        self.name = kname
        self.nick = nick if nick is not None else kname
        self.min_trans = min_trans

        # Determine where to load the filter from
        self.filename = self._resolve_filename(kname, directory) if data is None else None

        if data is not None:
            wave, trans = data
            self._process_filter_data(wave, trans)
        elif self.filename is not None:
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(f"Filter transmission file '{self.filename}' does not exist.")
            self.load_filter(self.filename)

        # Optionally rebin to log-lambda grid
        if dlnlam is not None:
            self.gridify_transmission(dlnlam, wmin=wmin)

        # Compute effective wavelength, width, AB/Vega zero points, etc.
        self.get_properties()

    def __repr__(self):
        """
        Human-friendly summary of the filter. Uses `rich` if available,
        otherwise falls back to plain text.
        """
        table_data = {
            "Name": self.name,
            "Nickname": self.nick,
            "λ_eff [Å]": f"{getattr(self, 'wave_effective', 'n/a'):.1f}",
            "Pivot λ [Å]": f"{getattr(self, 'wave_pivot', 'n/a'):.1f}",
            "Width [Å]": f"{getattr(self, 'rectangular_width', 'n/a'):.1f}",
            "AB→Vega": f"{getattr(self, '_ab_to_vega', 'n/a'):.4f}",
            "N pts": self.npts,
        }

        if _HAS_RICH:
            table = Table(title=f"Filter: {self.nick}")
            for k in table_data:
                table.add_column(k)
            table.add_row(*[str(v) for v in table_data.values()])
            console = Console()
            console.print(table)
            return ""
        else:
            # Fallback to plain text table
            rows = [f"{k:>14}: {v}" for k, v in table_data.items()]
            return f"Filter: {self.nick}\n" + "\n".join(rows)

    def _resolve_filename(self, kname: str, directory: Optional[str]) -> str:
        if directory is not None:
            return os.path.join(directory, f"{kname}.par")

        try:
            from importlib.resources import files
            return str(files("ceridwen").joinpath("data", "filters", f"{kname}.par"))
        except Exception:
            # Fall back to the data directory relative to this file
            _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(_pkg_dir, "data", "filters", f"{kname}.par")
        
    @property
    def transmission(self):
        return self._transmission

    @property
    def wavelength(self):
        return self._wavelength

    def load_filter(self, filename: str):
        """Loads and processes a filter from a 2-column ASCII file."""
        wave, trans = self._read_filter_file(filename)
        self._process_filter_data(wave, trans)

    @staticmethod
    def _read_filter_file(filename: str):
        """Loads wave and trans using NumPy (non-JAX)."""
        wave, trans = np.genfromtxt(filename, usecols=(0, 1), unpack=True)
        return wave, trans

    def _process_filter_data(self, wave, trans):
        """Cleans, sorts, and assigns wavelength and transmission arrays."""
        wave = jnp.array(wave)
        trans = jnp.array(trans)

        valid = jnp.isfinite(trans) & (trans >= 0.0)
        wave_valid = wave[valid]
        trans_valid = trans[valid]

        sort_idx = jnp.argsort(wave_valid)
        self.npts = valid.sum().item() 
        self._wavelength = wave_valid[sort_idx]
        self._transmission = trans_valid[sort_idx]

        self._remove_extra_zeros(self.min_trans)

    def _remove_extra_zeros(self, min_trans=1e-5):
        """
        Trim leading/trailing transmission values that are effectively zero,
        leaving one zero point before and after the active region.
        """
        trans = self.transmission
        mask = trans > jnp.max(trans) * min_trans
        indices = jnp.where(mask)[0]

        if indices.size == 0:
            # No non-zero region found — skip trimming
            return

        start = jnp.maximum(indices[0] - 1, 0)
        stop = jnp.minimum(indices[-1] + 2, trans.shape[0])
        inds = slice(start, stop)

        self._wavelength = self._wavelength[inds]
        self._transmission = self._transmission[inds]
        self.npts = self._wavelength.shape[0]

        '''ind_min = int(np.floor((np.log(self.wavelength.min()) - np.log(wmin)) / dlnlam))
        ind_max = int(np.ceil((np.log(self.wavelength.max()) - np.log(wmin)) / dlnlam))
        lnlam = np.linspace(ind_min * dlnlam + np.log(wmin),
                            ind_max * dlnlam + np.log(wmin), ind_max - ind_min)
        lam = np.exp(lnlam)
        # TODO: replace with a rebinning
        trans = np.interp(lam, self.wavelength, self.transmission,
                          left=0., right=0.)

        self.wmin = wmin
        self.dlnlam = dlnlam
        self.inds = slice(ind_min, ind_max)
        self._wavelength = lam
        self._transmission = trans
        self.dwave = np.gradient(lam)'''
    
    def gridify_transmission(self, dlnlam, wmin=1e2):
        """Resample transmission curve onto a regular log-lambda grid."""
        wave = self.wavelength
        trans = self.transmission

        ln_wave = jnp.log(wave)
        ln_wmin = jnp.log(wmin)
        ln_wave_min = ln_wave.min()
        ln_wave_max = ln_wave.max()

        ind_min = jnp.floor((ln_wave_min - ln_wmin) / dlnlam).astype(int)

        ind_max = jnp.ceil((ln_wave_max - ln_wmin) / dlnlam).astype(int)


        lnlam_grid = jnp.linspace(ind_min * dlnlam + ln_wmin,
                                ind_max * dlnlam + ln_wmin,
                                ind_max - ind_min)
        lam_grid = jnp.exp(lnlam_grid)

        trans_grid = jax_interp(lam_grid, wave, trans, left=0.0, right=0.0)

        self.wmin = wmin
        self.dlnlam = dlnlam
        self.inds = slice(int(ind_min), int(ind_max))
        self._wavelength = lam_grid
        self._transmission = trans_grid
        self.dwave = jnp.gradient(lam_grid)

    def get_properties(self):
        props = compute_filter_properties(
            self.wavelength,
            self.transmission,
            ab_gnu=self.ab_gnu,
            lightspeed=self.lightspeed,
            vega=vega,  # ← provide externally 
            solar=solar
        )
        for k, v in props.items():
            setattr(self, k, v)
    
    @property
    def ab_to_vega(self):
        """
        Conversion from AB to Vega for this filter.
        
        m_Vega = m_AB + ab_to_vega
        """
        return self._ab_to_vega
    
    def display(self, normalize=False, ax=None):
        """Plot the filter transmission curve using matplotlib."""
        import matplotlib.pyplot as plt

        if self.npts == 0:
            return ax  # nothing to display

        if ax is None:
            _, ax = plt.subplots()
            ax.set_title(self.nick)

        y = self.transmission
        if normalize:
            y = y / jnp.max(y)

        ax.plot(self.wavelength, y)
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Transmission")
        return ax

    def obj_counts_hires(self, sourcewave, sourceflux):
        sourcewave = jnp.array(sourcewave)
        sourceflux = jnp.array(sourceflux)

        newtrans = jax_interp(sourcewave, self.wavelength, self.transmission, left=0., right=0.)

        mask = newtrans > 0.
        if not jnp.any(mask):
            return jnp.nan

        positive = jnp.where(mask)[0]
        imin = jnp.maximum(positive[0] - 1, 0)
        imax = jnp.minimum(positive[-1] + 2, len(sourcewave))
        sl = slice(imin, imax)

        integrand = sourcewave[sl] * newtrans[sl] * sourceflux[..., sl]
        counts = jnp.trapezoid(integrand, sourcewave[sl], axis=-1)
        return counts
    
    def obj_counts_lores(self, sourcewave, sourceflux):
        sourcewave = jnp.array(sourcewave)
        sourceflux = jnp.squeeze(jnp.array(sourceflux))

        assert sourceflux.ndim == 1, "Only single source spectrum allowed in lores mode."

        newflux = jax_interp(self.wavelength, sourcewave, sourceflux, left=0., right=0.)

        mask = newflux > 0.
        if not jnp.any(mask):
            return jnp.nan

        integrand = self.wavelength * self.transmission * newflux
        counts = jnp.trapezoid(integrand, self.wavelength)
        return counts
    
    def obj_counts_grid(self, sourceflux, source_offset=None):
        sourceflux = jnp.array(sourceflux)

        if source_offset is None:
            valid = self.inds
        else:
            valid = slice(self.inds.start + source_offset, self.inds.stop + source_offset)

        sl_flux = sourceflux[..., valid]
        integrand = sl_flux * self.transmission * self.wavelength * self.dwave
        counts = jnp.sum(integrand, axis=-1)
        return counts
    
    def obj_counts(self, sourcewave, sourceflux, lores=False, gridded=False, **extras):
        if gridded:
            return self.obj_counts_grid(sourceflux, **extras)
        elif lores:
            return self.obj_counts_lores(sourcewave, sourceflux)
        else:
            return self.obj_counts_hires(sourcewave, sourceflux)

    def ab_mag(self, sourcewave, sourceflux, **extras):
        """
        Compute the AB magnitude of the source(s) through this filter.
        """
        counts = self.obj_counts(sourcewave, sourceflux, **extras)
        return -2.5 * jnp.log10(counts / self.ab_zero_counts)

    def vega_mag(self, sourcewave, sourceflux, **extras):
        """
        Compute the Vega magnitude of the source(s) through this filter.
        """
        counts = self.obj_counts(sourcewave, sourceflux, **extras)
        return -2.5 * jnp.log10(counts / self.vega_zero_counts)



class FilterSet:
    """
    A collection of filters that share a common wavelength grid, enabling
    fast, vectorized projection of source spectra onto multiple filters.

    Attributes
    ----------
    filters : List[Filter]
        The list of Filter objects in this set.

    trans : (N_filters, N_lam) array
        Precomputed transmission matrix for efficient dot product with source spectra.

    lam : (N_lam,) array
        Shared wavelength grid for all filters.

    filternames : List[str]
        Names of the filters included in the set.
    """

    def __init__(
        self,
        filterlist: List[str],
        wmin: Optional[float] = None,
        dlnlam: Optional[float] = None,
        **loading_kwargs
    ):
        """
        Initialize the FilterSet from a list of filter names.

        Parameters
        ----------
        filterlist : list of str
            Names of the filters to include in the set.

        wmin : float, optional
            Minimum wavelength for the shared grid. If None, inferred from filters.

        dlnlam : float, optional
            Logarithmic spacing for the shared wavelength grid. If None, inferred.

        loading_kwargs : dict
            Additional keyword arguments passed to the filter loader.
        """
        self.filternames = filterlist

        # Load the native filters (may have different wavelength grids initially)
        native = load_filters(self.filternames, **loading_kwargs)

        # Resample filters onto a shared wavelength grid
        self._set_filters(native, wmin=wmin, dlnlam=dlnlam, **loading_kwargs)

        # Build the matrix for fast SED projections: (N_filters, N_lam)
        self._build_super_trans()

    def __repr__(self):
        """
        Print a summary table of all filters in the FilterSet.
        Uses `rich` if available, falls back to plain text otherwise.
        """
        rows = []
        for i, f in enumerate(self.filters):
            rows.append({
                "Index": str(i),
                "Name": f.name,
                "λ_eff [Å]": f"{getattr(f, 'wave_effective', 0):.1f}",
                "Width [Å]": f"{getattr(f, 'rectangular_width', 0):.1f}",
                "AB→Vega": f"{getattr(f, '_ab_to_vega', 0):.4f}"
            })

        if _HAS_RICH:
            table = Table(title="FilterSet Summary")
            for col in rows[0].keys():
                table.add_column(col)
            for row in rows:
                table.add_row(*row.values())
            console = Console()
            console.print(table)
            return ""
        else:
            header = "Index | Name       | λ_eff [Å] | Width [Å] | AB→Vega"
            lines = [header, "-" * len(header)]
            for row in rows:
                line = f"{row['Index']:>5} | {row['Name']:<10} | {row['λ_eff [Å]']:>10} | {row['Width [Å]']:>9} | {row['AB→Vega']:>8}"
                lines.append(line)
            return "\n".join(lines)

    def _set_filters(self, native, wmin=None, wmax=None, dlnlam=None, **loading_kwargs):
        """
        Set filters and initialize shared log-wavelength grid for gridded projections.
        """
        # Native grid spacings (min ∆lnλ from each filter)
        self.dlnlam_native = jnp.array([jnp.diff(jnp.log(f.wavelength)).min()
                                        for f in native])

        # Choose global dlnlam
        if dlnlam is None:
            dlnlam = jnp.minimum(jnp.min(self.dlnlam_native), 1e-3)

        # Set wavelength range
        if wmin is None:
            wmin = jnp.min(jnp.array([f.wavelength[0] for f in native]))

        if wmax is None:
            wmax = jnp.max(jnp.array([f.wavelength[-1] for f in native]))

        self.wmin = wmin
        self.dlnlam = dlnlam
        self.wmax = jnp.exp(jnp.log(wmax) + dlnlam)  # pad upper edge

        # Build global wavelength grid
        self.lnlam = jnp.arange(jnp.log(wmin), jnp.log(self.wmax), dlnlam)
        self.lam = jnp.exp(self.lnlam)


        # Reload filters onto this shared grid
        self.filters = load_filters(self.filternames, wmin=wmin, dlnlam=dlnlam, **loading_kwargs)
        
    def _build_super_trans(self):
        """
        Build (n_filters, n_lam) matrix where each row contains:
        R * λ * Δλ / ab_zero_counts in the active slice, 0 elsewhere.
        """
        n_filters = len(self.filters)
        n_lam = len(self.lam)  # global wavelength grid length

        ab_counts = jnp.array([f.ab_zero_counts for f in self.filters])
        inds_start = jnp.array([f.inds.start for f in self.filters])
        inds_stop  = jnp.array([f.inds.stop for f in self.filters])

        # Stack padded transmission, wavelength, and dwave arrays
        trans = jnp.stack([pad_to(self.lam, f.transmission, f.inds) for f in self.filters])
        wave  = jnp.stack([pad_to(self.lam, f.wavelength, f.inds) for f in self.filters])
        dwave = jnp.stack([pad_to(self.lam, jnp.gradient(f.wavelength), f.inds) for f in self.filters])

        def compute_row(trans_j, wave_j, dwave_j, ab_j, istart, istop):
            full = trans_j * wave_j * dwave_j / ab_j
            mask = (jnp.arange(n_lam) >= istart) & (jnp.arange(n_lam) < istop)
            return jnp.where(mask, full, 0.0)

        self.trans = vmap(compute_row)(trans, wave, dwave, ab_counts, inds_start, inds_stop)

        self.trans_list = [self.trans[i, inds_start[i]:inds_stop[i]] for i in range(n_filters)]
        self.frange = jnp.stack([inds_start, inds_stop], axis=1)

    def interp_source(self, inwave, sourceflux):
        """
        Interpolate input source SEDs onto the FilterSet's wavelength grid.
        inwave: (N_pix,)
        sourceflux: (N_source, N_pix) or (N_pix,)
        Returns: interpolated_flux: (N_source, N_lam)
        """
        inwave = jnp.array(inwave)
        sourceflux = jnp.atleast_2d(jnp.array(sourceflux))  # (N_source, N_pix)
        lam_grid = self.lam  # (N_lam,)

        # vmap over each SED row
        interp_fn = lambda sflux: jax_interp(lam_grid, inwave, sflux, left=0.0, right=0.0)
        interp_flux = vmap(interp_fn)(sourceflux)  # (N_source, N_lam)

        return interp_flux.T
    
    def get_sed_maggies(self, sourceflux, sourcewave=None):
        """
        Project a set of source SEDs onto the filter response matrix to get
        fluxes in maggies (AB units).

        Parameters
        ----------
        sourceflux : (N_source, N_pix) or (N_pix,) JAX array
            Source flux in erg/s/cm^2/Å

        sourcewave : (N_pix,), optional
            Input wavelength grid. If None, assumes it matches self.lam.

        Returns
        -------
        maggies : (N_filter,) or (N_source, N_filter) JAX array
        """
        sourceflux = jnp.atleast_2d(sourceflux)  # ensure (N_source, N_pix)

        if sourcewave is not None:
            interp_flux = self.interp_source(sourcewave, sourceflux)  # (N_source, N_lam)
        else:
            assert sourceflux.shape[1] == self.lam.shape[0], \
                "Input flux must be on the same wavelength grid as FilterSet.lam"
            interp_flux = sourceflux

        # (N_source, N_filter) = (N_source, N_lam) @ (N_filter, N_lam).T
        maggies = jnp.dot(interp_flux.T, self.trans.T)

        # Return shape: (N_filter,) if only one SED, else (N_source, N_filter)
        return maggies.squeeze()


    def display(self, normalize=False, ax=None, colormap="coolwarm"):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize
        """
        Plot all filters in the FilterSet, colored by a colormap based on
        each filter's effective wavelength.

        Parameters
        ----------
        normalize : bool
            Whether to normalize each transmission curve.

        ax : matplotlib.axes.Axes, optional
            Axes to plot into. If None, creates new figure and axes.

        colormap : str or Colormap, optional
            A matplotlib colormap name or object used to color the filters
            based on their effective wavelengths.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Extract effective wavelengths
        wave_eff = jnp.array([f.wave_effective for f in self.filters])

        # Set up color normalization and colormap
        cmap = cm.get_cmap(colormap)
        norm = Normalize(vmin=wave_eff.min(), vmax=wave_eff.max())
        colors = [cmap(norm(w)) for w in wave_eff]

        # Plot each filter with corresponding color
        for f, color in zip(self.filters, colors):
            trans = f.transmission / jnp.max(f.transmission) if normalize else f.transmission
            ax.plot(f.wavelength, trans, label=f.nick, color=color)
            ax.fill_between(f.wavelength, 0, trans, color="lightgray", alpha=0.4, zorder=0)

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Transmission" + (" (normalized)" if normalize else ""))
        ax.set_title("FilterSet Transmission Curves")
        ax.legend(fontsize=8, ncol=2, loc="best") 

        return ax


def compute_filter_properties(wave, trans, ab_gnu, lightspeed, vega=None, solar=None):
    logwave = jnp.log(wave)
    tmax = jnp.max(trans)

    i0 = jnp.trapezoid(trans * logwave, logwave)
    i1 = jnp.trapezoid(trans, logwave)
    i2 = jnp.trapezoid(trans * wave, wave)
    i3 = jnp.trapezoid(trans, wave)

    wave_effective = jnp.exp(i0 / i1)
    wave_pivot = jnp.sqrt(i2 / i1)
    wave_average = i2 / i3
    rectangular_width = i3 / tmax
    wpeak = wave[jnp.argmax(trans)]

    sel_blue = (wave < wpeak) & (trans < 0.5 * tmax)
    blue_edge = jnp.where(sel_blue, wave, -1)
    blue_edge = blue_edge[jnp.argmax(blue_edge)]

    sel_red = (wave > wpeak) & (trans < 0.5 * tmax)
    red_edge = jnp.where(sel_red, wave, 1e10)
    red_edge = red_edge[jnp.argmin(red_edge)]

    i4 = jnp.trapezoid(trans * (jnp.log(wave / wave_effective))**2, logwave)
    gauss_width = jnp.sqrt(i4 / i1)
    effective_width = 2.0 * jnp.sqrt(2. * jnp.log(2.)) * gauss_width * wave_effective

    flux_ab = ab_gnu * lightspeed / (wave**2)
    ab_zero_counts = jnp.trapezoid(wave * flux_ab * trans, wave)
    ab_to_vega = jnp.nan
    vega_zero_counts = jnp.nan
    solar_ab_mag = jnp.nan

    if wave_effective < 1e6 and vega is not None:
        vega_flux = jax_interp(wave, vega[:, 0], vega[:, 1])
        vega_zero_counts = jnp.trapezoid(wave * vega_flux * trans, wave)
        ab_to_vega = -2.5 * jnp.log10(ab_zero_counts / vega_zero_counts)

    if wave_effective < 1e5 and solar is not None:
        solar_flux = jax_interp(wave, solar[:, 0], solar[:, 1])
        flux_ratio = wave * solar_flux * trans
        solar_counts = jnp.trapezoid(flux_ratio, wave)
        solar_ab_mag = -2.5 * jnp.log10(solar_counts / ab_zero_counts)
        
    return {
        "wave_effective": wave_effective,
        "wave_pivot": wave_pivot,
        "wave_mean": wave_effective,
        "wave_average": wave_average,
        "rectangular_width": rectangular_width,
        "blue_edge": blue_edge,
        "red_edge": red_edge,
        "gauss_width": gauss_width,
        "effective_width": effective_width,
        "ab_zero_counts": ab_zero_counts,
        "vega_zero_counts": vega_zero_counts,
        "_ab_to_vega": ab_to_vega,
        "solar_ab_mag": solar_ab_mag
    }

def jax_interp(x, xp, fp, left=0.0, right=0.0):
    """JAX-compatible 1D linear interpolation like np.interp"""
    idx = jnp.searchsorted(xp, x, side='left') - 1
    idx = jnp.clip(idx, 0, len(xp) - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x - x0)
    
    y = jnp.where(x < xp[0], left, y)
    y = jnp.where(x > xp[-1], right, y)
    return y
        
def pad_to(grid, array, inds):
    """Pad a filter's array into a full-length grid array using slice `inds`."""
    full = jnp.zeros_like(grid)
    full = full.at[inds].set(array)
    return full

def load_filters(filternamelist, **kwargs):
    """Given a list of filter names, this method returns a list of Filter
    objects.

    Parameters
    ----------
    filternamelist : list of strings
        The names of the filters.

    Returns
    -------
    filterlist : list of Filter() instances
        A list of filter objects.
    """
    return [Filter(f, **kwargs) for f in filternamelist]

def getSED(sourcewave, sourceflux, filterset=None, filterlist = None, linear_flux=False, **kwargs):
    """
    Project a source spectrum (or spectra) onto a filterlist or FilterSet,
    returning either AB magnitudes or maggies.

    Parameters
    ----------
    sourcewave : (N_pix,) array
        Wavelengths in Angstroms

    sourceflux : (N_pix,) or (N_source, N_pix)
        Input spectrum or batch of spectra

    filterset : FilterSet instance (JAX-compatible) or list of Filter objects
        Filters to use

    linear_flux : bool (default=False)
        If True, return maggies (linear flux). If False, return AB magnitudes.

    Returns
    -------
    sed : (N_filter,) or (N_source, N_filter)
        Output SED in AB magnitudes or maggies
    """
    # Use FilterSet — optimized path
    if hasattr(filterset, "get_sed_maggies"):
        maggies = filterset.get_sed_maggies(sourceflux, sourcewave=sourcewave)
        return maggies if linear_flux else -2.5 * jnp.log10(jnp.clip(maggies, 1e-30))

    elif filterset is None:
        return None

    # Fallback: loop over individual Filter objects (NumPy-based)
    # but we should only allow for filtersets.
    import numpy as np
    sourceflux = np.squeeze(np.array(sourceflux))
    sedshape = sourceflux.shape[:-1] + (len(filterlist),)
    sed = np.zeros(sedshape)

    for i, f in enumerate(filterlist):
        sed[..., i] = f.ab_mag(sourcewave, sourceflux, **kwargs)

    if linear_flux:
        return 10**(-0.4 * sed)
    else:
        return sed
    
def list_available_filters(startswith=None):
    """
    Return a sorted list of available filter names.

    Parameters
    ----------
    startswith : str, optional
        If given, only return filters whose names start with this string.

    Returns
    -------
    list of str
        Filter names without the '.par' extension.
    """
    import os
    from importlib.resources import files

    try:
        filter_dir = files("ceridwen").joinpath("data/filters")
        names = [f.name for f in filter_dir.iterdir() if f.name.endswith(".par")]
    except Exception:
        names = os.listdir(os.path.join(_PKGDIR, "data", "filters"))

    parfiles = sorted(n[:-4] for n in names if n.endswith(".par"))

    if startswith:
        parfiles = [name for name in parfiles if name.startswith(startswith)]

    return parfiles

def Lbol(wave, spec, wave_min=90.0, wave_max=1e6):
    """
    Compute the bolometric luminosity by integrating F_lambda over a wavelength range.

    Parameters
    ----------
    wave : (N_wave,) array
        Wavelength array in Angstroms.

    spec : (..., N_wave) array
        Spectrum or spectra in F_lambda units.

    wave_min : float
        Lower bound for integration [Å].

    wave_max : float
        Upper bound for integration [Å].

    Returns
    -------
    lbol : array of shape (...)
        Bolometric luminosity integrated over wavelength.
    """
    wave = jnp.array(wave)
    spec = jnp.array(spec)

    mask = (wave >= wave_min) & (wave < wave_max)
    wave_sel = wave[mask]
    spec_sel = spec[..., mask]

    return jnp.trapz(spec_sel, wave_sel, axis=-1)

def air2vac(air):
    """
    Convert from in-air to vacuum wavelengths (JAX version).
    Based on Allen's Astrophysical Quantities.

    Parameters
    ----------
    air : (N_pix,) array
        In-air wavelengths in Angstroms.

    Returns
    -------
    vac : (N_pix,) array
        Vacuum wavelengths in Angstroms.
    """
    ss = 1e4 / air
    vac = air * (
        1 + 6.4328e-5 +
        2.94981e-2 / (146 - ss**2) +
        2.5540e-4 / (41 - ss**2)
    )
    return vac

def vac2air(vac):
    """
    Convert from vacuum to in-air wavelengths (JAX version).
    Based on Morton 1991 ApJS (used by SDSS).

    Parameters
    ----------
    vac : (N_pix,) array
        Vacuum wavelengths in Angstroms.

    Returns
    -------
    air : (N_pix,) array
        In-air wavelengths in Angstroms.
    """
    conv = (
        1.0 +
        2.735182e-4 +
        131.4182 / vac**2 +
        2.76249e8 / vac**4
    )
    return vac / conv




# _________NOT TESTED_________

def rebin(outwave, wave, trans):
    """
    Rebin a transmission array onto a new wavelength grid, conserving total transmission.
    Works best when output bins are coarser than the input.

    Parameters
    ----------
    outwave : (N_out,) array
        Output wavelength grid (assumed to be centers).
    wave : (N_in,) array
        Native wavelength grid (assumed to be centers).
    trans : (N_in,) array
        Native transmission values.

    Returns
    -------
    rebinned : (N_out,) array
        Rebinned transmission values on the output grid.
    """
    # Calculate edges of bins from centers
    def centers_to_edges(x):
        mid = 0.5 * (x[1:] + x[:-1])
        first = 2 * mid[0] - mid[1]
        last = 2 * mid[-1] - mid[-2]
        return jnp.concatenate([[first], mid, [last]])

    in_edges = centers_to_edges(wave)
    out_edges = centers_to_edges(outwave)

    inlo, inhi = in_edges[:-1], in_edges[1:]
    outlo, outhi = out_edges[:-1], out_edges[1:]

    # Compute overlap between input and output bins
    l_inf = jnp.maximum(outlo[:, None], inlo[None, :])  # (N_out, N_in)
    l_sup = jnp.minimum(outhi[:, None], inhi[None, :])  # (N_out, N_in)
    overlap = jnp.clip(l_sup - l_inf, a_min=0.0)         # (N_out, N_in)

    in_widths = inhi - inlo                             # (N_in,)
    resamp_mat = overlap * in_widths                   # weight by input bin widths

    # Mask edges where output bins fall outside input range
    valid = (outlo >= inlo[0]) & (outhi <= inhi[-1])   # (N_out,)
    norm = jnp.sum(resamp_mat, axis=-1)                # (N_out,)
    rebinned = jnp.dot(resamp_mat, trans) / jnp.where(norm > 0, norm, 1.0)

    return jnp.where(valid, rebinned, 0.0)