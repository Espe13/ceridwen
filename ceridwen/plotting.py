"""Per-galaxy figures from a ``PostProcess`` output: a summary page (SED,
lines, SFH, marginals), a corner plot and a sampling-diagnostic page.
Plain matplotlib; the colours in ``COLORS`` are the only styling."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["COLORS", "summary_figure", "corner_figure", "diagnostic_figure", "make_figures"]

COLORS = {
    "posterior": "#2f4fa0",     # medians, model lines
    "band":      "#8fa5d9",     # posterior 16-84 % bands
    "band2":     "#c7d1ec",     # posterior 2.5-97.5 % bands
    "prior":     "#d9dbe3",     # prior 16-84 % shading
    "data":      "#1f1f1f",     # observed points
    "bestfit":   "#b03a2e",     # maximum-likelihood sample
    "grid":      "#9a9a9a",
    "truth":     "#2e8b57",     # injected values (mock tests)
    "chains":    ["#2f4fa0", "#5b8def", "#0f2a6b", "#7aa6f0", "#1c3b8a", "#a3bff5"],
}

_MAGGIE_TO_NJY = 3.631e12
_CGS_FNU_TO_NJY = 1e32
# 1 maggie = 3631 Jy = 3.631e-20 erg s^-1 cm^-2 Hz^-1: Photometry divides the band-averaged
# F_nu by this, also for a zred = 0 model whose F_nu is in model units (L_sun/Hz x 10^logmass),
# so the same factor brings such "maggies" back to the spectrum's units.
_MAGGIE_TO_CGS = 3.631e-20


def _visible_blues():
    """matplotlib's Blues without its near-white end (the lowest weights stay visible)."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    base = matplotlib.colormaps["Blues"]
    return LinearSegmentedColormap.from_list("ceridwen_blues", base(np.linspace(0.3, 1.0, 256)))


class _LazyCmap:
    """Build the colormap on first use, so importing this module does not import matplotlib."""
    _cmap = None

    def __call__(self):
        if _LazyCmap._cmap is None:
            _LazyCmap._cmap = _visible_blues()
        return _LazyCmap._cmap
_LABELS = {
    "logmass": r"$\log M_\mathrm{formed}/\mathrm{M}_\odot$", "zred": r"$z$",
    "logzsol": r"$\log(Z_\star/\mathrm{Z}_\odot)$",
    "logzsol_hist": r"$\log(Z_\star/\mathrm{Z}_\odot)$",
    "logzsol_total": r"$[Z/\mathrm{H}]$",
    "gas_logz": r"$\log(Z_\mathrm{gas}/\mathrm{Z}_\odot)$", "gas_logu": r"$\log U$",
    "frac_obrun": r"$f_\mathrm{esc}$", "eline_scaling": r"$s_\mathrm{line}$",
    "spectrum_scaling": r"$s_\mathrm{spec}$", "sigma_gal": r"$\sigma_\star$ [km/s]", "sigma_gas": r"$\sigma_\mathrm{gas}$ [km/s]",
    "igm_factor": r"$f_\mathrm{IGM}$", "afe": r"[$\alpha$/Fe]",
    "x_HI": r"$x_\mathrm{HI}$", "logN_HI": r"$\log N_\mathrm{HI}$ [cm$^{-2}$]",
    "z_dla": r"$z_\mathrm{DLA}$",
}


_FEH_LABELS = {"logzsol": r"$[\mathrm{Fe}/\mathrm{H}]$",
               "logzsol_hist": r"$[\mathrm{Fe}/\mathrm{H}]$"}


def _label(name: str, feh: bool = False) -> str:
    """LaTeX label; ``feh`` (an alpha / MIST grid, axis_meaning 'feh') labels logzsol [Fe/H]."""
    labels = {**_LABELS, **(_FEH_LABELS if feh else {})}
    if name in labels:
        return labels[name]
    base, _, idx = name.partition("[")
    if base in labels and idx:
        return labels[base][:-1] + rf"_{{{idx[:-1]}}}$"
    return name.replace("_", " ")


def _feh_axis(out) -> bool:
    """True when the fit's grid labels logzsol as [Fe/H] (MIST / aMIST)."""
    try:
        return out["meta"]["metallicity"]["axis_meaning"] == "feh"
    except (KeyError, TypeError):
        return False


def _flat_params(theta: dict, names: Optional[Sequence[str]] = None):
    """{column label: (N,) array} for every scalar parameter and vector element."""
    cols = {}
    for k in (names or theta.keys()):
        a = np.asarray(theta[k])
        if a.ndim == 1:
            cols[k] = a
        else:
            for j in range(a.shape[1]):
                cols[f"{k}[{j}]"] = a[:, j]
    return cols


def _flat_point(theta: dict, names) -> dict:
    out = {}
    for k in names:
        a = np.atleast_1d(np.asarray(theta[k], dtype=float))
        if a.size == 1:
            out[k] = float(a[0])
        else:
            for j in range(a.size):
                out[f"{k}[{j}]"] = float(a[j])
    return out


def _quantiles(x, q=(0.16, 0.5, 0.84)):
    return np.nanquantile(np.asarray(x, dtype=float), q, axis=0)


def _weighted_quantiles(x, w, q=(0.16, 0.5, 0.84)):
    """Quantiles of ``x`` under the normalised weights ``w`` (points with w = 0 or a
    non-finite x carry no mass)."""
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o]); c /= c[-1]
    return x[o][np.minimum(np.searchsorted(c, q), x.size - 1)]


def _fmt_q(lo, med, hi, nd=2):
    return rf"${med:.{nd}f}^{{+{hi - med:.{nd}f}}}_{{-{med - lo:.{nd}f}}}$"


def _prior_draws(model, n, seed=0):
    """Free-parameter draws from the model priors (None when a parameter has no prior)."""
    import jax
    key = jax.random.PRNGKey(seed)
    draws = {}
    for name in model.param_names:
        if name not in model.priors:
            return None
        key, sub = jax.random.split(key)
        shape = tuple(np.shape(model.theta_init[name]))
        from .sampler.priors import sample_for_parameter
        try:
            draws[name] = np.asarray(sample_for_parameter(model.priors[name], sub, n, shape))
        except ValueError:                 # prior shape does not fit the parameter
            return None
    return draws


def _nice_log_ticks(axis, lo, hi):
    """Labelled ticks at 1, 2, 3, 5 x 10^k inside [lo, hi] on a log axis, plain numbers, no
    minor labels (matplotlib's defaults give one label per decade or 'a x 10^b' clutter)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ticks = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1,
                                          int(np.ceil(np.log10(hi))) + 1)
             for m in (1, 2, 3, 5) if lo <= m * 10.0 ** k <= hi]
    if len(ticks) > 9:                         # wide ranges: 1 and 3 only
        ticks = [t for t in ticks if f"{t:.0e}"[0] in "13"]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _truth_sfr_per_bin(model, out, truths):
    """Per-bin physical SFR [M_sun/yr] of the injected parameters, or None.  Drawn only when
    ``truths`` gives the SFH parameters (a vector-valued free parameter such as
    ``logsfr_ratios``); any other free parameter it lacks takes its posterior median."""
    if not truths:
        return None
    vec = [k for k in model.param_names if np.size(model.theta_init[k]) > 1]
    if not vec or not all(k in truths for k in vec):
        return None
    import jax.numpy as jnp
    from .postprocess import _per_bin_and_nodes, model_theta
    free = {}
    for k in model.param_names:
        v = truths[k] if k in truths else np.median(np.asarray(out["theta"][k]), axis=0)
        free[k] = jnp.asarray(np.reshape(np.asarray(v, dtype=float),
                                         np.shape(model.theta_init[k])))
    t = model_theta(model, free)
    psi = np.ravel(np.asarray(t["sfh"], dtype=float))
    n_time = int(np.asarray(out["extras"]["sfh"]["lookback_gyr"]).shape[-1])
    bar, _nodes = _per_bin_and_nodes(psi, n_time)
    return bar * (10.0 ** float(np.ravel(np.asarray(t["logmass"]))[0]) if "logmass" in t else 1.0)


def _data_wave_range(obs_by_kind):
    """(lo, hi) observed wavelength [um] covered by the photometry and spectra, padded by
    ~25 % in log; None when there are none (lines only)."""
    lo, hi = [], []
    for o in obs_by_kind["photometry"]:
        w = np.asarray(o.wavelength, dtype=float)
        fs = getattr(o, "filterset", None)
        edges = None
        if fs is not None and hasattr(fs, "filters"):
            try:
                edges = [(float(np.min(f.wavelength[f.transmission > 0.01 * f.transmission.max()])),
                          float(np.max(f.wavelength[f.transmission > 0.01 * f.transmission.max()])))
                         for f in fs.filters]
            except Exception:
                edges = None
        if edges:
            lo.append(min(e[0] for e in edges)); hi.append(max(e[1] for e in edges))
        elif w.size:
            lo.append(w.min()); hi.append(w.max())
    for o in obs_by_kind["spectrum"]:
        w = np.asarray(o.wavelength, dtype=float)
        if w.size:
            lo.append(w.min()); hi.append(w.max())
    if not lo:
        return None
    return min(lo) / 1e4 / 1.25, max(hi) / 1e4 * 1.25


def summary_figure(out, model, *, title=None, prior_draws=500, params=None, truths=None,
                   savepath=None, figsize=(15, 11)):
    """Summary page: SED with residuals (photometry and model in the same units, limited to
    the observed wavelengths), emission lines, the SFH (per-bin log10 SFR vs lookback time,
    posterior 16-84 % and median), and 1-D marginals of the fitted parameters (median,
    16-84 %, best fit, prior 16-84 %).  ``out`` is ``PostProcess.run()``'s dict; ``model``
    the fitted ``SedModel``; ``truths`` ({name: value or array}) marks injected values in
    green, and draws the injected SFH when it gives the SFH parameters (e.g. logsfr_ratios)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    C = COLORS
    theta = out["theta"]
    feh = _feh_axis(out)
    cols = _flat_params(theta, params)
    best = _flat_point(out["bestfit"]["theta"], (params or list(theta.keys())))
    prior = None
    if prior_draws:
        pd = _prior_draws(model, int(prior_draws))
        if pd is not None:
            prior = {"theta": _flat_params(pd)}

    obs_by_kind = {"photometry": [], "spectrum": [], "lines": []}
    for o in model.observations:
        obs_by_kind.get(getattr(o, "_kind", ""), []).append(o)
    has_lines = bool(obs_by_kind["lines"])
    n_par = len(cols)
    n_marg_cols = 4
    n_marg_rows = int(np.ceil(n_par / n_marg_cols))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.35, 1.0],
                  left=0.06, right=0.98, top=0.90, bottom=0.07, hspace=0.32, wspace=0.22)
    gs_top = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_sed = fig.add_subplot(gs_top[0]); ax_chi = fig.add_subplot(gs_top[1], sharex=ax_sed)
    ax_lines = fig.add_subplot(gs[0, 1]) if has_lines else None
    ax_sfh = fig.add_subplot(gs[1, 0])
    gs_marg = gs[1, 1].subgridspec(n_marg_rows, n_marg_cols, hspace=0.9, wspace=0.35)

    chi2, ndata = 0.0, 0
    pred = out["prediction"]
    best_pred = out["bestfit"]["prediction"]
    truth_pt = _flat_point(truths, list(truths)) if truths else {}
    observed = "spectra_observed" in pred
    unit = _MAGGIE_TO_NJY if observed else _MAGGIE_TO_CGS     # photometry -> spectrum's units

    wave_rest = np.asarray(pred["wave_rest"])
    if observed:
        z = np.asarray(pred["zred"])
        wobs = (1.0 + np.median(z)) * wave_rest
        spec = np.asarray(pred["spectra_observed"], dtype=float) * _CGS_FNU_TO_NJY
    else:
        wobs = wave_rest
        spec = np.asarray(pred["spectra_model"], dtype=float)
    lo, med, hi = _quantiles(spec)
    xr = _data_wave_range(obs_by_kind)                 # [um], observed frame; None: no data
    m = (med > 0) & np.isfinite(med)
    if xr is not None:
        m &= (wobs / 1e4 >= xr[0]) & (wobs / 1e4 <= xr[1])
    ax_sed.fill_between(wobs[m] / 1e4, lo[m], hi[m], color=C["band"], alpha=0.6, lw=0,
                        label="model (16$-$84%)")
    ax_sed.plot(wobs[m] / 1e4, med[m], color=C["posterior"], lw=0.9)
    y_seen = [lo[m], hi[m]]
    unit_spec = _CGS_FNU_TO_NJY if observed else 1.0     # spectra are cgs f_nu, photometry maggies
    for o in obs_by_kind["spectrum"]:
        w = np.asarray(o.wavelength, dtype=float) / 1e4
        y = np.asarray(o.flux, dtype=float) * unit_spec
        s = np.asarray(o.uncertainty, dtype=float) * unit_spec
        mk = np.asarray(o.mask, dtype=bool)
        ax_sed.fill_between(w[mk], (y - s)[mk], (y + s)[mk], color=C["data"], alpha=0.15, lw=0)
        ax_sed.plot(w[mk], y[mk], color=C["data"], lw=0.6, label=f"observed spectrum ({o.name})")
        y_seen.append(y[mk])
        if o.name in pred["spectra"]:
            p = np.asarray(pred["spectra"][o.name], dtype=float) * unit_spec
            plo, pmed, phi = _quantiles(p)
            ax_sed.plot(w[mk], pmed[mk], color=C["bestfit"], lw=0.6, alpha=0.8, label="posterior spectrum")
            r = ((pmed - y) / s)[mk]
            ax_chi.plot(w[mk], r, color=C["posterior"], lw=0.5)
            pb = np.asarray(best_pred["spectra"][o.name], dtype=float) * unit_spec
            chi2 += float(np.sum(((pb - y) / s)[mk] ** 2)); ndata += int(mk.sum())
    for o in obs_by_kind["photometry"]:
        w = np.asarray(o.wavelength, dtype=float) / 1e4
        y = np.asarray(o.flux, dtype=float) * unit
        s = np.asarray(o.uncertainty, dtype=float) * unit
        mk = np.asarray(o.mask, dtype=bool)
        ul = np.asarray(o.upper_limit, dtype=bool) if getattr(o, "upper_limit", None) is not None else np.zeros_like(mk)
        det = mk & ~ul
        ax_sed.errorbar(w[det], y[det], yerr=s[det], fmt="o", color=C["data"], ms=5, capsize=2,
                        label="observed", zorder=5)
        y_seen.append(y[det])
        if ul.any():
            ax_sed.errorbar(w[ul], y[ul], yerr=0.3 * y[ul], uplims=True, fmt="v", color=C["grid"],
                            ms=5, label="upper limit", zorder=5)
        if o.name in pred["photometry"]:
            p = np.asarray(pred["photometry"][o.name], dtype=float) * unit
            plo, pmed, phi = _quantiles(p)
            ax_sed.errorbar(w, pmed, yerr=[pmed - plo, phi - pmed], fmt="s", mfc="none",
                            color=C["bestfit"], ms=6, capsize=0, label="posterior photometry", zorder=6)
            r = (pmed - y) / s
            rlo, rhi = (plo - y) / s, (phi - y) / s
            ax_chi.errorbar(w[det], r[det], yerr=[(r - rlo)[det], (rhi - r)[det]], fmt="o",
                            color=C["bestfit"], ms=4, capsize=0)
            pb = np.asarray(best_pred["photometry"][o.name], dtype=float) * unit
            rb = (pb - y) / s
            chi2 += float(np.sum(np.where(ul, np.maximum(rb, 0.0), rb)[mk] ** 2)); ndata += int(mk.sum())
    ax_sed.set_xscale("log"); ax_sed.set_yscale("log")
    if xr is not None:
        ax_sed.set_xlim(*xr)
    ys = np.concatenate([np.ravel(v) for v in y_seen]) if y_seen else np.array([])
    ys = ys[np.isfinite(ys) & (ys > 0)]
    if ys.size:
        ax_sed.set_ylim(ys.min() / 3.0, ys.max() * 3.0)
    ax_sed.set_ylabel(r"$F_\nu$ [nJy]" if observed
                      else r"$L_\nu$ [$\mathrm{L}_\odot\,\mathrm{Hz}^{-1}$] ($z = 0$: no distance)")
    ax_sed.legend(fontsize=7, loc="lower right", frameon=False)
    ax_sed.tick_params(labelbottom=False)
    ax_chi.axhline(0, color=C["data"], lw=0.8)
    ax_chi.axhspan(-1, 1, color=C["prior"], alpha=0.6, lw=0)
    ax_chi.set_ylim(-4, 4); ax_chi.set_ylabel(r"$\chi$")
    ax_chi.set_xlabel(r"observed wavelength [$\mu$m]")
    if xr is not None:
        _nice_log_ticks(ax_chi.xaxis, *xr)
    for ax in (ax_sed, ax_chi):
        ax.grid(False)

    # emission lines: (model - observed) / sigma per line
    if has_lines:
        yy = 0
        for o in obs_by_kind["lines"]:
            names = list(o.line_names) if getattr(o, "line_names", None) else [f"{float(l):.0f}" for l in np.asarray(o.wavelength)]
            y = np.asarray(o.flux, dtype=float); s = np.asarray(o.uncertainty, dtype=float)
            mk = np.asarray(o.mask, dtype=bool)
            ul = np.asarray(o.upper_limit, dtype=bool) if getattr(o, "upper_limit", None) is not None else np.zeros_like(mk)
            p = np.asarray(pred["lines"][o.name], dtype=float) if o.name in pred["lines"] else None
            pb = np.asarray(best_pred["lines"][o.name], dtype=float) if o.name in best_pred["lines"] else None
            for i, nm in enumerate(names):
                if not mk[i]:
                    continue
                snr = y[i] / s[i]
                if p is not None:
                    lo, med, hi = _quantiles((p[:, i] - y[i]) / s[i])
                    ax_lines.errorbar(med, yy, xerr=[[med - lo], [hi - med]], fmt="o",
                                      color=C["bestfit"] if snr >= 3 else C["band"],
                                      mfc=C["bestfit"] if snr >= 3 else "white", ms=5, capsize=0)
                    if pb is not None:
                        rb = (pb[i] - y[i]) / s[i]
                        ax_lines.plot(rb, yy, marker="|", color=C["data"], ms=8, lw=0)
                        chi2 += float(max(rb, 0.0) ** 2 if ul[i] else rb ** 2); ndata += 1
                ax_lines.text(4.9, yy, f"{snr:.0f}", ha="right", va="center", fontsize=7, color=C["grid"])
                ax_lines.text(-5.2, yy, nm, ha="right", va="center", fontsize=8)
                yy += 1
        ax_lines.axvline(0, color=C["data"], lw=0.8)
        ax_lines.axvspan(-1, 1, color=C["prior"], alpha=0.6, lw=0)
        ax_lines.set_xlim(-5.4, 5.2); ax_lines.set_ylim(-0.7, yy - 0.3); ax_lines.invert_yaxis()
        ax_lines.set_yticks([]); ax_lines.set_xlabel(r"(model $-$ observed) / $\sigma$")
        ax_lines.text(5.05, -0.55, "S/N", ha="right", va="center", fontsize=7, color=C["grid"])
        ax_lines.plot([], [], "o", color=C["bestfit"], label=r"S/N $\geq$ 3")
        ax_lines.plot([], [], "o", color=C["band"], mfc="white", label="S/N < 3")
        ax_lines.plot([], [], "|", color=C["data"], label="best fit")
        ax_lines.legend(fontsize=7, loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.08))

    # SFH (as the paper's figures): per-bin log10 SFR against lookback time [Gyr] on a log
    # axis; posterior 16-84 % per bin, posterior median, and the injected SFH when given
    sfh = out["extras"]["sfh"]
    T = np.median(np.asarray(sfh["lookback_gyr"], dtype=float), axis=0)
    edges = T.copy()
    if edges[0] <= 0.0:
        edges[0] = 0.5 * edges[1]              # a log axis has no 0: first bin from t1 / 2
    if "sfr_per_bin" in sfh:
        per_bin = np.asarray(sfh["sfr_per_bin"], dtype=float)
    else:
        from .postprocess import _per_bin_and_nodes
        per_bin = np.array([_per_bin_and_nodes(r, T.size)[0]
                            for r in np.asarray(sfh["sfr"], dtype=float)])
    with np.errstate(divide="ignore"):
        lsfr = np.log10(per_bin)
    lsfr = np.where(np.isfinite(lsfr), lsfr, np.nan)
    lo, med, hi = _quantiles(lsfr)
    ax_sfh.stairs(hi, edges, baseline=lo, fill=True, color=C["band"], alpha=0.6, lw=0,
                  label=r"posterior 16$-$84%")
    ax_sfh.stairs(med, edges, color=C["posterior"], lw=1.6, label="posterior median")
    t_sfr = _truth_sfr_per_bin(model, out, truths)
    if t_sfr is not None:
        with np.errstate(divide="ignore"):
            lt = np.log10(t_sfr)
        ax_sfh.stairs(np.where(np.isfinite(lt), lt, np.nan), edges, color=C["truth"], lw=1.4,
                      ls="--", label="truth")
    ax_sfh.set_xscale("log")
    ax_sfh.set_xlim(edges[0], edges[-1])
    _nice_log_ticks(ax_sfh.xaxis, edges[0], edges[-1])
    ax_sfh.set_xlabel("lookback time [Gyr]")
    ax_sfh.set_ylabel(r"$\log_{10}$ SFR [M$_\odot$ yr$^{-1}$]")
    ax_sfh.legend(fontsize=8, loc="best", frameon=False)

    # marginals
    for i, (name, x) in enumerate(cols.items()):
        ax = fig.add_subplot(gs_marg[i // n_marg_cols, i % n_marg_cols])
        x = x[np.isfinite(x)]
        q = _quantiles(x)
        if prior is not None and name in prior["theta"]:
            plo, _, phi = _quantiles(prior["theta"][name])
            ax.axvspan(plo, phi, color=C["prior"], alpha=0.8, lw=0)
        ax.hist(x, bins=30, color=C["band"], histtype="stepfilled", alpha=0.9)
        ax.hist(x, bins=30, color=C["posterior"], histtype="step", lw=1.0)
        ax.axvline(q[1], color=C["posterior"], lw=1.2)
        if name in best:
            ax.axvline(best[name], color=C["bestfit"], lw=1.0)
        if name in truth_pt:
            ax.axvline(truth_pt[name], color=C["truth"], lw=1.2, ls="--")
        ax.set_yticks([]); ax.tick_params(axis="x", labelsize=7)
        ax.set_title(_label(name, feh), fontsize=9, pad=3)
        ax.set_xlabel(_fmt_q(*q), fontsize=8, labelpad=1)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)

    # header
    meta = out["meta"]
    bits = []
    if "zred" in theta:
        q = _quantiles(theta["zred"]); bits.append(r"$z$ = " + _fmt_q(*q, nd=3))
    else:
        bits.append(rf"$z$ = {meta.get('zred_fixed', 0.0):.3f} (fixed)")
    if "logmass" in theta:
        bits.append(r"$\log M_\mathrm{formed}/\mathrm{M}_\odot$ = " + _fmt_q(*_quantiles(theta["logmass"])))
    if "sfr10" in sfh:
        with np.errstate(divide="ignore"):
            s10 = np.log10(np.asarray(sfh["sfr10"]))
        s10 = s10[np.isfinite(s10)]
        if s10.size:
            bits.append(r"$\log\,\mathrm{SFR}_{10}$ = " + _fmt_q(*_quantiles(s10)))
    nfree = sum(int(np.size(v)) for v in model.theta_init.values())
    if ndata:
        # diagonal chi^2 with the quoted uncertainties; a spectrum fitted with a GP likelihood
        # is scored by the full covariance in the fit, which this number does not include
        diag = " diagonal, no GP;" if _has_gp(model) else ""
        bits.append(rf"$\chi^2/\nu$ = {chi2 / max(ndata - nfree, 1):.2f} (best fit, quoted uncertainties;{diag} $N_\mathrm{{data}}$ = {ndata})")
    if "log_evidence" in meta and np.isfinite(meta["log_evidence"]):
        bits.append(rf"$\ln Z$ = {meta['log_evidence']:.1f}")
    fig.text(0.06, 0.965, title or "CERIDWEN fit", fontsize=14, weight="bold", va="top")
    fig.text(0.06, 0.93, "    ".join(bits) + f"    sampler: {meta.get('sampler', '')}", fontsize=9.5, va="top")
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def _has_gp(model) -> bool:
    """True when some Spectrum of ``model`` is fitted with the GP likelihood."""
    from .model.obs_params import GP_FAMILIES
    given = set(getattr(model, "param_names", ())) | set(getattr(model, "transforms", {}) or {})
    return (any(getattr(o, "noise", None) is not None for o in model.observations)
            or any(f.claims(n) for f in GP_FAMILIES for n in given))


def _smooth(H, sigma=1.0):
    """Gaussian-smoothed 2-D histogram (separable kernel, no scipy)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    Hp = np.pad(H, r, mode="edge")
    Hs = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, Hp)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, Hs)


def corner_figure(out, *, params=None, truths=None, savepath=None, bins=30, panel=1.6):
    """Corner plot of all fitted parameters: 1-D marginals on the diagonal, 2-D
    histograms with 1 and 2 sigma contours below; posterior median (blue, dashed),
    maximum-likelihood sample (red) and, when given, ``truths`` (green) marked."""
    import matplotlib.pyplot as plt
    C = COLORS
    feh = _feh_axis(out)
    cols = dict(_flat_params(out["theta"], params))
    best = _flat_point(out["bestfit"]["theta"], (params or list(out["theta"].keys())))
    tot = out.get("extras", {}).get("metallicity", {}).get("logzsol_total")
    best_tot = out.get("bestfit", {}).get("extras", {}).get("metallicity", {}).get("logzsol_total")
    if params is None and tot is not None and best_tot is not None:
        cols["logzsol_total"] = np.asarray(tot).reshape(-1)
        best["logzsol_total"] = float(np.asarray(best_tot).reshape(-1)[0])
    truth_pt = _flat_point(truths, list(truths)) if truths else {}
    names = list(cols)
    K = len(names)
    X = np.column_stack([cols[n] for n in names])
    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]
    fig, axes = plt.subplots(K, K, figsize=(panel * K, panel * K))
    axes = np.atleast_2d(axes)
    lims = [(np.quantile(X[:, i], 0.001), np.quantile(X[:, i], 0.999)) for i in range(K)]
    lims = [(a - 0.05 * (b - a), b + 0.05 * (b - a)) if b > a else (a - 1, a + 1) for a, b in lims]
    for i in range(K):                   # a supplied truth is always inside its panel
        t = truth_pt.get(names[i])
        if t is not None and np.isfinite(t):
            a, b = lims[i]
            pad = 0.05 * (b - a)
            lims[i] = (min(a, float(t) - pad), max(b, float(t) + pad))
    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue
            if i == j:
                x = X[:, i]
                q = _quantiles(x)
                ax.hist(x, bins=bins, range=lims[i], color=C["band"], histtype="stepfilled", alpha=0.9)
                ax.hist(x, bins=bins, range=lims[i], color=C["posterior"], histtype="step", lw=1.0)
                ax.axvline(q[1], color=C["posterior"], ls="--", lw=1.0)
                ax.axvline(q[0], color=C["posterior"], ls=":", lw=0.7); ax.axvline(q[2], color=C["posterior"], ls=":", lw=0.7)
                ax.axvline(best[names[i]], color=C["bestfit"], lw=1.2)
                if names[i] in truth_pt:
                    ax.axvline(truth_pt[names[i]], color=C["truth"], lw=1.2, ls="--")
                ax.set_title(_label(names[i], feh) + "\n" + _fmt_q(*q), fontsize=8)
                ax.set_yticks([]); ax.set_xlim(lims[i])
            else:
                x, y = X[:, j], X[:, i]
                H, xe, ye = np.histogram2d(x, y, bins=bins, range=[lims[j], lims[i]])
                Hs = _smooth(H.T)
                flat = np.sort(Hs.ravel())[::-1]
                csum = np.cumsum(flat) / flat.sum()
                levels = [flat[np.searchsorted(csum, f)] for f in (0.865, 0.393)]
                levels = sorted(set([max(l, 1e-12) for l in levels]))
                xc = 0.5 * (xe[1:] + xe[:-1]); yc = 0.5 * (ye[1:] + ye[:-1])
                ax.pcolormesh(xe, ye, np.ma.masked_where(Hs == 0, Hs), cmap="Blues", shading="flat", rasterized=True)
                if len(levels) >= 1:
                    ax.contour(xc, yc, Hs, levels=levels + [Hs.max() + 1], colors=C["posterior"], linewidths=0.8)
                ax.axvline(np.median(x), color=C["posterior"], ls="--", lw=0.7)
                ax.axhline(np.median(y), color=C["posterior"], ls="--", lw=0.7)
                ax.plot(best[names[j]], best[names[i]], "o", color=C["bestfit"], ms=4)
                if names[j] in truth_pt and names[i] in truth_pt:
                    ax.plot(truth_pt[names[j]], truth_pt[names[i]], "s", color=C["truth"], ms=4)
                ax.set_xlim(lims[j]); ax.set_ylim(lims[i])
            if i < K - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(_label(names[j], feh), fontsize=8)
            if j > 0 or i == 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(_label(names[i], feh), fontsize=8)
            ax.tick_params(labelsize=6)
    fig.text(0.62, 0.95, "dashed: posterior median (dotted: 16 / 84 %)\nred: maximum-likelihood sample"
             + ("\ngreen: injected truth" if truth_pt else ""), fontsize=9, color=C["data"], va="top")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.95, hspace=0.08, wspace=0.08)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def _split_rhat(chains):
    """Split-R-hat of (n_chains, n_samples) draws."""
    c = np.asarray(chains, dtype=float)
    half = c.shape[1] // 2
    if half < 2:
        return np.nan
    c = np.concatenate([c[:, :half], c[:, half:2 * half]], axis=0)
    m, n = c.shape
    means = c.mean(axis=1); W = c.var(axis=1, ddof=1).mean(); B = n * means.var(ddof=1)
    var = (n - 1) / n * W + B / n
    return float(np.sqrt(var / W)) if W > 0 else np.nan


def _ess(x):
    """Effective sample size of one chain from the autocorrelation (initial positive sequence)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    if n < 4 or np.all(x == 0):
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    acf /= acf[0]
    s = 0.0
    for t in range(1, n - 1, 2):
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        s += pair
    return float(n / (1 + 2 * s))


def diagnostic_figure(result, out=None, *, params=None, savepath=None, max_points=4000):
    """Sampling diagnostics.  Nested sampling: every parameter's dead points in
    deletion order coloured by posterior weight, the log-likelihood run and the
    cumulative evidence.  MCMC: per-chain traces with split-R-hat and ESS."""
    import matplotlib.pyplot as plt
    C = COLORS
    feh = _feh_axis(out) if out is not None else False     # label logzsol as the others do
    theta = {k: np.asarray(v) for k, v in result.samples.items()}
    if params:
        theta = {k: theta[k] for k in params}
    cols = _flat_params(theta)
    names = list(cols); K = len(names)
    raw = getattr(result, "raw", None) or {}
    n_chains = int(raw.get("num_chains", 0) or 0)
    is_mcmc = n_chains > 0 and getattr(result, "log_likelihoods_birth", None) is None
    ncol = 2
    nrow = int(np.ceil(K / ncol)) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.0 * nrow))
    axes = np.atleast_2d(axes)
    ll = np.asarray(result.log_likelihoods, dtype=float)
    n = ll.size
    if is_mcmc:
        S = n // n_chains
        for i, name in enumerate(names):
            ax = axes[i // ncol, i % ncol]
            ch = cols[name][: n_chains * S].reshape(n_chains, S)
            for c in range(n_chains):
                ax.plot(ch[c], color=C["chains"][c % len(C["chains"])], lw=0.4, alpha=0.8)
            rhat = _split_rhat(ch); ess = sum(_ess(ch[c]) for c in range(n_chains))
            ax.set_title(f"{_label(name, feh)}    $\\hat R$ = {rhat:.3f}    ESS = {ess:.0f}", fontsize=8)
            ax.tick_params(labelsize=7)
        ax = axes[-1, 0]
        for c in range(n_chains):
            ax.plot(ll[c * S:(c + 1) * S], color=C["chains"][c % len(C["chains"])], lw=0.4, alpha=0.8)
        ax.set_title("log-likelihood per chain", fontsize=8)
        ax2 = axes[-1, 1]
        div = raw.get("total_divergences", None)
        ax2.axis("off")
        ax2.text(0.0, 0.8, f"{n_chains} chains x {S} draws" + (f", {div} divergences" if div is not None else ""),
                 fontsize=9, transform=ax2.transAxes)
        fig.suptitle("MCMC traces", fontsize=11)
    else:
        from .sampler.ns_weights import nested_log_weights
        births = getattr(result, "log_likelihoods_birth", None)
        lw = (nested_log_weights(ll, np.asarray(births)) if births is not None
              else np.asarray(result.log_weights, dtype=float))
        fin = np.isfinite(lw)
        w = np.zeros(n); w[fin] = np.exp(lw[fin] - lw[fin].max()); w /= w.sum()
        order = np.arange(n)
        step = max(1, n // max_points)
        sel = order[::step]
        for i, name in enumerate(names):
            ax = axes[i // ncol, i % ncol]
            ax.scatter(sel, cols[name][sel], c=np.clip(lw[sel] - lw[fin].max(), -12, 0), cmap=_LazyCmap()(),
                       s=3, vmin=-12, vmax=0, rasterized=True)
            qs = _weighted_quantiles(cols[name], w, (0.16, 0.5, 0.84))
            ax.set_title(f"{_label(name, feh)}    posterior " + _fmt_q(*qs), fontsize=8)
            ax.tick_params(labelsize=7)
        ax = axes[-1, 0]
        ax.plot(order, ll, color=C["posterior"], lw=0.6)
        ax.set_ylabel(r"$\ln L$", fontsize=8); ax.set_xlabel("dead point (deletion order)", fontsize=8)
        fl = ll[fin]
        if fl.size:
            ax.set_ylim(np.quantile(fl, 0.02), fl.max() + 0.05 * (fl.max() - np.quantile(fl, 0.02)))
        ax.set_title("log-likelihood run", fontsize=8)
        ax2 = axes[-1, 1]
        ax2.plot(order, np.cumsum(w), color=C["posterior"], lw=1.0)
        ax2.set_ylim(0, 1.02); ax2.set_xlabel("dead point (deletion order)", fontsize=8)
        ax2.set_ylabel("cumulative posterior weight", fontsize=8)
        ess = 1.0 / np.sum(w ** 2)
        lz = getattr(result, "log_evidence", np.nan); lze = getattr(result, "log_evidence_err", np.nan)
        ax2.set_title(rf"$\ln Z$ = {lz:.2f} $\pm$ {lze:.2f}    ESS = {ess:.0f} of {n}", fontsize=8)
        fig.suptitle("nested-sampling run: dead points coloured by posterior weight", fontsize=11)
    for k in range(K, (nrow - 1) * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def make_figures(out, model, result, outdir, *, prefix="", title=None, truths=None, fmt="pdf"):
    """Write summary, corner and diagnostic figures to ``outdir``; returns their paths."""
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn, args, kw in (("summary", summary_figure, (out, model), {"title": title, "truths": truths}),
                               ("corner", corner_figure, (out,), {"truths": truths}),
                               ("diagnostics", diagnostic_figure, (result, out), {})):
        p = outdir / f"{prefix}{name}.{fmt}"
        plt.close(fn(*args, savepath=p, **kw))
        paths[name] = p
    return paths
