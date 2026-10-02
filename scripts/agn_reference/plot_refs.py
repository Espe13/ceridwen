"""One figure per AGN reference npz (A1-A5), house style (paper-figures/figkit).

    SPS_HOME=$SPS_HOME $CERIDWEN_PYTHON scripts/agn_reference/plot_refs.py [OUTDIR]

Needs smplotlib (ceridwen311 has it); reads tests/reference/agn/*.npz and
recomputes the A1 port comparison with fsps_port (numpy only). Writes
<name>.pdf/.png, <name>_numbers.json and captions.json to OUTDIR (default
~/Desktop/PhD/Tools/ceridwen_workspace/agents/agn/ref_figures).
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent.parent / "tests" / "reference" / "agn"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))
from figkit import *  # noqa: E402,F401,F403
import fsps_port  # noqa: E402

C_AA = 2.99792458e18
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path.home() / "Desktop/PhD/Tools/ceridwen_workspace/agents/agn/ref_figures")
OUT.mkdir(parents=True, exist_ok=True)
CAPS = {}


def load(name):
    return np.load(REF / name)


def um(w):
    return np.asarray(w) / 1e4


def nuLnu(w, lnu):
    return C_AA / np.asarray(w) * lnu


def tau_colors(taus, lo=None, hi=None):
    lt = np.log10(taus)
    lo = lt.min() if lo is None else lo
    hi = lt.max() if hi is None else hi
    return [GRAD(0.08 + 0.9 * (x - lo) / (hi - lo)) for x in lt]


def fmt(x, p=2):
    m, e = f"{x:.{p}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


# ---------------------------------------------------------------- A1
def fig_a1():
    a = load("A1_torus_fsps.npz")
    w, d, s0, lb = a["wave"], a["dspec"], a["spec0"], a["log_lbol"]
    fagn, taus, d2s = a["fagn"], a["agn_tau"], a["dust2"]
    jf = int(np.where(fagn == 1.0)[0][0])
    fig, ax = plt.subplots(1, 3, figsize=(PAGE, 2.6))
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.99, bottom=0.2, top=0.95)
    x = um(w)
    # (a) torus shape vs tau, z=0, dust2=0, fagn=1, with the stellar spectrum
    ax[0].plot(x, nuLnu(w, s0[0, 0]), color=GREY, lw=1.0)
    for c, it in zip(tau_colors(taus), range(len(taus))):
        ax[0].plot(x, nuLnu(w, d[0, 0, it, jf]), color=c, lw=1.1, label=f"${taus[it]:g}$")
    ax[0].set(xscale="log", yscale="log", xlim=(0.1, 1e3), ylim=(1e-6, 3),
              xlabel=r"$\lambda$  [$\mu$m]", ylabel=r"$\nu L_\nu$  [L$_\odot$ M$_\odot^{-1}$]")
    ax[0].legend(title=r"$\tau_{\rm AGN}$", loc="lower left", ncol=3, fontsize=FS_LEGEND,
                 columnspacing=0.8, handlelength=1.2)
    lab(ax[0], "(a)")
    # (b) dust2 screen on the torus, tau=10
    it = int(np.where(taus == 10.0)[0][0])
    for c, idd in zip([BLUE, CORAL, CRIM], range(len(d2s))):
        ax[1].plot(x, d[0, idd, it, jf] / d[0, 0, it, jf], color=c, lw=1.2,
                   label=f"${d2s[idd]:g}$")
    ax[1].set(xscale="log", xlim=(0.1, 30), ylim=(0, 1.2), xlabel=r"$\lambda$  [$\mu$m]",
              ylabel=r"$\Delta L_\nu(\tau_2) / \Delta L_\nu(0)$")
    ax[1].legend(title=r"$\tau_2$", loc="lower right")
    lab(ax[1], "(b)")
    # (c) FSPS vs float64 port, all 180 spectra: max over configs per pixel
    eps = np.finfo(float).eps
    rel_max = np.zeros_like(w)
    floor_max = np.zeros_like(w)
    n_cfg = 0
    worst_excess = 0.0
    for iz in range(d.shape[0]):
        for idd, d2 in enumerate(d2s):
            for it2, tau in enumerate(taus):
                for jf2, fa in enumerate(fagn):
                    p = fsps_port.agn_torus_fsps(w, lb[iz, idd], fa, tau, d2, 0, -0.7)
                    dd = d[iz, idd, it2, jf2]
                    m = p > 1e-6 * p.max()
                    r = np.zeros_like(w)
                    f = np.zeros_like(w)
                    r[m] = np.abs(dd[m] - p[m]) / p[m]
                    f[m] = 8 * eps * (np.abs(s0[iz, idd][m]) + p[m]) / p[m]
                    rel_max = np.maximum(rel_max, r)
                    floor_max = np.maximum(floor_max, f)
                    worst_excess = max(worst_excess, float(np.max(r - f)))
                    n_cfg += 1
    ok = floor_max > 0
    ax[2].plot(x[ok], floor_max[ok], color=GREY, lw=1.0, zorder=1)
    ax[2].plot(x[ok], np.maximum(rel_max[ok], 1e-18), ".", ms=2.2, color=DKRED, zorder=2)
    ax[2].axhline(1e-5, color=CRIM, lw=0.8, ls=":")
    ax[2].set(xscale="log", yscale="log", xlim=(0.01, 1e4), ylim=(1e-18, 1e-3),
              xlabel=r"$\lambda$  [$\mu$m]", ylabel=r"max $|\Delta_{\rm FSPS} - \Delta_{\rm port}| / \Delta_{\rm port}$")
    lab(ax[2], "(c)")
    num = dict(n_configs=n_cfg, max_rel=float(rel_max.max()), max_floor=float(floor_max[ok].max()),
               max_rel_minus_floor=worst_excess, n_wave=int(len(w)), log_lbol=float(lb[0, 0]),
               repeat_maxabs=float(a["noise_repeat_maxabs"]))
    save_num(num, "A1_torus_fsps", OUT)
    save(fig, "A1_torus_fsps", OUT)
    CAPS["A1_torus_fsps"] = (
        "Reference A1: the Nenkova et al. (2008) torus as python-fsps 0.5.0 adds it, "
        "$\\Delta L_\\nu = L_\\nu(f_{\\rm AGN}) - L_\\nu(0)$, for a 1 Gyr solar-metallicity SSP "
        f"(log $L_{{\\rm bol}}$ = {num['log_lbol']:.4f}, per M$_\\odot$ formed) on the "
        f"{num['n_wave']}-pixel FSPS grid. (a) $\\nu\\Delta L_\\nu$ at $f_{{\\rm AGN}}=1$, $z=0$, "
        "$\\tau_2=0$ for the five $\\tau_{\\rm AGN}$ of the test grid (colour, blue to dark red "
        "with increasing $\\tau_{\\rm AGN}$); grey: the stellar+dust spectrum without AGN. "
        "(b) The diffuse-dust screen FSPS applies to the torus, ratio of $\\Delta L_\\nu$ at "
        "$\\tau_2$ = 0.5 (coral) and 1.5 (red) to $\\tau_2=0$, power-law curve with index $-0.7$, "
        "$\\tau_{\\rm AGN}=10$. (c) Per pixel, the maximum over all "
        f"{num['n_configs']} configurations ($f_{{\\rm AGN}}$ x $\\tau_{{\\rm AGN}}$ x $\\tau_2$ x $z$) "
        "of the relative difference between python-fsps and an independent float64 numpy port "
        "of agn\\_dust.f90 and the sps\\_setup.f90 resampling (dark red points), on pixels "
        "where the torus exceeds $10^{-6}$ of its peak; grey line: the float64 cancellation "
        "floor $8\\epsilon(|L_\\nu(0)| + \\Delta L_\\nu)/\\Delta L_\\nu$ of the FSPS difference; "
        "dotted red: the $10^{-5}$ target of AGN\\_PLAN A1. Points at $10^{-18}$ are exact "
        f"agreement. The maximum difference, ${fmt(num['max_rel'])}$, never exceeds the floor "
        f"(max excess {num['max_rel_minus_floor']:.1g}); repeating a python-fsps call changes "
        f"no bit (max $|\\Delta|$ = {num['repeat_maxabs']:g}).")


# ---------------------------------------------------------------- A2
def fig_a2():
    a = load("A2_torus_prospector.npz")
    w, T, taus = a["wave_fsps"], a["torus_fsps"], a["agn_tau"]
    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 2.7))
    fig.subplots_adjust(wspace=0.5, left=0.08, right=0.99, bottom=0.19, top=0.95)
    x = um(w)
    cols = tau_colors(taus)
    for c, t, row in zip(cols, taus, T):
        ext = (t < 5) or (t > 150)
        ax[0].plot(x, np.where(row > 0, nuLnu(w, row), np.nan), color=c, lw=1.0,
                   ls="--" if ext else "-", label=f"${t:g}$")
    ax[0].set(xscale="log", yscale="log", xlim=(0.1, 1e3), ylim=(1e-3, 3),
              xlabel=r"$\lambda$  [$\mu$m]", ylabel=r"$\nu T'_\nu$  [$L_{\rm bol}$ units]")
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    lt = np.log10(taus)
    sm = ScalarMappable(norm=Normalize(lt.min() - 0.08 / 0.9 * np.ptp(lt), lt.max() + 0.02 / 0.9 * np.ptp(lt)), cmap=GRAD)
    cb = fig.colorbar(sm, ax=ax[0], pad=0.02, fraction=0.05)
    cb.set_label(r"log $\tau_{\rm AGN}$")
    lab(ax[0], "(a)")
    # (b) prospect T' vs FSPS resampled nodes (port) at the 9 node taus
    tau_n, Tn = fsps_port.resample_torus(w)
    neg = {}
    for c, t in zip(tau_colors(tau_n), tau_n):
        i = int(np.where(taus == t)[0][0]) if t in taus else None
        if i is None:
            row = None
        else:
            row = T[i]
        if row is None:
            continue
        ref = Tn[:, list(tau_n).index(t)]
        m = ref > 1e-6 * ref.max()
        ax[1].plot(x[m], np.maximum(np.abs(row[m] / ref[m] - 1), 1e-17), ".", ms=2.2, color=c)
    for t in (1.0, 3.0, 200.0, 300.0):
        neg[t] = int((T[list(taus).index(t)] < 0).sum())
    ax[1].set(xscale="log", yscale="log", xlim=(0.01, 1e4), ylim=(1e-17, 1e-8),
              xlabel=r"$\lambda$  [$\mu$m]", ylabel=r"$|T'_\nu / T_\nu^{\rm FSPS} - 1|$")
    lab(ax[1], "(b)")
    # numbers
    rr, wworst = [], []
    for t in tau_n:
        if t in taus:
            ref = Tn[:, list(tau_n).index(t)]
            m = ref > 1e-6 * ref.max()
            q = np.abs(T[list(taus).index(t)][m] / ref[m] - 1)
            rr.append(float(q.max()))
            wworst.append(float(w[m][np.argmax(q)]))
            n_gt = int((q > 1e-14).sum())
    num = dict(max_abs_rel_vs_fsps_at_nodes=max(rr), wave_of_max_A=wworst[int(np.argmax(rr))],
               n_pix_above_1e14_last_node=n_gt, n_negative_pixels=neg,
               taus=[float(t) for t in taus])
    save_num(num, "A2_torus_prospector", OUT)
    save(fig, "A2_torus_prospector", OUT)
    nodes_in = ", ".join(f"{t:g}" for t in tau_n if t in taus)
    CAPS["A2_torus_prospector"] = (
        "Reference A2: Prospector's own torus $T'_\\nu$ (prospect.sources.fake\\_fsps.agn\\_torus, "
        "unscaled, i.e. per unit $L_{\\rm bol}$ before the factor $10^{\\log L_{\\rm bol}} f_{\\rm AGN}$) "
        "on the FSPS wavelength grid. (a) $\\nu T'_\\nu$ for the twelve $\\tau_{\\rm AGN}$ of the "
        "reference (colour, blue to dark red with increasing $\\tau_{\\rm AGN}$); dashed: values "
        "outside the tabulated 5 to 150, which scipy's interp1d extrapolates linearly rather "
        "than clamping as FSPS does (colour bar: log $\\tau_{\\rm AGN}$). (b) Absolute relative "
        "difference between $T'_\\nu$ and the FSPS torus columns resampled as sps\\_setup.f90 "
        f"does (numpy port), at the node values $\\tau_{{\\rm AGN}}$ = {nodes_in} (same colours), "
        "where the torus exceeds $10^{-6}$ of its peak; exact agreement is drawn at $10^{-17}$. "
        "On this grid the two resamplings agree to rounding except at the table's blue edge, "
        f"largest difference {num['max_abs_rel_vs_fsps_at_nodes']:.1e} at "
        f"{num['wave_of_max_A']:.0f} $\\rm\\AA$, so at the node values the two tori are the same. "
        "Extrapolated spectra with negative pixels: "
        + ", ".join(f"$\\tau_{{\\rm AGN}}={t:g}$: {n}" for t, n in neg.items()) + ".")


# ---------------------------------------------------------------- A3
def fig_a3():
    a = load("A3_disk_screen_prospector.npz")
    w = a["wave_fsps"]
    cfg, asp, nod = a["configs_fsps"], a["aspec_fsps"], a["agn_nodust_fsps"]
    gal = a["young_fsps"] + a["old_fsps"]
    x = um(w)

    def pick(**kw):
        cols = dict(zip([str(c) for c in a["columns"]], range(8)))
        m = np.ones(len(cfg), bool)
        for k, v in kw.items():
            m &= np.isclose(cfg[:, cols[k]], v)
        i = np.where(m)[0]
        assert len(i) == 1, (kw, i)
        return int(i[0])

    base = dict(fagn_bbb=1.0, dust_index=-0.7, dust4_type=0, dust4=0.0, dust4_index=-0.8)
    fig, ax = plt.subplots(1, 3, figsize=(PAGE, 2.6))
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.99, bottom=0.2, top=0.95)
    # (a) galaxy and disc at the normalisation
    i0 = pick(dust_type=0, dust2=0.0, zred=0.0, **base)
    ax[0].plot(x, nuLnu(w, gal), color=GREY, lw=1.0)
    ax[0].plot(x, nuLnu(w, asp[i0]), color=DKRED, lw=1.2)
    ax[0].axvspan(0.548, 0.552, color=GREY_LT, alpha=0.6, lw=0)
    top = nuLnu(w, gal)[(x > 0.09) & (x < 10)].max()
    ax[0].set(xscale="log", yscale="log", xlim=(0.09, 10), ylim=(top * 1e-4, top * 5),
              xlabel=r"$\lambda$  [$\mu$m]", ylabel=r"$\nu L_\nu$  [L$_\odot$]")
    lab(ax[0], "(a)")
    # (b) transmission of the disc for the four dust laws, dust2=1, plus dust4
    for c, dt in zip([BLUE, PINK, CORAL, CRIM], (0, 2, 4, 6)):
        i = pick(dust_type=dt, dust2=1.0, zred=0.0, **base)
        ax[1].plot(x, asp[i] / nod[i], color=c, lw=1.1, label=f"${dt}$")
    i = pick(dust_type=0, dust2=1.0, zred=0.0, fagn_bbb=1.0, dust_index=-0.7, dust4_type=1,
             dust4=0.5, dust4_index=-1.8)
    ax[1].plot(x, asp[i] / nod[i], color=DKRED, lw=1.1, ls="--")
    ax[1].set(xscale="log", xlim=(0.09, 10), ylim=(0, 1.05), xlabel=r"$\lambda$  [$\mu$m]",
              ylabel=r"$L_\nu^{\rm disc} / L_\nu^{\rm disc}(\tau=0)$")
    ax[1].legend(loc="lower right")
    lab(ax[1], "(b)")
    # (c) IGM at z=6 on disc and torus
    i6 = pick(dust_type=0, dust2=0.0, zred=6.0, **base)
    tr = asp[i6] / asp[i0]
    tc = a["torus_configs_fsps"]
    it0 = int(np.where((tc[:, 1] == 20.0) & (tc[:, 2] == 0.0))[0][0])
    it6 = int(np.where((tc[:, 1] == 20.0) & (tc[:, 2] == 6.0))[0][0])
    tt = a["aspec_torus_fsps"]
    with np.errstate(divide="ignore", invalid="ignore"):
        trt = tt[it6] / tt[it0]
    ax[2].plot(w, tr, color=DKRED, lw=1.2)
    ax[2].plot(w, trt, color=BLUE, lw=0.9, ls="--")
    ax[2].axvline(1215.67, color=GREY, lw=0.6)
    ax[2].axvline(911.75, color=GREY, lw=0.6)
    ax[2].set(xlim=(500, 1600), ylim=(-0.02, 1.05), xlabel=r"$\lambda_{\rm rest}$  [$\rm\AA$]",
              ylabel=r"$L_\nu(z=6) / L_\nu(z=0)$")
    lab(ax[2], "(c)")
    win = (w >= 5480) & (w <= 5520)
    num = dict(scale_fsps=float(a["scale_fsps"]), scale_ceridwen=float(a["scale_ceridwen"]),
               npix_window_fsps=int(win.sum()), npix_window_ceridwen=int(a["npix_5500_window_ceridwen"]),
               n_configs_fsps=int(len(cfg)), n_configs_ceridwen=int(len(a["configs_ceridwen"])),
               n_clipped_z6=int((asp[i6] <= 1e-70 * 1.0000001).sum()))
    save_num(num, "A3_disk_screen_prospector", OUT)
    save(fig, "A3_disk_screen_prospector", OUT)
    CAPS["A3_disk_screen_prospector"] = (
        "Reference A3: the accretion disc of Prospector's AGNSSP path (Temple et al. 2021 "
        "three-piece power law; Wang et al. 2024), computed by prospect.sources.agnssp\\_basis."
        "get\\_spectrum with a stub ssp holding fixed python-fsps galaxy spectra (constant SFR "
        "over 1 Gyr, no dust, no IGM). (a) $\\nu L_\\nu$ of the dust-free galaxy, young plus old "
        "(grey; $10^9$ M$_\\odot$ formed), and of the disc at $f_{\\rm AGN,bbb}=1$, $z=0$ (dark red), which by construction "
        "equals the galaxy in the median of the ratio over 5480 to 5520 $\\rm\\AA$ (grey band; "
        f"only {num['npix_window_fsps']} FSPS pixels fall in it, {num['npix_window_ceridwen']} "
        "on the CERIDWEN grid). (b) Disc transmission at $\\tau_2=1$ for dust\\_type 0 "
        "(power law, index $-0.7$, blue), 2 (Calzetti, pink), 4 (Kriek \\& Conroy, coral) and "
        "6 (Reddy, red) as fake\\_fsps.attenuate implements them, and dust\\_type 0 with the extra "
        "AGN screen dust4\\_type = 1, $\\tau_4 = 0.5$, index $-1.8$ (dashed dark red). "
        "(c) Ratio of the $z=6$ to the $z=0$ spectrum, i.e. the Madau (1995) IGM of "
        "fake\\_fsps.add\\_igm, for the disc (dark red) and for the torus $T'$ at "
        "$\\tau_{\\rm AGN}=20$ (dashed blue); grey lines: Ly$\\alpha$ and the Lyman limit. "
        f"add\\_igm clips at $10^{{-70}}$, which affects {num['n_clipped_z6']} pixels of the $z=6$ "
        f"disc. The reference holds {num['n_configs_fsps']} configurations on the FSPS grid and "
        f"{num['n_configs_ceridwen']} on the CERIDWEN grid.")


# ---------------------------------------------------------------- A4
def fig_a4():
    a = load("A4_lines_prospector.npz")
    ow, z = a["obs_wave"], float(a["zred"])
    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 2.7))
    fig.subplots_adjust(wspace=0.3, left=0.09, right=0.99, bottom=0.19, top=0.95)
    rw = ow / (1 + z)
    for c, sig, sp in zip([BLUE, CRIM], a["agn_eline_sigma"], a["spec_pred"]):
        ax[0].plot(rw, sp * 1e9, color=c, lw=0.9, label=f"${sig:g}$")
    ax[0].set(xlim=(4820, 5040), xlabel=r"$\lambda_{\rm rest}$  [$\rm\AA$]",
              ylabel=r"$f_\nu$  [$10^{-9}$ maggies]")
    ax[0].legend(title=r"$\sigma_{\rm AGN}$  [km s$^{-1}$]", loc="upper right")
    lab(ax[0], "(a)")
    inds = a["aline_inds"]
    lw_ = a["line_wave"][inds]
    rat = a["aline_lum_ratio"][inds]
    ax[1].vlines(lw_, 0.01, rat, color=DKRED, lw=1.0)
    ax[1].plot(lw_, rat, "o", ms=3, color=DKRED)
    ax[1].plot(lw_[inds == 59], rat[inds == 59], "o", ms=5, mfc="white", color=DKRED)
    ax[1].set(yscale="log", ylim=(0.01, 15), xlim=(lw_.min() - 200, lw_.max() + 200),
              xlabel=r"$\lambda_{\rm rest}$  [$\rm\AA$]", ylabel=r"$L_{\rm line} / L_{\rm H\beta}$")
    lab(ax[1], "(b)")
    nu = 3e18 / ow
    flux_cons = []
    for sp in a["spec_pred"]:
        tot = -np.trapezoid(sp, nu) * 3631e-23
        inw = (lw_ * (1 + z) > ow[0]) & (lw_ * (1 + z) < ow[-1])
        flux_cons.append(float(tot / a["line_pred"][0][inw].sum()))
    num = dict(n_lines=int(len(inds)), zred=z, agn_elum=float(a["agn_elum"]),
               flux_ratio_spec_to_lines=dict(zip([f"{s:g}" for s in a["agn_eline_sigma"]], flux_cons)),
               n_painted_pixels=[int(m.sum()) for m in a["eline_pixmask"]])
    save_num(num, "A4_lines_prospector", OUT)
    save(fig, "A4_lines_prospector", OUT)
    s0, s1 = [f"{s:g}" for s in a["agn_eline_sigma"]]
    CAPS["A4_lines_prospector"] = (
        "Reference A4: AGN narrow lines of prospect.models.sedmodel.AGNSpecModel "
        "(Richardson et al. 2014 'a42' ratios on FSPS line indices) with a stub sps of zero "
        f"continuum and zero star-forming lines, at $z={z:g}$, $M_{{\\rm formed}}=10^{{10}}$ "
        f"M$_\\odot$ and agn\\_elum = {num['agn_elum']:g} L$_\\odot$(H$\\beta$)/M$_\\odot$. "
        "(a) The predicted observed-frame spectrum (2 $\\rm\\AA$ pixels, no instrumental "
        f"profile) around H$\\beta$ and [O III] 4959, 5007 for $\\sigma_{{\\rm AGN}}$ = {s0} (blue) "
        f"and {s1} (red) km s$^{{-1}}$; the line width is $\\sigma_{{\\rm AGN}}$ alone, with no "
        "library or instrumental term. Lines are painted only on pixels within "
        "$5\\sigma_{\\rm AGN}/\\sigma_{\\rm eline}$ star-forming line widths "
        f"({num['n_painted_pixels'][0]} and {num['n_painted_pixels'][1]} pixels), and each "
        "Gaussian is normalised over those pixels, so the frequency integral of the spectrum "
        f"recovers {num['flux_ratio_spec_to_lines'][s0]:.5f} ({s0} km s$^{{-1}}$) and "
        f"{num['flux_ratio_spec_to_lines'][s1]:.5f} ({s1} km s$^{{-1}}$) of the summed line "
        f"fluxes in the window. (b) The {num['n_lines']} line ratios to H$\\beta$ (open "
        "circle, FSPS index 59) against rest wavelength.")


# ---------------------------------------------------------------- A5
def fig_a5():
    a = load("A5_whole_model_prospector.npz")
    w, z = a["model_wave"], float(a["zred"])
    names = [c[0] for c in a["configs"]]
    io, it, ia = names.index("off"), names.index("torus"), names.index("all")
    ow = w * (1 + z)
    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 2.8))
    fig.subplots_adjust(wspace=0.3, left=0.09, right=0.99, bottom=0.19, top=0.95)
    fnu = lambda s: s * 3631e9  # maggies -> nJy
    ax[0].plot(um(ow), fnu(a["galspec"][io]), color=GREY, lw=0.9)
    ax[0].plot(um(ow), fnu(a["agnspec"][ia]), color=BLUE, lw=1.0)
    ax[0].plot(um(ow), fnu(a["agnspec_torus"][ia]), color=CORAL, lw=1.1)
    ax[0].plot(um(ow), fnu(a["fsps_torus_in_galspec"][ia]), color=CRIM, lw=1.1, ls="--")
    ax[0].plot(um(ow), fnu(a["smooth_spec"][ia]), color=DKRED, lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlim=(0.5, 30), ylim=(1, 3e4),
              xlabel=r"$\lambda_{\rm obs}$  [$\mu$m]", ylabel=r"$f_\nu$  [nJy]")
    lab(ax[0], "(a)")
    from ceridwen.observation.filters import load_filters
    piv = np.array([float(f.wave_pivot) for f in load_filters([str(f) for f in a["filters"]])]) / 1e4
    dph = a["phot"][it] - a["phot"][io]
    ax[1].plot(piv, dph / a["phot_torus"][it], "o-", color=CRIM, ms=3.5, lw=1.0)
    ax[1].axhline(1, color=GREY, lw=0.8)
    ax[1].set(xscale="log", xlim=(0.8, 24), ylim=(0.9, 2.1), xlabel=r"$\lambda_{\rm piv}$  [$\mu$m]",
              ylabel=r"$\Delta f^{\rm phot} / f^{\rm phot}_{T'}$")
    lab(ax[1], "(b)")
    r = dph / a["phot_torus"][it]
    num = dict(zred=z, ratio_min=float(r.min()), ratio_max=float(r.max()),
               fsps_torus_port_maxrel=float(a["fsps_torus_in_galspec_maxrel"][it]),
               filters=[str(f) for f in a["filters"]])
    save_num(num, "A5_whole_model_prospector", OUT)
    save(fig, "A5_whole_model_prospector", OUT)
    CAPS["A5_whole_model_prospector"] = (
        "Reference A5: the full Prospector AGN model, python-fsps 0.5.0 with prospect "
        "AGNSSPBasis and AGNPolySpecModel (continuity SFH, $\\log M = 10$, $\\tau_2 = 0.5$, "
        f"$z={z:g}$). (a) Observed-frame components for $f_{{\\rm AGN}} = 0.3$, $\\tau_{{\\rm AGN}} = 20$, "
        "$f_{\\rm AGN,bbb} = 1$, $\\tau_4 = 0.5$: the galaxy without AGN (grey), the attenuated "
        "disc (blue), Prospector's torus $T'$ (coral), the FSPS torus that python-fsps adds "
        "inside the galaxy spectrum because $f_{\\rm AGN}$ reaches it (dashed red; numpy port, "
        f"agreeing with the measured galaxy-spectrum change to {num['fsps_torus_port_maxrel']:.1e}), "
        "and the total model spectrum (dark red). (b) AGN-only photometry, "
        "$\\Delta f^{\\rm phot} = f^{\\rm phot}(f_{\\rm AGN}=0.3) - f^{\\rm phot}(0)$, divided by the "
        "photometry of $T'$ alone, in the 15 NIRCam and MIRI bands of the reference (F090W to "
        "F2100W) at their pivot wavelengths; the grey line is what a single torus would give. "
        f"The ratio runs from {num['ratio_min']:.2f} to {num['ratio_max']:.2f}: Prospector's "
        "AGNSSP path counts the torus twice.")


if __name__ == "__main__":
    for f in (fig_a1, fig_a2, fig_a3, fig_a4, fig_a5):
        f()
    (OUT / "captions.json").write_text(json.dumps(CAPS, indent=1))
    print("wrote", OUT)
