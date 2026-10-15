"""What a ``SedModel`` is made of, for people: ``SedModel.latex()`` (a table of every free,
derived and fixed parameter with its prior) and ``SedModel.display()`` (a diagram of the
model, with its prior-predictive range against the data).

Both read the model's structure, not its parameter names: a transform's inputs are the free
parameters its traced program actually reads (:func:`transform_inputs`), a constant transform
is a fixed value, and a parameter belongs to the forward-model stage whose component lists it
(``csp.dust_param_names``, ``csp.neb_param_names``, ...).  The stages are drawn in the order
the CSP applies them (``csp.py``, ``get_spectrum_*`` and ``predict``): stellar populations,
nebular emission, dust attenuation, dust emission, AGN, mass and redshift, IGM, broadening
into each observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# colour roles: the house palette (scripts/paper_style_sm.py), one colour per kind of physics;
# crimson is reserved for data
BLUE, CORAL, TEAL, PLUM = "#6170b9", "#ff7f50", "#3a9d7a", "#a94fb5"
SLATE, GREY, CRIM, INK = "#5b6770", "#8a8f98", "#dc143c", "#1f2328"

#: the attenuation laws of dust/attenuation_laws.py, by their papers
LAW_NAMES = {"powerlaw": "power law", "calzetti": "Calzetti et al. (2000)",
             "kriek_conroy": "Kriek & Conroy (2013)", "noll": "Noll et al. (2009)",
             "chevallard": "Chevallard et al. (2013)", "conroy": "Conroy et al. (2010)",
             "cardelli": "Cardelli et al. (1989)", "smc": "SMC (Pei 1992)",
             "lmc": "LMC (Pei 1992)", "gordon03_smcbar": "SMC bar (Gordon et al. 2003)",
             "reddy15": "Reddy et al. (2015)", "drude": "Drude profile"}

#: FSPS imf_type (src/imf.f90)
IMF_NAMES = {0: "Salpeter (1955)", 1: "Chabrier (2003)", 2: "Kroupa (2001)",
             3: "van Dokkum (2008)", 4: "Davé (2008)"}


@dataclass
class Stage:
    key: str
    title: str
    color: str
    details: list = field(default_factory=list)


@dataclass
class Param:
    name: str
    kind: str                      # "free" | "derived" | "fixed"
    stage: str                     # key of the Stage (or "obs:<name>") it enters
    prior: object = None
    shape: tuple = ()
    inputs: tuple = ()             # derived: the free parameters it reads
    fn_name: str = ""
    value: object = None           # fixed: its value


@dataclass
class Structure:
    stages: list
    params: list
    observations: list             # (obs, [detail lines])
    config: list                   # (label, value) of fixed settings


# --------------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------------

def transform_inputs(fn, theta: dict):
    """The keys of ``theta`` that ``fn(theta)`` depends on, from its traced program (an
    equation's outputs depend on all of its inputs, so a dependency inside a sub-program is
    never missed; it may be over-counted).  ``None`` when ``fn`` cannot be traced."""
    import jax
    keys = list(theta)
    try:
        closed = jax.make_jaxpr(lambda *xs: fn(dict(zip(keys, xs))))(*[theta[k] for k in keys])
    except Exception:                                         # noqa: BLE001  (not traceable)
        return None
    jaxpr = closed.jaxpr
    dep = {v: {i} for i, v in enumerate(jaxpr.invars)}

    def _of(v):
        return dep.get(v, set()) if not hasattr(v, "val") else set()   # a Literal has .val

    for eqn in jaxpr.eqns:
        s = set().union(*[_of(v) for v in eqn.invars]) if eqn.invars else set()
        for o in eqn.outvars:
            dep[o] = s
    used = set().union(*[_of(v) for v in jaxpr.outvars]) if jaxpr.outvars else set()
    return tuple(keys[i] for i in sorted(used))


def _fn_name(fn) -> str:
    name = getattr(fn, "__name__", None) or type(fn).__name__
    return "lambda" if name == "<lambda>" else name


def _obs_param_map(model, names) -> dict:
    """{parameter name: observation} for the per-observation families (noise, outlier,
    calibration, GP), via the same rule the likelihood uses (``model.obs_params``)."""
    from .obs_params import (CALIB_FAMILIES, GP_FAMILIES, NOISE_FAMILIES, OUTLIER_FAMILIES,
                             check_names)
    try:
        return check_names(model.observations,
                           OUTLIER_FAMILIES + NOISE_FAMILIES + CALIB_FAMILIES + GP_FAMILIES,
                           set(names))
    except ValueError:
        return {}


def _stage_of(model, key: str, obs_map: dict) -> str:
    """The forward-model stage a CSP / model key enters."""
    from ..csp.csp import LOGZSOL_KEYS
    csp = model.csp
    if key in obs_map:
        return "obs:" + obs_map[key].name
    if key in model._instrument_scale_keys():
        for o in model.observations:
            if key in getattr(getattr(o, "instrument", None), "free_keys", ()):
                return "obs:" + o.name
    if key in ("sfh", "lookback_time") or key in LOGZSOL_KEYS or key.startswith("afe"):
        return "stars"
    for stage, attr in (("dust", "dust_param_names"), ("dust", "diff_param_names"),
                        ("neb", "neb_param_names"), ("duste", "emi_param_names"),
                        ("agn", "agn_param_names")):
        if key in (getattr(csp, attr, None) or ()):
            return stage
    if key in ("frac_obrun",):
        return "dust"
    if key in tuple(getattr(getattr(csp, "igm", None), "param_names", ()) or ()):
        return "igm"
    if key in ("zred", "lumdist_mpc", "logmass"):
        return "flux"
    if key in model.kinematics.free_keys:
        return "kin"
    if key.startswith(("gas_", "fesc")):
        return "neb"
    return "stars"


def _mpc(d) -> str:
    d = float(d)
    return f"{d:,.0f} Mpc" if d >= 100 else f"{d:.3g} Mpc"


def _short_list(names, n=3):
    names = list(names)
    return ", ".join(names) if len(names) <= n else f"{', '.join(names[:n])}, ... ({len(names)})"


def model_structure(model) -> Structure:
    """Everything ``latex()`` and ``display()`` show, read from the model."""
    csp = model.csp
    free_theta = {k: model.theta_init[k] for k in model.param_names}
    all_names = list(model.param_names) + list(model.transforms)
    obs_map = _obs_param_map(model, all_names)

    params: list[Param] = []
    derived_inputs: dict = {}
    for name, fn in model.transforms.items():
        inputs = transform_inputs(fn, free_theta)
        stage = _stage_of(model, name, obs_map)
        if inputs == ():
            try:
                value = np.asarray(fn(free_theta), dtype=float)
            except Exception:                                 # noqa: BLE001
                value = None
            params.append(Param(name, "fixed", stage, value=value, fn_name=_fn_name(fn)))
        else:
            # inputs None: the transform could not be traced, its inputs are unknown
            params.append(Param(name, "derived", stage, inputs=inputs, fn_name=_fn_name(fn)))
            for k in inputs or ():
                derived_inputs.setdefault(k, []).append(stage)
    for name in model.param_names:
        stage = (derived_inputs[name][0] if name in derived_inputs
                 and _stage_of(model, name, obs_map) == "stars" and name not in ("logmass",)
                 else _stage_of(model, name, obs_map))
        params.append(Param(name, "free", stage, prior=model.priors.get(name),
                            shape=tuple(np.shape(model.theta_init[name]))))

    # ---- stages, in the order the CSP applies them -----------------------------------------
    st: list[Stage] = []
    iso = getattr(csp, "_ssp_isoc_type", None)
    lib = getattr(csp, "_ssp_spec_library", None)
    imf = IMF_NAMES.get(getattr(csp, "_ssp_imf_type", None))
    grid = " + ".join(str(x).upper() if len(str(x)) <= 5 else str(x)
                      for x in (iso, lib) if x) or "SSP grid"
    sfh_t = np.asarray(getattr(csp, "sfh_times", []), dtype=float) / 1e9
    stars = Stage("stars", "Stellar populations", BLUE,
                  [grid + (f", {imf} IMF" if imf else "")])
    if sfh_t.size:
        stars.details.append(f"SFH on {sfh_t.size} lookback nodes ({getattr(csp, 'sfh_interp', '')}),"
                             f" {sfh_t.min():.3g}-{sfh_t.max():.3g} Gyr")
    stars.details.append("metallicity constant in time" if getattr(csp, "zh_const", False)
                         else "metallicity history")
    st.append(stars)
    if getattr(csp, "neb", None) is not None:
        neb = Stage("neb", "Nebular emission", TEAL,
                    [f"CLOUDY grid {getattr(getattr(csp.neb, 'cont_file', None), 'stem', '')}"
                     .rstrip(), "young stars ionise their birth cloud"])
        if getattr(csp, "gas_tied", False):
            neb.details.append("gas metallicity = stellar")
        st.append(neb)
    dust_lines = []
    if getattr(csp, "dust_attn", None) is not None:
        laws = getattr(csp.dust_attn, "law_names_resolved", [])
        edges = np.asarray(getattr(csp.dust_attn, "bin_edges", []), dtype=float)
        for law, e in zip(laws, edges):
            hi = 10.0 ** e[1] * 1e3 if np.isfinite(e[1]) else None
            dust_lines.append(f"birth cloud: {LAW_NAMES.get(law, law)}"
                              + (f", ages < {hi:.3g} Myr" if hi else ""))
    if getattr(csp, "diff_dust", None) is not None:
        dust_lines.append("diffuse: " + ", ".join(
            LAW_NAMES.get(x, x) for x in getattr(csp.diff_dust, "law_names_resolved", [])))
    if dust_lines:
        st.append(Stage("dust", "Dust attenuation", CORAL, dust_lines))
    if getattr(csp, "dust_emi", None) is not None:
        st.append(Stage("duste", "Dust emission", CORAL,
                        [f"{getattr(csp.dust_emi, 'duste_model', '')} templates",
                         "energy balance with the attenuation"]))
    if getattr(csp, "_agn", None) is not None:
        st.append(Stage("agn", "AGN", PLUM,
                        [_short_list(getattr(csp, "agn_param_names", []) or ["components"])]))
    k = model.kinematics

    def _w(v):
        return "sampled" if isinstance(v, str) else f"{float(v):g} km/s"
    from ..broadening import TIED
    kin = Stage("kin", "Galaxy kinematics", SLATE,
                [f"stars σ {_w(k.sigma_gal)}",
                 "gas σ = stars" if k.sigma_gas is TIED else f"gas σ {_w(k.sigma_gas)}"])
    st.append(kin)
    if model.zred_is_free:
        zline = "redshift sampled"
    elif model.lumdist_mpc is not None:
        zline = f"z = {model.zred:g}, D_L = {_mpc(model.lumdist_mpc)}"
    elif model.zred == 0.0:
        zline = "z = 0: no flux factor (L_sun/Hz)"
    else:
        zline = f"z = {model.zred:g}, D_L = {_mpc(model.cosmo.luminosity_distance(model.zred))}"
    flux = Stage("flux", "Mass, redshift, distance", SLATE, [zline])
    if "logmass" in all_names:
        flux.details.append("spectrum scaled by 10^logmass")
    st.append(flux)
    if getattr(csp, "igm", None) is not None:
        st.append(Stage("igm", "IGM absorption", TEAL, [type(csp.igm).__name__]))

    observations = []
    for o in model.observations:
        observations.append((o, _obs_details(o)))

    config = [("Cosmology", model.cosmo.describe())]
    return Structure(st, params, observations, config)


def _obs_details(o) -> list:
    kind = getattr(o, "_kind", getattr(o, "kind", ""))
    lines = []
    if kind == "photometry":
        f = [getattr(x, "name", str(x)) for x in getattr(o, "filters", [])]
        w = np.asarray(o.wavelength, dtype=float)
        lines.append(f"{len(f)} filters, {w.min() / 1e4:.3g}-{w.max() / 1e4:.3g} μm")
        if f:
            lines.append(f"{f[0]} ... {f[-1]}" if len(f) > 2 else ", ".join(f))
    elif kind == "spectrum":
        w = np.asarray(o.wavelength, dtype=float)
        lines.append(f"{w.size} pixels, {w.min() / 1e4:.4g}-{w.max() / 1e4:.4g} μm")
        ins = getattr(o, "instrument", None)
        if ins is not None:
            lines.append(_instrument_line(ins))
    elif kind == "lines":
        n = int(np.size(getattr(o, "flux", []) or []))
        lines.append(f"{n} line fluxes")
    lik = type(getattr(o, "noise", None)).__name__ if getattr(o, "noise", None) is not None else None
    if lik and lik != "NoneType":
        lines.append(f"noise model: {lik}")
    return lines


_INSTRUMENT_KIND = {"R_fwhm": ("R", ""), "R_sigma": ("R (σ)", ""), "fwhm_aa": ("FWHM", "Å"),
                    "sigma_aa": ("σ", "Å"), "sigma_kms": ("σ", "km/s"),
                    "fwhm_kms": ("FWHM", "km/s")}


def _instrument_line(ins) -> str:
    """The resolution of a Spectrum's Instrument in words."""
    name, unit = _INSTRUMENT_KIND.get(getattr(ins, "kind", ""), (str(getattr(ins, "kind", "")), ""))
    v = np.atleast_1d(np.asarray(getattr(ins, "value", np.nan), dtype=float))
    val = f"{v[0]:.4g}" if v.size == 1 else f"{np.nanmin(v):.4g}-{np.nanmax(v):.4g}"
    sep = " = " if name.startswith("R") else " "
    line = f"instrument {name}{sep}{val}" + (f" {unit}" if unit else "")
    sc = getattr(ins, "scale", 1.0)
    if isinstance(sc, str):
        line += " × sampled scale"
    elif float(sc) != 1.0:
        line += f" × {float(sc):g}"
    return line


# --------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------

def _num(x, nd=3) -> str:
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        m, e = f"{x:.{nd - 1}e}".split("e")
        return rf"{float(m):g}\times10^{{{int(e)}}}"
    return f"{x:.{nd}g}"


def _arr(v, nd=3) -> str:
    a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if a.size == 1 or np.all(a == a[0]):
        return _num(a[0], nd)
    if a.size <= 4:
        return "(" + ", ".join(_num(x, nd) for x in a) + ")"
    return "(" + ", ".join(_num(x, nd) for x in a[:3]) + rf", \ldots)"


def prior_latex(prior) -> str:
    """The prior as LaTeX maths (no surrounding $)."""
    if prior is None:
        return r"\text{flat (improper)}"
    cls = type(prior).__name__
    p = getattr(prior, "params", {})
    try:
        if cls in ("Uniform", "TopHat"):
            return rf"\mathcal{{U}}({_arr(p['low'])},\,{_arr(p['high'])})"
        if cls == "Normal":
            return rf"\mathcal{{N}}({_arr(p['mean'])},\,{_arr(p['sigma'])}^2)"
        if cls == "ClippedNormal":
            return (rf"\mathcal{{N}}({_arr(p['mean'])},\,{_arr(p['sigma'])}^2)"
                    rf"\ \text{{on}}\ [{_arr(p['low'])},\,{_arr(p['high'])}]")
        if cls == "LogNormal":
            k = [x for x in ("mode", "mean", "loc") if x in p]
            s = [x for x in ("sigma", "scale") if x in p]
            if k and s:
                return rf"\ln\mathcal{{N}}({_arr(p[k[0]])},\,{_arr(p[s[0]])}^2)"
        if cls == "LogUniform":
            lo = p.get("mini", p.get("low")); hi = p.get("maxi", p.get("high"))
            return rf"\log\mathcal{{U}}({_arr(lo)},\,{_arr(hi)})"
        if cls == "StudentT":
            return (rf"t_{{{_arr(p['df'])}}}(\mu={_arr(p['mean'])},\,"
                    rf"s={_arr(p['scale'])})")
        if "Multivariate" in cls:
            d = int(np.shape(p["mean"])[0])
            return rf"\mathcal{{N}}_{{{d}}}(\boldsymbol{{\mu}},\,\boldsymbol{{\Sigma}})"
    except Exception:                                         # noqa: BLE001
        pass
    return rf"\text{{{cls}}}"


def prior_text(prior) -> str:
    """The prior in plain text (for the diagram's cards)."""
    if prior is None:
        return "flat (no prior)"
    cls = type(prior).__name__
    p = getattr(prior, "params", {})

    def f(v):
        a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        x = a[0] if (a.size == 1 or np.all(a == a[0])) else None
        return f"{x:.3g}" if x is not None else "vector"
    try:
        if cls in ("Uniform", "TopHat"):
            return f"Uniform [{f(p['low'])}, {f(p['high'])}]"
        if cls == "Normal":
            return f"Normal({f(p['mean'])}, {f(p['sigma'])})"
        if cls == "ClippedNormal":
            return (f"Normal({f(p['mean'])}, {f(p['sigma'])}) on "
                    f"[{f(p['low'])}, {f(p['high'])}]")
        if cls == "StudentT":
            return f"Student-t (df {f(p['df'])}, {f(p['mean'])}, {f(p['scale'])})"
        if cls == "LogUniform":
            return f"log-Uniform [{f(p.get('mini', p.get('low')))}, {f(p.get('maxi', p.get('high')))}]"
        if "Multivariate" in cls:
            return f"{int(np.shape(p['mean'])[0])}-d Normal"
    except Exception:                                         # noqa: BLE001
        pass
    return cls


def prior_curve(prior, shape=(), n=120):
    """``(x, pdf)`` of the prior's first element over its central 99%, or None."""
    if prior is None or "Multivariate" in type(prior).__name__:
        return None
    try:
        import jax.numpy as jnp
        u = jnp.linspace(0.005, 0.995, n)
        p = prior.params
        scalar = all(np.size(v) == 1 for v in p.values())
        try:
            blo, bhi = (float(np.ravel(v)[0]) for v in prior.bounds)
        except Exception:                                     # noqa: BLE001
            blo = bhi = float("nan")
        if scalar and np.isfinite(blo) and np.isfinite(bhi) and bhi > blo:
            x = np.linspace(blo, bhi, n)
            lp = np.array(prior.logpdf(jnp.asarray(x)), dtype=float).ravel()
            lp[[0, -1]] = lp[[1, -2]]                         # the closed edges of the support
        elif scalar:
            x = np.asarray(prior.unit_transform(u), dtype=float).ravel()
            lp = np.asarray(prior.logpdf(jnp.asarray(x)), dtype=float).ravel()
        else:                                                 # a vector prior: element 0
            k = int(np.prod(shape)) or 1
            uu = jnp.repeat(u[:, None], k, axis=1)
            x = np.asarray(prior.unit_transform(uu), dtype=float)[:, 0]
            xx = jnp.repeat(jnp.asarray(x)[:, None], k, axis=1)
            lp = np.asarray(prior.logpdf(xx), dtype=float)
            lp = lp[:, 0] if lp.ndim == 2 else lp
        y = np.exp(lp - np.nanmax(lp))
        ok = np.isfinite(x) & np.isfinite(y)
        return (x[ok], y[ok]) if ok.sum() > 3 else None
    except Exception:                                         # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------------
# what each key means: symbol (LaTeX maths), description (LaTeX text), unit (LaTeX text).
# Read from the code that consumes each value (csp/csp.py, dust/attenuation_laws.py,
# neb/NebularGridModel.py, dust/DustEmission.py, igm.py, broadening.py, agn/*,
# likelihood/noise_model.py, likelihood/gp_likelihood.py, csp/spectrum_calibration.py).
# --------------------------------------------------------------------------------------------

_T5500 = r"optical depth at 5500\,\AA{}"
PARAM_INFO = {
    # stars, SFH, mass
    "sfh": (r"\mathrm{SFR}(t)", "star-formation rate at each lookback node",
            r"$M_\odot\,\mathrm{yr}^{-1}$ per $10^{\log M}$"),
    "logsfr_ratios": (r"\log(\mathrm{SFR}_i/\mathrm{SFR}_{i+1})",
                      "log ratio of the SFR in adjacent SFH nodes (index 0 = today)", "dex"),
    "lookback_time": (r"t_\mathrm{lb}", "SFH lookback-time nodes", "Gyr"),
    "logzsol": (r"\log(Z_\star/Z_\odot)", "stellar metallicity, constant in time",
                r"dex ($Z_\odot$ of the grid)"),
    "logzsol_hist": (r"\log(Z_\star/Z_\odot)(t)", "stellar metallicity at each lookback node",
                     r"dex ($Z_\odot$ of the grid)"),
    "afe": (r"[\alpha/\mathrm{Fe}]", r"$\alpha$-element enhancement of the SSPs", "dex"),
    "logmass": (r"\log(M/M_\odot)",
                r"mass scale: the spectrum is multiplied by $10^{\log M}$ (the mass formed "
                "when the SFH integrates to one solar mass)", r"dex ($M_\odot$)"),
    # distance
    "zred": (r"z", "redshift", ""),
    "lumdist_mpc": (r"D_L", "luminosity distance", "Mpc"),
    # escape / geometry
    "frac_obrun": (r"f_\mathrm{esc}",
                   "fraction of the young light that bypasses the birth cloud", ""),
    # nebular
    "gas_logz": (r"\log(Z_\mathrm{gas}/Z_\odot)", "gas-phase metallicity (CLOUDY grid axis)",
                 "dex"),
    "gas_logu": (r"\log U", "ionisation parameter (CLOUDY grid axis)", "dex"),
    # dust emission
    "duste_qpah": (r"q_\mathrm{PAH}", "PAH fraction of the dust-emission templates", r"\%"),
    "duste_umin": (r"U_\mathrm{min}", "minimum starlight intensity heating the dust", ""),
    "duste_gamma": (r"\gamma", r"dust mass fraction exposed to $U > U_\mathrm{min}$", ""),
    # IGM
    "igm_factor": (r"f_\mathrm{IGM}", "scales the IGM optical depth", ""),
    "x_HI": (r"x_\mathrm{HI}", "IGM neutral fraction (damping wing)", ""),
    "logN_HI": (r"\log N_\mathrm{HI}", "column density of a damped Ly$\\alpha$ absorber",
                r"dex (cm$^{-2}$)"),
    "z_dla": (r"z_\mathrm{DLA}", "redshift of the damped Ly$\\alpha$ absorber", ""),
    # lines
    "eline_scaling": (r"s_\mathrm{lines}", "scales the predicted line fluxes (Lines data)", ""),
    "eline_delta_zred": (r"\Delta z_\mathrm{lines}",
                         "velocity offset of the emission lines in a spectrum", ""),
    # AGN
    "fagn": (r"f_\mathrm{AGN}", "AGN torus luminosity relative to the stars", ""),
    "agn_tau": (r"\tau_\mathrm{AGN}", "optical depth of the torus templates", ""),
    "fagn_bbb": (r"f_\mathrm{disc}", r"AGN disc flux relative to the galaxy at 5500\,\AA", ""),
    "dust4": (r"\tau_\mathrm{AGN,screen}", _T5500 + " of the AGN dust screen", ""),
    "dust4_index": (r"n_\mathrm{AGN,screen}", "power-law index of the AGN dust screen", ""),
    "agn_elum": (r"L_{\mathrm{H}\beta,\mathrm{AGN}}", r"AGN narrow-line H$\beta$ luminosity",
                 r"$L_\odot$ per $M_\odot$ formed"),
    "agn_eline_sigma": (r"\sigma_\mathrm{NLR}", "velocity dispersion of the AGN narrow lines",
                        r"km\,s$^{-1}$"),
    "nlr_logz": (r"\log(Z_\mathrm{NLR}/Z_\odot)", "narrow-line-region metallicity", "dex"),
    "nlr_logu": (r"\log U_\mathrm{NLR}", "narrow-line-region ionisation parameter", "dex"),
    "nlr_xid": (r"\xi_d", "narrow-line-region dust-to-metal ratio", ""),
    "nlr_lognh": (r"\log n_\mathrm{H}", "narrow-line-region density", r"dex (cm$^{-3}$)"),
    "nlr_alpha": (r"\alpha_\mathrm{NLR}", "slope of the ionising AGN continuum", ""),
    "blr_lum": (r"L_{\mathrm{H}\alpha,\mathrm{BLR}}", r"broad-line H$\alpha$ luminosity",
                r"$L_\odot$ per $M_\odot$ formed"),
    "blr_sigma": (r"\sigma_\mathrm{BLR}", "velocity dispersion of the broad lines",
                  r"km\,s$^{-1}$"),
    "polar_ebv": (r"E(B-V)_\mathrm{polar}", "reddening by AGN polar dust", "mag"),
    "polar_temp": (r"T_\mathrm{polar}", "temperature of the AGN polar dust", "K"),
    "polar_emis": (r"\beta_\mathrm{polar}", "emissivity index of the AGN polar dust", ""),
    "skirtor_t": (r"t_\mathrm{SKIRTOR}", "torus optical depth (SKIRTOR grid)", ""),
    "skirtor_pl": (r"p_\mathrm{SKIRTOR}", "radial density slope (SKIRTOR grid)", ""),
    "skirtor_q": (r"q_\mathrm{SKIRTOR}", "polar density slope (SKIRTOR grid)", ""),
    "skirtor_oa": (r"\Delta_\mathrm{SKIRTOR}", "torus opening angle (SKIRTOR grid)", "deg"),
    "skirtor_i": (r"i_\mathrm{SKIRTOR}", "viewing angle from the pole (SKIRTOR grid)", "deg"),
}

#: attenuation-law keys: (symbol, description), as dust/attenuation_laws.py uses them
_DUST_KEYS = {
    "tau_pow": (r"\tau_\mathrm{pow}", _T5500 + " (power law)"),
    "alpha_pow": (r"\alpha_\mathrm{pow}", "slope of the power-law attenuation"),
    "tau_cal00": (r"\tau_\mathrm{Cal}", _T5500 + " (Calzetti 2000)"),
    "R_v_cal00": (r"R_{V,\mathrm{Cal}}", "total-to-selective attenuation (Calzetti 2000)"),
    "tau_kc": (r"\tau_\mathrm{KC}", _T5500 + " (Kriek \\& Conroy 2013)"),
    "dust_index": (r"\delta", "slope offset from the Calzetti curve (Kriek \\& Conroy 2013)"),
    "tau_noll": (r"\tau_\mathrm{Noll}", _T5500 + " (Noll et al. 2009)"),
    "delta": (r"\delta_\mathrm{Noll}", "slope offset (Noll et al. 2009)"),
    "c_r": (r"c_R", r"$R_V$ modification (Noll et al. 2009)"),
    "Ebump": (r"E_\mathrm{bump}", "2175\\,\\AA\\ bump strength (Noll et al. 2009)"),
    "tau_chev": (r"\tau_\mathrm{Chev}", _T5500 + " (Chevallard et al. 2013)"),
    "tau_con": (r"\tau_\mathrm{Con}", _T5500 + " (Conroy et al. 2010)"),
    "R_v_con": (r"R_{V,\mathrm{Con}}", "total-to-selective attenuation (Conroy et al. 2010)"),
    "f_bump": (r"f_\mathrm{bump}", "UV bump strength (Conroy et al. 2010)"),
    "tau_card": (r"\tau_\mathrm{CCM}", _T5500 + " (Cardelli et al. 1989)"),
    "R_v_card": (r"R_{V,\mathrm{CCM}}", "total-to-selective extinction (Cardelli et al. 1989)"),
    "tau_smc": (r"\tau_\mathrm{SMC}", _T5500 + " (SMC, Pei 1992)"),
    "tau_lmc": (r"\tau_\mathrm{LMC}", _T5500 + " (LMC, Pei 1992)"),
    "tau_g03smc": (r"\tau_\mathrm{G03}", _T5500 + " (SMC bar, Gordon et al. 2003)"),
    "tau_reddy": (r"\tau_\mathrm{R15}", _T5500 + " (Reddy et al. 2015)"),
    "x0": (r"x_0", "centre of the Drude profile"),
    "gamma": (r"\gamma_\mathrm{D}", "width of the Drude profile"),
}

#: per-observation families: stem -> (symbol, description, unit)
_OBS_KEYS = {
    "log_err_scale": (r"\ln s", "scales the quoted uncertainties by $s$", ""),
    "log_jitter": (r"\ln \sigma_\mathrm{jit}", "noise added in quadrature", "ln (data units)"),
    "log_f_calib": (r"\ln f_\mathrm{model}", "fractional model error added in quadrature", ""),
    "log_f_data": (r"\ln f_\mathrm{data}", "fractional data error added in quadrature", ""),
    "f_outlier": (r"f_\mathrm{out}", "fraction of outlying data points", ""),
    "nsigma_outlier": (r"n_\mathrm{out}", r"width of the outlier distribution in $\sigma$", ""),
    "spectrum_scaling": (r"s_\mathrm{spec}", "grey scaling of the model spectrum", ""),
    "spectrum_calib": (r"c_m", "Legendre coefficients of the spectrum calibration", ""),
    "log_gp_amp": (r"\ln a_\mathrm{GP}",
                   r"amplitude of the correlated residuals (in units of $\sigma$)", ""),
    "log_gp_length": (r"\ln \ell_\mathrm{GP}", "length scale of the correlated residuals",
                      r"ln \AA"),
}


def _tex_escape(s: str) -> str:
    """Plain text (the diagram's, with its Unicode) as LaTeX text."""
    s = (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
         .replace("&", r"\&").replace("#", r"\#"))
    s = s.replace("^", r"\^{}").replace(r"10\^{}logmass", r"$10^{\log M}$")
    for u, t in (("<", r"$<$"), (">", r"$>$"), ("μm", r"$\mu$m"), ("Å", r"\AA"), ("σ", r"$\sigma$"), ("×", r"$\times$"),
                 ("α", r"$\alpha$"), ("é", r"\'e"), ("D\_L", r"$D_L$")):
        s = s.replace(u, t)
    return s


def param_info(model, name: str):
    """``(symbol, description, unit)`` of a parameter, in LaTeX; unknown names get their own
    name as the symbol and an empty description."""
    if name in PARAM_INFO:
        return PARAM_INFO[name]
    k = model.kinematics
    if isinstance(k.sigma_gal, str) and name == k.sigma_gal:
        return (r"\sigma_\star", "stellar velocity dispersion", r"km\,s$^{-1}$")
    if isinstance(k.sigma_gas, str) and name == k.sigma_gas:
        return (r"\sigma_\mathrm{gas}", "gas velocity dispersion", r"km\,s$^{-1}$")
    for o in model.observations:
        if name in getattr(getattr(o, "instrument", None), "free_keys", ()):
            return (r"s_\mathrm{LSF}",
                    f"scales the instrumental resolution of \\texttt{{{_tex_escape(o.name)}}}", "")
    # attenuation laws: diffuse_<key>, <key>, <key><k> for a law used k times
    base, where = name, "birth cloud"
    if base.startswith("diffuse_"):
        base, where = base[len("diffuse_"):], "diffuse"
    stripped = base.rstrip("0123456789")
    if stripped in _DUST_KEYS:
        sym, desc = _DUST_KEYS[stripped]
        tag = where if stripped == base else f"{where}, age bin {base[len(stripped):]}"
        return (sym + (r"^\mathrm{diff}" if where == "diffuse" else ""), f"{desc}, {tag}", "")
    obs_map = _obs_param_map(model, [name])
    if name in obs_map:
        for stem, (sym, desc, unit) in sorted(_OBS_KEYS.items(), key=lambda kv: -len(kv[0])):
            if name == stem or name.startswith(stem + "_"):
                return (sym, f"{desc} (\\texttt{{{_tex_escape(obs_map[name].name)}}})", unit)
    return (rf"\texttt{{{_tex_escape(name)}}}", "", "")


# --------------------------------------------------------------------------------------------
# latex()
# --------------------------------------------------------------------------------------------

def latex(model, *, caption: str | None = None, label: str = "tab:sedmodel",
          settings: bool = True, table: bool = True) -> str:
    """A LaTeX table (``booktabs``) of every free, derived and fixed parameter of ``model``
    with its prior or value, and (``settings``) the fixed model settings."""
    s = model_structure(model)
    by_kind = {k: [p for p in s.params if p.kind == k] for k in ("free", "derived", "fixed")}
    order = [st.key for st in s.stages]

    def _rank(p):
        return (order.index(p.stage) if p.stage in order else len(order), p.name)

    rows = []

    def head(title):
        rows.append(r"\midrule" if rows else "")
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\")

    def row(p, value):
        sym, desc, unit = param_info(model, p.name)
        n = int(np.prod(p.shape)) if p.shape else 1
        key = rf"\texttt{{{_tex_escape(p.name)}}}" + (f" ({n})" if n > 1 else "")
        rows.append(rf"${sym}$ & {key} & {desc} & {value} & {unit} \\")

    if by_kind["free"]:
        head("Free parameters")
        for p in sorted(by_kind["free"], key=_rank):
            row(p, f"${prior_latex(p.prior)}$")
    if by_kind["derived"]:
        head("Derived parameters")
        for p in sorted(by_kind["derived"], key=_rank):
            if p.inputs is None:
                src = rf"\texttt{{{_tex_escape(p.fn_name)}}}(\ldots)"
            else:
                src = ", ".join(rf"\texttt{{{_tex_escape(k)}}}" for k in p.inputs)
                src = rf"\texttt{{{_tex_escape(p.fn_name)}}}({src})"
            row(p, src)
    if by_kind["fixed"]:
        head("Fixed parameters")
        for p in sorted(by_kind["fixed"], key=_rank):
            row(p, "--" if p.value is None else f"${_arr(p.value)}$")
    if settings:
        head("Model settings")
        for st in s.stages:
            text = "; ".join(_tex_escape(d) for d in st.details)
            rows.append(rf"\multicolumn{{2}}{{l}}{{{_tex_escape(st.title)}}} & "
                        rf"\multicolumn{{3}}{{p{{0.62\textwidth}}}}{{{text}}} \\")
        for o, d in s.observations:
            text = "; ".join(_tex_escape(x) for x in d)
            rows.append(rf"\multicolumn{{2}}{{l}}{{\texttt{{{_tex_escape(o.name)}}} "
                        rf"({_tex_escape(getattr(o, 'kind', ''))})}} & "
                        rf"\multicolumn{{3}}{{p{{0.62\textwidth}}}}{{{text}}} \\")
        for lab, val in s.config:
            rows.append(rf"\multicolumn{{2}}{{l}}{{{_tex_escape(lab)}}} & "
                        rf"\multicolumn{{3}}{{p{{0.62\textwidth}}}}{{{_tex_escape(val)}}} \\")
    body = "\n".join(r for r in rows if r)
    tab = "\n".join([r"\begin{tabular}{llp{0.3\textwidth}p{0.22\textwidth}l}", r"\toprule",
                     r"Symbol & Key & Description & Prior / value & Unit \\", r"\midrule",
                     body.removeprefix(r"\midrule" + "\n") if body.startswith(r"\midrule")
                     else body,
                     r"\bottomrule", r"\end{tabular}"])
    if not table:
        return tab
    cap = caption or ("Parameters of the SED model. Free parameters are sampled with the "
                      "priors given; derived parameters are computed from them; fixed "
                      "parameters are held at the values given.")
    return "\n".join([r"\begin{table*}", r"\centering", rf"\caption{{{cap}}}",
                      rf"\label{{{label}}}", tab, r"\end{table*}"])


# --------------------------------------------------------------------------------------------
# display()
# --------------------------------------------------------------------------------------------

def _mpl_math(sym: str) -> str:
    """A PARAM_INFO symbol as matplotlib mathtext."""
    return "$" + sym.replace(r"\texttt", r"\mathtt").replace(r"\,", r"\,") + "$"


def _renders(text: str) -> bool:
    from matplotlib.mathtext import MathTextParser
    try:
        MathTextParser("path").parse(text)
        return True
    except Exception:                                         # noqa: BLE001
        return False


def _symbol(model, name: str) -> str:
    sym = _mpl_math(param_info(model, name)[0])
    return sym if _renders(sym) else name


def _curve(ax, x0, y0, x1, y1, color, *, lw=1.1, alpha=0.75, ls="-", head=True, z=1):
    """A horizontal S-shaped edge from (x0, y0) to (x1, y1)."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    dx = max(0.35, 0.45 * (x1 - x0))
    path = Path([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-",
                                 mutation_scale=8, color=color, lw=lw, alpha=alpha,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=1))


def _card(ax, x, y, w, h, accent, *, face="white", edge="#d5d8dc", lw=0.9, z=3, bar=True):
    """A rounded card with a coloured bar on its left edge; (x, y) is its top-left corner."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=face, edgecolor=edge, lw=lw, zorder=z))
    if bar:
        ax.add_patch(Rectangle((x, y - h), 0.07, h, facecolor=accent, edgecolor="none",
                               zorder=z + 0.1))


def _sparkline(ax, prior, shape, x, y, w, h, color, z=4):
    """The prior's density inside the box (x, y-h)-(x+w, y), with its range below."""
    c = prior_curve(prior, shape)
    if c is None:
        return
    px, py = c
    lo, hi = float(px.min()), float(px.max())
    if not hi > lo:
        return
    X = x + (px - lo) / (hi - lo) * w
    Y = (y - h) + 0.12 + py * (h - 0.2)
    ax.fill_between(X, (y - h) + 0.12, Y, color=color, alpha=0.18, lw=0, zorder=z)
    ax.plot(X, Y, color=color, lw=1.1, zorder=z + 0.1, solid_capstyle="round")
    ax.plot([x, x + w], [(y - h) + 0.12] * 2, color="#c4c8cc", lw=0.6, zorder=z)
    for xx, v, ha in ((x, lo, "left"), (x + w, hi, "right")):
        ax.text(xx, (y - h) + 0.035, f"{v:.3g}", ha=ha, va="bottom", fontsize=5.6,
                color="#8a8f98", zorder=z)


def _stage_color(s: Structure, key: str) -> str:
    for st in s.stages:
        if st.key == key:
            return st.color
    return INK


def _stack(heights, top, gap):
    """Top y of each item, stacked down from ``top``; and the total height."""
    ys, y = [], top
    for h in heights:
        ys.append(y)
        y -= h + gap
    return ys, (top - y - gap) if heights else 0.0


def _value_text(p) -> str:
    return "$=\\," + ("?" if p.value is None else _arr(p.value)) + "$"


def _fixed_lines(model, ps, per_line=2):
    """The fixed parameters of a stage as text lines, ``per_line`` to a line."""
    items = [f"{_symbol(model, p.name)} {_value_text(p)}" for p in ps]
    return ["      ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def _free_card(ax, model, p, x, y, w, h, col):
    _card(ax, x, y, w, h, col)
    n = int(np.prod(p.shape)) if p.shape else 1
    ax.text(x + 0.2, y - 0.24, _symbol(model, p.name), fontsize=10.5, color=INK, va="center")
    ax.text(x + 0.2, y - h + 0.16, p.name + (f"  ×{n}" if n > 1 else ""), fontsize=6.8,
            family="monospace", color="#8a8f98", va="center")
    sx = x + w - 1.75
    ax.text(x + w - 0.1, y - 0.13, prior_text(p.prior), fontsize=6.2, color="#3d444b",
            ha="right", va="center")
    _sparkline(ax, p.prior, p.shape, sx, y - 0.21, 1.65, h - 0.24, col)


def display(model, ax=None, figsize=None, return_fig: bool = False, *, predictive: bool = True,
            n_draws: int = 64, seed: int = 0, title: str | None = None):
    """Draw ``model`` (see ``SedModel.display``)."""
    import matplotlib.pyplot as plt

    s = model_structure(model)
    stages = s.stages
    skeys = [st.key for st in stages]
    obs_names = ["obs:" + o.name for o, _ in s.observations]
    free = [p for p in s.params if p.kind == "free"]
    derived = [p for p in s.params if p.kind == "derived"]
    fixed = [p for p in s.params if p.kind == "fixed"]

    def rank(p):
        order = skeys + obs_names
        return order.index(p.stage) if p.stage in order else len(order)

    free.sort(key=lambda p: (rank(p), p.name))
    derived.sort(key=lambda p: (rank(p), p.name))
    model_free = [p for p in free if not p.stage.startswith("obs:")]
    obs_free = {n: [p for p in free if p.stage == n] for n in obs_names}
    fixed_by = {k: [p for p in fixed if p.stage == k] for k in skeys}
    fixed_lines = {k: _fixed_lines(model, fixed_by[k]) for k in skeys}

    # ---- geometry (inches) --------------------------------------------------------------------
    W = 18.0
    X1, W1 = 0.25, 4.45
    X2, W2 = 5.75, 2.55
    X3, W3 = 9.35, 3.85
    X4, W4 = 13.95, 3.85
    H_FREE, H_MINI, H_HEAD = 0.70, 0.62, 0.27
    LINE = 0.19

    col1 = []
    last = None
    for p in model_free:
        if p.stage != last:
            col1.append(("head", p.stage, H_HEAD))
            last = p.stage
        col1.append(("free", p, H_FREE))
    h_stage = {st.key: 0.44 + LINE * (len(st.details) + len(fixed_lines[st.key]))
               for st in stages}
    h_obs = {}
    for (o, d) in s.observations:
        n = "obs:" + o.name
        h_obs[n] = 0.48 + LINE * len(d)
    col4_h = [h_obs[n] + sum(H_MINI + 0.08 for _ in obs_free[n]) for n in obs_names]

    _, h1 = _stack([c[-1] for c in col1], 0, 0.07)
    _, h3 = _stack([h_stage[k] for k in skeys], 0, 0.32)
    _, h4 = _stack(col4_h, 0, 0.45)
    body = max(h1, h3, h4, 2.6)
    H = body + 1.25

    created = ax is None
    H_pred = 3.4 if (created and predictive) else 0.0
    if created:
        fw, fh = figsize if figsize is not None else (W * 0.8, (H + H_pred) * 0.8)
        fig = plt.figure(figsize=(fw, fh), facecolor="white")
        ax = fig.add_axes([0.0, H_pred / (H + H_pred), 1.0, H / (H + H_pred)])
    else:
        fig = ax.get_figure()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = H - 1.0

    def centred(total):
        return top - max(0.0, (body - total) / 2.0)

    # ---- header -------------------------------------------------------------------------------
    n_obs = len(s.observations)
    t = ax.text(0.25, H - 0.34, title or "SedModel", fontsize=15, fontweight="bold",
                color=INK, va="center")
    ax.annotate(f"{len(free)} free  ·  {len(derived)} derived  ·  {len(fixed)} fixed  ·  "
                f"{n_obs} observation{'s' if n_obs != 1 else ''}",
                xy=(1, 0.5), xycoords=t, xytext=(12, 0), textcoords="offset points",
                fontsize=10, color=SLATE, va="center")
    for x, w, lab in ((X1, W1, "FREE PARAMETERS AND PRIORS"), (X2, W2, "DERIVED"),
                      (X3, W3, "FORWARD MODEL"), (X4, W4, "OBSERVATIONS")):
        ax.text(x, H - 0.76, lab, fontsize=7.5, color=GREY, fontweight="bold", va="center")
        ax.plot([x, x + w], [H - 0.87] * 2, color="#e3e5e8", lw=0.8)

    edges = []                       # (x0, y0, target key, colour, kwargs)
    anchor = {}                      # key -> (x_left, top, height)

    # ---- column 3: the forward model ----------------------------------------------------------
    ys3, _ = _stack([h_stage[k] for k in skeys], centred(h3), 0.32)
    for st, y in zip(stages, ys3):
        h = h_stage[st.key]
        _card(ax, X3, y, W3, h, st.color, face="#fbfbfc", edge=st.color, lw=1.0)
        ax.text(X3 + 0.2, y - 0.21, st.title, fontsize=9.5, fontweight="bold", color=st.color,
                va="center")
        for i, d in enumerate(st.details):
            ax.text(X3 + 0.2, y - 0.44 - LINE * i, d.replace("10^logmass", "$10^{\\log M}$"),
                    fontsize=7.2, color="#3d444b", va="center")
        for i, line in enumerate(fixed_lines[st.key]):
            yy = y - 0.44 - LINE * (len(st.details) + i)
            if i == 0:
                ax.text(X3 + 0.2, yy, "fixed", fontsize=6.5, color=GREY, fontweight="bold",
                        va="center")
            ax.text(X3 + 0.62, yy, line, fontsize=7.4, color=INK, va="center")
        anchor[st.key] = (X3, y, h)
    for i in range(1, len(stages)):
        y0 = ys3[i - 1] - h_stage[skeys[i - 1]]
        ax.annotate("", xy=(X3 + W3 / 2, ys3[i] + 0.01), xytext=(X3 + W3 / 2, y0 - 0.01),
                    arrowprops=dict(arrowstyle="-|>", color="#9aa0a6", lw=1.3,
                                    mutation_scale=10), zorder=2)

    # ---- column 4: observations and their own parameters -------------------------------------
    ys4, _ = _stack(col4_h, centred(h4), 0.45)
    last_st = anchor[skeys[-1]]
    for (o, d), n, y in zip(s.observations, obs_names, ys4):
        h = h_obs[n]
        kind = getattr(o, "kind", "")
        _card(ax, X4, y, W4, h, INK, face="white", edge="#b9bec4", lw=1.0)
        tt = ax.text(X4 + 0.2, y - 0.21, kind.capitalize(), fontsize=9.5, fontweight="bold",
                     color=INK, va="center")
        ax.annotate(o.name, xy=(1, 0.5), xycoords=tt, xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, family="monospace", color=SLATE, va="center")
        for i, line in enumerate(d):
            ax.text(X4 + 0.2, y - 0.44 - LINE * i, line, fontsize=7.2, color="#3d444b",
                    va="center")
        f = getattr(o, "flux", None)
        nd = int(np.size(f)) if f is not None else 0
        if nd:
            ax.plot(X4 + W4 - 0.25, y - 0.21, "o", ms=6.5, color=CRIM, zorder=6)
            ax.text(X4 + W4 - 0.4, y - 0.21, f"data ({nd})", fontsize=6.8, color=CRIM,
                    ha="right", va="center")
        anchor[n] = (X4, y, h)
        _curve(ax, X3 + W3, last_st[1] - last_st[2] / 2, X4, y - min(h / 2, 0.3), "#9aa0a6",
               lw=1.4, alpha=0.9)
        ym = y - h - 0.08
        for p in obs_free[n]:
            _free_card(ax, model, p, X4 + 0.3, ym, W4 - 0.3, H_MINI, INK)
            ax.plot([X4 + 0.15, X4 + 0.15, X4 + 0.3], [y - h, ym - H_MINI / 2, ym - H_MINI / 2],
                    color="#b9bec4", lw=1.0, zorder=2)
            ym -= H_MINI + 0.08

    # ---- column 2: derived parameters, level with the stage they enter ----------------------
    H_DER = 0.52
    next_free = top
    for p in derived:
        col = _stage_color(s, p.stage) if p.stage in anchor and not p.stage.startswith("obs:") \
            else INK
        tgt = anchor.get(p.stage)
        y = min(tgt[1] if tgt else next_free, next_free)
        _card(ax, X2, y, W2, H_DER, col, face="white", edge=col, lw=0.9)
        ax.text(X2 + 0.2, y - 0.18, _symbol(model, p.name), fontsize=10, color=INK, va="center")
        ax.text(X2 + W2 - 0.1, y - 0.18, p.name, fontsize=6.8, family="monospace", color=GREY,
                ha="right", va="center")
        ax.text(X2 + 0.2, y - 0.38, f"= {p.fn_name}(" + (", ".join(p.inputs) if p.inputs
                                                         else "...") + ")",
                fontsize=6.6, color=SLATE, family="monospace", va="center")
        anchor["derived:" + p.name] = (X2, y, H_DER)
        if tgt is not None:
            edges.append((X2 + W2, y - H_DER / 2, p.stage, col, dict(lw=1.5)))
        next_free = y - H_DER - 0.12

    # ---- column 1: the model's free parameters -----------------------------------------------
    heads = {"stars": "stellar populations", "neb": "nebular emission", "dust": "dust",
             "duste": "dust emission", "agn": "AGN", "flux": "mass and distance",
             "igm": "IGM", "kin": "kinematics"}
    ys1, _ = _stack([c[-1] for c in col1], centred(h1), 0.07)
    for (kind, payload, h), y in zip(col1, ys1):
        if kind == "head":
            ax.text(X1 + 0.02, y - 0.15, heads.get(payload, payload).upper(), fontsize=6.8,
                    fontweight="bold", color=_stage_color(s, payload), va="center")
            continue
        p = payload
        col = _stage_color(s, p.stage)
        _free_card(ax, model, p, X1, y, W1, h, col)
        outs = [d for d in derived if d.inputs and p.name in d.inputs]
        for d in outs:
            edges.append((X1 + W1, y - h / 2, "derived:" + d.name, col, {}))
        if not outs:
            edges.append((X1 + W1, y - h / 2, p.stage, col, {}))

    # ---- edges: spread the arrivals along each target's left edge ---------------------------
    by_target: dict = {}
    for e in edges:
        by_target.setdefault(e[2], []).append(e)
    for key, es in by_target.items():
        if key not in anchor:
            continue
        x, y, h = anchor[key]
        es.sort(key=lambda e: -e[1])
        lo, hi = y - h + 0.12, y - 0.12
        ts = [0.5] if len(es) == 1 else np.linspace(0, 1, len(es))
        for e, tt in zip(es, ts):
            yy = hi - tt * (hi - lo) if len(es) > 1 else y - min(0.24, h / 2)
            _curve(ax, e[0], e[1], x, yy, e[3], **e[4])

    if created and predictive:
        _predictive_panel(fig, model, H_pred / (H + H_pred), n_draws, seed)

    if return_fig:
        return fig, ax
    if created:
        plt.show()
    return None


def _predictive_panel(fig, model, frac, n_draws, seed):
    """Prior-predictive 68/95% range of every Photometry and Spectrum, with the data."""
    import jax.numpy as jnp
    from ..plotting import _nice_log_ticks, _prior_draws
    draws = _prior_draws(model, n_draws, seed=seed)
    pax = fig.add_axes([0.06, 0.2 * frac, 0.92, 0.68 * frac])
    if draws is None:
        pax.text(0.5, 0.5, "prior predictive: a free parameter has no prior", ha="center",
                 va="center", transform=pax.transAxes, color="#8a8f98")
        pax.axis("off")
        return
    pred = model.predict_vmap({k: jnp.asarray(v) for k, v in draws.items()})
    physical = model._zred_fixed is not None or model.zred_is_free
    to_ujy = {"photometry": 3631e6 if physical else 1.0, "spectrum": 1e29 if physical else 1.0}
    lo_all, hi_all, phot_q = [], [], []
    for o in model.observations:
        kind = getattr(o, "kind", "")
        if kind not in to_ujy or o.name not in pred:
            continue
        c = to_ujy[kind]
        w = np.asarray(o.wavelength, dtype=float) / 1e4
        y = np.asarray(pred[o.name], dtype=float) * c
        q = np.nanpercentile(y, [2.5, 16, 50, 84, 97.5], axis=0)
        m = np.asarray(getattr(o, "mask", np.ones(w.size, bool)), dtype=bool)
        m = m if m.shape == w.shape else np.ones(w.size, bool)
        if kind == "photometry":
            pax.vlines(w, q[0], q[4], color=BLUE, alpha=0.35, lw=5, zorder=2)
            pax.vlines(w, q[1], q[3], color=BLUE, alpha=0.8, lw=5, zorder=3)
            pax.plot(w, q[2], "_", color="white", ms=7, mew=1.2, zorder=4)
            phot_q.append((w, q[3]))
        else:
            pax.fill_between(w, q[0], q[4], color=BLUE, alpha=0.15, lw=0, zorder=1)
            pax.fill_between(w, q[1], q[3], color=BLUE, alpha=0.35, lw=0, zorder=1)
        lo_all.append(np.nanmin(q[1][q[1] > 0]) if np.any(q[1] > 0) else np.nan)
        hi_all.append(np.nanmax(q[3]))
        lo_all[-1] = max(lo_all[-1], hi_all[-1] * 1e-5)       # not decades below the band
        f = getattr(o, "flux", None)
        if f is not None:
            f = np.asarray(f, dtype=float) * c
            e = np.asarray(getattr(o, "uncertainty", np.zeros_like(f)), dtype=float) * c
            if kind == "photometry":
                pax.errorbar(w[m], f[m], yerr=e[m], fmt="o", ms=4, color=CRIM, mec="white",
                             mew=0.6, elinewidth=1, zorder=6)
            else:
                pax.plot(w[m], f[m], color=CRIM, lw=0.6, alpha=0.9, zorder=5)
            pos = f[m][f[m] > 0]
            if pos.size:
                lo_all.append(np.nanpercentile(pos, 1))
                hi_all.append(np.nanpercentile(pos, 99))
    pax.set_xscale("log")
    pax.set_yscale("log")
    lo, hi = np.nanmin(lo_all), np.nanmax(hi_all)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo > 0:
        pax.set_ylim(lo / 3.0, hi * 3.0)
        for w, q84 in phot_q:                 # bands entirely below the axis (e.g. a Lyman break)
            below = q84 < lo / 3.0
            if below.any():
                pax.plot(w[below], np.full(below.sum(), lo / 3.0 * 1.6), "v", color=BLUE,
                         ms=6, alpha=0.8, zorder=4)
    xl = pax.get_xlim()
    _nice_log_ticks(pax.xaxis, xl[0], xl[1])
    pax.set_xlabel("observed wavelength [μm]", fontsize=9)
    pax.set_ylabel("$F_\\nu$ [μJy]" if physical else "$L_\\nu$ [model units]", fontsize=9)
    pax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        pax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    pax.legend(handles=[Patch(color=BLUE, alpha=0.8, label="prior predictive, 68%"),
                        Patch(color=BLUE, alpha=0.3, label="95%"),
                        Line2D([], [], marker="o", color=CRIM, ls="", label="data")],
               fontsize=7.5, frameon=False, loc="upper left")
    pax.set_title(f"What the priors allow: {n_draws} prior draws through the model",
                  fontsize=9.5, loc="left", color=INK)
