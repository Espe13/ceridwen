"""
Misuse report for CERIDWEN — exercises the common user-error scenarios from the
footgun analysis and records, for each, how the package now responds:

  ERROR     — raises a clear exception (loud, good)
  WARN      — emits a warnings.warn (caught, good)
  BY-DESIGN — intentional supported fallback (no error expected)
  SILENT    — returns a (possibly wrong) value with no error/warning (bad)

Produces ``figures/misuse_report.png`` (green = caught loudly/with a warning or
by design; red = still silent) and prints a table.  No FSPS needed
(add_neb / add_dust off), so it is fast.

Run:  python tests/regression/misuse_report.py
"""
from __future__ import annotations

import os
import sys
import warnings
import pathlib

# Make the shared test helper importable when run as a standalone script
# (under pytest, tests/conftest.py already puts tests/ on sys.path).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ceridwen.ssps.ssp_data import SSPData
from ceridwen.csp.csp import CSPBasis
from ceridwen.cosmology import Cosmology
from ceridwen.observation.observation import Photometry, Spectrum, Lines
from ceridwen.model.model import SedModel
from ceridwen.broadening import Instrument
from ceridwen.likelihood import DiagonalNoiseModel, DiagonalGaussianLikelihood
from ceridwen.likelihood.eline_marginal import refuse_outlier_with_elines
from ceridwen.fit import (_check_outlier_setup, _check_noise_setup, _poly_calibration_for,
                          _resolve_fit_mfrac)
from ceridwen.model.obs_params import check_names, CALIB_FAMILIES
from ceridwen.priors import LogUniform, Normal, TopHat
from types import SimpleNamespace

from _gridfixture import require_test_grid

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
REPO = HERE.parent.parent
SSP = str(require_test_grid())

_ssp = SSPData.load(SSP)
_T = 13.8
_lb = jnp.linspace(0.0, _T, 10)   # NEW convention: today @ idx 0
_sfr = jnp.exp(-0.5 * ((_lb - 0.05) / 0.03) ** 2) + 0.7 * jnp.exp(-0.5 * ((_lb - 11.) / 0.8) ** 2)


def _kw(**over):
    kw = dict(cosmo=Cosmology.planck18(), zh_const=True, add_dust=False, add_diffuse_dust=False,
              add_dust_emission=False, add_neb=False, add_igm=False,
              verbose=False, sfh_interp="linear")
    kw.update(over)
    return kw


def _run(fn):
    """Run fn, classify outcome as ERROR / WARN / SILENT."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            return ("WARN", str(w[-1].message)[:90]) if w else ("SILENT", "")
        except Exception as e:
            return ("ERROR", f"{type(e).__name__}: {str(e)[:80]}")


def _without_table(csp):
    """A copy of ``csp`` without the grid's surviving-mass table (the published grids carry one)."""
    import copy
    c = copy.copy(csp)
    c.ssp_stellar_mass = None
    return c


def _good_csp():
    return CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "logzsol": jnp.array([-0.15])},
                    **_kw())


def _outlier_model(names, priors, n_spec=1):
    """The attributes fit._check_outlier_setup reads, for a spectrum-only model."""
    return SimpleNamespace(observations=[SimpleNamespace(kind="spectrum", name=f"s{i}")
                                         for i in range(n_spec)],
                           param_names=list(names), transforms={}, priors=priors,
                           theta_init={n: jnp.array([0.1]) for n in names})


def _gp_setup(sampled, n_spec=1, transforms=None, **spec_kw):
    """fit._likelihood_for on the first of ``n_spec`` spectra with the GP names ``sampled``."""
    from ceridwen.fit import _likelihood_for
    specs = [Spectrum(wavelength=jnp.linspace(5000, 5600, 40), flux=jnp.ones(40),
                      uncertainty=jnp.ones(40), name=f"s{i}", **(spec_kw if i == 0 else {}))
             for i in range(n_spec)]
    m = SimpleNamespace(observations=specs, param_names=list(sampled),
                        transforms=dict(transforms or {}), priors={},
                        theta_init={n: jnp.array([0.1]) for n in sampled})
    return _likelihood_for(specs[0], m.param_names, model=m)


_GP = ["log_gp_amp_spec", "log_gp_length_spec"]


def _short_csp(**over):
    """A CSP whose oldest node fits inside the Universe at zred = 0.5 (8.59 Gyr)."""
    lb = jnp.linspace(0.0, 8.0, 10)
    return CSPBasis(_ssp, theta={"lookback_time": lb, "sfh": jnp.ones(10),
                                 "logzsol": jnp.array([-0.15])}, **_kw(**over))


def _model_with(priors=None, csp_obj=None):
    """A SedModel on a short-grid CSP with one photometric observation (for prior checks)."""
    c = csp_obj if csp_obj is not None else _short_csp()
    ph = Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1e-9, 1e-9],
                    uncertainty=[1e-10, 1e-10], name="p")
    return SedModel(c, [ph], priors=priors or {}, zred=0.5)


def _unknown_zsun_grid():
    """An in-memory grid that is in no metadata table and records no Z_sun."""
    import dataclasses
    lgmet = jnp.asarray(np.asarray(_ssp.ssp_lgmet) + 0.123)      # perturbed -> new chash
    return SSPData(lgmet, _ssp.ssp_lg_age_gyr, _ssp.ssp_wave, _ssp.ssp_flux,
                   ssp_resolution=_ssp.ssp_resolution)


def _tied_gas_with_prior():
    if not os.environ.get("SPS_HOME"):
        raise RuntimeError("SPS_HOME unset: gas_tied needs the nebular grids (scenario n/a)")
    c = _short_csp(add_neb=True, gas_tied=True, sps_home=os.environ["SPS_HOME"])
    return _model_with(priors={"gas_logz": TopHat(low=-1.0, high=0.2)}, csp_obj=c)


def _afe_refused_cell():
    """The (logzsol, afe) corner built on FSPS's duplicated isoc_feh_p050_afe_p6 file."""
    from ceridwen.ssps.ssp_data_afe import SSPDataAfe
    from ceridwen.csp.csp_afe import CSPBasis_afe
    path = REPO / "ceridwen" / "data" / "test_data" / "amist_c3k_lr_chab_afe.h5"
    if not path.is_file():
        raise RuntimeError("alpha grid not present (scenario n/a)")
    ssp = SSPDataAfe.load(str(path))
    c = CSPBasis_afe(ssp, theta={"lookback_time": jnp.linspace(0.0, 8.0, 10),
                                 "sfh": jnp.ones(10),
                                 "logzsol": jnp.array([0.0]), "afe": jnp.array([0.0])},
                     **{k: v for k, v in _kw().items() if k != "add_neb"})
    return _model_with(priors={"logzsol": TopHat(low=-1.0, high=0.5),
                               "afe": TopHat(low=-0.2, high=0.6)}, csp_obj=c)


def _dla_csp(igm, **over):
    from ceridwen.igm import MadauDampingDLA  # noqa: F401
    return _short_csp(add_igm=True, igm_model=igm, **over)


def _sampled_x_hi_without_ob0():
    c = _dla_csp("madau1995_damping_dla")
    th = dict(c.theta_init, zred=jnp.array([0.5]), x_HI=jnp.array([0.5]))
    ph = Photometry(filters=["sdss_g0", "sdss_r0"], name="p")
    ph.setup_for_model(c.wave, zred=0.5)
    return c.predict(th, [ph])


def _noll_old_bump_name():
    c = _short_csp(add_dust=True, init_dust_params={"bin_edges": [(-jnp.inf, -1.97)],
                                                    "laws": ["noll"]})
    return c.get_spectrum_components(dict(c.theta_init, E_bump=jnp.array([2.0])))


def run_scenarios():
    csp = _good_csp()
    th = dict(csp.theta_init)

    scenarios = [
        # label, expected-good-kinds, fn
        ("zh_const=True, no 'logzsol'", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr}, **_kw())),
        ("zh_const=False, no 'logzsol_hist'", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "logzsol": jnp.array([-0.15])},
                          **_kw(zh_const=False))),
        ("NaN in sfh", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr.at[0].set(jnp.nan),
                                       "logzsol": jnp.array([-0.15])}, **_kw())),
        ("negative SFR", {"WARN"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": -jnp.abs(_sfr),
                                       "logzsol": jnp.array([-0.15])}, **_kw())),
        ("missing 'sfh'", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "logzsol": jnp.array([-0.15])}, **_kw())),
        ("wrong 'logzsol' shape", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "logzsol": jnp.zeros(10)}, **_kw())),
        ("wrong 'sfh' length", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": jnp.ones(13),
                                       "logzsol": jnp.array([-0.15])}, **_kw())),
        ("sfh_interp typo", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr, "logzsol": jnp.array([-0.15])},
                          **_kw(sfh_interp="steppe"))),
        ("typo theta key (logmas)", {"WARN"},
         lambda: csp.get_spectrum_components({**th, "logmas": jnp.array([10.0])})),
        ("logzsol outside metallicity grid", {"WARN"},
         lambda: csp.check_param_ranges({**th, "logzsol": jnp.array([3.0])})),
        ("old absolute key theta['Z']", {"ERROR"},
         lambda: csp.check_param_ranges({**th, "Z": jnp.array([-1.85])})),  # deliberate misuse
        ("old absolute key theta['zh']", {"ERROR"},
         lambda: CSPBasis(_ssp, theta={"lookback_time": _lb, "sfh": _sfr,
                                       "zh": jnp.full(10, -1.85)},   # deliberate misuse
                          **_kw(zh_const=False))),
        ("old absolute prior name 'Z' on the model", {"ERROR"},
         lambda: _model_with(priors={"Z": TopHat(low=-3.5, high=-1.5)})),  # deliberate misuse
        ("absolute-looking logzsol value (-1.85)", {"WARN"},
         lambda: csp.check_param_ranges({**th, "logzsol": jnp.array([-1.85])})),
        ("absolute-looking logzsol prior", {"WARN"},
         lambda: _model_with(priors={"logzsol": TopHat(low=-2.2, high=-1.5)})),
        ("logzsol prior wider than the grid", {"ERROR"},
         lambda: _model_with(priors={"logzsol": TopHat(low=-4.0, high=1.0)})),
        ("grid with unknown Z_sun (no provenance, not in the table)", {"ERROR"},
         lambda: _unknown_zsun_grid()),
        ("zsun= disagreeing with the grid's own solar node", {"ERROR"},
         lambda: SSPData.load(SSP, zsun=0.0142)),
        ("gas_tied=True plus an explicit gas_logz prior", {"ERROR"},
         lambda: _tied_gas_with_prior()),
        ("alpha grid: logzsol x afe priors reach the refused cell", {"ERROR"},
         lambda: _afe_refused_cell()),
        ("Spectrum.predict before setup", {"ERROR"},
         lambda: Spectrum(wavelength=jnp.linspace(4000, 7000, 40), flux=jnp.ones(40),
                          uncertainty=jnp.ones(40), name="s").predict(
                          csp.get_spectrum(th), csp.wave)),
        ("Lines.predict before setup", {"ERROR"},
         lambda: Lines(line_ind=[0, 1], line_names=["a", "b"], wavelength=[4861., 6563.],
                       name="l").predict(csp.get_spectrum(th), csp.wave)),
        ("unknown filter name", {"ERROR"},
         lambda: Photometry(filters=["not_a_filter_xyz"], name="p").setup_for_model(csp.wave)),
        ("all-zero uncertainty", {"ERROR"},
         lambda: (lambda sp: (sp.setup_for_model(csp.wave), sp.chi_sq(sp.predict(csp.get_spectrum(th), csp.wave)))) (
                  Spectrum(wavelength=jnp.linspace(4000, 7000, 40), flux=jnp.ones(40),
                           uncertainty=jnp.zeros(40), name="s"))),
        ("Photometry.predict before setup", {"ERROR"},
         lambda: Photometry(filters=["sdss_g0"], name="p").predict(csp.get_spectrum(th), csp.wave)),
        ("eline prior width without a nebular model", {"ERROR"},
         lambda: SedModel(csp, [Spectrum(wavelength=jnp.linspace(4000, 7000, 400), flux=jnp.ones(400),
                                         uncertainty=jnp.ones(400), name="s",
                                         instrument=Instrument.R_fwhm(1000.0),
                                         marginalize_elines=True, eline_prior_width=0.2)], zred=0.0)),
        ("Spectrum(eline_sigma=...)", {"ERROR"},
         lambda: Spectrum(wavelength=jnp.linspace(4000, 7000, 40), name="s",  # deliberate misuse
                          instrument=Instrument.R_fwhm(1000.0), marginalize_elines=True,
                          eline_sigma=100.0)),
        ("marginalize_elines without instrument", {"ERROR"},
         lambda: Spectrum(wavelength=jnp.linspace(4000, 7000, 40), name="s",
                          marginalize_elines=True)),
        ("fixed f_outlier outside [0, 1)", {"ERROR"},
         lambda: DiagonalNoiseModel(f_outlier=1.5)),
        ("outlier mixture inside marginalize_elines", {"ERROR"},
         lambda: refuse_outlier_with_elines(("s",), (DiagonalGaussianLikelihood(
             DiagonalNoiseModel(f_outlier=0.1)),), ("s",))),
        ("sampled f_outlier missing from theta", {"ERROR"},
         lambda: DiagonalGaussianLikelihood(DiagonalNoiseModel(f_outlier="f_outlier_spec"))(
             jnp.ones(3), jnp.ones(3), jnp.ones(3), jnp.ones(3, bool), {})),
        ("f_outlier_spec with two spectra", {"ERROR"},
         lambda: _check_outlier_setup(_outlier_model(["f_outlier_spec"],
                                                     {"f_outlier_spec": TopHat(low=1e-5, high=0.5)},
                                                     n_spec=2))),
        ("f_outlier_spec_<unknown obs name>", {"ERROR"},
         lambda: _check_outlier_setup(_outlier_model(["f_outlier_spec_nope"],
                                                     {"f_outlier_spec_nope": TopHat(low=1e-5, high=0.5)}))),
        ("NaN flux not in the mask", {"WARN"},
         lambda: Photometry(filters=["sdss_g0", "sdss_r0"], flux=[1.0, float("nan")],
                            uncertainty=[0.1, 0.1], name="p")),
        ("nsigma_outlier without f_outlier", {"ERROR"},
         lambda: _check_outlier_setup(_outlier_model(["nsigma_outlier_spec"],
                                                     {"nsigma_outlier_spec": TopHat(low=2.0, high=80.0)}))),
        ("f_outlier_phot without Photometry", {"ERROR"},
         lambda: _check_outlier_setup(_outlier_model(["f_outlier_phot"],
                                                     {"f_outlier_phot": TopHat(low=0.0, high=0.5)}))),
        ("mfrac on a grid without a surviving-mass table", {"ERROR"},
         lambda: _without_table(csp).surviving_mass_fraction(th)),
        ("fitSED(mfrac=True) on a grid without a surviving-mass table", {"ERROR"},
         lambda: _resolve_fit_mfrac(SimpleNamespace(csp=_without_table(csp)), True)),
        ("fitSED(mfrac='yes') (not a bool)", {"ERROR"},
         lambda: _resolve_fit_mfrac(SimpleNamespace(csp=csp), "yes")),
        ("surviving-mass table of the wrong shape", {"ERROR"},
         lambda: _ssp.with_stellar_mass(np.ones((2, 2)), source="misuse")),
        ("old shared noise name log_jitter (v1.0.7)", {"ERROR"},
         lambda: _check_noise_setup(_outlier_model(["log_jitter"], {}))),
        ("log_jitter_spec with two spectra", {"ERROR"},
         lambda: _check_noise_setup(_outlier_model(["log_jitter_spec"], {}, n_spec=2))),
        ("log_jitter_spec_<unknown obs name>", {"ERROR"},
         lambda: _check_noise_setup(_outlier_model(["log_jitter_spec_nope"], {}))),
        ("plain spectrum_scaling with two spectra", {"ERROR"},
         lambda: check_names(_outlier_model([], {}, n_spec=2).observations, CALIB_FAMILIES,
                             {"spectrum_scaling"})),
        ("profiled polynomial + sampled spectrum_calib", {"ERROR"},
         lambda: _poly_calibration_for(
             Spectrum(wavelength=jnp.linspace(4000, 7000, 40), name="s0", polynomial_order=2),
             _outlier_model(["spectrum_calib"], {}))),
        ("negative polynomial_order", {"ERROR"},
         lambda: Spectrum(wavelength=jnp.linspace(4000, 7000, 40), name="s",
                          polynomial_order=-1)),
        ("unbounded prior on f_outlier_spec", {"ERROR"},
         lambda: _check_outlier_setup(_outlier_model(["f_outlier_spec"],
                                                     {"f_outlier_spec": Normal(mean=0.1, sigma=0.1)}))),
        ("GP: plain log_gp_amp_spec with two spectra", {"ERROR"},
         lambda: _gp_setup(_GP, n_spec=2)),
        ("GP: log_gp_amp_spec without log_gp_length_spec", {"ERROR"},
         lambda: _gp_setup(_GP[:1])),
        ("GP: GaussianProcess object and sampled GP names", {"ERROR"},
         lambda: _gp_setup(_GP, noise=__import__("ceridwen.observation", fromlist=["x"])
                           .GaussianProcess(1.0, 10.0))),
        ("GP: log_gp_amp_phot (only spectra take a GP)", {"ERROR"},
         lambda: _gp_setup(_GP + ["log_gp_amp_phot"])),
        ("GP + outlier mixture on one spectrum", {"ERROR"},
         lambda: _gp_setup(_GP, transforms={"f_outlier_spec": lambda th: jnp.array([0.1])})),
        ("GP + profiled calibration polynomial", {"ERROR"},
         lambda: _gp_setup(_GP, polynomial_order=2)),
        ("GP + marginalize_elines", {"ERROR"},
         lambda: _gp_setup(_GP, marginalize_elines=True, instrument=Instrument.R_fwhm(2000.0))),
        ("IGM damping wing (x_HI > 0) without Ob0", {"ERROR"},
         lambda: __import__("ceridwen.igm", fromlist=["x"]).MadauDampingDLA(x_HI=0.5)),
        ("sampled x_HI on a DLA model without Ob0", {"ERROR"}, _sampled_x_hi_without_ob0),
        ("IGM model cosmology != CSP cosmology", {"ERROR"},
         lambda: _dla_csp(__import__("ceridwen.igm", fromlist=["x"]).MadauDampingDLA(
             Ob0=0.05, cosmo=Cosmology.wmap9()))),
        ("duste_model='THEMIS' without dust emission", {"ERROR"},
         lambda: _short_csp(duste_model="THEMIS")),
        ("noll bump under its old name E_bump", {"WARN"}, _noll_old_bump_name),
        ("LogUniform with mini <= 0 (log of 0)", {"ERROR"},
         lambda: LogUniform(mini=0.0, maxi=1.0)),
        ("Instrument LSF scale <= 0", {"ERROR"},
         lambda: Instrument.R_fwhm(1000.0, scale=0.0)),
        ("sampled LSF scale with an unbounded prior", {"ERROR"},
         lambda: SedModel(csp, [Spectrum(wavelength=jnp.linspace(4000, 7000, 400), name="s",
                                         instrument=Instrument.R_fwhm(1000.0, scale="lsf_scale"))],
                          priors={"lsf_scale": Normal(mean=1.0, sigma=0.1)},
                          free_param_init={"lsf_scale": 1.0}, zred=0.0)),
        ("sampled LSF scale not in theta", {"ERROR"},
         lambda: SedModel(csp, [Spectrum(wavelength=jnp.linspace(4000, 7000, 400), name="s",
                                         instrument=Instrument.R_fwhm(1000.0, scale="lsf_scale"))],
                          zred=0.0)),
    ]

    rows = []
    for label, good, fn in scenarios:
        kind, detail = _run(fn)
        ok = kind in good
        rows.append((label, kind, ok, detail, good))
    return rows


def make_misuse_figure(rows=None, outdir=FIG_DIR):
    if rows is None:
        rows = run_scenarios()
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in rows]
    kinds = [r[1] for r in rows]
    oks = [r[2] for r in rows]
    color = {"ERROR": "#2ca02c", "WARN": "#1f9e1f", "BY-DESIGN": "#7fbf7f", "SILENT": "#d62728"}
    # green if handled as expected, red otherwise
    bar_colors = ["#2ca02c" if ok else "#d62728" for ok in oks]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(labels) + 1.5))
    ax.barh(y, [1] * len(labels), color=bar_colors, alpha=0.85)
    for yi, (lab, kind, ok, detail, good) in zip(y, rows):
        ax.text(0.02, yi, f"{lab}", va="center", ha="left", fontsize=8,
                color="white", fontweight="bold")
        ax.text(0.98, yi, kind, va="center", ha="right", fontsize=8,
                color="white", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([])
    n_ok = sum(oks)
    ax.set_title(f"CERIDWEN misuse report — {n_ok}/{len(labels)} user mistakes "
                 f"now caught (green=loud error/warning or by-design, red=silent)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "misuse_report.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 78)
    print(f"{'scenario':<42s}{'outcome':<10s}{'handled?':<9s}")
    print("-" * 78)
    for lab, kind, ok, detail, good in rows:
        print(f"{lab:<42s}{kind:<10s}{'yes' if ok else 'NO':<9s} {detail}")
    print("=" * 78)
    print(f"Figure: {outdir / 'misuse_report.png'}")
    return rows


if __name__ == "__main__":
    make_misuse_figure()
