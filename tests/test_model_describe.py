"""``SedModel.latex()`` and ``SedModel.display()`` (``ceridwen.model.describe``): the model's
structure is read from the model, a transform's inputs from its traced program."""
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import ceridwen  # noqa: F401  (sets the precision)
from ceridwen.model.describe import (model_structure, param_info, prior_curve, prior_latex,
                                     transform_inputs)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _gridfixture import find_test_grid  # noqa: E402

GRID = find_test_grid()
needs_grid = pytest.mark.skipif(GRID is None, reason="no test SSP grid")
FILTERS = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]


def test_transform_inputs_are_traced_not_guessed():
    theta = {"a": jnp.ones(3), "b": jnp.array([2.0]), "c": jnp.array([0.5])}
    assert transform_inputs(lambda th: th["a"] * 2.0, theta) == ("a",)
    assert transform_inputs(lambda th: th["b"] + jnp.sum(th["c"]), theta) == ("b", "c")
    assert transform_inputs(lambda th: jnp.array([1.0]), theta) == ()          # fixed
    # a branch on a traced value cannot be traced: inputs unknown, not "all of them"
    assert transform_inputs(lambda th: th["a"] if float(th["b"][0]) > 0 else th["c"], theta) is None


def test_prior_curve_spans_the_bounds():
    from ceridwen.priors import ClippedNormal, StudentT, Uniform
    x, y = prior_curve(Uniform(low=-2.0, high=0.2))
    assert x.min() == -2.0 and x.max() == 0.2 and np.allclose(y, 1.0)
    x, y = prior_curve(ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0))
    assert x.min() == 0.0 and x.max() == 4.0 and y.argmax() < y.size // 4
    x, y = prior_curve(StudentT(mean=0.0, scale=1.0, df=2.0))
    assert abs(x.min() + x.max()) < 1e-6 and y.max() == 1.0


def test_prior_latex():
    from ceridwen.priors import ClippedNormal, Uniform
    assert prior_latex(Uniform(low=9.0, high=12.0)) == r"\mathcal{U}(9,\,12)"
    assert prior_latex(ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0)) == \
        r"\mathcal{N}(0.3,\,1^2)\ \text{on}\ [0,\,4]"
    assert prior_latex(None) == r"\text{flat (improper)}"


@pytest.fixture(scope="module")
def model():
    from ceridwen import CSPBasis, Instrument, Kinematics, SedModel, SSPData
    from ceridwen.cosmology import Cosmology
    from ceridwen.model import logsfr_ratios_to_sfh
    from ceridwen.observation import Photometry, Spectrum
    from ceridwen.priors import ClippedNormal, Normal, StudentT, Uniform

    ssp = SSPData.load(str(GRID))
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, 10.0, 5), zh_const=True,
                   sfh_interp="step", add_dust=True, add_diffuse_dust=True, add_neb=False,
                   verbose=False, cosmo=Cosmology.planck18())
    t_yr = np.array(csp.sfh_times)
    rng = np.random.default_rng(3)
    wave = np.linspace(4000.0, 8000.0, 200)
    phot = 1e-9 * (1.0 + 0.1 * rng.standard_normal(len(FILTERS)))
    spec = 1e-29 * (1.0 + 0.1 * rng.standard_normal(wave.size))
    obs = [Photometry(filters=FILTERS, flux=phot, uncertainty=0.05 * phot, name="phot"),
           Spectrum(wavelength=wave, flux=spec, uncertainty=0.05 * spec,
                    instrument=Instrument.sigma_kms(100.0), name="spec")]
    priors = {"logzsol": Uniform(low=-2.0, high=0.2), "logmass": Uniform(low=9.0, high=12.0),
              "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
              "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
              "logsfr_ratios": StudentT(df=2.0, mean=0.0, scale=0.3),
              "sigma_gal": Uniform(low=50.0, high=400.0),
              "log_jitter_spec": Normal(mean=-1.0, sigma=1.0)}
    fixed = {k: (lambda th, _v=jnp.atleast_1d(jnp.asarray(v)): _v)
             for k, v in csp.theta_init.items() if k in csp.dust_param_names}
    return SedModel(csp, obs, priors=priors,
                    transforms={"sfh": lambda th, _t=t_yr: logsfr_ratios_to_sfh(
                        th["logsfr_ratios"], sfh_times_yr=_t), **fixed},
                    free_param_init={"logsfr_ratios": jnp.zeros(4), "logmass": jnp.array([10.0]),
                                     "sigma_gal": jnp.array([200.0]),
                                     "log_jitter_spec": jnp.array([-1.0])},
                    zred=0.1, kinematics=Kinematics(sigma_gal="sigma_gal"))


@needs_grid
def test_structure_reads_the_model(model):
    s = model_structure(model)
    kinds = {p.name: (p.kind, p.stage) for p in s.params}
    assert kinds["sfh"] == ("derived", "stars")
    assert {p.name: p.inputs for p in s.params if p.kind == "derived"} == {"sfh": ("logsfr_ratios",)}
    for k in model.csp.dust_param_names:                       # constant transforms
        assert kinds[k] == ("fixed", "dust")
    assert kinds["diffuse_tau_kc"] == ("free", "dust")
    assert kinds["logmass"] == ("free", "flux")
    assert kinds["sigma_gal"] == ("free", "kin")
    assert kinds["log_jitter_spec"] == ("free", "obs:spec")
    # the stages in the order the CSP applies them (csp.py, predict)
    assert [st.key for st in s.stages] == ["stars", "dust", "kin", "flux"]


@needs_grid
def test_param_info_names_every_key(model):
    for name in model.param_names + list(model.transforms):
        sym, desc, _ = param_info(model, name)
        assert desc, f"{name} has no description"
        assert not sym.startswith(r"\texttt"), f"{name} has no symbol"
    assert "diffuse" in param_info(model, "diffuse_tau_kc")[1]
    assert "spec" in param_info(model, "log_jitter_spec")[1]


@needs_grid
def test_latex_lists_every_parameter(model):
    tex = model.latex()
    assert tex.startswith(r"\begin{table*}") and tex.rstrip().endswith(r"\end{table*}")
    for name in model.param_names + list(model.transforms):
        assert r"\texttt{" + name.replace("_", r"\_") + "}" in tex, name
    for part in ("Free parameters", "Derived parameters", "Fixed parameters", "Model settings",
                 r"\mathcal{U}(9,\,12)", r"\texttt{logsfr\_ratios} (4)"):
        assert part in tex, part
    assert model.latex(table=False).startswith(r"\begin{tabular}")
    assert "Model settings" not in model.latex(settings=False)
    # balanced braces and $ signs: LaTeX that compiles starts with that
    assert tex.count("{") == tex.count("}") and tex.count("$") % 2 == 0


@needs_grid
def test_display_draws(model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = model.display(return_fig=True, predictive=False)
    assert len(fig.axes) == 1 and ax.texts
    plt.close(fig)
    fig, ax = model.display(return_fig=True, n_draws=8)       # with the prior predictive
    assert len(fig.axes) == 2 and fig.axes[1].get_xscale() == "log"
    plt.close(fig)
    fig, own = plt.subplots()
    out = model.display(ax=own, return_fig=True)                # into a given axes: no panel
    assert out[1] is own and len(fig.axes) == 1
    plt.close(fig)
