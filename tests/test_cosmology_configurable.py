"""The cosmology must be user-settable, and must enter the pipeline EXACTLY
ONCE -- through the CSP object.

Cosmology reaches the forward model in two arithmetic places:

  1. the luminosity-distance flux factor applied to the spectrum, photometry
     and line predictions (``flux_factor_maggies``);
  2. the age of the universe used to rescale the SFH lookback grid when
     ``zred`` is sampled (``age_gyr`` via ``_lookback_from_zred``).

Both read ``self.cosmo`` on the CSP, and the CSP is the only object that
stores it: ``SedModel.cosmo`` is a read-only view that forwards to
``model.csp.cosmo``, so the two can never disagree.

Regression guard for the pre-v0.2.1 behaviour, in which ``Cosmology`` was
fully parametrised but no call site ever passed ``cosmo=``, so the parameters
were unreachable from the public API.  Monkey-patching ``DEFAULT_COSMO`` did
not help either, because Python binds default arguments at definition time.
"""
import ast
import inspect
import pathlib

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

import ceridwen
from ceridwen.cosmology import (
    DEFAULT_COSMO,
    Cosmology,
    age_gyr,
    flux_factor_maggies,
    luminosity_distance_mpc,
    resolve_cosmology,
)
from ceridwen.csp.csp import CSPBasis
from ceridwen.csp.csp_afe import CSPBasis_afe
from ceridwen.model.model import SedModel
from ceridwen.ssps.ssp_data import SSPData

from _gridfixture import require_test_grid

ALT = Cosmology(H0=70.0, Om0=0.30)
_NB = 8


# ------------------------------------------------------------------ unit ---
def test_default_is_planck18():
    assert DEFAULT_COSMO.is_planck18
    assert not ALT.is_planck18
    assert resolve_cosmology(None) is DEFAULT_COSMO
    assert resolve_cosmology(ALT) is ALT


def test_reference_values_match_astropy_planck18():
    # astropy.cosmology.Planck18: D_L(1) = 6791.5 Mpc, age(0) = 13.787 Gyr.
    assert float(luminosity_distance_mpc(1.0)) == pytest.approx(6791.5, rel=1e-3)
    assert float(age_gyr(0.0)) == pytest.approx(13.787, rel=1e-3)


def _relative_difference(a, b):
    """|a/b - 1|, for comparisons that must not depend on absolute scale.

    NB: ``pytest.approx(x, rel=...)`` also carries a default ``abs=1e-12``
    and passes if EITHER tolerance is met.  The flux factor is of order
    1e-24, so that absolute tolerance swallows every conceivable difference
    and ``!= pytest.approx(...)`` would be vacuously false.  Compare
    relative differences explicitly instead.
    """
    a, b = float(a), float(b)
    return abs(a / b - 1.0)


@pytest.mark.parametrize("z", [0.5, 3.0, 11.2])
def test_custom_cosmology_moves_distance_age_and_flux(z):
    """A different cosmology must move all three, not just the distance."""
    assert _relative_difference(
        luminosity_distance_mpc(z, ALT), luminosity_distance_mpc(z)) > 1e-3
    assert _relative_difference(age_gyr(z, ALT), age_gyr(z)) > 1e-3
    assert _relative_difference(
        flux_factor_maggies(z, ALT), flux_factor_maggies(z)) > 1e-3


def test_flux_factor_at_z_zero_is_cosmology_independent():
    """The CSP normalisation is 'source at 10 pc'; z <= 0 has no dimming."""
    assert _relative_difference(
        flux_factor_maggies(0.0, ALT), flux_factor_maggies(0.0)) < 1e-12


def test_default_cosmo_rebinding_is_honoured(monkeypatch):
    """``cosmo=None`` resolves at CALL time, not at definition time."""
    import ceridwen.cosmology as cosmo_mod
    baseline = float(luminosity_distance_mpc(3.0))
    monkeypatch.setattr(cosmo_mod, "DEFAULT_COSMO", ALT)
    assert _relative_difference(
        luminosity_distance_mpc(3.0), luminosity_distance_mpc(3.0, ALT)) < 1e-12
    assert _relative_difference(luminosity_distance_mpc(3.0), baseline) > 1e-3


def test_from_astropy_roundtrip_and_flatness_guard():
    pytest.importorskip("astropy")
    from astropy.cosmology import LambdaCDM, Planck18
    assert Cosmology.from_astropy(Planck18).is_planck18
    with pytest.raises(ValueError, match="flatness"):
        Cosmology.from_astropy(LambdaCDM(H0=70, Om0=0.3, Ode0=0.6))


@pytest.mark.parametrize("neff", [3.046, 3.0, 2.0, 0.0])
def test_to_astropy_neutrino_species_count(neff):
    """astropy validates len(m_nu) == floor(Neff); ceil() raised ValueError."""
    pytest.importorskip("astropy")
    cosmo = Cosmology(Neff=neff)
    ap = cosmo.to_astropy()              # must not raise
    assert float(ap.H0.value) == pytest.approx(cosmo.H0)
    assert float(ap.Neff) == pytest.approx(neff)


def test_to_astropy_round_trips_through_from_astropy():
    pytest.importorskip("astropy")
    for cosmo in (DEFAULT_COSMO, ALT, Cosmology(H0=72.0, Om0=0.26, Neff=3.0)):
        assert Cosmology.from_astropy(cosmo.to_astropy()) == cosmo


def test_astropy_backend_honours_custom_cosmology():
    """``backend='astropy'`` used to silently revert to Planck18."""
    pytest.importorskip("astropy")
    from astropy.cosmology import FlatLambdaCDM, Planck18

    import math

    import astropy.units as u

    z = 2.0
    ours = float(luminosity_distance_mpc(z, ALT, backend="astropy"))
    # astropy wants exactly floor(Neff) neutrino masses -- three for the
    # Planck18 Neff = 3.046, not four.
    masses = [0.0] * math.floor(ALT.Neff)
    masses[-1] = ALT.m_nu_ev_sum
    theirs = float(FlatLambdaCDM(
        H0=ALT.H0 * u.km / u.s / u.Mpc, Om0=ALT.Om0, Tcmb0=ALT.Tcmb0 * u.K,
        Neff=ALT.Neff, m_nu=u.Quantity(masses, u.eV),
    ).luminosity_distance(z).to("Mpc").value)
    assert _relative_difference(ours, theirs) < 1e-6
    assert _relative_difference(
        ours, float(Planck18.luminosity_distance(z).to("Mpc").value)) > 1e-4


def test_cosmology_is_exported_at_top_level():
    assert ceridwen.Cosmology is Cosmology
    assert ceridwen.DEFAULT_COSMO is DEFAULT_COSMO


# -------------------------------------------------- single entry point ------
def test_only_the_csp_classes_take_a_cosmology():
    """Every CSP flavour accepts ``cosmo=``; SedModel deliberately does not
    store one."""
    for cls in (CSPBasis, CSPBasis_afe):
        assert "cosmo" in inspect.signature(cls.__init__).parameters, cls.__name__


def test_no_cosmology_call_site_uses_the_default_implicitly():
    """Structural guard: every call to ``flux_factor_maggies``/``age_gyr``
    inside the forward model must pass a cosmology explicitly.

    This is the bug the whole change exists to prevent: a call site that
    silently falls back to Planck 2018 while the user believes their own
    cosmology is in force.
    """
    root = pathlib.Path(ceridwen.__file__).parent
    targets = {"flux_factor_maggies", "age_gyr"}
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "cosmology.py":       # the definitions themselves
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name not in targets:
                continue
            has_cosmo = len(node.args) >= 2 or any(
                kw.arg == "cosmo" for kw in node.keywords)
            if not has_cosmo:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {name}()")
    assert not offenders, (
        "cosmology call sites falling back to the package default:\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------------- CSP -------
def _build_csp(cosmo=None, track=True):
    ssp = SSPData.load(str(require_test_grid()))
    lb = np.linspace(0.0, float(age_gyr(2.0)), _NB)
    theta = {"lookback_time": jnp.asarray(lb),
             "sfh": jnp.ones(_NB), "Z": jnp.array([-2.0])}
    return CSPBasis(ssp, theta=theta, tuniv=13.8, zh_const=True,
                    sfh_interp="step", add_dust=False, add_diffuse_dust=False,
                    add_neb=False, add_igm=False, track_zred_age=track,
                    verbose=False, cosmo=cosmo)


def test_csp_stores_cosmology_and_defaults_to_planck18():
    assert _build_csp().cosmo.is_planck18
    assert _build_csp(cosmo=ALT).cosmo == ALT


def test_csp_age_grid_uses_the_supplied_cosmology():
    """The SFH lookback grid must be built from age_gyr under ``cosmo``."""
    z = 7.0
    default_grid = np.asarray(_build_csp()._lookback_from_zred(jnp.array([z])))
    alt_grid = np.asarray(_build_csp(cosmo=ALT)._lookback_from_zred(jnp.array([z])))
    assert default_grid[-1] / 1e9 == pytest.approx(float(age_gyr(z)), abs=1e-6)
    assert alt_grid[-1] / 1e9 == pytest.approx(float(age_gyr(z, ALT)), abs=1e-6)
    assert abs(default_grid[-1] - alt_grid[-1]) > 0.0


# ----------------------------------------------------------- SedModel ------
class _StubCSP:
    """Minimal duck-typed CSP: SedModel only needs these four attributes."""
    def __init__(self, cosmo):
        self.cosmo = cosmo
        self.theta_init = {"logmass": jnp.array([10.0])}
        self.param_names = ["logmass"]
        self.wave = jnp.linspace(1000.0, 10000.0, 16)


def test_sedmodel_reports_the_csp_cosmology():
    assert SedModel(_StubCSP(ALT), [], priors={}).cosmo == ALT


def test_sedmodel_cosmo_is_a_live_view_not_a_copy():
    """No second copy that could drift out of step with the CSP."""
    csp = _StubCSP(DEFAULT_COSMO)
    model = SedModel(csp, [], priors={})
    assert model.cosmo.is_planck18
    csp.cosmo = ALT                       # change the single source of truth
    assert model.cosmo == ALT             # the view follows immediately


def test_sedmodel_cosmo_is_read_only():
    model = SedModel(_StubCSP(DEFAULT_COSMO), [], priors={})
    with pytest.raises(AttributeError):
        model.cosmo = ALT


def test_sedmodel_refuses_a_cosmology_kwarg():
    """Passing it here would create a second, desyncable entry point."""
    with pytest.raises(TypeError, match="the CSP owns it"):
        SedModel(_StubCSP(DEFAULT_COSMO), [], priors={}, cosmo=ALT)


def test_sedmodel_falls_back_when_csp_predates_the_attribute():
    csp = _StubCSP(DEFAULT_COSMO)
    del csp.cosmo
    assert SedModel(csp, [], priors={}).cosmo is DEFAULT_COSMO
