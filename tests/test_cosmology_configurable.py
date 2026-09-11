"""The cosmology is set ONCE, on the CSP, and is never silent.

Cosmology reaches the forward model in two arithmetic places:

  1. the luminosity-distance flux factor applied to the spectrum, photometry
     and line predictions (``flux_factor_cgs``);
  2. the age of the universe used to rescale the SFH lookback grid when
     ``zred`` is sampled (``age_gyr`` via ``_lookback_from_zred``).

Both read ``self.cosmo`` on the CSP, which requires it at construction
(``Cosmology.planck18()``, ``wmap9()``, ``flat(H0, Om0)``, ``from_name``,
``from_astropy``); ``SedModel.cosmo`` is a read-only view onto the CSP's,
``SedModel(cosmo=...)`` is accepted only when equal, and the cosmology is
printed by ``CSPBasis.__repr__`` / ``SedModel.summary`` and written to the
HDF5 result.  ``tuniv`` is gone: ``csp.age_at(z)`` replaces it and
``SedModel`` refuses an SFH grid older than the Universe at the fixed
redshift.  ``lumdist_mpc`` gives nearby objects physical units.
"""
import ast
import inspect
import pathlib
import warnings

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
    available_cosmologies,
    flux_factor_cgs,
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


# --------------------------------------------------------------- presets ---
def test_presets_names_and_round_trips():
    assert available_cosmologies() == ("Planck18", "Planck15", "WMAP9")
    for name in available_cosmologies():
        c = Cosmology.from_name(name)
        assert c.name == name
        assert Cosmology.from_name(name.lower()) == c
        assert Cosmology.from_dict(c.to_dict()) == c
        assert Cosmology.from_dict(c.to_dict()).name == name
        assert name in str(c) and "H0" in str(c)
    assert Cosmology.planck18() == Cosmology()          # name takes no part in equality
    assert Cosmology.planck18().is_planck18 and not Cosmology.wmap9().is_planck18
    assert Cosmology.flat(70.0, 0.3).H0 == 70.0 and Cosmology.flat(70.0, 0.3).name is None
    with pytest.raises(KeyError):
        Cosmology.from_name("Planck99")
    with pytest.raises(ValueError):
        Cosmology.flat(-70.0, 0.3)
    with pytest.raises(ValueError):
        Cosmology.flat(70.0, 1.2)


def test_from_dict_reads_hdf5_style_attrs():
    attrs = {f"cosmo_{k}": v for k, v in Cosmology.wmap9().to_dict().items()}
    attrs["zred"] = 0.5                                     # other attrs are ignored
    c = Cosmology.from_dict(attrs)
    assert c == Cosmology.wmap9() and c.name == "WMAP9"
    unnamed = Cosmology.from_dict(Cosmology.flat(70.0, 0.3).to_dict())
    assert unnamed.name is None


def test_presets_match_astropy():
    ap = pytest.importorskip("astropy.cosmology")
    for name, ours in (("Planck18", Cosmology.planck18()),
                       ("Planck15", Cosmology.planck15()),
                       ("WMAP9", Cosmology.wmap9())):
        theirs = getattr(ap, name)
        assert ours.H0 == pytest.approx(theirs.H0.value)
        assert ours.Om0 == pytest.approx(theirs.Om0)
        assert ours.Tcmb0 == pytest.approx(theirs.Tcmb0.value)
        assert ours.Neff == pytest.approx(theirs.Neff)
        assert ours.m_nu_ev_sum == pytest.approx(float(np.sum(theirs.m_nu.to("eV").value)))
        assert Cosmology.from_astropy(theirs) == ours


def test_methods_match_helpers():
    c = Cosmology.wmap9()
    assert float(c.age(1.0)) == float(age_gyr(1.0, c))
    assert float(c.luminosity_distance(1.0)) == float(luminosity_distance_mpc(1.0, c))


def test_flux_factor_with_explicit_lumdist():
    """lumdist replaces D_L(z); (1+z) still comes from z; no 10 pc pin at z = 0."""
    ff = float(flux_factor_cgs(0.0, ALT, lumdist_mpc=3.5))
    expected = (10.0 / 3.5e6) ** 2 * 3.1967965e-7
    assert _relative_difference(ff, expected) < 1e-12
    z = 0.01
    ff_z = float(flux_factor_cgs(z, ALT, lumdist_mpc=3.5))
    assert _relative_difference(ff_z, expected * (1.0 + z)) < 1e-12
    # against the 10 pc convention at z = 0 the ratio is (10 pc / 3.5 Mpc)^2
    assert ff / float(flux_factor_cgs(0.0, ALT)) == pytest.approx((10.0 / 3.5e6) ** 2, rel=1e-12)


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
        flux_factor_cgs(z, ALT), flux_factor_cgs(z)) > 1e-3


def test_flux_factor_at_z_zero_is_cosmology_independent():
    """The CSP normalisation is 'source at 10 pc'; z <= 0 has no dimming."""
    assert _relative_difference(
        flux_factor_cgs(0.0, ALT), flux_factor_cgs(0.0)) < 1e-12


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
    """Structural guard: every call to ``flux_factor_cgs``/``age_gyr``
    inside the forward model must pass a cosmology explicitly.

    This is the bug the whole change exists to prevent: a call site that
    silently falls back to Planck 2018 while the user believes their own
    cosmology is in force.
    """
    root = pathlib.Path(ceridwen.__file__).parent
    targets = {"flux_factor_cgs", "flux_factor_maggies", "age_gyr"}
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
def _build_csp_kw(cosmo=DEFAULT_COSMO, track=True, oldest=None, **extra):
    ssp = SSPData.load(str(require_test_grid()))
    lb = np.linspace(0.0, float(age_gyr(2.0)) if oldest is None else oldest, _NB)
    theta = {"lookback_time": jnp.asarray(lb),
             "sfh": jnp.ones(_NB), "Z": jnp.array([-2.0])}
    return CSPBasis(ssp, theta=theta, zh_const=True,
                    sfh_interp="step", add_dust=False, add_diffuse_dust=False,
                    add_neb=False, add_igm=False, track_zred_age=track,
                    verbose=False, cosmo=cosmo, **extra)


def _build_csp(cosmo=DEFAULT_COSMO, track=True, oldest=None):
    return _build_csp_kw(cosmo=cosmo, track=track, oldest=oldest)


def test_csp_requires_and_stores_the_cosmology():
    assert _build_csp().cosmo.is_planck18
    assert _build_csp(cosmo=ALT).cosmo == ALT
    with pytest.raises(TypeError, match="explicit cosmology"):
        _build_csp(cosmo=None)


def test_csp_repr_and_age_at():
    csp = _build_csp(cosmo=Cosmology.wmap9())
    assert "WMAP9" in repr(csp) and "tuniv" not in repr(csp)
    assert csp.age_at(2.0) == pytest.approx(float(age_gyr(2.0, Cosmology.wmap9())))
    assert isinstance(csp.age_at(2.0), float)


def test_csp_age_grid_uses_the_supplied_cosmology():
    """The SFH lookback grid must be built from age_gyr under ``cosmo``."""
    z = 7.0
    default_grid = np.asarray(_build_csp()._lookback_from_zred(jnp.array([z])))
    alt_grid = np.asarray(_build_csp(cosmo=ALT)._lookback_from_zred(jnp.array([z])))
    assert default_grid[-1] / 1e9 == pytest.approx(float(age_gyr(z)), abs=1e-6)
    assert alt_grid[-1] / 1e9 == pytest.approx(float(age_gyr(z, ALT)), abs=1e-6)
    assert abs(default_grid[-1] - alt_grid[-1]) > 0.0


def test_sedmodel_refuses_a_grid_older_than_the_universe():
    csp = _build_csp(oldest=13.8, track=False)         # fine at z = 0 (0.1 % above 13.787)
    SedModel(csp, [], priors={}, zred=0.0)
    with pytest.raises(ValueError, match="predates the Universe"):
        SedModel(csp, [], priors={}, zred=2.0)
    SedModel(_build_csp(track=False), [], priors={}, zred=2.0)   # grid built from age_gyr(2.0)
    # track_zred_age=True: the CSP rescales the grid to age(zred) itself
    # (SedModel injects the fixed zred), so there is nothing to refuse.
    SedModel(_build_csp(oldest=13.8, track=True), [], priors={}, zred=2.0)


def test_lumdist_is_injected_and_applied():
    csp = _build_csp()
    model = SedModel(csp, [], priors={}, zred=0.0, lumdist_mpc=3.5)
    theta = dict(model.theta_init)
    assert "zred" not in theta and "lumdist_mpc" not in theta   # not sampled
    plain = SedModel(csp, [], priors={}, zred=0.0)
    assert plain.lumdist_mpc is None and plain._zred_fixed is None   # zred = 0: nothing injected
    assert model._zred_fixed is not None and float(model._lumdist_fixed[0]) == 3.5
    ff = float(flux_factor_cgs(0.0, csp.cosmo, lumdist_mpc=3.5))
    t = dict(theta, zred=jnp.array([0.0]), lumdist_mpc=jnp.array([3.5]))
    assert _relative_difference(float(csp._flux_factor(t)), ff) < 1e-12
    assert "lumdist_mpc" in csp._known_theta_keys
    with pytest.raises(ValueError, match="lumdist_mpc"):
        SedModel(csp, [], priors={}, lumdist_mpc=-1.0)


# ----------------------------------------------------------- SedModel ------
class _StubCSP:
    """Minimal duck-typed CSP: what SedModel.__init__ and summary() touch."""
    def __init__(self, cosmo):
        self.cosmo = cosmo
        self.theta_init = {"logmass": jnp.array([10.0])}
        self.param_names = ["logmass"]
        self.wave = jnp.linspace(1000.0, 10000.0, 16)

    def get_spectrum(self, theta):            # summary() prints its __name__
        return jnp.zeros_like(self.wave)


def test_sedmodel_reports_the_csp_cosmology():
    assert SedModel(_StubCSP(ALT), [], priors={}).cosmo == ALT


def test_sedmodel_cosmo_is_a_live_view_not_a_copy():
    """No second copy that could drift out of step with the CSP (duck-typed
    stub; the real CSP refuses reassignment, see test_csp_cosmo_is_read_only)."""
    csp = _StubCSP(DEFAULT_COSMO)
    model = SedModel(csp, [], priors={})
    assert model.cosmo.is_planck18
    csp.cosmo = ALT                       # change the single source of truth
    assert model.cosmo == ALT             # the view follows immediately


def test_csp_cosmo_is_read_only():
    csp = _build_csp()
    with pytest.raises(AttributeError, match="fixed at construction"):
        csp.cosmo = ALT
    assert csp.cosmo.is_planck18


def test_csp_refuses_unknown_kwargs():
    with pytest.raises(TypeError, match="tuniv"):
        _build_csp_kw(tuniv=13.8)
    with pytest.raises(TypeError, match="cosmolgy"):
        _build_csp_kw(cosmolgy=ALT)


def test_h0_range_catches_swapped_arguments():
    with pytest.raises(ValueError, match="argument order"):
        Cosmology.flat(0.3, 0.7)


def test_from_dict_accepts_bytes_names():
    d = Cosmology.wmap9().to_dict(); d["name"] = b"WMAP9"
    assert Cosmology.from_dict(d).name == "WMAP9"
    with pytest.raises(KeyError, match="missing"):
        Cosmology.from_dict({"H0": 70.0})


def test_zred_zero_with_observations_warns_and_lumdist_at_positive_zred_warns():
    from ceridwen.observation import Photometry
    phot = Photometry(filters=["sdss_g0"], flux=[1e-9], uncertainty=[1e-10])
    csp = _build_csp(track=False)
    with pytest.warns(UserWarning, match="NO flux factor"):
        SedModel(csp, [phot], priors={})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="no prior for sampled")
        SedModel(csp, [phot], priors={}, zred=0.5)             # no warning
        SedModel(csp, [phot], priors={}, lumdist_mpc=3.5)      # no warning
    with pytest.warns(UserWarning, match="replaces D_L"):
        SedModel(csp, [phot], priors={}, zred=0.5, lumdist_mpc=3000.0)


def test_sedmodel_cosmo_is_read_only():
    model = SedModel(_StubCSP(DEFAULT_COSMO), [], priors={})
    with pytest.raises(AttributeError):
        model.cosmo = ALT


def test_sedmodel_accepts_equal_and_refuses_different_cosmology():
    SedModel(_StubCSP(ALT), [], priors={}, cosmo=Cosmology.flat(70.0, 0.30))   # equal
    with pytest.raises(ValueError, match="differs from the CSP"):
        SedModel(_StubCSP(DEFAULT_COSMO), [], priors={}, cosmo=ALT)


def test_sedmodel_refuses_a_csp_without_cosmology():
    csp = _StubCSP(DEFAULT_COSMO)
    del csp.cosmo
    with pytest.raises(TypeError, match="no .cosmo"):
        SedModel(csp, [], priors={})


def test_sedmodel_summary_prints_the_cosmology_and_redshift():
    text = SedModel(_StubCSP(Cosmology.wmap9()), [], priors={}, zred=0.5).summary()
    assert "WMAP9" in text and "zred = 0.5" in text and "D_L" in text
    text0 = SedModel(_StubCSP(Cosmology.wmap9()), [], priors={}).summary()
    assert "no flux factor" in text0
    textd = SedModel(_StubCSP(Cosmology.wmap9()), [], priors={}, lumdist_mpc=3.5).summary()
    assert "3.5 Mpc" in textd


def test_hdf5_attrs_round_trip(tmp_path):
    h5py = pytest.importorskip("h5py")
    from ceridwen.fit import result_cosmology
    path = tmp_path / "r.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("model")
        for k, v in Cosmology.planck15().to_dict().items():
            g.attrs[f"cosmo_{k}"] = v
        g.attrs["zred"] = 0.5
    c = result_cosmology(path)
    assert c == Cosmology.planck15() and c.name == "Planck15"
