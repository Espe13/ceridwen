"""A copied ``CSPBasis`` built with ``agn=`` must evaluate the COPY's model.

Regression test for ``explorer/reports/AGN_COPY_BUG.md``: ``CSPBasis._with_agn`` used to return a
closure over the kernel bound to the ORIGINAL instance, and a shallow copy shares that function
object.  The shared-template recipe (``sfmsfit.model.Shared.csp_for``: ``copy.copy``, re-bind
every bound method stored on the instance to the copy, ``initialize_model_structure`` with new
age bins) re-binds bound methods only -- it cannot reach into a closure -- so the copy returned
the template's stars and the template's AGN.  The wrapper is now a bound method that resolves
``self._agn_base`` and ``self._agn_rest`` at call time, so the recipe works.

Each test builds a template on one lookback grid, a copy re-initialised to a second grid, and a
fresh CSP built on the second grid, and demands copy == fresh byte for byte (and copy != template,
so the test can fail).
"""
from __future__ import annotations

import copy
import types

import numpy as np
import pytest

import jax.numpy as jnp

from ceridwen.agn import AGNConfig

MODES = ("fsps", "prospector", "corrected")

#: the template's and the copy's lookback nodes [Gyr] -- same length, different bins
NODES_A = jnp.linspace(0.0, 0.5, 5)
NODES_B = jnp.asarray([0.0, 0.03, 0.2, 0.4, 0.8])


def _csp_for(tpl, nodes):
    """The shared-template recipe of ``sfmsfit.model.Shared.csp_for``."""
    csp = copy.copy(tpl)
    for k, v in list(vars(csp).items()):
        if isinstance(v, types.MethodType) and v.__self__ is tpl:
            setattr(csp, k, types.MethodType(v.__func__, csp))
    theta = {"lookback_time": nodes}
    theta.update(dict(tpl.theta_init))
    csp.initialize_model_structure(theta)
    return csp


def _leftovers(csp, tpl):
    """Attributes of ``csp`` that still reach ``tpl`` -- bound methods AND closures (the guard
    ``csp_for`` is missing, point 3 of the report)."""
    bad = []
    for k, v in vars(csp).items():
        if isinstance(v, types.MethodType) and v.__self__ is tpl:
            bad.append(k)
        fn = v.__func__ if isinstance(v, types.MethodType) else v
        if isinstance(fn, types.FunctionType):
            for cell in (fn.__closure__ or ()):
                try:
                    held = cell.cell_contents
                except ValueError:          # empty cell
                    continue
                if held is tpl or (isinstance(held, types.MethodType) and held.__self__ is tpl):
                    bad.append(k)
    return sorted(set(bad))


def _theta(csp, fagn):
    th = dict(csp.theta_init)
    th["sfh"] = jnp.linspace(3.0, 0.1, np.shape(th["sfh"])[0])
    th["fagn"] = jnp.asarray([fagn])
    return th


def _trio(build_csp, mode, **kw):
    cfg = dict(agn_mode=mode, disk=False, lines=False)
    tpl = build_csp(AGNConfig(**cfg), lookback_time=NODES_A, add_igm=True, **kw)
    return tpl, _csp_for(tpl, NODES_B), build_csp(AGNConfig(**cfg), lookback_time=NODES_B,
                                                  add_igm=True, **kw)


@pytest.mark.parametrize("fagn", [0.0, 0.3])
@pytest.mark.parametrize("mode", MODES)
def test_copy_rebound_equals_fresh(build_csp, mode, fagn):
    """copy == fresh byte for byte, at fagn = 0 and fagn > 0; and copy != template."""
    tpl, cp, fresh = _trio(build_csp, mode)
    assert _leftovers(cp, tpl) == []
    assert cp.get_spectrum.__self__ is cp            # a bound method of the copy, not a closure
    th = _theta(fresh, fagn)
    a = np.asarray(cp.get_spectrum(theta=dict(th)))
    b = np.asarray(fresh.get_spectrum(theta=dict(th)))
    t = np.asarray(tpl.get_spectrum(theta=dict(th)))
    assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert t.tobytes() != b.tobytes()                # the two grids really differ
    # the AGN term and the wrapped kernel both follow the copy
    assert np.asarray(cp._agn_rest(dict(th))).tobytes() \
        == np.asarray(fresh._agn_rest(dict(th))).tobytes()
    assert np.asarray(cp._agn_base(dict(th))).tobytes() \
        == np.asarray(fresh._agn_base(dict(th))).tobytes()


@pytest.mark.parametrize("mode", MODES)
def test_copy_rebound_keeps_per_pixel_and_support(build_csp, mode):
    """The copy keeps the ``_per_pixel`` mark and the ``support=`` signature (wavelength_support)."""
    tpl, cp, fresh = _trio(build_csp, mode, wavelength_support=True, add_neb=False)
    assert getattr(fresh.get_spectrum, "_per_pixel", False)
    assert getattr(cp.get_spectrum, "_per_pixel", False)
    th = _theta(fresh, 0.3)
    sup = slice(100, 400)
    a = np.asarray(cp.get_spectrum(theta=dict(th), support=sup))
    b = np.asarray(fresh.get_spectrum(theta=dict(th), support=sup))
    assert a.tobytes() == b.tobytes()
    # support= reaches the wrapped kernel: only those pixels, each the kernel's own support
    # output plus the sliced AGN term. (Not whole_grid[sup]: the support path differs from the
    # whole grid in the last float32 bit with or without agn=, GOTCHAS section 24.)
    kern = getattr(cp, cp._agn_base_name)
    assert kern.__self__ is cp
    k = kern(dict(th), include_lines=None, support=sup)
    # in jnp, as the wrapper does: XLA's f64 -> f32 cast flushes subnormals, NumPy's does not
    exp = np.asarray(k + cp._agn_rest(dict(th))[..., sup].astype(k.dtype))
    assert a.shape == (sup.stop - sup.start,)
    assert a.tobytes() == exp.tobytes()


@pytest.fixture(scope="module")
def afe_ssp(ssp_data):
    """The two-plane [alpha/Fe] grid of ``test_hook.py`` (BPASS test grid, planes x1 and x0.9)."""
    from types import SimpleNamespace
    s = SimpleNamespace(**{k: getattr(ssp_data, k) for k in vars(ssp_data)})
    f = np.asarray(ssp_data.ssp_flux, dtype=np.float64)
    s.ssp_flux = np.stack([f, 0.9 * f])
    s.ssp_afe = np.array([0.0, 0.4])
    if s.ssp_stellar_mass is not None:
        m = np.asarray(s.ssp_stellar_mass)
        s.ssp_stellar_mass = np.stack([m, m])
    return s


@pytest.mark.parametrize("dem", [False, True], ids=["dattn_nodem", "dattn_dem"])
@pytest.mark.parametrize("mode", MODES)
def test_copy_rebound_afe(afe_ssp, torus_path, mode, dem):
    """``CSPBasis_afe`` overrides the kernels under the same names: the copy's wrapper resolves
    the SUBCLASS kernel on the copy and equals a fresh a/Fe CSP at a sampled afe (the per-pixel
    wrapper without dust emission, the whole-grid one with it)."""
    from ceridwen import Cosmology
    from ceridwen.csp import CSPBasis_afe

    def build(nodes):
        return CSPBasis_afe(afe_ssp, lookback_time=nodes, zh_const=True, verbose=False,
                            cosmo=Cosmology.planck18(), add_igm=True, add_dust_emission=dem,
                            agn=AGNConfig(agn_mode=mode, disk=False, lines=False))
    tpl = build(NODES_A)
    cp, fresh = _csp_for(tpl, NODES_B), build(NODES_B)
    assert _leftovers(cp, tpl) == []
    kern = getattr(cp, cp._agn_base_name)
    assert kern.__self__ is cp
    assert kern.__func__ is getattr(CSPBasis_afe, cp._agn_base_name)
    assert getattr(cp.get_spectrum, "_per_pixel", False) is (not dem)
    th = {**_theta(fresh, 0.3), "afe": jnp.asarray([0.2])}
    a = np.asarray(cp.get_spectrum(theta=dict(th)))
    b = np.asarray(fresh.get_spectrum(theta=dict(th)))
    t = np.asarray(tpl.get_spectrum(theta=dict(th)))
    assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert t.tobytes() != b.tobytes()


@pytest.mark.parametrize("mode", MODES)
def test_copy_rebound_keeps_kernel_name_provenance(build_csp, mode):
    """``fit.py`` records ``<kernel>_agn``; the copy carries the same name."""
    from ceridwen.csp.csp import spectrum_model_name
    tpl, cp, fresh = _trio(build_csp, mode)
    # the kernel build_csp selects (add_dust default, add_neb=False), as 90be87f named it
    assert spectrum_model_name(fresh) == "get_spectrum_dattn_nodem_noneb_agn"
    assert spectrum_model_name(cp) == spectrum_model_name(fresh)
    assert cp._agn_base_name == fresh._agn_base_name
