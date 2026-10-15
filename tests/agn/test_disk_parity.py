"""A3 (AGN_PLAN §4): disc D + screen S vs Prospector ``agnssp_basis.get_spectrum``.

Reference: ``tests/reference/agn/A3_disk_screen_prospector.npz`` written by session REF
(``scripts/agn_reference/make_refs.py::make_a3``) from prospect 91f8e13 with a stub ``ssp``
(fixed python-fsps young/old spectra), on the FSPS grid (all 140 configs) and on the
CERIDWEN grid (10 configs). ``$CERIDWEN_AGN_REF_DIR`` overrides the directory.

What is compared, at rel <= 1e-12 on every pixel whose reference is a normal float64:

* ``temple_on_grid`` and ``scale``: the template on the grid and the 5500 A median.
* ``agn_nodust``: ``TempleDisk(mode="prospector").rest_spectrum`` (D alone).
* ``aspec``: D attenuated by ``AGNScreen`` with ``ctx.diffuse_curve`` from
  ``ProspectorDiffuseCurve`` (S), then Prospector's IGM. The IGM is not this module's:
  ``_np_add_igm`` below is a line-by-line numpy transcription of ``fake_fsps.add_igm``
  (Madau 1995 as Prospector writes it, clipped at 1e-70), used only so that the z = 6
  references can be compared. Whether the CSP's prospector mode applies that IGM or
  CERIDWEN's ``Madau1995`` to the AGN is the hook's decision.

Reference pixels below the float64 normal range (Prospector computes subnormals, XLA on
CPU flushes them to zero) must come out < 2.3e-308 here.
"""
import os
import pathlib

import numpy as np
import pytest

import jax.numpy as jnp

from ceridwen.agn import AGNContext
from ceridwen.agn.disk import TempleDisk, load_temple21
from ceridwen.agn.screen import AGNScreen, ProspectorDiffuseCurve

REF_NAME = "A3_disk_screen_prospector.npz"
RTOL = 1e-12
TINY = np.finfo(np.float64).tiny


def _ref_path():
    d = os.environ.get("CERIDWEN_AGN_REF_DIR")
    root = pathlib.Path(d) if d else pathlib.Path(__file__).resolve().parents[1] / "reference" / "agn"
    return root / REF_NAME


@pytest.fixture(scope="module")
def ref():
    p = _ref_path()
    if not p.exists():
        pytest.skip(f"A3 reference {p} not found (session REF writes it; set "
                    "$CERIDWEN_AGN_REF_DIR to its tests/reference/agn)")
    return dict(np.load(p, allow_pickle=False))


def _np_add_igm(wave, spec, zred, igm_factor=1.0):
    """prospect/sources/fake_fsps.py::add_igm (91f8e13), transcribed line by line."""
    redshifted_wave = np.asarray(wave) * (1 + zred)
    lylim = 911.75
    lyw = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150,
                    920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919,
                    914.576])
    lycoeff = np.array([0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960, 0.0006967, 0.0006236,
                        0.0005665, 0.0005200, 0.0004817, 0.0004487, 0.0004200, 0.0003947,
                        0.000372, 0.000352, 0.0003334, 0.00031644])
    tau_line = np.zeros_like(redshifted_wave)
    for i in range(len(lyw)):
        lmin, lmax = lyw[i], lyw[i] * (1.0 + zred)
        idx0 = np.where((redshifted_wave >= lmin) & (redshifted_wave <= lmax))[0]
        tau_line[idx0] += lycoeff[i] * np.exp(3.46 * np.log(redshifted_wave[idx0] / lyw[i]))
    xc = redshifted_wave / lylim
    xem = 1.0 + zred
    xc[xc < 1.0] = 1.0
    xc[xc > xem] = xem
    tau_cont = 0.25 * xc ** 3 * (np.exp(0.46 * np.log(xem)) - np.exp(0.46 * np.log(xc)))
    tau_cont = tau_cont + 9.4 * np.exp(1.5 * np.log(xc)) * (np.exp(0.18 * np.log(xem))
                                                            - np.exp(0.18 * np.log(xc)))
    tau_cont = tau_cont - 0.7 * xc ** 3 * (np.exp(-1.32 * np.log(xc)) - np.exp(-1.32 * np.log(xem)))
    tau_cont = tau_cont - 0.023 * (np.exp(1.68 * np.log(xem)) - np.exp(1.68 * np.log(xc)))
    res = spec * np.exp(-(tau_line + tau_cont) * igm_factor)
    return np.clip(res, a_min=10 ** (-70.0), a_max=None)


def _assert_rel(got, want, what):
    normal = np.abs(want) >= TINY
    rel = np.abs(got[normal] / want[normal] - 1.0)
    assert rel.max() <= RTOL, f"{what}: max rel {rel.max():.3g} at {np.argmax(rel)}"
    assert np.all(np.abs(got[~normal]) < TINY), what
    return rel.max()


def test_A3_template_file_is_the_reference_file(ref):
    w, f = load_temple21()
    assert np.array_equal(w, ref["temple_wave"]) and np.array_equal(f, ref["temple_fnu"])


@pytest.mark.parametrize("grid", ["fsps", "ceridwen"])
def test_A3_template_and_scale(ref, grid):
    wave = ref[f"wave_{grid}"]
    gal = ref[f"young_{grid}"] + ref[f"old_{grid}"]
    disk = TempleDisk(wave, mode="prospector")
    # np.interp of numpy 2.4 (ceridwen311) and of the reference env differ by 1 ulp on a
    # few pixels (measured: 3 of 1936 on the fsps grid, 32 of 15000 on the ceridwen grid)
    np.testing.assert_allclose(np.asarray(disk.template), ref[f"temple_on_grid_{grid}"],
                               rtol=2.3e-16, atol=0)
    assert len(disk.win_idx) == int(ref[f"npix_5500_window_{grid}"])
    scale, ok = disk.norm_scale(jnp.asarray(gal))
    assert bool(ok)
    assert float(scale) == pytest.approx(float(ref[f"scale_{grid}"]), rel=RTOL)


@pytest.mark.parametrize("grid", ["fsps", "ceridwen"])
def test_A3_disc_and_screen(ref, grid):
    wave = ref[f"wave_{grid}"]
    gal = jnp.asarray(ref[f"young_{grid}"] + ref[f"old_{grid}"])
    cols = list(ref["columns"])
    disk = TempleDisk(wave, mode="prospector")
    screens = {t: AGNScreen(wave, dust4_type=t) for t in (0, 1)}
    curves = {t: ProspectorDiffuseCurve(wave, dust_type=t) for t in (0, 2, 4, 6)}
    worst_nodust = worst = 0.0
    cfgs = ref[f"configs_{grid}"]
    for i, c in enumerate(cfgs):
        p = dict(zip(cols, c))
        theta = {k: float(p[k]) for k in ("fagn_bbb", "dust_index", "dust2", "dust4", "dust4_index")}
        curve = curves[int(p["dust_type"])]
        ctx = AGNContext(wave=jnp.asarray(wave), L_bol_star=jnp.asarray(1.0),
                         gal_nodust_young_old=gal, diffuse_curve=curve.transmission(theta),
                         mode="prospector")
        d = disk.rest_spectrum(theta, ctx)
        worst_nodust = max(worst_nodust, _assert_rel(np.asarray(d), ref[f"agn_nodust_{grid}"][i],
                                                     f"{grid} agn_nodust {c}"))
        # dust4_type 0: get_spectrum passes dust4 = 0 and the screen does not read it
        att = np.asarray(screens[int(p["dust4_type"])].attenuate(d, theta, ctx))
        got = _np_add_igm(wave, att, p["zred"])
        worst = max(worst, _assert_rel(got, ref[f"aspec_{grid}"][i], f"{grid} aspec {c}"))
    print(f"A3 {grid}: {len(cfgs)} configs, max rel agn_nodust {worst_nodust:.3g}, "
          f"aspec {worst:.3g}")
