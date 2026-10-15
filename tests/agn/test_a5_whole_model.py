"""A5 (AGN_PLAN §4): the whole model, CERIDWEN ``agn_mode="prospector"`` against python-fsps +
prospect ``AGNSSPBasis`` / ``AGNPolySpecModel`` (``tests/reference/agn/A5_whole_model_prospector.npz``,
session REF: continuity SFH, z = 3, dust2 0.5 power law, IGM on, 15 NIRCam + MIRI bands; five
configs off / torus / disk / disk_dust4 / all).

The plan's target, "AGN-only difference at rel <= 1e-4", cannot hold literally: the AGN-only
difference depends on the galaxy through two scalars that differ between two SPS codes by
design (the torus is ``fagn * L_bol``, the disc is normalised to the no-dust young + old galaxy
at 5500 A), and CERIDWEN's vendored filter curves are not sedpy 1.0.0's. So A5 is tested as the
product of its parts, each part as tightly as it can be:

1. Spectrum, torus: CERIDWEN's prospector-mode torus through the hook (``fake_fsps.agn_torus``
   port, scaled by L_bol, Prospector's IGM) with Prospector's own L_bol fed in equals
   ``agnspec_torus`` to 1e-12.
2. Spectrum, disc (+ dust4 screen): equal to ``agnspec`` up to ONE scalar, the same for every
   config (Prospector's galaxy L_nu at 5500 A), to 1e-12 in shape.
3. Photometry: CERIDWEN's filters on Prospector's own spectra reproduce ``phot_torus`` /
   ``phot_disk`` / ``phot_gal`` to <= 2e-3 (filter-curve versions; the galaxy shows the same
   offset, so it is not the AGN).
4. The full CSP (``CSPBasis(agn=AGNConfig(agn_mode="prospector"))`` on the test grid, the same
   SFH, dust, z): the AGN-only photometry ``predict(config) - predict(off)`` equals Prospector's
   ``phot_torus + phot_disk`` once the two galaxy scalars (CERIDWEN / Prospector L_bol and
   5500 A luminosity, both measured) and the flux-normalisation ratio are applied, to 2e-3
   per band (the filter term of 3).
5. Prospector counts the torus twice (python-fsps adds its own torus inside the
   galaxy spectrum, ``agnspec_torus`` adds T' on top). The prospector mode reproduces T' alone
   (Prospector's evident intent); this test records the size of the double count.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import jax
import ceridwen  # noqa: F401  (sets the precision: float64, or float32 on an Apple GPU)
import jax.numpy as jnp  # noqa: E402

from ceridwen.agn import AGNConfig, AGNContext  # noqa: E402

NAME = "A5_whole_model_prospector.npz"
TINY = np.finfo(np.float64).tiny


@pytest.fixture(scope="module")
def ref(agn_ref, torus_path):
    return dict(agn_ref(NAME))


def _vals(cfg):
    name, fagn, tau, fbbb, _d4t, d4, d4i = cfg
    return name, {"fagn": float(fagn), "agn_tau": float(tau), "fagn_bbb": float(fbbb),
                  "dust4": float(d4), "dust4_index": float(d4i)}


@pytest.fixture(scope="module")
def chain(ref):
    """CERIDWEN's prospector-mode AGN on the FSPS grid, observed frame (maggies), with
    Prospector's L_bol and a unit galaxy for the disc normalisation."""
    from ceridwen.agn.hook import AGNHook
    from ceridwen.agn.screen import ProspectorDiffuseCurve
    wave = ref["model_wave"]
    z = float(ref["zred"])
    fn = float(ref["flux_norm"][0])
    mtot = float(ref["mass"].sum())
    hook = AGNHook(AGNConfig(agn_mode="prospector", lines=False, dust4_type=1), wave,
                   add_dust_emission=False)
    t_igm = np.asarray(hook.agn_transmission(z, 1.0))
    dc = ProspectorDiffuseCurve(wave, dust_type=0).transmission({"dust2": 0.5,
                                                                  "dust_index": -0.7})
    out = []
    for i, cfg in enumerate(ref["configs"]):
        name, v = _vals(cfg)
        th = hook.filled({k: jnp.array([x]) for k, x in v.items()})
        ctx = AGNContext(jnp.asarray(wave), jnp.asarray(10 ** ref["log_lbol"][i] / mtot),
                         jnp.ones(wave.shape), dc, "prospector")
        torus = np.asarray(hook.model.components["torus"].rest_spectrum(th, ctx))
        disk = (np.asarray(hook.model.components["disk"].rest_spectrum(th, ctx))
                * np.asarray(hook.model.screen.transmission(th, ctx)))
        out.append((name, v, torus * t_igm * fn, disk * t_igm * fn))
    return out


def test_A5_torus_spectrum(ref, chain):
    worst = 0.0
    for i, (name, v, torus, _disk) in enumerate(chain):
        want = ref["agnspec_torus"][i]
        if v["fagn"] == 0.0:
            assert np.all(torus == 0.0)
            continue
        m = want > 1e-6 * want.max()
        rel = np.abs(torus[m] / want[m] - 1.0)
        worst = max(worst, rel.max())
    assert worst <= 1e-12, worst
    print(f"A5 torus spectrum (Prospector L_bol fed in): max rel {worst:.3g}")


def test_A5_disc_spectrum_up_to_the_galaxy_scalar(ref, chain):
    scalars = []
    worst = 0.0
    for i, (name, v, _torus, disk) in enumerate(chain):
        want = ref["agnspec"][i]
        if v["fagn_bbb"] == 0.0:
            assert np.all(disk == 0.0)
            continue
        m = want > 1e-6 * want.max()
        r = want[m] / disk[m]
        s = np.median(r)
        scalars.append(s)
        worst = max(worst, np.abs(r / s - 1.0).max())
    assert worst <= 1e-12, worst
    assert np.ptp(scalars) <= 1e-12 * np.mean(scalars), scalars
    print(f"A5 disc spectrum: shape max rel {worst:.3g}; Prospector galaxy L_nu(5500 A) per "
          f"M_formed = {scalars[0]:.6e} L_sun/Hz (the one scalar)")


def _phot(ref, wave):
    from ceridwen.observation import Photometry
    ph = Photometry(filters=list(ref["filters"]), flux=np.ones(len(ref["filters"])),
                    uncertainty=np.ones(len(ref["filters"])), name="a5")
    ph.setup_for_model(jnp.asarray(wave), zred=float(ref["zred"]))
    return ph


def test_A5_photometry_of_prospector_spectra(ref):
    """CERIDWEN's filters vs sedpy 1.0.0's on identical spectra: the galaxy and the AGN show the
    same few-1e-4 offsets, so the term is the filter data, not the AGN."""
    wave = ref["model_wave"]
    ph = _phot(ref, wave)
    worst = {}
    for i in range(len(ref["configs"])):
        for spec, pk in (("agnspec_torus", "phot_torus"), ("agnspec", "phot_disk"),
                         ("galspec", "phot_gal")):
            want = ref[pk][i]
            if want.max() <= 1e-60:
                continue
            got = np.asarray(ph.predict(jnp.asarray(ref[spec][i] * 3.631e-20), jnp.asarray(wave)))
            worst[pk] = max(worst.get(pk, 0.0), np.abs(got / want - 1.0).max())
    assert max(worst.values()) <= 2e-3, worst
    assert worst["phot_gal"] >= 3e-4      # the offset is there for the galaxy too
    print("A5 photometry, CERIDWEN filters on Prospector spectra: max rel "
          + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))


def test_A5_full_csp_prospector_mode(ref, chain, ssp_data, torus_path):
    from ceridwen import CSPBasis, Cosmology
    from ceridwen.agn.disk import TempleDisk
    ab, mass = ref["agebins"], ref["mass"]
    mtot = float(mass.sum())
    edges_yr = np.concatenate([[0.0], 10 ** ab[1:, 0], [10 ** ab[-1, 1]]])
    sfr = mass / np.diff(edges_yr) / mtot                     # unit formed mass per bin
    z = float(ref["zred"])
    csp = CSPBasis(ssp_data, theta={"lookback_time": jnp.asarray(edges_yr / 1e9),
                                    "sfh": jnp.asarray(sfr), "logzsol": jnp.array([-0.5])},
                   zh_const=True, sfh_interp="step", add_neb=False, add_dust=False,
                   add_diffuse_dust=True, diffuse_law="powerlaw", add_igm=True, verbose=False,
                   cosmo=Cosmology.wmap9(),
                   agn=AGNConfig(agn_mode="prospector", lines=False, dust4_type=1))
    th = dict(csp.theta_init)
    th.update({"logmass": jnp.array([np.log10(mtot)]), "zred": jnp.array([z]),
               "diffuse_tau_pow": jnp.array([0.5]), "diffuse_alpha_pow": jnp.array([-0.7]),
               "logzsol": jnp.array([-0.5])})                # Prospector's theta (dust2, logzsol)
    ph = _phot(ref, csp.wave)

    def phot(v):
        t = {**th, **{k: jnp.array([x]) for k, x in v.items()}}
        return np.asarray(csp.predict(t, [ph])["a5"], np.float64)

    # CERIDWEN / Prospector: L_bol (torus), the disc normalisation, the flux normalisation
    W = csp.calculate_ssp_weights(theta=th)
    ctx = csp._agn_context(th, W)
    r_lbol = float(ctx.L_bol_star) / (10 ** ref["log_lbol"][0] / mtot)
    # disc = template * fagn_bbb / median(template / galaxy) over 5480-5520 A. The chain used a
    # unit galaxy on the FSPS grid; Prospector's disc is s times it (test 2): so Prospector's
    # median(template / galaxy) = median(template) / s on that grid, CERIDWEN's is its own.
    i_disk = [i for i, c in enumerate(ref["configs"]) if c[0] == "disk"][0]
    want = ref["agnspec"][i_disk]
    m = want > 1e-6 * want.max()
    s = np.median(want[m] / chain[i_disk][3][m])
    unit = TempleDisk(ref["model_wave"], mode="prospector")
    scale_unit = float(unit.norm_scale(jnp.ones(ref["model_wave"].shape))[0])
    scale_c = float(csp._agn.model.components["disk"].norm_scale(
        csp._dustfree_stellar(th, W).astype(jnp.float64))[0])
    r_disc = (scale_unit / s) / scale_c
    r_ff = float(csp._flux_factor(th)) * mtot / 3.631e-20 / float(ref["flux_norm"][0])
    off = phot({"fagn": 0.0, "fagn_bbb": 0.0})
    worst = 0.0
    for i, cfg in enumerate(ref["configs"]):
        name, v = _vals(cfg)
        if name == "off":
            continue
        got = phot(v) - off
        want = (r_lbol * ref["phot_torus"][i] + r_disc * ref["phot_disk"][i]) * r_ff
        worst = max(worst, np.abs(got / want - 1.0).max())
    assert worst <= 2e-3, worst
    print(f"A5 full CSP (prospector mode, test grid): AGN-only photometry = Prospector's x the "
          f"galaxy scalars to max rel {worst:.2e}; scalars CERIDWEN/Prospector: L_bol "
          f"{r_lbol:.4f}, disc normalisation {r_disc:.4f}, flux normalisation {r_ff:.5f}")


def test_A5_prospector_double_counts_the_torus(ref):
    """Recorded, not reproduced: with fagn > 0 python-fsps puts its own torus in the galaxy
    spectrum (``fsps_torus_in_galspec``, REF's port to 1.7e-11) and AGNSSPBasis adds T' on top;
    ``phot[torus] - phot[off]`` is 1.39-1.90 x ``phot_torus`` in real Prospector."""
    i_off = [i for i, c in enumerate(ref["configs"]) if c[0] == "off"][0]
    for i, c in enumerate(ref["configs"]):
        if float(c[1]) == 0.0:
            continue
        d_gal = ref["galspec"][i] - ref["galspec"][i_off]
        ft = ref["fsps_torus_in_galspec"][i]
        m = ft > 1e-6 * ft.max()
        assert np.abs(d_gal[m] / ft[m] - 1.0).max() <= 1e-10
        dphot = ref["phot"][i] - ref["phot"][i_off]
        ratio = dphot / (ref["phot_torus"][i] + ref["phot_disk"][i])
        if float(c[3]) == 0.0:                 # torus only
            assert ratio.min() > 1.3 and ratio.max() < 2.0, ratio
