"""Build the AGN reference npz files (AGN_PLAN.md section 4, A1-A5).

Run in the reference environment, NOT ceridwen311:

    SPS_HOME=$SPS_HOME \\
    PYTHONPATH=$HOME/Desktop/PhD/Tools/ceridwen_workspace/refs/prospector \\
    /Users/amanda/opt/anaconda3/envs/prospector/bin/python \\
        scripts/agn_reference/make_refs.py a1 a2 a3 a4 a5

- python-fsps 0.5.0 (libfsps 05b5e55; AGN/dust/IGM Fortran identical to FSPS
  master b6600ff), SPS_HOME data as recorded in each npz.
- prospect: fresh clone of bd-j/prospector master, imported via PYTHONPATH.

Every npz holds its inputs, outputs, wavelength grids, the L_bol used and a
``provenance`` JSON string. Outputs go to ``tests/reference/agn/``.
"""
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "tests", "reference", "agn")
sys.path.insert(0, HERE)

import fsps_port  # noqa: E402

REFS = os.path.expanduser("~/Desktop/PhD/Tools/ceridwen_workspace/refs")
CERIDWEN_SSP = os.environ.get(
    "CERIDWEN_TEST_SSP",
    os.path.expanduser("~/Desktop/PhD/Tools/ceridwen/tests/fixtures/ssp_data_bpass_agb_dust.h5"))


def _git(path, *args):
    try:
        return subprocess.check_output(["git", "-C", path, *args], text=True,
                                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e})"


def provenance(extra=None):
    import fsps
    import prospect
    import scipy
    p = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python_fsps": fsps.__version__,
        "python_fsps_file": fsps.__file__,
        "python_fsps_libfsps_pin": "05b5e550ddd7ffb1e71102b75cfbfdf057c211fd (python-fsps v0.5.0 submodule)",
        "fsps_master_read": _git(os.path.join(REFS, "fsps"), "rev-parse", "HEAD"),
        "SPS_HOME": os.environ.get("SPS_HOME"),
        "SPS_HOME_git": _git(os.environ.get("SPS_HOME", "."), "rev-parse", "HEAD"),
        "prospect_file": prospect.__file__,
        "prospect_git": _git(os.path.dirname(os.path.dirname(prospect.__file__)), "rev-parse", "HEAD"),
        "make_refs_git": _git(REPO, "rev-parse", "HEAD"),
        "make_refs_dirty": _git(REPO, "status", "--porcelain", "--", "scripts/agn_reference"),
    }
    if extra:
        p.update(extra)
    return json.dumps(p, indent=1, sort_keys=True)


def save(name, prov_extra=None, **arrays):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    np.savez_compressed(path, provenance=np.array(provenance(prov_extra)), **arrays)
    print(f"wrote {path}  {os.path.getsize(path) / 1e6:.2f} MB", flush=True)


def ceridwen_wave():
    import h5py
    with h5py.File(CERIDWEN_SSP, "r") as f:
        return np.asarray(f["ssp_wave"][:], dtype=np.float64)


# ---------------------------------------------------------------------------
# A1: FSPS torus via python-fsps
# ---------------------------------------------------------------------------
A1_FAGN = np.array([1e-3, 0.1, 1.0, 3.0])
A1_TAU = np.array([5.0, 7.5, 10.0, 40.0, 150.0])
A1_DUST2 = np.array([0.0, 0.5, 1.5])
A1_ZRED = np.array([0.0, 2.0, 7.0])


def _a1_block(sp, fagns, taus, dust2s, zreds, tage, rep_check=True):
    """spec(fagn) - spec(0) for the full product, plus FSPS-noise measurements."""
    nw = len(sp.wavelengths)
    shape = (len(zreds), len(dust2s), len(taus), len(fagns), nw)
    dspec = np.zeros(shape)
    spec0 = np.zeros(shape[:2] + (nw,))
    lbol = np.zeros(shape[:2])
    rep_max = 0.0
    for iz, z in enumerate(zreds):
        for idd, d2 in enumerate(dust2s):
            sp.params["zred"] = z
            sp.params["dust2"] = d2
            sp.params["fagn"] = 0.0
            _, s0 = sp.get_spectrum(tage=tage, peraa=False)
            spec0[iz, idd] = s0
            lbol[iz, idd] = sp.log_lbol
            for it, tau in enumerate(taus):
                sp.params["agn_tau"] = tau
                for jf, fa in enumerate(fagns):
                    sp.params["fagn"] = fa
                    _, s1 = sp.get_spectrum(tage=tage, peraa=False)
                    assert sp.log_lbol == lbol[iz, idd], "log_lbol changed with fagn"
                    dspec[iz, idd, it, jf] = s1 - s0
            if rep_check:
                # FSPS noise, measure 1: the same call twice
                _, s1b = sp.get_spectrum(tage=tage, peraa=False)
                rep_max = max(rep_max, float(np.max(np.abs(s1b - s1))))
                sp.params["fagn"] = 0.0
                _, s0b = sp.get_spectrum(tage=tage, peraa=False)
                rep_max = max(rep_max, float(np.max(np.abs(s0b - s0))))
    return dspec, spec0, lbol, rep_max


def _port_compare(wave, dspec, spec0, lbol, fagns, taus, dust2s, dust_type, dust_index):
    """Max relative difference FSPS vs the float64 port, where flux > 1e-6 peak,
    and the cancellation floor eps*|spec0|/|d| at the same pixels."""
    eps = np.finfo(np.float64).eps
    worst, worst_info, floor_at_worst, worst_excess = 0.0, None, 0.0, 0.0
    port = np.zeros_like(dspec)
    for iz in range(dspec.shape[0]):
        for idd, d2 in enumerate(dust2s):
            for it, tau in enumerate(taus):
                for jf, fa in enumerate(fagns):
                    p = fsps_port.agn_torus_fsps(wave, lbol[iz, idd], fa, tau, d2,
                                                 dust_type, dust_index)
                    port[iz, idd, it, jf] = p
                    d = dspec[iz, idd, it, jf]
                    m = p > 1e-6 * p.max()
                    rel = np.abs(d[m] - p[m]) / p[m]
                    floor = 4 * eps * (np.abs(spec0[iz, idd][m]) + p[m]) / p[m]
                    k = int(np.argmax(rel))
                    if rel[k] > worst:
                        worst = float(rel[k])
                        floor_at_worst = float(floor[k])
                        worst_info = dict(iz=iz, idust2=idd, itau=it, ifagn=jf,
                                          wave=float(wave[m][k]))
                    worst_excess = max(worst_excess, float(np.max(rel - floor)))
    return port, worst, worst_info, floor_at_worst, worst_excess


def make_a1():
    import fsps
    t0 = time.time()
    tage = 1.0
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, logzsol=0.0, dust_type=0,
                                dust_index=-0.7, dust1=0.0, add_igm_absorption=True,
                                imf_type=2)
    wave = np.asarray(sp.wavelengths, dtype=np.float64)
    dspec, spec0, lbol, rep_max = _a1_block(sp, A1_FAGN, A1_TAU, A1_DUST2, A1_ZRED, tage)
    print(f"A1 main grid {time.time() - t0:.0f}s, repeat max |diff| = {rep_max}", flush=True)
    port, worst, info, floor_w, excess = _port_compare(wave, dspec, spec0, lbol, A1_FAGN,
                                                       A1_TAU, A1_DUST2, 0, -0.7)
    print(f"A1 FSPS vs float64 port: max rel {worst:.3e} at {info}, "
          f"cancellation floor there {floor_w:.3e}, max(rel - floor) {excess:.3e}", flush=True)

    # IGM transmission FSPS applied to the stellar spectrum (torus is added after it)
    igm_trans = spec0[:, 0] / spec0[0, 0][None]

    # resampled torus nodes recovered from FSPS itself: z=0, dust2=0, fagn=1
    jf1 = int(np.where(A1_FAGN == 1.0)[0][0])
    tau_nodes, tnodes_port = fsps_port.resample_torus(wave)
    tnodes_fsps = np.zeros_like(tnodes_port)
    sp.params["zred"], sp.params["dust2"] = 0.0, 0.0
    sp.params["fagn"] = 0.0
    _, s0 = sp.get_spectrum(tage=tage, peraa=False)
    lb = sp.log_lbol
    for i, tau in enumerate(tau_nodes):
        sp.params["agn_tau"], sp.params["fagn"] = tau, 1.0
        _, s1 = sp.get_spectrum(tage=tage, peraa=False)
        tnodes_fsps[:, i] = (s1 - s0) / 10 ** lb
        sp.params["fagn"] = 0.0
    lam_n, _, spec_n = fsps_port.read_nenkova()

    # extra slices: other dust types, and a CSP (tau model) for the L_bol plumbing
    ext = {}
    for dt in (2, 4):
        sp.params["dust_type"] = dt
        d, s0x, lbx, _ = _a1_block(sp, np.array([0.1]), np.array([10.0, 40.0]),
                                   np.array([0.5, 1.5]), np.array([0.0]), tage, rep_check=False)
        _, w, inf, fl, ex = _port_compare(wave, d, s0x, lbx, [0.1], [10.0, 40.0], [0.5, 1.5], dt, -0.7)
        print(f"A1 dust_type={dt}: port max rel {w:.3e} floor {fl:.3e} excess {ex:.3e}", flush=True)
        ext[f"dt{dt}_dspec"], ext[f"dt{dt}_spec0"], ext[f"dt{dt}_log_lbol"] = d, s0x, lbx
        ext[f"dt{dt}_port_maxrel"] = np.array(w)
    sp.params["dust_type"] = 0
    sp.params["sfh"], sp.params["tau"], sp.params["const"] = 1, 1.0, 0.0
    d, s0x, lbx, _ = _a1_block(sp, np.array([0.1, 1.0]), np.array([10.0, 40.0]),
                               np.array([0.0, 0.5]), np.array([0.0, 2.0]), 3.0, rep_check=False)
    _, w, inf, fl, ex = _port_compare(wave, d, s0x, lbx, [0.1, 1.0], [10.0, 40.0], [0.0, 0.5], 0, -0.7)
    print(f"A1 CSP (sfh=1, tau=1 Gyr, tage=3): port max rel {w:.3e} floor {fl:.3e}, "
          f"log_lbol={lbx[0, 0]:.6f}", flush=True)
    ext.update(csp_dspec=d, csp_spec0=s0x, csp_log_lbol=lbx, csp_port_maxrel=np.array(w),
               csp_fagn=np.array([0.1, 1.0]), csp_agn_tau=np.array([10.0, 40.0]),
               csp_dust2=np.array([0.0, 0.5]), csp_zred=np.array([0.0, 2.0]))

    # recommended tolerance: relative, on pixels with |d| > 1e-6 * peak(d),
    # rel <= max(A1_RTOL_FLOOR, 8 * eps * (|spec0| + |d|) / |d|)
    m_all = dspec > 1e-6 * dspec.max(axis=-1, keepdims=True)
    floor_all = 8 * np.finfo(np.float64).eps * (np.abs(spec0[:, :, None, None]) + dspec) / dspec
    floor_max = float(floor_all[m_all].max())
    print(f"A1 cancellation floor 8*eps*(|spec0|+|d|)/|d| over masked pixels: max {floor_max:.3e}",
          flush=True)
    rtol = 1e-10 if excess < 1e-12 else 10 * excess
    save("A1_torus_fsps.npz",
         prov_extra={"component": "A1 torus, FSPS path, spec(fagn)-spec(0) from python-fsps",
                     "sp_params": "zcontinuous=1 sfh=0 tage=1 logzsol=0 imf_type=2 dust_type=0 "
                                  "dust_index=-0.7 dust1=0 add_igm_absorption=True "
                                  "add_dust_emission=True(default) add_neb_emission=False(default)",
                     "libraries": [x.decode() for x in sp.libraries]},
         wave=wave, fagn=A1_FAGN, agn_tau=A1_TAU, dust2=A1_DUST2, zred=A1_ZRED,
         tage=np.array(tage), dust_type=np.array(0), dust_index=np.array(-0.7),
         dspec=dspec, spec0=spec0, log_lbol=lbol,
         igm_trans=igm_trans, torus_tau_nodes=tau_nodes,
         torus_nodes_fsps=tnodes_fsps, torus_nodes_port=tnodes_port,
         nenkova_lam=lam_n, nenkova_spec=spec_n,
         noise_repeat_maxabs=np.array(rep_max), noise_port_maxrel=np.array(worst),
         noise_port_floor_at_worst=np.array(floor_w), noise_port_excess=np.array(excess),
         recommended_rtol=np.array(rtol), cancellation_floor_max=np.array(floor_max),
         recommended_rule=np.array("mask: |d| > 1e-6*max(d) per spectrum; "
                                   "rel = |ceridwen - dspec| / |dspec| <= recommended_rtol "
                                   "+ 8*eps*(|spec0| + |dspec|)/|dspec| (cancellation in the "
                                   "FSPS difference; eps = float64 eps)"),
         **ext)


# ---------------------------------------------------------------------------
# A2: prospect fake_fsps.agn_torus (T')
# ---------------------------------------------------------------------------
A2_TAU = np.array([1.0, 3.0, 5.0, 7.5, 10.0, 25.0, 40.0, 50.0, 100.0, 150.0, 200.0, 300.0])


def make_a2():
    import fsps
    from prospect.sources import fake_fsps
    sp = fsps.StellarPopulation(zcontinuous=1)
    grids = {"fsps": np.asarray(sp.wavelengths, dtype=np.float64), "ceridwen": ceridwen_wave()}
    arrays = {}
    for g, w in grids.items():
        arrays[f"wave_{g}"] = w
        arrays[f"torus_{g}"] = np.stack([fake_fsps.agn_torus(w, t) for t in A2_TAU])
        # the default when agn_tau is absent from params: agnssp_basis.get_spectrum
        # uses params.get('agn_tau', 50)
        assert np.array_equal(arrays[f"torus_{g}"][list(A2_TAU).index(50.0)],
                              fake_fsps.agn_torus(w, 50))
    arrays["nenkova_lam"] = np.asarray(fake_fsps.agndust_lam, dtype=np.float64)
    arrays["nenkova_spec"] = np.asarray(fake_fsps.agndust_specinit, dtype=np.float64)
    arrays["nenkova_tau"] = np.asarray(fake_fsps.agndust_tau, dtype=np.float64)
    save("A2_torus_prospector.npz",
         prov_extra={"component": "A2 torus T', prospect.sources.fake_fsps.agn_torus(wave, agn_tau)",
                     "note": "rows = agn_tau; unscaled (Prospector multiplies by "
                             "10**ssp.log_lbol * fagn, then add_igm); tau outside [5,150] "
                             "is linearly extrapolated by scipy interp1d",
                     "ceridwen_wave_source": CERIDWEN_SSP},
         agn_tau=A2_TAU, default_agn_tau=np.array(50.0), **arrays)


# ---------------------------------------------------------------------------
# A3: disc D + screen S (+ T' with IGM) through agnssp_basis.get_spectrum
# ---------------------------------------------------------------------------
def emline_wavelengths():
    info = np.genfromtxt(os.path.join(os.environ["SPS_HOME"], "data", "emlines_info.dat"),
                         dtype=[("wave", "f8"), ("name", "<U20")], delimiter=",")
    return info


class StubSSP:
    """The attributes agnssp_basis.get_spectrum reads from ``ssp``
    (agnssp_basis.py:80-82, 70, 146): fixed spectra, no SPS."""

    def __init__(self, wave, young, old, tspec, log_lbol):
        self.wavelengths = wave
        self._young, self._old, self._tspec = young, old, tspec
        self.log_lbol = log_lbol
        self.emline_wavelengths = emline_wavelengths()["wave"]

    @property
    def _csp_young_old(self):
        return self._young.copy(), self._old.copy()

    def get_spectrum(self, tage=0, peraa=False):
        assert not peraa
        return self.wavelengths, self._tspec.copy()


def galaxy_young_old():
    """Fixed young/old dust-free galaxy spectra from python-fsps: constant SFR
    over the last 1 Gyr, 1 Msun formed (tabular sfh=3), no dust, no IGM."""
    import fsps
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=3, logzsol=0.0, dust2=0.0,
                                add_igm_absorption=False, imf_type=2)
    t = np.linspace(0.0, 1.0, 101)
    sp.set_tabular_sfh(t, np.ones_like(t))  # 1 Msun/yr
    wave, tspec = sp.get_spectrum(tage=1.0, peraa=False)
    young, old = sp._csp_young_old
    return (np.asarray(wave, np.float64), np.asarray(young, np.float64),
            np.asarray(old, np.float64), np.asarray(tspec, np.float64), float(sp.log_lbol))


def _loglog(x, xp, fp):
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(np.maximum(fp, 1e-300))))


def a3_configs():
    cfg = []
    for dt in (0, 2, 4, 6):
        for d2 in (0.0, 1.0):
            for d4t in (0, 1):
                for d4 in (0.0, 0.5):
                    for d4i in (-1.8, -0.8):
                        for z in (0.0, 6.0):
                            cfg.append((1.0, dt, -0.7, d2, d4t, d4, d4i, z))
    for fb in (1e-3, 30.0):
        for d2 in (0.0, 1.0):
            for z in (0.0, 6.0):
                cfg.append((fb, 0, -0.7, d2, 1, 0.5, -0.8, z))
    for dt in (0, 4):
        for z in (0.0, 6.0):
            cfg.append((1.0, dt, -0.3, 1.0, 1, 0.5, -1.8, z))
    return np.array(cfg)


A3_COLS = ("fagn_bbb", "dust_type", "dust_index", "dust2", "dust4_type", "dust4",
           "dust4_index", "zred")


def make_a3():
    from prospect.sources import agnssp_basis
    wave_f, young_f, old_f, tspec_f, log_lbol = galaxy_young_old()
    cfg_all = a3_configs()
    # the ceridwen grid is 15000 points: keep a subset there (size budget)
    # every A3 operation is pointwise in wavelength except the 5500 A median
    # scale, so a few configs suffice there; the full product is on the fsps grid
    sub = np.array([i for i, c in enumerate(cfg_all)
                    if (c[0] == 1.0 and c[2] == -0.7 and tuple(c[3:7]) == (1.0, 1, 0.5, -0.8))
                    or (c[0] != 1.0 and c[3] == 1.0 and c[7] == 6.0)])
    out = {"log_lbol": np.array(log_lbol), "columns": np.array(A3_COLS)}
    for g, w in (("fsps", wave_f), ("ceridwen", ceridwen_wave())):
        if g == "fsps":
            young, old, tspec, cfgs = young_f, old_f, tspec_f, cfg_all
        else:
            young, old, tspec = (_loglog(w, wave_f, a) for a in (young_f, old_f, tspec_f))
            cfgs = cfg_all[sub]
        ssp = StubSSP(w, young, old, tspec, log_lbol)
        aspec = np.zeros((len(cfgs), len(w)))
        agn_nodust = np.zeros_like(aspec)
        for i, c in enumerate(cfgs):
            p = dict(zip(A3_COLS, c))
            p["dust_type"], p["dust4_type"] = int(p["dust_type"]), int(p["dust4_type"])
            p.update(add_agn_bbb_cont=True, add_igm_absorption=True, fagn=0.0)
            blob = agnssp_basis.get_spectrum(ssp, p, tage=1.0)
            aspec[i] = blob["aspec"]
            agn_nodust[i] = blob["agn_nodust"]
        # the scale factor of agnssp_basis.py:96, for the record
        tmpl = agnssp_basis.temple_template(ssp)
        win = (w >= 5480) & (w <= 5520)
        out.update({f"wave_{g}": w, f"young_{g}": young, f"old_{g}": old, f"tspec_{g}": tspec,
                    f"configs_{g}": cfgs, f"aspec_{g}": aspec, f"agn_nodust_{g}": agn_nodust,
                    f"temple_on_grid_{g}": tmpl,
                    f"scale_{g}": np.array(np.median(tmpl[win] / (young + old)[win])),
                    f"npix_5500_window_{g}": np.array(int(win.sum()))})
        # torus T' through the same function: scaling 10**log_lbol*fagn and IGM
        tcfg, tspecs = [], []
        for tau in (20.0, None, 200.0):
            for z in (0.0, 6.0):
                p = dict(dust_type=0, dust_index=-0.7, dust2=1.0, zred=z, fagn=0.3,
                         add_agn_bbb_cont=False, add_igm_absorption=True)
                if tau is not None:
                    p["agn_tau"] = tau
                tspecs.append(agnssp_basis.get_spectrum(ssp, p, tage=1.0)["aspec_torus"])
                tcfg.append((0.3, np.nan if tau is None else tau, z))
        out[f"torus_configs_{g}"] = np.array(tcfg)
        out[f"aspec_torus_{g}"] = np.array(tspecs)
    w = np.loadtxt(os.path.join(os.path.dirname(agnssp_basis.__file__), "agn_data", "temple.txt"),
                   unpack=True)
    out["temple_wave"], out["temple_fnu"] = w
    save("A3_disk_screen_prospector.npz",
         prov_extra={"component": "A3 disc D + dust4 screen S via prospect "
                                  "agnssp_basis.get_spectrum(StubSSP, params, tage=1)",
                     "configs_columns": list(A3_COLS),
                     "torus_configs_columns": ["fagn", "agn_tau (nan = key absent -> 50)", "zred"],
                     "note": "aspec = disc after add_dust(dust2[,dust4]) and add_igm (clipped at "
                             "1e-70 by add_igm); galaxy young/old = python-fsps constant SFR "
                             "1 Gyr, no dust; ceridwen-grid young/old/tspec are log-log "
                             "interpolations of the fsps-grid ones; ceridwen grid holds a "
                             "config subset (size budget)",
                     "ceridwen_wave_source": CERIDWEN_SSP},
         **out)


# ---------------------------------------------------------------------------
# A4: AGN narrow lines via prospect AGNSpecModel
# ---------------------------------------------------------------------------
A4_FILTERS = ["jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w", "jwst_f356w",
              "jwst_f444w", "jwst_f770w", "jwst_f1800w"]


class StubLineSPS:
    """sps for SpecModel.predict_init (sedmodel.py:177-215): zero continuum and
    zero star-forming line luminosities, so every prediction is AGN lines only."""

    def __init__(self, wave, ewave):
        self.wave, self.ewave = wave, ewave
        self.spectral_resolution = np.zeros_like(wave)

    def get_galaxy_spectrum(self, **params):
        return self.wave, np.zeros_like(self.wave), 1.0

    def get_galaxy_elines(self):
        return self.ewave, np.zeros_like(self.ewave)


def make_a4():
    import fsps
    from prospect.models import sedmodel
    from prospect.models.templates import TemplateLibrary
    from prospect.observation import Lines, Photometry, Spectrum

    wave = np.asarray(fsps.StellarPopulation(zcontinuous=1).wavelengths, np.float64)
    info = emline_wavelengths()
    ewave = np.asarray(info["wave"], np.float64)
    zred, agn_elum = 2.0, 1e-3
    mp = TemplateLibrary["parametric_sfh"]
    mp.update(TemplateLibrary["nebular"])
    mp.update(TemplateLibrary["agn_eline"])
    mp["zred"]["init"] = zred
    mp["mass"]["init"] = 1e10
    mp["nebemlineinspec"]["init"] = False
    mp["agn_elum"]["init"] = agn_elum
    mp["eline_sigma"] = dict(N=1, isfree=False, init=100.0)
    mp["sigma_smooth"] = dict(N=1, isfree=False, init=100.0)

    # observed-frame spectrum covering rest 3400-7000 A at z=2, 2 A pixels, no LSF
    owave = np.arange(3400.0 * (1 + zred), 7000.0 * (1 + zred), 2.0)
    spec_obs = Spectrum(wavelength=owave, flux=np.zeros_like(owave),
                        uncertainty=np.ones_like(owave), resolution=None)
    phot_obs = Photometry(filters=A4_FILTERS, flux=np.zeros(len(A4_FILTERS)),
                          uncertainty=np.ones(len(A4_FILTERS)))
    sps = StubLineSPS(wave, ewave)

    out = {"line_wave": ewave, "line_names": np.array(info["name"]), "obs_wave": owave,
           "filters": np.array(A4_FILTERS), "zred": np.array(zred), "mass": np.array(1e10),
           "agn_elum": np.array(agn_elum), "eline_sigma": np.array(100.0),
           "model_wave": wave}
    sigmas = np.array([50.0, 300.0])
    specs, phots, lines, alspecs, emasks, gsig = [], [], [], [], [], []
    for sig in sigmas:
        mp["agn_eline_sigma"]["init"] = sig
        model = sedmodel.AGNSpecModel(mp)
        ainds = np.nonzero(model._aline_lum)[0]
        line_obs = Lines(line_ind=ainds, wavelength=ewave[ainds] * (1 + zred), flux=np.zeros(len(ainds)),
                         uncertainty=np.ones(len(ainds)))
        # prospect master defect: Lines has no `resolution` attribute but
        # SpecModel.cache_eline_parameters reads obs.resolution (sedmodel.py:575)
        line_obs.resolution = None
        for o in (spec_obs, phot_obs, line_obs):
            o.rectify()
        # one observation per predict() so the cached spectrum state is the spectrum's
        (sp_pred,), _ = model.predict(model.theta, observations=[spec_obs], sps=sps)
        specs.append(sp_pred)
        em = np.zeros(len(owave), bool)
        em[model._fix_eline_pixelmask] = True
        emasks.append(em)
        full = np.zeros(len(owave))
        full[em] = model._agn_eline_spec
        alspecs.append(full)
        gsig.append(np.array(model._eline_sigma_kms, dtype=np.float64))
        (ph_pred,), _ = model.predict(model.theta, observations=[phot_obs], sps=sps)
        (li_pred,), _ = model.predict(model.theta, observations=[line_obs], sps=sps)
        phots.append(ph_pred)
        lines.append(li_pred)
        out.update(aline_lum_ratio=model._aline_lum.copy(), aline_inds=ainds,
                   flux_norm=np.array(model.flux_norm()), line_norm=np.array(model.line_norm))
    # Hbeta check: the 'a42' table is normalised to index 59
    assert abs(ewave[59] - 4863) < 2
    out.update(agn_eline_sigma=sigmas, spec_pred=np.array(specs), phot_pred=np.array(phots),
               line_pred=np.array(lines), aline_spec=np.array(alspecs), eline_pixmask=np.array(emasks),
               eline_sigma_kms_restored=np.array(gsig),
               line_lum_lsun_per_mformed=out["aline_lum_ratio"][None] * agn_elum)
    save("A4_lines_prospector.npz",
         prov_extra={"component": "A4 AGN lines via prospect AGNSpecModel.predict with a stub "
                                  "sps (zero continuum, zero SF lines)",
                     "units": "spec_pred, aline_spec: maggies on obs_wave; phot_pred: maggies; "
                              "line_pred: erg/s/cm^2 for aline_inds; "
                              "line_lum_lsun_per_mformed = agn_elum * ratio (Hbeta=idx 59 = 1)",
                     "notes": ["AGN line sigma = agn_eline_sigma alone (no library/instrument "
                               "term): predict_aline_spec overwrites _eline_sigma_kms "
                               "(sedmodel.py AGNSpecModel.predict_aline_spec)",
                               "line Gaussians normalised by -trapezoid over the masked "
                               "pixels only (get_eline_gaussians)",
                               "painted only on pixels within nsigma=5*agn_eline_sigma/eline_sigma "
                               "SF-line widths of lines that are _fix_eline & _valid_eline",
                               "photometry: all 23 AGN lines, nebline_photometry, needs "
                               "add_neb_emission=True and nebemlineinspec=False"],
                     "cosmology": "prospect.sources.constants.cosmo (flux_norm stored)"},
         **out)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["a1", "a2", "a3", "a4", "a5"]
    for item in todo:
        t = time.time()
        globals()[f"make_{item}"]()
        print(f"{item} done in {time.time() - t:.0f}s", flush=True)
