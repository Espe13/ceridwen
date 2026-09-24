"""10**logmass is the formed stellar mass on every SFH path (B1-001), and the "linear" SFH
scheme's SSP weights equal a brute-force quadrature of SFR(t) x log-age tent functions (B1-016).

B1-001: with ``track_zred_age=True`` the lookback grid is rescaled to age(zred) inside the
forward model.  The rescaling moves the SFH in time and keeps the mass theta["sfh"] forms on the
construction grid, so under the unit-mass transform ``logsfr_ratios_to_sfh(..., sfh_times_yr=
csp.sfh_times)`` sum(W) = 1 and PostProcess ``mass_formed`` = 10**logmass at every zred, in both
schemes, per node and per bin, for CSPBasis and CSPBasis_afe.  The fixed-z path is unchanged.

B1-016: independent reference = numerical quadrature on a fine log-time grid of SFR(t) times the
tent function of each SSP node in log age.  ``_scheme_brute`` applies the scheme's one convention
(each SFH bin keeps its formed mass, spread over the part of the bin inside the SSP node range),
and must match to quadrature precision; ``_naive_brute`` puts SFR below the youngest node on that
node, and differs only there (the youngest-node effect)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _gridfixture import TEST_DATA_DIR, find_test_grid                    # noqa: E402

from ceridwen import SSPData, CSPBasis, SedModel, Cosmology               # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh               # noqa: E402
from ceridwen.postprocess import PostProcess                              # noqa: E402
from ceridwen.sampler.runner import SamplingResult                        # noqa: E402

COSMO = Cosmology.planck18()
N = 7
ZS = (0.0, 1.0, 3.0, 6.0)
RATIOS = jnp.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2])      # a non-flat SFH shape


def _bpass():
    path = find_test_grid()
    if path is None:
        pytest.skip("no test grid")
    return SSPData.load(str(path))


def _csp(ssp, interp, per_bin, zh_const, track=True, cls=CSPBasis, T=None):
    T = float(COSMO.age(0.0)) if T is None else T
    lb = np.linspace(0.0, T, N)
    kw = dict(lookback_time=lb, zh_const=zh_const, sfh_interp=interp, sfh_per_bin=per_bin,
              add_dust=False, add_diffuse_dust=False, track_zred_age=track, verbose=False,
              cosmo=COSMO)
    if cls is CSPBasis:
        kw.update(add_neb=False, add_igm=False)
    return cls(ssp, **kw)


def _unit_sfh(csp, per_bin):
    """Unit formed mass on the construction grid: the shipped transform per node; per bin, the
    same SFR values normalised by sum(sfr * dt)."""
    t = np.asarray(csp.sfh_times)
    if not per_bin:
        return logsfr_ratios_to_sfh(RATIOS, sfh_times_yr=t)
    sfr = 10.0 ** np.concatenate([[0.0], -np.cumsum(np.asarray(RATIOS)[:-1])])
    return jnp.asarray(sfr / np.sum(sfr * np.diff(t)))


def _theta(csp, sfh, z, zh_const):
    th = {"sfh": sfh, "zred": jnp.array([z])}
    if zh_const:
        th["logzsol"] = jnp.array([-0.3])
    else:
        th["logzsol_hist"] = jnp.linspace(-0.3, -1.0, N)
    return th


SCHEMES = [("step", False), ("step", True), ("linear", False), ("linear", True)]


@pytest.mark.parametrize("zh_const", [True, False])
@pytest.mark.parametrize("interp,per_bin", SCHEMES)
@pytest.mark.parametrize("z", ZS)
def test_tracked_grid_forms_unit_mass(interp, per_bin, zh_const, z):
    ssp = _bpass()
    csp = _csp(ssp, interp, per_bin, zh_const)
    W = csp.calculate_ssp_weights(_theta(csp, _unit_sfh(csp, per_bin), z, zh_const))
    assert float(jnp.sum(W)) == pytest.approx(1.0, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("interp,per_bin", SCHEMES)
def test_tracked_weights_are_rescaled_grid_weights_times_inverse_stretch(interp, per_bin):
    """Independent statement of the fix: stretching the grid by r = age(z)/T_grid multiplies
    every bin's formed mass by r, so the tracked weights equal the explicit-grid weights (a
    theta['lookback_time'] = the rescaled grid, no correction) divided by r."""
    ssp = _bpass()
    csp = _csp(ssp, interp, per_bin, True)
    sfh = _unit_sfh(csp, per_bin)
    T_grid = float(csp.sfh_times[-1]) / 1e9
    for z in (1.0, 6.0):
        r = float(COSMO.age(z)) / T_grid
        tracked = np.asarray(csp.calculate_ssp_weights(_theta(csp, sfh, z, True)))
        explicit = np.asarray(csp.calculate_ssp_weights(dict(
            _theta(csp, sfh, z, True), lookback_time=np.asarray(csp._lookback_from_zred(
                jnp.array([z]))) / 1e9)))
        np.testing.assert_allclose(tracked, explicit / r, rtol=1e-10, atol=1e-300)


def test_fixed_grid_path_is_the_untracked_path():
    """track_zred_age=True without a zred in theta, and track_zred_age=False with one, both give
    the construction-grid weights bit for bit (the correction exists only on the tracked grid)."""
    ssp = _bpass()
    for interp, per_bin in SCHEMES:
        a = _csp(ssp, interp, per_bin, True, track=True)
        b = _csp(ssp, interp, per_bin, True, track=False)
        sfh = _unit_sfh(a, per_bin)
        th = _theta(a, sfh, 3.0, True)
        Wa = np.asarray(a.calculate_ssp_weights({k: v for k, v in th.items() if k != "zred"}))
        Wb = np.asarray(b.calculate_ssp_weights(th))
        assert Wa.tobytes() == Wb.tobytes()


@pytest.mark.parametrize("interp", ["step", "linear"])
def test_grad_wrt_zred_is_finite(interp):
    ssp = _bpass()
    csp = _csp(ssp, interp, False, True)
    sfh = _unit_sfh(csp, False)

    def f(z):
        th = dict(_theta(csp, sfh, 0.0, True), zred=jnp.reshape(z, (1,)),
                  logmass=jnp.array([10.0]))
        return jnp.sum(csp.get_spectrum(theta=th)), jnp.sum(csp.calculate_ssp_weights(th))

    for z in (1.0, 3.0, 6.0):
        gs, gw = jax.grad(lambda z: f(z)[0])(z), jax.grad(lambda z: f(z)[1])(z)
        assert np.isfinite(float(gs)) and float(gs) != 0.0   # the grid moves the spectrum
        assert abs(float(gw)) < 1e-8                         # but not the mass


# ------------------------------------------------------------------ PostProcess ----
def _model(csp, free_z, zred):
    t_yr = np.array(csp.sfh_times)

    def _sfh(t, _t=t_yr):
        return logsfr_ratios_to_sfh(t["logsfr_ratios"], sfh_times_yr=_t)
    init = {"logsfr_ratios": RATIOS, "logmass": jnp.array([10.3])}
    if free_z:
        init["zred"] = jnp.array([zred])
    return SedModel(csp, [], priors={}, transforms={"sfh": _sfh}, free_param_init=init,
                    zred=0.0 if free_z else zred)


def _result(model, zs):
    n = len(zs)
    rng = np.random.default_rng(0)
    samples = {"logsfr_ratios": jnp.asarray(np.asarray(RATIOS)[None] + 0.1 * rng.standard_normal(
                   (n, N - 1))),
               "logmass": jnp.asarray(10.3 + 0.2 * rng.standard_normal(n))}
    if "zred" in model.param_names:
        samples["zred"] = jnp.asarray(np.asarray(zs, dtype=float))
    for name in model.param_names:
        if name not in samples:
            base = np.asarray(model.theta_init[name], dtype=float)
            samples[name] = jnp.asarray(np.repeat(base.reshape(1, -1), n, 0)[:, 0]
                                        if base.size == 1 else np.repeat(base[None], n, 0))
    return SamplingResult(samples=samples, log_evidence=float("nan"),
                          log_evidence_err=float("nan"), log_weights=jnp.zeros(n),
                          log_likelihoods=jnp.zeros(n), param_names=list(model.param_names),
                          n_likelihood_calls=n, wall_time_s=0.0, sampler_name="fake")


def _pp_mass(model, zs):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = PostProcess(model, _result(model, zs), mfrac=False, uv=False, ionizing=False,
                          predictions=False).run()
    return out["extras"]["sfh"]["mass_formed"], 10.0 ** out["theta"]["logmass"], out


@pytest.mark.parametrize("interp", ["step", "linear"])
def test_postprocess_mass_formed_free_z_tracked(interp):
    csp = _csp(_bpass(), interp, False, True)
    got, want, out = _pp_mass(_model(csp, True, 1.0), [0.0, 1.0, 3.0, 6.0])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)
    # the reported SFR is the forward model's: the lookback grid ends at age(z)
    zs = np.asarray(out["theta"]["zred"])
    np.testing.assert_allclose(out["extras"]["sfh"]["lookback_gyr"][:, -1],
                               [float(COSMO.age(z)) for z in zs], rtol=1e-6)


@pytest.mark.parametrize("track", [True, False])
@pytest.mark.parametrize("interp", ["step", "linear"])
def test_postprocess_mass_formed_fixed_z(interp, track):
    z = 0.5
    csp = _csp(_bpass(), interp, False, True, track=track,
               T=None if track else float(COSMO.age(z)))
    got, want, _ = _pp_mass(_model(csp, False, z), [z] * 4)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)


def test_postprocess_mass_formed_free_z_untracked():
    csp = _csp(_bpass(), "step", False, True, track=False, T=float(COSMO.age(6.0)))
    got, want, _ = _pp_mass(_model(csp, True, 3.0), [3.0, 4.0, 6.0])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)



@pytest.mark.parametrize("interp", ["step", "linear"])
def test_summary_truth_sfr_is_the_postprocess_sfr(interp):
    """plotting's truth SFH (summary figure) uses the SFR convention of PostProcess on a
    tracked grid: truths = one draw give that draw's sfr_per_bin."""
    from ceridwen.plotting import _truth_sfr_per_bin
    csp = _csp(_bpass(), interp, False, True)
    model = _model(csp, True, 1.0)
    _got, _want, out = _pp_mass(model, [1.0, 3.0, 6.0])
    for i in range(3):
        truths = {k: np.asarray(v)[i] for k, v in out["theta"].items()}
        np.testing.assert_allclose(_truth_sfr_per_bin(model, out, truths),
                                   out["extras"]["sfh"]["sfr_per_bin"][i], rtol=1e-10)

@pytest.mark.parametrize("interp", ["step", "linear"])
def test_afe_tracked_grid_forms_unit_mass(interp):
    path = TEST_DATA_DIR / "amist_c3k_lr_chab_afe.h5"
    if not path.exists():
        pytest.skip("alpha test grid not present")
    from ceridwen.csp import CSPBasis_afe
    from ceridwen.ssps import SSPDataAfe
    csp = _csp(SSPDataAfe.load(str(path)), interp, False, True, cls=CSPBasis_afe)
    sfh = _unit_sfh(csp, False)
    for z in ZS:
        th = dict(_theta(csp, sfh, z, True), afe=jnp.array([0.2]))
        assert float(jnp.sum(csp.calculate_ssp_weights(th))) == pytest.approx(1.0, rel=1e-10)
    got, want, _ = _pp_mass(_model(csp, True, 1.0), list(ZS))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)


# ------------------------------------------------------ B1-016: linear-scheme weights --
def _tents(la, lt):
    """(n_ssp, n_t) tent functions in log age; zero outside [la[0], la[-1]]."""
    T = np.zeros((la.size, lt.size))
    for j in range(la.size):
        if j > 0:
            m = (lt >= la[j - 1]) & (lt <= la[j])
            T[j, m] = (lt[m] - la[j - 1]) / (la[j] - la[j - 1])
        if j < la.size - 1:
            m = (lt >= la[j]) & (lt < la[j + 1])
            T[j, m] = (la[j + 1] - lt[m]) / (la[j + 1] - la[j])
    return T


def _scheme_brute(la, T_yr, sfh, npts=20001):
    """Per SFH bin: quadrature of SFR(t) x tent_j(log t) over the part of the bin inside the node
    range, scaled so the bin keeps its trapezoidal formed mass."""
    B = np.zeros(la.size)
    for lo, hi, s0, s1 in zip(T_yr[:-1], T_yr[1:], sfh[:-1], sfh[1:]):
        x0, x1 = max(np.log10(max(lo, 1.0)), la[0]), min(np.log10(hi), la[-1])
        if x1 <= x0:
            continue
        lt = np.linspace(x0, x1, npts)
        t = 10.0 ** lt
        sfr = s0 + (s1 - s0) * (t - lo) / (hi - lo)
        b = np.trapezoid(_tents(la, lt) * sfr * t * np.log(10.0), lt, axis=1)
        B += b * (0.5 * (s0 + s1) * (hi - lo)) / b.sum()
    return B


def _naive_brute(la, T_yr, sfh, npts=400001):
    """Whole-grid quadrature, SFR younger than the youngest node on the youngest node."""
    t = np.geomspace(1e2, T_yr[-1], npts)
    lt = np.log10(t)
    tents = _tents(la, lt)
    tents[0, lt < la[0]] = 1.0
    return np.trapezoid(tents * np.interp(t, T_yr, sfh), t, axis=1)


GRIDS_016 = [("ssp_data_bpass.h5", 13.0), ("ssp_data_bpass.h5", 12.0),
             ("ssp_data_mist_miles.h5", 13.7), ("ssp_data_bpass.h5", 13.8)]


@pytest.mark.parametrize("grid,T_max", GRIDS_016)
def test_linear_weights_match_brute_force_quadrature(grid, T_max):
    path = TEST_DATA_DIR / grid
    if not path.exists():
        pytest.skip(f"{grid} not present")
    T = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 3.0, 8.0, T_max])
    sfh = np.array([3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 1.0, 1.0])
    csp = CSPBasis(SSPData.load(str(path)), lookback_time=T, zh_const=True, add_neb=False,
                   add_dust=False, add_diffuse_dust=False, sfh_interp="linear", verbose=False,
                   cosmo=COSMO)
    W = np.asarray(csp.calculate_ssp_weights(
        {"sfh": jnp.asarray(sfh), "logzsol": jnp.array([0.0])}), float).sum(0)
    la = np.asarray(csp.ssp_ages_lgyr, float)            # log10(age / yr)
    B = _scheme_brute(la, T * 1e9, sfh)
    np.testing.assert_allclose(W, B, rtol=1e-6, atol=1e-9 * B.sum())
    # vs the naive quadrature: equal but for the youngest node's share of the youngest bin
    Bn = _naive_brute(la, T * 1e9, sfh)
    d = np.abs(W / W.sum() - Bn / Bn.sum())
    older = la > la[0] + 1.0          # nodes more than 1 dex older than the youngest
    assert d[older].max() < 1e-4 * np.abs(W).max() / W.sum()


@pytest.mark.parametrize("interp", ["step", "linear"])
def test_display_sfh_on_the_tracked_grid(interp):
    """display_sfh draws the SFH the forward model forms: on the age(zred) grid, with the
    construction-grid mass (unit here) in its title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    csp = _csp(_bpass(), interp, False, True)
    sfh = _unit_sfh(csp, False)
    for z in (1.0, 6.0):
        ax = csp.display_sfh(_theta(csp, sfh, z, True))
        x_max = max(float(np.max(line.get_xdata())) for line in ax.get_lines())
        assert x_max == pytest.approx(float(COSMO.age(z)), rel=1e-6)
        assert "M_total = 1.000e+00 M_sun" in ax.get_title()
        plt.close(ax.figure)


def test_golden_linear_weights_match_brute_force_quadrature():
    """The golden "linear" configuration (tests/baselines/manifest.json: psi = exp(-t / 1 Gyr) on
    0-13.79 Gyr, BPASS, log Z = -2.0 = grid node 7): the code's weights equal the quadrature at
    every node, and the stored W_linear_*.npy equal it at every node younger than 10^10 yr (the
    three oldest nodes are the ones B1-016 moves; after their re-capture all nodes agree)."""
    import json
    base = pathlib.Path(__file__).resolve().parents[1] / "baselines"
    man = json.loads((base / "manifest.json").read_text())
    ssp = _bpass()
    T = np.asarray(man["lb_old_gyr"], float)            # increasing, index 0 = today
    psi = np.asarray(man["psi_old"], float)
    csp = CSPBasis(ssp, lookback_time=T, zh_const=True, add_neb=False, add_dust=False,
                   add_diffuse_dust=False, sfh_interp="linear", verbose=False, cosmo=COSMO)
    iz = 7
    th = {"sfh": jnp.asarray(psi), "logzsol": jnp.array([float(np.asarray(ssp.ssp_lgmet)[iz])
                                                          - float(ssp.log10_zsun)])}
    W = np.asarray(csp.calculate_ssp_weights(th), float)
    assert np.all(np.delete(W, iz, axis=0) == 0.0)
    la = np.asarray(csp.ssp_ages_lgyr, float)
    B = _scheme_brute(la, T * 1e9, psi)
    np.testing.assert_allclose(W[iz], B, rtol=1e-6, atol=1e-9 * B.sum())
    young = la < 10.0
    for tag in ("linear_constZ_pernode", "linear_varZ_pernode"):
        Wg = np.load(base / f"W_{tag}.npy").sum(axis=0)
        np.testing.assert_allclose(Wg[young], B[young], rtol=1e-6, atol=1e-9 * B.sum())
