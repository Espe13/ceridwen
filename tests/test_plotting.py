"""ceridwen.plotting on a synthetic PostProcess output and synthetic results
(no grid, no sampler): the three figures are produced and saved."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ceridwen import plotting as P


class O:
    pass


def test_figures_on_synthetic_output(tmp_path):
    rng = np.random.default_rng(1)

    N, n_time, n_wave = 400, 6, 500
    wave = np.linspace(900, 20000, n_wave); z = 3.0
    theta = {"logmass": rng.normal(9.5, 0.1, N), "logsfr_ratios": rng.normal(0, 0.5, (N, n_time-1)), "logzsol": rng.normal(-0.5, 0.2, N),
             "gas_logu": rng.normal(-2.5, 0.2, N), "frac_obrun": np.clip(rng.normal(0.1, 0.05, N), 0, 1)}
    sfr = np.abs(rng.lognormal(0, 0.5, (N, n_time-1))) * 10
    T = np.tile(np.array([0, 0.01, 0.03, 0.1, 0.3, 1.0]), (N, 1))
    spec_obs = np.outer(10**(theta["logmass"]-9.5), 1e-9 * (wave/5000)**-1.0)
    lam_eff = np.array([9000, 11500, 15000, 20000, 27700, 35600, 44400.])
    phot = np.array([[np.interp(l/(1+z), wave, s) for l in lam_eff] for s in spec_obs])
    lines_pred = np.abs(rng.normal(1e-18, 2e-19, (N, 4)))
    ll = rng.normal(-50, 3, N)
    class O: pass
    ph = O(); ph._kind="photometry"; ph.name="phot"; ph.wavelength=lam_eff; ph.flux=phot[0]*1.05; ph.uncertainty=phot[0]*0.1; ph.mask=np.ones(7,bool); ph.upper_limit=np.array([1,0,0,0,0,0,0],bool)
    ln = O(); ln._kind="lines"; ln.name="lines"; ln.line_names=["Hb","[OIII]5007","Ha","[NII]"]; ln.wavelength=np.array([4861,5007,6563,6584.]); ln.flux=lines_pred[0]*1.1; ln.uncertainty=lines_pred[0]*0.2; ln.mask=np.ones(4,bool); ln.upper_limit=None
    model = O(); model.observations=[ph, ln]; model.theta_init={"logmass":np.zeros(1),"logsfr_ratios":np.zeros(5),"logzsol":np.zeros(1),"gas_logu":np.zeros(1),"frac_obrun":np.zeros(1)}
    model.param_names=list(model.theta_init); model.priors={}
    ib = int(np.argmax(ll))
    out = {"theta": theta, "log_likelihood": ll,
           "extras": {"sfh": {"lookback_gyr": T, "sfr": sfr, "sfr10": sfr[:,0], "sfr100": sfr[:,2], "mass_formed": 10**theta["logmass"]}},
           "prediction": {"wave_rest": wave, "spectra_observed": spec_obs, "zred": np.full(N, z), "photometry": {"phot": phot}, "lines": {"lines": lines_pred}, "spectra": {}},
           "bestfit": {"theta": {k: v[ib] for k,v in theta.items()}, "extras": {"sfh": {"sfr": sfr[ib]}}, "prediction": {"photometry": {"phot": phot[ib]}, "lines": {"lines": lines_pred[ib]}, "spectra": {}}},
           "meta": {"zred_fixed": z, "sampler": "blackjax.nss", "log_evidence": -123.4, "log_evidence_err": 0.3}}
    fig = P.summary_figure(out, model, title="ID 12345", prior_draws=0, savepath=tmp_path / "summary.pdf")
    fig = P.corner_figure(out, savepath=tmp_path / "corner.pdf")
    # NS result
    R = O(); R.samples = {k: v for k,v in theta.items()}; R.log_likelihoods = np.sort(ll); R.log_likelihoods_birth = np.sort(ll) - 2; R.log_weights = np.zeros(N); R.raw=None; R.log_evidence=-123.4; R.log_evidence_err=0.3
    fig = P.diagnostic_figure(R, out, savepath=tmp_path / "diag_ns.pdf")
    R2 = O(); R2.samples = {k: v for k,v in theta.items()}; R2.log_likelihoods = ll; R2.log_likelihoods_birth=None; R2.log_weights=np.zeros(N); R2.raw={"num_chains":4,"num_samples":100,"total_divergences":0}
    fig = P.diagnostic_figure(R2, out, savepath=tmp_path / "diag_mcmc.pdf")
    plt.close("all")
    for f in ("summary.pdf", "corner.pdf", "diag_ns.pdf", "diag_mcmc.pdf"):
        assert (tmp_path / f).stat().st_size > 1000


def test_zred0_photometry_on_the_spectrum_and_data_range(tmp_path):
    """zred = 0: PostProcess's spectra are L_sun/Hz x 10^logmass and Photometry divides the
    band-averaged F_nu by 3631 Jy, so the panel must bring the maggies back to the spectrum's
    units (they were drawn ~2.8e19 too high).  The axes cover the observed wavelengths only,
    and the SFH panel shows per-bin log10 SFR against lookback time in Gyr."""
    rng = np.random.default_rng(3)
    N, n_time = 200, 5
    wave = np.geomspace(100.0, 1e8, 3000)
    spec = np.outer(10 ** rng.normal(10.5, 0.05, N), 1e-15 * (wave / 5000.0) ** -0.5)
    lam_eff = np.array([3500.0, 6200.0, 12000.0, 22000.0])
    phot = np.array([[np.interp(l, wave, s) for l in lam_eff] for s in spec]) / 3.631e-20
    ph = O(); ph._kind = "photometry"; ph.name = "phot"; ph.wavelength = lam_eff
    ph.flux = phot[0]; ph.uncertainty = phot[0] * 0.1; ph.mask = np.ones(4, bool); ph.upper_limit = None
    model = O(); model.observations = [ph]; model.theta_init = {"logmass": np.zeros(1)}
    model.param_names = ["logmass"]; model.priors = {}
    T = np.tile(np.array([0.0, 0.1, 1.0, 5.0, 13.0]), (N, 1))
    sfr = np.abs(rng.lognormal(0, 0.3, (N, n_time - 1)))
    theta = {"logmass": rng.normal(10.5, 0.05, N)}
    out = {"theta": theta, "log_likelihood": rng.normal(size=N),
           "extras": {"sfh": {"lookback_gyr": T, "sfr": sfr, "sfr_per_bin": sfr}},
           "prediction": {"wave_rest": wave, "spectra_model": spec, "photometry": {"phot": phot},
                          "lines": {}, "spectra": {}},
           "bestfit": {"theta": {"logmass": theta["logmass"][0]}, "extras": {"sfh": {"sfr": sfr[0]}},
                       "prediction": {"photometry": {"phot": phot[0]}, "lines": {}, "spectra": {}}},
           "meta": {"zred_fixed": 0.0, "sampler": "x", "log_evidence": 0.0}}
    fig = P.summary_figure(out, model, prior_draws=0, savepath=tmp_path / "s.pdf")
    ax_sed, ax_sfh = fig.axes[0], fig.axes[2]
    x_line, y_line = ax_sed.lines[0].get_data()                  # model median
    pts = [c for c in ax_sed.containers if len(getattr(c, "lines", ())) and c.lines[0] is not None]
    xo, yo = pts[0].lines[0].get_data()                          # observed photometry
    ratio = np.asarray(yo) / np.interp(xo, x_line, y_line)
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio          # on the spectrum, not 1e19 away
    lo, hi = ax_sed.get_xlim()
    assert lo > lam_eff.min() / 1e4 / 2 and hi < lam_eff.max() / 1e4 * 2
    assert x_line.min() >= lo and x_line.max() <= hi
    assert ax_sfh.get_xscale() == "log" and "Gyr" in ax_sfh.get_xlabel()
    assert "log" in ax_sfh.get_ylabel() and ax_sfh.get_yscale() == "linear"
    plt.close("all")


def test_ns_diagnostic_titles_use_weighted_quantiles():
    """B3-001: the nested-sampling panels title 'posterior' quantiles; they must be the
    importance-weighted ones, not the quantiles of the unweighted dead points."""
    import types
    import numpy as np
    from ceridwen import plotting as P
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, 5000)
    logl = -0.5 * ((x - 3.0) / 0.1) ** 2
    lw = logl - np.logaddexp.reduce(logl)
    res = types.SimpleNamespace(samples={"x": x}, log_likelihoods=logl, log_weights=lw,
                                log_likelihoods_birth=None, raw=None,
                                log_evidence=0.0, log_evidence_err=0.0)
    title = P.diagnostic_figure(res).axes[0].get_title()
    assert "posterior $3.01^{+0.10}_{-0.10}$" in title, title


def test_ns_diagnostic_labels_logzsol_like_the_summary():
    """B3-021: with a PostProcess output from an [Fe/H] grid the diagnostic panels label
    logzsol [Fe/H], as the summary and corner figures do."""
    import types
    import numpy as np
    from ceridwen import plotting as P
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 0, 500)
    res = types.SimpleNamespace(samples={"logzsol": x}, log_likelihoods=-x ** 2,
                                log_weights=-x ** 2, log_likelihoods_birth=None, raw=None,
                                log_evidence=0.0, log_evidence_err=0.0)
    out = {"meta": {"metallicity": {"axis_meaning": "feh"}}}
    assert P.diagnostic_figure(res, out).axes[0].get_title().startswith(
        P._label("logzsol", True))


def test_corner_axes_include_the_truths():
    """Q1-008: a truth outside the 0.1-99.9 % posterior range was silently not drawn."""
    import numpy as np
    from ceridwen import plotting as P
    rng = np.random.default_rng(2)
    th = {"logmass": rng.normal(10.42, 0.03, 400), "logzsol": rng.normal(-0.2, 0.03, 400)}
    out = {"theta": th, "bestfit": {"theta": {"logmass": [10.42], "logzsol": [-0.2]}}}
    fig = P.corner_figure(out, truths={"logmass": 10.6, "logzsol": -0.2})
    lo, hi = fig.axes[0].get_xlim()
    assert lo < 10.6 < hi


def test_summary_sfh_prior_band_linear_axis_and_full_width_sed(tmp_path):
    """Q2-012: the SFH panel draws the prior 16-84 % band (logmass at its posterior median);
    a uniform node grid gets a linear time axis starting at 0, so the youngest bin is whole;
    without Lines the SED spans the top row instead of leaving the right half empty."""
    import jax.numpy as jnp
    from ceridwen.priors import TopHat
    rng = np.random.default_rng(5)
    N, n_time = 300, 5
    wave = np.geomspace(100.0, 1e8, 2000)
    spec = np.outer(10 ** rng.normal(10.5, 0.05, N), 1e-15 * (wave / 5000.0) ** -0.5)
    lam_eff = np.array([3500.0, 6200.0, 12000.0])
    phot = np.array([[np.interp(l, wave, s) for l in lam_eff] for s in spec]) / 3.631e-20
    ph = O(); ph._kind = "photometry"; ph.name = "phot"; ph.wavelength = lam_eff
    ph.flux = phot[0]; ph.uncertainty = phot[0] * 0.1; ph.mask = np.ones(3, bool); ph.upper_limit = None

    def sfh_of(r):          # per-bin SFR per unit mass from n_time-2 log ratios (any map will do)
        s = jnp.exp(jnp.concatenate([jnp.zeros(r.shape[:-1] + (1,)), jnp.cumsum(r, -1)], -1))
        return s / s.sum(-1, keepdims=True)

    model = O(); model.observations = [ph]
    model.theta_init = {"logmass": np.zeros(1), "logsfr_ratios": np.zeros(n_time - 2)}
    model.param_names = list(model.theta_init)
    model.priors = {"logmass": TopHat(low=8.0, high=12.0), "logsfr_ratios": TopHat(low=-2.0, high=2.0)}
    model.apply_transforms = lambda th: {**th, "sfh": sfh_of(th["logsfr_ratios"])}
    model._zred_fixed = None
    T = np.tile(np.linspace(0.0, 12.0, n_time), (N, 1))
    theta = {"logmass": rng.normal(10.5, 0.05, N), "logsfr_ratios": rng.normal(0, 0.3, (N, n_time - 2))}
    per_bin = np.asarray(sfh_of(jnp.asarray(theta["logsfr_ratios"]))) * 10 ** theta["logmass"][:, None]
    out = {"theta": theta, "log_likelihood": rng.normal(size=N),
           "extras": {"sfh": {"lookback_gyr": T, "sfr": per_bin, "sfr_per_bin": per_bin}},
           "prediction": {"wave_rest": wave, "spectra_model": spec, "photometry": {"phot": phot},
                          "lines": {}, "spectra": {}},
           "bestfit": {"theta": {k: v[0] for k, v in theta.items()}, "extras": {"sfh": {"sfr": per_bin[0]}},
                       "prediction": {"photometry": {"phot": phot[0]}, "lines": {}, "spectra": {}}},
           "meta": {"zred_fixed": 0.0, "sampler": "x", "log_evidence": 0.0}}
    fig = P.summary_figure(out, model, prior_draws=400, savepath=tmp_path / "s.pdf")
    ax_sed, ax_sfh = fig.axes[0], fig.axes[2]
    assert ax_sed.get_position().width > 0.8                     # the whole top row
    assert ax_sfh.get_xscale() == "linear" and ax_sfh.get_xlim() == (0.0, 12.0)
    assert ax_sfh.get_ylim()[0] > 8.0                            # no step down to log SFR = 0
    labels = ax_sfh.get_legend_handles_labels()[1]
    assert "prior 16$-$84% (at the median mass)" in labels
    # the prior band, recomputed independently from the same prior draws
    band = [p for p in ax_sfh.patches if p.get_label().startswith("prior")][0]
    hi, edges, lo = band.get_data()
    draws = P._prior_draws(model, 400)
    m_med = np.median(theta["logmass"])
    want = np.log10(np.asarray(sfh_of(jnp.asarray(draws["logsfr_ratios"])))) + m_med
    np.testing.assert_allclose(lo, np.quantile(want, 0.16, axis=0), rtol=1e-10)
    np.testing.assert_allclose(hi, np.quantile(want, 0.84, axis=0), rtol=1e-10)
    np.testing.assert_allclose(edges, np.linspace(0.0, 12.0, n_time))
    plt.close("all")
