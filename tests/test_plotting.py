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
