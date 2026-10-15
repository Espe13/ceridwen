"""Capture the forward model and a mock likelihood in the precision mode of THIS process, for
``tests/test_float32_parity.py`` (float32 mode: Apple GPU, ``docs/mac_gpu.md``).

``jax_enable_x64`` is process-global and set once by ``import ceridwen`` from the environment,
so the parity test runs this file in subprocesses: with float64 (the default; the reference),
with ``CERIDWEN_X64=0`` on the CPU, and, when jax-graft is installed, in float32 on the Apple
GPU.  Nothing here forces ``jax_enable_x64``: ``import ceridwen`` decides it.

Usage::

    python tests/baselines/_capture_f32.py OUT.npz [--mock-data REF.npz]

Writes, for every golden config of ``tests/baselines/manifest.json``, ``W_<tag>``,
``spec_<tag>``, ``lines_<tag>`` and ``maggies_<tag>`` computed exactly as
``tests/csp/test_lookback_flip_invariant.py`` computes them, and for the mock of
:func:`build_mock` its predictions (``pred_<key>``), its data (``data_<key>``,
``sigma_<key>``), the ln posterior and its gradient at ``theta_init`` and the ln posterior at a
second theta.  ``--mock-data REF.npz`` makes the mock fit the data arrays stored in ``REF.npz``
instead of generating them, so that a float32 run and the float64 reference see the same
observations.  ``precision`` and ``backend`` record what produced the file.

``scripts/mac_metal_probe.py --with-model`` imports :func:`build_mock` and
:func:`log_posterior` from here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))                 # tests/_gridfixture.py

import jax                                                 # noqa: E402
import jax.numpy as jnp                                    # noqa: E402

import ceridwen                                            # noqa: E402,F401  (sets x64 from the env)
from ceridwen import SSPData, CSPBasis, SedModel, Cosmology            # noqa: E402
from ceridwen.observation import Photometry, Spectrum, Lines           # noqa: E402
from ceridwen.broadening import Kinematics, Instrument                 # noqa: E402
from ceridwen.model.transforms import logsfr_ratios_to_sfh             # noqa: E402
from ceridwen.precision import precision_name                          # noqa: E402

from _gridfixture import find_test_grid                                # noqa: E402

# --------------------------------------------------------------------------- golden configs
# (the builder of tests/csp/test_lookback_flip_invariant.py; that module forces x64 on at
# import, so it cannot be imported here)
with open(THIS_DIR / "manifest.json") as f:
    MANIFEST = json.load(f)
N_TIME = MANIFEST["n_time"]
LOGZSOL_BASELINE = -0.30102999566398125
CONFIGS = [(c["sfh_interp"], c["zh_const"], c["per_bin"], c["tag"]) for c in MANIFEST["configs"]]


def _golden_theta(zh_const, per_bin):
    lb = jnp.asarray(MANIFEST["lb_old_gyr"])
    psi = jnp.asarray(MANIFEST["psi_old"])
    sfh = 0.5 * (psi[:-1] + psi[1:]) if per_bin else psi
    theta = {"lookback_time": lb, "sfh": sfh}
    if zh_const:
        theta["logzsol"] = jnp.asarray([LOGZSOL_BASELINE])
    else:
        theta["logzsol_hist"] = jnp.asarray(np.full(N_TIME, LOGZSOL_BASELINE))
    return theta


def _golden_csp(ssp, theta, *, sfh_interp, zh_const):
    return CSPBasis(ssp, theta=theta, cosmo=Cosmology.planck18(), zh_const=zh_const,
                    add_dust=False, add_diffuse_dust=False, add_dust_emission=False,
                    add_neb=True, init_neb_params={"cloudy_dust": True}, nebemlineinspec=True,
                    verbose=False, sfh_interp=sfh_interp)


def _golden_phot():
    return Photometry(filters=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
                      flux=jnp.zeros(5), uncertainty=jnp.ones(5) * 1e-12,
                      mask=jnp.ones(5, dtype=bool), name="phot")


def golden_arrays(ssp) -> dict:
    """``{W_<tag>, spec_<tag>, lines_<tag>, maggies_<tag>}`` for the manifest's configs."""
    out = {}
    phot = _golden_phot()
    for sfh_interp, zh_const, per_bin, tag in CONFIGS:
        csp = _golden_csp(ssp, _golden_theta(zh_const, per_bin), sfh_interp=sfh_interp,
                          zh_const=zh_const)
        theta = {k: jnp.asarray(v) for k, v in csp.theta_init.items()}
        theta.setdefault("logmass", jnp.zeros(1))
        theta.setdefault("zred", jnp.zeros(1))
        theta.setdefault("igm_factor", jnp.ones(1))
        theta.setdefault("eline_scaling", jnp.ones(1))
        out[f"W_{tag}"] = np.asarray(csp.calculate_ssp_weights(theta))
        out[f"spec_{tag}"] = np.asarray(csp.get_spectrum(theta))
        out[f"lines_{tag}"] = np.asarray(csp.get_line_spec(theta))
        spec_phot = csp._apply_mass_redshift_igm(*csp._assemble_observer_spectra(theta), theta)[0]
        out[f"maggies_{tag}"] = np.asarray(phot.get_maggies(csp.wave, spec_phot))
        print(f"  golden {tag}: done", flush=True)
    return out


# --------------------------------------------------------------------------- the mock
# A galaxy at z = 0.8 with photometry, a spectrum and three line fluxes, all in CERIDWEN's
# physical units (cgs F_nu, AB maggies, erg/s/cm^2): the magnitudes a float32 kernel has to
# survive (sigma ~ 1e-31 for the spectrum).  Deterministic: no random numbers.
ZRED = 0.8
MOCK_N_TIME = 5
FILTERS = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
SPEC_WAVE_REST = np.linspace(3800.0, 8800.0, 600)
LINES = {"Hbeta": 4862.71, "OIII5007": 5008.24, "Halpha": 6564.61}
MOCK_KEYS = ("phot", "spec", "lines")
THETA_TRUE = {"logsfr_ratios": np.array([0.3, 0.1, -0.2, -0.4]), "logmass": np.array([10.0]),
              "logzsol": np.array([-0.3]), "tau_pow": np.array([0.6]),
              "alpha_pow": np.array([-0.7]), "diffuse_tau_kc": np.array([0.4]),
              "diffuse_dust_index": np.array([-0.2]), "gas_logz": np.array([-0.4]),
              "gas_logu": np.array([-2.5])}


def _mock_data_from(pred: dict) -> dict:
    """Mock data and uncertainties from a prediction: a 3 per cent ripple on the flux and 5 per
    cent errors (plus 1 per cent of the median, so no sigma is 0)."""
    out = {}
    for key in MOCK_KEYS:
        p = np.asarray(pred[key], dtype=np.float64)
        i = np.arange(p.size)
        out[f"data_{key}"] = p * (1.0 + 0.03 * np.cos(2.0 * np.pi * i / 37.0))
        out[f"sigma_{key}"] = 0.05 * np.abs(p) + 0.01 * np.median(np.abs(p))
    return out


def _mock_model(grid_path: str, data: dict | None):
    ssp = SSPData.load(grid_path)
    cosmo = Cosmology.planck18()
    import os
    csp = CSPBasis(ssp, lookback_time=jnp.linspace(0.0, float(cosmo.age(ZRED)), MOCK_N_TIME),
                   zh_const=True, sfh_interp="step", add_dust=True, add_diffuse_dust=True,
                   add_neb=True, add_igm=True, add_dust_emission=False, verbose=False,
                   cosmo=cosmo, sps_home=os.environ["SPS_HOME"])
    t_yr = np.array(csp.sfh_times)

    def _sfh(t, _t=t_yr):
        return logsfr_ratios_to_sfh(t["logsfr_ratios"], sfh_times_yr=_t)
    _sfh.__name__ = "logsfr_ratios_to_sfh"

    def dat(key, n):
        if data is None:
            return np.ones(n), 0.1 * np.ones(n)
        return np.asarray(data[f"data_{key}"]), np.asarray(data[f"sigma_{key}"])

    f, s = dat("phot", 5)
    obs = [Photometry(filters=FILTERS, flux=f, uncertainty=s, name="phot")]
    f, s = dat("spec", SPEC_WAVE_REST.size)
    obs.append(Spectrum(wavelength=SPEC_WAVE_REST * (1.0 + ZRED), flux=f, uncertainty=s,
                        instrument=Instrument.sigma_kms(150.0), name="spec"))
    f, s = dat("lines", len(LINES))
    obs.append(Lines(line_ind=np.arange(len(LINES)), line_names=list(LINES),
                     wavelength=np.array(list(LINES.values())), flux=f, uncertainty=s,
                     name="lines"))
    init = {k: jnp.asarray(v) for k, v in THETA_TRUE.items()}
    return SedModel(csp, obs, priors=mock_priors(), transforms={"sfh": _sfh},
                    free_param_init=init, zred=ZRED, kinematics=Kinematics(sigma_gal=250.0))


def mock_priors() -> dict:
    from ceridwen.sampler.priors import Uniform, StudentT
    return {"logmass": Uniform(low=8.0, high=12.0),
            "logsfr_ratios": StudentT(mean=0.0, scale=1.0, df=2.0),
            "logzsol": Uniform(low=-1.8, high=0.2), "tau_pow": Uniform(low=0.0, high=2.0),
            "alpha_pow": Uniform(low=-2.0, high=0.0),
            "diffuse_tau_kc": Uniform(low=0.0, high=2.0),
            "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
            "gas_logz": Uniform(low=-1.3, high=0.3), "gas_logu": Uniform(low=-4.0, high=-1.0)}


def build_mock(grid_path: str, data: dict | None = None):
    """The mock :class:`SedModel`; ``data`` = ``{data_<key>, sigma_<key>}`` to fit given
    observations, else the data are generated from the model's own prediction at
    ``THETA_TRUE`` (built twice: the observations need the data at construction)."""
    model = _mock_model(grid_path, data)
    if data is None:
        pred = model.predict(model.theta_init)
        model = _mock_model(grid_path, _mock_data_from(pred))
    return model


def log_posterior(model):
    """``(lnprob, offset)``: the jitted ln posterior ``fitSED`` would sample (the
    MultiObservationLikelihood of the model's observations plus ``model.ln_prior``) and the
    float32-mode constant it is offset by (0.0 under float64; ``likelihood.loglike_offset``)."""
    from ceridwen.fit import _likelihood_for
    from ceridwen.likelihood import MultiObservationLikelihood
    from ceridwen.likelihood.likelihood import float32_rescaling, loglike_offset
    keys = tuple(model.obs_dict)
    lhs = tuple(_likelihood_for(model.obs_dict[k], model.param_names, model=model) for k in keys)
    multi = MultiObservationLikelihood(keys=keys, likelihoods=lhs)

    class _Prior:
        @staticmethod
        def log_prob(theta):
            return model.ln_prior(theta)

    offset = loglike_offset(float32_rescaling(model.obs_dict[k], lh) for k, lh in zip(keys, lhs))
    return multi.make_lnprobfn(model.obs_dict, model, _Prior()), offset


def theta_second(model) -> dict:
    """A second theta: ``THETA_TRUE`` moved by a fraction of each prior's width."""
    shift = {"logmass": 0.15, "logsfr_ratios": -0.2, "logzsol": 0.2, "tau_pow": 0.3,
             "alpha_pow": 0.25, "diffuse_tau_kc": 0.3, "diffuse_dust_index": 0.15,
             "gas_logz": 0.3, "gas_logu": 0.4}
    return {k: jnp.asarray(np.asarray(THETA_TRUE[k]) + shift[k]) for k in model.theta_init}


def mock_arrays(grid_path: str, data: dict | None) -> dict:
    model = build_mock(grid_path, data)
    out = {}
    pred = model.predict(model.theta_init)
    for key in MOCK_KEYS:
        out[f"pred_{key}"] = np.asarray(pred[key])
        out[f"data_{key}"] = np.asarray(model.obs_dict[key].flux)
        out[f"sigma_{key}"] = np.asarray(model.obs_dict[key].uncertainty)
    lnprob, offset = log_posterior(model)
    th0 = model.theta_init
    out["lnprob_offset"] = np.asarray(offset)
    out["lnprob_0"] = np.asarray(lnprob(th0)) + offset
    out["lnprob_1"] = np.asarray(lnprob(theta_second(model))) + offset
    grad = jax.jit(jax.grad(lnprob))(th0)
    for k in sorted(grad):
        out[f"grad_{k}"] = np.asarray(grad[k])
    print(f"  mock: lnprob_0 {float(out['lnprob_0']):.6f} (kernel offset {offset:.3f})", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--mock-data", default=None, help="REF.npz whose data_* / sigma_* to fit")
    ap.add_argument("--no-goldens", action="store_true")
    args = ap.parse_args(argv)
    grid = find_test_grid()
    if grid is None:
        raise SystemExit("no test grid (tests/_gridfixture.py)")
    data = None
    if args.mock_data:
        with np.load(args.mock_data) as ref:
            data = {k: ref[k] for k in ref.files if k.startswith(("data_", "sigma_"))}
    out = {"precision": np.asarray(precision_name()),
           "backend": np.asarray(jax.default_backend()),
           "jax_version": np.asarray(jax.__version__)}
    print(f"precision {precision_name()}  backend {jax.default_backend()}", flush=True)
    if not args.no_goldens:
        out.update(golden_arrays(SSPData.load(str(grid))))
    out.update(mock_arrays(str(grid), data))
    np.savez(args.out, **out)
    print(f"saved {len(out)} arrays -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
