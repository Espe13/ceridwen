"""Posterior post-processing of a fit: parameters, SFH and its time averages,
UV and ionising-photon properties, posterior-predictive observations, model-grid
spectra, the best-fit point and user-defined derived quantities.

    pp  = PostProcess(model, result)                 # result: SamplingResult or .h5 path
    out = pp.run()                                   # nested dict of numpy arrays
    pp.save("post.npz")                              # flat .npz + json meta; load_postprocess()

    out["theta"]["logmass"]                          # (N,)  equal-weight posterior draws
    out["extras"]["sfh"]["sfr"]                      # (N, n_bins)   M_sun / yr per SFH bin
    out["extras"]["sfh"]["sfr10"]                    # (N,)  mean SFR over the last 10 Myr
    out["extras"]["sfh"]["ssfr10"]                   # (N,)  sfr10 / formed mass  [1/yr]
    out["extras"]["uv"]["MUV"]                       # (N,)  absolute AB mag at 1500 A (dust-attenuated)
    out["extras"]["ionizing"]["nion"]                # (N,)  Q(H) [s^-1] from the intrinsic spectrum
    out["extras"]["elines"]["mean"]                  # (N, m) posterior line fluxes [erg/s/cm^2] (Spectrum(marginalize_elines=True) only;
                                                     #        also "sd", "cloudy", "names", "wave_rest"; predictions then carry these lines)
    out["prediction"]["photometry"]["phot"]          # (N, n_bands) maggies
    out["prediction"]["spectra"]["spec"]             # (N, n_pix)   as the Spectrum observation
    out["prediction"]["spectra_model"]               # (N, n_wave)  rest-frame L_nu [L_sun/Hz] on wave_rest, model resolution (no kinematic broadening)
    out["prediction"]["spectra_observed"]            # (N, n_wave)  observed-frame f_nu [erg/s/cm^2/Hz, cgs] at (1+z) wave_rest
    pp.figures("post_figs/")                         # summary, corner and sampling-diagnostic figures
    out["bestfit"]["theta"]["logmass"]               # the highest-likelihood sample, with all of the above

Custom quantities are functions of one sample::

    def A_V(s):                                      # s: SpectrumSample
        i = s.index_of(5500.0)
        return 2.5 * np.log10(s.dustfree[i] / s.full[i])

    pp = PostProcess(model, result, derived={"A_V": A_V})
    out["derived"]["A_V"]                            # (N,)

Conventions
-----------
* Draws are resampled to equal weight (nested-sampling weights are recomputed
  from ``log_likelihoods`` / ``log_likelihoods_birth``); uniform-weight results
  are used as they are.
* ``logmass`` is the FORMED mass; the physical SFR is ``theta['sfh']`` times
  ``10**logmass`` (the shape alone when there is no ``logmass``).  ``sfrW`` is the
  mean SFR over the last W Myr of lookback time on the ``sfh_interp`` piecewise
  function ("step" per bin, "linear" between nodes); ``ssfrW = sfrW / mass_formed``
  with no return fraction.
* Model-grid spectra are rest-frame L_nu [L_sun/Hz] times ``10**logmass``, no
  distance, (1+z) or IGM.  ``spectra_model``: the fitted model; ``spectra_intrinsic``:
  stellar continuum only (ionising continuum included); ``spectra_dustfree``: stars
  plus nebular, no dust.  Per-observation predictions are in the observation's units.
* ``LUV`` is the mean L_nu over 1450-1550 A rest [erg/s/Hz];
  ``MUV = -2.5 log10(LUV / (4 pi (10 pc)^2) / 3631 Jy)``; both from ``spectra_model``,
  ``*_intrinsic`` from ``spectra_intrinsic``.
* ``nion = Q(H) = (L_sun/h) int_{lambda<912 A} L_nu / lambda dlambda`` [s^-1] from the
  intrinsic spectrum; ``xion = nion / LUV_intrinsic`` [Hz/erg]; ``fesc`` is
  ``frac_obrun`` when the model has it.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["PostProcess", "SpectrumSample", "load_postprocess"]

_LSUN_ERG_S = 3.839e33
_HPLANCK_ERG_S = 6.6261e-27
_LYMAN_LIMIT_AA = 912.0
_PC_CM = 3.0856775814913673e18
_AB_ZERO_FNU = 3.631e-20                      # erg s^-1 cm^-2 Hz^-1
_UV_WINDOW_AA = (1450.0, 1550.0)

DEFAULT_WINDOWS_MYR = (3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0)

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@dataclass
class SpectrumSample:
    """One posterior draw as handed to a ``derived`` function: rest-frame L_nu
    [L_sun/Hz] x 10**logmass on ``wave_rest`` [A]; ``theta`` after the model transforms."""
    wave_rest: np.ndarray
    full: np.ndarray
    intrinsic: np.ndarray
    dustfree: np.ndarray
    theta: dict
    zred: float
    logmass: Optional[float]
    sfr: np.ndarray
    lookback_gyr: np.ndarray
    cosmo: Any

    def index_of(self, wave_aa: float) -> int:
        """Index of the model pixel closest to ``wave_aa`` (rest frame)."""
        return int(np.argmin(np.abs(self.wave_rest - float(wave_aa))))

    def mean_lnu(self, spectrum: np.ndarray, lo_aa: float, hi_aa: float) -> float:
        """Wavelength-averaged L_nu of ``spectrum`` over [lo, hi] A (trapezoid)."""
        m = (self.wave_rest >= lo_aa) & (self.wave_rest <= hi_aa)
        if m.sum() < 2:
            raise ValueError(f"fewer than two model pixels in [{lo_aa}, {hi_aa}] A")
        w = self.wave_rest[m]
        return float(_trapz(spectrum[m], w) / (w[-1] - w[0]))

    def luminosity(self, spectrum: np.ndarray, lo_aa: float, hi_aa: float) -> float:
        """int L_nu dnu over [lo, hi] A of ``spectrum`` [erg/s]."""
        m = (self.wave_rest >= lo_aa) & (self.wave_rest <= hi_aa)
        if m.sum() < 2:
            raise ValueError(f"fewer than two model pixels in [{lo_aa}, {hi_aa}] A")
        nu = 2.99792458e18 / self.wave_rest[m]
        return float(-_trapz(spectrum[m], nu) * _LSUN_ERG_S)


def _per_bin_and_nodes(psi: np.ndarray, n_time: int):
    """(psi per bin, psi per node) from a per-node or per-bin SFH."""
    psi = np.atleast_1d(np.asarray(psi, dtype=float)).ravel()
    if psi.size == n_time - 1:
        bar = psi
        nodes = np.empty(n_time)
        nodes[0], nodes[-1] = psi[0], psi[-1]
        nodes[1:-1] = 0.5 * (psi[:-1] + psi[1:])
    elif psi.size == n_time:
        bar = 0.5 * (psi[:-1] + psi[1:])
        nodes = psi
    else:
        raise ValueError(f"sfh has {psi.size} values for a {n_time}-node grid")
    return bar, nodes


def _mean_sfr_window(T_yr: np.ndarray, bar: np.ndarray, nodes: np.ndarray,
                     w_yr: float, interp: str) -> float:
    """Mean SFR over lookback [0, w_yr]; zero beyond the oldest node."""
    if interp == "step":
        lo, hi = T_yr[:-1], T_yr[1:]
        overlap = np.clip(np.minimum(hi, w_yr) - np.maximum(lo, 0.0), 0.0, None)
        return float(np.sum(bar * overlap) / w_yr)
    t_end = min(w_yr, T_yr[-1])
    if t_end <= T_yr[0]:
        return 0.0
    grid = np.concatenate([T_yr[T_yr < t_end], [t_end]])
    vals = np.interp(grid, T_yr, nodes)
    return float(_trapz(vals, grid) / w_yr)


def _formed_mass(T_yr, bar, nodes, interp):
    if interp == "step":
        return float(np.sum(bar * np.diff(T_yr)))
    return float(_trapz(nodes, T_yr))


class PostProcess:
    """Posterior post-processing of a fit.

    Parameters
    ----------
    model : SedModel -- the model the samples were drawn with; parameter names must match the result's
    result : SamplingResult or path -- sampler output or the HDF5 file it was written to
    n_samples : int, optional -- equal-weight draws (default 2000 for weighted results, all for uniform)
    windows_myr : sequence of float, Myr -- averaging windows W for ``sfrW`` / ``ssfrW``
    derived : dict[str, callable] -- ``name -> f(SpectrumSample) -> float or 1-D array``, evaluated per draw
    batch_size : int -- draws per compiled batch through the forward model
    """

    def __init__(self, model, result, *, n_samples: Optional[int] = None,
                 seed: int = 0, windows_myr: Sequence[float] = DEFAULT_WINDOWS_MYR,
                 sfr: bool = True, ssfr: bool = True, uv: bool = True,
                 ionizing: bool = True, predictions: bool = True,
                 derived: Optional[dict] = None, batch_size: int = 256):
        self.model = model
        self.csp = model.csp
        self.result = self._load_result(result)
        self.seed = int(seed)
        self.windows_myr = tuple(float(w) for w in windows_myr)
        if any(w <= 0 for w in self.windows_myr):
            raise ValueError("windows_myr must be positive")
        self.want = dict(sfr=bool(sfr), ssfr=bool(ssfr), uv=bool(uv),
                         ionizing=bool(ionizing), predictions=bool(predictions))
        if self.want["ssfr"] and not self.want["sfr"]:
            raise ValueError("ssfr=True needs sfr=True")
        self.derived = dict(derived or {})
        for k, f in self.derived.items():
            if not callable(f):
                raise TypeError(f"derived[{k!r}] is not callable")
            if k in ("sfh", "uv", "ionizing"):
                raise ValueError(f"derived name {k!r} collides with a built-in block")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._samples_np = {k: np.asarray(v, dtype=float) for k, v in self.result.samples.items()}
        self._logl_np = np.asarray(self.result.log_likelihoods, dtype=float)
        self._check_names()
        self.log_weights = self._log_weights()
        self.n_samples = n_samples
        self.output: Optional[dict] = None

    def _load_result(self, result):
        if isinstance(result, (str, Path)):
            from .fit import load_result_h5
            self._check_file_against_model(result)
            return load_result_h5(result)
        for attr in ("samples", "log_weights", "log_likelihoods", "param_names"):
            if not hasattr(result, attr):
                raise TypeError(f"result lacks '{attr}': pass a SamplingResult or an HDF5 path")
        return result

    def _check_file_against_model(self, path):
        """Refuse a file whose recorded zred / lumdist / cosmology differ from the model's."""
        import h5py
        from .cosmology import Cosmology
        m = self.model
        with h5py.File(Path(path), "r") as f:
            a = dict(f["model"].attrs)
        if "zred" in a and abs(float(a["zred"]) - float(m.zred)) > 1e-12:
            raise ValueError(f"the result was fitted at zred = {float(a['zred']):g} but the "
                             f"model has zred = {float(m.zred):g}")
        ld_file = a.get("lumdist_mpc", None)
        ld_model = getattr(m, "lumdist_mpc", None)
        if (ld_file is None) != (ld_model is None) or (
                ld_file is not None and abs(float(ld_file) - float(ld_model)) > 1e-9):
            raise ValueError(f"the result was fitted with lumdist_mpc = {ld_file} but the "
                             f"model has lumdist_mpc = {ld_model}")
        if any(k.startswith("cosmo_") for k in a):
            c_file = Cosmology.from_dict(a)
            if c_file != self.csp.cosmo:
                raise ValueError("the result was fitted with a different cosmology:\n"
                                 f"  file : {c_file.describe()}\n"
                                 f"  model: {self.csp.cosmo.describe()}")
        else:
            warnings.warn(f"{path}: no cosmology recorded in the file (written before it was "
                          "stored); cannot verify it against the model")
        csp = self.csp
        if hasattr(csp, "log10_zsun"):
            f_l0 = a.get("log10_zsun", None)
            if f_l0 is not None and float(f_l0) != float(csp.log10_zsun):
                raise ValueError(
                    f"the result was fitted on a grid with log10 Z_sun = {float(f_l0)!r} but "
                    f"the model's grid has {float(csp.log10_zsun)!r}: its logzsol samples mean "
                    "a different metallicity.  Rebuild the model with the grid of that fit.")
            f_ch = a.get("grid_chash", None)
            if f_ch is not None and str(f_ch) != str(getattr(csp, "grid_chash", "")):
                warnings.warn(
                    f"{path}: fitted with SSP grid {str(f_ch)} but the model uses "
                    f"{getattr(csp, 'grid_chash', None)}; Z_sun agrees, the grids differ "
                    "otherwise (library, IMF or ages).")
        kin = getattr(m, "kinematics", None)
        if kin is not None and "kinematics_sigma_gal" in a:
            got = (str(a["kinematics_sigma_gal"]), str(a["kinematics_sigma_gas"]),
                   bool(a.get("broaden_photometry", True)))
            want = (str(kin.sigma_gal), str(kin.effective_sigma_gas), bool(m.broaden_photometry))
            if got != want:
                raise ValueError("the result was fitted with different kinematics:\n"
                                 f"  file : sigma_gal={got[0]}, sigma_gas={got[1]}, "
                                 f"broaden_photometry={got[2]}\n"
                                 f"  model: sigma_gal={want[0]}, sigma_gas={want[1]}, "
                                 f"broaden_photometry={want[2]}")

    def _log_weights(self) -> np.ndarray:
        """Posterior log-weights aligned with the samples (recomputed from logL, logL_birth when available)."""
        stored = np.asarray(self.result.log_weights, dtype=float)
        births = getattr(self.result, "log_likelihoods_birth", None)
        if births is None:
            self._weights_source = "stored"
            return stored
        from .sampler.ns_weights import nested_log_weights
        lw = nested_log_weights(self._logl_np, np.asarray(births))
        self._weights_source = "recomputed from logL and logL_birth"
        fin = np.isfinite(lw) & np.isfinite(stored)
        if fin.any():
            a = lw[fin] - np.max(lw[fin]); b = stored[fin] - np.max(stored[fin])
            a -= np.log(np.sum(np.exp(a))); b -= np.log(np.sum(np.exp(b)))
            if np.max(np.abs(np.exp(a) - np.exp(b))) > 1e-6:
                warnings.warn("the stored log_weights differ from the weights recomputed from "
                              "(logL, logL_birth); the recomputed ones are used.  Stored weights "
                              "from before 2026-09-03 were sorted by likelihood and misaligned "
                              "with the samples for the final live points.")
        return lw

    def _check_names(self):
        have = set(self.result.param_names)
        need = set(self.model.param_names)
        if have != need:
            raise ValueError(
                "the result and the model do not describe the same parameters:\n"
                f"  only in result: {sorted(have - need)}\n"
                f"  only in model : {sorted(need - have)}\n"
                "Rebuild the SedModel exactly as it was for the fit (same "
                "transforms and free_param_init).")
        for name in self.model.param_names:
            s = self._samples_np[name]
            want = int(np.size(self.model.theta_init[name]))
            got = 1 if s.ndim == 1 else int(np.prod(s.shape[1:]))
            if got != want:
                raise ValueError(f"samples['{name}'] has {got} values per draw; the model "
                                 f"expects {want}")
        n = {self._samples_np[k].shape[0] for k in self.model.param_names}
        if len(n) != 1:
            raise ValueError(f"inconsistent sample counts across parameters: {n}")
        self.n_raw = n.pop()
        if self._logl_np.shape[0] != self.n_raw:
            raise ValueError("log_likelihoods length does not match the samples")
        if np.asarray(self.result.log_weights).shape[0] != self.n_raw:
            raise ValueError("log_weights length does not match the samples")

    def _uniform_weights(self) -> bool:
        lw = self.log_weights
        finite = np.isfinite(lw)
        if not finite.any():
            raise ValueError("no finite log_weights in the result")
        return bool(np.allclose(lw[finite], lw[finite][0]))

    def _draw_indices(self) -> np.ndarray:
        lw = self.log_weights
        finite = np.isfinite(lw)
        uniform = self._uniform_weights()
        rng = np.random.default_rng(self.seed)
        if uniform:
            idx = np.flatnonzero(finite)
            if self.n_samples is not None and self.n_samples < idx.size:
                idx = np.sort(rng.choice(idx, size=self.n_samples, replace=False))
            return idx
        n = 2000 if self.n_samples is None else int(self.n_samples)
        w = np.where(finite, lw, -np.inf)
        w = np.exp(w - np.max(w[finite]))
        w /= w.sum()
        return rng.choice(self.n_raw, size=n, replace=True, p=w)

    def _theta_batch(self, idx: np.ndarray) -> dict:
        """Free-parameter dict with a leading draw axis."""
        out = {}
        for name in self.model.param_names:
            arr = self._samples_np[name][idx]
            shape = np.shape(self.model.theta_init[name])
            out[name] = jnp.asarray(arr.reshape((arr.shape[0],) + tuple(shape)))
        return out

    def _model_theta(self, free_theta):
        """Transformed theta for one draw with fixed zred / lumdist injected."""
        m = self.model
        t = m.apply_transforms(free_theta)
        if t is free_theta:
            t = dict(t)
        if m._zred_fixed is not None and "zred" not in t:
            t["zred"] = m._zred_fixed
        if getattr(m, "_lumdist_fixed", None) is not None and "lumdist_mpc" not in t:
            t["lumdist_mpc"] = m._lumdist_fixed
        return t

    def _spectra_one(self, free_theta):
        """(full, intrinsic, dustfree) rest-frame L_nu x 10**logmass for one draw."""
        csp = self.csp
        t = self._model_theta(free_theta)
        full = csp.get_spectrum(t, include_lines=True)
        intrinsic = csp.get_spectrum_nodattn_nodem_noneb(t)
        if getattr(csp, "neb", None) is not None:
            dustfree = csp.get_spectrum_nodattn_nodem_neb(t, include_lines=True)
        else:
            dustfree = intrinsic
        out = {}
        if "zred" in t:
            out["observed"] = csp._apply_mass_redshift_igm(full, full, t)[0]     # cgs f_nu at (1+z) wave
            out["zred"] = jnp.ravel(t["zred"])[0]
        if "logmass" in t:
            scale = 10.0 ** jnp.ravel(t["logmass"])[0]
            full, intrinsic, dustfree = full * scale, intrinsic * scale, dustfree * scale
        out.update({"full": full, "intrinsic": intrinsic, "dustfree": dustfree})
        return out

    def _sfh_one(self, free_theta):
        """(lookback_yr (n_time,), sfh shape) for one draw."""
        csp = self.csp
        t = self._model_theta(free_theta)
        if "lookback_time" in t:
            T = jnp.atleast_1d(jnp.asarray(t["lookback_time"], dtype=float)) * 1e9
        elif csp.track_zred_age and "zred" in t:
            T = csp._lookback_from_zred(t["zred"])
        else:
            T = csp.sfh_times
        return {"T_yr": T, "sfh": jnp.ravel(jnp.asarray(t["sfh"], dtype=float))}

    def _compile(self):
        if getattr(self, "_f_spectra", None) is None:
            self._f_spectra = jax.jit(jax.vmap(self._spectra_one))
            self._f_sfh = jax.jit(jax.vmap(self._sfh_one))

    def _predictor(self):
        """Batched posterior-predictive observations: ``model.predict_vmap``, or, when a
        Spectrum marginalises the emission lines, the predictions with each draw's
        posterior-mean line fluxes (the likelihood fitSED builds) plus the line posteriors."""
        es = getattr(self.model, "_eline_system", None)
        if es is None:
            return self.model.predict_vmap
        if getattr(self, "_f_elines", None) is None:
            from .fit import _likelihood_for
            from .likelihood.likelihood import MultiObservationLikelihood
            from .likelihood.eline_marginal import eline_line_fluxes
            model = self.model
            lh = MultiObservationLikelihood(
                keys=tuple(model.obs_dict),
                likelihoods=tuple(_likelihood_for(o, model.param_names, model=model)
                                  for o in model.observations))

            def one(th):
                pred, aux = model.predict_with_elines(th)
                post = eline_line_fluxes(model, th, lh)
                out = dict(pred)
                for k, A in aux["cols"].items():
                    out[k] = pred[k] + (A @ post["mean"]).astype(pred[k].dtype)
                out["__elines_mean"] = post["mean"]
                out["__elines_sd"] = post["sd"]
                out["__elines_cloudy"] = post["cloudy"]
                return out
            self._f_elines = jax.jit(jax.vmap(one))
        return self._f_elines

    def _run_batches(self, theta_batch: dict) -> dict:
        n = next(iter(theta_batch.values())).shape[0]
        want_pred = self.want["predictions"] and self.model.observations
        spec, sfh, pred = [], [], []
        pending = None
        for a in range(0, n, self.batch_size):
            b = min(a + self.batch_size, n)
            tb = {k: v[a:b] for k, v in theta_batch.items()}
            current = (self._f_spectra(tb), self._f_sfh(tb),
                       self._predictor()(tb) if want_pred else None)
            if pending is not None:
                self._collect(pending, spec, sfh, pred)
            pending = current
        self._collect(pending, spec, sfh, pred)
        cat = lambda parts: {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in parts[0]}
        return {"spectra": cat(spec), "sfh": cat(sfh), "pred": cat(pred) if pred else {}}

    @staticmethod
    def _collect(batch, spec, sfh, pred):
        s, f, p = jax.device_get(batch)
        spec.append(s)
        sfh.append(f)
        if p is not None:
            pred.append(p)

    def _derive(self, theta_batch: dict, raw: dict) -> dict:
        """All output blocks except 'theta'."""
        m, csp = self.model, self.csp
        n = next(iter(theta_batch.values())).shape[0]
        wave = np.asarray(csp.wave, dtype=float)
        full, intr, dfree = (np.asarray(raw["spectra"][k], dtype=float) for k in ("full", "intrinsic", "dustfree"))
        T_yr = np.asarray(raw["sfh"]["T_yr"], dtype=float)
        sfh = np.asarray(raw["sfh"]["sfh"], dtype=float)              # (n, n_time) or (n, n_time-1)
        n_time = T_yr.shape[1]
        interp = csp.sfh_interp
        theta_np = {k: np.asarray(v) for k, v in theta_batch.items()}
        logmass = theta_np["logmass"].reshape(n) if "logmass" in theta_np else None
        mass_scale = 10.0 ** logmass if logmass is not None else np.ones(n)

        if "zred" in raw["spectra"]:
            zred = np.asarray(raw["spectra"]["zred"], dtype=float).reshape(n)
        else:
            zred = np.full(n, float(m.zred))

        out: dict = {"extras": {}, "prediction": {}, "derived": {}}

        bars = np.empty((n, n_time - 1)); nodes = np.empty((n, n_time))
        for i in range(n):
            bars[i], nodes[i] = _per_bin_and_nodes(sfh[i], n_time)
        bars *= mass_scale[:, None]; nodes *= mass_scale[:, None]
        mass_formed = np.array([_formed_mass(T_yr[i], bars[i], nodes[i], interp) for i in range(n)])
        sfr_native = (nodes if sfh.shape[1] == n_time else bars)
        blk = {"lookback_gyr": T_yr / 1e9, "sfr": sfr_native, "sfr_per_bin": bars,
               "mass_formed": mass_formed, "sfh_interp": interp}
        if self.want["sfr"]:
            for w in self.windows_myr:
                key = f"sfr{w:g}"
                blk[key] = np.array([_mean_sfr_window(T_yr[i], bars[i], nodes[i], w * 1e6, interp)
                                     for i in range(n)])
                if self.want["ssfr"]:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        blk[f"ssfr{w:g}"] = blk[key] / mass_formed
        out["extras"]["sfh"] = blk

        win = (wave >= _UV_WINDOW_AA[0]) & (wave <= _UV_WINDOW_AA[1])
        if self.want["uv"]:
            if win.sum() < 2:
                warnings.warn("model grid has fewer than two pixels in 1450-1550 A rest; UV block skipped")
            else:
                ww = wave[win]; span = ww[-1] - ww[0]
                def _luv(spec):
                    return _trapz(spec[:, win], ww, axis=1) / span * _LSUN_ERG_S
                def _muv(luv):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return -2.5 * np.log10(luv / (4.0 * np.pi * (10.0 * _PC_CM) ** 2) / _AB_ZERO_FNU)
                luv, luv_i = _luv(full), _luv(intr)
                out["extras"]["uv"] = {"LUV": luv, "MUV": _muv(luv),
                                       "LUV_intrinsic": luv_i, "MUV_intrinsic": _muv(luv_i)}

        if self.want["ionizing"]:
            ion = wave < _LYMAN_LIMIT_AA
            if ion.sum() < 2:
                warnings.warn("model grid has fewer than two pixels below 912 A; ionizing block skipped")
            else:
                qq = _trapz(intr[:, ion] / wave[ion], wave[ion], axis=1)
                nion = qq * _LSUN_ERG_S / _HPLANCK_ERG_S
                blk = {"nion": nion}
                if "uv" in out["extras"]:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        blk["xion"] = nion / out["extras"]["uv"]["LUV_intrinsic"]
                mt = self._model_theta({k: v[0] for k, v in theta_batch.items()})
                if "frac_obrun" in theta_np:
                    blk["fesc"] = theta_np["frac_obrun"].reshape(n)
                elif "frac_obrun" in mt:
                    blk["fesc"] = np.full(n, float(np.ravel(np.asarray(mt["frac_obrun"]))[0]))
                out["extras"]["ionizing"] = blk

        pred = {"wave_rest": wave, "spectra_model": full.astype(np.float32),
                "spectra_intrinsic": intr.astype(np.float32),
                "spectra_dustfree": dfree.astype(np.float32),
                "photometry": {}, "spectra": {}, "lines": {}}
        if "observed" in raw["spectra"]:
            pred["spectra_observed"] = np.asarray(raw["spectra"]["observed"]).astype(np.float32)   # cgs f_nu
            pred["zred"] = np.asarray(raw["spectra"]["zred"], dtype=float).reshape(n)
        if raw["pred"]:
            for obs in m.observations:
                kind = getattr(obs, "_kind", "observation")
                slot = {"photometry": "photometry", "spectrum": "spectra", "lines": "lines"}.get(kind)
                if slot is None:
                    continue
                pred[slot][obs.name] = np.asarray(raw["pred"][obs.name], dtype=float)
        out["prediction"] = pred
        es = getattr(m, "_eline_system", None)
        if es is not None and raw["pred"]:
            out["extras"]["elines"] = {
                "names": list(es.names), "wave_rest": np.asarray(es.wave_rest),
                "mean": np.asarray(raw["pred"]["__elines_mean"], dtype=float),
                "sd": np.asarray(raw["pred"]["__elines_sd"], dtype=float),
                "cloudy": np.asarray(raw["pred"]["__elines_cloudy"], dtype=float)}

        tot = self._logzsol_total_batch(theta_np, n)
        if tot is not None:
            out["extras"]["metallicity"] = {"logzsol_total": tot}

        if self.derived:
            samples = []
            for i in range(n):
                th = self._model_theta_np(theta_np, i)
                samples.append(SpectrumSample(
                    wave_rest=wave, full=full[i], intrinsic=intr[i], dustfree=dfree[i],
                    theta=th, zred=float(zred[i]),
                    logmass=(None if logmass is None else float(logmass[i])),
                    sfr=sfr_native[i], lookback_gyr=T_yr[i] / 1e9, cosmo=csp.cosmo))
            for name, fn in self.derived.items():
                vals = []
                for i, s in enumerate(samples):
                    v = np.asarray(fn(s), dtype=float)
                    if v.ndim > 1:
                        raise ValueError(f"derived[{name!r}] must return a scalar or 1-D array, got shape {v.shape}")
                    vals.append(v)
                shapes = {v.shape for v in vals}
                if len(shapes) != 1:
                    raise ValueError(f"derived[{name!r}] returned inconsistent shapes {shapes}")
                out["derived"][name] = np.stack(vals) if vals[0].ndim else np.array(vals)
        return out

    def _logzsol_total_batch(self, theta_np: dict, n: int):
        """[Z/H] = logzsol + f([alpha/Fe]) per draw on an alpha grid, else None."""
        csp = self.csp
        if not hasattr(csp, "logzsol_total") or int(getattr(csp, "_n_afe", 1)) == 1:
            return None
        vals = []
        for i in range(n):
            th = self._model_theta_np(theta_np, i)
            vals.append(np.asarray(csp.logzsol_total(th), dtype=float).reshape(-1))
        return np.stack(vals)

    def _model_theta_np(self, theta_np: dict, i: int) -> dict:
        if not self.model.transforms:
            one = {k: v[i] for k, v in theta_np.items()}
            return {k: np.asarray(v) for k, v in self._model_theta(one).items()}
        one = {k: jnp.asarray(v[i]) for k, v in theta_np.items()}
        return {k: np.asarray(v) for k, v in self._model_theta(one).items()}

    def run(self) -> dict:
        """Compute and return (and store in ``self.output``) the nested dict of the module docstring."""
        self._compile()
        idx = self._draw_indices()
        theta_batch = self._theta_batch(idx)
        raw = self._run_batches(theta_batch)
        out = self._derive(theta_batch, raw)
        out["theta"] = {k: np.asarray(v).reshape((v.shape[0], -1)).squeeze(axis=-1)
                        if np.asarray(v).shape[1:] == (1,) else np.asarray(v)
                        for k, v in theta_batch.items()}
        out["draw_index"] = idx
        out["log_likelihood"] = np.asarray(self.result.log_likelihoods, dtype=float)[idx]

        ll = np.asarray(self.result.log_likelihoods, dtype=float)
        ibest = int(np.nanargmax(ll))
        tb = self._theta_batch(np.array([ibest]))
        rb = self._run_batches(tb)
        best = self._derive(tb, rb)
        best["theta"] = {k: np.asarray(v)[0] for k, v in tb.items()}
        best["index"] = ibest
        best["log_likelihood"] = float(ll[ibest])
        best = _squeeze_leading(best)
        out["bestfit"] = best

        out["meta"] = {
            "n_samples": int(idx.size), "n_raw": int(self.n_raw), "seed": self.seed,
            "resampled": not self._uniform_weights(),
            "windows_myr": list(self.windows_myr), "param_names": list(self.model.param_names),
            "zred_fixed": float(self.model.zred), "cosmology": self.csp.cosmo.to_dict(),
            "sampler": getattr(self.result, "sampler_name", ""),
            "weights": self._weights_source,
            "log_evidence": float(getattr(self.result, "log_evidence", float("nan"))),
            "log_evidence_err": float(getattr(self.result, "log_evidence_err", float("nan"))),
            "observations": [(o.name, getattr(o, "_kind", "")) for o in self.model.observations],
            "sfh_interp": self.csp.sfh_interp, "sfh_per_bin": bool(getattr(self.csp, "sfh_per_bin", False)),
            "metallicity": self._metallicity_meta(),
        }
        self.output = out
        return out

    def _metallicity_meta(self) -> dict:
        """Convention and grid Z_sun behind every metallicity in this output."""
        csp = self.csp
        if not hasattr(csp, "log10_zsun"):
            return {}
        return {"convention": "logzsol = log10(Z/Z_sun)",
                "log10_zsun": float(csp.log10_zsun),
                "zsun_nominal": (float(csp.zsun_nominal)
                                 if getattr(csp, "zsun_nominal", None) is not None else None),
                "axis_meaning": getattr(csp, "axis_meaning", None),
                "zsun_source": getattr(csp, "zsun_source", None),
                "grid_chash": getattr(csp, "grid_chash", None),
                "gas_tied": bool(getattr(csp, "gas_tied", False))}

    def figures(self, outdir, *, prefix="", title=None, truths=None, fmt="pdf"):
        """Write the summary, corner and sampling-diagnostic figures
        (:mod:`ceridwen.plotting`) to ``outdir``; returns their paths.
        ``truths`` ({name: value}) marks injected values in mock tests."""
        from .plotting import make_figures
        if self.output is None:
            self.run()
        return make_figures(self.output, self.model, self.result, outdir,
                            prefix=prefix, title=title, truths=truths, fmt=fmt)

    def save(self, path) -> Path:
        """Write ``self.output`` (running first if needed) as a flat ``.npz`` with '/'-joined keys."""
        if self.output is None:
            self.run()
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        flat = {}
        _flatten(self.output, "", flat)
        np.savez(path, **flat)
        return path


def _squeeze_leading(d):
    if isinstance(d, dict):
        return {k: _squeeze_leading(v) for k, v in d.items()}
    a = np.asarray(d)
    return a[0] if a.ndim >= 1 and a.shape[0] == 1 and not isinstance(d, (str, float, int)) else d


def _flatten(d, prefix, flat):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, key + "/", flat)
        elif isinstance(v, (str, bool, int, float)) or v is None or isinstance(v, (list, tuple)):
            flat[key] = np.array(json.dumps(v))
        else:
            flat[key] = np.asarray(v)


def load_postprocess(path) -> dict:
    """Rebuild the nested dict written by :meth:`PostProcess.save`."""
    out: dict = {}
    with np.load(path, allow_pickle=False) as f:
        for key in f.files:
            parts = key.split("/")
            d = out
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            v = f[key]
            if v.dtype.kind in ("U", "S"):
                try:
                    v = json.loads(str(v))
                except json.JSONDecodeError:
                    v = str(v)
            d[parts[-1]] = v
    return out
