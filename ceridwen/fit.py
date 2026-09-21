"""
``fitSED``: fit a configured ``SedModel`` to observations and store the posterior in HDF5,
plus readers for the result file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger("ceridwen")

Array = jax.Array


def fitSED(
    model,
    observations: Sequence | None = None,
    output_dir: str | Path = ".",
    *,
    sampler: str = "nested",
    rng_key: Array | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
    vi: "str | Any | None" = None,
    vi_kwargs: dict[str, Any] | None = None,
    filename: str = "ceridwen_result.h5",
    overwrite: bool = True,
    verbose: bool = True,
):
    """Fit ``model`` to ``observations`` with ``sampler`` ("nested" or "nuts") and return the
    ``SamplingResult``; writes ``output_dir/filename`` (HDF5) and a ``.log`` with the same stem.

    Parameters
    ----------
    observations : list[Observation] -- replaces ``model.observations`` and re-runs ``setup_for_model`` at ``model.zred``
    sampler_kwargs : dict -- forwarded to the sampler adapter constructor
    vi : None, 'tril', 'iaf', or a VI map -- NUTS-only variational preconditioning
    vi_kwargs : dict -- forwarded to VI training / map constructor
    """
    from .likelihood.likelihood import MultiObservationLikelihood
    from .sampler.runner import run_sampler

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    if sampler_kwargs is None:
        sampler_kwargs = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists.  Pass overwrite=True to replace."
        )

    if observations is not None:
        model.observations = list(observations)
        model.setup_observations()

    if not model.observations:
        raise ValueError("No observations attached to model.")

    _t0_likelihood = time.perf_counter()
    obs_dict = model.obs_dict
    keys = tuple(obs_dict.keys())
    likelihoods = tuple(_likelihood_for(obs_dict[k], model.param_names) for k in keys)
    multi_likelihood = MultiObservationLikelihood(
        keys=keys,
        likelihoods=likelihoods,
    )
    _t_likelihood = time.perf_counter() - _t0_likelihood

    logger.setLevel(logging.INFO)
    logger.propagate = False
    if verbose and not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_handler)

    log_path = output_path.with_suffix(".log")
    _file_handler = logging.FileHandler(log_path, mode="w")
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_file_handler)

    try:
        _devices = jax.devices()
        _backend = jax.default_backend().upper()
        _device_str = ", ".join(str(d) for d in _devices)

        logger.info(f"ceridwen.fitSED")
        logger.info(f"  Device      : {_backend}  ({_device_str})")
        if _backend == "CPU":
            logger.info(
                "                ^ running on CPU — for GPU check that a "
                "CUDA-enabled jaxlib is installed and JAX_PLATFORMS is unset"
            )
        logger.info(f"  Sampler     : {sampler}")
        logger.info(f"  Cosmology   : {model.cosmo.describe()}")
        logger.info(f"  Redshift    : {model._redshift_line()}")
        logger.info(f"  Kinematics  : {model._kinematics_line()}")
        logger.info(f"  Parameters  : {model.param_names}  ({sum(int(jnp.size(v)) for v in model.theta_init.values())} dims)")
        logger.info(f"  Observations: {list(keys)}")
        for k, lh in zip(keys, likelihoods):
            logger.info(f"    {k}: {_describe_likelihood(obs_dict[k], lh)}")
        logger.info(f"  Output      : {output_path}")
        logger.info(f"  Log         : {log_path}")
        logger.info(f"  Likelihood build: {_t_likelihood:.3f} s")

        _t0_adapter = time.perf_counter()
        adapter = _build_adapter(
            sampler, model, sampler_kwargs, verbose,
            vi=vi, vi_kwargs=vi_kwargs,
        )
        _t_adapter = time.perf_counter() - _t0_adapter
        logger.info(f"  Adapter build:    {_t_adapter:.3f} s")
        logger.info(f"  Sampler settings: {_describe_adapter(adapter, model)}")

        _t0_sampler = time.perf_counter()
        result = run_sampler(model, multi_likelihood, adapter, rng_key)
        _t_sampler = time.perf_counter() - _t0_sampler

        logger.info(f"\n{result.summary()}")

        _t0_h5 = time.perf_counter()
        write_result_h5(output_path, model, result, verbose=verbose)
        _t_h5 = time.perf_counter() - _t0_h5

        _t_total = _t_likelihood + _t_adapter + _t_sampler + _t_h5
        logger.info(f"\n  fitSED timing breakdown:")
        logger.info(f"    Likelihood build : {_t_likelihood:>8.3f} s")
        logger.info(f"    Adapter build    : {_t_adapter:>8.3f} s")
        logger.info(f"    Sampler run      : {_t_sampler:>8.1f} s")
        logger.info(f"    HDF5 write       : {_t_h5:>8.3f} s")
        logger.info(f"    Total fitSED     : {_t_total:>8.1f} s")

        return result
    finally:
        logger.removeHandler(_file_handler)
        _file_handler.close()


def _build_adapter(sampler: str, model, sampler_kwargs: dict, verbose: bool,
                   vi=None, vi_kwargs=None):
    sampler = sampler.lower().strip()

    if sampler in ("nested", "ns", "nss"):
        if vi is not None:
            raise ValueError(
                "vi=... is only supported with sampler='nuts'."
            )
        from .sampler.nested import BlackJAXNestedSamplerAdapter
        defaults = dict(
            priors=model.priors,
            num_live=500,
            num_inner_steps=30,
            verbose=verbose,
        )
        defaults.update(sampler_kwargs)
        return BlackJAXNestedSamplerAdapter(**defaults)

    elif sampler in ("nuts", "hmc"):
        from .sampler.nuts import BlackJAXNUTSAdapter

        bounds = sampler_kwargs.pop("bounds", None)
        if bounds is None:
            bounds = _detect_bounds(model)
            if verbose and bounds:
                logger.info(f"  Auto-detected bounded priors: {bounds}")

        defaults = dict(
            num_samples=2000,
            num_chains=4,
            target_acceptance=0.95,
            bounds=bounds,
            verbose=verbose,
            vi=vi,
            vi_kwargs=vi_kwargs,
        )
        defaults.update(sampler_kwargs)
        return BlackJAXNUTSAdapter(**defaults)

    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'.  Choose 'nested' or 'nuts'."
        )


def _likelihood_for(obs, param_names=()):
    """Diagonal Gaussian likelihood for ``obs`` (one-sided kernel when it flags upper limits);
    the noise nuisance terms ``log_jitter`` / ``log_f_calib`` / ``log_f_data`` / ``log_err_scale``
    are switched on when the model samples them (one value shared by every observation)."""
    from .likelihood.likelihood import (
        DiagonalGaussianLikelihood, DiagonalGaussianLikelihoodWithUpperLimits)
    from .likelihood.noise_model import DiagonalNoiseModel
    if getattr(obs, "logify_spectrum", False):
        raise NotImplementedError(
            f"Spectrum {obs.name!r}: logify_spectrum=True is not available in the "
            "sampled likelihood")
    if getattr(obs, "noise", None) is not None:
        raise NotImplementedError(
            f"Spectrum {obs.name!r}: a GaussianProcess noise model is not available "
            "in the sampled likelihood (host-side Cholesky); use noise_floor= for a "
            "diagonal floor")
    nm = DiagonalNoiseModel(noise_floor=float(getattr(obs, "noise_floor", 0.0) or 0.0),
                            use_jitter="log_jitter" in param_names,
                            use_fractional="log_f_calib" in param_names,
                            use_data_fractional="log_f_data" in param_names,
                            use_error_scale="log_err_scale" in param_names)
    ul = getattr(obs, "upper_limit", None)
    if ul is not None and bool(jnp.any(ul)):
        return DiagonalGaussianLikelihoodWithUpperLimits(noise_model=nm)
    return DiagonalGaussianLikelihood(noise_model=nm)


def _describe_likelihood(obs, lh) -> str:
    bits = [type(lh).__name__]
    nf = getattr(lh.noise_model, "noise_floor", 0.0)
    if nf:
        bits.append(f"noise_floor={nf:g} x model")
    names = getattr(lh.noise_model, "nuisance_param_names", ())
    if names:
        bits.append("sampled noise terms " + ", ".join(names))
    if getattr(obs, "sky", None) is not None:
        bits.append("sky subtracted")
    if getattr(obs, "calibration", None) is not None:
        bits.append("fixed calibration vector on the model")
    ul = getattr(obs, "upper_limit", None)
    if ul is not None and bool(jnp.any(ul)):
        bits.append(f"{int(jnp.sum(ul))} upper limits")
    return ", ".join(bits)


def _describe_adapter(adapter, model) -> str:
    cls = type(adapter).__name__
    if cls == "BlackJAXNestedSamplerAdapter":
        n_dims = sum(int(jnp.size(v)) for v in model.theta_init.values())
        inner = adapter._num_inner_steps if adapter._num_inner_steps is not None else n_dims * 5
        ndel = adapter._num_delete if adapter._num_delete is not None else max(1, adapter.num_live // 5)
        return (f"nested: num_live={adapter.num_live}, num_inner_steps={inner}, "
                f"num_delete={ndel}, logZ_tol={adapter.logZ_tol:g}")
    keys = ("num_warmup", "num_samples", "num_chains", "target_acceptance")
    return cls + ": " + ", ".join(f"{k}={getattr(adapter, k)}" for k in keys if hasattr(adapter, k))


def _detect_bounds(model) -> dict[str, tuple[float, float]]:
    """(low, high) bounds of Uniform/TopHat/ClippedNormal priors, keyed by parameter name."""
    bounds = {}
    for name, prior in model.priors.items():
        cls_name = type(prior).__name__
        if cls_name in ("Uniform", "TopHat"):
            lo = float(prior.params["low"])
            hi = float(prior.params["high"])
            bounds[name] = (lo, hi)
        elif cls_name == "ClippedNormal":
            if "low" in prior.params and "high" in prior.params:
                lo = float(prior.params["low"])
                hi = float(prior.params["high"])
                bounds[name] = (lo, hi)
    return bounds


def write_result_h5(
    path: str | Path,
    model,
    result,
    verbose: bool = True,
):
    """Write ``result`` and the model/observation metadata to HDF5 ``path``::

        /obs/<obs_name>/
            flux           (n_data,)
            uncertainty    (n_data,)
            wavelength     (n_data,)
            mask           (n_data,)  bool
            attrs: type, name, [instrument_kind, subtract_library, filternames, ...]

        /model/
            param_names    (n_params,)  variable-length string
            wave           (n_wave,)
            theta_init/<param_name>    (shape,)
            priors/  attrs: <param_name> -> JSON string
            attrs: zred, kinematics_*, broaden_photometry, cosmo_*, transforms (JSON list of "derived <- fn")

        /samples/
            <param_name>       (n_samples, *shape)
            log_likelihoods    (n_samples,)
            log_weights        (n_samples,)
            attrs: log_evidence, log_evidence_err, sampler_name,
                   wall_time_s, n_likelihood_calls
    """
    import h5py

    path = Path(path)

    with h5py.File(path, "w") as f:
        obs_grp = f.create_group("obs")
        for obs in model.observations:
            og = obs_grp.create_group(obs.name)
            og.create_dataset("flux", data=np.asarray(obs.flux))
            og.create_dataset("uncertainty", data=np.asarray(obs.uncertainty))
            og.create_dataset("wavelength", data=np.asarray(obs.wavelength))
            og.create_dataset("mask", data=np.asarray(obs.mask))
            og.attrs["type"] = type(obs).__name__
            og.attrs["name"] = obs.name

            ins = getattr(obs, "instrument", None)
            if ins is not None:
                og.attrs["instrument_kind"] = str(ins.kind)
                og.create_dataset("instrument_value", data=np.asarray(ins.value))
                if ins.wave is not None:
                    og.create_dataset("instrument_wave", data=np.asarray(ins.wave))
                og.attrs["subtract_library"] = bool(obs.subtract_library)
                proj = getattr(obs, "_proj", None)
                if proj is not None and proj.free_z:
                    og.attrs["zred_range"] = np.asarray(proj.zred_range, dtype=float)
                    og.attrs["zred_ref"] = float(proj.opz_ref - 1.0)
            if hasattr(obs, "filternames"):
                og.attrs["filternames"] = json.dumps(obs.filternames)
            if hasattr(obs, "line_names") and obs.line_names is not None:
                og.attrs["line_names"] = json.dumps(list(obs.line_names))
            if hasattr(obs, "line_ind") and obs.line_ind is not None:
                og.create_dataset("line_ind", data=np.asarray(obs.line_ind))

        mod_grp = f.create_group("model")

        dt = h5py.string_dtype()
        canonical_names = (
            list(result.param_names)
            if getattr(result, "param_names", None)
            else list(model.param_names)
        )
        mod_grp.create_dataset(
            "param_names",
            data=np.array(canonical_names, dtype=object),
            dtype=dt,
        )

        mod_grp.create_dataset("wave", data=np.asarray(model.csp.wave))

        init_grp = mod_grp.create_group("theta_init")
        for name, val in model.theta_init.items():
            init_grp.create_dataset(name, data=np.asarray(val))

        prior_grp = mod_grp.create_group("priors")
        for name, prior in model.priors.items():
            if hasattr(prior, "serialize"):
                prior_grp.attrs[name] = json.dumps(prior.serialize())
            else:
                prior_grp.attrs[name] = repr(prior)

        if model.transforms:
            tx_names = []
            for derived, fn in model.transforms.items():
                fn_name = getattr(fn, "__name__", repr(fn))
                tx_names.append(f"{derived} <- {fn_name}")
            mod_grp.attrs["transforms"] = json.dumps(tx_names)

        mod_grp.attrs["zred"] = float(getattr(model, "zred", 0.0))
        mod_grp.attrs["zred_free"] = bool(getattr(model, "zred_is_free", False))
        kin = getattr(model, "kinematics", None)
        if kin is not None:
            mod_grp.attrs["kinematics_sigma_gal"] = str(kin.sigma_gal)
            mod_grp.attrs["kinematics_sigma_gas"] = str(kin.effective_sigma_gas)
            mod_grp.attrs["kinematics_sigma_max"] = float(kin.sigma_max)
            mod_grp.attrs["broaden_photometry"] = bool(model.broaden_photometry)
        if getattr(model, "lumdist_mpc", None) is not None:
            mod_grp.attrs["lumdist_mpc"] = float(model.lumdist_mpc)
        for k, v in model.cosmo.to_dict().items():
            mod_grp.attrs[f"cosmo_{k}"] = v
        mod_grp.attrs["n_time"] = int(model.csp.n_time) if hasattr(model.csp, "n_time") else -1
        mod_grp.attrs["n_ssp_ages"] = int(model.csp.ages.shape[0]) if hasattr(model.csp, "ages") else -1
        mod_grp.attrs["n_metallicities"] = int(model.csp.zmet.shape[0]) if hasattr(model.csp, "zmet") else -1

        samp_grp = f.create_group("samples")

        for name in result.param_names:
            arr = np.asarray(result.samples[name])
            samp_grp.create_dataset(name, data=arr, compression="gzip")

        samp_grp.create_dataset(
            "log_likelihoods",
            data=np.asarray(result.log_likelihoods),
            compression="gzip",
        )
        samp_grp.create_dataset(
            "log_weights",
            data=np.asarray(result.log_weights),
            compression="gzip",
        )

        if result.log_likelihoods_birth is not None:
            samp_grp.create_dataset(
                "log_likelihoods_birth",
                data=np.asarray(result.log_likelihoods_birth),
                compression="gzip",
            )

        samp_grp.attrs["log_evidence"] = float(result.log_evidence)
        samp_grp.attrs["log_evidence_err"] = float(result.log_evidence_err)
        samp_grp.attrs["sampler_name"] = result.sampler_name
        samp_grp.attrs["wall_time_s"] = result.wall_time_s
        samp_grp.attrs["n_likelihood_calls"] = result.n_likelihood_calls
        samp_grp.attrs["n_samples"] = int(result.log_likelihoods.shape[0])
        if isinstance(result.raw, dict):
            if "num_chains" in result.raw:
                samp_grp.attrs["num_chains"] = int(result.raw["num_chains"])
            if "num_samples" in result.raw:
                samp_grp.attrs["num_samples_per_chain"] = int(
                    result.raw["num_samples"]
                )
            if "num_warmup" in result.raw:
                samp_grp.attrs["num_warmup"] = int(result.raw["num_warmup"])

    if verbose:
        size_mb = path.stat().st_size / 1024**2
        logger.info(f"  Wrote {path}  ({size_mb:.1f} MB)")


def load_result_h5(path: str | Path):
    """Rebuild a ``SamplingResult`` from a result HDF5 file (the forward model itself is not restored)."""
    import h5py
    from .sampler.runner import SamplingResult

    path = Path(path)
    with h5py.File(path, "r") as f:
        samp = f["samples"]
        param_names = list(f["model"]["param_names"].asstr()[()])

        samples = {p: jnp.asarray(np.array(samp[p])) for p in param_names}
        log_likelihoods = jnp.asarray(np.array(samp["log_likelihoods"]))
        log_weights     = jnp.asarray(np.array(samp["log_weights"]))
        llb = (jnp.asarray(np.array(samp["log_likelihoods_birth"]))
               if "log_likelihoods_birth" in samp else None)

        raw = {}
        if "num_chains" in samp.attrs:
            raw["num_chains"] = int(samp.attrs["num_chains"])
        if "num_samples_per_chain" in samp.attrs:
            raw["num_samples"] = int(samp.attrs["num_samples_per_chain"])
        if "num_warmup" in samp.attrs:
            raw["num_warmup"] = int(samp.attrs["num_warmup"])

        return SamplingResult(
            samples               = samples,
            log_evidence          = float(samp.attrs.get("log_evidence", float("nan"))),
            log_evidence_err      = float(samp.attrs.get("log_evidence_err", float("nan"))),
            log_weights           = log_weights,
            log_likelihoods       = log_likelihoods,
            param_names           = param_names,
            n_likelihood_calls    = int(samp.attrs.get("n_likelihood_calls", -1)),
            wall_time_s           = float(samp.attrs.get("wall_time_s", float("nan"))),
            sampler_name          = str(samp.attrs.get("sampler_name", "unknown")),
            log_likelihoods_birth = llb,
            raw                   = raw or None,
        )


def result_cosmology(path: str | Path):
    """The ``Cosmology`` a result file was fitted with (``/model`` attrs ``cosmo_*``); KeyError if absent."""
    import h5py
    from .cosmology import Cosmology

    with h5py.File(Path(path), "r") as f:
        attrs = dict(f["model"].attrs)
    if "cosmo_H0" not in attrs:
        raise KeyError(f"{path} carries no cosmology attributes (written before "
                       "they were recorded); rebuild with the cosmology you used")
    return Cosmology.from_dict(attrs)


def read_result_h5(path: str | Path) -> dict:
    """Read a result HDF5 file into a nested dict with keys ``'obs'``, ``'model'``, ``'samples'``."""
    import h5py

    out = {"obs": {}, "model": {}, "samples": {}}

    with h5py.File(path, "r") as f:
        for obs_name in f["obs"]:
            og = f["obs"][obs_name]
            obs_data = {k: np.array(og[k]) for k in og}
            for attr_name in og.attrs:
                obs_data[attr_name] = og.attrs[attr_name]
            out["obs"][obs_name] = obs_data

        mod = f["model"]
        out["model"]["param_names"] = list(mod["param_names"].asstr()[()])
        out["model"]["wave"] = np.array(mod["wave"])

        out["model"]["theta_init"] = {}
        for name in mod["theta_init"]:
            out["model"]["theta_init"][name] = np.array(mod["theta_init"][name])

        out["model"]["priors"] = {}
        if "priors" in mod:
            for name in mod["priors"].attrs:
                raw = mod["priors"].attrs[name]
                try:
                    out["model"]["priors"][name] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    out["model"]["priors"][name] = raw

        for attr_name in mod.attrs:
            out["model"][attr_name] = mod.attrs[attr_name]
        if "cosmo_H0" in mod.attrs:
            from .cosmology import Cosmology
            out["model"]["cosmo"] = Cosmology.from_dict(dict(mod.attrs))

        samp = f["samples"]
        for dset_name in samp:
            out["samples"][dset_name] = np.array(samp[dset_name])

        for attr_name in samp.attrs:
            out["samples"][attr_name] = samp.attrs[attr_name]

    return out
