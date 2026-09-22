"""
``fitSED``: fit a configured ``SedModel`` to observations and store the posterior in HDF5,
plus readers for the result file.
"""

from __future__ import annotations

import json
import shutil
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

    Noise terms and the outlier mixture are switched on by the names the model samples (see
    ``_likelihood_for``); all outlier fractions default to 0 (off).
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
    likelihoods = tuple(_likelihood_for(obs_dict[k], model.param_names, model=model)
                        for k in keys)
    multi_likelihood = MultiObservationLikelihood(
        keys=keys,
        likelihoods=likelihoods,
    )
    if getattr(model, "_eline_system", None) is not None:
        from .likelihood.eline_marginal import refuse_outlier_with_elines
        refuse_outlier_with_elines(keys, likelihoods, model._eline_system.keys)
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
        if getattr(model, "_eline_system", None) is not None:
            logger.info(f"  Emission lines: {model._eline_system.describe()}")
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
        write_result_h5(output_path, model, result, verbose=verbose,
                        likelihood=multi_likelihood, adapter=adapter, rng_key=rng_key)
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


# Outlier-mixture parameter names.  Prospector's template (prospect/models/templates.py,
# TemplateLibrary["outlier_model"]) has f_outlier_spec / f_outlier_phot (+ nsigma_outlier_*);
# Lines follow the same pattern.  Never shared between observations: with several
# observations of one kind each takes its own ``f_outlier_<kind>_<obs.name>``; the plain name
# is accepted only when the model has exactly one observation of that kind.
OUTLIER_KIND_SUFFIX = {"photometry": "phot", "spectrum": "spec", "lines": "lines"}


def _outlier_names(obs):
    """``((f, nsigma) plain names, (f, nsigma) per-observation names)`` of ``obs``, or None
    for a kind without an outlier model."""
    sfx = OUTLIER_KIND_SUFFIX.get(getattr(obs, "kind", None))
    if sfx is None:
        return None
    return ((f"f_outlier_{sfx}", f"nsigma_outlier_{sfx}"),
            (f"f_outlier_{sfx}_{obs.name}", f"nsigma_outlier_{sfx}_{obs.name}"))


def _constant_transform(model, name):
    """Value of the transform ``name`` if it is a constant (the fixed-parameter idiom), else
    raise: an outlier parameter derived from sampled ones is not supported."""
    fn = model.transforms[name]
    th0 = {k: jnp.asarray(v) for k, v in model.theta_init.items()}
    th1 = {k: v + 0.137 * (1.0 + jnp.abs(v)) for k, v in th0.items()}
    v0, v1 = np.asarray(fn(th0), dtype=float), np.asarray(fn(th1), dtype=float)
    if v0.size != 1 or not np.array_equal(v0, v1):
        raise NotImplementedError(
            f"the transform for {name!r} must be a constant scalar (fix it with "
            f"transforms={{{name!r}: lambda th: jnp.array([value])}}); a value derived from "
            "sampled parameters is not supported for the outlier model")
    return float(v0.ravel()[0])


def _outlier_terms(obs, param_names, model=None):
    """``(f_outlier, nsigma_outlier)`` for ``obs``'s DiagonalNoiseModel: a theta key when the
    model samples the name, a float when a constant transform fixes it, else off / 50.  The
    per-observation name ``f_outlier_<kind>_<obs.name>`` is looked up before the plain one."""
    names = _outlier_names(obs)
    if names is None:
        return None, 50.0
    transforms = getattr(model, "transforms", {}) if model is not None else {}
    given = set(param_names) | set(transforms)

    def resolve(i, default):
        name = next((n[i] for n in (names[1], names[0]) if n[i] in given), None)
        if name is None:
            return default
        return name if name in param_names else _constant_transform(model, name)

    # a fixed 0.0 is turned off by DiagonalNoiseModel itself (the ordinary Gaussian)
    return resolve(0, None), resolve(1, 50.0)


def _check_outlier_setup(model):
    """Setup-time checks of the outlier parameters (fitSED): every f_outlier_* /
    nsigma_outlier_* name must belong to one observation, the plain name only when the
    model has one observation of that kind, nsigma needs its f, priors bounded in range."""
    given = set(model.param_names) | set(getattr(model, "transforms", {}))
    wanted = {n for n in given if n.startswith(("f_outlier_", "nsigma_outlier_"))}
    if not wanted:
        return
    by_kind = {}
    for o in model.observations:
        if _outlier_names(o) is not None:
            by_kind.setdefault(o.kind, []).append(o)
    valid = {}
    for kind, obs in by_kind.items():
        for o in obs:
            plain, own = _outlier_names(o)
            for i in (0, 1):
                valid[own[i]] = o
                if len(obs) == 1:
                    valid[plain[i]] = o
            used = [n for n in (*plain, *own) if n in given]
            if len(obs) > 1 and any(n in given for n in plain):
                raise ValueError(
                    f"{[n for n in plain if n in given]} is ambiguous: the model has "
                    f"{len(obs)} {kind} observations {[x.name for x in obs]}, and each takes its "
                    f"own fraction; use {[_outlier_names(x)[1][0] for x in obs]}")
            for i in (0, 1):
                if plain[i] in given and own[i] in given:
                    raise ValueError(f"both {plain[i]!r} and {own[i]!r} are set for "
                                     f"{kind} {o.name!r}; keep one")
            f_set = plain[0] in given or own[0] in given
            n_set = plain[1] in given or own[1] in given
            if n_set and not f_set:
                raise ValueError(
                    f"{[n for n in used if n.startswith('nsigma')]} set without an "
                    f"f_outlier for {kind} {o.name!r}: the outlier width has no effect unless "
                    "the fraction is sampled or fixed")
    unknown = sorted(n for n in wanted if n not in valid)
    if unknown:
        raise ValueError(
            f"{unknown} match no observation, so they would be sampled without entering the "
            f"likelihood; the outlier names of this model are {sorted(valid)}")
    bounds = _detect_bounds(model)
    for n in sorted(wanted & set(model.param_names)):
        lo, hi = bounds.get(n, (None, None))
        if n.startswith("f_outlier_") and (lo is None or lo < 0.0 or hi > 1.0):
            raise ValueError(
                f"{n!r} needs a bounded prior inside [0, 1] (Prospector's template: "
                f"TopHat(low=1e-5, high=0.5)), got {model.priors.get(n)!r}")
        if n.startswith("nsigma_outlier_") and (lo is None or lo <= 0.0):
            raise ValueError(
                f"{n!r} needs a bounded prior with a positive lower bound, got "
                f"{model.priors.get(n)!r}")


def _likelihood_for(obs, param_names=(), model=None):
    """Diagonal Gaussian likelihood for ``obs`` (one-sided kernel when it flags upper limits);
    the noise nuisance terms ``log_jitter`` / ``log_f_calib`` / ``log_f_data`` / ``log_err_scale``
    are switched on when the model samples them (one value shared by every observation).
    The outlier mixture is switched on per observation by ``f_outlier_<kind>`` (one
    observation of that kind) or ``f_outlier_<kind>_<obs.name>``, kind = phot / spec / lines,
    with ``nsigma_outlier_*`` (default 50), sampled or, given ``model``, fixed by a constant
    transform.  Every fraction defaults to 0 (off): no name in the model, or a fixed 0, gives
    the ordinary Gaussian."""
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
    if model is not None:
        _check_outlier_setup(model)
    f_out, nsigma_out = _outlier_terms(obs, param_names, model)
    nm = DiagonalNoiseModel(noise_floor=float(getattr(obs, "noise_floor", 0.0) or 0.0),
                            use_jitter="log_jitter" in param_names,
                            use_fractional="log_f_calib" in param_names,
                            use_data_fractional="log_f_data" in param_names,
                            use_error_scale="log_err_scale" in param_names,
                            f_outlier=f_out, nsigma_outlier=nsigma_out)
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
    if not getattr(lh.noise_model, "use_outlier", False):
        bits.append("outlier mixture off")
    else:
        def _v(v):
            return f"{v} (sampled)" if isinstance(v, str) else f"{v:g} (fixed)"
        bits.append(f"outlier mixture f = {_v(lh.noise_model.f_outlier)}, "
                    f"nsigma = {_v(lh.noise_model.nsigma_outlier)}")
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
    likelihood=None,
    adapter=None,
    rng_key=None,
):
    """Write ``result`` and the model/observation metadata to HDF5 ``path``::

        /obs/<obs_name>/
            flux           (n_data,)
            uncertainty    (n_data,)
            wavelength     (n_data,)
            mask           (n_data,)  bool
            sky, calibration, upper_limit   (n_data,)  when the observation has them
            attrs: type, name, noise_floor, [instrument_kind, subtract_library, filternames, ...],
                   likelihood_json (with ``likelihood``: kernel class and noise-model settings)

        /provenance/        attrs: ceridwen_version, ceridwen_githash (build stamp),
                            git_head / git_dirty (live, when the package is a git checkout),
                            jax_version, written_utc, sampler_json (the adapter's settings
                            actually used, with ``adapter``); dataset rng_key (with ``rng_key``)

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

        /elines/            (only with Spectrum(marginalize_elines=True) and ``likelihood``)
            names          (m,)  FSPS line names
            wave_rest      (m,)  vacuum rest wavelengths [A]
            mean, sd       (n_samples, m)  posterior line fluxes per draw [erg s^-1 cm^-2]
            cloudy         (n_samples, m)  the CLOUDY (grid) fluxes per draw
            attrs: spectrum, observations (JSON), prior_width, not_fitted, ignored (JSON)
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
            for extra in ("sky", "calibration", "upper_limit"):
                val = getattr(obs, extra, None)
                if val is not None:
                    og.create_dataset(extra, data=np.asarray(val))
            og.attrs["noise_floor"] = float(getattr(obs, "noise_floor", 0.0) or 0.0)
            if likelihood is not None and obs.name in tuple(likelihood.keys):
                lh = likelihood.likelihoods[tuple(likelihood.keys).index(obs.name)]
                og.attrs["likelihood_json"] = json.dumps(_likelihood_config(lh))

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
        _write_metallicity_provenance(mod_grp, model)

        _write_run_provenance(f, adapter, rng_key, model)

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

    if likelihood is not None and getattr(model, "_eline_system", None) is not None:
        _write_eline_group(path, model, result, likelihood)

    if verbose:
        size_mb = path.stat().st_size / 1024**2
        logger.info(f"  Wrote {path}  ({size_mb:.1f} MB)")


def _likelihood_config(lh) -> dict:
    """JSON-able description of an observation's likelihood: kernel class + noise model."""
    import dataclasses
    nm = getattr(lh, "noise_model", None)
    cfg = {"class": type(lh).__name__}
    if nm is not None:
        cfg["noise_model"] = {"class": type(nm).__name__}
        if dataclasses.is_dataclass(nm):
            cfg["noise_model"].update(
                {fld.name: getattr(nm, fld.name) for fld in dataclasses.fields(nm)})
        cfg["noise_model"]["sampled_parameters"] = list(
            getattr(nm, "nuisance_param_names", ())) + list(getattr(nm, "outlier_param_names", ()))
    return cfg


def _git_state(path) -> tuple:
    """``(head, dirty)`` of the git checkout containing ``path``, or (None, None)."""
    import os
    import subprocess
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], env=env,
                              capture_output=True, text=True, timeout=10)
        if head.returncode:
            return None, None
        st = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                             "--untracked-files=no"], env=env,
                            capture_output=True, text=True, timeout=10)
        return head.stdout.strip(), (bool(st.stdout.strip()) if st.returncode == 0 else None)
    except (OSError, subprocess.SubprocessError):
        return None, None


def adapter_settings(adapter, model=None) -> dict:
    """The sampler settings an adapter actually runs with (defaults resolved as the adapter
    resolves them; ``num_inner_steps`` needs ``model`` for its 5 x n_dims default)."""
    if adapter is None:
        return {}
    cls = type(adapter).__name__
    out = {"adapter": cls}
    if cls == "BlackJAXNestedSamplerAdapter":
        inner = adapter._num_inner_steps
        if inner is None and model is not None:
            inner = 5 * sum(int(jnp.size(v)) for v in model.theta_init.values())
        out.update(num_live=adapter.num_live,
                   num_inner_steps=inner,
                   num_delete=(adapter._num_delete if adapter._num_delete is not None
                               else max(1, adapter.num_live // 5)),
                   logZ_tol=adapter.logZ_tol,
                   checkpoint_interval_s=adapter.checkpoint_interval_s)
        return out
    for k in ("num_warmup", "num_samples", "num_chains", "target_acceptance",
              "initial_step_size", "max_num_doublings", "dense_mass", "bounds"):
        if hasattr(adapter, k):
            v = getattr(adapter, k)
            out[k] = ({n: list(b) for n, b in v.items()} if isinstance(v, dict) else v)
    if hasattr(adapter, "vi"):
        vi = adapter.vi
        out["vi"] = vi if (vi is None or isinstance(vi, str)) else type(vi).__name__
        out["vi_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None
                                else repr(v)) for k, v in getattr(adapter, "vi_kwargs", {}).items()}
    return out


def _write_run_provenance(f, adapter, rng_key, model=None) -> None:
    import datetime
    import ceridwen
    g = f.create_group("provenance")
    g.attrs["ceridwen_version"] = str(ceridwen.__version__)
    g.attrs["ceridwen_githash"] = str(getattr(ceridwen, "__githash__", None))
    g.attrs["ceridwen_githash_note"] = ("build stamp (_buildstamp.py): stale in an editable "
                                        "install; git_head is the checkout's live HEAD")
    head, dirty = _git_state(Path(ceridwen.__file__).resolve().parent)
    g.attrs["git_head"] = str(head) if head is not None else "unavailable (not a git checkout)"
    if dirty is not None:
        g.attrs["git_dirty"] = bool(dirty)
    g.attrs["jax_version"] = str(jax.__version__)
    g.attrs["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    g.attrs["sampler_json"] = json.dumps(adapter_settings(adapter, model))
    if rng_key is not None:
        key = rng_key
        if jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key):
            key = jax.random.key_data(key)
        g.create_dataset("rng_key", data=np.asarray(key))


def read_provenance(path) -> dict:
    """The ``/provenance`` group of a result file (JSON attrs decoded); {} for older files."""
    import h5py
    with h5py.File(Path(path), "r") as f:
        return _provenance_from_group(f["provenance"]) if "provenance" in f else {}


def _provenance_from_group(g) -> dict:
    out = {}
    for k, v in g.attrs.items():
        out[k[:-5] if k.endswith("_json") else k] = json.loads(v) if k.endswith("_json") else v
    for k in g:
        out[k] = np.array(g[k])
    return out


METALLICITY_CONVENTION = "logzsol"


def _write_metallicity_provenance(mod_grp, model) -> None:
    """Record the metallicity convention, the grid's Z_sun / axis / identity and the nebular
    reference, so a result file can never be read in the wrong units."""
    csp = model.csp
    if not hasattr(csp, "log10_zsun"):
        return
    mod_grp.attrs["metallicity_convention"] = METALLICITY_CONVENTION
    mod_grp.attrs["log10_zsun"] = float(csp.log10_zsun)
    if getattr(csp, "zsun_nominal", None) is not None:
        mod_grp.attrs["zsun_nominal"] = float(csp.zsun_nominal)
    for key, val in (("metallicity_axis_meaning", getattr(csp, "axis_meaning", None)),
                     ("zsun_source", getattr(csp, "zsun_source", None)),
                     ("grid_chash", getattr(csp, "grid_chash", None))):
        if val is not None:
            mod_grp.attrs[key] = str(val)
    from .ssps.grid_metadata import CHASH_TABLE
    meta = CHASH_TABLE.get(getattr(csp, "grid_chash", None))
    if meta is not None:
        mod_grp.attrs["grid_name"] = meta.name
        mod_grp.attrs["grid_file_sha256"] = json.dumps(list(meta.file_sha256))
    mod_grp.create_dataset("metallicity_axis_native", data=np.asarray(csp.zmet_native))
    mod_grp.create_dataset("metallicity_axis_logzsol", data=np.asarray(csp.zmet))
    mod_grp.attrs["gas_tied"] = bool(getattr(csp, "gas_tied", False))
    neb = getattr(csp, "neb", None)
    if neb is not None:
        iso = str(getattr(neb, "isoc_type", "") or "")
        mod_grp.attrs["nebular_isoc_type"] = iso
        axis = np.asarray(neb.nebem_logz, dtype=float)
        mod_grp.attrs["nebular_axis_logz"] = axis
        # Byler+17: gas abundances Dopita+00 on Anders & Grevesse 1989.  The ZAU_* axis is
        # log10(Z_gas / Z_sun,neb); the nominal Z_sun,neb is identified from the NODE SET
        # itself (never from isoc_type), and omitted when the axis matches neither.
        nom, src = _nebular_zsun_from_axis(axis)
        if nom is not None:
            mod_grp.attrs["nebular_zsun_nominal"] = nom
        mod_grp.attrs["nebular_zsun_source"] = src


#: the two CLOUDY node sets ceridwen can load, as log10(Z_gas / Z_sun,neb) (Byler+2017):
#: BPASS zlegend[1:] / 0.020, and the Padova2007 subset / 0.019 (MIST / Padova / PARSEC files)
_NEB_AXES = {
    0.020: np.array([-1.3, -1.0, -0.82, -0.7, -0.52, -0.4, -0.3, -0.15, 0.0, 0.18, 0.3]),
    0.019: np.array([-1.98, -1.5, -0.98, -0.58, -0.39, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]),
}


def _nebular_zsun_from_axis(axis):
    """(nominal Z_sun,neb, provenance string) identified from the CLOUDY axis node set."""
    for nom, ref in _NEB_AXES.items():
        if axis.shape == ref.shape and np.allclose(axis, ref, rtol=0, atol=5e-3):
            return float(nom), (
                f"Byler+2017 CLOUDY axis log10(Z_gas/Z_sun,neb); Z_sun,neb = {nom:g} "
                "identified from the node set itself, not from isoc_type")
    return None, ("Byler+2017 CLOUDY axis log10(Z_gas/Z_sun,neb); its node set matches "
                  "neither shipped set, so no nominal Z_sun,neb is recorded")


def _legacy_metallicity_keys(param_names, extra=()) -> list:
    from .csp.csp import REMOVED_METALLICITY_KEYS
    seen = list(param_names) + list(extra)
    return [k for k in REMOVED_METALLICITY_KEYS if k in seen]


def _require_logzsol_result(path, f) -> None:
    """Refuse a pre-v1.0.5 result whose metallicity samples are absolute log10 Z."""
    mod = f["model"]
    if str(mod.attrs.get("metallicity_convention", "")) == METALLICITY_CONVENTION:
        return
    names = list(mod["param_names"].asstr()[()]) if "param_names" in mod else []
    extra = list(f["samples"]) + list(mod["theta_init"]) if "theta_init" in mod else list(f["samples"])
    if "priors" in mod:
        extra += list(mod["priors"].attrs)
    legacy = _legacy_metallicity_keys(names, extra)
    if not legacy:
        return
    raise ValueError(
        f"{path}: this result was written before v1.0.5 and its metallicity parameter(s) "
        f"{legacy} are log10 of ABSOLUTE Z, not logzsol = log10(Z/Z_sun); reading it as "
        "logzsol would silently misstate the metallicity by log10 Z_sun (0.15-0.85 dex).  "
        "Convert it with\n"
        "    from ceridwen.fit import convert_result\n"
        f"    convert_result({str(path)!r}, ssp_grid=<the grid the fit used>)\n"
        "which checks the grid against the file before shifting the samples, priors and "
        "theta_init, or refit.")


def convert_result(path, ssp_grid, out=None, *, overwrite=False):
    """Convert a pre-v1.0.5 result file to the logzsol convention, writing a NEW file.

    ``ssp_grid`` is the grid the fit used (``SSPData``/``SSPDataAfe`` or a path); its shape is
    checked against the file, and every absolute metallicity (samples, theta_init, priors) is
    shifted by -log10 Z_sun.  Returns the new path.  The input file is never modified.
    """
    import h5py
    from .ssps.ssp_data import SSPData

    path = Path(path)
    grid = ssp_grid if hasattr(ssp_grid, "log10_zsun") else SSPData.load(str(ssp_grid))
    l0 = float(grid.log10_zsun)
    n_z = int(np.asarray(grid.ssp_lgmet).size)
    n_age = int(np.asarray(grid.ssp_lg_age_gyr).size)
    out = Path(out) if out is not None else path.with_name(path.stem + "_logzsol.h5")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True or another out=")
    with h5py.File(path, "r") as f:
        mod = f["model"]
        if str(mod.attrs.get("metallicity_convention", "")) == METALLICITY_CONVENTION:
            raise ValueError(f"{path} is already in the logzsol convention")
        names = list(mod["param_names"].asstr()[()])
        legacy = _legacy_metallicity_keys(names, list(f["samples"]) + list(mod["theta_init"]))
        if not legacy:
            raise ValueError(f"{path} has no absolute-metallicity parameter to convert "
                             f"(parameters: {names})")
        f_nz = int(mod.attrs.get("n_metallicities", -1))
        f_nage = int(mod.attrs.get("n_ssp_ages", -1))
        if f_nz not in (-1, n_z) or f_nage not in (-1, n_age):
            raise ValueError(
                f"{path} was fitted with a grid of {f_nz} metallicities x {f_nage} SSP ages, "
                f"but {getattr(grid, 'chash', 'the given grid')} has {n_z} x {n_age}: this is "
                "not the grid of that fit, and its Z_sun would be wrong.  Pass the grid the "
                "fit actually used.")
    shutil.copyfile(path, out)
    with h5py.File(out, "r+") as f:
        mod, samp = f["model"], f["samples"]
        rename = {"Z": "logzsol", "zh": "logzsol_hist"}
        for old, new in rename.items():
            if old in samp:
                samp[new] = np.asarray(samp[old]) - l0
                del samp[old]
            if "theta_init" in mod and old in mod["theta_init"]:
                mod["theta_init"][new] = np.asarray(mod["theta_init"][old]) - l0
                del mod["theta_init"][old]
            if "priors" in mod and old in mod["priors"].attrs:
                mod["priors"].attrs[new] = json.dumps(
                    _shift_prior(json.loads(mod["priors"].attrs[old]), -l0, old))
                del mod["priors"].attrs[old]
        names = [rename.get(n, n) for n in list(mod["param_names"].asstr()[()])]
        del mod["param_names"]
        mod.create_dataset("param_names", data=np.array(names, dtype=object),
                           dtype=h5py.string_dtype())
        if "transforms" in mod.attrs:
            tx = json.loads(mod.attrs["transforms"])
            mod.attrs["transforms"] = json.dumps(
                [t.replace("Z <-", "logzsol <-").replace("zh <-", "logzsol_hist <-") for t in tx])
        mod.attrs["metallicity_convention"] = METALLICITY_CONVENTION
        mod.attrs["log10_zsun"] = l0
        if getattr(grid, "zsun_nominal", None) is not None:
            mod.attrs["zsun_nominal"] = float(grid.zsun_nominal)
        if getattr(grid, "axis_meaning", None):
            mod.attrs["metallicity_axis_meaning"] = str(grid.axis_meaning)
        if getattr(grid, "chash", None):
            mod.attrs["grid_chash"] = str(grid.chash)
        mod.attrs["converted_from"] = str(path)
        mod.attrs["converted_note"] = (
            f"metallicity samples/priors shifted by -log10 Z_sun = {-l0!r} (v1.0.5 conversion)")
    return out


_SHIFTABLE_PRIOR_PARAMS = {"Uniform": ("low", "high"), "TopHat": ("low", "high"),
                           "Normal": ("mean",), "ClippedNormal": ("mean", "low", "high"),
                           "StudentT": ("mean",)}


def _shift_prior(spec: dict, shift: float, name: str) -> dict:
    kind = spec.get("type")
    if kind not in _SHIFTABLE_PRIOR_PARAMS:
        raise ValueError(
            f"cannot convert the {kind!r} prior on {name!r}: its location parameters are not "
            f"known to be a shift of logzsol (supported: {sorted(_SHIFTABLE_PRIOR_PARAMS)}).  "
            "Rebuild the prior in logzsol by hand.")
    out = dict(spec)
    for k in _SHIFTABLE_PRIOR_PARAMS[kind]:
        if k in out:
            v = np.asarray(out[k], dtype=float) + shift
            out[k] = v.tolist() if v.ndim else float(v)
    return out


def eline_fluxes_for_samples(model, samples, likelihood, chunk=256):
    """Posterior line fluxes of every draw in ``samples`` ({param: (n, ...)}): dict of
    ``mean``, ``sd``, ``cloudy`` arrays (n, m), evaluated with ``jax.vmap`` in chunks."""
    from .likelihood.eline_marginal import eline_line_fluxes
    names = [p for p in model.theta_init if p in samples]
    n = int(np.asarray(samples[names[0]]).shape[0])
    theta = {p: jnp.asarray(np.asarray(samples[p]).reshape((n,) + tuple(np.shape(model.theta_init[p]))))
             for p in names}

    def one(th):
        post = eline_line_fluxes(model, th, likelihood)
        return post["mean"], post["sd"], post["cloudy"]
    f = jax.jit(jax.vmap(one))
    parts = [f({p: v[i:i + chunk] for p, v in theta.items()}) for i in range(0, n, chunk)]
    mean, sd, cloudy = (np.concatenate([np.asarray(q[k]) for q in parts]) for k in range(3))
    return dict(mean=mean, sd=sd, cloudy=cloudy)


def _write_eline_group(path, model, result, likelihood):
    import h5py
    es = model._eline_system
    vals = eline_fluxes_for_samples(model, result.samples, likelihood)
    with h5py.File(path, "a") as f:
        g = f.create_group("elines")
        g.create_dataset("names", data=np.array(es.names, dtype=object), dtype=h5py.string_dtype())
        g.create_dataset("wave_rest", data=np.asarray(es.wave_rest))
        for k, v in vals.items():
            g.create_dataset(k, data=v, compression="gzip")
        g.attrs["spectrum"] = es.spec_key
        g.attrs["observations"] = json.dumps(list(es.keys))
        g.attrs["prior_width"] = float(es.prior_width)
        g.attrs["not_fitted"] = json.dumps([list(t) for t in es.not_fitted])
        g.attrs["ignored"] = json.dumps(list(es.ignored))


def load_result_h5(path: str | Path):
    """Rebuild a ``SamplingResult`` from a result HDF5 file (the forward model itself is not restored)."""
    import h5py
    from .sampler.runner import SamplingResult

    path = Path(path)
    with h5py.File(path, "r") as f:
        _require_logzsol_result(path, f)
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
    """Read a result HDF5 file into a nested dict with keys ``'obs'``, ``'model'``, ``'samples'``,
    ``'provenance'`` (files written by fitSED since v1.0.6) and ``'elines'`` (when the fit
    marginalised the emission lines)."""
    import h5py

    out = {"obs": {}, "model": {}, "samples": {}}

    with h5py.File(path, "r") as f:
        _require_logzsol_result(path, f)
        for obs_name in f["obs"]:
            og = f["obs"][obs_name]
            obs_data = {k: np.array(og[k]) for k in og}
            for attr_name in og.attrs:
                obs_data[attr_name] = og.attrs[attr_name]
            if "likelihood_json" in og.attrs:
                obs_data["likelihood"] = json.loads(og.attrs["likelihood_json"])
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

        if "provenance" in f:
            out["provenance"] = _provenance_from_group(f["provenance"])

        if "elines" in f:
            g = f["elines"]
            out["elines"] = {k: (list(g[k].asstr()[()]) if k == "names" else np.array(g[k]))
                             for k in g}
            for attr_name in g.attrs:
                out["elines"][attr_name] = g.attrs[attr_name]

    return out
