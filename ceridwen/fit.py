"""
ceridwen/fit.py
===============
High-level convenience function for SED fitting.

``fitSED`` is the single entry-point for users who want to go from a
configured model and observations to a posterior stored on disk, without
manually wiring likelihoods, sampler adapters, or HDF5 I/O.

Example
-------
::

    import jax
    from ceridwen.fit import fitSED

    result = fitSED(
        model,
        observations,
        output_dir = "./my_fit",
        sampler    = "nested",          # or "nuts"
        rng_key    = jax.random.PRNGKey(42),
    )

This writes ``my_fit/ceridwen_result.h5`` with subgroups::

    /obs/<obs_name>/flux
    /obs/<obs_name>/uncertainty
    /obs/<obs_name>/wavelength
    /obs/<obs_name>/mask
    /model/param_names
    /model/priors/<param_name>   (JSON string)
    /model/theta_init/<param>
    /model/wave
    /samples/<param_name>        (n_samples, *shape)
    /samples/log_likelihoods
    /samples/log_weights
    /samples/log_evidence
    /samples/log_evidence_err
    /samples/sampler_name
    /samples/wall_time_s
    /samples/n_likelihood_calls
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger("ceridwen")

Array = jax.Array


# ======================================================================
# Public API
# ======================================================================

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
    """
    Fit an SED model to observations and write results to HDF5.

    This is the main user-facing function.  It constructs the likelihood,
    configures and runs the requested sampler, and writes the posterior
    samples, model metadata, and observation data to disk.

    Parameters
    ----------
    model : SedModel
        A fully configured ``SedModel`` instance.  If ``observations``
        is not None, ``model.observations`` is replaced before fitting.
    observations : list[Observation], optional
        Observations to fit.  If None, uses ``model.observations``.
        Passing observations here triggers ``obs.setup_for_model()``
        automatically.
    output_dir : str or Path
        Directory for the output HDF5 file.  Created if it does not
        exist.
    sampler : str
        ``"nested"`` (default) for BlackJAX nested sampling, or
        ``"nuts"`` for BlackJAX NUTS HMC.
    rng_key : Array, optional
        JAX PRNG key.  Defaults to ``jax.random.PRNGKey(0)``.
    sampler_kwargs : dict, optional
        Extra keyword arguments forwarded to the sampler adapter
        constructor (e.g. ``num_warmup``, ``num_samples``,
        ``num_chains``, ``dense_mass``, etc.).  Any argument accepted
        by ``BlackJAXNestedSamplerAdapter`` or ``BlackJAXNUTSAdapter``
        can be passed here.
    vi : None, str, VariationalMap, or TrainedMap, optional
        Variational-inference preconditioning (only used when
        ``sampler='nuts'``).  See :class:`BlackJAXNUTSAdapter` for
        accepted values.  In short:

        - ``None`` (default): plain window-adaptation NUTS.
        - ``'tril'``: train a full-rank Gaussian transport map
          (paper's TriL baseline).
        - ``'iaf'``: train a stacked inverse-autoregressive-flow map
          (paper's NeuTra neural transport).

        When a map is supplied, NUTS samples in the whitened
        z-space, giving dramatically shorter warmup.  See
        Hoffman et al. 2019, arXiv:1903.03704.

    vi_kwargs : dict, optional
        Forwarded to :func:`train_vi` (e.g. ``num_steps=1500``,
        ``batch_size=16``, ``lr0=1e-2``) and/or the VI map
        constructor (e.g. ``init_scale=0.1`` for TriL, ``n_flows=3``
        for IAF).
    filename : str
        Name of the output HDF5 file.  Default ``ceridwen_result.h5``.
    overwrite : bool
        If True (default), overwrite an existing file.
    verbose : bool
        Print progress.  Default True.

    Returns
    -------
    SamplingResult
        The sampling result object (same as returned by ``run_sampler``).
        The HDF5 file is written as a side effect.
    """
    from .likelihood.likelihood import (
        DiagonalGaussianLikelihood,
        MultiObservationLikelihood,
    )
    from .sampler.runner import run_sampler

    # ── Validate inputs ───────────────────────────────────────────────
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

    # ── Attach observations if provided ───────────────────────────────
    if observations is not None:
        model.observations = list(observations)
        # Forward model.zred so the per-observation projection matrices
        # (Photometry._T, Spectrum._H, Lines._W) are baked at the
        # *observed-frame* wavelength grid (1+z) * wave_rest.  Calling
        # setup_for_model(...) with the default zred=0 here would clobber
        # the correctly-redshifted projection that SedModel.__init__
        # built (model.py L200), so any non-zero fixed redshift would
        # silently degrade to a rest-frame filter integral while
        # CSPBasis.predict still applied flux_factor_maggies(z).  At
        # z ~ 2.7 that produces 10-20 sigma photometric residuals
        # because the filters sample the wrong intrinsic wavelengths.
        for obs in model.observations:
            obs.setup_for_model(model.csp.wave, zred=model.zred)

    if not model.observations:
        raise ValueError("No observations attached to model.")

    # ── Build likelihood ──────────────────────────────────────────────
    _t0_likelihood = time.perf_counter()
    obs_dict = model.obs_dict
    keys = tuple(obs_dict.keys())
    likelihoods = tuple(DiagonalGaussianLikelihood() for _ in keys)
    multi_likelihood = MultiObservationLikelihood(
        keys=keys,
        likelihoods=likelihoods,
    )
    _t_likelihood = time.perf_counter() - _t0_likelihood

    # Route verbose diagnostics through the package logger.
    if verbose:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            _handler = logging.StreamHandler()
            _handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(_handler)
            logger.propagate = False

    # Report the XLA backend up front: a fit silently falling back to CPU
    # (missing CUDA jaxlib, JAX_PLATFORMS=cpu leaking from a mock script,
    # driver mismatch) looks identical except for a ~10-100x slowdown.
    _devices = jax.devices()
    _backend = jax.default_backend().upper()   # 'CPU', 'GPU', or 'TPU'
    _device_str = ", ".join(str(d) for d in _devices)

    if verbose:
        logger.info(f"ceridwen.fitSED")
        logger.info(f"  Device      : {_backend}  ({_device_str})")
        if _backend == "CPU":
            logger.info(
                "                ^ running on CPU — for GPU check that a "
                "CUDA-enabled jaxlib is installed and JAX_PLATFORMS is unset"
            )
        logger.info(f"  Sampler     : {sampler}")
        logger.info(f"  Parameters  : {model.param_names}  ({sum(int(jnp.size(v)) for v in model.theta_init.values())} dims)")
        logger.info(f"  Observations: {list(keys)}")
        logger.info(f"  Output      : {output_path}")
        logger.info(f"  Likelihood build: {_t_likelihood:.3f} s")

    # ── Configure sampler ─────────────────────────────────────────────
    _t0_adapter = time.perf_counter()
    adapter = _build_adapter(
        sampler, model, sampler_kwargs, verbose,
        vi=vi, vi_kwargs=vi_kwargs,
    )
    _t_adapter = time.perf_counter() - _t0_adapter

    if verbose:
        logger.info(f"  Adapter build:    {_t_adapter:.3f} s")

    # ── Run ────────────────────────────────────────────────────────────
    _t0_sampler = time.perf_counter()
    result = run_sampler(model, multi_likelihood, adapter, rng_key)
    _t_sampler = time.perf_counter() - _t0_sampler

    if verbose:
        logger.info(f"\n{result.summary()}")

    # ── Write HDF5 ────────────────────────────────────────────────────
    _t0_h5 = time.perf_counter()
    write_result_h5(output_path, model, result, verbose=verbose)
    _t_h5 = time.perf_counter() - _t0_h5

    if verbose:
        _t_total = _t_likelihood + _t_adapter + _t_sampler + _t_h5
        logger.info(f"\n  fitSED timing breakdown:")
        logger.info(f"    Likelihood build : {_t_likelihood:>8.3f} s")
        logger.info(f"    Adapter build    : {_t_adapter:>8.3f} s")
        logger.info(f"    Sampler run      : {_t_sampler:>8.1f} s")
        logger.info(f"    HDF5 write       : {_t_h5:>8.3f} s")
        logger.info(f"    Total fitSED     : {_t_total:>8.1f} s")

    return result


# ======================================================================
# Sampler factory
# ======================================================================

def _build_adapter(sampler: str, model, sampler_kwargs: dict, verbose: bool,
                   vi=None, vi_kwargs=None):
    """Construct the appropriate SamplerAdapter."""
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
            num_delete=250,
            num_inner_steps=30,
        )
        defaults.update(sampler_kwargs)
        return BlackJAXNestedSamplerAdapter(**defaults)

    elif sampler in ("nuts", "hmc"):
        from .sampler.nuts import BlackJAXNUTSAdapter

        # Auto-detect bounded priors for sigmoid reparameterisation
        bounds = sampler_kwargs.pop("bounds", None)
        if bounds is None:
            bounds = _detect_bounds(model)
            if verbose and bounds:
                logger.info(f"  Auto-detected bounded priors: {bounds}")

        # The adapter's own __init__ already applies VI-aware defaults
        # (shorter warmup, larger initial step size, diagonal mass
        # matrix) when ``vi`` is not None.  We only set the common
        # defaults here and let the user override via ``sampler_kwargs``.
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


def _detect_bounds(model) -> dict[str, tuple[float, float]]:
    """
    Inspect model.priors to find parameters with Uniform/TopHat priors
    and extract their (low, high) bounds for sigmoid reparameterisation.
    """
    bounds = {}
    for name, prior in model.priors.items():
        cls_name = type(prior).__name__
        if cls_name in ("Uniform", "TopHat"):
            lo = float(prior.params["low"])
            hi = float(prior.params["high"])
            bounds[name] = (lo, hi)
        elif cls_name == "ClippedNormal":
            # ClippedNormal also has hard bounds
            if "low" in prior.params and "high" in prior.params:
                lo = float(prior.params["low"])
                hi = float(prior.params["high"])
                bounds[name] = (lo, hi)
    return bounds


# ======================================================================
# HDF5 I/O
# ======================================================================

def write_result_h5(
    path: str | Path,
    model,
    result,
    verbose: bool = True,
):
    """
    Write fit results to an HDF5 file with subgroups obs, model, samples.

    Parameters
    ----------
    path : str or Path
        Output file path.
    model : SedModel
        The model used for fitting.
    result : SamplingResult
        The sampling result.
    verbose : bool
        Print confirmation.

    HDF5 structure
    --------------
    ::

        /obs/<obs_name>/
            flux           (n_data,)
            uncertainty    (n_data,)
            wavelength     (n_data,)
            mask           (n_data,)  bool
            attrs: type, name, [resolution, smoothtype, filternames, ...]

        /model/
            param_names    (n_params,)  variable-length string
            wave           (n_wave,)
            theta_init/<param_name>    (shape,)
            priors/<param_name>        scalar string (JSON)
            transforms     list of (derived, free_param) pairs

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
        # ── /obs ──────────────────────────────────────────────────────
        obs_grp = f.create_group("obs")
        for obs in model.observations:
            og = obs_grp.create_group(obs.name)
            og.create_dataset("flux", data=np.asarray(obs.flux))
            og.create_dataset("uncertainty", data=np.asarray(obs.uncertainty))
            og.create_dataset("wavelength", data=np.asarray(obs.wavelength))
            og.create_dataset("mask", data=np.asarray(obs.mask))
            og.attrs["type"] = type(obs).__name__
            og.attrs["name"] = obs.name

            # Observation-specific metadata
            if hasattr(obs, "resolution") and obs.resolution is not None:
                og.attrs["resolution"] = float(obs.resolution)
            if hasattr(obs, "smoothtype") and obs.smoothtype is not None:
                og.attrs["smoothtype"] = str(obs.smoothtype)
            if hasattr(obs, "filternames"):
                og.attrs["filternames"] = json.dumps(obs.filternames)
            # Persist FSPS line names for Lines observations so PPC / plotting
            # code can label rows without needing an FSPS install (HDF5 would
            # otherwise carry only wavelengths).
            if hasattr(obs, "line_names") and obs.line_names is not None:
                og.attrs["line_names"] = json.dumps(list(obs.line_names))
            if hasattr(obs, "line_ind") and obs.line_ind is not None:
                og.create_dataset("line_ind", data=np.asarray(obs.line_ind))

        # ── /model ────────────────────────────────────────────────────
        mod_grp = f.create_group("model")

        # Parameter names as variable-length strings.  Write the
        # canonical list from result.param_names (which the sampler
        # built from theta_init.keys() — matching the per-parameter
        # sample datasets we create below) rather than model.param_names
        # (which can silently drift if a caller appends to theta_init
        # after SedModel.__init__ without also touching param_names —
        # the symptom is parameters that get sampled and converge but
        # never appear in corner / trace plots).
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

        # Model wavelength grid
        mod_grp.create_dataset("wave", data=np.asarray(model.csp.wave))

        # Initial parameter values
        init_grp = mod_grp.create_group("theta_init")
        for name, val in model.theta_init.items():
            init_grp.create_dataset(name, data=np.asarray(val))

        # Priors (serialised as JSON strings)
        prior_grp = mod_grp.create_group("priors")
        for name, prior in model.priors.items():
            if hasattr(prior, "serialize"):
                prior_grp.attrs[name] = json.dumps(prior.serialize())
            else:
                prior_grp.attrs[name] = repr(prior)

        # Transforms (record which derived params map to which free params)
        if model.transforms:
            tx_names = []
            for derived, fn in model.transforms.items():
                fn_name = getattr(fn, "__name__", repr(fn))
                tx_names.append(f"{derived} <- {fn_name}")
            mod_grp.attrs["transforms"] = json.dumps(tx_names)

        # CSP configuration
        mod_grp.attrs["n_time"] = int(model.csp.ages.shape[0]) if hasattr(model.csp, "ages") else -1
        mod_grp.attrs["n_metallicities"] = int(model.csp.zmet.shape[0]) if hasattr(model.csp, "zmet") else -1

        # ── /samples ─────────────────────────────────────────────────
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

        # Scalar metadata as attributes
        samp_grp.attrs["log_evidence"] = float(result.log_evidence)
        samp_grp.attrs["log_evidence_err"] = float(result.log_evidence_err)
        samp_grp.attrs["sampler_name"] = result.sampler_name
        samp_grp.attrs["wall_time_s"] = result.wall_time_s
        samp_grp.attrs["n_likelihood_calls"] = result.n_likelihood_calls
        samp_grp.attrs["n_samples"] = int(result.log_likelihoods.shape[0])
        # Persist chain layout so post-hoc trace plots can reshape the
        # flat (n_chains * n_per_chain, ...) sample arrays back into
        # (n_chains, n_per_chain, ...).  Pulled from result.raw which
        # the BlackJAX-NUTS adapter populates with the actual run-time
        # values.  Absent for nested sampling (single-chain by design).
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


def read_result_h5(path: str | Path) -> dict:
    """
    Read an HDF5 result file into a plain dictionary.

    Returns a dict with keys ``'obs'``, ``'model'``, ``'samples'``,
    each containing sub-dicts of arrays and metadata.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file written by ``write_result_h5``.

    Returns
    -------
    dict
        Nested dictionary mirroring the HDF5 group structure.
    """
    import h5py

    out = {"obs": {}, "model": {}, "samples": {}}

    with h5py.File(path, "r") as f:
        # ── obs ───────────────────────────────────────────────────────
        for obs_name in f["obs"]:
            og = f["obs"][obs_name]
            obs_data = {
                "flux": np.array(og["flux"]),
                "uncertainty": np.array(og["uncertainty"]),
                "wavelength": np.array(og["wavelength"]),
                "mask": np.array(og["mask"]),
            }
            for attr_name in og.attrs:
                obs_data[attr_name] = og.attrs[attr_name]
            out["obs"][obs_name] = obs_data

        # ── model ─────────────────────────────────────────────────────
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

        # ── samples ───────────────────────────────────────────────────
        samp = f["samples"]
        for dset_name in samp:
            out["samples"][dset_name] = np.array(samp[dset_name])

        for attr_name in samp.attrs:
            out["samples"][attr_name] = samp.attrs[attr_name]

    return out
