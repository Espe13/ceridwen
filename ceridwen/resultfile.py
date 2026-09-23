"""Rebuild a ``SedModel`` from a ``fitSED`` result file, and check a model against one.

A result file (``fit.write_result_h5``) records the priors, the free parameters and their
``theta_init``, the transforms *by name*, zred, the kinematics, the cosmology, the grid
provenance and the observations, but not the CSP itself nor the transform callables.  So:

* :func:`priors_from_result` rebuilds the prior objects (exactly: every prior class serialises
  its constructor arguments);
* :func:`rebuild_model` builds the ``SedModel`` from a CSP, observations and transforms that
  you supply, taking everything else from the file, and then checks the result;
* :func:`check_model_against_result` compares any model with the file and says precisely what
  differs.  It writes the model's own record through ``write_result_h5`` (to a temporary file)
  and compares the two records, so the comparison uses the same code that wrote the file.

What cannot be checked because the file does not record it is listed in ``NOT_RECORDED``.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["priors_from_result", "kinematics_from_result", "check_model_against_result",
           "rebuild_model", "ResultComparison", "NOT_RECORDED"]

#: Parts of a model that a result file does not record, so no comparison can see them.
NOT_RECORDED = (
    "the bodies of the transform callables (only 'derived <- function name' is stored)",
    "CSP options beyond csp_config, sfh_times_yr and the grid provenance (dust-law choices "
    "and parameters fixed inside the CSP, nebular / dust-emission grid options)",
    "observation options not stored in /obs (noise_floor, upper limits, calibration, sky, "
    "PhotometricBroadener settings)",
)

# /model entries that are not part of the model definition
_IGNORED_MODEL_KEYS = {"converted_from", "converted_note"}
# recorded only by files written after the entry was added: absent from an older file is a
# note ("not recorded"), not a difference
_ADDED_LATER = {"/model@csp_config", "/model/sfh_times_yr"}


@dataclass
class ResultComparison:
    """Outcome of :func:`check_model_against_result`.

    differences : list of (path, value in the file, value in the model) -- the model differs
    notes       : list of (path, file, model) -- differences that do not change the model
                  definition (``theta_init`` values: the starting point), and entries an
                  older file does not record
    not_recorded: what the file cannot tell (``NOT_RECORDED``)
    """
    path: str
    differences: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    not_recorded: tuple = NOT_RECORDED

    @property
    def ok(self) -> bool:
        return not self.differences

    def __str__(self) -> str:
        head = (f"model matches {self.path}" if self.ok else
                f"model differs from {self.path} in {len(self.differences)} place(s):")
        out = [head]
        for p, a, b in self.differences:
            out.append(f"  {p}\n      file : {a}\n      model: {b}")
        if self.notes:
            out.append("  (not model differences: starting point, or not recorded by the file)")
            out += [f"  {p}: file {a}, model {b}" for p, a, b in self.notes]
        out.append("  not checked (not recorded in the file): " + "; ".join(self.not_recorded))
        return "\n".join(out)


def _read_record(path) -> dict:
    """{'/model/...' or '/obs/...': value} for every dataset and attribute of /model and /obs."""
    import h5py
    out = {}

    def visit(grp, prefix):
        for k, v in grp.attrs.items():
            out[f"{prefix}@{k}"] = v
        for k, v in grp.items():
            if isinstance(v, h5py.Group):
                visit(v, f"{prefix}/{k}")
            else:
                out[f"{prefix}/{k}"] = v.asstr()[()] if v.dtype.kind == "O" else v[()]

    with h5py.File(path, "r") as f:
        for top in ("model", "obs"):
            if top in f:
                visit(f[top], f"/{top}")
    return out


def _same(a, b) -> bool:
    if isinstance(a, bytes):
        a = a.decode()
    if isinstance(b, bytes):
        b = b.decode()
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "O" or b.dtype.kind == "O":
        return a.shape == b.shape and all(str(x) == str(y) for x, y in zip(a.ravel(), b.ravel()))
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _short(v) -> str:
    if v is None:
        return "(absent)"
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        return v if len(v) <= 200 else v[:200] + " ..."
    a = np.asarray(v)
    if a.size <= 8:
        return repr(a.tolist())
    return f"array{a.shape} {a.dtype}, first {a.ravel()[:4].tolist()}"


def _model_record(model) -> dict:
    """The /model and /obs record that ``write_result_h5`` would write for ``model``."""
    import jax.numpy as jnp
    from .fit import write_result_h5
    from .sampler.runner import SamplingResult

    names = list(model.theta_init)          # the samplers' order (run_sampler -> adapter.run)
    stub = SamplingResult(
        samples={k: jnp.asarray(model.theta_init[k])[None] for k in names},
        log_evidence=float("nan"), log_evidence_err=float("nan"),
        log_weights=jnp.zeros(1), log_likelihoods=jnp.zeros(1), param_names=names,
        n_likelihood_calls=0, wall_time_s=0.0, sampler_name="record")
    fd, tmp = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        write_result_h5(tmp, model, stub, verbose=False)
        return _read_record(tmp)
    finally:
        os.remove(tmp)


def check_model_against_result(model, path, *, raise_on_difference: bool = False
                               ) -> ResultComparison:
    """Compare ``model`` with the model recorded in the result file ``path``.

    Every recorded attribute and dataset of ``/model`` and ``/obs`` is compared exactly
    (priors, parameter names and shapes, transform names, zred, kinematics, cosmology, grid
    provenance and metallicity axes, the model wavelength grid, and every observation's data,
    mask, filters and instrument).  ``theta_init`` values are reported as notes only.
    Returns a :class:`ResultComparison`; with ``raise_on_difference`` a difference raises.
    """
    from .fit import _require_logzsol_result
    import h5py

    path = str(path)
    with h5py.File(path, "r") as f:
        _require_logzsol_result(path, f)
    in_file = _read_record(path)
    in_model = _model_record(model)
    cmp = ResultComparison(path=path)
    for key in sorted(set(in_file) | set(in_model)):
        if key.split("@")[-1] in _IGNORED_MODEL_KEYS:
            continue
        a, b = in_file.get(key), in_model.get(key)
        if a is not None and b is not None and _same(a, b):
            continue
        entry = (key, _short(a), _short(b))
        is_init = key.startswith("/model/theta_init/")
        if a is None and key in _ADDED_LATER:
            cmp.notes.append((key, "(not recorded by this file)", entry[2]))
        elif is_init and a is not None and b is not None and np.shape(a) == np.shape(b):
            cmp.notes.append(entry)
        else:
            cmp.differences.append(entry)
    if raise_on_difference and not cmp.ok:
        raise ValueError(str(cmp))
    return cmp


def priors_from_result(path) -> dict:
    """The prior objects of a result file, ``{name: Prior}``, rebuilt from their serialisation."""
    from .sampler import priors as _priors
    from .fit import read_result_h5

    out = {}
    for name, spec in read_result_h5(path)["model"]["priors"].items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"{path}: the prior of {name!r} was stored as {spec!r}, not as a "
                             "serialised ceridwen prior; rebuild it by hand")
        cls = getattr(_priors, spec["type"], None)
        if not (isinstance(cls, type) and issubclass(cls, _priors.Prior)):
            raise ValueError(f"{path}: the prior of {name!r} has unknown type {spec['type']!r}")
        kwargs = {k: v for k, v in spec.items() if k not in ("type", "name")}
        out[name] = cls(name=spec.get("name", ""), **kwargs)
    return out


def kinematics_from_result(path):
    """The ``Kinematics`` of a result file (``DEFAULT_KINEMATICS`` when it records none)."""
    from .broadening import DEFAULT_KINEMATICS, TIED, Kinematics
    from .fit import read_result_h5

    mod = read_result_h5(path)["model"]
    if "kinematics_sigma_gal" not in mod:
        return DEFAULT_KINEMATICS

    def width(v):
        v = v.decode() if isinstance(v, bytes) else str(v)
        try:
            return float(v)
        except ValueError:
            return v                                   # a theta key

    gal, gas = width(mod["kinematics_sigma_gal"]), width(mod["kinematics_sigma_gas"])
    # the file stores the effective gas width; equal to sigma_gal is TIED (same predictions)
    return Kinematics(sigma_gal=gal, sigma_gas=TIED if gas == gal else gas,
                      sigma_max=float(mod["kinematics_sigma_max"]))


def rebuild_model(path, csp, observations, transforms=None, *, check: bool = True):
    """``SedModel`` of the fit in ``path``, from a CSP, observations and transforms you supply.

    Taken from the file: priors, the free parameters and their ``theta_init``, zred,
    lumdist_mpc, kinematics and ``broaden_photometry``.  Not recorded, so supplied by you: the
    CSP (built as for the fit, with the same grid and cosmology), the observation objects and
    the transform callables (the file lists them as ``derived <- function name``; the derived
    names must match).  With ``check`` (default) the rebuilt model is compared with the file by
    :func:`check_model_against_result` and any difference raises a ``ValueError`` saying which.
    """
    from .fit import read_result_h5
    from .model import SedModel

    rec = read_result_h5(path)
    mod = rec["model"]
    transforms = dict(transforms or {})
    stored = json.loads(mod["transforms"]) if "transforms" in mod else []
    want = [t.split(" <- ")[0] for t in stored]
    if sorted(want) != sorted(transforms):
        raise ValueError(
            f"{path} was fitted with transforms {stored}; the transforms given are for "
            f"{sorted(transforms)}.  Supply one callable per derived name "
            f"{sorted(want)} (the file stores their names, not their code)")
    kwargs = dict(
        priors=priors_from_result(path),
        transforms=transforms,
        # in the fit's parameter order (h5py hands the theta_init group back sorted)
        free_param_init={k: np.asarray(mod["theta_init"][k]) for k in mod["param_names"]},
        zred=float(mod.get("zred", 0.0)),
        kinematics=kinematics_from_result(path),
    )
    if "lumdist_mpc" in mod:
        kwargs["lumdist_mpc"] = float(mod["lumdist_mpc"])
    if "broaden_photometry" in mod:
        kwargs["broaden_photometry"] = bool(mod["broaden_photometry"])
    model = SedModel(csp, observations, **kwargs)
    if check:
        check_model_against_result(model, path, raise_on_difference=True)
    return model
