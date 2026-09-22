"""Analytic marginalisation over emission-line fluxes (``Spectrum(marginalize_elines=True)``).

The fluxes ``alpha`` (m,) [erg s^-1 cm^-2, observed frame, IGM-transmitted] of the fitted
nebular lines are linear nuisance parameters shared by every observation that sees them: the
marginalising Spectrum (unit-flux line profiles times the spectrum calibration), each
Photometry (the static line-to-band basis) and each Lines observation (the blend matrix times
``eline_scaling``).  With independent Gaussian noise, block k contributes

    y_k = mu_k + A_k alpha + noise,     W_k = diag(1/sigma_eff^2) on unmasked data

where ``mu_k`` is the model with the fitted lines removed.  The prior on alpha_j is flat, or
Gaussian N(F_j, (w F_j)^2) around the CLOUDY flux F_j (``eline_prior_width`` = w); Ly-alpha is
always flat.  The exact marginal likelihood is (``docs/dev/eline_marginalisation_design.md``)

    ln Z = -1/2 r'^T W r' + 1/2 b'^T P^-1 b' - 1/2 ln|P| + 1/2 sum_prior ln(1/s_j^2)
           + (m_flat/2) ln 2 pi - sum_k sum_i ln sqrt(2 pi sigma_eff,i^2)

with ``r' = y - mu - A abar`` (abar = prior mean, 0 on flat lines), ``M = A^T W A``,
``b' = A^T W r'`` and ``P = M + diag(1/s_j^2)`` (0 on flat lines).  It is evaluated with the
per-line rescaling ``d_j = s_j`` (prior) or ``M_jj^-1/2`` (flat), ``P~ = D M D + diag(prior)``,
which is well conditioned and exact also for s_j = 0 (a line pinned at its prior mean).
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

__all__ = ["ElineSystem", "build_eline_system", "eline_marginal_loglike",
           "eline_marginal_loglike_static",
           "joint_loglike", "eline_line_fluxes", "line_table_for", "line_profiles_on_grid",
           "LYA_REST_AA", "COND_MAX"]

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)
LYA_REST_AA = 1215.67
COND_MAX = 1e10


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------

def eline_marginal_loglike(blocks, prior_mean, prior_sd, is_flat, *, return_posterior=False):
    """Exact Gaussian marginal ln-likelihood of the stacked ``blocks`` over the line fluxes.

    blocks : sequence of ``(r, A, w, log_det_sum)`` per observation: residual ``y - mu`` (n,) of
        the model WITHOUT the fitted lines (0 where masked), design ``A`` (n, m), weights ``w``
        (n,) = 1/sigma_eff^2 on used data and 0 elsewhere, and the summed Gaussian
        normalisation sum_i ln sqrt(2 pi sigma_eff,i^2) over used data.
    prior_mean, prior_sd : (m,) prior of the Gaussian lines (ignored where ``is_flat``).
    is_flat : (m,) bool, static -- flat (improper, uniform on R) prior for these lines.

    Returns ln Z, or ``(ln Z, mean, cov)`` of the line-flux posterior with ``return_posterior``.
    """
    is_flat = np.asarray(is_flat, dtype=bool)
    prior = jnp.asarray(~is_flat, dtype=jnp.float64)
    abar = jnp.where(is_flat, 0.0, prior_mean)
    M = 0.0
    b = 0.0
    chi2 = 0.0
    lognorm = 0.0
    for r, A, w, log_det_sum in blocks:
        r1 = r - A @ abar
        wA = w[:, None] * A
        M = M + A.T @ wA
        b = b + wA.T @ r1
        chi2 = chi2 + jnp.sum(w * r1 * r1)
        lognorm = lognorm + log_det_sum
    diag = jnp.diagonal(M)
    tiny = jnp.finfo(jnp.float64).tiny
    d = jnp.where(is_flat, 1.0 / jnp.sqrt(jnp.maximum(diag, tiny)), prior_sd)
    P = d[:, None] * M * d[None, :] + jnp.diag(prior)
    L = jnp.linalg.cholesky(P)
    v = solve_triangular(L, d * b, lower=True)
    lnz = (-0.5 * chi2 + 0.5 * jnp.dot(v, v) - jnp.sum(jnp.log(jnp.diagonal(L)))
           + jnp.sum(jnp.where(is_flat, jnp.log(d) + _HALF_LOG_2PI, 0.0)) - lognorm)
    if not return_posterior:
        return lnz
    Pinv_db = cho_solve((L, True), d * b)
    mean = abar + d * Pinv_db
    cov = d[:, None] * cho_solve((L, True), jnp.eye(d.shape[0])) * d[None, :]
    return lnz, mean, cov


# ---------------------------------------------------------------------------
# static configuration, built once by SedModel.setup_observations
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class ElineSystem:
    """Static (setup-time) description of the joint line marginalisation."""
    spec_key: str
    phot_keys: tuple
    lines_keys: tuple
    fit_rows: np.ndarray          # (m,) rows in predict_line_fluxes order
    fit_pos: np.ndarray           # (m,) positions of fit_rows in the projector's kept lines
    names: tuple                  # (m,) FSPS names
    wave_rest: np.ndarray         # (m,) vacuum rest wavelengths [A]
    is_flat: np.ndarray           # (m,) bool
    prior_width: float
    keep_grid: np.ndarray         # (n_lines,) 0 on fitted and ignored rows, 1 elsewhere
    phot_cols: dict = field(default_factory=dict)    # key -> (n_band, m)
    lines_cols: dict = field(default_factory=dict)   # key -> (n_obs_lines, m)
    not_fitted: tuple = ()        # (name, reason) of covered lines left at the CLOUDY flux
    ignored: tuple = ()
    static: Optional[dict] = None  # precomputed design/factorisation (fixed z, width, noise, calib)
    has_grid: bool = True          # False: no nebular grid (add_neb=False / CSPBasis_afe), flat prior

    @property
    def keys(self) -> tuple:
        return (self.spec_key, *self.phot_keys, *self.lines_keys)

    @property
    def m(self) -> int:
        return int(self.fit_rows.size)

    def prior_sd(self, prior_mean):
        return self.prior_width * jnp.abs(prior_mean)

    def describe(self) -> str:
        prior = ("flat" if self.prior_width == 0.0
                 else f"Gaussian, width {self.prior_width:g} x CLOUDY (Ly-alpha flat)")
        if not self.has_grid:
            prior += "; no nebular grid, line list from emlines_info.dat"
        s = (f"{self.m} lines marginalised jointly over {list(self.keys)} ({prior}): "
             + ", ".join(self.names))
        if self.not_fitted:
            s += "; kept at CLOUDY: " + ", ".join(f"{n} ({why})" for n, why in self.not_fitted)
        if self.ignored:
            s += "; ignored: " + ", ".join(self.ignored)
        return s


def _resolve_names(neb, names, what):
    """Grid rows of FSPS line ``names``; raises with close matches for unknown names."""
    if names is None:
        return []
    table = getattr(neb, "nebem_line_names", None)
    if table is None:
        raise ValueError(
            f"{what} needs the FSPS line names from $SPS_HOME/data/emlines_info.dat, "
            "which could not be read")
    rows = []
    for n in names:
        hits = [i for i, t in enumerate(table) if t == n]
        if not hits:
            close = difflib.get_close_matches(n, [t for t in table if t], n=4)
            raise ValueError(
                f"{what}: {n!r} is not an FSPS line name"
                + (f"; did you mean one of {close}?" if close else "")
                + " (names as in $SPS_HOME/data/emlines_info.dat)")
        rows.append(hits[0])
    return rows


def _used_mask(obs, n):
    m = getattr(obs, "mask", None)
    if m is None or isinstance(m, slice):
        return np.ones(n, dtype=bool)
    m = np.asarray(m, dtype=bool)
    if m.shape != (n,):            # no flux yet (predictive container): nothing masked
        return np.ones(n, dtype=bool)
    return m


def _is_afe(csp) -> bool:
    return type(csp).__name__ == "CSPBasis_afe"


def line_table_for(csp) -> dict:
    """``{"wave", "names"}`` of FSPS's emission lines from ``<sps_home>/data/emlines_info.dat``
    (vacuum rest wavelengths [A]), for a basis without a nebular grid (``add_neb=False`` or
    ``CSPBasis_afe``)."""
    sps_home = getattr(csp, "sps_home", None) or os.environ.get("SPS_HOME")
    path = None if not sps_home else Path(sps_home) / "data" / "emlines_info.dat"
    if path is None or not path.is_file():
        raise ValueError(
            "marginalize_elines=True without a nebular grid reads the emission-line list from "
            "$SPS_HOME/data/emlines_info.dat, which "
            + ("was not found at " + str(path) if path else "needs $SPS_HOME")
            + ": set $SPS_HOME to your FSPS data directory or pass sps_home=... to the basis")
    wave, names = [], []
    with open(path) as f:
        for row in f:
            parts = row.split(",")
            if len(parts) >= 2:
                wave.append(float(parts[0]))
                names.append(parts[1].strip())
    return {"wave": np.asarray(wave, dtype=np.float64), "names": names}


def line_profiles_on_grid(wave, pos, sigma_kms=0.0, res_floor_factor=2.0):
    """(n_wave, n_line) f_nu profiles of unit-luminosity lines at ``pos`` on the model grid
    ``wave``, exactly as ``NebularModel.line_profiles(sigma_kms)`` builds them (default
    NebularModel settings: smooth_velocity, sigma_smooth = 0): a Gaussian in lambda of width
    sqrt(floor^2 + (pos sigma/c)^2), floor = res_floor_factor x the local pixel width."""
    from ..neb.NebularGridModel import CLIGHT_AA_S, SQRT_2PI
    lam = np.asarray(wave, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    idx = np.clip(np.searchsorted(lam, pos, side="right") - 1, 1, lam.size - 2)
    floor = (lam[idx + 1] - lam[idx]) * res_floor_factor
    base = pos * 0.0 / CLIGHT_AA_S * 1.0e13
    dl0 = np.maximum(base, floor)
    dl = np.sqrt(dl0 ** 2 + (pos * float(sigma_kms) / CLIGHT_AA_S * 1.0e13) ** 2)
    prof = np.exp(-0.5 * ((lam[:, None] - pos[None, :]) / dl[None, :]) ** 2)
    return prof / (SQRT_2PI * dl[None, :]) * (pos[None, :] ** 2 / CLIGHT_AA_S)


def refuse_without_grid(csp, spec, observations):
    """Setup-time refusals for a marginalising ``spec`` on a basis without a nebular grid:
    everything that needs the CLOUDY fluxes (a prior, fixed lines, Lines observations).
    Called before the line list is read, so these messages win over a missing $SPS_HOME."""
    from ..observation.lines import Lines
    what = ("CSPBasis_afe has no nebular grid (there are no alpha-enhanced CLOUDY grids)"
            if _is_afe(csp) else "this CSP has no nebular model (add_neb=False)")
    if spec.eline_prior_width > 0.0:
        raise ValueError(
            f"Spectrum {spec.name!r}: eline_prior_width={spec.eline_prior_width:g} needs the "
            f"CLOUDY line fluxes to centre the prior on, and {what}. Use the flat prior, "
            "eline_prior_width=0 (the default), or build the basis with add_neb=True")
    if spec.elines_to_fix:
        raise ValueError(
            f"Spectrum {spec.name!r}: elines_to_fix={list(spec.elines_to_fix)} keeps lines at "
            f"their CLOUDY flux, and {what}; fit them, or ignore them (elines_to_ignore)")
    if any(isinstance(o, Lines) for o in observations):
        raise ValueError(
            f"a Lines observation needs the nebular grid's line fluxes, and {what}; remove "
            "the Lines observation or build the basis with add_neb=True")


def build_eline_system(model) -> Optional[ElineSystem]:
    """Validate the configuration and build the ``ElineSystem`` (None when no Spectrum
    marginalises its lines).  Every refusal happens here, at setup, never in a kernel."""
    from ..observation.spectrum import Spectrum
    from ..observation.photometry import Photometry
    from ..observation.lines import Lines

    specs = [o for o in model.observations
             if isinstance(o, Spectrum) and getattr(o, "marginalize_elines", False)]
    if not specs:
        return None
    if len(specs) > 1:
        raise ValueError(
            f"marginalize_elines=True is set on {len(specs)} spectra "
            f"({[o.name for o in specs]}); only one marginalising Spectrum is supported")
    spec = specs[0]
    csp = model.csp
    neb = getattr(csp, "neb", None)
    has_grid = neb is not None
    if neb is None:
        # no nebular grid (add_neb=False, or CSPBasis_afe: there are no alpha-enhanced CLOUDY
        # grids): the line list, rest wavelengths and names come from FSPS's emlines_info.dat
        # (none of them depends on [alpha/Fe] or Z), the widths from sigma_gas and the
        # instrument as always; without grid fluxes the prior must be flat
        refuse_without_grid(csp, spec, model.observations)
        tab = line_table_for(csp)
        neb = SimpleNamespace(nebem_line_pos=tab["wave"], nebem_line_names=tab["names"])
    fitted_neb = sorted(set(getattr(csp, "neb_param_names", [])) & set(model.param_names))
    if fitted_neb:
        example = ", ".join(f'"{k}": lambda th: jnp.array([...])' for k in fitted_neb)
        raise ValueError(
            f"marginalize_elines=True with the nebular parameter(s) {fitted_neb} sampled: with "
            "the line fluxes marginalised the lines no longer constrain the nebular model, so "
            "it must not be fitted. Fix them with a constant transform, "
            f"SedModel(..., transforms={{{example}}}) (and drop their priors), or fit the "
            "lines at their CLOUDY fluxes instead")
    phots = [o for o in model.observations if isinstance(o, Photometry)]
    lines_obs = [o for o in model.observations if isinstance(o, Lines)]
    for o in phots + lines_obs:
        ul = getattr(o, "upper_limit", None)
        if ul is not None and bool(np.any(np.asarray(ul))):
            raise ValueError(
                f"observation {o.name!r} has upper limits; the joint line marginalisation "
                "is Gaussian and cannot include one-sided data")
    gas = model.kinematics.effective_sigma_gas
    if phots and model.zred_is_free:
        raise NotImplementedError(
            "marginalize_elines=True with Photometry and a sampled zred: the photometric line "
            "contribution is then painted on the model grid, which the joint marginalisation "
            "does not support yet. Fix zred, or fit the spectrum without the photometry")
    if phots and isinstance(gas, str) and model.broaden_photometry:
        raise NotImplementedError(
            "marginalize_elines=True with Photometry, a sampled sigma_gas and "
            "broaden_photometry=True: the photometric lines are then painted on the model grid, "
            "which the joint marginalisation does not support yet. Fix sigma_gas, or use "
            "broaden_photometry=False")

    proj = spec._proj
    kept = np.asarray(proj.line_idx, dtype=np.int64)
    n_lines = int(np.asarray(neb.nebem_line_pos).size)
    pos = np.asarray(neb.nebem_line_pos, dtype=np.float64)
    table = getattr(neb, "nebem_line_names", None) or [f"{w:.2f}A" for w in pos]
    name_of = lambda r: table[r] or f"{pos[r]:.2f}A"

    to_fit = _resolve_names(neb, spec.elines_to_fit, "elines_to_fit")
    to_fix = set(_resolve_names(neb, spec.elines_to_fix, "elines_to_fix"))
    to_ignore = _resolve_names(neb, spec.elines_to_ignore, "elines_to_ignore")
    explicit = spec.elines_to_fit is not None
    candidates = to_fit if explicit else list(kept)
    candidates = [r for r in candidates if r not in to_fix and r not in set(to_ignore)]

    # coverage at every redshift the projector serves, at the start value of sigma_gas
    wave = np.asarray(spec.wavelength, dtype=np.float64)
    used = _used_mask(spec, wave.size)
    s_tab = np.asarray(proj.sigma_inst_kms, dtype=np.float64)
    if isinstance(gas, str):
        s_gas = float(np.ravel(np.asarray(model.theta_init[gas]))[0])
    else:
        s_gas = float(gas)
    zs = ([proj.opz_ref - 1.0] if not proj.free_z
          else [proj.zred_range[0], proj.opz_ref - 1.0, proj.zred_range[1]])
    fit_rows, not_fitted = [], []
    for r in candidates:
        why = None
        inside = all(wave[0] < pos[r] * (1.0 + z) < wave[-1] for z in zs)
        if r not in set(kept) or not inside:
            if explicit:
                not_fitted.append((name_of(r), "centre outside the spectrum"))
            continue
        else:
            for z in zs:
                lo = pos[r] * (1.0 + z)
                s = np.hypot(s_gas, np.interp(lo, wave, s_tab)) / 2.99792458e5
                if not (wave[0] * np.exp(3 * s) < lo < wave[-1] * np.exp(-3 * s)):
                    why = "within 3 sigma of the spectrum edge"
                    break
                if int(np.sum(used & (np.abs(np.log(wave / lo)) < 2 * s))) < 3:
                    why = "fewer than 3 unmasked pixels within 2 sigma"
                    break
        if why is None:
            fit_rows.append(r)
        else:
            not_fitted.append((name_of(r), why))
    if explicit and not_fitted:
        raise ValueError(
            "elines_to_fit lists lines that the spectrum cannot constrain: "
            + ", ".join(f"{n} ({why})" for n, why in not_fitted))
    if not fit_rows:
        raise ValueError(
            f"Spectrum {spec.name!r} (marginalize_elines=True) covers no fittable nebular line "
            f"in [{wave[0]:.0f}, {wave[-1]:.0f}] A observed")
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    kept_pos = {int(r): i for i, r in enumerate(kept)}
    fit_pos = np.asarray([kept_pos[int(r)] for r in fit_rows], dtype=np.int64)
    keep_grid = np.ones(n_lines, dtype=np.float64)
    keep_grid[fit_rows] = 0.0
    keep_grid[np.asarray(to_ignore, dtype=np.int64)] = 0.0
    wave_rest = pos[fit_rows]
    is_flat = (np.full(fit_rows.size, spec.eline_prior_width == 0.0)
               | (np.abs(wave_rest - LYA_REST_AA) < 1.0))

    zfix = float(model.zred)
    phot_cols = {}
    for o in phots:
        if has_grid:
            basis = getattr(o, "_line_basis", None)
            gnb = np.asarray(basis if basis is not None else neb.gaussnebarr,
                             dtype=np.float64)[:, fit_rows]
        else:           # the same profiles NebularModel.line_profiles gives (setup_broadening)
            s_phot = (float(gas) if (model.broaden_photometry and not isinstance(gas, str)
                                     and float(gas) > 0.0) else 0.0)
            gnb = line_profiles_on_grid(np.asarray(csp.wave), pos[fit_rows], s_phot)
        T = np.asarray(o._T, dtype=np.float64)
        phot_cols[o.name] = (1.0 + zfix) * (T @ gnb)
    lines_cols = {}
    for o in lines_obs:
        B = csp._neb_blend_matrix_for(o)
        if B is None:
            rows = np.asarray(csp._neb_cube_rows_for(o), dtype=np.int64)
            B = np.zeros((rows.size, n_lines))
            B[np.arange(rows.size), rows] = 1.0
        lines_cols[o.name] = np.asarray(B, dtype=np.float64)[:, fit_rows]

    es = ElineSystem(
        spec_key=spec.name, phot_keys=tuple(o.name for o in phots),
        lines_keys=tuple(o.name for o in lines_obs), fit_rows=fit_rows, fit_pos=fit_pos,
        names=tuple(name_of(r) for r in fit_rows), wave_rest=wave_rest, is_flat=is_flat,
        prior_width=float(spec.eline_prior_width), keep_grid=keep_grid,
        phot_cols=phot_cols, lines_cols=lines_cols, not_fitted=tuple(not_fitted),
        ignored=tuple(name_of(r) for r in to_ignore), has_grid=has_grid)
    _check_conditioning(es, spec, proj, s_gas, used)
    es.static = _static_precompute(model, es, spec, phots, lines_obs, proj, s_gas, used, gas)
    return es


_NOISE_KEYS = ("log_err_scale", "log_jitter", "log_f_calib", "log_f_data")


def _static_precompute(model, es, spec, phots, lines_obs, proj, s_gas, used, gas):
    """When nothing the line solve depends on can change between likelihood calls (fixed
    redshift and sigma_gas, no eline_delta_zred, no sampled spectrum calibration or
    eline_scaling, and noise weights 1/sigma^2 only), precompute the line profiles, the
    design matrices, the weights and -- for a flat prior on every line -- the whole
    factorisation.  Returns None otherwise (the general per-call path is used)."""
    names = set(model.param_names) | set(model.transforms)
    if (proj.free_z or isinstance(gas, str) or "eline_delta_zred" in names
            or names & {"spectrum_scaling", "spectrum_calib"}
            or (lines_obs and "eline_scaling" in names) or names & set(_NOISE_KEYS)):
        return None
    obs_by_key = {o.name: o for o in [spec, *phots, *lines_obs]}
    if any(float(getattr(obs_by_key[k], "noise_floor", 0.0) or 0.0) > 0.0 for k in es.keys):
        return None
    if any(getattr(obs_by_key[k], "uncertainty", None) is None for k in es.keys):
        return None
    basis = np.asarray(proj.line_basis(s_gas, proj.opz_ref), dtype=np.float64)
    A = {}
    calib = getattr(spec, "calibration", None)
    A[spec.name] = basis[:, es.fit_pos] * (1.0 if calib is None else np.asarray(calib)[:, None])
    A.update({k: np.asarray(v, dtype=np.float64) for k, v in es.phot_cols.items()})
    A.update({k: np.asarray(v, dtype=np.float64) for k, v in es.lines_cols.items()})
    w, lognorm = {}, 0.0
    for k in es.keys:
        o = obs_by_key[k]
        sig = np.asarray(o.uncertainty, dtype=np.float64)
        msk = _used_mask(o, sig.size) & np.isfinite(sig)
        var = np.maximum(np.where(msk, sig, 1.0) ** 2, np.finfo(np.float64).tiny)
        w[k] = np.where(msk, 1.0 / var, 0.0)
        lognorm += float(np.sum(np.where(msk, 0.5 * np.log(var) + _HALF_LOG_2PI, 0.0)))
    M = sum(A[k].T @ (w[k][:, None] * A[k]) for k in es.keys)
    st = {"basis": basis, "A": A, "w": w, "lognorm": lognorm, "M": M,
          "B": {k: A[k].T * w[k][None, :] for k in es.keys}, "all_flat": bool(np.all(es.is_flat))}
    if st["all_flat"]:
        d = 1.0 / np.sqrt(np.maximum(np.diag(M), np.finfo(np.float64).tiny))
        L = np.linalg.cholesky(d[:, None] * M * d[None, :])
        Linv = np.linalg.inv(L)
        st["G"] = {k: (Linv * d[None, :]) @ st["B"][k] for k in es.keys}
        st["const"] = float(-np.sum(np.log(np.diag(L))) + np.sum(np.log(d) + _HALF_LOG_2PI))
    return st


def eline_marginal_loglike_static(r, st, prior_mean, prior_sd, is_flat):
    """``eline_marginal_loglike`` with the design, weights and (flat prior) factorisation of
    ``st`` precomputed; ``r`` = {key: data - model without the fitted lines}."""
    is_flat = np.asarray(is_flat, dtype=bool)
    keys = list(st["A"])
    r = {k: jnp.where(st["w"][k] > 0.0, r[k], 0.0) for k in keys}
    if st["all_flat"]:
        chi2 = sum(jnp.sum(jnp.asarray(st["w"][k]) * r[k] * r[k]) for k in keys)
        v = sum(jnp.asarray(st["G"][k]) @ r[k] for k in keys)
        return -0.5 * chi2 + 0.5 * jnp.dot(v, v) + st["const"] - st["lognorm"]
    prior = jnp.asarray(~is_flat, dtype=jnp.float64)
    abar = jnp.where(is_flat, 0.0, prior_mean)
    r1 = {k: r[k] - jnp.asarray(st["A"][k]) @ abar for k in keys}
    chi2 = sum(jnp.sum(jnp.asarray(st["w"][k]) * r1[k] * r1[k]) for k in keys)
    b = sum(jnp.asarray(st["B"][k]) @ r1[k] for k in keys)
    M = jnp.asarray(st["M"])
    d = jnp.where(is_flat, 1.0 / jnp.sqrt(jnp.maximum(jnp.diagonal(M), jnp.finfo(jnp.float64).tiny)),
                  prior_sd)
    L = jnp.linalg.cholesky(d[:, None] * M * d[None, :] + jnp.diag(prior))
    v = solve_triangular(L, d * b, lower=True)
    return (-0.5 * chi2 + 0.5 * jnp.dot(v, v) - jnp.sum(jnp.log(jnp.diagonal(L)))
            + jnp.sum(jnp.where(is_flat, jnp.log(d) + _HALF_LOG_2PI, 0.0)) - st["lognorm"])


def _check_conditioning(es, spec, proj, s_gas, used):
    """Refuse near-degenerate line sets at setup: the spectrum's information matrix at the
    reference redshift and sigma_gas, unit-diagonal scaled, with cond > COND_MAX raises."""
    if spec.uncertainty is None:
        return
    unc = np.asarray(spec.uncertainty, dtype=np.float64)
    w = np.where(used & np.isfinite(unc) & (unc > 0), 1.0 / np.where(unc > 0, unc, 1.0) ** 2, 0.0)
    A = np.asarray(proj.line_basis(s_gas, proj.opz_ref), dtype=np.float64)[:, es.fit_pos]
    M = A.T @ (w[:, None] * A)                 # spectrum only: it is what resolves the lines
    d =1.0 / np.sqrt(np.maximum(np.diag(M), np.finfo(float).tiny))
    C = d[:, None] * M * d[None, :]
    cond = float(np.linalg.cond(C))
    if not np.isfinite(cond) or cond > COND_MAX:
        off = np.abs(C - np.diag(np.diag(C)))
        i, j = np.unravel_index(int(np.argmax(off)), off.shape)
        raise ValueError(
            f"the marginalised lines are nearly degenerate in spectrum {spec.name!r} "
            f"(condition number {cond:.2e} > {COND_MAX:.0e}); the worst pair is "
            f"{es.names[i]!r} / {es.names[j]!r} (correlation {C[i, j]:.6f}). Put one of "
            "them in elines_to_fix (kept at its CLOUDY flux) or elines_to_ignore")


# ---------------------------------------------------------------------------
# likelihood assembly (used by run_sampler and MultiObservationLikelihood)
# ---------------------------------------------------------------------------

def _blocks(model, predictions, aux, static_data, lhood_of, theta):
    es = model._eline_system
    blocks, rest = [], 0.0
    for key, lhood in lhood_of.items():
        y, sig, mask, calib, ul = static_data[key]
        mu = predictions[key]
        if calib is not None:
            mu = mu * calib
        if key not in es.keys:
            if ul is not None:
                lnl_k, _ = lhood(y, mu, sig, mask, params=theta, is_upper_limit=ul)
            else:
                lnl_k, _ = lhood(y, mu, sig, mask, params=theta)
            rest = rest + lnl_k
            continue
        A = aux["cols"][key]
        if calib is not None:
            A = A * calib[:, None]
        nm = lhood.noise_model.compute(sig, mu, mask, theta, data=y)
        w = jnp.where(mask, nm.inv_var, 0.0)
        r = jnp.where(mask, y - mu, 0.0)
        blocks.append((r, A, w, jnp.sum(jnp.where(mask, nm.log_det, 0.0))))
    return blocks, rest


def joint_loglike(model, keys, likelihoods, static_data, theta):
    """Total ln-likelihood of all observations with the fitted line fluxes marginalised:
    the observations in the ElineSystem jointly, the others as usual."""
    es = model._eline_system
    predictions, aux = model.predict_with_elines(theta)
    if es.static is not None:
        rest, r = 0.0, {}
        for key, lhood in zip(keys, likelihoods):
            y, sig, mask, calib, ul = static_data[key]
            mu = predictions[key] if calib is None else predictions[key] * calib
            if key in es.keys:
                r[key] = y - mu
            elif ul is not None:
                rest = rest + lhood(y, mu, sig, mask, params=theta, is_upper_limit=ul)[0]
            else:
                rest = rest + lhood(y, mu, sig, mask, params=theta)[0]
        mean = aux["prior_mean"]
        return rest + eline_marginal_loglike_static(r, es.static, mean, es.prior_sd(mean),
                                                    es.is_flat)
    blocks, rest = _blocks(model, predictions, aux, static_data,
                           dict(zip(keys, likelihoods)), theta)
    mean = aux["prior_mean"]
    return rest + eline_marginal_loglike(blocks, mean, es.prior_sd(mean), es.is_flat)


def eline_line_fluxes(model, theta, likelihood):
    """Posterior of the marginalised line fluxes at ``theta``: dict with ``names``,
    ``wave_rest``, ``mean`` (m,) and ``sd`` (m,) [erg s^-1 cm^-2, observed frame], ``cov``,
    and ``cloudy`` (the grid fluxes).  ``likelihood`` is the MultiObservationLikelihood of
    the fit (its noise models set the weights)."""
    es = model._eline_system
    if es is None:
        raise ValueError("no Spectrum of this model has marginalize_elines=True")
    static_data = _static_data(model, likelihood.keys)
    predictions, aux = model.predict_with_elines(theta)
    blocks, _ = _blocks(model, predictions, aux, static_data,
                        dict(zip(likelihood.keys, likelihood.likelihoods)), theta)
    prior_mean = aux["prior_mean"]
    _, mean, cov = eline_marginal_loglike(blocks, prior_mean, es.prior_sd(prior_mean),
                                          es.is_flat, return_posterior=True)
    cloudy = prior_mean if es.has_grid else jnp.full_like(prior_mean, jnp.nan)
    return dict(names=es.names, wave_rest=es.wave_rest, mean=mean,
                sd=jnp.sqrt(jnp.diagonal(cov)), cov=cov, cloudy=cloudy)


def _static_data(model, keys):
    """``{key: (y - sky, sigma, mask, calibration, upper_limit)}`` as run_sampler builds it."""
    out = {}
    obs = model.obs_dict
    for key in keys:
        o = obs[key]
        y = o.flux
        sky = getattr(o, "sky", None)
        if sky is not None:
            y = y - sky
        ul = getattr(o, "upper_limit", None)
        ul = None if ul is None or not bool(jnp.any(ul)) else jnp.asarray(ul, dtype=bool)
        out[key] = (y, o.uncertainty, o.mask, getattr(o, "calibration", None), ul)
    return out
