"""Nested-sampling posterior log-weights aligned with the input point order."""
from __future__ import annotations

import numpy as np

__all__ = ["nested_log_weights", "log_evidence_from_weights"]


def nested_log_weights(log_likelihoods, log_likelihoods_birth) -> np.ndarray:
    """Log posterior weights, one per point, aligned with the inputs; -inf for
    non-finite logL or logL <= logL_birth."""
    logL = np.asarray(log_likelihoods, dtype=float).ravel()
    logLb = np.asarray(log_likelihoods_birth, dtype=float).ravel()
    if logL.shape != logLb.shape:
        raise ValueError(f"log_likelihoods {logL.shape} and log_likelihoods_birth "
                         f"{logLb.shape} differ in length")
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(logL) & (logL > logLb)
    out = np.full(logL.shape, -np.inf)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("no point has a finite log-likelihood above its birth contour")
    L, B = logL[idx], logLb[idx]
    order = np.argsort(L, kind="stable")
    Ls = L[order]
    n_born = np.searchsorted(np.sort(B), Ls, side="left")
    n_dead = np.searchsorted(Ls, Ls, side="left")
    nlive = (n_born - n_dead).astype(float)                    # >= 1: the point itself counts
    logX = np.cumsum(np.log(nlive / (nlive + 1.0)))
    logX_prev = np.concatenate([[0.0], logX[:-1]])
    logX_next = np.concatenate([logX[1:], [-np.inf]])
    with np.errstate(divide="ignore"):
        logdX = logX_prev + np.log1p(-np.exp(logX_next - logX_prev)) - np.log(2.0)
    out[idx[order]] = logdX + Ls
    return out


def log_evidence_from_weights(log_weights) -> float:
    lw = np.asarray(log_weights, dtype=float)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return float("-inf")
    m = lw.max()
    return float(m + np.log(np.sum(np.exp(lw - m))))
