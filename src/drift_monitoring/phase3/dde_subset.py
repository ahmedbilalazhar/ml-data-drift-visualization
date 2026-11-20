"""Small, explicit primitives for the Phase 3 DDE-style subset audit.

This is an empirical co-drift proxy inspired by the Domino drift-effect paper;
it is not an implementation of the paper's complete online system.
"""
from __future__ import annotations

import numpy as np


def pair_correlation(reference: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation estimated on historical reference rows."""
    x = np.asarray(reference, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("reference must be a finite 2D array with at least two rows/features")
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def fit_pooled_dde_curve(correlations: list[np.ndarray], drift_flags: list[np.ndarray],
                         thresholds: np.ndarray | None = None) -> dict:
    """Pool pair counts across development drift episodes (Eq. 14--15 proxy).

    Empty episodes are omitted because they provide no observed drift anchor.
    The curve is fit without generator truth and with development streams only.
    """
    if len(correlations) != len(drift_flags) or not correlations:
        raise ValueError("correlations and flags must be nonempty paired episode lists")
    z = np.asarray(thresholds if thresholds is not None else np.linspace(0, .9, 10), dtype=float)
    if z.ndim != 1 or not len(z) or np.any((z < 0) | (z > 1)) or np.any(np.diff(z) < 0):
        raise ValueError("correlation thresholds must be sorted values in [0, 1]")
    total = np.zeros(len(z), dtype=np.int64)
    both = np.zeros(len(z), dtype=np.int64)
    episode_count = 0
    for corr, flags in zip(correlations, drift_flags, strict=True):
        corr = np.asarray(corr, dtype=float)
        flags = np.asarray(flags, dtype=bool)
        if corr.shape != (len(flags), len(flags)) or not np.isfinite(corr).all():
            raise ValueError("each correlation matrix must match its feature flag vector")
        if not flags.any():
            continue
        i, j = np.triu_indices(len(flags), k=1)
        edge_corr = np.abs(corr[i, j])
        edge_both = flags[i] & flags[j]
        total += np.array([(edge_corr >= cut).sum() for cut in z], dtype=np.int64)
        both += np.array([np.sum((edge_corr >= cut) & edge_both) for cut in z], dtype=np.int64)
        episode_count += 1
    curve = np.divide(both, total, out=np.zeros_like(both, dtype=float), where=total > 0)
    return {"thresholds": z, "curve": curve, "pair_count": total, "both_drifted_count": both,
            "episodes_used": int(episode_count), "status": "dde_style_empirical_proxy"}


def score_unmonitored(curve: dict, correlations: np.ndarray, drifted_monitored: np.ndarray,
                      unmonitored: np.ndarray) -> np.ndarray:
    """Average learned DDE probability over currently flagged monitored anchors."""
    corr = np.asarray(correlations, dtype=float)
    anchors = np.asarray(drifted_monitored, dtype=int)
    candidates = np.asarray(unmonitored, dtype=int)
    z = np.asarray(curve["thresholds"], dtype=float)
    p = np.asarray(curve["curve"], dtype=float)
    supported = np.asarray(curve.get("pair_count", np.ones(len(z))), dtype=np.int64) > 0
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or len(p) != len(z) or len(supported) != len(z):
        raise ValueError("invalid correlation matrix or fitted curve")
    if not supported.any():
        return np.full(len(candidates), np.nan, dtype=float)
    if np.any((anchors < 0) | (anchors >= len(corr))) or np.any((candidates < 0) | (candidates >= len(corr))):
        raise ValueError("feature index outside correlation matrix")
    if not len(anchors):
        return np.zeros(len(candidates), dtype=float)
    out = np.zeros(len(candidates), dtype=float)
    for row, feature in enumerate(candidates):
        rho = np.abs(corr[feature, anchors])
        values = []
        for value in rho:
            eligible = np.flatnonzero(supported & (z <= value))
            # A zero-denominator high-z bin is unsupported, not probability 0.
            index = int(eligible[-1]) if len(eligible) else int(np.flatnonzero(supported)[0])
            values.append(p[index])
        out[row] = float(np.mean(values))
    return out


def binary_ranking_metrics(scores: np.ndarray, truth: np.ndarray, k: int = 3) -> dict:
    """Average precision, AUROC and precision@k with explicit undefined cases."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = np.asarray(scores, dtype=float)
    y = np.asarray(truth, dtype=bool)
    if score.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("scores and binary truth must be aligned finite vectors")
    positives = int(y.sum())
    if not len(y) or positives == 0:
        ap = auc = float("nan")
        p_at_k = float("nan") if not positives else 0.0
    else:
        ap = float(average_precision_score(y, score))
        auc = float(roc_auc_score(y, score)) if positives < len(y) else float("nan")
        k_eff = min(k, len(score))
        boundary = np.sort(score)[-k_eff]
        above = score > boundary
        tied = score == boundary
        needed = k_eff - int(above.sum())
        expected_hits = float(y[above].sum()) + needed * float(y[tied].mean())
        p_at_k = expected_hits / k_eff if k_eff else float("nan")
    return {"average_precision": ap, "auroc": auc, "precision_at_k": p_at_k,
            "positive_count": positives, "feature_count": len(y), "empty_truth": positives == 0}
