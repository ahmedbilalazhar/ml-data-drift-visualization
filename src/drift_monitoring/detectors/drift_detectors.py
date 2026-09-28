"""Baseline detector family (roadmap Sec. 8.1, E1).

- Input-change: per-feature KS statistic (numeric) + chi-square (categorical),
  aggregated with Holm adjustment; normalized Wasserstein variant.
- Joint-change: reserved for E2 extension (two-sample classifier) — not in E0.
Same reference / windows / calibration for every comparator (fairness rule).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def ks_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                    numeric_cols: list[str]) -> dict:
    """Per-feature KS statistic + Holm-adjusted p-values; global = max D."""
    stats_list, pvals = [], []
    for c in numeric_cols:
        r = ref[c].to_numpy(float)
        v = cur[c].to_numpy(float)
        if len(np.unique(np.concatenate([r, v]))) < 2:
            stats_list.append(0.0); pvals.append(1.0); continue
        res = stats.ks_2samp(r, v)
        stats_list.append(float(res.statistic)); pvals.append(float(res.pvalue))
    pvals = np.array(pvals)
    adj = _holm(pvals) if len(pvals) else pvals
    per = {c: {"D": float(d), "p_holm": float(p)}
           for c, d, p in zip(numeric_cols, stats_list, adj)}
    return {"per_feature": per,
            "global_D": float(max(stats_list)) if stats_list else 0.0,
            "min_p_holm": float(min(adj)) if len(adj) else 1.0}


def wasserstein_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                             numeric_cols: list[str]) -> dict:
    """Normalized Wasserstein: |W| / (pooled std); global = max."""
    per, vals = {}, []
    for c in numeric_cols:
        r = ref[c].to_numpy(float); v = cur[c].to_numpy(float)
        w = float(stats.wasserstein_distance(r, v))
        scale = float(np.std(np.concatenate([r, v]))) + 1e-9
        nw = w / scale
        per[c] = {"W": w, "W_norm": nw}
        vals.append(nw)
    return {"per_feature": per, "global_W_norm": float(max(vals)) if vals else 0.0}


def categorical_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                             col: str = "cat") -> dict:
    """Chi-square homogeneity on valid codes; never invents codes."""
    cats = sorted(set(ref[col].unique()) | set(cur[col].unique()))
    table = np.array([[int((ref[col] == k).sum()) for k in cats],
                      [int((cur[col] == k).sum()) for k in cats]])
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
    except ValueError:
        chi2, p = 0.0, 1.0
    return {"chi2": float(chi2), "p": float(p), "codes": cats}


def calibrate_threshold(scores: list[float], target_alerts_per_1000: float = 10.0) -> float:
    """Stationary-calibration threshold: quantile leaving `target` alerts/1000.

    Pilot default ~10 raw alerted windows per 1000 stationary windows
    (roadmap Sec. 8.5) — a proposed operating point, not a guarantee.
    """
    q = 1.0 - target_alerts_per_1000 / 1000.0
    return float(np.quantile(scores, q))
