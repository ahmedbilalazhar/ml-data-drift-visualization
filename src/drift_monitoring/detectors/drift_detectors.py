"""Baseline detector family (roadmap Sec. 8.1, E1).

- Input-change: per-feature KS statistic (numeric) + chi-square (categorical),
  aggregated with Holm adjustment; normalized Wasserstein variant;
  Jensen-Shannon distance variant (Szucs DDE paper's JS detector family).
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


def js_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                    numeric_cols: list[str], bins: int = 20) -> dict:
    """Per-feature Jensen-Shannon distance via fixed-bin histograms.

    Histograms share the pooled min/max range; bounded in [0, 1];
    global = max. Same windows/reference as KS/Wasserstein (fairness rule).
    """
    from scipy.spatial import distance
    per, vals = {}, []
    for c in numeric_cols:
        r = ref[c].to_numpy(float)
        v = cur[c].to_numpy(float)
        lo = float(min(r.min(initial=0.0), v.min(initial=0.0)))
        hi = float(max(r.max(initial=0.0), v.max(initial=0.0)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            per[c] = {"JS": 0.0}
            vals.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        pr, _ = np.histogram(r, bins=edges)
        pv, _ = np.histogram(v, bins=edges)
        pr = pr.astype(float) + 1e-9
        pv = pv.astype(float) + 1e-9
        pr /= pr.sum()
        pv /= pv.sum()
        js = float(distance.jensenshannon(pr, pv, base=2.0))
        if not np.isfinite(js):
            js = 0.0
        per[c] = {"JS": min(max(js, 0.0), 1.0)}
        vals.append(per[c]["JS"])
    return {"per_feature": per, "global_JS": float(max(vals)) if vals else 0.0}


def adwin_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                       numeric_cols: list[str]) -> dict:
    """ADWIN-inspired mean-change score (Guerrero/DDE detector family).

    Full ADWIN adaptively cuts windows via Hoeffding bounds on error rates;
    here windows are fixed by the replay clock, so this scores the
    standardized mean shift |mean_ref - mean_cur| / pooled_std per feature.
    Global = max. Deterministic, same windows/reference as KS/JS/W
    (fairness rule). NOT a full ADWIN implementation — named for family.
    """
    per, vals = {}, []
    for c in numeric_cols:
        r = ref[c].to_numpy(float)
        v = cur[c].to_numpy(float)
        scale = float(np.std(np.concatenate([r, v]))) + 1e-9
        dm = abs(float(r.mean()) - float(v.mean())) / scale
        if not np.isfinite(dm):
            dm = 0.0
        per[c] = {"ADWIN": float(dm)}
        vals.append(float(dm))
    return {"per_feature": per, "global_ADWIN": float(max(vals)) if vals else 0.0}


def mmd_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                     numeric_cols: list[str], max_n: int = 400,
                     seed: int = 0) -> dict:
    """Multivariate RBF-MMD (Greco/Roschewitz family), subsampled for speed.

    Unbiased MMD^2 with median-heuristic bandwidth on standardized features;
    max_n caps rows per side (deterministic subsample). Returns global score
    only (multivariate — no per-feature decomposition); ranking falls back
    to univariate detectors. Nonnegative up to float noise (clipped at 0).
    """
    rng = np.random.default_rng(seed)
    R = ref[numeric_cols].to_numpy(float)
    V = cur[numeric_cols].to_numpy(float)
    if len(R) > max_n:
        R = R[rng.choice(len(R), max_n, replace=False)]
    if len(V) > max_n:
        V = V[rng.choice(len(V), max_n, replace=False)]
    mu = R.mean(axis=0)
    sd = R.std(axis=0) + 1e-9
    Rz = (R - mu) / sd
    Vz = (V - mu) / sd
    Z = np.vstack([Rz, Vz])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    tri = d2[np.triu_indices(len(Z), k=1)]
    bw = float(np.median(tri)) + 1e-9
    if not np.isfinite(bw) or bw <= 0:
        return {"global_MMD": 0.0, "bandwidth": 0.0, "n": [len(R), len(V)]}

    def K(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        dd = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-dd / (2.0 * bw))

    m, n = len(Rz), len(Vz)
    Krr = K(Rz, Rz)
    Kvv = K(Vz, Vz)
    Krv = K(Rz, Vz)
    mmd2 = (Krr.sum() - np.trace(Krr)) / max(m * (m - 1), 1) + \
           (Kvv.sum() - np.trace(Kvv)) / max(n * (n - 1), 1) - \
           2.0 * Krv.mean()
    return {"global_MMD": float(max(mmd2, 0.0)), "bandwidth": bw,
            "n": [m, n]}


def joint_window_score(ref: pd.DataFrame, cur: pd.DataFrame,
                       numeric_cols: list[str], max_n: int = 1000,
                       seed: int = 0) -> dict:
    """Joint two-sample classifier detector (E2 extension, P2-4).

    Logistic regression distinguishes reference (0) vs current (1) windows;
    2-fold CV AUC in [0.5, 1.0]; ~0.5 = indistinguishable (no joint shift).
    Same windows/reference as univariate family (fairness rule). Multivariate
    complement to KS/JS/Wasserstein; S6/S7 joint cases are its design target.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    R = ref[numeric_cols].to_numpy(float)
    V = cur[numeric_cols].to_numpy(float)
    rng = np.random.default_rng(seed)
    if len(R) > max_n:
        R = R[rng.choice(len(R), max_n, replace=False)]
    if len(V) > max_n:
        V = V[rng.choice(len(V), max_n, replace=False)]
    X = np.vstack([R, V])
    y = np.array([0] * len(R) + [1] * len(V))
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd
    try:
        auc = float(cross_val_score(
            LogisticRegression(max_iter=500), Xz, y, cv=2,
            scoring="roc_auc").mean())
    except Exception:
        auc = 0.5
    if not np.isfinite(auc):
        auc = 0.5
    return {"auc": min(max(auc, 0.0), 1.0), "n": [len(R), len(V)]}


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
