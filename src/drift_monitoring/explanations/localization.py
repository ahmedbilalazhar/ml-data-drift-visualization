"""Direct distribution explanations (roadmap Sec. 9.2).

Core explanation = raw per-feature distance ranking (measured evidence).
Historical predictive importance is a SEPARATE weak comparator column —
never presented as drift evidence. Inferred features (DDE-style) are E5
extension only; this module always marks status='measured'.
"""
from __future__ import annotations
import numpy as np


def rank_features(detector_per_feature: dict, k: int = 3) -> dict:
    """Rank by KS D (or JS / ADWIN / W_norm fallback). Full ranking + top-k."""
    rows = []
    for feat, v in detector_per_feature.items():
        score = float(v.get("D", v.get("JS", v.get("ADWIN", v.get("W_norm", 0.0)))))
        rows.append((feat, score))
    rows.sort(key=lambda t: -t[1])
    return {"ranking": [f for f, _ in rows],
            "scores": {f: s for f, s in rows},
            f"top{k}": [f for f, _ in rows[:k]],
            "status": "measured"}


def localization_scores(pred_ranking: list[str], true_set: set[str],
                        k: int = 3) -> dict:
    """Precision/recall/F1@k + average precision vs independent truth."""
    pred_k = pred_ranking[:k]
    tp = len(set(pred_k) & true_set)
    prec = tp / max(len(pred_k), 1)
    rec = tp / max(len(true_set), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # average precision over full ranking
    ap_num, hits = 0.0, 0
    for i, f in enumerate(pred_ranking, 1):
        if f in true_set:
            hits += 1
            ap_num += hits / i
    ap = ap_num / max(len(true_set), 1)
    return {"p@k": prec, "r@k": rec, "f1@k": f1, "ap": ap,
            "empty_truth": len(true_set) == 0}


def fit_dde_curve(corr_abs, drifted: set, monitored: list,
                  thresholds: list | None = None) -> dict:
    """Domino Drift Effect lift curve (Szucs DDE, transparent proxy).

    corr_abs: DataFrame of |Pearson rho| on reference data (or dict-like).
    drifted: set of monitored features flagged drifted at the fitting drift.
    Curve DDE(z) = fraction of monitored pairs with |rho|>=z where BOTH
    drifted. Lift = DDE(z)/DDE(0). Flat curve => independent drift (no DDE),
    upward curve => co-drifting (cf. Szucs Covertype vs Insects/Heartbeats).
    """
    import pandas as pd
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    drifted = set(drifted) & set(monitored)
    curve, lift = {}, {}
    base = None
    for z in thresholds:
        tot, both = 0, 0
        for i, a in enumerate(monitored):
            for b in monitored[i + 1:]:
                try:
                    rho = abs(float(corr_abs.loc[a, b]))
                except Exception:
                    rho = abs(float(corr_abs[a][b]))
                if rho >= z:
                    tot += 1
                    if a in drifted and b in drifted:
                        both += 1
        p = both / tot if tot else 0.0
        curve[float(z)] = float(p)
        if base is None:
            base = p
    for z, p in curve.items():
        lift[z] = float(p / base) if base else 0.0
    return {"curve": curve, "lift": lift, "n_monitored": len(monitored),
            "n_drifted": len(drifted), "status": "inferred-dde"}


def dde_scores(curve: dict, corr_abs, drifted_monitored: set,
               unmonitored: list) -> dict:
    """DDE score per unmonitored feature (Szucs Eq.25 averaging).

    score(fi) = mean over drifted monitored fj of DDE(|rho_ij|), where
    DDE(.) is the stepwise-fitted curve (largest threshold <= |rho|).
    Marked status='inferred-dde', never 'measured'.
    """
    drifted_monitored = set(drifted_monitored)
    zs = sorted(curve.keys())
    out = {}
    for fi in unmonitored:
        vals = []
        for fj in drifted_monitored:
            try:
                rho = abs(float(corr_abs.loc[fi, fj]))
            except Exception:
                rho = abs(float(corr_abs[fi][fj]))
            cand = [z for z in zs if z <= rho]
            vals.append(curve[max(cand)] if cand else curve[zs[0]])
        out[fi] = float(sum(vals) / len(vals)) if vals else 0.0
    return {"scores": out, "status": "inferred-dde"}
