"""Direct distribution explanations (roadmap Sec. 9.2).

Core explanation = raw per-feature distance ranking (measured evidence).
Historical predictive importance is a SEPARATE weak comparator column —
never presented as drift evidence. Inferred features (DDE-style) are E5
extension only; this module always marks status='measured'.
"""
from __future__ import annotations
import numpy as np


def rank_features(detector_per_feature: dict, k: int = 3) -> dict:
    """Rank by KS D (or W_norm fallback). Returns full ranking + top-k."""
    rows = []
    for feat, v in detector_per_feature.items():
        score = float(v.get("D", v.get("W_norm", 0.0)))
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
