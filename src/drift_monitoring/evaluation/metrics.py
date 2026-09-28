"""Event/feature matching + performance metrics (roadmap Sec. 9.1-9.3, 8.5).

Matching: abrupt = first alarm within 5 windows of onset; gradual = start
through 5 windows after transition end. One-to-one event/episode match.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score


def match_events(alert_windows: list[int], onset_window: int | None,
                 gradual_end_window: int | None = None,
                 horizon: int = 5) -> dict:
    """Return {detected, delay_windows, matched_window, burden, false_pre}.

    burden = total alerted windows (repeated alarms counted, never double-credit).
    false_pre = alerts strictly BEFORE onset (genuinely false). Post-onset alerts
    beyond the matched one are correct persistence under a persistent shift, NOT
    false alarms — they appear in burden, never in false_pre.
    """
    burden = len(alert_windows)
    if onset_window is None:  # S0 stationary: every alert is genuinely false
        return {"detected": False, "delay_windows": None,
                "matched_window": None, "false_alerts": burden,
                "burden": burden, "false_pre": burden}
    lo, hi = onset_window, (gradual_end_window or onset_window) + horizon
    cands = sorted(w for w in alert_windows if lo <= w <= hi)
    false_pre = len([w for w in alert_windows if w < onset_window])
    if not cands:
        return {"detected": False, "delay_windows": None,
                "matched_window": None, "false_alerts": false_pre,
                "burden": burden, "false_pre": false_pre}
    return {"detected": True, "delay_windows": cands[0] - onset_window,
            "matched_window": cands[0], "false_alerts": false_pre,
            "burden": burden, "false_pre": false_pre}


def window_macro_f1(y_true, y_pred) -> float:
    if len(set(np.asarray(y_true).tolist())) < 2:
        return float("nan")  # single-class window: undefined, NOT zero
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def paired_ci(diffs: np.ndarray, n_boot: int = 2000,
              seed: int = 0) -> tuple[float, float]:
    """Block-note: diffs are per-STREAM paired differences (independent units).
    Within-stream windows are never bootstrapped as independent rows."""
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
             for _ in range(n_boot)]
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
