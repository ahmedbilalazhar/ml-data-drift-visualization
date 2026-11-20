"""Feature-only dependence change statistic for the bounded follow-up study."""
from __future__ import annotations

import numpy as np


def max_correlation_change(reference: np.ndarray, current: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return the largest absolute pairwise correlation change and its pair.

    The score uses only feature matrices. Threshold selection and operating
    characteristics belong to the experiment protocol, not this function.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if (reference.ndim != 2 or current.ndim != 2 or reference.shape[1] != current.shape[1]
            or reference.shape[1] < 2 or len(reference) < 3 or len(current) < 3
            or not np.isfinite(reference).all() or not np.isfinite(current).all()):
        raise ValueError("reference and current inputs must be finite 2D matrices with matching width and at least 3 rows")
    if np.any(np.ptp(reference, axis=0) == 0) or np.any(np.ptp(current, axis=0) == 0):
        raise ValueError("correlation change is undefined for a constant feature")
    delta = np.abs(np.corrcoef(current, rowvar=False) - np.corrcoef(reference, rowvar=False))
    np.fill_diagonal(delta, -np.inf)
    i, j = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return float(delta[i, j]), (min(int(i), int(j)), max(int(i), int(j)))


def calibrate_threshold(null_scores: np.ndarray, quantile: float = 0.95) -> float:
    """Calibrate an empirical maximum-pair threshold on development null scores."""
    scores = np.asarray(null_scores, dtype=float)
    if (scores.ndim != 1 or len(scores) < 20 or not np.isfinite(scores).all()
            or not 0.5 < quantile < 1.0):
        raise ValueError("need at least 20 finite null scores and a quantile in (0.5, 1)")
    return float(np.quantile(scores, quantile, method="higher"))


def calibrate_reference_threshold(reference: np.ndarray, calibration: np.ndarray, *,
                                  batch_size: int, random_state: int, n_null: int = 100,
                                  quantile: float = 0.95) -> float:
    """Calibrate a maximum-correlation threshold from historical inputs only.

    The reference cohort is fixed; pseudo-current cohorts are sampled from the
    separate historical calibration partition at the monitoring batch size.
    Labels are intentionally not accepted by this function.
    """
    reference = np.asarray(reference, dtype=float)
    calibration = np.asarray(calibration, dtype=float)
    if (reference.ndim != 2 or calibration.ndim != 2 or
            reference.shape[1] != calibration.shape[1] or
            batch_size < 3 or len(calibration) < batch_size or n_null < 20):
        raise ValueError("invalid historical inputs or insufficient null cohorts")
    rng = np.random.default_rng(random_state)
    scores = np.empty(n_null, dtype=float)
    for i in range(n_null):
        indices = rng.choice(len(calibration), size=batch_size, replace=False)
        scores[i] = max_correlation_change(reference, calibration[indices])[0]
    return calibrate_threshold(scores, quantile)
