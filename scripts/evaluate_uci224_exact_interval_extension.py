"""Compare exact and normal plug-in Poisson-binomial intervals on cached UCI rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from drift_monitoring.phase3.core import poisson_binomial
from evaluate_uci224_day2 import load_cohorts


def interval_exact(q: np.ndarray) -> tuple[float, float]:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or len(q) == 0 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("q must be a non-empty vector of finite probabilities")
    pmf = poisson_binomial(q)
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return tuple((np.searchsorted(cdf, [0.025, 0.975]) / len(q)).tolist())


def interval_normal(q: np.ndarray) -> tuple[float, float]:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or len(q) == 0 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("q must be a non-empty vector of finite probabilities")
    n = len(q)
    mean = float(q.sum())
    sd = float(np.sqrt(np.sum(q * (1.0 - q))))
    lo = max(0.0, mean - 0.5 - norm.ppf(0.975) * sd) / n
    hi = min(float(n), mean + 0.5 + norm.ppf(0.975) * sd) / n
    return lo, hi


def main(run: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"choose a fresh output directory: {out}")
    lock_path = run / "protocol_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    archive = ROOT / lock["dataset"]["archive"]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != lock["dataset"]["archive_sha256"]:
        raise ValueError("UCI source archive hash does not match the locked study")
    cohorts = load_cohorts()
    primary = pd.read_csv(run / "cohort_metrics.csv").set_index("batch")
    rows = []
    for batch in range(2, 11):
        stored = np.load(run / f"batch{batch:02}_predictions.npz")
        q = np.asarray(stored["calibrated_correctness"], dtype=float)
        prediction = np.asarray(stored["prediction"])
        y = cohorts[batch][1]
        if q.ndim != 1 or len(q) != len(y) or len(prediction) != len(y):
            raise ValueError(f"cached cohort shape mismatch for batch {batch}")
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError(f"invalid q values in batch {batch}")
        accuracy = float(np.mean(prediction == y))
        exact_lo, exact_hi = interval_exact(q)
        normal_lo, normal_hi = interval_normal(q)
        old = primary.loc[batch]
        if not np.isclose(accuracy, float(old.accuracy), rtol=0, atol=1e-12):
            raise ValueError(f"cached prediction alignment mismatch for batch {batch}")
        rows.append({
            "batch": batch, "n": len(y), "accuracy": accuracy,
            "estimate": float(q.mean()), "exact_lower": exact_lo, "exact_upper": exact_hi,
            "exact_width": exact_hi - exact_lo,
            "exact_covered": exact_lo <= accuracy <= exact_hi,
            "normal_lower": normal_lo, "normal_upper": normal_hi,
            "normal_width": normal_hi - normal_lo,
            "normal_covered": normal_lo <= accuracy <= normal_hi,
            "width_difference_exact_minus_normal": (exact_hi - exact_lo) - (normal_hi - normal_lo),
            "lower_difference_exact_minus_normal": exact_lo - normal_lo,
            "upper_difference_exact_minus_normal": exact_hi - normal_hi,
        })
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "cohort_interval_comparison.csv", index=False)
    summary = {
        "status": "exploratory_complete", "cohorts": len(rows),
        "cohort_unit_warning": "nine sequential cohorts from one dataset are not independent replications",
        "exact_coverage_count": int(frame.exact_covered.sum()),
        "normal_coverage_count": int(frame.normal_covered.sum()),
        "mean_exact_width": float(frame.exact_width.mean()),
        "mean_normal_width": float(frame.normal_width.mean()),
        "mean_width_difference_exact_minus_normal": float(frame.width_difference_exact_minus_normal.mean()),
        "mean_absolute_endpoint_difference": float(np.mean(np.abs(frame.lower_difference_exact_minus_normal) +
                                                              np.abs(frame.upper_difference_exact_minus_normal)) / 2),
        "calibration_transfer_error_unchanged": float(np.mean(np.abs(frame.estimate - frame.accuracy))),
        "source_sha256": digest,
        "limitations": ["plug-in q treated as known", "independent correctness assumption",
                        "no reference or calibration parameter uncertainty", "no arbitrary-shift guarantee"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "protocol_copy.md").write_text(
        (ROOT / "docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md").read_text(encoding="utf-8"),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "results/phase3/external_uci224_day2_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
