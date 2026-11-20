from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]


def ece10(q, correct):
    q = np.asarray(q, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    total = 0.0
    for i in range(10):
        mask = (q >= edges[i]) & (q < edges[i + 1] if i < 9 else q <= edges[i + 1])
        if mask.any():
            total += mask.mean() * abs(float(q[mask].mean() - correct[mask].mean()))
    return float(total)


def evaluate(run: Path):
    run = run.resolve()
    lock_path = run / "protocol_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "online_forecasts_complete_evaluation_pending":
        raise ValueError("the run is not awaiting evaluation")
    rows = []
    for batch in range(2, 11):
        pred_cache = np.load(run / "predictions" / f"batch{batch:02}_online_inputs.npz")
        y = np.load(run / "evaluator_truth" / f"batch{batch:02}_truth.npz")["truth"]
        pred = pred_cache["prediction"]
        conf = pred_cache["confidence"]
        frozen = pred_cache["frozen_q"]
        correct = (pred == y).astype(int)
        acc = float(accuracy_score(y, pred))
        ref_acc = float(pd.read_csv(ROOT / "results/phase3/external_uci224_day2_v2/cohort_metrics.csv")
                        .loc[lambda frame: frame.batch.eq(1), "accuracy"].iloc[0])
        if not (len(pred) == len(y) == len(conf) == len(frozen)):
            raise ValueError(f"cohort {batch} evaluator arrays are misaligned")
        for delay in (0, 1, 2):
            audit = pd.read_csv(run / f"delay_{delay}_release_audit.csv")
            a = audit.loc[audit.batch.eq(batch)].iloc[0]
            if a.max_release_clock is not None and not pd.isna(a.max_release_clock) and int(a.max_release_clock) >= batch:
                raise ValueError("current or future released labels appear in fit audit")
            q_online = np.load(run / "predictions" / f"delay{delay}_batch{batch:02}_q.npz")["q_recalibrated"]
            methods = {
                "frozen_calibration": frozen,
                "cumulative_released_logistic": q_online,
                "raw_average_confidence": conf,
                "historical_reference_accuracy": np.full(len(y), ref_acc),
            }
            for method, q in methods.items():
                q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
                rows.append({
                    "delay_cohorts": delay, "batch": batch, "method": method,
                    "n": len(y), "accuracy": acc,
                    "estimated_accuracy": float(q.mean()),
                    "signed_estimation_error": float(q.mean() - acc),
                    "absolute_estimation_error": float(abs(q.mean() - acc)),
                    "brier_correctness": float(np.mean((q - correct) ** 2)),
                    "ece10": ece10(q, correct),
                    "available_monitoring_labels": int(a.available_monitoring_labels),
                    "available_source_batches": str(a.available_source_batches),
                    "forecast_before_current_cohort_release": True,
                })
    per = pd.DataFrame(rows)
    out = run / "evaluation"
    out.mkdir(exist_ok=False)
    per.to_csv(out / "per_batch_metrics.csv", index=False)
    summaries = []
    for (delay, method), group in per.groupby(["delay_cohorts", "method"], sort=True):
        summaries.append({
            "delay_cohorts": int(delay), "method": method, "batches": int(group.batch.nunique()),
            "total_rows": int(group.n.sum()),
            "equal_batch_mae": float(group.absolute_estimation_error.mean()),
            "equal_batch_bias": float(group.signed_estimation_error.mean()),
            "row_weighted_brier": float(np.average(group.brier_correctness, weights=group.n)),
            "equal_batch_ece10": float(group.ece10.mean()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    # Recheck the release contract from persisted audit data.
    audit = pd.read_csv(run / "fit_audit.csv")
    for row in audit.itertuples():
        if row.max_release_clock == row.max_release_clock and int(row.max_release_clock) >= int(row.batch):
            raise AssertionError("fit audit violates strict release-before-forecast ordering")
    validation = {
        "status": "passed", "forecast_rows": int(len(per)), "target_cohorts": 9,
        "delay_values": [0, 1, 2], "methods": sorted(per.method.unique().tolist()),
        "strict_release_clock_passed": True,
        "online_prediction_cache_has_no_truth": all(
            "truth" not in np.load(p).files
            for p in (run / "predictions").glob("*.npz")),
        "same_predictions_across_methods_and_delays": True,
        "interpretation": "one observationally ordered dataset sequence; cohort-level release simulation only",
        "summary": summary.to_dict(orient="records"),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    lock["status"] = "complete"
    lock["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (run / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run)
