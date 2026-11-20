"""Run the frozen chronological NOAA Weather transfer evaluation.

This is a retrospective same-day rain-classification benchmark using completed
daily summaries. It is not an operational weather forecast. The online scoring
path receives features/IDs/timestamps only; truth enters LabelBroker after each
prediction batch has been written.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.contracts import (  # noqa: E402
    CheckpointedRun, atomic_write_bytes, atomic_write_json, canonical_hash, sha256_file,
)
from drift_monitoring.phase3.core import FeatureBatch, Partition, fit_artifacts, score_batch  # noqa: E402
from drift_monitoring.phase3.replay import LabelBroker, replay  # noqa: E402

FEATURES = ("temp", "dewp", "slp", "visib", "wdsp", "mxspd", "max", "min")
BATCH_SIZE = 500


def split_counts(n: int) -> dict[str, int]:
    """Chronological 20/10/10/60 split using explicit floor allocation."""
    if n < 10:
        raise ValueError("at least 10 eligible observations are required")
    train = int(np.floor(.20 * n))
    calibration = int(np.floor(.10 * n))
    reference = int(np.floor(.10 * n))
    monitoring = n - train - calibration - reference
    return {"train": train, "calibration": calibration, "reference": reference,
            "monitoring": monitoring}


def _csv_bytes(rows) -> bytes:
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def evaluate(processed_path: Path, manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(processed_path) != manifest["processed"]["sha256"]:
        raise ValueError("processed Weather checksum differs from its source manifest")
    df = pd.read_csv(processed_path, parse_dates=["timestamp"])
    if list(c.lower() for c in manifest["preprocessing"]["feature_columns"]) != list(FEATURES):
        raise ValueError("Weather feature schema does not match the runner")
    if (len(df) != manifest["processed"]["row_count"] or df.timestamp.isna().any()
            or not df.timestamp.is_monotonic_increasing or df.timestamp.duplicated().any()
            or not np.array_equal(df.row_id.to_numpy(), np.arange(len(df)))):
        raise ValueError("processed Weather chronology, row count, or stable IDs failed validation")
    x = df.loc[:, FEATURES].to_numpy(dtype=float)
    y = df.label_rain.to_numpy(dtype=np.int8)
    ids = df.row_id.to_numpy(dtype=np.int64)
    counts = split_counts(len(df))
    cuts = np.cumsum([counts[k] for k in ("train", "calibration", "reference")])
    a, b, c = map(int, cuts)
    train = Partition(x[:counts["train"]], y[:counts["train"]], ids[:counts["train"]])
    calibration = Partition(x[a:b], y[a:b], ids[a:b])
    reference = Partition(x[b:c], y[b:c], ids[b:c])
    monitor = df.iloc[c:].reset_index(drop=True)
    full_batches = len(monitor) // BATCH_SIZE
    used_rows = full_batches * BATCH_SIZE
    discarded = len(monitor) - used_rows
    if full_batches < 2:
        raise ValueError("monitoring stream must contain at least two complete 500-row batches")
    monitor = monitor.iloc[:used_rows].reset_index(drop=True)
    mx = monitor.loc[:, FEATURES].to_numpy(dtype=float)
    my = monitor.label_rain.to_numpy(dtype=np.int8)
    mids = monitor.row_id.to_numpy(dtype=np.int64)

    fitted = fit_artifacts(seed=0, train=train, calibration=calibration, reference=reference,
                           n_pca_components=3, n_clusters=3, model_max_iter=2000)
    online, caches = [], []
    for batch_idx in range(full_batches):
        lo, hi = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        batch = FeatureBatch(mx[lo:hi], mids[lo:hi], seed=0, condition="NOAA_GSOD_Weather",
                             batch=batch_idx + 1)
        record, cache, _ = score_batch(fitted, batch)
        record.update(timestamp_start=monitor.timestamp.iloc[lo].strftime("%Y-%m-%d"),
                      timestamp_end=monitor.timestamp.iloc[hi - 1].strftime("%Y-%m-%d"),
                      source_rows=int(hi - lo))
        online.append(record)
        caches.append({"ids": cache["ids"], "prediction": cache["prediction"],
                       "correctness_probability": cache["correctness_probability"]})

    # Persist the label-free online artifact before constructing the broker.
    lock = CheckpointedRun(output)
    run_contract = {"protocol": "noaa-weather-chronological-holdout-v2",
                    "source_data_sha256": sha256_file(processed_path), "feature_schema": FEATURES,
                    "split_counts": counts, "batch_size": BATCH_SIZE, "complete_batches": full_batches,
                    "delay_batches": 0, "model": "LogisticRegression(C=1,max_iter=2000), frozen",
                    "calibration": "historical calibration partition only; isotonic correctness map"}
    source_hashes = {"processed_weather": sha256_file(processed_path),
                     "weather_manifest": sha256_file(manifest_path),
                     "runner": sha256_file(Path(__file__)),
                     "core": sha256_file(ROOT / "src/drift_monitoring/phase3/core.py"),
                     "replay": sha256_file(ROOT / "src/drift_monitoring/phase3/replay.py"),
                     "phase2_docx": sha256_file(ROOT / "docs/23i-2604_23i-2581DAV-phase2.docx")}
    lock.initialize(config_digest=canonical_hash(run_contract), source_hashes=source_hashes,
                    expected_cells=["label_free_online_scoring", "released_label_events",
                                    "released_label_verification", "evaluator_batch_metrics"])
    online_path = output / "online" / "batch_scores.csv"
    atomic_write_bytes(online_path, _csv_bytes(online))
    lock.commit_cell("label_free_online_scoring", online_path)

    # Evaluator truth is supplied only after all current batch scores are persisted.
    broker = LabelBroker(mids, my, BATCH_SIZE, delay=0)
    events, verified = replay(online, caches, broker)
    verification_path = output / "evaluator" / "released_verification.csv"
    atomic_write_bytes(verification_path, _csv_bytes(verified))
    lock.commit_cell("released_label_verification", verification_path)
    event_path = output / "run" / "label_release_events.csv"
    atomic_write_bytes(event_path, _csv_bytes(events))
    lock.commit_cell("released_label_events", event_path)

    metrics = []
    for row, result in zip(online, verified, strict=True):
        lo = (int(row["batch"]) - 1) * BATCH_SIZE
        hi = lo + BATCH_SIZE
        truth, pred = my[lo:hi], caches[row["batch"] - 1]["prediction"]
        tn, fp, fn, tp = map(int, confusion_matrix(truth, pred, labels=[0, 1]).ravel())
        q = caches[row["batch"] - 1]["correctness_probability"]
        correct = (truth == pred).astype(float)
        metrics.append({"batch": row["batch"], "timestamp_start": row["timestamp_start"],
                        "timestamp_end": row["timestamp_end"], "n": BATCH_SIZE,
                        "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                        "rain_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                        "rain_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "rain_prevalence": float(truth.mean()),
                        "calibrated_accuracy_estimate": float(row["cal_estimate"]),
                        "raw_average_confidence_estimate": float(row["raw_estimate"]),
                        "atc_estimate": float(row["atc_estimate"]),
                        "historical_constant_estimate": float(row["constant_estimate"]),
                        "calibration_mae": abs(float(row["cal_estimate"]) - result["accuracy"]),
                        "calibration_bias": float(row["cal_estimate"]) - result["accuracy"],
                        "correctness_brier": float(np.mean((q - correct) ** 2)),
                        "plug_in_interval_lower": float(row["cal_lower"]),
                        "plug_in_interval_upper": float(row["cal_upper"]),
                        "plug_in_interval_covers": bool(row["cal_lower"] <= result["accuracy"] <= row["cal_upper"]),
                        "input_alarm": bool(row["input_alarm"]),
                        "available_at": result["available_at"], "evidence_age_batches": result["evidence_age_batches"]})
    metrics_path = output / "evaluator" / "batch_metrics.csv"
    atomic_write_bytes(metrics_path, _csv_bytes(metrics))
    lock.commit_cell("evaluator_batch_metrics", metrics_path)
    summary = {
        "status": "complete", "run_id": "noaa-weather-1949-1999-v2",
        "dataset_id": manifest["dataset_id"], "data_sha256": sha256_file(processed_path),
        "row_counts": {**counts, "monitoring_complete_batches": full_batches,
                       "monitoring_evaluated_rows": used_rows, "monitoring_remainder_dropped": discarded},
        "date_ranges": {"train": [df.timestamp.iloc[0].strftime("%Y-%m-%d"), df.timestamp.iloc[counts["train"]-1].strftime("%Y-%m-%d")],
                        "calibration": [df.timestamp.iloc[counts["train"]].strftime("%Y-%m-%d"), df.timestamp.iloc[a-1].strftime("%Y-%m-%d")],
                        "reference": [df.timestamp.iloc[a].strftime("%Y-%m-%d"), df.timestamp.iloc[b-1].strftime("%Y-%m-%d")],
                        "monitoring": [monitor.timestamp.iloc[0].strftime("%Y-%m-%d"), monitor.timestamp.iloc[-1].strftime("%Y-%m-%d")]},
        "primary_holdout_summary": {"mean_batch_accuracy": float(np.mean([r["accuracy"] for r in metrics])),
                                    "mean_batch_macro_f1": float(np.mean([r["macro_f1"] for r in metrics])),
                                    "mean_batch_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in metrics])),
                                    "mean_calibration_mae": float(np.mean([r["calibration_mae"] for r in metrics])),
                                    "mean_calibration_bias": float(np.mean([r["calibration_bias"] for r in metrics])),
                                    "mean_correctness_brier": float(np.mean([r["correctness_brier"] for r in metrics])),
                                    "plug_in_interval_coverage": float(np.mean([r["plug_in_interval_covers"] for r in metrics])),
                                    "input_alarm_fraction": float(np.mean([r["input_alarm"] for r in metrics]))},
        "interpretation": "descriptive chronological holdout on one station; batches are serial temporal blocks, not independent replications",
        "timing_boundary": "features scored and written first; delay-zero labels released after the batch prediction; queue drainage creates no input",
        "label_timing_limitation": "Features and rain occurrence are same-day GSOD summaries. This benchmark is retrospective classification and cannot support an ahead-of-time rain forecast.",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "model": {"name": "LogisticRegression", "C": 1.0, "max_iter": 2000,
                  "frozen": True, "actual_pca_components": int(fitted.pca.n_components_),
                  "clusters": int(fitted.clusters.n_clusters)},
        "split_seed": None,
    }
    atomic_write_json(output / "run_summary.json", summary)
    atomic_write_json(output / "run_contract.json", run_contract)
    lock.complete()
    lock.lock["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(lock.lock_path, lock.lock)
    integrity = {p.relative_to(output).as_posix(): sha256_file(p)
                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "integrity_manifest.json"}
    atomic_write_json(output / "integrity_manifest.json", {
        "run_id": summary["run_id"], "protocol_lock_sha256": sha256_file(lock.lock_path),
        "files_sha256": integrity,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/noaa_weather_1949_1999_v1")
    args = parser.parse_args()
    result = evaluate(args.data.resolve(), args.manifest.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "rows": result["row_counts"], "summary": result["primary_holdout_summary"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
