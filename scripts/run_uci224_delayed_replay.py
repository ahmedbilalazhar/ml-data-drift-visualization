from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_uci224_day2 import (  # noqa: E402
    ARCHIVE, CLASSES, historical_split, load_cohorts,
)

DELAYS = (0, 1, 2)


def asof_recalibration(cal_conf, cal_correct, confidence, predictions, truth, delay):
    """Make each cohort forecast before releasing labels due at or after its clock."""
    if delay not in DELAYS:
        raise ValueError(f"delay must be one of {DELAYS}")
    if set(confidence) != set(predictions) or set(confidence) != set(truth):
        raise ValueError("cohort keys differ")
    keys = sorted(confidence)
    outputs, audits = {}, []
    initial_x = np.asarray(cal_conf, dtype=float).reshape(-1, 1)
    initial_y = np.asarray(cal_correct, dtype=int)
    if len(initial_x) != len(initial_y) or len(np.unique(initial_y)) != 2:
        raise ValueError("historical calibration needs aligned examples from both correctness classes")

    for batch in keys:
        eligible = [source for source in keys if source + delay < batch]
        x_parts = [initial_x]
        y_parts = [initial_y]
        for source in eligible:
            conf = np.asarray(confidence[source], dtype=float).reshape(-1, 1)
            pred = np.asarray(predictions[source])
            y = np.asarray(truth[source])
            if not (len(conf) == len(pred) == len(y)):
                raise ValueError(f"unaligned released cohort {source}")
            x_parts.append(conf)
            y_parts.append((pred == y).astype(int))
        fit_x, fit_y = np.concatenate(x_parts), np.concatenate(y_parts)
        unique = np.unique(fit_y)
        fallback = len(unique) < 2
        if fallback:
            # Keep a valid forecast if the released history contains only one correctness class.
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
            model.fit(initial_x, initial_y)
        else:
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
            model.fit(fit_x, fit_y)
        outputs[batch] = np.clip(model.predict_proba(
            np.asarray(confidence[batch], dtype=float).reshape(-1, 1))[:, 1], 0.0, 1.0)
        release_clocks = [source + delay for source in eligible]
        if any(clock >= batch for clock in release_clocks):
            raise AssertionError("current/future cohort outcome entered forecast")
        audits.append({
            "batch": int(batch), "delay_cohorts": int(delay),
            "available_source_batches": ";".join(map(str, eligible)),
            "available_monitoring_labels": int(sum(len(truth[s]) for s in eligible)),
            "fit_rows_including_historical_calibration": int(len(fit_y)),
            "max_release_clock": max(release_clocks) if release_clocks else None,
            "forecast_before_current_cohort_release": True,
            "single_class_history_fallback": bool(fallback),
        })
    return outputs, pd.DataFrame(audits)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source_run: Path, out: Path):
    if out.exists():
        raise FileExistsError(f"use a fresh versioned directory: {out}")
    source_run = source_run.resolve()
    source_lock_path = source_run / "protocol_lock.json"
    if not source_lock_path.is_file():
        raise FileNotFoundError(source_lock_path)
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if source_lock.get("status") not in {"complete", "executed"}:
        raise ValueError("source study is not a completed UCI 224 run")
    if _hash(ARCHIVE) != source_lock["dataset"]["archive_sha256"]:
        raise ValueError("UCI source archive hash differs from the locked source study")

    cohorts = load_cohorts()
    x_hist, y_hist = cohorts[1]
    train_idx, calibration_idx, reference_idx = historical_split(y_hist)
    scaler = StandardScaler().fit(x_hist[train_idx])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
    model.fit(scaler.transform(x_hist[train_idx]), y_hist[train_idx])
    p_cal = model.predict_proba(scaler.transform(x_hist[calibration_idx]))
    cal_conf = p_cal.max(axis=1)
    cal_pred = model.classes_[p_cal.argmax(axis=1)]
    cal_correct = (cal_pred == y_hist[calibration_idx]).astype(int)
    frozen_calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
    frozen_calibrator.fit(cal_conf.reshape(-1, 1), cal_correct)

    confidence, predictions, truth = {}, {}, {}
    for batch in range(2, 11):
        saved = np.load(source_run / f"batch{batch:02}_predictions.npz")
        conf = np.asarray(saved["confidence"], dtype=float)
        pred = np.asarray(saved["prediction"])
        x, y = cohorts[batch]
        if len(conf) != len(y) or len(pred) != len(y):
            raise ValueError(f"cached predictions do not align to source cohort {batch}")
        p_cal = model.predict_proba(scaler.transform(x))
        if not np.array_equal(pred, model.classes_[p_cal.argmax(axis=1)]):
            raise ValueError(f"fresh frozen-model predictions differ from admitted cache in cohort {batch}")
        if not np.array_equal(conf, p_cal.max(axis=1)):
            raise ValueError(f"fresh confidence values differ from admitted cache in cohort {batch}")
        expected_frozen = frozen_calibrator.predict_proba(conf.reshape(-1, 1))[:, 1]
        if not np.array_equal(saved["calibrated_correctness"], expected_frozen):
            raise ValueError(f"frozen calibration differs from admitted cache in cohort {batch}")
        confidence[batch], predictions[batch], truth[batch] = conf, pred, y

    out.mkdir(parents=True)
    predictions_dir = out / "predictions"
    truth_dir = out / "evaluator_truth"
    predictions_dir.mkdir()
    truth_dir.mkdir()
    truth_audit = []
    for batch in range(2, 11):
        conf = confidence[batch]
        frozen_q = frozen_calibrator.predict_proba(conf.reshape(-1, 1))[:, 1]
        np.savez_compressed(predictions_dir / f"batch{batch:02}_online_inputs.npz",
                            prediction=predictions[batch], confidence=conf,
                            frozen_q=np.clip(frozen_q, 0, 1))
        np.savez_compressed(truth_dir / f"batch{batch:02}_truth.npz", truth=truth[batch])
        truth_audit.append({"batch": batch, "rows": len(truth[batch]),
                            "class_ids": ";".join(map(str, sorted(np.unique(truth[batch]).tolist())))})

    audits = []
    for delay in DELAYS:
        q_by, audit = asof_recalibration(cal_conf, cal_correct, confidence,
                                         predictions, truth, delay)
        audit.to_csv(out / f"delay_{delay}_release_audit.csv", index=False)
        for batch, q in q_by.items():
            np.savez_compressed(predictions_dir / f"delay{delay}_batch{batch:02}_q.npz",
                                q_recalibrated=q)
        audit.insert(0, "method", "cumulative_logistic_recalibration")
        audits.append(audit)
    pd.concat(audits, ignore_index=True).to_csv(out / "fit_audit.csv", index=False)
    pd.DataFrame(truth_audit).to_csv(truth_dir / "cohort_manifest.csv", index=False)

    lock = {
        "study": "UCI 224 delayed cohort-label replay",
        "version": "uci224-delayed-replay-v1",
        "status": "online_forecasts_complete_evaluation_pending",
        "source_run": str(source_run),
        "source_run_lock_sha256": _hash(source_lock_path),
        "dataset_archive_sha256": _hash(ARCHIVE),
        "runner_sha256": _hash(Path(__file__).resolve()),
        "delays_in_whole_cohort_steps": list(DELAYS),
        "forecast_clock": "forecast batch b before release; source j eligible iff j+delay < b",
        "classifier_and_preprocessing_frozen": True,
        "historical_calibration_only_from_batch1": True,
        "updated_component": "one-dimensional logistic correctness calibrator only",
        "online_prediction_artifacts_exclude_truth": True,
        "truth_isolated_under": "evaluator_truth/",
        "current_future_label_invariance_test": "see tests/phase3/test_uci224_delayed_replay.py",
        "caveat": "offline simulation of coarse cohort releases, not observed label-arrival timestamps",
    }
    (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run, args.out)
