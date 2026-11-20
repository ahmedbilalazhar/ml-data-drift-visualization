"""Run the locked, outcome-informed released-label adaptation add-on."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.adaptation import (  # noqa: E402
    BATCHES, BATCH_SIZE, CumulativeLogisticAdapter, DELAYS, DEVELOPMENT_SEEDS,
    FINAL_SEEDS, LabelReleaseBroker, TASK_CONDITION, digest_ids, label_budget_before,
    make_stream,
)


ONLINE_FIELDS = (
    "task", "seed", "condition", "delay", "method", "cohort", "batch", "n",
    "available_monitoring_labels", "training_rows", "fit_count", "monitor_id_sha256",
    "max_available_at", "prediction_persisted", "probability_mean", "prediction_mean",
)
RELEASE_FIELDS = (
    "task", "seed", "condition", "delay", "source_batch", "available_at",
    "release_clock", "after_forecast_batch", "queue_draining", "n", "ids_sha256",
)
FIT_FIELDS = (
    "task", "seed", "condition", "delay", "method", "forecast_batch",
    "released_monitoring_n", "training_n", "fit_count", "monitor_ids_sha256",
    "max_available_at", "fit_performed_after_clock",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_lists(mode: str) -> dict[str, tuple[int, ...]]:
    if mode == "development":
        return {task: DEVELOPMENT_SEEDS for task, _ in TASK_CONDITION}
    if mode == "final":
        return {task: FINAL_SEEDS for task, _ in TASK_CONDITION}
    raise ValueError(mode)


def _as_json(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def run(mode: str, output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing adaptation run: {output}")
    seeds = _seed_lists(mode)
    output.mkdir(parents=True)
    (output / "features").mkdir()
    (output / "evaluator_truth").mkdir()
    (output / "predictions").mkdir()
    started = time.perf_counter()
    source_names = [
        "src/drift_monitoring/phase3/adaptation.py",
        "src/drift_monitoring/phase3/v2.py",
        "scripts/run_phase3_adaptation.py",
        "scripts/evaluate_phase3_adaptation.py",
        "docs/phase3/ADAPTATION_PROTOCOL.md",
    ]
    lock = dict(
        protocol="Released-label adaptation add-on",
        mode=mode,
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        seeds_by_task={key: list(value) for key, value in seeds.items()},
        conditions={task: condition for task, condition in TASK_CONDITION},
        delays=list(DELAYS), batch_size=BATCH_SIZE, batches=BATCHES,
        outcome_informed=True,
        source_sha256={name: _sha(ROOT / name) for name in source_names},
        prediction_order="persist all methods for current batch, then release labels at current clock",
    )
    (output / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    packages = ("numpy", "scikit-learn", "pandas", "threadpoolctl", "psutil")
    environment = dict(
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in packages},
        thread_limits={name: os.environ[name] for name in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        v1_final_archive_sha256=_sha(ROOT / "output/phase3/IDM_Viz_Phase3_v1_full_reproducibility.zip"),
    )
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    online_path = output / "online_batches.csv"
    release_path = output / "release_log.csv"
    released_path = output / "released_labels.csv"
    fit_path = output / "fit_audit.csv"
    online_file = online_path.open("w", encoding="utf-8", newline="")
    release_file = release_path.open("w", encoding="utf-8", newline="")
    released_file = released_path.open("w", encoding="utf-8", newline="")
    fit_file = fit_path.open("w", encoding="utf-8", newline="")
    online_writer = csv.DictWriter(online_file, fieldnames=ONLINE_FIELDS)
    release_writer = csv.DictWriter(release_file, fieldnames=RELEASE_FIELDS)
    released_writer = csv.DictWriter(released_file,
        fieldnames=("task", "seed", "condition", "delay", "source_batch", "available_at", "row_id", "label"))
    fit_writer = csv.DictWriter(fit_file, fieldnames=FIT_FIELDS)
    for writer in (online_writer, release_writer, released_writer, fit_writer):
        writer.writeheader()

    feature_stream_count = 0
    delay_stream_count = 0
    batch_records = 0
    try:
        with threadpool_limits(limits=1):
            for task, condition in TASK_CONDITION:
                for seed in seeds[task]:
                    train, calibration, reference, monitor = make_stream(seed, task, condition)
                    historical = [train, calibration, reference]
                    hashes = {"train": digest_ids(train.ids),
                              "calibration": digest_ids(calibration.ids),
                              "reference": digest_ids(reference.ids)}
                    short_task = "original" if task == "v1_synthetic" else "xor"
                    key = f"{seed}/{condition}"
                    feature_dir = output / "features" / short_task / str(seed)
                    truth_dir = output / "evaluator_truth" / short_task / str(seed)
                    feature_dir.mkdir(parents=True, exist_ok=True)
                    truth_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(feature_dir / f"{condition}.npz", ids=monitor.ids, x=monitor.x)
                    np.savez_compressed(truth_dir / f"{condition}.npz", ids=monitor.ids, y=monitor.y)
                    (output / "features" / short_task / str(seed) / f"{condition}_partitions.json").write_text(
                        json.dumps({"n_train": len(train.y), "n_calibration": len(calibration.y),
                                    "n_reference": len(reference.y), "n_monitoring": len(monitor.y),
                                    "ids_sha256": hashes}, indent=2), encoding="utf-8")
                    feature_stream_count += 1

                    scaler = StandardScaler().fit(train.x)
                    train_x = scaler.transform(train.x)
                    frozen = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
                    frozen.fit(train_x, train.y)

                    for delay in DELAYS:
                        broker = LabelReleaseBroker(monitor, delay)
                        adapter = CumulativeLogisticAdapter(seed, train)
                        pred_dir = output / "predictions" / short_task / str(seed) / condition / f"delay{delay}"
                        pred_dir.mkdir(parents=True, exist_ok=True)
                        received_ids: list[np.ndarray] = []
                        received_y: list[np.ndarray] = []
                        due_batches = set()

                        def log_release(clock: int, draining: bool = False) -> None:
                            events = broker.release(clock)
                            for event in events:
                                if event.source_batch in due_batches:
                                    raise RuntimeError("duplicate release event")
                                due_batches.add(event.source_batch)
                                release_writer.writerow({
                                    "task": task, "seed": seed, "condition": condition,
                                    "delay": delay, "source_batch": event.source_batch,
                                    "available_at": event.available_at,
                                    "release_clock": clock,
                                    "after_forecast_batch": (clock if clock <= BATCHES else BATCHES),
                                    "queue_draining": bool(draining),
                                    "n": len(event.ids), "ids_sha256": digest_ids(event.ids),
                                })
                                for row_id, label in zip(event.ids, event.y):
                                    released_writer.writerow({
                                        "task": task, "seed": seed, "condition": condition,
                                        "delay": delay, "source_batch": event.source_batch,
                                        "available_at": event.available_at, "row_id": int(row_id),
                                        "label": int(label),
                                    })
                                release_file.flush(); os.fsync(release_file.fileno())
                                released_file.flush(); os.fsync(released_file.fileno())
                                received_ids.append(event.ids.copy())
                                received_y.append(event.y.copy())
                                adapter.add_released([event])

                        for batch in range(1, BATCHES + 1):
                            start = (batch - 1) * BATCH_SIZE
                            stop = start + BATCH_SIZE
                            x = monitor.x[start:stop]
                            ids = monitor.ids[start:stop]
                            frozen_probability = frozen.predict_proba(scaler.transform(x))[:, 1]
                            frozen_prediction = (frozen_probability >= 0.5).astype(np.int8)
                            adapted_prediction, adapted_probability = adapter.predict(x)
                            expected_budget = label_budget_before(batch, delay)
                            adapted_audit = adapter.audit(batch)
                            if adapted_audit["released_monitoring_n"] != expected_budget:
                                raise AssertionError("released-label budget does not match locked clock")
                            arrays = dict(ids=ids, frozen_prediction=frozen_prediction,
                                          frozen_probability=frozen_probability,
                                          adapted_prediction=adapted_prediction,
                                          adapted_probability=adapted_probability)
                            # This closed NPZ is the persisted score artifact. Labels are released below.
                            np.savez_compressed(pred_dir / f"batch{batch:02}.npz", **arrays)
                            for method, probability, prediction, audit in (
                                ("frozen", frozen_probability, frozen_prediction,
                                 dict(training_rows=len(train.y), released_monitoring_n=0,
                                      monitor_id_sha256=digest_ids([]), max_available_at=None, fit_count=1)),
                                ("cumulative_refit", adapted_probability, adapted_prediction, adapted_audit),
                            ):
                                cohort = f"{task}:{seed}:{condition}:{delay}:{batch}:{method}"
                                online_writer.writerow(dict(
                                    task=task, seed=seed, condition=condition, delay=delay, method=method,
                                    cohort=cohort, batch=batch, n=BATCH_SIZE,
                                    available_monitoring_labels=expected_budget,
                                    training_rows=audit["training_rows"] if "training_rows" in audit else audit["training_n"],
                                    fit_count=audit["fit_count"], monitor_id_sha256=audit["monitor_id_sha256"],
                                    max_available_at=audit["max_available_at"], prediction_persisted=True,
                                    probability_mean=float(probability.mean()),
                                    prediction_mean=float(prediction.mean())))
                                fit_writer.writerow(dict(
                                    task=task, seed=seed, condition=condition, delay=delay,
                                    method=method, forecast_batch=batch,
                                    released_monitoring_n=audit["released_monitoring_n"],
                                    training_n=audit["training_rows"] if "training_rows" in audit else audit["training_n"],
                                    fit_count=audit["fit_count"], monitor_ids_sha256=audit["monitor_id_sha256"],
                                    max_available_at=audit["max_available_at"],
                                    fit_performed_after_clock=(audit["max_available_at"]
                                        if method == "cumulative_refit" else None)))
                                batch_records += 1
                            online_file.flush(); os.fsync(online_file.fileno())
                            fit_file.flush(); os.fsync(fit_file.fileno())
                            log_release(batch)

                        for clock in range(BATCHES + 1, BATCHES + delay + 1):
                            log_release(clock, draining=True)
                        if due_batches != set(range(1, BATCHES + 1)):
                            raise AssertionError("release queue failed to drain")
                        if len(np.concatenate(received_ids)) != BATCH_SIZE * BATCHES:
                            raise AssertionError("incorrect total label budget after drain")
                        delay_stream_count += 1
                    if seed % 5 == 0:
                        print(f"{task} seed {seed}: completed", flush=True)
    except Exception:
        lock["status"] = "failed"
        lock["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        lock["elapsed_seconds"] = time.perf_counter() - started
        (output / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
        raise
    finally:
        online_file.close(); release_file.close(); released_file.close(); fit_file.close()

    lock.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter() - started,
                feature_streams=feature_stream_count,
                delay_streams=delay_stream_count,
                model_batch_records=batch_records)
    (output / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({k: _as_json(v) for k, v in lock.items()
                      if k in ("status", "elapsed_seconds", "feature_streams", "delay_streams",
                               "model_batch_records")}, indent=2))
    return lock


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.out)
