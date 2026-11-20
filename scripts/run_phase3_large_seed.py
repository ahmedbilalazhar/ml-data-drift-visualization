"""Run the locked aggregate-only, large-seed v2 robustness study."""
from __future__ import annotations

import argparse
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

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.v2 import (  # noqa: E402
    BATCHES, BATCH_SIZE, MODELS, TASKS, V1_CONDITIONS, XOR_CONDITIONS,
    FeatureBatch, fit_models, historical_partition, monitoring_partition, score_model,
)
from drift_monitoring.phase3.core import is_harm  # noqa: E402

MANIFEST = ROOT / "docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md"
SEEDS = tuple(range(1000, 2500))
DEV_SEEDS = tuple(range(990001, 990006))
ONLINE_FORBIDDEN = {"label", "y", "accuracy", "macro_f1", "correct", "harm", "truth", "f1"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(mode: str, out: Path) -> None:
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite robustness output: {out}")
    seeds = DEV_SEEDS if mode == "development" else SEEDS if mode == "final" else None
    if seeds is None:
        raise ValueError(mode)
    if set(seeds) & (set(range(100, 130)) | set(range(200, 230)) | set(range(300, 330)) |
                     set(range(190001, 190006)) | set(range(290001, 290006))):
        raise RuntimeError("seed collision with existing evidence")
    out.mkdir(parents=True)
    conditions = {"v1_synthetic": V1_CONDITIONS, "xor_synthetic": XOR_CONDITIONS}
    lock = {
        "protocol": "large-seed robustness replication of frozen model/task extension",
        "status": "running", "mode": mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds_by_task": {task: list(seeds) for task in TASKS},
        "conditions_by_task": {task: list(conditions[task]) for task in TASKS},
        "models": list(MODELS), "seed_count_per_task": len(seeds),
        "manifest_sha256": sha(MANIFEST),
        "source_sha256": {p: sha(ROOT / p) for p in (
            "src/drift_monitoring/phase3/v2.py", "src/drift_monitoring/phase3/core.py",
            "scripts/run_phase3_large_seed.py", "scripts/evaluate_phase3_large_seed.py",
            "docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md")},
        "thread_limit": 1, "output_contract": "online batch features-only; evaluator batch truth separate",
        "cache_policy": "aggregate-only deterministic regeneration; no per-instance cache",
    }
    (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    packages = ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in packages},
                   "thread_limit": 1}
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    online_fields = None
    evaluator_fields = ("task", "seed", "condition", "model", "batch", "n", "correct",
                        "accuracy", "macro_f1", "class0_n", "class1_n", "reference_correct",
                        "reference_n", "harm")
    online_path = out / "online_batches.csv"
    evaluator_path = out / "evaluator_batches.csv"
    per_seed_path = out / "per_seed_condition.csv"
    totals = {"streams": 0, "model_streams": 0, "batches": 0, "rows": 0, "failed_seeds": []}
    started = time.perf_counter()
    try:
        with online_path.open("w", newline="", encoding="utf-8") as of, \
             evaluator_path.open("w", newline="", encoding="utf-8") as ef, \
             per_seed_path.open("w", newline="", encoding="utf-8") as sf, \
             threadpool_limits(limits=1):
            online_writer = None
            eval_writer = csv.DictWriter(ef, fieldnames=evaluator_fields)
            eval_writer.writeheader()
            seed_fields = ("task", "seed", "condition", "model", "regime", "n_batches",
                           "accuracy", "macro_f1", "cal_mae", "cal_bias", "raw_mae",
                           "cal_coverage", "cal_width", "raw_coverage", "raw_width",
                           "harm_n", "cal_tp", "cal_fp", "cal_fn", "cal_tn", "cal_unflagged_harm",
                           "input_alarm_n", "original_alarm_n", "latent_alarm_n", "id_sha256")
            seed_writer = csv.DictWriter(sf, fieldnames=seed_fields)
            seed_writer.writeheader()
            for task in TASKS:
                for index, seed in enumerate(seeds, start=1):
                    seed_started = time.perf_counter()
                    train = historical_partition(task, seed, 0, 4000)
                    calibration = historical_partition(task, seed, 1, 2000)
                    reference = historical_partition(task, seed, 2, 2000)
                    artifacts = fit_models(seed, train, calibration, reference)
                    for condition in conditions[task]:
                        part = monitoring_partition(task, seed, condition)
                        model_rows = {}
                        for model_name, artifact in artifacts.items():
                            caches, records = [], []
                            for batch in range(1, BATCHES + 1):
                                sl = slice((batch - 1) * BATCH_SIZE, batch * BATCH_SIZE)
                                feature = FeatureBatch(part.x[sl], part.ids[sl], seed, condition, batch)
                                record, cache = score_model(artifact, feature, task, model_name)
                                records.append(record)
                                caches.append(cache)
                            # Persist all feature-only model outputs before evaluator truth is used.
                            if online_writer is None:
                                online_fields = tuple(records[0].keys()) + ("prediction_id_sha256",)
                                overlap = ONLINE_FORBIDDEN.intersection(online_fields)
                                if overlap:
                                    raise AssertionError(f"online schema includes truth: {overlap}")
                                online_writer = csv.DictWriter(of, fieldnames=online_fields)
                                online_writer.writeheader()
                            for row, cache in zip(records, caches):
                                row = dict(row)
                                row["prediction_id_sha256"] = hashlib.sha256(
                                    np.asarray(cache["ids"], dtype="<i8").tobytes()).hexdigest()
                                online_writer.writerow(row)
                            model_rows[model_name] = (records, caches)
                            totals["model_streams"] += 1
                        of.flush()
                        # Evaluator-only join follows persisted predictions; labels never enter score_model.
                        for model_name, (records, caches) in model_rows.items():
                            pred = np.stack([cache["prediction"] for cache in caches])
                            truth = part.y.reshape(BATCHES, BATCH_SIZE)
                            f1_by_batch = []
                            for batch in range(BATCHES):
                                labels, predicted = truth[batch], pred[batch]
                                correct = int((labels == predicted).sum())
                                cm = confusion_matrix(labels, predicted, labels=(0, 1))
                                tn, fp, fn, tp = (int(v) for v in cm.ravel())
                                ref = artifacts[model_name]
                                macro_f1 = float(np.mean([
                                    (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else np.nan,
                                    (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan]))
                                evaluator = dict(task=task, seed=seed, condition=condition,
                                    model=model_name, batch=batch + 1, n=BATCH_SIZE,
                                    correct=correct, accuracy=correct / BATCH_SIZE,
                                    macro_f1=macro_f1,
                                    class0_n=int((labels == 0).sum()), class1_n=int((labels == 1).sum()),
                                    reference_correct=ref.reference_correct, reference_n=ref.reference_n,
                                    harm=bool(is_harm(correct, BATCH_SIZE, ref.reference_correct, ref.reference_n)))
                                f1_by_batch.append(macro_f1)
                                eval_writer.writerow(evaluator)
                            # Per-seed condition metrics by pre, transition/onset, post, recovery and all.
                            correct_rows = np.sum(pred == truth, axis=1)
                            accuracy = correct_rows / BATCH_SIZE
                            cal_est = np.asarray([r["cal_estimate"] for r in records])
                            raw_est = np.asarray([r["raw_estimate"] for r in records])
                            cal_covered = np.asarray([r["cal_lower"] <= acc <= r["cal_upper"]
                                                      for r, acc in zip(records, accuracy)])
                            raw_covered = np.asarray([r["raw_lower"] <= acc <= r["raw_upper"]
                                                      for r, acc in zip(records, accuracy)])
                            harmful = np.asarray([r["cal_concern_flag"] for r in records], bool)
                            actual_harm = np.asarray([is_harm(int(c), BATCH_SIZE,
                                artifacts[model_name].reference_correct, artifacts[model_name].reference_n)
                                for c in correct_rows], bool)
                            regimes = {"all": np.arange(24), "pre": np.arange(0, 8),
                                       "onset": np.arange(8, 12), "post": np.arange(8, 24)}
                            if (task, condition) in (("v1_synthetic", "C5"),):
                                regimes["recovery"] = np.arange(16, 24)
                            for regime, idx in regimes.items():
                                tp_n = int((harmful[idx] & actual_harm[idx]).sum())
                                fp_n = int((harmful[idx] & ~actual_harm[idx]).sum())
                                fn_n = int((~harmful[idx] & actual_harm[idx]).sum())
                                tn_n = int((~harmful[idx] & ~actual_harm[idx]).sum())
                                digest = hashlib.sha256(np.asarray(part.ids, dtype="<i8").tobytes()).hexdigest()
                                seed_writer.writerow(dict(task=task, seed=seed, condition=condition,
                                    model=model_name, regime=regime, n_batches=len(idx),
                                    accuracy=float(accuracy[idx].mean()),
                                    macro_f1=float(np.nanmean(np.asarray(f1_by_batch)[idx])),
                                    cal_mae=float(np.mean(np.abs(cal_est[idx] - accuracy[idx]))),
                                    cal_bias=float(np.mean(cal_est[idx] - accuracy[idx])),
                                    raw_mae=float(np.mean(np.abs(raw_est[idx] - accuracy[idx]))),
                                    cal_coverage=float(cal_covered[idx].mean()),
                                    cal_width=float(np.mean([records[j]["cal_upper"] - records[j]["cal_lower"] for j in idx])),
                                    raw_coverage=float(raw_covered[idx].mean()),
                                    raw_width=float(np.mean([records[j]["raw_upper"] - records[j]["raw_lower"] for j in idx])),
                                    harm_n=int(actual_harm[idx].sum()), cal_tp=tp_n, cal_fp=fp_n,
                                    cal_fn=fn_n, cal_tn=tn_n, cal_unflagged_harm=fn_n,
                                    input_alarm_n=int(sum(bool(records[j]["input_alarm"]) for j in idx)),
                                    original_alarm_n=int(sum(bool(records[j]["original_alarm"]) for j in idx)),
                                    latent_alarm_n=int(sum(bool(records[j]["latent_rejection"]) for j in idx)),
                                    id_sha256=digest))
                            totals["batches"] += BATCHES
                        ef.flush(); sf.flush()
                        totals["streams"] += 1
                        totals["rows"] += len(part.y)
                    if index % 25 == 0 or index == len(seeds):
                        print(f"{task} seed {seed} ({index}/{len(seeds)}): "
                              f"{time.perf_counter()-seed_started:.1f}s; elapsed {time.perf_counter()-started:.1f}s",
                              flush=True)
        if online_writer is None:
            raise RuntimeError("no records generated")
        # Check output contracts and deterministic completeness without outcome filtering.
        import pandas as pd
        online_df = pd.read_csv(online_path)
        evaluator_df = pd.read_csv(evaluator_path)
        per_seed_df = pd.read_csv(per_seed_path)
        if ONLINE_FORBIDDEN.intersection(online_df.columns):
            raise AssertionError("online truth field detected")
        expected_streams = len(seeds) * sum(len(v) for v in conditions.values())
        if totals["streams"] != expected_streams or len(evaluator_df) != expected_streams * BATCHES * len(MODELS):
            raise AssertionError("stream or evaluator row denominator mismatch")
        if len(per_seed_df) != expected_streams * len(MODELS) * 4 + len(seeds) * len(MODELS):
            # 4 standard regimes plus C5 recovery rows.
            raise AssertionError("per-seed summary denominator mismatch")
        if online_df.duplicated(["task", "seed", "condition", "model", "batch"]).any():
            raise AssertionError("duplicate online batch record")
        lock.update(status="complete", ended_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-started, streams=totals["streams"],
                    model_streams=totals["model_streams"], online_batches=len(online_df),
                    evaluator_batches=len(evaluator_df), monitoring_rows=totals["rows"],
                    per_seed_condition_rows=len(per_seed_df), failed_seeds=totals["failed_seeds"])
        (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
        validation = {"status": "passed", "online_truth_columns_absent": True,
                      "expected_seed_count_per_task": len(seeds), "unique_seed_count": int(online_df.seed.nunique()),
                      "online_batches": len(online_df), "evaluator_batches": len(evaluator_df),
                      "per_seed_condition_rows": len(per_seed_df), "all_finite_scores": bool(
                          np.isfinite(online_df.select_dtypes(include=["number"]).to_numpy()).all()),
                      "paired_model_rows": bool((online_df.groupby(["task", "seed", "condition", "batch"]).model.nunique()==2).all()),
                      "online_evaluator_join_keys_unique": not online_df.duplicated(["task", "seed", "condition", "model", "batch"]).any(),
                      "queue_delays_not_replayed": True, "seed_vector_unit": "seed"}
        (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps({"lock": lock, "validation": validation}, indent=2), flush=True)
    except BaseException as exc:
        lock.update(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter()-started,
                    failed_seeds=totals["failed_seeds"])
        (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.out)
