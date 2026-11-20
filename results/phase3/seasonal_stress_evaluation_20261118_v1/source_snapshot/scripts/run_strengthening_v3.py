"""Run paired, label-safe IDM-Viz strengthening cells with atomic resume checkpoints."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import zipfile

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_monitoring.phase3.contracts import (  # noqa: E402
    CheckpointedRun, atomic_write_bytes, atomic_write_json, canonical_hash, sha256_file,
    load_experiment_config,
)
from drift_monitoring.phase3.core import FeatureBatch, fit_artifacts, score_batch  # noqa: E402
from drift_monitoring.phase3.replay import LabelBroker, replay  # noqa: E402
from drift_monitoring.phase3.dependence import (  # noqa: E402
    calibrate_reference_threshold, max_correlation_change,
)
from drift_monitoring.phase3.strengthening_generators import (  # noqa: E402
    MECHANISMS, SEASONAL, TASKS, generate_historical, generate_monitoring,
)


def _csv_bytes(rows):
    if not rows:
        return b""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _npz_bytes(**arrays):
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def _cell_zip(path, online, cache, truth, metadata, delays):
    """Build a single immutable cell artifact, then atomically install it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("online/batch_scores.csv", _csv_bytes(online))
            zf.writestr("online/prediction_cache.npz", _npz_bytes(
                ids=cache["ids"], prediction=cache["prediction"], probability=cache["probability"],
                correctness_probability=cache["correctness_probability"]))
            zf.writestr("evaluator/truth.npz", _npz_bytes(
                labels=truth["labels"], expected_label_probability=truth["expected_label_probability"],
                region_id=truth["region_id"]))
            zf.writestr("evaluator/truth_metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
            zf.writestr("run/label_release_events.csv", _csv_bytes(delays["events"]))
            zf.writestr("run/delayed_verification.csv", _csv_bytes(delays["verification"]))
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _seeds(config, profile):
    if profile == "smoke":
        return [int(config["seed_banks"]["development"][0])]
    return list(map(int, config["seed_banks"][profile]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3_strengthening_v3.json")
    parser.add_argument("--profile", choices=("smoke", "development", "validation", "evaluation"), default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--mechanism", choices=(*MECHANISMS, SEASONAL), action="append")
    parser.add_argument("--dimension", type=int, action="append")
    args = parser.parse_args()

    validated_config = load_experiment_config(args.config)
    config = dict(validated_config.config)
    tasks = args.task or list(config["tasks"])
    mechanisms = args.mechanism or list(config["mechanisms"])
    dimensions = args.dimension or [int(config["feature_dimensions"]["primary"])]
    if set(tasks) - set(TASKS) or set(mechanisms) - (set(MECHANISMS) | {SEASONAL}):
        raise SystemExit("configuration contains an unsupported task or mechanism")
    allowed_dims = {int(config["feature_dimensions"]["primary"]),
                    *map(int, config["feature_dimensions"].get("targeted_sensitivity", []))}
    if not dimensions or set(dimensions) - allowed_dims:
        raise SystemExit(f"dimensions must be selected from {sorted(allowed_dims)}")
    if SEASONAL in mechanisms:
        requested_feature = int(config["seasonal_stress"]["feature"])
        if dimensions != [8] or requested_feature != 7:
            raise SystemExit("the locked seasonal stress is defined only for d=8, nuisance feature index 7")
    seeds = _seeds(config, args.profile)
    cells = [f"{task}|{mechanism}|{seed}|d{dimension}"
             for task in tasks for mechanism in mechanisms for seed in seeds for dimension in dimensions]
    output = args.output or ROOT / "results" / "phase3" / f"strengthening_v3_{args.profile}"
    output = output.resolve()
    source_paths = [args.config, Path(__file__), ROOT / "src/drift_monitoring/phase3/core.py",
                    ROOT / "src/drift_monitoring/phase3/contracts.py",
                    ROOT / "src/drift_monitoring/phase3/strengthening_generators.py",
                    ROOT / "src/drift_monitoring/phase3/dependence.py",
                    ROOT / "src/drift_monitoring/phase3/replay.py",
                    ROOT / config["phase2_basis"]["path"]]
    source_hashes = {p.resolve().relative_to(ROOT).as_posix(): sha256_file(p) for p in source_paths}
    run_contract = {"config": config, "profile": args.profile, "tasks": tasks,
                    "mechanisms": mechanisms, "seeds": seeds, "dimensions": dimensions,
                    "model": config["models"]["primary"], "cell_artifact_schema": 1}
    run = CheckpointedRun(output)
    run.initialize(config_digest=canonical_hash(run_contract), source_hashes=source_hashes,
                   expected_cells=cells, resume=args.resume)
    for source in source_paths:
        relative = source.resolve().relative_to(ROOT).as_posix()
        snapshot = output / "source_snapshot" / relative
        if snapshot.exists():
            if sha256_file(snapshot) != source_hashes[relative]:
                raise RuntimeError(f"source snapshot checksum mismatch: {relative}")
        else:
            atomic_write_bytes(snapshot, source.read_bytes())
    if not (output / "run_metadata.json").exists():
        atomic_write_json(output / "run_metadata.json", {
            "protocol_id": config["protocol_id"], "profile": args.profile,
            "config_sha256": canonical_hash(config), "run_contract_sha256": canonical_hash(run_contract),
            "source_sha256": source_hashes, "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "platform": platform.platform(), "clock_order": "features scored and prediction persisted before label release",
        })

    completed = set(run.lock["completed_cells"])
    artifact_cache = {}
    start = time.perf_counter()
    try:
        for index, (task, mechanism, seed, dimension) in enumerate(
                ((t, m, s, d) for t in tasks for m in mechanisms for s in seeds for d in dimensions), 1):
            cell = f"{task}|{mechanism}|{seed}|d{dimension}"
            if cell in completed:
                continue
            cache_key = (task, seed, dimension)
            if cache_key not in artifact_cache:
                hist = [generate_historical(task, seed, stage, rows, dimension)
                        for stage, rows in enumerate((config["history"]["train_rows"],
                                                      config["history"]["calibration_rows"],
                                                      config["history"]["reference_rows"]), 1)]
                fitted = fit_artifacts(
                    seed, *hist,
                    n_pca_components=int(config["representations"]["pca_requested_components"]),
                    n_clusters=int(config["representations"]["clusters_primary"]),
                    model_max_iter=int(config["models"]["primary"]["max_iter"]))
                calibration_x = fitted.scaler.transform(hist[1].x)
                correlation_threshold = calibrate_reference_threshold(
                    fitted.reference_x, calibration_x,
                    batch_size=int(config["stream"]["batch_size"]),
                    random_state=seed + dimension * 17 + 991, n_null=100, quantile=.95)
                artifact_cache[cache_key] = (fitted, correlation_threshold)
            fitted, correlation_threshold = artifact_cache[cache_key]
            generated = generate_monitoring(task, mechanism, seed, feature_count=dimension,
                                            batch_size=int(config["stream"]["batch_size"]),
                                            batches=int(config["stream"]["batches"]))
            online, caches = [], []
            for batch_no in range(int(config["stream"]["batches"])):
                lo = batch_no * int(config["stream"]["batch_size"])
                hi = lo + int(config["stream"]["batch_size"])
                batch = FeatureBatch(generated.monitor.x[lo:hi], generated.monitor.ids[lo:hi],
                                     seed, mechanism, batch_no + 1)
                record, cache, _ = score_batch(fitted, batch)
                correlation_score, correlation_pair = max_correlation_change(
                    fitted.reference_x, fitted.scaler.transform(batch.x))
                record.update(correlation_discrepancy=correlation_score,
                              correlation_threshold=correlation_threshold,
                              correlation_alarm=bool(correlation_score > correlation_threshold),
                              correlation_pair_i=correlation_pair[0],
                              correlation_pair_j=correlation_pair[1])
                record.update(task=task, feature_dimension=dimension,
                              cohort=f"{task}:{seed}:{mechanism}:{dimension}:{batch_no + 1}")
                online.append(record)
                caches.append(cache)
            pred_cache = {"ids": np.concatenate([c["ids"] for c in caches]),
                          "prediction": np.concatenate([c["prediction"] for c in caches]),
                          "probability": np.concatenate([c["probability"] for c in caches]),
                          "correctness_probability": np.concatenate([c["correctness_probability"] for c in caches])}
            label_events, verification_rows = [], []
            for delay in map(int, config["delays_batches"]):
                broker = LabelBroker(generated.monitor.ids, generated.monitor.y,
                                     int(config["stream"]["batch_size"]), delay)
                events, verified = replay(online, [
                    {"ids": c["ids"], "prediction": c["prediction"]} for c in caches], broker)
                for row in events:
                    row.update(task=task, mechanism=mechanism, feature_dimension=dimension)
                for row in verified:
                    row.update(task=task, mechanism=mechanism, feature_dimension=dimension)
                label_events.extend(events)
                verification_rows.extend(verified)
            truth = {"labels": generated.monitor.y,
                     "expected_label_probability": generated.expected_label_probability,
                     "region_id": generated.region_id}
            metadata = {k: v for k, v in generated.truth.items()
                        if k not in {"true_region_ids", "expected_label_probability"}}
            metadata.update({"feature_dimension": dimension, "fit_seconds": fitted.fit_seconds,
                             "reference_correct": fitted.reference_correct,
                             "reference_n": fitted.reference_n,
                             "calibration_threshold_atc": fitted.atc,
                             "calibration_threshold_correlation": correlation_threshold,
                             "pca_actual_components": int(fitted.pca.n_components_),
                             "prediction_cache_hash": None})
            artifact_path = output / "cells" / f"{canonical_hash(cell)[:24]}.zip"
            _cell_zip(artifact_path, online, pred_cache, truth, metadata,
                      {"events": label_events, "verification": verification_rows})
            run.commit_cell(cell, artifact_path)
            elapsed = time.perf_counter() - start
            print(json.dumps({"completed": index, "total": len(cells), "cell": cell,
                              "cell_seconds": round(elapsed, 2), "elapsed_seconds": round(elapsed, 2),
                              "remaining_cells": len(cells) - len(run.lock["completed_cells"])}), flush=True)
        run.complete()
        run.lock["completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(run.lock_path, run.lock)
    except BaseException as exc:
        run.fail(exc)
        raise
    print(json.dumps({"state": run.lock["state"], "output": str(output),
                      "cells": len(run.lock["completed_cells"]),
                      "elapsed_seconds": round(time.perf_counter() - start, 2)}), flush=True)


if __name__ == "__main__":
    main()
