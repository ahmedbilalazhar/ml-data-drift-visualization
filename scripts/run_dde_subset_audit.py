"""Run a reproducible DDE-style marginal-localization subset audit.

The fitted co-drift curve and detector threshold use development streams only.
Validation seeds estimate transfer checks; untouched evaluation seeds produce
the final exploratory results. This is not a reproduction of Domino's complete
online algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.dde_subset import (  # noqa: E402
    binary_ranking_metrics, fit_pooled_dde_curve, pair_correlation, score_unmonitored,
)
from drift_monitoring.phase3.strengthening_generators import (  # noqa: E402
    generate_historical, generate_monitoring,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def verify_completed_cell(path: Path, config: dict) -> str:
    """Validate resumed seed output structure before treating it as complete."""
    match = re.fullmatch(r"(validation|evaluation)_seed_(\d+)_(.+)_d(\d+)", path.stem)
    if not match:
        raise RuntimeError(f"unexpected completed-cell filename: {path.name}")
    partition, seed, task, dim = match.group(1), int(match.group(2)), match.group(3), int(match.group(4))
    expected_seeds = (config["validation_seeds"] if partition == "validation"
                      else [*config["evaluation_seeds"], config["smoke_seed"]])
    if seed not in expected_seeds or task not in config["tasks"] or dim not in config["dimensions"]:
        raise RuntimeError(f"cell is outside the locked design: {path.name}")
    frame = pd.read_csv(path)
    expected_rows = len(config["evaluation_check_batches"]) * len(config["evaluation_mechanisms"]) * 4
    if len(frame) != expected_rows:
        raise RuntimeError(f"incomplete resumed cell {path.name}: {len(frame)} != {expected_rows} rows")
    fixed = {"partition": partition, "seed": seed, "task": task, "dimension": dim}
    if any(not frame[column].eq(value).all() for column, value in fixed.items()):
        raise RuntimeError(f"cell identity mismatch: {path.name}")
    methods = {"dde", "correlation_only", "direct_full_ks_hidden", "random"}
    expected_keys = {(mechanism, batch, method)
                     for mechanism in config["evaluation_mechanisms"]
                     for batch in config["evaluation_check_batches"] for method in methods}
    observed_keys = set(map(tuple, frame[["mechanism", "batch", "method"]]
                            .itertuples(index=False, name=None)))
    if observed_keys != expected_keys:
        raise RuntimeError(f"cell condition/method grid mismatch: {path.name}")
    if frame["online_evaluator_truth_access"].fillna(True).astype(bool).any():
        raise RuntimeError(f"online evaluator truth access recorded in {path.name}")
    return sha256(path)


def ks_vector(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.fromiter((ks_2samp(reference[:, j], current[:, j], method="asymp").statistic
                        for j in range(reference.shape[1])), dtype=float, count=reference.shape[1])


def design_for(seed: int, task_id: int, dimension: int, feature_count: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, dimension, 87112]))
    return np.sort(rng.choice(dimension, size=feature_count, replace=False))


def calibrate(cfg: dict, task_dimensions: list[tuple[str, int]] | None = None) -> dict:
    """Use development seeds only for stationary max-statistic threshold/curve."""
    calibration = {}
    bins = np.asarray(cfg["curve_correlation_bins"], dtype=float)
    combinations = task_dimensions or [(task, dim) for task in cfg["tasks"] for dim in cfg["dimensions"]]
    for task, dim in combinations:
        null_max, episodes = [], []
        for seed in cfg["development_seeds"]:
            ref = generate_historical(task, seed, 3, cfg["reference_rows"], dim).x
            corr = pair_correlation(ref)
            stationary = generate_monitoring(task, "M0_stationary", seed, feature_count=dim,
                                             batch_size=cfg["batch_size"], batches=cfg["batches"])
            for b in range(24):
                lo = b * cfg["batch_size"]
                d = ks_vector(ref, stationary.monitor.x[lo:lo + cfg["batch_size"]])
                null_max.append(float(d.max()))
            for mechanism in cfg["development_mechanisms"]:
                stream = generate_monitoring(task, mechanism, seed, feature_count=dim,
                                             batch_size=cfg["batch_size"], batches=cfg["batches"])
                for batch_number in cfg["development_check_batches"]:
                    b = batch_number - 1
                    lo = b * cfg["batch_size"]
                    d = ks_vector(ref, stream.monitor.x[lo:lo + cfg["batch_size"]])
                    episodes.append((corr, d))
        threshold = float(np.quantile(null_max, cfg["null_quantile"], method="higher"))
        curve_corr = [corr for corr, _ in episodes]
        curve_flags = [d > threshold for _, d in episodes]
        curve = fit_pooled_dde_curve(curve_corr, curve_flags, bins)
        calibration[f"{task}|{dim}"] = {
            "threshold": threshold, "null_max_count": len(null_max),
            "null_quantile": cfg["null_quantile"], "correlation_bins": bins.tolist(),
            "dde_curve": curve["curve"].tolist(), "pair_count": curve["pair_count"].tolist(),
            "both_drifted_count": curve["both_drifted_count"].tolist(),
            "episodes_used": curve["episodes_used"],
            "development_seed_bank": cfg["development_seeds"],
            "curve_training_uses_evaluator_truth": False,
        }
    return calibration


def evaluate_seed(cfg: dict, calibration: dict, task_id: int, task: str,
                  dim: int, seed: int, partition: str, mechanisms: list[str]) -> list[dict]:
    locked = calibration[f"{task}|{dim}"]
    threshold = float(locked["threshold"])
    curve = {"thresholds": np.asarray(locked["correlation_bins"], dtype=float),
             "curve": np.asarray(locked["dde_curve"], dtype=float),
             "pair_count": np.asarray(locked["pair_count"], dtype=np.int64)}
    ref = generate_historical(task, seed, 3, cfg["reference_rows"], dim).x
    corr = pair_correlation(ref)
    subset_n = max(1, int(round(dim * cfg["monitored_fraction"])))
    subset = design_for(seed, task_id, dim, subset_n)
    hidden = np.setdiff1d(np.arange(dim), subset, assume_unique=True)
    outputs = []
    for mechanism in mechanisms:
        stream = generate_monitoring(task, mechanism, seed, feature_count=dim,
                                     batch_size=cfg["batch_size"], batches=cfg["batches"])
        changed = np.asarray(stream.truth_metadata["changed_marginal_features"], dtype=int)
        for batch_number in cfg["evaluation_check_batches"]:
            b = batch_number - 1
            lo = b * cfg["batch_size"]
            current = stream.monitor.x[lo:lo + cfg["batch_size"]]
            full_d = ks_vector(ref, current)
            detected_subset = subset[full_d[subset] > threshold]
            dde = score_unmonitored(curve, corr, detected_subset, hidden)
            cor = (np.mean(corr[np.ix_(hidden, detected_subset)], axis=1)
                   if len(detected_subset) else np.zeros(len(hidden), dtype=float))
            target = np.zeros(dim, dtype=bool)
            if mechanism != "M0_stationary" and b + 1 >= 25:
                target[changed] = True
            y_hidden = target[hidden]
            random_scores = np.random.default_rng(np.random.SeedSequence([seed, task_id, dim, b, 9901])).random(len(hidden))
            metrics = {
                "dde": binary_ranking_metrics(dde, y_hidden, cfg["ranking_k"]),
                "correlation_only": binary_ranking_metrics(cor, y_hidden, cfg["ranking_k"]),
                "direct_full_ks_hidden": binary_ranking_metrics(full_d[hidden], y_hidden, cfg["ranking_k"]),
                "random": binary_ranking_metrics(random_scores, y_hidden, cfg["ranking_k"]),
            }
            # Label availability is intentionally absent: this study concerns
            # feature-distribution localization, not model-error/harm estimates.
            for method, values in metrics.items():
                outputs.append({"partition": partition, "task": task, "dimension": dim, "seed": seed,
                                "mechanism": mechanism, "batch": batch_number, "method": method,
                                "threshold": threshold, "subset_size": len(subset),
                                "hidden_feature_count": len(hidden), "detected_anchor_count": len(detected_subset),
                                "target_feature_count": int(y_hidden.sum()),
                                "average_precision": values["average_precision"],
                                "auroc": values["auroc"], "precision_at_k": values["precision_at_k"],
                                "empty_truth": values["empty_truth"],
                                "online_evaluator_truth_access": False})
    return outputs


def evaluate_cell_job(job: tuple) -> tuple[str, str, int, int, str, list[dict]]:
    """Top-level, pickleable worker for one immutable seed cell."""
    cfg, calibration_values, task_id, task, dim, seed, partition, mechanisms = job
    rows = evaluate_seed(cfg, calibration_values, task_id, task, dim, seed, partition, mechanisms)
    return partition, task, dim, seed, "ok", rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dde_subset_audit_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/dde_subset_audit_20261118_v4")
    parser.add_argument("--smoke", action="store_true", help="one task/dimension/seed for contract validation only")
    parser.add_argument("--workers", type=int, default=2, help="bounded independent seed-cell workers (default: 2)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    preexisting_hashes = {p.name: verify_completed_cell(p, cfg)
                          for p in sorted(args.output.glob("*_seed_*.csv"))}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_hash or manifest.get("protocol") != cfg["protocol"]:
            raise RuntimeError("existing run directory uses a different config; choose a new output path")
        if manifest.get("status") != "complete":
            manifest.setdefault("resume_history", []).append({
                "at_unix": time.time(),
                "previous_runner_sha256": "a5bab2e95e10a523427e27a0ad387d24cd004baef9e282ac5e4879eaacdba2ec",
                "change": "resumed with bounded process workers; seed cells, data, threshold, scoring and metrics unchanged",
                "verified_existing_cells": len(preexisting_hashes),
                "verified_existing_cell_hashes_sha256": hashlib.sha256(
                    json.dumps(preexisting_hashes, sort_keys=True).encode()).hexdigest(),
            })
        manifest["worker_count"] = args.workers
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
    else:
        manifest = {"protocol": cfg["protocol"], "config_sha256": config_hash,
                    "config_path": config_path.relative_to(ROOT).as_posix(), "status": "running",
                    "smoke_only": bool(args.smoke), "started_unix": time.time(),
                    "worker_count": args.workers}
        atomic_json(manifest_path, manifest)

    calibration_path = args.output / "development_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("config_sha256") != config_hash:
            raise RuntimeError("calibration config hash mismatch")
    else:
        calibration_scope = ([(cfg["tasks"][0], cfg["dimensions"][0])] if args.smoke else None)
        calibration_values = calibrate(cfg, calibration_scope)
        calibration = {"config_sha256": config_hash, "training_seed_bank": cfg["development_seeds"],
                       "values": calibration_values,
                       "threshold_fit": "95th percentile of development stationary per-batch maximum KS D",
                       "curve_fit": "pooled pair counts from development intervention streams; no generator truth used"}
        atomic_json(calibration_path, calibration)

    task_list = [(0, cfg["tasks"][0])] if args.smoke else list(enumerate(cfg["tasks"]))
    dim_list = [cfg["dimensions"][0]] if args.smoke else cfg["dimensions"]
    seed_partitions = [("evaluation", cfg["evaluation_seeds"])]
    if not args.smoke:
        seed_partitions.insert(0, ("validation", cfg["validation_seeds"]))
    else:
        seed_partitions = [("evaluation", [cfg["smoke_seed"]])]
    mechanisms = cfg["evaluation_mechanisms"]
    jobs = []
    for task_id, task in task_list:
        for dim in dim_list:
            for partition, seeds in seed_partitions:
                for seed in seeds:
                    cell = args.output / f"{partition}_seed_{seed}_{task}_d{dim}.csv"
                    if cell.exists():
                        continue
                    jobs.append((cell, (cfg, calibration["values"], task_id, task, dim,
                                        seed, partition, mechanisms)))
    if args.workers == 1:
        completed = ((cell, evaluate_cell_job(job)) for cell, job in jobs)
        for cell, result in completed:
            partition, task, dim, seed, _, rows = result
            temp = cell.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temp, index=False)
            os.replace(temp, cell)
            print(f"completed {partition}: task={task} d={dim} seed={seed}", flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_cell_job, job): cell for cell, job in jobs}
            for future in as_completed(futures):
                cell = futures[future]
                partition, task, dim, seed, _, rows = future.result()
                temp = cell.with_suffix(".csv.tmp")
                pd.DataFrame(rows).to_csv(temp, index=False)
                os.replace(temp, cell)
                print(f"completed {partition}: task={task} d={dim} seed={seed}", flush=True)
    manifest["worker_count"] = args.workers
    manifest["status"] = "complete" if not args.smoke else "smoke_complete"
    manifest["finished_unix"] = time.time()
    manifest["validation_cells_completed"] = len(list(args.output.glob("validation_seed_*.csv")))
    manifest["evaluation_cells_completed"] = len(list(args.output.glob("evaluation_seed_*.csv")))
    manifest["source_sha256"] = {
        "runner": sha256(Path(__file__)),
        "dde_primitives": sha256(ROOT / "src/drift_monitoring/phase3/dde_subset.py"),
        "generator": sha256(ROOT / "src/drift_monitoring/phase3/strengthening_generators.py"),
        "config": config_hash,
    }
    manifest["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    manifest["completed_cells_sha256"] = {p.name: sha256(p) for p in sorted(args.output.glob("*_seed_*.csv"))}
    manifest["temporal_sampling"] = {"development_batches": cfg["development_check_batches"],
                                     "validation_and_evaluation_batches": cfg["evaluation_check_batches"],
                                     "interpretation": "predefined temporal checkpoints, not a per-batch exhaustive trajectory"}
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "output": str(args.output),
                      "completed_seed_cells": manifest["evaluation_cells_completed"],
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
