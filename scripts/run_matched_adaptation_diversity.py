"""Run a matched-clock, matched-feedback-row global/local refit comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.replay import LabelBroker, release_feedback
from drift_monitoring.phase3.strengthening_generators import (  # noqa: E402
    generate_historical,
    generate_monitoring,
    TASKS as TASK_FAMILIES, MECHANISMS as MECHANISM_FAMILIES,
)

BATCH_SIZE = 500
BATCHES = 96
UPDATE_CLOCKS = (25, 49, 73, 89)
WINDOW_ROWS = 2_000
MAX_UPDATES = 4
METHODS = ("frozen", "matched_global", "matched_local")
MECHANISMS = ("M0_stationary", "M4_abrupt_conditional_reversal")
DELAYS = (0, 5, 20)


def fit_model(seed: int, x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("a refit requires both classes")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    return model.fit(x, y)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3_strengthening_v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/matched_adaptation_diversity_20261118_v1")
    parser.add_argument("--seed-limit", type=int, help="smoke testing only")
    parser.add_argument("--tasks", nargs="+", choices=TASK_FAMILIES, default=list(TASK_FAMILIES))
    parser.add_argument("--mechanisms", nargs="+", choices=MECHANISM_FAMILIES, default=["M0_stationary", "M2_gradual_predictive", "M5_recurrent_conditional_reversal", "M7_out_of_support"])
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = list(map(int, config["seed_banks"]["evaluation"]))
    if args.seed_limit is not None:
        if args.seed_limit < 1:
            raise ValueError("seed-limit must be positive")
        seeds = seeds[:args.seed_limit]

    train_n = int(config["history"]["train_rows"])
    batch_rows: list[dict] = []
    run_rows: list[dict] = []
    fit_rows: list[dict] = []
    release_event_rows: list[dict] = []
    method_wall_seconds = {m: 0.0 for m in METHODS}

    for seed in seeds:
        for task in args.tasks:
            train = generate_historical(task, seed, 1, train_n, 8)
            scaler = StandardScaler().fit(train.x)
            train_x = scaler.transform(train.x)
            clusterer = MiniBatchKMeans(n_clusters=3, n_init=3, batch_size=256, random_state=seed).fit(train_x)
    
            for mechanism in args.mechanisms:
                stream = generate_monitoring(task, mechanism, seed, feature_count=8)
                monitor_x = scaler.transform(stream.monitor.x)
                monitor_y = stream.evaluator.labels
                monitor_ids = stream.monitor.ids
                monitor_cluster = clusterer.predict(monitor_x)
    
                for delay in DELAYS:
                    states = {}
                    for method in METHODS:
                        states[method] = {
                            "global_model": fit_model(seed, train_x, train.y),
                            "local_models": {}, "predictions": {},
                            "released_x": [], "released_y": [], "released_ids": [],
                            "last_source": 0, "opportunities": 0, "successful_updates": 0,
                            "row_uses": 0, "failed_local_region_fits": 0,
                            "started": time.perf_counter(),
                        }
    
                    broker = LabelBroker(monitor_ids, monitor_y, BATCH_SIZE, delay)
                    for clock in range(1, BATCHES + 1):
                        lo = (clock - 1) * BATCH_SIZE
                        hi = lo + BATCH_SIZE
                        for released in release_feedback(broker, monitor_x, monitor_ids, clock - 1):
                            release_event_rows.append({"seed": seed, "mechanism": mechanism, "delay": delay,
                                                           "source_batch": released.source_batch, "available_at": released.available_at,
                                                           "release_processed_clock": clock - 1, "prediction_clock": clock,
                                                           "event": "labels_available_before_prediction", "row_count": len(released.ids),
                                                           "ids_sha256": hashlib.sha256(released.ids.tobytes()).hexdigest()})
                            for state in states.values():
                                state["released_x"].append(released.features.copy())
                                state["released_y"].append(released.labels.copy())
                                state["released_ids"].append(released.ids.copy())
                                state["last_source"] = released.source_batch
                        for method, state in states.items():
                            if method != "frozen" and clock in UPDATE_CLOCKS:
                                if state["opportunities"] >= MAX_UPDATES or not state["released_y"]:
                                    raise AssertionError("locked update opportunity unavailable")
                                rx = np.vstack(state["released_x"])
                                ry = np.concatenate(state["released_y"])
                                rids = np.concatenate(state["released_ids"])
                                n = min(WINDOW_ROWS, len(ry))
                                wx, wy, wids = rx[-n:], ry[-n:], rids[-n:]
                                if len(np.unique(wy)) != 2:
                                    raise RuntimeError(f"global refit has one class at seed={seed}, clock={clock}")
                                ids_hash = hashlib.sha256(np.sort(wids).tobytes()).hexdigest()
                                started = time.perf_counter()
                                state["opportunities"] += 1
    
                                if method == "matched_global":
                                    state["global_model"] = fit_model(seed, wx, wy)
                                    used = n
                                    fit_rows.append({
                                        "seed": seed, "task": task, "mechanism": mechanism, "delay": delay,
                                        "method": method, "clock": clock, "cluster": "all",
                                        "common_feedback_ids_sha256": ids_hash,
                                        "window_rows_available": n, "rows_actually_fit": n,
                                        "fit_success": True, "fallback_reason": "not_applicable",
                                    })
                                    state["successful_updates"] += 1
                                else:
                                    # Clusterer was fitted once on historical training features.
                                    row_cluster = clusterer.predict(wx)
                                    candidate_models = {}
                                    region_outcomes = []
                                    used = 0
                                    failures = 0
                                    for cluster in range(3):
                                        mask = row_cluster == cluster
                                        count = int(mask.sum())
                                        labels = wy[mask]
                                        success = count >= 20 and len(np.unique(labels)) == 2
                                        reason = "not_applicable"
                                        if success:
                                            candidate_models[cluster] = fit_model(seed, wx[mask], labels)
                                            used += count
                                        elif count < 20:
                                            reason = "insufficient_rows"
                                            failures += 1
                                        else:
                                            reason = "single_class_region"
                                            failures += 1
                                        region_outcomes.append((cluster, count, success, reason))
                                    state["local_models"] = candidate_models
                                    state["failed_local_region_fits"] += failures
                                    state["successful_updates"] += int(bool(candidate_models))
                                    for cluster, count, success, reason in region_outcomes:
                                        fit_rows.append({
                                            "seed": seed, "task": task, "mechanism": mechanism, "delay": delay,
                                            "method": method, "clock": clock, "cluster": cluster,
                                            "common_feedback_ids_sha256": ids_hash,
                                            "window_rows_available": n,
                                            "rows_actually_fit": count if success else 0,
                                            "fit_success": success, "fallback_reason": reason,
                                        })
                                method_wall_seconds[method] += time.perf_counter() - started
                                state["row_uses"] += used
                                if state["row_uses"] > MAX_UPDATES * WINDOW_ROWS:
                                    raise AssertionError("cumulative adaptation row-use cap exceeded")
    
                            a, b = (clock - 1) * BATCH_SIZE, clock * BATCH_SIZE
                            batch_x = monitor_x[a:b]
                            pred = state["global_model"].predict(batch_x).astype(np.int8)
                            if method == "matched_local":
                                clusters_now = monitor_cluster[a:b]
                                for cluster, local_model in state["local_models"].items():
                                    mask = clusters_now == cluster
                                    if mask.any():
                                        pred[mask] = local_model.predict(batch_x[mask]).astype(np.int8)
                            state["predictions"][clock] = pred
                            latest_clock = state["last_source"] + delay if state["last_source"] else None
                            batch_rows.append({
                                "seed": seed, "task": task, "mechanism": mechanism, "delay": delay,
                                "method": method, "batch": clock, "prediction_persisted": True,
                                "available_source_through": state["last_source"],
                                "latest_available_at": latest_clock,
                                "update_opportunities": state["opportunities"],
                                "successful_updates": state["successful_updates"],
                                "adaptation_row_uses": state["row_uses"],
                                "active_local_models": len(state["local_models"]),
                            })
    
                    drained = release_feedback(broker, monitor_x, monitor_ids, BATCHES + delay)
                    for released in drained:
                        release_event_rows.append({"seed": seed, "mechanism": mechanism, "delay": delay,
                            "source_batch": released.source_batch, "available_at": released.available_at,
                            "release_processed_clock": BATCHES + delay, "prediction_clock": None,
                            "event": "queue_drain_after_stream", "row_count": len(released.ids),
                            "ids_sha256": hashlib.sha256(released.ids.tobytes()).hexdigest()})
                    if sum(len(v.labels) for v in drained) + sum(len(y) for y in states["frozen"]["released_y"]) != len(monitor_y):
                        raise AssertionError("label queue did not drain exactly once")
                    for method, state in states.items():
                        pred = np.concatenate([state["predictions"][c] for c in range(1, BATCHES + 1)])
                        truth = monitor_y
                        correct = pred == truth
                        tp = int(np.sum((pred == 1) & (truth == 1)))
                        fp = int(np.sum((pred == 1) & (truth == 0)))
                        fn = int(np.sum((pred == 0) & (truth == 1)))
                        tn = int(np.sum((pred == 0) & (truth == 0)))
                        run_rows.append({
                            "seed": seed, "task": task, "mechanism": mechanism,
                            "delay": delay, "method": method, "accuracy": float(correct.mean()),
                            "post_onset_accuracy": float(correct[24 * BATCH_SIZE:].mean()),
                            "macro_f1": float((2 * tp / max(1, 2 * tp + fp + fn) +
                                                2 * tn / max(1, 2 * tn + fp + fn)) / 2),
                            "balanced_accuracy": float((tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2),
                            "initial_training_rows": train_n,
                            "update_opportunities": state["opportunities"],
                            "successful_updates": state["successful_updates"],
                            "adaptation_row_uses": state["row_uses"],
                            "adaptation_row_use_cap": MAX_UPDATES * WINDOW_ROWS,
                            "failed_local_region_fits": state["failed_local_region_fits"],
                            "active_local_models_at_end": len(state["local_models"]),
                            "wall_seconds": time.perf_counter() - state["started"],
                            "released_rows_by_final_clock": state["last_source"] * BATCH_SIZE,
                            "queue_drain_rows_after_stream": sum(len(v.labels) for v in drained),
                        })
    
    args.output.mkdir(parents=True)
    frames = {
        "online_batch_audit.csv": pd.DataFrame(batch_rows),
        "seed_metrics.csv": pd.DataFrame(run_rows),
        "fit_audit.csv": pd.DataFrame(fit_rows),
        "label_release_events.csv": pd.DataFrame(release_event_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(args.output / name, index=False)

    # Bootstrap matched seed differences; keep every condition from a seed together.
    runs = frames["seed_metrics.csv"]
    rng = np.random.default_rng(261117)
    references = {
        "matched_global": runs[runs.method == "frozen"].set_index(["seed", "task", "mechanism", "delay"]),
        "matched_local": runs[runs.method == "frozen"].set_index(["seed", "task", "mechanism", "delay"]),
        "matched_local_vs_global": runs[runs.method == "matched_global"].set_index(["seed", "task", "mechanism", "delay"]),
    }
    contrasts = []
    comparisons = (("matched_global", "matched_global", "_difference_vs_frozen"),
                   ("matched_local", "matched_local", "_difference_vs_frozen"),
                   ("matched_local", "matched_local_vs_global", "_difference_vs_global"))
    for policy, label, suffix in comparisons:
        for (task, mechanism, delay), group in runs[runs.method == policy].groupby(["task", "mechanism", "delay"], sort=True):
            current = group.set_index("seed").sort_index()
            reference = references[label].xs((task, mechanism, delay), level=("task", "mechanism", "delay")).sort_index()
            for metric in ("accuracy", "post_onset_accuracy", "macro_f1", "balanced_accuracy"):
                delta = current[metric].to_numpy() - reference.loc[current.index, metric].to_numpy()
                boot = delta[rng.integers(0, len(delta), size=(10_000, len(delta)))].mean(axis=1)
                contrasts.append({
                    "task": task, "mechanism": mechanism, "delay": delay, "method": label,
                    "metric": metric + suffix, "seeds": len(delta),
                    "mean": float(delta.mean()), "lower": float(np.quantile(boot, .025)),
                    "upper": float(np.quantile(boot, .975)),
                })
    frames["paired_seed_differences.csv"] = pd.DataFrame(contrasts)
    frames["paired_seed_differences.csv"].to_csv(args.output / "paired_seed_differences.csv", index=False)

    source_paths = [Path(__file__), args.config, ROOT / "src/drift_monitoring/phase3/strengthening_generators.py", ROOT / "src/drift_monitoring/phase3/replay.py"]
    output_names = [*frames.keys()]
    receipt = {
        "status": "complete", "protocol": "matched-clock-global-vs-cluster-local-refit-diversity-v1",
        "tasks": list(args.tasks), "mechanisms": list(args.mechanisms), "delays": list(DELAYS),
        "seed_bank": seeds, "batch_size": BATCH_SIZE, "batches": BATCHES,
        "update_clocks": list(UPDATE_CLOCKS), "maximum_updates": MAX_UPDATES,
        "per_update_feedback_window_rows": WINDOW_ROWS,
        "maximum_cumulative_adaptation_row_uses": MAX_UPDATES * WINDOW_ROWS,
        "initial_training_rows_reported_separately": train_n,
        "matched_feedback": "both adaptive arms receive the same latest released row IDs at each identical clock; LabelBroker releases due labels at clock t-1 before prediction at t; delay-zero releases follow the source prediction",
        "local_policy": "three-cluster MiniBatchKMeans fitted on historical training features only; each region receives a logistic model fit on its slice of the shared released-label window; unsupported regions retain the original frozen global fallback",
        "minimum_local_fit_rows": 20,
        "statistical_unit": "seed; batch outcomes aggregated within seed",
        "paired_seed_bootstrap_replicates": 10_000,
        "aggregate_fit_wall_seconds": method_wall_seconds,
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths},
        "output_sha256": {name: digest(args.output / name) for name in output_names},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__},
    }
    (args.output / "audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(batch_rows):,} forecasts, {len(run_rows):,} seed-policy cells, "
          f"and {len(fit_rows):,} fit-audit rows in {args.output}")


if __name__ == "__main__":
    main()
