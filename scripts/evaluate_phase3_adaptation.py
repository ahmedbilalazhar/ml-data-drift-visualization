"""Validate and summarize the released-label adaptation add-on."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.adaptation import (  # noqa: E402
    BATCHES, BATCH_SIZE, DELAYS, TASK_CONDITION, digest_ids, label_budget_before,
)

ONLINE_FORBIDDEN = {"label", "y", "accuracy", "macro_f1", "correct", "harm", "truth"}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bootstrap(table: pd.DataFrame, group_cols, metrics, replicates=2000):
    rng = np.random.default_rng(15112026)
    rows = []
    for key, group in table.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values):
                draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                mean = float(values.mean())
            else:
                low = high = mean = np.nan
            rows.append(dict(**base, metric=metric, mean=mean, lower=float(low), upper=float(high),
                             seeds=len(values), replicates=replicates))
    return pd.DataFrame(rows)


def validate(run: Path) -> dict:
    run = Path(run)
    lock = json.loads((run / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock.get("status") != "complete":
        raise ValueError("adaptation run is not complete")
    online = pd.read_csv(run / "online_batches.csv")
    forbidden = ONLINE_FORBIDDEN.intersection(column.lower() for column in online.columns)
    if forbidden:
        raise AssertionError(f"online artifact includes truth columns: {sorted(forbidden)}")
    expected_columns = set(online.columns)
    if not {"task", "seed", "condition", "delay", "method", "cohort", "batch",
            "available_monitoring_labels", "training_rows", "prediction_persisted"}.issubset(expected_columns):
        raise AssertionError("online schema incomplete")
    if len(online) != lock["delay_streams"] * 2 * BATCHES:
        raise AssertionError("wrong online batch-row count")
    if set(online.method) != {"frozen", "cumulative_refit"}:
        raise AssertionError("unexpected adaptation method set")
    if not online.prediction_persisted.astype(bool).all():
        raise AssertionError("a batch was scored without persistence")
    expected_by_task = {k: set(v) for k, v in lock["seeds_by_task"].items()}
    for task, condition in TASK_CONDITION:
        got = set(online.loc[online.task == task, "seed"].astype(int))
        if got != expected_by_task[task] or set(online.loc[online.task == task, "condition"]) != {condition}:
            raise AssertionError(f"seed/condition mismatch: {task}")

    model_batches = online.set_index(["task", "seed", "condition", "delay", "method", "batch"])
    if model_batches.index.has_duplicates:
        raise AssertionError("duplicate method-batch cohort")
    fit = pd.read_csv(run / "fit_audit.csv")
    if len(fit) != len(online):
        raise AssertionError("fit audit row count mismatch")
    fit_adapt = fit[fit.method == "cumulative_refit"].copy()
    for row in fit_adapt.itertuples(index=False):
        expected = label_budget_before(int(row.forecast_batch), int(row.delay))
        if int(row.released_monitoring_n) != expected or int(row.training_n) != 4000 + expected:
            raise AssertionError("adapter training count does not match exact release budget")
        if expected:
            if pd.isna(row.max_available_at) or int(row.max_available_at) >= int(row.forecast_batch):
                raise AssertionError("same-clock or future labels entered adaptation fit")
            if int(row.fit_performed_after_clock) != int(row.forecast_batch) - 1:
                raise AssertionError("adaptation update was not after the prior release clock")
        elif not pd.isna(row.max_available_at):
            raise AssertionError("unreleased labels appear in initial fit")
        short = "original" if row.task == "v1_synthetic" else "xor"
        with np.load(run / "features" / short / str(row.seed) / f"{row.condition}.npz") as data:
            expected_ids = data["ids"][:expected]
        if digest_ids(expected_ids) != row.monitor_ids_sha256:
            raise AssertionError("adapter fit IDs differ from exact available prefix")

    # Validate each scheduled label batch, its ID digest, the original truth and full drainage.
    releases = pd.read_csv(run / "release_log.csv")
    labels = pd.read_csv(run / "released_labels.csv")
    expected_release_rows = lock["delay_streams"] * BATCHES
    if len(releases) != expected_release_rows or len(labels) != expected_release_rows * BATCH_SIZE:
        raise AssertionError("release artifact counts are incomplete")
    release_groups = releases.groupby(["task", "seed", "condition", "delay"], sort=False)
    if len(release_groups) != lock["delay_streams"]:
        raise AssertionError("release stream count mismatch")
    label_groups = {key: part for key, part in labels.groupby(
        ["task", "seed", "condition", "delay"], sort=False)}
    release_id_checks = 0
    for (task, seed, condition, delay), group in release_groups:
        group = group.sort_values("source_batch")
        if group.source_batch.astype(int).tolist() != list(range(1, BATCHES + 1)):
            raise AssertionError("a release queue omitted or duplicated a batch")
        if not (group.n.astype(int) == BATCH_SIZE).all():
            raise AssertionError("incorrect labels released for a batch")
        if not (group.available_at.astype(int) == group.source_batch.astype(int) + int(delay)).all():
            raise AssertionError("release clock differs from locked delay")
        expected_after = np.minimum(group.available_at.astype(int), BATCHES)
        if not np.array_equal(group.after_forecast_batch.to_numpy(np.int64), expected_after.to_numpy()):
            raise AssertionError("release was not logged after the latest persisted forecast")
        if not np.array_equal(group.queue_draining.astype(bool).to_numpy(),
                              (group.available_at.astype(int).to_numpy() > BATCHES)):
            raise AssertionError("queue-drain clock status is incorrect")
        if not np.array_equal(group.release_clock.to_numpy(np.int64), group.available_at.to_numpy(np.int64)):
            raise AssertionError("release clock mismatch")
        short = "original" if task == "v1_synthetic" else "xor"
        with np.load(run / "features" / short / str(seed) / f"{condition}.npz") as feature_data:
            ids = feature_data["ids"]
        with np.load(run / "evaluator_truth" / short / str(seed) / f"{condition}.npz") as truth_data:
            truth_ids, truth_y = truth_data["ids"], truth_data["y"]
        if not np.array_equal(ids, truth_ids):
            raise AssertionError("feature/truth cache IDs differ")
        stream_labels = label_groups[(task, seed, condition, delay)]
        if len(stream_labels) != BATCHES * BATCH_SIZE:
            raise AssertionError("released label stream is incomplete")
        for event in group.itertuples(index=False):
            start = (int(event.source_batch) - 1) * BATCH_SIZE
            stop = start + BATCH_SIZE
            current = stream_labels[stream_labels.source_batch == event.source_batch].sort_values("row_id")
            if not np.array_equal(current.row_id.to_numpy(np.int64), ids[start:stop]):
                raise AssertionError("released labels use different instance IDs")
            if not np.array_equal(current.label.to_numpy(np.int8), truth_y[start:stop]):
                raise AssertionError("released label values differ from evaluator truth")
            if digest_ids(current.row_id.to_numpy(np.int64)) != event.ids_sha256:
                raise AssertionError("release ID digest mismatch")
            release_id_checks += BATCH_SIZE
        if group.available_at.max() != BATCHES + int(delay):
            raise AssertionError("queue did not drain through clock 24+d")

    # Join offline predictions with evaluator-only truth, not with broker labels.
    evaluator_rows = []
    id_joins = 0
    truth_cache = {}
    for (task, seed, condition), group in online.groupby(["task", "seed", "condition"], sort=False):
        short = "original" if task == "v1_synthetic" else "xor"
        with np.load(run / "evaluator_truth" / short / str(seed) / f"{condition}.npz") as truth_data:
            truth_cache[(task, int(seed), condition)] = (truth_data["ids"].copy(), truth_data["y"].copy())
        truth_ids, truth_y = truth_cache[(task, int(seed), condition)]
        for delay in DELAYS:
            for batch in range(1, BATCHES + 1):
                start = (batch - 1) * BATCH_SIZE
                stop = start + BATCH_SIZE
                path = (run / "predictions" / short / str(seed) / condition /
                        f"delay{delay}" / f"batch{batch:02}.npz")
                with np.load(path) as pred_data:
                    ids = pred_data["ids"]
                    if not np.array_equal(ids, truth_ids[start:stop]):
                        raise AssertionError("prediction/evaluator IDs do not join exactly")
                    y = truth_y[start:stop]
                    for method, pred_key, probability_key in (
                        ("frozen", "frozen_prediction", "frozen_probability"),
                        ("cumulative_refit", "adapted_prediction", "adapted_probability"),
                    ):
                        prediction = pred_data[pred_key]
                        probability = pred_data[probability_key]
                        if not np.isfinite(probability).all() or not np.isin(prediction, (0, 1)).all():
                            raise AssertionError("invalid persisted prediction")
                        online_row = model_batches.loc[(task, int(seed), condition, delay, method, batch)]
                        if not np.isclose(float(probability.mean()), float(online_row.probability_mean), atol=1e-14):
                            raise AssertionError("online summary differs from saved predictions")
                        evaluator_rows.append(dict(
                            task=task, seed=int(seed), condition=condition, delay=int(delay),
                            method=method, batch=batch, n=BATCH_SIZE,
                            accuracy=float(accuracy_score(y, prediction)),
                            macro_f1=float(f1_score(y, prediction, average="macro", zero_division=0)),
                            label0_n=int((y == 0).sum()), label1_n=int((y == 1).sum()),
                            available_monitoring_labels=label_budget_before(batch, delay)))
                        id_joins += len(ids)
    evaluator = pd.DataFrame(evaluator_rows)
    evaluator.to_csv(run / "evaluator_batches.csv", index=False)

    evaluator["regime"] = np.select(
        [evaluator.batch <= 8, evaluator.batch >= 17], ["pre_onset", "late_post"], default="post_onset")
    paired = (evaluator.pivot(index=["task", "seed", "condition", "delay", "batch", "regime"],
                              columns="method", values=["accuracy", "macro_f1"])
              .reset_index())
    paired.columns = ["_".join(str(item) for item in col if item) if isinstance(col, tuple) else col
                      for col in paired.columns]
    per_seed = []
    for (task, seed, condition, delay, regime), group in paired.groupby(
            ["task", "seed", "condition", "delay", "regime"], sort=True):
        per_seed.append(dict(
            task=task, seed=seed, condition=condition, delay=delay, regime=regime,
            frozen_accuracy=float(group.accuracy_frozen.mean()),
            adapted_accuracy=float(group.accuracy_cumulative_refit.mean()),
            accuracy_delta=float((group.accuracy_cumulative_refit - group.accuracy_frozen).mean()),
            frozen_macro_f1=float(group.macro_f1_frozen.mean()),
            adapted_macro_f1=float(group.macro_f1_cumulative_refit.mean()),
            macro_f1_delta=float((group.macro_f1_cumulative_refit - group.macro_f1_frozen).mean()),
            seeds=1, batches=len(group),
        ))
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(run / "paired_seed_differences.csv", index=False)
    _bootstrap(per_seed_frame, ["task", "condition", "delay", "regime"],
               ["frozen_accuracy", "adapted_accuracy", "accuracy_delta",
                "frozen_macro_f1", "adapted_macro_f1", "macro_f1_delta"]).to_csv(
        run / "paired_bootstrap.csv", index=False)

    budgets = evaluator.drop_duplicates(["task", "seed", "condition", "delay", "batch"])
    budgets = (budgets.groupby(["task", "condition", "delay", "batch"], as_index=False)
               .agg(available_monitoring_labels=("available_monitoring_labels", "first"),
                    unresolved_before_prediction=("available_monitoring_labels", lambda x: 12000 - int(x.iloc[0]))))
    budgets.to_csv(run / "budget_by_clock.csv", index=False)
    validation = dict(
        status="passed", feature_streams=lock["feature_streams"], delay_streams=lock["delay_streams"],
        model_batch_records=len(online), prediction_instance_joins=id_joins,
        scheduled_release_events=len(releases), released_label_rows=len(labels),
        released_id_checks=release_id_checks, adapter_fit_rows=len(fit_adapt),
        exact_available_budget_checked=True, future_label_fit_check=True,
        online_truth_columns_absent=True, prediction_persisted_before_release=True,
        same_delay_release_ids_match=True, queue_drainage_checked=True,
        calibration_or_harm_claims_added=False,
    )
    (run / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    validate(args.run)
