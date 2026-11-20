"""Validate and evaluate the outcome-informed v2 extension."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.core import is_harm
from drift_monitoring.phase3.v2 import (BATCHES, BATCH_SIZE, DELAYS, MODELS, TASKS,
    V1_CONDITIONS, XOR_CONDITIONS)

DETECTION_SUMMARY_METRICS = (
    "null_alerts_per_1000", "post_alert_fraction", "first_alarm_delay_batches", "missed_event")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def flag_metrics(truth, flagged, score=None):
    truth, flagged = np.asarray(truth, bool), np.asarray(flagged, bool)
    tp = int((truth & flagged).sum())
    fp = int((~truth & flagged).sum())
    fn = int((truth & ~flagged).sum())
    tn = int((~truth & ~flagged).sum())
    auroc = (float(roc_auc_score(truth, score)) if score is not None and
             len(np.unique(truth)) == 2 else np.nan)
    return dict(n=len(truth), harm_n=int(truth.sum()), flagged_n=int(flagged.sum()),
                tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn), false_flag_rate=ratio(fp, fp + tn),
                harmful_unflagged_fraction=ratio(fn, tp + fn),
                flagged_fraction=ratio(tp + fp, len(truth)), harm_auroc=auroc)


def adaptive_calibration_error(probability, correct, bins=20):
    probability = np.asarray(probability, float)
    correct = np.asarray(correct, float)
    order = np.argsort(probability, kind="stable")
    return float(sum(len(index) * abs(probability[index].mean() - correct[index].mean())
                     for index in np.array_split(order, min(bins, len(order))) if len(index)) / len(order))


def regime(task, condition, batch):
    if batch <= 8:
        return "pre"
    stationary = (task == "v1_synthetic" and condition == "C0") or (
        task == "xor_synthetic" and condition == "X0")
    if stationary:
        return "stationary_late"
    if task == "v1_synthetic" and condition == "C5" and batch >= 17:
        return "recovery"
    if task == "v1_synthetic" and condition in ("C2", "C4") and batch <= 12:
        return "transition"
    if batch == 9:
        return "onset"
    return "post"


def validate_run(out):
    out = Path(out)
    lock = json.loads((out / "protocol_lock.json").read_text())
    if lock["status"] != "complete":
        raise ValueError("incomplete runs are not final evidence")
    online = pd.read_csv(out / "online_batches.csv")
    verified = pd.read_csv(out / "released_verification.csv")
    events = pd.read_csv(out / "clock_events.csv")
    expected = sum(len(lock["seeds_by_task"][task]) *
                   len(lock["conditions_by_task"][task]) * len(MODELS) * BATCHES for task in TASKS)
    if len(online) != expected or not online.cohort.is_unique:
        raise AssertionError("online batch count or cohort uniqueness failed")
    forbidden = {"accuracy", "macro_f1", "harm", "correct", "labels", "y", "class0_n", "class1_n"}
    if forbidden.intersection(online.columns):
        raise AssertionError("online record contains evaluator truth")
    np.testing.assert_array_equal(online.cal_estimate, online.calibrated_ac_alone)
    if verified.duplicated(["cohort", "delay"]).any():
        raise AssertionError("duplicate model/cohort/delay verification rows")
    np.testing.assert_array_equal(verified.available_at, verified.batch + verified.delay)
    if set(verified.delay) != set(DELAYS) or set(verified.cohort) != set(online.cohort):
        raise AssertionError("release schedule/cohort coverage mismatch")
    reference_by_cohort = online.set_index("cohort")[["reference_correct", "reference_n"]]
    for row in verified.itertuples(index=False):
        reference = reference_by_cohort.loc[row.cohort]
        if bool(row.harm) != bool(is_harm(row.correct, row.n,
                                          reference.reference_correct, reference.reference_n)):
            raise AssertionError("model-specific harm target mismatch")
    expected_releases = expected * len(DELAYS)
    if len(verified) != expected_releases:
        raise AssertionError("release row count mismatch")
    first = online.iloc[0]
    feature_count = int(first.get("n_features", sum(c.startswith("x") and c.endswith("_ks") for c in online.columns)))
    component_count = int(first.get("n_pca_components", sum(c.startswith("pc") and c.endswith("_ks") for c in online.columns)))
    shared_columns = ["input_alarm", "original_alarm", "latent_rejection", "min_adjusted_p",
                      "min_original_adjusted_p"] + [f"x{index}_ks" for index in range(feature_count)] + \
                     [f"pc{index}_ks" for index in range(component_count)]
    for _, group in online.groupby(["task", "seed", "condition", "batch"], sort=False):
        if len(group) != len(MODELS):
            raise AssertionError("model pairing is incomplete")
        logistic = group[group.model == "logistic"].iloc[0]
        tree = group[group.model == "shallow_tree"].iloc[0]
        np.testing.assert_array_equal(logistic[shared_columns].to_numpy(),
                                      tree[shared_columns].to_numpy())
    for delay, group in events.groupby("delay", sort=False):
        if group[group.event == "prediction_persisted"].clock.max() != BATCHES:
            raise AssertionError("prediction clock did not end at the input horizon")
        if group[group.event == "labels_verified"].clock.max() != BATCHES + delay:
            raise AssertionError("release queue did not drain to 24+d")
        for _, stream in group.groupby(["task", "model", "seed", "condition"], sort=False):
            if list(zip(stream.clock, stream.phase)) != sorted(zip(stream.clock, stream.phase)):
                raise AssertionError("event phase ordering failed")

    id_joins = 0
    for row in online[["task", "seed", "condition", "model"]].drop_duplicates().itertuples(index=False):
        task, seed, condition, model = row
        cache_path = out / "predictions" / f"{task}_{seed}_{condition}_{model}.npz"
        truth_path = out / "evaluator_truth" / f"{task}_{seed}_{condition}.npz"
        with np.load(cache_path) as cache, np.load(truth_path) as truth:
            np.testing.assert_array_equal(cache["ids"].ravel(), truth["ids"])
            if len(np.unique(truth["ids"])) != BATCHES * BATCH_SIZE:
                raise AssertionError("duplicate instance IDs")
            np.testing.assert_array_equal(cache["prediction"], cache["probability"] >= 0.5)
            rows = online[(online.task == task) & (online.seed == seed) &
                          (online.condition == condition) & (online.model == model)]
            np.testing.assert_allclose(cache["correctness_probability"].mean(axis=1),
                                       rows.cal_estimate, atol=1e-14)
            np.testing.assert_allclose(cache["raw_confidence"].mean(axis=1), rows.raw_estimate, atol=1e-14)
            labels = truth["labels"].reshape(BATCHES, BATCH_SIZE)
            counts = (cache["prediction"] == labels).sum(axis=1)
            released = verified[(verified.task == task) & (verified.seed == seed) &
                                (verified.condition == condition) & (verified.model == model) &
                                (verified.delay == 0)].sort_values("batch")
            np.testing.assert_array_equal(counts, released.correct)
            id_joins += len(truth["ids"])
    v1_match = None
    v1_dir = ROOT / "results/phase3/final_v1"
    if lock["mode"] == "final" and v1_dir.is_dir():
        checked_streams = 0
        checked_historical_partitions = 0
        checked_truth_streams = 0
        equal_arrays = 0
        equal_array_values = 0
        array_keys = ("ids", "probability", "prediction", "raw_confidence",
                      "correctness_probability", "pca", "cal_pmf", "raw_pmf")
        record_keys = ("reference_correct", "reference_n", "reference_accuracy", "atc_threshold",
                       "input_alarm", "original_alarm", "latent_rejection", "cal_estimate",
                       "cal_lower", "cal_upper", "cal_concern", "cal_concern_flag",
                       "raw_estimate", "raw_lower", "raw_upper", "raw_concern", "raw_concern_flag",
                       "min_adjusted_p", "min_original_adjusted_p")
        for seed in lock["seeds_by_task"]["v1_synthetic"]:
            with np.load(v1_dir / "predictions" / f"{seed}_historical_ids.npz") as old_ids, \
                 np.load(out / "historical_ids" / f"v1_synthetic_{seed}.npz") as new_ids:
                for key in ("train", "calibration", "reference"):
                    np.testing.assert_array_equal(old_ids[key], new_ids[key])
                checked_historical_partitions += 1
            for condition in V1_CONDITIONS:
                old_cache_path = v1_dir / "predictions" / f"{seed}_{condition}.npz"
                new_cache_path = out / "predictions" / f"v1_synthetic_{seed}_{condition}_logistic.npz"
                if not old_cache_path.is_file():
                    raise FileNotFoundError(old_cache_path)
                with np.load(old_cache_path) as old, np.load(new_cache_path) as new:
                    for key in array_keys:
                        np.testing.assert_array_equal(old[key], new[key])
                        equal_arrays += 1
                        equal_array_values += old[key].size
                with np.load(v1_dir / "evaluator_truth" / f"{seed}_{condition}.npz") as old_truth, \
                     np.load(out / "evaluator_truth" /
                             f"v1_synthetic_{seed}_{condition}.npz") as new_truth:
                    np.testing.assert_array_equal(old_truth["ids"], new_truth["ids"])
                    np.testing.assert_array_equal(old_truth["labels"], new_truth["labels"])
                    checked_truth_streams += 1
                old_rows = pd.read_csv(v1_dir / "predictions" / f"{seed}_{condition}_online.csv")
                new_rows = pd.read_csv(out / "predictions" /
                    f"v1_synthetic_{seed}_{condition}_logistic_online.csv")
                for key in record_keys:
                    np.testing.assert_array_equal(old_rows[key], new_rows[key])
                checked_streams += 1
        v1_match = dict(status="passed", streams=checked_streams,
                        historical_partitions=checked_historical_partitions,
                        truth_streams=checked_truth_streams,
                        arrays_compared=equal_arrays, array_values_compared=equal_array_values,
                        v1_source="results/phase3/final_v1",
                        comparison="exact logistic prediction/proxy/cache and scientific batch evidence")
    return dict(status="passed", expected_online_batches=expected,
                model_streams=expected // BATCHES, verification_rows=len(verified),
                instance_id_joins_checked=id_joins, queue_drainage_checked=True,
                online_truth_columns_absent=True, model_specific_harm_checked=True,
                calibration_identity_checked=True,
                v1_logistic_extension_comparison=v1_match)


def joined_batches(out):
    online = pd.read_csv(Path(out) / "online_batches.csv")
    verified = pd.read_csv(Path(out) / "released_verification.csv")
    verified = verified[verified.delay == 0].drop(columns=["delay", "seed", "condition", "batch"])
    joined = online.merge(verified, on=["task", "model", "cohort"], validate="one_to_one")
    joined["regime"] = [regime(task, condition, batch) for task, condition, batch in
                         zip(joined.task, joined.condition, joined.batch)]
    return joined


def impact_table(batches):
    rows = []
    fields = [
        ("IDM_calibrated_AC", "cal_estimate", "cal", "cal_concern_flag"),
        ("calibrated_AC_alone", "calibrated_ac_alone", "cal", "cal_concern_flag"),
        ("raw_AC_no_calibration", "raw_estimate", "raw", "raw_concern_flag"),
        ("ATC_style", "atc_estimate", None, None),
        ("constant_historical", "constant_estimate", None, None),
        ("no_latent_same_impact", "cal_estimate", "cal", "cal_concern_flag"),
        ("point_rule_no_interval", "cal_estimate", None, "cal_point_flag"),
    ]
    for (task, seed, condition, model), whole in batches.groupby(
            ["task", "seed", "condition", "model"], sort=False):
        regimes = [("all", whole)] + list(whole.groupby("regime", sort=False))
        for regime_name, group in regimes:
            for method, estimate_col, prefix, flag_col in fields:
                estimate = group[estimate_col].to_numpy(float)
                if flag_col is None:
                    if method == "ATC_style":
                        flagged = group.reference_accuracy.to_numpy() - estimate > 0.05
                    else:
                        flagged = np.zeros(len(group), bool)
                    score = group.reference_accuracy.to_numpy() - estimate
                else:
                    flagged = group[flag_col].to_numpy(bool)
                    score = group[f"{prefix}_concern"].to_numpy(float) if prefix else estimate
                truth = group.harm.to_numpy(bool)
                row = dict(task=task, seed=seed, condition=condition, model=model, regime=regime_name,
                           method=method, mae=float(np.abs(estimate - group.accuracy.to_numpy()).mean()),
                           bias=float((estimate - group.accuracy.to_numpy()).mean()),
                           actual_accuracy=float(group.accuracy.mean()),
                           macro_f1=float(group.macro_f1.mean()),
                           class0_n=int(group.class0_n.sum()), class1_n=int(group.class1_n.sum()),
                           **flag_metrics(truth, flagged, score))
                if prefix:
                    covered = ((group[f"{prefix}_lower"] <= group.accuracy) &
                               (group.accuracy <= group[f"{prefix}_upper"]))
                    row.update(covered_n=int(covered.sum()), coverage=float(covered.mean()),
                               interval_width=float((group[f"{prefix}_upper"] -
                                                     group[f"{prefix}_lower"]).mean()))
                else:
                    row.update(covered_n=np.nan, coverage=np.nan, interval_width=np.nan)
                rows.append(row)
            for method, column in (("original_KS_harm_proxy", "original_alarm"),
                                   ("full_KS_PCA_harm_proxy", "input_alarm")):
                rows.append(dict(task=task, seed=seed, condition=condition, model=model,
                                 regime=regime_name, method=method, mae=np.nan, bias=np.nan,
                                 coverage=np.nan, interval_width=np.nan,
                                 **flag_metrics(group.harm, group[column], group[column].astype(float))))
    return pd.DataFrame(rows)


def _bootstrap_rows(table, group_columns, metrics, replicates=2000):
    rng = np.random.default_rng(7312026)
    output = []
    for key, group in table.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_columns, key))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values):
                draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                mean = float(values.mean())
            else:
                low = high = mean = np.nan
            row = dict(base)
            row.update(mean=mean, lower=low, upper=high, defined_seeds=len(values),
                       total_seeds=len(group), replicates=replicates)
            if "metric" in base:
                row["summarized_column"] = metric
            else:
                row["metric"] = metric
            output.append(row)
    return pd.DataFrame(output)


def evaluate(out):
    out = Path(out)
    validation = validate_run(out)
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    batches = joined_batches(out)
    batches.to_csv(out / "evaluator_batches.csv", index=False)
    impact = impact_table(batches)
    impact.to_csv(out / "impact_per_seed.csv", index=False)
    metric_names = ["mae", "bias", "coverage", "interval_width", "precision", "recall",
                    "harmful_unflagged_fraction", "flagged_fraction", "false_flag_rate", "harm_auroc"]
    keys = ["task", "condition", "regime", "model", "method"]
    _bootstrap_rows(impact, keys, metric_names).to_csv(out / "impact_summary.csv", index=False)
    pooled = impact.groupby(keys)[["n", "harm_n", "flagged_n", "tp", "fp", "fn", "tn"]].sum().reset_index()
    pooled["pooled_precision"] = pooled.tp / (pooled.tp + pooled.fp).replace(0, np.nan)
    pooled["pooled_recall"] = pooled.tp / (pooled.tp + pooled.fn).replace(0, np.nan)
    pooled.to_csv(out / "harm_counts.csv", index=False)

    primary = impact[(impact.regime == "all") & (impact.method == "IDM_calibrated_AC")]
    overall = primary.groupby(["task", "seed", "model"], as_index=False)[metric_names].mean()
    overall.to_csv(out / "overall_per_seed.csv", index=False)
    _bootstrap_rows(overall, ["task", "model"], metric_names).to_csv(
        out / "overall_summary.csv", index=False)

    shared = batches.drop_duplicates(["task", "seed", "condition", "batch"])
    detection = []
    for (task, seed, condition), whole in shared.groupby(["task", "seed", "condition"], sort=False):
        eligible = ((task == "v1_synthetic" and condition in ("C1", "C2", "C6")) or
                    (task == "xor_synthetic" and condition in ("X1", "X2")))
        null = whole if condition in ("C0", "X0") else whole[whole.batch <= 8]
        matched = whole[(whole.batch >= 9) & (whole.batch <= 12) & whole.input_alarm]
        detection.append(dict(task=task, seed=seed, condition=condition,
            eligible_input_event=int(eligible), null_batches=len(null),
            null_alerts=int(null.input_alarm.sum()),
            null_alerts_per_1000=1000 * float(null.input_alarm.mean()),
            total_alerts=int(whole.input_alarm.sum()), total_batches=len(whole),
            matched_event=int(eligible and len(matched) > 0) if eligible else np.nan,
            missed_event=int(eligible and len(matched) == 0) if eligible else np.nan,
            first_alarm_delay_batches=float(matched.batch.min() - 9)
                if eligible and len(matched) else np.nan,
            post_alert_fraction=float(whole[whole.batch >= 9].input_alarm.mean()),
            input_seconds=float((whole.original_ks_seconds + whole.latent_ks_seconds).sum())))
    detection_frame = pd.DataFrame(detection)
    detection_frame.to_csv(out / "input_detection_per_seed.csv", index=False)
    # Input wall time remains a per-run engineering diagnostic. It is not a
    # scientific outcome and must not receive a seed-bootstrap interval.
    _bootstrap_rows(detection_frame, ["task", "condition"],
                    DETECTION_SUMMARY_METRICS).to_csv(
        out / "input_detection_summary.csv", index=False)

    # Complete-seed paired contrasts; neither batches nor delay copies are resampling units.
    complete = impact[impact.regime == "all"].copy()
    estimates = complete[complete.method.isin(("IDM_calibrated_AC", "raw_AC_no_calibration",
                                                "ATC_style", "constant_historical"))]
    paired = []
    for comparator in ("raw_AC_no_calibration", "ATC_style", "constant_historical"):
        wide = estimates.pivot(index=["task", "seed", "condition", "model"],
                               columns="method", values="mae")
        if comparator not in wide or "IDM_calibrated_AC" not in wide:
            continue
        differences = (wide["IDM_calibrated_AC"] - wide[comparator]).rename("mae_difference").reset_index()
        differences["comparator"] = comparator
        paired.append(differences)
    pd.concat(paired, ignore_index=True).to_csv(out / "paired_estimator_differences.csv", index=False)
    (_bootstrap_rows(pd.concat(paired, ignore_index=True),
                     ["task", "condition", "model", "comparator"], ["mae_difference"])
     .to_csv(out / "paired_estimator_bootstrap.csv", index=False))

    model_primary = complete[complete.method == "IDM_calibrated_AC"]
    model_wide = model_primary.pivot(index=["task", "seed", "condition", "regime"],
                                     columns="model", values=["mae", "actual_accuracy", "bias"])
    model_rows = []
    for metric in ("mae", "actual_accuracy", "bias"):
        if (metric, "logistic") in model_wide and (metric, "shallow_tree") in model_wide:
            diff = (model_wide[(metric, "shallow_tree")] - model_wide[(metric, "logistic")])
            for (task, seed, condition, regime_name), value in diff.items():
                model_rows.append(dict(task=task, seed=seed, condition=condition, regime=regime_name,
                                       metric=f"tree_minus_logistic_{metric}", value=value))
    model_differences = pd.DataFrame(model_rows)
    model_differences.to_csv(out / "paired_model_differences.csv", index=False)
    (_bootstrap_rows(model_differences, ["task", "condition", "regime", "metric"], ["value"])
     .to_csv(out / "paired_model_bootstrap.csv", index=False))

    calibration = []
    for key, group in batches.groupby(["task", "seed", "condition", "model"], sort=False):
        task, seed, condition, model = key
        with np.load(out / "predictions" / f"{task}_{seed}_{condition}_{model}.npz") as pred, \
             np.load(out / "evaluator_truth" / f"{task}_{seed}_{condition}.npz") as truth:
            labels = truth["labels"].reshape(BATCHES, BATCH_SIZE)
            correctness = pred["prediction"] == labels
            batch_ids = np.arange(1, BATCHES + 1)
            for regime_name in sorted(group.regime.unique()):
                mask = np.asarray([regime(task, condition, batch) == regime_name for batch in batch_ids])
                for method, key_name in (("calibrated", "correctness_probability"),
                                         ("raw_confidence", "raw_confidence")):
                    probability = pred[key_name][mask].ravel()
                    correct = correctness[mask].ravel().astype(float)
                    calibration.append(dict(task=task, seed=seed, condition=condition, model=model,
                        regime=regime_name, method=method, n=len(correct),
                        observed_accuracy=float(correct.mean()), mean_confidence=float(probability.mean()),
                        adaptive_calibration_error=adaptive_calibration_error(probability, correct),
                        brier=float(np.mean((probability - correct) ** 2))))
    calibration_frame = pd.DataFrame(calibration)
    calibration_frame.to_csv(out / "calibration_error_per_seed.csv", index=False)
    (_bootstrap_rows(calibration_frame, ["task", "condition", "regime", "model", "method"],
                     ["adaptive_calibration_error", "brier", "mean_confidence", "observed_accuracy"])
     .to_csv(out / "calibration_error_summary.csv", index=False))

    delays = pd.read_csv(out / "released_verification.csv")
    availability = []
    for (task, seed, condition, model, delay), group in delays.groupby(
            ["task", "seed", "condition", "model", "delay"], sort=False):
        at_input_end = group.available_at <= BATCHES
        availability.append(dict(task=task, seed=seed, condition=condition, model=model, delay=delay,
            total_batches=len(group), verified_at_input_end=int(at_input_end.sum()),
            unresolved_at_input_end=int((~at_input_end).sum()), verified_after_drain=len(group),
            mean_evidence_age_at_release=float(group.evidence_age_batches.mean())))
    availability_frame = pd.DataFrame(availability)
    availability_frame.to_csv(out / "delay_availability.csv", index=False)
    availability_seed = availability_frame.groupby(["task", "seed", "model", "delay"],
        as_index=False)[["verified_at_input_end", "unresolved_at_input_end", "verified_after_drain",
                         "mean_evidence_age_at_release"]].mean()
    _bootstrap_rows(availability_seed, ["task", "model", "delay"],
                    ["verified_at_input_end", "unresolved_at_input_end",
                     "mean_evidence_age_at_release"]).to_csv(
        out / "delay_availability_summary.csv", index=False)
    fit = pd.read_csv(out / "fit_records.csv")
    timing = batches.groupby(["task", "seed", "model"])[[c for c in batches if c.endswith("_seconds")]].sum().reset_index()
    timing = timing.merge(fit[["task", "seed", "model", "fit_seconds"]],
                          on=["task", "seed", "model"], validate="one_to_one")
    timing.to_csv(out / "runtime_per_seed.csv", index=False)
    print(json.dumps(validation, indent=2))
    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run.resolve())
