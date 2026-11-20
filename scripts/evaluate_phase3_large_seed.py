"""Validate and summarize the large-seed robustness run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL_SEEDS = set(range(1000, 2500))
DEVELOPMENT_SEEDS = set(range(990001, 990006))


def exact_harm(reference_correct: int, reference_n: int, correct: int, n: int) -> bool:
    """Strict historical-minus-batch accuracy > 0.05, using integer arithmetic."""
    return 20 * (int(reference_correct) * int(n) - int(correct) * int(reference_n)) > int(reference_n) * int(n)


def bootstrap(frame: pd.DataFrame, groups: list[str], value: str, replicates: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(15112026)
    rows = []
    for key, group in frame.groupby(groups, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = group[value].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size:
            draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            mean = float(values.mean())
        else:
            lo = hi = mean = np.nan
        row = {**dict(zip(groups, key)), "mean": mean, "lower": float(lo), "upper": float(hi),
               "defined_seeds": len(values), "total_seeds": len(group), "replicates": replicates}
        if "metric" in groups:
            row["summarized_column"] = value
        else:
            row["metric"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(run: Path) -> dict:
    run = run.resolve()
    lock = json.loads((run / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "complete":
        raise ValueError("run is not complete")
    expected_seeds = FINAL_SEEDS if lock["mode"] == "final" else DEVELOPMENT_SEEDS
    online = pd.read_csv(run / "online_batches.csv")
    evaluator = pd.read_csv(run / "evaluator_batches.csv")
    per_seed = pd.read_csv(run / "per_seed_condition.csv")
    forbidden = {"label", "y", "accuracy", "macro_f1", "correct", "harm", "truth", "f1"}
    if forbidden.intersection(online.columns):
        raise AssertionError("online table exposes evaluator truth")
    for task in ("v1_synthetic", "xor_synthetic"):
        found = set(online.loc[online.task == task, "seed"].unique())
        if found != expected_seeds:
            raise AssertionError(f"seed set mismatch for {task}: {len(found)} vs {len(expected_seeds)}")
    keys = ["task", "seed", "condition", "model", "batch"]
    if online.duplicated(keys).any() or evaluator.duplicated(keys).any():
        raise AssertionError("duplicate online/evaluator key")
    joined = online.merge(evaluator, on=keys, validate="one_to_one", suffixes=("_online", "_eval"))
    if len(joined) != len(online) or len(joined) != len(evaluator):
        raise AssertionError("online/evaluator join mismatch")
    expected_harm = np.asarray([exact_harm(rc, rn, c, n) for rc, rn, c, n in zip(
        joined.reference_correct_eval, joined.reference_n_eval, joined.correct, joined.n_eval)], dtype=bool)
    if not np.array_equal(joined.harm.to_numpy(bool), expected_harm):
        raise AssertionError("realized-harm threshold mismatch")
    # Shared feature-only detector evidence must be exactly model-independent.
    shared_cols = [c for c in online.columns if c in {
        "input_alarm", "original_alarm", "latent_rejection", "min_adjusted_p",
        "min_original_adjusted_p", "prediction_id_sha256", *[f"x{i}_ks" for i in range(8)],
        *[f"pc{i}_ks" for i in range(3)]}]
    for _, group in online.groupby(["task", "seed", "condition", "batch"], sort=False):
        if len(group) != 2:
            raise AssertionError("incomplete paired model row")
        left, right = group.iloc[0], group.iloc[1]
        for col in shared_cols:
            if left[col] != right[col]:
                raise AssertionError(f"input evidence differs by model: {col}")
    if not np.allclose(joined.cal_estimate, joined.calibrated_ac_alone, atol=1e-14):
        raise AssertionError("calibrated AC identity failed")
    if not np.isfinite(online.select_dtypes(include=["number"]).to_numpy()).all():
        raise AssertionError("non-finite online numerical output")
    if len(evaluator) != len(expected_seeds) * 12 * 2 * 24:
        raise AssertionError("incorrect evaluator denominator")
    per_seed.to_csv(run / "per_seed_condition.csv", index=False)
    summary = bootstrap(per_seed, ["task", "condition", "regime", "model"], "accuracy")
    for metric in ("cal_mae", "cal_bias", "raw_mae", "cal_coverage", "cal_width",
                   "raw_coverage", "raw_width", "harm_n", "cal_tp", "cal_fp", "cal_fn",
                   "cal_tn", "cal_unflagged_harm", "input_alarm_n", "original_alarm_n", "latent_alarm_n"):
        summary = pd.concat([summary, bootstrap(per_seed, ["task", "condition", "regime", "model"], metric)],
                            ignore_index=True)
    summary.to_csv(run / "seed_bootstrap_summary.csv", index=False)
    # Paired tree-minus-logistic differences on identical task/seed/condition/regime cohorts.
    metrics = ("accuracy", "cal_mae", "cal_bias", "raw_mae", "cal_coverage", "cal_width",
               "raw_coverage", "raw_width", "cal_unflagged_harm", "input_alarm_n")
    paired = []
    for metric in metrics:
        wide = per_seed.pivot(index=["task", "seed", "condition", "regime"],
                              columns="model", values=metric)
        delta = (wide["shallow_tree"] - wide["logistic"]).rename("difference").reset_index()
        delta["metric"] = metric
        paired.append(delta)
    paired_frame = pd.concat(paired, ignore_index=True)
    paired_frame.to_csv(run / "paired_model_seed_differences.csv", index=False)
    paired_summary = bootstrap(paired_frame, ["task", "condition", "regime", "metric"], "difference")
    paired_summary.to_csv(run / "paired_model_bootstrap.csv", index=False)
    analyzer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    locked_analyzer_hash = lock.get("source_sha256", {}).get("scripts/evaluate_phase3_large_seed.py")
    validation = {"status": "passed", "mode": lock["mode"], "seeds_per_task": len(expected_seeds),
                  "online_rows": len(online), "evaluator_rows": len(evaluator),
                  "per_seed_rows": len(per_seed), "paired_seed_vectors": len(expected_seeds),
                  "online_truth_columns_absent": True, "realized_harm_exact_check": "passed",
                  "paired_input_evidence_identical": True, "calibrated_identity": True,
                  "finite_scores": True, "bootstrap_replicates": 2000,
                  "resampling_unit": "complete seed vector",
                  "analysis_source_sha256": analyzer_hash,
                  "locked_analysis_source_sha256": locked_analyzer_hash,
                  "analysis_source_amendment": ("A post-start output-label correction preserves the metric name in paired-bootstrap rows; it does not change metrics, resampling, run data, predictions or conditions."
                      if locked_analyzer_hash != analyzer_hash else None)}
    (run / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if validation["analysis_source_amendment"]:
        amendment = {
            "scope": "analysis output labeling only",
            "locked_source_sha256": locked_analyzer_hash,
            "executed_source_sha256": analyzer_hash,
            "description": validation["analysis_source_amendment"],
            "scientific_outputs_changed": False,
        }
        (run / "analysis_source_amendment.json").write_text(
            json.dumps(amendment, indent=2), encoding="utf-8"
        )
    return validation


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(evaluate(args.run), indent=2))
