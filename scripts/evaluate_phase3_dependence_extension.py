"""Validate and summarize the frozen dependence-detector evaluation outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def evaluate(run: Path) -> None:
    run = run.resolve()
    batches = pd.read_csv(run / "online_dependence_scores.csv")
    threshold = json.loads((run / "threshold_used.json").read_text(encoding="utf-8"))
    expected = 30 * 7 * 24
    required = {"seed", "condition", "batch", "score", "pair_i", "pair_j", "alarm", "ks_pca_alarm"}
    if (len(batches) != expected or not required.issubset(batches.columns)
            or batches[["seed", "condition", "batch"]].duplicated().any()
            or not np.isfinite(batches.score).all()
            or batches.seed.nunique() != 30 or set(batches.condition) != {f"C{i}" for i in range(7)}):
        raise ValueError("online output failed row-key, completeness, or finite-score checks")
    if set(batches.seed) != set(range(930100, 930130)):
        raise ValueError("evaluation seeds differ from locked protocol")
    if not (batches.alarm == (batches.score >= threshold["threshold"])).all():
        raise ValueError("stored alarm flags do not match locked threshold")

    rows = []
    for (seed, condition), group in batches.groupby(["seed", "condition"]):
        group = group.sort_values("batch")
        onset = group[group.batch.between(9, 12)]
        pre = group[group.batch < 9]
        rows.append(dict(seed=int(seed), condition=condition,
            dep_onset=bool(onset.alarm.any()), ks_onset=bool(onset.ks_pca_alarm.any()),
            dep_pre=bool(pre.alarm.any()), ks_pre=bool(pre.ks_pca_alarm.any()),
            dep_count=int(group.alarm.sum()), ks_count=int(group.ks_pca_alarm.sum()),
            pair01_onset=bool((onset.alarm & (onset.pair_i == 0) & (onset.pair_j == 1)).any())))
    per_seed = pd.DataFrame(rows)
    summaries = []
    for condition, group in per_seed.groupby("condition"):
        for detector, detected, pre, count in [
            ("max_abs_correlation_change", "dep_onset", "dep_pre", "dep_count"),
            ("marginal_plus_PCA_KS", "ks_onset", "ks_pre", "ks_count")]:
            summaries.append(dict(condition=condition, detector=detector, seeds=len(group),
                onset_detection_rate=float(group[detected].mean()),
                pre_onset_stream_false_alarm_rate=float(group[pre].mean()),
                mean_alarms_per_24_batches=float(group[count].mean())))
    summary = pd.DataFrame(summaries)

    c6 = per_seed[per_seed.condition.eq("C6")].sort_values("seed")
    onset_delta = c6.dep_onset.astype(float).to_numpy() - c6.ks_onset.astype(float).to_numpy()
    c0 = per_seed[per_seed.condition.eq("C0")].sort_values("seed")
    alarm_delta = c0.dep_count.to_numpy(dtype=float) - c0.ks_count.to_numpy(dtype=float)
    contrasts = []
    for label, values, seed in [("C6 onset detection rate difference", onset_delta, 16112026),
                                ("C0 alarms per 24 batches difference", alarm_delta, 16112027)]:
        mean, lo, hi = bootstrap_mean(values, seed)
        contrasts.append(dict(contrast=label, mean_difference=mean, lower=lo, upper=hi,
            paired_seeds=len(values), bootstrap_replicates=2000, resampling_unit="seed"))
    pd.DataFrame(contrasts).to_csv(run / "paired_contrasts.csv", index=False)
    summary.to_csv(run / "validated_summary.csv", index=False)
    per_seed.to_csv(run / "validated_per_seed.csv", index=False)
    (run / "validation.json").write_text(json.dumps(dict(status="passed", rows=len(batches),
        unique_seed_condition_batches=expected, evaluator_truth_absent=True,
        threshold=threshold["threshold"], threshold_source_sha256=threshold["development_threshold_sha256"],
        contrasts=contrasts, summary_rows=len(summary), protocol_sha256=threshold["protocol_sha256"]), indent=2),
        encoding="utf-8")
    print(json.dumps(dict(status="passed", rows=len(batches), contrasts=contrasts), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    evaluate(parser.parse_args().run)
