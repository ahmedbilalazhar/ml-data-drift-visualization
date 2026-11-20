"""Targeted historical-only cluster-region and dependence-pair localization audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.core import Partition, fit_artifacts  # noqa: E402
from drift_monitoring.phase3.dependence import (  # noqa: E402
    calibrate_reference_threshold, max_correlation_change,
)
from drift_monitoring.phase3.strengthening_generators import (  # noqa: E402
    generate_historical, generate_monitoring, historical_region_ids,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_cluster_mapping(model, scaler, historical_x: np.ndarray, historical_regions: np.ndarray,
                          n_regions: int = 5) -> tuple[dict[int, int], dict]:
    """One-to-one Hungarian mapping from clusters to historical true regions."""
    cluster_ids = model.predict(scaler.transform(historical_x))
    n_clusters = int(model.n_clusters)
    table = np.zeros((n_clusters, n_regions), dtype=np.int64)
    for c, r in zip(cluster_ids, historical_regions, strict=True):
        table[int(c), int(r)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(c): int(r) for c, r in zip(rows, cols, strict=True)}
    return mapping, {"contingency": table.tolist(), "mapping_cluster_to_region": mapping}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3_strengthening_v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/localization_audit_20261117_v3")
    parser.add_argument("--null-cohorts", type=int, default=100)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = list(map(int, cfg["seed_banks"]["evaluation"]))
    rows, mappings, pair_rows = [], [], []
    # Five-region cluster sensitivity; correspondences use historical training
    # rows only, then the frozen map is scored on the entire monitoring stream.
    for seed in seeds:
        train = generate_historical("five_region", seed, 1, int(cfg["history"]["train_rows"]), 8)
        cal = generate_historical("five_region", seed, 2, int(cfg["history"]["calibration_rows"]), 8)
        ref = generate_historical("five_region", seed, 3, int(cfg["history"]["reference_rows"]), 8)
        train_regions = historical_region_ids("five_region", seed, 1, len(train.x), 8)
        monitor = generate_monitoring("five_region", "M0_stationary", seed, feature_count=8)
        for k in (3, 5, 10):
            fitted = fit_artifacts(seed, train, cal, ref, n_pca_components=3, n_clusters=k,
                                   model_max_iter=2000)
            mapping, table = train_cluster_mapping(fitted.clusters, fitted.scaler, train.x, train_regions)
            pred_cluster = fitted.clusters.predict(fitted.scaler.transform(monitor.monitor.x))
            truth_region = monitor.evaluator.region_id.astype(int)
            mapped = np.array([mapping.get(int(c), -1) for c in pred_cluster], dtype=int)
            valid = mapped >= 0
            region2 = truth_region == 2
            pred2 = mapped == 2
            rows.append({"seed": seed, "clusters": k,
                         "ari": float(adjusted_rand_score(truth_region, pred_cluster)),
                         "mapped_coverage": float(valid.mean()),
                         "mapped_accuracy_on_covered": float(np.mean(mapped[valid] == truth_region[valid])),
                         "macro_f1_on_covered": float(f1_score(truth_region[valid], mapped[valid],
                                                                labels=list(range(5)), average="macro",
                                                                zero_division=0)),
                         "region2_precision": float(np.sum(pred2 & region2) / np.sum(pred2)) if np.any(pred2) else np.nan,
                         "region2_recall": float(np.sum(pred2 & region2) / np.sum(region2)),
                         "region2_predicted_fraction": float(pred2.mean()),
                         "model_fit_seconds": float(fitted.fit_seconds)})
            mappings.append({"seed": seed, "clusters": k, **table})
            # Evaluate spatial cluster allocation by batch for conditional
            # reversal and stationary control using the same frozen assignment.
            for mechanism in ("M0_stationary", "M4_abrupt_conditional_reversal",
                              "M5_recurrent_conditional_reversal"):
                stream = generate_monitoring("five_region", mechanism, seed, feature_count=8)
                assigned = fitted.clusters.predict(fitted.scaler.transform(stream.monitor.x))
                mapped_test = np.array([mapping.get(int(c), -1) for c in assigned])
                for batch in range(96):
                    lo, hi = batch * 500, (batch + 1) * 500
                    actual = stream.evaluator.region_id[lo:hi].astype(int)
                    estimate = mapped_test[lo:hi]
                    target_active = mechanism != "M0_stationary" and (25 <= batch + 1 <= 96 if mechanism.startswith("M4") else
                                                                        25 <= batch + 1 <= 40 or 65 <= batch + 1 <= 80)
                    rows.append({"seed": seed, "clusters": k, "mechanism": mechanism,
                                 "batch": batch + 1, "region2_recall": float(np.mean(estimate[actual == 2] == 2)),
                                 "region2_precision": float(np.mean(actual[estimate == 2] == 2)) if np.any(estimate == 2) else np.nan,
                                 "target_region_intervention_active": bool(target_active),
                                 "mapped_assignment_coverage": float(np.mean(estimate >= 0))})

    # Dependence-pair top-1 identification for the three task families, with a
    # per-seed threshold calibrated only from the historical reference/calibration.
    for task in cfg["tasks"]:
        for seed in seeds:
            cal = generate_historical(task, seed, 2, int(cfg["history"]["calibration_rows"]), 8)
            ref = generate_historical(task, seed, 3, int(cfg["history"]["reference_rows"]), 8)
            threshold = calibrate_reference_threshold(ref.x, cal.x, batch_size=500,
                                                       random_state=seed + 811, n_null=args.null_cohorts,
                                                       quantile=.95)
            for mechanism in ("M0_stationary", "M6_dependence_marginals_preserved"):
                stream = generate_monitoring(task, mechanism, seed, feature_count=8)
                target_pairs = {tuple(sorted(map(int, p))) for p in stream.truth_metadata["changed_dependence_pairs"]}
                for batch in range(96):
                    lo, hi = batch * 500, (batch + 1) * 500
                    score, pair = max_correlation_change(ref.x, stream.monitor.x[lo:hi])
                    active = mechanism == "M6_dependence_marginals_preserved" and batch + 1 >= 25
                    pair_rows.append({"task": task, "seed": seed, "mechanism": mechanism,
                                      "batch": batch + 1, "score": score, "threshold": threshold,
                                      "alarm": bool(score > threshold), "top_pair_i": pair[0], "top_pair_j": pair[1],
                                      "target_pair_count": len(target_pairs), "target_pair_top1":
                                          bool(pair in target_pairs) if target_pairs else None,
                                      "intervention_active": active})

    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output / "region_localization.csv", index=False)
    pd.DataFrame(mappings).to_json(args.output / "historical_cluster_mappings.json", orient="records", indent=2)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(args.output / "dependence_pair_localization.csv", index=False)
    cluster_frame = pd.DataFrame(rows)
    group_fields = [c for c in ("clusters", "mechanism", "target_region_intervention_active") if c in cluster_frame]
    region_summary = cluster_frame.groupby(group_fields, dropna=False).agg(
        seeds=("seed", "nunique"), region2_recall=("region2_recall", "mean"),
        region2_precision=("region2_precision", "mean"), coverage=("mapped_assignment_coverage", "mean")
    ).reset_index() if group_fields else pd.DataFrame()
    pair_summary = pair_frame.groupby(["task", "mechanism", "intervention_active"], dropna=False).agg(
        cells=("batch", "size"), alarm_rate=("alarm", "mean"), top_pair_accuracy=("target_pair_top1", "mean")
    ).reset_index()
    region_summary.to_csv(args.output / "region_localization_summary.csv", index=False)
    pair_summary.to_csv(args.output / "dependence_pair_summary.csv", index=False)
    source_files = [Path(__file__), args.config,
                    ROOT / "src/drift_monitoring/phase3/strengthening_generators.py",
                    ROOT / "src/drift_monitoring/phase3/core.py",
                    ROOT / "src/drift_monitoring/phase3/dependence.py"]
    output_files = [args.output / "region_localization.csv",
                    args.output / "historical_cluster_mappings.json",
                    args.output / "dependence_pair_localization.csv",
                    args.output / "region_localization_summary.csv",
                    args.output / "dependence_pair_summary.csv"]
    receipt = {"status": "complete", "protocol": "posthoc-targeted-localization-audit-v1",
               "seed_bank": seeds, "cluster_counts": [3, 5, 10], "n_null_cohorts": args.null_cohorts,
               "mapping_fit_data": "historical training rows only; no monitoring region IDs enter the map",
               "pair_threshold_fit_data": "historical reference versus pseudo-current calibration cohorts only",
               "config_sha256": sha256(args.config),
               "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
               "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in output_files},
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "numpy": np.__version__, "pandas": pd.__version__,
                               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
               "scope_note": "exploratory diagnostic on already locked evaluation seeds; no threshold or model tuning on monitor truth"}
    (args.output / "audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cluster_rows": len(rows), "pair_rows": len(pair_rows),
                      "output": str(args.output), "region_summary": region_summary.to_dict(orient="records"),
                      "pair_summary": pair_summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
