"""Historical-only mapping and frozen-representation sensitivity for region localization."""
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             f1_score, log_loss, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.strengthening_generators import (  # noqa: E402
    generate_historical, generate_monitoring, historical_region_ids,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3_strengthening_v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/cluster_mapping_sensitivity_20261117_v2")
    args = parser.parse_args()
    args.config, args.output = args.config.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = list(map(int, cfg["seed_banks"]["evaluation"]))
    train_n = int(cfg["history"]["train_rows"])
    rows, maps = [], []
    for seed in seeds:
        train = generate_historical("five_region", seed, 1, train_n, 8)
        cal = generate_historical("five_region", seed, 2, int(cfg["history"]["calibration_rows"]), 8)
        ref = generate_historical("five_region", seed, 3, int(cfg["history"]["reference_rows"]), 8)
        # The coordinate system is frozen from training data only. PCA-3 is the
        # predeclared descriptive anchor; PCA-2 is included as a sensitivity.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train.x)
        ztrain = scaler.transform(train.x)
        zcal, zref = scaler.transform(cal.x), scaler.transform(ref.x)
        zmon = scaler.transform(generate_monitoring("five_region", "M0_stationary", seed,
                                                    feature_count=8).monitor.x)
        regions = historical_region_ids("five_region", seed, 1, len(train.x), 8).astype(int)
        pcas = {"scaled_full": None}
        for dim in (2, 3):
            pcas[f"frozen_pca{dim}"] = PCA(n_components=dim, svd_solver="full").fit(ztrain)
        for representation, pca in pcas.items():
            tr = ztrain if pca is None else pca.transform(ztrain)
            te = zmon if pca is None else pca.transform(zmon)
            for k in (3, 5, 10):
                km = MiniBatchKMeans(n_clusters=k, random_state=seed + 49000 + k,
                                     n_init=10, batch_size=1024, max_iter=300,
                                     reassignment_ratio=0.01).fit(tr)
                train_cluster = km.predict(tr)
                test_cluster = km.predict(te)
                contingency = np.zeros((k, 5), dtype=np.int64)
                for c, r in zip(train_cluster, regions, strict=True):
                    contingency[c, r] += 1
                # Dominant-region map is many-to-one: multiple clusters may
                # represent the same region, unlike a forced Hungarian match.
                dominant = contingency.argmax(axis=1)
                probs = (contingency + 0.5) / (contingency.sum(axis=1, keepdims=True) + 2.5)
                truth = generate_monitoring("five_region", "M0_stationary", seed,
                                           feature_count=8).evaluator.region_id.astype(int)
                mapped = dominant[test_cluster]
                p = probs[test_cluster]
                target = (truth == 2).astype(int)
                rows.append({"seed": seed, "representation": representation, "clusters": k,
                             "ari": float(adjusted_rand_score(truth, test_cluster)),
                             "many_to_one_accuracy": float(np.mean(mapped == truth)),
                             "many_to_one_macro_f1": float(f1_score(truth, mapped, labels=list(range(5)),
                                                                     average="macro", zero_division=0)),
                             "region2_precision": float(np.sum((mapped == 2) & (truth == 2)) / max(1, np.sum(mapped == 2))),
                             "region2_recall": float(np.mean(mapped[truth == 2] == 2)),
                             "region2_ap": float(average_precision_score(target, p[:, 2])),
                             "region2_auc": float(roc_auc_score(target, p[:, 2])),
                             "region2_brier": float(np.mean((p[:, 2] - target) ** 2)),
                             "region2_logloss": float(log_loss(target, p[:, 2], labels=[0, 1])),
                             "mapping_coverage": 1.0,
                             "nonempty_clusters": int(np.sum(contingency.sum(axis=1) > 0)),
                             "cluster_entropy_nats": float(np.mean(-np.sum(probs * np.log(probs), axis=1)))})
                rr, cc = linear_sum_assignment(-contingency)
                maps.append({"seed": seed, "representation": representation, "clusters": k,
                             "contingency": contingency.tolist(), "dominant_many_to_one": dominant.tolist(),
                             "dominant_probabilities": probs.tolist(),
                             "hungarian": {str(int(c)): int(r) for c, r in zip(rr, cc, strict=True)},
                             "cluster_entropy_nats": (-np.sum(probs * np.log(probs), axis=1)).tolist()})
    args.output.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "cluster_mapping_sensitivity.csv", index=False)
    pd.DataFrame(maps).to_json(args.output / "historical_mappings.json", orient="records", indent=2)
    summary = frame.groupby(["representation", "clusters"], as_index=False).agg(
        seeds=("seed", "nunique"), ari_mean=("ari", "mean"), ari_sd=("ari", "std"),
        macro_f1_mean=("many_to_one_macro_f1", "mean"), macro_f1_sd=("many_to_one_macro_f1", "std"),
        region2_ap_mean=("region2_ap", "mean"), region2_ap_sd=("region2_ap", "std"),
        region2_brier_mean=("region2_brier", "mean"), recall_mean=("region2_recall", "mean"),
        precision_mean=("region2_precision", "mean"), entropy_mean=("cluster_entropy_nats", "mean")
    )
    summary.to_csv(args.output / "cluster_mapping_summary.csv", index=False)
    sources = [Path(__file__), args.config, ROOT / "src/drift_monitoring/phase3/strengthening_generators.py"]
    output_files = [args.output / n for n in ("cluster_mapping_sensitivity.csv", "historical_mappings.json", "cluster_mapping_summary.csv")]
    receipt = {"status": "complete", "protocol": "frozen-training-representation-mapping-sensitivity-v1",
               "seed_bank": seeds, "representations": list(pcas), "cluster_counts": [3, 5, 10],
               "mapping_fit_data": "historical training rows only", "evaluation_truth_use": "metrics only",
               "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
               "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in output_files},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
               "scope_note": "posthoc diagnostic on locked evaluation seeds; does not alter primary detector"}
    (args.output / "audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {len(frame)} seed-level rows to {args.output}")


if __name__ == "__main__":
    main()
