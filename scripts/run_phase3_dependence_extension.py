"""Run the locked exploratory pairwise-dependence detector extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_monitoring.phase3.core import historical_partition, monitoring_partition, holm
from drift_monitoring.phase3.dependence import calibrate_threshold, max_correlation_change

PROTOCOL = ROOT / "docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md"
DETECTOR = ROOT / "src/drift_monitoring/phase3/dependence.py"
CONDITIONS = tuple(f"C{i}" for i in range(7))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(mode: str, out: Path, threshold_file: Path | None = None):
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if mode == "development":
        seeds, conditions = range(930000, 930030), ("C0",)
        threshold = None
    elif mode == "final":
        if threshold_file is None:
            raise ValueError("final mode requires the frozen development threshold JSON")
        lock = json.loads(threshold_file.read_text(encoding="utf-8"))
        if (lock.get("mode") != "development" or lock.get("protocol_sha256") != digest(PROTOCOL)
                or lock.get("source_sha256") != digest(Path(__file__))
                or lock.get("detector_sha256") != digest(DETECTOR)):
            raise ValueError("threshold file is not from this protocol's development run")
        threshold = float(lock["threshold"])
        seeds, conditions = range(930100, 930130), CONDITIONS
    else:
        raise ValueError(mode)

    out.mkdir(parents=True)
    rows = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            train = historical_partition(seed, 0, 4000).x
            ref = historical_partition(seed, 2, 2000).x[-1000:]
            scaler = StandardScaler().fit(train)
            train_z = scaler.transform(train)
            ref_z = scaler.transform(ref)
            pca = PCA(n_components=3, svd_solver="full").fit(train_z)
            ref_pca = pca.transform(ref_z)
            for condition in conditions:
                stream = monitoring_partition(seed, condition, n=12000)
                for batch in range(1, 25):
                    x = stream.x[(batch-1)*500:batch*500]
                    score, pair = max_correlation_change(ref, x)
                    current_z = scaler.transform(x)
                    pvals = [ks_2samp(ref_z[:, j], current_z[:, j], method="auto").pvalue
                             for j in range(ref_z.shape[1])]
                    current_pca = pca.transform(current_z)
                    pvals.extend(ks_2samp(ref_pca[:, j], current_pca[:, j], method="auto").pvalue
                                  for j in range(current_pca.shape[1]))
                    ks_pca_alarm = bool(np.any(holm(pvals) <= .05))
                    rows.append(dict(seed=seed, condition=condition, batch=batch,
                        score=score, pair_i=pair[0], pair_j=pair[1],
                        ks_pca_alarm=ks_pca_alarm,
                        alarm=(score >= threshold) if threshold is not None else False))
    batches = pd.DataFrame(rows)
    if mode == "development":
        threshold = calibrate_threshold(batches.loc[batches.condition.eq("C0"), "score"].to_numpy(), .95)
        batches["alarm"] = batches.score >= threshold
        threshold_record = dict(mode="development", threshold=threshold, quantile=.95,
            quantile_method="higher", seeds=[930000, 930029], stationary_batch_scores=len(batches),
            protocol_sha256=digest(PROTOCOL), source_sha256=digest(Path(__file__)),
            detector_sha256=digest(DETECTOR),
            statistic="maximum absolute pairwise Pearson correlation difference")
        (out / "threshold.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")
    else:
        threshold_record = dict(mode="final", threshold=threshold,
            development_threshold_sha256=digest(threshold_file), protocol_sha256=digest(PROTOCOL),
            source_sha256=digest(Path(__file__)), detector_sha256=digest(DETECTOR))
        (out / "threshold_used.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")

    batches.to_csv(out / "online_dependence_scores.csv", index=False)
    seed_rows = []
    for (seed, condition), group in batches.groupby(["seed", "condition"]):
        onset = group[group.batch.between(9, 12)]
        pre = group[group.batch < 9]
        seed_rows.append(dict(seed=int(seed), condition=condition,
            onset_detected=bool(onset.alarm.any()), detected_at_any_time=bool(group.alarm.any()),
            pre_onset_false_alarm=bool(pre.alarm.any()),
            pair01_selected_onset=bool((onset.alarm & (onset.pair_i == 0) & (onset.pair_j == 1)).any()),
            ks_pca_onset_detected=bool(onset.ks_pca_alarm.any()),
            alarms=int(group.alarm.sum()), batches=len(group)))
    summary = pd.DataFrame(seed_rows)
    summary.to_csv(out / "per_seed_summary.csv", index=False)
    if mode == "final":
        condition_summary = summary.groupby("condition").agg(seeds=("seed", "nunique"),
            onset_detection_rate=("onset_detected", "mean"),
            any_time_detection_rate=("detected_at_any_time", "mean"),
            pre_onset_false_alarm_rate=("pre_onset_false_alarm", "mean"),
            ks_pca_onset_detection_rate=("ks_pca_onset_detected", "mean"),
            pair01_onset_selection_rate=("pair01_selected_onset", "mean"),
            mean_alarms=("alarms", "mean")).reset_index()
        condition_summary["false_alarms_per_1000_batches"] = np.where(
            condition_summary.condition.eq("C0"), condition_summary.mean_alarms / 24 * 1000, np.nan)
        condition_summary.to_csv(out / "condition_summary.csv", index=False)
        # Paired same-stream comparison against the locked marginal/PCA family.
        paired = []
        rng = np.random.default_rng(16112026)
        eval_summ = summary[summary.condition.eq("C6")].set_index("seed").sort_index()
        vals = (eval_summ.onset_detected.astype(float) - eval_summ.ks_pca_onset_detected.astype(float)).to_numpy()
        boot = vals[rng.integers(0, len(vals), size=(2000, len(vals)))].mean(axis=1)
        paired.append(dict(condition="C6", metric="dependence_detector_onset_detection_rate",
            mean_difference=float(vals.mean()), lower=float(np.quantile(boot, .025)),
            upper=float(np.quantile(boot, .975)), seeds=len(vals),
            comparator="same-stream original-feature plus PCA KS alarm"))
        pd.DataFrame(paired).to_csv(out / "bootstrap_summary.csv", index=False)
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        protocol_sha256=digest(PROTOCOL), source_sha256=digest(Path(__file__)),
        detector_sha256=digest(DETECTOR), mode=mode)
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path)
    args = parser.parse_args()
    run(args.mode, args.out, args.threshold_file)
