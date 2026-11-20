from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/uci224_source/uci224_curl.zip"
LOCK = ROOT / "docs/phase3/research_improvement_day2/PROTOCOL_LOCK.json"
OUT = ROOT / "results/phase3/external_uci224_day2_v2"
CLASSES = np.arange(1, 7)


def load_cohorts():
    cohorts = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for batch in range(1, 11):
            lines = archive.read(f"Dataset/batch{batch}.dat").decode("ascii").splitlines()
            x = np.zeros((len(lines), 128), dtype=np.float64)
            y = np.empty(len(lines), dtype=np.int8)
            for row, line in enumerate(lines):
                tokens = line.split()
                y[row] = int(tokens[0])
                for token in tokens[1:]:
                    feature, value = token.split(":")
                    x[row, int(feature) - 1] = float(value)
            cohorts[batch] = (x, y)
    return cohorts


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    size = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (size - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def main():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if actual_hash != lock["dataset"]["archive_sha256"]:
        raise RuntimeError("source archive hash differs from locked protocol")
    cohorts = load_cohorts()
    x_hist, y_hist = cohorts[1]
    indices = np.arange(len(y_hist))
    train_indices, holdout_indices = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.4, random_state=2242026).split(indices, y_hist))
    calibration_local, reference_local = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=2242027).split(
            holdout_indices, y_hist[holdout_indices]))
    calibration_indices = holdout_indices[calibration_local]
    reference_indices = holdout_indices[reference_local]
    train_x, train_y = x_hist[train_indices], y_hist[train_indices]
    calibration_x, calibration_y = x_hist[calibration_indices], y_hist[calibration_indices]
    reference_x, reference_y = x_hist[reference_indices], y_hist[reference_indices]
    if set(np.concatenate([train_indices, calibration_indices, reference_indices])) != set(indices):
        raise AssertionError("historical partition does not exactly cover batch 1")

    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
    model.fit(scaler.transform(train_x), train_y)
    train_scores = scaler.transform(train_x)
    pca = PCA(n_components=3, svd_solver="full").fit(train_scores)
    calibration_probability = model.predict_proba(scaler.transform(calibration_x))
    calibration_confidence = calibration_probability.max(axis=1).reshape(-1, 1)
    calibration_correct = (model.classes_[calibration_probability.argmax(axis=1)] == calibration_y).astype(int)
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
    calibrator.fit(calibration_confidence, calibration_correct)
    reference_prediction = model.predict(scaler.transform(reference_x))
    reference_accuracy = accuracy_score(reference_y, reference_prediction)
    reference_input = reference_x
    reference_pca = pca.transform(scaler.transform(reference_x))

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "protocol_lock.json").write_text(json.dumps({
        **lock,
        "status": "executed",
        "source_archive_sha256_verified": actual_hash,
        "historical_partition_row_indices": {
            "batch1_train": train_indices.tolist(),
            "batch1_calibration": calibration_indices.tolist(),
            "batch1_reference": reference_indices.tolist(),
        },
        "historical_partition_rows": {"train": len(train_indices),
                                       "calibration": len(calibration_indices),
                                       "reference": len(reference_indices)},
        "fit_class_order": model.classes_.tolist(),
    }, indent=2), encoding="utf-8")

    cohort_rows, class_rows, test_rows = [], [], []
    for batch in range(1, 11):
        x, y = cohorts[batch]
        counts = Counter(map(int, y))
        for label in CLASSES:
            n_class = counts[int(label)]
            class_rows.append({"batch": batch, "class": int(label), "count": n_class,
                               "proportion": n_class / len(y)})
        if batch == 1:
            x_eval, y_eval = reference_x, reference_y
        else:
            x_eval, y_eval = x, y

        standardized = scaler.transform(x_eval)
        probabilities = model.predict_proba(standardized)
        prediction = model.classes_[probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)
        calibrated_correctness = calibrator.predict_proba(confidence.reshape(-1, 1))[:, 1]
        expected_correct = float(calibrated_correctness.sum())
        standard_deviation = float(np.sqrt(np.sum(
            calibrated_correctness * (1.0 - calibrated_correctness))))
        interval_low = max(0.0, expected_correct - 0.5 - norm.ppf(0.975) * standard_deviation) / len(y_eval)
        interval_high = min(float(len(y_eval)), expected_correct + 0.5 + norm.ppf(0.975) * standard_deviation) / len(y_eval)
        pvalues = [ks_2samp(reference_input[:, j], x_eval[:, j], method="auto").pvalue
                   for j in range(x_eval.shape[1])]
        projected = pca.transform(standardized)
        pvalues.extend(ks_2samp(reference_pca[:, j], projected[:, j], method="auto").pvalue
                       for j in range(projected.shape[1]))
        adjusted = holm(pvalues)
        per_class_p, per_class_r, per_class_f, support = precision_recall_fscore_support(
            y_eval, prediction, labels=CLASSES, zero_division=0)
        for index, label in enumerate(CLASSES):
            cohort_class = next(r for r in class_rows if r["batch"] == batch and r["class"] == int(label))
            cohort_class.update(precision=float(per_class_p[index]), recall=float(per_class_r[index]),
                                f1=float(per_class_f[index]), support=int(support[index]))
        acc = float(accuracy_score(y_eval, prediction))
        estimate = float(calibrated_correctness.mean())
        cohort_rows.append({
            "batch": batch, "cohort_rows": len(y_eval),
            "accuracy": acc, "macro_f1_fixed_six_classes": float(f1_score(
                y_eval, prediction, labels=CLASSES, average="macro", zero_division=0)),
            "estimated_accuracy": estimate, "absolute_estimation_error": abs(estimate - acc),
            "raw_mean_max_probability": float(confidence.mean()),
            "historical_reference_accuracy": float(reference_accuracy),
            "prediction_interval_lower": interval_low, "prediction_interval_upper": interval_high,
            "interval_method": "plug-in Poisson-binomial normal approximation with continuity correction",
            "interval_width": interval_high - interval_low,
            "interval_covers_observed_accuracy": bool(interval_low <= acc <= interval_high),
            "input_tests_rejected": int((adjusted < 0.05).sum()),
            "input_test_family_size": len(pvalues), "any_input_alarm": bool((adjusted < 0.05).any()),
            "class_counts": json.dumps(dict(sorted(counts.items()))),
            "class_proportions": json.dumps({str(k): v / len(y_eval) for k, v in sorted(counts.items())}),
        })
        np.savez_compressed(OUT / f"batch{batch:02}_predictions.npz", prediction=prediction,
                            probability=probabilities, confidence=confidence,
                            calibrated_correctness=calibrated_correctness)
        for feature, pvalue in enumerate(pvalues):
            test_rows.append({"batch": batch, "test": f"feature_or_pc_{feature:03}",
                              "raw_p": float(pvalue), "holm_adjusted_p": float(adjusted[feature]),
                              "rejected": bool(adjusted[feature] < 0.05)})

    pd.DataFrame(cohort_rows).to_csv(OUT / "cohort_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "class_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT / "input_tests.csv", index=False)
    summary = pd.DataFrame(cohort_rows)
    summary.to_json(OUT / "summary.json", orient="records", indent=2)
    print(json.dumps({
        "status": "complete", "cohorts": len(cohort_rows),
        "monitoring_rows": int(summary.loc[summary.batch > 1, "cohort_rows"].sum()),
        "macro_f1": summary.loc[summary.batch > 1, "macro_f1_fixed_six_classes"].tolist(),
        "accuracy_estimation_mae": float(summary.loc[summary.batch > 1, "absolute_estimation_error"].mean()),
        "interval_coverage_over_9_cohorts": float(summary.loc[summary.batch > 1,
            "interval_covers_observed_accuracy"].mean()),
        "interval_mean_width": float(summary.loc[summary.batch > 1, "interval_width"].mean()),
        "input_alarm_batches": int(summary.loc[summary.batch > 1, "any_input_alarm"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
