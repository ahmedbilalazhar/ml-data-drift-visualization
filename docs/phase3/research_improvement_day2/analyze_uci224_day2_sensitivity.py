from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/uci224_source/uci224_curl.zip"
RUN = ROOT / "results/phase3/external_uci224_day2_v2"
OUT = ROOT / "results/phase3/external_uci224_day2_sensitivity_v2"


def load_batch(archive, batch):
    lines = archive.read(f"Dataset/batch{batch}.dat").decode("ascii").splitlines()
    x = np.zeros((len(lines), 128), dtype=float)
    y = np.empty(len(lines), dtype=np.int8)
    for row, line in enumerate(lines):
        values = line.split()
        y[row] = int(values[0])
        for item in values[1:]:
            column, value = item.split(":")
            x[row, int(column) - 1] = float(value)
    return x, y


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as archive:
        cohorts = {batch: load_batch(archive, batch) for batch in range(1, 11)}
    x_hist, y_hist = cohorts[1]
    indices = np.arange(len(y_hist))
    train_indices, holdout_indices = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.4, random_state=2242026).split(indices, y_hist))
    calibration_local, reference_local = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=2242027).split(
            holdout_indices, y_hist[holdout_indices]))
    calibration_indices = holdout_indices[calibration_local]
    reference_indices = holdout_indices[reference_local]
    scaler = StandardScaler().fit(x_hist[train_indices])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2242026)
    model.fit(scaler.transform(x_hist[train_indices]), y_hist[train_indices])
    x_cal, y_cal = x_hist[calibration_indices], y_hist[calibration_indices]
    p_cal = model.predict_proba(scaler.transform(x_cal))
    pred_cal = model.classes_[p_cal.argmax(axis=1)]
    calibration_accuracy = float(np.mean(pred_cal == y_cal))
    reference_x, reference_y = x_hist[reference_indices], y_hist[reference_indices]
    reference_prediction = model.predict(scaler.transform(reference_x))
    ref_props = Counter(map(int, reference_y))
    total_ref = len(reference_y)
    reference_weights = {label: count / total_ref for label, count in ref_props.items()}

    rows = []
    cal_confidence = p_cal.max(axis=1)
    rows.append({"cohort": "calibration", "n": len(y_cal), "accuracy": calibration_accuracy,
                 "mean_confidence": float(cal_confidence.mean()),
                 "mean_confidence_minus_correctness": float(cal_confidence.mean() - calibration_accuracy)})
    p_ref = model.predict_proba(scaler.transform(reference_x))
    rows.append({"cohort": "batch1_reference", "n": len(reference_y),
                 "accuracy": float(np.mean(reference_prediction == reference_y)),
                 "mean_confidence": float(p_ref.max(axis=1).mean()),
                 "mean_confidence_minus_correctness": float(p_ref.max(axis=1).mean() -
                                                               np.mean(reference_prediction == reference_y))})
    decomposition = []
    classes = np.arange(1, 7)
    for batch in range(2, 11):
        x, y = cohorts[batch]
        stored = np.load(RUN / f"batch{batch:02}_predictions.npz")
        prediction = stored["prediction"]
        confidence = stored["confidence"]
        correctness = prediction == y
        supported = [int(label) for label in classes if np.any(y == label)]
        weight_sum = sum(reference_weights.get(label, 0.0) for label in supported)
        standardized = 0.0
        for label in classes:
            mask = y == label
            if not mask.any():
                class_accuracy = np.nan
            else:
                class_accuracy = float(correctness[mask].mean())
                standardized += reference_weights.get(int(label), 0.0) / weight_sum * class_accuracy
            decomposition.append({"batch": batch, "class": int(label), "support": int(mask.sum()),
                                  "accuracy": class_accuracy,
                                  "batch1_reference_proportion": reference_weights.get(int(label), 0.0),
                                  "class_present": bool(mask.any())})
        observed = float(correctness.mean())
        rows.append({"cohort": f"batch{batch}", "n": len(y), "accuracy": observed,
                     "mean_confidence": float(confidence.mean()),
                     "mean_confidence_minus_correctness": float(confidence.mean() - observed)})
        decomposition.append({"batch": batch, "class": "COMMON_CLASS_STANDARDIZED",
                              "support": len(y), "accuracy": standardized,
                              "batch1_reference_proportion": 1.0,
                              "class_present": True})
        decomposition.append({"batch": batch, "class": "AGGREGATE_OBSERVED",
                              "support": len(y), "accuracy": observed,
                              "batch1_reference_proportion": 1.0,
                              "class_present": True})

    pd.DataFrame(rows).to_csv(OUT / "confidence_transfer.csv", index=False)
    pd.DataFrame(decomposition).to_csv(OUT / "class_mix_sensitivity.csv", index=False)
    report = {
        "status": "exploratory_complete",
        "question": "How much of cohort accuracy variation is associated with changing class proportions, and does source calibration transfer?",
        "protocol": "SENSITIVITY_QUESTION.md",
        "calibration_n": len(y_cal),
        "calibration_correct": int(np.sum(pred_cal == y_cal)),
        "calibration_accuracy": calibration_accuracy,
        "reference_n": len(reference_y),
        "reference_accuracy": float(np.mean(reference_prediction == reference_y)),
        "interpretation": "Common-class standardization is descriptive, weights only classes present per cohort, and is not a causal decomposition. Absent classes are not estimated.",
    }
    (OUT / "sensitivity_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
