"""Recreate a predeclared seed subset in fresh memory and compare saved rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.v2 import (  # noqa: E402
    BATCHES, BATCH_SIZE, MODELS, TASKS, V1_CONDITIONS, XOR_CONDITIONS,
    FeatureBatch, fit_models, historical_partition, monitoring_partition, score_model,
)

SAMPLE = (1000, 1375, 1750, 2125, 2499)
TIMING = {"prediction_seconds", "original_ks_seconds", "latent_ks_seconds",
          "calibrated_impact_seconds", "raw_impact_seconds", "score_seconds"}


def compare(run: Path, out: Path) -> dict:
    run, out = run.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    online_saved = pd.read_csv(run / "online_batches.csv")
    evaluator_saved = pd.read_csv(run / "evaluator_batches.csv")
    tasks = {"v1_synthetic": V1_CONDITIONS, "xor_synthetic": XOR_CONDITIONS}
    online_rows, evaluator_rows = [], []
    for task in TASKS:
        for seed in SAMPLE:
            train = historical_partition(task, seed, 0, 4000)
            calibration = historical_partition(task, seed, 1, 2000)
            reference = historical_partition(task, seed, 2, 2000)
            artifacts = fit_models(seed, train, calibration, reference)
            for condition in tasks[task]:
                part = monitoring_partition(task, seed, condition)
                for model, artifact in artifacts.items():
                    records, caches = [], []
                    for batch in range(1, BATCHES + 1):
                        sl = slice((batch-1)*BATCH_SIZE, batch*BATCH_SIZE)
                        record, cache = score_model(artifact,
                            FeatureBatch(part.x[sl], part.ids[sl], seed, condition, batch), task, model)
                        records.append(record); caches.append(cache)
                    for record, cache in zip(records, caches):
                        record = dict(record)
                        record["prediction_id_sha256"] = hashlib.sha256(
                            np.asarray(cache["ids"], dtype="<i8").tobytes()).hexdigest()
                        online_rows.append(record)
                    pred = np.stack([cache["prediction"] for cache in caches])
                    labels = part.y.reshape(BATCHES, BATCH_SIZE)
                    for batch in range(BATCHES):
                        cm = confusion_matrix(labels[batch], pred[batch], labels=(0,1))
                        tn, fp, fn, tp = (int(v) for v in cm.ravel())
                        macro = np.mean([2*tn/(2*tn+fp+fn) if (2*tn+fp+fn) else np.nan,
                                         2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan])
                        correct = int((labels[batch] == pred[batch]).sum())
                        evaluator_rows.append(dict(task=task, seed=seed, condition=condition,
                            model=model, batch=batch+1, n=BATCH_SIZE, correct=correct,
                            accuracy=correct/BATCH_SIZE, macro_f1=float(macro),
                            class0_n=int((labels[batch] == 0).sum()),
                            class1_n=int((labels[batch] == 1).sum()),
                            reference_correct=artifact.reference_correct,
                            reference_n=artifact.reference_n,
                            harm=bool(20*(artifact.reference_correct*BATCH_SIZE-
                                          correct*artifact.reference_n) > artifact.reference_n*BATCH_SIZE)))
    online_rebuilt, evaluator_rebuilt = pd.DataFrame(online_rows), pd.DataFrame(evaluator_rows)
    idx = ["task", "seed", "condition", "model", "batch"]
    saved_o = online_saved[online_saved.seed.isin(SAMPLE)].sort_values(idx).reset_index(drop=True)
    rebuilt_o = online_rebuilt.sort_values(idx).reset_index(drop=True)
    common_o = [c for c in saved_o.columns if c in rebuilt_o.columns and c not in TIMING]
    max_float_diff = 0.0
    float_columns = []
    for col in common_o:
        left, right = saved_o[col], rebuilt_o[col]
        if pd.api.types.is_numeric_dtype(left) and (pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right)):
            a, b = left.to_numpy(float), right.to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise AssertionError(f"sample reproduction NaN mask differs: {col}")
            finite = np.isfinite(a) & np.isfinite(b)
            delta = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
            max_float_diff = max(max_float_diff, delta)
            float_columns.append(col)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14, equal_nan=True,
                                       err_msg=f"sample reproduction numeric mismatch: {col}")
        else:
            pd.testing.assert_series_equal(left, right, check_dtype=False, check_exact=True,
                                           check_names=True, obj=col)
    saved_e = evaluator_saved[evaluator_saved.seed.isin(SAMPLE)].sort_values(idx).reset_index(drop=True)
    rebuilt_e = evaluator_rebuilt.sort_values(idx).reset_index(drop=True)
    common_e = list(saved_e.columns)
    for col in common_e:
        left, right = saved_e[col], rebuilt_e[col]
        if pd.api.types.is_numeric_dtype(left) and (pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right)):
            a, b = left.to_numpy(float), right.to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise AssertionError(f"evaluator NaN mask differs: {col}")
            finite = np.isfinite(a) & np.isfinite(b)
            delta = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
            max_float_diff = max(max_float_diff, delta)
            float_columns.append(col)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14, equal_nan=True,
                                       err_msg=f"evaluator numeric mismatch: {col}")
        else:
            pd.testing.assert_series_equal(left, right, check_dtype=False, check_exact=True,
                                           check_names=True, obj=col)
    receipt = {"status":"passed", "sample_seeds_per_task":list(SAMPLE),
        "conditions":{k:list(v) for k,v in tasks.items()},
        "online_rows_compared":len(saved_o), "online_columns_compared":len(common_o),
        "timing_fields_excluded":sorted(TIMING), "evaluator_rows_compared":len(saved_e),
        "evaluator_columns_compared":len(common_e),
        "equality":"non-floating scientific fields exact; floating fields allclose rtol=1e-12 atol=1e-14; timings excluded",
        "maximum_absolute_float_difference":max_float_diff,
        "floating_columns_compared":sorted(set(float_columns)),
        "method":"fresh deterministic generation and scoring outside the main run output; tolerance accounts for floating-point/library rounding only"}
    (out / "sample_reproduction_comparison.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(compare(a.run,a.out),indent=2))
