"""Evaluate fixed future-harm predictors on paired, unseen synthetic mechanisms."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time

# This module sets numerical thread limits before importing NumPy.
from run_future_harm_unseen_family import DEV, FEATURES, TRAIN_RUN
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drift_monitoring.phase3.core import (
    FeatureBatch, baseline_x, fit_artifacts, historical_partition, innovations,
    label_probability, score_batch,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/phase3/SIX_HOUR_STRENGTHENING_20261120.md"
FAMILIES = ("stationary", "c7_flatten", "regional_reversal",
            "boundary_concentration", "sensor_sign_inversion")
DELAYS = (0, 5, 20)
ORIGINS = range(1, 20)
KEYS = ["seed", "family", "delay", "origin"]
MODELS = (*FEATURES, "development_prevalence", "last_released_harm")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def condition(x_clean, uniforms, family, start):
    x = x_clean.copy()
    p = label_probability(x_clean)
    if family == "c7_flatten":
        p[start:] = .30 * p[start:] + .70 * .5
    elif family == "regional_reversal":
        changed = (np.arange(len(x)) >= start) & (x[:, 0] >= 0)
        p[changed] = 1 - p[changed]
    elif family == "boundary_concentration":
        x[start:, :3] *= .25
        p = label_probability(x)
    elif family == "sensor_sign_inversion":
        x[start:, 0] *= -1
    elif family != "stationary":
        raise ValueError(family)
    return x, (uniforms < p).astype(np.int8)


def released_features(accuracy, ref_acc, delay, origin):
    end = max(0, origin - delay - 1)
    known = accuracy[:end]
    if not end:
        return dict(latest_accuracy=None, rolling_accuracy=None,
                    released_harm_fraction=None, latest_released_harm=None,
                    released_n=0, latest_age=25)
    return dict(latest_accuracy=float(known[-1]), rolling_accuracy=float(known[-5:].mean()),
                released_harm_fraction=float((known[-5:] < ref_acc - .05).mean()),
                latest_released_harm=int(known[-1] < ref_acc - .05),
                released_n=end * 500, latest_age=origin - end)


def seed_run(job):
    seed, onset, deadline = job
    if time.monotonic() >= deadline:
        raise TimeoutError("Forecast time budget exhausted")
    with threadpool_limits(limits=1):
        art = fit_artifacts(seed, *(historical_partition(seed, stage, size)
                                    for stage, size in enumerate((4000, 2000, 2000))))
        ref = art.reference_correct / art.reference_n
        z, u = innovations(seed, 3, 12000)
        clean = baseline_x(z)
        ids = seed * 1_000_000 + 500_000 + np.arange(len(u))
        score_caches = {}
        online, truth = [], []
        for family in FAMILIES:
            x, y = condition(clean, u, family, (onset - 1) * 500)
            key = "stationary" if family in FAMILIES[:3] else family
            if key not in score_caches:
                rows = []
                for batch in range(1, 25):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Forecast time budget exhausted")
                    sl = slice((batch - 1) * 500, batch * 500)
                    rec, cache, _ = score_batch(art, FeatureBatch(x[sl], ids[sl], seed, key, batch))
                    marginal = [rec[f"x{i}_adjusted_p"] for i in range(8)]
                    pca = [rec[f"pc{i}_adjusted_p"] for i in range(3)]
                    features = dict(estimate=rec["cal_estimate"], concern=rec["cal_concern"],
                                    alarm_count=sum(bool(rec[f"x{i}_flag"]) for i in range(8))
                                    + sum(p <= .05 for p in pca),
                                    min_adjusted_p=min(marginal + pca), input_alarm=int(rec["input_alarm"]))
                    rows.append((features, cache["prediction"]))
                score_caches[key] = rows
            rows = score_caches[key]
            accuracy = np.asarray([(pred == y[b*500:(b+1)*500]).mean()
                                   for b, (_, pred) in enumerate(rows)])
            for delay in DELAYS:
                for origin in ORIGINS:
                    keys = dict(seed=seed, family=family, delay=delay, origin=origin)
                    past = released_features(accuracy, ref, delay, origin)
                    end = max(0, origin - delay - 1)
                    counterfactual = accuracy.copy()
                    counterfactual[end:] = 1 - counterfactual[end:]
                    if past != released_features(counterfactual, ref, delay, origin):
                        raise AssertionError(f"Future truth leaked at {keys}")
                    future = float(accuracy[origin:origin + 5].mean())
                    online.append(dict(**keys, time_norm=origin/24, delay_norm=delay/20,
                                       **rows[origin - 1][0], **past))
                    truth.append(dict(**keys, onset=None if family == "stationary" else onset,
                                      future_accuracy=future, reference_accuracy=ref,
                                      future_harm=int(future < ref - .05)))
        return online, truth


def model_frame(frame, cols):
    return frame[cols].replace([np.inf, -np.inf], np.nan).fillna(-1)


def fit_models():
    source_lock = json.loads((TRAIN_RUN / "protocol_lock.json").read_text(encoding="utf-8"))
    if source_lock["protocol"] != "future_harm_forecast_20261116_v1":
        raise ValueError("Unexpected forecast training source")
    source_features = TRAIN_RUN / "predictions/online_features.csv"
    source_truth = TRAIN_RUN / "evaluator_truth/future_targets.csv"
    x, y = pd.read_csv(source_features), pd.read_csv(source_truth)
    x, y = x[x.seed.isin(DEV)], y[y.seed.isin(DEV)]
    train = x.merge(y, on=["seed", "condition", "delay", "batch"], validate="one_to_one")
    if len(train) != len(x) or len(train) != len(y) or set(train.seed) != set(DEV):
        raise ValueError("Incomplete training join")
    models = {}
    with threadpool_limits(limits=1):
        for name, cols in FEATURES.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000,
                                                                        random_state=2026))
            models[name] = model.fit(model_frame(train, cols), train.future_harm)
    return models, train.future_harm.mean(), [source_features, source_truth, TRAIN_RUN/"protocol_lock.json"]


def summarize(online, truth, models, prevalence, out):
    if online.duplicated(KEYS).any() or truth.duplicated(KEYS).any():
        raise AssertionError("Duplicate forecast keys")
    expected_labels = (online.origin - online.delay - 1).clip(lower=0) * 500
    if not online.released_n.eq(expected_labels).all():
        raise AssertionError("Availability count mismatch")
    joined_keys = online[KEYS].merge(truth[KEYS], on=KEYS, validate="one_to_one")
    if len(joined_keys) != len(truth) or len(joined_keys) != len(online):
        raise AssertionError("Forecast and truth keys differ")
    results = online[KEYS].copy()
    for name, cols in FEATURES.items():
        results[name] = models[name].predict_proba(model_frame(online, cols))[:, 1]
    results["development_prevalence"] = prevalence
    results["last_released_harm"] = online["latest_released_harm"].fillna(prevalence)
    joined = results.merge(truth[KEYS + ["future_harm"]], on=KEYS, validate="one_to_one")
    stats, seed_losses = [], []
    for (family, delay), rows in joined.groupby(["family", "delay"]):
        y = rows.future_harm
        for model in MODELS:
            loss = (rows[model] - y)**2
            stats.append(dict(family=family, delay=delay, model=model, origins=len(rows),
                              seeds=rows.seed.nunique(), prevalence=y.mean(), brier=loss.mean(),
                              auc=roc_auc_score(y, rows[model]) if y.nunique() == 2 else None))
            for seed, value in loss.groupby(rows.seed).mean().items():
                seed_losses.append(dict(family=family, delay=delay, seed=seed, model=model,
                                        brier=value))
    seed_table = pd.DataFrame(seed_losses).pivot(index=["seed", "family", "delay"],
                                                columns="model", values="brier")
    gains = pd.DataFrame({
        "monitor_vs_time": seed_table.time_only - seed_table.unlabeled_monitor,
        "released_vs_monitor": seed_table.unlabeled_monitor - seed_table.full_released,
    })
    matrix = gains.unstack(["family", "delay"])
    means = matrix.mean()
    rng = np.random.default_rng(972999)
    values = matrix.to_numpy()
    resampled = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    width = float(np.quantile(np.abs(resampled - means.to_numpy()).max(axis=1), .95))
    contrasts = means.rename("brier_gain").rename_axis(["contrast", "family", "delay"]).reset_index()
    contrasts["simultaneous_low"] = contrasts.brier_gain - width
    contrasts["simultaneous_high"] = contrasts.brier_gain + width
    contrasts["seeds"] = len(matrix)
    online.to_csv(out/"online_features.csv", index=False)
    truth.to_csv(out/"evaluator_targets.csv", index=False)
    results.to_csv(out/"forecast_probabilities.csv", index=False)
    pd.DataFrame(stats).to_csv(out/"metrics_by_family_delay.csv", index=False)
    gains.reset_index().to_csv(out/"paired_seed_gains.csv", index=False)
    contrasts.to_csv(out/"simultaneous_contrasts.csv", index=False)


def run(out, smoke, workers, max_seconds):
    start = time.monotonic()
    seeds = list(range(972000, 972002) if smoke else range(972100, 972180))
    onsets = [5, 9] if smoke else [5, 9, 13, 17] * 20
    if set(seeds) & set(DEV):
        raise AssertionError("Evaluation seeds overlap training")
    models, prevalence, sources = fit_models()
    out.mkdir(parents=True, exist_ok=False)
    lock = dict(status="running", smoke=smoke, scope="synthetic forecast stress study",
                evaluation_seeds=seeds, training_seeds=DEV, onset_batches=dict(zip(seeds, onsets)),
                families=FAMILIES, delays=DELAYS, origins=list(ORIGINS), models=FEATURES,
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__,
                source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in
                               [*sources, PLAN, Path(__file__).resolve(),
                                ROOT/"scripts/run_future_harm_unseen_family.py",
                                ROOT/"src/drift_monitoring/phase3/core.py"]})
    lock_path = out/"protocol_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    features, targets = [], []
    jobs = [(seed, onset, start + max_seconds) for seed, onset in zip(seeds, onsets)]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (x, y) in enumerate(pool.map(seed_run, jobs), start=1):
                features.extend(x)
                targets.extend(y)
                if n % 5 == 0 or n == len(jobs):
                    print(f"{n}/{len(jobs)} seeds, {time.monotonic()-start:.1f}s", flush=True)
        online, truth = pd.DataFrame(features), pd.DataFrame(targets)
        n_expected = len(seeds)*len(FAMILIES)*len(DELAYS)*len(ORIGINS)
        if len(online) != n_expected or len(truth) != n_expected:
            raise AssertionError("Missing forecast combinations")
        if {"future_harm", "future_accuracy", "reference_accuracy"} & set(online.columns):
            raise AssertionError("Evaluator fields present in online output")
        summarize(online, truth, models, prevalence, out)
        validation = dict(status="passed", seed_count=len(seeds), forecasts_per_model=n_expected,
                          outcome_mutation_checks=n_expected, strict_release_clock=True,
                          source_sha256=lock["source_sha256"],
                          output_sha256={p.name:digest(p) for p in sorted(out.glob("*.csv"))},
                          elapsed_seconds=time.monotonic()-start)
        (out/"validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        lock["status"] = "complete"
    except BaseException:
        lock["status"] = "incomplete"
        if features:
            pd.DataFrame(features).to_csv(out/"partial_online_features.csv", index=False)
            pd.DataFrame(targets).to_csv(out/"partial_evaluator_targets.csv", index=False)
        raise
    finally:
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-seconds", type=int, default=14400)
    args = parser.parse_args()
    run(args.out, args.smoke, args.workers, args.max_seconds)
