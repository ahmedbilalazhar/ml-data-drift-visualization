"""Outcome-informed IDM-Viz v2 extension primitives.

The extension remains separate from the frozen v1 implementation and result paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import warnings

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .core import (FeatureBatch, historical_partition as v1_historical_partition,
                   holm, impact, monitoring_partition as v1_monitoring_partition)


TASKS = ("v1_synthetic", "xor_synthetic")
MODELS = ("logistic", "shallow_tree")
V1_CONDITIONS = tuple(f"C{i}" for i in range(7))
XOR_CONDITIONS = tuple(f"X{i}" for i in range(5))
DELAYS = (0, 5, 20)
BATCH_SIZE = 500
BATCHES = 24
CONCERN_THRESHOLD = 0.95
HARM_MARGIN = Fraction(1, 20)


@dataclass(frozen=True)
class Part:
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray


@dataclass
class Artifacts:
    scaler: StandardScaler
    model: object
    pca: object
    calibration: IsotonicRegression
    reference_x: np.ndarray
    reference_z: np.ndarray
    reference_correct: int
    reference_n: int
    atc: float
    fit_seconds: float
    fit_ids: dict


def _xor_law(x: np.ndarray, noise_uniform: np.ndarray, noise_rate: float) -> np.ndarray:
    base = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
    return np.logical_xor(base, noise_uniform < noise_rate).astype(np.int8)


def _xor_historical(seed: int, stage: int, n: int) -> Part:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stage), 2]))
    x = rng.normal(size=(n, 8))
    noise = rng.random(n)
    ids = int(seed) * 1_000_000_000 + stage * 100_000 + np.arange(n, dtype=np.int64)
    return Part(x, _xor_law(x, noise, 0.10), ids)


def historical_partition(task: str, seed: int, stage: int, n: int) -> Part:
    if task == "v1_synthetic":
        p = v1_historical_partition(seed, stage, n)
        return Part(p.x, p.y, p.ids)
    if task == "xor_synthetic":
        return _xor_historical(seed, stage, n)
    raise ValueError(f"unknown task {task}")


def _xor_monitoring(seed: int, condition: str, n: int = 12000, onset: int = 4000) -> Part:
    if condition not in XOR_CONDITIONS:
        raise ValueError(condition)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 3, 2]))
    x = rng.normal(size=(n, 8))
    noise = rng.random(n)
    rows = np.arange(n)
    post = rows >= onset
    if condition == "X1":
        x[post, 7] += 1.5
    elif condition == "X2":
        x[post, 0] += 0.75
    y = _xor_law(x, noise, 0.10)
    if condition == "X3":
        y[post] = 1 - y[post]
    elif condition == "X4":
        y[post] = _xor_law(x[post], noise[post], 0.35)
    ids = int(seed) * 1_000_000_000 + 300_000 + int(condition[1:]) * 20_000 + rows
    return Part(x, y, ids)


def monitoring_partition(task: str, seed: int, condition: str) -> Part:
    if task == "v1_synthetic":
        p = v1_monitoring_partition(seed, condition)
        return Part(p.x, p.y, p.ids)
    if task == "xor_synthetic":
        return _xor_monitoring(seed, condition)
    raise ValueError(f"unknown task {task}")


def _atc_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct)
    if (confidence.ndim != 1 or correct.ndim != 1 or len(confidence) == 0
            or len(confidence) != len(correct) or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))
            or not np.isin(correct, (0, 1)).all()):
        raise ValueError("ATC calibration requires aligned finite confidence and binary correctness vectors")
    scores = np.sort(confidence)
    candidates = np.unique(np.r_[0.0, scores, 1.0, np.nextafter(1.0, np.inf)])
    accepted_accuracy = (len(scores) - np.searchsorted(scores, candidates, side="left")) / len(scores)
    errors = np.abs(accepted_accuracy - np.mean(correct))
    return float(candidates[np.flatnonzero(errors == errors.min())[-1]])


def fit_models(seed: int, train: Part, calibration: Part, reference: Part, *, n_pca_components=3):
    parts = (train, calibration, reference)
    feature_count = train.x.shape[1] if train.x.ndim == 2 else 0
    if feature_count < 1:
        raise ValueError("training features must have at least one column")
    if (not isinstance(n_pca_components, (int, np.integer))
            or isinstance(n_pca_components, (bool, np.bool_)) or n_pca_components < 1):
        raise ValueError("requested PCA components must be a positive integer")
    for name, part in zip(("training", "calibration", "reference"), parts):
        if part.x.ndim != 2 or part.x.shape[1] != feature_count or len(part.x) == 0:
            raise ValueError(f"{name} features must be non-empty and match the training width")
        if part.y.ndim != 1 or part.ids.ndim != 1 or len(part.y) != len(part.x) or len(part.ids) != len(part.x):
            raise ValueError(f"{name} labels and IDs must be one-dimensional and row-aligned")
        if not np.isfinite(part.x).all() or not np.isin(part.y, (0, 1)).all():
            raise ValueError(f"{name} features must be finite and labels binary")
        if len(np.unique(part.ids)) != len(part.ids):
            raise ValueError(f"{name} IDs must be unique")
    if len(np.unique(train.y)) != 2:
        raise ValueError("training partition must contain both classes")
    ids = [set(part.ids.tolist()) for part in parts]
    if any(ids[left] & ids[right] for left in range(3) for right in range(left)):
        raise ValueError("historical partitions overlap")
    from time import perf_counter
    preprocessing_started = perf_counter()
    scaler = StandardScaler().fit(train.x)
    train_x = scaler.transform(train.x)
    actual_components = min(int(n_pca_components), feature_count, len(train_x) - 1)
    if actual_components < 1:
        raise ValueError("training set is too small to fit one PCA component")
    pca = PCA(n_components=actual_components, svd_solver="full").fit(train_x)
    preprocessing_seconds = perf_counter() - preprocessing_started
    model_specs = {
        "logistic": LogisticRegression(C=1.0, max_iter=500, random_state=seed),
        "shallow_tree": DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=6,
            min_samples_leaf=25, random_state=seed),
    }
    artifacts = {}
    for name, model in model_specs.items():
        model_started = perf_counter()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train.y)
        if any(issubclass(item.category, ConvergenceWarning) for item in seen):
            raise RuntimeError(f"{name} convergence warning; paired seed retained as failure")
        cal_p = model.predict_proba(scaler.transform(calibration.x))[:, 1]
        raw_confidence = np.maximum(cal_p, 1.0 - cal_p)
        cal_correct = ((cal_p >= 0.5) == calibration.y).astype(np.int8)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_confidence, cal_correct)
        ref_x = scaler.transform(reference.x)
        ref_p = model.predict_proba(ref_x)[:, 1]
        artifacts[name] = Artifacts(
            scaler=scaler, model=model, pca=pca, calibration=calibrator,
            reference_x=ref_x[-1000:], reference_z=pca.transform(ref_x[-1000:]),
            reference_correct=int(((ref_p >= 0.5) == reference.y).sum()),
            reference_n=len(reference.y), atc=_atc_threshold(raw_confidence, cal_correct),
            fit_seconds=preprocessing_seconds + perf_counter() - model_started,
            fit_ids={"train": train.ids, "calibration": calibration.ids, "reference": reference.ids})
    return artifacts


def score_model(a: Artifacts, batch: FeatureBatch, task: str, model_name: str):
    from time import perf_counter
    started = perf_counter()
    if (batch.x.ndim != 2 or batch.x.shape[1] != a.scaler.n_features_in_ or len(batch.x) == 0
            or batch.ids.ndim != 1 or len(batch.x) != len(batch.ids)):
        raise ValueError("feature cohort must be non-empty, match fitted width, and align with 1D IDs")
    if not np.isfinite(batch.x).all() or len(np.unique(batch.ids)) != len(batch.ids):
        raise ValueError("feature cohort values must be finite and IDs unique")
    if any(np.intersect1d(batch.ids, fit_ids).size for fit_ids in a.fit_ids.values()):
        raise ValueError("monitoring IDs overlap historical fit IDs")
    prediction_started = perf_counter()
    x = a.scaler.transform(batch.x)
    p = a.model.predict_proba(x)[:, 1]
    prediction = (p >= 0.5).astype(np.int8)
    raw = np.maximum(p, 1.0 - p)
    q = a.calibration.predict(raw)
    prediction_seconds = perf_counter() - prediction_started
    z = a.pca.transform(x)
    original_started = perf_counter()
    feature_count = a.scaler.n_features_in_
    original_tests = [ks_2samp(a.reference_x[:, j], x[:, j], method="auto")
                      for j in range(feature_count)]
    original_ks_seconds = perf_counter() - original_started
    latent_started = perf_counter()
    latent_count = a.pca.n_components_
    latent_tests = [ks_2samp(a.reference_z[:, j], z[:, j], method="auto")
                    for j in range(latent_count)]
    latent_ks_seconds = perf_counter() - latent_started
    tests = original_tests + latent_tests
    adjusted = holm([test.pvalue for test in tests])
    original_adjusted = holm([test.pvalue for test in original_tests])
    calibrated_started = perf_counter()
    cal, cal_pmf = impact(q, a.reference_correct, a.reference_n)
    calibrated_impact_seconds = perf_counter() - calibrated_started
    raw_started = perf_counter()
    raw_result, raw_pmf = impact(raw, a.reference_correct, a.reference_n)
    raw_impact_seconds = perf_counter() - raw_started
    record = dict(
        task=task, seed=batch.seed, condition=batch.condition, model=model_name,
        batch=batch.batch, cohort=f"{task}:{batch.seed}:{batch.condition}:{batch.batch}:{model_name}",
        paired_cohort=f"{task}:{batch.seed}:{batch.condition}:{batch.batch}", n=len(x),
        observed_at=batch.batch, prediction_phase="before_label_release",
        reference_correct=a.reference_correct, reference_n=a.reference_n,
        reference_accuracy=a.reference_correct / a.reference_n,
        input_alarm=bool(np.any(adjusted <= 0.05)),
        original_alarm=bool(np.any(original_adjusted <= 0.05)),
        latent_rejection=bool(np.any(adjusted[feature_count:] <= 0.05)),
        n_features=feature_count, n_original_tests=feature_count,
        n_pca_components=latent_count,
        min_adjusted_p=float(adjusted.min()), min_original_adjusted_p=float(original_adjusted.min()),
        atc_estimate=float(np.mean(raw >= a.atc)), atc_threshold=a.atc,
        constant_estimate=a.reference_correct / a.reference_n,
        prediction_seconds=prediction_seconds, original_ks_seconds=original_ks_seconds,
        latent_ks_seconds=latent_ks_seconds, calibrated_impact_seconds=calibrated_impact_seconds,
        raw_impact_seconds=raw_impact_seconds,
    )
    for prefix, result in (("cal", cal), ("raw", raw_result)):
        record.update({f"{prefix}_{key}": value for key, value in result.items()})
    record["calibrated_ac_alone"] = cal["estimate"]
    for j, test in enumerate(original_tests):
        record.update({f"x{j}_ks": float(test.statistic), f"x{j}_p": float(test.pvalue),
                       f"x{j}_adjusted_p": float(adjusted[j]),
                       f"x{j}_flag": bool(adjusted[j] <= 0.05)})
    for j, test in enumerate(latent_tests):
        record[f"pc{j}_ks"] = float(test.statistic)
        record[f"pc{j}_adjusted_p"] = float(adjusted[feature_count + j])
    record["score_seconds"] = perf_counter() - started
    cache = dict(ids=batch.ids, probability=p, prediction=prediction, raw_confidence=raw,
                 correctness_probability=q, pca=z, cal_pmf=cal_pmf, raw_pmf=raw_pmf)
    return record, cache
