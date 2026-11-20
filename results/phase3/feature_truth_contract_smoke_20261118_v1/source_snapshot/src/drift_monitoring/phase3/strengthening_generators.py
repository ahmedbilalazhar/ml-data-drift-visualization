"""Paired 96-batch generators for the Phase 3 strengthening protocol."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .core import Partition


TASKS = ("linear_probabilistic", "xor_nonlinear", "five_region")
MECHANISMS = (
    "M0_stationary", "M1_abrupt_nuisance", "M2_gradual_predictive",
    "M3_incremental_repeated_small", "M4_abrupt_conditional_reversal",
    "M5_recurrent_conditional_reversal", "M6_dependence_marginals_preserved",
    "M7_out_of_support",
)
SEASONAL = "S1_seasonal_nuisance"
TASK_CODE = {name: i + 1 for i, name in enumerate(TASKS)}
MECHANISM_CODE = {name: i + 1 for i, name in enumerate((*MECHANISMS, SEASONAL))}
REGION_CENTERS = np.array([[-1.8, -1.0], [-1.8, 1.0], [0.0, 0.0],
                           [1.8, -1.0], [1.8, 1.0]], dtype=float)
REGION_SLOPES = np.array([[1.0, 0.7], [1.0, -0.7], [-1.0, 0.7],
                          [-1.0, -0.7], [0.2, 1.0]], dtype=float)


def _readonly_vector(value, *, dtype=None):
    out = np.array(value, dtype=dtype, copy=True)
    if out.ndim != 1:
        raise ValueError("stream boundary arrays must be one-dimensional")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class MonitorFeatures:
    """Live feature source: IDs and features only, never evaluator truth."""
    x: np.ndarray
    ids: np.ndarray

    def __post_init__(self):
        x = np.array(self.x, dtype=float, copy=True)
        ids = _readonly_vector(self.ids)
        if (x.ndim != 2 or len(x) != len(ids) or not np.isfinite(x).all()
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("monitor features require finite rows and unique aligned IDs")
        x.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "ids", ids)


@dataclass(frozen=True)
class EvaluatorArtifacts:
    """Retrospective truth isolated from the live feature-source interface."""
    ids: np.ndarray
    labels: np.ndarray
    region_id: np.ndarray
    expected_label_probability: np.ndarray

    def __post_init__(self):
        ids = _readonly_vector(self.ids)
        labels = _readonly_vector(self.labels, dtype=np.int8)
        region = _readonly_vector(self.region_id, dtype=np.int8)
        probability = _readonly_vector(self.expected_label_probability, dtype=float)
        if not (len(ids) == len(labels) == len(region) == len(probability)):
            raise ValueError("evaluator arrays must align")
        if (len(np.unique(ids)) != len(ids) or not np.isin(labels, (0, 1)).all()
                or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
            raise ValueError("invalid evaluator truth arrays")
        for name, value in (("ids", ids), ("labels", labels), ("region_id", region),
                            ("expected_label_probability", probability)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GeneratedStream:
    monitor: MonitorFeatures
    evaluator: EvaluatorArtifacts
    truth_metadata: dict

    def __post_init__(self):
        if not np.array_equal(self.monitor.ids, self.evaluator.ids):
            raise ValueError("monitor and evaluator row IDs must align exactly")


def _rng(seed: int, task: str, stage: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), TASK_CODE[task], int(stage), 73116]))


def _base_features(z: np.ndarray, task: str, region_id: np.ndarray) -> np.ndarray:
    x = np.array(z, dtype=float, copy=True)
    if task == "five_region":
        centers = REGION_CENTERS[region_id]
        x[:, 0] = centers[:, 0] + z[:, 0]
        x[:, 1] = centers[:, 1] + 0.3 * z[:, 0] + np.sqrt(1 - 0.3**2) * z[:, 1]
    else:
        x[:, 1] = 0.3 * z[:, 0] + np.sqrt(1 - 0.3**2) * z[:, 1]
    return x


def _label_probability(x: np.ndarray, task: str, region_id: np.ndarray) -> np.ndarray:
    if task == "linear_probabilistic":
        weights = np.zeros(x.shape[1], dtype=float)
        weights[:min(3, x.shape[1])] = np.array([1.2, -0.9, 0.6])[:min(3, x.shape[1])]
        return expit(x @ weights - 0.2)
    if task == "xor_nonlinear":
        xor = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
        return np.where(xor, 0.90, 0.10)
    if task == "five_region":
        local = x[:, :2] - REGION_CENTERS[region_id]
        logits = np.sum(local * REGION_SLOPES[region_id], axis=1) + np.array([-0.25, 0.1, 0.35, -0.1, 0.2])[region_id]
        return expit(logits)
    raise ValueError(f"unknown task family: {task}")


def generate_historical(task: str, seed: int, stage: int, rows: int,
                        feature_count: int = 8) -> Partition:
    """Generate an independent historical partition under the stationary source law."""
    if task not in TASKS or feature_count < 2 or rows < 1:
        raise ValueError("historical generator requires a known task, at least 2 features, and positive rows")
    rng = _rng(seed, task, stage)
    z = rng.normal(size=(rows, feature_count))
    regions = rng.integers(0, 5, size=rows) if task == "five_region" else np.full(rows, -1, dtype=np.int8)
    x = _base_features(z, task, regions)
    probability = _label_probability(x, task, regions)
    y = (rng.random(rows) < probability).astype(np.int8)
    ids = _row_ids(seed, task, 0, stage, rows)
    return Partition(x, y, ids)


def historical_region_ids(task: str, seed: int, stage: int, rows: int,
                          feature_count: int = 8) -> np.ndarray:
    """Recreate historical generating-region IDs for cluster-label matching.

    IDs are derived from the same historical random stream used by
    ``generate_historical``. They are intended only to fit a cluster-to-region
    correspondence on historical training rows; never use monitoring truth to
    choose this mapping.
    """
    if task not in TASKS or rows < 1 or feature_count < 2:
        raise ValueError("historical region matching requires a known task, >=2 features and positive rows")
    rng = _rng(seed, task, stage)
    rng.normal(size=(rows, feature_count))  # consume the feature draw in the generator's fixed order
    if task == "five_region":
        return rng.integers(0, 5, size=rows, dtype=np.int64)
    return np.full(rows, -1, dtype=np.int64)


def _row_ids(seed: int, task: str, mechanism: int, stage: int, rows: int) -> np.ndarray:
    # < 2**63 for the seed ranges locked in the v3 configuration.
    base = int(seed) * 1_000_000_000 + TASK_CODE[task] * 100_000_000 + mechanism * 10_000_000 + stage * 1_000_000
    ids = base + np.arange(rows, dtype=np.int64)
    if ids[-1] >= np.iinfo(np.int64).max:
        raise OverflowError("row identifier exceeds signed 64-bit range")
    return ids


def generate_monitoring(task: str, mechanism: str, seed: int, *, feature_count: int = 8,
                        batch_size: int = 500, batches: int = 96) -> GeneratedStream:
    """Generate paired mechanisms using common feature and label innovations.

    Batches 1--24 share the same stationary prefix across all mechanisms within
    a task/seed. Region membership and all outcome truth are evaluator artifacts.
    """
    allowed = set(MECHANISMS) | {SEASONAL}
    if task not in TASKS or mechanism not in allowed:
        raise ValueError("unknown task or mechanism")
    if feature_count < 2 or batch_size < 1 or batches < 65:
        raise ValueError("the locked mechanism schedule requires >=2 features and >=65 batches")
    n = batch_size * batches
    rng = _rng(seed, task, 3000)
    z = rng.normal(size=(n, feature_count))
    outcome_u = rng.random(n)
    if task == "five_region":
        region = rng.integers(0, 5, size=n, dtype=np.int8)
    else:
        region = np.full(n, -1, dtype=np.int8)
    x = _base_features(z, task, region)
    batch_number = np.arange(n, dtype=np.int64) // batch_size + 1
    intervention = batch_number >= 25
    affected_features: list[int] = []
    affected_pairs: list[list[int]] = []
    affected_regions: list[int] = []
    intervention_windows: list[list[int]] = []
    probabilities = _label_probability(x, task, region)

    if mechanism == "M1_abrupt_nuisance":
        nuisance_feature = feature_count - 1
        x[intervention, nuisance_feature] += 1.5
        affected_features = [nuisance_feature]
        probabilities = _label_probability(x, task, region)
    elif mechanism == "M2_gradual_predictive":
        ramp = np.clip((batch_number - 24) / (40 - 24), 0, 1)
        x[:, 0] += 1.2 * ramp
        affected_features = [0]
        probabilities = _label_probability(x, task, region)
        intervention_windows = [[25, 40]]
    elif mechanism == "M3_incremental_repeated_small":
        change_points = np.array([25, 37, 49, 61, 73, 85])
        increments = (batch_number[:, None] >= change_points[None, :]).sum(axis=1)
        nuisance_feature = feature_count - 1
        x[:, nuisance_feature] += 0.25 * increments
        affected_features = [nuisance_feature]
        intervention_windows = [[int(point), batches] for point in change_points]
    elif mechanism == "M6_dependence_marginals_preserved":
        rho = np.where(intervention, 0.9, 0.3)
        if task == "five_region":
            centers = REGION_CENTERS[region]
            x[:, 0] = centers[:, 0] + z[:, 0]
            x[:, 1] = centers[:, 1] + rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        else:
            x[:, 0] = z[:, 0]
            x[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        probabilities = _label_probability(x, task, region)
        affected_features = []
        affected_pairs = [[0, 1]]
        intervention_windows = [[25, batches]]
    elif mechanism == "M7_out_of_support":
        x[intervention, 0] += 2.5
        x[intervention, 1] += 2.5
        affected_features = [0, 1]
        probabilities = _label_probability(x, task, region)
        intervention_windows = [[25, batches]]
    elif mechanism == SEASONAL:
        seasonal = 0.6 * np.sin(2 * np.pi * (batch_number - 1) / 24.0)
        seasonal_feature = feature_count - 1
        x[:, seasonal_feature] += seasonal
        affected_features = [seasonal_feature]
        probabilities = _label_probability(x, task, region)
        intervention_windows = [[1, batches]]

    y = (outcome_u < probabilities).astype(np.int8)
    if mechanism in {"M4_abrupt_conditional_reversal", "M5_recurrent_conditional_reversal"}:
        if mechanism == "M4_abrupt_conditional_reversal":
            active = intervention.copy()
            intervention_windows = [[25, batches]]
        else:
            active = ((batch_number >= 25) & (batch_number < 41)) | ((batch_number >= 65) & (batch_number < 81))
            intervention_windows = [[25, 40], [65, 80]]
        if task == "five_region":
            target_region = 2
            active &= region == target_region
            affected_regions = [target_region]
        else:
            affected_regions = []
        y[active] = 1 - y[active]
        probabilities[active] = 1.0 - probabilities[active]
        affected_features = list(range(min(3 if task == "linear_probabilistic" else 2, feature_count)))

    ids = _row_ids(seed, task, MECHANISM_CODE[mechanism], 3000, n)
    truth = {
        "task": task, "mechanism": mechanism, "seed": int(seed),
        "batch_size": int(batch_size), "batches": int(batches),
        "stationary_prefix_batches": [1, 24], "intervention_windows_batches": intervention_windows,
        "changed_marginal_features": affected_features,
        "changed_dependence_pairs": affected_pairs,
        "changed_generating_regions": affected_regions,
        "conditional_law_changed": mechanism in {"M4_abrupt_conditional_reversal", "M5_recurrent_conditional_reversal"},
        "region_membership_evaluator_only": task == "five_region",
        "paired_innovations_id": f"{task}:{seed}:monitoring-v1",
        "no_intervention_control": mechanism == "M0_stationary",
    }
    monitor = MonitorFeatures(x=x, ids=ids)
    evaluator = EvaluatorArtifacts(ids=ids, labels=y, region_id=region,
                                   expected_label_probability=probabilities)
    return GeneratedStream(monitor=monitor, evaluator=evaluator, truth_metadata=truth)
