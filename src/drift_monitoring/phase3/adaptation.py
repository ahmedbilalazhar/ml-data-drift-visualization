"""Small outcome-informed delayed-label adaptation study primitives."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .v2 import Part, historical_partition, monitoring_partition


TASK_CONDITION = (("v1_synthetic", "C3"), ("xor_synthetic", "X3"))
DEVELOPMENT_SEEDS = tuple(range(290001, 290006))
FINAL_SEEDS = tuple(range(300, 330))
DELAYS = (0, 5, 20)
BATCH_SIZE = 500
BATCHES = 24


def label_budget_before(batch: int, delay: int) -> int:
    """Number of monitoring labels available before scoring this batch.

    A label released at the current clock arrives after that clock's prediction.
    """
    if (not isinstance(batch, (int, np.integer)) or isinstance(batch, (bool, np.bool_))
            or not isinstance(delay, (int, np.integer)) or isinstance(delay, (bool, np.bool_))
            or not 1 <= batch <= BATCHES or delay not in DELAYS):
        raise ValueError("invalid batch or label delay")
    return BATCH_SIZE * min(BATCHES, max(0, int(batch) - int(delay) - 1))


def digest_ids(ids: Iterable[int]) -> str:
    values = np.asarray(list(ids), dtype="<i8")
    return sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True)
class ReleaseEvent:
    source_batch: int
    available_at: int
    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray


class LabelReleaseBroker:
    """Release labels only at their scheduled clock, after prediction persistence."""

    def __init__(self, part: Part, delay: int):
        if (not isinstance(delay, (int, np.integer)) or isinstance(delay, (bool, np.bool_))
                or delay not in DELAYS or part.x.ndim != 2 or part.y.ndim != 1 or part.ids.ndim != 1
                or len(part.y) != BATCH_SIZE * BATCHES or len(part.x) != len(part.y)
                or len(part.ids) != len(part.y) or not np.isfinite(part.x).all()
                or not np.isin(part.y, (0, 1)).all() or len(np.unique(part.ids)) != len(part.ids)):
            raise ValueError("invalid delay or monitoring partition length")
        self._part = part
        self.delay = int(delay)
        self._released: set[int] = set()

    def release(self, clock: int) -> list[ReleaseEvent]:
        events = []
        for batch in range(1, BATCHES + 1):
            if batch in self._released or batch + self.delay != clock:
                continue
            start = (batch - 1) * BATCH_SIZE
            stop = start + BATCH_SIZE
            self._released.add(batch)
            events.append(ReleaseEvent(
                source_batch=batch, available_at=int(clock),
                ids=self._part.ids[start:stop].copy(),
                x=self._part.x[start:stop].copy(), y=self._part.y[start:stop].copy()))
        return events

    @property
    def released_batches(self) -> set[int]:
        return set(self._released)


class CumulativeLogisticAdapter:
    """Refit a fixed logistic model on initial training plus due labels only."""

    def __init__(self, seed: int, train: Part):
        if (train.x.ndim != 2 or train.y.ndim != 1 or train.ids.ndim != 1
                or len(train.x) == 0 or len(train.x) != len(train.y) or len(train.x) != len(train.ids)
                or not np.isfinite(train.x).all() or not np.isin(train.y, (0, 1)).all()
                or len(np.unique(train.ids)) != len(train.ids) or len(np.unique(train.y)) != 2):
            raise ValueError("training partition must be finite, aligned, unique, and contain both classes")
        self.seed = int(seed)
        self.train = train
        self.scaler = StandardScaler().fit(train.x)
        self.train_x = self.scaler.transform(train.x)
        self.released_x: list[np.ndarray] = []
        self.released_y: list[np.ndarray] = []
        self.released_ids: list[np.ndarray] = []
        self.released_times: list[np.ndarray] = []
        self.model = self._new_model()
        self.model.fit(self.train_x, train.y)
        self.fit_count = 1

    def _new_model(self):
        return LogisticRegression(C=1.0, max_iter=500, random_state=self.seed)

    def add_released(self, events: Iterable[ReleaseEvent]) -> None:
        due = list(events)
        if not due:
            return
        known = set(np.concatenate(self.released_ids).tolist()) if self.released_ids else set()
        for event in due:
            if (event.x.ndim != 2 or event.x.shape[1] != self.train.x.shape[1]
                    or event.y.ndim != 1 or event.ids.ndim != 1
                    or len(event.ids) == 0 or len(event.x) != len(event.ids) or len(event.y) != len(event.ids)
                    or not np.isfinite(event.x).all() or not np.isin(event.y, (0, 1)).all()
                    or len(np.unique(event.ids)) != len(event.ids)
                    or not isinstance(event.available_at, (int, np.integer)) or event.available_at < 1):
                raise ValueError("released label event is malformed")
            if np.intersect1d(event.ids, self.train.ids).size:
                raise ValueError("released monitoring IDs overlap initial training IDs")
            if known.intersection(event.ids.tolist()):
                raise ValueError("duplicate released label IDs")
            known.update(event.ids.tolist())
            self.released_x.append(self.scaler.transform(event.x))
            self.released_y.append(event.y.copy())
            self.released_ids.append(event.ids.copy())
            self.released_times.append(np.full(len(event.ids), event.available_at, dtype=np.int16))
        x = np.concatenate([self.train_x, *self.released_x], axis=0)
        y = np.concatenate([self.train.y, *self.released_y])
        updated = self._new_model()
        updated.fit(x, y)
        self.model = updated
        self.fit_count += 1

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (features.ndim != 2 or features.shape[1] != self.train.x.shape[1]
                or len(features) == 0 or not np.isfinite(features).all()):
            raise ValueError("invalid feature matrix")
        p = self.model.predict_proba(self.scaler.transform(features))[:, 1]
        return (p >= 0.5).astype(np.int8), p

    def audit(self, forecast_batch: int) -> dict:
        if (not isinstance(forecast_batch, (int, np.integer))
                or isinstance(forecast_batch, (bool, np.bool_)) or forecast_batch < 1):
            raise ValueError("forecast batch must be a positive one-based integer")
        if not self.released_ids:
            ids = np.empty(0, dtype=np.int64)
            times = np.empty(0, dtype=np.int16)
        else:
            ids = np.concatenate(self.released_ids)
            times = np.concatenate(self.released_times)
        if len(times) and np.any(times >= forecast_batch):
            raise ValueError("same-clock or future label entered a forecast fit")
        return dict(
            training_n=len(self.train.y) + len(ids),
            released_monitoring_n=len(ids),
            monitor_id_sha256=digest_ids(ids),
            max_available_at=int(times.max()) if len(times) else None,
            fit_count=self.fit_count,
        )


def make_stream(seed: int, task: str, condition: str):
    if (task, condition) not in TASK_CONDITION:
        raise ValueError("unlocked task-condition pair")
    train = historical_partition(task, seed, 0, 4000)
    calibration = historical_partition(task, seed, 1, 2000)
    reference = historical_partition(task, seed, 2, 2000)
    monitor = monitoring_partition(task, seed, condition)
    historical_sets = [set(p.ids.tolist()) for p in (train, calibration, reference)]
    if any(historical_sets[i] & historical_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("historical partitions overlap")
    if any(np.intersect1d(part.ids, monitor.ids).size for part in (train, calibration, reference)):
        raise ValueError("monitoring IDs overlap historical partitions")
    return train, calibration, reference, monitor
