"""Explicit two-phase clock and scheduled-label boundary, with queue drainage."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.metrics import f1_score
from .core import is_harm


@dataclass(frozen=True)
class ReleasedLabels:
    batch: int
    available_at: int
    ids: np.ndarray
    labels: np.ndarray


class LabelBroker:
    """Stores evaluator truth; exposes it only at/after scheduled release.

    This is an API boundary, not a security sandbox. The monitor never receives
    this object. IDs must be exact and ordered before a release is accepted.
    """
    def __init__(self, ids, labels, batch_size, delay):
        ids = np.asarray(ids)
        labels = np.asarray(labels)
        if (ids.ndim != 1 or labels.ndim != 1 or len(ids) == 0 or len(ids) != len(labels)
                or not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, (bool, np.bool_))
                or batch_size <= 0 or len(ids) % batch_size
                or not isinstance(delay, (int, np.integer)) or isinstance(delay, (bool, np.bool_))
                or delay < 0):
            raise ValueError('invalid release schedule')
        if not np.isin(labels, (0, 1)).all():
            raise ValueError('released labels must be binary')
        if len(np.unique(ids)) != len(ids):
            raise ValueError('duplicate IDs')
        self.__ids = ids.copy()
        self.__labels = labels.copy()
        self.batch_size = batch_size
        self.delay = delay
        self.batches = len(ids)//batch_size
        self.__released = set()

    def release(self, clock):
        if not isinstance(clock, (int, np.integer)) or isinstance(clock, (bool, np.bool_)) or clock < 0:
            raise ValueError('release clock must be a non-negative integer')
        released = []
        for b in range(1, self.batches+1):
            if b+self.delay <= clock and b not in self.__released:
                sl = slice((b-1)*self.batch_size, b*self.batch_size)
                released.append(ReleasedLabels(b, b+self.delay,
                                               self.__ids[sl].copy(), self.__labels[sl].copy()))
                self.__released.add(b)
        return released



@dataclass(frozen=True)
class ReleasedFeedback:
    source_batch: int
    available_at: int
    ids: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def release_feedback(broker: LabelBroker, features, ids, clock):
    """Release due labels and their matching source features for an orchestrator.

    Call with ``clock=t-1`` immediately before prediction at forecast clock t.
    This enforces the Phase 3 rule that even delay-zero feedback follows that
    cohort's prediction. Queue drainage can call it after the last prediction.
    """
    x = np.asarray(features)
    row_ids = np.asarray(ids)
    if x.ndim != 2 or row_ids.ndim != 1 or len(x) != len(row_ids):
        raise ValueError("feature source must be a row-aligned matrix and ID vector")
    if len(row_ids) != broker.batch_size * broker.batches:
        raise ValueError("feature source length differs from broker schedule")
    released = []
    for batch in broker.release(clock):
        lo = (batch.batch - 1) * broker.batch_size
        hi = lo + broker.batch_size
        if not np.array_equal(batch.ids, row_ids[lo:hi]):
            raise ValueError(f"released-label IDs do not align with source batch {batch.batch}")
        fx = np.array(x[lo:hi], copy=True)
        ids_copy = batch.ids.copy()
        labels_copy = batch.labels.copy()
        for arr in (fx, ids_copy, labels_copy):
            arr.setflags(write=False)
        released.append(ReleasedFeedback(batch.batch, batch.available_at,
                                         ids_copy, fx, labels_copy))
    return released

def replay(records, prediction_cache, broker):
    """Input rows precede release rows even when d=0; no fake input on drain."""
    if len(records) != broker.batches or len(prediction_cache) != broker.batches:
        raise ValueError('records and prediction cache must cover every broker batch exactly once')
    batches = [r.get('batch') for r in records]
    if len(set(batches)) != len(batches) or set(batches) != set(range(1, broker.batches + 1)):
        raise ValueError('record batches must be unique and cover the broker schedule')
    if any(r.get('n') != broker.batch_size for r in records):
        raise ValueError('record cohort size differs from broker batch size')
    events, verification = [], []
    by_batch = {r['batch']: r for r in records}
    for clock in range(1, broker.batches+broker.delay+1):
        if clock <= broker.batches:
            r = by_batch[clock]
            events.append(dict(cohort=r['cohort'], seed=r['seed'], condition=r['condition'],
                               delay=broker.delay, clock=clock, phase=0,
                               event='prediction_persisted', batch=clock))
        for label in broker.release(clock):
            r = by_batch[label.batch]
            cache = prediction_cache[label.batch-1]
            ids = np.asarray(cache.get('ids'))
            prediction = np.asarray(cache.get('prediction'))
            if ids.ndim != 1 or prediction.ndim != 1 or len(prediction) != len(label.ids):
                raise ValueError('prediction cache arrays must be one-dimensional and match batch size')
            if not np.array_equal(label.ids, ids):
                raise ValueError('prediction/label ID mismatch')
            if not np.isin(prediction, (0, 1)).all():
                raise ValueError('cached predictions must be binary')
            correct = int((prediction == label.labels).sum())
            row = dict(seed=r['seed'], condition=r['condition'], cohort=r['cohort'],
                       batch=label.batch, delay=broker.delay, available_at=label.available_at,
                       evidence_age_batches=clock-label.batch, n=len(label.ids), correct=correct,
                       accuracy=correct/len(label.ids),
                       harm=is_harm(correct, len(label.ids), r['reference_correct'], r['reference_n']),
                       macro_f1=float(f1_score(label.labels, prediction, average='macro',
                                              labels=[0, 1], zero_division=0)),
                       class0_n=int((label.labels == 0).sum()), class1_n=int((label.labels == 1).sum()),
                       ids_sha256=hashlib.sha256(label.ids.tobytes()).hexdigest())
            verification.append(row)
            events.append(dict(cohort=r['cohort'], seed=r['seed'], condition=r['condition'],
                               delay=broker.delay, clock=clock, phase=1,
                               event='labels_verified', batch=label.batch))
    if len(verification) != broker.batches:
        raise RuntimeError('release queue not fully drained')
    return events, verification


def as_of(records, verification, clock):
    """UI-ready join; unknown truth is absent, never zero-filled or backdated."""
    if not isinstance(clock, (int, np.integer)) or isinstance(clock, (bool, np.bool_)) or clock < 0:
        raise ValueError('as-of clock must be a non-negative integer')
    cohorts = [row.get('cohort') for row in records]
    if len(set(cohorts)) != len(cohorts):
        raise ValueError('online records must have unique cohort IDs')
    released_cohorts = [row.get('cohort') for row in verification]
    if len(set(released_cohorts)) != len(released_cohorts):
        raise ValueError('verification records must have unique cohort IDs')
    visible = {v['cohort']: v for v in verification if v['available_at'] <= clock}
    rows = []
    for record in records:
        if record['observed_at'] > clock:
            continue
        row = record.copy()
        v = visible.get(row['cohort'])
        row.update(verified_accuracy=None if v is None else v['accuracy'],
                   verified_harm=None if v is None else v['harm'],
                   label_status='unresolved' if v is None else 'verified',
                   evidence_age_batches=None if v is None else clock-v['batch'])
        rows.append(row)
    return rows
