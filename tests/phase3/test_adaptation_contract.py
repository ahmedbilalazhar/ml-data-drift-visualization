import inspect

import numpy as np

from drift_monitoring.phase3.adaptation import (
    DELAYS,
    DEVELOPMENT_SEEDS,
    FINAL_SEEDS,
    CumulativeLogisticAdapter,
    LabelReleaseBroker,
    label_budget_before,
)
from drift_monitoring.phase3.v2 import Part


def test_locked_adaptation_seeds_are_fresh_and_disjoint():
    used = set(range(100, 130)) | set(range(200, 230)) | set(range(190001, 190006))
    assert not (used & set(DEVELOPMENT_SEEDS))
    assert not (used & set(FINAL_SEEDS))
    assert DEVELOPMENT_SEEDS == tuple(range(290001, 290006))
    assert FINAL_SEEDS == tuple(range(300, 330))


def test_exact_pre_prediction_label_budgets_for_all_delays():
    assert {delay: label_budget_before(1, delay) for delay in DELAYS} == {0: 0, 5: 0, 20: 0}
    assert {delay: label_budget_before(24, delay) for delay in DELAYS} == {
        0: 11500, 5: 9000, 20: 1500,
    }
    assert label_budget_before(7, 5) == 500
    assert label_budget_before(22, 20) == 500


def test_broker_releases_only_due_batch_and_same_clock_after_prediction():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(12000, 2))
    y = (x[:, 0] > 0).astype(np.int8)
    ids = np.arange(12000, dtype=np.int64) + 1_000_000
    part = Part(x, y, ids)
    broker = LabelReleaseBroker(part, delay=5)
    assert broker.release(5) == []
    event = broker.release(6)[0]
    assert event.source_batch == 1
    assert event.available_at == 6
    np.testing.assert_array_equal(event.ids, ids[:500])
    np.testing.assert_array_equal(event.y, y[:500])
    assert broker.released_batches == {1}


def test_adapter_fit_uses_only_labels_released_before_forecast():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(1000, 2))
    y = (x[:, 0] - 0.4 * x[:, 1] > 0).astype(np.int8)
    train = Part(x, y, np.arange(1000, dtype=np.int64))
    monitor_x = rng.normal(size=(12000, 2))
    monitor_y = (monitor_x[:, 0] < 0).astype(np.int8)
    monitor_ids = np.arange(12000, dtype=np.int64) + 10_000
    monitor = Part(monitor_x, monitor_y, monitor_ids)
    adapter = CumulativeLogisticAdapter(5, train)
    broker = LabelReleaseBroker(monitor, delay=0)

    assert list(inspect.signature(adapter.predict).parameters) == ["features"]
    batch1, _ = adapter.predict(monitor_x[:500])
    assert len(batch1) == 500
    assert adapter.audit(1)["released_monitoring_n"] == 0
    event = broker.release(1)
    assert len(event) == 1
    adapter.add_released(event)
    audit = adapter.audit(2)
    assert audit["released_monitoring_n"] == 500
    assert audit["training_n"] == 1500
    assert audit["max_available_at"] == 1 < 2
    assert audit["fit_count"] == 2
    batch2, _ = adapter.predict(monitor_x[500:1000])
    assert len(batch2) == 500
