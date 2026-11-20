import numpy as np
import pytest

from drift_monitoring.phase3.core import FeatureBatch, fit_artifacts, harm_kmax, holm, historical_partition, score_batch
from drift_monitoring.phase3.adaptation import CumulativeLogisticAdapter, Part, ReleaseEvent, label_budget_before
from drift_monitoring.phase3.replay import LabelBroker, as_of, replay
from scripts.evaluate_uci224_day2 import historical_split, parse_batch


def test_external_historical_split_matches_locked_partition_and_is_reproducible():
    labels = np.repeat(np.arange(1, 7), [90, 98, 83, 30, 70, 74])
    first = historical_split(labels)
    second = historical_split(labels)
    assert [len(part) for part in first] == [267, 89, 89]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert set(np.concatenate(first)) == set(range(len(labels)))
    for part in first:
        assert set(labels[part]) == set(range(1, 7))


@pytest.mark.parametrize("contents", [
    "1 1:1.0 1:2.0",
    "1 129:0.0",
    "7 1:0.0",
    "1 1:nan",
    "1 malformed",
])
def test_external_parser_rejects_malformed_or_invalid_rows(contents):
    with pytest.raises(ValueError):
        parse_batch(contents, 1)


def test_external_parser_accepts_sparse_zero_features_and_checks_shape():
    x, y = parse_batch("1 1:2.5 128:-1.0\n2 2:3.0", 2)
    assert x.shape == (2, 128)
    np.testing.assert_array_equal(y, [1, 2])
    assert x[0, 1] == 0 and x[1, 0] == 0


@pytest.mark.parametrize("pvalues", [[], [np.nan], [-0.1], [1.1], [[0.1, 0.2]]])
def test_holm_rejects_invalid_pvalue_vectors(pvalues):
    with pytest.raises(ValueError):
        holm(pvalues)


def test_harm_boundary_rejects_invalid_counts():
    for args in [(0, 0, 0), (10, -1, 10), (10, 11, 10), (10, 1, 0)]:
        with pytest.raises(ValueError):
            harm_kmax(*args)


@pytest.mark.parametrize("batch,delay", [(1.5, 0), (1, 5.0), (True, 0)])
def test_label_budget_rejects_non_integer_clock_values(batch, delay):
    with pytest.raises(ValueError):
        label_budget_before(batch, delay)


def test_adaptation_api_rejects_bad_feature_width_and_duplicate_release_ids():
    rng = np.random.default_rng(809)
    train_x = rng.normal(size=(80, 2))
    train_y = (train_x[:, 0] > 0).astype(np.int8)
    train = Part(train_x, train_y, np.arange(80))
    adapter = CumulativeLogisticAdapter(7, train)
    with pytest.raises(ValueError, match="feature matrix"):
        adapter.predict(np.ones((4, 3)))
    bad_release = ReleaseEvent(1, 1, np.array([100, 100]), np.ones((2, 2)), np.array([0, 1]))
    with pytest.raises(ValueError, match="malformed"):
        adapter.add_released([bad_release])


@pytest.mark.parametrize("batch_size,delay", [(0, 0), (-1, 0), (2, -1), (True, 0)])
def test_label_broker_rejects_invalid_clock_configuration(batch_size, delay):
    with pytest.raises(ValueError):
        LabelBroker(np.arange(4), np.array([0, 1, 0, 1]), batch_size, delay)


def test_label_broker_rejects_empty_or_multidimensional_input():
    with pytest.raises(ValueError):
        LabelBroker([], [], 1, 0)
    with pytest.raises(ValueError):
        LabelBroker(np.zeros((2, 2)), np.zeros((2, 2)), 1, 0)


def test_replay_rejects_incomplete_records_and_malformed_prediction_cache():
    broker = LabelBroker(np.arange(4), np.array([0, 1, 0, 1]), 2, 0)
    with pytest.raises(ValueError, match="cover every broker batch"):
        replay([], [], broker)

    records = [
        {"batch": 1, "n": 2, "cohort": "c1", "seed": 1, "condition": "C0",
         "reference_correct": 1, "reference_n": 2},
        {"batch": 2, "n": 2, "cohort": "c2", "seed": 1, "condition": "C0",
         "reference_correct": 1, "reference_n": 2},
    ]
    cache = [{"ids": np.arange(2), "prediction": np.array([0, 1])}]
    with pytest.raises(ValueError, match="cover every broker batch"):
        replay(records, cache, broker)


@pytest.mark.parametrize("clock", [-1, 1.5, True])
def test_release_and_as_of_reject_invalid_clocks(clock):
    broker = LabelBroker(np.arange(2), np.array([0, 1]), 2, 0)
    with pytest.raises(ValueError):
        broker.release(clock)
    with pytest.raises(ValueError):
        as_of([], [], clock)


@pytest.mark.parametrize("duplicate", [True, False])
def test_monitor_score_rejects_malformed_or_training_reused_instances(duplicate):
    train = historical_partition(190991, 0, 4000)
    calibration = historical_partition(190991, 1, 2000)
    reference = historical_partition(190991, 2, 2000)
    fitted = fit_artifacts(190991, train, calibration, reference)
    ids = np.arange(500, dtype=np.int64) + 99_000_000
    if duplicate:
        ids[1] = ids[0]
    batch = FeatureBatch(np.zeros((500, 8)), ids, 190991, "C0", 1)
    if duplicate:
        with pytest.raises(ValueError, match="IDs unique"):
            score_batch(fitted, batch)
    else:
        overlap_ids = train.ids[:500].copy()
        with pytest.raises(ValueError, match="overlap historical fit IDs"):
            score_batch(fitted, FeatureBatch(np.zeros((500, 8)), overlap_ids, 190991, "C0", 1))
        with pytest.raises(ValueError, match="non-empty"):
            score_batch(fitted, FeatureBatch(np.empty((0, 8)), np.array([], dtype=np.int64), 190991, "C0", 1))
