import numpy as np

from scripts.run_uci224_delayed_replay import asof_recalibration


def fixture():
    cal_conf = np.array([0.2, 0.35, 0.55, 0.7, 0.82, 0.92])
    cal_correct = np.array([0, 0, 1, 0, 1, 1])
    confidence = {
        2: np.array([0.3, 0.4, 0.7, 0.8]),
        3: np.array([0.25, 0.5, 0.75, 0.9]),
        4: np.array([0.2, 0.45, 0.65, 0.85]),
    }
    predictions = {b: np.array([1, 1, 0, 0]) for b in confidence}
    truth = {
        2: np.array([1, 0, 0, 0]),
        3: np.array([0, 0, 0, 1]),
        4: np.array([1, 0, 1, 0]),
    }
    return cal_conf, cal_correct, confidence, predictions, truth


def test_release_clock_is_strict_and_delay_is_cohort_based():
    c, y, conf, pred, truth = fixture()
    _, audit0 = asof_recalibration(c, y, conf, pred, truth, 0)
    _, audit1 = asof_recalibration(c, y, conf, pred, truth, 1)
    a0 = audit0.set_index("batch")
    a1 = audit1.set_index("batch")
    assert a0.loc[2, "available_source_batches"] == ""
    assert a0.loc[3, "available_source_batches"] == "2"
    assert a1.loc[3, "available_source_batches"] == ""
    assert a1.loc[4, "available_source_batches"] == "2"
    assert (audit0.max_release_clock.dropna() < audit0.loc[audit0.max_release_clock.notna(), "batch"]).all()


def test_unreleased_future_outcome_cannot_change_earlier_forecasts():
    c, y, conf, pred, truth = fixture()
    base, _ = asof_recalibration(c, y, conf, pred, truth, 0)
    changed = dict(truth)
    changed[4] = 1 - changed[4]
    altered, _ = asof_recalibration(c, y, conf, pred, changed, 0)
    for batch in (2, 3, 4):
        np.testing.assert_array_equal(base[batch], altered[batch])


def test_delay_zero_does_not_release_same_batch_labels_before_forecast():
    c, y, conf, pred, truth = fixture()
    base, audit = asof_recalibration(c, y, conf, pred, truth, 0)
    changed = dict(truth)
    changed[3] = 1 - changed[3]
    altered, _ = asof_recalibration(c, y, conf, pred, changed, 0)
    np.testing.assert_array_equal(base[3], altered[3])
    assert audit.set_index("batch").loc[3, "available_source_batches"] == "2"
