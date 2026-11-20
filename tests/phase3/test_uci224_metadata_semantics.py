import numpy as np
import pytest

from scripts.evaluate_uci224_day2 import population_metadata


def test_source_and_evaluation_denominators_are_explicit_for_batch_one():
    source = np.array([1] * 200 + [2] * 100 + [3] * 80 + [4] * 40 + [5] * 20 + [6] * 5)
    evaluation = np.array([1] * 30 + [2] * 20 + [3] * 15 + [4] * 10 + [5] * 8 + [6] * 6)
    meta = population_metadata(source, evaluation)
    assert meta["source_cohort_rows"] == 445
    assert meta["evaluation_rows"] == 89
    assert sum(meta["source_class_counts"].values()) == 445
    assert sum(meta["evaluation_class_counts"].values()) == 89
    assert sum(meta["source_class_proportions"].values()) == pytest.approx(1.0)
    assert sum(meta["evaluation_class_proportions"].values()) == pytest.approx(1.0)
    assert meta["evaluation_class_proportions"]["1"] == pytest.approx(30 / 89)


def test_matching_populations_have_matching_denominators():
    labels = np.array([1, 1, 2, 3, 3, 6])
    meta = population_metadata(labels, labels)
    assert meta["source_cohort_rows"] == meta["evaluation_rows"] == 6
    assert meta["source_class_counts"] == meta["evaluation_class_counts"]
    assert meta["source_class_proportions"] == meta["evaluation_class_proportions"]
