import numpy as np
import pytest

from scripts.evaluate_uci224_exact_interval_extension import interval_exact, interval_normal


def test_exact_interval_uses_discrete_prediction_count_quantiles():
    lo, hi = interval_exact(np.full(4, 0.5))
    assert lo == 0.0
    assert hi == 1.0


def test_exact_and_normal_intervals_are_bounded_for_extreme_probabilities():
    q = np.r_[np.zeros(20), np.ones(30)]
    for interval in (interval_exact(q), interval_normal(q)):
        assert 0 <= interval[0] <= interval[1] <= 1


def test_exact_interval_rejects_empty_input():
    with pytest.raises(ValueError):
        interval_exact(np.array([]))
