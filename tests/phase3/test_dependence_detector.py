import numpy as np
import pytest

from drift_monitoring.phase3.dependence import (
    calibrate_reference_threshold,
    calibrate_threshold,
    max_correlation_change,
)


def test_detects_paired_correlation_change_without_marginal_variance_change():
    rng = np.random.default_rng(44)
    z = rng.normal(size=(20000, 2))
    reference = z.copy()
    changed = z.copy()
    changed[:, 1] = .9*z[:, 0] + np.sqrt(1-.9**2)*z[:, 1]
    score, pair = max_correlation_change(reference, changed)
    assert pair == (0, 1)
    assert score > .5
    assert np.allclose(reference.var(axis=0), changed.var(axis=0), atol=.03)


def test_stationary_score_is_small_and_threshold_validation_is_explicit():
    rng = np.random.default_rng(45)
    reference = rng.normal(size=(1000, 8))
    current = rng.normal(size=(500, 8))
    score, _ = max_correlation_change(reference, current)
    threshold = calibrate_threshold(np.linspace(.01, .2, 100))
    assert score < threshold
    with pytest.raises(ValueError):
        max_correlation_change(np.ones((10, 2)), np.ones((10, 2)))
    with pytest.raises(ValueError):
        calibrate_threshold(np.arange(10.0))


def test_historical_only_threshold_flags_preserved_marginal_dependence_change():
    rng = np.random.default_rng(771)
    z = rng.normal(size=(6000, 8))
    reference = z[:1000]
    calibration = z[1000:3000]
    threshold = calibrate_reference_threshold(reference, calibration, batch_size=500,
                                              random_state=772, n_null=100)
    current = z[3000:3500].copy()
    current[:, 1] = .9 * current[:, 0] + np.sqrt(1 - .9**2) * current[:, 1]
    score, pair = max_correlation_change(reference, current)
    assert pair == (0, 1)
    assert score > threshold
    np.testing.assert_allclose(current[:, :2].std(axis=0), z[3000:3500, :2].std(axis=0), atol=.06)
