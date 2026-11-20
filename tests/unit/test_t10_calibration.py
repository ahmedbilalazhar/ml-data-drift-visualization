"""T10/T11: calibration quantile + matching determinism checks."""
from drift_monitoring.detectors.drift_detectors import calibrate_threshold
from drift_monitoring.evaluation.metrics import match_events


def test_calibration_quantile_direction():
    thr_hi = calibrate_threshold([0.1 * i for i in range(100)], 10.0)
    thr_lo = calibrate_threshold([0.1 * i for i in range(100)], 100.0)
    assert thr_hi > thr_lo  # fewer allowed alerts -> higher threshold


def test_match_events_deterministic_and_bounded():
    a = match_events([4, 5], 4)
    b = match_events([4, 5], 4)
    assert a == b
    assert a["detected"] is True and a["delay_windows"] == 0
