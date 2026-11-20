import pytest

from drift_monitoring.phase3.released_error import ReleasedErrorADWINStyle


def test_released_error_detector_does_not_alarm_on_fixed_error_rate():
    detector = ReleasedErrorADWINStyle(delta=.002, min_window_batches=2)
    outputs = [detector.update(100, 500) for _ in range(96)]
    assert not any(result["alarm"] for result in outputs)


def test_released_error_detector_finds_large_shift_from_released_cohorts():
    detector = ReleasedErrorADWINStyle(delta=.002, min_window_batches=2)
    outputs = [detector.update(50, 500) for _ in range(24)]
    outputs += [detector.update(300, 500) for _ in range(40)]
    alarms = [index + 1 for index, result in enumerate(outputs) if result["alarm"]]
    assert alarms
    assert 25 <= alarms[0] <= 32
    assert all(result["updates"] == index + 1 for index, result in enumerate(outputs))


@pytest.mark.parametrize("errors,n", [(-1, 500), (501, 500), (1.5, 500), (1, 0), (True, 500)])
def test_released_error_detector_rejects_invalid_counts(errors, n):
    detector = ReleasedErrorADWINStyle()
    with pytest.raises(ValueError):
        detector.update(errors, n)

