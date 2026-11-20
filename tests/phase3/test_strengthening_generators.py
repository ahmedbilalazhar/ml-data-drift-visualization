import numpy as np
import pytest

from drift_monitoring.phase3.strengthening_generators import (
    MECHANISMS,
    TASKS,
    generate_historical,
    generate_monitoring,
)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_all_primary_mechanisms_share_the_same_stationary_prefix(task, mechanism):
    control = generate_monitoring(task, "M0_stationary", 73116001)
    candidate = generate_monitoring(task, mechanism, 73116001)
    n = 24 * 500
    np.testing.assert_array_equal(candidate.monitor.x[:n], control.monitor.x[:n])
    np.testing.assert_array_equal(candidate.evaluator.labels[:n], control.evaluator.labels[:n])
    assert len(np.unique(candidate.monitor.ids)) == 96 * 500


def test_abrupt_nuisance_changes_input_but_not_outcome_law_or_labels():
    control = generate_monitoring("linear_probabilistic", "M0_stationary", 73116002)
    changed = generate_monitoring("linear_probabilistic", "M1_abrupt_nuisance", 73116002)
    start = 24 * 500
    np.testing.assert_array_equal(changed.evaluator.labels, control.evaluator.labels)
    np.testing.assert_array_equal(changed.evaluator.expected_label_probability,
                                  control.evaluator.expected_label_probability)
    np.testing.assert_array_equal(changed.monitor.x[:start], control.monitor.x[:start])
    np.testing.assert_allclose(changed.monitor.x[start:, :7], control.monitor.x[start:, :7])
    np.testing.assert_allclose(changed.monitor.x[start:, 7] - control.monitor.x[start:, 7], 1.5)


def test_dependence_intervention_changes_pair_correlation_and_preserves_marginal_moments():
    control = generate_monitoring("linear_probabilistic", "M0_stationary", 73116003)
    changed = generate_monitoring("linear_probabilistic", "M6_dependence_marginals_preserved", 73116003)
    start = 24 * 500
    before, after = control.monitor.x[start:, :2], changed.monitor.x[start:, :2]
    assert np.corrcoef(before.T)[0, 1] < 0.35
    assert np.corrcoef(after.T)[0, 1] > 0.85
    np.testing.assert_allclose(before.mean(axis=0), after.mean(axis=0), atol=0.025)
    np.testing.assert_allclose(before.std(axis=0), after.std(axis=0), atol=0.025)
    assert changed.truth_metadata["changed_dependence_pairs"] == [[0, 1]]
    assert changed.truth_metadata["changed_marginal_features"] == []


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("mechanism", ["M4_abrupt_conditional_reversal", "M5_recurrent_conditional_reversal"])
def test_conditional_reversal_probability_truth_matches_label_flip(task, mechanism):
    control = generate_monitoring(task, "M0_stationary", 73116004)
    changed = generate_monitoring(task, mechanism, 73116004)
    active = np.zeros(len(changed.evaluator.labels), dtype=bool)
    batches = np.arange(len(active)) // 500 + 1
    if mechanism == "M4_abrupt_conditional_reversal":
        active = batches >= 25
    else:
        active = ((batches >= 25) & (batches <= 40)) | ((batches >= 65) & (batches <= 80))
    if task == "five_region":
        active &= changed.evaluator.region_id == 2
    np.testing.assert_array_equal(changed.evaluator.labels[~active], control.evaluator.labels[~active])
    np.testing.assert_array_equal(changed.evaluator.labels[active], 1 - control.evaluator.labels[active])
    np.testing.assert_allclose(changed.evaluator.expected_label_probability[active],
                               1 - control.evaluator.expected_label_probability[active])


@pytest.mark.parametrize("feature_count", [2, 8, 32, 128])
def test_generator_supports_locked_and_small_feature_dimensions(feature_count):
    hist = generate_historical("xor_nonlinear", 73116005, 1, 300, feature_count)
    stream = generate_monitoring("xor_nonlinear", "M1_abrupt_nuisance", 73116005,
                                 feature_count=feature_count, batch_size=10)
    assert hist.x.shape == (300, feature_count)
    assert stream.monitor.x.shape == (960, feature_count)


def test_generator_rejects_one_feature_for_two_coordinate_mechanisms():
    with pytest.raises(ValueError):
        generate_monitoring("xor_nonlinear", "M0_stationary", 1, feature_count=1)


def test_gradual_and_repeated_mechanisms_follow_the_locked_batch_schedule():
    control = generate_monitoring("linear_probabilistic", "M0_stationary", 73116006)
    gradual = generate_monitoring("linear_probabilistic", "M2_gradual_predictive", 73116006)
    repeated = generate_monitoring("linear_probabilistic", "M3_incremental_repeated_small", 73116006)
    x0 = np.arange(96 * 500) // 500 + 1
    ramp = np.clip((x0 - 24) / 16, 0, 1) * 1.2
    np.testing.assert_allclose(gradual.monitor.x[:, 0] - control.monitor.x[:, 0], ramp)
    changes = np.array([25, 37, 49, 61, 73, 85])
    increments = (x0[:, None] >= changes[None, :]).sum(axis=1) * .25
    np.testing.assert_allclose(repeated.monitor.x[:, 7] - control.monitor.x[:, 7], increments)


def test_out_of_support_displacement_is_post_intervention_and_truth_tracks_new_law():
    control = generate_monitoring("linear_probabilistic", "M0_stationary", 73116007)
    changed = generate_monitoring("linear_probabilistic", "M7_out_of_support", 73116007)
    start = 24 * 500
    np.testing.assert_array_equal(changed.monitor.x[:start], control.monitor.x[:start])
    np.testing.assert_allclose(changed.monitor.x[start:, :2] - control.monitor.x[start:, :2], 2.5)
    assert not np.allclose(changed.evaluator.expected_label_probability[start:],
                           control.evaluator.expected_label_probability[start:])



def test_seasonal_stress_is_nuisance_only_and_separate_from_primary_matrix():
    from drift_monitoring.phase3.strengthening_generators import SEASONAL
    control = generate_monitoring("linear_probabilistic", "M0_stationary", 73116101)
    seasonal = generate_monitoring("linear_probabilistic", SEASONAL, 73116101)
    np.testing.assert_array_equal(seasonal.evaluator.labels, control.evaluator.labels)
    np.testing.assert_array_equal(seasonal.evaluator.expected_label_probability,
                                  control.evaluator.expected_label_probability)
    np.testing.assert_array_equal(seasonal.monitor.x[:, :7], control.monitor.x[:, :7])
    expected = .6 * np.sin(2 * np.pi * (np.arange(96 * 500) // 500) / 24.0)
    np.testing.assert_allclose(seasonal.monitor.x[:, 7] - control.monitor.x[:, 7], expected, atol=1e-12)
    assert seasonal.truth_metadata["intervention_windows_batches"] == [[1, 96]]


def test_monitor_feature_source_cannot_expose_or_mutate_evaluator_truth():
    from dataclasses import fields
    stream = generate_monitoring("five_region", "M4_abrupt_conditional_reversal", 73116102)
    names = {field.name for field in fields(stream.monitor)}
    assert names == {"x", "ids"}
    assert not hasattr(stream.monitor, "y")
    assert not hasattr(stream.monitor, "labels")
    assert not hasattr(stream.monitor, "region_id")
    assert not hasattr(stream.monitor, "expected_label_probability")
    np.testing.assert_array_equal(stream.monitor.ids, stream.evaluator.ids)
    with pytest.raises(ValueError):
        stream.monitor.x[0, 0] = 123.0
    with pytest.raises(ValueError):
        stream.evaluator.labels[0] = 1 - stream.evaluator.labels[0]
