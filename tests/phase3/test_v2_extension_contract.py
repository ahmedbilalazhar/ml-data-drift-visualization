import numpy as np

from drift_monitoring.phase3.v2 import (
    MODELS,
    _xor_monitoring,
    fit_models,
    historical_partition,
    score_model,
)
from drift_monitoring.phase3.core import FeatureBatch


def test_v2_seed_spaces_are_disjoint_and_task_partition_ids_are_unique():
    development = set(range(190001, 190006))
    v1_final = set(range(100, 130))
    xor_final = set(range(200, 230))
    assert not development & v1_final
    assert not development & xor_final
    assert not v1_final & xor_final
    parts = [historical_partition("xor_synthetic", 200, stage, size)
             for stage, size in ((0, 4000), (1, 2000), (2, 2000))]
    assert all(len(part.ids) == len(np.unique(part.ids)) for part in parts)
    assert not any(set(parts[i].ids) & set(parts[j].ids) for i in range(3) for j in range(i))


def test_xor_interventions_share_inputs_and_change_only_the_declared_mechanism():
    stationary = _xor_monitoring(200, "X0")
    conditional = _xor_monitoring(200, "X3")
    noise = _xor_monitoring(200, "X4")
    np.testing.assert_array_equal(stationary.x, conditional.x)
    np.testing.assert_array_equal(stationary.x, noise.x)
    np.testing.assert_array_equal(stationary.y[:4000], conditional.y[:4000])
    np.testing.assert_array_equal(stationary.y[:4000], noise.y[:4000])
    assert np.all(conditional.y[4000:] == 1 - stationary.y[4000:])
    assert np.mean(stationary.y[4000:] != noise.y[4000:]) > 0.2
    assert len(np.unique(stationary.ids)) == 12000


def test_both_frozen_models_use_training_only_preprocessing_and_separate_calibration():
    train = historical_partition("xor_synthetic", 200, 0, 4000)
    calibration = historical_partition("xor_synthetic", 200, 1, 2000)
    reference = historical_partition("xor_synthetic", 200, 2, 2000)
    artifacts = fit_models(200, train, calibration, reference)
    assert set(artifacts) == set(MODELS)
    np.testing.assert_allclose(artifacts["logistic"].scaler.mean_, train.x.mean(axis=0))
    np.testing.assert_allclose(artifacts["shallow_tree"].scaler.mean_, train.x.mean(axis=0))
    assert artifacts["shallow_tree"].model.get_params()["max_depth"] == 6
    assert artifacts["shallow_tree"].model.get_params()["min_samples_leaf"] == 25
    assert all(artifact.reference_n == 2000 for artifact in artifacts.values())
    assert artifacts["logistic"].calibration is not artifacts["shallow_tree"].calibration


def test_scoring_is_feature_only_and_preserves_fixed_class_predictions():
    train = historical_partition("xor_synthetic", 201, 0, 4000)
    calibration = historical_partition("xor_synthetic", 201, 1, 2000)
    reference = historical_partition("xor_synthetic", 201, 2, 2000)
    part = _xor_monitoring(201, "X0")
    artifact = fit_models(201, train, calibration, reference)["shallow_tree"]
    batch = FeatureBatch(part.x[:500], part.ids[:500], 201, "X0", 1)
    record, cache = score_model(artifact, batch, "xor_synthetic", "shallow_tree")
    assert not {"accuracy", "labels", "correct", "harm", "y"}.intersection(record)
    np.testing.assert_array_equal(cache["prediction"], cache["probability"] >= 0.5)
    assert record["cohort"].endswith(":shallow_tree")
    assert 0 <= record["cal_concern"] <= 1
    assert record["cal_lower"] <= record["cal_upper"]
