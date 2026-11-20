import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.dde_subset import (  # noqa: E402
    binary_ranking_metrics, fit_pooled_dde_curve, pair_correlation, score_unmonitored,
)
from scripts import run_dde_subset_audit as runner  # noqa: E402
from scripts import report_dde_subset_audit as report  # noqa: E402


def test_dde_curve_is_pooled_only_from_observed_development_flags():
    corr = np.array([[1.0, .9, .1], [.9, 1.0, .2], [.1, .2, 1.0]])
    fit = fit_pooled_dde_curve([corr, corr], [np.array([1, 1, 0]), np.zeros(3, bool)],
                               np.array([0.0, .5, .8]))
    assert fit["episodes_used"] == 1
    assert fit["pair_count"].tolist() == [3, 1, 1]
    assert fit["both_drifted_count"].tolist() == [1, 1, 1]
    assert fit["curve"].tolist() == [1 / 3, 1.0, 1.0]


def test_unmonitored_score_uses_only_frozen_reference_correlation_and_flags():
    corr = np.array([[1.0, .9, .2], [.9, 1.0, .1], [.2, .1, 1.0]])
    curve = {"thresholds": np.array([0.0, .5]), "curve": np.array([.1, .8])}
    scores = score_unmonitored(curve, corr, np.array([0]), np.array([1, 2]))
    assert np.allclose(scores, [.8, .1])
    assert np.array_equal(score_unmonitored(curve, corr, np.array([], dtype=int), np.array([1, 2])), [0, 0])


def test_unsupported_correlation_bins_fall_back_to_nearest_lower_supported_bin():
    corr = np.array([[1.0, .9, .1], [.9, 1.0, .1], [.1, .1, 1.0]])
    curve = {"thresholds": np.array([0.0, .5, .8]), "curve": np.array([.1, .8, 0.0]),
             "pair_count": np.array([10, 5, 0])}
    scores = score_unmonitored(curve, corr, np.array([0]), np.array([1, 2]))
    assert np.allclose(scores, [.8, .1])


def test_curve_without_any_supported_bins_returns_undefined_scores():
    corr = np.eye(3)
    curve = {"thresholds": np.array([0.0, .5]), "curve": np.array([0.0, 0.0]),
             "pair_count": np.array([0, 0])}
    assert np.isnan(score_unmonitored(curve, corr, np.array([0]), np.array([1, 2]))).all()


def test_reference_correlation_rejects_invalid_values_and_produces_finite_scores():
    x = np.array([[0, 0], [1, 2], [2, 4], [3, 6]], dtype=float)
    corr = pair_correlation(x)
    assert np.isfinite(corr).all()
    assert corr[0, 1] > .99
    try:
        pair_correlation(np.array([[np.nan, 0], [1, 2]]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite reference data must be rejected")


def test_empty_truth_is_explicitly_undefined_not_counted_as_success():
    result = binary_ranking_metrics(np.array([.8, .1]), np.array([False, False]))
    assert result["empty_truth"] is True
    assert np.isnan(result["average_precision"])
    assert np.isnan(result["auroc"])
    assert np.isnan(result["precision_at_k"])


def test_precision_at_k_averages_ties_instead_of_using_feature_index():
    result = binary_ranking_metrics(np.array([1.0, 1.0, 0.0]), np.array([True, False, True]), k=1)
    assert result["precision_at_k"] == 0.5
    all_tied = binary_ranking_metrics(np.zeros(4), np.array([True, False, False, False]), k=2)
    assert all_tied["precision_at_k"] == 0.25


def test_locked_development_validation_evaluation_seed_banks_are_disjoint():
    cfg = json.loads((ROOT / "configs/dde_subset_audit_v1.json").read_text(encoding="utf-8"))
    banks = [set(cfg["development_seeds"]), set(cfg["validation_seeds"]), set(cfg["evaluation_seeds"])]
    assert all(not (banks[i] & banks[j]) for i in range(3) for j in range(i + 1, 3))
    assert cfg["smoke_seed"] not in set.union(*banks)
    assert len(banks[0]) == 5 and len(banks[1]) == 10 and len(banks[2]) == 30


def test_temporal_checkpoints_cover_onset_transition_recovery_and_endpoints():
    cfg = json.loads((ROOT / "configs/dde_subset_audit_v1.json").read_text(encoding="utf-8"))
    checks = set(cfg["evaluation_check_batches"])
    assert {12, 24, 25, 40, 41, 64, 65, 80, 81, 96}.issubset(checks)
    assert len(checks) == len(cfg["evaluation_check_batches"])
    assert all(1 <= b <= cfg["batches"] for b in checks)


def test_evaluation_scores_do_not_depend_on_monitoring_labels(monkeypatch):
    cfg = json.loads((ROOT / "configs/dde_subset_audit_v1.json").read_text(encoding="utf-8"))
    task, dim, seed = cfg["tasks"][0], cfg["dimensions"][0], cfg["evaluation_seeds"][0]
    locked = {"threshold": .10, "correlation_bins": cfg["curve_correlation_bins"],
              "dde_curve": [.05] * len(cfg["curve_correlation_bins"]),
              "pair_count": [10] * len(cfg["curve_correlation_bins"])}
    calibration = {f"{task}|{dim}": locked}
    run = runner.generate_monitoring
    expected = runner.evaluate_seed(cfg, calibration, 0, task, dim, seed, "evaluation",
                                    ["M1_abrupt_nuisance"])

    def with_flipped_labels(*args, **kwargs):
        stream = run(*args, **kwargs)
        evaluator = replace(stream.evaluator, labels=1 - stream.evaluator.labels)
        return replace(stream, evaluator=evaluator)

    monkeypatch.setattr(runner, "generate_monitoring", with_flipped_labels)
    actual = runner.evaluate_seed(cfg, calibration, 0, task, dim, seed, "evaluation",
                                  ["M1_abrupt_nuisance"])
    assert_frame_equal(pd.DataFrame(expected), pd.DataFrame(actual))
    assert all(row["online_evaluator_truth_access"] is False for row in actual)


def test_report_completion_gate_rejects_nonterminal_runs():
    import pytest

    with pytest.raises(RuntimeError, match="report requires complete run"):
        report.require_complete({}, {"status": "running"}, {}, pd.DataFrame())


def test_resume_validator_accepts_only_complete_grid_cells():
    cfg = json.loads((ROOT / "configs/dde_subset_audit_v1.json").read_text(encoding="utf-8"))
    cell = ROOT / "results/phase3/dde_subset_audit_smoke_20261118_v5/evaluation_seed_8732999_linear_probabilistic_d32.csv"
    assert len(runner.verify_completed_cell(cell, cfg)) == 64
