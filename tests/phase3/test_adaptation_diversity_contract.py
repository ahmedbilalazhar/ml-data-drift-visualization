import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase3/matched_adaptation_diversity_20261118_v1"


def test_full_diversity_matrix_is_complete_and_uses_fresh_collision_audited_seeds():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    lock = json.loads((RUN / "audit_receipt.json").read_text(encoding="utf-8"))
    collision = json.loads((ROOT / "docs/phase3/SEED_COLLISION_AUDIT_ADAPTATION_DIVERSITY_20261118.json").read_text(encoding="utf-8"))
    assert len(metrics) == 30 * 3 * 4 * 3 * 3
    assert metrics.groupby(["seed", "task", "mechanism", "delay", "method"]).size().eq(1).all()
    assert len(lock["seed_bank"]) == 30
    assert lock["seed_bank"] == list(range(73116301, 73116331))
    assert collision["matches_outside_reserved_config"] == []
    assert collision["result"] == "no exact evaluation-seed collisions found"
    assert set(metrics.task) == {"linear_probabilistic", "xor_nonlinear", "five_region"}
    assert set(metrics.mechanism) == {
        "M0_stationary", "M2_gradual_predictive",
        "M5_recurrent_conditional_reversal", "M7_out_of_support",
    }


def test_adaptation_diversity_preserves_clock_budget_and_matched_feedback_contracts():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    batch = pd.read_csv(RUN / "online_batch_audit.csv")
    fits = pd.read_csv(RUN / "fit_audit.csv")
    lock = json.loads((RUN / "audit_receipt.json").read_text(encoding="utf-8"))
    assert len(batch) == len(metrics) * 96
    assert batch.prediction_persisted.all()
    known = batch.latest_available_at.notna()
    assert (batch.loc[known, "latest_available_at"] < batch.loc[known, "batch"]).all()
    adaptive = metrics[metrics.method != "frozen"]
    assert adaptive.update_opportunities.eq(4).all()
    assert adaptive.adaptation_row_uses.le(8_000).all()
    assert metrics[metrics.method == "matched_global"].adaptation_row_uses.eq(8_000).all()
    assert set(fits.clock) == {25, 49, 73, 89}
    keys = ["seed", "task", "mechanism", "delay", "clock"]
    global_hashes = fits[fits.method == "matched_global"].set_index(keys).common_feedback_ids_sha256
    for key, group in fits[fits.method == "matched_local"].groupby(keys):
        assert group.common_feedback_ids_sha256.nunique() == 1
        assert group.common_feedback_ids_sha256.iloc[0] == global_hashes.loc[key]
        assert group.rows_actually_fit.sum() <= group.window_rows_available.iloc[0]
    assert lock["status"] == "complete"
    for name, expected in lock["output_sha256"].items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected


def test_report_outputs_are_hash_receipted_and_results_remain_paired_by_seed():
    report = ROOT / "reports/phase3/matched_adaptation_diversity_20261118_v1"
    paired = pd.read_csv(RUN / "paired_seed_differences.csv")
    receipt = json.loads((report / "report_receipt.json").read_text(encoding="utf-8"))
    local_global = paired[(paired.method == "matched_local_vs_global") & (paired.metric == "accuracy_difference_vs_global")]
    assert len(local_global) == 3 * 4 * 3
    assert local_global.seeds.eq(30).all()
    assert (local_global.lower <= local_global["mean"]).all()
    assert (local_global["mean"] <= local_global.upper).all()
    assert (report / "matched_adaptation_diversity.tex").is_file()
    assert (report / "matched_adaptation_diversity.pdf").is_file()
    for name, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((report / name).read_bytes()).hexdigest() == expected
