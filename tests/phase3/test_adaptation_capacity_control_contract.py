import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase3/adaptation_capacity_control_20261118_v2"
REPORT = ROOT / "reports/phase3/adaptation_capacity_control_20261118_v9"


def test_corrected_run_has_complete_paired_matrix_and_supersedes_invalid_initialization():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    lock = json.loads((RUN / "audit_receipt.json").read_text(encoding="utf-8"))
    assert len(metrics) == 30 * 3 * 4 * 3 * 4
    assert metrics.groupby(["seed", "task", "mechanism", "delay", "method"]).size().eq(1).all()
    assert lock["protocol"] == "matched-capacity-global-interaction-control-v2"
    assert "unlike other arms" in lock["supersedes"]
    assert lock["seed_bank"] == list(range(73116301, 73116331))
    assert set(metrics.method) == {"frozen", "matched_global", "matched_local", "matched_interaction"}


def test_capacity_run_obeys_temporal_feedback_cost_and_output_hash_contracts():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    batch = pd.read_csv(RUN / "online_batch_audit.csv")
    fits = pd.read_csv(RUN / "fit_audit.csv")
    lock = json.loads((RUN / "audit_receipt.json").read_text(encoding="utf-8"))
    assert len(batch) == 30 * 3 * 4 * 3 * 4 * 96
    assert batch.prediction_persisted.all()
    released = batch.latest_available_at.notna()
    assert (batch.loc[released, "latest_available_at"] < batch.loc[released, "batch"]).all()
    assert metrics[metrics.method != "frozen"].successful_updates.eq(4).all()
    assert metrics.adaptation_row_uses.le(8_000).all()
    assert set(fits.clock) == {25, 49, 73, 89}
    ids = fits.groupby(["seed", "task", "mechanism", "delay", "clock"]).common_feedback_ids_sha256.nunique()
    assert ids.eq(1).all()
    assert metrics.failed_local_region_fits.sum() == 549
    for name, expected in lock["output_sha256"].items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected


def test_report_hashes_and_local_interaction_similarity_are_preserved():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    receipt = json.loads((REPORT / "report_receipt.json").read_text(encoding="utf-8"))
    assert receipt["report_script_sha256"] == hashlib.sha256((ROOT / "scripts/report_adaptation_capacity_control.py").read_bytes()).hexdigest()
    assert receipt["reproduction_script_sha256"] == hashlib.sha256((ROOT / "scripts/reproduce_adaptation_capacity_control.ps1").read_bytes()).hexdigest()
    means = metrics.groupby(["task", "method"]).accuracy.mean().unstack()
    assert (means.matched_local - means.matched_interaction).abs().max() < 0.0002
    effects = pd.read_csv(REPORT / "local_vs_global_accuracy_effects.csv")
    simultaneous = pd.read_csv(REPORT / "local_vs_global_simultaneous_bootstrap.csv")
    assert len(effects) == 3 * 4 * 3
    assert len(simultaneous) == 3 * 4 * 3
    assert (simultaneous.simultaneous_lower <= simultaneous["mean"]).all()
    assert (simultaneous["mean"] <= simultaneous.simultaneous_upper).all()
    assert (REPORT / "adaptation_capacity_control.tex").is_file()
    assert (REPORT / "adaptation_capacity_control.pdf").is_file()
    for name, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected
