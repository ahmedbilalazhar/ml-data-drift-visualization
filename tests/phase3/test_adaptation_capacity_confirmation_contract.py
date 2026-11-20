import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase3/adaptation_capacity_control_confirmation_20261118_v1"
REPORT = ROOT / "reports/phase3/adaptation_capacity_confirmation_20261118_v4"
COLLISION = ROOT / "docs/phase3/SEED_COLLISION_AUDIT_ADAPTATION_CAPACITY_CONFIRMATION_20261118.json"


def test_independent_confirmation_uses_reserved_seed_bank_and_complete_matrix():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    receipt = json.loads((RUN / "audit_receipt.json").read_text(encoding="utf-8"))
    collision = json.loads(COLLISION.read_text(encoding="utf-8"))
    assert len(metrics) == 10 * 3 * 4 * 3 * 4
    assert receipt["seed_bank"] == list(range(73116401, 73116411))
    assert collision["matches_outside_reserved_config"] == []
    assert collision["result"] == "no exact evaluation-seed collisions found"


def test_confirmation_run_preserves_timing_matched_feedback_and_caps():
    metrics = pd.read_csv(RUN / "seed_metrics.csv")
    batches = pd.read_csv(RUN / "online_batch_audit.csv")
    fits = pd.read_csv(RUN / "fit_audit.csv")
    assert len(batches) == 10 * 3 * 4 * 3 * 4 * 96
    released = batches.latest_available_at.notna()
    assert (batches.loc[released, "latest_available_at"] < batches.loc[released, "batch"]).all()
    assert metrics[metrics.method != "frozen"].successful_updates.eq(4).all()
    assert metrics.adaptation_row_uses.le(8_000).all()
    assert fits.groupby(["seed", "task", "mechanism", "delay", "clock"]).common_feedback_ids_sha256.nunique().eq(1).all()
    run_receipt = json.loads((RUN / "audit_receipt.json").read_text(encoding="utf-8"))
    for name, expected in run_receipt["output_sha256"].items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected


def test_confirmation_report_is_honest_and_receipted():
    report = json.loads((REPORT / "report_receipt.json").read_text(encoding="utf-8"))
    assert report["reproduction_script_sha256"] == hashlib.sha256((ROOT / "scripts/reproduce_adaptation_capacity_confirmation.ps1").read_bytes()).hexdigest()
    effects = pd.read_csv(REPORT / "independent_seed_effects.csv")
    assert len(effects) == 9
    assert effects.seeds.eq(10).all()
    tex = (REPORT / "adaptation_capacity_confirmation.tex").read_text(encoding="utf-8")
    assert "exploratory" in tex
    assert "not pooled with the 30-seed primary run" in tex
    for name, expected in report["output_sha256"].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected
