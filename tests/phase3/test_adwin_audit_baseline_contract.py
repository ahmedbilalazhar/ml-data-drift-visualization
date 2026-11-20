import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_adwin_audit_baseline import make_schedule
from scripts.run_adwin_midpoint_probe_sensitivity import engine as midpoint_engine

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase3/adwin_audit_baseline_20261117_v1"
REPORT = ROOT / "reports/phase3/adwin_audit_baseline_20261117_v3"


def _toy_stream():
    group = pd.DataFrame({
        "batch": np.arange(1, 25), "cohort": [f"c{i}" for i in range(1, 25)],
        "reference_accuracy": np.repeat(.8, 24),
    })
    truth = pd.DataFrame({"cohort": group.cohort, "accuracy": np.repeat(.8, 24),
                          "harm": np.repeat(False, 24)}).set_index("cohort", verify_integrity=True)
    return group, truth


def test_adwin_receives_only_selected_labels_after_release_clock():
    group, truth = _toy_stream()
    errors = np.zeros((24, 500), dtype=bool)
    altered_unselected = errors.copy()
    altered_unselected[0, :] = True
    base, _ = make_schedule(group, errors, truth, 5, 4, "strict_cap")
    changed, _ = make_schedule(group, altered_unselected, truth, 5, 4, "strict_cap")
    assert base[:11] == changed[:11]
    assert [r["batch"] for r in base if r["selected"]] == [6]
    assert base[10]["labels_processed_so_far"] == 0
    assert base[11]["batch"] == 12
    assert base[11]["new_released_source_batches"] == "[6]"
    assert base[11]["labels_processed_so_far"] == 500
    delayed, metrics = make_schedule(group, errors, truth, 100, 4, "strict_cap")
    assert sum(r["labels_processed_so_far"] for r in delayed) == 0
    assert metrics["adwin_row_updates"] == 0


def test_full_adwin_run_respects_budget_and_label_contracts():
    per_seed = pd.read_csv(RUN / "adwin_per_seed.csv")
    decisions = pd.read_csv(RUN / "adwin_decisions.csv")
    lock = json.loads((RUN / "protocol_lock.json").read_text(encoding="utf-8"))
    assert len(per_seed) == 30 * 7 * 4 * 3 * 2
    assert len(decisions) == len(per_seed) * 24
    strict = per_seed[per_seed.cost_mode == "strict_cap"]
    assert (strict.audited_batches <= strict.budget_cohorts).all()
    exact = per_seed[per_seed.cost_mode == "exact_quota"]
    assert exact.audited_batches.eq(exact.budget_cohorts).all()
    assert per_seed[per_seed.delay == 100].adwin_row_updates.eq(0).all()
    assert (per_seed.adwin_row_updates <= per_seed.labels_requested).all()
    assert lock["detector"] == "river.drift.ADWIN"
    assert lock["river_version"] == "0.27.0"
    assert lock["evaluation_seeds"] == list(range(100, 130))


def test_adwin_report_is_compiled_and_records_its_paired_unit():
    effects = pd.read_csv(REPORT / "adwin_paired_aggregate_effects.csv")
    summary = pd.read_csv(REPORT / "adwin_delay_summary.csv")
    receipt = json.loads((REPORT / "report_receipt.json").read_text(encoding="utf-8"))
    assert len(effects) == 2 * 2 * 4 * 2
    assert effects.paired_seeds.eq(30).all()
    assert effects.bootstrap_unit.eq("seed after averaging seven conditions and three budgets").all()
    strict = summary[summary.cost_mode == "strict_cap"]
    assert strict.detection_cells.eq(0).all()
    assert (REPORT / "adwin_baseline.pdf").is_file()
    assert (REPORT / "adwin_baseline.tex").is_file()
    assert "adwin_baseline.pdf" in receipt["outputs_sha256"]


def test_midpoint_probes_are_selected_by_remaining_cohort_budget():
    group, truth = _toy_stream()
    errors = np.zeros((24, 500), dtype=bool)
    midpoint_engine.BOOTSTRAP_CLOCKS = (1, 12)
    for budget in (2, 4, 8):
        for delay in (0, 5, 20, 100):
            decisions, metrics = midpoint_engine.schedule_two_probes(
                group, errors, truth, delay, budget, "strict_cap")
            selected_probe_clocks = [
                r["batch"] for r in decisions if r["selected"] and r["bootstrap_probe"]
            ]
            assert selected_probe_clocks == [1, 12]
            assert metrics["audited_batches"] == 2
            assert metrics["audited_batches"] <= budget


def test_corrected_midpoint_run_has_two_selected_probes_and_valid_hashes():
    run = ROOT / "results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2"
    per_seed = pd.read_csv(run / "two_probe_per_seed.csv")
    decisions = pd.read_csv(run / "two_probe_decisions.csv")
    lock = json.loads((run / "protocol_lock.json").read_text(encoding="utf-8"))
    assert len(per_seed) == 30 * 7 * 4 * 3 * 2
    assert len(decisions) == len(per_seed) * 24
    assert lock["bootstrap_clocks"] == [1, 12]
    probes = decisions[decisions.bootstrap_probe]
    assert probes.groupby(["seed", "condition", "delay", "budget_cohorts", "cost_mode"]).selected.sum().eq(2).all()
    strict = per_seed[per_seed.cost_mode == "strict_cap"]
    assert (strict.audited_batches <= strict.budget_cohorts).all()
    assert per_seed[per_seed.delay == 100].adwin_row_updates.eq(0).all()
    assert lock["supersedes"].startswith("adwin_midpoint_probe_sensitivity_20261118_v1")
    import hashlib
    for name, digest in lock["output_sha256"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
