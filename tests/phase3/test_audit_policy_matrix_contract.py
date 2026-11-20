import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_audit_policy_matrix import decision_schedule

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase3/audit_policy_matrix_20261117_v1"
REPORT = ROOT / "reports/phase3/audit_policy_matrix_20261117_v7"


def _toy_stream():
    group = pd.DataFrame({
        "batch": np.arange(1, 25),
        "cohort": [f"c{i}" for i in range(1, 25)],
        "reference_accuracy": np.repeat(.8, 24),
        "min_adjusted_p": np.repeat(.5, 24),
        "cal_concern": np.repeat(0., 24),
    })
    truth = pd.DataFrame({"cohort": group.cohort, "accuracy": np.repeat(.8, 24),
                          "harm": np.repeat(False, 24)}).set_index("cohort", verify_integrity=True)
    return group, truth


def test_released_error_cannot_use_label_before_strict_release_clock():
    group, truth = _toy_stream()
    baseline, _ = decision_schedule(group, truth, "released_error", 5, 4,
                                    "strict_cap", 1., 1.)
    altered = truth.copy()
    altered.loc["c6", "accuracy"] = .5
    changed, _ = decision_schedule(group, altered, "released_error", 5, 4,
                                   "strict_cap", 1., 1.)
    # The probe is selected at clock 6. With delay 5, source 6 first passes
    # the strict s+d<t gate at clock 12. Altering its label cannot affect 1-11.
    assert [r["batch"] for r in baseline if r["selected"]] == [6]
    assert baseline[:11] == changed[:11]
    assert baseline[11]["batch"] == changed[11]["batch"] == 12
    assert changed[11]["selected"] is True
    assert np.isclose(changed[11]["released_error_score"], .3)


def test_full_matrix_obeys_policy_cost_and_release_contracts():
    per_seed = pd.read_csv(RUN / "policy_per_seed.csv")
    decisions = pd.read_csv(RUN / "online_audit_decisions.csv")
    lock = json.loads((RUN / "protocol_lock.json").read_text(encoding="utf-8"))
    assert len(per_seed) == 30 * 7 * 4 * 3 * 2 * 6
    assert len(decisions) == len(per_seed) * 24
    assert set(per_seed.delay.unique()) == {0, 5, 20, 100}
    assert set(per_seed.budget_cohorts.unique()) == {2, 4, 8}
    assert set(per_seed.cost_mode.unique()) == {"strict_cap", "exact_quota"}
    assert (per_seed[per_seed.cost_mode == "strict_cap"].audited_batches <=
            per_seed[per_seed.cost_mode == "strict_cap"].budget_cohorts).all()
    exact = per_seed[per_seed.cost_mode == "exact_quota"]
    assert exact.audited_batches.eq(exact.budget_cohorts).all()
    assert decisions[decisions.delay == 100].prior_released_selected_batches.eq("[]").all()
    selected = decisions[decisions.selected]
    for row in selected.itertuples():
        assert row.selected_labels_available_at == row.batch + row.delay
        assert row.selected_labels_first_usable_before_decision == row.batch + row.delay + 1
    for row in decisions.itertuples():
        sources = json.loads(row.prior_released_selected_batches)
        assert all(source + row.delay < row.batch for source in sources)
    assert lock["outcome_informed"] is True
    assert lock["final_seeds"] == list(range(100, 130))


def test_paired_aggregate_report_uses_seeds_as_the_resampling_unit():
    effects = pd.read_csv(REPORT / "paired_overall_effects.csv")
    receipt = json.loads((REPORT / "report_receipt.json").read_text(encoding="utf-8"))
    assert len(effects) == 5 * 4 * 2 * 2
    assert effects.paired_seeds.eq(30).all()
    assert (effects.lower <= effects.mean_difference_vs_uniform).all()
    assert (effects.mean_difference_vs_uniform <= effects.upper).all()
    assert effects.bootstrap_unit.eq("seed after averaging matched conditions and budgets").all()
    assert "paired_overall_effects.csv" in receipt["outputs_sha256"]
    assert "audit_policy_matrix.pdf" in receipt["outputs_sha256"]
    acquisition = pd.read_csv(REPORT / "policy_acquisition_summary.csv")
    uniform_count = acquisition[(acquisition.policy == "uniform") &
                                (acquisition.cost_mode == "strict_cap")].selected.iloc[0]
    conjunction_count = acquisition[(acquisition.policy == "conjunction") &
                                    (acquisition.cost_mode == "strict_cap")].selected.iloc[0]
    assert uniform_count > conjunction_count
    assert "policy_acquisition_summary.csv" in receipt["outputs_sha256"]
    for name in ("audit_policy_matrix.tex", "audit_policy_matrix.pdf", "audit_policy_effects.pdf",
                 "audit_policy_effects.png", "audit_policy_effects.svg"):
        assert (REPORT / name).is_file()
