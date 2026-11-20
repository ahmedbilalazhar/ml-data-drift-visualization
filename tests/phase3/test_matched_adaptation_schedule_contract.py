from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase3/matched_adaptation_schedule_20261117_v2"
REPORT = ROOT / "reports/phase3/matched_adaptation_schedule_20261117_v2"


def test_full_matched_run_obeys_clock_label_and_row_use_contracts():
    batch = pd.read_csv(RUN / "online_batch_audit.csv")
    runs = pd.read_csv(RUN / "seed_metrics.csv")
    fits = pd.read_csv(RUN / "fit_audit.csv")
    receipt = pd.read_json(RUN / "audit_receipt.json", typ="series")

    assert len(receipt["seed_bank"]) == 30
    assert set(receipt["update_clocks"]) == {25, 49, 73, 89}
    assert len(runs) == 30 * 2 * 3 * 3
    assert len(batch) == 30 * 2 * 3 * 3 * 96
    assert batch.prediction_persisted.all()
    available = batch.latest_available_at.notna()
    assert (batch.loc[available, "latest_available_at"] < batch.loc[available, "batch"]).all()
    assert runs[runs.method != "frozen"].update_opportunities.eq(4).all()
    assert runs.adaptation_row_uses.le(8_000).all()
    assert runs[runs.method == "matched_global"].adaptation_row_uses.eq(8_000).all()
    assert runs[runs.method == "matched_local"].adaptation_row_uses.le(8_000).all()
    assert set(fits.clock.unique()) == {25, 49, 73, 89}

    global_fits = fits[fits.method == "matched_global"].set_index(["seed", "mechanism", "delay", "clock"])
    local_fits = fits[fits.method == "matched_local"].groupby(["seed", "mechanism", "delay", "clock"])
    assert global_fits.index.nunique() == 30 * 2 * 3 * 4
    for key, group in local_fits:
        assert group.common_feedback_ids_sha256.nunique() == 1
        assert group.common_feedback_ids_sha256.iloc[0] == global_fits.loc[key, "common_feedback_ids_sha256"]
        assert group.rows_actually_fit.sum() <= global_fits.loc[key, "window_rows_available"]


def test_matched_schedule_report_has_paired_seed_intervals_and_editable_source():
    paired = pd.read_csv(REPORT / "paired_seed_differences.csv")
    assert len(paired) == 72
    assert paired.seeds.eq(30).all()
    assert (paired.lower <= paired["mean"]).all() and (paired["mean"] <= paired.upper).all()
    direct = paired[paired.metric == "accuracy_difference_vs_global"]
    assert len(direct) == 6
    for name in (
        "matched_adaptation_schedule.tex", "matched_adaptation_schedule.pdf",
        "matched_adaptation_effects.pdf", "matched_adaptation_effects.png",
        "matched_adaptation_effects.svg", "report_receipt.json",
    ):
        assert (REPORT / name).is_file()
