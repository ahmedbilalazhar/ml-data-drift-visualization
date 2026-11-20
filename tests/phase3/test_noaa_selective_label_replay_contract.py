import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"results/phase3/noaa_selective_label_replay_20261117_v1"
REPORT=ROOT/"reports/phase3/noaa_selective_label_replay_20261117_v3"

def test_noaa_finite_stream_sampling_design_and_row_log():
    receipt=json.loads((RUN/"audit_receipt.json").read_text(encoding="utf-8"))
    metrics=pd.read_csv(RUN/"design_replicate_metrics.csv")
    summary=pd.read_csv(RUN/"design_summary.csv").set_index("policy")
    assert receipt["design_replicates"]==1000
    assert receipt["batches"]==21 and receipt["rows_scored"]==10500
    assert "not prospective" in receipt["interpretation"]
    assert len(metrics)==3000
    assert metrics.groupby("policy").replicate.nunique().eq(1000).all()
    assert summary.loc["targeted_10pct","population_estimable"]==False
    assert pd.isna(summary.loc["targeted_10pct","ht_bias"])
    assert abs(summary.loc["stratified_random_audit","ht_bias"])<.01
    assert abs(summary.loc["simple_random_budget_matched","ht_bias"])<.01
    assert .90<summary.loc["stratified_random_audit","ht_ci_coverage"]<.99
    sample=pd.read_csv(RUN/"selected_label_log.csv.gz",nrows=3000)
    assert set(sample.loc[sample.policy=="targeted_10pct","inclusion_probability"])=={1.0}
    assert set(sample.loc[sample.policy=="stratified_random_audit","inclusion_probability"]).issubset({.1,1.0})
    assert set(sample.loc[sample.policy=="simple_random_budget_matched","inclusion_probability"])=={.19}
    assert sample.correctness.isin([0,1]).all()
    for name,digest in receipt["output_sha256"].items():
        path=ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest

def test_noaa_selective_report_and_reproduction_wrapper_exist():
    assert (REPORT/"NOAA_SELECTIVE_LABEL_REPLAY.pdf").is_file()
    assert (REPORT/"NOAA_SELECTIVE_LABEL_REPLAY.tex").is_file()
    assert (REPORT/"noaa_selective_design.png").is_file()
    assert (REPORT/"noaa_selective_design.svg").is_file()
    wrapper=(ROOT/"scripts/reproduce_noaa_selective_label_replay.ps1").read_text(encoding="utf-8")
    assert "reproduce" in wrapper.lower() or "RunDir" in wrapper
    tex=(REPORT/"NOAA_SELECTIVE_LABEL_REPLAY.tex").read_text(encoding="utf-8")
    assert "not prospective" in tex.lower()
    assert "1,000 randomized sampling" in tex or "1,000 independent sampling" in tex or "1,000 times" in tex
