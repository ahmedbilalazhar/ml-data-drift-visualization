import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_noaa_uncertainty_bootstrap_separates_components_and_is_reproducible():
    run = ROOT / "results/phase3/noaa_reference_calibration_uncertainty_20261117_v1"
    receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run / "uncertainty_summary.csv")
    draws = pd.read_csv(run / "bootstrap_draws.csv")
    assert receipt["status"] == "complete"
    assert receipt["bootstrap_replicates_per_component"] == 200
    assert receipt["block_length_rows"] == 30
    assert receipt["components"] == ["calibrator_only", "reference_only", "joint"]
    assert receipt["monitoring_labels"].startswith("evaluator-only")
    assert len(draws) == 3 * 200 * 21
    assert set(summary.component) == {"calibrator_only", "reference_only", "joint"}
    assert summary.groupby("component").batch.nunique().eq(21).all()
    assert summary.loc[summary.component.eq("reference_only"), "estimate_upper"].sub(
        summary.loc[summary.component.eq("reference_only"), "estimate_lower"]).max() < 1e-12
    for rel, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected


def test_noaa_uncertainty_report_is_compiled_and_receipt_bound():
    report = ROOT / "reports/phase3/noaa_reference_calibration_uncertainty_20261117_v2"
    receipt = json.loads((report / "REPORT_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert (report / "NOAA_UNCERTAINTY.pdf").stat().st_size > 100_000
    assert (report / "noaa_uncertainty.png").is_file()
    for rel, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
