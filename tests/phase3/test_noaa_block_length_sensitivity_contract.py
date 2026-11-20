import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_noaa_uncertainty_block_length_runs_are_complete_and_sensitivity_is_ordered():
    runs = {
        7: ROOT / "results/phase3/noaa_reference_calibration_uncertainty_block7_20261117_v1",
        30: ROOT / "results/phase3/noaa_reference_calibration_uncertainty_20261117_v2",
        90: ROOT / "results/phase3/noaa_reference_calibration_uncertainty_block90_20261117_v1",
    }
    for length, run in runs.items():
        receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
        draws = pd.read_csv(run / "bootstrap_draws.csv")
        assert receipt["status"] == "complete"
        assert receipt["block_length_rows"] == length
        assert receipt["bootstrap_replicates_per_component"] == 200
        assert len(draws) == 3 * 200 * 21
        for rel, expected in receipt["output_sha256"].items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
    summary = pd.read_csv(ROOT / "reports/phase3/noaa_block_length_sensitivity_20261117_v3/block_length_summary.csv")
    assert summary.block_length_days.tolist() == [7, 30, 90]
    assert summary.mean_accuracy_interval_width.is_monotonic_increasing
    assert summary.mean_concern_interval_width.is_monotonic_increasing


def test_noaa_block_length_report_is_compiled_and_receipt_bound():
    report = ROOT / "reports/phase3/noaa_block_length_sensitivity_20261117_v3"
    receipt = json.loads((report / "REPORT_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["block_lengths_days"] == [7, 30, 90]
    assert (report / "NOAA_BLOCK_LENGTH_SENSITIVITY.pdf").stat().st_size > 100_000
    for rel, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
