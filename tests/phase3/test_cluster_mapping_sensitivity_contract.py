import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_cluster_mapping_sensitivity_is_complete_frozen_and_seed_complete():
    run = ROOT / "results/phase3/cluster_mapping_sensitivity_20261117_v2"
    receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run / "cluster_mapping_sensitivity.csv")
    mappings = json.loads((run / "historical_mappings.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["mapping_fit_data"] == "historical training rows only"
    assert receipt["evaluation_truth_use"] == "metrics only"
    assert len(receipt["seed_bank"]) == 30
    assert set(frame.representation) == {"scaled_full", "frozen_pca2", "frozen_pca3"}
    assert set(frame.clusters) == {3, 5, 10}
    assert len(frame) == 30 * 3 * 3
    assert len(mappings) == len(frame)
    assert frame.mapping_coverage.eq(1.0).all()
    for rel, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected


def test_cluster_mapping_sensitivity_report_receipt_binds_tex_pdf_and_figures():
    report = ROOT / "reports/phase3/cluster_mapping_sensitivity_20261117_v1"
    receipt = json.loads((report / "REPORT_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["bootstrap_replicates"] == 5000
    assert (report / "CLUSTER_MAPPING_SENSITIVITY.pdf").stat().st_size > 100_000
    assert (report / "cluster_mapping_sensitivity.png").is_file()
    assert (report / "cluster_mapping_sensitivity.svg").is_file()
    for rel, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
