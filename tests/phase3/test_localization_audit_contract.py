import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_historical_region_ids_use_the_historical_random_stream_only():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from drift_monitoring.phase3.strengthening_generators import _rng, historical_region_ids

    seed, rows, dimensions = 73116201, 4000, 8
    rng = _rng(seed, "five_region", 1)
    rng.normal(size=(rows, dimensions))
    expected = rng.integers(0, 5, size=rows)
    actual = historical_region_ids("five_region", seed, 1, rows, dimensions)
    np.testing.assert_array_equal(actual, expected)


def test_localization_audit_is_seed_complete_and_mapping_fit_is_historical_only():
    run = ROOT / "results/phase3/localization_audit_20261117_v3"
    receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    region = pd.read_csv(run / "region_localization.csv")
    pairs = pd.read_csv(run / "dependence_pair_localization.csv")
    assert receipt["status"] == "complete"
    assert len(receipt["seed_bank"]) == 30
    assert receipt["mapping_fit_data"].startswith("historical training rows only")
    assert "run_localization_audit.py" in ";".join(receipt["source_sha256"])
    assert "region_localization.csv" in ";".join(receipt["output_sha256"])
    assert len(region[region.batch.isna()]) == 90
    assert len(region[region.batch.notna()]) == 25_920
    assert len(pairs) == 17_280
    changed = pairs[(pairs.mechanism == "M6_dependence_marginals_preserved")
                    & pairs.intervention_active]
    assert changed.target_pair_top1.mean() > .98


def test_verified_real_dataset_loaders_use_readable_v2_and_weather_split_contract():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from drift_monitoring.data.real import load_real

    elec = load_real("elec2")
    cov = load_real("covtype_subset")
    weather = load_real("weather")
    assert (elec["n"], cov["n"], weather["n"]) == (45_312, 83_665, 18_210)
    assert len(elec["numeric"]) == 7 and len(cov["numeric"]) == 54 and len(weather["numeric"]) == 8
    assert weather["splits"]["train"] == [0, 3641]
    assert weather["splits"]["calib"] == [3642, 5462]
    assert weather["splits"]["reference"] == [5463, 7283]
    assert weather["splits"]["test"] == [7284, 18209]


def test_full_dataset_audit_reads_all_three_processed_streams():
    report = json.loads((ROOT / "data/interim/audit_all_datasets_20261117.json")
                        .read_text(encoding="utf-8"))
    assert not report["audit_load_errors"]
    assert all(not issue for issue in report["checksum_issues"].values())
    assert set(report["audits"]) == {"elec2", "covtype_subset", "weather_noaa_gsod"}


def test_localization_report_receipt_binds_outputs_to_input_run():
    receipt = json.loads((ROOT / "reports/phase3/localization_audit_20261117_v5/REPORT_RECEIPT.json")
                         .read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["bootstrap_replicates"] == 2000
    assert "results/phase3/localization_audit_20261117_v3/region_localization.csv" in receipt["input_sha256"]
    assert "reports/phase3/localization_audit_20261117_v5/LOCALIZATION_AUDIT.tex" in receipt["output_sha256"]

