import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv"
MANIFEST = DATA.with_name("weather_manifest.json")
RUN = ROOT / "results/phase3/noaa_weather_1949_1999_v1"


def test_noaa_weather_processed_contract_and_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    assert manifest["processed"]["row_count"] == len(df) == 18_210
    assert df.timestamp.is_monotonic_increasing and not df.timestamp.duplicated().any()
    assert np.array_equal(df.row_id.to_numpy(), np.arange(len(df)))
    assert set(df.label_rain.unique()) == {0, 1}
    assert set(df.columns) == {"timestamp", "row_id", "temp", "dewp", "slp", "visib",
                               "wdsp", "mxspd", "max", "min", "label_rain"}
    assert np.isfinite(df.drop(columns="timestamp").to_numpy()).all()


def test_noaa_weather_online_predictions_precede_truth_release():
    online = pd.read_csv(RUN / "online/batch_scores.csv")
    events = pd.read_csv(RUN / "run/label_release_events.csv")
    verified = pd.read_csv(RUN / "evaluator/released_verification.csv")
    metrics = pd.read_csv(RUN / "evaluator/batch_metrics.csv")
    assert len(online) == len(verified) == len(metrics) == 21
    assert len(events) == 42
    assert not ({"accuracy", "macro_f1", "truth", "label_rain"} & set(online.columns))
    assert events.groupby("batch").size().tolist() == [2] * 21
    assert (events["delay"] == 0).all()
    assert verified.batch.tolist() == online.batch.tolist()
    lock = json.loads((RUN / "protocol_lock.json").read_text(encoding="utf-8"))
    assert lock["state"] == "complete"
    integrity = json.loads((RUN / "integrity_manifest.json").read_text(encoding="utf-8"))
    assert "online/batch_scores.csv" in integrity["files_sha256"]


def test_noaa_weather_report_receipt_exists_and_has_four_pages():
    receipt = json.loads((ROOT / "reports/phase3/weather_noaa_1949_1999_v2/report_build_receipt.json")
                         .read_text(encoding="utf-8"))
    assert receipt["report_status"].startswith("generated from complete")
    assert (ROOT / "reports/phase3/weather_noaa_1949_1999_v2/REPORT.pdf").is_file()

