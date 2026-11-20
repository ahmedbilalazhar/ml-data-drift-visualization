"""Validate and preprocess the official NOAA GSOD Offutt AFB weather stream.

The target is NOAA's same-day rain/drizzle occurrence flag. Precipitation and
all weather-event flags are excluded from predictors to avoid target leakage.
This creates a retrospective same-day classification stream, not a forecast.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATION = "72554014949"
START_YEAR, END_YEAR = 1949, 1999
FEATURES = ("TEMP", "DEWP", "SLP", "VISIB", "WDSP", "MXSPD", "MAX", "MIN")
FRSHTT_FIELDS = ("fog", "rain_or_drizzle", "snow_or_ice_pellets", "hail", "thunder", "tornado_or_funnel_cloud")
SOURCE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/72554014949.csv"
METADATA_URL = "https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00516/html"
BENCHMARK_MIRROR = "https://github.com/scikit-multiflow/streaming-datasets/blob/master/weather.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        frame.to_csv(temp, index=False, lineterminator="\n", float_format="%.6g")
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def preprocess(raw_dir: Path, out_dir: Path, reference: Path | None = None) -> dict:
    expected = {f"{STATION}-{year}.csv" for year in range(START_YEAR, END_YEAR + 1)}
    found = {p.name for p in raw_dir.glob(f"{STATION}-*.csv")}
    missing_files = sorted(expected - found)
    extra_files = sorted(found - expected)
    if missing_files or extra_files:
        raise ValueError(f"station-year inventory mismatch; missing={missing_files}, extra={extra_files}")

    frames = []
    source_files = []
    required = {"STATION", "DATE", *FEATURES, "FRSHTT"}
    for year in range(START_YEAR, END_YEAR + 1):
        path = raw_dir / f"{STATION}-{year}.csv"
        frame = pd.read_csv(path, dtype={"STATION": str}, parse_dates=["DATE"])
        if not required.issubset(frame.columns):
            raise ValueError(f"missing required fields in {path.name}: {sorted(required - set(frame.columns))}")
        if set(frame["STATION"].astype(str)) != {STATION}:
            raise ValueError(f"unexpected station ID in {path.name}")
        if not frame["DATE"].dt.year.eq(year).all():
            raise ValueError(f"unexpected dates in {path.name}")
        frame["_source_file"] = path.name
        frames.append(frame)
        source_files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                             "rows": int(len(frame))})

    raw = pd.concat(frames, ignore_index=True).sort_values("DATE", kind="mergesort").reset_index(drop=True)
    if raw["DATE"].duplicated().any() or not raw["DATE"].is_monotonic_increasing:
        raise ValueError("station dates are not strictly ordered and unique")
    if raw["DATE"].min() != pd.Timestamp(f"{START_YEAR}-01-01") or raw["DATE"].max() != pd.Timestamp(f"{END_YEAR}-12-31"):
        raise ValueError("unexpected coverage endpoints")

    features = raw.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    # GSOD missing-value codes are 999.9 for the listed continuous fields.
    # 999.9 is the documented GSOD missing code; larger SLP values such as
    # 1000.0 millibars are valid and must not be dropped.
    features = features.mask(features.eq(999.9) | ~np.isfinite(features))
    flags = raw["FRSHTT"]
    valid_flags = flags.notna()
    flag_text = flags.loc[valid_flags].map(lambda v: str(int(float(v))).zfill(6))
    flag_is_valid = flag_text.map(lambda value: len(value) == 6 and set(value) <= {"0", "1"})
    valid_flags.loc[valid_flags] = flag_is_valid.to_numpy()
    flag_text = flag_text.loc[flag_is_valid]
    # FRSHTT positions are fog, rain/drizzle, snow/ice pellets, hail, thunder, tornado.
    rain = pd.Series(pd.NA, index=raw.index, dtype="Int8")
    rain.loc[valid_flags] = flag_text.str[1].astype(int).to_numpy()
    keep = features.notna().all(axis=1) & valid_flags
    n_dropped_feature = int((~features.notna().all(axis=1)).sum())
    n_dropped_label = int((~valid_flags).sum())
    data = pd.DataFrame({"timestamp": raw.loc[keep, "DATE"].dt.strftime("%Y-%m-%d"),
                         "row_id": np.arange(int(keep.sum()), dtype=np.int64),
                         **{col.lower(): features.loc[keep, col].to_numpy(dtype=float) for col in FEATURES},
                         "label_rain": rain.loc[keep].to_numpy(dtype=np.int8)})
    if not data.timestamp.is_monotonic_increasing or data.timestamp.duplicated().any():
        raise ValueError("processed timestamps must be strictly increasing and unique")
    if not np.isfinite(data.loc[:, [c.lower() for c in FEATURES]].to_numpy()).all():
        raise ValueError("processed feature matrix contains non-finite values")
    if set(data.label_rain.unique()) != {0, 1}:
        raise ValueError("processed label must contain both classes")
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_path = out_dir / "weather_ordered.csv"
    _atomic_csv(data, processed_path)

    mirror_audit = {"available": bool(reference and reference.is_file()), "url": BENCHMARK_MIRROR}
    if reference and reference.is_file():
        old = pd.read_csv(reference)
        mirror_audit.update({"sha256": sha256(reference), "rows": int(len(old)),
                             "columns": list(old.columns),
                             "exact_feature_and_label_equivalence": False,
                             "note": "The mirror is retained only as a legacy benchmark reference. The current official GSOD release is timestamped and is authoritative for this processing run; no exact row equivalence is asserted."})

    manifest = {
        "dataset_id": "noaa-gsod-offutt-weather-1949-1999-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"dataset": "NOAA Global Summary of the Day (GSOD), NCEI DSI 3505 / gov.noaa.ncdc:C00516",
                   "metadata_url": METADATA_URL, "station_id": STATION,
                   "station_name": "BELLEVUE OFFUTT AFB, NE US", "coverage": ["1949-01-01", "1999-12-31"],
                   "station_year_csv_url_template": SOURCE_URL,
                   "raw_files": source_files, "raw_row_count": int(len(raw)),
                   "raw_station_dates_unique": True, "raw_station_dates_monotonic": True},
        "preprocessing": {"feature_columns": list(FEATURES),
                          "feature_units": {"TEMP": "degrees Fahrenheit", "DEWP": "degrees Fahrenheit",
                                            "SLP": "millibars", "VISIB": "miles", "WDSP": "knots",
                                            "MXSPD": "knots", "MAX": "degrees Fahrenheit", "MIN": "degrees Fahrenheit"},
                          "missing_code": "999.9 or non-finite values in selected numeric predictors; corresponding rows dropped",
                          "target": "second FRSHTT digit (rain or drizzle occurrence on the same GSOD summary date)",
                          "target_source_field": "FRSHTT", "target_flag_fields": list(FRSHTT_FIELDS),
                          "excluded_predictors": ["PRCP", "SNDP", "FRSHTT", "MAX_ATTRIBUTES", "MIN_ATTRIBUTES"],
                          "ordering": "ascending NOAA DATE, one station, unique days",
                          "dropped_for_predictor_missingness": n_dropped_feature,
                          "dropped_for_invalid_label_flags": n_dropped_label,
                          "row_id": "zero-based stable ID assigned after timestamp sorting and row filtering",
                          "interpretation": "same-day retrospective classification benchmark; not a forward rain forecast"},
        "processed": {"path": processed_path.relative_to(ROOT).as_posix(), "sha256": sha256(processed_path),
                      "row_count": int(len(data)), "positive_count": int(data.label_rain.sum()),
                      "positive_rate": float(data.label_rain.mean()),
                      "timestamp_start": data.timestamp.iloc[0], "timestamp_end": data.timestamp.iloc[-1],
                      "feature_columns": [c.lower() for c in FEATURES]},
        "legacy_benchmark_comparison": mirror_audit,
        "license_and_citation": "NOAA/NCEI GSOD public data; cite NOAA National Centers of Environmental Information, Global Surface Summary of the Day (GSOD), dataset identifier gov.noaa.ncdc:C00516; accessed 2026-11-17.",
        "claims_not_supported": ["causal drift attribution", "future rainfall forecast", "independent dataset replication from temporal blocks", "general transfer"],
    }
    manifest_path = out_dir / "weather_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/external/noaa_gsod_weather_1949_1999_v1/raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/weather_noaa_1949_1999_v1")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/external/noaa_gsod_weather_1949_1999_v1/reference/scikit_multiflow_weather.csv")
    args = parser.parse_args()
    result = preprocess(args.raw_dir.resolve(), args.out_dir.resolve(), args.reference.resolve())
    print(json.dumps({"dataset_id": result["dataset_id"], "rows": result["processed"]["row_count"],
                      "positive_rate": result["processed"]["positive_rate"],
                      "sha256": result["processed"]["sha256"], "manifest": str(args.out_dir.resolve() / "weather_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
