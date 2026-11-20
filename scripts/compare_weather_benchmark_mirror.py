"""Compare the legacy un-timestamped weather benchmark mirror to NOAA GSOD."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("TEMP", "DEWP", "SLP", "VISIB", "WDSP", "MXSPD", "MAX", "MIN")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference: Path, raw_dir: Path, processed_path: Path) -> dict:
    legacy = pd.read_csv(reference)
    files = sorted(raw_dir.glob("72554014949-*.csv"))
    gsod = pd.concat([pd.read_csv(p, parse_dates=["DATE"]) for p in files], ignore_index=True)
    gsod = gsod.sort_values("DATE", kind="mergesort").reset_index(drop=True)
    lookup: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(gsod.loc[:, FEATURES].round(1).itertuples(index=False, name=None)):
        lookup[row].append(i)
    matched_indices, last = [], -1
    for row in legacy.iloc[:, :8].round(1).itertuples(index=False, name=None):
        found = next((i for i in lookup.get(row, ()) if i > last), None)
        matched_indices.append(found)
        if found is not None:
            last = found
    pairs = [(k, i) for k, i in enumerate(matched_indices) if i is not None]
    legacy_labels = legacy.iloc[[k for k, _ in pairs], 8].to_numpy(dtype=int)
    def flag(v: object) -> int:
        return int(str(int(float(v))).zfill(6)[1])
    source_labels = np.array([flag(gsod.FRSHTT.iloc[i]) for _, i in pairs], dtype=int)
    return {
        "comparison_id": "noaa-gsod-vs-legacy-weather-mirror-v1",
        "reference_path": reference.relative_to(ROOT).as_posix(),
        "reference_sha256": digest(reference),
        "NOAA_station_id": "72554014949",
        "raw_source_file_count": len(files),
        "legacy_rows": int(len(legacy)), "legacy_positive_count": int(legacy.iloc[:, 8].sum()),
        "NOAA_raw_rows": int(len(gsod)),
        "processed_NOAA_rows": int(len(pd.read_csv(processed_path))),
        "processed_NOAA_positive_count": int(pd.read_csv(processed_path).label_rain.sum()),
        "processed_NOAA_sha256": digest(processed_path),
        "monotone_exact_feature_tuple_matches": len(pairs),
        "unmatched_legacy_feature_rows": int(len(legacy) - len(pairs)),
        "rain_flag_mismatches_on_matched_rows": int(np.sum(legacy_labels != source_labels)),
        "exact_row_and_label_equivalence": False,
        "matching_method": "Greedy monotone sequence alignment on all eight listed features rounded to 0.1, then compare the legacy class to NOAA FRSHTT rain bit. This is a diagnostic, not a proof of a unique alignment.",
        "interpretation": "The timestamped NOAA station-year archive is used as the primary stream. The older generic-feature mirror is not used for timestamps or evaluator labels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/external/noaa_gsod_weather_1949_1999_v1/reference/scikit_multiflow_weather.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/external/noaa_gsod_weather_1949_1999_v1/raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/noaa_gsod_weather_1949_1999_v1/benchmark_comparison.json")
    args = parser.parse_args()
    result = compare(args.reference.resolve(), args.raw_dir.resolve(), args.processed.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
