"""Real-data loader: verified Elec2, Covertype subset and NOAA Weather.

Single entry point so scripts stop re-implementing splits. Contract:
- event_id ascending = replay order (Elec2 chronological; Covertype explicitly
  constructed; NOAA Weather chronological same-day summaries).
- returns frame + 20/20/60 chronological splits + numeric cols + manifest meta.
- Weather is a CSV same-day NOAA summary stream, not a forecast benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
MAN = ROOT / "data" / "manifests"

FILES = {
    "elec2": ("data/processed/verified_v2/elec2_ordered.parquet",
              "data/manifests/elec2_verified_v2_manifest.json"),
    "covtype_subset": ("data/processed/verified_v2/covtype_subset100k.parquet",
                       "data/manifests/covtype_subset_verified_v2_manifest.json"),
    "weather": ("data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv",
                "data/processed/weather_noaa_1949_1999_v2/weather_manifest.json"),
}


class PendingAcquisition(RuntimeError):
    pass


def load_real(name: str, drop_like: tuple = ("date",)) -> dict:
    if name not in FILES:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(FILES)}")
    pq, mf = FILES[name]
    p, m = ROOT / pq, ROOT / mf
    if not p.exists():
        raise PendingAcquisition(
            f"{name}: {pq} not materialized. "
            f"{'See configs/datasets/weather.yaml and scripts/prepare_noaa_weather.py.' if name == 'weather' else 'Run scripts/rebuild_processed_datasets_v2.py after restoring verified raw inputs.'}")
    df = (pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p))
    if "event_id" not in df and "row_id" in df:
        df = df.rename(columns={"row_id": "event_id", "label_rain": "label"})
    df = df.sort_values("event_id").reset_index(drop=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].cat.codes.astype(float)
    feats = [c for c in df.columns if c not in ("event_id", "label", "timestamp")
             and not any(d in c.lower() for d in drop_like)]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    n = len(df)
    if name == "weather":
        train_end, calibration_end, reference_end = int(n * .2), int(n * .3), int(n * .4)
        splits = {"train": [0, train_end - 1], "calib": [train_end, calibration_end - 1],
                  "reference": [calibration_end, reference_end - 1],
                  "test": [reference_end, n - 1]}
    else:
        splits = {"train": [0, int(n * 0.2) - 1],
                  "calib": [int(n * 0.2), int(n * 0.4) - 1],
                  "test": [int(n * 0.4), n - 1]}
    meta = json.loads(m.read_text()) if m.exists() else {"dataset": name,
                                                         "manifest": "missing"}
    return {"frame": df, "numeric": num, "splits": splits, "meta": meta,
            "n": n}
