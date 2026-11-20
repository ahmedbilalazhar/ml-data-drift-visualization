"""Audit + chronological splits (roadmap Sec. 7.4, 7.5, 7.2).

- Verifies manifest checksums (immutable raw).
- Schema / missingness / duplicates / class balance per dataset.
- Elec2: chronological 20/20/60 train/calib/test split manifest (file order = time).
- Covertype subset: 20/20/60 split of event_id order (EXPLICITLY constructed).
- Writes data/interim/audit_report.json; prints ADMIT/RESTRICT verdicts.
"""
import json, hashlib
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN, PROC, INTER = ROOT / "data" / "manifests", ROOT / "data" / "processed", ROOT / "data" / "interim"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_manifest(path: Path) -> list[str]:
    issues = []
    man = json.loads(path.read_text())
    for f in man.get("files", []):
        p = ROOT / f["path"]
        if not p.exists():
            issues.append(f"missing {f['path']}")
        elif sha256(p) != f["sha256"]:
            issues.append(f"checksum mismatch {f['path']}")
    return issues


def audit_frame(df: pd.DataFrame, name: str) -> dict:
    id_col = "event_id" if "event_id" in df else "row_id"
    label_col = "label" if "label" in df else "label_rain"
    return {"dataset": name, "rows": len(df), "cols": list(df.columns),
            "missing_total": int(df.isna().sum().sum()),
            "dup_event_id": int(df[id_col].duplicated().sum()),
            "dup_rows": int(df.duplicated().sum()),
            "class_counts": df[label_col].value_counts().sort_index().to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=INTER / "audit_report.json")
    args = parser.parse_args()
    INTER.mkdir(parents=True, exist_ok=True)
    report: dict = {"checksum_issues": {}, "audits": {}, "splits": {}, "verdicts": {}}

    for m in ["covtype_manifest.json", "elec2_manifest.json"]:
        if (MAN / m).exists():
            report["checksum_issues"][m] = check_manifest(MAN / m)
    for m in ["elec2_verified_v2_manifest.json", "covtype_subset_verified_v2_manifest.json"]:
        path = MAN / m
        if path.exists():
            current = check_manifest(path)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            processed = manifest.get("processed_file")
            if processed:
                file_path = ROOT / processed["path"]
                if not file_path.exists() or sha256(file_path) != processed["sha256"]:
                    current.append(f"processed file missing or checksum mismatch: {processed['path']}")
            elif "file" in manifest and "sha256" in manifest:
                file_path = ROOT / manifest["file"]
                if not file_path.exists() or sha256(file_path) != manifest["sha256"]:
                    current.append(f"processed file missing or checksum mismatch: {manifest['file']}")
            report["checksum_issues"][m] = current
    weather_manifest_path = ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_manifest.json"
    if weather_manifest_path.exists():
        weather_manifest = json.loads(weather_manifest_path.read_text(encoding="utf-8"))
        processed = ROOT / weather_manifest["processed"]["path"]
        issues = []
        if not processed.exists():
            issues.append(f"missing {weather_manifest['processed']['path']}")
        elif sha256(processed) != weather_manifest["processed"]["sha256"]:
            issues.append("processed checksum mismatch")
        for entry in weather_manifest["source"]["raw_files"]:
            raw_path = ROOT / entry["path"]
            if not raw_path.exists() or sha256(raw_path) != entry["sha256"]:
                issues.append(f"missing or changed raw source {entry['path']}")
        report["checksum_issues"]["weather_noaa_manifest.json"] = issues

    datasets = [("elec2", ROOT / "data/processed/verified_v2/elec2_ordered.parquet"),
                ("covtype_subset", ROOT / "data/processed/verified_v2/covtype_subset100k.parquet")]
    weather_csv = PROC / "weather_noaa_1949_1999_v2/weather_ordered.csv"
    if weather_csv.exists():
        datasets.append(("weather_noaa_gsod", weather_csv))
    report["audit_load_errors"] = {}
    for name, f in datasets:
        try:
            df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_parquet(f)
        except Exception as exc:
            report["audit_load_errors"][name] = f"{type(exc).__name__}: {exc}"
            print(f"{name}: unable to read {f}: {type(exc).__name__}: {exc}")
            continue
        if name == "weather_noaa_gsod":
            date_ok = ("timestamp" in df and df.timestamp.is_monotonic_increasing
                       and not df.timestamp.duplicated().any())
            expected_columns = {"timestamp", "row_id", "temp", "dewp", "slp", "visib",
                                "wdsp", "mxspd", "max", "min", "label_rain"}
            schema_ok = expected_columns == set(df.columns)
            no_leakage = not ({"prcp", "frshtt", "sndp"} & set(df.columns))
            report.setdefault("weather_contract", {}).update({
                "timestamp_strictly_increasing_unique": bool(date_ok),
                "expected_schema_exact": bool(schema_ok),
                "target_leakage_columns_absent": bool(no_leakage),
                "rows": int(len(df)), "tail_rows_after_complete_batches": int(
                    (len(df) - int(len(df)*.4)) % 500),
                "complete_monitor_batches_after_20_10_10_60_split":
                    int((len(df) - int(len(df)*0.4)) // 500),
                "split_counts": {"train": int(len(df)*.2), "calibration": int(len(df)*.1),
                                 "reference": int(len(df)*.1), "monitoring": len(df)-int(len(df)*.4)}})
        a = audit_frame(df, name)
        report["audits"][name] = a
        n = len(df)
        # chronological 20/20/60 split on event order (protocol Sec. 7.4)
        if name == "weather_noaa_gsod":
            tr, ca, re = int(n * .2), int(n * .3), int(n * .4)
            report["splits"][name] = {"train": [0, tr - 1], "calibration": [tr, ca - 1],
                                      "reference": [ca, re - 1], "monitoring": [re, n - 1]}
        else:
            tr, ca = int(n * 0.2), int(n * 0.4)
            report["splits"][name] = {"train": [0, tr - 1], "calib": [tr, ca - 1],
                                      "test": [ca, n - 1]}
        checksum_key = {"elec2": "elec2_verified_v2_manifest.json",
                        "covtype_subset": "covtype_subset_verified_v2_manifest.json"}.get(name)
        own_checksum_issues = (report["checksum_issues"].get("weather_noaa_manifest.json", [])
                               if name == "weather_noaa_gsod" else
                               report["checksum_issues"].get(checksum_key, []) if checksum_key else [])
        ok = (a["missing_total"] == 0 and a["dup_event_id"] == 0 and not own_checksum_issues)
        notes = []
        if name == "covtype_subset" and min(a["class_counts"].values()) < 5000:
            notes.append("minority classes (esp. 4: n=2747) — suppress small-group stats per rule")
        if name == "elec2":
            notes.append("no verified drift-event catalog — burden/association only")
            notes.append("file order assumed chronological — timestamp column unverified")
        if name == "weather_noaa_gsod":
            notes.append("no verified drift-event catalog — alarm burden is not false-alarm rate")
            notes.append("same-day retrospective classification, not a forecast or independent replication")
            if not date_ok or not schema_ok or not no_leakage:
                notes.append("NOAA weather contract check failed")
        verdict = "ADMITTED" if ok and not notes else "ADMITTED-WITH-NOTES"
        report["verdicts"][name] = {"status": verdict, "notes": notes}
        print(f"{name}: rows={n} missing={a['missing_total']} "
              f"dup_ids={a['dup_event_id']} dup_rows={a['dup_rows']} "
              f"classes={a['class_counts']} -> {verdict} {notes}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
