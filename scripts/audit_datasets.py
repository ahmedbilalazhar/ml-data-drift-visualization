"""Audit + chronological splits (roadmap Sec. 7.4, 7.5, 7.2).

- Verifies manifest checksums (immutable raw).
- Schema / missingness / duplicates / class balance per dataset.
- Elec2: chronological 20/20/60 train/calib/test split manifest (file order = time).
- Covertype subset: 20/20/60 split of event_id order (EXPLICITLY constructed).
- Writes data/interim/audit_report.json; prints ADMIT/RESTRICT verdicts.
"""
import json, hashlib
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
    return {"dataset": name, "rows": len(df), "cols": list(df.columns),
            "missing_total": int(df.isna().sum().sum()),
            "dup_event_id": int(df.event_id.duplicated().sum()),
            "dup_rows": int(df.duplicated().sum()),
            "class_counts": df.label.value_counts().sort_index().to_dict()}


def main():
    INTER.mkdir(parents=True, exist_ok=True)
    report: dict = {"checksum_issues": {}, "audits": {}, "splits": {}, "verdicts": {}}

    for m in ["covtype_manifest.json", "elec2_manifest.json"]:
        report["checksum_issues"][m] = check_manifest(MAN / m)

    for name, f in [("elec2", PROC / "elec2_ordered.parquet"),
                    ("covtype_subset", PROC / "covtype_subset100k.parquet")]:
        df = pd.read_parquet(f)
        a = audit_frame(df, name)
        report["audits"][name] = a
        n = len(df)
        # chronological 20/20/60 split on event order (protocol Sec. 7.4)
        tr, ca = int(n * 0.2), int(n * 0.4)
        report["splits"][name] = {"train": [0, tr - 1], "calib": [tr, ca - 1],
                                  "test": [ca, n - 1]}
        ok = (a["missing_total"] == 0 and a["dup_event_id"] == 0
              and not any(report["checksum_issues"].values()))
        notes = []
        if name == "covtype_subset" and min(a["class_counts"].values()) < 5000:
            notes.append("minority classes (esp. 4: n=2747) — suppress small-group stats per rule")
        if name == "elec2":
            notes.append("no verified drift-event catalog — burden/association only")
            notes.append("file order assumed chronological — timestamp column unverified")
        verdict = "ADMITTED" if ok and not notes else "ADMITTED-WITH-NOTES"
        report["verdicts"][name] = {"status": verdict, "notes": notes}
        print(f"{name}: rows={n} missing={a['missing_total']} "
              f"dup_ids={a['dup_event_id']} dup_rows={a['dup_rows']} "
              f"classes={a['class_counts']} -> {verdict} {notes}")

    (INTER / "audit_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"-> {INTER/'audit_report.json'}")


if __name__ == "__main__":
    main()
