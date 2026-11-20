"""Check the locked v3 evaluation seeds against prior textual manifests/logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3_strengthening_v3.json"
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".py", ".yaml", ".yml", ".toml"}
ZIP_METADATA_TOKENS = ("manifest", "protocol_lock", "metadata", "seed", "validation", "readme", "protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/phase3/SEED_COLLISION_AUDIT_20261116.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = sorted(map(int, config["seed_banks"]["evaluation"]))
    pattern = re.compile(rb"(?<![0-9])(?:" + b"|".join(str(s).encode() for s in seeds) + rb")(?![0-9])")
    scanned_files = 0
    scanned_zip_metadata = 0
    matches = []
    skip_parts = {".git", "__pycache__", "strengthening_v3_evaluation_d8", "strengthening_v3_smoke_m1_retry",
                  "strengthening_v3_smoke_m6", "strengthening_v3_development_d8"}
    roots = [ROOT / "configs", ROOT / "docs/phase3", ROOT / "scripts", ROOT / "src",
             ROOT / "results/phase3", ROOT / "output/phase3"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or any(part in skip_parts for part in path.parts):
                continue
            if path.resolve() == config_path or path.suffix.lower() in {".npz", ".png", ".pdf", ".parquet", ".arff", ".bin"}:
                continue
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as zf:
                        for info in zf.infolist():
                            name = info.filename.lower()
                            if (info.file_size > 10_000_000 or Path(name).suffix not in TEXT_SUFFIXES
                                    or not any(token in name for token in ZIP_METADATA_TOKENS)):
                                continue
                            scanned_zip_metadata += 1
                            data = zf.read(info)
                            hit = pattern.search(data)
                            if hit:
                                matches.append({"path": str(path.relative_to(ROOT)),
                                                "archive_member": info.filename,
                                                "matched_seed": int(hit.group().decode())})
                except (zipfile.BadZipFile, OSError):
                    continue
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 50_000_000:
                continue
            scanned_files += 1
            try:
                data = path.read_bytes()
            except OSError:
                continue
            hit = pattern.search(data)
            if hit:
                matches.append({"path": str(path.relative_to(ROOT)),
                                "matched_seed": int(hit.group().decode())})
    record = {
        "audit_id": "strengthening-v3-evaluation-seed-collision-audit-2026-11-16",
        "evaluation_seed_count": len(seeds), "evaluation_seeds": seeds,
        "scanned_text_files": scanned_files, "scanned_archive_metadata_members": scanned_zip_metadata,
        "excluded": ["generated v3 run outputs containing their own reserved evaluation IDs",
                     "numeric arrays and binary data files", "test code that cites seeds as negative collision cases"],
        "matches_outside_reserved_config": matches,
        "result": "no exact evaluation-seed collisions found" if not matches else "collision review required"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("evaluation_seed_count", "scanned_text_files",
                                              "scanned_archive_metadata_members", "matches_outside_reserved_config", "result")},
                     indent=2))
    if matches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
