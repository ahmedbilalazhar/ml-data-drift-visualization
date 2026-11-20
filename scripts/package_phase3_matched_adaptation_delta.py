"""Package the matched adaptation extension as a verifiable overlay on v20."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/phase3/IDM_Viz_Phase3_PostDeadline_Addendum_20261117_v20.zip"
OUT = ROOT / "output/phase3/IDM_Viz_Phase3_Matched_Adaptation_Delta_20261117_v3.zip"
RECEIPT = OUT.with_suffix(".receipt.json")
RUN = ROOT / "results/phase3/matched_adaptation_schedule_20261117_v2"
REPORT = ROOT / "reports/phase3/matched_adaptation_schedule_20261117_v2"
ROBUSTNESS_REPORT = ROOT / "reports/phase3/large_seed_robustness_final_reprowording_v1"
FILES = [
    "scripts/run_matched_adaptation_schedule.py",
    "scripts/report_matched_adaptation_schedule.py",
    "scripts/report_phase3_large_seed.py",
    "scripts/package_phase3_matched_adaptation_delta.py",
    "tests/phase3/test_matched_adaptation_schedule_contract.py",
    "tests/phase3/test_robustness_report_reproduction_wording.py",
    "reports/phase3/overnight_robustness_20261115/sample_reproduction_tolerant/sample_reproduction_comparison.json",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json",
    "output/phase3/IDM_Viz_Phase3_Addendum_Package_Status_20261117.json",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    selected: dict[str, Path] = {p: ROOT / p for p in FILES}
    for directory in (RUN, REPORT, ROBUSTNESS_REPORT):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if path.is_file():
                selected[path.relative_to(ROOT).as_posix()] = path
    missing = [name for name, path in selected.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required delta files missing: {missing}")

    readme = """# Matched adaptation schedule delta

This overlay supplements `IDM_Viz_Phase3_PostDeadline_Addendum_20261117_v20.zip`; it is not a replacement for that archive. Extract the delta over the extracted v20 package to add the matched-clock, matched-feedback-row global versus cluster-local adaptation study and the corrected large-seed robustness report reproduction wording.

The study contains 30 paired seeds, stationary and abrupt conditional-reversal streams, and label delays 0/5/20. Global and local arms share the same latest released row-ID windows at clocks 25, 49, 73 and 89. Each opportunity allows 2,000 fit rows; repeated uses count toward 8,000 total adaptation-row uses. Initial training is reported separately. Labels obey `source_batch + delay < forecast_batch`.

On the tested reversal generator both refit policies improve mean accuracy over frozen by about 22 percentage points. Global refitting is higher than local refitting by 0.08, 0.11 and 0.10 percentage points at delays 0/5/20; paired 95% seed-bootstrap intervals exclude zero. Under stationarity, both refit policies slightly reduce accuracy versus frozen. The conclusions are limited to this task, model, partition, delay set and schedule.

Reproduce from the workspace root:

```powershell
python scripts/run_matched_adaptation_schedule.py
python scripts/report_matched_adaptation_schedule.py --run results/phase3/matched_adaptation_schedule_20261117_v1 --out reports/phase3/matched_adaptation_schedule_20261117_v1
tectonic -X compile reports/phase3/matched_adaptation_schedule_20261117_v1/matched_adaptation_schedule.tex --outdir reports/phase3/matched_adaptation_schedule_20261117_v1
python -m pytest -p no:cacheprovider tests/phase3/test_matched_adaptation_schedule_contract.py -q
```

The robustness report distinguishes exact discrete-field comparison from floating-point allclose tolerance (rtol=1e-12, atol=1e-14; observed maximum absolute difference 1.11e-16). The base v20 archive, Phase 2 submission, implementation brief and original submission archive are preserved unchanged. The Phase 2 DOCX SHA-256 recorded in the audit is `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4`.
"""
    manifest = {
        "delta_id": OUT.stem.lower().replace("_", "-"),
        "base_package": BASE.name,
        "base_package_sha256": sha(BASE),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                  for name, path in sorted(selected.items())},
        "file_count": len(selected),
    }
    temp = OUT.with_name("." + OUT.stem + ".tmp.zip")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(selected.items()):
            archive.write(path, name)
        archive.writestr("MATCHED_ADAPTATION_DELTA_README.md", readme)
        archive.writestr("MATCHED_ADAPTATION_DELTA_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    temp.replace(OUT)
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("delta ZIP CRC validation failed")
        names = set(archive.namelist())
        if not set(selected).issubset(names):
            raise RuntimeError("delta ZIP is missing selected files")
        embedded = json.loads(archive.read("MATCHED_ADAPTATION_DELTA_MANIFEST.json"))
        for name, record in embedded["files"].items():
            payload = archive.read(name)
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise RuntimeError(f"delta overlay hash mismatch: {name}")
    receipt = {
        "status": "validated", "package": OUT.name, "sha256": sha(OUT),
        "bytes": OUT.stat().st_size, "base_package": BASE.name,
        "base_package_sha256": manifest["base_package_sha256"],
        "overlay_file_count": len(selected), "zip_crc": "passed", "overlay_hashes": "passed",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
