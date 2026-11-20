"""Package the audit-policy matrix as a verified additive overlay on v20."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/phase3/IDM_Viz_Phase3_PostDeadline_Addendum_20261117_v20.zip"
PREVIOUS = ROOT / "output/phase3/IDM_Viz_Phase3_Matched_Adaptation_Delta_20261117_v3.zip"
OUT = ROOT / "output/phase3/IDM_Viz_Phase3_Audit_Policy_Delta_20261117_v4.zip"
RECEIPT = OUT.with_suffix(".receipt.json")
RUN = ROOT / "results/phase3/audit_policy_matrix_20261117_v1"
REPORT = ROOT / "reports/phase3/audit_policy_matrix_20261117_v3"
FILES = [
    "scripts/run_audit_policy_matrix.py", "scripts/report_audit_policy_matrix.py",
    "scripts/update_audit_policy_matrix_registry.py", "scripts/package_phase3_audit_policy_delta.py",
    "tests/phase3/test_audit_policy_matrix_contract.py",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json",
]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def main() -> None:
    if not BASE.is_file() or not PREVIOUS.is_file():
        raise FileNotFoundError("base v20 or previous matched-adaptation delta is missing")
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not RUN.is_dir() or not REPORT.is_dir():
        raise FileNotFoundError("audit matrix run or report directory missing")
    overlays: dict[str, bytes] = {}
    with zipfile.ZipFile(PREVIOUS) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("previous delta CRC validation failed")
        for name in archive.namelist():
            if name in {"MATCHED_ADAPTATION_DELTA_MANIFEST.json", "MATCHED_ADAPTATION_DELTA_README.md"}:
                continue
            if name == "output/phase3/IDM_Viz_Phase3_Addendum_Package_Status_20261117.json":
                continue
            overlays[name] = archive.read(name)
    for directory in (RUN, REPORT):
        for path in directory.rglob("*"):
            if path.is_file() and "rendered_page-" not in path.name:
                overlays[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for rel in FILES:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        overlays[rel] = path.read_bytes()
    # The external status file is updated after this ZIP is sealed, avoiding a
    # self-referential archive hash. The adjacent receipt records its digest.
    manifest = {
        "delta_id": OUT.stem.lower().replace("_", "-"),
        "base_package": BASE.name,
        "base_package_sha256": sha_file(BASE),
        "previous_delta": PREVIOUS.name,
        "previous_delta_sha256": sha_file(PREVIOUS),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": sha_bytes(data), "bytes": len(data)}
                  for name, data in sorted(overlays.items())},
        "file_count": len(overlays),
    }
    readme = """# IDM-Viz Phase 3 audit-policy delta v4

Apply this additive overlay after extracting the v20 base addendum. It also carries forward the matched-adaptation v3 overlay. It adds a complete declared delayed-audit policy matrix and its editable report, figures, data, tests, and updated Phase 3 audit registry.

The matrix evaluates six policies, strict-cap and exact-quota modes, budgets 2/4/8 cohorts, delays 0/5/20/100, seven synthetic conditions and 30 locked seeds. It records 30,240 policy-seed cells and 725,760 decisions. Labels first affect a decision only when source batch + delay < current clock. The replay uses cached predictions and is outcome-informed for scoring; it is not fresh independent or prospective confirmation. Delay 100 exceeds the 24-batch stream horizon. Exact-quota force-fills and unused strict-cap budgets are reported separately.

The report PDF is `reports/phase3/audit_policy_matrix_20261117_v3/audit_policy_matrix.pdf`; the editable source and vector/raster plots are adjacent. Raw per-seed data, every decision, summaries, paired bootstrap intervals and a protocol lock are under `results/phase3/audit_policy_matrix_20261117_v1/`.

The submitted Phase 2 DOCX and prior submission snapshots are preserved. This is a post-deadline, versioned improvement.
"""
    temp = OUT.with_name("." + OUT.stem + ".tmp.zip")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(overlays.items()):
            archive.writestr(name, data)
        archive.writestr("AUDIT_POLICY_DELTA_README.md", readme)
        archive.writestr("AUDIT_POLICY_DELTA_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    temp.replace(OUT)
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("delta ZIP CRC validation failed")
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or sha_bytes(data) != record["sha256"]:
                raise RuntimeError(f"overlay integrity failed: {name}")
    receipt = {"status": "validated", "package": OUT.name, "sha256": sha_file(OUT),
               "bytes": OUT.stat().st_size, "base_package": BASE.name,
               "base_package_sha256": manifest["base_package_sha256"],
               "previous_delta": PREVIOUS.name, "previous_delta_sha256": manifest["previous_delta_sha256"],
               "overlay_file_count": len(overlays), "zip_crc": "passed", "overlay_hashes": "passed"}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
