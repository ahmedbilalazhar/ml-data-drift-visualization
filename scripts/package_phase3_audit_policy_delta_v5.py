"""Package the paired-contrast report as a verified v5 overlay on the v20 base."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/phase3/IDM_Viz_Phase3_PostDeadline_Addendum_20261117_v20.zip"
PREVIOUS = ROOT / "output/phase3/IDM_Viz_Phase3_Audit_Policy_Delta_20261117_v4.zip"
OUT = ROOT / "output/phase3/IDM_Viz_Phase3_Audit_Policy_Delta_20261117_v5.zip"
RECEIPT = OUT.with_suffix(".receipt.json")
REPORT = ROOT / "reports/phase3/audit_policy_matrix_20261117_v7"
FILES = [
    "scripts/run_audit_policy_matrix.py", "scripts/report_audit_policy_matrix.py",
    "scripts/update_audit_policy_matrix_registry.py", "scripts/package_phase3_audit_policy_delta.py",
    "scripts/package_phase3_audit_policy_delta_v5.py", "scripts/update_phase3_addendum_package_status.py",
    "tests/phase3/test_audit_policy_matrix_contract.py",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not BASE.is_file() or not PREVIOUS.is_file():
        raise FileNotFoundError("v20 base or v4 audit-policy delta missing")
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    overlays: dict[str, bytes] = {}
    with zipfile.ZipFile(PREVIOUS) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("v4 CRC check failed")
        for name in archive.namelist():
            if name in {"AUDIT_POLICY_DELTA_MANIFEST.json", "AUDIT_POLICY_DELTA_README.md"}:
                continue
            if name == "output/phase3/IDM_Viz_Phase3_Addendum_Package_Status_20261117.json":
                continue
            overlays[name] = archive.read(name)
    for path in REPORT.rglob("*"):
        if path.is_file() and "rendered_page-" not in path.name:
            overlays[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for rel in FILES:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        overlays[rel] = path.read_bytes()
    manifest = {
        "delta_id": OUT.stem.lower().replace("_", "-"), "base_package": BASE.name,
        "base_package_sha256": digest(BASE.read_bytes()), "previous_delta": PREVIOUS.name,
        "previous_delta_sha256": digest(PREVIOUS.read_bytes()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": digest(data), "bytes": len(data)}
                  for name, data in sorted(overlays.items())}, "file_count": len(overlays),
    }
    readme = """# IDM-Viz Phase 3 audit-policy delta v5

Apply this additive overlay after extracting the v20 base and the matched-adaptation v3 and audit-policy v4 deltas. It carries their contents forward and updates the audit-policy report to include seed-level paired contrasts against uniform, with each seed averaged over matched conditions and budgets before bootstrap resampling. The report also discloses unequal strict-cap acquisition volume, so episode-count differences are not misrepresented as equal-spend rankings.

The six-policy matrix contains 30,240 policy-seed cells and 725,760 sequential decision records across strict-cap/exact-quota modes, budgets 2/4/8, delays 0/5/20/100, seven synthetic conditions and 30 seeds. Delayed accuracy enters the released-error signal only after `source_batch + delay < decision_clock`. This is outcome-informed cached synthetic replay, not fresh or prospective confirmation; delay 100 exceeds the 24-batch stream horizon.

The final editable TeX and two-page PDF are at `reports/phase3/audit_policy_matrix_20261117_v7/`. Its report receipt hashes the PDF and data summaries. The submitted Phase 2 document and earlier snapshots remain unchanged.
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
            raise RuntimeError("v5 ZIP CRC validation failed")
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or digest(data) != record["sha256"]:
                raise RuntimeError(f"v5 overlay hash mismatch: {name}")
    receipt = {"status": "validated", "package": OUT.name, "sha256": digest(OUT.read_bytes()),
               "bytes": OUT.stat().st_size, "base_package": BASE.name,
               "base_package_sha256": manifest["base_package_sha256"],
               "previous_delta": PREVIOUS.name, "previous_delta_sha256": manifest["previous_delta_sha256"],
               "overlay_file_count": len(overlays), "zip_crc": "passed", "overlay_hashes": "passed"}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
