"""Package the ADWIN baseline and updated registry as a verified v6 overlay."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/phase3/IDM_Viz_Phase3_PostDeadline_Addendum_20261117_v20.zip"
PREVIOUS = ROOT / "output/phase3/IDM_Viz_Phase3_Audit_Policy_Delta_20261117_v5.zip"
OUT = ROOT / "output/phase3/IDM_Viz_Phase3_ADWIN_Baseline_Delta_20261117_v6.zip"
RECEIPT = OUT.with_suffix(".receipt.json")
RUN = ROOT / "results/phase3/adwin_audit_baseline_20261117_v1"
REPORT = ROOT / "reports/phase3/adwin_audit_baseline_20261117_v3"
FILES = [
    "pyproject.toml", "scripts/run_adwin_audit_baseline.py",
    "scripts/report_adwin_audit_baseline.py", "scripts/update_audit_policy_matrix_registry.py",
    "scripts/update_phase3_addendum_package_status.py", "scripts/package_phase3_adwin_delta_v6.py",
    "tests/phase3/test_adwin_audit_baseline_contract.py",
    "docs/phase3/BASELINE_FIDELITY_AUDIT_20261117.md",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md",
    "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not BASE.is_file() or not PREVIOUS.is_file() or not RUN.is_dir() or not REPORT.is_dir():
        raise FileNotFoundError("Required v20 base, v5 delta, ADWIN run or report directory is missing")
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    overlays: dict[str, bytes] = {}
    with zipfile.ZipFile(PREVIOUS) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("previous delta CRC check failed")
        for name in archive.namelist():
            if name in {"AUDIT_POLICY_DELTA_MANIFEST.json", "AUDIT_POLICY_DELTA_README.md"}:
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
    manifest = {
        "delta_id": OUT.stem.lower().replace("_", "-"), "base_package": BASE.name,
        "base_package_sha256": digest(BASE.read_bytes()), "previous_delta": PREVIOUS.name,
        "previous_delta_sha256": digest(PREVIOUS.read_bytes()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": digest(data), "bytes": len(data)}
                  for name, data in sorted(overlays.items())}, "file_count": len(overlays),
    }
    readme = """# IDM-Viz Phase 3 ADWIN baseline delta v6

This is a versioned post-deadline addendum overlay on base package v20. It carries forward the previous matched-adaptation and audit-policy improvements and adds a canonical River ADWIN baseline under the same cached predictions, sampled 500-row audit cohorts, delays, budgets, 30 seed bank, seven synthetic conditions, and strict-cap/exact-quota rules.

ADWIN consumes row-level binary errors only from selected cohorts after `source_batch + delay < decision_clock`. It uses River 0.27.0, delta 0.002, clock 1, and the documented default bucket/window/grace settings. In 5,040 policy cells it fired at least once in 212, all exact-quota cells; it did not fire under strict caps after the single initial probe. This exposes a sparse-feedback deadlock for this policy. It does not establish general ADWIN weakness or superiority of IDM-Viz.

The two-page editable report and compiled PDF are in `reports/phase3/adwin_audit_baseline_20261117_v3/`. Its receipt includes the PDF hash. The input Phase 2 document and historical submission archives are preserved.
"""
    temp = OUT.with_name("." + OUT.stem + ".tmp.zip")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(overlays.items()):
            archive.writestr(name, data)
        archive.writestr("ADWIN_DELTA_README.md", readme)
        archive.writestr("ADWIN_DELTA_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    temp.replace(OUT)
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("v6 ZIP CRC validation failed")
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or digest(data) != record["sha256"]:
                raise RuntimeError(f"v6 overlay hash mismatch: {name}")
    receipt = {"status": "validated", "package": OUT.name, "sha256": digest(OUT.read_bytes()),
               "bytes": OUT.stat().st_size, "base_package": BASE.name,
               "base_package_sha256": manifest["base_package_sha256"],
               "previous_delta": PREVIOUS.name, "previous_delta_sha256": manifest["previous_delta_sha256"],
               "overlay_file_count": len(overlays), "zip_crc": "passed", "overlay_hashes": "passed"}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
