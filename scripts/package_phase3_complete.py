"""Build a separate compact follow-up archive without changing the submitted ZIP."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "output/phase3/23I-2581_23I-2604_Phase3_Research_Extension.zip"
OUT = ROOT / "output/phase3/23I-2581_23I-2604_Phase3_Research_Complete.zip"
SUBMITTED = ROOT / "output/phase3/23I-2581_23I-2604_Phase3.zip"
SUBMITTED_SHA = "279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if sha_file(SUBMITTED) != SUBMITTED_SHA:
        raise RuntimeError("Submitted archive checksum changed; refusing to proceed")
    if not CORE.is_file():
        raise FileNotFoundError(CORE)
    with tempfile.TemporaryDirectory(prefix="phase3_complete_") as temp:
        stage = Path(temp)
        with zipfile.ZipFile(CORE) as zf:
            zf.extractall(stage)

        def copy(source: Path, target: str) -> None:
            dest = stage / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

        def text(source: Path, target: str, replacements=()) -> None:
            value = source.read_text(encoding="utf-8")
            for old, new in replacements:
                value = value.replace(old, new)
            dest = stage / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(value, encoding="utf-8", newline="\n")

        r = ROOT / "reports/phase3/adaptation_final"
        rr = ROOT / "reports/phase3/adaptation_reproduction"
        run = ROOT / "results/phase3/adaptation_final"
        repro = ROOT / "results/phase3/adaptation_reproduction"
        docs = ROOT / "docs/phase3"
        copy(r / "23I-2581_23I-2604_Phase3_Adaptation_Report.pdf", "Adaptation.pdf")
        copy(r / "23I-2581_23I-2604_Phase3_Adaptation_Report.md", "Adaptation.md")
        copy(r / "adaptation_accuracy.png", "Figures/adaptation_accuracy.png")
        copy(r / "adaptation_accuracy.svg", "Figures/adaptation_accuracy.svg")
        copy(r / "label_budgets.png", "Figures/label_budgets.png")
        copy(r / "label_budgets.svg", "Figures/label_budgets.svg")
        copy(r / "pdf_render_record.json", "Adaptation_Render.json")
        copy(r / "visual_review.json", "Adaptation_Visual.json")
        copy(run / "protocol_lock.json", "Adaptation_Protocol.json")
        copy(run / "environment.json", "Adaptation_Environment.json")
        copy(run / "validation.json", "Adaptation_Validation.json")
        copy(rr / "reproduction_comparison.json", "Adaptation_Reproduction.json")
        copy(repro / "validation.json", "Adaptation_Reproduction_Validation.json")
        text(docs / "ADAPTATION_PROTOCOL.md", "Adaptation_Protocol.md")

        for source, target in {
            "online_batches.csv": "Results/adaptation_online.csv",
            "released_labels.csv": "Results/adaptation_released_labels.csv",
            "release_log.csv": "Results/adaptation_release_log.csv",
            "evaluator_batches.csv": "Results/adaptation_evaluator.csv",
            "fit_audit.csv": "Results/adaptation_fit_audit.csv",
            "budget_by_clock.csv": "Results/adaptation_label_budgets.csv",
            "paired_seed_differences.csv": "Results/adaptation_seed_pairs.csv",
            "paired_bootstrap.csv": "Results/adaptation_bootstrap.csv",
        }.items():
            copy(run / source, target)

        src = ROOT / "src/drift_monitoring/phase3/adaptation.py"
        text(src, "Code/src/drift_monitoring/phase3/adaptation.py",
             (("from .v2 import", "from .extension import"),))
        for source, target in {
            "run_phase3_adaptation.py": "Code/scripts/run_adaptation.py",
            "evaluate_phase3_adaptation.py": "Code/scripts/evaluate_adaptation.py",
            "report_phase3_adaptation.py": "Code/scripts/report_adaptation.py",
            "compare_phase3_adaptation.py": "Code/scripts/compare_adaptation.py",
        }.items():
            text(ROOT / "scripts" / source, target)
        test = ROOT / "tests/phase3/test_adaptation_contract.py"
        text(test, "Code/tests/test_adaptation_contract.py",
             (("drift_monitoring.phase3.v2", "drift_monitoring.phase3.extension"),))

        # Include the clean reproduction receipt and exact workspace commands.
        repro_notes = (
            "# Reproduction\n\n"
            "The compact archive includes frozen outputs and the exact environment. "
            "Instance caches are omitted to keep the submission compact. From the original "
            "workspace root, use the commands in docs/phase3/REPRODUCTION.md and run "
            "scripts/reproduce_phase3_adaptation.ps1 with new output directories.\n\n"
            "The adaptation reproduction was independently run from an empty output directory. "
            "Its comparison excludes timing and run-metadata fields; all scientific outputs "
            "matched exactly. This package is a local prepared artifact, not an upload receipt.\n"
        )
        (stage / "Reproduce.md").write_text(repro_notes, encoding="utf-8", newline="\n")

        # Replace prior inventory so it covers the complete payload, excluding itself.
        inventory_path = stage / "Checksums.json"
        inventory_path.unlink(missing_ok=True)
        inventory = {}
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            inventory[path.relative_to(stage).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha_file(path),
            }
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

        # Stage the ZIP outside its own input tree, then atomically replace only OUT.
        temp_zip = ROOT / "tmp/phase3_complete_package.zip"
        temp_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(p for p in stage.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(stage).as_posix())
        OUT.parent.mkdir(parents=True, exist_ok=True)
        temp_zip.replace(OUT)

    with zipfile.ZipFile(OUT) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"CRC failure: {bad}")
        names = zf.namelist()
        forbidden = [n for n in names if "v1" in n.lower() or "v2" in n.lower() or "idm-viz" in n.lower()]
        if forbidden:
            raise RuntimeError("Forbidden archive member names: " + ", ".join(forbidden))
        inventory = json.loads(zf.read("Checksums.json"))
        for name, record in inventory.items():
            if sha_bytes(zf.read(name)) != record["sha256"]:
                raise RuntimeError("Member checksum mismatch: " + name)
    receipt = {
        "archive": str(OUT), "bytes": OUT.stat().st_size, "sha256": sha_file(OUT),
        "members": len(names), "crc": "passed", "member_sha256": "passed",
        "submitted_archive_sha256": SUBMITTED_SHA, "uploaded": False,
    }
    OUT.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
