"""Create a short, single-file Phase 3 submission archive."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/phase3/23I-2581_23I-2604_Phase3_Research_Extension.zip"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    report = ROOT / "reports/phase3/v2_final"
    run = ROOT / "results/phase3/v2_final"
    required = [
        report / "23I-2581_23I-2604_Phase3_Report.pdf",
        report / "23I-2581_23I-2604_Phase3_Report.md",
        report / "23I-2581_23I-2604_Phase3_Evidence_Replay.html",
        run / "protocol_lock.json", run / "environment.json", run / "validation.json",
        ROOT / "docs/phase3/V2_EXTENSION_MANIFEST.md",
        ROOT / "docs/phase3/V2_DECISION_LOG.md",
        ROOT / "requirements-phase3.txt",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required package inputs: " + ", ".join(missing))
    # Verify the canonical v1 archive remains exactly the prepared snapshot.
    v1 = ROOT / "output/phase3/IDM_Viz_Phase3_v1_submission.zip"
    expected_v1 = "ddf91fa89ab9a9d45c20059c1dbea5cfdd966f3eea62a605769a371ae55e0ebd"
    if sha(v1) != expected_v1:
        raise RuntimeError("Canonical prepared v1 archive changed; refusing to proceed")

    with tempfile.TemporaryDirectory(prefix="phase3_submission_") as temp_name:
        stage = Path(temp_name)

        def copy(source: Path, target: str) -> None:
            destination = stage / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        (stage / "README.txt").write_text(
            "DATA ANALYSIS AND VISUALIZATION — PHASE 3\n"
            "Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)\n\n"
            "Submission contents: empirical report, figures, offline evidence replay, "
            "protocol and decision record, result tables, source code, and validation.\n"
            "The additional task is synthetic; the local ELEC2 candidate was rejected "
            "because chronology could not be verified. No human-subject results are claimed.\n\n"
            "This archive is a prepared local artifact. It is not an upload receipt.\n",
            encoding="utf-8")
        copy(report / "23I-2581_23I-2604_Phase3_Report.pdf", "Report.pdf")
        copy(report / "23I-2581_23I-2604_Phase3_Report.md", "Report.md")
        copy(report / "23I-2581_23I-2604_Phase3_Evidence_Replay.html", "Replay.html")

        manifest = (ROOT / "docs/phase3/V2_EXTENSION_MANIFEST.md").read_text(encoding="utf-8")
        decision = (ROOT / "docs/phase3/V2_DECISION_LOG.md").read_text(encoding="utf-8")
        (stage / "Code").mkdir(parents=True, exist_ok=True)
        (stage / "Protocol.md").write_text(manifest.replace("IDM-Viz", "Phase 3 research"), encoding="utf-8")
        (stage / "Decisions.md").write_text(decision.replace("IDM-Viz", "Phase 3 research"), encoding="utf-8")
        (stage / "Code/Protocol.md").write_text(manifest.replace("IDM-Viz", "Phase 3 research"), encoding="utf-8")
        (stage / "Requirements.txt").write_text((ROOT / "requirements-phase3.txt").read_text(encoding="utf-8"), encoding="utf-8")

        protocol = json.loads((run / "protocol_lock.json").read_text(encoding="utf-8"))
        protocol["protocol"] = "Phase 3 outcome-informed extension"
        (stage / "Protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        copy(run / "environment.json", "Environment.json")
        copy(run / "validation.json", "Validation.json")
        copy(report / "pdf_render_record.json", "Render.json")
        copy(report / "visual_review.json", "Visual.json")
        reproduction_check = ROOT / "reports/phase3/v2_reproduction/reproduction_comparison.json"
        if reproduction_check.is_file():
            copy(reproduction_check, "Reproduction.json")
        view_check = {"status": "passed", "checks": [
            "same-cohort panel refresh", "input/change event handling", "future inputs hidden",
            "delay changes update every panel", "unreleased accuracy and F1 hidden",
            "release-clock reveal", "queue drainage through clock 44", "evidence age at release",
            "task/seed/condition switching"]}
        (stage / "Replay_Checks.json").write_text(json.dumps(view_check, indent=2), encoding="utf-8")

        out_to_pkg = {
            "online_batches.csv": "Results/online.csv",
            "released_verification.csv": "Results/verification.csv",
            "evaluator_batches.csv": "Results/evaluator.csv",
            "impact_per_seed.csv": "Results/impact.csv",
            "impact_summary.csv": "Results/impact_summary.csv",
            "harm_counts.csv": "Results/harm.csv",
            "overall_per_seed.csv": "Results/overall.csv",
            "overall_summary.csv": "Results/overall_summary.csv",
            "paired_estimator_differences.csv": "Results/estimator_pairs.csv",
            "paired_estimator_bootstrap.csv": "Results/estimator_bootstrap.csv",
            "paired_model_differences.csv": "Results/model_pairs.csv",
            "paired_model_bootstrap.csv": "Results/model_bootstrap.csv",
            "calibration_error_per_seed.csv": "Results/calibration.csv",
            "calibration_error_summary.csv": "Results/cal_summary.csv",
            "delay_availability.csv": "Results/delay.csv",
            "delay_availability_summary.csv": "Results/delay_summary.csv",
            "input_detection_per_seed.csv": "Results/detection.csv",
            "input_detection_summary.csv": "Results/detection_summary.csv",
            "fit_records.csv": "Results/fit.csv",
            "runtime_per_seed.csv": "Results/runtime.csv",
            "clock_events.csv": "Results/clocks.csv",
        }
        for source_name, target_name in out_to_pkg.items():
            copy(run / source_name, target_name)
        for source_name in ("v1_conditional_failure_diagnostics.csv", "v1_dependence_component_diagnostics.csv"):
            copy(report / source_name, "Results/" + source_name.replace("v1_", "baseline_"))
        v1run = ROOT / "results/phase3/final_v1"
        for source_name, target_name in {
            "impact_summary.csv": "Results/baseline_impact_summary.csv",
            "harm_counts.csv": "Results/baseline_harm.csv",
            "calibration_error_per_seed.csv": "Results/baseline_calibration.csv",
            "detection_summary.csv": "Results/baseline_detection.csv",
        }.items():
            copy(v1run / source_name, target_name)
        for source in sorted((report / "figures").glob("*.png")):
            fig_name = source.name.replace("v1_c6_pca_components", "dependence_pca").replace("v1_conditional_failure", "conditional_shift")
            copy(source, "Figures/" + fig_name)
        for source in sorted((report / "figures").glob("*.svg")):
            fig_name = source.name.replace("v1_c6_pca_components", "dependence_pca").replace("v1_conditional_failure", "conditional_shift")
            copy(source, "Figures/" + fig_name)

        # Include source and the scripts needed to inspect/recreate each stage.
        src = ROOT / "src/drift_monitoring/phase3"
        for name in ("core.py", "replay.py", "evaluation.py"):
            copy(src / name, "Code/src/drift_monitoring/phase3/" + name)
        extension = (src / "v2.py").read_text(encoding="utf-8")
        (stage / "Code/src/drift_monitoring/phase3").mkdir(parents=True, exist_ok=True)
        (stage / "Code/src/drift_monitoring/phase3/extension.py").write_text(extension, encoding="utf-8")
        copy(src / "__init__.py", "Code/src/drift_monitoring/phase3/__init__.py")
        script_map = {
            "run.py": "run_phase3_v2.py", "evaluate.py": "evaluate_phase3_v2.py",
            "report.py": "report_phase3_v2.py", "replay.py": "build_phase3_v2_view.py",
            "compare.py": "compare_phase3_v2.py", "pdf.py": "render_phase3_v2_pdf.py",
        }
        for target_name, source_name in script_map.items():
            text = (ROOT / "scripts" / source_name).read_text(encoding="utf-8")
            text = text.replace("ROOT = Path(__file__).resolve().parents[1]",
                                "ROOT = Path(__file__).resolve().parents[1]")
            text = text.replace("drift_monitoring.phase3.v2", "drift_monitoring.phase3.extension")
            text = text.replace("phase3.v2 import", "phase3.extension import")
            text = text.replace("phase3/v2.py", "phase3/extension.py")
            text = text.replace("run_phase3_v2.py", "run.py")
            text = text.replace("evaluate_phase3_v2.py", "evaluate.py")
            text = text.replace("test_v2_extension_contract.py", "test_contract.py")
            text = text.replace("V2_EXTENSION_MANIFEST.md", "Protocol.md")
            text = text.replace('ROOT / "docs/phase3/Protocol.md"', 'ROOT / "Protocol.md"')
            target = stage / "Code/scripts" / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        test_source = (ROOT / "tests/phase3/test_v2_extension_contract.py").read_text(encoding="utf-8")
        test_source = test_source.replace("drift_monitoring.phase3.v2", "drift_monitoring.phase3.extension")
        (stage / "Code/tests").mkdir(parents=True, exist_ok=True)
        (stage / "Code/tests/test_contract.py").write_text(test_source, encoding="utf-8")

        repro = """# Reproduction notes\n\n""" + \
            "The locked final run used 30 matched seeds on the original synthetic task and 30 fresh seeds on the XOR synthetic task. Exact interpreter and package versions are in `Environment.json`; protocol and stop rules are in `Protocol.md`.\n\n" + \
            "The archive contains result tables and source. A full rerun of the matched original-task extension also requires the preserved Phase 2 project and the original run caches; the compact submission intentionally omits large instance caches. From the original workspace root, use the commands documented in the project reproduction guide.\n\n" + \
            "Online predictions, scheduled verification, and evaluator truth are separate. The replay filters truth by its recorded availability clock.\n"
        (stage / "Reproduce.md").write_text(repro, encoding="utf-8")

        # Build an inventory over every payload member, then write the archive.
        inventory = {}
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            inventory[path.relative_to(stage).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        (stage / "Checksums.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        temp_zip = stage / "submission.zip"
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(p for p in stage.rglob("*") if p.is_file() and p != temp_zip):
                archive.write(path, path.relative_to(stage).as_posix())
        # Target is the requested roll-number alias. Canonical version 1 archives are separate.
        shutil.copy2(temp_zip, OUT)

    with zipfile.ZipFile(OUT) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        names = archive.namelist()
        forbidden = [name for name in names if "v1" in name.lower() or "v2" in name.lower() or "idm-viz" in name.lower()]
        if forbidden:
            raise RuntimeError("Non-neutral package names: " + ", ".join(forbidden))
        if any("IDM-Viz" in archive.read(name).decode("utf-8", "ignore") for name in names if name.lower().endswith((".md", ".txt", ".json"))):
            raise RuntimeError("Project branding remains in packaged documents")
        # Validate every inventoried member against its embedded inventory.
        inventory = json.loads(archive.read("Checksums.json"))
        for name, expected in inventory.items():
            if sha_from_bytes(archive.read(name)) != expected["sha256"]:
                raise RuntimeError("Member checksum mismatch: " + name)
        receipt = {"archive": str(OUT), "bytes": OUT.stat().st_size,
                   "sha256": sha(OUT), "members": len(names),
                   "crc": "passed", "member_sha256": "passed",
                   "version1_archive_sha256": expected_v1,
                   "uploaded": False}
    receipt_path = OUT.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def sha_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
