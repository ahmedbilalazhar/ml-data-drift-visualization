"""Build a non-destructive post-deadline package layered over preserved v14."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v14.zip"
OUT = ROOT / "output/phase3/IDM_Viz_Phase3_PostDeadline_Addendum_20261117_v20.zip"
RECEIPT = OUT.with_suffix(".receipt.json")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_files() -> dict[str, Path]:
    roots = [
        "scripts/audit_datasets.py", "scripts/download_datasets.py",
        "scripts/prepare_noaa_weather.py", "scripts/compare_weather_benchmark_mirror.py",
        "scripts/run_noaa_weather_transfer.py", "scripts/report_noaa_weather.py",
        "scripts/reproduce_noaa_weather.ps1", "scripts/rebuild_processed_datasets_v2.py",
        "scripts/run_localization_audit.py", "scripts/summarize_localization_audit.py",
        "scripts/run_cluster_mapping_sensitivity.py", "scripts/summarize_cluster_mapping_sensitivity.py",
        "scripts/run_noaa_reference_calibration_uncertainty.py",
        "scripts/summarize_noaa_reference_calibration_uncertainty.py",
        "scripts/summarize_noaa_block_length_sensitivity.py",
        "scripts/reproduce_noaa_block_length_sensitivity.ps1",
        "scripts/run_selective_label_audit.py", "scripts/summarize_selective_label_audit.py",
        "scripts/run_adaptation_schedule_audit.py", "scripts/report_adaptation_schedule_audit.py",
        "scripts/reproduce_adaptation_schedule_audit.ps1",
        "scripts/run_noaa_selective_label_replay.py", "scripts/summarize_noaa_selective_label_replay.py",
        "scripts/reproduce_noaa_selective_label_replay.ps1",
        "scripts/run_noaa_training_uncertainty.py", "scripts/summarize_noaa_training_uncertainty.py",
        "scripts/package_phase3_addendum.py",
        "configs/phase3_strengthening_v3.json",
        "src/drift_monitoring/data/real.py", "src/drift_monitoring/phase3",
        "tests/phase3/test_noaa_weather_contract.py",
        "tests/phase3/test_localization_audit_contract.py",
        "tests/phase3/test_cluster_mapping_sensitivity_contract.py",
        "tests/phase3/test_noaa_uncertainty_contract.py",
        "tests/phase3/test_noaa_block_length_sensitivity_contract.py",
        "tests/phase3/test_selective_label_audit_contract.py",
        "tests/phase3/test_adaptation_schedule_audit_contract.py",
        "tests/phase3/test_noaa_selective_label_replay_contract.py",
        "tests/phase3/test_noaa_training_uncertainty_contract.py",
        "configs/datasets/weather.yaml",
        "docs/23i-2604_23i-2581DAV-phase2.docx", "docs/Phase 03 - Implementation.pdf",
        "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md",
        "docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json",
        "docs/phase3/WEATHER_PROVENANCE_AUDIT_20261117.md",
        "docs/phase3/LOCALIZATION_AUDIT_20261117.md",
        "docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md",
        "docs/phase3/BASELINE_FIDELITY_AUDIT_20261117.md",
        "results/phase3/adaptation_schedule_audit_20261117_v1",
        "reports/phase3/adaptation_schedule_audit_20261117_v8",
        "results/phase3/adaptation_schedule_audit_20261117_v2",
        "reports/phase3/adaptation_schedule_audit_20261117_v11",
        "results/phase3/noaa_selective_label_replay_20261117_v1",
        "reports/phase3/noaa_selective_label_replay_20261117_v3",
        "results/phase3/noaa_training_uncertainty_20261117_v2",
        "reports/phase3/noaa_training_uncertainty_20261117_v3",
        "results/phase3/selective_label_audit_20261117_v2",
        "reports/phase3/selective_label_audit_20261117_v5",
        "data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv",
        "data/processed/weather_noaa_1949_1999_v2/weather_manifest.json",
        "data/processed/weather_noaa_1949_1999_v1/REJECTED_INTERMEDIATE.md",
        "data/processed/verified_v2/elec2_ordered.parquet",
        "data/processed/verified_v2/covtype_subset100k.parquet",
        "data/processed/verified_v2/covtype_source_row_ids.csv",
        "data/manifests/elec2_manifest.json", "data/manifests/covtype_manifest.json",
        "data/manifests/covtype_subset_manifest.json",
        "data/manifests/elec2_verified_v2_manifest.json",
        "data/manifests/covtype_subset_verified_v2_manifest.json",
        "data/interim/audit_all_datasets_20261117.json",
        "data/interim/audit_weather_noaa_20261117.json",
        "data/external/noaa_gsod_weather_1949_1999_v1/benchmark_comparison_20261117.json",
        "results/phase3/noaa_weather_1949_1999_v1",
        "results/phase3/localization_audit_20261117_v3",
        "reports/phase3/weather_noaa_1949_1999_v2",
        "reports/phase3/localization_audit_20261117_v5",
        "results/phase3/cluster_mapping_sensitivity_20261117_v2",
        "reports/phase3/cluster_mapping_sensitivity_20261117_v1",
        "results/phase3/noaa_reference_calibration_uncertainty_20261117_v1",
        "results/phase3/noaa_reference_calibration_uncertainty_20261117_v2",
        "results/phase3/noaa_reference_calibration_uncertainty_block7_20261117_v1",
        "results/phase3/noaa_reference_calibration_uncertainty_block90_20261117_v1",
        "reports/phase3/noaa_reference_calibration_uncertainty_20261117_v2",
        "reports/phase3/noaa_block_length_sensitivity_20261117_v3",
        "reports/phase3/strengthening_v3_evaluation_with_sensitivity_20261117",
        "data/raw/covtype/covtype.data.gz",
        "data/raw/openml",
        "data/external/noaa_gsod_weather_1949_1999_v1/raw",
        "data/external/noaa_gsod_weather_1949_1999_v1/reference/scikit_multiflow_weather.csv",
    ]
    files: dict[str, Path] = {}
    for item in roots:
        path = ROOT / item
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    files[child.relative_to(ROOT).as_posix()] = child
        elif path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
        else:
            raise FileNotFoundError(f"required package file is missing: {item}")
    return files


def main() -> None:
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite package or receipt: {OUT}")
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    overlay = select_files()
    overlay_names = set(overlay)
    readme = r'''# IDM-Viz Phase 3 post-deadline strengthening addendum

This is an additive update layered over `IDM_Viz_Phase3_Final_Strengthening_20261116_v14.zip`. It does not replace the original Phase 2 submission, the pre-deadline ZIP, or the v14 improvement package. The Phase 2 DOCX and implementation brief are included unchanged; their SHA-256 values are in `docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json`.

The addendum adds the NOAA GSOD one-station same-day classification exercise, audited 3/5/10 cluster-region and dependence-pair localization, a 30-seed full/PCA region-mapping sensitivity with many-to-one mappings, 200-replicate reference/calibrator uncertainty with 7/30/90-row block sensitivity, 200 training-only pipeline refits whose bands contain the original fitted estimate on 21/21 batches (not truth-coverage evidence), 30-seed synthetic and 1,000-draw retrospective NOAA probability-sampled label audits, a 30-seed frozen/periodic/global-trigger/region-local-trigger adaptation comparison, a primary-source baseline-fidelity audit, verified v2 Elec2/Covertype processed tables, tests, TeX sources and figures. It does not claim forecasting, broad transfer, prospective label operations, causal attribution or analyst benefit. The local trigger improved accuracy in the tested synthetic reversal and made one stationary false trigger in one seed; this result is generator-specific. Read the 17 November audit reports for complete result limits.

## Environment and clean-extraction verification

The scripts use the project Anaconda environment at `C:\Users\hp\anaconda3\python.exe`, pandas, numpy, scikit-learn, scipy, pyarrow, matplotlib, pytest, and Tectonic. Tectonic is required to compile the NOAA reproduction report.

From the extracted package root in PowerShell:

```powershell
C:\Users\hp\anaconda3\python.exe scripts\rebuild_processed_datasets_v2.py
C:\Users\hp\anaconda3\python.exe scripts\audit_datasets.py --output data\interim\audit_all_datasets_20261117.json
C:\Users\hp\anaconda3\python.exe -m pytest tests\phase3\test_noaa_weather_contract.py tests\phase3\test_localization_audit_contract.py tests\phase3\test_cluster_mapping_sensitivity_contract.py tests\phase3\test_noaa_uncertainty_contract.py tests\phase3\test_noaa_block_length_sensitivity_contract.py tests\phase3\test_selective_label_audit_contract.py tests\phase3\test_adaptation_schedule_audit_contract.py tests\phase3\test_noaa_selective_label_replay_contract.py tests\phase3\test_noaa_training_uncertainty_contract.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_noaa_weather.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_noaa_block_length_sensitivity.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_adaptation_schedule_audit.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_noaa_selective_label_replay.ps1 -Replicates 1000
C:\Users\hp\anaconda3\python.exe scripts\run_noaa_training_uncertainty.py --replicates 200 --block-length 30 --output results\phase3\noaa_training_uncertainty_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\summarize_noaa_training_uncertainty.py --run results\phase3\noaa_training_uncertainty_cleancheck --output reports\phase3\noaa_training_uncertainty_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\run_localization_audit.py --output results\phase3\localization_audit_20261117_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\summarize_localization_audit.py --run results\phase3\localization_audit_20261117_cleancheck --output reports\phase3\localization_audit_20261117_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\run_cluster_mapping_sensitivity.py --output results\phase3\cluster_mapping_sensitivity_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\summarize_cluster_mapping_sensitivity.py --run results\phase3\cluster_mapping_sensitivity_cleancheck --output reports\phase3\cluster_mapping_sensitivity_cleancheck
tectonic -X compile reports\phase3\cluster_mapping_sensitivity_cleancheck\CLUSTER_MAPPING_SENSITIVITY.tex --outdir reports\phase3\cluster_mapping_sensitivity_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\run_noaa_reference_calibration_uncertainty.py --output results\phase3\noaa_reference_calibration_uncertainty_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\summarize_noaa_reference_calibration_uncertainty.py --run results\phase3\noaa_reference_calibration_uncertainty_cleancheck --output reports\phase3\noaa_reference_calibration_uncertainty_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\run_selective_label_audit.py --output results\phase3\selective_label_audit_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\summarize_selective_label_audit.py --run results\phase3\selective_label_audit_cleancheck --output reports\phase3\selective_label_audit_cleancheck
```

The first build command verifies included v2 files by checksum and read-back without overwriting them. The nine Phase 3 contract-test files validate delayed label timing, adaptation schedule timing, NOAA training refits, natural-stream sampling probabilities, output integrity, historical-only mappings, and all three dataset loaders. The complete project suite (191 tests) passes in the source workspace; package commands reproduce the NOAA uncertainty, cluster-mapping, selective-label, training-refit and adaptation-schedule analyses in fresh output directories. All raw sources needed for NOAA and processed-dataset rebuilding are included with source manifests. The NOAA runs are same-day retrospective classification and are not forecasts; the selective-label replay is retrospective sampling on one fixed stream, not a prospective audit.

## Known limitations

The five-region cluster audit found weak region recovery. The dependence-pair test is one synthetic positive control. Research Question 3 is unanswered and no human study was conducted. Named-literature baseline fidelity, natural-stream calibration under shift, prospective selective-label operations, alternative region-local policies, equal consumed-label adaptation budgets, and joint model/calibrator/reference uncertainty guarantees remain open. No instructor feedback was received.
'''
    temp = OUT.with_name(f".{OUT.stem}.tmp.zip")
    with zipfile.ZipFile(BASE, "r") as old, zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                                                               compresslevel=6) as new:
        for info in old.infolist():
            if info.filename in overlay_names or info.filename in {"ADDENDUM_README.md", "ADDENDUM_FILE_MANIFEST.json"}:
                continue
            new.writestr(info, old.read(info.filename))
        for name, path in sorted(overlay.items()):
            new.write(path, name)
        new.writestr("ADDENDUM_README.md", readme)
        file_manifest = {name: {"sha256": digest(path), "bytes": path.stat().st_size}
                         for name, path in sorted(overlay.items())}
        new.writestr("ADDENDUM_FILE_MANIFEST.json", json.dumps({
            "package_id": OUT.stem.lower().replace("_", "-"),
            "base_package": BASE.name, "base_package_sha256": digest(BASE),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "overlay_file_count": len(file_manifest), "overlay_files": file_manifest,
            "phase2_docx_sha256": digest(ROOT / "docs/23i-2604_23i-2581DAV-phase2.docx"),
            "predeadline_zip_sha256": digest(ROOT / "output/phase3/23I-2581_23I-2604_Phase3.zip")
        }, indent=2) + "\n")
    temp.replace(OUT)
    with zipfile.ZipFile(OUT) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError(f"ZIP CRC validation failed at {bad}")
        expected = file_manifest.keys()
        for name in expected:
            if digest_bytes(bundle.read(name)) != file_manifest[name]["sha256"]:
                raise ValueError(f"ZIP overlay content hash mismatch: {name}")
    receipt = {"status": "validated", "package": OUT.name, "sha256": digest(OUT),
               "bytes": OUT.stat().st_size, "base_package": BASE.name,
               "overlay_file_count": len(overlay), "zip_members": len(zipfile.ZipFile(OUT).namelist()),
               "zip_crc": "passed", "overlay_sha256_checks": "passed"}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
