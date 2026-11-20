"""Package and clean-extraction validate the UCI 224 delayed-label extension."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("C:/Users/hp/anaconda3/python.exe")
ARCHIVE = ROOT / "output/phase3/IDM_Viz_UCI224_Delayed_Replay_Addendum_20261116_v2.zip"

PROTECTED = {
    "docs/23i-2604_23i-2581DAV-phase2.docx": "786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4",
    "docs/Phase 03 - Implementation.pdf": "9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115",
    "output/phase3/23I-2581_23I-2604_Phase3.zip": "279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_path(files: dict[str, Path], relative: str):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    files[path.relative_to(ROOT).as_posix()] = path


def add_tree(files: dict[str, Path], relative: str):
    path = ROOT / relative
    for child in path.rglob("*"):
        if child.is_file() and "__pycache__" not in child.parts and not child.name.endswith(".tmp"):
            files[child.relative_to(ROOT).as_posix()] = child


def compare_trees(old: Path, new: Path, paths: list[str]):
    for rel in paths:
        a, b = old / rel, new / rel
        if a.suffix == ".csv":
            pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b), check_exact=True)
        elif a.suffix == ".json":
            assert json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8")), rel
        else:
            aa, bb = np.load(a), np.load(b)
            assert set(aa.files) == set(bb.files), rel
            for key in aa.files:
                np.testing.assert_array_equal(aa[key], bb[key], err_msg=f"{rel}:{key}")


def main():
    if ARCHIVE.exists():
        raise FileExistsError(ARCHIVE)
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    for rel, expected in PROTECTED.items():
        if sha(ROOT / rel) != expected:
            raise AssertionError(f"protected file changed: {rel}")

    files: dict[str, Path] = {}
    for rel in [
        "scripts/evaluate_uci224_day2.py",
        "scripts/run_uci224_delayed_replay.py",
        "scripts/evaluate_uci224_delayed_replay.py",
        "scripts/report_uci224_delayed_replay.py",
        "scripts/package_uci224_delayed_replay.py",
        "tests/phase3/test_uci224_delayed_replay.py",
        "tests/phase3/test_uci224_metadata_semantics.py",
        "docs/phase3/UCI224_DELAYED_REPLAY_PROTOCOL_20261116.md",
        "docs/phase3/PHASE2_TRACEABILITY.md",
        "docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md",
        "docs/phase3/DECISION_STRENGTHENING_CHANGELOG.md",
        "docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md",
        "docs/phase3/research_improvement_day2/PROTOCOL_LOCK.json",
        "data/external/uci224_source/uci224_curl.zip",
        "reports/phase3/uci224_delayed_replay_20261116_v5/REPORT.tex",
        "reports/phase3/uci224_delayed_replay_20261116_v5/REPORT.pdf",
        "reports/phase3/uci224_delayed_replay_20261116_v5/recalibration_by_delay.pdf",
        "reports/phase3/uci224_delayed_replay_20261116_v5/recalibration_by_delay.svg",
        "reports/phase3/uci224_delayed_replay_20261116_v5/recalibration_by_delay.png",
        "reports/phase3/uci224_delayed_replay_20261116_v5/cohort_accuracy_forecasts.pdf",
        "reports/phase3/uci224_delayed_replay_20261116_v5/cohort_accuracy_forecasts.svg",
        "reports/phase3/uci224_delayed_replay_20261116_v5/cohort_accuracy_forecasts.png",
    ]:
        add_path(files, rel)
    add_tree(files, "results/phase3/external_uci224_day2_v2")
    add_tree(files, "results/phase3/external_uci224_day2_metadata_v3")
    add_tree(files, "results/phase3/uci224_delayed_replay_20261116_v2")

    readme = """# UCI 224 delayed-label replay addendum

This package follows the validated Day 3 external-evaluation improvement report. It preserves the frozen batch-1 classifier and source run, adds a label-safe cohort-clock recalibration replay at delays 0/1/2, and corrects batch-1 source/evaluation class metadata with explicit denominators.

From this extracted root, with the configured Anaconda environment:

```powershell
$python = 'C:/Users/hp/anaconda3/python.exe'
& $python -m pytest tests/phase3/test_uci224_delayed_replay.py tests/phase3/test_uci224_metadata_semantics.py -q
& $python scripts/evaluate_uci224_day2.py --out results/phase3/metadata_NEW
& $python scripts/run_uci224_delayed_replay.py --source-run results/phase3/external_uci224_day2_v2 --out results/phase3/replay_NEW
& $python scripts/evaluate_uci224_delayed_replay.py --run results/phase3/replay_NEW
& $python scripts/report_uci224_delayed_replay.py --run results/phase3/replay_NEW --out reports/replay_NEW
tectonic reports/replay_NEW/REPORT.tex --outdir reports/replay_NEW --keep-logs
```

Every output directory must be new. Release eligibility is `source_batch + delay < forecast_batch`; labels from the target cohort never enter its own forecast. The replay uses simulated whole-cohort delays over one ordered data sequence, not observed label-arrival timestamps. Read the protocol and report for denominators and claim limits. `MEMBER_SHA256.json` inventories all packaged files; the adjacent receipt records fresh clean-extraction runs and protected Phase 2 hashes.
"""
    inventory = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in sorted(files.items())}
    with zipfile.ZipFile(ARCHIVE, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, path in files.items():
            zf.write(path, name)
        zf.writestr("REPRODUCE.md", readme)
        zf.writestr("MEMBER_SHA256.json", json.dumps(inventory, indent=2))
        zf.writestr("PRESERVATION.json", json.dumps(PROTECTED, indent=2))

    runs, checks = [], []
    with zipfile.ZipFile(ARCHIVE) as zf:
        if zf.testzip() is not None:
            raise AssertionError("archive CRC failure")
        for name, item in inventory.items():
            if hashlib.sha256(zf.read(name)).hexdigest() != item["sha256"]:
                raise AssertionError(f"member hash mismatch: {name}")
        with tempfile.TemporaryDirectory(prefix="uci224_replay_extract_") as temporary:
            work = Path(temporary)
            zf.extractall(work)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(work)
            env["PYTHONNOUSERSITE"] = "1"
            commands = [
                ("tests", [str(PYTHON), "-m", "pytest", "tests/phase3/test_uci224_delayed_replay.py",
                           "tests/phase3/test_uci224_metadata_semantics.py", "-q"]),
                ("metadata_runner", [str(PYTHON), "scripts/evaluate_uci224_day2.py", "--out", "results/phase3/metadata_EXTRACTED"]),
                ("delayed_runner", [str(PYTHON), "scripts/run_uci224_delayed_replay.py", "--source-run",
                                    "results/phase3/external_uci224_day2_v2", "--out", "results/phase3/replay_EXTRACTED"]),
                ("delayed_evaluator", [str(PYTHON), "scripts/evaluate_uci224_delayed_replay.py", "--run", "results/phase3/replay_EXTRACTED"]),
                ("report", [str(PYTHON), "scripts/report_uci224_delayed_replay.py", "--run",
                            "results/phase3/replay_EXTRACTED", "--out", "reports/replay_EXTRACTED"]),
            ]
            for stage, command in commands:
                completed = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                           text=True, timeout=300)
                runs.append({"stage": stage, "command": command, "exit_code": completed.returncode,
                             "stdout": completed.stdout, "stderr": completed.stderr})
                if completed.returncode:
                    raise RuntimeError(json.dumps(runs, indent=2))

            compare_trees(work / "results/phase3/external_uci224_day2_metadata_v3",
                          work / "results/phase3/metadata_EXTRACTED",
                          ["cohort_metrics.csv", "class_metrics.csv", "input_tests.csv", "summary.json", "protocol_lock.json"])
            checks += ["corrected external metrics/class composition/lock exact"]
            saved, fresh = (work / "results/phase3/uci224_delayed_replay_20261116_v2",
                            work / "results/phase3/replay_EXTRACTED")
            compare_trees(saved, fresh, ["fit_audit.csv", "delay_0_release_audit.csv", "delay_1_release_audit.csv",
                                         "delay_2_release_audit.csv", "evaluation/per_batch_metrics.csv",
                                         "evaluation/summary.csv", "evaluation/validation.json"])
            for rel in ["evaluator_truth/batch02_truth.npz"] + [f"predictions/batch{b:02}_online_inputs.npz" for b in range(2, 11)] + [
                f"predictions/delay{d}_batch{b:02}_q.npz" for d in (0, 1, 2) for b in range(2, 11)]:
                compare_trees(saved, fresh, [rel])
            checks += ["delayed replay audits, outcomes, predictions, and metrics exact"]
            regenerated = work / "reports/replay_EXTRACTED"
            original = work / "reports/phase3/uci224_delayed_replay_20261116_v5"
            for name in ["REPORT.tex"]:
                if (original / name).read_bytes() != (regenerated / name).read_bytes():
                    raise AssertionError(f"report source mismatch: {name}")
            for name in ["recalibration_by_delay.png", "cohort_accuracy_forecasts.png"]:
                if not np.array_equal(np.asarray(Image.open(original / name).convert("RGBA")),
                                      np.asarray(Image.open(regenerated / name).convert("RGBA"))):
                    raise AssertionError(f"report image mismatch: {name}")
            checks += ["TeX and scientific figure pixels exact after clean report regeneration"]

    if any(sha(ROOT / rel) != expected for rel, expected in PROTECTED.items()):
        raise AssertionError("protected submission hash changed during packaging")
    receipt = {
        "status": "passed", "archive": str(ARCHIVE), "archive_sha256": sha(ARCHIVE),
        "bytes": ARCHIVE.stat().st_size, "members": len(inventory) + 3,
        "crc": "passed", "member_hashes": "passed", "clean_extraction": "passed",
        "fresh_commands": runs, "exact_comparisons": checks,
        "preserved_hashes": PROTECTED, "uploaded": False,
    }
    ARCHIVE.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
