from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "output/phase3/IDM_Viz_Phase3_Implementation_Hardening_20261115_v2.zip"
PATHS = (
    Path("docs/phase3/implementation_hardening_20261115"),
    Path("src/drift_monitoring/phase3/core.py"),
    Path("src/drift_monitoring/phase3/replay.py"),
    Path("src/drift_monitoring/phase3/v2.py"),
    Path("src/drift_monitoring/phase3/adaptation.py"),
    Path("scripts/evaluate_uci224_day2.py"),
    Path("tests/phase3/test_hardened_contracts.py"),
    Path("requirements-phase3.txt"),
    Path("results/phase3/external_uci224_day2_hardening_check"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the versioned Phase 3 hardening overlay.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().out
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    files = []
    for item in PATHS:
        source = ROOT / item
        if source.is_dir():
            files.extend(path for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            files.append(source)
        else:
            raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(set(files)):
            relative = source.relative_to(ROOT).as_posix()
            info = ZipInfo(relative, date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    print(f"Created {output} with {len(files)} files")


if __name__ == "__main__":
    main()
