"""Build the full-cache companion archive with neutral, short member paths."""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPACT = ROOT / "output/phase3/23I-2581_23I-2604_Phase3_Research_Complete.zip"
OUT = ROOT / "output/phase3/23I-2581_23I-2604_Phase3_Research_Full.zip"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def neutral_name(value: str) -> str:
    value = value.replace("v1_synthetic", "original_synthetic")
    value = value.replace("final_v1", "original_baseline")
    value = value.replace("v2_final", "model_final")
    value = value.replace("v2_reproduction", "model_reproduction")
    value = value.replace("adaptation_final", "adapt_final")
    value = value.replace("adaptation_reproduction", "adapt_reproduction")
    value = value.replace("v1_", "original_")
    value = value.replace("v2_", "model_")
    return value


def main() -> None:
    if not COMPACT.is_file():
        raise FileNotFoundError(COMPACT)
    cache_sources = (
        (ROOT / "results/phase3/final_v1", "Cache/OriginalBaseline"),
        (ROOT / "results/phase3/v2_final", "Cache/ModelFinal"),
        (ROOT / "results/phase3/v2_reproduction", "Cache/ModelReproduction"),
        (ROOT / "results/phase3/adaptation_final", "Cache/AdaptationFinal"),
        (ROOT / "results/phase3/adaptation_reproduction", "Cache/AdaptationReproduction"),
    )
    with tempfile.TemporaryDirectory(prefix="phase3_full_") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(COMPACT) as zf:
            zf.extractall(stage)
        notes = stage / "Full_Archive.md"
        notes.write_text(
            "# Full reproducibility archive\n\n"
            "This archive supplements the compact report package with complete saved "
            "features, predictions, evaluator truth, delayed-release queues, and output "
            "tables for the original baseline, model/task extension, and adaptation study. "
            "The cache folder names are shortened for readability.\n\n"
            "Exact clean-run commands are in `Reproduce.md`. The independent model/task "
            "and adaptation reproductions were generated in fresh output directories and "
            "compared against their respective final runs. Scientific result files matched "
            "exactly; timing and run-metadata fields are excluded as documented.\n",
            encoding="utf-8", newline="\n")
        for source, prefix in cache_sources:
            if not source.is_dir():
                raise FileNotFoundError(source)
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(source).as_posix()
                safe_rel = "/".join(neutral_name(part) for part in rel.split("/"))
                dest = stage / prefix / safe_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())

        inventory_path = stage / "Checksums.json"
        inventory_path.unlink(missing_ok=True)
        inventory = {}
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            rel = path.relative_to(stage).as_posix()
            if "v1" in rel.lower() or "v2" in rel.lower() or "idm-viz" in rel.lower():
                raise RuntimeError("Unneutralized member path: " + rel)
            inventory[rel] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        temp_zip = ROOT / "tmp/phase3_full_package.zip"
        temp_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
            for path in sorted(p for p in stage.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(stage).as_posix())
        OUT.parent.mkdir(parents=True, exist_ok=True)
        temp_zip.replace(OUT)

    with zipfile.ZipFile(OUT) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError("ZIP CRC failure: " + bad)
        names = zf.namelist()
        if any("v1" in n.lower() or "v2" in n.lower() or "idm-viz" in n.lower() for n in names):
            raise RuntimeError("Non-neutral member path remains")
        inventory = json.loads(zf.read("Checksums.json"))
        if len(inventory) + 1 != len(names):
            raise RuntimeError("Inventory membership mismatch")
        for name, record in inventory.items():
            if sha_bytes(zf.read(name)) != record["sha256"]:
                raise RuntimeError("Member checksum mismatch: " + name)
    receipt = {"archive": str(OUT), "bytes": OUT.stat().st_size,
               "sha256": sha_file(OUT), "members": len(names),
               "crc": "passed", "member_sha256": "passed", "uploaded": False}
    OUT.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
