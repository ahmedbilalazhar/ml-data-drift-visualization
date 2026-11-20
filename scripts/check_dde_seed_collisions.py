"""Check the locked DDE audit seed banks against existing project JSON artifacts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dde_subset_audit_v1.json"
AUDIT = ROOT / "docs/phase3/DDE_SEED_COLLISION_AUDIT_20261118.json"
SCAN_ROOTS = [ROOT / name for name in ("configs", "studies", "docs", "results", "reports", "scripts", "tests")]


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    banks = {name: list(map(int, config[f"{name}_seeds"]))
             for name in ("development", "validation", "evaluation")}
    all_seeds = [s for values in banks.values() for s in values]
    if len(set(all_seeds)) != len(all_seeds):
        raise SystemExit("seed banks overlap internally")
    smoke_seed = int(config["smoke_seed"])
    if smoke_seed in set(all_seeds):
        raise SystemExit("smoke seed must not overlap a validation or evaluation seed bank")
    checked_seeds = [*all_seeds, smoke_seed]
    # The config being audited declares these values and is excluded. The audit
    # output is also excluded on reruns; all other JSON evidence is searched.
    pattern = re.compile(r"(?<!\d)(?:" + "|".join(map(str, sorted(checked_seeds, key=lambda value: len(str(value)), reverse=True))) + r")(?!\d)")
    scanned = []
    collisions = []
    excluded_run_prefixes = (
        "results/phase3/dde_subset_audit_20261118_",
        "reports/phase3/dde_subset_audit_20261118_",
    )
    for base in SCAN_ROOTS:
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            rel = path.relative_to(ROOT).as_posix()
            if path.resolve() in {CONFIG.resolve(), AUDIT.resolve()}:
                continue
            if rel.startswith(excluded_run_prefixes):
                continue
            scanned.append(rel)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in pattern.finditer(content):
                collisions.append({"seed": int(match.group()), "path": rel})
    receipt = {
        "audit_id": "dde-seed-collision-audit-20261118-v1",
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "config_seed_banks": banks,
        "smoke_seed": smoke_seed,
        "seed_banks_pairwise_disjoint": True,
        "smoke_seed_disjoint": True,
        "searched_artifact_scope": "all existing JSON files under configs, studies, docs, results, reports, scripts and tests; exact integer-token matching",
        "excluded": [CONFIG.relative_to(ROOT).as_posix(), AUDIT.relative_to(ROOT).as_posix(),
                     *excluded_run_prefixes],
        "scanned_json_count": len(scanned),
        "scanned_paths_sha256": hashlib.sha256("\n".join(sorted(scanned)).encode()).hexdigest(),
        "collisions": collisions,
        "status": "clear" if not collisions else "collision_found",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "scanned_json_count": len(scanned),
                      "collision_count": len(collisions), "audit": str(AUDIT)}, indent=2))
    if collisions:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
