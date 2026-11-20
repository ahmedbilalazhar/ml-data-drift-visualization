"""Exploratory ADWIN probe placement sensitivity with an early/midpoint pair.

It reuses the tested two-probe scheduling engine while changing only the
predeclared probe clocks from (1, 2) to (1, 12). The policy was motivated by
the one-probe deadlock and early-probe failure; the result is exploratory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

if __package__:
    from . import run_adwin_bootstrap_sensitivity as engine
else:
    import run_adwin_bootstrap_sensitivity as engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    engine.BOOTSTRAP_CLOCKS = (1, 12)
    sys.argv = [str(Path(__file__).resolve()), "--out", str(OUT)]
    engine.main()
    lock_path = OUT / "protocol_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.update({
        "protocol": "river-adwin-early-midpoint-probes-sensitivity-v1",
        "bootstrap_clocks": [1, 12],
        "bootstrap_rule": "one early probe at batch 1 and one fixed midpoint probe at batch 12, both within the declared audit budget",
        "variant": "midpoint two-probe schedule; exploratory sensitivity following one-probe deadlock and adjacent early-probe non-detection",
        "supersedes": "adwin_midpoint_probe_sensitivity_20261118_v1 (invalid: compared stream clock to cohort budget and skipped clock-12 probe for budgets below 12)",
        "probe_budget_gate": "probe is selected when its fixed stream clock is reached and at least one cohort of budget remains",
    })
    for path in (Path(__file__), ROOT / "scripts/run_adwin_bootstrap_sensitivity.py"):
        lock["source_sha256"][path.relative_to(ROOT).as_posix()] = sha(path)
    per_seed_path = OUT / "two_probe_per_seed.csv"
    per_seed = pd.read_csv(per_seed_path)
    per_seed["policy"] = "river_adwin_early_midpoint_probes"
    per_seed.to_csv(per_seed_path, index=False)
    lock["output_sha256"][per_seed_path.name] = sha(per_seed_path)
    contrast_path = OUT / "paired_one_probe_contrasts.csv"
    contrasts = pd.read_csv(contrast_path)
    contrasts["contrast"] = "early_midpoint_two_probe_minus_one_probe"
    contrasts.to_csv(contrast_path, index=False)
    lock["output_sha256"][contrast_path.name] = sha(contrast_path)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Locked midpoint-probe sensitivity at {OUT}")


if __name__ == "__main__":
    main()
