"""Replay label-release availability at delay 100 using frozen cached cohorts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(run: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"choose a fresh output directory: {out}")
    online_path = run / "online_batches.csv"
    evaluator_path = run / "evaluator_batches.csv"
    if not online_path.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError("run must contain online_batches.csv and evaluator_batches.csv")
    online = pd.read_csv(online_path)
    evaluator = pd.read_csv(evaluator_path)
    keys = ["seed", "condition", "batch"]
    for frame, name in ((online, "online"), (evaluator, "evaluator")):
        if not set(keys).issubset(frame.columns) or frame.duplicated(keys).any():
            raise ValueError(f"{name} stream keys are absent or duplicated")
    if "correct" not in evaluator or len(online) != 5040 or len(evaluator) != 5040:
        raise ValueError("expected the complete 30-seed, seven-condition, 24-batch study")
    if set(map(tuple, online[keys].to_numpy())) != set(map(tuple, evaluator[keys].to_numpy())):
        raise ValueError("online and evaluator cohort keys do not align")
    online_only_columns = set(online.columns) - set(evaluator.columns)
    if any(c in online.columns for c in ("correct", "accuracy", "harm", "y", "label")):
        raise ValueError("online stream exposes evaluator truth")
    protocol_path = run / "protocol_lock.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    delay = 100
    events = online[keys].copy()
    events["available_at"] = events.batch + delay
    events["available_by_input_end"] = events.available_at <= 24
    events["release_clock_after_drain"] = events.available_at.clip(lower=25, upper=124)
    if (events.release_clock_after_drain != events.available_at).any():
        raise AssertionError("drain clock does not reach the final release")
    out.mkdir(parents=True, exist_ok=False)
    events.to_csv(out / "release_schedule.csv", index=False)
    grouped = events.groupby(["seed", "condition"], as_index=False).agg(
        batches=("batch", "size"), available_at_input_end=("available_by_input_end", "sum"),
        first_release=("available_at", "min"), last_release=("available_at", "max"))
    grouped["unresolved_at_input_end"] = grouped.batches - grouped.available_at_input_end
    grouped["available_after_drain"] = grouped.batches
    grouped.to_csv(out / "per_stream_availability.csv", index=False)
    summary = {
        "status": "passed", "delay_batches": delay, "streams": len(grouped),
        "cohorts": len(events), "labels_available_at_input_end": int(events.available_by_input_end.sum()),
        "labels_unresolved_at_input_end": int((~events.available_by_input_end).sum()),
        "labels_released_after_drain": len(events), "drain_final_clock": 124,
        "online_truth_columns_absent": True,
        "online_columns_not_in_evaluator": sorted(online_only_columns),
        "input_hashes": {"online_batches.csv": sha(online_path),
                         "evaluator_batches.csv": sha(evaluator_path),
                         "protocol_lock.json": sha(protocol_path)},
        "scope": "label availability only; no adaptation or performance claim",
    }
    if summary["labels_available_at_input_end"] != 0 or summary["labels_released_after_drain"] != len(events):
        raise AssertionError("delay-100 release boundary failed")
    (out / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "protocol_copy.md").write_text(
        (ROOT / "docs/phase3/DELAY100_AVAILABILITY_EXTENSION_20261116.md").read_text(encoding="utf-8"),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "results/phase3/final_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
