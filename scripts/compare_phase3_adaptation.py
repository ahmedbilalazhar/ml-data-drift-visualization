"""Exact comparison for a clean adaptation reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(first: Path, second: Path, output: Path):
    first, second, output = first.resolve(), second.resolve(), output.resolve()
    lock_a = json.loads((first / "protocol_lock.json").read_text(encoding="utf-8"))
    lock_b = json.loads((second / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock_a["status"] != "complete" or lock_b["status"] != "complete":
        raise ValueError("both adaptation runs must be complete")
    if lock_a["mode"] != lock_b["mode"] or lock_a["seeds_by_task"] != lock_b["seeds_by_task"]:
        raise AssertionError("adaptation modes or seed sets differ")
    csv_a = sorted(p.relative_to(first) for p in first.rglob("*.csv"))
    csv_b = sorted(p.relative_to(second) for p in second.rglob("*.csv"))
    if csv_a != csv_b:
        raise AssertionError("CSV artifact path sets differ")
    cells = 0
    for relative in csv_a:
        a, b = pd.read_csv(first / relative), pd.read_csv(second / relative)
        if a.columns.tolist() != b.columns.tolist():
            raise AssertionError(f"columns differ: {relative}")
        timing = [column for column in a.columns if column.endswith(("_seconds", "_time"))]
        a, b = a.drop(columns=timing), b.drop(columns=timing)
        if not a.equals(b):
            raise AssertionError(f"scientific CSV differs: {relative}")
        cells += a.size
    npz_a = sorted(p.relative_to(first) for p in first.rglob("*.npz"))
    npz_b = sorted(p.relative_to(second) for p in second.rglob("*.npz"))
    if npz_a != npz_b:
        raise AssertionError("array artifact path sets differ")
    arrays = values = 0
    for relative in npz_a:
        with np.load(first / relative) as a, np.load(second / relative) as b:
            if a.files != b.files:
                raise AssertionError(f"array keys differ: {relative}")
            for key in a.files:
                x, y = a[key], b[key]
                if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                    raise AssertionError(f"array values differ: {relative}:{key}")
                arrays += 1
                values += x.size
    result = dict(
        status="passed", first=str(first), second=str(second),
        seeds_by_task=lock_a["seeds_by_task"], csv_files=len(csv_a), csv_cells=cells,
        npz_archives=len(npz_a), arrays=arrays, array_values=values,
        comparison="exact outputs; timing fields and run metadata excluded",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.first, args.second, args.out)
