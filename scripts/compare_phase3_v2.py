"""Compare final/reproduction v2 scientific CSVs and cached arrays exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(first: Path, second: Path, output: Path):
    first, second, output = first.resolve(), second.resolve(), output.resolve()
    first_lock = json.loads((first / "protocol_lock.json").read_text())
    second_lock = json.loads((second / "protocol_lock.json").read_text())
    if first_lock["status"] != "complete" or second_lock["status"] != "complete":
        raise ValueError("both runs must be complete")
    if first_lock["seeds_by_task"] != second_lock["seeds_by_task"]:
        raise AssertionError("seed sets differ")

    csv_paths = sorted(path.relative_to(first) for path in first.rglob("*.csv"))
    other_csv_paths = sorted(path.relative_to(second) for path in second.rglob("*.csv"))
    if csv_paths != other_csv_paths:
        raise AssertionError("CSV path sets differ")
    cells = 0
    for relative in csv_paths:
        left = pd.read_csv(first / relative)
        right = pd.read_csv(second / relative)
        if left.columns.tolist() != right.columns.tolist():
            raise AssertionError(f"columns differ in {relative}")
        timed = [column for column in left.columns if column.endswith("_seconds")]
        left, right = left.drop(columns=timed), right.drop(columns=timed)
        if not left.equals(right):
            raise AssertionError(f"scientific CSV differs: {relative}")
        cells += left.size

    npz_paths = sorted(path.relative_to(first) for path in first.rglob("*.npz"))
    other_npz_paths = sorted(path.relative_to(second) for path in second.rglob("*.npz"))
    if npz_paths != other_npz_paths:
        raise AssertionError("NPZ path sets differ")
    arrays = values = 0
    for relative in npz_paths:
        with np.load(first / relative) as left, np.load(second / relative) as right:
            if left.files != right.files:
                raise AssertionError(f"array keys differ in {relative}")
            for name in left.files:
                a, b = left[name], right[name]
                if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                    raise AssertionError(f"array values differ in {relative}:{name}")
                arrays += 1
                values += a.size
    result = dict(status="passed", first=str(first), second=str(second),
                  final_seeds_by_task=first_lock["seeds_by_task"],
                  csv_files=len(csv_paths), csv_cells=cells,
                  npz_archives=len(npz_paths), arrays=arrays, array_values=values,
                  comparison="exact scientific CSVs and arrays; timing columns and protocol metadata excluded")
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
