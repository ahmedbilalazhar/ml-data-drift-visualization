"""Replay fixed accuracy forecasts on three existing frozen-model streams."""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/phase3"
PLAN = ROOT / "docs/phase3/EXTERNAL_FUTURE_HARM_REPLAY_20261120.md"
DELAYS = (0, 2, 5)
HORIZONS = (1, 5)
METHODS = ("historical", "current_monitor", "last_released", "mean3_released")

SOURCES = {
    "ELEC2": ("elec2_temporal_20261116_v2/online_batches.csv",
              "elec2_temporal_20261116_v2/evaluation/evaluator_batches.csv",
              "estimate", "historical_reference_accuracy", "n", 1, 55),
    "NOAA": ("noaa_weather_1949_1999_v1/online/batch_scores.csv",
             "noaa_weather_1949_1999_v1/evaluator/batch_metrics.csv",
             "cal_estimate", "reference_accuracy", "n", 1, 21),
    "UCI224": ("external_uci224_day2_v2/cohort_metrics.csv",
               "external_uci224_day2_v2/cohort_metrics.csv",
               "estimated_accuracy", "historical_reference_accuracy", "cohort_rows", 2, 9),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, start):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if int(row["batch"]) >= start]
    rows.sort(key=lambda row: int(row["batch"]))
    return rows


def load_stream(spec):
    online_file, truth_file, estimate_field, ref_field, n_field, start, count = spec
    online = read(RUNS / online_file, start)
    truth = read(RUNS / truth_file, start)
    batches = list(range(start, start + count))
    for rows in (online, truth):
        if [int(row["batch"]) for row in rows] != batches:
            raise ValueError(f"Missing, duplicate or unexpected cohort in {spec[0]}")
    n = [int(row[n_field]) for row in truth]
    if n != [int(row[n_field]) for row in online] or min(n) < 1:
        raise ValueError(f"Online/evaluator cohort size mismatch in {spec[0]}")
    refs = {float(row[ref_field]) for row in online}
    if len(refs) != 1:
        raise ValueError(f"Historical reference changed in {spec[0]}")
    ref = refs.pop()
    estimates = [float(row[estimate_field]) for row in online]
    accuracy = [float(row["accuracy"]) for row in truth]
    if not all(0 <= x <= 1 for x in [ref, *estimates, *accuracy]):
        raise ValueError(f"Invalid accuracy or estimate in {spec[0]}")
    return batches, ref, estimates, accuracy, n


def forecast(accuracy, estimate, reference, index, delay):
    """The source and target clocks share consecutive cohort IDs."""
    released_n = max(0, index - delay)
    history = accuracy[:released_n]
    latest = history[-1] if history else reference
    trailing = mean(history[-3:]) if history else reference
    return released_n, (reference, estimate, latest, trailing)


def evaluate(stream, spec):
    batches, ref, estimates, accuracy, n = load_stream(spec)
    online, targets, metrics = [], [], []
    checks = 0
    for horizon in HORIZONS:
        actual = {}
        for index in range(len(batches) - horizon):
            value = mean(accuracy[index + 1:index + 1 + horizon])
            actual[index] = value
            targets.append(dict(stream=stream, horizon=horizon, origin=batches[index],
                                last_target=batches[index + horizon],
                                target_rows=sum(n[index + 1:index + horizon + 1]),
                                future_accuracy=value))
        for delay in DELAYS:
            by_method = {method: [] for method in METHODS}
            for index, value in actual.items():
                usable, predictions = forecast(accuracy, estimates[index], ref, index, delay)
                perturbed = accuracy[:usable] + [1 - x for x in accuracy[usable:]]
                if (usable, predictions) != forecast(perturbed, estimates[index], ref, index, delay):
                    raise AssertionError("Unavailable outcomes changed the forecasts")
                checks += 1
                for method, prediction in zip(METHODS, predictions):
                    row = dict(stream=stream, horizon=horizon, delay=delay, origin=batches[index],
                               released_cohorts=usable,
                               latest_released=batches[usable - 1] if usable else "",
                               method=method, prediction=prediction)
                    online.append(row)
                    by_method[method].append((index, value, prediction))
            for method, subset in product(METHODS, ("all", "common_released")):
                pairs = [(i, y, p) for i, y, p in by_method[method]
                         if subset == "all" or i > max(DELAYS)]
                error = [p - y for _, y, p in pairs]
                harm = [(y < ref - .05, p < ref - .05) for _, y, p in pairs]
                metrics.append(dict(stream=stream, horizon=horizon, delay=delay, subset=subset,
                                    method=method, origins=len(pairs), reference_accuracy=ref,
                                    origins_with_labels=sum(i > delay for i, _, _ in pairs),
                                    mae=mean(abs(e) for e in error) if error else "",
                                    bias=mean(error) if error else "",
                                    observed_harm=sum(y for y, _ in harm),
                                    predicted_harm=sum(p for _, p in harm),
                                    false_positives=sum(p and not y for y, p in harm),
                                    false_negatives=sum(y and not p for y, p in harm)))
    return online, targets, metrics, checks


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    sources = {RUNS / name for spec in SOURCES.values() for name in spec[:2]}
    sources |= {Path(__file__).resolve(), PLAN}
    metadata = dict(status="running", python=platform.python_version(),
                    delays=DELAYS, horizons=HORIZONS, source_rows_by_stream=SOURCES,
                    source_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                                   for p in sorted(sources)})
    lock = out / "protocol_lock.json"
    lock.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    online, targets, metrics, checks = [], [], [], 0
    for stream, spec in SOURCES.items():
        p, y, m, c = evaluate(stream, spec)
        online.extend(p)
        targets.extend(y)
        metrics.extend(m)
        checks += c
    write_csv(out / "online_forecasts.csv", online)
    write_csv(out / "evaluator_targets.csv", targets)
    write_csv(out / "summary.csv", metrics)
    metadata.update(status="complete", forecast_rows=len(online), unavailable_outcome_checks=checks,
                    output_sha256={p.name: digest(p) for p in sorted(out.glob("*.csv"))})
    lock.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{len(online)} forecast rows; {checks} release-boundary checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
