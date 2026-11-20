"""Exploratory follow-up: give ADWIN two early audited cohorts to avoid cold-start deadlock.

This protocol was motivated by the one-probe ADWIN result. It is therefore an
exploratory policy amendment, not a confirmatory, previously locked comparison.
It reuses the same cached final streams and applies the ordinary budget caps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import river
from river.drift import ADWIN

if __package__:
    from .run_adwin_audit_baseline import BATCH_SIZE, COST_MODES, DELAYS, BUDGETS, RUN_DIR, sha
else:
    from run_adwin_audit_baseline import BATCH_SIZE, COST_MODES, DELAYS, BUDGETS, RUN_DIR, sha

ROOT = Path(__file__).resolve().parents[1]
STREAM_BATCHES = 24
BOOTSTRAP_CLOCKS = (1, 2)
BOOTSTRAP_REPLICATES = 2_000


def find_episodes(harm: np.ndarray) -> list[tuple[int, int]]:
    result = []
    start = None
    for i, value in enumerate(harm, 1):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(harm)):
            result.append((start, i if value and i == len(harm) else i - 1))
            start = None
    return result


def schedule_two_probes(group: pd.DataFrame, row_errors: np.ndarray, truth: pd.DataFrame,
                        delay: int, budget: int, cost_mode: str) -> tuple[list[dict], dict]:
    g = group.sort_values("batch").reset_index(drop=True)
    if len(g) != 24 or not np.array_equal(g.batch.to_numpy(), np.arange(1, 25)):
        raise ValueError("expected locked batches 1..24")
    if row_errors.shape != (24, 500):
        raise ValueError("expected row-level errors with shape (24, 500)")
    outcome = truth.loc[g.cohort.to_list()]
    harm, accuracy = outcome.harm.to_numpy(bool), outcome.accuracy.to_numpy(float)
    detector = ADWIN(delta=.002, clock=1)
    selected: list[int] = []
    consumed: set[int] = set()
    decisions = []
    update_count = 0
    detection_count = 0
    for clock in range(1, STREAM_BATCHES + 1):
        released = [s for s in selected if s + delay < clock and s not in consumed]
        events = []
        for source in released:
            for row_index, error in enumerate(row_errors[source - 1]):
                detector.update(float(error))
                update_count += 1
                if detector.drift_detected:
                    detection_count += 1
                    events.append({"source_batch": source, "row_index_1based": row_index + 1,
                                   "decision_clock": clock, "window_width": int(detector.width),
                                   "error_estimate": float(detector.estimation)})
            consumed.add(source)
        # Probe clocks are stream positions; budget is a number of cohorts.
        # Gate probes by remaining cohort capacity, never by comparing a clock
        # value to the cohort count (which silently drops late probes).
        bootstrap = clock in BOOTSTRAP_CLOCKS and len(selected) < budget
        signal = bool(bootstrap or events)
        remaining = budget - len(selected)
        select = bool(signal and remaining > 0)
        forced = False
        if cost_mode == "exact_quota" and not select and remaining > 0 and 25 - clock <= remaining:
            select, forced = True, True
        if select:
            selected.append(clock)
        decisions.append({
            "batch": clock, "selected": select, "policy_signal": signal,
            "bootstrap_probe": bootstrap, "adwin_events_from_released_labels": len(events),
            "released_sources": json.dumps(released), "labels_processed_so_far": update_count,
            "adwin_detections_so_far": detection_count, "adwin_width": int(detector.width),
            "adwin_error_estimate": float(detector.estimation),
            "forced_quota_acquisition": forced,
            "selected_labels_available_at": clock + delay if select else None,
            "selected_labels_first_usable_clock": clock + delay + 1 if select else None,
        })

    if cost_mode == "strict_cap" and len(selected) > budget:
        raise AssertionError("strict cap exceeded")
    if cost_mode == "exact_quota" and len(selected) != budget:
        raise AssertionError("exact quota not met")
    harmful = [s for s in selected if harm[s - 1]]
    verified = [s for s in harmful if s + delay + 1 <= 24]
    eps = find_episodes(harm)
    verified_eps = 0
    delays = []
    for start, end in eps:
        hits = [s for s in harmful if start <= s <= end]
        if hits:
            release = min(s + delay + 1 for s in hits)
            if release <= 24:
                verified_eps += 1
                delays.append(release - start)
    nh = int(harm.sum())
    nsel = len(selected)
    metrics = {
        "audited_batches": nsel, "labels_requested": nsel * 500,
        "budget_labels": budget * 500, "unspent_budget_cohorts": budget - nsel,
        "forced_quota_acquisitions": sum(d["forced_quota_acquisition"] for d in decisions),
        "harmful_batches": nh, "harmful_audits": len(harmful),
        "verified_harmful_audits_by_stream_end": len(verified),
        "unnecessary_audits": nsel - len(harmful), "missed_harmful_batches": nh - len(harmful),
        "audit_precision": len(harmful) / nsel if nsel else np.nan,
        "harm_recall": len(harmful) / nh if nh else np.nan,
        "true_harmful_episodes": len(eps), "verified_harmful_episodes_by_stream_end": verified_eps,
        "mean_episode_verification_delay": float(np.mean(delays)) if delays else np.nan,
        "adwin_detection_events": detection_count, "adwin_row_updates": update_count,
        "selected_batches": json.dumps(selected),
    }
    return decisions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/phase3/adwin_two_probe_sensitivity_20261118_v1")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    started = time.perf_counter()
    final = pd.read_csv(RUN_DIR / "online_batches.csv")
    raw_truth = pd.read_csv(RUN_DIR / "released_verification.csv")
    truth = raw_truth[raw_truth.delay == 0].set_index("cohort", verify_integrity=True)
    if __package__:
        from .run_adwin_audit_baseline import row_errors_for
    else:
        from run_adwin_audit_baseline import row_errors_for

    rows, log = [], []
    for (seed, condition), group in final.groupby(["seed", "condition"], sort=True):
        error = row_errors_for(int(seed), str(condition))
        for delay in DELAYS:
            for budget in BUDGETS:
                for mode in COST_MODES:
                    decisions, metrics = schedule_two_probes(group, error, truth, delay, budget, mode)
                    rows.append({"seed": int(seed), "condition": condition, "delay": delay,
                                 "budget_cohorts": budget, "cost_mode": mode,
                                 "policy": "river_adwin_two_early_probes", **metrics})
                    log.extend({"seed": int(seed), "condition": condition, "delay": delay,
                                "budget_cohorts": budget, "cost_mode": mode, **record}
                               for record in decisions)
    elapsed = time.perf_counter() - started
    out.mkdir(parents=True)
    result = pd.DataFrame(rows)
    decisions = pd.DataFrame(log)
    result.to_csv(out / "two_probe_per_seed.csv", index=False)
    decisions.to_csv(out / "two_probe_decisions.csv", index=False)
    summary = result.groupby(["delay", "budget_cohorts", "cost_mode"], as_index=False).mean(numeric_only=True)
    summary.to_csv(out / "two_probe_summary.csv", index=False)

    baseline = pd.read_csv(ROOT / "results/phase3/adwin_audit_baseline_20261117_v1/adwin_per_seed.csv")
    keys = ["seed", "condition", "delay", "budget_cohorts", "cost_mode"]
    contrasts = []
    joined = result.set_index(keys).join(
        baseline.set_index(keys)[["verified_harmful_episodes_by_stream_end", "audited_batches"]],
        rsuffix="_one_probe", how="left")
    joined["episode_delta"] = (joined.verified_harmful_episodes_by_stream_end -
                               joined.verified_harmful_episodes_by_stream_end_one_probe)
    joined["audit_delta"] = joined.audited_batches - joined.audited_batches_one_probe
    seed_delta = joined.reset_index().groupby(["seed", "delay", "cost_mode"], as_index=False)[
        ["episode_delta", "audit_delta"]].mean()
    rng = np.random.default_rng(20261118)
    for (delay, mode), group in seed_delta.groupby(["delay", "cost_mode"], sort=True):
        for metric in ("episode_delta", "audit_delta"):
            v = group.sort_values("seed")[metric].to_numpy(float)
            boot = v[rng.integers(0, len(v), size=(BOOTSTRAP_REPLICATES, len(v)))].mean(axis=1)
            contrasts.append({"contrast": "two_probe_minus_one_probe", "delay": int(delay),
                              "cost_mode": mode, "metric": metric, "mean_difference": float(v.mean()),
                              "lower": float(np.quantile(boot, .025)), "upper": float(np.quantile(boot, .975)),
                              "paired_seeds": len(v), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                              "bootstrap_unit": "seed after averaging seven matched conditions and three budgets"})
    pd.DataFrame(contrasts).to_csv(out / "paired_one_probe_contrasts.csv", index=False)
    sources = [RUN_DIR / "online_batches.csv", RUN_DIR / "released_verification.csv",
               Path(__file__), ROOT / "scripts/run_adwin_audit_baseline.py", ROOT / "pyproject.toml"]
    output_files = ["two_probe_per_seed.csv", "two_probe_decisions.csv", "two_probe_summary.csv",
                    "paired_one_probe_contrasts.csv"]
    lock = {
        "status": "exploratory_complete", "protocol": "river-adwin-two-early-probes-sensitivity-v1",
        "motivation": "Post hoc follow-up to the strict-cap one-probe ADWIN deadlock; not confirmatory.",
        "adwin_version": river.__version__, "adwin_parameters": {"delta": .002, "clock": 1,
            "max_buckets": 5, "min_window_length": 5, "grace_period": 10},
        "bootstrap_clocks": list(BOOTSTRAP_CLOCKS), "bootstrap_rule": "two earliest batches, within the budget",
        "selection_signal": "two initial probes or an ADWIN event from selected, released row-level errors",
        "release_clock": "source_batch + delay < current_decision_clock",
        "evaluation_seeds": sorted(map(int, result.seed.unique())),
        "conditions": sorted(map(str, result.condition.unique())), "cells": len(result),
        "decision_rows": len(decisions), "elapsed_seconds": elapsed,
        "warnings": [
            "This schedule was devised after inspecting the one-probe result and is exploratory.",
            "It reuses the same predictions and labels; it is not independent confirmation.",
            "Two early probes consume two of the available audit cohorts; strict-cap policy volume differs from one-probe ADWIN.",
            "ADWIN monitors supervised error and cannot detect input drift without labels.",
        ],
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "output_sha256": {name: sha(out / name) for name in output_files},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "river": river.__version__},
    }
    (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Completed exploratory two-probe ADWIN sensitivity: {len(result):,} cells, {len(decisions):,} clocks, {elapsed:.2f}s")


if __name__ == "__main__":
    main()
