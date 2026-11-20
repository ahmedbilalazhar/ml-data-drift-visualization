"""Run canonical River ADWIN on the same delayed, budgeted audit cohorts.

ADWIN consumes row-level 0/1 prediction errors only when the corresponding
selected cohort's labels have been released. It does not see unselected or
unreleased labels. A fixed bootstrap audit is shared with the released-error
baseline; all arms respect the same cohort caps and exact-quota tail rule.
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

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/phase3/final_v1"
DELAYS = (0, 5, 20, 100)
BUDGETS = (2, 4, 8)
COST_MODES = ("strict_cap", "exact_quota")
STREAM_BATCHES = 24
BATCH_SIZE = 500
BOOTSTRAP_REPLICATES = 2_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episodes(harm: np.ndarray) -> list[tuple[int, int]]:
    result = []
    start = None
    for i, value in enumerate(harm, 1):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(harm)):
            result.append((start, i if value and i == len(harm) else i - 1))
            start = None
    return result


def make_schedule(group: pd.DataFrame, row_errors: np.ndarray, truth: pd.DataFrame,
                  delay: int, budget: int, cost_mode: str) -> tuple[list[dict], dict]:
    """Sequentially update ADWIN from only released, previously selected cohorts."""
    g = group.sort_values("batch").reset_index(drop=True)
    if len(g) != STREAM_BATCHES or not np.array_equal(g.batch.to_numpy(), np.arange(1, 25)):
        raise ValueError("expected batches 1..24")
    if row_errors.shape != (STREAM_BATCHES, BATCH_SIZE):
        raise ValueError(f"expected 24x500 error matrix, got {row_errors.shape}")
    outcomes = truth.loc[g.cohort.to_list()]
    harm = outcomes.harm.to_numpy(bool)
    accuracy = outcomes.accuracy.to_numpy(float)
    selected: list[int] = []
    consumed: set[int] = set()
    decisions = []
    adwin = ADWIN(delta=0.002, clock=1)
    update_count = 0
    detection_count = 0
    probe_clock = int(np.ceil(STREAM_BATCHES / budget))

    for clock in range(1, STREAM_BATCHES + 1):
        released = [s for s in selected if s + delay < clock and s not in consumed]
        batch_events = []
        for source in released:
            for row_index, error in enumerate(row_errors[source - 1]):
                adwin.update(float(error))
                update_count += 1
                if adwin.drift_detected:
                    detection_count += 1
                    batch_events.append({"source_batch": source, "row_index_1based": row_index + 1,
                                         "release_decision_clock": clock,
                                         "estimation": float(adwin.estimation),
                                         "window_width": int(adwin.width)})
            consumed.add(source)
        probe = clock == probe_clock
        adwin_signal = bool(batch_events)
        signal = adwin_signal or probe
        remaining = budget - len(selected)
        choose = bool(signal and remaining > 0)
        forced = False
        if cost_mode == "exact_quota" and not choose and remaining > 0 and STREAM_BATCHES - clock + 1 <= remaining:
            choose = True
            forced = True
        if choose:
            selected.append(clock)
        decisions.append({
            "batch": clock, "selected": choose, "policy_signal": signal,
            "adwin_drift_events_from_released_labels": len(batch_events),
            "bootstrap_probe": probe, "forced_quota_acquisition": forced,
            "new_released_source_batches": json.dumps(released),
            "labels_processed_so_far": update_count, "adwin_detections_so_far": detection_count,
            "adwin_window_width": int(adwin.width), "adwin_error_estimate": float(adwin.estimation),
            "selected_labels_available_at": clock + delay if choose else None,
            "selected_labels_first_usable_clock": clock + delay + 1 if choose else None,
        })

    if cost_mode == "strict_cap" and len(selected) > budget:
        raise AssertionError("ADWIN exceeded the strict audit cap")
    if cost_mode == "exact_quota" and len(selected) != budget:
        raise AssertionError("ADWIN exact quota failed")
    ep = episodes(harm)
    harmful_audits = [s for s in selected if harm[s - 1]]
    verified = [s for s in harmful_audits if s + delay + 1 <= STREAM_BATCHES]
    verified_ep_delays = []
    eventually_hit = 0
    verified_hit = 0
    for start, end in ep:
        hits = [s for s in harmful_audits if start <= s <= end]
        if hits:
            eventually_hit += 1
            release = min(s + delay + 1 for s in hits)
            if release <= STREAM_BATCHES:
                verified_hit += 1
                verified_ep_delays.append(release - start)
    n = len(selected)
    n_harm = int(harm.sum())
    metrics = {
        "audited_batches": n, "labels_requested": n * BATCH_SIZE,
        "budget_labels": budget * BATCH_SIZE, "unspent_budget_cohorts": budget - n,
        "forced_quota_acquisitions": sum(d["forced_quota_acquisition"] for d in decisions),
        "harmful_batches": n_harm, "harmful_audits": len(harmful_audits),
        "verified_harmful_audits_by_stream_end": len(verified),
        "unnecessary_audits": n - len(harmful_audits),
        "missed_harmful_batches": n_harm - len(harmful_audits),
        "audit_precision": len(harmful_audits) / n if n else np.nan,
        "harm_recall": len(harmful_audits) / n_harm if n_harm else np.nan,
        "true_harmful_episodes": len(ep), "eventually_audited_harmful_episodes": eventually_hit,
        "verified_harmful_episodes_by_stream_end": verified_hit,
        "mean_episode_verification_delay": float(np.mean(verified_ep_delays)) if verified_ep_delays else np.nan,
        "adwin_row_updates": update_count, "adwin_detection_events": detection_count,
        "selected_batches": json.dumps(selected),
    }
    return decisions, metrics


def row_errors_for(seed: int, condition: str) -> np.ndarray:
    pred_file = RUN_DIR / f"predictions/{seed}_{condition}.npz"
    truth_file = RUN_DIR / f"evaluator_truth/{seed}_{condition}.npz"
    pred = np.load(pred_file)
    labels = np.load(truth_file)
    pred_ids = pred["ids"].reshape(-1)
    true_ids = labels["ids"].reshape(-1)
    if not np.array_equal(pred_ids, true_ids):
        raise ValueError(f"prediction/truth IDs differ for {seed}/{condition}")
    errors = pred["prediction"].reshape(-1) != labels["labels"]
    return errors.reshape(STREAM_BATCHES, BATCH_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/phase3/adwin_audit_baseline_20261117_v1")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    final = pd.read_csv(RUN_DIR / "online_batches.csv")
    truth_raw = pd.read_csv(RUN_DIR / "released_verification.csv")
    truth = truth_raw[truth_raw.delay == 0].set_index("cohort", verify_integrity=True)
    if len(final.seed.unique()) != 30 or len(final.condition.unique()) != 7:
        raise ValueError("locked evaluation must contain 30 seeds and 7 conditions")
    source_data = [RUN_DIR / "online_batches.csv", RUN_DIR / "released_verification.csv"]
    per_seed_rows = []
    decision_rows = []
    started = time.perf_counter()
    for (seed, condition), group in final.groupby(["seed", "condition"], sort=True):
        errors = row_errors_for(int(seed), str(condition))
        for delay in DELAYS:
            for budget in BUDGETS:
                for cost_mode in COST_MODES:
                    decisions, metrics = make_schedule(group, errors, truth, delay, budget, cost_mode)
                    per_seed_rows.append({"seed": int(seed), "condition": condition, "delay": delay,
                                          "budget_cohorts": budget, "cost_mode": cost_mode,
                                          "policy": "river_adwin", **metrics})
                    decision_rows.extend({"seed": int(seed), "condition": condition, "delay": delay,
                                          "budget_cohorts": budget, "cost_mode": cost_mode,
                                          "policy": "river_adwin", **row} for row in decisions)
    elapsed = time.perf_counter() - started
    out.mkdir(parents=True)
    per_seed = pd.DataFrame(per_seed_rows)
    decisions = pd.DataFrame(decision_rows)
    per_seed.to_csv(out / "adwin_per_seed.csv", index=False)
    decisions.to_csv(out / "adwin_decisions.csv", index=False)
    summary = per_seed.groupby(["delay", "budget_cohorts", "cost_mode"], as_index=False).mean(numeric_only=True)
    summary.to_csv(out / "adwin_summary.csv", index=False)

    policies = pd.read_csv(ROOT / "results/phase3/audit_policy_matrix_20261117_v1/policy_per_seed.csv")
    merge_key = ["seed", "condition", "delay", "budget_cohorts", "cost_mode"]
    uniform = policies[policies.policy == "uniform"].set_index(merge_key)
    released = policies[policies.policy == "released_error"].set_index(merge_key)
    adwin = per_seed.set_index(merge_key)
    comparisons = []
    rng = np.random.default_rng(20261117)
    for other_name, other in (("uniform", uniform), ("released_error", released)):
        for (delay, budget, mode), group in adwin.groupby(level=["delay", "budget_cohorts", "cost_mode"], sort=True):
            idx = group.index
            baseline = other.loc[idx]
            for metric in ("verified_harmful_episodes_by_stream_end", "unnecessary_audits", "audited_batches"):
                delta = group[metric].to_numpy(float) - baseline[metric].to_numpy(float)
                # Preserve seed pairing while averaging all seven conditions first.
                frame = pd.DataFrame({"seed": group.index.get_level_values("seed"), "delta": delta})
                per_seed_delta = frame.groupby("seed").delta.mean().to_numpy(float)
                boot = per_seed_delta[rng.integers(0, len(per_seed_delta), size=(BOOTSTRAP_REPLICATES, len(per_seed_delta)))].mean(axis=1)
                comparisons.append({"comparison": f"river_adwin_minus_{other_name}", "delay": int(delay),
                                    "budget_cohorts": int(budget), "cost_mode": mode, "metric": metric,
                                    "mean_difference": float(per_seed_delta.mean()),
                                    "lower": float(np.quantile(boot, .025)), "upper": float(np.quantile(boot, .975)),
                                    "paired_seeds": len(per_seed_delta), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                                    "bootstrap_unit": "seed after averaging seven matched conditions"})
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)

    sources = [*source_data, ROOT / "results/phase3/final_v1/evaluator_truth/100_C0.npz",
               ROOT / "results/phase3/final_v1/predictions/100_C0.npz", Path(__file__),
               ROOT / "pyproject.toml"]
    outputs = ["adwin_per_seed.csv", "adwin_decisions.csv", "adwin_summary.csv", "paired_comparisons.csv"]
    lock = {
        "status": "complete", "protocol": "river-adwin-delayed-budgeted-audit-v1",
        "river_version": river.__version__, "detector": "river.drift.ADWIN",
        "parameters": {"delta": 0.002, "clock": 1, "max_buckets": 5,
                       "min_window_length": 5, "grace_period": 10,
                       "note": "Only clock=1 differs from River defaults, so checking occurs after every received error."},
        "input": "row-level binary error from cached frozen predictions; values are ingested only from selected cohorts after release",
        "label_release_rule": "source_batch + delay < current decision clock",
        "initial_probe": "ceil(24 / budget_cohorts), same as released-error baseline",
        "budgets": list(BUDGETS), "delays": list(DELAYS), "cost_modes": list(COST_MODES),
        "batch_size": BATCH_SIZE, "stream_batches": STREAM_BATCHES,
        "evaluation_seeds": sorted(map(int, per_seed.seed.unique())),
        "conditions": sorted(map(str, per_seed.condition.unique())),
        "policy_cells": len(per_seed), "decision_rows": len(decisions),
        "elapsed_seconds": elapsed,
        "limitations": [
            "Same synthetic cached predictions as the primary run; not an independent dataset or fresh model rerun.",
            "ADWIN is supervised and can update only on audited, released rows; sparse audit feedback may make detection underpowered.",
            "This baseline monitors prediction error and does not detect input drift without labels.",
            "The strict-cap policy can consume fewer cohorts than uniform; event-yield contrasts are not equal-spend unless forced exact-quota mode is used.",
            "ADWIN parameters are fixed at River defaults except clock=1 to check each arriving error; no final-seed tuning was performed.",
        ],
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "output_sha256": {name: sha(out / name) for name in outputs},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "river": river.__version__},
    }
    (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(per_seed):,} ADWIN policy cells and {len(decisions):,} decisions in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
