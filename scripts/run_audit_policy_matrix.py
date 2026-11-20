"""Run a locked, delayed-label acquisition-policy matrix on cached streams.

This is an outcome-informed policy simulation over preserved predictions. The
policy receives current features and only selected labels whose release clock
is strictly before the current decision clock. The evaluator truth table is
used only after the complete policy schedule has been generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV_RUNS = ("development_v1", "development_remaining_v1")
FINAL_RUN = "final_v1"
POLICIES = ("uniform", "input", "concern", "released_error", "conjunction", "disjunction")
COST_MODES = ("strict_cap", "exact_quota")
BUDGETS = (2, 4, 8)
DELAYS = (0, 5, 20, 100)
STREAM_BATCHES = 24
BATCH_SIZE = 500
HARM_MARGIN = 0.05
BOOTSTRAP_REPLICATES = 2_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contiguous_episodes(harm: np.ndarray) -> list[tuple[int, int]]:
    episodes: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(harm, start=1):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(harm)):
            end = i if value and i == len(harm) else i - 1
            episodes.append((start, end))
            start = None
    return episodes


def decision_schedule(
    group: pd.DataFrame,
    truth: pd.DataFrame,
    policy: str,
    delay: int,
    budget: int,
    cost_mode: str,
    input_threshold: float,
    concern_threshold: float,
) -> tuple[list[dict], dict]:
    """Make sequential decisions without exposing unreleased evaluator rows."""
    g = group.sort_values("batch").reset_index(drop=True)
    if len(g) != STREAM_BATCHES or not np.array_equal(g.batch.to_numpy(), np.arange(1, 25)):
        raise ValueError("each policy stream must contain the locked 24 batches")
    outcomes = truth.loc[g.cohort.to_list()]
    harm = outcomes.harm.to_numpy(bool)
    accuracy = outcomes.accuracy.to_numpy(float)
    reference = float(g.reference_accuracy.iloc[0])
    input_score = -np.log10(np.maximum(g.min_adjusted_p.to_numpy(float), 1e-300))
    concern_score = g.cal_concern.to_numpy(float)
    selected: list[int] = []
    decisions: list[dict] = []
    probe_clock = int(np.ceil(STREAM_BATCHES / budget))
    uniform_clocks = set(np.ceil(np.arange(1, budget + 1) * STREAM_BATCHES / budget).astype(int))

    for clock in range(1, STREAM_BATCHES + 1):
        # The decision precedes this clock's label release. A selected source s
        # is usable only when s + delay < clock (strict, including delay zero).
        released_sources = [s for s in selected if s + delay < clock]
        released_drops = [reference - accuracy[s - 1] for s in released_sources]
        released_error = max(released_drops, default=None)
        input_flag = bool(input_score[clock - 1] >= input_threshold)
        concern_flag = bool(concern_score[clock - 1] >= concern_threshold)
        error_flag = released_error is not None and released_error > HARM_MARGIN
        reasons = []
        if policy == "uniform":
            signal = clock in uniform_clocks
            if signal:
                reasons.append("uniform_schedule")
        elif policy == "input":
            signal = input_flag
            if input_flag:
                reasons.append("input_signal")
        elif policy == "concern":
            signal = concern_flag
            if concern_flag:
                reasons.append("concern_signal")
        elif policy == "released_error":
            probe = clock == probe_clock
            signal = bool(error_flag or probe)
            if error_flag:
                reasons.append("released_error_signal")
            if probe:
                reasons.append("bootstrap_probe")
        elif policy == "conjunction":
            signal = input_flag and concern_flag
            if signal:
                reasons.append("input_and_concern")
        elif policy == "disjunction":
            probe = clock == probe_clock
            signal = bool(input_flag or concern_flag or error_flag or probe)
            if input_flag:
                reasons.append("input_signal")
            if concern_flag:
                reasons.append("concern_signal")
            if error_flag:
                reasons.append("released_error_signal")
            if probe:
                reasons.append("bootstrap_probe")
        else:
            raise ValueError(f"unknown policy {policy}")

        remaining = budget - len(selected)
        chosen = bool(signal and remaining > 0)
        forced = False
        if cost_mode == "exact_quota" and not chosen and remaining > 0 and (STREAM_BATCHES - clock + 1) <= remaining:
            chosen = True
            forced = True
            reasons.append("quota_tail_fill")
        if not reasons:
            reasons.append("no_signal")
        if chosen:
            selected.append(clock)
        decisions.append({
            "batch": clock, "selected": chosen, "policy_signal": bool(signal),
            "warning_active": bool(signal and policy != "uniform" and "bootstrap_probe" not in reasons),
            "reason": ";".join(reasons), "cost_mode": cost_mode,
            "prior_released_selected_batches": json.dumps(released_sources),
            "released_error_score": released_error,
            "input_score": float(input_score[clock - 1]),
            "concern_score": float(concern_score[clock - 1]),
            "selected_labels_available_at": clock + delay if chosen else None,
            "selected_labels_first_usable_before_decision": clock + delay + 1 if chosen else None,
            "forced_quota_acquisition": forced,
        })

    if cost_mode == "exact_quota" and len(selected) != budget:
        raise AssertionError("exact quota did not fill the locked budget")
    if cost_mode == "strict_cap" and len(selected) > budget:
        raise AssertionError("strict cost cap exceeded")
    if len(set(selected)) != len(selected):
        raise AssertionError("duplicate audit selection")

    episodes = contiguous_episodes(harm)
    selected_harm = [s for s in selected if harm[s - 1]]
    verified_harm = [s for s in selected_harm if s + delay + 1 <= STREAM_BATCHES]
    verified_episode_delays = []
    eventual_episodes = 0
    verified_episodes = 0
    for start, end in episodes:
        hits = [s for s in selected_harm if start <= s <= end]
        if hits:
            eventual_episodes += 1
            release_clock = min(s + delay + 1 for s in hits)
            if release_clock <= STREAM_BATCHES:
                verified_episodes += 1
                verified_episode_delays.append(release_clock - start)
    count = len(selected)
    true_harmful_batches = int(harm.sum())
    warning_batches = sum(bool(row["warning_active"]) for row in decisions)
    outcomes_row = {
        "audited_batches": count,
        "labels_requested": BATCH_SIZE * count,
        "budget_labels": BATCH_SIZE * budget,
        "unspent_budget_cohorts": budget - count,
        "forced_quota_acquisitions": sum(row["forced_quota_acquisition"] for row in decisions),
        "harmful_batches": true_harmful_batches,
        "harmful_audits": len(selected_harm),
        "verified_harmful_audits_by_stream_end": len(verified_harm),
        "unnecessary_audits": count - len(selected_harm),
        "missed_harmful_batches": true_harmful_batches - len(selected_harm),
        "audit_precision": len(selected_harm) / count if count else np.nan,
        "harm_recall": len(selected_harm) / true_harmful_batches if true_harmful_batches else np.nan,
        "true_harmful_episodes": len(episodes),
        "eventually_audited_harmful_episodes": eventual_episodes,
        "verified_harmful_episodes_by_stream_end": verified_episodes,
        "mean_episode_verification_delay": float(np.mean(verified_episode_delays)) if verified_episode_delays else np.nan,
        "warning_batches": warning_batches,
        "warning_fraction": warning_batches / STREAM_BATCHES,
        "selected_batches": json.dumps(selected),
    }
    return decisions, outcomes_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/phase3/audit_policy_matrix_20261117_v1")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    dev_paths = [ROOT / f"results/phase3/{name}/online_batches.csv" for name in DEV_RUNS]
    final_path = ROOT / f"results/phase3/{FINAL_RUN}/online_batches.csv"
    truth_path = ROOT / f"results/phase3/{FINAL_RUN}/released_verification.csv"
    dev = pd.concat([pd.read_csv(path) for path in dev_paths], ignore_index=True)
    final = pd.read_csv(final_path)
    truth_raw = pd.read_csv(truth_path)
    truth = truth_raw[truth_raw.delay == 0].set_index("cohort", verify_integrity=True)
    if dev.duplicated(["seed", "condition", "batch"]).any():
        raise ValueError("duplicate development batches")
    if final.duplicated(["seed", "condition", "batch"]).any():
        raise ValueError("duplicate final batches")
    if not set(dev.seed).isdisjoint(final.seed):
        raise ValueError("development and final seed banks overlap")
    if set(final.cohort) != set(truth.index):
        raise ValueError("cached prediction and evaluator cohort keys differ")

    for frame in (dev, final):
        frame["input_score"] = -np.log10(np.maximum(frame.min_adjusted_p.to_numpy(float), 1e-300))
    thresholds = {}
    for budget in BUDGETS:
        q = 1 - budget / STREAM_BATCHES
        thresholds[str(budget)] = {
            "input": float(np.quantile(dev.input_score, q, method="higher")),
            "concern": float(np.quantile(dev.cal_concern, q, method="higher")),
        }
    dev_seeds = sorted(map(int, dev.seed.unique()))
    final_seeds = sorted(map(int, final.seed.unique()))
    if len(final_seeds) != 30:
        raise ValueError(f"expected 30 locked evaluation seeds, found {len(final_seeds)}")

    policy_rows: list[dict] = []
    decision_rows: list[dict] = []
    for (seed, condition), group in final.groupby(["seed", "condition"], sort=True):
        for delay in DELAYS:
            for budget in BUDGETS:
                cut = thresholds[str(budget)]
                for cost_mode in COST_MODES:
                    for policy in POLICIES:
                        decisions, metrics = decision_schedule(
                            group, truth, policy, delay, budget, cost_mode,
                            cut["input"], cut["concern"],
                        )
                        policy_rows.append({
                            "seed": int(seed), "condition": condition, "delay": delay,
                            "budget_cohorts": budget, "cost_mode": cost_mode,
                            "policy": policy, **metrics,
                        })
                        decision_rows.extend({
                            "seed": int(seed), "condition": condition, "delay": delay,
                            "budget_cohorts": budget, "cost_mode": cost_mode,
                            "policy": policy, **decision,
                        } for decision in decisions)

    args.out.mkdir(parents=True)
    per_seed = pd.DataFrame(policy_rows)
    decisions = pd.DataFrame(decision_rows)
    per_seed.to_csv(args.out / "policy_per_seed.csv", index=False)
    decisions.to_csv(args.out / "online_audit_decisions.csv", index=False)
    summary = per_seed.groupby(["condition", "delay", "budget_cohorts", "cost_mode", "policy"],
                               as_index=False).mean(numeric_only=True)
    summary.to_csv(args.out / "policy_summary.csv", index=False)

    rng = np.random.default_rng(17112026)
    paired = []
    for key, group in per_seed.groupby(["condition", "delay", "budget_cohorts", "cost_mode"], sort=True):
        base = group[group.policy == "uniform"].set_index("seed").sort_index()
        for policy in POLICIES[1:]:
            other = group[group.policy == policy].set_index("seed").loc[base.index]
            for metric in ("harmful_audits", "verified_harmful_episodes_by_stream_end",
                           "unnecessary_audits", "warning_fraction"):
                delta = other[metric].to_numpy(float) - base[metric].to_numpy(float)
                boot = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))].mean(axis=1)
                paired.append({
                    "condition": key[0], "delay": key[1], "budget_cohorts": key[2],
                    "cost_mode": key[3], "policy": policy,
                    "metric": metric + "_difference_vs_uniform", "mean": float(delta.mean()),
                    "lower": float(np.quantile(boot, .025)), "upper": float(np.quantile(boot, .975)),
                    "paired_seeds": len(delta), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                })
    pd.DataFrame(paired).to_csv(args.out / "paired_seed_bootstrap.csv", index=False)

    sources = [*dev_paths, final_path, truth_path, Path(__file__)]
    output_names = ("policy_per_seed.csv", "online_audit_decisions.csv", "policy_summary.csv", "paired_seed_bootstrap.csv")
    lock = {
        "status": "complete", "protocol": "delayed-audit-policy-matrix-v1",
        "outcome_informed": True,
        "development_seeds": dev_seeds, "final_seeds": final_seeds,
        "conditions": sorted(map(str, final.condition.unique())),
        "policies": list(POLICIES), "cost_modes": list(COST_MODES),
        "budgets_cohorts": list(BUDGETS), "labels_per_cohort": BATCH_SIZE,
        "delays_batches": list(DELAYS), "stream_batches": STREAM_BATCHES,
        "input_thresholds": thresholds,
        "combined_conjunction": "development-quantile input score AND development-quantile concern score for the budget rate",
        "disjunction": "input OR concern OR released reference-minus-accuracy greater than 0.05; fixed bootstrap probe at ceil(24/budget) for label evidence",
        "released_error_policy": "fixed bootstrap probe at ceil(24/budget); then maximum reference-minus-accuracy among selected labels satisfying source_batch + delay < current_decision_clock exceeds 0.05",
        "strict_cap": "select only while the evidence rule is active and budget remains; unused budget is retained",
        "exact_quota": "same evidence rule plus forced selection in the last remaining clocks when necessary to spend exactly the quota; forced acquisitions are separately counted",
        "warning_fraction": "fraction of decision clocks with the policy evidence signal active; bootstrap probes and forced quota fills are excluded",
        "confirmation_clock": "selected source_batch + delay + 1; this respects source_batch + delay < decision_clock",
        "primary_outcome": "harm is evaluator accuracy more than 0.05 below the frozen historical reference; strictly greater boundary",
        "episodes": "maximal contiguous runs of harmful cohorts in the 24-batch evaluator horizon",
        "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "seed within condition; conditions are not independent domains",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "output_sha256": {name: sha(args.out / name) for name in output_names},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": [
            "Outcome-informed replay on one preserved final synthetic run, not fresh confirmation.",
            "Threshold selection uses five development seeds.",
            "No operational acquisition cost or analyst benefit is measured.",
            "At delay 100, no selected label is released within the 24-batch decision horizon.",
        ],
    }
    (args.out / "protocol_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(per_seed):,} policy cells and {len(decisions):,} sequential decisions in {args.out}")


if __name__ == "__main__":
    main()
