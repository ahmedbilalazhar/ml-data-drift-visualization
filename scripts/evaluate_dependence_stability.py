"""Seed-level evaluator for the locked dependence stability experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def seed_bootstrap(values, rng, reps=2000):
    x = np.asarray(values, dtype=float)
    draw = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(draw, .025)), float(np.quantile(draw, .975))


def evaluate(run: Path) -> dict:
    run = run.resolve()
    batches = pd.read_csv(run / "online_dependence_stability.csv")
    lock = json.loads((run / "threshold_used.json").read_text(encoding="utf-8"))
    threshold = {int(k): float(v) for k, v in lock["thresholds"].items()}
    seeds = list(range(931100, 931150))
    scenarios = ["C0"] + [f"rho{rho:.2f}_pair{i}{j}" for rho in (.30, .60, .90)
                            for i, j in ((0, 1), (2, 5), (6, 7))]
    sizes = (250, 500, 1000)
    expected = len(seeds) * len(scenarios) * len(sizes) * 24
    keys = ["seed", "scenario", "batch_size", "batch"]
    required = {*keys, "dependence_score", "dependence_alarm", "ks_pca_alarm",
                "target_pair", "found_i", "found_j", "intervention_active"}
    if len(batches) != expected or not required.issubset(batches.columns):
        raise ValueError(f"expected {expected} complete batch records with required fields")
    if batches[keys].duplicated().any() or set(batches.seed.unique()) != set(seeds):
        raise ValueError("duplicate or unexpected held-out seed/batch keys")
    if set(batches.scenario.unique()) != set(scenarios) or set(batches.batch_size.unique()) != set(sizes):
        raise ValueError("scenario or batch-size grid differs from protocol")
    if not np.isfinite(batches.dependence_score).all():
        raise ValueError("non-finite detector scores")
    if not (batches.dependence_alarm.astype(bool) ==
            (batches.dependence_score >= batches.batch_size.map(threshold))).all():
        raise ValueError("alarm flags disagree with frozen size-specific thresholds")
    if not (batches.intervention_active.astype(bool) == batches.batch.ge(9).where(
            batches.scenario.ne("C0"), False)).all():
        raise ValueError("intervention clock differs from the locked batch-9 onset")

    seed_rows = []
    for (seed, scenario, n), group in batches.groupby(["seed", "scenario", "batch_size"], sort=True):
        group = group.sort_values("batch")
        onset = group[group.batch.between(9, 12)]
        pre = group[group.batch.lt(9)]
        dep_batches = onset.loc[onset.dependence_alarm.astype(bool), "batch"]
        ks_batches = onset.loc[onset.ks_pca_alarm.astype(bool), "batch"]
        target = str(group.target_pair.iloc[0])
        seed_rows.append(dict(seed=int(seed), scenario=scenario, batch_size=int(n), target_pair=target,
            dependence_detected=not dep_batches.empty, ks_detected=not ks_batches.empty,
            dependence_delay=(int(dep_batches.iloc[0])-9 if len(dep_batches) else np.nan),
            ks_delay=(int(ks_batches.iloc[0])-9 if len(ks_batches) else np.nan),
            pair_recovered=bool(len(dep_batches) and
                ((onset.loc[onset.dependence_alarm.astype(bool)].iloc[0].found_i,
                  onset.loc[onset.dependence_alarm.astype(bool)].iloc[0].found_j) ==
                 tuple(map(int, target.split("-"))))),
            dep_pre_alarms=int(pre.dependence_alarm.astype(bool).sum()),
            ks_pre_alarms=int(pre.ks_pca_alarm.astype(bool).sum()),
            dep_alarms_24=int(group.dependence_alarm.astype(bool).sum()),
            ks_alarms_24=int(group.ks_pca_alarm.astype(bool).sum())))
    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(run / "per_seed_metrics.csv", index=False)

    rng = np.random.default_rng(20261116)
    setting_rows = []
    for (scenario, n), group in per_seed[per_seed.scenario.ne("C0")].groupby(
            ["scenario", "batch_size"], sort=True):
        for method, detected, delay, pre, alarm in [
            ("max_pairwise_correlation", "dependence_detected", "dependence_delay", "dep_pre_alarms", "dep_alarms_24"),
            ("marginal_plus_PCA_Holm_KS", "ks_detected", "ks_delay", "ks_pre_alarms", "ks_alarms_24")]:
            mean, lo, hi = seed_bootstrap(group[detected].astype(float).to_numpy(), rng)
            delays = group.loc[group[detected], delay].dropna().to_numpy(dtype=float)
            setting_rows.append(dict(scenario=scenario, batch_size=int(n), method=method, seeds=len(group),
                onset_detected=int(group[detected].sum()), detection_rate=mean,
                detection_rate_ci95_low=lo, detection_rate_ci95_high=hi,
                missed_seeds=int((~group[detected]).sum()),
                mean_delay_given_detection=(float(delays.mean()) if len(delays) else np.nan),
                pair_recovery_rate=(float(group.pair_recovered.mean()) if method == "max_pairwise_correlation" else np.nan),
                pre_onset_alarms_per_8_batches=float(group[pre].mean()),
                mean_alarms_per_24_batches=float(group[alarm].mean())))
        dep = group.dependence_detected.astype(float).to_numpy()
        ks = group.ks_detected.astype(float).to_numpy()
        paired_mean, paired_lo, paired_hi = seed_bootstrap(dep-ks, rng)
        setting_rows.append(dict(scenario=scenario, batch_size=int(n), method="paired_difference_dependence_minus_KS",
            seeds=len(group), onset_detected=np.nan, detection_rate=paired_mean,
            detection_rate_ci95_low=paired_lo, detection_rate_ci95_high=paired_hi,
            missed_seeds=np.nan, mean_delay_given_detection=np.nan, pair_recovery_rate=np.nan,
            pre_onset_alarms_per_8_batches=np.nan, mean_alarms_per_24_batches=np.nan))
    settings = pd.DataFrame(setting_rows)
    settings.to_csv(run / "setting_summary.csv", index=False)

    stationary = []
    c0 = per_seed[per_seed.scenario.eq("C0")]
    for n, group in c0.groupby("batch_size"):
        for method, alarm in [("max_pairwise_correlation", "dep_alarms_24"),
                              ("marginal_plus_PCA_Holm_KS", "ks_alarms_24")]:
            mean, lo, hi = seed_bootstrap(group[alarm].to_numpy(dtype=float), rng)
            stationary.append(dict(batch_size=int(n), method=method, seeds=len(group),
                mean_false_alarms_per_24_batches=mean, ci95_low=lo, ci95_high=hi,
                batches_per_seed=24))
    stationary = pd.DataFrame(stationary)
    stationary.to_csv(run / "stationary_burden.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, n in zip(axes, sizes):
        for method, label, color, marker in [
            ("max_pairwise_correlation", "Pairwise correlation", "#167d83", "o"),
            ("marginal_plus_PCA_Holm_KS", "KS + PCA", "#d07845", "s")]:
            sub = settings[(settings.batch_size.eq(n)) & settings.method.eq(method)]
            for pair_idx, pair in enumerate(("0-1", "2-5", "6-7")):
                ss = sub[sub.scenario.str.endswith(f"pair{pair.replace('-', '')}")].copy()
                ss["rho"] = ss.scenario.str.extract(r"rho([0-9.]+)").astype(float)
                ss = ss.sort_values("rho")
                ax.plot(ss.rho, ss.detection_rate, color=color, marker=marker,
                        linestyle=["-", "--", ":"][pair_idx], alpha=.85,
                        label=f"{label}, pair {pair}" if n == sizes[0] else None)
                ax.fill_between(ss.rho.to_numpy(), ss.detection_rate_ci95_low.to_numpy(),
                                ss.detection_rate_ci95_high.to_numpy(), color=color, alpha=.08)
        ax.set(title=f"Batch size {n}", xlabel="Intervention correlation (rho)", ylim=(-.03, 1.03))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Detection by batch 12 (50 held-out seeds)")
    axes[-1].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Dependence detector stability across selected Gaussian interventions", y=1.02)
    fig.tight_layout()
    fig.savefig(run / "dependence_stability.png", dpi=180, bbox_inches="tight")
    fig.savefig(run / "dependence_stability.pdf", bbox_inches="tight")
    fig.savefig(run / "dependence_stability.svg", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Dependence detector stability results", "",
        "Outcome-informed exploratory extension; the unit is the held-out generator seed. Batch size, pair and strength are repeated settings, not extra replicates.", "",
        "## Held-out detection by batch 12", "",
        "| Setting | n | Detector | Detected / 50 | Rate (95% seed bootstrap) | Mean delay when detected | Correct pair rate |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in settings[settings.method.ne("paired_difference_dependence_minus_KS")].itertuples():
        pair_rate = "—" if pd.isna(r.pair_recovery_rate) else f"{r.pair_recovery_rate:.2f}"
        detected = f"{int(r.onset_detected)}/50"
        delay = "—" if pd.isna(r.mean_delay_given_detection) else f"{r.mean_delay_given_detection:.2f} batches"
        lines.append(f"| {r.scenario} | {r.batch_size} | {r.method} | {detected} | {r.detection_rate:.2f} ({r.detection_rate_ci95_low:.2f}–{r.detection_rate_ci95_high:.2f}) | {delay} | {pair_rate} |")
    lines += ["", "## Stationary alarm burden", "", "| n | Detector | Mean alarms / 24 batches (95% seed bootstrap) |", "|---:|---|---:|"]
    for r in stationary.itertuples():
        lines.append(f"| {r.batch_size} | {r.method} | {r.mean_false_alarms_per_24_batches:.2f} ({r.ci95_low:.2f}–{r.ci95_high:.2f}) |")
    lines += ["", "## Interpretation", "",
        "All intervals resample the 50 held-out seeds. The study evaluates only the programmed Gaussian pairwise-correlation changes with preserved univariate marginals. It does not establish arbitrary dependence detection, causal localization, model-harm prediction, or natural-stream validity.", ""]
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    validation = dict(status="passed", rows=len(batches), expected_rows=expected,
        heldout_seeds=len(seeds), resampling_unit="seed", bootstrap_replicates=2000,
        batch_sizes=list(sizes), scenarios=scenarios, onset_batch=9, detection_window=[9, 12],
        evaluator_truth_absent=True, threshold_lock_sha256=__import__("hashlib").sha256(
            (run / "threshold_used.json").read_bytes()).hexdigest(),
        setting_rows=len(settings), stationary_rows=len(stationary))
    (run / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    evaluate(parser.parse_args().run)
