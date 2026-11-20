"""Summarize and report a complete DDE-style subset audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase3/dde_subset_audit_20261118_v4"
OUT = ROOT / "reports/phase3/dde_subset_audit_20261118_v4"
CONFIG = ROOT / "configs/dde_subset_audit_v1.json"

COLORS = {"dde": "#087E8B", "correlation_only": "#E07A2D",
          "direct_full_ks_hidden": "#315A7D", "random": "#8C6BB1"}
LABELS = {"dde": "DDE-style inference", "correlation_only": "Mean |correlation|",
          "direct_full_ks_hidden": "Full KS, scored hidden", "random": "Random ranking"}
MECH_LABELS = {"M1_abrupt_nuisance": "Abrupt nuisance", "M2_gradual_predictive": "Gradual predictive",
               "M3_incremental_repeated_small": "Repeated small", "M7_out_of_support": "Out-of-support"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def boot_mean(values: np.ndarray, seed: int, reps: int = 2000) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(x.mean())
    if len(x) == 1:
        return mean, float("nan"), float("nan"), 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(reps, len(x)))
    means = x[draws].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return mean, float(lo), float(hi), int(len(x))


def require_complete(config: dict, manifest: dict, calibration: dict,
                     raw: pd.DataFrame) -> None:
    if manifest.get("status") != "complete":
        raise RuntimeError(f"runner status is {manifest.get('status')!r}; report requires complete run")
    expect_eval = len(config["tasks"]) * len(config["dimensions"]) * len(config["evaluation_seeds"])
    expect_val = len(config["tasks"]) * len(config["dimensions"]) * len(config["validation_seeds"])
    if manifest.get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("run manifest config hash differs from current config")
    if manifest.get("evaluation_cells_completed") != expect_eval or manifest.get("validation_cells_completed") != expect_val:
        raise RuntimeError("manifest cell counts do not match locked design")
    expected_cells = set()
    for partition, seeds in (("validation", config["validation_seeds"]),
                              ("evaluation", config["evaluation_seeds"])):
        expected_cells.update((partition, task, dim, seed)
                              for task in config["tasks"] for dim in config["dimensions"] for seed in seeds)
    if set(calibration["values"]) != {f"{task}|{dim}" for task in config["tasks"] for dim in config["dimensions"]}:
        raise RuntimeError("development calibration is missing task/dimension strata")
    expected_rows = len(config["evaluation_check_batches"]) * len(config["evaluation_mechanisms"]) * 4
    for (partition, task, dim, seed), group in raw.groupby(["partition", "task", "dimension", "seed"]):
        if len(group) != expected_rows:
            raise RuntimeError(f"incomplete seed cell: {(partition, task, dim, seed)} has {len(group)} rows")
    observed_cells = set(map(tuple, raw[["partition", "task", "dimension", "seed"]]
                             .drop_duplicates().itertuples(index=False, name=None)))
    if observed_cells != expected_cells:
        raise RuntimeError(f"seed cell set mismatch: missing={len(expected_cells-observed_cells)}, extra={len(observed_cells-expected_cells)}")
    if raw["online_evaluator_truth_access"].astype(bool).any():
        raise RuntimeError("an online row claims evaluator truth access")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((RUN / "development_calibration.json").read_text(encoding="utf-8"))
    paths = sorted(RUN.glob("*_seed_*.csv"))
    if not paths:
        raise RuntimeError("no seed outputs found")
    raw = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    require_complete(config, manifest, calibration, raw)
    OUT.mkdir(parents=True, exist_ok=True)

    metric_cols = ["average_precision", "auroc", "precision_at_k"]
    active = raw[(raw["mechanism"] != "M0_stationary") & (raw["target_feature_count"] > 0)].copy()
    # First average correlated temporal checkpoints within a seed. Seeds, not
    # repeated batches or feature counts, are the inferential units.
    per_seed = active.groupby(["partition", "task", "dimension", "mechanism", "seed", "method"],
                              as_index=False)[metric_cols].mean()
    per_seed.to_csv(OUT / "per_seed_localization_metrics.csv", index=False)

    summary_rows = []
    group_fields = ["partition", "task", "dimension", "mechanism", "method"]
    for key, group in per_seed.groupby(group_fields):
        row = dict(zip(group_fields, key, strict=True))
        for j, metric in enumerate(metric_cols):
            stable_key = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:4], "big")
            mean, lo, hi, n = boot_mean(group[metric].to_numpy(), seed=7300 + j + stable_key)
            row.update({f"{metric}_mean": mean, f"{metric}_ci_low": lo,
                        f"{metric}_ci_high": hi, f"{metric}_n_seeds": n})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "localization_summary.csv", index=False)

    # Observable inference coverage and workload, including stationary nulls.
    coverage = raw.groupby(["partition", "task", "dimension", "mechanism", "seed"], as_index=False).agg(
        target_hidden_coverage=("target_feature_count", lambda x: float((x > 0).mean())),
        anchor_checkpoint_fraction=("detected_anchor_count", lambda x: float((x > 0).mean())),
        mean_detected_anchors=("detected_anchor_count", "mean"),
        mean_target_features_hidden=("target_feature_count", "mean"),
    )
    coverage.to_csv(OUT / "seed_coverage_and_workload.csv", index=False)

    # Paired, seed-level differences. These are exploratory 95% percentile
    # intervals; no superiority claim or multiplicity-adjusted decision is made.
    eval_seed = per_seed[per_seed["partition"] == "evaluation"]
    effect_rows = []
    for (task, dim, mechanism), group in eval_seed.groupby(["task", "dimension", "mechanism"]):
        wide = group.pivot(index="seed", columns="method", values="average_precision")
        if "dde" not in wide:
            continue
        for comparator in ("correlation_only", "direct_full_ks_hidden", "random"):
            if comparator not in wide:
                continue
            paired = (wide["dde"] - wide[comparator]).dropna().to_numpy()
            mean, lo, hi, n = boot_mean(paired, seed=9100 + len(effect_rows))
            effect_rows.append({"task": task, "dimension": dim, "mechanism": mechanism,
                                "contrast": f"dde - {comparator}", "mean_difference": mean,
                                "ci_low": lo, "ci_high": hi, "n_paired_seeds": n})
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(OUT / "paired_seed_effects.csv", index=False)

    plot_calibration(calibration, config)
    plot_performance(per_seed, config)
    plot_temporal(raw, config)
    write_report(summary, effects, coverage, raw, per_seed, config, manifest, calibration)
    compile_report()
    receipt = {
        "status": "complete", "protocol": config["protocol"],
        "run_manifest_sha256": sha256(RUN / "run_manifest.json"),
        "run_config_sha256": manifest["config_sha256"],
        "input_seed_cells": len(paths), "evaluation_seed_cells": manifest["evaluation_cells_completed"],
        "validation_seed_cells": manifest["validation_cells_completed"],
        "evaluation_seeds": config["evaluation_seeds"], "bootstrap_replicates": 2000,
        "bootstrap_unit": "seed; repeated checkpoints averaged within seed first",
        "worker_count": manifest.get("worker_count"),
        "output_sha256": {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        "visual_reviewed": False,
        "visual_review_status": "pending rendered-page inspection",
    }
    (OUT / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(OUT / "dde_subset_audit.pdf"),
                      "evaluation_cells": manifest["evaluation_cells_completed"],
                      "validation_cells": manifest["validation_cells_completed"],
                      "summary_rows": len(summary), "output": str(OUT)}, indent=2))


def plot_calibration(calibration: dict, config: dict) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 9.2), sharex=True, sharey=True)
    for i, task in enumerate(config["tasks"]):
        for j, dim in enumerate(config["dimensions"]):
            value = calibration["values"][f"{task}|{dim}"]
            ax = axes[i, j]
            supported_curve = np.where(np.asarray(value["pair_count"]) > 0,
                                       np.asarray(value["dde_curve"], dtype=float), np.nan)
            ax.plot(value["correlation_bins"], supported_curve, marker="o", lw=2.2,
                    color=COLORS["dde"], markersize=4)
            ax.axhline(0, color="#AAB6C2", lw=.7)
            ax.set_title(f"{task.replace('_', ' ').title()} · {dim} features", loc="left", fontsize=10, weight="bold")
            ax.text(.98, .96, f"{value['episodes_used']:,} nonempty dev episodes",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#536577")
            ax.grid(axis="y", color="#DCE3EA", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Reference |Pearson correlation| threshold z")
    fig.supylabel("Pooled fraction of eligible pairs with both features flagged")
    fig.suptitle("Development-only empirical co-drift curves", x=.08, ha="left", y=.995,
                 fontsize=15, color="#16324F", weight="bold")
    fig.tight_layout(rect=(.05, .03, 1, .96))
    save_figure(fig, "development_dde_curves")


def summary_lookup(group: pd.DataFrame, method: str, metric: str) -> tuple[float, float, float]:
    rows = group[group["method"] == method]
    if rows.empty:
        return np.nan, np.nan, np.nan
    r = rows.iloc[0]
    return (float(r[f"{metric}_mean"]), float(r[f"{metric}_ci_low"]), float(r[f"{metric}_ci_high"]))


def plot_performance(per_seed: pd.DataFrame, config: dict) -> None:
    active = per_seed[(per_seed["partition"] == "evaluation") & (per_seed["mechanism"] != "M0_stationary")]
    # The panel estimates equal-weighted seed-means across intervention mechanisms.
    agg = active.groupby(["task", "dimension", "seed", "method"], as_index=False).mean(numeric_only=True)
    rng = np.random.default_rng(8192)
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 9.4), sharey=True)
    methods = ["dde", "correlation_only", "direct_full_ks_hidden", "random"]
    for i, task in enumerate(config["tasks"]):
        for j, dim in enumerate(config["dimensions"]):
            ax = axes[i, j]
            block = agg[(agg.task == task) & (agg.dimension == dim)]
            for offset, method in enumerate(methods):
                vals = block[block.method == method]["average_precision"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                mean, lo, hi, _ = boot_mean(vals, seed=int(rng.integers(0, 2**31 - 1)))
                lower = max(0.0, mean - lo) if np.isfinite(lo) else 0.0
                upper = max(0.0, hi - mean) if np.isfinite(hi) else 0.0
                ax.errorbar(offset, mean, yerr=[[lower], [upper]], fmt="o", ms=7,
                            capsize=3.5, lw=1.6, color=COLORS[method], label=LABELS[method])
            ax.set_title(f"{task.replace('_', ' ').title()} · {dim} features", loc="left", fontsize=10, weight="bold")
            ax.set_ylim(-.03, 1.03)
            ax.set_xticks(range(4), ["DDE proxy", "Correlation", "Full KS", "Random"], rotation=16, ha="right", fontsize=8)
            ax.grid(axis="y", color="#DCE3EA", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Average precision over held-out changed features")
    axes[1, 0].set_ylabel("Average precision over held-out changed features")
    axes[2, 0].set_ylabel("Average precision over held-out changed features")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .0), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Held-out feature localization · evaluation seeds", x=.07, ha="left", y=.995,
                 fontsize=15, color="#16324F", weight="bold")
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, "heldout_feature_localization")


def plot_temporal(raw: pd.DataFrame, config: dict) -> None:
    active = raw[(raw.partition == "evaluation") & (raw.mechanism != "M0_stationary") & (raw.target_feature_count > 0)]
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 9.4), sharex=True, sharey=True)
    for i, task in enumerate(config["tasks"]):
        for j, dim in enumerate(config["dimensions"]):
            ax = axes[i, j]
            block = active[(active.task == task) & (active.dimension == dim)]
            for method in ("dde", "direct_full_ks_hidden"):
                method_data = block[block.method == method]
                grouped = method_data.groupby(["mechanism", "batch"])["average_precision"]
                for mechanism in config["evaluation_mechanisms"]:
                    if mechanism == "M0_stationary":
                        continue
                    values = method_data[method_data.mechanism == mechanism].groupby("batch")["average_precision"]
                    means = values.mean()
                    std = values.std()
                    counts = values.count()
                    lo = means - 1.96 * std / np.sqrt(counts.clip(lower=1))
                    hi = means + 1.96 * std / np.sqrt(counts.clip(lower=1))
                    color = COLORS[method]
                    ax.plot(means.index, means.values, marker="o", ms=2.8, lw=1.7,
                            color=color, alpha=.92, label=LABELS[method] if mechanism == "M1_abrupt_nuisance" else None)
                    ax.fill_between(means.index, lo, hi, color=color, alpha=.09, linewidth=0)
            ax.axvline(25, color="#AAB6C2", ls="--", lw=.8)
            ax.set_title(f"{task.replace('_', ' ').title()} · {dim} features", loc="left", fontsize=10, weight="bold")
            ax.set_ylim(-.03, 1.03)
            ax.grid(axis="y", color="#DCE3EA", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[2, 0].set_xlabel("Predeclared stream checkpoint (batch)")
    axes[2, 1].set_xlabel("Predeclared stream checkpoint (batch)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Average precision")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .0), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Localization over sampled checkpoints · DDE proxy vs full measurement", x=.07, ha="left", y=.995,
                 fontsize=15, color="#16324F", weight="bold")
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, "temporal_localization_checkpoints")


def save_figure(fig, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(summary: pd.DataFrame, effects: pd.DataFrame, coverage: pd.DataFrame,
                 raw: pd.DataFrame, per_seed: pd.DataFrame, config: dict,
                 manifest: dict, calibration: dict) -> None:
    # Equal-weighted mechanism-level seed estimates; print n to expose missing
    # hidden positives. No NaN metric is coerced to success.
    eval_rows = summary[(summary.partition == "evaluation") & (summary.mechanism != "M0_stationary")]
    table_rows = []
    for task in config["tasks"]:
        for dim in config["dimensions"]:
            block = eval_rows[(eval_rows.task == task) & (eval_rows.dimension == dim)]
            values = {}
            for method in ("dde", "correlation_only", "direct_full_ks_hidden", "random"):
                tmp = block[block.method == method]
                means = tmp["average_precision_mean"].to_numpy(float)
                means = means[np.isfinite(means)]
                values[method] = float(np.mean(means)) if len(means) else float("nan")
            cov = coverage[(coverage.partition == "evaluation") & (coverage.task == task)
                           & (coverage.dimension == dim) & (coverage.mechanism != "M0_stationary")]
            target_cov = float(cov.groupby(["mechanism", "seed"])["target_hidden_coverage"].mean().mean()) if len(cov) else float("nan")
            table_rows.append((task.replace("_", " "), dim, values["dde"], values["correlation_only"],
                               values["direct_full_ks_hidden"], values["random"], target_cov))

    result_table = "\n".join(
        f"{task.title()} & {dim} & {dde:.3f} & {corr:.3f} & {full:.3f} & {rnd:.3f} & {cov:.2f} \\\\"
        for task, dim, dde, corr, full, rnd, cov in table_rows)
    null_rows = []
    stationary = raw[(raw.partition == "evaluation") & (raw.mechanism == "M0_stationary")
                     & (raw.method == "dde")]
    for task in config["tasks"]:
        for dim in config["dimensions"]:
            block = stationary[(stationary.task == task) & (stationary.dimension == dim)]
            false_anchor = float((block["detected_anchor_count"] > 0).mean()) if len(block) else float("nan")
            null_rows.append((task.replace("_", " ").title(), dim, len(block), false_anchor))
    null_table = "\n".join(f"{task} & {dim} & {n:,} & {rate:.3f} \\\\" for task, dim, n, rate in null_rows)
    effect_rows = []
    eval_per_seed = per_seed[per_seed["partition"] == "evaluation"]
    for task in config["tasks"]:
        for dim in config["dimensions"]:
            block = eval_per_seed[(eval_per_seed.task == task) & (eval_per_seed.dimension == dim)]
            wide = block.pivot(index=["seed", "mechanism"], columns="method", values="average_precision")
            for comparator in ("correlation_only", "direct_full_ks_hidden", "random"):
                if "dde" not in wide or comparator not in wide:
                    continue
                paired = (wide["dde"] - wide[comparator]).rename("difference").reset_index()
                by_seed = paired.groupby("seed")["difference"].mean().to_numpy(float)
                stable_key = int.from_bytes(hashlib.sha256(f"{task}|{dim}|{comparator}".encode()).digest()[:4], "big")
                mean, lo, hi, n = boot_mean(by_seed, seed=9900 + stable_key)
                effect_rows.append((task.replace("_", " ").title(), dim, comparator.replace("_", " "),
                                    f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]", n))
    effect_table = "\n".join(f"{task} & {dim} & {comp} & {delta} \\\\" for task, dim, comp, delta, _ in effect_rows)
    overall_anchor = raw[(raw.partition == "evaluation") & (raw.mechanism != "M0_stationary")]
    anchor_fraction = float((overall_anchor[overall_anchor.method == "dde"]["detected_anchor_count"] > 0).mean())
    cfg_hash = manifest["config_sha256"]
    phase2_hash = config["phase2_basis_sha256"]
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype,graphicx,booktabs,tabularx,placeins,needspace,xcolor,hyperref}}
\definecolor{{navy}}{{HTML}}{{16324F}}\definecolor{{teal}}{{HTML}}{{087E8B}}\definecolor{{muted}}{{HTML}}{{536577}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=teal}}
\setlength{{\parskip}}{{4pt}}\setlength{{\parindent}}{{0pt}}
\title{{\textcolor{{navy}}{{Localizing Hidden Feature Drift with a DDE-Style Subset Proxy}}\\[3pt]\large A post-deadline synthetic evaluation addendum}}
\author{{23I-2581 and 23I-2604\\Data Analysis and Visualization --- IDM-Viz Phase 3}}
\date{{18 November 2026}}
\begin{{document}}\maketitle
\begin{{abstract}}
This addendum evaluates whether a co-drift curve learned from development streams can rank changed marginal features that are not directly monitored. A frozen random quarter of features is tested using a development-calibrated KS threshold; a pooled correlation-conditioned co-drift curve assigns scores to the remaining features. The study uses 5 development, 10 validation and 30 evaluation seeds across three synthetic task families, two feature dimensions, four intervention mechanisms and 14 predeclared temporal checkpoints. Results are an exploratory transfer test of an explicitly DDE-style proxy, not a reproduction of the complete Domino method and not evidence about natural-stream performance, model harm, analyst benefit or causal feature responsibility.
\end{{abstract}}

\section{{Question and protocol}}
The project fidelity audit had already marked Domino as unreproduced. The previous demonstration selected its monitored features and drift threshold from the same current batch, which creates a circular evaluation. This extension removes that defect: a per-feature detector threshold is the development-only 95th percentile of the batch maximum KS statistic in stationary streams; a pooled co-drift curve is fit from seven predeclared checkpoints per development intervention stream without using generator truth. Evaluation uses fresh seed banks and a deterministic, seed-specific random subset containing 25\% of the dimensions. No evaluation truth enters feature selection, detector decisions, or scores. This addendum builds on the submitted Phase 2 DOCX without editing it (SHA-256: \texttt{{{phase2_hash}}}). No instructor feedback was received; this addendum is a team-authored post-deadline extension, not an instructor-requested method change.

There are 32- and 128-feature settings for linear probabilistic, XOR nonlinear and five-region generators. The four marginal-change mechanisms are abrupt nuisance shift, gradual predictive shift, repeated small shifts and out-of-support displacement. The held-out target is the generator-declared set of changed marginal features, scored only when the feature is outside the monitored subset. Stationarity is retained as a null control; empty-target ranking metrics remain undefined. The 14 checkpoints cover the stationary prefix, intervention onset, gradual endpoint and later stream. They are sampled time points, not a full 96-batch trajectory.

The DDE-style score averages the development probability curve at the reference-correlation magnitudes connecting each hidden feature to currently flagged monitored anchors. When the next higher-correlation bin has no eligible pairs, scoring falls back to the largest lower supported bin; it never treats an unsupported bin as probability zero. Reference correlations are frozen from historical reference rows; this is another simplification relative to the original online method. Comparators are mean absolute correlation to those same anchors, direct KS ranking of the hidden features (a full-measurement comparator with greater feature-testing cost), and random ranking. The seed is the uncertainty unit: checkpoints are averaged within seed before 2,000 seed-bootstrap replicates, with paired differences resampled within seed. Intervals are descriptive 95\% percentile intervals; no multiple-comparison-adjusted superiority test is claimed. The configuration SHA-256 is \texttt{{{cfg_hash}}}.

\Needspace{{14\baselineskip}}
\section{{Results}}
\begin{{figure}}[ht]
\centering\includegraphics[width=\linewidth]{{development_dde_curves.pdf}}
\caption{{Pooled development-only co-drift curves. Denominators are eligible feature pairs at each reference-correlation threshold; only episodes with at least one flagged feature contribute. Unsupported bins (zero eligible pairs) are shown as gaps, not as zero-probability estimates.}}
\end{{figure}}
\begin{{figure}}[ht]
\centering\includegraphics[width=\linewidth]{{heldout_feature_localization.pdf}}
\caption{{Average precision for changed features withheld from direct monitoring. Points summarize seed-level means across intervention mechanisms; bars show descriptive 95\% bootstrap intervals. The direct-KS comparator tests all features and therefore spends more feature-level tests than the subset methods.}}
\end{{figure}}
\FloatBarrier
\Needspace{{12\baselineskip}}
\begin{{table}}[ht]
\centering\small\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule Task & $d$ & DDE proxy & Correlation & Full KS & Random & Hidden-target coverage\\\midrule
{result_table}
\bottomrule\end{{tabular}}
\caption{{Equal-weighted mean average precision across the four intervention mechanisms. Hidden-target coverage is the fraction of intervention seed/checkpoint cells containing at least one changed feature outside the fixed monitored subset. Missing or empty-target cells are not counted as successful predictions.}}
\end{{table}}

\begin{{table}}[ht]
\centering\small\begin{{tabular}}{{@{{}}llrr@{{}}}}
\toprule Task & $d$ & Null seed-checkpoints & Any-anchor rate\\\midrule
{null_table}
\bottomrule\end{{tabular}}
\caption{{Stationary evaluation controls. Any-anchor rate is the fraction of evaluation seed/checkpoint cells in which at least one checked feature crosses the frozen threshold.}}
\end{{table}}

\begin{{table}}[ht]
\centering\small\begin{{tabularx}}{{\linewidth}}{{@{{}}lrXr@{{}}}}
\toprule Task & $d$ & Comparator & Mean paired $\Delta$ AP (95\% CI)\\\midrule
{effect_table}
\bottomrule\end{{tabularx}}
\caption{{DDE-style minus comparator average-precision differences, paired within seed and mechanism, then averaged within seed across eligible mechanisms. Intervals use 2,000 seed bootstrap replicates. Positive values favor the DDE-style ranking. These post-hoc intervals are exploratory and not multiplicity-adjusted.}}
\end{{table}}

\begin{{figure}}[ht]
\centering\includegraphics[width=\linewidth]{{temporal_localization_checkpoints.pdf}}
\caption{{Average precision across the predeclared checkpoints for the DDE-style proxy and full-measurement KS comparator. Shaded bands show pointwise mean $\pm 1.96$ standard errors across evaluation seeds; they are not simultaneous confidence bands.}}
\end{{figure}}
\FloatBarrier

\section{{Interpretation and limitations}}
The results should be read as a test of whether this implementation can transfer co-drift structure across new seeds under the same generator mechanisms. Development episodes and evaluation episodes share the same intervention families; this is not held-out-mechanism generalization. Features and correlations are synthetic and at most 128-dimensional. The random subset can omit every changed feature, which is why hidden-target coverage is reported separately. A DDE inference score had a detected monitored anchor in {anchor_fraction:.1%} of evaluated intervention checkpoints; without an anchor the proxy emits zero scores. The full-KS comparator is not matched for measurement cost; correlation-only is the same-subset structural baseline. The experiment does not evaluate dependence-pair localization, natural datasets, calibration, delayed labels, accuracy estimation, observed harm, future risk or actions. It cannot support claims of production savings or superiority to Domino.

The analysis uses one pooled curve per task/dimension and no model-based smoothing. Probability estimates can be unstable when few flagged co-drift pairs support a correlation bin; pair counts and the fitted values are preserved in the run calibration artifact. Confidence intervals resample seeds and average repeated checkpoints first, but the synthetic seeds do not represent independent real-world datasets. The result therefore strengthens the localization evidence while leaving the baseline-fidelity limitation explicit.

\section{{Reproduction and artifacts}}
From the project root, run \texttt{{powershell -ExecutionPolicy Bypass -File .\\scripts\\reproduce\_dde\_subset\_audit.ps1}}. The runner resumes completed seed cells only after validating their identity and full row grid against the locked configuration. Two bounded worker processes compute independent seed cells, and each result is written atomically. The runner records its worker count, source hashes, calibration and per-cell checksums. Reporting refuses to proceed unless all locked validation and evaluation cells and row counts are present. The independent paper is Sz\H{{u}}cs and N\'emeth, ``Domino drift effect approach for probability estimation of feature drift in high-dimensional data,'' \emph{{Knowledge and Information Systems}} 67 (2025), 4597--4621, DOI: \href{{https://doi.org/10.1007/s10115-025-02362-0}}{{10.1007/s10115-025-02362-0}}. The implementation here remains a scoped proxy.

\end{{document}}
"""
    (OUT / "dde_subset_audit.tex").write_text(tex, encoding="utf-8")


def compile_report() -> None:
    cmd = ["tectonic", "-X", "compile", "--keep-logs", "dde_subset_audit.tex"]
    subprocess.run(cmd, cwd=OUT, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
