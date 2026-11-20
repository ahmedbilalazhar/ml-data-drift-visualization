"""Write the v2 empirical report and failure-analysis figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _value(table, query, metric):
    selected = table
    for key, value in query.items():
        selected = selected[selected[key] == value]
    selected = selected[selected.metric == metric]
    return selected.iloc[0] if len(selected) else None


def _fmt(value, digits=4):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def _plot_model_sensitivity(run, figures):
    summary = pd.read_csv(run / "impact_summary.csv")
    tasks = (("v1_synthetic", "Original synthetic task", [f"C{i}" for i in range(7)]),
             ("xor_synthetic", "New XOR synthetic task", [f"X{i}" for i in range(5)]))
    models = (("logistic", "Logistic regression"), ("shallow_tree", "Shallow tree"))
    methods = (("IDM_calibrated_AC", "Calibrated AC", "#176B87"),
               ("raw_AC_no_calibration", "Raw AC", "#E07A5F"),
               ("ATC_style", "ATC-style", "#5B8E7D"))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharey="row")
    for row, (task, task_title, conditions) in enumerate(tasks):
        for col, (model, model_title) in enumerate(models):
            ax = axes[row, col]
            for method, method_title, color in methods:
                subset = summary[(summary.task == task) & (summary.model == model) &
                                 (summary.regime == "all") & (summary.method == method) &
                                 (summary.metric == "mae")].set_index("condition")
                means = [subset.loc[c, "mean"] if c in subset.index else np.nan for c in conditions]
                lows = [subset.loc[c, "lower"] if c in subset.index else np.nan for c in conditions]
                highs = [subset.loc[c, "upper"] if c in subset.index else np.nan for c in conditions]
                means, lows, highs = np.asarray(means), np.asarray(lows), np.asarray(highs)
                x = np.arange(len(conditions))
                ax.errorbar(x, means, yerr=np.vstack((means - lows, highs - means)),
                            marker="o", capsize=2.5, linewidth=1.6, color=color, label=method_title)
            ax.set_title(f"{task_title} · {model_title}")
            ax.set_xticks(np.arange(len(conditions)), conditions)
            ax.set_ylabel("Accuracy-estimation MAE")
            ax.grid(axis="y", alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Paired confidence-estimator sensitivity by task and learner", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "model_sensitivity_mae.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "model_sensitivity_mae.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_intervals(run, figures):
    summary = pd.read_csv(run / "impact_summary.csv")
    tasks = (("v1_synthetic", "Original task", [f"C{i}" for i in range(7)]),
             ("xor_synthetic", "XOR task", [f"X{i}" for i in range(5)]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for row, (task, task_title, conditions) in enumerate(tasks):
        for col, (model, model_title, color) in enumerate((
                ("logistic", "Logistic regression", "#176B87"),
                ("shallow_tree", "Shallow tree", "#E07A5F"))):
            ax = axes[row, col]
            width_ax = ax.twinx()
            for offset, (metric, marker, label) in enumerate((
                    ("coverage", "o", "Coverage"), ("interval_width", "s", "Mean interval width"))):
                subset = summary[(summary.task == task) & (summary.model == model) &
                                 (summary.regime == "all") &
                                 (summary.method == "IDM_calibrated_AC") &
                                 (summary.metric == metric)].set_index("condition")
                values = [subset.loc[c, "mean"] if c in subset.index else np.nan for c in conditions]
                x = np.arange(len(conditions)) + (offset - 0.5) * 0.12
                target = ax if offset == 0 else width_ax
                line_color = color if offset == 0 else "#7768AE"
                target.plot(x, values, marker=marker, color=line_color,
                            linewidth=1.4, label=label)
            ax.axhline(0.95, color="#555555", linestyle="--", linewidth=1, label="Nominal 0.95 coverage")
            ax.set_title(f"{task_title} · {model_title}")
            ax.set_xticks(np.arange(len(conditions)), conditions)
            ax.set_ylabel("Empirical coverage")
            width_ax.set_ylabel("Mean interval width")
            ax.grid(axis="y", alpha=0.25)
            if row == 0 and col == 0:
                lines, labels = ax.get_legend_handles_labels()
                width_lines, width_labels = width_ax.get_legend_handles_labels()
                ax.legend(lines + width_lines, labels + width_labels, frameon=False, fontsize=8)
    fig.suptitle("Empirical coverage and interval width (nominal line is a target)", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "interval_coverage_width.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "interval_coverage_width.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_harm(run, figures):
    counts = pd.read_csv(run / "harm_counts.csv")
    counts = counts[(counts.regime == "all") & (counts.method == "IDM_calibrated_AC")]
    conditions = sorted(counts.condition.unique(), key=lambda c: (c[0], int(c[1:])))
    groups = list(counts.groupby(["task", "model"], sort=False))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    for ax, ((task, model), group) in zip(axes.flat, groups):
        table = group.set_index("condition")
        labels = [c for c in conditions if c in table.index]
        x = np.arange(len(labels))
        fn = [table.loc[c, "fn"] for c in labels]
        tp = [table.loc[c, "tp"] for c in labels]
        fp = [table.loc[c, "fp"] for c in labels]
        ax.bar(x, fn, color="#D98C5F", label="Harmful unflagged")
        ax.bar(x, tp, bottom=fn, color="#438B78", label="Harm flagged")
        ax.bar(x, fp, bottom=np.asarray(fn) + np.asarray(tp), color="#7868A6", label="False flags")
        title = "Original task" if task == "v1_synthetic" else "XOR task"
        ax.set_title(f"{title} · {model.replace('_', ' ')}")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Batch count across final seeds")
        ax.grid(axis="y", alpha=0.22)
        if ax is axes.flat[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Concern-rule outcomes with misses shown explicitly", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "harm_concern_outcomes.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "harm_concern_outcomes.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(run, figures):
    summary = pd.read_csv(run / "calibration_error_summary.csv")
    tasks = (("v1_synthetic", "Original task"), ("xor_synthetic", "XOR task"))
    regimes = ["pre", "onset", "transition", "post", "recovery", "stationary_late"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharey=True)
    for row, (task, title) in enumerate(tasks):
        for col, (model, model_title) in enumerate((
                ("logistic", "Logistic regression"), ("shallow_tree", "Shallow tree"))):
            ax = axes[row, col]
            subset = summary[(summary.task == task) & (summary.model == model) &
                             (summary.metric == "adaptive_calibration_error")]
            for method, color, label in (("raw_confidence", "#E07A5F", "Raw confidence"),
                                         ("calibrated", "#176B87", "Isotonic calibrated")):
                series = subset[subset.method == method].set_index("regime")
                values = [series.loc[r, "mean"].mean() if r in series.index else np.nan
                          for r in regimes]
                ax.plot(np.arange(len(regimes)), values, marker="o", color=color, label=label)
            ax.set_title(f"{title} · {model_title}")
            ax.set_xticks(np.arange(len(regimes)), regimes, rotation=25, ha="right")
            ax.set_ylabel("Adaptive calibration error")
            ax.grid(axis="y", alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Correctness reliability by stream regime", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "calibration_by_regime.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "calibration_by_regime.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_v1_failure(run, report_dir):
    v1 = run.parents[2] / "results/phase3/final_v1"
    data = pd.read_csv(v1 / "evaluator_batches.csv")
    figure_dir = report_dir / "figures"
    rows = []
    for condition in ("C3", "C4", "C5"):
        group = data[data.condition == condition]
        summary = group.groupby("batch", as_index=False).agg(
            calibrated_estimate=("cal_estimate", "mean"), actual_accuracy=("accuracy", "mean"),
            raw_estimate=("raw_estimate", "mean"), calibrated_lower=("cal_lower", "mean"),
            calibrated_upper=("cal_upper", "mean"), confidence=("cal_estimate", "mean"),
            harm_count=("harm", "sum"), stream_n=("seed", "nunique"))
        summary.insert(0, "condition", condition)
        rows.append(summary)
    pd.concat(rows, ignore_index=True).to_csv(report_dir / "v1_conditional_failure_diagnostics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for ax, condition in zip(axes, ("C3", "C4", "C5")):
        group = pd.concat(rows).query("condition == @condition")
        ax.fill_between(group.batch, group.calibrated_lower, group.calibrated_upper,
                        color="#176B87", alpha=0.15, label="Mean plug-in interval")
        ax.plot(group.batch, group.calibrated_estimate, color="#176B87", label="Calibrated estimate")
        ax.plot(group.batch, group.actual_accuracy, color="#D95F59", label="Evaluator-only accuracy")
        ax.axvline(8.5, color="#555", linestyle="--", linewidth=1)
        if condition == "C5":
            ax.axvline(16.5, color="#6D597A", linestyle=":", linewidth=1.2, label="Return")
        ax.set_title(f"{condition}: conditional change")
        ax.set_xlabel("Batch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Version 1 conditional-only failure · retrospective evaluator truth", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "v1_conditional_failure.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "v1_conditional_failure.svg", bbox_inches="tight")
    plt.close(fig)

    c6 = pd.read_csv(v1 / "online_batches.csv")
    c6 = c6[c6.condition == "C6"]
    c6_window = c6[c6.batch.between(9, 12)]
    component_rows = []
    for component in range(3):
        component_rows.append(dict(component=f"PC{component}",
            seed_events=int(c6_window.groupby("seed")[f"pc{component}_adjusted_p"].min().le(0.05).sum()),
            seeds=int(c6_window.seed.nunique())))
    component = pd.DataFrame(component_rows)
    component.to_csv(report_dir / "v1_dependence_component_diagnostics.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(component.component, component.seed_events, color=["#176B87", "#438B78", "#E07A5F"])
    ax.axhline(30, linestyle="--", linewidth=1, color="#555", label="All 30 seeds")
    ax.set_ylim(0, 31)
    ax.set_ylabel("Seeds with adjusted p ≤ .05 in batches 9–12")
    ax.set_title("V1 C6: which frozen PCA coordinates signaled")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "v1_c6_pca_components.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "v1_c6_pca_components.svg", bbox_inches="tight")
    plt.close(fig)
    return data


def build_report(run: Path, report_dir: Path):
    run, report_dir = run.resolve(), report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = report_dir / "figures"
    figures.mkdir(exist_ok=True)
    validation = pd.read_json(run / "validation.json", typ="series")
    if validation.get("status") != "passed":
        raise ValueError("report requires passed evaluation validation")
    _plot_model_sensitivity(run, figures)
    _plot_intervals(run, figures)
    _plot_harm(run, figures)
    _plot_calibration(run, figures)
    v1_data = _plot_v1_failure(run, report_dir)

    overall = pd.read_csv(run / "overall_summary.csv")
    impact = pd.read_csv(run / "impact_summary.csv")
    counts = pd.read_csv(run / "harm_counts.csv")
    models = pd.read_csv(run / "paired_model_bootstrap.csv")
    delays = pd.read_csv(run / "delay_availability_summary.csv")
    intervals = pd.read_csv(run / "impact_summary.csv")

    def overall_value(task, model, metric):
        row = overall[(overall.task == task) & (overall.model == model) & (overall.metric == metric)]
        return row.iloc[0] if len(row) else None

    v1_per_seed = pd.read_csv(run.parents[2] / "results/phase3/final_v1/impact_per_seed.csv")
    v1_main = v1_per_seed[(v1_per_seed.regime == "all") &
        (v1_per_seed.method.isin(("IDM_calibrated_AC", "raw_AC_no_calibration")))].groupby(
            "method", as_index=False).agg(mae=("mae", "mean"), coverage=("coverage", "mean"))
    v1_cal = v1_main[v1_main.method == "IDM_calibrated_AC"]
    v1_raw = v1_main[v1_main.method == "raw_AC_no_calibration"]
    v1_cal_mae = float(v1_cal.iloc[0].mae) if len(v1_cal) else np.nan
    v1_raw_mae = float(v1_raw.iloc[0].mae) if len(v1_raw) else np.nan
    v1_cover = float(v1_cal.iloc[0].coverage) if len(v1_cal) else np.nan

    def task_metric(task, model, metric):
        row = overall_value(task, model, metric)
        if row is None:
            return "undefined"
        return f"{row['mean']:.6f} [{row['lower']:.6f}, {row['upper']:.6f}]"

    total_streams = int(json.loads((run / "protocol_lock.json").read_text())["streams"])
    lock = json.loads((run / "protocol_lock.json").read_text())
    v1_compare = validation.get("v1_logistic_extension_comparison")
    v1_pair_text = (f"Exact extension check passed for {v1_compare['streams']} v1-task streams, "
                    f"{v1_compare['arrays_compared']} cached arrays and "
                    f"{v1_compare['array_values_compared']:,} array values.") if v1_compare else (
                        "V1 cache comparison was not available in this package context.")
    overall_table = []
    for task, task_title in (("v1_synthetic", "Original synthetic task"),
                             ("xor_synthetic", "XOR synthetic task")):
        for model, model_title in (("logistic", "Logistic"), ("shallow_tree", "Shallow tree")):
            overall_table.append(f"| {task_title} | {model_title} | {task_metric(task, model, 'mae')} | "
                                 f"{task_metric(task, model, 'coverage')} | "
                                 f"{task_metric(task, model, 'interval_width')} |")
    tree_delta = models[(models.metric == "tree_minus_logistic_actual_accuracy") &
                        (models.regime == "all")]
    xor_delta = tree_delta[(tree_delta.task == "xor_synthetic") & (tree_delta.condition == "X0")]
    v1_tree_delta = tree_delta[(tree_delta.task == "v1_synthetic") & (tree_delta.condition == "C0")]
    delta_text = ""
    if len(xor_delta):
        r = xor_delta.iloc[0]
        delta_text += f"On stationary XOR, tree minus logistic accuracy was {r['mean']:.4f} " \
                      f"(seed-bootstrap 95% interval {r['lower']:.4f} to {r['upper']:.4f}). "
    if len(v1_tree_delta):
        r = v1_tree_delta.iloc[0]
        delta_text += f"On v1 stationary data the paired accuracy difference was {r['mean']:.4f} " \
                      f"({r['lower']:.4f} to {r['upper']:.4f})."

    conditional_v1 = v1_data[v1_data.condition.isin(("C3", "C4", "C5"))]
    v1_cond_harms = int(conditional_v1.harm.sum())
    v1_cond_flags = int(conditional_v1.cal_concern_flag.sum())
    v1_cond_accuracy = float(conditional_v1.accuracy.mean())
    v1_cond_estimate = float(conditional_v1.cal_estimate.mean())
    v1_calibration = pd.read_csv(run.parents[2] / "results/phase3/final_v1/calibration_error_per_seed.csv")
    c6_hits = int((v1_data[(v1_data.condition == "C6") & v1_data.batch.between(9, 12)]
                   .groupby("seed").input_alarm.max().sum()))
    c6_original_hits = int((v1_data[(v1_data.condition == "C6") & v1_data.batch.between(9, 12)]
                            .groupby("seed").original_alarm.max().sum()))
    c6_components = pd.read_csv(report_dir / "v1_dependence_component_diagnostics.csv")
    delay_rows = delays[(delays.task == "v1_synthetic") & (delays.model == "logistic")]
    delay_counts = {}
    for delay in (0, 5, 20):
        verified = delay_rows[(delay_rows.delay == delay) &
                              (delay_rows.metric == "verified_at_input_end")]
        unresolved = delay_rows[(delay_rows.delay == delay) &
                                (delay_rows.metric == "unresolved_at_input_end")]
        delay_counts[delay] = (int(verified.iloc[0]["mean"]), int(unresolved.iloc[0]["mean"]))

    comparison_path = run.parents[2] / "reports/phase3/v2_reproduction/reproduction_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else None
    reproduction_lines = []
    if comparison and comparison.get("status") == "passed":
        reproduction_lines = [
            f"**Independent reproduction:** a clean run matched {comparison['csv_files']:,} CSVs "
            f"({comparison['csv_cells']:,} cells) and {comparison['npz_archives']:,} compressed arrays "
            f"({comparison['arrays']:,} arrays; {comparison['array_values']:,} values) exactly. "
            "Timing columns and protocol metadata were excluded from scientific equality checks.",
            "The first comparator attempt differed only in bootstrap summaries of wall-clock `input_seconds`. "
            "That engineering timing was removed from uncertainty summaries, and both saved runs were reevaluated; "
            "no inputs, seeds, predictions, model settings or scientific thresholds changed.",
            "",
        ]

    lines = [
        "# Phase 3: Research Implementation and Experimental Evaluation",
        "",
        "**Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)**  ",
        "Department of AI and Data Science, FAST-NUCES Islamabad  ",
        "Data Analysis and Visualization",
        "",
        "**Protocol status:** Outcome-informed extension; final evidence generated from the locked extension protocol.",
        "",
        "## Summary",
        "",
        f"Version 2 adds a fixed shallow tree comparison on the original controlled task and a new XOR synthetic task. The final run has {total_streams} feature streams and {len(pd.read_csv(run / 'online_batches.csv')):,} model-batch records. It evaluates both learners with their own correctness calibrator, ATC-style threshold and historical reference. The XOR task is synthetic and is not external-domain validation.",
        "",
        delta_text,
        "",
        f"{v1_pair_text} V1 logistic predictions, confidence arrays and original condition records are preserved as the matched comparison; no v1 seed or outcome is replaced.",
        "",
        "## What the extension tests",
        "",
        "The shallow tree is fixed at Gini criterion, depth 6 and minimum leaf size 25. Logistic regression retains `C=1`, `max_iter=500` and a 0.5 class cutoff. Both models use training-only scaling and independent isotonic calibrators fitted on the same correctness-calibration rows. No hyperparameter search is performed.",
        "",
        "The new XOR task uses eight independent standard-normal features and the noisy concept `1{(X0 > 0) XOR (X1 > 0)}` with 10% baseline label noise. Its locked interventions are stationary, irrelevant mean shift, relevant mean shift, concept complement and increased label noise. The last two preserve inputs and alter only the label law. Synthetic event times are known from construction; no onset labels are fabricated for a natural dataset.",
        "",
        "The local ELEC2 candidate was rejected: the raw ARFF’s `date` is non-monotone, 528 `(date, period)` pairs repeat, and 11 date values have 96 rows instead of 48. The date/period fields therefore do not uniquely reconstruct row chronology. The processed parquet also fails to read. OpenML’s local metadata reports its license field as `Public`; this report does not turn that label into a specific permissive license. The source record and original study are listed at the end.",
        "",
        "## Overall results",
        "",
        "| Task | Model | Calibrated AC MAE, mean [seed-bootstrap 95% interval] | Coverage | Mean interval width |",
        "|---|---|---:|---:|---:|",
        *overall_table,
        "",
        f"For context, the earlier study reported calibrated AC MAE {v1_cal_mae:.6f} versus raw AC MAE {v1_raw_mae:.6f}, and calibrated coverage {v1_cover:.6f}. Its retained baseline tables give the corresponding seed rows and denominators. These cohorts are not pooled with the two new synthetic tasks.",
        "",
        "Per-condition summaries, complete seed rows, paired seed contrasts, pooled harm counts, undefined metrics and bootstrap intervals are provided in the CSV files. The tree’s model-specific historical reference changes its realised-harm label; differences in harm counts alone do not rank the models. Matched-cohort accuracy and estimation error are reported separately.",
        "",
        "## Failure analysis",
        "",
        f"In v1 C3-C5, the calibrated confidence channel averaged {v1_cond_estimate:.3f} while evaluator-only accuracy averaged {v1_cond_accuracy:.3f}; {v1_cond_harms:,} batches met the v1 realised-harm rule and the calibrated concern rule flagged {v1_cond_flags}. This is measured. Those conditions reuse identical input arrays while changing the conditional label law, so the input detector has no distributional input evidence by construction. The model receives familiar inputs and retains its source confidence even when its predictions become wrong. The frozen source calibrator cannot infer a label-law reversal from confidence alone.",
        "",
        f"Version 1 calibrated AC MAE ({v1_cal_mae:.6f}) exceeded raw AC MAE ({v1_raw_mae:.6f}) by {v1_cal_mae - v1_raw_mae:.6f}; calibrated interval coverage was {v1_cover:.3f}. The plausible explanation that an isotonic map learned source reliability but did not transfer to changed regimes is consistent with the conditional-only results, but it is not a unique causal explanation. V2 adds pre/onset/post/recovery reliability summaries and reports calibration error and Brier score by model and regime.",
        "",
        f"For C6, v1 matched the dependence onset in {c6_hits}/30 seeds with full KS/PCA and {c6_original_hits}/30 with original-feature KS. Individual frozen PCA-coordinate adjusted-p detections in the onset window were: {', '.join(f'{r.component} {int(r.seed_events)}/30' for r in c6_components.itertuples())}. PCA therefore supplied some additional task-specific marginal evidence, but did not solve the missed-event problem; the PCA basis is descriptive and fixed from training.",
        "",
        "Version 2 reports interval width alongside empirical coverage and concern TP/FP/FN/TN counts for each task, model, condition and regime. A no-flag outcome is not a safety verdict. The plug-in Poisson-binomial interval and concern score are not calibrated harm probabilities or arbitrary-shift coverage guarantees.",
        "",
        f"At the end of 24 input batches, delayed verification leaves {delay_counts[0][1]} batches unresolved for delay 0, {delay_counts[5][1]} for delay 5, and {delay_counts[20][1]} for delay 20; after drainage all 24 are verified at ages 0, 5 and 20 batches respectively. Delay changes evidence availability and age, not the frozen estimator’s accuracy.",
        "",
        "## Figures",
        "",
        "**Paired model sensitivity:** estimation error by task, condition and learner.",
        "",
        "![Paired model sensitivity](figures/model_sensitivity_mae.png)",
        "",
        "**Interval behavior:** empirical coverage and width; the nominal line is a target.",
        "",
        "![Coverage and interval width](figures/interval_coverage_width.png)",
        "",
        "**Concern-rule outcomes:** harmful batches left unflagged are shown directly.",
        "",
        "![Harm flags and misses](figures/harm_concern_outcomes.png)",
        "",
        "**Reliability:** calibrated and raw scores across stream regimes.",
        "",
        "![Calibration by regime](figures/calibration_by_regime.png)",
        "",
        "**V1 failure analysis:** conditional-only shift with evaluator-only truth.",
        "",
        "![Conditional-only failure](figures/v1_conditional_failure.png)",
        "",
        "**V1 dependence analysis:** adjusted-p detections by frozen PCA coordinate.",
        "",
        "![Dependence coordinate detection](figures/v1_c6_pca_components.png)",
        "",
        "",
        "## Version comparison and research questions",
        "",
        "- **Established beyond v1:** model sensitivity on matched original-task observations; an independently seeded synthetic task with an axis-aligned nonlinear boundary; model-specific calibration, harm targets, intervals and paired uncertainty; failure reliability split by task, model and regime.",
        "- **Contradicts a universal model claim:** neither learner dominates every task/regime on accuracy, estimation error, coverage and harm flags. Interpret comparisons only within matched cohorts and state the model-specific reference target.",
        "- **Unchanged:** v1’s logistic method, original condition evidence, thresholds, delays and negative findings remain intact. V2 does not retrofit v1 outcomes or claim a natural-domain replication.",
        "- **RQ1:** current-batch realised deterioration is evaluated in controlled synthetic streams; future forecasting and certified population harm remain untested.",
        "- **RQ2:** feature evidence remains descriptive marginal ranking; causal attribution and cluster localization accuracy remain unanswered.",
        "- **RQ3:** analyst benefit remains unanswered; there is no human-subject evaluation.",
        "- **RQ4:** robustness evidence now spans two synthetic task families, not external or natural-domain validation.",
        "",
        "## Limitations and provenance",
        "",
        "The evidence uses one linear probabilistic learner and one fixed shallow tree on two synthetic families. It does not include external-domain validation, neural autoencoders, causal discovery, online adaptation, selective-label analysis or a human study. Calibration is frozen, confidence-based, and dependent on the source law. Interval width and empirical coverage are descriptive. Bootstrap intervals resample complete seed vectors; batches, delay copies, models and ablations are not treated as independent datasets. Version 2 hypotheses were informed by observed v1 outcomes, so the extension is not described as an untouched preregistration.",
        "",
        "The submitted Phase 2 DOCX remains the methodological basis; the older TeX is only supporting material, and no instructor feedback was received. The original two-page implementation brief was consulted and preserved. The existing offline replay is a prototype, contains retrospective release information and is not a security boundary or human-evaluation result.",
        "",
        "**Sources consulted for the rejected ELEC2 candidate:** [OpenML dataset 151](https://www.openml.org/d/151); Gama, Medas, Castillo and Rodrigues, *Learning with Drift Detection* (2004), [primary paper](https://sweet.ua.pt/gladys/Papers/GamaMedasCastilloRodriguesSBIA04.pdf). The published task results are not treated as comparable benchmarks.",
        "",
        "## Reproduction and artifact semantics",
        "",
        *reproduction_lines,
        "Use `Reproduce.md` for the package contents and run separation; the project workspace guide `docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md` gives the exact full-rerun commands. The final run has a protocol lock and exact environment versions. `predictions/` contain persisted online scores; `released_verification.csv` contains scheduled label results; `evaluator_truth/` and `evaluator_batches.csv` are retrospective. Historical source documents and run archives are preserved.",
        "",
    ]
    report_path = report_dir / "23I-2581_23I-2604_Phase3_Report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run, args.out)
