"""Create a concise outcome-informed adaptation report and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_NAMES = {"v1_synthetic": "Original synthetic task", "xor_synthetic": "XOR synthetic task"}
CONDITION_NAMES = {"C3": "conditional reversal", "X3": "XOR concept complement"}
COLORS = {0: "#176B87", 5: "#E07A5F", 20: "#5B8E7D"}


def _trajectory_figure(run: Path, out: Path) -> Path:
    batches = pd.read_csv(run / "evaluator_batches.csv")
    rng = np.random.default_rng(15112027)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, (task, condition) in zip(axes, (("v1_synthetic", "C3"), ("xor_synthetic", "X3"))):
        subset = batches[(batches.task == task) & (batches.condition == condition)]
        for delay in (0, 5, 20):
            color = COLORS[delay]
            for method, style in (("frozen", "--"), ("cumulative_refit", "-")):
                values = subset[(subset.delay == delay) & (subset.method == method)]
                matrix = values.pivot(index="seed", columns="batch", values="accuracy").sort_index(axis=0)
                arr = matrix.to_numpy(float)
                indices = rng.integers(0, len(arr), size=(2000, len(arr)))
                boot = arr[indices].mean(axis=1)
                mean = arr.mean(axis=0)
                low, high = np.quantile(boot, [0.025, 0.975], axis=0)
                label = f"{method.replace('_', ' ')} · delay {delay}"
                ax.plot(np.arange(1, 25), mean, linestyle=style, color=color, linewidth=1.8, label=label)
                if method == "cumulative_refit":
                    ax.fill_between(np.arange(1, 25), low, high, color=color, alpha=0.13)
        ax.axvline(8.5, color="#59636e", linestyle=":", linewidth=1.2)
        ax.set_title(f"{TASK_NAMES[task]} · {CONDITION_NAMES[condition]}")
        ax.set_xlabel("Forecast batch")
        ax.set_xlim(1, 24)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Batch accuracy (evaluator-only after release)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    fig.suptitle("Frozen logistic model and cumulative released-label refit", y=1.02)
    fig.tight_layout()
    target = out / "adaptation_accuracy.png"
    fig.savefig(target, dpi=220, bbox_inches="tight")
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return target


def _budget_figure(out: Path) -> Path:
    batch = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for delay in (0, 5, 20):
        count = 500 * np.minimum(24, np.maximum(0, batch - delay - 1))
        ax.step(batch, count, where="mid", color=COLORS[delay], linewidth=2,
                label=f"delay {delay}")
    ax.axvline(8.5, color="#59636e", linestyle=":", linewidth=1.2, label="label-law change")
    ax.set(xlabel="Forecast batch", ylabel="Monitoring labels available before prediction",
           xlim=(1, 24), ylim=(0, 12000))
    ax.set_yticks(np.arange(0, 12001, 2000))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Exact cumulative released-label budget at each forecast")
    fig.tight_layout()
    target = out / "label_budgets.png"
    fig.savefig(target, dpi=220, bbox_inches="tight")
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return target


def build(run: Path, out: Path) -> Path:
    run, out = run.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bootstrap = pd.read_csv(run / "paired_bootstrap.csv")
    budget = pd.read_csv(run / "budget_by_clock.csv")
    validation = pd.read_json(run / "validation.json", typ="series")
    protocol = json.loads((run / "protocol_lock.json").read_text(encoding="utf-8"))
    comparison_path = run.parents[2] / "reports/phase3/adaptation_reproduction/reproduction_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else None
    reproduction_lines = []
    if comparison and comparison.get("status") == "passed":
        reproduction_lines = [
            f"An independent clean run matched {comparison['csv_files']:,} CSVs "
            f"({comparison['csv_cells']:,} cells) and {comparison['npz_archives']:,} compressed arrays "
            f"({comparison['arrays']:,} arrays; {comparison['array_values']:,} values) exactly. "
            "Timing and run-metadata fields were excluded.", "",
        ]
    seeds = len(pd.read_csv(run / "paired_seed_differences.csv").seed.unique())
    run_description = "final evidence" if protocol["mode"] == "final" else "development-only feasibility evidence"
    trajectory = _trajectory_figure(run, out)
    budget_figure = _budget_figure(out)

    selected = bootstrap[(bootstrap.metric == "accuracy_delta") &
                         bootstrap.regime.isin(("pre_onset", "post_onset", "late_post"))]
    table = [
        "| Condition | Delay | Regime | Adapted minus frozen accuracy, mean [seed-bootstrap 95% interval] | Seeds |",
        "|---|---:|---|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        table.append(f"| {CONDITION_NAMES[row.condition]} | {row.delay} | {row.regime.replace('_', ' ')} | "
                     f"{row.mean:+.4f} [{row.lower:+.4f}, {row.upper:+.4f}] | {row.seeds} |")

    at_end = budget[budget.batch == 24].groupby("delay").available_monitoring_labels.first().to_dict()
    late = bootstrap[(bootstrap.regime == "late_post") &
                     bootstrap.metric.isin(("frozen_accuracy", "adapted_accuracy", "accuracy_delta"))]
    late_lookup = {(r.condition, int(r.delay), r.metric): float(r.mean)
                   for r in late.itertuples(index=False)}
    c3_lower = late_lookup[("C3", 0, "accuracy_delta")]
    c3_frozen = late_lookup[("C3", 0, "frozen_accuracy")]
    c3_adapted = late_lookup[("C3", 0, "adapted_accuracy")]
    c3_full_post = bootstrap[(bootstrap.condition == "C3") & (bootstrap.delay == 0) &
                             (bootstrap.regime == "post_onset") & (bootstrap.metric == "accuracy_delta")].iloc[0]
    xor_late_range = (min(late_lookup[("X3", d, "accuracy_delta")] for d in (0, 5, 20)),
                      max(late_lookup[("X3", d, "accuracy_delta")] for d in (0, 5, 20)))
    post_labels_by_delay = {
        d: 500 * max(0, min(24, 24 - d - 1) - 8) for d in (0, 5, 20)
    }
    lines = [
        "# Released-label adaptation in controlled synthetic streams",
        "",
        "**Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)**  ",
        "Data Analysis and Visualization · Phase 3 supplementary experiment",
        "",
        "**Status:** Outcome-informed exploratory add-on, protocol locked before its development and final runs.",
        "",
        "## Question and scope",
        "",
        "This small experiment asks whether cumulative refitting on labels after their scheduled release changes batch accuracy under two selected synthetic conditional changes: C3 (label-probability reversal with unchanged inputs) and X3 (XOR concept complement with unchanged inputs). Selection was informed by the earlier study's conditional-only failures and the extension's XOR task. The experiment is not external validation and is kept separate from the frozen-model results.",
        "",
        f"This {run_description} uses {seeds} seeds per condition, 3 delays, and 24 batches of 500 rows. It created {int(validation['feature_streams'])} feature streams, {int(validation['delay_streams'])} delay streams, {int(validation['model_batch_records'])} online method-batch records, {int(validation['prediction_instance_joins']):,} exact prediction/truth joins and {int(validation['released_label_rows']):,} scheduled label rows.",
        "",
        "## Locked comparison",
        "",
        "Both arms use LogisticRegression(C=1, max_iter=500), training-only StandardScaler and the same initial 4,000 training rows. The frozen arm never updates. The adaptation arm refits on those training rows plus all unique monitoring labels available strictly before the next batch prediction. Calibration and reference partitions are not used for adaptation. No current or future label is passed to prediction; each batch's two prediction arrays are persisted before broker release, including delay zero.",
        "",
        "At forecast batch `t` and delay `d`, the exact available monitoring-label count is `500 × min(24, max(0, t−d−1))`. At batch 24, delays 0, 5 and 20 make 11,500, 9,000 and 1,500 labels available before scoring; after drainage each releases the same 12,000 labels by clock 24+d. The frozen model is a zero-label update baseline, so gains versus it represent access-and-update effects. The only adaptive procedure is evaluated at exact, logged release budgets; no adaptive-method leaderboard is made.",
        "",
        f"The primary metric is paired-seed accuracy change over post-onset batches 9–24, with late post-onset batches 17–24 and pre-onset batches 1–8 shown separately. Macro F1 is secondary. Bootstrap intervals resample complete {seeds}-seed vectors, never batches or delays. Shaded plot intervals use the same complete-seed-vector resampling.",
        "",
        "## Results",
        "",
        "| Condition | Delay | Regime | Adapted minus frozen accuracy, mean [seed-bootstrap 95% interval] | Seeds |",
        "|---|---:|---|---:|---:|",
        *table[2:],
        "",
        "![Accuracy trajectories](adaptation_accuracy.png)",
        "",
        "![Released-label budgets](label_budgets.png)",
        "",
        "The table reports the measured paired difference; intervals crossing zero do not establish a directional effect. C3 and X3 remain generator-specific examples. Their different results do not prove that adaptation generally repairs delayed monitoring or confidence estimation.",
        "",
        f"In C3 at delay 0, late post-onset accuracy rose from {c3_frozen:.4f} frozen to {c3_adapted:.4f} adapted (paired difference {c3_lower:+.4f}); however, across all post-onset batches the paired change was only {c3_full_post['mean']:+.4f} [{c3_full_post['lower']:+.4f}, {c3_full_post['upper']:+.4f}]. At delays 5 and 20, the late-post differences were {late_lookup[('C3', 5, 'accuracy_delta')]:+.4f} and {late_lookup[('C3', 20, 'accuracy_delta')]:+.4f}. For X3, late-post adapted accuracy stayed near 0.50 and differences ranged from {xor_late_range[0]:+.4f} to {xor_late_range[1]:+.4f}; there was no meaningful overall post-onset improvement. This is limited recovery in one condition, not a general adaptation result.",
        "",
        f"At the batch-24 prediction, delays 0, 5 and 20 had {int(at_end[0]):,}, {int(at_end[5]):,} and {int(at_end[20]):,} labels available, of which {post_labels_by_delay[0]:,}, {post_labels_by_delay[5]:,} and {post_labels_by_delay[20]:,} came from post-onset batches. Thus delay 20 had no post-onset labels available to learn from by the final input prediction. The cumulative model also retains all pre-onset training examples. A plausible explanation for the weak X3 response is that a linear logistic boundary cannot represent the XOR interaction; the experiment did not directly test alternative nonlinear update mechanisms.",
        "",
        "## Evidence separation and validation",
        "",
        *reproduction_lines,
        "`online_batches.csv` and per-batch prediction archives contain no outcome fields. `released_labels.csv` carries IDs and labels with availability clocks; `evaluator_truth/` is retrospective. The evaluator checked exact availability budgets before each fit, that no fitting label had an availability clock at or after its forecast, same-clock persistence ordering, identical release IDs across delays, exact prediction/truth joins, and drainage without new input. All metrics are evaluator-only. This is not a security boundary or human study.",
        "",
        "## Limitations",
        "",
        "This add-on studies one cumulative refit rule and one linear logistic model on two selected synthetic conditional-change cases. It does not add confidence calibration, prediction intervals, harm probabilities, or detector claims. The frozen comparison intentionally uses zero monitoring labels for fitting; only the availability schedules are shared. Cumulative retraining repeatedly reuses older released labels, so compute exposure differs from incremental training even though each adaptive run receives the same unique released IDs. No external data, natural onset labels, human participants, or analyst-benefit claims are present.",
        "",
        "**Reproduction:** from the workspace root, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\reproduce_phase3_adaptation.ps1 -RunDir results/phase3/adaptation_new_reproduction -ReportDir reports/phase3/adaptation_new_reproduction`. Use fresh, unused output directories. The wrapper evaluates the saved labels, checks exact equality with this final run, and renders the report. Exact Python package versions are in `environment.json`.",
        "",
    ]
    path = out / "23I-2581_23I-2604_Phase3_Adaptation_Report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.run, args.out)
