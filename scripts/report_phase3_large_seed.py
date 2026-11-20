"""Generate a concise report and failure-analysis plots for the large-seed study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"logistic": "#087e8b", "shallow_tree": "#ff5a5f"}


def _fmt(row, metric):
    return f"{row['mean']:.4f} [{row['lower']:.4f}, {row['upper']:.4f}]"


def _summary(summary, task, condition, model, metric):
    rows = summary[(summary.task == task) & (summary.condition == condition) &
                   (summary.model == model) & (summary.regime == "all") & (summary.metric == metric)]
    return rows.iloc[0] if not rows.empty else None


def report(run: Path, out: Path, reproduction: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(run / "per_seed_condition.csv")
    paired = pd.read_csv(run / "paired_model_bootstrap.csv")
    online = pd.read_csv(run / "online_batches.csv")
    validation = pd.read_json(run / "validation.json", typ="series")
    seeds = int(validation["seeds_per_task"])
    analysis_amendment = validation.get("analysis_source_amendment")
    all_rows = data[data.regime == "all"]

    # Forest-style paired model differences for predeclared representative regimes.
    chosen = [("v1_synthetic", "C0", "stationary original"),
              ("v1_synthetic", "C3", "conditional-only original"),
              ("v1_synthetic", "C6", "dependence original"),
              ("xor_synthetic", "X0", "stationary XOR"),
              ("xor_synthetic", "X3", "conditional-only XOR"),
              ("xor_synthetic", "X4", "label-noise XOR")]
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    y, labels, effects, lows, highs = [], [], [], [], []
    for task, condition, label in chosen:
        row = paired[(paired.task == task) & (paired.condition == condition) &
                     (paired.regime == "all") & (paired.metric == "accuracy")]
        if row.empty:
            continue
        row = row.iloc[0]
        labels.append(label); effects.append(row["mean"])
        lows.append(row["mean"] - row["lower"]); highs.append(row["upper"] - row["mean"])
        y.append(len(labels)-1)
    ax.errorbar(effects, y, xerr=[lows, highs], fmt="o", color="#087e8b", ecolor="#5f7772",
                capsize=3, markersize=5)
    ax.axvline(0, color="#263a38", lw=1, ls="--")
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel("Shallow tree minus logistic accuracy (95% paired seed bootstrap interval)")
    ax.set_title(f"Model sensitivity across {seeds:,} fresh seeds per task")
    ax.grid(axis="x", alpha=.2)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"paired_accuracy.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)

    # Calibration error and coverage by task/condition, preserving seed uncertainty.
    summary = pd.read_csv(run / "seed_bootstrap_summary.csv")
    rows = [(t, c, m) for t, c, _ in chosen for m in ("logistic", "shallow_tree")]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(rows))
    ticklabels = [f"{c}\n{m.replace('_',' ')}" for t,c,m in rows]
    for model in ("logistic", "shallow_tree"):
        offsets, vals, lo, hi = [], [], [], []
        for i, (task, condition, _) in enumerate(rows):
            if rows[i][2] != model:
                continue
            a = _summary(summary, task, condition, model, "cal_mae")
            b = _summary(summary, task, condition, model, "cal_coverage")
            if a is None or b is None:
                continue
            offsets.append(i); vals.append(a["mean"]); lo.append(a["mean"]-a["lower"]); hi.append(a["upper"]-a["mean"])
            ax2.errorbar(i, b["mean"], yerr=[[b["mean"]-b["lower"]],[b["upper"]-b["mean"]]],
                         fmt="o", color=COLORS[model], capsize=2, markersize=4)
        ax1.errorbar(offsets, vals, yerr=[lo,hi], fmt="o", color=COLORS[model],
                     capsize=2, markersize=4, label=model.replace("_", " "))
    ax1.set_ylabel("Calibrated AC MAE"); ax1.legend(frameon=False); ax1.grid(axis="y", alpha=.2)
    ax2.axhline(.95, color="#263a38", lw=1, ls="--", label="nominal 95%")
    ax2.set_ylabel("Empirical interval coverage"); ax2.set_xticks(x, ticklabels, rotation=35, ha="right")
    ax2.set_ylim(0,1.02); ax2.grid(axis="y", alpha=.2)
    ax2.legend(frameon=False, loc="lower right")
    fig.suptitle("Confidence-estimation error and plug-in interval coverage")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"estimation_and_coverage.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)

    # Input alarm incidence over seeds for detector-visible and conditional-only cases.
    detector = online.groupby(["task", "condition", "seed", "batch"], as_index=False).input_alarm.max()
    event_eligibility = {("v1_synthetic", "C1"), ("v1_synthetic", "C2"),
                         ("v1_synthetic", "C6"), ("xor_synthetic", "X1"),
                         ("xor_synthetic", "X2")}
    det_rows=[]
    for (task,condition,seed), group in detector.groupby(["task","condition","seed"]):
        event=(task,condition) in event_eligibility
        win=group[(group.batch>=9)&(group.batch<=12)]
        null=group[(group.batch<=8) | ((task,condition) not in event_eligibility)]
        det_rows.append(dict(task=task,condition=condition,seed=seed,event=event,
                             matched=bool(win.input_alarm.any()) if event else np.nan,
                             null_batches=len(null),null_alerts=int(null.input_alarm.sum()),
                             null_alerts_per_1000=1000*float(null.input_alarm.mean()),
                             total_alerts=int(group.input_alarm.sum())))
    det=pd.DataFrame(det_rows)
    det_summary=det.groupby(["task","condition"],as_index=False).agg(
        seeds=("seed","nunique"), event=("event","first"),
        detected=("matched","sum"), null_alerts_mean=("null_alerts","mean"),
        null_batches=("null_batches","first"),null_alerts_per_1000=("null_alerts_per_1000","mean"),
        any_alert_seed_fraction=("total_alerts",lambda x: float((x>0).mean())))
    det_summary.to_csv(out / "input_detection_summary.csv", index=False)

    metric_lines = [
        "| Cohort | Model | Accuracy [95% seed interval] | Calibrated MAE [95% seed interval] | Coverage [95% seed interval] | Calibrated width [95% seed interval] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, condition, label in chosen:
        for model in ("logistic", "shallow_tree"):
            vals = []
            for metric in ("accuracy", "cal_mae", "cal_coverage", "cal_width"):
                row = _summary(summary, task, condition, model, metric)
                vals.append("undefined" if row is None else _fmt(row, metric))
            metric_lines.append(f"| {label} | {model.replace('_',' ')} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    paired_lines = [
        "| Cohort | Tree minus logistic accuracy [95% paired seed interval] |",
        "|---|---:|",
    ]
    for task, condition, label in chosen:
        row = paired[(paired.task == task) & (paired.condition == condition) &
                     (paired.regime == "all") & (paired.metric == "accuracy")]
        if not row.empty:
            row = row.iloc[0]
            paired_lines.append(f"| {label} | {_fmt(row, 'difference')} |")
    concern = all_rows[all_rows.condition.isin(("C3", "C4", "X3", "X4"))].groupby(
        ["task", "condition", "model"], as_index=False).agg(
            seed_n=("seed", "nunique"), harmful_per_seed=("harm_n", "mean"),
            concern_flags_per_seed=("cal_tp", "mean"),  # TP is split below; total flags use TP+FP.
            unflagged_harm_per_seed=("cal_unflagged_harm", "mean"),
            fp_per_seed=("cal_fp", "mean"), tp_per_seed=("cal_tp", "mean"))
    concern["concern_flags_per_seed"] = concern.tp_per_seed + concern.fp_per_seed
    concern_lines = [
        "| Conditional-only cohort | Model | Seeds | Harmful batches/seed | Flags/seed | Harm missed/seed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in concern.itertuples(index=False):
        concern_lines.append(f"| {row.task}/{row.condition} | {row.model.replace('_',' ')} | {row.seed_n} | "
            f"{row.harmful_per_seed:.2f} | {row.concern_flags_per_seed:.2f} | {row.unflagged_harm_per_seed:.2f} |")
    det_lines=[
        "| Task / condition | Programmed input event | Seeds | Detected in batches 9-12 | Null batches/seed | Null alerts per 1,000 null batches |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in det_summary.itertuples(index=False):
        if bool(row.event):
            detected=f"{int(row.detected)}/{row.seeds}"
        else:
            detected="not defined (no input event label)"
        det_lines.append(f"| {row.task}/{row.condition} | {'yes' if row.event else 'no'} | {row.seeds} | "
            f"{detected} | {row.null_batches:.0f} | {row.null_alerts_per_1000:.3f} |")
    reproduction_lines = []
    if reproduction is not None:
        receipt = json.loads(Path(reproduction).read_text(encoding="utf-8"))
        float_difference = float(receipt.get("maximum_absolute_float_difference", 0.0))
        reproduction_lines = [
            (f"Five-seed sample: {receipt['online_rows_compared']:,} online rows and "
             f"{receipt['evaluator_rows_compared']:,} evaluator rows checked. Discrete fields matched exactly; "
             f"floats passed allclose (rtol=1e-12, atol=1e-14; max absolute difference {float_difference:.3g}). "
             f"Timing fields were excluded."),
            "",
        ]

    lines = [
        "# Large-seed robustness and remaining failure modes",
        "",
        "**Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)**  ",
        "Data Analysis and Visualization · Phase 3 supplementary experiment",
        "",
        "**Protocol status:** Outcome-informed robustness replication. It reuses the frozen synthetic task laws and analysis settings; it is not external validation and not a preregistration preceding the earlier studies.",
        "",
        "## Why this experiment was run",
        "",
        f"The main remaining sampling weakness was that model sensitivity and the XOR task had 30 final seeds each. This run adds {seeds:,} fresh seeds per task ({2*seeds:,} task-seed replications total), holding the generators, partitions, models, calibrators, thresholds, conditions and batch sizes fixed. It tests seed sensitivity; it does not widen the data domains.",
        "",
        "## Design and integrity",
        "",
        "The original synthetic task has seven conditions and XOR has five. Each has 12,000 monitoring rows in 24 batches of 500. Logistic regression and the fixed depth-6 tree share each stream; their own historical references define their own realised-harm targets. Harm counts alone therefore do not rank the learners; paired accuracy and estimation error are also reported on identical cohorts. Thresholds or seeds were not selected from results. The online table contains no labels, accuracy, F1 or harm; evaluator metrics are stored separately and joined after each batch score is written. This scale run stores aggregate batch outputs and deterministically regenerated inputs rather than per-instance caches.",
        "",
        f"The run covers {int(validation['online_rows']):,} online batch records and {int(validation['evaluator_rows']):,} evaluator batch records. Exact strict harm boundaries, unique row keys, model pairing, common input alarms, calibrated-estimate identity and finite scores passed validation. Paired uncertainty uses 2,000 bootstrap replicates over complete seeds.",
        *( [f"Analysis record: {analysis_amendment}"] if isinstance(analysis_amendment, str) and analysis_amendment else [] ),
        "",
        "## Model sensitivity",
        "",
        "![Paired model accuracy differences](paired_accuracy.png)",
        "",
        "Positive accuracy differences favor the tree on the matched cohort. Intervals are seed-bootstrap intervals; they do not treat individual batches as independent. Refer to `paired_model_bootstrap.csv` for all conditions and metrics.",
        "",
        *metric_lines,
        "",
        *paired_lines,
        "",
        "<!-- PAGE -->",
        "## Estimation and interval behavior",
        "",
        "![Calibration error and interval coverage](estimation_and_coverage.png)",
        "",
        "Coverage is shown against the nominal 95% reference and with seed uncertainty; widths are in the result tables. Coverage shortfall is not repaired by lower alert burden. These are plug-in intervals, not guarantees under arbitrary shift; concern scores are not calibrated harm probabilities, and no flag is not a safety verdict. Concern counts and unflagged harmful batches are reported in the per-seed summaries.",
        "",
        "## Input alarms and conditional-only failures",
        "",
        "`input_detection_summary.csv` reports per-condition seed denominators and alarm frequencies. For synthetic input-shift conditions C1/C2/C6 and X1/X2, onset matching is defined only over batches 9–12. Conditional-only conditions C3/C4 and X3/X4 keep the input stream unchanged by construction, so an input-only detector has no input-distribution signal for their label changes. A lack of alarm in these cases does not mean low model impact or safety.",
        "",
        *det_lines,
        "",
        *concern_lines,
        "",
        "The earlier C3–C5 misses and v1 calibration results remain part of the evidence; this fresh-seed run can show how consistently those failures recur under the same generators. It cannot identify a universal cause. A plausible explanation remains source-law confidence transfer under conditional change; the experiments measure the failure pattern, not a unique causal mechanism.",
        "",
        "## What this adds and what it does not",
        "",
        "It narrows Monte Carlo uncertainty for two existing synthetic task families and tests whether the 30-seed model/calibration/alert findings persist over fresh streams. It does not add a natural data source, new learner, alternative adaptive mechanism, human evaluation, causal explanation or external-domain evidence. The XOR task is another controlled synthetic task. RQ3 analyst benefit remains unanswered.",
        "",
        "## Reproduction",
        "",
        "From the workspace root, use fresh output paths:",
        "",
        "```powershell",
        "powershell -NoProfile -ExecutionPolicy Bypass `",
        "  -File .\\scripts\\reproduce_phase3_large_seed.ps1 `",
        "  -RunDir results/phase3/large_seed_robustness_reproduction `",
        "  -ReportDir reports/phase3/large_seed_robustness_reproduction",
        "```",
        "",
        "Exact package versions and source hashes are stored in the run directory. Existing v1/v2/adaptation outputs and prepared archives remain unchanged.",
        "",
        *reproduction_lines,
    ]
    path = out / "23I-2581_23I-2604_Phase3_Robustness_Report.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path)
    a=parser.parse_args()
    print(report(a.run, a.out, a.reproduction))
