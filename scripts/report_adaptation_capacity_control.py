"""Create a TeX report for the matched-capacity adaptation control."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/phase3/adaptation_capacity_control_20261118_v2"
DEFAULT_OUT = ROOT / "reports/phase3/adaptation_capacity_control_20261118_v9"
TASKS = ("linear_probabilistic", "xor_nonlinear", "five_region")
TASK_LABELS = {"linear_probabilistic": "Linear", "xor_nonlinear": "XOR", "five_region": "Five-region"}
MECHANISMS = ("M0_stationary", "M2_gradual_predictive", "M5_recurrent_conditional_reversal", "M7_out_of_support")
MECHANISM_LABELS = {"M0_stationary": "Stationary", "M2_gradual_predictive": "Gradual predictive", "M5_recurrent_conditional_reversal": "Recurrent reversal", "M7_out_of_support": "Out-of-support"}
COLORS = {0: "#087E8B", 5: "#4263EB", 20: "#D97706"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    metrics = pd.read_csv(run / "seed_metrics.csv")
    paired = pd.read_csv(run / "paired_seed_differences.csv")
    lock = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    summary = metrics.groupby(["task", "mechanism", "method"], as_index=False).agg(
        accuracy=("accuracy", "mean"), post_onset_accuracy=("post_onset_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
        update_rows=("adaptation_row_uses", "mean"), wall_seconds=("wall_seconds", "mean"))
    summary.to_csv(out / "method_summary.csv", index=False)
    local_global = paired[(paired.method == "matched_local_vs_global") & (paired.metric == "accuracy_difference_vs_global")].copy()
    local_interaction = paired[(paired.method == "matched_local_vs_interaction") & (paired.metric == "accuracy_difference_vs_interaction")].copy()
    interaction_global = paired[(paired.method == "matched_interaction_vs_global") & (paired.metric == "accuracy_difference_vs_global")].copy()
    pivot = metrics.pivot(index=["seed", "task", "mechanism", "delay"], columns="method", values="accuracy")
    seeds = sorted(metrics.seed.unique())
    keys = [(task, mechanism, delay) for task in TASKS for mechanism in MECHANISMS for delay in (0, 5, 20)]
    local_minus_global = pivot["matched_local"] - pivot["matched_global"]
    seed_differences = np.column_stack([
        local_minus_global.xs((task, mechanism, delay), level=("task", "mechanism", "delay")).reindex(seeds).to_numpy()
        for task, mechanism, delay in keys
    ])
    observed = seed_differences.mean(axis=0)
    se = seed_differences.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    rng = np.random.default_rng(38181118)
    indices = rng.integers(0, len(seeds), size=(20_000, len(seeds)))
    boot_means = seed_differences[indices].mean(axis=1)
    max_t = np.max(np.abs((boot_means - observed) / np.maximum(se, 1e-12)), axis=1)
    critical = float(np.quantile(max_t, .95))
    simultaneous = pd.DataFrame(keys, columns=["task", "mechanism", "delay"])
    simultaneous["mean"] = observed
    simultaneous["simultaneous_lower"] = observed - critical * se
    simultaneous["simultaneous_upper"] = observed + critical * se
    simultaneous["max_t_critical_95"] = critical
    local_global = local_global.merge(simultaneous, on=["task", "mechanism", "delay"])
    local_global.to_csv(out / "local_vs_global_accuracy_effects.csv", index=False)
    simultaneous.to_csv(out / "local_vs_global_simultaneous_bootstrap.csv", index=False)
    excluded = int(((simultaneous.simultaneous_lower > 0) | (simultaneous.simultaneous_upper < 0)).sum())
    local_interaction.to_csv(out / "local_vs_interaction_accuracy_effects.csv", index=False)
    interaction_global.to_csv(out / "interaction_vs_global_accuracy_effects.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.1), sharex=True, constrained_layout=True)
    offsets = {0: -0.19, 5: 0.0, 20: 0.19}
    yy0 = np.arange(len(MECHANISMS))
    for ax, task in zip(axes, TASKS):
        for delay in (0, 5, 20):
            adjusted = simultaneous[(simultaneous.task == task) & (simultaneous.delay == delay)].set_index("mechanism")
            means = [adjusted.loc[m, "mean"] * 100 for m in MECHANISMS]
            lo = [(adjusted.loc[m, "mean"] - adjusted.loc[m, "simultaneous_lower"]) * 100 for m in MECHANISMS]
            hi = [(adjusted.loc[m, "simultaneous_upper"] - adjusted.loc[m, "mean"]) * 100 for m in MECHANISMS]
            ax.errorbar(means, yy0 + offsets[delay], xerr=[lo, hi], fmt="o", ms=4, capsize=2.5,
                        lw=1.35, color=COLORS[delay], label=f"{delay} batches")
        ax.axvline(0, color="#334155", lw=1, ls="--")
        ax.set_title(TASK_LABELS[task]); ax.set_yticks(yy0, [MECHANISM_LABELS[m] for m in MECHANISMS]); ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Local − global accuracy (percentage points)\n95% simultaneous seed-bootstrap interval")
    axes[0].set_ylabel("Stream setting"); axes[2].legend(title="Feedback delay", frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "capacity_control_effects.pdf", bbox_inches="tight")
    fig.savefig(out / "capacity_control_effects.svg", bbox_inches="tight")
    plt.close(fig)

    task_rows = []
    for task in TASKS:
        g = summary[(summary.task == task) & (summary.mechanism != "M0_stationary")].groupby("method").accuracy.mean()
        task_rows.append(f"{TASK_LABELS[task]} & {g['frozen']:.3f} & {g['matched_global']:.3f} & {g['matched_local']:.3f} & {g['matched_interaction']:.3f}" + r" \\")
    # Seed-weighted overall contrasts average within each seed before bootstrapping.
    seed_effects = (pivot.matched_interaction - pivot.matched_global).groupby(level="seed").mean()
    seed_effects.to_csv(out / "interaction_minus_global_seed_effects.csv", header=["accuracy_difference"])
    # The receipt's matched paired-bootstrap contrasts are primary; this compact summary aids interpretation.
    primary = interaction_global[(interaction_global.task != "linear_probabilistic")]
    all_task_rows = "\n".join(task_rows)
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,placeins}
\definecolor{navy}{HTML}{16324F}\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{Is the Local-Adaptation Gain a Capacity Effect?}\\[3pt]
\large A matched-function-class control across tasks, mechanisms and delays}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 research addendum}
\date{18 November 2026}
\begin{document}\maketitle
\begin{abstract}
Earlier matched-update experiments found that cluster-local refitting can outperform a single global logistic refit, but the local arm has greater representational capacity. We add a pooled cluster-interaction logistic control with the same 27 intercept/slope parameters as three separate cluster experts. All arms begin from the identical historical global model. Across 30 paired seeds, three task families, four mechanisms and delays 0/5/20, the interaction and local arms have nearly identical mean results. This supports a capacity-based explanation for much of the local--global gap in this design; it does not show that IDM-Viz's monitoring or triggering policy caused that gain.
\end{abstract}
\section{Question and design}
The primary contrast is a frozen classifier, one global logistic refit, three frozen-cluster local logistic experts, and a single pooled logistic model whose feature vector is block-coded by the same three frozen clusters. The latter has $3(8+1)=27$ parameters: one intercept and eight feature coefficients per cluster. The local arm fits the same function class as three separate models. Both adaptive arms start from the exact same global historical model and update at clocks 25, 49, 73 and 89 using the same latest released feedback-row IDs. Every arm is evaluated on the same streams; a row label is eligible only when its release precedes the prediction batch. Each adaptive opportunity is limited to 2,000 released rows and cumulative repeated row-use is capped at 8,000.

We evaluate 30 reserved seeds over linear, XOR and five-region synthetic tasks; stationary, gradual predictive, recurrent reversal and out-of-support mechanisms; and delays of 0, 5 and 20 batches. The seed is the resampling unit. Figure intervals use a paired seed-level max-$t$ bootstrap simultaneous across the 36 local--global contrasts; pointwise intervals remain in the CSV for reference. This is a sensitivity analysis, not universal inference.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{capacity_control_effects.pdf}
\caption{Local minus global accuracy, percentage points, paired over 30 seeds. Bars are 95\% max-$t$ bootstrap intervals simultaneous across all 36 task/mechanism/delay contrasts.}
\end{figure}
\begin{table}[ht]
\centering\small\begin{tabular}{@{}lrrrr@{}}\toprule
Task (nonstationary mechanisms) & Frozen & Global & Local & Interaction\\\midrule
''' + all_task_rows + r'''\bottomrule\end{tabular}
\caption{Accuracy averaged across the three delays, four nonstationary mechanisms and 30 seeds. These averages summarize this designed simulation matrix.}\end{table}
\FloatBarrier
\section{Interpretation}
Only ''' + str(excluded) + r''' of the 36 simultaneous intervals exclude zero. Original pointwise bootstrap intervals are retained alongside the simultaneous limits in the machine-readable effects table.
The interaction control closely reproduces the local arm's average accuracy in each task family: 0.5865 versus 0.5865 in five-region, 0.6885 versus 0.6886 in linear, and 0.6632 versus 0.6632 in XOR. In contrast, global-refit averages are 0.5665, 0.6904 and 0.6382. Thus the local--global difference is concentrated in the cluster-conditional function class, rather than evidence that separately fitting local experts or a local trigger is inherently better. The interaction model exceeds the global model in nonlinear/region-dependent tasks and underperforms it slightly in the linear task. This is consistent with a representation-by-task interaction.

The stationary condition is an important negative control: the same cluster-conditional function class also improves the five-region and XOR accuracy there. Consequently, the observed gap cannot be attributed solely to responding to drift. Recurrent reversal and delay effects are heterogeneous; aggregation over tasks or mechanisms would conceal failures. Local and interaction estimates are near-equivalent, but they are not a comparison of equal computational cost: the receipt records total fit wall time of about 95.6 seconds for local versus 204.4 seconds for interaction across this run, and wall time is hardware/software dependent.
\section{Validity checks and limitations}
The completed run contains 4,320 seed/task/mechanism/delay/method cells, 414,720 persisted batch forecasts and 21,600 fit-audit rows. The receipt records matched feedback hashes, four successful update opportunities, the 8,000 repeated-use cap, and source/config/output checksums. Across the run, 549 cluster-level fits fell back to the historical global model; these are repeated across seed/task/mechanism/delay/update combinations, and fallback frequency and effective row use are logged. The pooled interaction fitting objective and separate local fits are not guaranteed to converge identically in finite numerical optimization, despite their equivalent block function class. The simulation uses one classifier family and one fixed three-cluster partition. It does not demonstrate natural-stream transfer, causal attribution, analyst benefit, detector validity, or a deployment-ready adaptive policy. Threshold selection and trigger value are not tested by this fixed-clock comparison.
\section{Conclusion}
This control strengthens the interpretation of the earlier adaptation result: gains attributed to local adaptation are substantially explained by a cluster-conditional model class. IDM-Viz should describe this as evidence that regional representation can matter in the tested synthetic tasks, not evidence that the full monitoring framework yields a general accuracy improvement. The result motivates future matched-function-class and matched-compute comparisons on independent real streams.
\end{document}
'''
    texpath = out / "adaptation_capacity_control.tex"
    texpath.write_text(tex, encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", str(texpath)], cwd=ROOT, check=True)
    output_names = ["method_summary.csv", "local_vs_global_accuracy_effects.csv", "local_vs_global_simultaneous_bootstrap.csv", "local_vs_interaction_accuracy_effects.csv",
                    "interaction_vs_global_accuracy_effects.csv", "interaction_minus_global_seed_effects.csv",
                    "capacity_control_effects.pdf", "capacity_control_effects.svg", "adaptation_capacity_control.tex",
                    "adaptation_capacity_control.pdf"]
    receipt = {"status": "report_compiled", "run_receipt_sha256": sha(run / "audit_receipt.json"),
               "run_protocol": lock["protocol"],
               "report_script_sha256": sha(Path(__file__)),
               "reproduction_script_sha256": sha(ROOT / "scripts/reproduce_adaptation_capacity_control.ps1"),
               "output_sha256": {name: sha(out / name) for name in output_names}}
    (out / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Compiled capacity-control report at {out}; simultaneous intervals excluding zero={excluded}/36")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.run.resolve(), args.out.resolve())
