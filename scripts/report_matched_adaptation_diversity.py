"""Build a TeX addendum for the fresh-seed adaptation-diversity sensitivity."""
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
DEFAULT_RUN = ROOT / "results/phase3/matched_adaptation_diversity_20261118_v1"
DEFAULT_OUT = ROOT / "reports/phase3/matched_adaptation_diversity_20261118_v1"
TASKS = ("linear_probabilistic", "xor_nonlinear", "five_region")
TASK_LABELS = {"linear_probabilistic": "Linear", "xor_nonlinear": "XOR", "five_region": "Five-region"}
MECHANISMS = ("M0_stationary", "M2_gradual_predictive",
              "M5_recurrent_conditional_reversal", "M7_out_of_support")
MECHANISM_LABELS = {"M0_stationary": "Stationary", "M2_gradual_predictive": "Gradual predictive",
                    "M5_recurrent_conditional_reversal": "Recurrent reversal",
                    "M7_out_of_support": "Out-of-support"}
DELAY_COLORS = {0: "#087E8B", 5: "#4263EB", 20: "#D97706"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def build(run: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    metrics = pd.read_csv(run / "seed_metrics.csv")
    paired = pd.read_csv(run / "paired_seed_differences.csv")
    lock = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))

    summary = metrics.groupby(["task", "mechanism", "method"], as_index=False).agg(
        accuracy=("accuracy", "mean"), post_onset_accuracy=("post_onset_accuracy", "mean"),
        row_uses=("adaptation_row_uses", "mean"), failed_local_region_fits=("failed_local_region_fits", "mean"),
        wall_seconds=("wall_seconds", "mean"))
    summary.to_csv(out / "method_summary.csv", index=False)

    local_global = paired[(paired.method == "matched_local_vs_global") &
                          (paired.metric == "accuracy_difference_vs_global")].copy()
    local_global.to_csv(out / "local_vs_global_accuracy_effects.csv", index=False)
    plot = local_global.set_index(["task", "mechanism", "delay"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharex=True, constrained_layout=True)
    offsets = {0: -0.18, 5: 0.0, 20: 0.18}
    mechanisms_y = np.arange(len(MECHANISMS))
    for ax, task in zip(axes, TASKS):
        for delay in (0, 5, 20):
            means, lower, upper = [], [], []
            for mechanism in MECHANISMS:
                row = plot.loc[(task, mechanism, delay)]
                means.append(row["mean"] * 100)
                lower.append((row["mean"] - row["lower"]) * 100)
                upper.append((row["upper"] - row["mean"]) * 100)
            yy = mechanisms_y + offsets[delay]
            ax.errorbar(means, yy, xerr=[lower, upper], fmt="o", ms=4.7,
                        capsize=2.5, lw=1.4, color=DELAY_COLORS[delay], label=f"{delay} batches")
        ax.axvline(0, color="#334155", lw=1, ls="--")
        ax.set_title(TASK_LABELS[task])
        ax.set_yticks(mechanisms_y, [MECHANISM_LABELS[m] for m in MECHANISMS])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Local − global accuracy (percentage points)\n95% paired seed-bootstrap interval")
    axes[0].set_ylabel("Controlled stream setting")
    axes[2].legend(title="Feedback delay", frameon=False, fontsize=8, loc="lower right")
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"adaptation_diversity_effects.{ext}", bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for task in TASKS:
        for mechanism in MECHANISMS:
            group = summary[(summary.task == task) & (summary.mechanism == mechanism)].set_index("method")
            delta = local_global[(local_global.task == task) &
                                 (local_global.mechanism == mechanism)]["mean"].mean() * 100
            table_rows.append(
                f"{TASK_LABELS[task]} & {MECHANISM_LABELS[mechanism]} & "
                f"{group.loc['frozen','accuracy']:.3f} & {group.loc['matched_global','accuracy']:.3f} & "
                f"{group.loc['matched_local','accuracy']:.3f} & {delta:+.2f}"
                + r" \\")
    table = "\n".join(table_rows)
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,placeins}
\definecolor{navy}{HTML}{16324F}\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{Does Cluster-Local Adaptation Transfer?}\\[3pt]
\large A Fresh-Seed Comparison Across Task Families and Drift Mechanisms}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 research addendum}
\date{18 November 2026}
\begin{document}\maketitle
\begin{abstract}
We broaden the matched-clock adaptation comparison beyond one linear abrupt-reversal example. Thirty fresh evaluation seeds cover three task families, four stream mechanisms and feedback delays 0, 5 and 20. Frozen, global-refit and cluster-local-refit models share update clocks and released row-ID windows. The comparison reveals strong task dependence: local models improve accuracy in the five-region and XOR generators, including stationary streams, but slightly underperform global refitting in most linear settings. The stationary gains show that this design does not isolate a drift-specific benefit of local adaptation.
\end{abstract}
\section{Protocol}
Each stream contains 96 batches of 500 rows, with 4,000 initial training rows. A training-fitted scaler and three-cluster MiniBatchKMeans assignment remain frozen. The two adaptive arms receive the same latest released feedback IDs at clocks 25, 49, 73 and 89; each refit uses at most 2,000 rows and repeated row uses count toward an 8,000-row cap. The local arm fits separate logistic models by frozen cluster and falls back to the initial global model when a cluster has fewer than 20 rows or only one observed class. Both arms receive the same opportunity schedule and feedback window, but they do not have equal parameter counts or equal CPU cost. Source labels obey $s+d<b$ for a prediction at batch $b$.

The task families are linear probabilistic classification, XOR classification and a five-region generator. The mechanisms are stationary, gradual predictive-feature shift, recurrent conditional reversal and out-of-support displacement. The 30-seed evaluation bank (73116301--73116330) was checked against existing text manifests and archive metadata; no exact collisions were found. Within each seed, comparisons are paired across task, mechanism and delay. The run is a broad sensitivity extension, not a newly tuned primary analysis; the 95\% intervals shown are per-setting and unadjusted across the 36 local--global comparisons.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{adaptation_diversity_effects.pdf}
\caption{Cluster-local minus global-refit all-stream accuracy, in percentage points. Points and intervals are paired across the same 30 seeds. Intervals are descriptive and unadjusted for the 36 settings. Positive values favor the local arm in this specific design.}
\end{figure}
\begin{table}[ht]
\centering\small\begin{tabular}{@{}llrrrr@{}}\toprule
Task & Mechanism & Frozen & Global & Local & Local--global (pp)\\\midrule
''' + table + r'''\bottomrule\end{tabular}
\caption{Accuracy averages first over the three delays and then over the 30 seeds. Local--global values average the paired setting-level differences over delays.}\end{table}
\FloatBarrier
\section{Interpretation}
The local--global effect changes sign across task families. For the five-region task, the local arm is higher in all four mechanisms, with delay-averaged differences ranging from roughly 0.88 to 2.91 percentage points. The XOR task shows larger local gains under stationarity and gradual shift, but the recurrent-reversal effect is small and becomes negative at delay 20; out-of-support differences are small and their intervals include zero. In the linear task, local refitting is generally slightly worse, although the recurrent-reversal comparison at delay 20 is a small positive exception. At the same time, local-vs-frozen post-onset accuracy in stationary data rises by about 3.96 points for the five-region task and 7.49 points for XOR, while it falls by about 0.40 points for the linear task. Therefore, improvements in regional and XOR cases can arise from the cluster-gated model's greater representational capacity, not solely from adaptation to drift. This is an important comparator limitation and a reason not to claim local adaptation is generally superior.

Global refitting also does not guarantee improvement: under recurrent reversal in the linear and XOR families, both adaptive arms underperform their frozen comparators on average. Out-of-support shifts benefit the five-region arm but not the linear or XOR arms. These heterogeneous effects argue against collapsing mechanisms or task families into one overall ranking.
\section{Limitations}
This experiment holds update clocks and label windows fixed; it does not compare the trigger policy's value of choosing when to adapt. It uses a single logistic family, one three-cluster partition and simulated labels. The global and local arms have different effective model capacities and the local arm can consume fewer fitting rows when fallbacks occur (549 unsupported local-cluster fits across the evaluated cells). No natural-stream adaptation is included. Wall time is recorded, but memory, energy and acquisition prices are not measured. The seed bootstrap supports within-generator variability only; it does not make task generators equivalent to independent real-world datasets. No human decision-effect or causal claim is made.
\section{Reproduction}
From the workspace root:
\begin{verbatim}
powershell -ExecutionPolicy Bypass -File scripts/reproduce_adaptation_diversity.ps1
\end{verbatim}
The reproduction script creates new timestamped output paths and uses the same locked configuration. The run receipt includes source/config/output hashes, environment versions, seeds, timing and the exact matrix. The accompanying contract tests check cell counts, timing, matched IDs, row-use limits and report hashes.
\end{document}
'''
    tex_path = out / "matched_adaptation_diversity.tex"
    tex_path.write_text(tex, encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", str(tex_path)], cwd=ROOT, check=True)
    outputs = ["method_summary.csv", "local_vs_global_accuracy_effects.csv",
               "adaptation_diversity_effects.pdf", "adaptation_diversity_effects.svg",
               "matched_adaptation_diversity.tex", "matched_adaptation_diversity.pdf"]
    receipt = {"status": "report_compiled", "run_receipt_sha256": sha(run / "audit_receipt.json"),
               "run_protocol": lock["protocol"],
               "reproduction_script_sha256": sha(ROOT / "scripts/reproduce_adaptation_diversity.ps1"),
               "output_sha256": {name: sha(out / name) for name in outputs}}
    (out / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Compiled report at {out}; {len(local_global)} paired setting effects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.run.resolve(), args.out.resolve())
