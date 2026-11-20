"""Create paired-effect figures and a TeX report for matched adaptation runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"matched_global": "#2673a8", "matched_local": "#16866f"}
LABELS = {"matched_global": "Global sliding window", "matched_local": "Cluster-local"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    runs = pd.read_csv(run / "seed_metrics.csv")
    paired = pd.read_csv(run / "paired_seed_differences.csv")
    means = runs.groupby(["mechanism", "delay", "method"], as_index=False).agg(
        accuracy=("accuracy", "mean"), post_onset_accuracy=("post_onset_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
        successful_updates=("successful_updates", "mean"), row_uses=("adaptation_row_uses", "mean"),
        wall_seconds=("wall_seconds", "mean"),
    )
    means.to_csv(out / "policy_means.csv", index=False)
    paired.to_csv(out / "paired_seed_differences.csv", index=False)

    acc = paired[paired.metric == "accuracy_difference_vs_frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True, constrained_layout=True)
    cases = (("M0_stationary", "Stationary stream"),
             ("M4_abrupt_conditional_reversal", "Abrupt conditional reversal"))
    for ax, (mechanism, title) in zip(axes, cases):
        for method in COLORS:
            group = acc[(acc.mechanism == mechanism) & (acc.method == method)].sort_values("delay")
            offset = -0.10 if method == "matched_global" else 0.10
            y = np.arange(3) + offset
            ax.errorbar(group["mean"], y,
                        xerr=[group["mean"] - group.lower, group.upper - group["mean"]],
                        fmt="o", capsize=3, lw=1.7, ms=5, color=COLORS[method], label=LABELS[method])
        ax.axvline(0, color="#334155", lw=1, ls="--")
        ax.set_yticks(range(3), ["0 batches", "5 batches", "20 batches"])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Accuracy difference vs frozen\n(95% paired seed-bootstrap interval)")
        ax.grid(axis="x", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Label-release delay")
    axes[1].legend(frameon=False, loc="lower right")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"matched_adaptation_effects.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)

    table = []
    for mechanism, label in cases:
        short = "Stationary" if mechanism == "M0_stationary" else "Reversal"
        for delay in (0, 5, 20):
            group = means[(means.mechanism == mechanism) & (means.delay == delay)].set_index("method")
            table.append(
                f"{short} & {delay} & {group.loc['frozen','accuracy']:.3f} & "
                f"{group.loc['matched_global','accuracy']:.3f} & {group.loc['matched_local','accuracy']:.3f} & "
                f"{group.loc['matched_global','row_uses']:.0f} & {group.loc['matched_local','row_uses']:.0f} \\\\"
            )
    reversal_local = acc[(acc.mechanism == "M4_abrupt_conditional_reversal") &
                         (acc.method == "matched_local")].set_index("delay")
    effects = "; ".join(
        f"{100*reversal_local.loc[d,'mean']:+.2f} pp ({100*reversal_local.loc[d,'lower']:+.2f}, "
        f"{100*reversal_local.loc[d,'upper']:+.2f})"
        for d in (0, 5, 20)
    )
    direct = paired[paired.metric == "accuracy_difference_vs_global"]
    reversal_direct = direct[direct.mechanism == "M4_abrupt_conditional_reversal"].set_index("delay")
    direct_effects = "; ".join(
        f"{100*reversal_direct.loc[d,'mean']:+.2f} pp ({100*reversal_direct.loc[d,'lower']:+.2f}, "
        f"{100*reversal_direct.loc[d,'upper']:+.2f})"
        for d in (0, 5, 20)
    )
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{placeins}
\usepackage{xcolor}
\usepackage{hyperref}
\definecolor{navy}{HTML}{16324F}
\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{Matched-Opportunity Adaptation}\\[3pt]\large Global and Cluster-Local Refitting Under a Shared Row Budget}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 addendum}
\date{17 November 2026}
\begin{document}\maketitle
\begin{abstract}
This controlled extension holds update clocks and released feedback rows fixed while comparing a frozen classifier with global and cluster-local logistic refits. Thirty paired seeds, three label delays and stationary or abrupt conditional-reversal streams are evaluated. Each adaptive policy receives four opportunities and may use at most 2,000 released rows per opportunity; repeated uses count toward an 8,000-row adaptation cap. Initial training is reported separately.
\end{abstract}
\section{Protocol}
Each stream has 96 batches of 500 rows. The logistic model, scaler and three-cluster MiniBatchKMeans partition are initialized from the 4,000-row training partition; the clusterer is frozen thereafter. Both adaptive policies receive the same latest released row IDs at forecast clocks 25, 49, 73 and 89. The global policy fits one logistic model on the shared window. The local policy fits one replacement model per cluster on that cluster's rows from the same window; a cluster with fewer than 20 rows or a single observed class retains the original global fallback. Its actual fit-row use can therefore be lower than the shared window allowance.

For delay $d$, source batch $s$ can affect forecast batch $b$ only if $s+d<b$. Current-batch labels are excluded before prediction. A fitted label may recur in later windows, and each recurrence counts against the row-use total. Uncertainty resamples paired seeds; batches are not treated as independent replicates.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{matched_adaptation_effects.pdf}
\caption{Paired all-stream accuracy differences from the frozen classifier. Each interval is a 95\% bootstrap interval over the 30 matched seeds.}
\end{figure}
\begin{table}[ht]
\centering\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Stream & Delay & Frozen & Global & Local & Global rows & Local rows\\
\midrule
'''
    tex += "\n".join(table)
    tex += r'''
\bottomrule
\end{tabular}
\caption{Mean all-stream accuracy and mean cumulative adaptation-row uses per seed. Initial training is excluded from adaptation-row uses.}
\end{table}
For abrupt conditional reversal, local-minus-frozen accuracy differences at delays 0, 5 and 20 are ''' + effects + r''' percentage points, respectively; parentheses give 95\% paired seed-bootstrap intervals in percentage points. Direct local-minus-global differences at the same delays are ''' + direct_effects + r''' points. Here global refitting is slightly better at all delays, while both adaptive arms recover substantially relative to the frozen classifier. This comparative result is specific to this generator, partition, window and update schedule; it does not establish superiority on other mechanisms or natural streams.

The matched design answers whether global and local training choices behave differently when their four update opportunities and feedback row windows are held constant. It does not answer whether an error-triggered policy is better than periodic adaptation; that distinct budget-cap comparison is reported in the companion trigger audit. In the stationary condition, updates can be unnecessary by construction, and any accuracy loss is a measured cost rather than a false alarm.

\FloatBarrier
\section{Limitations}
The experiment uses one linear-probabilistic generator, one classifier and one training-fitted three-cluster partition. The local arm sees the same released IDs but fits separate models on region slices; its cumulative model-fit rows are at most the global arm's because disjoint regions partition the shared window. This is not a comparison of equal compute or equal wall time. The row-use cap measures examples passed to fitting, not CPU, memory, energy or acquisition cost. More delays, nonlinear and regional tasks, alternative cluster counts and natural streams remain untested here.

\section{Reproduction}
From the workspace root, run:
\begin{verbatim}
python scripts/run_matched_adaptation_schedule.py
python scripts/report_matched_adaptation_schedule.py \\
  --run results/phase3/matched_adaptation_schedule_20261117_v1 \\
  --out reports/phase3/matched_adaptation_schedule_20261117_v1
tectonic -X compile \\
  reports/phase3/matched_adaptation_schedule_20261117_v1/matched_adaptation_schedule.tex \\
  --outdir reports/phase3/matched_adaptation_schedule_20261117_v1
\end{verbatim}
The run receipt records source, configuration and output hashes; the fit audit stores the shared feedback-ID hash at each update opportunity.
\end{document}
'''
    (out / "matched_adaptation_schedule.tex").write_text(tex, encoding="utf-8")
    source_receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    outputs = [out / name for name in (
        "policy_means.csv", "paired_seed_differences.csv", "matched_adaptation_schedule.tex",
        "matched_adaptation_effects.png", "matched_adaptation_effects.pdf",
        "matched_adaptation_effects.svg")]
    receipt = {
        "status": "report_sources_complete", "run_receipt_sha256": digest(run / "audit_receipt.json"),
        "run_protocol": source_receipt["protocol"],
        "output_sha256": {p.name: digest(p) for p in outputs},
    }
    (out / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}; {len(paired)} paired contrasts; seeds={len(source_receipt['seed_bank'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    build(options.run, options.out)
