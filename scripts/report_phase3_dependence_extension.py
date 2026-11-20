"""Create figures and an editable LaTeX supplement for the C6 study."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase3/dependence_extension_20261116_final_v2"
OUT = ROOT / "reports/phase3/dependence_extension_20261116"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(RUN / "validated_summary.csv")
    paired = pd.read_csv(RUN / "paired_contrasts.csv")
    dev = json.loads((ROOT / "results/phase3/dependence_extension_20261116_development_v4/threshold.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    colors = {"max_abs_correlation_change": "#167c80", "marginal_plus_PCA_KS": "#cc7043"}
    labels = {"max_abs_correlation_change": "Pairwise correlation", "marginal_plus_PCA_KS": "KS + PCA"}
    c6 = summary[summary.condition.eq("C6")]
    for row in c6.itertuples():
        axes[0].bar(labels[row.detector], row.onset_detection_rate, color=colors[row.detector], width=.58)
    axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("Detected by batches 9–12")
    axes[0].set_title("Dependence change (C6)")
    c0 = summary[summary.condition.eq("C0")]
    for row in c0.itertuples():
        rate = row.mean_alarms_per_24_batches / 24 * 1000
        axes[1].bar(labels[row.detector], rate, color=colors[row.detector], width=.58)
    axes[1].set_ylabel("Alarms per 1,000 stationary batches")
    axes[1].set_title("Stationary false-alarm burden (C0)")
    for ax in axes:
        ax.grid(axis="y", alpha=.22); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=10)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"dependence_tradeoff.{ext}", dpi=220)
    plt.close(fig)

    cond = summary.pivot(index="condition", columns="detector", values="onset_detection_rate").reset_index()
    cond = cond.rename(columns={"max_abs_correlation_change": "Correlation", "marginal_plus_PCA_KS": "KS_PCA"})
    cond["Correlation"] = cond.Correlation.map(lambda x: f"{x:.3f}")
    cond["KS_PCA"] = cond.KS_PCA.map(lambda x: f"{x:.3f}")
    contrast = paired.iloc[0]
    c6c = summary[(summary.condition == "C6")].set_index("detector")
    c0c = summary[(summary.condition == "C0")].set_index("detector")
    table_rows = "\n".join(f"{r.condition} & {r.Correlation} & {r.KS_PCA} " + r"\tabularnewline" for r in cond.itertuples(index=False))
    tex = r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,tabularx,graphicx,xcolor}
\usepackage{hyperref}
\hypersetup{hidelinks}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\title{IDM-Viz: Exploratory Dependence-Change Detector Evaluation}
\author{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}
\date{16 November 2026}
\begin{document}\maketitle
\begin{abstract}
We evaluated a feature-only maximum pairwise-correlation-change score against the existing eight-feature plus three-PCA-coordinate KS alarm. On 30 fresh paired synthetic seeds, the new score detected the programmed C6 dependence change in all 30 onset windows, compared with 16 of 30 for KS/PCA. It selected the programmed pair (X0,X1) in all 30 seeds. The gain came with higher stationary alarm burden: 1.70 versus 0.83 alarms per 24 batches. This exploratory result supports a targeted complement for the tested dependence intervention; it does not establish general dependence detection or harm prediction.
\end{abstract}
\section{Protocol and scope}
The statistic is the maximum absolute off-diagonal difference between current-batch and historical-reference Pearson correlation matrices. The reference is the final 1,000 rows of the canonical 2,000-row reference partition. A single 95th-percentile higher-rule threshold was chosen using 720 stationary development batch scores from seeds 930000--930029. The frozen threshold is ''' + f"{dev['threshold']:.6f}" + r'''. Evaluation used seeds 930100--930129 and all seven original conditions, with 24 batches of 500 rows. The same streams were scored by a recomputed training-only standardized KS family over eight original features and three training-fitted PCA coordinates, with Holm correction at 0.05.

The study was designed after the previous 1,500-seed study revealed weak C6 sensitivity. It is therefore an exploratory follow-up, despite using fresh evaluation seeds. No labels enter either detector. The maximizing pair is descriptive and is not a causal explanation.
\section{Results}
\begin{table}[ht]\centering\small
\caption{Onset detection frequency by condition. Onset is batches 9--12; values are proportions over 30 seeds.}
\begin{tabular}{lrr}\toprule Condition & Correlation & KS/PCA \\\midrule
''' + table_rows + r'''
\bottomrule\end{tabular}\end{table}

For C6, the paired detection-rate difference was ''' + f"{contrast.mean_difference:.3f}" + r''' (95\% seed-bootstrap interval ''' + f"[{contrast.lower:.3f}, {contrast.upper:.3f}]" + r'''; 2,000 resamples). The correlation detector selected pair (X0,X1) during an alarm in 30/30 C6 onset windows. This is exact recovery of the programmed pair in this generator, not general feature-pair localization.

On stationary C0, the correlation score produced ''' + f"{c0c.loc['max_abs_correlation_change','mean_alarms_per_24_batches']:.2f}" + r''' alarms per 24 batches (70.8 per 1,000 batches) versus ''' + f"{c0c.loc['marginal_plus_PCA_KS','mean_alarms_per_24_batches']:.2f}" + r''' (34.7 per 1,000) for KS/PCA. The paired difference was ''' + f"{paired.iloc[1].mean_difference:.2f}" + r''' alarms per stream (95\% seed-bootstrap interval ''' + f"[{paired.iloc[1].lower:.2f}, {paired.iloc[1].upper:.2f}]" + r'''). The correlation detector's pre-onset stream false-alarm rate in C0 was 40\% versus 16.7\% for KS/PCA. The threshold's per-batch development quantile did not control stream-level false alarms to the comparator's burden.

\begin{figure}[ht]\centering\includegraphics[width=.9\linewidth]{dependence_tradeoff.pdf}
\caption{Onset detection and stationary alarm burden. The detector improves sensitivity for C6 while increasing alarms under stationarity.}\end{figure}
\section{Interpretation and limits}
The result justifies retaining pairwise correlation evidence as a separate targeted signal in this synthetic setting, with explicit alert-burden costs. It does not show that correlation is the cause of model harm, that alarms predict loss, or that operators benefit. C1/C2 marginal shifts remain more readily detected by the KS/PCA family. The max-pair statistic can miss higher-order dependence changes and its threshold needs a burden-aware calibration rule before deployment. Only one synthetic dependence mechanism, one learner-independent input stream and 30 fresh seeds were evaluated.

\section{Reproduction}
From the repository root, run development before evaluation:
\begin{verbatim}
python scripts/run_phase3_dependence_extension.py --mode development `
  --out results/dev_NEW
python scripts/run_phase3_dependence_extension.py --mode final `
  --threshold-file results/dev_NEW/threshold.json `
  --out results/final_NEW
python scripts/evaluate_phase3_dependence_extension.py --run results/final_NEW
\end{verbatim}
The actual development and final CSVs, threshold locks, source hashes and validation receipt accompany this report.\end{document}
'''
    (OUT / "DEPENDENCE_EXTENSION.tex").write_text(tex, encoding="utf-8")
    (OUT / "DEPENDENCE_EXTENSION.md").write_text(
        "# Dependence detector follow-up\n\n" +
        f"C6 onset detection: 30/30 versus 16/30 for KS/PCA; paired difference {contrast.mean_difference:.3f} "
        f"(95% seed bootstrap [{contrast.lower:.3f}, {contrast.upper:.3f}]). Stationary alarms per 24 batches: "
        f"{c0c.loc['max_abs_correlation_change','mean_alarms_per_24_batches']:.2f} versus "
        f"{c0c.loc['marginal_plus_PCA_KS','mean_alarms_per_24_batches']:.2f}. This is an exploratory, mechanism-specific result.\n",
        encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
