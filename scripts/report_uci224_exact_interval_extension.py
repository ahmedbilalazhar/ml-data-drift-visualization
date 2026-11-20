"""Create a vector comparison figure and LaTeX note for the exact UCI interval check."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/phase3/external_uci224_exact_interval_20261116_v1"
OUT = ROOT / "reports/phase3/uci224_exact_intervals_20261116"

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    d = pd.read_csv(DATA / "cohort_interval_comparison.csv")
    s = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(8.7, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    x = d.batch.to_numpy()
    axes[0].fill_between(x, d.exact_lower, d.exact_upper, color="#16817a", alpha=.18,
                         label="Exact plug-in interval")
    axes[0].errorbar(x, d.estimate, yerr=[d.estimate-d.exact_lower, d.exact_upper-d.estimate],
                     fmt="s", ms=5, color="#16817a", capsize=3, label="Calibrated estimate + exact interval")
    axes[0].scatter(x, d.accuracy, marker="D", s=34, color="#c45e37", zorder=4,
                    label="Observed accuracy (evaluator only)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.04)
    axes[0].set_title("Exact plug-in intervals do not resolve calibration transfer")
    axes[0].grid(axis="y", alpha=.22)
    axes[0].legend(frameon=False, ncol=1, loc="lower left", fontsize=8)
    width_pct = 100 * d[["exact_width", "normal_width"]].to_numpy()
    axes[1].plot(x, width_pct[:, 0], "o-", color="#16817a", lw=1.8, label="Exact")
    axes[1].plot(x, width_pct[:, 1], "s--", color="#536578", lw=1.6, label="Normal + continuity correction")
    axes[1].set_ylabel("Interval width (%)")
    axes[1].set_xlabel("UCI 224 acquisition cohort")
    axes[1].set_xticks(x)
    axes[1].grid(axis="y", alpha=.22)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"interval_comparison.{ext}", dpi=240)
    plt.close(fig)
    tex = rf'''\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype,booktabs,graphicx,xcolor,hyperref}}
\hypersetup{{hidelinks}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{6pt}}
\title{{IDM-Viz: Exact-Interval Sensitivity on UCI 224}}
\author{{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}}
\date{{16 November 2026}}
\begin{{document}}\maketitle
\begin{{abstract}}
We recomputed the existing UCI 224 plug-in correctness intervals using the exact Poisson-binomial distribution and compared them with the original continuity-corrected normal approximation on identical cached predictions. Exact intervals covered 0 of 9 monitoring cohorts, the same count as the normal approximation. Their mean width was {s['mean_exact_width']:.4f}, versus {s['mean_normal_width']:.4f}. Thus normal approximation error does not explain the external calibration failure.
\end{{abstract}}
\section{{Design}}
This post-hoc sensitivity extension used frozen probabilities from the existing single-model UCI 224 cohort study. No model or calibrator was refit and no cohort outcome was used to select a method. For cohort $b$, the exact calculation forms the distribution of $S_b=\sum_i Z_i$ with independent $Z_i\sim\mathrm{{Bernoulli}}(q_i)$, then reports discrete 2.5\% and 97.5\% quantiles of $S_b/n_b$. The comparator recomputes the original normal approximation with continuity correction from the same $q_i$. The nine sequential cohorts are descriptive, not independent replications.
\section{{Results}}
The mean absolute error of the calibrated accuracy point estimate remained {s['calibration_transfer_error_unchanged']:.3f}. Exact intervals were narrower by {abs(s['mean_width_difference_exact_minus_normal']):.4f} on average; mean absolute endpoint displacement was {s['mean_absolute_endpoint_difference']:.4f}. Both approaches covered 0/9 realized cohort accuracies. The substantial gap between estimated and realized accuracy is therefore not fixed by replacing the approximation algorithm.
\begin{{figure}}[ht]\centering
\includegraphics[width=0.93\linewidth]{{interval_comparison.pdf}}
\caption{{Observed cohort accuracy compared with the fixed calibrated correctness estimate and its exact plug-in interval (top); exact and normal-approximation interval widths (bottom). Outcomes are shown for retrospective evaluation, never as online inputs.}}
\end{{figure}}
\section{{Interpretation and limits}}
The exact calculation is numerically preferable for this finite Poisson-binomial model, but it does not repair a correctness model that transfers poorly across cohorts. Both intervals condition on fitted $q_i$ values, assume independent correctness indicators, omit calibration and reference uncertainty, and lack arbitrary-shift coverage guarantees. The one UCI dataset and its acquisition cohorts do not establish domain generality. This analysis is exploratory and does not supersede the original external result.
\section{{Reproduction}}
\begin{{verbatim}}
python scripts/evaluate_uci224_exact_interval_extension.py \\
  --out results/phase3/external_uci224_exact_interval_NEW
python scripts/report_uci224_exact_interval_extension.py
\end{{verbatim}}
The original Phase 2 DOCX and earlier reports are unchanged.\end{{document}}
'''
    (OUT / "UCI224_EXACT_INTERVALS.tex").write_text(tex, encoding="utf-8")
    (OUT / "UCI224_EXACT_INTERVALS.md").write_text(
        f"# Exact interval sensitivity on UCI 224\n\nExact interval coverage was 0/9, matching the normal approximation (0/9). "
        f"Mean width was {s['mean_exact_width']:.4f} versus {s['mean_normal_width']:.4f}; calibrated point-estimate MAE remained "
        f"{s['calibration_transfer_error_unchanged']:.3f}. Exact calculation does not correct calibration transfer. See the TeX report and figure.",
        encoding="utf-8")
    print(OUT)

if __name__ == "__main__":
    main()
