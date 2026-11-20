from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/phase3/external_uci224_day2_v2"
OUT = ROOT / "reports/phase3/day2_external_v2/figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)

cohorts = pd.read_csv(RESULTS / "cohort_metrics.csv").query("batch >= 2")
positions = np.arange(len(cohorts))
width = 0.34
fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.2, 1]})
axes[0].bar(positions - width / 2, cohorts.accuracy, width, label="Observed accuracy",
            color="#20639B")
axes[0].bar(positions + width / 2, cohorts.estimated_accuracy, width,
            label="Estimated accuracy", color="#F6A04D")
axes[0].errorbar(positions + width / 2, cohorts.estimated_accuracy,
                 yerr=np.vstack([cohorts.estimated_accuracy - cohorts.prediction_interval_lower,
                                 cohorts.prediction_interval_upper - cohorts.estimated_accuracy]),
                 fmt="none", ecolor="#6B4C9A", capsize=2, linewidth=1.2,
                 label="95% plug-in predictive interval")
axes[0].set_ylim(0, 1.04)
axes[0].set_ylabel("Accuracy")
axes[0].legend(ncol=3, loc="lower left", frameon=True)
axes[0].set_title("Frozen model performance and correctness-based estimate")
axes[1].bar(positions, cohorts.macro_f1_fixed_six_classes, color="#41A690")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Macro-F1\n(fixed 6 classes)")
axes[1].set_xticks(positions, [f"Batch {b}" for b in cohorts.batch])
axes[1].set_xlabel("Whole acquisition cohort (coarse order; no within-batch times)")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"cohort_performance.{ext}", dpi=260, bbox_inches="tight")
plt.close(fig)

class_rows = pd.read_csv(RESULTS / "class_metrics.csv")
proportions = class_rows.pivot(index="batch", columns="class", values="proportion")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
sns.heatmap(proportions, annot=True, fmt=".2f", cmap="crest", vmin=0, vmax=0.6,
            linewidths=0.5, cbar_kws={"label": "Within-cohort class proportion"}, ax=ax)
ax.set_xlabel("Gas class ID")
ax.set_ylabel("Acquisition batch")
ax.set_title("Class composition varies substantially across acquisition cohorts")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"class_composition.{ext}", dpi=260, bbox_inches="tight")
plt.close(fig)
print(f"Wrote figures to {OUT}")
