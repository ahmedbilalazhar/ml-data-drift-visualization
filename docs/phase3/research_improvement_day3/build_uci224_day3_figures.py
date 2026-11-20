from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/phase3/external_uci224_day2_v2"
SENSITIVITY = ROOT / "results/phase3/external_uci224_day2_sensitivity_v2"
OUT = ROOT / "reports/phase3/day3_final/figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)

confidence = pd.read_csv(SENSITIVITY / "confidence_transfer.csv")
confidence = confidence[confidence.cohort.str.fullmatch(r"batch\d+")].copy()
confidence["batch"] = confidence.cohort.str.replace("batch", "", regex=False).astype(int)
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(confidence.batch, confidence.accuracy, marker="o", lw=2, label="Observed accuracy")
ax.plot(confidence.batch, confidence.mean_confidence, marker="s", lw=2,
        label="Mean maximum class probability")
ax.set(xticks=range(2, 11), ylim=(0, 1.02), xlabel="Whole acquisition cohort",
       ylabel="Mean / accuracy", title="Source confidence does not transfer to later cohorts")
ax.legend(frameon=True)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"confidence_transfer.{ext}", dpi=260, bbox_inches="tight")
plt.close(fig)

decomp = pd.read_csv(SENSITIVITY / "class_mix_sensitivity.csv")
observed = decomp[decomp["class"] == "AGGREGATE_OBSERVED"].sort_values("batch")
standard = decomp[decomp["class"] == "COMMON_CLASS_STANDARDIZED"].sort_values("batch")
fig, ax = plt.subplots(figsize=(9, 4.6))
positions = np.arange(len(observed))
ax.bar(positions - .18, observed.accuracy, .36, label="Aggregate observed", color="#20639B")
ax.bar(positions + .18, standard.accuracy, .36,
       label="Common-class standardized", color="#F6A04D")
ax.set(xticks=positions, xticklabels=[f"{b}" for b in observed.batch], ylim=(0, 1),
       xlabel="Whole acquisition cohort", ylabel="Accuracy",
       title="Class-mixture sensitivity (classes present in each cohort only)")
ax.legend(frameon=True)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"class_mix_sensitivity.{ext}", dpi=260, bbox_inches="tight")
plt.close(fig)
print(f"Wrote figures to {OUT}")
