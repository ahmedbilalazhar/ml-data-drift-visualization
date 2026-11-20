"""Create an editable, reproducible report for the exploratory midpoint-probe study."""
from __future__ import annotations

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
RUN = ROOT / "results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2"
ONE = ROOT / "results/phase3/adwin_audit_baseline_20261117_v1"
OUT = ROOT / "reports/phase3/adwin_midpoint_probe_sensitivity_20261118_v1"
COLORS = {"strict_cap": "#7C3AED", "exact_quota": "#087E8B"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    results = pd.read_csv(RUN / "two_probe_per_seed.csv")
    decisions = pd.read_csv(RUN / "two_probe_decisions.csv")
    contrasts = pd.read_csv(RUN / "paired_one_probe_contrasts.csv")
    previous = pd.read_csv(ONE / "adwin_per_seed.csv")

    detection = results.assign(detected=results.adwin_detection_events.gt(0)).groupby(
        ["cost_mode", "delay"], as_index=False
    ).agg(detection_cells=("detected", "sum"), cells=("detected", "size"),
          mean_audits=("audited_batches", "mean"),
          verified_episodes=("verified_harmful_episodes_by_stream_end", "mean"),
          mean_unspent=("unspent_budget_cohorts", "mean"))
    detection["detection_fraction"] = detection.detection_cells / detection.cells
    detection.to_csv(OUT / "midpoint_detection_summary.csv", index=False)

    previous_summary = previous.assign(detected=previous.adwin_detection_events.gt(0)).groupby(
        ["cost_mode", "delay"], as_index=False
    ).agg(one_probe_detection_cells=("detected", "sum"), one_probe_cells=("detected", "size"),
          one_probe_mean_audits=("audited_batches", "mean"))
    comparison = detection.merge(previous_summary, on=["cost_mode", "delay"], validate="one_to_one")
    comparison.to_csv(OUT / "one_vs_midpoint_summary.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    x = np.arange(4)
    delays = [0, 5, 20, 100]
    for mode, label in (("strict_cap", "Strict cap"), ("exact_quota", "Exact quota")):
        z = comparison[comparison.cost_mode == mode].set_index("delay").loc[delays]
        axes[0].plot(x, z.detection_fraction, marker="o", ms=6, lw=2.3,
                     color=COLORS[mode], label=f"Two probes · {label}")
        axes[0].plot(x, z.one_probe_detection_cells / z.one_probe_cells, marker="s",
                     ms=5, lw=1.4, ls="--", color=COLORS[mode], alpha=.58,
                     label=f"One probe · {label}")
    axes[0].set_xticks(x, ["0", "5", "20", "100"])
    axes[0].set_ylim(-.04, 1.04)
    axes[0].set_xlabel("Feedback delay (batches)")
    axes[0].set_ylabel("Cells with ≥1 ADWIN event")
    axes[0].set_title("Probe spacing restores short-delay feedback")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    axes[0].spines[["top", "right"]].set_visible(False)

    forest = contrasts[contrasts.metric == "episode_delta"]
    offset = {"strict_cap": -.09, "exact_quota": .09}
    for mode, label in (("strict_cap", "Strict cap"), ("exact_quota", "Exact quota")):
        z = forest[forest.cost_mode == mode].set_index("delay").loc[delays]
        yy = x + offset[mode]
        axes[1].errorbar(z.mean_difference, yy,
                         xerr=[z.mean_difference-z.lower, z.upper-z.mean_difference],
                         fmt="o", ms=5, capsize=3, lw=1.7, color=COLORS[mode], label=label)
    axes[1].axvline(0, color="#334155", lw=1, ls="--")
    axes[1].set_yticks(x, ["0", "5", "20", "100"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Verified harmful episodes by horizon\n(midpoint pair − one-probe; 95% paired seed bootstrap)")
    axes[1].set_ylabel("Feedback delay (batches)")
    axes[1].set_title("Episode verification changes at short delays")
    axes[1].grid(axis="x", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"midpoint_probe_effects.{ext}", dpi=240 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    rows = []
    for row in comparison.sort_values(["cost_mode", "delay"]).itertuples():
        rows.append(f"{row.cost_mode.replace('_', ' ')} & {row.delay} & "
                    f"{int(row.one_probe_detection_cells)}/{int(row.one_probe_cells)} & "
                    f"{int(row.detection_cells)}/{int(row.cells)} & {row.mean_audits:.2f} & "
                    f"{row.verified_episodes:.3f}" + r" \\")
    table = "\n".join(rows)
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,placeins}
\definecolor{navy}{HTML}{16324F}\definecolor{teal}{HTML}{087E8B}\definecolor{violet}{HTML}{7C3AED}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{When Supervised Drift Detectors Need Exploration}\\[3pt]
\large An Early--Midpoint Audit Sensitivity for Delayed ADWIN}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 research addendum}
\date{18 November 2026}
\begin{document}\maketitle
\begin{abstract}
We investigate whether placing two label-independent audit probes at stream batches 1 and 12 enables canonical River ADWIN to operate under strict audit caps. The exploratory replay uses the same frozen prediction streams, truth, 30 seeds, seven conditions, four feedback delays, three cohort budgets and exact-quota/strict-cap rules as the one-probe baseline. With the corrected schedule, ADWIN fires in 273/630 strict-cap cells at delays 0 and 5, compared with zero under one probe. This comes with greater audit use (2.29--2.38 versus one cohort on average); it is not an equal-spend detector comparison. At delays 20 and 100, the 24-batch horizon still prevents timely error feedback.\end{abstract}
\section{Question and protocol}
The previous strict-cap result exposed a cold-start deadlock: one bootstrap cohort was insufficient for an error-change detector to request a second cohort. We therefore tested a fixed, label-independent pair at batches 1 and 12. Each selected cohort has 500 rows, and the detector receives row-level zero/one prediction errors only after that cohort's labels are released. Source batch $s$ is usable at decision clock $t$ iff $s+d<t$. The cohort budget is 2, 4 or 8; probes are selected only while a cohort remains available. All other settings match the River 0.27.0 ADWIN baseline ($\delta=0.002$, \texttt{clock=1}).

This was a post-hoc exploratory sensitivity motivated by the prior one-probe and adjacent-probe findings. It reuses the same cached streams and is not independent confirmation. The one-probe contrast is paired by seed after averaging seven conditions and three budgets; a 2,000-replicate seed bootstrap forms the intervals. Under strict caps, the probe pair changes audit volume, so episode-yield changes combine feedback opportunity and spend.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{midpoint_probe_effects.pdf}
\caption{Left: fraction of seed--condition--budget cells with an ADWIN event, comparing the one-probe and corrected early--midpoint schedules. Right: paired change in harmful episodes verified by stream end; intervals resample 30 seeds after averaging matched conditions and budgets. Detection is not equivalent to correct harm localization.}
\end{figure}
\begin{table}[ht]
\centering\small\begin{tabular}{@{}llrrrr@{}}\toprule
Cost mode & Delay & One-probe events & Two-probe events & Mean audits & Verified episodes\\\midrule
''' + table + r'''\bottomrule\end{tabular}
\caption{Counts are out of 630 seed--condition--budget cells per delay and cost mode. Mean audits and verified episodes are descriptive means across those cells.}\end{table}
\FloatBarrier
\section{Interpretation}
The corrected pair prevents the strict-cap deadlock for short feedback delays: the detector obtains a second error sample at batch 12, and events occur in 273/630 cells at delay 0 and 273/630 at delay 5. The corresponding paired change in verified episodes versus one probe is +0.381 (95\% seed-bootstrap interval 0.327--0.457) at both delays. However, mean strict-cap audit use rises from 1.00 to 2.29 and 2.38 cohorts, respectively. These differences therefore do not isolate detector quality at equal cost. Under exact quotas, audit volume is fixed; paired episode changes are +0.133 (0.075--0.211) at delay 0 and +0.248 (0.194--0.329) at delay 5. For delays 20 and 100, no ADWIN events occur: released supervision is too late relative to the 24-batch observation horizon.

The v1 exploratory output is retained but superseded. Its implementation incorrectly compared a batch clock with a cohort count, causing the batch-12 probe to be skipped for all declared budgets. The v2 runner gates probes on remaining cohort capacity, and its contract tests verify both probes are selected, budget caps hold, delayed labels are not ingested early, and output hashes match.
\section{Limits and implications}
This result supports only a narrow engineering conclusion: supervised drift monitoring under sparse delayed labels needs an explicit exploration or bootstrap policy. It does not show that ADWIN is generally effective, that fixed probes are optimal, or that they outperform alternative detectors. The schedule consumes budget before evidence of harm and can waste audits in stationary streams; this report does not estimate acquisition cost or deployment utility. All comparisons reuse synthetic cached streams; no natural-stream transfer, causal attribution, analyst benefit or human-subject claim is made. A prospectively locked exploration policy and equal-spend comparison are appropriate next experiments.
\section{Reproduction}
From the project root:
\begin{verbatim}
python scripts/run_adwin_midpoint_probe_sensitivity.py
python scripts/report_adwin_midpoint_probe_sensitivity.py
python -m pytest tests/phase3/test_adwin_audit_baseline_contract.py -q
\end{verbatim}
The run lock records code/data/output SHA-256 hashes, dependencies, probe clocks, seeds and runtime. The TeX source compiles with Tectonic and the receipt records the generated figure and PDF hashes.
\end{document}
'''
    tex_path = OUT / "adwin_midpoint_sensitivity.tex"
    tex_path.write_text(tex, encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", str(tex_path)], cwd=ROOT, check=True)
    run_lock = json.loads((RUN / "protocol_lock.json").read_text(encoding="utf-8"))
    names = ["midpoint_detection_summary.csv", "one_vs_midpoint_summary.csv",
             "midpoint_probe_effects.pdf", "midpoint_probe_effects.png", "midpoint_probe_effects.svg",
             "adwin_midpoint_sensitivity.tex", "adwin_midpoint_sensitivity.pdf"]
    receipt = {"status": "report_compiled", "run_protocol": run_lock["protocol"],
               "run_lock_sha256": sha(RUN / "protocol_lock.json"),
               "outputs_sha256": {name: sha(OUT / name) for name in names}}
    (OUT / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Compiled TeX report and figures: {OUT}")


if __name__ == "__main__":
    main()
