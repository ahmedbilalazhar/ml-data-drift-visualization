"""Summarize the locked delayed-label audit-policy matrix with figures and TeX."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"uniform": "#64748b", "input": "#2673a8", "concern": "#16866f",
          "released_error": "#d97706", "conjunction": "#7c3aed", "disjunction": "#c2415c"}
LABELS = {"uniform": "Uniform", "input": "Input alarm", "concern": "Concern",
          "released_error": "Released-label error", "conjunction": "Input AND concern",
          "disjunction": "Input OR concern OR error"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(run: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    per_seed = pd.read_csv(run / "policy_per_seed.csv")
    lock = json.loads((run / "protocol_lock.json").read_text(encoding="utf-8"))
    metrics = ["audited_batches", "harmful_audits", "verified_harmful_episodes_by_stream_end",
               "unnecessary_audits", "warning_fraction", "missed_harmful_batches"]
    means = per_seed.groupby(["condition", "delay", "budget_cohorts", "cost_mode", "policy"],
                             as_index=False)[metrics].mean()
    means.to_csv(out / "policy_means.csv", index=False)
    summary = means.groupby(["delay", "cost_mode", "policy"], as_index=False)[metrics].mean()
    summary.to_csv(out / "policy_summary_across_conditions.csv", index=False)

    keys = ["seed", "condition", "delay", "budget_cohorts", "cost_mode"]
    uniform = per_seed[per_seed.policy == "uniform"].set_index(keys)
    targeted = per_seed[per_seed.policy != "uniform"].set_index(keys)
    paired = targeted.join(uniform[["verified_harmful_episodes_by_stream_end", "unnecessary_audits"]],
                           how="left", rsuffix="_uniform")
    paired["episode_difference"] = (paired.verified_harmful_episodes_by_stream_end -
                                     paired.verified_harmful_episodes_by_stream_end_uniform)
    paired["unnecessary_audit_difference"] = paired.unnecessary_audits - paired.unnecessary_audits_uniform
    per_seed_effects = paired.reset_index().groupby(
        ["seed", "cost_mode", "policy", "delay"], as_index=False
    )[["episode_difference", "unnecessary_audit_difference"]].mean()
    rng = np.random.default_rng(17112026)
    effect_rows = []
    for (mode, policy, delay), group in per_seed_effects.groupby(
            ["cost_mode", "policy", "delay"], sort=True):
        for metric in ("episode_difference", "unnecessary_audit_difference"):
            values = group.sort_values("seed")[metric].to_numpy(float)
            boot = values[rng.integers(0, len(values), size=(2_000, len(values)))].mean(axis=1)
            effect_rows.append({
                "cost_mode": mode, "policy": policy, "delay": int(delay), "metric": metric,
                "mean_difference_vs_uniform": float(values.mean()),
                "lower": float(np.quantile(boot, .025)), "upper": float(np.quantile(boot, .975)),
                "paired_seeds": len(values), "bootstrap_replicates": 2_000,
                "bootstrap_unit": "seed after averaging matched conditions and budgets",
            })
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(out / "paired_overall_effects.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.05), constrained_layout=True)
    # Bootstrap seeds after averaging their matched conditions and budgets.
    forest = effects[(effects.cost_mode == "strict_cap") &
                     (effects.metric == "episode_difference")].copy()
    forest["policy_order"] = forest.policy.map({p: i for i, p in enumerate(COLORS)})
    forest = forest.sort_values(["policy_order", "delay"]).reset_index(drop=True)
    y = np.arange(len(forest))
    for i, row in forest.iterrows():
        axes[0].errorbar(row.mean_difference_vs_uniform, y[i],
                         xerr=[[row.mean_difference_vs_uniform - row.lower],
                               [row.upper - row.mean_difference_vs_uniform]],
                         fmt="o", ms=4.5, capsize=2.5, lw=1.2, color=COLORS[row.policy])
    axes[0].axvline(0, color="#334155", lw=1, ls="--")
    axes[0].set_yticks(y, [f"{LABELS[row.policy]} · d={row.delay}" for row in forest.itertuples()], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Verified harmful episodes difference vs uniform\n(95% paired seed-bootstrap interval)")
    axes[0].set_title("Strict-cap policy contrasts", weight="bold")
    axes[0].grid(axis="x", alpha=.22)
    axes[0].spines[["top", "right"]].set_visible(False)

    heat = summary[summary.cost_mode == "strict_cap"].pivot(index="policy", columns="delay",
                                                    values="verified_harmful_episodes_by_stream_end")
    heat = heat.reindex(list(COLORS))
    image = axes[1].imshow(heat.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0)
    axes[1].set_yticks(np.arange(len(heat)), [LABELS[p] for p in heat.index])
    axes[1].set_xticks(np.arange(len(heat.columns)), [f"{d}" for d in heat.columns])
    axes[1].set_xlabel("Label delay (batches)")
    axes[1].set_title("Verified harmful episodes\n(strict cap; mean per stream)", weight="bold")
    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            v = heat.iloc[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > heat.to_numpy().max() * .55 else "#16324f", fontsize=8)
    fig.colorbar(image, ax=axes[1], shrink=.82, label="Episode count")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"audit_policy_effects.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)

    strict = summary[summary.cost_mode == "strict_cap"]
    pivot = strict[strict.delay.isin([0, 5, 20, 100])].pivot(index="policy", columns="delay",
                                                              values="verified_harmful_episodes_by_stream_end")
    table = []
    for p in COLORS:
        row = pivot.loc[p]
        table.append(" & ".join([LABELS[p], *[f"{row[d]:.2f}" for d in (0, 5, 20, 100)]]) + r" \\")
    acquisition = per_seed.groupby(["policy", "cost_mode"], as_index=False).agg(
        selected=("audited_batches", "mean"), unspent=("unspent_budget_cohorts", "mean"),
        forced=("forced_quota_acquisitions", "mean"))
    acquisition.to_csv(out / "policy_acquisition_summary.csv", index=False)
    acquisition_idx = acquisition.set_index(["policy", "cost_mode"])
    quota_rows = "\n".join(
        f"{LABELS[p]} & {acquisition_idx.loc[(p, 'strict_cap'), 'selected']:.2f} & "
        f"{acquisition_idx.loc[(p, 'strict_cap'), 'unspent']:.2f} & "
        f"{acquisition_idx.loc[(p, 'exact_quota'), 'forced']:.2f}" + r" \\" for p in COLORS)
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,placeins}
\definecolor{navy}{HTML}{16324F}\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{When Should IDM-Viz Request Labels?}\\[3pt]
\large Delayed Audit Policies Under Explicit Acquisition Budgets}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 addendum}
\date{17 November 2026}
\begin{document}\maketitle
\begin{abstract}
We replay six predeclared audit policies on cached predictions from the 30-seed locked synthetic evaluation. The matrix crosses four label delays (0, 5, 20, 100 batches), three audit budgets (2, 4, 8 cohorts), and strict-cap versus exact-quota spending. It evaluates whether selected audits identify harmful batches and whether harmful episodes can be verified before the 24-batch horizon ends. The replay is outcome-informed for evaluation and is not fresh independent confirmation.
\end{abstract}
\section{Protocol and label timing}
Input-alarm and calibrated-concern thresholds are quantiles set only on five disjoint development seeds at a rate of $B/24$, where $B$ is the cohort budget. The conjunction requires both signals; the disjunction accepts either signal or a previously released error signal. The released-error policy starts with a fixed probe at $\lceil24/B\rceil$, then compares the historical reference accuracy with selected-label accuracy once labels are available. Uniform selection spaces acquisitions across the horizon. Every selected cohort contains 500 rows.

At decision clock $t$, a selected source cohort $s$ is usable only if $s+d<t$. Thus its first usable clock is $s+d+1$. The strict-cap mode acquires only on a positive policy signal and may leave budget unused. Exact-quota mode fills any remaining quota in the tail; those forced acquisitions are separately counted and must not be interpreted as signal-driven evidence. A harmful cohort is defined as evaluator accuracy strictly more than 0.05 below the frozen historical reference. Episodes are maximal contiguous runs of such cohorts.

The 30 evaluation seeds are paired across all policies. The run-level receipt includes within-condition paired intervals. This report additionally averages each seed over matched conditions and budgets before bootstrapping the 30 seed-level differences against uniform. Batches, budgets, and conditions are not treated as independent replicates. Conditions are not independent real-world domains.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{audit_policy_effects.pdf}
\caption{Left: strict-cap paired differences versus uniform in harmful episodes verified by stream end, averaged within seed across matched conditions and budgets; intervals are 95\% seed-bootstrap intervals. Right: strict-cap verification yield by policy and delay, averaged across budgets, conditions, and seeds. A delay of 100 exceeds the simulated stream horizon, so no selected label can be verified online within that horizon.}
\end{figure}
\begin{table}[ht]
\centering\small\begin{tabular}{@{}lrrr@{}}\toprule
Policy & Strict audits & Strict unused & Exact forced\\\midrule
''' + quota_rows + r'''\bottomrule\end{tabular}
\caption{Average acquisition use across seeds, conditions, delays, and budgets. Exact-quota counts show tail acquisitions not triggered by a policy signal.}
\end{table}
\FloatBarrier
\section{Interpretation}
The matrix separates two questions that the earlier fixed-quota comparison conflated: whether a policy chooses informative audits under a hard cost cap, and what happens when an operator insists on spending the full quota. Strict-cap contrasts are not matched by acquisition volume: uniform uses a mean 4.67 cohorts, while signal policies range from 0.71 (conjunction) to 3.87 (disjunction). Their lower verified-episode counts therefore mix policy selectivity with fewer audit opportunities and do not establish inferior ranking at equal spend. Exact-quota performance includes the forced tail acquisitions shown in Table 1, so it is a budget-completion scenario rather than a pure alarm policy. Longer delays reduce opportunities to verify episodes, while delay 100 makes online verification impossible over this finite horizon by design.

All metrics involving harmful labels use evaluator truth after decisions are simulated. The released-error arm only consumes selected labels after the release clock condition. This replay cannot show analyst utility, monetary savings, or downstream intervention benefit. Results depend on this synthetic stream, frozen reference, 0.05 harm margin, threshold quantiles, and horizon; no claim of superiority on natural drift is supported.
\section{Reproduction}
From the project root:
\begin{verbatim}
python scripts/run_audit_policy_matrix.py
python scripts/report_audit_policy_matrix.py
The report command compiles the editable TeX with Tectonic and records a hash for the PDF.
\end{verbatim}
The run lock records all source and output hashes, policy rules, seed banks, thresholds, and limitations. The complete per-seed data, sequential decisions, and paired bootstrap intervals are included alongside this report.
\end{document}
'''
    (out / "audit_policy_matrix.tex").write_text(tex, encoding="utf-8")
    tex_path = out / "audit_policy_matrix.tex"
    subprocess.run(["tectonic", "-X", "compile", str(tex_path)], cwd=ROOT, check=True)
    files = [out / n for n in ("policy_means.csv", "policy_summary_across_conditions.csv", "paired_overall_effects.csv", "policy_acquisition_summary.csv", "audit_policy_effects.png", "audit_policy_effects.pdf", "audit_policy_effects.svg", "audit_policy_matrix.tex", "audit_policy_matrix.pdf")]
    receipt = {"status": "report_sources_complete", "run_protocol_lock_sha256": sha(run / "protocol_lock.json"),
               "run_protocol": lock["protocol"], "outputs_sha256": {p.name: sha(p) for p in files}}
    (out / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report and figures to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "results/phase3/audit_policy_matrix_20261117_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/phase3/audit_policy_matrix_20261117_v7")
    args = parser.parse_args()
    make(args.run.resolve(), args.out.resolve())
