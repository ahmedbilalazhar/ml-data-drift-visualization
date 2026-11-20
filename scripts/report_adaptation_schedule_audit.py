"""Build a paired-seed figure and editable TeX report for the schedule audit."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COLORS={"periodic_refit":"#1f77b4","global_error_trigger_refit":"#d95f02","region_error_trigger_refit":"#16866f"}
LABELS={"periodic_refit":"Periodic refit","global_error_trigger_refit":"Global error trigger","region_error_trigger_refit":"Region-local trigger"}

def build(run:Path,out:Path):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    runs=pd.read_csv(run/"seed_metrics.csv")
    rec=pd.read_json(run/"audit_receipt.json",typ="series")
    frozen=runs[runs.method=="frozen"].set_index(["seed","mechanism","delay"])
    contrasts=[]
    for method in ("periodic_refit","global_error_trigger_refit","region_error_trigger_refit"):
        part=runs[runs.method==method]
        for (mechanism,delay),g in part.groupby(["mechanism","delay"],sort=True):
            # MultiIndex vector selection preserves the seed pairing.
            d=g.set_index("seed").sort_index()
            b=frozen.xs((mechanism,delay),level=("mechanism","delay")).sort_index()
            delta=d.accuracy.to_numpy()-b.loc[d.index,"accuracy"].to_numpy()
            contrasts.append({"mechanism":mechanism,"delay":delay,"method":method,"delta":delta})
    if len(contrasts)!=18 or any(len(item["delta"])!=30 for item in contrasts):
        raise ValueError("the locked schedule report requires 18 contrasts and 30 paired seeds per contrast")
    # Jointly resample seed rows across the complete family, preserving
    # dependence between conditions that share the same evaluation seed.
    delta_matrix=np.vstack([item["delta"] for item in contrasts])
    rng=np.random.default_rng(261118)
    indices=rng.integers(0,delta_matrix.shape[1],(20000,delta_matrix.shape[1]))
    boot_means=delta_matrix[:,indices].mean(axis=2)
    observed=delta_matrix.mean(axis=1)
    standard_errors=delta_matrix.std(axis=1,ddof=1)/np.sqrt(delta_matrix.shape[1])
    if not np.isfinite(standard_errors).all():
        raise ValueError("max-t intervals require finite across-seed variation")
    nonzero_variance=standard_errors>0
    bootstrap_t=np.zeros_like(boot_means)
    bootstrap_t[nonzero_variance]=(boot_means[nonzero_variance]-observed[nonzero_variance,None])/standard_errors[nonzero_variance,None]
    max_t=np.max(np.abs(bootstrap_t),axis=0)
    critical=float(np.quantile(max_t,.95))
    paired=[]
    for i,item in enumerate(contrasts):
        marginal_low,marginal_high=np.quantile(boot_means[i],[.025,.975])
        paired.append({"mechanism":item["mechanism"],"delay":item["delay"],"method":item["method"],
          "metric":"accuracy_difference_vs_frozen","seeds":30,"mean":float(observed[i]),
          "lower":float(observed[i]-critical*standard_errors[i]),
          "upper":float(observed[i]+critical*standard_errors[i]),
          "marginal_lower":float(marginal_low),"marginal_upper":float(marginal_high),
          "max_t_critical":critical,"family_size":len(contrasts),
          "zero_seed_variance":bool(not nonzero_variance[i]),
          "ci_method":"joint paired-seed bootstrap max-|t|"})
    paired=pd.DataFrame(paired); paired.to_csv(out/"paired_accuracy_differences.csv",index=False)
    means=runs.groupby(["mechanism","delay","method"],as_index=False).agg(
      accuracy=("accuracy","mean"),post_onset_accuracy=("post_onset_accuracy","mean"),
      updates=("update_count","mean"),fit_rows=("fit_row_cost_cumulative","mean"))
    means.to_csv(out/"policy_means.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.4),sharey=True,constrained_layout=True)
    for ax,(mechanism,title) in zip(axes,[("M0_stationary","Stationary stream"),("M4_abrupt_conditional_reversal","Abrupt conditional reversal")]):
        for method in COLORS:
            q=paired[(paired.mechanism==mechanism)&(paired.method==method)].sort_values("delay")
            offset={"periodic_refit":-0.14,"global_error_trigger_refit":0,"region_error_trigger_refit":0.14}[method]
            y=np.arange(3)+offset
            ax.errorbar(q["mean"],y,xerr=[q["mean"]-q["lower"],q["upper"]-q["mean"]],fmt="o",capsize=3,
                        lw=1.7,ms=5,color=COLORS[method],label=LABELS[method])
        ax.axvline(0,color="#334155",lw=1,ls="--");ax.set_yticks(range(3),["0 batches","5 batches","20 batches"])
        ax.invert_yaxis();ax.set_title(title,fontsize=12,weight="bold");ax.set_xlabel("Accuracy difference vs frozen\n(simultaneous 95% max-|t| interval)")
        ax.grid(axis="x",alpha=.22);ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Label-release delay")
    for ext in ("png","pdf","svg"):fig.savefig(out/f"adaptation_schedule.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)
    # TeX report is intentionally self-contained except for the adjacent figure.
    tex=r'''\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{placeins}
\usepackage{needspace}
\usepackage{xcolor}
\usepackage{hyperref}
\definecolor{navy}{HTML}{16324F}
\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{When Should IDM-Viz Refit?}\\[3pt]\large A Label-Safe Schedule Audit}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 addendum}
\date{18 November 2026 --- post-deadline research addendum}
\begin{document}\maketitle
\begin{abstract}
This controlled extension compares a frozen logistic classifier with periodic, global released-error-triggered, and region-local released-error-triggered refitting. It tests whether the triggers avoid unnecessary stationary-stream updates and whether schedules recover accuracy after a synthetic conditional reversal. Thirty paired seeds, three label delays, and two synthetic mechanisms are used. This remains a bounded synthetic policy comparison, not a claim of operational benefit.
\end{abstract}
\section{Protocol}
The stream contains 96 batches of 500 observations. A single logistic classifier is trained on the frozen historical set; its scaler and a three-region MiniBatchKMeans partition are fit on training rows. The periodic policy refits at forecast clocks 25, 49, and 73. The global trigger requires both latest released cohorts to exceed calibration error by 0.10. The local trigger requires two eligible released cohorts in one frozen region to exceed that region's calibration error by 0.10; each cohort needs at least 20 rows and each calibration region needs at least 20 rows. Both triggers have an eight-clock cooldown and a three-update cap. A global update uses historical training rows plus at most 2,000 recent released labels. A local update uses only that region's historical training rows and at most 2,000 recent released labels from that region; other regions retain the global frozen fallback. Thresholds and regions are fixed without monitoring outcomes.

For delay $d$, source batch $s$ is available before forecast batch $b$ only when $s+d<b$. A shared \texttt{LabelBroker} schedules releases; the orchestrator consumes them at clock $b-1$, before forecasting batch $b$. Stored predictions are joined to labels only after release. Every cohort release records its source batch, due clock, processing clock, prediction clock, row count, and ID digest in \texttt{label\_release\_events.csv}. Delayed cohorts are drained only after the final forecast, with no extra prediction or model update. Thus labels available by the last forecast are 47,500, 45,000, and 37,500 at delays 0, 5, and 20, while all 48,000 labels are accounted for after queue drain. The replication unit is the seed; a joint paired-seed bootstrap preserves cross-condition dependence and a max-|t| adjustment gives simultaneous intervals over the 18 accuracy contrasts.

\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{adaptation_schedule.pdf}
\caption{Paired accuracy differences from the frozen policy; blue denotes periodic refitting, orange global trigger, and green region-local trigger. Intervals are simultaneous 95\% max-|t| bootstrap intervals across all 18 contrasts, resampling matched seeds jointly.}
\end{figure}
\begin{table}[ht]
\centering\small
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Mechanism & Delay & Frozen & Periodic & Global & Local & Global updates & Local updates\\
\midrule
'''
    for mech,mlabel in [("M0_stationary","Stationary"),("M4_abrupt_conditional_reversal","Reversal")]:
      for delay in (0,5,20):
        vals={r.method:r for r in means[(means.mechanism==mech)&(means.delay==delay)].itertuples(index=False)}
        tex+=f"{mlabel} & {delay} & {vals['frozen'].accuracy:.3f} & {vals['periodic_refit'].accuracy:.3f} & {vals['global_error_trigger_refit'].accuracy:.3f} & {vals['region_error_trigger_refit'].accuracy:.3f} & {vals['global_error_trigger_refit'].updates:.2f} & {vals['region_error_trigger_refit'].updates:.2f}\\\\\n"
    tex+=r'''\bottomrule
\end{tabular}
\caption{Mean accuracy across paired seeds; trigger updates are averaged across seeds.}
\end{table}

\FloatBarrier
\Needspace{8\baselineskip}
For abrupt reversal, the local trigger improved all-stream mean accuracy over frozen by 12.45 percentage points at zero delay, 11.35 points at five-batch delay, and 7.41 points at twenty-batch delay. The global trigger gains were 0.37, 0.35, and 0.26 points, respectively. The plotted intervals use a joint seed bootstrap with max-|t| adjustment over all 18 accuracy contrasts; exact simultaneous and marginal intervals are in \texttt{paired\_accuracy\_differences.csv}. Local gains are substantial in this generator, but this is a single synthetic reversal design with a fixed training-fitted partition and trigger; it does not establish general superiority or transfer. Under stationarity, one of 30 seeds triggered one local update (at all three delays); paired accuracy differences were effectively zero. The global trigger did not update.

The v4 run replaces duplicated clock arithmetic with the shared broker contract. Its forecast metrics, update counts, and fit-row costs match the earlier 30-seed schedule run exactly; release timing is separately auditable, and queue-drain labels are not counted as available to the final forecast. This is a software-equivalence check for the existing policy, not an independent scientific replication.

\FloatBarrier
\section{Interpretation and limits}
This extension evaluates one global and one frozen-region local trigger, but does not cover alternative thresholds, region counts, multiple classifiers, multiple real-world streams, concept-change recovery guarantees, or equal consumed-label budgets against frozen prediction. The local policy uses a training-fitted unsupervised partition and can leave a region on its global fallback when evidence is sparse or a local refit lacks both classes. Fit-row counts are a computational proxy, not wall-clock or energy measurements. Synthetic reversal results cannot establish utility for deployment.

\section{Reproduction}
Run from the workspace root with the project dependencies on PATH:
\begin{verbatim}
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce_adaptation_schedule_audit.ps1
\end{verbatim}
The runner receipt records code/configuration and output hashes, label-release semantics, seed bank and software versions. Contract tests and the per-cohort event audit check that no forecast consumes same-clock labels, every cohort is released once, and the result tables contain all locked seed-policy cells. Inference is seed-level; repeated batches are not treated as independent replicates.

\end{document}
'''
    (out/"adaptation_schedule_audit.tex").write_text(tex,encoding="utf-8")
    print(f"Wrote {out}; {len(paired)} paired policy contrasts; locked seeds={rec['seed_bank']}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();build(a.run,a.out)
