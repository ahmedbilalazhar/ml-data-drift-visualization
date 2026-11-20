"""Summarize NOAA finite-stream label-sampling draws and compile the TeX report."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={"targeted_10pct":"#C45A4A","stratified_random_audit":"#167C80","simple_random_budget_matched":"#315B96"}
LABELS={"targeted_10pct":"Targeted sample mean*","stratified_random_audit":"Targeted + random audit (HT)","simple_random_budget_matched":"Uniform random (HT)"}

def wilson(success,n,z=1.96):
    phat=success/n;den=1+z*z/n;mid=(phat+z*z/(2*n))/den;half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

def build(run:Path,out:Path):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    metrics=pd.read_csv(run/"design_replicate_metrics.csv");summary=pd.read_csv(run/"design_summary.csv")
    receipt=json.loads((run/"audit_receipt.json").read_text(encoding="utf-8"));reps=int(receipt["design_replicates"])
    truth=float(receipt["finite_stream_accuracy"])
    order=["targeted_10pct","stratified_random_audit","simple_random_budget_matched"]
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.8),constrained_layout=True,gridspec_kw={"width_ratios":[1.35,1]})
    values=[metrics.loc[metrics.policy==p,"selected_naive_accuracy"].to_numpy(float) for p in order]
    bp=axs[0].boxplot(values,vert=False,patch_artist=True,tick_labels=[LABELS[p] for p in order],showfliers=False,medianprops={"color":"#142B3B","linewidth":1.6})
    for patch,p in zip(bp["boxes"],order):patch.set_facecolor(COLORS[p]);patch.set_alpha(.55);patch.set_edgecolor(COLORS[p])
    axs[0].axvline(truth,color="#142B3B",ls="--",lw=1.6,label=f"Full-stream accuracy = {truth:.3f}")
    axs[0].set_xlim(.45,.83);axs[0].set_xlabel("Accuracy estimate across randomized label draws")
    axs[0].set_title("Repeated sampling designs",weight="bold");axs[0].grid(axis="x",alpha=.22);axs[0].spines[["top","right"]].set_visible(False);axs[0].legend(frameon=False,fontsize=8,loc="lower right")
    est=summary.set_index("policy").loc[["stratified_random_audit","simple_random_budget_matched"]]
    y=np.arange(2);cover=est.ht_ci_coverage.to_numpy(float);lohi=np.array([wilson(round(c*reps),reps) for c in cover])
    axs[1].axvline(.95,color="#142B3B",ls="--",lw=1.4,label="Nominal 95%")
    axs[1].errorbar(cover,y,xerr=[cover-lohi[:,0],lohi[:,1]-cover],fmt="o",capsize=4,ms=7,lw=1.6,color="#167C80")
    axs[1].set_xlim(.88,1.005);axs[1].set_ylim(-.55,1.55);axs[1].set_yticks(y,["Targeted + audit","Uniform random"]);axs[1].invert_yaxis()
    axs[1].set_xlabel("Empirical coverage of normal HT intervals")
    axs[1].set_title("Coverage conditional on this stream",weight="bold");axs[1].grid(axis="x",alpha=.22);axs[1].spines[["top","right"]].set_visible(False);axs[1].legend(frameon=False,fontsize=8,loc="lower right")
    fig.suptitle("A random audit restores population estimability",fontsize=14,weight="bold")
    fig.savefig(out/"noaa_selective_design.png",dpi=230);fig.savefig(out/"noaa_selective_design.svg");fig.savefig(out/"noaa_selective_design.pdf");plt.close(fig)
    s=summary.set_index("policy")
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,tcolorbox,parskip}
\setlength{\emergencystretch}{3em}
\definecolor{ink}{HTML}{172B4D}\definecolor{teal}{HTML}{167C80}\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\title{\vspace{-2em}\color{ink}\bfseries IDM-Viz label audits on NOAA observations\\[-.2em]\large A retrospective known-probability sampling replay}
\author{Phase 3 supplementary experiment}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1em}
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2mm]\textbf{Scope boundary.} This replays randomized label selection on one previously observed NOAA station chronology. It is not prospective label operations, independent domain replication, or a weather forecast.
\end{tcolorbox}
\section*{Design}
The existing frozen logistic model was refit from the same historical partitions and its batch accuracies were required to match the locked NOAA evaluator exactly. The monitoring sequence contributes 21 complete batches of 500 rows (10,500 total); the final 426 rows are excluded, matching the original evaluation. For each batch, policies select labels using confidence only, before evaluator correctness is joined. We repeat independent sampling 1,000 times on this same finite stream. These replicates quantify randomized-design behavior conditional on this single fixed stream; they do not create 1,000 independent datasets.

The targeted policy selects the 50 least-confident rows in each batch with certainty and zero inclusion probability elsewhere. It therefore cannot estimate population accuracy. The combined policy samples those 50 rows plus each remaining row with Bernoulli probability 0.10, so inclusion probabilities are known (1 or 0.10). Budget-matched uniform sampling uses Bernoulli probability 0.19. The combined policy uses the untruncated Horvitz--Thompson (HT) estimator and its Bernoulli design variance. Normal 95\% intervals are checked against the known finite-stream accuracy after selection.
\section*{Results}
The exact frozen-model finite-stream accuracy is \TRUTH. Confidence-only targeting labels \TARGETN{} of 10,500 rows per draw on average and its selected mean is \TARGET; its population estimand is not identified. The targeted-plus-random-audit design labels \COMBINEDN{} rows on average; its HT mean across draws is \COMBINEDHT with design bias \COMBINEDBIAS and normal-interval coverage \COMBINEDCOV. Budget-matched uniform sampling labels \UNIFORMN{} rows on average; its HT mean is \UNIFORMHT with design bias \UNIFORMBIAS and coverage \UNIFORMCOV.
\begin{center}\includegraphics[width=\linewidth]{noaa_selective_design.png}\end{center}
\begin{table}[ht]\centering\small
\begin{tabular}{lrrrr}\toprule
Policy & Mean labels & Mean estimate & HT bias & 95\% interval coverage\\\midrule
Targeted only (naive descriptor) & \TARGETN{} & \TARGET & Not estimable & Not defined\\
Targeted + random audit (HT) & \COMBINEDN{} & \COMBINEDHT & \COMBINEDBIAS & \COMBINEDCOV\\
Uniform random (HT) & \UNIFORMN{} & \UNIFORMHT & \UNIFORMBIAS & \UNIFORMCOV\\\bottomrule
\end{tabular}
\caption{Averages across independent sampling draws on the one fixed NOAA stream. Coverage is conditional on that stream.}
\end{table}
\section*{Interpretation and limitations}
For this stream, the least-confident decile is much less accurate than the full population, so targeted-only labels substantially understate population accuracy. Adding a known-probability random audit makes the full-stream target design-estimable and gives near-nominal empirical coverage in this replay. The combined design's certainty stratum reduces effective sample size relative to uniform sampling despite similar label counts. This experiment does not establish prospective operations, repeated-domain uncertainty, nonresponse behavior, acquisition cost, selected-label recalibration, or validity under changing inclusion probabilities. The data are NOAA same-day completed summaries, and the task is retrospective classification.
\section*{Reproduction}
Run from the project root:
\begin{verbatim}
powershell -NoProfile -ExecutionPolicy Bypass \\
  -File .\scripts\reproduce_noaa_selective_label_replay.ps1 -Replicates 1000
\end{verbatim}
The audit receipt hashes sources, the locked NOAA evaluator, and every output. Row-level logs retain selected IDs, their inclusion probabilities, confidence-only acquisition score, prediction and selected correctness.
\end{document}
'''
    macros={"\\TRUTH":f"{truth:.4f}","\\TARGETN":f"{s.loc['targeted_10pct','mean_selected_rows']:.0f}","\\TARGET":f"{s.loc['targeted_10pct','mean_selected_naive_accuracy']:.4f}",
      "\\COMBINEDN":f"{s.loc['stratified_random_audit','mean_selected_rows']:.1f}","\\COMBINEDHT":f"{s.loc['stratified_random_audit','mean_selected_naive_accuracy']:.4f}","\\COMBINEDBIAS":f"{s.loc['stratified_random_audit','ht_bias']:+.4f}","\\COMBINEDCOV":f"{100*s.loc['stratified_random_audit','ht_ci_coverage']:.1f}"+r"\%",
      "\\UNIFORMN":f"{s.loc['simple_random_budget_matched','mean_selected_rows']:.1f}","\\UNIFORMHT":f"{s.loc['simple_random_budget_matched','mean_selected_naive_accuracy']:.4f}","\\UNIFORMBIAS":f"{s.loc['simple_random_budget_matched','ht_bias']:+.4f}","\\UNIFORMCOV":f"{100*s.loc['simple_random_budget_matched','ht_ci_coverage']:.1f}"+r"\%"}
    for token,value in macros.items():tex=tex.replace(token,value)
    texpath=out/"NOAA_SELECTIVE_LABEL_REPLAY.tex";texpath.write_text(tex,encoding="utf-8")
    subprocess.run(["tectonic","-X","compile",str(texpath),"--outdir",str(out),"--keep-logs"],check=True)
    print(f"Wrote editable report, figure and PDF to {out}; design draws={reps}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();build(a.run.resolve(),a.output.resolve())
