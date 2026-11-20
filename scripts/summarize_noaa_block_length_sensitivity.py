"""Summarize the prespecified 7/30/90-day NOAA moving-block sensitivity."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUNS={7:ROOT/"results/phase3/noaa_reference_calibration_uncertainty_block7_20261117_v1",
      30:ROOT/"results/phase3/noaa_reference_calibration_uncertainty_20261117_v2",
      90:ROOT/"results/phase3/noaa_reference_calibration_uncertainty_block90_20261117_v1"}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"reports/phase3/noaa_block_length_sensitivity_20261117_v3")
 a=p.parse_args();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(f"refusing to overwrite {a.output}")
 rows=[];inputs=[]
 for length,run in RUNS.items():
  receipt=json.loads((run/"audit_receipt.json").read_text(encoding="utf-8"));s=pd.read_csv(run/"uncertainty_summary.csv")
  if receipt["status"]!="complete" or receipt["block_length_rows"]!=length:raise ValueError("block-length run receipt mismatch")
  q=s[s.component=="joint"]
  rows.append({"block_length_days":length,"replicates":receipt["bootstrap_replicates_per_component"],
   "mean_accuracy_interval_width":float((q.estimate_upper-q.estimate_lower).mean()),
   "mean_concern_interval_width":float((q.concern_upper-q.concern_lower).mean()),
   "mean_concern_flag_disagreement":float(q.concern_flag_disagreement_rate.mean()),
   "mean_reference_accuracy_sd":float(q.reference_accuracy_sd.mean()),
   "mean_bootstrap_estimate":float(q.estimate_mean.mean()),"mean_verified_accuracy":float(q.verified_accuracy.mean())})
  inputs += [run/"audit_receipt.json",run/"uncertainty_summary.csv"]
 summary=pd.DataFrame(rows);a.output.mkdir(parents=True);summary.to_csv(a.output/"block_length_summary.csv",index=False)
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
  "figure.facecolor":"#F4F7FA","axes.facecolor":"white","savefig.facecolor":"#F4F7FA","axes.edgecolor":"#D6DEE8",
  "axes.labelcolor":"#344054","text.color":"#172B4D","xtick.color":"#475467","ytick.color":"#475467","grid.color":"#E7ECF2"})
 fig,axs=plt.subplots(1,3,figsize=(10.8,3.8),constrained_layout=True)
 metrics=[("mean_accuracy_interval_width","Accuracy estimate CI width","#176B87"),
          ("mean_concern_interval_width","Expected-harm concern CI width","#D06A45"),
          ("mean_concern_flag_disagreement","Concern decision disagreement","#6A6C9B")]
 for ax,(col,title,color) in zip(axs,metrics):
  ax.plot(summary.block_length_days,summary[col],marker="o",lw=2,color=color)
  ax.set_title(title,weight="bold",fontsize=10);ax.set_xlabel("Moving-block length (days)");ax.set_xticks([7,30,90]);ax.grid(axis="y",alpha=.4)
 axs[0].set_ylabel("Mean over 21 serial batches")
 fig.suptitle("NOAA uncertainty expands with longer temporal resampling blocks",fontsize=13,weight="bold")
 fig.savefig(a.output/"block_length_sensitivity.png",dpi=250,bbox_inches="tight");fig.savefig(a.output/"block_length_sensitivity.svg",bbox_inches="tight");plt.close(fig)
 r7=summary[summary.block_length_days==7].iloc[0];r30=summary[summary.block_length_days==30].iloc[0];r90=summary[summary.block_length_days==90].iloc[0]
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,tcolorbox,parskip}
\definecolor{ink}{HTML}{172B4D}\definecolor{teal}{HTML}{137C8B}\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\setlength{\emergencystretch}{3em}
\title{\vspace{-2em}\color{ink}\bfseries NOAA block-length sensitivity\\[-.2em]\large Reference and correctness-calibrator uncertainty}
\author{Phase 3 natural-stream addendum}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1em}
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2mm]\textbf{Scope.} One station, one chronological same-day classification holdout. The classifier is fixed. These resampling diagnostics are conditional sensitivity summaries, not population guarantees.
\end{tcolorbox}
\section*{Design}
We repeated the same 200-replicate calibrator-only, reference-only and joint moving-block bootstrap with block lengths of 7, 30 and 90 consecutive daily rows. The 30-day run is the primary diagnostic; 7 and 90 days test dependence-length sensitivity. Each run resamples calibration confidence/correctness pairs and reference correctness, then exactly recomputes the plug-in Poisson-binomial concern score for the same 21 monitor batches. Model-fit uncertainty is not resampled. Monitor outcomes are evaluator-only.
\section*{Result}
Under joint resampling, the average per-batch 95\% accuracy-estimate interval width increased from \W7{} at 7 days, to \W30{} at 30 days, and \W90{} at 90 days. Expected-harm concern interval widths were \C7{}, \C30{} and \C90{}, respectively. The average fraction of bootstrap concern flags disagreeing with the original plug-in decision was \D7{}, \D30{} and \D90{}. Longer blocks produce wider uncertainty intervals and somewhat more decision disagreement in this selected diagnostic, so uncertainty depends on the temporal resampling scale.
\begin{center}\includegraphics[width=\linewidth]{block_length_sensitivity.png}\end{center}
\section*{Interpretation}
This sensitivity checks a single station's calibration/reference partitions under three block lengths; it does not identify the true dependence horizon. The 90-day scheme has relatively few effective blocks in these historical segments; 7-day blocks may understate longer dependence. Report the 30-day analysis as a transparent reference choice, and include this spread as a limitation. The result does not account for fitting uncertainty in the logistic classifier, alternative stations, other temporal splits, or distribution-free coverage.
\section*{Reproduction}
Reproduce the three block-bootstrap runs and report with:
\begin{verbatim}
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/reproduce_noaa_block_length_sensitivity.ps1
\end{verbatim}
Full per-draw CSVs and checksums are included.
\end{document}
'''
 values={"\\W7":f"{r7.mean_accuracy_interval_width:.3f}","\\W30":f"{r30.mean_accuracy_interval_width:.3f}","\\W90":f"{r90.mean_accuracy_interval_width:.3f}",
 "\\C7":f"{r7.mean_concern_interval_width:.3f}","\\C30":f"{r30.mean_concern_interval_width:.3f}","\\C90":f"{r90.mean_concern_interval_width:.3f}",
 "\\D7":f"{r7.mean_concern_flag_disagreement:.3f}","\\D30":f"{r30.mean_concern_flag_disagreement:.3f}","\\D90":f"{r90.mean_concern_flag_disagreement:.3f}"}
 for old,new in values.items():tex=tex.replace(old+"{}",new).replace(old,new)
 texpath=a.output/"NOAA_BLOCK_LENGTH_SENSITIVITY.tex";texpath.write_text(tex,encoding="utf-8")
 import subprocess
 subprocess.run(["tectonic","-X","compile",str(texpath),"--outdir",str(a.output)],check=True)
 sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();inputs += [RUNS[30]/"bootstrap_draws.csv",RUNS[7]/"bootstrap_draws.csv",RUNS[90]/"bootstrap_draws.csv"]
 outputs=[a.output/n for n in ["block_length_summary.csv","block_length_sensitivity.png","block_length_sensitivity.svg","NOAA_BLOCK_LENGTH_SENSITIVITY.tex","NOAA_BLOCK_LENGTH_SENSITIVITY.pdf"]]
 (a.output/"REPORT_RECEIPT.json").write_text(json.dumps({"status":"complete","input_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in inputs},
  "output_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in outputs},"block_lengths_days":[7,30,90],"replicates_per_component":200},indent=2)+"\n",encoding="utf-8")
 print(summary.to_string(index=False))
if __name__=="__main__":main()
