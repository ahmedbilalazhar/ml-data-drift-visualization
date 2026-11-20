"""Figures and TeX report for the NOAA reference/calibrator bootstrap."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--run",type=Path,default=ROOT/"results/phase3/noaa_reference_calibration_uncertainty_20261117_v1")
 p.add_argument("--output",type=Path,default=ROOT/"reports/phase3/noaa_reference_calibration_uncertainty_20261117_v2")
 a=p.parse_args();a.run=a.run.resolve();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(f"refusing to overwrite {a.output}")
 a.output.mkdir(parents=True);d=pd.read_csv(a.run/"uncertainty_summary.csv")
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
  "figure.facecolor":"#F4F7FA","axes.facecolor":"white","savefig.facecolor":"#F4F7FA","axes.edgecolor":"#D6DEE8",
  "axes.labelcolor":"#344054","text.color":"#172B4D","xtick.color":"#475467","ytick.color":"#475467","grid.color":"#E7ECF2"})
 x=np.arange(1,22); colors={"calibrator_only":"#137C8B","reference_only":"#6C63A8","joint":"#E76F51"}
 names={"calibrator_only":"Calibrator only","reference_only":"Reference only","joint":"Joint bootstrap"}
 fig,axs=plt.subplots(3,1,figsize=(10.5,8.1),sharex=True,constrained_layout=True)
 for comp in colors:
  q=d[d.component==comp].sort_values("batch")
  axs[0].plot(x,q.estimate_mean,color=colors[comp],lw=1.6,label=names[comp])
  axs[0].fill_between(x,q.estimate_lower.to_numpy(),q.estimate_upper.to_numpy(),color=colors[comp],alpha=.12)
  axs[1].plot(x,q.concern_mean,color=colors[comp],lw=1.6,label=names[comp])
  axs[1].fill_between(x,q.concern_lower.to_numpy(),q.concern_upper.to_numpy(),color=colors[comp],alpha=.12)
  axs[2].plot(x,q.flag_rate,color=colors[comp],lw=1.6,marker="o",ms=3,label=names[comp])
 joint=d[d.component=="joint"].sort_values("batch")
 axs[0].plot(x,joint.verified_accuracy,color="#172B4D",lw=1.2,ls="--",label="Verified batch accuracy")
 axs[0].set_ylabel("Estimated accuracy");axs[0].set_ylim(.5,1)
 axs[1].axhline(.95,color="#344054",ls=":",lw=1,label="Concern flag threshold (.95)")
 axs[1].set_ylabel("Expected-harm concern");axs[1].set_ylim(-.04,1.04)
 axs[2].set_ylabel("Bootstrap share flagged");axs[2].set_ylim(-.05,1.05);axs[2].set_xlabel("Chronological 500-row batch")
 for ax in axs:ax.grid(axis="y",alpha=.65)
 axs[0].legend(frameon=False,ncol=2,loc="lower left",fontsize=8);axs[1].legend(frameon=False,ncol=2,loc="upper right",fontsize=8)
 fig.suptitle("Reference and calibrator uncertainty on one NOAA holdout",x=.01,ha="left",fontsize=14,weight="bold")
 fig.savefig(a.output/"noaa_uncertainty.png",dpi=250,bbox_inches="tight");fig.savefig(a.output/"noaa_uncertainty.svg",bbox_inches="tight");plt.close(fig)
 joint=d[d.component=="joint"]
 width=float((joint.estimate_upper-joint.estimate_lower).mean())
 concern_width=float((joint.concern_upper-joint.concern_lower).mean())
 disagreement=float(joint.concern_flag_disagreement_rate.mean())
 refsd=float(joint.reference_accuracy_sd.mean())
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,tcolorbox,parskip}
\definecolor{ink}{HTML}{172B4D}\definecolor{teal}{HTML}{137C8B}\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\title{\vspace{-2em}\color{ink}\bfseries IDM-Viz reference and calibrator uncertainty\\[-.2em]\large Block-bootstrap sensitivity on NOAA GSOD}
\author{Phase 3 natural-stream addendum}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1em}
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2mm]\textbf{Scope.} One Offutt station, one chronological same-day classification holdout. These bootstrap intervals are conditional sensitivity summaries, not station-population confidence guarantees.
\end{tcolorbox}
\section*{Protocol}
We held the logistic classifier and its training fit fixed, then performed 200 noncircular 30-day moving-block resamples for each of three components: refit the isotonic correctness calibrator only; resample fixed-reference correctness only; and refit/resample both. Each draw yields the 21 batch accuracy estimates and plug-in Poisson-binomial expected-harm concern scores using the same frozen predictions. Monitoring labels are used only to compare estimates with realized accuracy after scoring. No threshold or method is selected from these results. This design isolates uncertainty due to the calibration and reference partitions; it omits model-training uncertainty and does not create independent temporal replications.
\section*{Results}
With the joint bootstrap, mean per-batch 95\% bootstrap interval width for estimated accuracy was \ESTWIDTH{}; mean interval width for expected-harm concern was \CONWIDTH{}. Across the 21 batches, the bootstrap concern decision disagreed with the original plug-in decision in \DISAGREE{} of batch-draws on average. The mean bootstrap standard deviation of reference accuracy was \REFSD{}. These values describe sensitivity to historical calibration/reference sampling under the chosen 30-day block scheme. They do not repair the original stream's weak empirical plug-in interval coverage and must not be generalized beyond this record.
\begin{center}\includegraphics[width=\linewidth]{noaa_uncertainty.png}\end{center}
\section*{Interpretation and limits}
The component comparison separates calibrator variation from reference-accuracy variation and shows where the concern score is unstable. Reference-only resampling leaves the point accuracy estimate unchanged by construction, but can move the expected-harm concern because its benchmark changes. Isotonic refits affect the estimated correctness probabilities. The classifier itself is not refit; station-year autocorrelation may extend beyond 30 days; finite bootstrap replicates add Monte Carlo error; and only one station and one split are available. Consequently these are conditional uncertainty diagnostics, not calibrated guarantees or evidence of natural-domain transfer.
\section*{Reproduction}
\texttt{python scripts/run\_noaa\_reference\_calibration\_uncertainty.py}\\
\texttt{python scripts/summarize\_noaa\_reference\_calibration\_uncertainty.py}\\
Compile this source with Tectonic. Full-precision draws, summaries, source hashes, and software versions are included with the report.
\end{document}
'''
 for old,new in {"\\ESTWIDTH":f"{width:.3f}","\\CONWIDTH":f"{concern_width:.3f}","\\DISAGREE":f"{disagreement:.3f}","\\REFSD":f"{refsd:.4f}"}.items():tex=tex.replace(old,new)
 texpath=a.output/"NOAA_UNCERTAINTY.tex";texpath.write_text(tex,encoding="utf-8")
 subprocess.run(["tectonic","-X","compile",str(texpath),"--outdir",str(a.output)],check=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 inputs=[a.run/"audit_receipt.json",a.run/"uncertainty_summary.csv",a.run/"bootstrap_draws.csv"]
 outputs=[a.output/n for n in ["noaa_uncertainty.png","noaa_uncertainty.svg","NOAA_UNCERTAINTY.tex","NOAA_UNCERTAINTY.pdf"]]
 (a.output/"REPORT_RECEIPT.json").write_text(json.dumps({"status":"complete","input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
  "output_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in outputs},"scope":"conditional block-bootstrap uncertainty; one chronological stream"},indent=2)+"\n",encoding="utf-8")
 print(f"Joint bootstrap: mean estimate interval width {width:.3f}; concern width {concern_width:.3f}; flag disagreement {disagreement:.3f}; reference accuracy SD {refsd:.4f}")
if __name__=="__main__":main()
