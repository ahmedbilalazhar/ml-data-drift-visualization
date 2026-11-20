"""Create a TeX/PDF report for training-only NOAA model-refit uncertainty."""
from __future__ import annotations
import argparse,subprocess,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(run:Path,out:Path):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    d=pd.read_csv(run/"training_uncertainty_summary.csv");r=json.loads((run/"audit_receipt.json").read_text(encoding="utf-8"))
    fig,axs=plt.subplots(2,1,figsize=(10.8,7.1),sharex=True,constrained_layout=True)
    b=d.batch.to_numpy(int)
    axs[0].fill_between(b,d.accuracy_lower,d.accuracy_upper,color="#167C80",alpha=.18,label="95% training-block bootstrap interval")
    axs[0].plot(b,d.accuracy_mean,color="#167C80",lw=1.8,label="Bootstrap mean")
    axs[0].plot(b,d.baseline_estimate,color="#315B96",lw=1.2,ls=":",label="Original frozen estimate")
    axs[0].plot(b,d.verified_accuracy,color="#C45A4A",marker="o",ms=3.8,lw=1.2,label="Original frozen model observed accuracy")
    axs[0].set_ylabel("Batch accuracy");axs[0].set_title("Refitting adds training-sample spread, not shift coverage",weight="bold")
    axs[0].grid(axis="y",alpha=.22);axs[0].spines[["top","right"]].set_visible(False);axs[0].legend(frameon=False,ncol=2,fontsize=8,loc="lower left")
    axs[1].fill_between(b,d.concern_lower,d.concern_upper,color="#D07A3E",alpha=.22,label="95% training-block interval")
    axs[1].plot(b,d.concern_mean,color="#D07A3E",lw=1.7,label="Bootstrap mean concern")
    axs[1].plot(b,d.baseline_concern,color="#315B96",lw=1.1,ls=":",label="Original frozen concern")
    axs[1].axhline(.95,color="#334155",ls="--",lw=1,label="Concern flag threshold (.95)")
    axs[1].set_ylabel("Plug-in concern");axs[1].set_xlabel("Chronological monitoring batch (21 serial batches)")
    axs[1].set_ylim(-.03,1.02);axs[1].set_xticks(np.arange(1,22,2));axs[1].grid(axis="y",alpha=.22)
    axs[1].spines[["top","right"]].set_visible(False);axs[1].legend(frameon=False,ncol=2,fontsize=8,loc="upper right")
    for ext in ("png","svg","pdf"):fig.savefig(out/f"training_uncertainty.{ext}",dpi=230 if ext=="png" else None)
    plt.close(fig)
    est_width=float((d.accuracy_upper-d.accuracy_lower).mean());conc_width=float((d.concern_upper-d.concern_lower).mean())
    coverage=float(d.accuracy_interval_covers_original_estimator.mean());ncover=int(d.accuracy_interval_covers_original_estimator.sum())
    mae=float(d.baseline_estimator_mae.mean());replicate_mae=float(d.replicate_estimator_mae.mean());ref_sd=float(d.reference_accuracy_sd.mean())
    flag_dis=float(d.baseline_bootstrap_flag_disagreement_rate.mean())
    tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,tcolorbox,parskip}
\setlength{\emergencystretch}{3em}
\definecolor{ink}{HTML}{172B4D}\definecolor{teal}{HTML}{167C80}\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\title{\vspace{-2em}\color{ink}\bfseries Training-data uncertainty in IDM-Viz\\[.2em]\large NOAA moving-block refit sensitivity}
\author{Phase 3 supplementary experiment}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1em}
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2mm]\textbf{Scope.} This estimates sensitivity to historical training rows by refitting the frozen pipeline on 200 moving-block bootstrap draws. It does not provide a generalization guarantee or make the NOAA same-day task a forecast.
\end{tcolorbox}
\section*{Protocol}
The chronological training partition contains \TRAIN{} processed rows. Each bootstrap replicate resamples contiguous blocks of \BLOCK{} processed rows to the original training length. For every draw we refit the training scaler, logistic classifier, PCA anchor and clusters, then refit the isotonic correctness map on the fixed calibration partition and recompute reference correctness on the fixed reference partition. Calibration, reference and monitoring rows are not resampled. The same 21 complete 500-row monitoring batches are used across all refits; the trailing 426 rows remain excluded as in the original run. Monitoring outcomes are consulted only after the model scores and concern values for a draw are computed.

This is a training-only component sensitivity. It does not combine training, calibrator and reference uncertainty into one joint procedure; the separate 7/30/90-row block analysis for calibration/reference remains a distinct analysis.
\section*{Findings}
The mean width of the 95\% bootstrap interval for batch accuracy estimates is \ESTWIDTH. The mean width for plug-in concern is \CONWIDTH. The interval contains the original frozen model accuracy estimate in \COVCOUNT{} of 21 batches (\COVERAGE). This is a descriptive bootstrap consistency check around one fitted-model estimate, not coverage for true or future accuracy. The original model mean absolute accuracy-estimation error against verified batch accuracy is \MAE; across training refits, the mean absolute difference between each estimate and its matching refit's observed accuracy is \REFITMAE. The refitted reference accuracy has mean bootstrap standard deviation \REFSD. Training refits changed no batch concern flags in this run (mean disagreement from the original policy: \FLAGDIS).

\begin{center}\includegraphics[width=\linewidth]{training_uncertainty.png}\end{center}
\section*{Interpretation and limits}
Retraining variability adds uncertainty to estimated correctness and plug-in concern, while the original frozen-model point estimate is inside most training-bootstrap estimate bands. Original-model estimation error and refit-specific estimation error are reported separately; bootstrap spread is not a calibrated population interval. The result is conditional on one station, a particular historical training window, a 30-row moving-block scheme, one logistic pipeline and a frozen evaluation chronology. The bootstrap cannot represent model misspecification, new-station transfer, unobserved future regimes, or arbitrary conditional shift. Block length is a design sensitivity, not an identified dependence horizon.
\section*{Reproduction}
From the project root, use fresh output directories:
\begin{verbatim}
python scripts/run_noaa_training_uncertainty.py --replicates 200 --block-length 30 \\
  --output results/phase3/noaa_training_uncertainty_cleancheck
python scripts/summarize_noaa_training_uncertainty.py \\
  --run results/phase3/noaa_training_uncertainty_cleancheck \\
  --output reports/phase3/noaa_training_uncertainty_cleancheck
\end{verbatim}
The run receipt records exact source/output hashes, bootstrap attempts, software versions and the fixed temporal partitions.
\end{document}
'''
    values={"\\TRAIN":str(r["training_rows"]),"\\BLOCK":str(r["training_block_length_rows"]),"\\ESTWIDTH":f"{est_width:.4f}","\\CONWIDTH":f"{conc_width:.4f}","\\COVCOUNT":str(ncover),"\\COVERAGE":f"{100*coverage:.1f}"+r"\%","\\MAE":f"{mae:.4f}","\\REFITMAE":f"{replicate_mae:.4f}","\\REFSD":f"{ref_sd:.4f}","\\FLAGDIS":f"{flag_dis:.3f}"}
    for k,v in values.items():tex=tex.replace(k,v)
    src=out/"NOAA_TRAINING_UNCERTAINTY.tex";src.write_text(tex,encoding="utf-8")
    subprocess.run(["tectonic","-X","compile",str(src),"--outdir",str(out),"--keep-logs"],check=True)
    print(f"Wrote TeX/PDF and figures to {out}; interval coverage={ncover}/21")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();build(a.run.resolve(),a.output.resolve())
