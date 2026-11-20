"""Generate an editable TeX comparison of matched-budget ELEC2 calibrators."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main(run:Path,platt:Path,isotonic:Path,results:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);s=pd.read_csv(results/'summary.csv')
 methods=['frozen_isotonic','cumulative_isotonic','cumulative_platt'];labels=['Frozen isotonic','Cumulative isotonic','Cumulative Platt'];colors=['#536578','#16817a','#cf8756'];delays=[0,5,20];x=range(3);w=.23
 fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 specs=[('equal_batch_mae','Equal-batch MAE',0,.08,False),('mean_correctness_brier','Correctness Brier',0,.26,False),('equal_batch_ece10','ECE (10 bins)',0,.13,False),('interval_coverage','Plug-in interval coverage',0,1,True)]
 for ax,(field,title,ymin,ymax,fraction) in zip(axes.flat,specs):
  for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
   z=s[s.method.eq(method)].set_index('delay').loc[delays];vals=z[field].to_numpy(dtype=float)
   if fraction:vals=vals/55
   bars=ax.bar([k+(j-1)*w for k in x],vals,w,color=color,label=label)
   if fraction:
    for bar,count in zip(bars,z[field].astype(int)):
     ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.018,f'{count}/55',ha='center',va='bottom',fontsize=6.5,rotation=0)
  ax.set_title(title,fontsize=10);ax.set_xticks(list(x),[str(d) for d in delays]);ax.set_xlabel('Label delay (batches)');ax.set_ylim(ymin,ymax);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 axes[0,0].set_ylabel('Mean over 55 batches');axes[0,1].set_ylabel('Mean over monitoring instances');axes[1,0].set_ylabel('Mean over 55 batches');axes[1,1].set_ylabel('Covered cohorts / 55')
 axes[0,0].legend(frameon=False,fontsize=7,loc='upper left')
 for ext in ('pdf','png','svg'):fig.savefig(out/f'calibration_family.{ext}',dpi=240)
 plt.close(fig)
 rows=[]
 for _,r in s.sort_values(['delay','method']).iterrows():
  label=dict(zip(methods,labels))[r.method]
  rows.append(f"{int(r.delay)} & {label} & {r.equal_batch_mae:.3f} & {r.equal_batch_bias:+.3f} & {r.mean_correctness_brier:.3f} & {r.equal_batch_ece10:.3f} & {int(r.interval_coverage)}/55 \\\\")
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=.72in]{geometry}\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref}
\hypersetup{hidelinks}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\emergencystretch=2em
\title{IDM-Viz: Matched-Budget Calibration Sensitivity on ELEC2}
\author{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}\date{16 November 2026}
\begin{document}\maketitle
\begin{abstract}
We compared the frozen historical isotonic correctness calibrator with cumulative isotonic and cumulative Platt-style logistic correctness recalibration on one fixed ELEC2 stream. Both updated methods received exactly the same calibration and monitoring labels released strictly before each forecast, while classifier predictions were unchanged. Neither update reduced equal-batch estimate MAE at any tested delay. This is an exploratory single-stream result, not a general calibration conclusion.
\end{abstract}
\section{Protocol}
The classifier, scaler, split, confidence values, predictions, reference and all 55 monitoring batches were frozen. At each batch, same-clock labels were unavailable to the forecast, including at delay zero. Cumulative isotonic and Platt calibrators used the original correctness-calibration set plus identical previously released monitoring examples. Platt's logistic regression (`C=1`) modeled correctness against logit predicted-class confidence. Delays were 0, 5 and 20. The evaluator verifies release clocks, matched label counts and exact prediction/ID equality. No hyperparameters or method were selected on these outcomes.
\section{Results}
\begin{table}[ht]\centering\small\begin{tabular}{rlrrrrr}\toprule Delay & Method & MAE & Bias & Brier & ECE$_{10}$ & Coverage \\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\caption{All 55 monitoring cohorts per delay. Coverage is the empirical count inside the plug-in Poisson-binomial interval; these are descriptive cohorts from one serial stream.} \end{table}
\begin{figure}[ht]\centering\includegraphics[width=.98\linewidth]{calibration_family.pdf}\caption{Estimation and correctness-calibration metrics by release delay. Lower MAE, Brier and ECE are preferable; coverage counts are shown as fractions out of 55.}
\end{figure}
The frozen calibrator had MAE 0.0586, correctness Brier 0.2160 and ECE$_{10}$ 0.0801 at every delay because its predictions did not change. Cumulative isotonic MAE was 0.0628, 0.0637 and 0.0591 for delays 0, 5 and 20; cumulative Platt MAE was 0.0653, 0.0666 and 0.0615. Both updated methods reduced signed bias toward zero but had worse MAE and ECE than frozen isotonic at every delay. Coverage was mixed (frozen 23/55; updated isotonic 18/55, 21/55, 24/55; Platt 21/55 at all delays). No independent-stream uncertainty interval is estimated.
\section{Interpretation and limitations}
On this one sequence, the lower-bias updates traded off against higher absolute error and poorer ECE. This cautions against treating cumulative recalibration as a fix for external calibration transfer. The single normalized-date ordering has known tied-date ambiguity and is not a second replicate. Serial cohorts are dependent; calibration, reference and interval uncertainty remain omitted. These results do not establish performance on another stream, improved classification, future-harm prediction, or an operational benefit.
\section{Reproduction}
\begin{verbatim}
& C:/Users/hp/anaconda3/python.exe scripts/run_elec2_platt_recalibration.py `
  --run results/phase3/elec2_temporal_20261116_v2 `
  --out results/phase3/elec2_platt_recalibration_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_elec2_platt_recalibration.py `
  --run results/phase3/elec2_temporal_20261116_v2 `
  --platt results/phase3/elec2_platt_recalibration_NEW `
  --isotonic results/phase3/elec2_recalibration_20261116_v1 `
  --out results/phase3/elec2_calibration_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/report_elec2_calibration_family.py `
  --run results/phase3/elec2_temporal_20261116_v2 `
  --platt results/phase3/elec2_platt_recalibration_NEW `
  --isotonic results/phase3/elec2_recalibration_20261116_v1 `
  --results results/phase3/elec2_calibration_family_NEW `
  --out reports/phase3/elec2_calibration_family_NEW
\end{verbatim}
\end{document}
'''
 (out/'ELEC2_CALIBRATION_FAMILY.tex').write_text(tex,encoding='utf-8');print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--platt',type=Path,required=True);p.add_argument('--isotonic',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.run,a.platt,a.isotonic,a.results,a.out)
