"""Plot and report the ELEC2 frozen versus released-label recalibration check."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/phase3/elec2_recalibration_20261116_v1/evaluation'
OUT=ROOT/'reports/phase3/elec2_recalibration_20261116'

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 d=pd.read_csv(DATA/'summary.csv')
 fig,axes=plt.subplots(1,3,figsize=(10,3.75),layout='constrained')
 delays=[0,5,20];xx=np.arange(3);w=.34
 frozen=d[d.method.eq('frozen')].set_index('delay').loc[delays]
 updated=d[d.method.eq('released_label_isotonic')].set_index('delay').loc[delays]
 series=[('equal_batch_mae','Estimation MAE',0,.08),('interval_coverage','Coverage (count / 55)',0,55),('row_weighted_brier','Correctness Brier score',0,.25)]
 for ax,(field,title,ymin,ymax) in zip(axes,series):
  a=frozen[field].to_numpy();b=updated[field].to_numpy()
  if field=='interval_coverage':a=a/55;b=b/55;ymax=1
  ax.bar(xx-w/2,a,w,color='#536578',label='Frozen')
  ax.bar(xx+w/2,b,w,color='#16817a',label='Released-label recalibration')
  ax.set_xticks(xx,[f'{v}' for v in delays]);ax.set_title(title,fontsize=10);ax.set_xlabel('Label delay (batches)')
  ax.set_ylim(ymin,ymax*1.12);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
  if field=='interval_coverage':
   for xval,val in zip(xx-w/2,a):ax.text(xval,val+.025,f'{int(round(val*55))}/55',ha='center',fontsize=7)
   for xval,val in zip(xx+w/2,b):ax.text(xval,val+.025,f'{int(round(val*55))}/55',ha='center',fontsize=7)
 axes[0].set_ylabel('Mean absolute error');axes[1].set_ylabel('Empirical cohort fraction');axes[2].set_ylabel('Mean Brier score')
 axes[0].legend(frameon=False,fontsize=7,loc='upper right')
 for ext in ('pdf','png','svg'):fig.savefig(OUT/f'recalibration_tradeoff.{ext}',dpi=240)
 plt.close(fig)
 f0=d[(d.delay==0)&(d.method=='frozen')].iloc[0]
 updated0=d[(d.delay==0)&(d.method=='released_label_isotonic')].iloc[0]
 table=[]
 for delay in delays:
  f=d[(d.delay==delay)&(d.method=='frozen')].iloc[0]
  u=d[(d.delay==delay)&(d.method=='released_label_isotonic')].iloc[0]
  table.append(f'{delay} & {f.equal_batch_mae:.3f} & {u.equal_batch_mae:.3f} & '
    f'{f.equal_batch_bias:.3f} & {u.equal_batch_bias:.3f} & {int(f.interval_coverage)}/55 & {int(u.interval_coverage)}/55 \\\\')
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=.8in]{geometry}\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref}
\hypersetup{hidelinks}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\title{IDM-Viz: Released-Label Recalibration on ELEC2}
\author{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}\date{16 November 2026}
\begin{document}\maketitle
\begin{abstract}
We tested whether updating only the correctness calibrator with monitoring labels released strictly before each forecast improves current-batch accuracy estimates on the ELEC2 stream. Recalibration reduced average signed bias toward zero but did not reduce equal-batch MAE at any tested delay; interval coverage also did not consistently improve. This is one outcome-informed stream analysis, not a general recalibration result.
\end{abstract}
\section{Protocol}
The classifier, features, training scaler and predictions are unchanged from the ELEC2 base run. A frozen isotonic calibrator is compared with a new isotonic calibrator refit on the original calibration examples plus all monitoring confidence/correctness pairs whose labels were released strictly before the current batch forecast. Same-clock labels are added only after that forecast is persisted, including at zero delay. Delays are 0, 5 and 20 batches. The classifier is not retrained. All 55 monitoring batches, including the final partial batch, are included.
\section{Results}
\begin{table}[ht]\centering\small
\caption{Equal-batch estimation errors and exact plug-in interval coverage. No inferential intervals are reported because this is one serial stream.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{rrrrrrr}\toprule Delay & Frozen MAE & Updated MAE & Frozen bias & Updated bias & Frozen coverage & Updated coverage \\\midrule
''' + '\n'.join(table) + r'''
\bottomrule\end{tabular}}
\end{table}
Frozen equal-batch MAE was 0.0586 for all delay settings. Recalibrated MAE was 0.0628, 0.0637 and 0.0591 at delays 0, 5 and 20, respectively. Recalibration moved signed bias from -0.0106 toward zero (to -0.0005, -0.0023 and -0.0018) but did not yield lower MAE. Empirical interval coverage was 23/55 for the frozen estimator and 18/55, 21/55 and 24/55 after recalibration. Correctness Brier scores were similar and mixed across delays.
\begin{figure}[ht]\centering\includegraphics[width=.98\linewidth]{recalibration_tradeoff.pdf}
\caption{Frozen versus released-label recalibration by delay. Coverage bars show cohort counts out of 55.}
\end{figure}
\section{Interpretation and limits}
The evidence does not support claiming that simple cumulative isotonic recalibration fixes this stream's estimator. Reduced aggregate bias can coexist with increased MAE and worse interval coverage. Delay comparisons have unequal available label budgets by design. Batches are serial and dependent, so the 55 cohorts are not independent replicates. The experiment uses one fixed stream, one model and an outcome-informed extension; calibration, reference and stream uncertainty are not certified. Recalibration does not establish improved classification accuracy or future-harm prediction.
\section{Reproduction}
\begin{verbatim}
python scripts/run_elec2_delayed_recalibration.py `
  --run results/phase3/elec2_NEW `
  --out results/phase3/recal_NEW
python scripts/evaluate_elec2_delayed_recalibration.py `
  --run results/phase3/elec2_NEW `
  --recalibration results/phase3/recal_NEW
python scripts/report_elec2_delayed_recalibration.py
\end{verbatim}
\end{document}
'''
 (OUT/'ELEC2_RECALIBRATION.tex').write_text(tex,encoding='utf-8')
 (OUT/'ELEC2_RECALIBRATION.md').write_text(
  '# ELEC2 released-label recalibration\n\nFrozen MAE was 0.0586; recalibration MAEs were 0.0628, 0.0637 and 0.0591 for delays 0/5/20. '
  'Bias moved toward zero, but MAE did not improve and interval coverage varied. This is a single outcome-informed stream analysis. See the TeX report and figure.',encoding='utf-8')
 print(OUT)

if __name__=='__main__':main()
