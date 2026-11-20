"""Build a concise LaTeX supplement for UCI split/model sensitivity."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation'
OUT=ROOT/'reports/phase3/uci224_split_model_sensitivity_20261116_v2'

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 d=pd.read_csv(RUN/'common_cohort_split_sensitivity.csv')
 metrics=pd.read_csv(RUN/'common_cohorts_per_split.csv')
 fig,axes=plt.subplots(1,2,figsize=(9.4,4.1),layout='constrained')
 palette={'logistic':'#167c80','tree_depth6':'#c66a42'}
 labels={'logistic':'Logistic','tree_depth6':'Depth-6 tree'}
 groups=[('batch1','One cohort'),('batch1_2','Two cohorts')]
 positions=[]
 for xi,(window,label) in enumerate(groups):
  g=d[d.history_window.eq(window)].set_index('model')
  x=xi*3
  for off,model in enumerate(('logistic','tree_depth6')):
   row=g.loc[model]; xpos=x+off
   axes[0].bar(xpos,row.calibrated_mae_mean,color=palette[model],width=.68,
               yerr=row.calibrated_mae_sd,capsize=3,edgecolor='white')
   axes[0].text(xpos,row.calibrated_mae_mean+.018,f"{row.calibrated_mae_mean:.2f}",ha='center',fontsize=8)
   axes[1].bar(xpos,row.macro_f1_mean,color=palette[model],width=.68,
               yerr=row.macro_f1_sd,capsize=3,edgecolor='white')
   axes[1].text(xpos,row.macro_f1_mean+.018,f"{row.macro_f1_mean:.2f}",ha='center',fontsize=8)
   positions.append(xpos)
 axes[0].set_title('Calibrated accuracy-estimate MAE')
 axes[0].set_ylabel('Equal-cohort mean absolute error')
 axes[0].set_ylim(0,.72)
 axes[1].set_title('Fixed six-class macro-F1')
 axes[1].set_ylabel('Equal-cohort mean macro-F1')
 axes[1].set_ylim(0,.75)
 ticks=[.5,1.5,3.5,4.5]
 axes[0].set_xticks(ticks,['Logistic','Depth-6 tree','Logistic','Depth-6 tree'],rotation=15)
 axes[1].set_xticks(ticks,['Logistic','Depth-6 tree','Logistic','Depth-6 tree'],rotation=15)
 for ax in axes:
  ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 for ax in axes: ax.axvline(2.5,color='#b8c2c6',lw=1)
 axes[0].text(1,.685,'History: batch 1',ha='center',fontsize=8,color='#536578')
 axes[0].text(4,.685,'History: batches 1–2',ha='center',fontsize=8,color='#536578')
 axes[1].text(1,.715,'History: batch 1',ha='center',fontsize=8,color='#536578')
 axes[1].text(4,.715,'History: batches 1–2',ha='center',fontsize=8,color='#536578')
 for ext in ('pdf','png','svg'): fig.savefig(OUT/f'split_model_sensitivity.{ext}',dpi=240)
 plt.close(fig)
 rows=[]
 for r in d.itertuples():
  hist='Batch 1' if r.history_window=='batch1' else 'Batches 1--2'
  model='Logistic' if r.model=='logistic' else 'Depth-6 tree'
  rows.append(f'{hist} & {model} & {r.accuracy_mean:.3f} $\\pm$ {r.accuracy_sd:.3f} & '
    f'{r.macro_f1_mean:.3f} $\\pm$ {r.macro_f1_sd:.3f} & '
    f'{r.calibrated_mae_mean:.3f} $\\pm$ {r.calibrated_mae_sd:.3f} & '
    f'{r.raw_confidence_mae_mean:.3f} & {r.reference_mae_mean:.3f} \\\\')
 table='\n'.join(rows)
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=.78in]{geometry}\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,tabularx,graphicx,xcolor,hyperref}
\hypersetup{hidelinks}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\title{IDM-Viz: External Split and Model Sensitivity on UCI 224}
\author{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}\date{16 November 2026}
\begin{document}\maketitle
\begin{abstract}
We tested whether external cohort results depended on one historical partition or the frozen logistic classifier. Six stratified historical splits, two models and one- versus two-cohort history windows were scored on shared UCI acquisition batches 3--10. Calibrated correctness transfer remained poor and varied materially by model/window. These repeated splits are sensitivity analyses on one dataset, not independent replications or evidence of general transfer.
\end{abstract}
\section{Locked extension}
We used UCI Gas Sensor Array Drift with six states (420001--420006), a 60/20/20 train/calibration/reference split, logistic regression and a fixed depth-6 decision tree. The one-cohort history uses batch 1; the pooled history uses batches 1--2. The latter has a larger training and labelled history budget. To compare windows fairly, reported cross-window metrics use the shared later batches 3--10; batch 2 is shown separately in raw result tables. All transformations and correctness calibrators use historical training/calibration data only. Monitoring labels appear only in evaluator outputs.
\section{Results}
\begin{table}[ht]\centering\small
\caption{Mean and standard deviation across six random historical splits, averaged equally over shared cohorts 3--10. Split variation is descriptive.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrr}\toprule History & Model & Accuracy & Macro-F1 & Calibrated MAE & Raw-confidence MAE & Reference-accuracy MAE \\\midrule
''' + table + r'''
\bottomrule\end{tabular}}
\end{table}
For batch-1-only history, calibrated accuracy-estimate MAE was 0.553 for logistic regression and 0.374 for the tree. With batches 1--2 pooled, it was 0.342 for logistic and 0.401 for the tree. The extra history changes the task and label budget, so these differences are not a causal estimate of the value of adding batch 2. Raw confidence was the strongest reported estimator for batch-1 logistic in the shared-window comparison (MAE 0.428), while the calibrated tree was best among the displayed estimates at MAE 0.374. No one estimator dominates across these configurations.
\begin{figure}[ht]\centering\includegraphics[width=.96\linewidth]{split_model_sensitivity.pdf}
\caption{Mean performance and calibrated-estimate error by model and history window. Error bars show standard deviation across six random splits, not confidence intervals over datasets.}
\end{figure}
\section{Scope and limitations}
This extension tests sensitivity to split and learner choice, not temporal adaptation or multiple-domain generalization. Splits overlap the same underlying records; six split seeds are not six independent datasets. The pooled-history window has more labelled history. The calibration transfer errors remain substantial and do not justify an all-purpose accuracy estimate under conditional shift. No interval guarantee or harmful-shift certainty follows from these metrics. The original Phase 2 DOCX and primary external outputs are unchanged.
\section{Reproduction}
\begin{verbatim}
python scripts/run_uci224_split_model_sensitivity.py --out results/phase3/uci224_NEW
python scripts/evaluate_uci224_split_model_sensitivity.py --run results/phase3/uci224_NEW
python scripts/report_uci224_split_model_sensitivity.py
\end{verbatim}
\end{document}
'''
 (OUT/'UCI224_SPLIT_MODEL_SENSITIVITY.tex').write_text(tex,encoding='utf-8')
 (OUT/'UCI224_SPLIT_MODEL_SENSITIVITY.md').write_text(
  '# UCI 224 split/model sensitivity\n\nSix stratified splits were run for logistic regression and a depth-6 tree under one- and two-cohort history windows. '
  'On shared cohorts 3--10, calibrated estimate MAE was 0.553 (logistic) and 0.374 (tree) with batch-1 history; '
  '0.342 and 0.401 with batches 1--2. These are split sensitivities on one dataset, and history windows have different label budgets. '
  'See the TeX report and figure.',encoding='utf-8')
 print(OUT)
if __name__=='__main__':main()
