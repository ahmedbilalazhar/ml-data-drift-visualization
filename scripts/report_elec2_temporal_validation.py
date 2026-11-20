"""Generate an editable TeX report and vector evidence figure for ELEC2."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/phase3/elec2_temporal_20261116_v2'
OUT=ROOT/'reports/phase3/elec2_temporal_20261116'

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 m=pd.read_csv(RUN/'online_batches.csv');e=pd.read_csv(RUN/'evaluation/evaluator_batches.csv')
 r=pd.read_csv(RUN/'evaluation/delay_release.csv');s=json.loads((RUN/'evaluation/summary.json').read_text(encoding='utf-8'))
 fig,axes=plt.subplots(2,1,figsize=(9.2,6.1),sharex=False,gridspec_kw={'height_ratios':[2.2,1]},layout='constrained')
 x=e.batch.to_numpy()
 axes[0].fill_between(x,e.interval_lower,e.interval_upper,color='#15827a',alpha=.15,label='Plug-in predictive interval')
 axes[0].plot(x,e.estimate,color='#15827a',lw=1.8,marker='o',ms=3.3,label='Calibrated estimate')
 axes[0].plot(x,e.accuracy,color='#c45e37',lw=1.45,marker='.',ms=5,label='Observed accuracy (evaluator only)')
 alarms=m.loc[m.input_alarm,'batch']
 axes[0].scatter(alarms,np.full(len(alarms),1.015),marker='v',s=18,color='#263b50',clip_on=False,label='Input alarm')
 axes[0].set_ylim(0,1.07);axes[0].set_ylabel('Batch accuracy');axes[0].set_title('Frozen IDM-Viz evidence on the ELEC2 temporal stream')
 axes[0].grid(axis='y',alpha=.2);axes[0].legend(frameon=False,ncol=2,fontsize=8,loc='lower left')
 clocks=np.arange(1,len(e)+1)
 colors={0:'#536578',5:'#cf8756',20:'#15827a'}
 for d in (0,5,20):
  z=r[r.delay.eq(d)]
  cumulative=z.groupby('available_at').n.sum().reindex(clocks,fill_value=0).cumsum()/z.n.sum()
  axes[1].step(clocks,cumulative,where='post',color=colors[d],lw=1.8,label=f'{d}-batch delay')
 axes[1].set_ylim(0,1.04);axes[1].set_xlim(1,len(e));axes[1].set_ylabel('Labels verified / total');axes[1].set_xlabel('Monitoring batch / input clock')
 axes[1].grid(axis='y',alpha=.2);axes[1].legend(frameon=False,ncol=3,fontsize=8)
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 for ext in ('pdf','png','svg'):fig.savefig(OUT/f'elec2_timeline.{ext}',dpi=240)
 plt.close(fig)
 tex=rf'''\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=.78in]{{geometry}}\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,xurl,seqsplit}}
\hypersetup{{hidelinks}}\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{6pt}}
\title{{IDM-Viz: Second-Domain Stream Evaluation on ELEC2}}
\author{{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}}\date{{16 November 2026}}
\begin{{document}}\maketitle
\begin{{abstract}}
We evaluated the frozen IDM-Viz monitor on a second external binary stream, OpenML Electricity (ELEC2), using 45,312 records and a coarse time ordering reconstructed from normalized date and period fields. On the final 27,188 observations, row-weighted accuracy was {s['row_weighted_accuracy']:.3f} and fixed-model calibrated-accuracy MAE averaged {s['equal_batch_estimate_mae']:.3f} across 55 cohorts. Input tests alarmed in 55/55 batches. This is a one-domain extension without known drift onset labels; the alarms cannot be scored as drift recall.
\end{{abstract}}
\section{{Source and protocol}}
The source is \href{{https://www.openml.org/search?id=151}}{{OpenML dataset 151, version 1}} (45,312 records; public dataset visibility), also wrapped as `Elec2` in the \href{{https://github.com/online-ml/river/blob/main/river/datasets/elec2.py}}{{River stream dataset}}. OpenML lists Harries (1999), \emph{{SPLICE-2 Comparative Evaluation: Electricity Pricing}}, and Gama et al. (2004), \emph{{Learning with Drift Detection}}, DOI \href{{https://doi.org/10.1007/978-3-540-28645-5\_29}}{{10.1007/978-3-540-28645-5\_29}}.

The downloaded ARFF source SHA-256 is \texttt{{\seqsplit{{2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2}}}}. Its row order is not monotone by its normalized date field. We reconstructed a coarse order by normalized date, source order of `(date, weekday)` groups, then period. Eleven normalized date values have multiple weekday groups, so ordering within those ties is approximate. The six model inputs exclude date (ordering only) and contemporaneous NSW price (target-proximal): day, period, NSW demand, Victoria price, Victoria demand and scheduled transfer.

Rows 1--10,800 train the frozen logistic model and training-only scaler/PCA/clusters; rows 10,801--14,462 fit isotonic correctness calibration; rows 14,463--18,124 form the labelled reference; remaining rows are monitored in batches of 500, with a final batch of 188. The input reference is the last 1,000 rows of the reference partition. Six input KS tests plus three frozen PCA tests use within-batch Holm correction at 0.05. Predictions and scores were persisted before label release. Delays 0, 5 and 20 are replayed over identical caches, with post-stream drain.
\section{{Results}}
Across monitored rows, accuracy was {s['row_weighted_accuracy']:.3f}, balanced accuracy {s['row_weighted_balanced_accuracy']:.3f}, and fixed two-class macro-F1 {s['row_weighted_macro_f1']:.3f}. Equal-batch calibrated-estimate MAE was {s['equal_batch_estimate_mae']:.3f}, with signed bias {s['equal_batch_estimate_bias']:.3f}. Row-weighted correctness Brier score was {s['row_weighted_correctness_brier']:.3f}; equal-batch 10-bin ECE was {s['equal_batch_ece10']:.3f}. The plug-in interval covered {s['interval_coverage_batches']}/55 realized cohort accuracies. All 55 batches produced an input alarm. This counts observed feature distribution differences, not detections of known concept drift.

At the input end, delay 0 had 55/55 batches verifiable, delay 5 had 50/55, and delay 20 had 35/55; all 55 cohorts release after the respective drain. The final partial batch is retained. Online output contains no current labels, accuracy or correctness.
\begin{{figure}}[ht]\centering\includegraphics[width=.96\linewidth]{{elec2_timeline.pdf}}
\caption{{Fixed-model estimates, retrospectively observed accuracy, plug-in intervals, input alarms and delayed-label availability in coarse stream order. The lower panel shows the fraction of monitoring labels verified by each input clock.}}
\end{{figure}}
\section{{Interpretation and limits}}
This second domain adds evidence that the integrated feature/estimate/release pipeline runs beyond the Gas Sensor dataset. It does not establish general transfer: there is one additional domain and one fixed chronological split. The 55/55 input alarms may reflect sustained temporal feature change, but no labeled onset or stationary control exists. The reconstructed normalized-time order has tied groups. Accuracy and calibration outcomes are retrospective evaluation only. The Poisson-binomial interval assumes independent correctness indicators and known calibrated probabilities; batch dependence and calibration/reference uncertainty are omitted. No causal explanation, future-risk forecast or analyst benefit is established.
\section{{Reproduction}}
\begin{{verbatim}}
python scripts/run_elec2_temporal_validation.py --out results/phase3/elec2_NEW
python scripts/evaluate_elec2_temporal_validation.py --run results/phase3/elec2_NEW
python scripts/report_elec2_temporal_validation.py
\end{{verbatim}}
The Phase 2 DOCX and prior result versions remain unchanged.\end{{document}}
'''
 (OUT/'ELEC2_TEMPORAL_EXTENSION.tex').write_text(tex,encoding='utf-8')
 (OUT/'ELEC2_TEMPORAL_EXTENSION.md').write_text(
  f"# ELEC2 extension\n\n{len(e)} batches, {s['row_weighted_accuracy']:.3f} row-weighted accuracy, "
  f"{s['equal_batch_estimate_mae']:.3f} equal-batch estimate MAE, {s['interval_coverage_batches']}/55 interval coverage, "
  f"55/55 input-alarm batches. No known onset is available, so alarm recall is not measured. See TeX report and figure.",encoding='utf-8')
 print(OUT)

if __name__=='__main__':main()
