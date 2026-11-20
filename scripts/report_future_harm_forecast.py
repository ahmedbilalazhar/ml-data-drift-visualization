"""Make editable TeX and vector figures for the held-out forecast study."""
from pathlib import Path
import json
import argparse
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; DEFAULT_RUN=ROOT/'results/phase3/future_harm_forecast_20261116_v1'; DEFAULT_OUT=ROOT/'reports/phase3/future_harm_forecast_20261116_v10'
def main(out=DEFAULT_OUT, run=DEFAULT_RUN):
 out=Path(out); run=Path(run); out.mkdir(parents=True,exist_ok=True); d=pd.read_csv(run/'metrics_by_delay_condition.csv'); boot=pd.read_csv(run/'paired_seed_bootstrap.csv'); s=pd.read_csv(run/'descriptive_threshold_metrics.csv')
 ds=d[d.delay.notna()].copy(); models=['time_only','unlabeled_monitor','full_released']; colors=['#536578','#cf8756','#16817a']
 fig,axes=plt.subplots(1,3,figsize=(9.5,3.4),layout='constrained')
 for ax,metric,title in zip(axes,['roc_auc','average_precision','brier'],['ROC AUC','Average precision','Brier score']):
  for model,color in zip(models,colors):
   g=ds[ds.model.eq(model)].sort_values('delay'); ax.plot(g.delay,g[metric],marker='o',lw=2,color=color,label=model.replace('_',' '))
  ax.set_title(title);ax.set_xlabel('Label delay (batches)');ax.set_xticks([0,5,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylim(0,1);axes[0].set_ylabel('Held-out seed rows');axes[0].legend(frameon=False,fontsize=7)
 for ext in ('pdf','png','svg'):fig.savefig(out/f'forecast_metrics.{ext}',dpi=240)
 plt.close(fig)
 overall=json.loads((run/'evaluation_summary.json').read_text(encoding='utf-8'))['overall']
 rows=[]
 for m in models:
  z=overall[m]; rows.append(f"{m.replace('_',' ')} & {z['roc_auc']:.3f} & {z['average_precision']:.3f} & {z['brier']:.3f} & {z['log_loss']:.3f} \\\\")
 for m in ['development_prevalence','last_released_harm','current_concern']:
  z=overall[m]; rows.append(f"{m.replace('_',' ')}$^*$ & {z['roc_auc']:.3f} & {z['average_precision']:.3f} & {z['brier']:.3f} & {z['log_loss']:.3f} \\\\")
 btxt='; '.join(f"{r.contrast.replace('_',' ')}: {r['mean']:+.4f} [{r.lower_95:+.4f}, {r.upper_95:+.4f}]" for _,r in boot.iterrows())
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=.76in]{geometry}\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref}
\hypersetup{hidelinks}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\emergencystretch=3em
\title{IDM-Viz: Seed-Held-Out Forecast of Future Harm}
\author{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}\date{16 November 2026}
\begin{document}\maketitle
\begin{abstract}
We tested a locked pooled logistic model for whether mean accuracy across the next five complete synthetic monitoring batches falls more than five percentage points below that stream's historical reference. Thirty development and thirty disjoint evaluation seeds span seven known generator conditions and delays 0, 5 and 20. The model using released-label history is compared with time-only and currently available unlabeled-monitor predictors. This evaluates interpolation within known synthetic laws, not transfer to unseen regimes or real streams.
\end{abstract}
\section{Protocol}
Forecasts are issued at batch t for batches t+1 through t+5; t=1,...,19. A monitoring label is available only when its batch plus delay is strictly less than t; same-clock labels arrive after the forecast. Three StandardScaler plus LogisticRegression(C=1) models are trained only on 30 development seeds. No seed or condition identifiers enter predictors and no threshold was tuned. The evaluation unit for uncertainty is the held-out seed, not the repeated row. Two thousand paired whole-seed bootstrap replicates compare Brier scores.
\section{Results}
\begin{table}[ht]\centering\small\begin{tabular}{lrrrr}\toprule Predictor set & ROC AUC & AP & Brier & Log loss \\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\caption{Pooled held-out performance over 11,970 future-window forecasts per predictor (30 seeds x 7 conditions x 3 delays x 19 origins). $^*$Heuristic score comparators; current concern is not trained or calibrated as a future-risk probability.} \end{table}
\begin{figure}[ht]\centering\includegraphics[width=.98\linewidth]{forecast_metrics.pdf}\caption{Discrimination and probability error by label delay. AP should be interpreted against the held-out target prevalence.}\end{figure}
Paired seed-bootstrap Brier differences (first model minus second; negative favors the first) were: ''' + btxt + r'''. The bootstrap resamples 30 seeds as clusters and retains each seed's seven conditions and three delays together.
\section{Interpretation and limitations}
This experiment directly evaluates a prospective five-batch outcome using only information available at forecast time. It does not establish that alarms cause harm, that a forecast is useful to an analyst, or that performance transfers beyond these seven synthetic mechanisms. The seven conditions are known during development, and the pooled model can exploit time-dependent prevalence; the held-out seeds are independent draws under those same laws. The synthetic labels share the generator construction, so this is not external validation. Small held-out seed count leaves material uncertainty. No calibrated risk certificate or intervention benefit is claimed. The online feature artifact excludes correctness and harm labels; future targets are stored separately for evaluation.
\section{Reproduction}
\begin{verbatim}
python scripts/run_future_harm_forecast.py --out results/forecast_NEW
python scripts/evaluate_future_harm_forecast.py results/forecast_NEW
python scripts/report_future_harm_forecast.py --run results/forecast_NEW --out reports/forecast_NEW
\end{verbatim}
The protocol lock, split, saved probabilities, evaluator targets, metrics and paired bootstrap are included with the editable \TeX source.
\end{document}
'''
 (out/'FUTURE_HARM_FORECAST.tex').write_text(tex,encoding='utf-8')
 print(out)
if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,default=DEFAULT_RUN); parser.add_argument('--out',type=Path,default=DEFAULT_OUT); args=parser.parse_args(); main(args.out,args.run)
